"""Train the code LM on the BPE shards. bf16 autocast, AdamW, cosine LR with warmup, grad accumulation; one GPU, or
several with `torchrun --nproc_per_node N train.py ...` (DDP: data groups striped by rank, --tokens-per-step is the global batch,
checkpoints from rank 0 carry every rank's stream state, so a run resumes only with the same world size).

Example (smoke):  python train.py --preset go31m --run smoke-go30 --tokens-per-step 524288 --micro-batch 32 \
                      --max-minutes 15 --eval-every 200 --ckpt-every 200 --compile
Checkpoints: <out>/<run>/ckpt-latest.pt (+ ckpt-<step>.pt every --keep-every), metrics in <out>/<run>/metrics.jsonl,
config in <out>/<run>/config.json. Re-running with the same --run resumes from ckpt-latest.pt unless --no-resume.
"""
import argparse
import contextlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CodeLM, ModelConfig, PRESETS, preset  # noqa: E402
import data as D  # noqa: E402

DEFAULT_OUT = os.path.expanduser("~/work/ml-data/go/nn")
RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))


def say(*args, **kw):
    """print from rank 0 only"""
    if RANK == 0:
        print(*args, flush=True)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--data", default=D.DEFAULT_DATA)
    ap.add_argument("--vocab", default=D.DEFAULT_VOCAB)
    ap.add_argument("--train-fold", default="lm")
    ap.add_argument("--eval-fold", default="test")
    # model
    ap.add_argument("--preset", default="go31m", choices=list(PRESETS))
    ap.add_argument("--d-model", type=int); ap.add_argument("--n-layers", type=int); ap.add_argument("--n-heads", type=int)
    ap.add_argument("--n-kv-heads", type=int); ap.add_argument("--ffn-dim", type=int)
    ap.add_argument("--seq-len", type=int, default=2048)
    # data
    ap.add_argument("--fim-rate", type=float, default=0.5)
    ap.add_argument("--spm-rate", type=float, default=0.5)
    ap.add_argument("--line-rate", type=float, default=0.5, help="share of FIM documents with a line-aligned middle (ends at a line end)")
    ap.add_argument("--single-line", type=float, default=0.5, help="probability that a line-aligned middle spans exactly one line")
    ap.add_argument("--t-min", type=float, default=2e6, help="repo down-weighting knee in tokens (0 = off)")
    ap.add_argument("--no-path", action="store_true", help="do not prepend the file path after <|file_sep|>")
    ap.add_argument("--max-file-tokens", type=int, default=0, help="skip files longer than this (0 = keep all)")
    ap.add_argument("--seed", type=int, default=1)
    # optimisation
    ap.add_argument("--micro-batch", type=int, default=32, help="sequences per forward pass")
    ap.add_argument("--tokens-per-step", type=int, default=1 << 19, help="rounded to micro-batch*seq-len multiples")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=500, help="warmup steps")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-tokens", type=float, default=0, help="stop after this many training tokens")
    ap.add_argument("--max-minutes", type=float, default=0, help="wall-clock limit (checkpoints at the end)")
    ap.add_argument("--beta1", type=float, default=0.9); ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-fused", action="store_true")
    # bookkeeping
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-windows", type=int, default=128, help="windows per eval variant (plain / FIM)")
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--keep-every", type=int, default=0, help="also keep a permanent ckpt-<step>.pt every N steps")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-warm-cache", action="store_true", help="do not pre-read the token shard into the page cache")
    ap.add_argument("--io-threads", type=int, default=4)
    return ap.parse_args()


def make_config(a):
    ov = {k: getattr(a, k) for k in ("d_model", "n_layers", "n_heads", "n_kv_heads", "ffn_dim") if getattr(a, k) is not None}
    return preset(a.preset, max_context=a.seq_len, **ov)


def lr_at(step, a, total_steps):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    if total_steps <= a.warmup:
        return a.lr
    t = min(1.0, (step - a.warmup) / max(1, total_steps - a.warmup))
    return a.lr * (a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def evaluate(model, windows, micro_batch, device):
    """Mean token loss over fixed windows [n, T+1] (numpy uint16)."""
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(windows), micro_batch):
        w = torch.from_numpy(windows[i:i + micro_batch].astype(np.int64)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(w[:, :-1], w[:, 1:])
        tot += loss.item() * w.shape[0]; n += w.shape[0]
    model.train()
    return tot / n


def save_ckpt(path, raw_model, opt, step, tokens, stream_state, a, cfg):
    tmp = path + ".tmp"
    torch.save({"model": raw_model.state_dict(), "optimizer": opt.state_dict(), "step": step, "tokens": tokens,
                "stream": stream_state, "args": vars(a), "config": cfg.to_dict(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, tmp)
    os.replace(tmp, path)


def main():
    a = parse()
    torch.manual_seed(a.seed)
    assert torch.cuda.is_available()
    ddp = WORLD > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = "cuda"
    main_proc = RANK == 0
    if ddp:
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
        say(f"DDP: {WORLD} ranks (NCCL {torch.cuda.nccl.version()})")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run_dir = os.path.join(a.out, a.run)
    os.makedirs(run_dir, exist_ok=True)
    cfg = make_config(a)
    ckpt_path = os.path.join(run_dir, "ckpt-latest.pt")
    resume = os.path.exists(ckpt_path) and not a.no_resume
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False) if resume else None
    if ck:
        cfg = ModelConfig.from_dict(ck["config"])
    if main_proc:
        json.dump({"config": cfg.to_dict(), "args": vars(a), "world_size": WORLD}, open(os.path.join(run_dir, "config.json"), "w"), indent=1)

    tokens_per_micro = a.micro_batch * a.seq_len
    accum = max(1, round(a.tokens_per_step / (tokens_per_micro * WORLD)))
    tokens_per_step = accum * tokens_per_micro * WORLD
    say(f"run {a.run}: {cfg.name} {cfg.param_count():,} params ({cfg.non_embedding_params():,} non-emb); "
          f"micro-batch {a.micro_batch} x {a.seq_len} x accum {accum} x {WORLD} ranks = {tokens_per_step:,} tokens/step", flush=True)

    # ---------------------------------------------------------------- data
    tok = D.Tokenizer(a.vocab)
    assert tok.vocab_size == cfg.vocab_size, (tok.vocab_size, cfg.vocab_size)
    train_sh = D.Shards(a.data, a.train_fold)
    if not a.no_warm_cache and main_proc:
        D.warm_page_cache(train_sh.tokens_path)
    if ddp and not main_proc:
        dist.barrier()          # rank 0 builds the path-token cache (<fold>.pathtok.u16) first; the others reuse it
    stream = D.PackedStream(train_sh, tok, seq_len=a.seq_len, seed=a.seed, fim_rate=a.fim_rate, spm_rate=a.spm_rate,
                            line_rate=a.line_rate, single_line=a.single_line,
                            t_min=a.t_min, with_path=not a.no_path, max_file_tokens=a.max_file_tokens,
                            io_threads=a.io_threads, rank=RANK, world_size=WORLD, name=f"train/{RANK}" if ddp else "train")
    if ck:
        st = ck["stream"]
        if isinstance(st, list):
            assert len(st) == WORLD, f"checkpoint was written by {len(st)} ranks, running with {WORLD}"
            st = st[RANK]
        else:
            assert WORLD == 1, "single-GPU checkpoint cannot be resumed under DDP (stream state is per rank)"
        stream.load_state_dict(st)
    eval_sh = D.Shards(a.data, a.eval_fold)
    t0 = time.time()
    ev_plain = D.eval_windows(eval_sh, tok, a.eval_windows, a.seq_len, fim=False, with_path=not a.no_path)
    ev_fim = D.eval_windows(eval_sh, tok, a.eval_windows, a.seq_len, fim=True, with_path=not a.no_path)
    if ddp and main_proc:
        dist.barrier()          # the eval fold's path-token cache is built above too: keep it in the rank-0-first section
    say(f"eval set: {a.eval_windows} plain + {a.eval_windows} FIM windows from '{a.eval_fold}' ({time.time()-t0:.1f}s)", flush=True)

    # ---------------------------------------------------------------- model / optimiser
    model = CodeLM(cfg).to(device)
    if ck:
        model.load_state_dict(ck["model"])
    opt = torch.optim.AdamW(model.param_groups(a.weight_decay), lr=a.lr, betas=(a.beta1, a.beta2), eps=1e-8,
                            fused=not a.no_fused)
    if ck:
        opt.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["torch_rng"]); torch.cuda.set_rng_state(ck["cuda_rng"])
    step = ck["step"] if ck else 0
    tokens = ck["tokens"] if ck else 0
    if ck:
        say(f"resumed from {ckpt_path}: step {step}, {tokens:,} tokens, stream epoch {stream.epoch} cursor {stream.cursor}", flush=True)
    raw_model = model
    fwd = torch.compile(model) if a.compile else model
    if ddp:   # DDP around the compiled module (nanoGPT pattern); rope buffers are constant, no buffer broadcast
        fwd = torch.nn.parallel.DistributedDataParallel(fwd, device_ids=[local_rank], broadcast_buffers=False)
    eval_model = raw_model if ddp else fwd

    def gather_stream(st):
        """stream state of every rank (list) for the checkpoint; a plain dict on one GPU"""
        if not ddp:
            return st
        objs = [None] * WORLD
        dist.all_gather_object(objs, st)
        return objs

    if a.max_steps:
        total_steps = a.max_steps
    elif a.max_tokens:
        total_steps = int(a.max_tokens // tokens_per_step)
    else:
        total_steps = 10 ** 9  # time-limited: constant LR after warmup
    say(f"schedule: warmup {a.warmup}, total steps {total_steps if total_steps < 10**9 else 'open'}, "
          f"lr {a.lr} -> {a.lr*a.min_lr_ratio}", flush=True)

    log = open(os.path.join(run_dir, "metrics.jsonl"), "a") if main_proc else None

    def write(rec):
        if log is None:
            return
        rec["time"] = time.time()
        log.write(json.dumps(rec) + "\n"); log.flush()

    pf = D.Prefetcher(stream, a.micro_batch)
    model.train()
    start = time.time()
    last_log = start
    toks_since_log = 0
    loss_acc = 0.0
    stream_state = stream.state_dict()
    torch.cuda.reset_peak_memory_stats()
    deadline = start + 60 * a.max_minutes if a.max_minutes else None
    stop_reason = None
    try:
        while True:
            if step >= total_steps:
                stop_reason = "max steps"; break
            if deadline:
                stop_t = torch.tensor([time.time() > deadline], device=device)
                if ddp:
                    dist.broadcast(stop_t, 0)     # every rank stops on the same step
                if stop_t.item():
                    stop_reason = "time limit"; break
            lr = lr_at(step, a, total_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            ts = time.time()
            loss_sum = 0.0
            for k in range(accum):
                x, y, stream_state = pf.next(device)
                if ddp:
                    fwd.require_backward_grad_sync = (k == accum - 1)   # all-reduce once per step
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = fwd(x, y)
                (loss / accum).backward()
                loss_sum += loss.detach()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            tokens += tokens_per_step
            toks_since_log += tokens_per_step
            loss_t = loss_sum / accum
            if ddp:
                dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
            loss_val = loss_t.item()   # sync point once per step
            loss_acc += loss_val
            step_time = time.time() - ts
            if step % a.log_every == 0 or step == 1:
                now = time.time()
                tps = toks_since_log / (now - last_log)
                rec = {"step": step, "tokens": tokens, "loss": loss_acc / (a.log_every if step % a.log_every == 0 else 1),
                       "lr": lr, "grad_norm": float(gn), "step_time": step_time, "tok_s": tps,
                       "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                       "gpu_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
                       "epoch": stream_state["epoch"], "elapsed": now - start}
                write(rec)
                say(f"step {step:6d} | loss {rec['loss']:.4f} | lr {lr:.2e} | gn {float(gn):.2f} | "
                      f"{step_time*1000:.0f} ms/step | {tps/1e3:.0f}k tok/s | mem {rec['gpu_mem_gb']:.1f} GB | "
                      f"{tokens/1e9:.3f} G tokens", flush=True)
                last_log = now; toks_since_log = 0; loss_acc = 0.0
            if a.eval_every and step % a.eval_every == 0 and main_proc:
                te = time.time()
                lp = evaluate(eval_model, ev_plain, a.micro_batch, device)
                lf = evaluate(eval_model, ev_fim, a.micro_batch, device)
                write({"step": step, "tokens": tokens, "eval_loss": lp, "eval_ppl": math.exp(lp),
                       "eval_fim_loss": lf, "eval_fim_ppl": math.exp(lf), "eval_time": time.time() - te})
                say(f"  eval @ {step}: plain loss {lp:.4f} ppl {math.exp(lp):.2f} | FIM loss {lf:.4f} ppl {math.exp(lf):.2f} "
                      f"({time.time()-te:.1f}s)", flush=True)
            if a.ckpt_every and step % a.ckpt_every == 0:
                st_all = gather_stream(stream_state)
                if main_proc:
                    save_ckpt(ckpt_path, raw_model, opt, step, tokens, st_all, a, cfg)
                    if a.keep_every and step % a.keep_every == 0:
                        save_ckpt(os.path.join(run_dir, f"ckpt-{step}.pt"), raw_model, opt, step, tokens, st_all, a, cfg)
                say(f"  checkpoint @ {step}", flush=True)
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    pf.close()
    say(f"stopping ({stop_reason}) at step {step}, {tokens:,} tokens, {(time.time()-start)/60:.1f} min", flush=True)
    st_all = gather_stream(stream_state)
    if main_proc:
        save_ckpt(ckpt_path, raw_model, opt, step, tokens, st_all, a, cfg)
        lp = evaluate(eval_model, ev_plain, a.micro_batch, device)
        lf = evaluate(eval_model, ev_fim, a.micro_batch, device)
        write({"step": step, "tokens": tokens, "eval_loss": lp, "eval_ppl": math.exp(lp), "eval_fim_loss": lf,
               "eval_fim_ppl": math.exp(lf), "final": True, "stop": stop_reason})
        say(f"final eval @ {step}: plain loss {lp:.4f} ppl {math.exp(lp):.2f} | FIM loss {lf:.4f} ppl {math.exp(lf):.2f}", flush=True)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
