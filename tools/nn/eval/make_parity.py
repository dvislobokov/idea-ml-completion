#!/usr/bin/env python
"""Parity fixture for the Kotlin inference (ml-core `io.github.completionml.core.nn`, test `NnParity`).

Re-derives the positions and prompts of an `eval_inline.py` run (same seed/stride/positions, so prompt i here is
prompt i of the eval JSON) and writes, under --out-dir:

  prompts.bin  32 prompts (plain + SPM, lengths spread over 50..2000 tokens): path / prefix / suffix bytes (for the
               tokenizer parity), prompt ids, float32 logits of the last position from the float model AND from the
               int8 fake-quantised model (export.fake_quantize_, i.e. exactly the weights in the .cml), and the greedy
               48-token continuation (no stop rule) with per-token log-probs from both models.
  behav.bin    N positions x {plain, spm}: prompt ids, true rest of line, the eval's generated line (bf16 batched run,
               from the JSON) and a fresh fp32 fake-quantised greedy run with the eval's stop rule.
  meta.json    provenance.

The model runs in true fp32 (no autocast, TF32 off), batch 1, KV cache — the reference the Kotlin code is measured
against. Binary layout: big-endian, Java DataInputStream-compatible (readUTF strings, int32 lengths, uint16 ids).

  ~/work/nn/.venv/bin/python -I make_parity.py --ckpt ~/work/ml-data/go/nn/go31m-e1/ckpt-latest.pt \
      --eval-json ~/work/ml-data/go/nn/eval-inline-step6484.json --out-dir ~/work/ml-data/go/nn/parity
  ~/work/nn/.venv/bin/python -I make_parity.py --lang csharp --ckpt ~/work/ml-data/csharp/nn/cs31m-e1/ckpt-latest.pt \
      --eval-json ~/work/ml-data/csharp/nn/eval-inline-step5388.json --out-dir ~/work/ml-data/csharp/nn/parity
The language (lexer, manifest, repos root, vocabulary) comes from the eval JSON (`info.lang`) or `--lang`; the prompts
are rebuilt with `--heal none` semantics (the cursor as the eval placed it), whatever the harness default is now.
"""
import argparse
import copy
import hashlib
import json
import os
import random
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "work/nn/eval"))
sys.path.insert(0, os.path.join(HOME, "work/nn/train"))
import eval_inline as E  # noqa: E402
from model import CodeLM, ModelConfig  # noqa: E402
import export as X  # noqa: E402


# ------------------------------------------------------------------------------------------------- fp32 inference

@torch.no_grad()
def fwd(m, ids, pos0, kv):
    """One sequence, no autocast. ids [1, T] at positions pos0.., kv: per-layer (k, v) or None. Returns last logits."""
    T = ids.shape[1]
    x = m.tok_emb(ids)
    cos = m.rope_cos[pos0:pos0 + T][None, None]
    sin = m.rope_sin[pos0:pos0 + T][None, None]
    for li, blk in enumerate(m.layers):
        h = blk.attn_norm(x)
        a = blk.attn
        q = a.wq(h).view(1, T, a.n_heads, a.hd).transpose(1, 2)
        k = a.wk(h).view(1, T, a.n_kv, a.hd).transpose(1, 2)
        v = a.wv(h).view(1, T, a.n_kv, a.hd).transpose(1, 2)
        q, k = E._rope(q, cos, sin), E._rope(k, cos, sin)
        fresh = kv[li] is None
        if not fresh:
            k = torch.cat((kv[li][0], k), dim=2)
            v = torch.cat((kv[li][1], v), dim=2)
        kv[li] = (k, v)
        rep = a.n_heads // a.n_kv
        y = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), is_causal=fresh and T > 1)
        x = x + a.wo(y.transpose(1, 2).reshape(1, T, -1))
        x = x + blk.mlp(blk.ffn_norm(x))
    x = m.final_norm(x[:, -1:])
    return m.logits(x)[0, -1].float()


@torch.no_grad()
def greedy(m, prompt, max_new, stop, dev, sb=0):
    """Greedy continuation. stop: bool tensor over the vocabulary or None (no stopping). Returns
    (last-prompt logits, gen ids, log-probs of gen ids, stop kind, stop prob). Mirrors eval_inline.Generator.generate:
    at most max_new tokens are emitted, the (max_new+1)-th argmax only decides newline/limit."""
    kv = [None] * len(m.layers)
    ids = torch.tensor([prompt], dtype=torch.long, device=dev)
    logits0 = fwd(m, ids, 0, kv)
    logits = logits0
    gen, lps = [], []
    stop_kind, stop_p = "limit", None
    for step in range(max_new + 1):
        lp = torch.log_softmax(logits, dim=-1)
        nxt = int(torch.argmax(logits))
        if stop is not None and bool(stop[nxt]):
            stop_kind = "special" if nxt >= sb else "newline"
            stop_p = float(torch.exp(lp[nxt]))
            break
        if step == max_new:
            break
        gen.append(nxt); lps.append(float(lp[nxt]))
        logits = fwd(m, torch.tensor([[nxt]], dtype=torch.long, device=dev), len(prompt) + step, kv)
    return logits0, gen, lps, stop_kind, stop_p


# ------------------------------------------------------------------------------------------------- binary writer

class W:
    def __init__(self, path):
        self.f = open(path, "wb")

    def i32(self, v): self.f.write(struct.pack(">i", v))
    def f32s(self, a): self.f.write(np.ascontiguousarray(a, dtype=">f4").tobytes())
    def u16s(self, a):
        assert all(0 <= x < 65536 for x in a)
        self.f.write(np.asarray(a, dtype=">u2").tobytes())
    def i32s(self, a): self.f.write(np.asarray(a, dtype=">i4").tobytes())
    def utf(self, s):
        b = s.encode("utf-8"); assert len(b) < 65536
        self.f.write(struct.pack(">H", len(b)) + b)
    def bytes(self, b): self.i32(len(b)); self.f.write(b)
    def close(self): self.f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval-json", required=True)
    ap.add_argument("--lang", choices=sorted(E.LANGS), help="default: the eval JSON's language")
    ap.add_argument("--vocab", help="default ~/work/ml-data/tokenizer/<go|cs>-16384.bpe")
    ap.add_argument("--manifest", help="default ~/work/ml-data/<lang>/prepared/manifest.jsonl")
    ap.add_argument("--repos", help="default ~/work/ml-data/<lang>/repos")
    ap.add_argument("--out-dir", help="default ~/work/ml-data/<lang>/nn/parity")
    ap.add_argument("--n-logits", type=int, default=32)
    ap.add_argument("--n-behav", type=int, default=500)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rep = json.load(open(a.eval_json))
    info = rep["info"]
    lang = a.lang or info.get("lang", "go")
    E.LANG = E.LANGS[lang]
    a.vocab = a.vocab or os.path.join(E.DATA, E.LANG["vocab"])
    a.manifest = a.manifest or os.path.join(E.DATA, E.LANG["manifest"])
    a.repos = a.repos or os.path.join(E.DATA, E.LANG["repos"])
    a.out_dir = a.out_dir or os.path.join(E.DATA, E.LANG["out_dir"], "parity")
    assert info.get("heal", "none") == "none", "the eval JSON must come from a --heal none run (prompts end at the cursor)"
    modes = ["plain", "spm"]
    os.makedirs(a.out_dir, exist_ok=True)
    for mo in modes:
        assert mo in rep["modes"], f"eval JSON has no mode {mo}"
    tok = E.Tokenizer(a.vocab)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    assert int(ck["step"]) == int(info["step"]), (ck["step"], info["step"])
    cfg = ModelConfig.from_dict(ck["config"])
    m = CodeLM(cfg); m.load_state_dict(ck["model"]); m.float().to(dev).eval()
    mq = copy.deepcopy(m); X.fake_quantize_(mq)
    del ck
    V = cfg.vocab_size
    stop = torch.zeros(V, dtype=torch.bool)
    for i in range(tok.special_base):
        stop[i] = tok.vocab.tokens[i][:1] in (b"\n", b"\r")
    stop[tok.special_base:] = True
    stop = stop.to(dev)

    t0 = time.time()
    files = E.read_manifest(a.manifest)
    positions = E.sample_positions(files, a.repos, info["n_positions"], info["stride"], info["seed"], info["n_in_string"] > 0)
    main_pos = [p for p in positions if p["kind"] != "in-string"]
    assert len(main_pos) == info["n_positions"], (len(main_pos), info["n_positions"])
    print(f"{len(main_pos)} positions rebuilt in {time.time() - t0:.0f} s", flush=True)
    recs = {mo: rep["modes"][mo]["positions"] for mo in modes}
    for i, p in enumerate(main_pos):   # alignment with the eval records
        r = recs["plain"][i]
        assert (r["repo"], r["path"], r["line"]) == (p["repo"], p["path"], p["line_no"]), (i, r, p["path"])
        assert r["true"] == p["true_code"].decode("utf-8", "replace")

    def prompt_parts(p):
        text, cur = p["text"], p["cursor"]
        aa = max(0, cur - 40000)
        if aa > 0:
            nl = text.find(b"\n", aa, cur)
            aa = nl + 1 if nl >= 0 else aa
        return p["path"].encode(), text[aa:cur], text[p["eol"]:p["eol"] + 16000]

    prompts = {mo: [E.build_prompt(tok, p, mo, info["ctx"], info["suffix_tokens"], info["with_path"], info["max_prefix"])
                    for p in main_pos] for mo in modes}

    # ---- 32 logit prompts: per mode, quantiles of the prompt length over [50, ctx]
    rng = random.Random(a.seed)
    chosen = []
    per = a.n_logits // len(modes)
    for mo in modes:
        cand = sorted((len(prompts[mo][i]), i) for i in range(len(main_pos)) if len(prompts[mo][i]) >= 50)
        for k in range(per):
            lo = int(len(cand) * k / per); hi = max(lo + 1, int(len(cand) * (k + 1) / per))
            chosen.append((mo, cand[rng.randrange(lo, hi)][1]))
    w = W(os.path.join(a.out_dir, "prompts.bin"))
    w.utf("nn-parity-1"); w.i32(V); w.i32(len(chosen))
    t0 = time.time()
    for mo, i in chosen:
        p = main_pos[i]; ids = prompts[mo][i]
        path, pre, suf = prompt_parts(p)
        w.utf(f"{mo}#{i}"); w.utf(mo); w.i32(i)
        w.bytes(path); w.bytes(pre); w.bytes(suf)
        w.i32(info["ctx"]); w.i32(info["max_prefix"]); w.i32(info["suffix_tokens"])
        w.i32(len(ids)); w.u16s(ids)
        lg, gen, lps, _, _ = greedy(m, ids, 48, None, dev)
        lgq, genq, lpsq, _, _ = greedy(mq, ids, 48, None, dev)
        w.i32(V); w.f32s(lg.cpu().numpy()); w.f32s(lgq.cpu().numpy())
        w.i32(len(gen)); w.i32s(gen); w.f32s(np.array(lps))
        w.i32(len(genq)); w.i32s(genq); w.f32s(np.array(lpsq))
        d = (lg - lgq).abs().max().item()
        print(f"  {mo}#{i}: {len(ids)} tokens, |float-fq| max {d:.3f}, argmax {int(lg.argmax())}/{int(lgq.argmax())}, "
              f"gen agree {sum(1 for x, y in zip(gen, genq) if x == y)}/48 ({tok.decode(gen[:12])!r})", flush=True)
    w.close()
    print(f"prompts.bin: {len(chosen)} prompts, {time.time() - t0:.0f} s", flush=True)

    # ---- behavioural set: n positions x modes, fp32 fake-quantised run with the eval stop rule
    sel = sorted(rng.sample(range(len(main_pos)), min(a.n_behav, len(main_pos))))
    w = W(os.path.join(a.out_dir, "behav.bin"))
    w.utf("nn-behav-1"); w.i32(V); w.i32(len(sel) * len(modes))
    t0 = time.time()
    agree = {mo: 0 for mo in modes}
    exact_eval = {mo: 0 for mo in modes}; exact_fq = {mo: 0 for mo in modes}
    for n, i in enumerate(sel):
        p = main_pos[i]
        for mo in modes:
            ids = prompts[mo][i]; r = recs[mo][i]
            _, gen, lps, kind, sp = greedy(mq, ids, 48, stop, dev, tok.special_base)
            gb = tok.decode(gen)
            gcode, _ = E.code_part(gb)
            ex = gcode == p["true_code"].rstrip() and len(gcode) > 0
            exact_fq[mo] += ex; exact_eval[mo] += r["exact"]
            agree[mo] += gb.decode("utf-8", "replace") == r["gen"]
            w.i32(i); w.utf(mo); w.utf(p["kind"])
            w.i32(len(ids)); w.u16s(ids)
            w.bytes(p["true_code"]); w.bytes(r["gen"].encode("utf-8")); w.utf(r["stop"])
            w.bytes(gb); w.i32(len(gen)); w.u16s(gen); w.utf(kind); w.i32(len(lps)); w.f32s(np.array(lps))
        if n % 100 == 99:
            print(f"  behav {n + 1}/{len(sel)} ({time.time() - t0:.0f} s)", flush=True)
    w.close()
    for mo in modes:
        k = len(sel)
        print(f"behav {mo}: fp32-fq vs eval(bf16) line agreement {agree[mo]}/{k}; line exact: eval {exact_eval[mo]}/{k}, fp32-fq {exact_fq[mo]}/{k}")
    meta = {"ckpt": a.ckpt, "step": info["step"], "eval_json": a.eval_json, "vocab": a.vocab, "lang": lang,
            "vocabSha256": hashlib.sha256(open(a.vocab, "rb").read()).hexdigest(), "seed": a.seed,
            "n_logits": len(chosen), "n_behav_positions": len(sel), "modes": modes, "ctx": info["ctx"],
            "max_prefix": info["max_prefix"], "suffix_tokens": info["suffix_tokens"], "max_new": 48,
            "behav_positions": sel, "logit_prompts": [f"{mo}#{i}" for mo, i in chosen],
            "python_line_exact_subset": {mo: exact_eval[mo] / len(sel) for mo in modes},
            "fq32_line_exact_subset": {mo: exact_fq[mo] / len(sel) for mo in modes},
            "fq32_vs_eval_agreement": {mo: agree[mo] / len(sel) for mo in modes}}
    json.dump(meta, open(os.path.join(a.out_dir, "meta.json"), "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
