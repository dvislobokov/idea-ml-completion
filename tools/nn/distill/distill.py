"""Sequence-level distillation data: a teacher (Hugging Face code model via vLLM) completes the rest of the line at sampled
FIM positions of the lm fold; the student (train.py --teacher) trains on those lines in place of random FIM middles.

  python -I distill.py sample --data ~/work/ml-data/csharp/bpe16k --vocab .../cs-16384.bpe --n 300000 --out pos.jsonl
  .venv-vllm/bin/python -I distill.py generate --model Qwen/Qwen2.5-Coder-1.5B --in pos.jsonl --out teacher.jsonl [--gpu 0 --part 0/2]
  python -I distill.py encode --vocab ... --in teacher.jsonl --out teacher.npz

Positions are sampled in TOKEN space from the encoded shards (so the student's prefix/suffix are exactly the shard tokens):
a cursor inside a line (line start with prob 0.5, else a uniform token of the line), the middle = the rest of that line
(<= 48 tokens), the suffix = what follows the line end. The teacher sees the decoded prefix tail / suffix head as a PSM
prompt (Qwen format) and must reproduce the typed remainder of the pre-token at the cursor (plain healing: we cut the
prompt at the last pre-token boundary before the cursor and strip the typed bytes from the output).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
sys.path.insert(0, os.path.join(HERE, "..", "tokenizer"))

MAX_MID = 48
PRE_TOK, SUF_TOK = 1024, 256


def cmd_sample(a):
    import data as D
    tok = D.Tokenizer(a.vocab)
    sh = D.Shards(a.data, a.fold)
    rng = np.random.default_rng(a.seed)
    paths = sh.paths()
    # files weighted uniformly (one position per draw; big files are not favoured)
    n_files = sh.n_files
    out = open(a.out, "w", encoding="utf-8")
    n = 0
    tries = 0
    while n < a.n and tries < a.n * 4:
        tries += 1
        fi = int(rng.integers(0, n_files))
        body = sh.file(fi)
        if len(body) < 64:
            continue
        nl = np.nonzero(tok.nl_start[body])[0]
        if len(nl) < 4:
            continue
        j = int(rng.integers(1, len(nl)))              # line j: tokens (start after nl[j-1] + ws) .. nl[j]
        b = int(nl[j])
        s = int(nl[j - 1]) + 1
        while s < b and tok.ws_only[body[s]]:
            s += 1
        if b - s < 1 or b - s > MAX_MID:
            continue
        a_ = s if rng.random() < 0.5 else int(rng.integers(s, b))
        pre_ids = body[max(0, a_ - PRE_TOK):a_]
        suf_ids = body[b:b + SUF_TOK]
        rec = {"fi": fi, "repo": sh.repos[int(sh.repo[fi])], "path": paths[fi], "a": a_, "b": b,
               "prefix": tok.decode(pre_ids),
               "suffix": tok.decode(suf_ids),
               "true": tok.decode(body[a_:b])}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
    out.close()
    print(f"{n} positions from {a.n} requested ({tries} draws) -> {a.out}")


def _is_word(c):
    return 65 <= c <= 90 or 97 <= c <= 122 or c == 95 or c >= 128 or 48 <= c <= 57


def heal_split(prefix: str):
    """Cut the prefix at the start of a trailing word (the typing situation); returns (prompt_prefix, typed)."""
    b = prefix.encode("utf-8")
    i = len(b)
    while i > 0 and _is_word(b[i - 1]):
        i -= 1
    if i < len(b) and i > 0 and b[i - 1] == 32:   # the pre-token owns its leading space
        i -= 1
    return b[:i].decode("utf-8", "replace"), b[i:].decode("utf-8", "replace")


def cmd_generate(a):
    from vllm import LLM, SamplingParams
    recs = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    if a.part:
        k, m = map(int, a.part.split("/"))
        recs = recs[k::m]
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    nl_ids = [i for i in range(len(tok)) if "\n" in tok.decode([i]) or "\r" in tok.decode([i])]
    fp, fs, fm, fsep = "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|file_sep|>"
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_new, stop=["\n"], stop_token_ids=nl_ids + [tok.eos_token_id],
                        logprobs=1, skip_special_tokens=True)
    out = open(a.out, "a", encoding="utf-8")
    B = a.batch
    for k in range(0, len(recs), B):
        chunk = recs[k:k + B]
        prompts, typed = [], []
        for r in chunk:
            pp, ty = heal_split(r["prefix"])
            prompts.append(f"{fp}{fsep}{r['path']}\n{pp}{fs}{r['suffix']}{fm}")
            typed.append(ty)
        outs = llm.generate(prompts, sp, use_tqdm=False)
        for r, o, ty in zip(chunk, outs, typed):
            text = o.outputs[0].text.split("\n")[0].split("\r")[0]
            lps = [list(d.values())[0].logprob for d in (o.outputs[0].logprobs or []) if d]
            conf = float(np.exp(sum(lps))) if lps else 0.0
            if ty:
                if not text.startswith(ty):
                    continue                     # the teacher did not reproduce the typed remainder: skip
                text = text[len(ty):]
            if not text.strip():
                continue
            out.write(json.dumps({"fi": r["fi"], "a": r["a"], "b": r["b"], "true": r["true"], "teacher": text, "conf": conf}, ensure_ascii=False) + "\n")
        if (k // B) % 20 == 0:
            print(f"{k + len(chunk)}/{len(recs)}", flush=True)
    out.close()


def cmd_encode(a):
    import data as D
    tok = D.Tokenizer(a.vocab)
    fi, aa, bb, off, ids = [], [], [], [0], []
    n = 0
    for l in open(a.inp, encoding="utf-8"):
        r = json.loads(l)
        t = tok.encode(r["teacher"])
        if not t or len(t) > MAX_MID:
            continue
        fi.append(r["fi"]); aa.append(r["a"]); bb.append(r["b"]); ids.extend(t); off.append(len(ids)); n += 1
    np.savez(a.out, fi=np.array(fi, np.int64), a=np.array(aa, np.int64), b=np.array(bb, np.int64),
             off=np.array(off, np.int64), ids=np.array(ids, np.uint16))
    print(f"{n} teacher samples, {len(ids)} tokens -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample"); s.add_argument("--data", required=True); s.add_argument("--vocab", required=True)
    s.add_argument("--fold", default="lm"); s.add_argument("--n", type=int, default=300000); s.add_argument("--seed", type=int, default=1)
    s.add_argument("--out", required=True)
    g = sub.add_parser("generate"); g.add_argument("--model", required=True); g.add_argument("--in", dest="inp", required=True)
    g.add_argument("--out", required=True); g.add_argument("--part", default=""); g.add_argument("--batch", type=int, default=256)
    g.add_argument("--max-new", type=int, default=48)
    e = sub.add_parser("encode"); e.add_argument("--vocab", required=True); e.add_argument("--in", dest="inp", required=True); e.add_argument("--out", required=True)
    a = ap.parse_args()
    {"sample": cmd_sample, "generate": cmd_generate, "encode": cmd_encode}[a.cmd](a)


if __name__ == "__main__":
    main()
