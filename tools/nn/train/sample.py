"""Greedy continuations from a checkpoint: plain prompts and FIM (PSM) prompts, built like the training documents.

  python sample.py --ckpt ~/work/ml-data/go/nn/smoke-go30/ckpt-latest.pt --snippets 3 --new 48
Snippets come from held-out 'test' fold files (deterministic by --seed) or from --prompt-file (raw Go source).
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CodeLM, ModelConfig  # noqa: E402
import data as D  # noqa: E402


def load(ckpt, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ck["config"])
    m = CodeLM(cfg); m.load_state_dict(ck["model"]); m.to(device).eval()
    return m, ck


def gen(model, tok, ids, n, device):
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        y = model.generate(x, n, stop_ids=(tok.eot, tok.file_sep))
    return y[0, len(ids):].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=D.DEFAULT_DATA)
    ap.add_argument("--vocab", default=D.DEFAULT_VOCAB)
    ap.add_argument("--snippets", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ctx", type=int, default=256, help="prompt tokens (plain) / prefix tokens (FIM)")
    ap.add_argument("--new", type=int, default=48)
    ap.add_argument("--no-path", action="store_true")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = D.Tokenizer(a.vocab)
    model, ck = load(a.ckpt, device)
    with_path = not (a.no_path or ck["args"].get("no_path"))
    sh = D.Shards(a.data, "test")
    paths = sh.paths() if with_path else None
    rng = np.random.default_rng(a.seed)
    cand = np.nonzero((sh.lengths > 2 * a.ctx + 64) & (sh.lengths < 4000))[0]
    picks = rng.choice(cand, size=a.snippets, replace=False)
    print(f"model {ck['config']['name']} step {ck['step']} ({ck['tokens']:,} tokens)")
    for k, i in enumerate(picks):
        body = sh.file(int(i)).astype(np.int64).tolist()
        hdr = [tok.file_sep] + (tok.encode(paths[i] + "\n") if with_path else [])
        print(f"\n===== snippet {k}: {sh.repos[sh.repo[i]]}/{paths[i] if paths else i} ({len(body)} tokens) =====")
        # plain continuation
        prompt = hdr + body[:a.ctx]
        out = gen(model, tok, prompt, a.new, device)
        print("--- PLAIN prompt tail:\n" + tok.decode(prompt[-40:]))
        print(">>> model:\n" + tok.decode(out))
        print("=== truth:\n" + tok.decode(body[a.ctx:a.ctx + a.new]))
        # FIM: prefix = first ctx tokens, suffix = tokens after a 32-token hole
        hole = 32
        pre, mid, suf = body[:a.ctx], body[a.ctx:a.ctx + hole], body[a.ctx + hole:a.ctx + hole + a.ctx]
        prompt = [tok.fim_prefix] + hdr + pre + [tok.fim_suffix] + suf + [tok.fim_middle]
        out = gen(model, tok, prompt, a.new, device)
        print("--- FIM (PSM) prefix tail:\n" + tok.decode(pre[-30:]) + "\n--- suffix head:\n" + tok.decode(suf[:30]))
        print(">>> model middle:\n" + tok.decode(out))
        print("=== truth middle:\n" + tok.decode(mid))


if __name__ == "__main__":
    main()
