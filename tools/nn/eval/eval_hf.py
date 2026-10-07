"""Whole-line completion ceiling check with a Hugging Face code model (teacher candidate), on the same test-fold positions,
lexer and metrics as eval_inline.py. Nothing here ships: it only tells how far a strong model gets on our positions
(context = prefix tail + suffix head of the same file, PSM FIM prompt), so that the distillation headroom is known.

  python -I eval_hf.py --model Qwen/Qwen2.5-Coder-1.5B --lang csharp --positions 500

Healing is the plain variant (prompt cut at the pre-token boundary, the typed remainder must be reproduced and is
stripped; no constrained decoding), confidence = product of the greedy token probabilities incl. the newline token.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_inline as E  # noqa: E402

FIM = {"qwen": ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|file_sep|>")}


def newline_ids(tok):
    ids = []
    for i in range(len(tok)):
        s = tok.decode([i])
        if "\n" in s or "\r" in s:
            ids.append(i)
    return set(ids)


def build_prompt(p, fmt, heal, pre_chars, suf_chars):
    text, cur = p["text"], p["boundary"] if heal else p["cursor"]
    a = max(0, cur - pre_chars)
    if a > 0:
        nl = text.find(b"\n", a, cur)
        a = nl + 1 if nl >= 0 else a
    pre = text[a:cur].decode("utf-8", "replace")
    suf = text[p["eol"]:p["eol"] + suf_chars].decode("utf-8", "replace")
    fp, fs, fm, fsep = fmt
    return f"{fp}{fsep}{p['path']}\n{pre}{fs}{suf}{fm}"


def score(p, gen: bytes, heal, heal_miss):
    gen_code, gen_toks = E.code_part(gen)
    gl, gln = E.lex_norm(gen_toks, False), E.lex_norm(gen_toks, True)
    tl, tln = p["true_lex"], p["true_lex_norm"]
    match = E.lcp(tl, gl)
    return {"repo": p["repo"], "path": p["path"], "line": p["line_no"], "kind": p["kind"],
            "true": p["true_code"].decode("utf-8", "replace"), "gen": gen.decode("utf-8", "replace"),
            "rest_len": len(tl), "match": match, "first_ok": match >= 1,
            "ok3": match >= min(3, len(tl)),
            "exact": (not heal_miss) and gen_code == p["true_code"].rstrip() and len(gen_code) > 0,
            "exact_norm": (not heal_miss) and len(tln) > 0 and gln == tln,
            "healed": bool(p["typed"]) if heal else False, "heal_miss": heal_miss,
            "punct_only": E.punct_only(gen)}


def summarise(recs):
    n = len(recs)
    r = lambda f: sum(1 for x in recs if f(x)) / max(1, n)
    le8 = [x for x in recs if 1 <= x["rest_len"] <= 8]
    s = {"n": n, "line_exact": r(lambda x: x["exact"]), "line_exact_norm": r(lambda x: x["exact_norm"]),
         "line_exact_le8": sum(1 for x in le8 if x["exact"]) / max(1, len(le8)), "n_le8": len(le8),
         "first_lex_ok": r(lambda x: x["first_ok"]), "ok3": r(lambda x: x["ok3"]),
         "healed": sum(1 for x in recs if x["healed"]), "heal_miss": sum(1 for x in recs if x["heal_miss"]),
         "thresholds": {}}
    for thr in E.THRESHOLDS:
        shown = [x for x in recs if x["conf_prod"] >= thr]
        s["thresholds"][str(thr)] = {"shown": len(shown) / max(1, n),
                                     "prec_exact": sum(1 for x in shown if x["exact"]) / max(1, len(shown)),
                                     "prec_ok3": sum(1 for x in shown if x["ok3"]) / max(1, len(shown))}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lang", default="csharp", choices=sorted(E.LANGS))
    ap.add_argument("--fmt", default="qwen", choices=sorted(FIM))
    ap.add_argument("--positions", type=int, default=500)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pre-chars", type=int, default=6000, help="prefix tail (≈1 500 tokens)")
    ap.add_argument("--suf-chars", type=int, default=2000, help="suffix head (≈500 tokens)")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--heal", default="boundary", choices=("none", "boundary"))
    ap.add_argument("--manifest", help="prepare manifest (default <lang>/prepared/manifest.jsonl; e.g. manifest-fresh.jsonl)")
    ap.add_argument("--out")
    ap.add_argument("--dump", type=int, default=0)
    a = ap.parse_args()
    E.LANG = E.LANGS[a.lang]
    manifest = a.manifest or os.path.join(E.DATA, E.LANG["manifest"]); repos = os.path.join(E.DATA, E.LANG["repos"])
    out = a.out or os.path.join(E.DATA, E.LANG["out_dir"], "eval-hf-" + a.model.split("/")[-1])
    heal = a.heal != "none"

    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    except Exception as e:   # multimodal checkpoints (Qwen3.5/3.6): text-only generation through the full model
        print(f"AutoModelForCausalLM failed ({str(e)[:120]}); loading AutoModelForImageTextToText", flush=True)
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.cuda().eval()
    nl_ids = newline_ids(tok)
    # every added token ends the middle: Qwen closes a FIM middle with <|fim_pad|> / <|endoftext|>, which are not
    # all flagged 'special' (so skip_special_tokens would not even strip them)
    added = set(tok.added_tokens_decoder) if hasattr(tok, 'added_tokens_decoder') else set()
    stop_ids = nl_ids | {tok.eos_token_id} | set(tok.all_special_ids) | added
    print(f"{a.model}: {sum(p.numel() for p in model.parameters())/1e6:.0f} M params, {len(nl_ids)} newline tokens, "
          f"loaded in {time.time()-t0:.0f} s", flush=True)

    files = E.read_manifest(manifest)
    positions = E.sample_positions(files, repos, a.positions, a.stride, a.seed, True, True, 400)
    main_pos = [p for p in positions if p["kind"] not in E.EXTRA_KINDS]
    print(f"{len(main_pos)} positions (+{len(positions) - len(main_pos)} extras) in {len({p['fi'] for p in positions})} files", flush=True)

    class StopOnNewline(StoppingCriteria):
        def __init__(self, start):
            self.start = start
        def __call__(self, input_ids, scores, **kw):
            new = input_ids[:, self.start:]
            done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            for b in range(input_ids.shape[0]):
                done[b] = any(int(t) in stop_ids for t in new[b].tolist())
            return done

    fmt = FIM[a.fmt]
    prompts = [build_prompt(p, fmt, heal, a.pre_chars, a.suf_chars) for p in positions]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    recs = [None] * len(prompts)
    t1 = time.time()
    with torch.no_grad():
        for k in range(0, len(order), a.batch):
            idx = order[k:k + a.batch]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
            L = enc.input_ids.shape[1]
            g = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, output_scores=True,
                               return_dict_in_generate=True, stopping_criteria=StoppingCriteriaList([StopOnNewline(L)]),
                               pad_token_id=tok.pad_token_id)
            seqs = g.sequences[:, L:]
            steps = len(g.scores)
            probs = torch.stack([torch.softmax(s.float(), -1).max(-1).values for s in g.scores], 1)   # [B, steps]
            for j, i in enumerate(idx):
                ids = seqs[j, :steps].tolist()
                n_gen, conf, kind = None, 1.0, "limit"
                for t, tid in enumerate(ids):
                    conf *= float(probs[j, t])
                    if tid in stop_ids:
                        n_gen, kind = t, ("newline" if tid in nl_ids else "special"); break
                if n_gen is None:
                    n_gen, conf = len(ids), 0.0
                text = tok.decode(ids[:n_gen + 1] if kind == "newline" else ids[:n_gen], skip_special_tokens=True)
                text = text.split("\n")[0].split("\r")[0]
                p = positions[i]
                heal_miss = False
                if heal and p["typed"]:
                    typed = p["typed"].decode("utf-8", "replace")
                    if text.startswith(typed):
                        text = text[len(typed):]
                    else:
                        heal_miss = True; text = ""
                r = score(p, text.encode("utf-8"), heal, heal_miss)
                r["conf_prod"] = conf; r["stop"] = kind
                recs[i] = r
            if (k // a.batch) % 20 == 0:
                print(f"  {k + len(idx)}/{len(prompts)} ({time.time() - t1:.0f} s)", flush=True)
    main_recs = [recs[i] for i, p in enumerate(positions) if p["kind"] not in E.EXTRA_KINDS]
    s = summarise(main_recs)
    report = {"model": a.model, "lang": a.lang, "manifest": manifest, "positions": len(main_recs), "seed": a.seed, "stride": a.stride,
              "heal": a.heal, "pre_chars": a.pre_chars, "suf_chars": a.suf_chars, "max_new": a.max_new,
              "gen_time_s": time.time() - t1, "summary": s, "records": main_recs}
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False)
    lines = [f"# {a.model} on {a.lang}: {len(main_recs)} positions (seed {a.seed}), heal={a.heal}, generation {report['gen_time_s']:.0f} s\n",
             "| metric | value |", "|---|---|",
             f"| rest of line exact, all positions [norm] | {100*s['line_exact']:.1f} [{100*s['line_exact_norm']:.1f}] % |",
             f"| rest of line exact, ≤ 8 tokens left ({s['n_le8']}) | {100*s['line_exact_le8']:.1f} % |",
             f"| first lexical token right | {s['first_lex_ok']:.3f} |",
             f"| ≥3 tokens right | {100*s['ok3']:.1f} % |",
             f"| healed positions / heal misses | {s['healed']} / {s['heal_miss']} |", "",
             "| conf_prod ≥ | shown % | line exact % | ≥3 right % |", "|---|---|---|---|"]
    for thr, t in s["thresholds"].items():
        lines.append(f"| {thr} | {100*t['shown']:.1f} | {100*t['prec_exact']:.1f} | {100*t['prec_ok3']:.1f} |")
    if a.dump:
        rng = np.random.default_rng(a.seed)
        lines += ["", "## sample"]
        for r in rng.choice(main_recs, min(a.dump, len(main_recs)), replace=False):
            lines.append(f"- {'OK ' if r['exact'] else 'NO '} {r['path']}:{r['line']} conf {r['conf_prod']:.2f}\n"
                         f"  - true: `{r['true']}`\n  - gen:  `{r['gen']}`")
    with open(out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:14]))
    print(f"wrote {out}.json / .md", flush=True)


if __name__ == "__main__":
    main()
