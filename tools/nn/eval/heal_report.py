#!/usr/bin/env python
"""Before/after tables for token healing and the show policy from two eval_inline JSONs per language
(`--heal none` and `--heal boundary` runs on the same positions). Prints Markdown.

  ~/work/nn/.venv/bin/python -I heal_report.py --go-none ... --go-heal ... --cs-none ... --cs-heal ...
"""
import argparse
import json
import os
import sys

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "work/nn/eval"))
import eval_inline as ei  # noqa: E402

THR = (0.6, 0.7, 0.8, 0.9)


def pct(x):
    return "n/a" if x != x else f"{100 * x:.1f}"


def load(path):
    return json.load(open(path))


def positions(rep, mode):
    return rep["modes"][mode]["positions"]


def policy(recs, thr, punct, rep):
    n = len(recs)
    shown = [r for r in recs if r["conf_prod"] >= thr and not (punct and r["punct_only"]) and not (rep and r["repeated"])]
    useful = [r for r in shown if r["useful"]]
    prec = ei.rate([r["exact"] for r in shown])
    return len(shown) / n, prec, len(useful) / n, ei.rate([r["exact"] for r in useful])


def fix_legacy(recs):
    """Records of a --heal none / --no-rep-guard run carry the same fields (the harness always writes them)."""
    for r in recs:
        r.setdefault("punct_only", ei.punct_only(r["gen"].encode("utf-8", "replace")))
        r.setdefault("repeated", r["stop"] in ("repeat", "limit"))
        r.setdefault("useful", not r["punct_only"] and r["gen_len"] >= 2)
        r.setdefault("healed", False)


def headline(name, none, heal, mode):
    a = positions(none, mode); b = positions(heal, mode)
    fix_legacy(a); fix_legacy(b)
    assert len(a) == len(b)
    sa = none["modes"][mode]["summary"]; sb = heal["modes"][mode]["summary"]
    rows = [("line exact, all positions", pct(sa["line_exact"]), pct(sb["line_exact"])),
            ("line exact, ≤8 lexical tokens left", pct(sa["line_exact_le8"]), pct(sb["line_exact_le8"])),
            ("first lexical token right", f"{sa['first_lex_ok']:.3f}", f"{sb['first_lex_ok']:.3f}"),
            ("next BPE token top-1", f"{sa['bpe_top1']:.3f}", f"{sb['bpe_top1']:.3f}"),
            ("≥3 lexical tokens right", pct(sa["ok3"]), pct(sb["ok3"]))]
    for thr in (0.8, 0.7):
        pa = policy(a, thr, False, False); pb = policy(b, thr, False, False)
        rows.append((f"conf_prod ≥{thr}, no filters: shown / precision", f"{pct(pa[0])} / {pct(pa[1])}", f"{pct(pb[0])} / {pct(pb[1])}"))
        pa = policy(a, thr, True, True); pb = policy(b, thr, True, True)
        rows.append((f"conf_prod ≥{thr}, policy (no punct-only, rep guard): shown / precision", f"{pct(pa[0])} / {pct(pa[1])}", f"{pct(pb[0])} / {pct(pb[1])}"))
        rows.append((f"  of which useful (≥2 lexical tokens): shown / precision", f"{pct(pa[2])} / {pct(pa[3])}", f"{pct(pb[2])} / {pct(pb[3])}"))
    L = [f"### {name} {mode} ({len(a)} positions)\n", "| metric | heal none | heal boundary |", "|---|---|---|"]
    for r in rows:
        L.append(f"| {r[0]} | {r[1]} | {r[2]} |")
    return "\n".join(L) + "\n"


def by_kind(name, none, heal, mode):
    a = positions(none, mode); b = positions(heal, mode)
    groups = {}
    for x, y in zip(a, b):
        groups.setdefault(x["kind"], []).append((x, y))
    # the healed subset (cursor inside a pre-token) as an extra row
    healed = [(x, y) for x, y in zip(a, b) if y["healed"]]
    L = [f"### {name} {mode}: by position kind (line exact % none → boundary; policy@0.8 shown / precision)\n",
         "| kind | n | line exact none | boundary | first tok none | boundary | policy@0.8 none | boundary |", "|---|---|---|---|---|---|---|---|"]
    def row(k, g):
        xs = [x for x, _ in g]; ys = [y for _, y in g]
        pa = policy(xs, 0.8, True, True); pb = policy(ys, 0.8, True, True)
        return (f"| {k} | {len(g)} | {pct(ei.rate([x['exact'] for x in xs]))} | {pct(ei.rate([y['exact'] for y in ys]))} | "
                f"{pct(ei.rate([x['first_ok'] for x in xs]))} | {pct(ei.rate([y['first_ok'] for y in ys]))} | "
                f"{pct(pa[0])} / {pct(pa[1])} | {pct(pb[0])} / {pct(pb[1])} |")
    for k in sorted(groups, key=lambda k: -len(groups[k])):
        L.append(row(k, groups[k]))
    L.append(row("**cursor inside a pre-token (healed)**", healed))
    L.append(row("cursor at a pre-token boundary", [(x, y) for x, y in zip(a, b) if not y["healed"]]))
    # typed remainders
    typed = {}
    for x, y in healed:
        typed.setdefault(y["typed"], []).append((x, y))
    L.append("")
    L.append(f"Healed positions by typed remainder ({len(healed)} = {pct(len(healed) / len(a))} % of positions): " +
             ", ".join(f"`{repr(k)}` {len(v)} ({pct(ei.rate([x['exact'] for x, _ in v]))} → {pct(ei.rate([y['exact'] for _, y in v]))} %)"
                       for k, v in sorted(typed.items(), key=lambda kv: -len(kv[1]))[:12]) + ".\n")
    return "\n".join(L) + "\n"


def extras(name, heal, mode):
    ex = heal["modes"][mode].get("extra_positions", [])
    if not ex:
        return ""
    L = [f"### {name} {mode}: extra position kinds (healed; not in the totals)\n",
         "| kind | n | first tok | line exact | ≤8 exact | policy@0.8 shown / precision | policy@0.7 shown / precision |", "|---|---|---|---|---|---|---|"]
    for kind in ("typed-space", "mid-ident"):
        rs = [r for r in ex if r["kind"] == kind]
        if not rs:
            continue
        p8 = policy(rs, 0.8, True, True); p7 = policy(rs, 0.7, True, True)
        le8 = [r for r in rs if 1 <= r["rest_len"] <= 8]
        L.append(f"| {kind} | {len(rs)} | {pct(ei.rate([r['first_ok'] for r in rs]))} | {pct(ei.rate([r['exact'] for r in rs]))} | "
                 f"{pct(ei.rate([r['exact'] for r in le8]))} | {pct(p8[0])} / {pct(p8[1])} | {pct(p7[0])} / {pct(p7[1])} |")
    return "\n".join(L) + "\n"


def policy_table(name, heal, mode):
    recs = positions(heal, mode)
    L = [f"### {name} {mode}: show policy on the healed run (shown % / line exact % of shown / useful shown % (precision %))\n",
         "| conf_prod ≥ | none | no punct-only | rep guard | both |", "|---|---|---|---|---|"]
    for thr in THR:
        cells = []
        for punct, rep in ((False, False), (True, False), (False, True), (True, True)):
            s, p, u, pu = policy(recs, thr, punct, rep)
            cells.append(f"{pct(s)} / {pct(p)} / {pct(u)} ({pct(pu)})")
        L.append(f"| {thr} | " + " | ".join(cells) + " |")
    # what the filters remove
    n = len(recs)
    po = [r for r in recs if r["punct_only"]]; rp = [r for r in recs if r["repeated"]]
    L.append("")
    L.append(f"Punctuation-only suggestions: {pct(len(po) / n)} % of positions ({pct(ei.rate([r['exact'] for r in po]))} % right; "
             f"at ≥0.8: {pct(sum(1 for r in po if r['conf_prod'] >= 0.8) / n)} % of positions, "
             f"{pct(ei.rate([r['exact'] for r in po if r['conf_prod'] >= 0.8]))} % right). "
             f"Repetition guard / token limit: {pct(len(rp) / n)} % of positions ({sum(r['exact'] for r in rp)} right), "
             f"{sum(1 for r in rp if r['conf_prod'] >= 0.8)} of them at conf_prod ≥ 0.8.\n")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go-none"); ap.add_argument("--go-heal"); ap.add_argument("--cs-none"); ap.add_argument("--cs-heal")
    a = ap.parse_args()
    out = []
    for name, lang, pn, ph in (("Go go31m-e1", "go", a.go_none, a.go_heal), ("C# cs31m-e1", "csharp", a.cs_none, a.cs_heal)):
        if not pn:
            continue
        ei.LANG = ei.LANGS[lang]
        none = load(pn); heal = load(ph)
        out.append(f"## {name}\n")
        info = heal["info"]
        out.append(f"Runs: `{os.path.basename(pn)}` (heal none, no rep guard) and `{os.path.basename(ph)}` (heal boundary, rep guard); "
                   f"{info['n_positions']} positions, prompt ≤{info['ctx']} tokens (prefix ≤{info['max_prefix']}, suffix ≤{info['suffix_tokens']}), "
                   f"≤{info['max_new']} new tokens, bf16, batch {info['batch']}; extras {info.get('n_extra')}.\n")
        for mode in ("spm", "plain"):
            out.append(headline(name, none, heal, mode))
        for mode in ("spm",):
            out.append(by_kind(name, none, heal, mode))
            out.append(extras(name, heal, mode))
            out.append(policy_table(name, heal, mode))
    print("\n".join(out))


if __name__ == "__main__":
    main()
