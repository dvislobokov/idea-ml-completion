#!/usr/bin/env python3
"""Statistics over a PSI-context dump (GoMlContextExport JSONL): timings, coverage of the context, and the headroom table:
for the identifiers of the true rest-of-line, the share found (a) in the file prefix, (b) only in the PSI context, (c) nowhere.
usage: context_stats.py context.jsonl [--md out.md]"""
import json, sys, statistics as st, collections

def pct(a, b): return "%.1f %%" % (100.0 * a / b) if b else "-"
def q(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float('nan')

def context_names(r):
    names = set()
    qd = r.get("qualifier") or {}
    for m in qd.get("members") or []: names.add(m["n"])
    if qd.get("package"): names.add(qd["package"].rsplit("/", 1)[-1])
    sc = r.get("scope") or {}
    for d in sc.get("locals") or []: names.add(d["n"])
    for d in sc.get("pkg") or []: names.add(d["n"])
    for i in r.get("imports") or []: names.add(i.get("a") or i["p"].rsplit("/", 1)[-1])
    for x in r.get("xrefs") or []: names.add(x["n"])
    return names

def member_names(r):
    qd = r.get("qualifier") or {}
    return {m["n"] for m in qd.get("members") or []}

def main():
    path = sys.argv[1]; md = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    errs = [r for r in recs if "error" in r]; ok = [r for r in recs if "error" not in r and "ms" in r]
    L = []
    L.append(f"records {len(recs)}, ok {len(ok)}, errors {len(errs)}, repos {len({r['repo'] for r in recs})}, files {len({(r['repo'], r['path']) for r in recs})}")
    sizes = [len(json.dumps(r, ensure_ascii=False).encode()) for r in ok]
    L.append(f"record bytes: median {int(st.median(sizes))}, p90 {int(q(sizes, .9))}, max {max(sizes)}, >4096: {pct(sum(s > 4096 for s in sizes), len(sizes))}")
    kinds = collections.Counter(r["kind"] for r in ok)
    L.append("kinds: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()))
    if errs: L.append("errors: " + "; ".join(collections.Counter(e["error"][:80] for e in errs).most_common(5).__repr__().split("), (")[:5]))
    # timings
    L.append("\n| section | median ms | p90 ms | mean ms | max ms |\n|---|---|---|---|---|")
    for sec in ["func", "qual", "exp", "scope", "xref", "total"]:
        xs = [r["ms"][sec] for r in ok if sec in r["ms"]]
        if xs: L.append(f"| {sec} | {st.median(xs):.1f} | {q(xs, .9):.1f} | {st.mean(xs):.1f} | {max(xs):.0f} |")
    # coverage of the context
    dots = [r for r in ok if r["kind"] == "after-dot"]
    qd = [r for r in dots if r.get("qualifier")]
    typed = [r for r in qd if r["qualifier"].get("type")]; pk = [r for r in qd if r["qualifier"].get("package")]
    L.append(f"\nafter-dot positions {len(dots)}: qualifier found {pct(len(qd), len(dots))}, value with known type {pct(len(typed), len(dots))}, package {pct(len(pk), len(dots))}, "
             f"members median {st.median([r['qualifier']['total'] for r in qd]) if qd else '-'}")
    exp = [r for r in ok if r.get("expected")]
    L.append(f"expected type known: {pct(len(exp), len(ok))} of positions; enclosing function known: {pct(sum(1 for r in ok if r.get('func')), len(ok))}")
    loc = [len(r["scope"]["locals"]) for r in ok if r.get("scope")]; pkgn = [r["scope"]["pkg_total"] for r in ok if r.get("scope")]
    xr = [len(r.get("xrefs") or []) for r in ok]
    L.append(f"locals in scope: median {st.median(loc) if loc else '-'}, p90 {q(loc, .9) if loc else '-'}; package-level declarations: median {st.median(pkgn) if pkgn else '-'}, p90 {q(pkgn, .9) if pkgn else '-'}; "
             f"cross-file refs per function: median {st.median(xr)}, p90 {q(xr, .9)}, 0 refs: {pct(sum(1 for x in xr if x == 0), len(xr))}")
    # headroom table
    def table(rows, title, first_only=False):
        out = [f"\n{title}\n\n| positions | identifiers | (a) in file prefix | (b) only in PSI context | (c) nowhere |\n|---|---|---|---|---|"]
        for name, rs in rows:
            a = b = c = 0
            for r in rs:
                ids = r.get("rest_idents") or []
                if first_only: ids = ids[:1]
                names = context_names(r)
                for i in ids:
                    if i["p"]: a += 1
                    elif i["n"] in names: b += 1
                    else: c += 1
            n = a + b + c
            out.append(f"| {name} ({len(rs)}) | {n} | {pct(a, n)} | {pct(b, n)} | {pct(c, n)} |")
        return out
    rows = [("all", ok)] + [(k, [r for r in ok if r["kind"] == k]) for k, _ in kinds.most_common()]
    rows += [("non-test files", [r for r in ok if not r["is_test"]]), ("test files", [r for r in ok if r["is_test"]])]
    L += table(rows, "All identifiers of the true rest of line")
    L += table([("all", ok), ("after-dot", dots)], "First identifier of the rest of line only", first_only=True)
    # which part of the context supplies (b)
    src = collections.Counter()
    for r in ok:
        qn = member_names(r); sc = r.get("scope") or {}
        ln = {d["n"] for d in sc.get("locals") or []}; pn = {d["n"] for d in sc.get("pkg") or []}
        im = {i.get("a") or i["p"].rsplit("/", 1)[-1] for i in r.get("imports") or []}; xn = {x["n"] for x in r.get("xrefs") or []}
        for i in r.get("rest_idents") or []:
            if i["p"]: continue
            n = i["n"]
            if n in qn: src["qualifier members"] += 1
            elif n in ln: src["locals"] += 1
            elif n in pn: src["package level"] += 1
            elif n in im: src["imports"] += 1
            elif n in xn: src["cross-file refs"] += 1
            else: src["nowhere"] += 1
    tot = sum(src.values())
    L.append("\nSource of identifiers not in the prefix: " + ", ".join(f"{k} {pct(v, tot)}" for k, v in src.most_common()))
    text = "\n".join(L); print(text)
    if md: open(md, "w").write(text + "\n")

main()
