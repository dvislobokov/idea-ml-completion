#!/usr/bin/env python3
"""Summarises the GoMlDatasetExport worker logs (logs/export-w*-{rank,test}.log): lists, positions, recall, failures.
usage: export_summary.py logs/ [--md out.md]"""
import re, sys, glob, collections, os, gzip, struct

def main():
    dirs = [a for a in sys.argv[1:] if not a.startswith("--") and a != (sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None)]; d = dirs[0]; md = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
    rx = re.compile(r"ml: (?!TOTAL)(\S+) files=(\d+) positions=(\d+) lists=(\d+) answer-missing=(\d+) no-list=(\d+) single-insert=(\d+) recall=([\d.]+) (\d+) ms/position")
    L = []
    for fold in ["rank", "test"]:
        rows = []; failed = []; skipped = []
        for f in sorted(sum((glob.glob(os.path.join(dd, f"export-w*-{fold}.log")) for dd in dirs), [])):
            for line in open(f, errors="replace"):
                m = rx.search(line)
                if m: rows.append((m.group(1),) + tuple(float(x) for x in m.groups()[1:]))
                elif "ml: skip" in line: skipped.append(line.strip())
                elif "FAILED" in line or ("Exception" in line and " at " not in line and "WARN" not in line): failed.append(line.strip()[:200])
        if not rows: continue
        n = len(rows); s = lambda i: sum(r[i] for r in rows)
        files, pos, lists, miss, nolist, single, ms = s(1), s(2), s(3), s(4), s(5), s(6), sum(r[8] * r[2] for r in rows)
        zero = [r[0] for r in rows if r[3] == 0]
        L.append(f"## {fold}: {n} repositories, {int(files)} files, {int(pos)} positions, {int(lists)} lists")
        L.append(f"- plugin shows a list at {pos - nolist - single:.0f} positions ({100 * (pos - nolist - single) / pos:.1f} %); no list {100 * nolist / pos:.1f} %, single candidate inserted directly {100 * single / pos:.1f} %")
        L.append(f"- answer in the list (recall over positions with a list): {100 * lists / (pos - nolist - single):.1f} %; answer missing {int(miss)}")
        L.append(f"- {ms / pos:.0f} ms per position (export, incl. completion); repositories with 0 lists: {len(zero)} {zero[:10]}")
        if skipped: L.append(f"- skipped: {len(skipped)} ({skipped[0][:100]} ...)")
        if failed: L.append(f"- failures: {len(failed)}: " + " | ".join(failed[:5]))
        # candidates per list from the shards
        sd = os.path.join(os.path.dirname(d.rstrip('/')), "..", "..", "ml-data", "go", "psi", fold)
    text = "\n".join(L); print(text)
    if md: open(md, "w").write(text + "\n")
main()
