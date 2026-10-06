"""Tables from a scan_secrets.py output directory.

    python -I report.py out/go-lm            # markdown tables: categories, policy impact, top repos
    python -I report.py out/go-lm --samples 20 [--cat credential]   # masked random samples for the manual FP look
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import secret_filter as sf  # noqa: E402


def fmt(n):
    return f"{n:,}"


def tables(out):
    s = json.load(open(f"{out}/summary.json"))
    files, nbytes = s["files"], s["bytes"]
    print(f"### {s['lang']}/{s['fold']}: {fmt(files)} files, {nbytes/1e9:.2f} GB, {fmt(s['files_test'])} test-path files "
          f"({100*s['files_test']/files:.1f} %), scan {s['seconds']/60:.1f} min\n")
    print("| category | files (test / prod) | % files | matches (test / prod) | scrubbed? |")
    print("|---|---|---|---|---|")
    for c in sf.CATEGORIES:
        v = s["categories"][c]
        ft, fp = v["files_test"], v["files_prod"]
        mt, mp = v["matches_test"], v["matches_prod"]
        print(f"| {c} | {fmt(ft+fp)} ({fmt(ft)} / {fmt(fp)}) | {100*(ft+fp)/files:.3f} | {fmt(mt+mp)} ({fmt(mt)} / {fmt(mp)}) | "
              f"{'drop file' if c == 'private_key' else 'yes' if c in sf.SCRUBBED else 'report only'} |")
    p = s["policy"]
    print("\n#### Policy impact (DEFAULT_POLICY)\n")
    print("| effect | files | bytes | % of files | % of bytes |")
    print("|---|---|---|---|---|")
    for key, label in (("private_key", "dropped: private key block"), ("credential_hits", "dropped: >= 3 credential hits"),
                       ("data_file", "dropped: >= 100 scrubbable matches (data file)")):
        f, b = p.get(f"dropped_files_{key}", 0), p.get(f"dropped_bytes_{key}", 0)
        print(f"| {label} | {fmt(f)} ({fmt(p.get(f'dropped_files_{key}_test', 0))} test / {fmt(p.get(f'dropped_files_{key}_prod', 0))} prod) "
              f"| {b/1e6:.1f} MB | {100*f/files:.3f} | {100*b/nbytes:.3f} |")
    print(f"| scrubbed in place (files) | {fmt(p.get('scrubbed_files', 0))} ({fmt(p.get('scrubbed_files_test', 0))} test / "
          f"{fmt(p.get('scrubbed_files_prod', 0))} prod) | {p.get('scrubbed_bytes', 0)/1e6:.2f} MB replaced | "
          f"{100*p.get('scrubbed_files', 0)/files:.3f} | {100*p.get('scrubbed_bytes', 0)/nbytes:.4f} |")
    print(f"| scrubbed lines | {fmt(p.get('scrubbed_lines', 0))} | | | |")
    print(f"| alt: keep private-key files, replace PEM block | {fmt(p.get('alt_kept_pk_files', 0))} kept | "
          f"{p.get('alt_kept_pk_bytes_affected', 0)/1e6:.2f} MB replaced | | |")
    print("\nper-file scrubbable-match histogram (files / matches): " + ", ".join(
        f"{b}: {fmt(p.get('hist_files_' + b, 0))} / {fmt(p.get('hist_matches_' + b, 0))}" for b in ("1", "2-4", "5-9", "10-49", "50-99", "100+")))
    print("\nscrubbed matches by category: " + ", ".join(f"{k[len('scrubbed_matches_'):]}={fmt(v)}" for k, v in sorted(p.items()) if k.startswith("scrubbed_matches_")))
    print("\n#### Subtypes (top 40)\n")
    print("| subtype | matches |\n|---|---|")
    for k, v in list(s["subtypes"].items())[:40]:
        print(f"| {k} | {fmt(v)} |")
    print("\n#### e-mail domains (top 30, non-allowlisted)\n")
    print(", ".join(f"{d} {n}" for d, n in s["email_domains_top"][:30]))
    print("\n#### public IP /16 (top 20)\n")
    print(", ".join(f"{d}.x.x {n}" for d, n in s["ip_nets_top"][:20]))
    print("\n#### credential key names (top 30)\n")
    print(", ".join(f"{d} {n}" for d, n in s["credential_keys_top"][:30]))
    # top repos
    print("\n#### Top-20 repos by scrubbed-category matches\n")
    rows = []
    with open(f"{out}/repos.tsv") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            rows.append(line.rstrip("\n").split("\t"))
    cats = header[3:]
    print("| repo | scrubbed-cat matches | breakdown |")
    print("|---|---|---|")
    for r in rows[:20]:
        cn = {c: int(v) for c, v in zip(cats, r[3:]) if int(v) and c in sf.SCRUBBED}
        print(f"| {r[0]} | {fmt(int(r[2]))} | " + ", ".join(f"{c} {fmt(v)}" for c, v in sorted(cn.items(), key=lambda kv: -kv[1])) + " |")
    print(f"\nrepos with >= 1 match in any category: {fmt(s['repos_with_matches'])}")


def samples(out, n, cat, seed):
    rnd = random.Random(seed)
    for c in ([cat] if cat else sf.CATEGORIES):
        p = f"{out}/samples/{c}.jsonl"
        if not os.path.exists(p):
            continue
        recs = [json.loads(l) for l in open(p, encoding="utf-8")]
        rnd.shuffle(recs)
        print(f"\n=== {c} ({len(recs)} in reservoir)")
        for i, r in enumerate(recs[:n], 1):
            print(f"{i:2} {r['kind']:4} {r['sub']:14} {r['repo'][:34]:34} {r['path'][-44:]:44} L{r['line']:<6} | {r['snippet'][:150]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--cat")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    if a.samples:
        samples(a.out, a.samples, a.cat, a.seed)
    else:
        tables(a.out)
