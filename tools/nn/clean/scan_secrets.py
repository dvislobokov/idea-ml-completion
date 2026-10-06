"""Audit the prepared corpus for secrets / personal data and measure the scrub policy in the same pass.

    python -I scan_secrets.py --lang go --fold lm --workers 8 --out out/go-lm [--sample-frac 0.05] [--max-files N]

Streams the status=ok files of the fold through secret_filter.find_matches, aggregates per category
(files / matches, split into test-fixture vs production paths), per subtype, per repo, per e-mail domain and
IP /16, and applies secret_filter.DEFAULT_POLICY to measure files dropped / lines scrubbed / bytes affected.
Writes (nothing in here contains a full secret - every sample is masked at the worker):
    <out>/summary.json        all counters
    <out>/repos.tsv           per-repo match counts by category (sorted by total)
    <out>/samples/<cat>.jsonl uniform-ish random sample (reservoir, <= --keep per category) of masked records
    <out>/full/<cat>.jsonl    every record for the rare, high-signal categories (private_key, cloud_key, jwt, conn_string)
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import secret_filter as sf  # noqa: E402

DATA = os.path.expanduser("~/work/ml-data")
FULL_CATS = {"private_key", "cloud_key", "jwt", "conn_string"}
PER_BATCH_SAMPLE = 12
ALT_POLICY = dict(sf.DEFAULT_POLICY, drop_on_private_key=False)  # samples kept per category per batch before the parent's reservoir


def read_manifest(lang, fold, status="ok"):
    with open(f"{DATA}/{lang}/prepared/manifest.jsonl", "rb") as f:
        for line in f:
            if not line.endswith(b"\n"):
                break
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d["fold"] == fold and d["status"] == status:
                yield d


def _work(args):
    lang, batch = args
    rnd = random.Random(hash(batch[0]["path"]) & 0xFFFFFFFF)
    cat_files = Counter()          # (cat, kind) -> files
    cat_matches = Counter()        # (cat, kind) -> matches
    sub = Counter()                # (cat, subtype) -> matches
    repo = Counter()               # (repo, cat) -> matches
    domains = Counter()            # email domain -> matches
    ipnets = Counter()             # /16 -> matches
    cred_keys = Counter()          # credential key name (lower) -> matches
    pol = Counter()                # policy counters
    samples = defaultdict(list)    # cat -> [(seen_count, record)]
    full = defaultdict(list)
    seen = Counter()
    nfiles = nbytes = 0
    for d in batch:
        p = f"{DATA}/{lang}/repos/{d['repo']}/{d['path']}"
        try:
            with open(p, "rb") as f:
                data = f.read()
        except OSError:
            continue
        nfiles += 1
        nbytes += len(data)
        text = data.decode("utf-8", errors="replace")
        ms = sf.find_matches(text, d["path"])
        kind = "test" if sf.is_test_path(d["path"]) else "prod"
        pol["files"] += 1
        pol["bytes"] += len(data)
        if not ms:
            continue
        cats_here = set()
        for m in ms:
            cats_here.add(m.category)
            cat_matches[(m.category, kind)] += 1
            sub[(m.category, m.subtype)] += 1
            repo[(d["repo"], m.category)] += 1
            if m.category == "email":
                domains[m.extra.get("domain", "")] += 1
            elif m.category == "public_ip":
                ipnets[m.extra.get("net", "")] += 1
            elif m.category == "credential":
                cred_keys[m.extra.get("key", m.subtype).lower()] += 1
            seen[m.category] += 1
            rec = None
            if m.category in FULL_CATS or len(samples[m.category]) < PER_BATCH_SAMPLE or rnd.random() < PER_BATCH_SAMPLE / seen[m.category]:
                rec = {"cat": m.category, "sub": m.subtype, "repo": d["repo"], "path": d["path"], "line": m.line,
                       "kind": kind, "mask": sf.mask(m, text), "snippet": sf.masked_line(text, ms, m.line)}
                if m.extra.get("host"):
                    rec["host"] = m.extra["host"]
                if m.category in FULL_CATS:
                    full[m.category].append(rec)
                lst = samples[m.category]
                if len(lst) < PER_BATCH_SAMPLE:
                    lst.append(rec)
                else:
                    lst[rnd.randrange(PER_BATCH_SAMPLE)] = rec
        for c in cats_here:
            cat_files[(c, kind)] += 1
        # policy measurement
        clean, st = sf.scrub_matches(text, ms)
        if st["matches"].get("private_key"):
            alt, ast = sf.scrub_matches(text, ms, ALT_POLICY)
            if ast["dropped"]:
                pol["alt_dropped_files"] += 1
                pol["alt_dropped_bytes"] += len(data)
            else:
                pol["alt_kept_pk_files"] += 1
                pol["alt_kept_pk_bytes_affected"] += ast["bytes_affected"]
        nscrub = sum(1 for m in ms if m.category in sf.SCRUBBED)
        if nscrub:
            b = "1" if nscrub == 1 else "2-4" if nscrub < 5 else "5-9" if nscrub < 10 else "10-49" if nscrub < 50 else "50-99" if nscrub < 100 else "100+"
            pol["hist_files_" + b] += 1
            pol["hist_matches_" + b] += nscrub
        if st["dropped"]:
            pol["dropped_files_" + st["dropped"]] += 1
            pol["dropped_bytes_" + st["dropped"]] += len(data)
            pol["dropped_files_" + st["dropped"] + "_" + kind] += 1
        elif st["lines_scrubbed"]:
            pol["scrubbed_files"] += 1
            pol["scrubbed_files_" + kind] += 1
            pol["scrubbed_lines"] += st["lines_scrubbed"]
            pol["scrubbed_bytes"] += st["bytes_affected"]
            for c, n in st["matches"].items():
                if c in sf.SCRUBBED:
                    pol["scrubbed_matches_" + c] += n
    return dict(cat_files=cat_files, cat_matches=cat_matches, sub=sub, repo=repo, domains=domains, ipnets=ipnets,
                cred_keys=cred_keys, pol=pol, samples=dict(samples), full=dict(full), nfiles=nfiles, nbytes=nbytes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="go")
    ap.add_argument("--fold", default="lm")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--keep", type=int, default=3000, help="reservoir size per category")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    workers = min(a.workers, 8)
    rnd = random.Random(a.seed)
    files = list(read_manifest(a.lang, a.fold))
    if a.sample_frac < 1:
        files = [d for d in files if rnd.random() < a.sample_frac]
    if a.max_files:
        files = files[: a.max_files]
    total_bytes = sum(d["bytes"] for d in files)
    print(f"{a.lang}/{a.fold}: {len(files)} files, {total_bytes/1e9:.3f} GB, workers={workers}", flush=True)
    batches = [(a.lang, files[i:i + a.batch]) for i in range(0, len(files), a.batch)]
    os.makedirs(f"{a.out}/samples", exist_ok=True)
    os.makedirs(f"{a.out}/full", exist_ok=True)
    agg = {k: Counter() for k in ("cat_files", "cat_matches", "sub", "repo", "domains", "ipnets", "cred_keys", "pol")}
    reservoir = defaultdict(list)
    seen = Counter()
    full_files = {c: open(f"{a.out}/full/{c}.jsonl", "w", encoding="utf-8") for c in FULL_CATS}
    nfiles = nbytes = 0
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_work, batches, chunksize=1), 1):
            for key in agg:
                agg[key].update(r[key])
            nfiles += r["nfiles"]
            nbytes += r["nbytes"]
            for c, recs in r["samples"].items():
                for rec in recs:
                    seen[c] += 1
                    lst = reservoir[c]
                    if len(lst) < a.keep:
                        lst.append(rec)
                    else:
                        j = rnd.randrange(seen[c])
                        if j < a.keep:
                            lst[j] = rec
            for c, recs in r["full"].items():
                for rec in recs:
                    full_files[c].write(json.dumps(rec, ensure_ascii=False) + "\n")
            if k % 100 == 0 or k == len(batches):
                el = time.time() - t0
                print(f"  {k}/{len(batches)} batches, {nfiles} files, {nbytes/1e9:.2f} GB, {nbytes/el/1e6:.1f} MB/s, "
                      f"eta {(total_bytes-nbytes)/max(nbytes/el,1)/60:.1f} min", flush=True)
    for f in full_files.values():
        f.close()
    for c, lst in reservoir.items():
        with open(f"{a.out}/samples/{c}.jsonl", "w", encoding="utf-8") as f:
            for rec in lst:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # per-repo table
    per_repo = defaultdict(Counter)
    for (repo, c), n in agg["repo"].items():
        per_repo[repo][c] += n
    cats = sf.CATEGORIES
    with open(f"{a.out}/repos.tsv", "w") as f:
        f.write("repo\ttotal\tscrubbed_cats_total\t" + "\t".join(cats) + "\n")
        for repo, cn in sorted(per_repo.items(), key=lambda kv: -sum(v for c, v in kv[1].items() if c in sf.SCRUBBED)):
            tot = sum(cn.values())
            sc = sum(v for c, v in cn.items() if c in sf.SCRUBBED)
            f.write(f"{repo}\t{tot}\t{sc}\t" + "\t".join(str(cn.get(c, 0)) for c in cats) + "\n")
    summary = {
        "lang": a.lang, "fold": a.fold, "files": nfiles, "bytes": nbytes, "sample_frac": a.sample_frac,
        "seconds": round(time.time() - t0, 1),
        "categories": {c: {"files_test": agg["cat_files"][(c, "test")], "files_prod": agg["cat_files"][(c, "prod")],
                           "matches_test": agg["cat_matches"][(c, "test")], "matches_prod": agg["cat_matches"][(c, "prod")]}
                       for c in cats},
        "subtypes": {f"{c}/{s}": n for (c, s), n in sorted(agg["sub"].items(), key=lambda kv: -kv[1])},
        "email_domains_top": agg["domains"].most_common(60),
        "ip_nets_top": agg["ipnets"].most_common(40),
        "credential_keys_top": agg["cred_keys"].most_common(60),
        "policy": dict(agg["pol"]),
        "repos_with_matches": len(per_repo),
        "files_test": sum(1 for d in files if sf.is_test_path(d["path"])),
    }
    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=1)
    print(json.dumps(summary["categories"], indent=1))
    print("policy", json.dumps(summary["policy"], indent=1))
    print(f"done in {summary['seconds']}s")


if __name__ == "__main__":
    main()
