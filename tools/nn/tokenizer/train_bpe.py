"""Byte-level BPE trainer (plain algorithm, deterministic).

1. sample files from the manifest (fold lm, status ok), stratified by repo (water-filling: every repo contributes at
   most `cap` bytes, `cap` chosen so the total is ~--sample-gb; files inside a repo picked at random with --seed);
2. pre-tokenize (cmlbpe.pretokenize) and count distinct pre-tokens (multiprocessing);
3. classic BPE on the weighted word table: repeatedly merge the most frequent adjacent pair, ties -> smallest
   (left_id, right_id); incremental pair counts + pair->words index + lazy max-heap.
Merge i creates token 256+i. Writes one vocab file per --vocab-size.
"""
import argparse, collections, heapq, multiprocessing as mp, os, random, sys, time
import cmlbpe
from manifest import read_manifest, read_file


def sample_files(lang, budget, max_file_bytes, seed):
    by_repo = collections.defaultdict(list)
    for d in read_manifest(lang, "lm"):
        if d["bytes"] <= max_file_bytes:
            by_repo[d["repo"]].append(d)
    tot = {r: sum(x["bytes"] for x in fs) for r, fs in by_repo.items()}
    lo, hi = 0, max(tot.values())
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(min(t, mid) for t in tot.values()) < budget: lo = mid + 1
        else: hi = mid
    cap = lo
    rnd = random.Random(seed)
    out = []
    for r in sorted(by_repo):
        fs = by_repo[r]; rnd.shuffle(fs); used = 0
        for d in fs:
            if used >= cap: break
            out.append(d); used += d["bytes"]
    print(f"repos {len(by_repo)}, per-repo cap {cap/1e6:.2f} MB, files {len(out)}, bytes {sum(d['bytes'] for d in out)/1e9:.3f} GB", flush=True)
    return out


def count_batch(args):
    lang, batch = args
    c = collections.Counter()
    for d in batch:
        try:
            c.update(cmlbpe.pretokenize(read_file(lang, d)))
        except OSError:
            pass
    return c


def count_words(lang, files, workers):
    batches = [(lang, files[i:i + 100]) for i in range(0, len(files), 100)]
    total = collections.Counter()
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for k, c in enumerate(pool.imap_unordered(count_batch, batches, chunksize=1)):
            total.update(c)
            if k % 100 == 0:
                print(f"  counted {k}/{len(batches)} batches, {len(total)} distinct, {time.time()-t0:.0f}s", flush=True)
    return total


def train(words, n_merges, log_every=500):
    """words: list of (bytes, count). Returns merges [(a, b)]."""
    W = [list(w) for w, _ in words]
    F = [c for _, c in words]
    counts = {}
    where = {}
    for wi, w in enumerate(W):
        f = F[wi]
        for i in range(len(w) - 1):
            k = (w[i] << 16) | w[i + 1]
            counts[k] = counts.get(k, 0) + f
            s = where.get(k)
            if s is None: where[k] = {wi}
            else: s.add(wi)
    heap = [(-c, k) for k, c in counts.items()]
    heapq.heapify(heap)
    merges = []
    t0 = time.time()
    while len(merges) < n_merges and heap:
        negc, k = heapq.heappop(heap)
        cur = counts.get(k, 0)
        if cur <= 0: continue
        if cur != -negc:
            heapq.heappush(heap, (-cur, k)); continue
        if cur < 2: break
        a, b = k >> 16, k & 0xFFFF
        new = 256 + len(merges)
        merges.append((a, b))
        fresh = set()
        for wi in where.pop(k):
            w = W[wi]; f = F[wi]; n = len(w)
            i = 0; out = []; changed = False
            while i < n:
                if i + 1 < n and w[i] == a and w[i + 1] == b:
                    changed = True
                    if out:
                        p = out[-1]
                        pk = (p << 16) | a
                        counts[pk] -= f
                        nk = (p << 16) | new
                        counts[nk] = counts.get(nk, 0) + f
                        s = where.get(nk)
                        if s is None: where[nk] = {wi}
                        else: s.add(wi)
                        fresh.add(nk)
                    if i + 2 < n:
                        q = w[i + 2]
                        qk = (b << 16) | q
                        counts[qk] -= f
                        nk = (new << 16) | q
                        counts[nk] = counts.get(nk, 0) + f
                        s = where.get(nk)
                        if s is None: where[nk] = {wi}
                        else: s.add(wi)
                        fresh.add(nk)
                    out.append(new); i += 2
                else:
                    out.append(w[i]); i += 1
            if changed: W[wi] = out
        counts.pop(k, None)
        for nk in fresh:
            c = counts.get(nk, 0)
            if c > 0: heapq.heappush(heap, (-c, nk))
        if len(merges) % log_every == 0:
            print(f"  merge {len(merges)} count {cur} {time.time()-t0:.0f}s", flush=True)
    return merges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="go")
    ap.add_argument("--sample-gb", type=float, default=1.5)
    ap.add_argument("--max-file-bytes", type=int, default=256 * 1024)
    ap.add_argument("--min-word-count", type=int, default=2)
    ap.add_argument("--vocab-sizes", default="16384,32768", help="total ids incl. 256 bytes and specials")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/work/ml-data/tokenizer"))
    ap.add_argument("--prefix", default="go")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    files = sample_files(a.lang, int(a.sample_gb * 1e9), a.max_file_bytes, a.seed)
    wc = count_words(a.lang, files, min(a.workers, 24))
    tot_bytes = sum(len(w) * c for w, c in wc.items())
    kept = sorted(((w, c) for w, c in wc.items() if c >= a.min_word_count), key=lambda x: (-x[1], x[0]))
    print(f"distinct pretokens {len(wc)}, kept {len(kept)} (count>={a.min_word_count}), "
          f"bytes covered {sum(len(w)*c for w,c in kept)/tot_bytes:.4f}", flush=True)
    del wc
    sizes = sorted(int(x) for x in a.vocab_sizes.split(","))
    nspec = len(cmlbpe.SPECIALS)
    merges = train(kept, sizes[-1] - 256 - nspec)
    for s in sizes:
        m = s - 256 - nspec
        v = cmlbpe.Vocab(merges[:m])  # a prefix of the merge list is exactly the smaller vocabulary
        p = f"{a.out_dir}/{a.prefix}-{s}.bpe"
        v.save(p); print("wrote", p, "merges", len(v.merges), "vocab", v.vocab_size, flush=True)


if __name__ == "__main__":
    main()
