"""Encode all status=ok files of a manifest fold into uint16 shards.

Output (in --out-dir, default ~/work/ml-data/<lang>/bpe/<vocab-name>/):
  <fold>.tokens.u16   little-endian uint16 ids of all files, concatenated in manifest order (no separators inserted;
                      the model/data loader adds <|endoftext|> / <|file_sep|> where it wants)
  <fold>.offsets.u64  little-endian uint64, N+1 entries: file i = tokens[offsets[i]:offsets[i+1]]
  <fold>.repo.u32     little-endian uint32, N entries: repo id of file i (index into <fold>.repos.txt)
  <fold>.repos.txt    one repo name per line (id = line number)
  <fold>.files.txt    "repo<TAB>path" per file (same order as offsets)
  <fold>.meta.json    vocab file name/sha256, counts
--dry-run: no output; encodes a random --sample-frac of the files (default 2%) and extrapolates the totals.
At most 24 workers (the machine is shared).
"""
import argparse, hashlib, json, multiprocessing as mp, os, random, struct, sys, time
from array import array
assert sys.byteorder == 'little'
import cmlbpe
from manifest import read_manifest, read_file

_enc = None


def _init(path):
    global _enc
    _enc = cmlbpe.Encoder(cmlbpe.Vocab.load(path))


def _work(args):
    lang, batch = args
    lens = []
    arr = array("H")
    nbytes = 0
    for d in batch:
        try:
            data = read_file(lang, d)
        except OSError:
            data = b""
        ids = _enc.encode_bytes(data)
        nbytes += len(data)
        lens.append(len(ids))
        arr.extend(ids)
    return lens, arr.tobytes(), nbytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--lang", default="go")
    ap.add_argument("--fold", default="lm")
    ap.add_argument("--out-dir")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sample-frac", type=float, default=None, help="dry-run only; default 0.02")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-files", type=int, default=0, help="debug: only the first N files")
    a = ap.parse_args()
    workers = min(a.workers, 24)
    files = list(read_manifest(a.lang, a.fold))
    if a.max_files: files = files[:a.max_files]
    total_bytes = sum(d["bytes"] for d in files)
    lexer = sum(d["tokens"] for d in files)
    print(f"{a.lang}/{a.fold}: {len(files)} ok files, {total_bytes/1e9:.3f} GB, lexer tokens {lexer/1e9:.3f} G", flush=True)
    if a.dry_run:
        frac = a.sample_frac if a.sample_frac is not None else 0.02
        rnd = random.Random(3)
        files = [d for d in files if rnd.random() < frac] if frac < 1 else files
    else:
        assert a.sample_frac is None, "--sample-frac is for --dry-run"
    batches = [(a.lang, files[i:i + a.batch]) for i in range(0, len(files), a.batch)]
    name = os.path.splitext(os.path.basename(a.vocab))[0]
    out = None
    if not a.dry_run:
        d = a.out_dir or os.path.expanduser(f"~/work/ml-data/{a.lang}/bpe/{name}")
        os.makedirs(d, exist_ok=True)
        pre = f"{d}/{a.fold}"
        ftok = open(pre + ".tokens.u16", "wb")
        fidx = open(pre + ".files.txt", "w", encoding="utf-8", newline="\n")
        offsets = [0]
        repos = {}
        repo_ids = []
    ntok = nbytes = 0
    t0 = time.time()
    with mp.Pool(workers, _init, (a.vocab,)) as pool:
        k = 0
        for lens, raw, nb in pool.imap(_work, batches):
            ntok += sum(lens); nbytes += nb
            if not a.dry_run:
                ftok.write(raw)
                for dd, n in zip(files[k * a.batch:(k + 1) * a.batch], lens):
                    offsets.append(offsets[-1] + n)
                    repo_ids.append(repos.setdefault(dd["repo"], len(repos)))
                    fidx.write(f"{dd['repo']}\t{dd['path']}\n")
            k += 1
            if k % 200 == 0:
                print(f"  {k}/{len(batches)} batches, {ntok/1e6:.1f} M tokens, {nbytes/(time.time()-t0)/1e6:.1f} MB/s", flush=True)
    bpt = nbytes / max(ntok, 1)
    print(f"encoded {len(files)} files, {nbytes/1e6:.1f} MB -> {ntok} tokens ({bpt:.3f} bytes/token, "
          f"{ntok/max(sum(x['tokens'] for x in files),1):.2f}x lexer) in {time.time()-t0:.0f}s")
    if a.dry_run:
        est_tokens = ntok / nbytes * total_bytes
        print(f"ESTIMATE for the whole fold: {est_tokens/1e9:.3f} G BPE tokens ({est_tokens/lexer:.2f}x lexer count {lexer/1e9:.3f} G)")
        return
    ftok.close(); fidx.close()
    array("Q", offsets).tofile(open(pre + ".offsets.u64", "wb"))
    array("I", repo_ids).tofile(open(pre + ".repo.u32", "wb"))
    with open(pre + ".repos.txt", "w", encoding="utf-8", newline="\n") as f:
        for r in repos: f.write(r + "\n")
    meta = {"vocab": os.path.basename(a.vocab), "vocab_sha256": hashlib.sha256(open(a.vocab, "rb").read()).hexdigest(),
            "lang": a.lang, "fold": a.fold, "files": len(files), "repos": len(repos), "tokens": ntok, "bytes": nbytes}
    json.dump(meta, open(pre + ".meta.json", "w"), indent=1)
    print("wrote", pre + ".*", meta)


if __name__ == "__main__":
    main()
