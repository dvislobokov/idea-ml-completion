"""bytes/token of one or more vocab files on a manifest fold (random sample of --max-files files, fixed seed)."""
import argparse, multiprocessing as mp, random, sys
import cmlbpe
from manifest import read_manifest, read_file

_enc = None


def _init(path):
    global _enc
    _enc = cmlbpe.Encoder(cmlbpe.Vocab.load(path))


def _work(args):
    lang, batch = args
    nb = nt = 0
    for d in batch:
        data = read_file(lang, d)
        ids = _enc.encode_bytes(data)
        assert _enc.decode_bytes(ids) == data
        nb += len(data); nt += len(ids)
    return nb, nt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vocabs", nargs="+")
    ap.add_argument("--lang", default="go")
    ap.add_argument("--fold", default="test")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    files = list(read_manifest(a.lang, a.fold))
    random.Random(7).shuffle(files)
    if a.max_files: files = files[:a.max_files]
    print(f"{a.fold}: {len(files)} files, {sum(d['bytes'] for d in files)/1e6:.1f} MB, lexer tokens {sum(d['tokens'] for d in files)}")
    lex = sum(d["tokens"] for d in files)
    batches = [(a.lang, files[i:i + 50]) for i in range(0, len(files), 50)]
    for v in a.vocabs:
        with mp.Pool(a.workers, _init, (v,)) as pool:
            res = pool.map(_work, batches)
        nb = sum(r[0] for r in res); nt = sum(r[1] for r in res)
        print(f"{v}: {nb/nt:.3f} bytes/token, {nt} tokens, {nt/lex:.2f}x lexer tokens")


if __name__ == "__main__":
    main()
