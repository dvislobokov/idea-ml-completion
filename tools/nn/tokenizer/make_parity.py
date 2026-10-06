"""Dump a parity fixture for the Kotlin tokenizer test.

Output dir gets `vocab.bpe` (copy) and `parity.bin.gz`:
    "CMLPAR1\\n", then gzip-compressed big-endian records:  int32 N; N x { int32 nbytes, bytes, int32 nids, nids x uint16 }
Cases = synthetic edge cases + random fuzz strings + files of a manifest fold (self-contained: the input bytes are
stored, so the Kotlin test does not need the corpus).
"""
import argparse, gzip, multiprocessing as mp, os, random, shutil, struct
import cmlbpe
from manifest import read_manifest, read_file

_enc = None


def _init(path):
    global _enc
    _enc = cmlbpe.Encoder(cmlbpe.Vocab.load(path))


def _read_encode(args):
    lang, d = args
    data = read_file(lang, d)
    return data, _enc.encode_bytes(data)


def edge_cases():
    c = [b"", b" ", b"\t", b"\n", b"\r", b"\r\n", b"\n\r", b"\r\r\n\n", b"   ", b" \t \t ", b"\n\n\n\t\t\t", b"  \n  \n  ",
         b"a", b"_", b"0", b"-", b"\x00", b"\x00\x01\x02\x7f", b"\x0b\x0c",
         b"x := 1\r\ny := 2\r\n", b"\xef\xbb\xbfpackage main\n", b"package main\n\nimport \"fmt\"\n\nfunc main() {\n\tfmt.Println(\"hello\")\n}\n",
         "func привет(über int) { /* 日本語 */ }\n".encode(),
         "log.Printf(\"started \U0001F680 ✓ %s\", name) // \U0001F468‍\U0001F469‍\U0001F467\n".encode(),
         b"\xff\xfe\xfd", b"abc\xc3", b"\xc3\x28", b"\xe2\x82", b"\xf0\x9f\x98", b"a\x80b\xbfc", b"\xed\xa0\x80 surrogate",
         b"<|endoftext|><|fim_prefix|> <|fim_middle|>", b"x" * 127, b"x" * 128, b"x" * 129, b"x" * 300,
         b" " * 127 + b"x", b" " * 128 + b"x", b" " * 129 + b"x", b" " * 500, b"\t" * 130 + b"y", b" x" + b"y" * 200,
         b"\n" * 200, b"\r\n" * 150, b"(" * 300, b"=" * 129 + b"a",
         b"0123456789" * 40, b"a1b2c3", b"x1", b"1x", b"int32 uint8 sha256 utf8 float64", b"fooBarBaz_qux HTTPServer parseURL",
         b"if err != nil {\n\t\treturn nil, fmt.Errorf(\"open %q: %w\", path, err)\n\t}\n",
         b"a  b   c\t\td \te\t f", b"x \n y \r\n z \t\n", b"\t \tfoo", b"  \tfoo", b"\t  foo", b"foo  ", b"foo\t", b"foo \t",
         b"//" + b"=" * 70, b"\"" * 5 + b"'" * 5 + b"`" * 5, b"a" * 5000, (b"word, " * 4000), b"A" * 10 + b"\n" + b"\xc3\xa9" * 200,
         ("x = \"" + "QUJD" * 1000 + "\"\n").encode(), b"\n" + b" " * 8 + b"foo", b"\n" + b"\t" * 3 + b" foo", b"  foo", b"\tfoo"]
    r = random.Random(12345)
    alpha = [b"a", b"B", b"_", b" ", b"  ", b"\t", b"\n", b"\r", b"\r\n", b"0", b"7", b"(", b")", b"\"", b":=", b"//", b"\xc3\xa9",
             b"\xf0\x9f\x98\x80", b"\x80", b"\xff", b"\x00", b"Foo", b"bar", b"for", b"err", b"nil", b"."]
    for _ in range(400):
        c.append(b"".join(r.choice(alpha) for _ in range(r.randint(1, 60))))
    for _ in range(100):
        c.append(bytes(r.randrange(256) for _ in range(r.randint(1, 400))))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lang", default="go")
    ap.add_argument("--fold", default="test")
    ap.add_argument("--max-files", type=int, default=10000)
    ap.add_argument("--max-file-bytes", type=int, default=1 << 20)
    ap.add_argument("--min-file-bytes", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    shutil.copy(a.vocab, f"{a.out_dir}/vocab.bpe")
    files = [d for d in read_manifest(a.lang, a.fold) if a.min_file_bytes <= d["bytes"] <= a.max_file_bytes]
    random.Random(99).shuffle(files)
    files = files[:a.max_files]
    enc = cmlbpe.Encoder(cmlbpe.Vocab.load(a.vocab))
    recs = []
    for data in edge_cases():
        ids = enc.encode_bytes(data)
        assert enc.decode_bytes(ids) == data
        recs.append((data, ids))
    n_edge = len(recs)
    with mp.Pool(a.workers, _init, (a.vocab,)) as pool:
        for data, ids in pool.imap(_read_encode, [(a.lang, d) for d in files], chunksize=16):
            recs.append((data, ids))
    nb = sum(len(r[0]) for r in recs); nt = sum(len(r[1]) for r in recs)
    with open(f"{a.out_dir}/parity.bin.gz", "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as f:
            f.write(b"CMLPAR1\n")
            f.write(struct.pack(">i", len(recs)))
            for data, ids in recs:
                f.write(struct.pack(">i", len(data))); f.write(data)
                f.write(struct.pack(">i", len(ids)))
                f.write(struct.pack(f">{len(ids)}H", *ids))
    print(f"{len(recs)} cases ({n_edge} synthetic, {len(recs)-n_edge} files), {nb} bytes, {nt} ids -> {a.out_dir}/parity.bin.gz "
          f"({os.path.getsize(a.out_dir + '/parity.bin.gz')/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
