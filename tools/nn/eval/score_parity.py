#!/usr/bin/env python
"""Scores the Kotlin generations of `NnParity` (kotlin-<kernels>-t<threads>.tsv in the fixture directory) with the
eval_inline metrics (`code_part` line-exact), next to the Python numbers on the same subset: the eval's bf16 batched
run (from the eval JSON, via behav.bin) and the fp32 fake-quantised reference run (behav.bin).

  ~/work/nn/.venv/bin/python -I score_parity.py [--dir ~/work/ml-data/go/nn/parity]
"""
import argparse
import base64
import glob
import os
import struct
import sys
from collections import defaultdict

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "work/nn/eval"))
import eval_inline as E  # noqa: E402


class R:
    def __init__(self, f):
        self.f = f

    def i32(self): return struct.unpack(">i", self.f.read(4))[0]
    def utf(self):
        n = struct.unpack(">H", self.f.read(2))[0]
        return self.f.read(n).decode("utf-8")
    def bytes(self): return self.f.read(self.i32())
    def u16s(self, n): return list(struct.unpack(">%dH" % n, self.f.read(2 * n)))
    def f32s(self, n): return list(struct.unpack(">%df" % n, self.f.read(4 * n)))


def read_behav(path):
    out = []
    with open(path, "rb") as f:
        r = R(f)
        assert r.utf() == "nn-behav-1"
        r.i32()
        for _ in range(r.i32()):
            pos = r.i32(); mode = r.utf(); kind = r.utf()
            ids = r.u16s(r.i32())
            true = r.bytes(); gen_eval = r.bytes(); stop_eval = r.utf()
            gen_fq = r.bytes(); r.u16s(r.i32()); stop_fq = r.utf(); r.f32s(r.i32())
            out.append(dict(pos=pos, mode=mode, kind=kind, n_ids=len(ids), true=true, gen_eval=gen_eval, stop_eval=stop_eval,
                            gen_fq=gen_fq, stop_fq=stop_fq))
    return out


def exact(gen: bytes, true: bytes):
    code, _ = E.code_part(gen)
    return code == true.rstrip() and len(code) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HOME, "work/ml-data/go/nn/parity"))
    a = ap.parse_args()
    recs = read_behav(os.path.join(a.dir, "behav.bin"))
    key = {(r["pos"], r["mode"]): r for r in recs}
    modes = sorted({r["mode"] for r in recs})
    print(f"{len(recs)} records ({', '.join(f'{m} {sum(1 for r in recs if r['mode'] == m)}' for m in modes)}), mean prompt {sum(r['n_ids'] for r in recs) / len(recs):.0f} tokens")
    print("\n| run | n | same line as fp32-fq | same line as eval bf16 | line exact vs truth | per mode (exact) |")
    print("|---|---|---|---|---|---|")
    def row(name, items):   # items: list of (rec, gen bytes, stop)
        n = len(items)
        sfq = sum(g == r["gen_fq"] for r, g, _ in items); sev = sum(g == r["gen_eval"] for r, g, _ in items)
        ex = sum(exact(g, r["true"]) for r, g, _ in items)
        per = []
        for m in modes:
            it = [(r, g) for r, g, _ in items if r["mode"] == m]
            per.append(f"{m} {100 * sum(exact(g, r['true']) for r, g in it) / max(1, len(it)):.1f} %")
        print(f"| {name} | {n} | {sfq} ({100 * sfq / n:.1f} %) | {sev} ({100 * sev / n:.1f} %) | {ex} ({100 * ex / n:.1f} %) | {', '.join(per)} |")
    row("python eval (bf16, batched, from JSON)", [(r, r["gen_eval"], r["stop_eval"]) for r in recs])
    row("python fp32 fake-quantised (reference)", [(r, r["gen_fq"], r["stop_fq"]) for r in recs])
    for tsv in sorted(glob.glob(os.path.join(a.dir, "kotlin-*.tsv"))):
        items = []
        with open(tsv, encoding="utf-8") as f:
            if not f.readline():
                continue      # still being written
            for line in f:
                pos, mode, stop, b64, _, ms = line.rstrip("\n").split("\t")
                r = key[(int(pos), mode)]
                items.append((r, base64.b64decode(b64), stop))
        row(os.path.basename(tsv)[len("kotlin-"):-4], items)
        # divergence analysis vs the fp32-fq reference
        diff = [(r, g, s) for r, g, s in items if g != r["gen_fq"]]
        if diff:
            kinds = defaultdict(int)
            for r, g, s in diff:
                k = "stop kind differs" if s != r["stop_fq"] else "first token differs" if (g[:1] != r["gen_fq"][:1]) else "later token differs"
                kinds[k] += 1
            print(f"    differences: " + ", ".join(f"{k} {v}" for k, v in kinds.items()) +
                  f"; exact-vs-truth among the differing lines: kotlin {sum(exact(g, r['true']) for r, g, _ in diff)}, reference {sum(exact(r['gen_fq'], r['true']) for r, _, _ in diff)}")


if __name__ == "__main__":
    main()
