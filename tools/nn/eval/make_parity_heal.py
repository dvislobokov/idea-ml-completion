#!/usr/bin/env python
"""Parity fixture for the Kotlin `NnCompletion` (token healing + constrained decoding + show policy), test `NnHealParityTest`.

Positions come from `eval_inline.sample_positions` (same seed/stride as the eval) with the typed extras on; the fixture
takes positions whose typed remainder is non-empty (cursor inside a punctuation run), `typed-space` and `mid-ident`
extras, a few plain at-boundary positions, and the hand-made `trailing-ws` cases of healing v2 (`TRAILING_WS_CASES`:
`return ⟨⟩`, two spaces, a tab, `foo( ⟨⟩)`, indentation only, a word being typed). For each: the text before the caret
(from up to 60 KB before it — the eval's 40 KB cut then lands at the same line start in Kotlin), the text after the
caret (rest of the line + 16 KB), the healed prompt ids, and the fp32 fake-quantised greedy generation (batch 1, KV cache,
no autocast, TF32 off) with the typed-remainder constraint, the eval stop rule and the repetition guard; plus conf_prod /
conf_min / punct_only / show at threshold 0.8. Binary layout: big-endian, Java DataInputStream-compatible; header
`nn-heal-2` carries the heal-ws flag (`nn-heal-1` fixtures were made with the rule of 2026-10-07, the test reads both).

  ~/work/nn/.venv/bin/python -I make_parity_heal.py --ckpt ~/work/ml-data/go/nn/go31m-e2/ckpt-latest.pt \
      --out-dir ~/work/ml-data/go/nn/parity-heal          # then copy to data/go-nn-parity-heal of the repo
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import struct
import sys
import time

import numpy as np
import torch

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "work/nn/eval"))
sys.path.insert(0, os.path.join(HOME, "work/nn/train"))
import eval_inline as E  # noqa: E402
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("eval_make_parity", os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_parity.py"))
_mp = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mp)   # ~/work/nn/tokenizer has an unrelated make_parity.py
fwd, W = _mp.fwd, _mp.W
from model import CodeLM, ModelConfig  # noqa: E402
import export as X  # noqa: E402


CARET = b"<CARET>"

# Healing v2 cases (the examples of the rule, see eval_inline.heal_trailing_ws): small files with the caret marked; the fixture
# records them as kind `trailing-ws`, with what the Python side types. Expected typed remainders are checked at generation time.
TRAILING_WS_CASES = {
    "go": [
        ("main.go", b"package main\n\nfunc main() {\n\tx := 1\n\treturn <CARET>\n}\n", b" "),
        ("main.go", b"package main\n\nfunc f() int {\n\tx := <CARET>\n\treturn x\n}\n", b" "),
        ("main.go", b"package main\n\nfunc f() int {\n\tx := 1\n\tx =  <CARET>\n\treturn x\n}\n", b"  "),
        ("main.go", b"package main\n\nfunc f() {\n\tif true {\n\t\tfoo(\t<CARET>\n\t}\n}\n", b"\t"),
        ("main.go", b"package main\n\nfunc f() {\n\tfoo( <CARET>)\n}\n", b" "),
        ("main.go", b"package main\n\nfunc f() {\n\tx := 1\n\t<CARET>\n}\n", b""),
        ("main.go", b"package main\n\nfunc f() string {\n\treturn Hel<CARET>\n}\n", b" Hel"),
        ("main.go", b"package main\n\nfunc f() int {\n\tx := 1\n\treturn x<CARET>\n}\n", b" x"),
    ],
    "csharp": [
        ("Demo/Service.cs", b"namespace Demo;\n\npublic sealed class Service\n{\n    public int Count()\n    {\n        return <CARET>\n    }\n}\n", b" "),
        ("Demo/Service.cs", b"namespace Demo;\n\npublic sealed class Service\n{\n    public void Fail()\n    {\n        throw new <CARET>\n    }\n}\n", b" "),
        ("Demo/Service.cs", b"namespace Demo;\n\npublic sealed class Service\n{\n    public void Run()\n    {\n        var exception = <CARET>\n    }\n}\n", b" "),
        ("Demo/Service.cs", b"namespace Demo;\n\npublic sealed class Service\n{\n    public void Run()\n    {\n        var x = 1;\n        x =  <CARET>\n    }\n}\n", b"  "),
        ("Demo/Service.cs", b"namespace Demo;\n\npublic sealed class Service\n{\n    public void Run()\n    {\n        Foo(\t<CARET>\n    }\n}\n", b"\t"),
        ("Demo/Service.cs", b"namespace Demo;\n\npublic sealed class Service\n{\n    public void Run()\n    {\n        Foo( <CARET>)\n    }\n}\n", b" "),
        ("Demo/Service.cs", b"namespace Demo;\n\npublic sealed class Service\n{\n    public void Run()\n    {\n        var x = 1;\n        <CARET>\n    }\n}\n", b""),
        ("Demo/Service.cs", b"namespace Demo;\n\npublic sealed class Service\n{\n    public string Name()\n    {\n        return Hel<CARET>\n    }\n}\n", b" Hel"),
        ("Demo/Service.cs", b"namespace Demo;\n\npublic sealed class Service\n{\n    public int Count()\n    {\n        var x = 1;\n        return x<CARET>;\n    }\n}\n", b" x"),
        ("Grpc/Greeter.cs", b"using Grpc.Core;\n\nnamespace GrpcPlayground.Services;\n\npublic sealed class GreeterService : Greeter.GreeterBase\n{\n    public override Task<HelloReply> SayHello(HelloRequest request, ServerCallContext context) =>\n        Task.FromResult(<CARET>)\n}\n", b"("),
    ],
}


def synthetic_position(text, cursor, path, kind, fi=-1):
    """A position dict like `eval_inline.make_position` for a hand-made file: the line of the cursor, its rest as the truth."""
    bol = text.rfind(b"\n", 0, cursor) + 1
    eol, _ = E.line_end(text, cursor)
    rest = text[cursor:eol]
    true_code, true_toks = E.code_part(rest)
    boundary = E.pretoken_boundary(text, bol, cursor, eol)
    return {"fi": fi, "repo": "synthetic", "path": path, "cursor": cursor, "bol": bol, "eol": eol, "boundary": boundary,
            "typed": text[boundary:cursor], "line_no": text.count(b"\n", 0, cursor) + 1, "kind": kind, "is_test": False,
            "in_string": False, "line": text[bol:eol], "true_rest": rest, "true_code": true_code,
            "true_lex": E.lex_norm(true_toks, False), "true_lex_norm": E.lex_norm(true_toks, True), "text": text}


def trailing_ws_positions(lang, heal_ws):
    out = []
    for i, (path, marked, expect_typed) in enumerate(TRAILING_WS_CASES[lang]):
        cursor = marked.index(CARET)
        text = marked.replace(CARET, b"")
        p = synthetic_position(text, cursor, path, "trailing-ws", fi=-1 - i)
        if heal_ws and p["typed"] != expect_typed:
            raise AssertionError(f"{path} case {i}: typed {p['typed']!r}, expected {expect_typed!r}")
        out.append(p)
    return out


@torch.no_grad()
def greedy_heal(m, tok, prompt, typed, max_new, stop, dev, rep_guard=True):
    """Mirrors eval_inline.Generator.generate for one prompt with a typed-remainder constraint, in fp32."""
    kv = [None] * len(m.layers)
    logits = fwd(m, torch.tensor([prompt], dtype=torch.long, device=dev), 0, kv)
    gen, lps = [], []
    rem = typed
    stop_kind, stop_lp = "limit", None
    for step in range(max_new + 1):
        if rem:
            mask = tok.allowed_mask(rem, dev)
            logits = logits.masked_fill(~mask, float("-inf"))
        lp = torch.log_softmax(logits, dim=-1)
        nxt = int(torch.argmax(logits))
        if bool(stop[nxt]):
            stop_kind = "special" if nxt >= tok.special_base else "newline"
            stop_lp = float(lp[nxt])
            break
        if step == max_new:
            break
        gen.append(nxt); lps.append(float(lp[nxt]))
        if rem:
            rem = tok.consume(rem, nxt)
        if rep_guard and E.repetition(gen, tok.tok_len):
            stop_kind = "repeat"
            break
        logits = fwd(m, torch.tensor([[nxt]], dtype=torch.long, device=dev), len(prompt) + step, kv)
    return gen, lps, stop_kind, stop_lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lang", default="go", choices=sorted(E.LANGS))
    ap.add_argument("--heal", default="word-eol", choices=("boundary", "word", "word-eol"), help="healing boundary rule; must match NnCompletion.Options.healMode of the test (default WORD_EOL)")
    ap.add_argument("--heal-ws", dest="heal_ws", action="store_true", default=True, help="healing v2, the whitespace before the cursor is typed (default on; written into the fixture header, the test follows it)")
    ap.add_argument("--no-heal-ws", dest="heal_ws", action="store_false")
    ap.add_argument("--vocab")
    ap.add_argument("--out-dir")
    ap.add_argument("--positions", type=int, default=3000, help="eval sample the fixture positions are drawn from")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--ctx", type=int, default=2000)
    ap.add_argument("--max-prefix", type=int, default=1450)
    ap.add_argument("--suffix-tokens", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=48)
    a = ap.parse_args()
    E.LANG = E.LANGS[a.lang]
    E.HEAL_MODE = a.heal
    E.HEAL_WS = a.heal_ws
    a.vocab = a.vocab or os.path.join(E.DATA, E.LANG["vocab"])
    a.out_dir = a.out_dir or os.path.join(E.DATA, E.LANG["out_dir"], "parity-heal")
    manifest = os.path.join(E.DATA, E.LANG["manifest"]); repos = os.path.join(E.DATA, E.LANG["repos"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out_dir, exist_ok=True)

    tok = E.Tokenizer(a.vocab)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ck["config"])
    with_path = not ck["args"].get("no_path", False)
    m = CodeLM(cfg); m.load_state_dict(ck["model"]); m.float().to(dev).eval()
    X.fake_quantize_(m)
    step = int(ck["step"]); del ck
    V = cfg.vocab_size
    stop = torch.zeros(V, dtype=torch.bool)
    for i in range(tok.special_base):
        stop[i] = tok.vocab.tokens[i][:1] in (b"\n", b"\r")
    stop[tok.special_base:] = True
    stop = stop.to(dev)

    t0 = time.time()
    files = E.read_manifest(manifest)
    positions = E.sample_positions(files, repos, a.positions, a.stride, a.seed, False, typed_extras=True, max_typed_extras=400)
    rng = random.Random(a.seed + 5)
    main_typed = [p for p in positions if p["kind"] not in E.EXTRA_KINDS and p["typed"]]
    main_plain = [p for p in positions if p["kind"] not in E.EXTRA_KINDS and not p["typed"]]
    space = [p for p in positions if p["kind"] == "typed-space"]
    mid = [p for p in positions if p["kind"] == "mid-ident"]
    half = a.n // 2
    quarter = a.n // 4
    ws_cases = trailing_ws_positions(a.lang, a.heal_ws)
    sel = (rng.sample(main_typed, min(half, len(main_typed))) + rng.sample(space, min(quarter // 2, len(space))) +
           rng.sample(mid, min(quarter // 2, len(mid))) + rng.sample(main_plain, min(a.n // 10, len(main_plain))) + ws_cases)
    print(f"{len(positions)} positions ({len(main_typed)} typed in the main sample), fixture {len(sel)}: "
          f"{min(half, len(main_typed))} punct-run, {min(quarter // 2, len(space))} typed-space, {min(quarter // 2, len(mid))} mid-ident, "
          f"{min(a.n // 10, len(main_plain))} at-boundary, {len(ws_cases)} trailing-ws (heal-ws {'on' if a.heal_ws else 'off'}); {time.time() - t0:.0f} s", flush=True)

    w = W(os.path.join(a.out_dir, "heal.bin"))
    w.utf("nn-heal-2"); w.i32(V); w.i32(int(a.heal_ws)); w.i32(sum(1 + (k % 6 == 0) for k in range(len(sel))))
    t0 = time.time()
    stats = {"n": 0, "healed": 0, "exact": 0, "shown": 0, "shown_exact": 0, "repeat": 0, "limit": 0}
    kinds = {}
    for k, p in enumerate(sel):
        modes = ["spm"] + (["plain"] if k % 6 == 0 else [])
        text, cur = p["text"], p["cursor"]
        before = text[max(0, cur - 60000):cur]
        after = text[cur:p["eol"] + 16000]
        for mode in modes:
            prompt = E.build_prompt(tok, p, mode, a.ctx, a.suffix_tokens, with_path, a.max_prefix, heal=True)
            gen, lps, kind, stop_lp = greedy_heal(m, tok, prompt, p["typed"], a.max_new, stop, dev)
            raw = tok.decode(gen)
            typed = p["typed"]
            miss = bool(typed) and not raw.startswith(typed)
            gen_text = raw[len(typed):] if typed and not miss else raw
            allp = lps + ([stop_lp] if stop_lp is not None else [])
            conf_prod = math.exp(sum(allp)) if allp else 0.0
            conf_min = math.exp(min(allp)) if allp else 0.0
            punct = E.punct_only(gen_text)
            repeated = kind in ("repeat", "limit")
            show = conf_prod >= 0.8 and not punct and not repeated and len(gen_text) > 0
            gcode, _ = E.code_part(gen_text)
            exact = gcode == p["true_code"].rstrip() and len(gcode) > 0
            w.utf(f"{p['kind']}#{k}#{mode}"); w.utf(mode); w.utf(p["kind"])
            w.bytes(p["path"].encode()); w.bytes(before); w.bytes(after)
            w.i32(a.ctx); w.i32(a.max_prefix); w.i32(a.suffix_tokens); w.i32(a.max_new)
            w.bytes(typed); w.i32(len(prompt)); w.u16s(prompt)
            w.i32(len(gen)); w.u16s(gen); w.f32s(np.array(lps, dtype=np.float32))
            w.utf(kind); w.f32s(np.array([float("nan") if stop_lp is None else stop_lp], dtype=np.float32))
            w.f32s(np.array([conf_prod, conf_min], dtype=np.float32))
            w.i32(int(punct)); w.i32(int(repeated)); w.i32(int(show)); w.i32(int(miss)); w.i32(int(exact))
            w.bytes(gen_text); w.bytes(p["true_code"])
            stats["n"] += 1; stats["healed"] += bool(typed); stats["exact"] += exact; stats["shown"] += show
            stats["shown_exact"] += show and exact; stats["repeat"] += kind == "repeat"; stats["limit"] += kind == "limit"
            kk = kinds.setdefault(p["kind"], [0, 0]); kk[0] += 1; kk[1] += exact
        if k % 50 == 49:
            print(f"  {k + 1}/{len(sel)} ({time.time() - t0:.0f} s)", flush=True)
    w.close()
    print(f"records {stats['n']}: healed {stats['healed']}, line exact {stats['exact']}, shown@0.8 {stats['shown']} "
          f"(exact {stats['shown_exact']}), repeat stops {stats['repeat']}, limit {stats['limit']}; per kind n/exact: "
          + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in kinds.items()), flush=True)
    meta = {"ckpt": a.ckpt, "step": step, "lang": a.lang, "vocab": a.vocab, "heal": a.heal, "heal_ws": a.heal_ws,
            "vocabSha256": hashlib.sha256(open(a.vocab, "rb").read()).hexdigest(), "seed": a.seed, "stride": a.stride,
            "positions": a.positions, "n_records": stats["n"], "ctx": a.ctx, "max_prefix": a.max_prefix,
            "suffix_tokens": a.suffix_tokens, "max_new": a.max_new, "stats": stats, "kinds": kinds}
    json.dump(meta, open(os.path.join(a.out_dir, "meta.json"), "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
