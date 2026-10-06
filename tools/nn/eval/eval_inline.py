#!/usr/bin/env python
"""Inline ("ghost text") evaluation of our neural code model, mirroring `ml-train eval-inline` (InlineEval.kt).

Protocol (n-gram baseline, e14-b): every `stride`-th lexer token of the held-out files is a position; the model greedily
continues; we measure how many tokens are right, whether the rest of the line is exact, and precision/coverage at
confidence thresholds (confidence = geometric mean of the first 3 token probabilities).

Here: same kind of positions (sampled, deterministic), prompts built exactly like the training documents
(`<|file_sep|> path\n body`, PSM fill-in-the-middle with the rest of the file as suffix), greedy decode until the first
newline token, and metrics computed at the CHARACTER / lexical-token level so that BPE vs. n-gram tokenisation does
not matter: a small Go-ish splitter (identifiers, numbers, strings, punctuation) is applied to the generated line and
to the true rest of the line (trailing comments dropped). "Normalised" variants map strings/chars/numbers to
<STR>/<CHAR>/<NUM> like the n-gram lexer does.

Usage (see --help):
  ~/work/nn/.venv/bin/python -I eval_inline.py --ckpt ~/work/ml-data/go/nn/go31m-e1/ckpt-latest.pt \
      --positions 3000 --modes plain,fim --dump 60
  ~/work/nn/.venv/bin/python -I eval_inline.py --lang csharp --ckpt ~/work/ml-data/csharp/nn/cs31m-e1/ckpt-latest.pt ...
Outputs <out>.json, <out>.md, <out>-dump.md (default out = ~/work/ml-data/<lang>/nn/eval-inline-step<N>).
`--lang csharp` switches to the C# manifest/repos/vocab, a C# lexical splitter (identifiers incl. `@ident`, numbers,
regular/verbatim/interpolated/raw strings as single tokens, `=>` `?.` `??=` …), the e15-a n-gram reference, and the
in-string detection for `"…"`, `@"…"`, `$"…"`. Lines are cut before a CR so CRLF files feed the model their own
`\r\n…` suffix token; generation stops at any token starting with CR or LF.
"""
import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "work/nn/tokenizer"))
sys.path.insert(0, os.path.join(HOME, "work/nn/train"))
import cmlbpe  # noqa: E402
from model import CodeLM, ModelConfig  # noqa: E402

DATA = os.path.join(HOME, "work/ml-data")

# n-gram references: Go e14-b (CHANGELOG: order 5, cache λ=0.3, test fold 300 repos, 46 271 positions),
# C# e15-a (order 5, cache λ=0.3, test fold 290 repos, 1 980 files / 27 408 positions)
NGRAM_E14B = {"name": "e14-b", "positions": 46271, "shown_0.8": 10.0, "ok3_0.8": 92.9, "tokprec_0.8": 0.956,
              "shown_0.7": 16.5, "ok3_0.7": 83.3, "line_exact": 35.1, "first_token": 0.652}
NGRAM_E15A = {"name": "e15-a", "positions": 27408, "shown_0.8": 3.1, "ok3_0.8": 87.9,
              "shown_0.7": 6.6, "ok3_0.7": 76.6, "line_exact": 31.5, "first_token": 0.570}

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)

# ----------------------------------------------------------------------------------------------------- lexers

GO_KEYWORDS = {b"break", b"case", b"chan", b"const", b"continue", b"default", b"defer", b"else", b"fallthrough",
               b"for", b"func", b"go", b"goto", b"if", b"import", b"interface", b"map", b"package", b"range",
               b"return", b"select", b"struct", b"switch", b"type", b"var"}

_GO_LEX = re.compile(
    rb'(?P<ws>[ \t\r\n\f\v]+)'
    rb'|(?P<comment>//[^\n]*|/\*(?:.*?\*/|.*))'
    rb'|(?P<string>"(?:\\.|[^"\\\n])*"?|`[^`]*`?)'
    rb"|(?P<char>'(?:\\.|[^'\\\n])*'?)"
    rb'|(?P<ident>[A-Za-z_\x80-\xff][A-Za-z0-9_\x80-\xff]*)'
    rb'|(?P<number>\.?[0-9](?:[eEpP][+-]|[0-9A-Za-z_.])*)'
    rb'|(?P<punct><<=|>>=|&\^=|\.\.\.|&&|\|\||<-|\+\+|--|==|!=|<=|>=|:=|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<|>>|&\^'
    rb'|[-+*/%&|^<>=!(){}\[\],;.:~])'
    rb'|(?P<other>.)', re.DOTALL)

# C# keywords (reserved + the contextual ones that behave like keywords at the cursor: var, async, await, ...)
CS_KEYWORDS = set(w.encode() for w in (
    "abstract as base bool break byte case catch char checked class const continue decimal default delegate do double "
    "else enum event explicit extern false finally fixed float for foreach goto if implicit in int interface internal is "
    "lock long namespace new null object operator out override params private protected public readonly ref return sbyte "
    "sealed short sizeof stackalloc static string struct switch this throw true try typeof uint ulong unchecked unsafe "
    "ushort using virtual void volatile while var async await yield get set init partial record nameof when").split())

# strings: raw ("""…""", C# 11), verbatim (@"…" / $@"…" / @$"…", "" escapes, may span lines), regular / interpolated ($"…");
# identifiers may carry the @ verbatim prefix; punctuation covers =>, ?., ??, ??=, ::, ->, <<=, >>=, >>>=
_CS_LEX = re.compile(
    rb'(?P<ws>[ \t\r\n\f\v]+)'
    rb'|(?P<comment>//[^\n]*|/\*(?:.*?\*/|.*))'
    rb'|(?P<string>(?:\$@|@\$|@|\$)?"""(?:(?!""").)*(?:"""|$)'
    rb'|(?:\$@|@\$|@)"(?:[^"]|"")*"?'
    rb'|\$?"(?:\\.|[^"\\\n])*"?)'
    rb"|(?P<char>'(?:\\.|[^'\\\n])*'?)"
    rb'|(?P<ident>@?[A-Za-z_\x80-\xff][A-Za-z0-9_\x80-\xff]*)'
    rb'|(?P<number>\.?[0-9](?:[eEpP][+-]|[0-9A-Za-z_.])*)'
    rb'|(?P<punct>>>>=|<<=|>>=|>>>|\?\?=|\?\?|\?\.|=>|->|::|\+\+|--|&&|\|\||==|!=|<=|>=|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<|>>|\.\.'
    rb'|[-+*/%&|^<>=!(){}\[\],;.:~?#])'
    rb'|(?P<other>.)', re.DOTALL)


def _is_test_go(path: str):
    return path.endswith("_test.go")


_CS_TEST = re.compile(r"(?:(?<![A-Za-z])(?:tests?|specs?)|(?<![A-Z])(?:Tests?|Specs?))(?![a-z])")


def _is_test_cs(path: str):
    """Test file heuristic for C#: a word "test(s)"/"spec(s)" in the path (`Foo.Tests/`, `BarTests.cs`, `test/`, `UnitTests`),
    not `Contest` or `Testing`."""
    return _CS_TEST.search(path) is not None


LANGS = {
    "go": {"lex": _GO_LEX, "keywords": GO_KEYWORDS, "ext": "go", "fence": "go", "is_test": _is_test_go,
           "manifest": "go/prepared/manifest.jsonl", "repos": "go/repos", "vocab": "tokenizer/go-16384.bpe",
           "out_dir": "go/nn", "ngram": NGRAM_E14B, "assign": (b"=", b":="), "dot": (b".",)},
    "csharp": {"lex": _CS_LEX, "keywords": CS_KEYWORDS, "ext": "cs", "fence": "csharp", "is_test": _is_test_cs,
               "manifest": "csharp/prepared/manifest.jsonl", "repos": "csharp/repos", "vocab": "tokenizer/cs-16384.bpe",
               "out_dir": "csharp/nn", "ngram": NGRAM_E15A, "assign": (b"=",), "dot": (b".", b"?.")},
}
LANG = LANGS["go"]          # set by main(); module-level so that the lexer helpers stay simple functions


class Tok:
    __slots__ = ("kind", "start", "end", "text")

    def __init__(self, kind, start, end, text):
        self.kind, self.start, self.end, self.text = kind, start, end, text


def lex(data: bytes, with_comments=True):
    """Tokens (kind, byte offsets, bytes) of the current language; whitespace dropped, comments optional."""
    out = []
    kw = LANG["keywords"]
    for m in LANG["lex"].finditer(data):
        kind = m.lastgroup
        if kind == "ws" or (kind == "comment" and not with_comments):
            continue
        if kind == "ident" and m.group() in kw:
            kind = "keyword"
        out.append(Tok(kind, m.start(), m.end(), m.group()))
    return out


go_lex = lex    # backwards-compatible name


def string_open_len(t: Tok):
    """Bytes of the opening delimiter of a string token (`"`, `$"`, `@"`, `$@"`, …; 3 quotes for raw strings)."""
    q = t.text.find(b'"')
    if q < 0:
        return 1
    return q + 3 if t.text[q:q + 3] == b'"""' else q + 1


def lex_norm(toks, normalise):
    """Lexical token strings for comparison; normalise -> <STR>/<CHAR>/<NUM> as the n-gram lexer does."""
    res = []
    for t in toks:
        if normalise and t.kind == "string":
            res.append(b"<STR>")
        elif normalise and t.kind == "char":
            res.append(b"<CHAR>")
        elif normalise and t.kind == "number":
            res.append(b"<NUM>")
        else:
            res.append(t.text)
    return res


def code_part(line: bytes):
    """The line without a trailing comment (and without a comment-only tail), right-stripped."""
    toks = lex(line, with_comments=True)
    code = [t for t in toks if t.kind != "comment"]
    if not code:
        return b"", []
    return line[:code[-1].end].rstrip(), code


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# ----------------------------------------------------------------------------------------------------- positions

def read_manifest(path, fold="test", status="ok"):
    out = []
    with open(path, "rb") as f:
        for line in f:
            if not line.endswith(b"\n"):
                break
            d = json.loads(line)
            if d["fold"] == fold and d["status"] == status:
                out.append(d)
    return out


def sample_positions(files, repos_root, n_positions, stride, seed, in_string_extra):
    """(file index, token index) pairs: uniform over the multiset of every stride-th lexer token of the fold, like
    InlineEval's `i % stride == 0` over all files (the manifest token counts come from the Kotlin lexer, close enough)."""
    rng = random.Random(seed)
    cand = [(fi, k * stride) for fi, d in enumerate(files) for k in range(1, d["tokens"] // stride + 1)]
    picks = rng.sample(cand, min(n_positions, len(cand)))
    by_file = defaultdict(list)
    for fi, i in picks:
        by_file[fi].append(i)
    positions = []
    for fi in sorted(by_file):
        d = files[fi]
        fpath = os.path.join(repos_root, d["repo"], d["path"])
        try:
            with open(fpath, "rb") as f:
                text = f.read()
        except OSError:
            continue
        toks = lex(text, with_comments=True)
        code_idx = [j for j, t in enumerate(toks) if t.kind != "comment"]
        if len(code_idx) < 2:
            continue
        for i in sorted(set(by_file[fi])):
            i = min(i, len(code_idx) - 1)
            p = make_position(text, toks, code_idx[i], d, fi)
            if p is not None:
                positions.append(p)
                if in_string_extra:
                    t = toks[code_idx[i]]
                    if t.kind == "string" and t.end - t.start >= 6 and b"\n" not in t.text and not t.text.startswith(b'"""'):
                        q = make_position(text, toks, code_idx[i], d, fi, cursor=t.start + string_open_len(t), kind="in-string")
                        if q is not None:
                            positions.append(q)
    return positions


def line_end(text, start):
    """(eol, eol_nl): byte offset where the line content ends (before a CR of a CRLF pair) and where the LF is.
    The suffix of a FIM prompt starts at `eol`, so a CRLF file hands the model its own `\\r\\n…` token."""
    eol_nl = text.find(b"\n", start)
    if eol_nl < 0:
        eol_nl = len(text)
    eol = eol_nl - 1 if eol_nl > start and text[eol_nl - 1:eol_nl] == b"\r" else eol_nl
    return eol, eol_nl


def make_position(text, toks, j, d, fi, cursor=None, kind=None):
    """Position before token j (index into the full token list). The cursor goes after the previous token of the same
    line (clean BPE pre-token boundary) or after the indentation when the token starts its line."""
    t = toks[j]
    bol = text.rfind(b"\n", 0, t.start) + 1
    eol, _ = line_end(text, t.start)
    prev = toks[j - 1] if j > 0 else None
    if cursor is None:
        if prev is None or prev.end < bol:
            cursor = t.start          # line start: prefix ends with "\n" + indentation
            kind = "line-start"
        else:
            cursor = prev.end
            pt = prev.text
            if prev.kind == "comment":
                kind = "after-comment"
            elif pt in LANG["dot"]:
                kind = "after-dot"
            elif pt in (b"(", b"[", b"{"):
                kind = "after-open"
            elif pt == b",":
                kind = "after-comma"
            elif pt in LANG["assign"]:
                kind = "after-assign"
            elif prev.kind == "keyword":
                kind = "after-keyword"
            elif prev.kind == "ident":
                kind = "after-ident"
            else:
                kind = "after-other"
    line = text[bol:eol]
    rest = text[cursor:eol]
    true_code, true_toks = code_part(rest)
    if not true_toks and kind != "in-string":
        return None
    if LANG["ext"] == "go":
        # inside-string heuristic from the prefix: odd number of unescaped quotes on the current line, or an open backtick
        lp = text[bol:cursor]
        dq = len(re.findall(rb'(?<!\\)"', lp))
        bt = text.count(b"`", 0, cursor)
        in_string = (dq % 2 == 1) or (bt % 2 == 1)
    else:
        # the lexer knows the literals ("…", @"…", $"…", """…"""): the cursor is inside one iff it is strictly within a string token
        in_string = any(x.kind == "string" and x.start < cursor < x.end for x in toks[max(0, j - 1):j + 1])
    return {"fi": fi, "repo": d["repo"], "path": d["path"], "cursor": cursor, "bol": bol, "eol": eol,
            "line_no": text.count(b"\n", 0, cursor) + 1, "kind": kind, "is_test": LANG["is_test"](d["path"]),
            "in_string": in_string, "line": line, "true_rest": rest, "true_code": true_code,
            "true_lex": lex_norm(true_toks, False), "true_lex_norm": lex_norm(true_toks, True), "text": text}


# ----------------------------------------------------------------------------------------------------- prompts

class Tokenizer:
    def __init__(self, path):
        self.vocab = cmlbpe.Vocab.load(path)
        self.enc = cmlbpe.Encoder(self.vocab)
        sid = self.vocab.special_id
        self.eot, self.fim_prefix, self.fim_middle, self.fim_suffix = sid("<|endoftext|>"), sid("<|fim_prefix|>"), sid("<|fim_middle|>"), sid("<|fim_suffix|>")
        self.file_sep, self.pad = sid("<|file_sep|>"), sid("<|pad|>")
        self.vocab_size = self.vocab.vocab_size
        self.special_base = self.vocab.special_base

    def encode(self, b: bytes):
        return self.enc.encode_bytes(b)

    def decode(self, ids):
        return self.enc.decode_bytes([int(i) for i in ids])


def build_prompt(tok, p, mode, ctx, suffix_tokens, with_path, max_prefix=1450):
    """Prompt of at most `ctx` tokens: header + prefix tail (<= max_prefix) [+ suffix head (<= suffix_tokens)].
    Mirrors the training documents (data.py: FIM doc <= seq_len - 8 with suffix <= 512, middle <= 256, prefix = the
    rest), so that ctx + max_new <= max_context and the model never runs past the trained context."""
    text, cur = p["text"], p["cursor"]
    hdr = [tok.file_sep] + (tok.encode(p["path"].encode() + b"\n") if with_path else [])
    # prefix: last ~40 KB, cut forward to a line start so that the pre-tokenisation is the same as in training
    a = max(0, cur - 40000)
    if a > 0:
        nl = text.find(b"\n", a, cur)
        a = nl + 1 if nl >= 0 else a
    pre = tok.encode(text[a:cur])
    if mode == "plain":
        budget = ctx - len(hdr)
        return hdr + pre[-budget:]
    suf = tok.encode(text[p["eol"]:p["eol"] + 16000])[:suffix_tokens]
    budget = min(max_prefix, ctx - len(hdr) - len(suf) - 3)
    assert budget > 0, (ctx, len(hdr), len(suf))
    if mode == "spm":   # Code Llama variant: the middle follows the prefix directly
        return [tok.fim_prefix, tok.fim_suffix] + suf + [tok.fim_middle] + hdr + pre[-budget:]
    assert mode == "fim", mode
    return [tok.fim_prefix] + hdr + pre[-budget:] + [tok.fim_suffix] + suf + [tok.fim_middle]


# ----------------------------------------------------------------------------------------------------- inference

def _rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c, s = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)


class Generator:
    """Batched greedy decoding with a KV cache and left padding (the model's own generate() recomputes the whole
    context at every step)."""

    def __init__(self, model, tok, device):
        self.m, self.tok, self.dev = model, tok, device
        V, sb = tok.vocab_size, tok.special_base
        stop = np.zeros(V, dtype=bool)
        for i in range(sb):
            stop[i] = tok.vocab.tokens[i][:1] in (b"\n", b"\r")
        stop[sb:] = True
        self.stop = torch.from_numpy(stop).to(device)

    def _fwd(self, ids, pos, mask, kv):
        m = self.m
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev == "cuda"):
            x = m.tok_emb(ids)
            cos, sin = m.rope_cos[pos][:, None], m.rope_sin[pos][:, None]      # [B,1,T,hd/2]
            for li, blk in enumerate(m.layers):
                h = blk.attn_norm(x)
                a = blk.attn
                B, T, _ = h.shape
                q = a.wq(h).view(B, T, a.n_heads, a.hd).transpose(1, 2)
                k = a.wk(h).view(B, T, a.n_kv, a.hd).transpose(1, 2)
                v = a.wv(h).view(B, T, a.n_kv, a.hd).transpose(1, 2)
                q, k = _rope(q, cos, sin), _rope(k, cos, sin)
                if kv[li] is not None:
                    k = torch.cat((kv[li][0], k), dim=2)
                    v = torch.cat((kv[li][1], v), dim=2)
                kv[li] = (k, v)
                rep = a.n_heads // a.n_kv
                y = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), attn_mask=mask)
                x = x + a.wo(y.transpose(1, 2).reshape(B, T, -1))
                x = x + blk.mlp(blk.ffn_norm(x))
            x = m.final_norm(x[:, -1:])
            return m.logits(x)[:, -1].float()

    @torch.no_grad()
    def generate(self, prompts, max_new, true_first=None):
        """prompts: list of id lists. Returns per prompt: (generated ids without the stop token, their probs,
        prob of the stop/last token, stop kind, first-step top-1 id, prob of the true first token)."""
        dev = self.dev
        B = len(prompts)
        L = max(len(p) for p in prompts)
        lens = torch.tensor([len(p) for p in prompts], device=dev)
        ids = torch.full((B, L), self.tok.pad, dtype=torch.long, device=dev)
        for b, p in enumerate(prompts):
            ids[b, L - len(p):] = torch.tensor(p, dtype=torch.long, device=dev)
        ar = torch.arange(L, device=dev)
        valid = ar[None, :] >= (L - lens)[:, None]
        pos = (ar[None, :] - (L - lens)[:, None]).clamp_(min=0)
        mask = (ar[None, None, :, None] >= ar[None, None, None, :]) & valid[:, None, None, :]
        mask = mask | torch.eye(L, dtype=torch.bool, device=dev)[None, None]
        kv = [None] * len(self.m.layers)
        logits = self._fwd(ids, pos, mask, kv)
        gen = [[] for _ in range(B)]; probs = [[] for _ in range(B)]
        stop_p = [None] * B; stop_kind = [None] * B
        done = [False] * B
        top1_first = None; p_true = None
        for step in range(max_new + 1):
            pr = torch.softmax(logits, dim=-1)
            pmax, nxt = pr.max(dim=-1)
            if step == 0:
                top1_first = nxt.tolist()
                if true_first is not None:
                    tf = torch.tensor(true_first, device=dev)
                    p_true = pr[torch.arange(B, device=dev), tf].tolist()
            st = self.stop[nxt].tolist(); nl = nxt.tolist(); pl = pmax.tolist()
            for b in range(B):
                if done[b]:
                    continue
                if st[b]:
                    done[b] = True; stop_p[b] = pl[b]
                    stop_kind[b] = "special" if nl[b] >= self.tok.special_base else "newline"
                elif step == max_new:
                    done[b] = True; stop_p[b] = None; stop_kind[b] = "limit"
                else:
                    gen[b].append(nl[b]); probs[b].append(pl[b])
            if all(done):
                break
            valid = torch.cat((valid, torch.ones(B, 1, dtype=torch.bool, device=dev)), dim=1)
            logits = self._fwd(nxt[:, None], (lens + step)[:, None], valid[:, None, None, :], kv)
        return [(gen[b], probs[b], stop_p[b], stop_kind[b], top1_first[b], p_true[b] if p_true else None) for b in range(B)]


def selftest(model, tok, gen, device):
    """Cached/batched decoding must agree with the model's own full-recompute generate()."""
    if LANG["ext"] == "go":
        snippets = [b"main.go\npackage main\n\nimport (\n\t\"fmt\"\n\t\"os\"\n)\n\nfunc main() {\n\tf, err := os.Open(\"x\")\n\tif err != nil {\n\t\t",
                    b"util/strings.go\npackage util\n\nfunc Join(parts []string, sep string) string {\n\tvar b strings.Builder\n\tfor i, p := range parts {\n\t\tif i > 0 {\n\t\t\tb.",
                    b"a.go\npackage a\n\nfunc f() error {\n\treturn"]
    else:
        snippets = [b"src/App/Program.cs\r\nusing System;\r\nusing System.IO;\r\n\r\nnamespace App\r\n{\r\n    public static class Program\r\n    {\r\n        public static void Main(string[] args)\r\n        {\r\n            if (args.Length == 0)\r\n            {\r\n                ",
                    b"src/Models/User.cs\nnamespace Models;\n\npublic class User\n{\n    public int Id { get; set; }\n    public string Name {",
                    b"src/Util/Strings.cs\nusing System.Text;\n\npublic static class Strings\n{\n    public static string Join(string[] parts, string sep)\n    {\n        var sb = new StringBuilder();\n        for (int i = 0; i < parts.Length; i++)\n        {\n            if (i > 0) sb."]
    prompts = [[tok.file_sep] + tok.encode(s) for s in snippets]
    ref = []
    for p in prompts:
        x = torch.tensor([p], dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            y = model.generate(x, 12, stop_ids=())
        ref.append(y[0, len(p):].tolist())
    # batched, no stopping: compare the first tokens up to the first stop token of the reference
    out = gen.generate(prompts, 12)
    ok = True
    for p, r, o in zip(prompts, ref, out):
        g = o[0]
        n = min(len(g), len(r))
        same = g[:n] == r[:n]
        ok &= same and n >= 1
        print(f"[selftest] ref={tok.decode(r)!r} cached={tok.decode(g)!r} stop={o[3]} agree={same} ({n} tokens)")
    print("[selftest]", "OK" if ok else "MISMATCH")
    return ok


# ----------------------------------------------------------------------------------------------------- metrics

def evaluate_position(tok, p, res, max_new):
    gen_ids, probs, stop_p, stop_kind, top1, p_true = res
    gen_bytes = tok.decode(gen_ids)
    gen_code, gen_toks = code_part(gen_bytes)
    gl, gln = lex_norm(gen_toks, False), lex_norm(gen_toks, True)
    tl, tln = p["true_lex"], p["true_lex_norm"]
    match = lcp(tl, gl); match_n = lcp(tln, gln)
    exact = gen_code == p["true_code"].rstrip() and len(gen_code) > 0
    exact_n = len(tln) > 0 and gln == tln
    n = len(probs)
    head = probs[:3]
    if len(head) < 3 and stop_p is not None:
        head = head + [stop_p]
    conf_gm3 = math.exp(sum(math.log(max(x, 1e-9)) for x in head) / len(head)) if head else 0.0
    allp = probs + ([stop_p] if stop_p is not None else [])
    conf_min = min(allp) if allp else 0.0
    conf_prod = float(np.prod(allp)) if allp else 0.0
    true_first_ids = tok.encode(p["true_rest"])
    return {"repo": p["repo"], "path": p["path"], "line": p["line_no"], "kind": p["kind"], "is_test": p["is_test"],
            "in_string": p["in_string"], "true": p["true_code"].decode("utf-8", "replace"),
            "gen": gen_bytes.decode("utf-8", "replace"), "n_bpe": n, "stop": stop_kind,
            "probs": [round(x, 4) for x in probs], "stop_p": None if stop_p is None else round(stop_p, 4),
            "conf_gm3": conf_gm3, "conf_min": conf_min, "conf_prod": conf_prod,
            "rest_len": len(tl), "gen_len": len(gl), "match": match, "match_norm": match_n,
            "first_ok": match >= 1, "first_ok_norm": match_n >= 1,
            "ok3": match >= min(3, len(tl)), "ok3_strict": match >= 3,
            "ok3_norm": match_n >= min(3, len(tln)), "exact": exact, "exact_norm": exact_n,
            "bpe_top1": bool(true_first_ids) and top1 == true_first_ids[0], "p_true_first": p_true}


def rate(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def summarise(recs):
    """Aggregate metrics over a list of position records."""
    n = len(recs)
    if n == 0:
        return {"n": 0}
    s = {"n": n,
         "first_lex_ok": rate([r["first_ok"] for r in recs]), "first_lex_ok_norm": rate([r["first_ok_norm"] for r in recs]),
         "ok3": rate([r["ok3"] for r in recs]), "ok3_strict": rate([r["ok3_strict"] for r in recs]), "ok3_norm": rate([r["ok3_norm"] for r in recs]),
         "line_exact": rate([r["exact"] for r in recs]), "line_exact_norm": rate([r["exact_norm"] for r in recs]),
         "line_exact_le8": rate([r["exact"] for r in recs if 1 <= r["rest_len"] <= 8]),
         "line_exact_norm_le8": rate([r["exact_norm"] for r in recs if 1 <= r["rest_len"] <= 8]),
         "n_le8": sum(1 for r in recs if 1 <= r["rest_len"] <= 8),
         "bpe_top1": rate([r["bpe_top1"] for r in recs]),
         "mean_match": rate([r["match"] for r in recs]), "mean_rest_len": rate([r["rest_len"] for r in recs]),
         "stop_newline": rate([r["stop"] == "newline" for r in recs]), "stop_limit": rate([r["stop"] == "limit" for r in recs]),
         "thresholds": {}, "calibration": {}}
    for cname in ("conf_gm3", "conf_min", "conf_prod"):
        tab = {}
        for thr in THRESHOLDS:
            shown = [r for r in recs if r[cname] >= thr]
            tp_num = sum(min(r["match"], 3) for r in shown)
            tp_den = sum(min(3, max(r["gen_len"], 1)) for r in shown)
            tab[str(thr)] = {"shown": len(shown) / n, "n_shown": len(shown),
                             "prec_ok3": rate([r["ok3"] for r in shown]), "prec_ok3_strict": rate([r["ok3_strict"] for r in shown]),
                             "prec_ok3_norm": rate([r["ok3_norm"] for r in shown]),
                             "prec_exact": rate([r["exact"] for r in shown]), "prec_exact_norm": rate([r["exact_norm"] for r in shown]),
                             "token_prec": tp_num / tp_den if tp_den else float("nan")}
        s["thresholds"][cname] = tab
        cal = []
        for lo in np.arange(0.0, 1.0, 0.1):
            hi = lo + 0.1
            b = [r for r in recs if lo <= r[cname] < hi or (hi >= 1.0 and r[cname] >= lo)]
            cal.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(b), "share": len(b) / n,
                        "ok3": rate([r["ok3"] for r in b]), "exact": rate([r["exact"] for r in b]),
                        "first_ok": rate([r["first_ok"] for r in b])})
        s["calibration"][cname] = cal
    return s


def len_bucket(k):
    return "1-2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"


def breakdowns(recs):
    groups = {"kind": lambda r: r["kind"], "is_test": lambda r: "test file" if r["is_test"] else "non-test",
              "rest_len": lambda r: len_bucket(r["rest_len"]), "in_string": lambda r: "in string" if r["in_string"] else "code"}
    out = {}
    for gname, key in groups.items():
        g = defaultdict(list)
        for r in recs:
            g[key(r)].append(r)
        out[gname] = {}
        for k in sorted(g, key=lambda x: (-len(g[x]), x)):
            rs = g[k]
            shown = [r for r in rs if r["conf_gm3"] >= 0.8]
            out[gname][k] = {"n": len(rs), "first_lex_ok": rate([r["first_ok"] for r in rs]), "ok3": rate([r["ok3"] for r in rs]),
                             "line_exact": rate([r["exact"] for r in rs]), "line_exact_norm": rate([r["exact_norm"] for r in rs]),
                             "bpe_top1": rate([r["bpe_top1"] for r in rs]),
                             "shown_0.8": len(shown) / len(rs), "prec_0.8": rate([r["ok3"] for r in shown])}
    return out


# ----------------------------------------------------------------------------------------------------- reporting

def pct(x):
    return "n/a" if x != x else f"{100 * x:.1f}"


def f3(x):
    return "n/a" if x != x else f"{x:.3f}"


def markdown(report):
    L = []
    info = report["info"]
    L.append(f"# Inline completion eval — {info['preset']} step {info['step']} ({info['tokens']/1e9:.2f} G tokens)\n")
    L.append(f"Checkpoint `{info['ckpt']}`; {info['n_positions']} positions in {info['n_files']} files of the test fold "
             f"(stride {info['stride']}, seed {info['seed']}); prompt ≤{info['ctx']} tokens (FIM prefix ≤{info.get('max_prefix', '?')}, suffix ≤{info['suffix_tokens']}), "
             f"≤{info['max_new']} new tokens, batch {info['batch']}; run time {info['time_total_s']:.0f} s "
             f"({', '.join(f'{m} {t:.0f} s' for m, t in info['time_modes_s'].items())}, prompts {info['time_prompts_s']:.0f} s).\n")
    ng = report["ngram_ref"]
    L.append(f"## Headline vs n-gram {ng['name']}\n")
    L.append("Confidence here = geometric mean of the first 3 generated BPE tokens' probabilities (the newline token counts when "
             "the line ends earlier) — the direct analogue of the n-gram definition (first 3 lexer tokens). "
             "`≥3 right` = the first 3 lexical tokens of the generated line are right (or the whole line is right when "
             "fewer than 3 remain); `strict` requires 3 matched tokens. `norm` = strings/chars/numbers compared as "
             "`<STR>/<CHAR>/<NUM>` like the n-gram lexer. `line exact` = rest of the line (comments stripped) byte-exact; "
             "`≤8` restricts to lines with 1–8 remaining lexical tokens, the n-gram eligibility.\n")
    modes = list(report["modes"])
    hdr = f"| metric | n-gram {ng['name']} | " + " | ".join(f"nn {m}" for m in modes) + " |"
    L.append(hdr); L.append("|" + "---|" * (len(modes) + 2))
    def row(name, ngv, f):
        L.append(f"| {name} | {ngv} | " + " | ".join(f(report["modes"][m]["summary"]) for m in modes) + " |")
    row("positions", ng["positions"], lambda s: str(s["n"]))
    for thr in ("0.8", "0.7"):
        ngs = f"{ng['shown_' + thr]} %" ; ngo = f"{ng['ok3_' + thr]} %"
        row(f"conf ≥{thr}: shown", ngs, lambda s, t=thr: pct(s["thresholds"]["conf_gm3"][t]["shown"]) + " %")
        row(f"conf ≥{thr}: ≥3 right (strict) [norm]", ngo,
            lambda s, t=thr: f"{pct(s['thresholds']['conf_gm3'][t]['prec_ok3'])} ({pct(s['thresholds']['conf_gm3'][t]['prec_ok3_strict'])}) [{pct(s['thresholds']['conf_gm3'][t]['prec_ok3_norm'])}] %")
        row(f"conf ≥{thr}: line exact [norm]", "—",
            lambda s, t=thr: f"{pct(s['thresholds']['conf_gm3'][t]['prec_exact'])} [{pct(s['thresholds']['conf_gm3'][t]['prec_exact_norm'])}] %")
        row(f"conf ≥{thr}: token precision", f3(ng["tokprec_" + thr]) if ("tokprec_" + thr) in ng else "—",
            lambda s, t=thr: f3(s["thresholds"]["conf_gm3"][t]["token_prec"]))
    row("rest of line exact, all positions [norm]", "—", lambda s: f"{pct(s['line_exact'])} [{pct(s['line_exact_norm'])}] %")
    row("rest of line exact, ≤8 tokens left [norm]", f"{ng['line_exact']} %", lambda s: f"{pct(s['line_exact_le8'])} [{pct(s['line_exact_norm_le8'])}] % (n={s['n_le8']})")
    row("first lexical token right [norm]", f3(ng["first_token"]), lambda s: f"{f3(s['first_lex_ok'])} [{f3(s['first_lex_ok_norm'])}]")
    row("≥3 right, unconditional (strict)", "—", lambda s: f"{pct(s['ok3'])} ({pct(s['ok3_strict'])}) %")
    row("next BPE token top-1", "—", lambda s: f3(s["bpe_top1"]))
    row("mean matched lexical tokens / mean rest length", "—", lambda s: f"{s['mean_match']:.2f} / {s['mean_rest_len']:.2f}")
    row("stopped at newline / hit 48-token limit", "—", lambda s: f"{pct(s['stop_newline'])} / {pct(s['stop_limit'])} %")
    L.append("")
    for m in modes:
        s = report["modes"][m]["summary"]
        L.append(f"## {m}: thresholds for the three confidence definitions\n")
        L.append("| confidence | thr | shown % | ≥3 right % | strict % | line exact % | token prec |")
        L.append("|---|---|---|---|---|---|---|")
        for cname in ("conf_gm3", "conf_min", "conf_prod"):
            for thr in THRESHOLDS:
                t = s["thresholds"][cname][str(thr)]
                L.append(f"| {cname} | {thr} | {pct(t['shown'])} | {pct(t['prec_ok3'])} | {pct(t['prec_ok3_strict'])} | {pct(t['prec_exact'])} | {f3(t['token_prec'])} |")
        L.append("")
        L.append(f"## {m}: calibration (conf_gm3 bin → precision)\n")
        L.append("| bin | n | share % | ≥3 right % | line exact % | first token % |")
        L.append("|---|---|---|---|---|---|")
        for c in s["calibration"]["conf_gm3"]:
            L.append(f"| {c['bin']} | {c['n']} | {pct(c['share'])} | {pct(c['ok3'])} | {pct(c['exact'])} | {pct(c['first_ok'])} |")
        L.append("")
        L.append(f"## {m}: breakdowns\n")
        L.append("| group | value | n | first tok % | ≥3 right % | line exact % | [norm] % | BPE top-1 | shown@0.8 % | prec@0.8 % |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for gname, g in report["modes"][m]["breakdowns"].items():
            for k, v in g.items():
                L.append(f"| {gname} | {k} | {v['n']} | {pct(v['first_lex_ok'])} | {pct(v['ok3'])} | {pct(v['line_exact'])} | {pct(v['line_exact_norm'])} | {f3(v['bpe_top1'])} | {pct(v['shown_0.8'])} | {pct(v['prec_0.8'])} |")
        L.append("")
        if report["modes"][m].get("in_string_extra"):
            e = report["modes"][m]["in_string_extra"]
            L.append(f"{m}: extra in-string positions (cursor after the opening quote, not in the totals): n={e['n']}, "
                     f"line exact {pct(e['line_exact'])} %, first lexical token {pct(e['first_lex_ok'])} %, "
                     f"shown@0.8 {pct(e['thresholds']['conf_gm3']['0.8']['shown'])} %, precision (line exact) "
                     f"{pct(e['thresholds']['conf_gm3']['0.8']['prec_exact'])} %.\n")
    if len(modes) >= 2:
        L.append("## Modes per position (line exact)\n")
        a = modes[0]; pa = report["modes"][a]["positions"]
        for b in modes[1:]:
            pb = report["modes"][b]["positions"]
            both = sum(1 for x, y in zip(pa, pb) if x["exact"] and y["exact"])
            only_a = sum(1 for x, y in zip(pa, pb) if x["exact"] and not y["exact"])
            only_b = sum(1 for x, y in zip(pa, pb) if y["exact"] and not x["exact"])
            L.append(f"- {a} vs {b}: both {both}, only {a} {only_a}, only {b} {only_b}, neither {len(pa) - both - only_a - only_b}.")
        L.append("")
    L.append("## Caveats\n")
    L.append("- The n-gram protocol scores lexer tokens (strings collapsed to `<STR>`, comments invisible, out-of-vocabulary "
             "identifiers `<ID>` never match, ~21 % of identifiers); here the generated text is compared byte-exact / by lexical tokens "
             "(`norm` columns collapse literals the same way). The n-gram generated up to 8 tokens across line breaks; here decoding stops at the first newline, "
             "so `≥3 right` counts lines shorter than 3 tokens as right when they are complete (the `strict` figure does not).")
    L.append(f"- The n-gram eval ran over every 50th token of the test fold ({info.get('n_fold_files', '?')} files); here {info['n_positions']} of those positions are sampled uniformly (same distribution, ±1–2 pp noise).")
    L.append("- Confidence over BPE tokens is not the same quantity as over lexer tokens (one lexer token ≈ 1–2 BPE tokens), so the thresholds are not strictly interchangeable.")
    L.append("- The cursor is placed after the previous token of the line (not after a typed space). Since the FIM data fix "
             "(bounded FIM documents that never straddle a window) the prompt layout — header, prefix tail, ≤512-token suffix — "
             "is exactly what training shows.")
    return "\n".join(L) + "\n"


def write_dump(path, report, positions_by_mode, n_dump, seed, max_new):
    rng = random.Random(seed + 1)
    cats = {"confident & right": lambda r: r["conf_gm3"] >= 0.8 and r["ok3"],
            "confident & wrong": lambda r: r["conf_gm3"] >= 0.8 and not r["ok3"],
            "not shown (conf < 0.8)": lambda r: r["conf_gm3"] < 0.8}
    per = max(1, n_dump // len(cats))
    out = [f"# Inline eval dump — step {report['info']['step']}; {per} random examples per category, modes mixed\n"]
    for cname, pred in cats.items():
        pool = []
        for mode, (recs, pos) in positions_by_mode.items():
            pool += [(mode, r, p) for r, p in zip(recs, pos) if pred(r)]
        out.append(f"\n## {cname} ({len(pool)} positions)\n")
        for mode, r, p in rng.sample(pool, min(per, len(pool))):
            text = p["text"]
            lines = text.split(b"\n")
            ln = p["line_no"] - 1
            ctx_before = lines[max(0, ln - 3):ln]
            ctx_after = lines[ln + 1:ln + 4]
            cur_line = text[p["bol"]:p["cursor"]].decode("utf-8", "replace")
            ctx_before = [l.rstrip(b"\r") for l in ctx_before]; ctx_after = [l.rstrip(b"\r") for l in ctx_after]
            out.append(f"### {mode} | {r['kind']} | conf_gm3={r['conf_gm3']:.2f} prod={r['conf_prod']:.2f} | matched {r['match']}/{r['rest_len']} lexical tokens"
                       f"{' | LINE EXACT' if r['exact'] else ''} | {r['repo']}/{r['path']}:{r['line']}\n")
            out.append("```" + LANG["fence"])
            for l in ctx_before:
                out.append(l.decode("utf-8", "replace"))
            out.append(cur_line + "⟨cursor⟩")
            for l in ctx_after:
                out.append(l.decode("utf-8", "replace"))
            out.append("```")
            out.append(f"TRUE: `{r['true']}`  ")
            out.append(f"GEN : `{r['gen']}`  (stop: {r['stop']}, probs {' '.join(f'{x:.2f}' for x in r['probs'][:8])}{' …' if len(r['probs']) > 8 else ''})\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


# ----------------------------------------------------------------------------------------------------- main

def load_model(ckpt, device, scratch):
    os.makedirs(scratch, exist_ok=True)
    tmp = os.path.join(scratch, "ckpt-eval-copy.pt")
    shutil.copyfile(ckpt, tmp)       # the trainer may be rewriting ckpt-latest.pt
    ck = torch.load(tmp, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ck["config"])
    m = CodeLM(cfg)
    m.load_state_dict(ck["model"])
    m.to(device).eval()
    info = {"preset": cfg.name, "step": int(ck["step"]), "tokens": int(ck["tokens"]), "config": cfg.to_dict(),
            "with_path": not ck["args"].get("no_path", False)}
    del ck
    os.remove(tmp)
    return m, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lang", default="go", choices=sorted(LANGS), help="selects lexer, manifest, repos root, vocab, output dir and n-gram reference")
    ap.add_argument("--vocab", help="BPE vocab (default ~/work/ml-data/tokenizer/<go|cs>-16384.bpe)")
    ap.add_argument("--manifest", help="prepare manifest (default ~/work/ml-data/<lang>/prepared/manifest.jsonl)")
    ap.add_argument("--repos", help="repos root the manifest paths are relative to (default ~/work/ml-data/<lang>/repos)")
    ap.add_argument("--out", help="output prefix (default ~/work/ml-data/<lang>/nn/eval-inline-step<N>)")
    ap.add_argument("--positions", type=int, default=3000)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--modes", default="plain,fim,spm", help="plain | fim (PSM) | spm")
    ap.add_argument("--ctx", type=int, default=0, help="prompt token budget incl. header and suffix (default max_context - max_new)")
    ap.add_argument("--max-prefix", type=int, default=1450, help="longest prefix tail in FIM/SPM prompts (training: <= ~1 500)")
    ap.add_argument("--suffix-tokens", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dump", type=int, default=0)
    ap.add_argument("--no-in-string", action="store_true", help="skip the extra in-string positions")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--scratch", help="where the checkpoint copy goes (default ~/work/ml-data/<lang>/nn/eval-tmp)")
    a = ap.parse_args()
    global LANG
    LANG = LANGS[a.lang]
    a.vocab = a.vocab or os.path.join(DATA, LANG["vocab"])
    a.manifest = a.manifest or os.path.join(DATA, LANG["manifest"])
    a.repos = a.repos or os.path.join(DATA, LANG["repos"])
    a.scratch = a.scratch or os.path.join(DATA, LANG["out_dir"], "eval-tmp")
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer(a.vocab)
    model, info = load_model(a.ckpt, device, a.scratch)
    gen = Generator(model, tok, device)
    print(f"model {info['preset']} step {info['step']} ({info['tokens']/1e9:.2f} G tokens) on {device}", flush=True)
    if a.selftest:
        if not selftest(model, tok, gen, device):
            sys.exit(1)
    out = a.out or os.path.join(DATA, LANG["out_dir"], f"eval-inline-step{info['step']}")
    max_context = int(info["config"].get("max_context", 2048))
    if not a.ctx:
        a.ctx = max_context - a.max_new
    assert a.ctx + a.max_new <= max_context, f"ctx {a.ctx} + max_new {a.max_new} > max_context {max_context}"
    assert a.max_prefix + a.suffix_tokens + 32 <= a.ctx, "prefix + suffix must leave room for the header"

    t0 = time.time()
    files = read_manifest(a.manifest)
    positions = sample_positions(files, a.repos, a.positions, a.stride, a.seed, not a.no_in_string)
    main_pos = [p for p in positions if p["kind"] != "in-string"]
    extra_pos = [p for p in positions if p["kind"] == "in-string"]
    print(f"{len(main_pos)} positions (+{len(extra_pos)} in-string) in {len({p['fi'] for p in positions})} files, "
          f"{time.time() - t0:.0f} s", flush=True)

    modes = a.modes.split(",")
    report = {"info": dict(info, ckpt=a.ckpt, lang=a.lang, vocab=a.vocab, manifest=a.manifest, n_fold_files=len(files),
                           n_positions=len(main_pos), n_in_string=len(extra_pos),
                           n_files=len({p["fi"] for p in main_pos}), stride=a.stride, seed=a.seed, ctx=a.ctx,
                           suffix_tokens=a.suffix_tokens, max_prefix=a.max_prefix, max_new=a.max_new, batch=a.batch, modes=modes,
                           time_modes_s={}, time_prompts_s=0.0), "ngram_ref": LANG["ngram"], "modes": {}}
    positions_by_mode = {}
    for mode in modes:
        t0 = time.time()
        allpos = main_pos + extra_pos
        prompts = [build_prompt(tok, p, mode, a.ctx, a.suffix_tokens, info["with_path"], a.max_prefix) for p in allpos]
        assert max(map(len, prompts)) + a.max_new <= max_context, (max(map(len, prompts)), a.max_new, max_context)
        true_first = [(tok.encode(p["true_rest"]) or [-1])[0] for p in allpos]
        t_prompt = time.time() - t0
        report["info"]["time_prompts_s"] += t_prompt
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))   # length-sorted batches: less padding
        results = [None] * len(prompts)
        t1 = time.time()
        for k in range(0, len(order), a.batch):
            idx = order[k:k + a.batch]
            tf = [true_first[i] if true_first[i] >= 0 else 0 for i in idx]
            res = gen.generate([prompts[i] for i in idx], a.max_new, true_first=tf)
            for i, r in zip(idx, res):
                results[i] = r
            if (k // a.batch) % 50 == 0:
                print(f"  {mode}: {k + len(idx)}/{len(prompts)} ({time.time() - t1:.0f} s)", flush=True)
        t_gen = time.time() - t1
        recs = [evaluate_position(tok, p, r, a.max_new) for p, r in zip(allpos, results)]
        main_recs, extra_recs = recs[:len(main_pos)], recs[len(main_pos):]
        report["modes"][mode] = {"summary": summarise(main_recs), "breakdowns": breakdowns(main_recs),
                                 "in_string_extra": summarise(extra_recs) if extra_recs else None,
                                 "positions": main_recs, "in_string_positions": extra_recs,
                                 "mean_prompt_tokens": float(np.mean([len(p) for p in prompts]))}
        report["info"]["time_modes_s"][mode] = t_gen
        positions_by_mode[mode] = (main_recs, main_pos)
        s = report["modes"][mode]["summary"]
        print(f"{mode}: prompts {t_prompt:.0f} s, generation {t_gen:.0f} s; first-tok {s['first_lex_ok']:.3f}, "
              f"line exact {100*s['line_exact']:.1f} %, ≥0.8 shown {100*s['thresholds']['conf_gm3']['0.8']['shown']:.1f} % "
              f"prec {100*s['thresholds']['conf_gm3']['0.8']['prec_ok3']:.1f} %", flush=True)
    report["info"]["time_total_s"] = time.time() - t_start
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False)
    with open(out + ".md", "w", encoding="utf-8") as f:
        f.write(markdown(report))
    if a.dump:
        write_dump(out + "-dump.md", report, positions_by_mode, a.dump, a.seed, a.max_new)
    print(f"wrote {out}.json / .md{' / -dump.md' if a.dump else ''}; total {report['info']['time_total_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
