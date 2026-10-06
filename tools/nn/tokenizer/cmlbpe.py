"""Byte-level BPE with a code-aware pre-tokenizer (spec: pretokenizer "go-code-1"). Reference implementation.

The Kotlin twin lives in ml-core (io.github.completionml.core.bpe) and must produce identical ids.

PRE-TOKENIZER (operates on raw bytes; no Unicode tables, so Python/Kotlin cannot disagree):
  byte classes: L = [A-Za-z_] and every byte >= 0x80 (all non-ASCII text, incl. emoji, is "letter")
                D = [0-9];  W = ' ' | '\\t';  N = '\\r' | '\\n';  P = every other byte (punctuation, control bytes)
  Rules, tried in order at the current position:
   1. newline run: (\\r\\n | \\n | \\r)+ followed by all W bytes (the indentation of the next line belongs to it)
   2. W run (maximal). If it is followed by a non-W non-N byte and ends with ' ', the last ' ' is detached and
      becomes the optional prefix of the next unit (the rest of the run, if any, is a token of its own);
      otherwise the whole run is one token (tabs, trailing whitespace, whitespace before a newline).
   3. unit = optional single ' ' + (L+ | one D | P+)      (maximal runs of one class; digits one by one)
  camelCase / snake_case are NOT split (BPE learns subwords); letters/digits/punctuation never merge.
  Finally every pre-token longer than 128 bytes is cut into consecutive 128-byte chunks.
"""
import re
import sys

MAX_PRETOKEN = 128
PRETOK_ID = "go-code-1"
FORMAT_MAGIC = "CMLBPE"
FORMAT_VERSION = 1

_PAT = re.compile(
    rb'(?:\r\n|\n|\r)+[ \t]*'
    rb'|[ \t]+(?= [^ \t\r\n])'
    rb'| ?[A-Za-z_\x80-\xff]+'
    rb'| ?[0-9]'
    rb'| ?[^A-Za-z0-9_\x80-\xff \t\r\n]+'
    rb'|[ \t]+')
_findall = _PAT.findall

SPECIALS = ["<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|file_sep|>",
            "<|repo_name|>", "<|pad|>"] + ["<|reserved_%d|>" % i for i in range(9)]


def _chunk(toks):
    out = []
    for t in toks:
        if len(t) > MAX_PRETOKEN:
            out.extend(t[i:i + MAX_PRETOKEN] for i in range(0, len(t), MAX_PRETOKEN))
        else:
            out.append(t)
    return out


def pretokenize(data: bytes):
    toks = _findall(data)
    if toks and max(map(len, toks)) > MAX_PRETOKEN:
        toks = _chunk(toks)
    return toks


def pretokenize_reference(b: bytes):
    """Hand-written scanner, mirrors the Kotlin implementation line by line (used to cross-check the regex)."""
    n = len(b)
    out = []

    def emit(s, e):
        while e - s > MAX_PRETOKEN:
            out.append(b[s:s + MAX_PRETOKEN])
            s += MAX_PRETOKEN
        out.append(b[s:e])

    def is_l(c): return 65 <= c <= 90 or 97 <= c <= 122 or c == 95 or c >= 128
    def is_d(c): return 48 <= c <= 57
    def is_w(c): return c == 32 or c == 9
    def is_n(c): return c == 10 or c == 13

    i = 0
    while i < n:
        c = b[i]
        if is_n(c):
            j = i
            while j < n:
                if b[j] == 13 and j + 1 < n and b[j + 1] == 10: j += 2
                elif is_n(b[j]): j += 1
                else: break
            while j < n and is_w(b[j]): j += 1
            emit(i, j); i = j
            continue
        if is_w(c):
            j = i
            while j < n and is_w(b[j]): j += 1
            if j < n and b[j - 1] == 32 and not is_w(b[j]) and not is_n(b[j]):
                if j - 1 > i: emit(i, j - 1)
                start = j - 1; k = j
            else:
                emit(i, j); i = j
                continue
        else:
            start = i; k = i
        c = b[k]
        if is_d(c): e = k + 1
        elif is_l(c):
            e = k
            while e < n and is_l(b[e]): e += 1
        else:
            e = k
            while e < n and not (is_l(b[e]) or is_d(b[e]) or is_w(b[e]) or is_n(b[e])): e += 1
        emit(start, e); i = e
    return out


class Vocab:
    def __init__(self, merges, specials=None):
        self.merges = [tuple(m) for m in merges]
        self.specials = list(specials if specials is not None else SPECIALS)
        self.n_merges = len(self.merges)
        self.ranks = {}
        self.tokens = [bytes([i]) for i in range(256)]
        for r, (a, b) in enumerate(self.merges):
            assert a < 256 + r and b < 256 + r, "merge %d references a later id" % r
            self.ranks.setdefault((a << 16) | b, r)
            self.tokens.append(self.tokens[a] + self.tokens[b])
        self.special_base = 256 + self.n_merges
        self.vocab_size = self.special_base + len(self.specials)

    def special_id(self, name):
        return self.special_base + self.specials.index(name)

    def save(self, path):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("%s %d\n" % (FORMAT_MAGIC, FORMAT_VERSION))
            f.write("pretokenizer %s\n" % PRETOK_ID)
            f.write("max_pretoken %d\n" % MAX_PRETOKEN)
            f.write("base 256\n")
            f.write("merges %d\n" % self.n_merges)
            f.write("specials %d\n" % len(self.specials))
            for s in self.specials: f.write(s + "\n")
            for a, b in self.merges: f.write("%d %d\n" % (a, b))

    @staticmethod
    def load(path):
        with open(path, encoding="utf-8") as f:
            lines = f.read().split("\n")
        assert lines[0] == "%s %d" % (FORMAT_MAGIC, FORMAT_VERSION), lines[0]
        hdr = {}
        i = 1
        while not lines[i].startswith("specials "):
            k, v = lines[i].split(" ", 1); hdr[k] = v; i += 1
        ns = int(lines[i].split()[1]); i += 1
        assert hdr["pretokenizer"] == PRETOK_ID and int(hdr["max_pretoken"]) == MAX_PRETOKEN
        specials = lines[i:i + ns]; i += ns
        nm = int(hdr["merges"])
        merges = [tuple(map(int, l.split())) for l in lines[i:i + nm]]
        assert len(merges) == nm
        return Vocab(merges, specials)


class Encoder:
    def __init__(self, vocab: Vocab, cache_limit=1 << 21):
        self.v = vocab
        self.cache = {}
        self.cache_limit = cache_limit

    def _bpe(self, tok: bytes):
        ids = list(tok)
        ranks = self.v.ranks
        merges = self.v.merges
        while len(ids) > 1:
            mr = 1 << 60
            for i in range(len(ids) - 1):
                r = ranks.get((ids[i] << 16) | ids[i + 1])
                if r is not None and r < mr: mr = r
            if mr == 1 << 60: break
            a, b = merges[mr]; new = 256 + mr
            out = []; i = 0; n = len(ids)
            while i < n:
                if i + 1 < n and ids[i] == a and ids[i + 1] == b:
                    out.append(new); i += 2
                else:
                    out.append(ids[i]); i += 1
            ids = out
        return tuple(ids)

    def encode_bytes(self, data: bytes):
        cache = self.cache
        out = []
        for t in pretokenize(data):
            r = cache.get(t)
            if r is None:
                if len(cache) >= self.cache_limit: cache.clear()
                r = cache[t] = self._bpe(t)
            out.extend(r)
        return out

    def encode(self, text: str):
        return self.encode_bytes(text.encode("utf-8"))

    def decode_bytes(self, ids):
        toks = self.v.tokens; sb = self.v.special_base; sp = self.v.specials
        return b"".join(toks[i] if i < sb else sp[i - sb].encode() for i in ids)

    def decode(self, ids):
        return self.decode_bytes(ids).decode("utf-8", errors="replace")
