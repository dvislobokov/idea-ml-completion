"""Data pipeline over the uint16 BPE shards written by tokenizer/encode_corpus.py.

Shard layout (per fold, in --data dir): <fold>.tokens.u16, <fold>.offsets.u64 (N+1), <fold>.repo.u32, <fold>.repos.txt,
<fold>.files.txt, <fold>.meta.json.  A file is tokens[offsets[i]:offsets[i+1]], read with os.pread (see Shards.file).

Epoch construction (deterministic from seed + epoch index):
  1. Repo down-weighting. Let T_r be the number of tokens of repo r. Each file of r is kept with probability
         p_r = min(1, sqrt(T_min / T_r))
     so a repo contributes ~T_r tokens if T_r <= T_min and ~sqrt(T_min * T_r) tokens otherwise, i.e. the expected share of
     a repo grows with sqrt(T_r) above the knee instead of linearly (a 100x bigger SDK repo gets 10x, not 100x, the
     presence of a normal repo). Different epochs draw different subsets of the big repos, so over several epochs their
     files are still covered. T_min defaults to 2M tokens (see README for the resulting token shares).
  2. Kept files are grouped by repo, shuffled within the repo, and cut into groups of <= group_tokens tokens
     (default 2 * seq_len) so that a context window usually sees a few files of the same repository (cross-file context,
     as the plugin will provide at inference). Groups are shuffled globally.
  3. Document format of a file inside a group (specials from the BPE vocabulary):
         plain:  <|file_sep|> path "\n" tokens <|endoftext|>
         FIM PSM: <|fim_prefix|> <|file_sep|> path "\n" pre <|fim_suffix|> suf <|fim_middle|> mid <|endoftext|>
         FIM SPM: <|fim_prefix|> <|fim_suffix|> suf <|fim_middle|> <|file_sep|> path "\n" pre mid <|endoftext|>
     FIM is applied per file (document level) with probability fim_rate; PSM/SPM with probability 1/2 each. The path
     header always stays with the prefix. The header can be disabled (--no-path); the repo name is NOT included (the
     plugin does not know GitHub names).
     FIM pieces are bounded so that the whole FIM document fits into one window with margin (total <= seq_len - 8):
         middle <= MAX_MID (256) tokens, suffix = the first <= MAX_SUF (512) tokens after the middle,
         prefix = the last tokens before the middle that still fit (-> up to ~1 250-1 400 with the default sizes).
     Cut placement (line_rate = 0.5 of the FIM documents are line-aligned, the inference positions are mid-line or
     line-start with the suffix always starting at the end of the cursor line):
       * token-level (1 - line_rate): start uniform over the body, middle length ~ U[0, 32] (50 %) or U[0, 256] (50 %);
       * line-aligned (line_rate): the middle ends at a line end (just before a newline token, or at EOF) and spans
         k lines (k = 1 with prob 0.5, else 1 + Geometric(0.35) capped at 8); the middle starts at the line start
         (after the newline + indentation token(s)) with prob 0.5, otherwise at a uniform token inside the span (mid-line
         cursor). Spans longer than MAX_MID are trimmed at the start.
  4. Document-aware packing into windows of seq_len + 1 tokens (x = w[:-1], y = w[1:]), no padding, no attention
     masking across documents, loss on every position. Plain documents are concatenated and may be split across
     windows. A FIM document is never split: if it does not fit into the remaining room of the window it is placed
     with its prefix shortened when room >= SHRINK_MIN (384) tokens and head + tail + MIN_PRE still fit; otherwise it is
     deferred (put back in front of the queue) and the room is filled with the following documents (plain ones are
     split to end the window exactly). Deferred FIM documents fit at the start of the next window (doc <= seq_len - 8).
     As a last resort (> MAX_SKIP deferred documents, never seen in practice) a FIM document is downgraded to plain.

Resume: the stream state is (epoch, group cursor, pending documents); PackedStream.state_dict()/load_state_dict()
restore it exactly (the documents of group g are a pure function of (seed, epoch, g), so groups are fetched ahead by a
thread pool, which helps when the page cache is cold). Prefetcher overlaps batch assembly with the GPU in one more
background thread; warm_page_cache() reads the token file sequentially at start.
"""
import json
import os
import queue
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

TOK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tokenizer")
if TOK_DIR not in sys.path:
    sys.path.insert(0, TOK_DIR)
import cmlbpe  # noqa: E402

DEFAULT_DATA = os.path.expanduser("~/work/ml-data/go/bpe16k")
DEFAULT_VOCAB = os.path.expanduser("~/work/ml-data/tokenizer/go-16384.bpe")


class Tokenizer:
    """Thin wrapper around cmlbpe with the special ids we need."""

    def __init__(self, vocab_path=DEFAULT_VOCAB):
        self.path = vocab_path
        self.vocab = cmlbpe.Vocab.load(vocab_path)
        self.enc = cmlbpe.Encoder(self.vocab)
        sid = self.vocab.special_id
        self.eot = sid("<|endoftext|>")
        self.fim_prefix = sid("<|fim_prefix|>")
        self.fim_middle = sid("<|fim_middle|>")
        self.fim_suffix = sid("<|fim_suffix|>")
        self.file_sep = sid("<|file_sep|>")
        self.repo_name = sid("<|repo_name|>")
        self.pad = sid("<|pad|>")
        self.vocab_size = self.vocab.vocab_size
        # line-boundary tables over the vocabulary (specials: neither). The pre-tokenizer makes "\n+[ \t]*" one
        # pre-token, so a token starting with a newline byte marks a line end (cut *before* it) and the line start is
        # after that token and any following whitespace-only tokens.
        toks = self.vocab.tokens
        self.nl_start = np.zeros(self.vocab_size, dtype=bool)
        self.ws_only = np.zeros(self.vocab_size, dtype=bool)
        for i, t in enumerate(toks):
            self.nl_start[i] = t[:1] in (b"\n", b"\r")
            self.ws_only[i] = len(t) > 0 and all(c in (32, 9) for c in t)

    def encode(self, text):
        return self.enc.encode(text)

    def decode(self, ids):
        return self.enc.decode([int(i) for i in ids])


class Shards:
    """One fold: offsets/repo ids in RAM, tokens read on demand with pread."""

    def __init__(self, data_dir, fold):
        pre = os.path.join(data_dir, fold)
        self.meta = json.load(open(pre + ".meta.json"))
        self.pre = pre
        self.tokens_path = pre + ".tokens.u16"
        self.n_tokens = os.path.getsize(self.tokens_path) // 2
        self._fd = os.open(self.tokens_path, os.O_RDONLY)
        self.offsets = np.fromfile(pre + ".offsets.u64", dtype=np.uint64).astype(np.int64)
        self.repo = np.fromfile(pre + ".repo.u32", dtype=np.uint32).astype(np.int64)
        self.n_files = len(self.offsets) - 1
        assert self.n_files == len(self.repo) == self.meta["files"]
        assert int(self.offsets[-1]) == self.n_tokens == self.meta["tokens"]
        with open(pre + ".repos.txt", encoding="utf-8") as f:
            self.repos = f.read().split("\n")[:-1]
        self.n_repos = len(self.repos)
        self.lengths = np.diff(self.offsets)
        self.repo_tokens = np.bincount(self.repo, weights=self.lengths, minlength=self.n_repos)
        self._paths = None

    def paths(self):
        """Relative path of every file (lazy, ~3M short strings)."""
        if self._paths is None:
            with open(self.pre + ".files.txt", encoding="utf-8") as f:
                self._paths = [line.rstrip("\n").split("\t", 1)[1] for line in f]
            assert len(self._paths) == self.n_files
        return self._paths

    def file(self, i):
        """Tokens of file i. os.pread, not memmap: on this kernel a page fault on the 13.7 GB mapping costs ~20 ms,
        pread of the same (cached) bytes 0.2 ms; pread also releases the GIL for the I/O threads."""
        a, n = int(self.offsets[i]), int(self.lengths[i])
        return np.frombuffer(os.pread(self._fd, 2 * n, 2 * a), dtype=np.uint16)

    def path_tokens(self, tok, workers=16):
        """BPE ids of `path + "\\n"` for every file, as (u16 array, int64 offsets N+1); built once into
        <fold>.pathtok.u16 / <fold>.pathoff.u64 next to the shards (the Python BPE is ~0.3 ms per path, so encoding
        the 3.2 M paths on the fly made the loader 20x slower than the GPU)."""
        ft, fo = self.pre + ".pathtok.u16", self.pre + ".pathoff.u64"
        if not (os.path.exists(ft) and os.path.exists(fo)):
            build_path_tokens(self.paths(), tok.path, ft, fo, workers)
        pt = np.fromfile(ft, dtype=np.uint16)
        po = np.fromfile(fo, dtype=np.uint64).astype(np.int64)
        assert len(po) == self.n_files + 1 and int(po[-1]) == pt.shape[0], "stale path token cache; delete " + ft
        return pt, po


_penc = None


def _path_init(vocab_path):
    global _penc
    _penc = cmlbpe.Encoder(cmlbpe.Vocab.load(vocab_path))


def _path_work(batch):
    lens, out = [], []
    for p in batch:
        ids = _penc.encode(p + "\n")
        lens.append(len(ids)); out.extend(ids)
    return lens, np.asarray(out, dtype=np.uint16).tobytes()


def build_path_tokens(paths, vocab_path, ft, fo, workers=16):
    import multiprocessing as mp
    t0 = time.time()
    uniq = sorted(set(paths))
    index = {p: i for i, p in enumerate(uniq)}
    bs = 2000
    batches = [uniq[i:i + bs] for i in range(0, len(uniq), bs)]
    ulens, ublobs = [], []
    with mp.Pool(min(workers, 20), _path_init, (vocab_path,)) as pool:
        for lens, blob in pool.imap(_path_work, batches):
            ulens.extend(lens); ublobs.append(blob)
    utok = np.frombuffer(b"".join(ublobs), dtype=np.uint16)
    uoff = np.r_[0, np.cumsum(ulens)]
    ids = np.fromiter((index[p] for p in paths), dtype=np.int64, count=len(paths))
    lens = uoff[ids + 1] - uoff[ids]
    off = np.r_[0, np.cumsum(lens)].astype(np.uint64)
    with open(ft + ".tmp", "wb") as f:
        for k in range(0, len(paths), 100_000):
            f.write(np.concatenate([utok[uoff[j]:uoff[j + 1]] for j in ids[k:k + 100_000]]).tobytes())
    off.tofile(fo + ".tmp")
    os.replace(ft + ".tmp", ft); os.replace(fo + ".tmp", fo)
    print(f"[paths] encoded {len(uniq)} unique paths for {len(paths)} files in {time.time()-t0:.0f}s -> {ft}", flush=True)


def repo_keep_prob(repo_tokens, t_min):
    rt = np.maximum(repo_tokens, 1.0)
    return np.minimum(1.0, np.sqrt(t_min / rt))


def build_epoch(sh: Shards, seed, epoch, t_min, group_tokens, max_file_tokens):
    """Return a list of groups; each group is an int64 array of file indices (same repo). Deterministic."""
    rng = np.random.default_rng([seed, epoch, 0xC0DE])
    p = repo_keep_prob(sh.repo_tokens, t_min)[sh.repo] if t_min > 0 else np.ones(sh.n_files)
    keep = rng.random(sh.n_files) < p
    if max_file_tokens:
        keep &= sh.lengths <= max_file_tokens
    keep &= sh.lengths > 0
    idx = np.nonzero(keep)[0]
    # shuffle, then stable sort by repo -> files of a repo are contiguous in random order
    idx = idx[rng.permutation(len(idx))]
    idx = idx[np.argsort(sh.repo[idx], kind="stable")]
    repo = sh.repo[idx]
    lens = sh.lengths[idx] + 3  # + specials (file_sep, eot, ~path)
    # cut repo runs into groups of <= group_tokens
    groups = []
    starts = np.r_[0, np.nonzero(np.diff(repo))[0] + 1, len(idx)]
    for a, b in zip(starts[:-1], starts[1:]):
        cum = np.cumsum(lens[a:b])
        s = a
        acc0 = 0
        for j in range(a, b):
            if cum[j - a] - acc0 > group_tokens and j > s:
                groups.append(idx[s:j]); s = j; acc0 = cum[j - a - 1]
        groups.append(idx[s:b])
    order = rng.permutation(len(groups))
    return [groups[i] for i in order], int(keep.sum()), int(sh.lengths[idx].sum())


# FIM document bounds (tokens); see module doc, item 3 / 4.
MAX_MID = 256        # longest middle
MAX_SUF = 512        # suffix = first MAX_SUF tokens after the middle
DOC_MARGIN = 8       # whole FIM document <= seq_len - DOC_MARGIN
SHRINK_MIN = 384     # shrink a FIM prefix into the remaining room only if the room is at least this
MIN_PRE = 32         # ... and at least this many prefix tokens survive
MAX_SKIP = 32        # deferred FIM documents per window before downgrading one to plain (never reached in practice)

_U16 = np.uint16


def _sp(x):
    return np.array([x], dtype=_U16)


class FimDoc:
    """A fill-in-the-middle document kept in pieces so that the packer can shorten the prefix.

    tokens(pre_keep) assembles PSM/SPM with the last `pre_keep` prefix tokens (None = all)."""
    __slots__ = ("spm", "hdr", "pre", "suf", "mid", "T")

    def __init__(self, T, spm, hdr, pre, suf, mid):
        self.T, self.spm, self.hdr, self.pre, self.suf, self.mid = T, spm, hdr, pre, suf, mid

    def fixed_len(self):
        """Tokens outside the prefix: 4 specials + header + suffix + middle."""
        return 4 + len(self.hdr) + len(self.suf) + len(self.mid)

    def __len__(self):
        return self.fixed_len() + len(self.pre)

    def tokens(self, pre_keep=None):
        T = self.T
        pre = self.pre if pre_keep is None else self.pre[len(self.pre) - pre_keep:]
        if self.spm:   # SPM (Code Llama / StarCoder variant)
            parts = [_sp(T.fim_prefix), _sp(T.fim_suffix), self.suf, _sp(T.fim_middle), self.hdr, pre, self.mid, _sp(T.eot)]
        else:          # PSM
            parts = [_sp(T.fim_prefix), self.hdr, pre, _sp(T.fim_suffix), self.suf, _sp(T.fim_middle), self.mid, _sp(T.eot)]
        return np.concatenate(parts)

    def plain(self):
        """Downgrade: the covered file span as a plain document (last resort of the packer)."""
        return np.concatenate([self.hdr, self.pre, self.mid, self.suf, _sp(self.T.eot)])

    # checkpoint form: plain tuple of numpy arrays (no class reference in the pickle, so that other tools can
    # torch.load the checkpoint without importing this module)
    def to_state(self):
        return ("fim", bool(self.spm), self.hdr, self.pre, self.suf, self.mid)


class PackedStream:
    """Infinite stream of packed windows (seq_len + 1 tokens) over one fold with the transforms above.

    The documents of group g of epoch e are a pure function of (seed, e, g) (its own numpy RNG), so groups are fetched
    ahead by a thread pool (pread releases the GIL) and the resume state is just (epoch, cursor, pending documents).
    A document in the queue is either a uint16 array (plain, splittable) or a FimDoc (atomic).
    """

    def __init__(self, sh: Shards, tok: Tokenizer, seq_len=2048, seed=1, fim_rate=0.5, spm_rate=0.5, line_rate=0.5,
                 t_min=2_000_000, group_tokens=None, with_path=True, max_file_tokens=0, epoch=0, name="train",
                 io_threads=4, lookahead=128):
        self.sh, self.tok, self.seq_len, self.seed = sh, tok, seq_len, seed
        self.fim_rate, self.spm_rate, self.line_rate, self.t_min = fim_rate, spm_rate, line_rate, t_min
        self.group_tokens = group_tokens or 2 * seq_len
        self.with_path = with_path
        self.max_file_tokens = max_file_tokens
        self.name = name
        self.epoch = epoch
        self.cursor = 0           # next group index to *consume*
        self.queue = deque()      # documents fetched from groups < cursor and not yet emitted
        self.windows = 0
        self.tokens_seen = 0
        self.stats = {"fim_docs": 0, "plain_docs": 0, "shrunk": 0, "deferred": 0, "downgraded": 0}
        self.groups = None
        self._ptok, self._poff = sh.path_tokens(tok) if with_path else (None, None)
        self._pool = ThreadPoolExecutor(max_workers=io_threads) if io_threads > 0 else None
        self._lookahead = lookahead if self._pool else 0
        self._pending = deque()   # futures for groups cursor .. cursor+len(pending)-1 of the current epoch
        self._submitted = 0
        assert MAX_MID + MAX_SUF + SHRINK_MIN < seq_len - DOC_MARGIN

    # ---------------------------------------------------------------- epoch / state

    def _start_epoch(self):
        self.groups, self.epoch_files, self.epoch_tokens = build_epoch(
            self.sh, self.seed, self.epoch, self.t_min, self.group_tokens, self.max_file_tokens)
        self.cursor = 0
        self._pending.clear(); self._submitted = 0
        print(f"[{self.name}] epoch {self.epoch}: {self.epoch_files} files, {self.epoch_tokens/1e9:.3f} G tokens, "
              f"{len(self.groups)} groups", flush=True)

    def state_dict(self):
        q = [d.to_state() if isinstance(d, FimDoc) else ("plain", d) for d in self.queue]
        return {"epoch": self.epoch, "cursor": self.cursor, "queue": q, "windows": self.windows,
                "tokens_seen": self.tokens_seen, "stats": dict(self.stats)}

    def load_state_dict(self, st):
        self.epoch, self.windows, self.tokens_seen = st["epoch"], st["windows"], st["tokens_seen"]
        if "queue" in st:
            self.queue = deque(FimDoc(self.tok, *d[1:]) if d[0] == "fim" else np.asarray(d[1], dtype=_U16)
                               for d in st["queue"])
        else:   # checkpoint from the flat-stream packer: the carry is a plain token run
            self.queue = deque([np.asarray(st["carry"], dtype=_U16)])
        self.stats.update(st.get("stats", {}))
        self._start_epoch()
        self.cursor = st["cursor"]
        self._submitted = self.cursor

    # ---------------------------------------------------------------- documents

    def _path_tokens(self, i):
        return np.asarray(self._ptok[self._poff[i]:self._poff[i + 1]])

    def _line_start_after(self, body, i):
        """Index of the first token of the line that follows the newline token at body[i]."""
        n = len(body)
        j = i + 1
        while j < n and (self.tok.nl_start[body[j]] or self.tok.ws_only[body[j]]):
            j += 1
        return j

    def fim_cuts(self, body, rng):
        """(a, b): middle = body[a:b], len <= MAX_MID. Token-level or line-aligned (see module doc)."""
        n = len(body)
        nl = np.nonzero(self.tok.nl_start[body])[0] if rng.random() < self.line_rate else None
        if nl is not None and len(nl) > 0:
            ends = np.append(nl, n)                      # line-end cut points (before the newline token / EOF)
            k = 1 if rng.random() < 0.5 else min(8, 1 + int(rng.geometric(0.35)))
            j = int(rng.integers(0, len(ends)))
            b = int(ends[j])
            a0 = 0 if j - k < 0 else self._line_start_after(body, int(nl[j - k]))
            a0 = min(a0, b)
            if rng.random() < 0.5 or a0 >= b:            # cursor at line start (after the indentation)
                a = a0
            else:                                        # cursor mid-line
                a = int(rng.integers(a0, b))
            a = max(a, b - MAX_MID)
        else:
            a = int(rng.integers(0, n + 1))
            L = int(rng.integers(0, 33)) if rng.random() < 0.5 else int(rng.integers(0, MAX_MID + 1))
            b = min(n, a + L)
        return a, b

    def document(self, i, rng, fim=None):
        """File i as a training document (see module doc): a uint16 array (plain) or a FimDoc.
        fim: None = random by fim_rate, True/False forced."""
        T = self.tok
        body = self.sh.file(i)
        hdr = [_sp(T.file_sep)]
        if self.with_path:
            hdr.append(self._path_tokens(i))
        hdr = np.concatenate(hdr)
        if fim is None:
            fim = rng.random() < self.fim_rate
        if not fim or len(body) < 2:
            return np.concatenate([hdr, body, _sp(T.eot)])
        a, b = self.fim_cuts(body, rng)
        mid = body[a:b]
        suf = body[b:b + MAX_SUF]
        pre_max = self.seq_len - DOC_MARGIN - 4 - len(hdr) - len(suf) - len(mid)
        pre = body[max(0, a - pre_max):a]
        return FimDoc(T, rng.random() < self.spm_rate, hdr, pre, suf, mid)

    def group_docs_of(self, epoch, gi):
        """Pure function of (seed, epoch, gi): the documents of one group, in order."""
        rng = np.random.default_rng([self.seed, epoch, gi, 0xF1E])
        return [self.document(int(i), rng) for i in self.groups[gi]]

    def group_tokens_of(self, epoch, gi):
        """Flat tokens of one group (documents concatenated, FIM documents complete) — inspection only."""
        return np.concatenate([d.tokens() if isinstance(d, FimDoc) else d for d in self.group_docs_of(epoch, gi)])

    # ---------------------------------------------------------------- windows

    def _next_group_docs(self):
        if self.groups is None:
            self._start_epoch()
        while self.cursor >= len(self.groups):
            self.epoch += 1
            self._start_epoch()
        if self._pool is None:
            docs = self.group_docs_of(self.epoch, self.cursor)
        else:
            while self._submitted < min(len(self.groups), self.cursor + self._lookahead):
                self._pending.append(self._pool.submit(self.group_docs_of, self.epoch, self._submitted))
                self._submitted += 1
            docs = self._pending.popleft().result()
        self.cursor += 1
        return docs

    def next_window(self):
        """Document-aware packing (module doc, item 4)."""
        need = self.seq_len + 1
        out, n = [], 0
        q = self.queue
        skipped = []
        while n < need:
            if not q:
                q.extend(self._next_group_docs())
                continue
            d = q.popleft()
            room = need - n
            if isinstance(d, FimDoc):
                L = len(d)
                if L <= room:
                    out.append(d.tokens()); n += L
                    self.stats["fim_docs"] += 1
                elif room >= SHRINK_MIN and d.fixed_len() + MIN_PRE <= room:
                    out.append(d.tokens(room - d.fixed_len())); n = need
                    self.stats["fim_docs"] += 1; self.stats["shrunk"] += 1
                elif len(skipped) < MAX_SKIP:
                    skipped.append(d)
                    self.stats["deferred"] += 1
                else:                       # last resort, keeps the stream finite; never observed
                    d = d.plain()
                    self.stats["downgraded"] += 1
                    out.append(d[:room]); n += min(room, len(d))
                    if len(d) > room:
                        q.appendleft(d[room:])
            else:
                L = len(d)
                if L <= room:
                    out.append(d); n += L
                    self.stats["plain_docs"] += 1
                else:
                    out.append(d[:room]); n = need
                    q.appendleft(d[room:])   # the rest continues in the next window
        if skipped:                            # deferred FIM documents open the next window
            q.extendleft(reversed(skipped))
        w = np.concatenate(out) if len(out) > 1 else out[0]
        assert len(w) == need
        self.windows += 1
        self.tokens_seen += self.seq_len
        return w

    def next_batch(self, batch_size):
        """numpy [B, seq_len+1] uint16."""
        return np.stack([self.next_window() for _ in range(batch_size)])


def warm_page_cache(path, chunk=64 << 20):
    """Read a file sequentially in a daemon thread so that later random access hits the page cache."""
    def run():
        t0 = time.time(); n = 0
        with open(path, "rb", buffering=0) as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                n += len(b)
        print(f"[cache] warmed {os.path.basename(path)}: {n/1e9:.1f} GB in {time.time()-t0:.0f}s", flush=True)
    th = threading.Thread(target=run, daemon=True); th.start()
    return th


class Prefetcher:
    """Background thread turning PackedStream batches into pinned int64 tensors (x, y)."""

    def __init__(self, stream: PackedStream, batch_size, depth=6):
        self.stream, self.bs = stream, batch_size
        self.q = queue.Queue(maxsize=depth)
        self.stop = False
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while not self.stop:
            try:
                b = self.stream.next_batch(self.bs)
            except RuntimeError:   # thread pool shut down at interpreter exit
                return
            st = self.stream.state_dict()
            t = torch.from_numpy(b.astype(np.int64)).pin_memory()
            self.q.put((t, st))

    def next(self, device):
        """Returns x, y on device, plus the stream state *after* this batch (save it with the checkpoint)."""
        t, st = self.q.get()
        t = t.to(device, non_blocking=True)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous(), st

    def close(self):
        self.stop = True
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass


def eval_windows(sh: Shards, tok: Tokenizer, n_windows, seq_len=2048, fim=False, seed=12345, with_path=True):
    """Fixed held-out windows (numpy [n, seq_len+1]) from a fold: fim=False -> plain documents, True -> every document FIM."""
    s = PackedStream(sh, tok, seq_len=seq_len, seed=seed, fim_rate=1.0 if fim else 0.0, spm_rate=0.5,
                     t_min=0, with_path=with_path, name="eval-fim" if fim else "eval")
    return s.next_batch(n_windows)


if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser(description="inspect shards / dump a few packed windows")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--fold", default="lm")
    ap.add_argument("--t-min", type=float, default=2e6)
    ap.add_argument("--show", type=int, default=1, help="decode N windows")
    ap.add_argument("--bench", type=int, default=0, help="time N windows")
    a = ap.parse_args()
    sh = Shards(a.data, a.fold)
    tok = Tokenizer()
    print(f"{a.fold}: {sh.n_files} files, {sh.n_repos} repos, {sh.n_tokens/1e9:.3f} G tokens")
    rt = np.sort(sh.repo_tokens)[::-1]
    print("repo tokens: max %.1fM, p99 %.1fM, p90 %.1fM, median %.1fk" % (rt[0]/1e6, rt[len(rt)//100]/1e6, rt[len(rt)//10]/1e6, np.median(rt)/1e3))
    p = repo_keep_prob(sh.repo_tokens, a.t_min)
    exp_tokens = (p * sh.repo_tokens)
    print(f"T_min={a.t_min:.0f}: expected epoch tokens {exp_tokens.sum()/1e9:.3f} G ({exp_tokens.sum()/sh.repo_tokens.sum()*100:.1f} %), "
          f"repos above knee {(p < 1).sum()} holding {sh.repo_tokens[p < 1].sum()/sh.repo_tokens.sum()*100:.1f} % of raw tokens")
    top = np.argsort(sh.repo_tokens)[::-1][:8]
    for r in top:
        print(f"  {sh.repos[r]:45} {sh.repo_tokens[r]/1e6:7.1f} M tokens  share raw {sh.repo_tokens[r]/sh.repo_tokens.sum()*100:5.2f} % -> "
              f"{exp_tokens[r]/exp_tokens.sum()*100:5.2f} %")
    s = PackedStream(sh, tok, t_min=a.t_min)
    for k in range(a.show):
        w = s.next_window()
        print(f"--- window {k} ({len(w)} tokens) ---")
        print(tok.decode(w)[:3000])
    if a.bench:
        t0 = time.time(); n = 0
        for _ in range(a.bench):
            n += len(s.next_window())
        dt = time.time() - t0
        print(f"{a.bench} windows: {n/dt/1e6:.2f} M tokens/s (single thread)")
