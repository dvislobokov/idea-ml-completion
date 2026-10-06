"""Tests for the document-aware FIM packer in data.py (run: python -I test_data.py  or  pytest test_data.py).

Uses the small `test` fold (42 700 files) so that an epoch builds in seconds.
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402

FOLD = os.environ.get("TEST_FOLD", "test")
_cache = {}


def setup():
    if not _cache:
        _cache["sh"] = D.Shards(D.DEFAULT_DATA, FOLD)
        _cache["tok"] = D.Tokenizer()
    return _cache["sh"], _cache["tok"]


def stream(**kw):
    sh, tok = setup()
    kw.setdefault("t_min", 0)
    kw.setdefault("io_threads", 0)
    return D.PackedStream(sh, tok, **kw)


def fim_spans(w, T):
    """Yield (prefix_pos, suffix_pos, middle_pos, eot_pos) for every <|fim_middle|> in the window; None where missing."""
    w = np.asarray(w)
    for m in np.nonzero(w == T.fim_middle)[0]:
        m = int(m)
        p = s = e = None
        for j in range(m - 1, -1, -1):          # back to the opening <|fim_prefix|>, no other FIM special in between
            if w[j] == T.fim_prefix:
                p = j; break
            if w[j] in (T.fim_middle, T.eot):
                break
        if p is not None:
            ss = np.nonzero(w[p:m] == T.fim_suffix)[0]
            s = p + int(ss[0]) if len(ss) == 1 else None
        for j in range(m + 1, len(w)):          # forward to the closing <|endoftext|>
            if w[j] == T.eot:
                e = j; break
            if w[j] in (T.fim_prefix, T.fim_middle, T.fim_suffix):
                break
        yield p, s, m, e


def test_fim_docs_never_straddle(n_windows=400):
    s = stream(fim_rate=0.5, seed=7)
    T = s.tok
    n_mid = n_pre = 0
    for _ in range(n_windows):
        w = s.next_window()
        assert len(w) == s.seq_len + 1
        for p, suf, m, e in fim_spans(w, T):
            n_mid += 1
            assert p is not None, "fim_middle without fim_prefix in the same window"
            assert suf is not None, "fim_middle without exactly one fim_suffix between prefix and middle"
            assert e is not None, "fim_middle without its <|endoftext|> in the same window"
            assert e - p <= s.seq_len - D.DOC_MARGIN
        # converse: every <|fim_prefix|> has its middle and eot inside the window
        n_pre_w = int((w == T.fim_prefix).sum())
        n_mid_w = int((w == T.fim_middle).sum())
        assert n_pre_w == n_mid_w, (n_pre_w, n_mid_w)
        n_pre += n_pre_w
    assert n_mid > n_windows // 2, n_mid
    print(f"[straddle] {n_windows} windows, {n_mid} FIM documents, all complete; stats {s.stats}")


def test_psm_reassembles(n_files=300):
    sh, tok = setup()
    rng = np.random.default_rng(3)
    s = stream(fim_rate=1.0, spm_rate=0.0)
    idx = rng.choice(sh.n_files, size=n_files, replace=False)
    s._start_epoch()
    short = 0
    for i in idx:
        i = int(i)
        body = sh.file(i)
        if len(body) < 2:
            continue
        d = s.document(i, np.random.default_rng([1, i]), fim=True)
        assert isinstance(d, D.FimDoc) and not d.spm
        assert len(d.mid) <= D.MAX_MID and len(d.suf) <= D.MAX_SUF
        assert len(d) <= s.seq_len - D.DOC_MARGIN
        span = np.concatenate([d.pre, d.mid, d.suf])
        # contiguous slice of the body ...
        a = len(body) - len(d.suf) - len(d.mid) - len(d.pre)
        hits = [k for k in range(len(body) - len(span) + 1) if np.array_equal(body[k:k + len(span)], span)]
        assert hits, "pre+mid+suf is not a slice of the body"
        # ... and the whole body for files that fit without truncation
        if len(body) + len(d.hdr) + 4 <= s.seq_len - D.DOC_MARGIN and len(d.suf) < D.MAX_SUF:
            assert np.array_equal(span, body); short += 1
        # the assembled PSM document parses back into the same pieces
        t = d.tokens()
        T = tok
        assert t[0] == T.fim_prefix and t[-1] == T.eot
        ps = int(np.nonzero(t == T.fim_suffix)[0][0]); pm = int(np.nonzero(t == T.fim_middle)[0][0])
        assert np.array_equal(t[1:1 + len(d.hdr)], d.hdr)
        assert np.array_equal(t[1 + len(d.hdr):ps], d.pre)
        assert np.array_equal(t[ps + 1:pm], d.suf)
        assert np.array_equal(t[pm + 1:-1], d.mid)
        assert tok.decode(np.concatenate([d.pre, d.mid, d.suf])) == tok.decode(span)
    assert short > 0
    print(f"[psm] {n_files} files checked, {short} short files reassemble to the full body")


def test_line_aligned_cuts(n_files=300):
    sh, tok = setup()
    s = stream(fim_rate=1.0, line_rate=1.0)
    s._start_epoch()
    rng = np.random.default_rng(5)
    n = 0
    for i in rng.choice(sh.n_files, size=n_files, replace=False):
        body = sh.file(int(i))
        if len(body) < 2 or not tok.nl_start[body].any():
            continue
        a, b = s.fim_cuts(body, np.random.default_rng([2, int(i)]))
        assert 0 <= a <= b <= len(body) and b - a <= D.MAX_MID
        assert b == len(body) or tok.nl_start[body[b]], "line-aligned middle must end at a line end"
        n += 1
    assert n > n_files // 2
    print(f"[line] {n} line-aligned cuts end at a line end")


def test_resume_exact(n_before=120, n_after=120):
    a = stream(fim_rate=0.5, seed=11, io_threads=2)
    for _ in range(n_before):
        a.next_window()
    st = pickle.loads(pickle.dumps(a.state_dict()))    # as it goes through torch.save / torch.load
    ref = [a.next_window() for _ in range(n_after)]
    b = stream(fim_rate=0.5, seed=11)
    b.load_state_dict(st)
    got = [b.next_window() for _ in range(n_after)]
    for k, (x, y) in enumerate(zip(ref, got)):
        assert np.array_equal(x, y), f"window {k} after resume differs"
    assert a.state_dict()["cursor"] == b.state_dict()["cursor"]
    print(f"[resume] {n_after} windows after resume identical (cursor {b.cursor}, queue {len(b.queue)} docs)")


def test_legacy_carry_state():
    """A checkpoint from the old flat packer carries a token array; it must open the first window."""
    s = stream(fim_rate=0.5, seed=11)
    carry = np.arange(1000, dtype=np.uint16)
    s.load_state_dict({"epoch": 0, "cursor": 5, "carry": carry, "windows": 3, "tokens_seen": 3 * 2048})
    w = s.next_window()
    assert np.array_equal(w[:1000], carry) and s.cursor >= 6 and s.windows == 4
    print("[legacy] old carry state restored")


def test_fim_share(n_windows=600, fim_rate=0.5):
    s = stream(fim_rate=fim_rate, seed=23)
    T = s.tok
    fim_tok = 0; with_fim = 0
    for _ in range(n_windows):
        w = s.next_window()
        spans = list(fim_spans(w, T))
        with_fim += bool(spans)
        fim_tok += sum(e - p + 1 for p, _, _, e in spans)
    docs = s.stats["fim_docs"] + s.stats["plain_docs"]
    share = s.stats["fim_docs"] / docs
    assert abs(share - fim_rate) < 0.05, share
    print(f"[share] FIM documents {share:.3f} of {docs} (target {fim_rate}); FIM tokens {fim_tok/(n_windows*(s.seq_len+1)):.3f} "
          f"of all tokens; windows with >=1 FIM doc {with_fim/n_windows:.3f}; stats {s.stats}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"all {len(tests)} tests passed")
