#!/usr/bin/env python3
"""LightGBM lambdarank ranker on the expanded feature matrices dumped by `ml-train dump-features` (e19).

    ml-train dump-features --shards data/go-psi/rank --out go-rank.cmlf.gz
    ml-train dump-features --shards data/go-psi/test --out go-test.cmlf.gz
    python -I tools/gbdt/train_gbdt.py --train go-rank.cmlf.gz --test go-test.cmlf.gz --trees 300 --export-trees go.trees.txt \
        --fixture go-parity.tsv
    ml-train import-gbdt --lang go --trees go.trees.txt --shards data/go-psi/rank --out models/go-rank-gbdt-e19.cml

Groups = completion lists (one relevant candidate each); the validation split for early stopping / tuning is a set of whole
repositories of the training fold, the test fold is only reported. Metrics are computed like `RankMetrics` (ties: answer last).
The text export is read by `TreeRanker.parseText` and reproduced in numpy here (`--check`) before anything goes to Kotlin.
"""
import argparse, gzip, hashlib, struct, sys, time
import numpy as np
import lightgbm as lgb

KINDS = ["AFTER_DOT", "STATEMENT_START", "ARGUMENT", "TYPE_POSITION", "ASSIGN_RHS", "OTHER"]


def read_cmlf(path):
    """Returns (names, repos[list], kinds[list], chosen[list], sizes[list], X[rows, n])."""
    data = gzip.open(path, "rb").read()
    pos = 0

    def u(fmt):
        nonlocal pos
        v = struct.unpack_from(fmt, data, pos); pos += struct.calcsize(fmt); return v[0]

    def utf():
        nonlocal pos
        n = u(">H"); s = data[pos:pos + n].decode("utf-8"); pos += n; return s

    assert data[:4] == b"CMLF", "not a dump-features file"; pos = 4
    assert u(">i") == 1
    n = u(">i"); names = [utf() for _ in range(n)]
    lists = u(">i")
    repos, kinds, chosen, sizes = [], np.zeros(lists, np.int32), np.zeros(lists, np.int32), np.zeros(lists, np.int32)
    for i in range(lists):
        repos.append(utf()); kinds[i] = u(">B"); chosen[i] = u(">i"); sizes[i] = u(">i")
    rows = u(">i")
    X = np.frombuffer(data, dtype=">f4", count=rows * n, offset=pos).astype(np.float32).reshape(rows, n)
    assert sizes.sum() == rows
    return names, repos, kinds, chosen, sizes, X


def labels_of(chosen, sizes):
    y = np.zeros(sizes.sum(), np.int32)
    off = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    y[off + chosen] = 1
    return y


def metrics(scores, chosen, sizes, kinds=None):
    """top-1 / top-5 / MRR with pessimistic ties (as ml-core RankMetrics)."""
    off = 0; rr = 0.0; t1 = 0; t5 = 0; per_kind = {}
    for i, n in enumerate(sizes):
        s = scores[off:off + n]; c = s[chosen[i]]
        rank = 1 + int(np.sum(s >= c)) - 1
        rr += 1.0 / rank; t1 += rank == 1; t5 += rank <= 5
        if kinds is not None:
            b = per_kind.setdefault(KINDS[kinds[i]], [0, 0.0]); b[0] += 1; b[1] += 1.0 / rank
        off += n
    m = len(sizes)
    return {"top1": t1 / m, "top5": t5 / m, "mrr": rr / m, "n": m, "by_kind": {k: v[1] / v[0] for k, v in per_kind.items()}}


def fmt(m):
    return "n=%d  top1=%.3f  top5=%.3f  MRR=%.3f" % (m["n"], m["top1"], m["top5"], m["mrr"])


def repo_split(repos, frac, salt="valid"):
    """Whole repositories into the validation set by md5 (deterministic)."""
    return np.array([int(hashlib.md5((salt + r).encode()).hexdigest()[:2], 16) < 256 * frac for r in repos])


# ---- tree export -----------------------------------------------------------------------------------------------------
MISSING = {"None": 0, "Zero": 1, "NaN": 2}


def flatten(tree):
    """Preorder node list of a LightGBM dump tree: (feature, threshold, default_left, missing, left, right, value)."""
    nodes = []

    def rec(n):
        idx = len(nodes); nodes.append(None)
        if "leaf_value" in n:
            nodes[idx] = (-1, 0.0, 0, 0, -1, -1, float(n["leaf_value"])); return idx
        assert n["decision_type"] == "<=", n["decision_type"]
        l = rec(n["left_child"]); r = rec(n["right_child"])
        nodes[idx] = (int(n["split_feature"]), float(n["threshold"]), 1 if n["default_left"] else 0, MISSING[n["missing_type"]], l, r, 0.0)
        return idx

    rec(tree); return nodes


def export_trees(booster, names, path, n_trees=None):
    dump = booster.dump_model(num_iteration=n_trees)
    assert not dump.get("average_output", False)
    trees = [flatten(t["tree_structure"]) for t in dump["tree_info"]]
    with open(path, "w") as f:
        f.write("tree-ranker 1\nfeatures %d\n" % len(names))
        for n in names: f.write(n + "\n")
        f.write("trees %d\n" % len(trees))
        for t in trees:
            f.write("tree %d\n" % len(t))
            for (feat, thr, dl, mt, l, r, v) in t:
                f.write("%d %s %d %d %d %d %s\n" % (feat, repr(thr), dl, mt, l, r, repr(v)))
    return trees


def predict_trees(trees, X):
    """numpy re-implementation of TreeRanker (LightGBM NumericalDecision incl. missing modes)."""
    out = np.zeros(len(X), np.float64)
    for t in trees:
        for i, x in enumerate(X):
            n = 0
            while True:
                feat, thr, dl, mt, l, r, v = t[n]
                if feat < 0: out[i] += v; break
                fv = float(x[feat]); nan = np.isnan(fv)
                if nan and mt != 2: fv = 0.0
                if (mt == 1 and abs(fv) <= 1e-35) or (mt == 2 and nan): n = l if dl else r
                else: n = l if fv <= thr else r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--trees", type=int, default=300); ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--lr", type=float, default=0.05); ap.add_argument("--min-leaf", type=int, default=50)
    ap.add_argument("--feature-fraction", type=float, default=0.8); ap.add_argument("--bagging", type=float, default=0.8)
    ap.add_argument("--valid-frac", type=float, default=0.15, help="share of training repositories held out for early stopping")
    ap.add_argument("--early-stop", type=int, default=50, help="0 = fixed --trees")
    ap.add_argument("--max-lists", type=int, default=0, help="subsample training lists (0 = all)")
    ap.add_argument("--export-trees", help="text trees for `ml-train import-gbdt`")
    ap.add_argument("--small", type=int, default=0, help="also export the first N trees to <export-trees>.small<N> (size/quality trade-off)")
    ap.add_argument("--fixture", help="parity fixture: TSV of test rows (group, label, features…, lightgbm score)")
    ap.add_argument("--fixture-lists", type=int, default=8)
    ap.add_argument("--check", type=int, default=2000, help="rows to verify the text export against booster.predict")
    ap.add_argument("--importance", type=int, default=15); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()

    t0 = time.time()
    names, repos, kinds, chosen, sizes, X = read_cmlf(a.train)
    tn, tr_repos, tk, tc, ts, Xt = read_cmlf(a.test)
    assert tn == names, "test schema differs"
    print("train %d lists / %d rows, test %d lists / %d rows, %d features (%.1f s)" % (len(sizes), len(X), len(ts), len(Xt), len(names), time.time() - t0))
    y = labels_of(chosen, sizes)

    valid = repo_split(repos, a.valid_frac)
    if a.max_lists and a.max_lists < len(sizes):
        rng = np.random.default_rng(a.seed); keep = np.zeros(len(sizes), bool); keep[rng.choice(len(sizes), a.max_lists, replace=False)] = True
        valid_keep = keep | valid
    else:
        valid_keep = np.ones(len(sizes), bool)
    row_of = np.repeat(np.arange(len(sizes)), sizes)
    tr_rows = (~valid[row_of]) & valid_keep[row_of]; va_rows = valid[row_of]
    tr_lists = (~valid) & valid_keep
    print("split: %d train lists (%d repos) / %d valid lists (%d repos)" % (tr_lists.sum(), len({r for r, v, k in zip(repos, valid, valid_keep) if not v and k}), valid.sum(), len({r for r, v in zip(repos, valid) if v})))

    dtrain = lgb.Dataset(X[tr_rows], y[tr_rows], group=sizes[tr_lists], feature_name=["f%d" % i for i in range(len(names))], free_raw_data=False)  # ":" is not allowed in LightGBM names
    dvalid = lgb.Dataset(X[va_rows], y[va_rows], group=sizes[valid], reference=dtrain)
    params = dict(objective="lambdarank", metric="map", eval_at=[100], learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=a.min_leaf,
                  feature_fraction=a.feature_fraction, bagging_fraction=a.bagging, bagging_freq=1 if a.bagging < 1 else 0,
                  lambdarank_truncation_level=20, num_threads=a.threads, seed=a.seed, verbose=-1, deterministic=True, force_row_wise=True)
    cb = [lgb.log_evaluation(50)]
    if a.early_stop: cb.append(lgb.early_stopping(a.early_stop, verbose=True))
    t1 = time.time()
    booster = lgb.train(params, dtrain, num_boost_round=a.trees, valid_sets=[dvalid], valid_names=["valid"], callbacks=cb)
    best = booster.best_iteration or a.trees
    print("trained %d trees (best %d) in %.1f s" % (booster.num_trees(), best, time.time() - t1))

    for label, n_it in [("gbdt %d trees" % best, best)] + ([("gbdt 100 trees", 100)] if best > 100 else []):
        s = booster.predict(Xt, num_iteration=n_it)
        m = metrics(s, tc, ts, tk)
        print("%s: %s" % (label, fmt(m)), "  ".join("%s=%.3f" % kv for kv in sorted(m["by_kind"].items())))
    base = metrics(-Xt[:, names.index("rule_rank_log")], tc, ts) if "rule_rank_log" in names else None
    if base: print("plugin rules: " + fmt(base))

    imp = booster.feature_importance("gain", iteration=best); order = np.argsort(-imp)[:a.importance]
    print("feature importance (gain, %% of total):")
    for i in order: print("  %-30s %5.1f %%  (splits %d)" % (names[i], 100 * imp[i] / imp.sum(), booster.feature_importance("split", iteration=best)[i]))

    if a.export_trees:
        trees = export_trees(booster, names, a.export_trees, best)
        nodes = sum(len(t) for t in trees); leaves = sum(1 for t in trees for n in t if n[0] < 0)
        print("exported %d trees, %d nodes, %d leaves to %s" % (len(trees), nodes, leaves, a.export_trees))
        if a.check:
            k = min(a.check, len(Xt)); ref = booster.predict(Xt[:k], num_iteration=best); mine = predict_trees(trees, Xt[:k])
            print("export check on %d rows: max |diff| = %.3g" % (k, np.max(np.abs(ref - mine))))
            assert np.max(np.abs(ref - mine)) < 1e-9
        if a.small and a.small < best:
            small = export_trees(booster, names, a.export_trees + ".small%d" % a.small, a.small)
            print("exported the first %d trees to %s.small%d" % (len(small), a.export_trees, a.small))
    if a.fixture:
        # a few whole test lists (features exactly as Kotlin expands them) with the booster's scores
        rng = np.random.default_rng(a.seed); pick = sorted(rng.choice(len(ts), a.fixture_lists, replace=False))
        off = np.concatenate([[0], np.cumsum(ts)[:-1]])
        with open(a.fixture, "w") as f:
            f.write("# group\tlabel\t" + "\t".join(names) + "\tscore\n")
            for g, li in enumerate(pick):
                rows = Xt[off[li]:off[li] + ts[li]]; sc = booster.predict(rows, num_iteration=best)
                for c in range(ts[li]):
                    f.write("%d\t%d\t%s\t%s\n" % (g, 1 if c == tc[li] else 0, "\t".join(repr(float(v)) for v in rows[c]), repr(float(sc[c]))))
        print("fixture: %d lists, %d rows to %s" % (len(pick), int(ts[pick].sum()), a.fixture))


if __name__ == "__main__":
    main()
