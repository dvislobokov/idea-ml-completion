#!/usr/bin/env python3
"""e20: evaluate an `imports` .cml artifact on a held-out fold.

    eval_imports.py --lang go --model models/go-imports-e20.cml --fold test [--lambdas 0,0.5,1,2] [--fixture parity.txt]

Query 1 (rankImports): for every file of the fold, for every referenced identifier whose import is present in the file, hide that
import and ask rankImports(name, other imports); top-1 / top-3 with lambda = 0 (prior only: most frequent path for the name) vs the
packed lambda / the --lambdas sweep. Go: `pkg.Ident` references resolved through the import list (exact). C#: identifiers whose
declaring namespace is known from the declaration index (--decl-counts: the mined pickles of the lm and rank folds + the fold's
own declarations, i.e. facts, not statistics) and exactly one of the file's usings declares it.
Query 2 (rankCoImports): hide one random import of each file, rank by the others; hit@1/5/10 vs the frequency order.
Writes a parity fixture (--fixture, --fixture-n queries of each kind) for ImportsParityTest.
"""
import argparse, collections, json, os, pickle, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imports_parse as P  # noqa: E402
from imports_model import ImportsModel  # noqa: E402
from mine_imports import load_tasks, read_text, go_local_prefix  # noqa: E402


def parse_fold(lang, tasks):
    """[(repo, file, imports(set), refs)] with refs = Go: set of (ident, path); C#: (idents(set), decls(set of (name, ns)))."""
    out = []
    for repo, files in tasks:
        if lang == 'go':
            local = go_local_prefix(repo)
            for f in files:
                t = read_text(f)
                if t is None:
                    continue
                imports, refs, _ = P.parse_go(t, local)
                if imports:
                    out.append((repo, f, imports, refs))
        else:
            parsed = []
            local_ns = set()
            for f in files:
                t = read_text(f)
                if t is None:
                    continue
                usings, idents, decls = P.parse_cs(t)
                parsed.append((f, usings, idents, decls))
                local_ns.update(ns for _, ns in decls)
            for f, usings, idents, decls in parsed:
                usings = usings - local_ns
                if usings:
                    out.append((repo, f, usings, (idents, decls)))
    return out


def build_queries(lang, records, decl_index, per_file, rng):
    qs = []
    for repo, f, imports, refs in records:
        if lang == 'go':
            cands = sorted(refs)
        else:
            idents, decls = refs
            cands = []
            for n in sorted(idents):
                nss = decl_index.get(n)
                if not nss:
                    continue
                hit = [u for u in imports if u in nss]
                if len(hit) == 1:
                    cands.append((n, hit[0]))
        if len(cands) > per_file:
            cands = rng.sample(cands, per_file)
        for n, p in cands:
            qs.append((n, p, sorted(imports - {p})))
    return qs


def eval_rank(model, qs, lam):
    top1 = top3 = known = 0
    for n, truth, ctx in qs:
        r = model.rank_imports(n, ctx, lam=lam)
        if not r:
            continue
        known += 1
        paths = [p for p, _ in r[:3]]
        if paths[0] == truth:
            top1 += 1
        if truth in paths:
            top3 += 1
    return dict(queries=len(qs), known=known, top1=top1 / len(qs), top3=top3 / len(qs),
                top1_known=top1 / max(1, known), top3_known=top3 / max(1, known))


def eval_co(model, records, rng, limit=10):
    hits = collections.Counter(); base = collections.Counter(); n = 0
    freq_order = sorted(range(len(model.paths)), key=lambda i: -model.path_doc[i])
    for repo, f, imports, _ in records:
        imps = sorted(p for p in imports if p in model.path_id)
        if len(imps) < 2:
            continue
        truth = rng.choice(imps)
        ctx = [p for p in imps if p != truth]
        n += 1
        r = [p for p, _ in model.rank_co_imports(ctx, limit)]
        ctx_set = set(ctx)
        b = []
        for i in freq_order:
            if model.paths[i] not in ctx_set:
                b.append(model.paths[i])
                if len(b) == limit:
                    break
        for k in (1, 5, 10):
            if truth in r[:k]:
                hits[k] += 1
            if truth in b[:k]:
                base[k] += 1
    return dict(files=n, **{f'hit{k}': hits[k] / max(1, n) for k in (1, 5, 10)}, **{f'freq_hit{k}': base[k] / max(1, n) for k in (1, 5, 10)})


def write_fixture(path, model, qs, records, rng, n):
    with open(path, 'w') as f:
        f.write(f'# imports parity fixture: {model.language}; R name<TAB>ctx(,)<TAB>expected path=score(;) ; C ctx<TAB>expected top-10\n')
        for name, truth, ctx in rng.sample(qs, min(n, len(qs))):
            r = model.rank_imports(name, ctx)
            f.write('R\t%s\t%s\t%s\n' % (name, ','.join(ctx), ';'.join(f'{p}={s:g}' for p, s in r)))
        recs = [r for r in records if len(r[2]) >= 2]
        for _, _, imports, _ in rng.sample(recs, min(n, len(recs))):
            ctx = sorted(imports)
            r = model.rank_co_imports(ctx, 10)
            f.write('C\t%s\t%s\n' % (','.join(ctx), ';'.join(f'{p}={s:g}' for p, s in r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lang', required=True, choices=['go', 'csharp'])
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', default=os.path.expanduser('~/work/ml-data'))
    ap.add_argument('--fold', default='test')
    ap.add_argument('--max-repos', type=int, default=0)
    ap.add_argument('--max-files-per-repo', type=int, default=200)
    ap.add_argument('--per-file', type=int, default=10)
    ap.add_argument('--lambdas', default='0,0.5,1,2')
    ap.add_argument('--decl-counts', default='', help='C#: comma-separated mined pickles whose decl Counters form the truth index')
    ap.add_argument('--fixture')
    ap.add_argument('--fixture-n', type=int, default=50)
    ap.add_argument('--dump', help='write a few true/ranked examples here')
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    t0 = time.time()
    model = ImportsModel.read(a.model)
    tasks = load_tasks(os.path.join(a.data, a.lang, 'prepared', 'manifest.jsonl'), os.path.join(a.data, a.lang, 'repos'),
                       a.fold, a.max_files_per_repo, a.max_repos)
    records = parse_fold(a.lang, tasks)
    decl_index = collections.defaultdict(set)
    if a.lang == 'csharp':
        for p in filter(None, a.decl_counts.split(',')):
            with open(p, 'rb') as f:
                for (n, ns) in pickle.load(f)['decl']:
                    decl_index[n].add(ns)
        for _, _, _, (idents, decls) in records:
            for n, ns in decls:
                decl_index[n].add(ns)
    qs = build_queries(a.lang, records, decl_index, a.per_file, rng)
    print(f'{len(tasks)} repos, {len(records)} files with imports, {len(qs)} queries ({time.time() - t0:.0f} s)', flush=True)
    res = dict(lang=a.lang, fold=a.fold, repos=len(tasks), files=len(records), queries=len(qs), model=os.path.basename(a.model),
               model_lambda=model.lam, rank={})
    for lam in [float(x) for x in a.lambdas.split(',')]:
        t1 = time.time()
        r = eval_rank(model, qs, lam)
        r['us_per_query'] = round((time.time() - t1) / max(1, len(qs)) * 1e6, 1)
        res['rank'][str(lam)] = r
        print(f'  lambda {lam}: top-1 {r["top1"]:.4f} top-3 {r["top3"]:.4f} (known {r["known"]}: {r["top1_known"]:.4f} / {r["top3_known"]:.4f})', flush=True)
    res['co'] = eval_co(model, records, random.Random(a.seed))
    print('  co-imports:', json.dumps(res['co']), flush=True)
    if a.fixture:
        write_fixture(a.fixture, model, qs, records, random.Random(a.seed + 1), a.fixture_n)
    if a.dump:
        with open(a.dump, 'w') as f:
            for name, truth, ctx in random.Random(a.seed + 2).sample(qs, min(40, len(qs))):
                r0 = model.rank_imports(name, ctx, lam=0)[:3]
                r1 = model.rank_imports(name, ctx)[:3]
                f.write(f'{name}\ttrue={truth}\tctx={",".join(ctx)[:160]}\n  prior: {r0}\n  ctx:   {r1}\n')
    print(json.dumps(res))


if __name__ == '__main__':
    main()
