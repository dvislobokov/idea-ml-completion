#!/usr/bin/env python3
"""e20: pack mined import counts (mine_imports.py) into the `imports` .cml artifact read by ml-core.

    pack_imports.py --lang go --counts ~/work/ml-data/go/imports/counts-lm.pkl --out models/go-imports-e20.cml [--lambda 1.0]

Name -> path prior p(path | name):
  go      file-level counts of `pkg.Ident` resolved through the import list (exact);
  csharp  `--prior decl|excess|both`: `decl` = files declaring the type in the namespace (exact but only for types declared in the
          corpus), `excess` = co-occurrence of the identifier with the using beyond chance, max(0, c(N,U) - c(N) c(U) / D)
          (covers the BCL and NuGet packages whose sources are not in the corpus), `both` = excess + decl-weight * decl.
Pruning: pairs with count < --min-count dropped, top --top-paths per name, co-import lists (PMI, top --top-co by pair count) only for
paths imported by at least --co-min-doc files. Scores are quantised to 1/16 nat (see imports_model.py).
"""
import argparse, collections, json, math, os, pickle, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imports_model import ImportsModel, q_logp, q_pmi  # noqa: E402


_GO_STD = None


def go_std():
    global _GO_STD
    if _GO_STD is None:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'go-std.txt')) as f:
            _GO_STD = {l.strip() for l in f if l.strip()}
    return _GO_STD


def path_ok(lang, p):
    """Importable from another project: Go = stdlib or a module path (dot in the first element), no `internal`/`vendor`
    elements; C# = a dotted or simple ASCII namespace."""
    if not p.isascii() or len(p) >= 200:
        return False
    if lang == 'go':
        parts = p.split('/')
        if 'internal' in parts or 'vendor' in parts or 'testdata' in parts:
            return False
        return p in go_std() or ('.' in parts[0] and len(parts) >= 2)
    return all(part and (part[0].isalpha() or part[0] == '_') for part in p.split('.'))


def build(counts, lang, min_count=3, top_paths=16, top_co=32, co_min_doc=20, prior='both', decl_weight=1.0, lam=1.0,
          name_min_doc=3, corpus_id='', ctx_norm='sum', excess_share=0.0, max_names=0, decl_min_use=20, excess_rel=0.0, decl_rel=0.0):
    D = counts['docs']
    path_doc = {p: c for p, c in counts['path_doc'].items() if path_ok(lang, p)}
    # ---- name -> [(path, weight)]
    per_name = collections.defaultdict(dict)
    weak = collections.defaultdict(list)
    if lang == 'go' or prior in ('excess', 'both'):
        name_doc = counts['name_doc']
        for (n, p), c in counts['name_path'].items():
            if c < min_count or not n.isascii() or p not in path_doc:
                continue
            if lang == 'go':
                w = float(c)
            else:
                w = c - name_doc[n] * path_doc[p] / D
                if w < min_count:
                    continue
                if w < excess_share * name_doc[n]:
                    weak[n].append((p, w))  # kept only relative to the strongest namespace of the name (second pass)
                    continue
            per_name[n][p] = per_name[n].get(p, 0.0) + w
    if lang != 'go' and excess_rel > 0:
        for n, lst in weak.items():
            if n not in per_name:
                continue
            top = max(per_name[n].values())
            for p, w in lst:
                if w >= excess_rel * top:
                    per_name[n][p] = per_name[n].get(p, 0.0) + w
    if lang == 'go':
        for (n, p), c in counts['qual'].items():  # package qualifiers as written (`http` -> net/http), keys are lower-case
            if c >= min_count and n.isascii() and p in path_doc:
                per_name[n][p] = per_name[n].get(p, 0.0) + c
    if lang != 'go' and prior in ('decl', 'both'):
        for (n, ns), c in counts['decl'].items():  # a type is declared once: keep namespaces other projects import
            if path_doc.get(ns, 0) < decl_min_use or not n.isascii():
                continue
            # a declaration is one file; give it decl_weight file-equivalents, or decl_rel of the strongest usage candidate
            # so that a declared-but-rarely-used namespace stays within reach of the context term
            top = max(per_name[n].values()) if per_name.get(n) else 0.0
            per_name[n][ns] = per_name[n].get(ns, 0.0) + max(decl_weight, decl_rel * top) * c
    names = {}
    used_paths = set()
    has_decl = set()
    if lang != 'go' and prior in ('decl', 'both'):
        has_decl = {n for (n, ns) in counts['decl'] if path_doc.get(ns, 0) >= decl_min_use}
    if max_names and len(per_name) > max_names:  # keep the most used identifiers
        keep = sorted(per_name, key=lambda n: -sum(per_name[n].values()))[:max_names]
        keep = set(keep) | has_decl
        per_name = {n: per_name[n] for n in keep}
    for n, d in per_name.items():
        items = sorted(d.items(), key=lambda t: (-t[1], t[0]))[:top_paths]
        total = sum(w for _, w in items)
        if total < name_min_doc and n not in has_decl:
            continue
        names[n] = (int(round(total)), [(p, q_logp(w / total)) for p, w in items])
        used_paths.update(p for p, _ in items)
    # ---- paths table: those used by a name entry or frequent enough for a co-import list
    co_paths = {p for p, c in path_doc.items() if c >= co_min_doc and p.isascii()}
    paths = sorted(used_paths | co_paths)
    pid = {p: i for i, p in enumerate(paths)}
    # ---- co-imports: PMI, top by pair count, both directions, among co_paths
    lists = collections.defaultdict(list)
    for (a, b), c in counts['pair'].items():
        if c < min_count or a not in co_paths or b not in co_paths:
            continue
        pmi = math.log(c * D / (path_doc[a] * path_doc[b]))
        q = q_pmi(pmi)
        lists[a].append((c, b, q))
        lists[b].append((c, a, q))
    co = {}
    for a, lst in lists.items():
        lst.sort(key=lambda t: (-t[0], t[1]))
        co[pid[a]] = [(pid[b], q) for _, b, q in lst[:top_co]]
    model = ImportsModel(lang, {'lambda': str(lam), 'ctx_norm': ctx_norm, 'min_count': str(min_count), 'prior': prior if lang != 'go' else 'refs'},
                         D, paths, [path_doc.get(p, 0) for p in paths],
                         {n: (doc, [(pid[p], q) for p, q in lst]) for n, (doc, lst) in names.items()}, co, corpus_id)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lang', required=True, choices=['go', 'csharp'])
    ap.add_argument('--counts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--min-count', type=int, default=3)
    ap.add_argument('--top-paths', type=int, default=16)
    ap.add_argument('--top-co', type=int, default=32)
    ap.add_argument('--co-min-doc', type=int, default=20)
    ap.add_argument('--prior', default='both', choices=['decl', 'excess', 'both'])
    ap.add_argument('--decl-weight', type=float, default=3.0, help='C#: file-equivalents per declaring file')
    ap.add_argument('--lambda', dest='lam', type=float, default=1.0)
    ap.add_argument('--name-min-doc', type=int, default=3)
    ap.add_argument('--ctx-norm', default='sum', choices=['sum', 'sqrt', 'mean'])
    ap.add_argument('--excess-share', type=float, default=0.05, help='C#: usage pair kept when excess >= share * files using the name')
    ap.add_argument('--max-names', type=int, default=0, help='keep only the N most used identifiers (0 = all)')
    ap.add_argument('--excess-rel', type=float, default=0.0, help='C#: also keep usage pairs with excess >= rel * the strongest candidate')
    ap.add_argument('--decl-rel', type=float, default=0.0, help='C#: declaration weight at least rel * the strongest usage candidate')
    ap.add_argument('--decl-min-use', type=int, default=20, help='C#: declarations count only in namespaces imported by >= N files')
    ap.add_argument('--decl-counts', default='', help='C#: extra mined pickles whose declarations (facts, not statistics) join the prior')
    a = ap.parse_args()
    with open(a.counts, 'rb') as f:
        counts = pickle.load(f)
    for extra in filter(None, a.decl_counts.split(',')):
        with open(extra, 'rb') as f:
            counts['decl'] = counts['decl'] + pickle.load(f)['decl']
    st = counts.get('stats', {})
    corpus_id = f"{a.lang} {st.get('fold', '?')} fold, {st.get('repos', '?')} repos, {counts['docs']} files with imports"
    m = build(counts, a.lang, a.min_count, a.top_paths, a.top_co, a.co_min_doc, a.prior, a.decl_weight, a.lam, a.name_min_doc, corpus_id, a.ctx_norm, a.excess_share, a.max_names, a.decl_min_use, a.excess_rel, a.decl_rel)
    m.write(a.out)
    info = dict(paths=len(m.paths), names=len(m.names), name_entries=sum(len(v[1]) for v in m.names.values()),
                co_lists=len(m.co), co_entries=sum(len(v) for v in m.co.values()), bytes=os.path.getsize(a.out), corpus=corpus_id)
    print(json.dumps(info))


if __name__ == '__main__':
    main()
