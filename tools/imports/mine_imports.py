#!/usr/bin/env python3
"""e20: mine import statistics from a corpus fold (file-level counts).

    mine_imports.py --lang go --fold lm --out ~/work/ml-data/go/imports/counts-lm.pkl [--max-files-per-repo 500] [--jobs 32]

Reads the `ml-train prepare` manifest (status ok, the requested fold), processes one repository per task so the repository's own
packages / namespaces can be dropped, and writes a pickle with document-level Counters:
  docs            number of files counted
  path_doc        import path (Go) / namespace (C#) -> files importing it
  name_doc        identifier -> files referencing it (Go: as `pkg.Ident`; C#: PascalCase identifier not declared in the file)
  name_path       (identifier, path) -> files where the identifier is used and the path is imported (Go: `pkg.Ident` resolved
                  through the import list; C#: co-occurrence with every using of the file — noisy, see pack_imports.py)
  pair            (path_a, path_b), a < b -> files importing both
  decl            C# only: (type name, namespace) -> files declaring it
  qual            Go only: (package qualifier as written, path) -> files
Per-worker Counters are pruned of singletons when they exceed --prune-at entries (the packer keeps count >= 3 anyway).
"""
import argparse, collections, json, os, pickle, sys, time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imports_parse as P  # noqa: E402

STATE = None  # per-worker accumulation


class State:
    def __init__(self, lang, prune_at):
        self.lang = lang
        self.prune_at = prune_at
        self.docs = 0
        self.repos = 0
        self.path_doc = collections.Counter()
        self.name_doc = collections.Counter()
        self.name_path = collections.Counter()
        self.pair = collections.Counter()
        self.decl = collections.Counter()
        self.qual = collections.Counter()

    def prune(self):
        for c in (self.name_path, self.pair, self.name_doc, self.qual):
            if len(c) > self.prune_at:
                for k in [k for k, v in c.items() if v == 1]:
                    del c[k]

    def to_dict(self):
        return dict(lang=self.lang, docs=self.docs, repos=self.repos, path_doc=self.path_doc, name_doc=self.name_doc,
                    name_path=self.name_path, pair=self.pair, decl=self.decl, qual=self.qual)


def _init(lang, prune_at):
    global STATE
    STATE = State(lang, prune_at)


def read_text(path):
    try:
        with open(path, 'rb') as f:
            return f.read().decode('utf-8', 'replace')
    except OSError:
        return None


def go_local_prefix(repo):
    owner, _, name = repo.partition('__')
    return f'github.com/{owner}/{name}'.lower() if name else None


def process_repo(task):
    """task = (repo, [absolute file paths])"""
    repo, files = task
    st = STATE
    st.repos += 1
    if st.lang == 'go':
        local = go_local_prefix(repo)
        for f in files:
            text = read_text(f)
            if text is None:
                continue
            imports, refs, quals = P.parse_go(text, local)
            if not imports:
                continue
            st.docs += 1
            st.path_doc.update(imports)
            st.name_path.update(refs)
            st.name_doc.update(n for n, _ in refs)
            st.qual.update(quals)
            ps = sorted(imports)
            st.pair.update((ps[i], ps[j]) for i in range(len(ps)) for j in range(i + 1, len(ps)))
    else:
        parsed = []
        local_ns = set()
        for f in files:
            text = read_text(f)
            if text is None:
                continue
            usings, idents, decls = P.parse_cs(text)
            parsed.append((usings, idents, decls))
            local_ns.update(ns for _, ns in decls)
        for usings, idents, decls in parsed:
            usings = usings - local_ns
            st.decl.update(decls)
            if not usings:
                continue
            st.docs += 1
            st.path_doc.update(usings)
            st.name_doc.update(idents)
            st.name_path.update((n, u) for n in idents for u in usings)
            ps = sorted(usings)
            st.pair.update((ps[i], ps[j]) for i in range(len(ps)) for j in range(i + 1, len(ps)))
    if st.repos % 50 == 0:
        st.prune()
    return len(files)


def flush(out_dir):
    st = STATE
    if st is None or st.docs == 0 and st.repos == 0:
        return None
    path = os.path.join(out_dir, f'part-{os.getpid()}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(st.to_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)
    _init(st.lang, st.prune_at)
    return path


def load_tasks(manifest, repos_dir, fold, max_files, max_repos):
    per_repo = collections.defaultdict(list)
    with open(manifest) as f:
        for line in f:
            d = json.loads(line)
            if d['status'] != 'ok' or d['fold'] != fold:
                continue
            per_repo[d['repo']].append(d['path'])
    tasks = []
    for repo in sorted(per_repo):
        files = sorted(per_repo[repo])
        if max_files and len(files) > max_files:
            files = files[:: max(1, len(files) // max_files)][:max_files]
        tasks.append((repo, [os.path.join(repos_dir, repo, p) for p in files]))
    if max_repos:
        tasks = tasks[:max_repos]
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lang', required=True, choices=['go', 'csharp'])
    ap.add_argument('--data', default=os.path.expanduser('~/work/ml-data'))
    ap.add_argument('--manifest')
    ap.add_argument('--repos')
    ap.add_argument('--fold', default='lm')
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=32)
    ap.add_argument('--max-files-per-repo', type=int, default=500)
    ap.add_argument('--max-repos', type=int, default=0)
    ap.add_argument('--prune-at', type=int, default=8_000_000)
    a = ap.parse_args()
    manifest = a.manifest or os.path.join(a.data, a.lang, 'prepared', 'manifest.jsonl')
    repos = a.repos or os.path.join(a.data, a.lang, 'repos')
    t0 = time.time()
    tasks = load_tasks(manifest, repos, a.fold, a.max_files_per_repo, a.max_repos)
    nfiles = sum(len(t[1]) for t in tasks)
    print(f'{len(tasks)} repos, {nfiles} files ({a.fold} fold)', flush=True)
    out_dir = a.out + '.parts'
    os.makedirs(out_dir, exist_ok=True)
    done = 0
    with Pool(a.jobs, initializer=_init, initargs=(a.lang, a.prune_at)) as pool:
        for n in pool.imap_unordered(process_repo, tasks, chunksize=4):
            done += n
            if done % 200_000 < n:
                print(f'  {done}/{nfiles} files, {time.time() - t0:.0f} s', flush=True)
        parts = [p for p in pool.map(flush, [out_dir] * a.jobs * 8, chunksize=1) if p]
    print(f'parsed in {time.time() - t0:.0f} s; merging {len(parts)} parts', flush=True)
    total = State(a.lang, 1 << 62)
    for p in sorted(set(parts)):
        with open(p, 'rb') as f:
            d = pickle.load(f)
        total.docs += d['docs']; total.repos += d['repos']
        for k in ('path_doc', 'name_doc', 'name_path', 'pair', 'decl', 'qual'):
            getattr(total, k).update(d[k])
        os.remove(p)
    os.rmdir(out_dir)
    res = total.to_dict()
    res['stats'] = dict(fold=a.fold, repos=len(tasks), files_listed=nfiles, max_files_per_repo=a.max_files_per_repo,
                        seconds=round(time.time() - t0, 1), sizes={k: len(res[k]) for k in ('path_doc', 'name_doc', 'name_path', 'pair', 'decl', 'qual')})
    with open(a.out, 'wb') as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps(res['stats']), f'docs={total.docs}', flush=True)


if __name__ == '__main__':
    main()
