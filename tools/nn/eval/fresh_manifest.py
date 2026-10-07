"""Fresh-repository eval set: repositories created on/after CUT (after the teachers' release), not in the lm fold (so our own
models never saw them either), >= MIN_LINES kept lines. Writes <lang>/prepared/manifest-fresh.jsonl with fold := test."""
import json, sys, collections
lang, cut, min_lines = sys.argv[1], sys.argv[2], int(sys.argv[3])
cat = {}
for l in open(f'/root/work/ml-data/catalog/{lang}-20.jsonl'):
    d = json.loads(l); cat[d['full_name'].replace('/', '__')] = d
src = f'/root/work/ml-data/{lang}/prepared/manifest.jsonl'
lines = collections.Counter(); files = collections.Counter(); fold = {}
rows = []
for l in open(src):
    r = json.loads(l)
    if r.get('status') != 'ok': continue
    c = cat.get(r['repo'])
    if not c or c['created_at'] < cut or r['fold'] == 'lm': continue
    lines[r['repo']] += r.get('lines', 0); files[r['repo']] += 1; fold[r['repo']] = r['fold']
    rows.append(r)
keep = {k for k, v in lines.items() if v >= min_lines}
n = 0
with open(f'/root/work/ml-data/{lang}/prepared/manifest-fresh.jsonl', 'w') as f:
    for r in rows:
        if r['repo'] in keep:
            r['fold'] = 'test'; f.write(json.dumps(r) + '\n'); n += 1
print(f'{lang}: created >= {cut}, not lm, >= {min_lines} lines: {len(keep)} repos, {n} files, {sum(lines[k] for k in keep):,} lines '
      f'(folds: {collections.Counter(fold[k] for k in keep)})')
for k in sorted(keep, key=lambda k: -lines[k])[:6]:
    print(f'   {k:45} {cat[k]["created_at"][:10]} {cat[k]["stars"]:5} stars {files[k]:5} files {lines[k]:8,} lines')
