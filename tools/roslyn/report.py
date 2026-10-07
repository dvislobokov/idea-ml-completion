#!/usr/bin/env python3
"""Headroom report for a `cmlroslyn filter` JSONL: coverage of the semantic model, unresolved-name classification,
the filter headroom table, receiver statistics and raw examples.

    python3 tools/roslyn/report.py filter.jsonl [--bpe tokenizer/cs-16384.bpe] [--examples 10] [--seed 1]
"""
import argparse, collections, json, random, re, sys, os

W = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')

def pct(a, b): return '—' if not b else f'{100.0 * a / b:.1f} %'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('jsonl'); ap.add_argument('--bpe'); ap.add_argument('--examples', type=int, default=10); ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    R = [json.loads(l) for l in open(a.jsonl)]
    ok = [r for r in R if r.get('status') == 'ok']
    P = print
    P(f'positions {len(R)}, with a verdict {len(ok)}; status: ' + ', '.join(f'{k} {v}' for k, v in collections.Counter(r.get('status') for r in R).items()))
    tr = [r for r in ok if r['true_resolvable']]
    P(f'true line resolves: {len(tr)} / {len(ok)} = {pct(len(tr), len(ok))}; generated line resolves: {pct(sum(r["gen_resolvable"] for r in ok), len(ok))}')
    nt = [r for r in ok if not r['true_resolvable']]
    P(f'true line does NOT resolve: {len(nt)}: ' + ', '.join(f'{k} {v}' for k, v in collections.Counter(
        ('member / receiver known' if r['true_recv_known'] else 'member / receiver unknown') if r['true_first_is_member'] else 'simple name' for r in nt).most_common()))
    P('top-30 first unresolved names (name, kind, count, repos):')
    c = collections.defaultdict(lambda: [0, collections.Counter()])
    for r in nt:
        k = (r['true_first_unresolved'], 'member' if r['true_first_is_member'] else 'simple'); c[k][0] += 1; c[k][1][r['repo']] += 1
    for (n, k), (cnt, reps) in sorted(c.items(), key=lambda kv: -kv[1][0])[:30]:
        P(f'  {n:<24} {k:<7} {cnt:>3}  ' + ', '.join(f'{x.split("__")[-1]} {v}' for x, v in reps.most_common(3)))
    P('repos with most unresolved true lines: ' + ', '.join(f'{k} {v}/{sum(1 for r in ok if r["repo"] == k)}' for k, v in collections.Counter(r['repo'] for r in nt).most_common(10)))
    # receivers
    ma = sum(r['true_members'] for r in ok); mk = sum(r['true_members_recv_known'] for r in ok)
    P(f'member accesses in true lines: {ma}, receiver type known: {mk} ({pct(mk, ma)})')
    ad = [r for r in ok if r['kind'] == 'after-dot']
    adk = [r for r in ad if r['true_members'] > 0 and r['true_members_recv_known'] == r['true_members']]
    P(f'after-dot positions: {len(ad)}; all receivers known: {len(adk)} ({pct(len(adk), len(ad))}); true line resolves there: {pct(sum(r["true_resolvable"] for r in ad), len(ad))}')
    cand = [r['gen_candidates'] for r in ad if r.get('gen_recv_known') and r['gen_candidates']]
    if cand:
        cand.sort(); P(f'member candidates on the receiver (after-dot, generated line fails on a known receiver): n={len(cand)}, mean {sum(cand)/len(cand):.0f}, median {cand[len(cand)//2]}')
    # headroom table
    P('')
    P('Headroom (shown = conf_prod >= threshold; filter drops a line only where the generated line does not resolve; "complete model" = only where the true line resolves):')
    P('| conf >= | shown | exact | +filter shown | exact | wrong removed | right removed | +filter (complete model only) shown | exact | wrong removed | right removed |')
    P('|---|---|---|---|---|---|---|---|---|---|---|')
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        s = [r for r in ok if r['conf_prod'] >= t]; right = sum(r['exact'] for r in s)
        k = [r for r in s if r['gen_resolvable']]; kr = sum(r['exact'] for r in k)
        i = [r for r in s if r['gen_resolvable'] or not r['true_resolvable']]; ir = sum(r['exact'] for r in i)
        P(f'| {t:.1f} | {len(s)} ({pct(len(s), len(ok))}) | {pct(right, len(s))} | {len(k)} ({pct(len(k), len(ok))}) | {pct(kr, len(k))} | {len(s)-right-(len(k)-kr)} of {len(s)-right} | {right-kr} of {right} | '
          f'{len(i)} ({pct(len(i), len(ok))}) | {pct(ir, len(i))} | {len(s)-right-(len(i)-ir)} of {len(s)-right} | {right-ir} of {right} |')
    P('')
    P('Right lines removed by the filter at conf >= 0.7 (why the correct line does not resolve):')
    rr = [r for r in ok if r['conf_prod'] >= 0.7 and r['exact'] and not r['gen_resolvable']]
    for k, v in collections.Counter((r['gen_first_unresolved'], 'member' if r['gen_first_is_member'] else 'simple', r['repo'].split('__')[-1]) for r in rr).most_common(15): P(f'  {k[0]:<24} {k[1]:<7} {k[2]:<28} {v}')
    if 'ctx_chars' in ok[0]:
        c = [r for r in ok if 'ctx_chars' in r]
        line = f'context block: mean {sum(r["ctx_chars"] for r in c)/len(c):.0f} chars'
        if a.bpe:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'nn', 'tokenizer'))
            import cmlbpe
            voc = cmlbpe.Vocab.load(a.bpe); toks = sorted(len(voc.encode(r['ctx'])) for r in c)
            line += f', BPE tokens mean {sum(toks)/len(toks):.0f}, median {toks[len(toks)//2]}, p90 {toks[int(len(toks)*0.9)]}'
        P(''); P(line + f', empty {pct(sum(r["ctx_chars"]==0 for r in c), len(c))}')
        ids = sum(r['ids'] for r in c); P(f'identifiers of the true rest: {ids}; in the file prefix {pct(sum(r["ids_in_prefix"] for r in c), ids)}, only in the context block {pct(sum(r["ids_in_ctx_only"] for r in c), ids)}, nowhere {pct(sum(r["ids_nowhere"] for r in c), ids)}')
        wi = [r for r in c if r['ids'] > 0]
        P(f'positions with identifiers {len(wi)}: some identifier only in the context {pct(sum(r["ids_in_ctx_only"]>0 for r in wi), len(wi))}, context needed and sufficient {pct(sum(r["ids_in_ctx_only"]>0 and r["ids_nowhere"]==0 for r in wi), len(wi))}')
    # examples
    random.seed(a.seed)
    P(''); P(f'{a.examples} raw examples (conf >= 0.5, true line resolves; mixed verdicts):')
    pool = [r for r in ok if r['conf_prod'] >= 0.5 and r['true_resolvable']]
    groups = [[r for r in pool if not r['exact'] and not r['gen_resolvable']], [r for r in pool if r['exact'] and not r['gen_resolvable']], [r for r in pool if not r['exact'] and r['gen_resolvable']], [r for r in pool if r['exact'] and r['gen_resolvable']]]
    want = [3, 3, 2, 2]
    for g, n in zip(groups, want):
        for r in random.sample(g, min(n, len(g))):
            why = '' if r['gen_resolvable'] else f" (unresolved `{r['gen_first_unresolved']}`" + (f" on {r['gen_recv_type']}" if r['gen_recv_known'] else (', receiver unknown' if r['gen_first_is_member'] else ', simple name')) + ')'
            P(f"- {r['repo']} {r['path']}:{r['line']} conf {r['conf_prod']:.2f} {'EXACT' if r['exact'] else 'wrong'}, filter {'keeps' if r['gen_resolvable'] else 'DROPS'}{why}\n    true: `{r['true'].strip()}`\n    gen:  `{r['gen'].strip()}`")

if __name__ == '__main__': main()
