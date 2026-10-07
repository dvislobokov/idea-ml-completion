#!/usr/bin/env bash
# Poor man's sampling profiler for a running export JVM: profile.sh <samples> [interval-s]; prints the most frequent plugin/platform
# frames (top of stack of threads that are inside io.github.golangsupport or completion code).
N=${1:-40}; I=${2:-1}
pid=$(pgrep -f "JUnitCore io.github.golangsupport.ml" | head -1)
for i in $(seq $N); do /root/work/idea/jbr/bin/jcmd $pid Thread.print 2>/dev/null; sleep $I; done > /root/work/go-psi/logs/profile-raw.txt
python3 - <<'PY'
import re, collections
raw = open("/root/work/go-psi/logs/profile-raw.txt").read()
top = collections.Counter(); deep = collections.Counter(); own = collections.Counter()
for th in raw.split('\n\n'):
    frames = [l.strip()[3:] for l in th.splitlines() if l.strip().startswith("at ")]
    if not frames or not any("io.github.golangsupport" in f for f in frames): continue
    if "java.lang.Thread.run" not in frames[-1] and "EventQueue" not in th: pass
    top[frames[0].split('(')[0]] += 1
    g = [f for f in frames if "io.github" in f]
    deep[g[0].split('(')[0]] += 1
    seen = set()
    for f in g:
        k = f.split('(')[0]
        if k not in seen: seen.add(k); own[k] += 1
print("== top frame"); [print(c, k) for k, c in top.most_common(12)]
print("== first plugin frame"); [print(c, k) for k, c in deep.most_common(15)]
print("== plugin frames on stack (inclusive)"); [print(c, k) for k, c in own.most_common(25)]
PY
