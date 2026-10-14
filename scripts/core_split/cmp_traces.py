"""Compare two traced runs, aligned by INPUT, not by call position.

    python cmp_traces.py <traceA> <traceB>

For every (function, input hash) seen in both runs the output hashes are
compared. A deterministic kernel given the same bytes must return the same
bytes, so any difference is the build moving bits. Alignment by input is
immune to xdist scheduling and warm caches changing which calls a test makes.
Functions whose output varies for a fixed input WITHIN one run (process
counters) are reported separately as stateful.
"""

import collections
import glob
import json
import os
import sys


def load(d):
    m = collections.defaultdict(set)
    n = 0
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        for line in open(f):
            node, idx, fn, hi, ho = json.loads(line)
            m[(fn, hi)].add(ho)
            n += 1
    return m, n


(a, na), (b, nb) = load(sys.argv[1]), load(sys.argv[2])
stateful = {k[0] for k, v in list(a.items()) + list(b.items()) if len(v) > 1}
common = [k for k in a if k in b and k[0] not in stateful]
per = collections.defaultdict(lambda: [0, 0])
moved = []
for k in common:
    same = a[k] == b[k]
    per[k[0]][0 if same else 1] += 1
    if not same:
        moved.append(k)
fa = {k[0] for k in a}
print(
    f"calls A={na} B={nb}; distinct (fn,input) A={len(a)} B={len(b)}; compared={len(common)}"
)
print(
    f"functions called A={len(fa)}, compared={len(per)}, stateful(excluded)={sorted(stateful)}"
)
print("functions with MOVED outputs:", {f: v[1] for f, v in per.items() if v[1]})
print("not compared (no shared input):", sorted(fa - set(per) - stateful))
# The stateful ones, judged leniently: for a shared input, do the two runs
# share at least one output? (A single-valued key in both runs is strict.)
st = collections.defaultdict(lambda: [0, 0, 0])
for k in a:
    if k in b and k[0] in stateful:
        if len(a[k]) == 1 and len(b[k]) == 1:
            st[k[0]][0 if a[k] == b[k] else 2] += 1
        else:
            st[k[0]][1 if a[k] & b[k] else 2] += 1
print("stateful fns [strict-same, multi-valued-overlapping, NO-overlap]:")
for f, v in sorted(st.items()):
    print("  ", f, v)
