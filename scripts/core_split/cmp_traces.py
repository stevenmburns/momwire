"""Compare two traced test runs (mwtrace_plugin.py), aligned by INPUT.

    python scripts/core_split/cmp_traces.py TRACE_A TRACE_B [--per-function REGEX]

For every (function, input hash) seen in both runs the output hashes are
compared. A deterministic kernel given the same bytes must return the same
bytes, so any difference is the build moving bits. Aligning by input rather
than by call position is immune to xdist scheduling and to warm caches
changing which calls a test makes.

A function whose output varies for one input WITHIN a run (a kernel reading a
cache object or other state) is "stateful": it is excluded from the strict
comparison and judged leniently instead (do the two runs share an output for
a shared input?). A stateful function only ever called with one input, i.e.
with no arguments, is a getter of a process counter (`*_calls`,
`somm_proj_lane_pairs`): its value is the history of the worker that called
it, not a computation, so it is listed and never judged.

Exit status 1 when any strictly compared output moved, or a stateful function
other than a getter shares no output for a shared input; 0 otherwise.
--per-function prints, for every compared function matching REGEX, how many
distinct inputs it was compared on.
"""

import argparse
import collections
import glob
import json
import os
import re
import sys


def load(directory):
    """{(function, input hash): {output hashes}} and the call count."""
    seen = collections.defaultdict(set)
    calls = 0
    files = glob.glob(os.path.join(directory, "*.jsonl"))
    if not files:
        sys.exit(f"{directory}: no trace files")
    for path in files:
        with open(path) as fh:
            for line in fh:
                _node, _idx, fn, h_in, h_out = json.loads(line)
                seen[(fn, h_in)].add(h_out)
                calls += 1
    return seen, calls


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("trace_a")
    p.add_argument("trace_b")
    p.add_argument("--per-function", metavar="REGEX")
    args = p.parse_args()
    (a, na), (b, nb) = load(args.trace_a), load(args.trace_b)

    stateful = {k[0] for k, v in [*a.items(), *b.items()] if len(v) > 1}
    inputs = collections.defaultdict(set)
    for fn, h_in in [*a, *b]:
        inputs[fn].add(h_in)
    getters = {f for f in stateful if len(inputs[f]) == 1}
    common = [k for k in a if k in b and k[0] not in stateful]
    per = collections.defaultdict(lambda: [0, 0])  # [same, moved]
    for k in common:
        per[k[0]][0 if a[k] == b[k] else 1] += 1
    moved = {f: v[1] for f, v in per.items() if v[1]}
    called_a = {k[0] for k in a}

    print(f"calls: A {na}, B {nb}")
    print(f"distinct (function, input): A {len(a)}, B {len(b)}, compared {len(common)}")
    print(f"functions: called in A {len(called_a)}, compared {len(per)}")
    print("not compared (no shared input):", sorted(called_a - set(per) - stateful))
    print("MOVED:", moved or "none")

    print("counter getters (not judged):", sorted(getters))
    lenient = collections.defaultdict(lambda: [0, 0, 0])  # strict-same, overlap, none
    for k in a:
        if k in b and k[0] in stateful - getters:
            if len(a[k]) == 1 and len(b[k]) == 1:
                lenient[k[0]][0 if a[k] == b[k] else 2] += 1
            else:
                lenient[k[0]][1 if a[k] & b[k] else 2] += 1
    print(
        "stateful [single-valued and same, multi-valued and overlapping, NO overlap]:"
    )
    for f, v in sorted(lenient.items()):
        print(f"   {f} {v}")

    if args.per_function:
        print(f"distinct inputs compared, functions matching {args.per_function!r}:")
        for f in sorted(per):
            if re.search(args.per_function, f):
                same, mv = per[f]
                print(f"   {f:44s} {same + mv:8d}  moved {mv}")
        for f in sorted(lenient):
            if re.search(args.per_function, f):
                print(f"   {f:44s} {sum(lenient[f]):8d}  (stateful, lenient)")

    bad = bool(moved) or any(v[2] for v in lenient.values())
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
