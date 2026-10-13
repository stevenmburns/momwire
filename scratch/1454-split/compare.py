"""Compare two `byte_gate.py` output directories (base, change).

Reports the population, how many decks took the exact-kernel correction, how
many printouts are identical (TIME and SEC. lines masked) and how many are not, and for those that
are not the worst differing line (by the largest relative change of any
number on it). Exit status 1 when any printout differs.
"""

import json
import re
import sys
from pathlib import Path

NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def text(d, sha):
    """A stored printout, less the NEC-4.2 dialect's own timing lines
    (`FILL= .. SEC., FACTOR= .. SEC.`), which `byte_gate.py`'s TIME mask
    does not catch."""
    lines = (Path(d) / f"{sha}.txt").read_text().splitlines()
    return "\n".join(x for x in lines if " SEC." not in x)


def load(d):
    return {
        (r["deck"]): r
        for r in map(json.loads, (Path(d) / "index.jsonl").read_text().splitlines())
    }


def worst_line(a, b):
    """(relative change, base line, change line) of the worst differing line."""
    la, lb = a.splitlines(), b.splitlines()
    if len(la) != len(lb):
        return (float("inf"), f"<{len(la)} lines>", f"<{len(lb)} lines>")
    worst = (0.0, "", "")
    for x, y in zip(la, lb):
        if x == y:
            continue
        nx, ny = NUM.findall(x), NUM.findall(y)
        rel = float("inf")
        if len(nx) == len(ny):
            rel = 0.0
            for p, q in zip(nx, ny):
                p, q = float(p), float(q)
                if p != q:
                    rel = max(rel, abs(p - q) / max(abs(p), abs(q)))
        if rel >= worst[0]:
            worst = (rel, x, y)
    return worst


def main(base, change):
    A, B = load(base), load(change)
    keys = [k for k in A if k in B]
    skipped = [k for k in keys if "skipped" in A[k] or "skipped" in B[k]]
    used = [k for k in keys if k not in skipped]
    same = diff = 0
    raised = sum(
        1
        for k in used
        for sha in A[k]["prints"].values()
        if (Path(base) / f"{sha}.txt").read_text()[:6] == "RAISED"
    )
    worst = (0.0, "", "", "")
    differing_decks = set()
    for k in used:
        for slot, sha in A[k]["prints"].items():
            other = B[k]["prints"].get(slot)
            if other == sha or text(base, sha) == text(change, other):
                same += 1
                continue
            diff += 1
            differing_decks.add(A[k]["path"])
            w = worst_line(text(base, sha), text(change, other))
            if w[0] >= worst[0]:
                worst = (*w, f"{A[k]['path']} [{slot}]")
    by = {}
    for k in used:
        by[A[k]["source"]] = by.get(A[k]["source"], 0) + 1
    print(f"decks compared: {len(used)} {by}; skipped: {len(skipped)}")
    for k in skipped:
        print(f"  skipped {A[k]['path']}: {A[k].get('skipped') or B[k].get('skipped')}")
    for name, R in (("base", A), ("change", B)):
        eng = sum(1 for k in used if R[k]["corrections"])
        tot = sum(R[k]["corrections"] for k in used)
        st = {}
        for k in used:
            for key, v in R[k]["stats"].items():
                st[key] = st.get(key, 0) + v
            for key, v in R[k].get("series", {}).items():
                st[f"series_{key}"] = st.get(f"series_{key}", 0) + v
        print(
            f"{name}: decks taking the correction {eng}, corrections {tot}, stats {st}"
        )
    print(
        f"printouts: {same + diff}  identical: {same}  differing: {diff}  (base printouts that are a raised exception: {raised})"
    )
    if diff:
        print(f"decks with a differing printout: {len(differing_decks)}")
        print(f"worst line (relative change {worst[0]:.3g}) in {worst[3]}:")
        print(f"  base  : {worst[1]}")
        print(f"  change: {worst[2]}")
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
