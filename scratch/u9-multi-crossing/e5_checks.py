"""U9 Amendment 8c: one step's pre-Z checks, from e4's JSONL. A missing,
duplicated or errored row is a failure, never a skip (the D3 lesson). No gated Z
is read here: C8c reads only the sign of Re Z12.

  python e5_checks.py nec5     checks 2 and 3, and check 4's NEC-5 half
  python e5_checks.py momwire  C8b, and check 4's momwire half
  python e5_checks.py same     C8b and C8d on the corner-omitted rows, then C8c
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECK_DIR = HERE / "a8c_decks"
JSONL = HERE / "e4_knot.jsonl"
SEPARATIONS = (3, 5, 8, 11)
SYM = [f"two_node_d{d}_{r}" for d in SEPARATIONS for r in ("r1", "far3", "all3")]
CTRL = ["ctrl_d3_r1", "ctrl_d3_far3"]
RECIP_NEC5 = 1e-2
RECIP_MW = 1e-9
ASYM_MIN = 10 * RECIP_NEC5
V_TOL = 1e-4
ZIN_TOL = 1e-3


def load():
    rows = {}
    if not JSONL.exists():
        return rows
    for line in JSONL.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows.setdefault((r["stem"], r["engine"], r["corner"]), []).append(r)
    return rows


def one(rows, key, fails):
    got = rows.get(key, [])
    if len(got) != 1:
        fails.append(f"{key}: {len(got)} rows, expected 1")
        return None
    if "error" in got[0]:
        fails.append(f"{key}: error {got[0]['error'][:200]}")
        return None
    return got[0]


def cplx(v):
    return complex(v[0], v[1])


def ys(row):
    """[Y11, Y12, Y21, Y22]."""
    return [cplx(v) for v in row["y"]]


def recip(y):
    return abs(y[1] - y[2]) / abs(y[1])


def asym(y):
    return abs(y[0] - y[3]) / abs(y[1])


def tag_offsets(path):
    """Segments before each tag, in GW order: NEC-5 prints absolute segments."""
    off, n = {}, 0
    for line in path.read_text().splitlines():
        f = line.split()
        if f and f[0] == "GW":
            off[int(f[1])] = n
            n += int(f[2])
    return off


def printed(run):
    return sorted(
        [r["tag"], r["seg"], r["sub"], *r["tokens"]] for r in run["input_parameters"]
    )


def check3(row, fails, rec):
    """One input-parameters row per EX card, at that card's knot, printing the
    card's voltage, with Z = V / I."""
    for exc, run in row["runs"].items():
        where = f"check 3 {row['stem']} {exc}"
        off = tag_offsets(DECK_DIR / f"{row['stem']}_{exc}.nec")
        ip = run["input_parameters"]
        if len(ip) != len(run["ex"]):
            fails.append(f"{where}: {len(ip)} rows for {len(run['ex'])} EX cards")
        for tag, seg, _end, v_card in run["ex"]:
            match = [r for r in ip if r["tag"] == tag]
            if len(match) != 1:
                fails.append(f"{where}: {len(match)} rows for tag {tag}")
                continue
            r = match[0]
            vc, vp, ic, zp = cplx(v_card), cplx(r["v"]), cplx(r["i"]), cplx(r["z"])
            if ic == 0 or zp == 0:
                fails.append(f"{where}: tag {tag} prints I = {ic} and Z = {zp}")
                continue
            v_err = abs(vp - vc) / abs(vc)
            zin_err = abs(vp / ic - zp) / abs(zp)
            abs_seg = off[tag] + seg
            rec.append(
                dict(
                    stem=row["stem"],
                    excitation=exc,
                    tag=tag,
                    printed_seg=r["seg"],
                    ex_abs_seg=abs_seg,
                    sub=r["sub"],
                    v_err=v_err,
                    zin_err=zin_err,
                )
            )
            if r["seg"] not in (abs_seg, abs_seg + 1):
                fails.append(f"{where}: tag {tag} printed at segment {r['seg']}")
            if v_err > V_TOL:
                fails.append(f"{where}: tag {tag} voltage error {v_err:.3e}")
            if zin_err > ZIN_TOL:
                fails.append(f"{where}: tag {tag} |V/I - Z|/|Z| = {zin_err:.3e}")


def nec5_step(rows, fails, rec):
    rec.update(check2=[], check3=[], check4_nec5=[], c8a_symmetric_info={})
    for stem in SYM + CTRL:
        row = one(rows, (stem, "nec5", "cross"), fails)
        if row is None:
            continue
        want = ("a", "b") if stem in CTRL else ("a", "b", "a0")
        missing = [e for e in want if e not in row["runs"]]
        if missing:
            fails.append(f"{stem}: NEC-5 runs {missing} missing")
        check3(row, fails, rec["check3"])
        y = ys(row)
        if stem in CTRL:
            r, s = recip(y), asym(y)
            rec["check4_nec5"].append(dict(stem=stem, reciprocity=r, asymmetry=s))
            if r > RECIP_NEC5:
                fails.append(f"check 4 {stem}: NEC-5 reciprocity {r:.3e}")
            if s < ASYM_MIN:
                fails.append(f"check 4 {stem}: NEC-5 asymmetry {s:.3e}")
            continue
        rec["c8a_symmetric_info"][stem] = recip(y)
        if "a" in row["runs"] and "a0" in row["runs"]:
            same = printed(row["runs"]["a"]) == printed(row["runs"]["a0"])
            rec["check2"].append(dict(stem=stem, identical=same))
            if not same:
                fails.append(f"check 2 {stem}: I4 = 2 and I4 = 0 print differently")
    c3 = rec["check3"]
    return dict(
        check3_rows=len(c3),
        worst_v_err=max((r["v_err"] for r in c3), default=None),
        worst_zin_err=max((r["zin_err"] for r in c3), default=None),
        printed_seg_is_ex_abs_seg=all(r["printed_seg"] == r["ex_abs_seg"] for r in c3),
        subs=sorted({r["sub"] for r in c3}),
        check2_identical=sum(r["identical"] for r in rec["check2"]),
        check2_rows=len(rec["check2"]),
        check4_nec5=rec["check4_nec5"],
    )


def momwire_step(rows, fails, rec):
    rec.update(c8b=[], check4_momwire=[])
    for stem in [s for s in SYM if not s.endswith("_all3")] + CTRL:
        row = one(rows, (stem, "momwire", "cross"), fails)
        if row is None:
            continue
        y = ys(row)
        r = recip(y)
        rec["c8b"].append(dict(stem=stem, reciprocity=r))
        if r > RECIP_MW:
            fails.append(f"C8b {stem}: momwire reciprocity {r:.3e}")
        if stem in CTRL:
            s = asym(y)
            rec["check4_momwire"].append(dict(stem=stem, reciprocity=r, asymmetry=s))
            if s < ASYM_MIN:
                fails.append(f"check 4 {stem}: momwire asymmetry {s:.3e}")
    return dict(
        c8b_rows=len(rec["c8b"]),
        c8b_worst=max((r["reciprocity"] for r in rec["c8b"]), default=None),
        check4_momwire=rec["check4_momwire"],
    )


def same_step(rows, fails, rec):
    rec.update(c8b_same=[], c8d=[], c8c=[])
    for d in SEPARATIONS:
        stem = f"two_node_d{d}_r1"
        row = one(rows, (stem, "momwire", "same"), fails)
        if row is None:
            continue
        r = recip(ys(row))
        dropped = row["cross_node_pairs_dropped"]
        loops = row.get("corner_calls_by_loop")
        rec["c8b_same"].append(dict(stem=stem, reciprocity=r))
        rec["c8d"].append(dict(stem=stem, dropped=dropped, by_loop=loops))
        if r > RECIP_MW:
            fails.append(f"C8b {stem} same: momwire reciprocity {r:.3e}")
        if dropped <= 0:
            fails.append(f"C8d {stem}: the corner patch dropped {dropped} pairs")
        if not loops:
            fails.append(f"C8d {stem}: no per-loop corner count")
        # A loop that reaches the corner but zeroes nothing fails, even when
        # the other loop's zeroes make the total pass (Laptop-builder review).
        for loop, n in (loops or {}).items():
            if n["calls"] > 0 and n["zeroed"] == 0:
                fails.append(
                    f"C8d {stem}: {loop} reached the corner {n['calls']} times"
                    " and zeroed no pair"
                )
    if not fails:  # C8c is read only after C8d passes
        for d in SEPARATIONS:
            stem = f"two_node_d{d}_r1"
            m = one(rows, (stem, "momwire", "cross"), fails)
            n = one(rows, (stem, "nec5", "cross"), fails)
            if m is None or n is None:
                continue
            agree = (cplx(m["z"][1]).real > 0) == (cplx(n["z"][1]).real > 0)
            rec["c8c"].append(dict(d=d, sign_re_z12_agrees=agree))
            if not agree:
                fails.append(f"C8c d = {d} m: sign(Re Z12) disagrees")
    return dict(
        c8b_same_worst=max((r["reciprocity"] for r in rec["c8b_same"]), default=None),
        c8d=rec["c8d"],
        c8c=rec["c8c"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("nec5", "momwire", "same"))
    args = ap.parse_args()
    rows = load()
    fails, rec = [], {}
    step = {"nec5": nec5_step, "momwire": momwire_step, "same": same_step}
    summary = step[args.step](rows, fails, rec)
    verdict = "FAIL" if fails else "PASS"
    (HERE / f"e5_checks_{args.step}.json").write_text(
        json.dumps(
            dict(step=args.step, verdict=verdict, failures=fails, records=rec),
            indent=1,
        )
    )
    for f in fails:
        print("FAIL", f)
    print(json.dumps(summary, indent=1))
    print(f"{args.step}: {verdict} ({len(fails)} failures)")


if __name__ == "__main__":
    main()
