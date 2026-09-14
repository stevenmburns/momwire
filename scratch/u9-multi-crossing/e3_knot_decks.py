"""U9 Amendment 8c, step 0: write the knot-fed two-node decks (no run), and, with
--check1, capture what antennaknobs builds for momwire from each momwire deck
(geometry only: construction raises before any fill, so no Z) for check 1.

  python e3_knot_decks.py --write
  PYTHONPATH=<momwire src>:<antennaknobs src> python e3_knot_decks.py \\
      --check1 --out e3_check1.json

Geometry is Amendment 8's (`e0_two_node_decks.py`): the rod 2 m down to each
node and the 10 m monopole up from it, radius 1 mm, twice at separation d, GE -1,
GN 2 over soil A, 7.0 MHz. `--write` asserts that every symmetric deck's GW, GE,
GN, FR, XQ and EN cards are byte-identical to Amendment 8's `both` deck at the
same (d, rung).

Every feed is the knot at z = 4.5 m on its monopole's long edge, spelled with an
explicit end field (`EX 0 tag seg 2`): antennaknobs then reads the deck in NEC-5's
dialect and feeds momwire at the point native NEC-5 feeds.

Decks, in a8c_decks/:
  two_node_d{3,5,8,11}_{r1,far3,all3}_{a,b,a0}.nec
      a    port voltages (1, 0.5), knot feeds;
      b    port voltages (0.5, 1), knot feeds;
      a0   a's voltages with I4 = 0: check 2's twin, and check 1's record of how
           the importer reads an undeclared I4 = 0 card;
  ctrl_d3_{r1,far3}_{a,b}.nec
      check 4's control: node 2's monopole is 11 m, so the deck is not
      mirror-symmetric and a port mislabel breaks reciprocity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECK_DIR = HERE / "a8c_decks"
A8_DIR = HERE / "a8_decks"
SEPARATIONS = (3.0, 5.0, 8.0, 11.0)
RUNGS = ("r1", "far3", "all3")
MOMWIRE_RUNGS = ("r1", "far3")
CTRL_D = 3.0
RADIUS = 0.001
FEED_Z = 4.5
TOP_Z = 10.0
CTRL_TOP_Z = 11.0
LONG_SEG_M = 0.5
VOLTS = {"a": (1.0, 0.5), "b": (0.5, 1.0), "a0": (1.0, 0.5)}
POINT_TOL_M = 1e-9
GEOMETRY_CARDS = ("GW", "GE", "GN", "FR", "XQ", "EN")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edges(rung, top_z):
    """[(z0, z1, n, node_adjacent)] for one rod (bottom -> node) and one monopole
    (node -> top): Amendment 8's level-1 grading, 0.5 m segments on the long
    edge at r1."""
    rod = [(-2.0, -0.5, 3, False), (-0.5, -0.1, 2, False), (-0.1, 0.0, 2, True)]
    mono = [
        (0.0, 0.1, 2, True),
        (0.1, 0.5, 2, False),
        (0.5, top_z, round((top_z - 0.5) / LONG_SEG_M), False),
    ]
    if rung == "r1":
        return rod, mono

    def scale(ws):
        return [
            (z0, z1, n if (rung == "far3" and adj) else 3 * n, adj)
            for z0, z1, n, adj in ws
        ]

    return scale(rod), scale(mono)


def stems():
    """(stem, d, rung, control, excitations) for every deck set."""
    out = [
        (f"two_node_d{d:g}_{rung}", d, rung, False, ("a", "b", "a0"))
        for d in SEPARATIONS
        for rung in RUNGS
    ]
    out += [
        (f"ctrl_d{CTRL_D:g}_{rung}", CTRL_D, rung, True, ("a", "b"))
        for rung in MOMWIRE_RUNGS
    ]
    return out


def deck_text(d, rung, control, exc):
    note = ", control: node 2's monopole is 11 m" if control else ""
    lines = [
        f"CM U9 amendment 8c: two-node soil-A deck, d = {d:g} m, {rung}, "
        f"excitation {exc}{note}",
        "CE",
    ]
    tag = 0
    ports = []
    for node, x in enumerate((0.0, d)):
        rod, mono = edges(rung, CTRL_TOP_Z if control and node == 1 else TOP_Z)
        for z0, z1, n, _adj in rod + mono:
            tag += 1
            lines.append(
                f"GW {tag} {n} {x:.6E} 0.000000E+00 {z0:.6E} {x:.6E} 0.000000E+00 {z1:.6E} {RADIUS:.6E}"
            )
        z0, z1, n, _adj = mono[-1]
        k = (FEED_Z - z0) / ((z1 - z0) / n)
        assert abs(k - round(k)) < 1e-9 and 1 <= round(k) < n, (d, rung, node, k)
        ports.append((tag, round(k)))
    lines.append("GE -1")
    lines.append("GN 2 0 0 0 1.300000E+01 5.000000E-03")
    end = 0 if exc == "a0" else 2
    for (t, s), v in zip(ports, VOLTS[exc], strict=True):
        lines.append(f"EX 0 {t} {s} {end} {v:.6E} 0.000000E+00")
    lines.append("FR 0 1 0 0 7.000000E+00 0.000000E+00")
    lines += ["XQ 0", "EN"]
    return "\n".join(lines) + "\n", ports


def geometry_cards(text):
    return [ln for ln in text.splitlines() if ln[:2] in GEOMETRY_CARDS]


def write():
    DECK_DIR.mkdir(exist_ok=True)
    rec = []
    for stem, d, rung, control, excs in stems():
        for exc in excs:
            text, ports = deck_text(d, rung, control, exc)
            if not control:
                a8 = (A8_DIR / f"two_node_d{d:g}_{rung}_both.nec").read_text()
                assert geometry_cards(text) == geometry_cards(a8), (stem, exc)
            path = DECK_DIR / f"{stem}_{exc}.nec"
            path.write_text(text)
            rec.append(
                dict(
                    stem=stem,
                    excitation=exc,
                    d=d,
                    rung=rung,
                    control=control,
                    path=str(path.relative_to(HERE)),
                    sha256=sha(path),
                    ports=[list(p) for p in ports],
                    volts=list(VOLTS[exc]),
                    end=0 if exc == "a0" else 2,
                )
            )
    (HERE / "e3_knot_decks.json").write_text(json.dumps(dict(decks=rec), indent=1))
    print(
        f"{len(rec)} decks written; every symmetric deck's geometry cards equal"
        " amendment 8's"
    )


def point_at(poly, s):
    """The point at arclength s along a polyline."""
    import numpy as np

    v = np.asarray(poly, dtype=float)
    seg = np.linalg.norm(np.diff(v, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    i = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1))
    return v[i] + (s - cum[i]) / seg[i] * (v[i + 1] - v[i])


def canon(v):
    """A JSON-comparable spelling of one solver kwarg."""
    import numpy as np

    if isinstance(v, np.ndarray):
        return canon(v.tolist())
    if isinstance(v, (list, tuple)):
        return [canon(x) for x in v]
    if isinstance(v, np.generic):
        return canon(v.item())
    if isinstance(v, complex):
        return [v.real, v.imag]
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return repr(v)


def solver_kwargs(path):
    """What antennaknobs parsed from one deck, and the kwargs its momwire engine
    builds from it (construction raises before any fill)."""
    from antennaknobs.cli import (
        _GROUND_UNSET,
        file_ground_default,
        make_engine_factory,
    )
    from antennaknobs.file_designs import builder_from_file
    from momwire import bspline
    from momwire.bspline import BSplineSolver

    bspline.below_reach_refusal = lambda *a, **k: None  # AK#1464 stub (Amendment 3)
    captured = []
    real_init = BSplineSolver.__init__

    class Stop(Exception):
        pass

    def grab(self, *a, **kw):
        captured.append(kw)
        raise Stop

    b = builder_from_file(str(path), refine=1)
    BSplineSolver.__init__ = grab
    try:
        make_engine_factory("momwire", file_ground_default(_GROUND_UNSET, b))(
            b()
        ).impedance()
    except Stop:
        pass
    finally:
        BSplineSolver.__init__ = real_init
    return b.file_deck_parsed, captured[0]


def capture(path):
    """`solver_kwargs`, with the served solver's geometry facts."""
    import numpy as np
    from momwire import bspline
    from momwire.bspline import BSplineSolver

    parsed, kw = solver_kwargs(path)
    feeds = list(kw.get("feeds") or [])
    s = BSplineSolver(**kw)
    geom = s._build_geometry()
    b_idx = np.nonzero(s._below_segments(geom))[0]
    obs_b = s._buried_nodes(geom, b_idx)[0]
    d_b = float(kw.get("ground_z", 0.0)) - obs_b[:, 2]
    _r1, th = bspline._pair_extents_below(obs_b[:, 0], obs_b[:, 1], d_b)
    facts = dict(
        nec5_dialect=bool(parsed.nec5_dialect),
        parsed_feeds=[[f.wire, f.seg, getattr(f, "edge", None)] for f in parsed.feeds],
        feeds=[[int(f[0]), float(f[1])] for f in feeds],
        feed_points=[
            point_at(kw["wires"][int(f[0])], float(f[1])).tolist() for f in feeds
        ],
        junction_ports=canon(kw.get("junction_ports")),
        node_gaps=canon(kw.get("node_gaps")),
        port_count=int(s._port_count()),
        crossing_junctions=len(s._crossing_junctions()),
        th_min_deg=math.degrees(th),
        refusal=s.buried_serve_refusal(),
    )
    rest = {k: canon(v) for k, v in kw.items() if k not in ("feeds", "cancel")}
    return facts, rest


def check1(out):
    rows, fails = [], []
    for stem, d, rung, control, _excs in stems():
        if rung not in MOMWIRE_RUNGS:
            continue
        got = {}
        for exc in ("a",) if control else ("a", "a0"):
            path = DECK_DIR / f"{stem}_{exc}.nec"
            try:
                facts, rest = capture(path)
            except Exception as err:  # noqa: BLE001 - a record, not a handler
                fails.append(
                    f"{stem}_{exc}: capture {type(err).__name__}: {err!s:.240}"
                )
                continue
            got[exc] = (facts, rest)
            rows.append(dict(stem=stem, excitation=exc, sha256=sha(path), **facts))
        if "a" not in got:
            continue
        facts = got["a"][0]
        want = [[0.0, 0.0, FEED_Z], [d, 0.0, FEED_Z]]
        bad = []
        if not facts["nec5_dialect"]:
            bad.append("the importer did not read the NEC-5 dialect")
        if len(facts["feeds"]) != 2 or facts["port_count"] != 2:
            bad.append(f"{len(facts['feeds'])} feeds, {facts['port_count']} ports")
        if facts["junction_ports"] is not None or facts["node_gaps"] is not None:
            bad.append("junction or node ports present")
        if len(facts["feed_points"]) != 2 or any(
            math.dist(p, w) > POINT_TOL_M
            for p, w in zip(facts["feed_points"], want, strict=True)
        ):
            bad.append(f"feed points {facts['feed_points']}, want {want} in order")
        if facts["refusal"] is not None or facts["crossing_junctions"] != 2:
            bad.append(
                f"refusal {facts['refusal']!r:.200},"
                f" {facts['crossing_junctions']} crossing junctions"
            )
        if "a0" in got and got["a"][1] != got["a0"][1]:
            keys = [k for k in got["a"][1] if got["a"][1][k] != got["a0"][1].get(k)]
            bad.append(f"kwargs other than feeds differ from the I4 = 0 twin: {keys}")
        fails += [f"{stem}_a: {b}" for b in bad]
    verdict = "FAIL" if fails else "PASS"
    out.write_text(
        json.dumps(dict(check1=verdict, failures=fails, rows=rows), indent=1)
    )
    for r in rows:
        print(
            r["stem"],
            r["excitation"],
            "dialect",
            r["nec5_dialect"],
            "feeds",
            r["feeds"],
            "z",
            [round(p[2], 6) for p in r["feed_points"]],
            "ports",
            r["port_count"],
            f"th_min {r['th_min_deg']:.4f}",
            "refusal",
            r["refusal"] is not None,
        )
    for f in fails:
        print("FAIL", f)
    print(f"check 1: {verdict} ({len(fails)} failures)")


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check1", action="store_true")
    ap.add_argument("--out", type=Path, default=HERE / "e3_check1.json")
    args = ap.parse_args()
    if args.write:
        write()
    else:
        check1(args.out)


if __name__ == "__main__":
    main()
