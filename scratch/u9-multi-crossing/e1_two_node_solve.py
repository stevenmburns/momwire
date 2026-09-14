"""U9 Amendment 8: the 2x2 Z matrix of one two-node deck, from one engine, as a
JSONL row.

  PYTHONPATH=<momwire src>:<antennaknobs src> python e1_two_node_solve.py \\
      --d 3 --rung r1 --engine momwire [--corner same] --out F.jsonl
  PYTHONPATH=<antennaknobs src> python e1_two_node_solve.py \\
      --d 3 --rung r1 --engine nec5 --nec5-exe PATH --out F.jsonl

momwire: the solver kwargs antennaknobs' momwire engine builds from the `both`
deck (the U2 path; AK#1464's vertex preflight stubbed for construction), then
`BSplineSolver(**kw).compute_port_solution().y`. `--corner same` drops the
cross-node corner pairs (the (b) probe's `same` mode) for Amendment 8's
non-vacuity guard, and records how many pairs it dropped.

NEC-5: native runs of the `p1` and `p2` decks (one EX each, the other port a
plain segment), the segment-centre current at both port segments read off each
printout with antennaknobs' own parser: Y[i][j] = I_i / V_j. Printout output
only (the licence courtesy rule).

Z = inv(Y) either way; reciprocity |Y12 - Y21| / |Y12| is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DECK_DIR = HERE / "a8_decks"


def deck_path(d, rung, excite):
    return DECK_DIR / f"two_node_d{d:g}_{rung}_{excite}.nec"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ports_of(path):
    """The EX (tag, seg) pairs of a deck, in card order."""
    out = []
    for line in path.read_text().splitlines():
        f = line.split()
        if f and f[0] == "EX":
            out.append((int(f[2]), int(f[3])))
    return out


def momwire_y(d, rung, corner):
    import momwire
    from momwire import _crossing_fill, _near_interface, bspline
    from momwire.bspline import BSplineSolver

    bspline.below_reach_refusal = lambda *a, **k: None  # AK#1464 stub
    from antennaknobs.cli import _GROUND_UNSET, file_ground_default, make_engine_factory
    from antennaknobs.file_designs import builder_from_file

    captured = []
    real_init = BSplineSolver.__init__

    class Stop(Exception):
        pass

    def capture(self, *a, **kw):
        captured.append(kw)
        raise Stop

    path = deck_path(d, rung, "both")
    BSplineSolver.__init__ = capture
    try:
        b = builder_from_file(str(path), refine=1)
        make_engine_factory("momwire", file_ground_default(_GROUND_UNSET, b))(
            b()
        ).impedance()
    except Stop:
        pass
    finally:
        BSplineSolver.__init__ = real_init
    kw = captured[0]
    dropped = [0]
    if corner == "same":
        real_eac = _crossing_fill._ends_and_corner

        def eac(
            ctx,
            A,
            B,
            eps_t,
            k_p,
            c1,
            gz,
            memo=None,
            *,
            corner=True,
            test_ends=True,
            source_ends=True,
        ):
            t_ab = real_eac(
                ctx,
                A,
                B,
                eps_t,
                k_p,
                c1,
                gz,
                memo=memo,
                corner=False,
                test_ends=test_ends,
                source_ends=source_ends,
            )
            if not corner:
                return t_ab
            a_wire = float(ctx.a_wire)
            v = None
            for pt_a, sig_a, fv_a in A["ends"]:
                if abs(pt_a[2] - gz) > 1e-12:
                    continue
                for pt_b, sig_b, fv_b in B["ends"]:
                    if abs(pt_b[2] - gz) > 1e-12:
                        continue
                    if (
                        float(np.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1]))
                        >= _crossing_fill._SAME_NODE_RHO
                    ):
                        dropped[0] += 1
                        continue
                    if v is None:
                        v = complex(
                            _near_interface.six_point(
                                eps_t,
                                k_p,
                                a_wire,
                                0.0,
                                0.0,
                                rtol=_crossing_fill._CORNER_RTOL,
                            )[1]
                        )
                    nza, nzb = np.flatnonzero(fv_a), np.flatnonzero(fv_b)
                    t_ab[np.ix_(nza, nzb)] += (-sig_a * sig_b * c1 * v) * np.outer(
                        fv_a[nza], fv_b[nzb]
                    )
            return t_ab

        _crossing_fill._ends_and_corner = eac
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = np.asarray(BSplineSolver(**kw).compute_port_solution().y, dtype=complex)
    src = Path(momwire.__file__).resolve().parent
    return y, dict(
        momwire=str(src),
        feeds=[list(f[:2]) for f in (kw.get("feeds") or [])],
        corner=corner,
        cross_node_pairs_dropped=dropped[0],
        deck_sha256=sha(path),
    )


def nec5_y(d, rung, exe):
    from antennaknobs.engines.nec5 import NEC5Engine, run_deck

    ports = ports_of(deck_path(d, rung, "both"))
    y = np.zeros((2, 2), dtype=complex)
    hashes = {}
    zin = {}
    for j, excite in enumerate(("p1", "p2")):
        path = deck_path(d, rung, excite)
        hashes[excite] = sha(path)
        text = run_deck(exe, path.read_text(), timeout=3600)
        cur = NEC5Engine._parse_wire_currents(text)[0]
        for i, (tag, seg) in enumerate(ports):
            y[i, j] = cur[tag][seg - 1] / 1.0
        rows = NEC5Engine._parse_input_parameters(text)[0]
        zin[excite] = [[t, s, z.real, z.imag] for t, s, z in rows]
    return y, dict(nec5_exe=exe, ports=ports, deck_sha256=hashes, input_parameters=zin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=float, required=True)
    ap.add_argument("--rung", choices=("r1", "far3", "all3"), required=True)
    ap.add_argument("--engine", choices=("momwire", "nec5"), required=True)
    ap.add_argument("--corner", choices=("cross", "same"), default="cross")
    ap.add_argument("--nec5-exe", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rec = dict(
        d=args.d,
        rung=args.rung,
        engine=args.engine,
        t_start=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    t0 = time.perf_counter()
    try:
        if args.engine == "momwire":
            y, info = momwire_y(args.d, args.rung, args.corner)
        else:
            y, info = nec5_y(args.d, args.rung, args.nec5_exe)
        z = np.linalg.inv(y)
        rec.update(info)
        rec["y"] = [[v.real, v.imag] for v in y.ravel()]
        rec["z"] = [[v.real, v.imag] for v in z.ravel()]
        rec["y_reciprocity"] = float(abs(y[0, 1] - y[1, 0]) / abs(y[0, 1]))
    except Exception as exc:  # noqa: BLE001 - a record, not a handler
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    rec["maxrss_mb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
    )
    with args.out.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec)[:600])


if __name__ == "__main__":
    main()
