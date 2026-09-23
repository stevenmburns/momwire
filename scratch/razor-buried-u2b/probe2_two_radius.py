"""U2b-2 gates on the real constructor: the two-radius crossing node.

a  route: scope passes, each cross block at its source radius, node term run
b  eps~=1 collapse per block against razor's free-space fill, by rule
c  node-term radius sensitivity in soil (the regulariser of a radius-free
   quantity): per-wing source rule vs one radius vs a/10, a/100, 10a
d  soil radius-response differential vs bspline at x4 (rise/2, rise/4, top/2)
   with red controls (observer cross blocks; single-radius node term)
e  reciprocity ladder, graded two-radius rod, asymmetric ports
f  convergence onto bspline: two-radius rod, #1140 deck
"""

import json
import sys
import time

import numpy as np
from common import BSplineSolver, RazorSolver, crossing_deck, refined

from momwire import _crossing_fill

sys.path.insert(0, "../../tests")
from test_crossing_serve_524 import above_side_deck  # noqa: E402

which = sys.argv[1] if len(sys.argv) > 1 and __name__ == "__main__" else "none"
A = 0.25e-3  # the U5 rod's radius
_f, _r = (
    _crossing_fill.cross_complete_block,
    _crossing_fill.cross_complete_block_reversed,
)
_nc = RazorSolver._crossing_node_charges


def out(**kw):
    print(json.dumps(kw, default=str), flush=True)


def strip(d):
    return {k: v for k, v in d.items() if k != "junctions"}


def rod(m=1, a_above=A, a_below=A, **kw):
    return refined(crossing_deck(2, wire_radius=[a_below, a_above], **kw), m)


def zr(d, **kw):
    return complex(
        RazorSolver(**strip(d), nec5_quadrature=True, **kw).compute_impedance()[0]
    )


def zb(d):
    return complex(BSplineSolver(**d).compute_impedance()[0])


class Rule:
    """Context manager forcing a cross-block radius rule and/or a node-term
    radius rule. None = shipped code."""

    def __init__(self, cross=None, node=None):
        self.cross, self.node = cross, node

    def __enter__(self):
        c, n = self.cross, self.node
        if c is not None:
            a_f, a_r = c

            def f(ctx, *a, **k):
                return _f(ctx._replace(a_wire=a_f), *a, **k)

            def r(ctx, *a, **k):
                return _r(ctx._replace(a_wire=a_r), *a, **k)

            _crossing_fill.cross_complete_block = f
            _crossing_fill.cross_complete_block_reversed = r
        if n is not None:

            def nc(self_, geom, *a, **k):
                orig = self_._seg_radius
                seg = orig(geom)
                self_._seg_radius = lambda g: n(seg)
                try:
                    return _nc(self_, geom, *a, **k)
                finally:
                    del self_._seg_radius

            RazorSolver._crossing_node_charges = nc
        return self

    def __exit__(self, *e):
        _crossing_fill.cross_complete_block = _f
        _crossing_fill.cross_complete_block_reversed = _r
        RazorSolver._crossing_node_charges = _nc


# ------------------------------------------------------------------ a
if which in ("all", "a"):
    calls = []
    nterm = []

    def f(ctx, *a, **k):
        calls.append(("fwd", ctx.a_wire))
        return _f(ctx, *a, **k)

    def r(ctx, *a, **k):
        calls.append(("rev", ctx.a_wire))
        return _r(ctx, *a, **k)

    def nc(self_, geom, tents, *a, **k):
        nterm.append(len(tents))
        return _nc(self_, geom, tents, *a, **k)

    _crossing_fill.cross_complete_block = f
    _crossing_fill.cross_complete_block_reversed = r
    RazorSolver._crossing_node_charges = nc
    for name, d in (
        ("rod rise/2", rod(1, A, A / 2)),
        ("rod top/2", rod(1, A / 2, A)),
        ("1140", above_side_deck(wire_radius=[0.002, 0.001, 0.004])),
    ):
        calls.clear()
        nterm.clear()
        s = RazorSolver(**strip(d), nec5_quadrature=True)
        z = complex(s.compute_impedance()[0])
        out(
            gate="a",
            deck=name,
            crossing=s._crossing,
            calls=calls[:],
            node_term=nterm[:],
            z=z,
        )
    _crossing_fill.cross_complete_block = _f
    _crossing_fill.cross_complete_block_reversed = _r
    RazorSolver._crossing_node_charges = _nc

# ------------------------------------------------------------------ b
if which in ("all", "b"):
    for name, a_above, a_below in (("rise/2", A, A / 2), ("top/4", A / 4, A)):
        for lane in (True, False):
            d1 = strip(rod(1, a_above, a_below, ground_eps=(1.0, 0.0)))
            df = {
                k: v
                for k, v in d1.items()
                if k not in ("ground_z", "ground_eps", "ground_model")
            }
            s1 = RazorSolver(**d1, nec5_quadrature=lane)
            sf = RazorSolver(**df, nec5_quadrature=lane)
            g1, gf = s1._build_geometry(), sf._build_geometry()
            assert np.array_equal(g1["wing_seg"], gf["wing_seg"])
            Zf = sf._assemble_Z(gf, sf.k)
            off = np.asarray(g1["basis_offsets"])
            groups = {
                "B": np.arange(off[0], off[1]),
                "A": np.arange(off[1], off[2]),
                "N": np.arange(off[2], g1["n_basis_total"]),
            }
            for rule, cross in (
                ("source", None),
                ("observer", (a_above, a_below)),
                ("min", (min(a_above, a_below),) * 2),
                ("max", (max(a_above, a_below),) * 2),
                ("wire0", (a_below, a_below)),
            ):
                with Rule(cross=cross):
                    Z1 = s1._assemble_Z(g1, s1.k)
                rel = {}
                for rn, rr in groups.items():
                    for cn, cc in groups.items():
                        b1, bf = Z1[np.ix_(rr, cc)], Zf[np.ix_(rr, cc)]
                        rel[rn + cn] = float(
                            np.max(np.abs(b1 - bf)) / np.max(np.abs(bf))
                        )
                whole = float(np.max(np.abs(Z1 - Zf)) / np.max(np.abs(Zf)))
                out(gate="b", deck=name, lane=lane, rule=rule, whole=whole, rel=rel)

# ------------------------------------------------------------------ c
if which in ("all", "c"):
    for name, a_above, a_below in (("rise/4", A, A / 4), ("top/4", A / 4, A)):
        d = rod(4, a_above, a_below)
        base = zr(d)
        for rule, fn in (
            ("all a_above", lambda s, v=a_above: np.full_like(s, v)),
            ("all a_below", lambda s, v=a_below: np.full_like(s, v)),
            ("source/10", lambda s: s / 10),
            ("source/100", lambda s: s / 100),
            ("source*10", lambda s: s * 10),
        ):
            with Rule(node=fn):
                z = zr(d)
            out(
                gate="c",
                deck=name,
                rule=rule,
                z_source=base,
                dz=z - base,
                abs_dz=abs(z - base),
            )

# ------------------------------------------------------------------ d
if which in ("all", "d"):
    M = 4
    t = time.time()
    r_eq = {"razor": zr(rod(M)).real, "bspline": zb(rod(M)).real}
    ctrl = {
        "observer cross": dict(cross="observer"),
        "node at min": dict(node="min"),
        "node at max": dict(node="max"),
    }
    for name, a_above, a_below in (
        ("rise/2", A, A / 2),
        ("rise/4", A, A / 4),
        ("top/2", A / 2, A),
        ("top/4", A / 4, A),
    ):
        d = rod(M, a_above, a_below)
        dr = zr(d).real - r_eq["razor"]
        db = zb(d).real - r_eq["bspline"]
        ctl = {}
        for cn, spec in ctrl.items():
            kw = {}
            if spec.get("cross") == "observer":
                kw["cross"] = (a_above, a_below)
            if spec.get("node") == "min":
                kw["node"] = lambda s, v=min(a_above, a_below): np.full_like(s, v)
            if spec.get("node") == "max":
                kw["node"] = lambda s, v=max(a_above, a_below): np.full_like(s, v)
            with Rule(**kw):
                ctl[cn] = zr(d).real - r_eq["razor"] - db
        out(
            gate="d",
            deck=name,
            m=M,
            dR_razor=dr,
            dR_bspline=db,
            diff=dr - db,
            controls_diff=ctl,
            secs=round(time.time() - t, 1),
        )

# ------------------------------------------------------------------ e
if which in ("all", "e"):
    for name, a_above, a_below in (("rise/2", A, A / 2), ("top/4", A / 4, A)):
        xs = []
        for m in (1, 2, 4, 8):
            d = rod(m, a_above, a_below)
            d["feeds"] = [(1, 4.5, 1 + 0j), (0, 1.0, 1 + 0j)]
            Y = np.asarray(
                RazorSolver(**strip(d), nec5_quadrature=True).compute_y_matrix()
            )
            xs.append(abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1]))
            out(gate="e", deck=name, m=m, nonrec=xs[-1])
        out(gate="e", deck=name, ratios=[a / b for a, b in zip(xs, xs[1:])])

# ------------------------------------------------------------------ f
if which in ("all", "f"):

    def d1140(m):
        d = above_side_deck(wire_radius=[0.002, 0.001, 0.004])
        d["feeds"] = [(2, 4.0, 1 + 0j)]
        return refined(d, m)

    for name, mk in (
        ("rod rise/2", lambda m: dict(rod(m, A, A / 2), feeds=[(1, 4.5, 1 + 0j)])),
        ("rod top/4", lambda m: dict(rod(m, A / 4, A), feeds=[(1, 4.5, 1 + 0j)])),
        ("1140", d1140),
    ):
        g = []
        for m in (1, 2, 4, 8):
            t = time.time()
            d = mk(m)
            a, b = zr(d), zb(d)
            g.append(abs(a - b))
            out(
                gate="f",
                deck=name,
                m=m,
                razor=a,
                bspline=b,
                gap=g[-1],
                secs=round(time.time() - t, 1),
            )
        out(gate="f", deck=name, ratios=[b / a for a, b in zip(g, g[1:])])
