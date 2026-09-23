"""U2b-3 gates on the real constructor: decks with several crossing nodes.

a  route: 2 tents counted, node term called once with both; cross-node
   entries of the node term nonzero (node 1's charge seen from node 2's rows)
b  eps~=1 collapse per block against razor's free-space fill
c  reciprocity decay, ASYMMETRIC ports (above on rod 1, buried on rod 2),
   every edge refined; red control: node term zeroed
d  convergence onto bspline's U9 route (Z11, Z22, Z12)
e  hub/fan pair: a hub screen at node 1, a three-leg fan at node 2
f  grazing-floor parity: razor vs bspline buried_serve_refusal over
   separation x mesh, and razor's pre-flight against its own fill
"""

import json
import sys
import time

import numpy as np
from common import (
    A_WIRE,
    SOIL_A,
    WL7,
    BSplineSolver,
    RazorSolver,
    hub_deck,
    refined,
    two_node_deck,
)

which = sys.argv[1] if len(sys.argv) > 1 and __name__ == "__main__" else "none"
_nc = RazorSolver._crossing_node_charges


def out(**kw):
    print(json.dumps(kw, default=str), flush=True)


def strip(d):
    return {k: v for k, v in d.items() if k != "junctions"}


def rz(d, **kw):
    return RazorSolver(**strip(d), nec5_quadrature=True, **kw)


def two(m, sep=2.0, **kw):
    d = two_node_deck(sep, **kw)
    d["feeds"] = [(1, 4.5, 1 + 0j), (2, 1.0, 1 + 0j)]
    return refined(d, m)


def y(s):
    Y = np.asarray(s.compute_y_matrix())
    return Y, np.linalg.inv(Y), abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1])


def hub_fan(m, sep=6.0):
    """Node 1: hub_deck(2) (two radials at 0.15 m into a hub, one rise, the
    10 m mast). Node 2, `sep` m along x: a 10 m mast over three 2 m legs
    leaning 45 deg down and out, the mast listed first so all three tents
    cross. Ports: mast 1 at 4.0, leg 0 of the fan at 1.0."""
    h = hub_deck(n_radials=2)
    w = list(h["wires"])
    npe = [list(e) for e in h["n_per_edge_per_wire"]]
    junctions = [list(j) for j in h["junctions"]]
    mast1 = len(w) - 1
    sh = np.array([sep, 0.0, 0.0])
    base = len(w)
    w.append(np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 10.0)]) + sh)
    npe.append([20])
    s = 2.0 / np.sqrt(2.0)
    legs = []
    for a in (0.3, 0.3 + 2 * np.pi / 3, 0.3 + 4 * np.pi / 3):
        legs.append(len(w))
        w.append(np.array([(s * np.cos(a), s * np.sin(a), -s), (0.0, 0.0, 0.0)]) + sh)
        npe.append([8])
    junctions.append([(i, "end") for i in legs] + [(base, "start")])
    d = dict(
        wires=w,
        n_per_edge_per_wire=npe,
        junctions=junctions,
        feeds=[(mast1, 4.0, 1 + 0j), (legs[0], 1.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )
    return refined(d, m)


# ------------------------------------------------------------------ a
if which in ("all", "a"):
    seen = []

    def nc(self_, geom, tents, *a, **k):
        o = _nc(self_, geom, tents, *a, **k)
        seen.append((len(tents), o))
        return o

    RazorSolver._crossing_node_charges = nc
    for name, d in (
        ("two 2m", two(1)),
        ("two 8m", two(1, 8.0)),
        ("hub/fan", hub_fan(1)),
    ):
        seen.clear()
        s = rz(d)
        g = s._build_geometry()
        tents = s._crossing_tents(g)
        z = s.compute_impedance()[0]
        n_t, o = seen[0]
        # rows of wires at node 2 vs the column of node 1's first tent
        off = np.asarray(g["basis_offsets"])
        media = s._wire_media()
        out(
            gate="a",
            deck=name,
            crossing_junctions=s._crossing_junctions(),
            tents=tents,
            node_term_calls=len(seen),
            node_term_tents=n_t,
            z=z,
            col_abs_max=[float(np.abs(o[:, j]).max()) for j in range(o.shape[1])],
        )
    RazorSolver._crossing_node_charges = _nc

# ------------------------------------------------------------------ b
if which in ("all", "b"):
    for name, mk in (
        ("two 2m", lambda **k: two(1, 2.0, **k)),
        ("two 8m", lambda **k: two(1, 8.0, **k)),
    ):
        for lane in (True, False):
            d1 = strip(mk(ground_eps=(1.0, 0.0)))
            df = {
                k: v
                for k, v in d1.items()
                if k not in ("ground_z", "ground_eps", "ground_model")
            }
            s1 = RazorSolver(**d1, nec5_quadrature=lane)
            sf = RazorSolver(**df, nec5_quadrature=lane)
            g1, gf = s1._build_geometry(), sf._build_geometry()
            assert np.array_equal(g1["wing_seg"], gf["wing_seg"])
            Z1, Zf = s1._assemble_Z(g1, s1.k), sf._assemble_Z(gf, sf.k)
            out(
                gate="b",
                deck=name,
                lane=lane,
                whole=float(np.max(np.abs(Z1 - Zf)) / np.max(np.abs(Zf))),
            )

# ------------------------------------------------------------------ c
if which in ("all", "c"):
    for name, mk in (("two 2m", lambda m: two(m, 2.0)), ("hub/fan 6m", hub_fan)):
        xs = []
        for m in (1, 2, 4):
            t = time.time()
            xs.append(y(rz(mk(m)))[2])
            out(gate="c", deck=name, m=m, nonrec=xs[-1], secs=round(time.time() - t, 1))
        out(gate="c", deck=name, ratios=[a / b for a, b in zip(xs, xs[1:])])
        RazorSolver._crossing_node_charges = lambda self_, g, t, *a, **k: np.zeros(
            (g["n_basis_total"], len(t)), dtype=np.complex128
        )
        xs0 = [y(rz(mk(m)))[2] for m in (1, 2)]
        RazorSolver._crossing_node_charges = _nc
        out(gate="c", deck=name, control="node term zeroed", nonrec=xs0)

# ------------------------------------------------------------------ d/e
if which in ("all", "d"):
    for name, mk in (("two 2m", lambda m: two(m, 2.0)), ("hub/fan 6m", hub_fan)):
        gaps = []
        for m in (1, 2, 4):
            t = time.time()
            d = mk(m)
            _, Zr, _ = y(rz(d))
            _, Zb, _ = y(BSplineSolver(**d))
            g = {
                k: float(abs(Zr[i, j] - Zb[i, j]))
                for k, (i, j) in (("z11", (0, 0)), ("z22", (1, 1)), ("z12", (0, 1)))
            }
            gaps.append(g)
            out(
                gate="d",
                deck=name,
                m=m,
                gap=g,
                zr=[str(Zr[0, 0]), str(Zr[1, 1]), str(Zr[0, 1])],
                zb=[str(Zb[0, 0]), str(Zb[1, 1]), str(Zb[0, 1])],
                secs=round(time.time() - t, 1),
            )
        for k in ("z11", "z22", "z12"):
            s = [g[k] for g in gaps]
            out(gate="d", deck=name, key=k, ratios=[b / a for a, b in zip(s, s[1:])])

# ------------------------------------------------------------------ f
if which in ("all", "f"):
    for m in (1, 2, 4):
        for sep in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0):
            d = two(m, sep)
            try:
                r = rz(d).buried_serve_refusal()
            except (ValueError, NotImplementedError) as e:
                r = "CTOR: " + str(e)
            try:
                b = BSplineSolver(**d).buried_serve_refusal()
            except (ValueError, NotImplementedError) as e:
                b = "CTOR: " + str(e)
            out(
                gate="f",
                m=m,
                sep=sep,
                razor=None if r is None else r[:90],
                bspline=None if b is None else b[:90],
                agree=(r is None) == (b is None),
            )
