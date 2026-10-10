"""C++ vs numpy agreement of the exact-kernel route (#1410).

Prints max relative differences for: the kernel, its split, pair moments on
#1409's pair-shape set, CoaxialRows windows, and Z on the decks.
Usage: agreement.py [big]   (big adds the 3000-segment wire)
"""

import sys
import time
import warnings

import numpy as np

import momwire
from momwire import BSplineSolver
from momwire import _exact_kernel as EK
from momwire._accel import acc

print("momwire", momwire.__file__, "variant", getattr(momwire._accel, "VARIANT", None))
K0 = 2 * np.pi
R = EK._accel_rules()


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def elem(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.abs(b) > 0
    return float(np.max(np.abs(a - b)[m] / np.abs(b)[m]))


for dl in (0.001, 0.0085, 0.05):
    a = dl / 2
    t = np.concatenate([np.geomspace(1e-4, 1e3, 701), [5.999999, 6.0, 6.000001]])
    print(
        f"kernel d/lam={dl}: elementwise {elem(acc.exact_kernel_ring(t * a, a, K0), EK.ring_kernel(t * a, a, K0)):.2e}"
    )
    ts = np.geomspace(1e-6, 5.99, 300) * a
    A1, B1 = acc.exact_kernel_ring_split(ts, a, K0)
    A0, B0 = EK.ring_kernel_split(ts, a, K0)
    print(f"  split A {elem(A1, A0):.2e}  B {elem(B1, B0):.2e}")

_A = 0.00425
PAIRS = [
    (6 * _A, 6 * _A, 0.0, 1, 1),
    (0.7 * _A, 0.7 * _A, 0.0, 1, 1),
    (3 * _A, 1.3 * _A, 3 * _A, 1, 1),
    (3 * _A, 2 * _A, 1 * _A, 1, 1),
    (4 * _A, 4 * _A, 4.5 * _A, 1, 1),
    (4 * _A, 4 * _A, 8.5 * _A, 1, -1),
    (4 * _A, 3 * _A, -0.5 * _A, -1, 1),
    (2 * _A, 2 * _A, 30 * _A, 1, 1),
    (100 * _A, 100 * _A, 100 * _A, 1, 1),
]
worst = 0
for nd in (3, 4):
    for pr in PAIRS:
        J0 = EK.pair_moments(*pr, _A, K0, nd)
        J1 = acc.exact_kernel_pair_moments(*pr, _A, K0, nd, R, 0)
        worst = max(worst, rel(J1, J0))
print(f"pair moments (near route), max over set: {worst:.2e}")


def group(n, flip=(), lengths=None, L=0.47):
    h = np.full(n, L / n) if lengths is None else np.asarray(lengths, float)
    x0 = np.concatenate([[0.0], np.cumsum(h)[:-1]])
    sgn = np.ones(n)
    for i in flip:
        sgn[i] = -1.0
        x0[i] += h[i]
    return x0, sgn, h


GROUPS = {
    "uniform41": group(41),
    "two runs + reversed": group(30, flip=(3, 17), lengths=[0.01] * 12 + [0.017] * 18),
    "graded": group(16, lengths=np.geomspace(0.004, 0.03, 16)),
    "graded symmetric": group(
        40, lengths=np.r_[np.geomspace(0.03, 0.004, 20), np.geomspace(0.004, 0.03, 20)]
    ),
}
for name, (x0, sgn, h) in GROUPS.items():
    n = x0.size
    whole = EK.coaxial_block(x0, sgn, h, _A, K0, 3)
    for sym in (False, True):
        cr = EK.CoaxialRowsAccel(x0, sgn, h, _A, K0, 3, symmetric=sym)
        got = np.concatenate(
            [cr.rows(r0, min(n, r0 + 7)) for r0 in range(0, n, 7)], axis=2
        )
        ref = EK.CoaxialRows(x0, sgn, h, _A, K0, 3)
        ref.rows(0, n)
        print(
            f"rows {name:22s} sym={sym!s:5s}: max|d|/max {rel(got, whole):.2e}  evaluated {cr.evaluated} (numpy {ref.evaluated})"
        )

L = 0.47
DIPOLE = [np.array([(0.0, 0.0, -L / 2), (0.0, 0.0, L / 2)])]


def halves(w):
    return [
        np.array([(0.0, 0.0, -L / 2), (0.0, 0.0, -w / 2)]),
        np.array([(0.0, 0.0, w / 2), (0.0, 0.0, L / 2)]),
    ]


def solver(wires=DIPOLE, n=None, a=0.00425, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BSplineSolver(
            wires=wires,
            n_per_edge_per_wire=n,
            wavelength=1.0,
            wire_radius=a,
            exact_kernel=True,
            **kw,
        )


def Z(s):
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = time.perf_counter()
        z = np.array(s._compute_Z_operator(geom, supp_seg, polys))
        return z, time.perf_counter() - t


def both(s):
    EK._USE_ACCEL = False
    z0, t0 = Z(s)
    EK._USE_ACCEL = True
    z1, t1 = Z(s)
    z2, _ = Z(s)
    return rel(z1, z0), elem(z1, z0), np.array_equal(z1, z2), t0, t1


DECKS = {
    "dipole37": dict(n=[[37]]),
    "dipole37 segment gap": dict(n=[[37]], feed_model="segment"),
    "end-port pair": dict(
        wires=halves(L / 75),
        n=[[37], [37]],
        feeds=[(0, (L - L / 75) / 4, 0j)],
        junctions=[[(0, "end")], [(1, "start")]],
        junction_ports=[(0, 0j), (1, 0j)],
    ),
    "collinear junction + bend": dict(
        wires=[
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.2)]),
            np.array([(0.0, 0.0, 0.2), (0.0, 0.0, 0.3), (0.1, 0.0, 0.4)]),
        ],
        n=[[19], [9, 13]],
        junctions=[[(0, "end"), (1, "start")]],
    ),
    "bent wire": dict(
        wires=[np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.25), (0.15, 0.0, 0.40)])],
        n=[[23, 17]],
    ),
    "degree 3": dict(n=[[29]], degree=3),
    "ek dipole37": dict(n=[[37]], extended_kernel=True),
}
for n in (9, 19, 37, 75, 151, 301, 601):
    DECKS[f"ladder n={n}"] = dict(n=[[n]])
for name, kw in DECKS.items():
    r, e, stable, t0, t1 = both(solver(**kw))
    print(
        f"Z {name:28s}: max|dZ|/max|Z| {r:.2e}  elementwise {e:.2e}  bit-stable {stable}  numpy {t0:.3f}s C++ {t1:.3f}s"
    )
if len(sys.argv) > 1:
    n = 3000
    s = solver(
        wires=[np.array([(0.0, 0.0, 0.0), (0.0, 0.0, n * 0.003)])],
        n=[[n]],
        a=0.002,
        degree=2,
    )
    r, e, stable, t0, t1 = both(s)
    print(
        f"Z fat wire n=3000: max|dZ|/max|Z| {r:.2e}  elementwise {e:.2e}  bit-stable {stable}  numpy {t0:.3f}s C++ {t1:.3f}s"
    )
