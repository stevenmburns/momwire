"""Shared builders for the momwire#1149 U3 / #1152 probes."""

import pathlib
import sys
import warnings

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tests"))
warnings.simplefilter("ignore")

from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from test_crossing_serve_524 import crossing_deck, hub_deck  # noqa: E402
from test_razor_detached_1149 import detached  # noqa: E402

__all__ = [
    "BSplineSolver",
    "RazorSolver",
    "collapse",
    "crossing_deck",
    "detached",
    "detached_hub",
    "hub_deck",
]

SOIL_A = (13.0, 0.005)


def detached_hub(m=1, gap=0.1, ground=True, eps=SOIL_A):
    """`scratch/razor-buried-u2b/common.py`'s deck, uniform radii: hub_deck(2)
    with the node pulled apart -- two radials at 0.15 m depth into a buried
    hub, a rise from the hub up to -gap, and the 10 m monopole (15 segments,
    0.67 m each) from +gap up. COAXIAL: the rise end sits 2*gap below the
    mast's long first segment, which is what momwire#1152 is about."""
    d = hub_deck(n_radials=2)
    w = d["wires"]
    w[2] = np.array([(0.0, 0.0, -0.15), (0.0, 0.0, -gap)])
    w[3] = np.array([(0.0, 0.0, 10.0), (0.0, 0.0, gap)])
    d["wires"] = w
    d["junctions"] = [[(0, "end"), (1, "end"), (2, "start")]]
    d["feeds"] = [(3, 4.0, 1 + 0j), (0, 1.0, 1 + 0j)]
    if not ground:
        for k in ("ground_z", "ground_eps", "ground_model"):
            d.pop(k)
    else:
        d["ground_eps"] = eps
    d["n_per_edge_per_wire"] = [[n * m for n in e] for e in d["n_per_edge_per_wire"]]
    return d


def collapse(mk):
    """Per-block eps~=1 collapse of razor's fill against its free-space fill."""
    s1 = RazorSolver(
        **{k: v for k, v in mk(eps=(1.0, 0.0)).items() if k != "junctions"},
        nec5_quadrature=True,
    )
    sf = RazorSolver(
        **{k: v for k, v in mk(ground=False).items() if k != "junctions"},
        nec5_quadrature=True,
    )
    g1, gf = s1._build_geometry(), sf._build_geometry()
    Z1, Zf = s1._assemble_Z(g1, s1.k), sf._assemble_Z(gf, sf.k)
    media = s1._wire_media()
    off = np.asarray(g1["basis_offsets"])
    B = np.concatenate(
        [np.arange(off[w], off[w + 1]) for w, m in enumerate(media) if m == "below"]
    )
    A = np.concatenate(
        [np.arange(off[w], off[w + 1]) for w, m in enumerate(media) if m == "above"]
    )
    out = {}
    for bn, rr, cc in (("AxB", A, B), ("BxA", B, A), ("AxA", A, A), ("BxB", B, B)):
        b1, bf = Z1[np.ix_(rr, cc)], Zf[np.ix_(rr, cc)]
        out[bn] = float(np.max(np.abs(b1 - bf)) / np.max(np.abs(bf)))
    return out
