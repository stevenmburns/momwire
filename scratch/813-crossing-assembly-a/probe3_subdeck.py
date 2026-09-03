"""#813 step 3 shape (a), probe 3: the below-only sub-deck and its basis map.

Shape (a): the below wires become their own RazorSolver, so #812's fill runs
on a geometry it will accept, and its bases map back into the full system's
index. The crossing wire's plane end is a grounded end there, carrying the
half tent razor already spells for contact decks.

What this asks, before any of it is written into `razor.py`:

  * does the sub-deck's basis set correspond one-for-one with the full
    deck's below bases plus one per crossing tent?
  * does the sub-solver's fill run at all, i.e. is the serve plan's theta
    artefact the only thing in the way?
"""

import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))

from momwire import _medium_spec as MS  # noqa: E402
from momwire import razor as _razor  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from test_crossing_serve_524 import crossing_deck, fan_rise_deck  # noqa: E402


def full_solver(deck):
    """The whole deck on razor, past the constructor's crossing refusal."""
    rd = {k: v for k, v in deck.items() if k != "junctions"}
    orig = RazorSolver._refuse_buried_geometry
    RazorSolver._refuse_buried_geometry = lambda self: None
    try:
        rs = RazorSolver(**rd, nec5_quadrature=False, n_qp_path=8)
        return rs, rs._build_geometry()
    finally:
        RazorSolver._refuse_buried_geometry = orig


def sub_deck(deck, media):
    """The below wires alone, same ground, same radius, same mesh."""
    below = [w for w, m in enumerate(media) if m == MS.BELOW]
    return below, dict(
        wires=[np.asarray(deck["wires"][w]) for w in below],
        n_per_edge_per_wire=[list(deck["n_per_edge_per_wire"][w]) for w in below],
        wire_radius=deck["wire_radius"],
        wavelength=deck["wavelength"],
        ground_z=deck["ground_z"],
        ground_eps=deck["ground_eps"],
        ground_model=deck["ground_model"],
        feeds=[(0, 0.5, 1 + 0j)],  # a placeholder port; the fill is what matters
    )


def report(name, deck):
    print(f"\n=== {name} ===")
    media = BSplineSolver(**deck)._wire_media()
    rs, geom = full_solver(deck)
    bo = np.asarray(geom["basis_offsets"])
    below_w, sd = sub_deck(deck, media)
    n_full_below = sum(int(bo[w + 1] - bo[w]) for w in below_w)
    # crossing tents in the full geometry
    so = np.asarray(geom["seg_offsets"])
    below_seg = set()
    for w in below_w:
        below_seg.update(range(int(so[w]), int(so[w + 1])))
    tents = [
        m
        for m in range(geom["n_basis_total"])
        if (int(geom["wing_seg"][m, 0]) in below_seg)
        != (int(geom["wing_seg"][m, 1]) in below_seg)
    ]
    print(
        f"  below wires {below_w};  full-deck below bases {n_full_below};  "
        f"crossing tents {tents}"
    )

    orig = RazorSolver._refuse_buried_geometry
    RazorSolver._refuse_buried_geometry = lambda self: (
        setattr(self, "_below_plane", True) or setattr(self, "_crossing", False)
    )
    try:
        sub = RazorSolver(**sd, nec5_quadrature=False, n_qp_path=8)
        g = sub._build_geometry()
    finally:
        RazorSolver._refuse_buried_geometry = orig
    print(
        f"  sub-deck bases {g['n_basis_total']}  "
        f"(grounded {np.asarray(g['grounded_bases']).tolist()})"
    )
    print(
        f"  expected = {n_full_below} below + {len(tents)} crossing = "
        f"{n_full_below + len(tents)}  ->  "
        f"{'MATCH' if g['n_basis_total'] == n_full_below + len(tents) else 'MISMATCH'}"
    )
    try:
        Z = sub._assemble_Z_from_prepared(
            g, sub._assemble_Z_prepare(g), sub.k, sub.omega
        )
        print(f"  sub-deck fill: OK, |Z|max = {np.abs(Z).max():.6g}")
    except Exception as e:  # noqa: BLE001 -- the probe's subject
        print(f"  sub-deck fill: {type(e).__name__}: {str(e)[:130]}")


def main():
    warnings.filterwarnings("ignore")
    _razor._SERVE_BELOW_PLANE = True
    report("crossing_deck(1)", crossing_deck(1))
    report("fan_rise_deck()", fan_rise_deck())


if __name__ == "__main__":
    main()
