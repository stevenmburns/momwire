"""#813 step 3, probe 8: the eps~ = 1 collapse with the cross axes graded fine.

Probe 7 found the node row's plateaus: `panel_order` alone takes 5.3e-5 to
2.2e-6 and stops, `q` alone does nothing, and the two together reach 7.7e-11.
The cheapest setting at that class is growth 2.0 / panel 8 / q 8, which costs
96 nodes on this deck's below axis against today's 40.

So: the whole-matrix collapse under that setting, on both decks.
`crossing_deck(1)` has both wires vertical, so t_z and F' are collinear on it
and it cannot separate them; `fan_rise_deck()` has horizontal below members
and can.
"""

import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire import RazorSolver  # noqa: E402
from momwire import _crossing_fill as CF  # noqa: E402
from momwire import _medium_spec as MS  # noqa: E402
from test_crossing_serve_524 import crossing_deck, fan_rise_deck  # noqa: E402

FINE = dict(growth=2.0, panel_order=8, q=8)


def setup_deck(mk, nec5=False):
    deck = mk()
    fs = {
        k: v
        for k, v in deck.items()
        if k not in ("ground_z", "ground_eps", "ground_model", "junctions")
    }
    rs = RazorSolver(**fs, nec5_quadrature=nec5, n_qp_path=8)
    geom = rs._build_geometry()
    Z = rs._assemble_Z_from_prepared(geom, rs._assemble_Z_prepare(geom), rs.k, rs.omega)
    # media off the GROUNDED deck, then applied to this free-space geometry
    from momwire.bspline import BSplineSolver

    media = BSplineSolver(**deck)._wire_media()
    so = np.asarray(geom["seg_offsets"])
    bo = np.asarray(geom["basis_offsets"])
    seg_of = lambda lab: np.concatenate(  # noqa: E731
        [np.arange(so[w], so[w + 1]) for w, m in enumerate(media) if m == lab]
    )
    bas_of = lambda lab: np.concatenate(  # noqa: E731
        [np.arange(bo[w], bo[w + 1]) for w, m in enumerate(media) if m == lab]
    )
    return dict(
        rs=rs,
        geom=geom,
        Z=Z,
        media=media,
        seg_below=seg_of(MS.BELOW),
        seg_above=seg_of(MS.ABOVE),
        b_below=bas_of(MS.BELOW),
        b_above=bas_of(MS.ABOVE),
        n_wire_bases=int(bo[-1]),
    )


def crossing_tents(f):
    """The junction tents that span the interface, and which wing is below."""
    geom = f["geom"]
    out = []
    for m in range(f["n_wire_bases"], geom["n_basis_total"]):
        sides = []
        for j in (0, 1):
            s = int(geom["wing_seg"][m, j])
            sides.append(MS.BELOW if s in f["seg_below"] else MS.ABOVE)
        if len(set(sides)) == 2:
            out.append((m, 0 if sides[0] == MS.BELOW else 1))
    return out


def assemble(f, fine):
    rs, geom = f["rs"], f["geom"]
    n = geom["n_basis_total"]
    tents = crossing_tents(f)
    kw = FINE if fine else {}

    # the two same-medium fills: each zeroes the OTHER medium's wing on every
    # crossing tent and chops that tent's path at the node.
    def masked(zero_side):
        g = dict(geom)
        g["wing_sigma"] = geom["wing_sigma"].copy()
        chop = {}
        for m, below_wing in tents:
            w = below_wing if zero_side == MS.BELOW else 1 - below_wing
            g["wing_sigma"][m, w] = 0.0
            # the surviving half: wing 0 alive -> "A", wing 1 alive -> "B"
            chop[m] = "A" if (1 - w) == 0 else "B"
        return rs._assemble_Z_from_prepared(
            g, rs._assemble_Z_prepare(g, chop=chop), rs.k, rs.omega
        )

    Z_ab = masked(MS.BELOW)  # below wings off -> the ABOVE fill
    Z_be = masked(MS.ABOVE)  # above wings off -> the BELOW fill

    def rows(bases, side):
        recs = [r for m in bases for r in rs._path_test_rows(geom, [m])]
        for m, below_wing in tents:
            half = "A" if (below_wing == 0) == (side == MS.BELOW) else "B"
            recs += rs._path_test_rows(geom, [m], halves=half)
        return CF.path_test_axis(n, recs)

    ctx = rs._crossing_context(geom, ground_eps=(1.0, 0.0))
    A = rows(f["b_above"], MS.ABOVE)
    P = rows(f["b_below"], MS.BELOW)
    ax_a = CF.axis_data(ctx, f["seg_above"], **kw)
    ax_b = CF.axis_data(ctx, f["seg_below"], **kw)

    jn_all = [m for m, _ in tents]
    R_a = np.concatenate([f["b_above"], jn_all]).astype(int)
    R_b = np.concatenate([f["b_below"], jn_all]).astype(int)
    M = np.zeros((n, n), dtype=complex)
    M[np.ix_(R_a, R_a)] += Z_ab[np.ix_(R_a, R_a)]
    M[np.ix_(R_b, R_b)] += Z_be[np.ix_(R_b, R_b)]
    M[np.ix_(R_a, R_b)] -= CF.cross_complete_block(ctx, A, ax_b, corner=False)[
        np.ix_(R_a, R_b)
    ]
    M[np.ix_(R_b, R_a)] -= CF.cross_complete_block_reversed(ctx, P, ax_a, corner=False)[
        np.ix_(R_b, R_a)
    ]
    return M, ax_a["nodes"].shape[0] + ax_b["nodes"].shape[0]


def main():
    warnings.filterwarnings("ignore")
    for name, mk in (
        ("crossing_deck(1)", lambda: crossing_deck(1)),
        ("fan_rise_deck()", fan_rise_deck),
    ):
        print(f"\n=== {name} ===")
        for nec5 in (False, True):
            f = setup_deck(mk, nec5)
            lane = "nec5" if nec5 else "gauss-legendre"
            for fine in (False, True):
                M, nodes = assemble(f, fine)
                d = np.abs(M - f["Z"]) / np.abs(f["Z"]).max()
                print(
                    f"  {lane:>15} {'fine' if fine else 'default':>8}: "
                    f"rel = {d.max():.3e}   axis nodes = {nodes}"
                )


if __name__ == "__main__":
    main()
