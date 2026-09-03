"""#813 step 3 shape (a), probe 5: the below-only GEOMETRY.

Shape (a) does not need a whole sub-solver. The below fill's path reads only
seven keys off `geom` — `seg_p0`, `seg_t`, `seg_h`, `wing_seg`, `wing_rise`,
`wing_sigma`, `grounded_bases` — and `_kernel_radius` returns the scalar,
since the crossing serve already refuses per-wire radii. So the sub-geometry
is a slice plus a wing table, and the basis map is the row order it is built
in.

Each crossing tent contributes its BELOW WING as a half tent: the other wing
becomes a ghost at sigma = 0, which is razor's own contact-tent shape, while
`grounded_bases` stays EMPTY so no plane reference is taken (momwire#813
derivation (a): at a medium interface Phi is not zero and not single-valued
across the families).
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


def below_geometry(rs, geom, media):
    """`(geom_below, rows)` — the below family's own geometry and the FULL
    basis index of each of its rows, in order."""
    so = np.asarray(geom["seg_offsets"])
    bo = np.asarray(geom["basis_offsets"])
    seg_b, bas_b = [], []
    for w, m in enumerate(media):
        if m == MS.BELOW:
            seg_b.append(np.arange(int(so[w]), int(so[w + 1])))
            bas_b.append(np.arange(int(bo[w]), int(bo[w + 1])))
    seg_b = np.concatenate(seg_b)
    bas_b = np.concatenate(bas_b) if bas_b else np.zeros(0, dtype=np.int64)
    below_seg = set(seg_b.tolist())
    remap = {int(s): i for i, s in enumerate(seg_b)}

    ws, wr, wg = geom["wing_seg"], geom["wing_rise"], geom["wing_sigma"]
    rows = list(bas_b)
    # junction tents whose wings are BOTH below belong to this family whole;
    # a crossing tent contributes its below wing as a half tent.
    for m in range(int(bo[-1]), geom["n_basis_total"]):
        sides = [int(ws[m, j]) in below_seg for j in (0, 1)]
        if any(sides):
            rows.append(m)
    n = len(rows)
    g_ws = np.empty((n, 2), dtype=np.int64)
    g_wr = np.empty((n, 2), dtype=bool)
    g_wg = np.empty((n, 2), dtype=np.float64)
    for i, m in enumerate(rows):
        sides = [int(ws[m, j]) in below_seg for j in (0, 1)]
        for j in (0, 1):
            keep = sides[j] if not all(sides) else True
            src = j if sides[j] else (1 - j)  # a ghost copies the live wing
            g_ws[i, j] = remap[int(ws[m, src])]
            g_wr[i, j] = bool(wr[m, src])
            g_wg[i, j] = float(wg[m, j]) if keep else 0.0
    return (
        dict(
            seg_p0=geom["seg_p0"][seg_b],
            seg_t=geom["seg_t"][seg_b],
            seg_h=geom["seg_h"][seg_b],
            wing_seg=g_ws,
            wing_rise=g_wr,
            wing_sigma=g_wg,
            grounded_bases=np.zeros(0, dtype=np.int64),
            n_basis_total=n,
        ),
        np.asarray(rows, dtype=np.int64),
    )


def report(name, deck):
    media = BSplineSolver(**deck)._wire_media()
    rd = {k: v for k, v in deck.items() if k != "junctions"}
    orig = RazorSolver._refuse_buried_geometry
    RazorSolver._refuse_buried_geometry = lambda self: None
    try:
        rs = RazorSolver(**rd, nec5_quadrature=False, n_qp_path=8)
        geom = rs._build_geometry()
    finally:
        RazorSolver._refuse_buried_geometry = orig
    gb, rows = below_geometry(rs, geom, media)
    print(
        f"\n{name}: full n_basis {geom['n_basis_total']}, "
        f"below rows {len(rows)} -> full indices {rows.tolist()}"
    )
    print(
        f"  below segments {gb['seg_h'].size};  "
        f"ghost wings {int((gb['wing_sigma'] == 0).sum())}"
    )
    # `_assemble_Z_from_prepared` dispatches on `_below_plane`; the stub above
    # never set it, and the first version of this probe therefore measured
    # razor's ORDINARY fill on the below geometry and called it a pass. Call
    # the below assembler directly so there is nothing to mis-read.
    try:
        prep = rs._assemble_Z_prepare(gb)
        node = rs._knot_points(geom)[
            [m for m in rows if m >= int(np.asarray(geom["basis_offsets"])[-1])]
        ]
        Z = rs._assemble_Z_below_plane(gb, prep, rs.k, rs.omega, plan_skip=node)
        print(f"  below fill (#812): OK, shape {Z.shape}, |Z|max {np.abs(Z).max():.6g}")
    except Exception as e:  # noqa: BLE001 -- the probe's subject
        print(f"  below fill (#812): {type(e).__name__}: {str(e)[:160]}")


def main():
    warnings.filterwarnings("ignore")
    _razor._SERVE_BELOW_PLANE = True
    report("crossing_deck(1)", crossing_deck(1))
    report("fan_rise_deck()", fan_rise_deck())


if __name__ == "__main__":
    main()
