"""`axis_data`'s density knobs, and the two plateaus behind them — momwire#813.

`_crossing_fill.axis_data` took its three densities from module constants:
`_NEAR_Q` (plain Gauss, every segment not touching the interface),
`_NEAR_GROWTH` and `_NEAR_GX` (the log-graded a-scale panels on the segments
that DO touch it). Those are a Galerkin axis's settings, and a PATH-tested
row that ends AT the node needs a finer axis than a Galerkin one does — so
they are per-axis arguments now, defaulting to exactly the constants.

**The two error plateaus are separate, and that is the point of this module.**
On razor's node row at `crossing_deck(1)`:

  * sweeping `q` alone leaves the residual at 5.3e-5 for every order from 4
    to 32 — it never moves;
  * sweeping `panel_order` alone takes it to 2.2e-6 and stops there at every
    `growth` from 4.0 down to 1.25;
  * `panel_order` = 8 AND `q` = 8 together reach 7.7e-11.

So a sweep of either knob alone reads as "converged" at the other's plateau.
That is how the residual came to be recorded in
`test_razor_crossing_axis_813.BAR_ROW_HALF` as a property of the source
Gauss, which it is not; the prose beside that bar is corrected in this change
and this module is what holds the correction.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

from momwire import RazorSolver
from momwire import _crossing_fill as CF

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from test_crossing_serve_524 import crossing_deck  # noqa: E402


@pytest.fixture(scope="module")
def fs():
    deck = crossing_deck(1)
    d = {
        k: v
        for k, v in deck.items()
        if k not in ("ground_z", "ground_eps", "ground_model")
    }
    rs = RazorSolver(**d, nec5_quadrature=False, n_qp_path=8)
    geom = rs._build_geometry()
    so = np.asarray(geom["seg_offsets"])
    return dict(
        rs=rs,
        geom=geom,
        ctx=rs._crossing_context(geom, ground_eps=(1.0, 0.0)),
        seg_below=np.arange(so[0], so[1]),
        seg_above=np.arange(so[1], so[2]),
    )


def _axes(f, **kw):
    return (
        CF.axis_data(f["ctx"], f["seg_below"], **kw),
        CF.axis_data(f["ctx"], f["seg_above"], **kw),
    )


@pytest.mark.parametrize("coarse", [False, True])
def test_the_defaults_are_the_constants(fs, coarse):
    """Passing nothing must build the axis the module constants build, key
    for key and bit for bit — the property `BSplineSolver`'s crossing fill
    rides on."""
    plain = CF.axis_data(fs["ctx"], fs["seg_below"], coarse=coarse)
    gx = CF._GX4 if coarse else CF._NEAR_GX
    spelt = CF.axis_data(
        fs["ctx"],
        fs["seg_below"],
        coarse=coarse,
        growth=CF._FAR_GROWTH if coarse else CF._NEAR_GROWTH,
        panel_order=len(gx),
        q=CF._FAR_Q if coarse else CF._NEAR_Q,
    )
    for key in ("nodes", "t", "w", "F", "Fd", "segof"):
        assert np.array_equal(np.asarray(plain[key]), np.asarray(spelt[key])), key
    assert len(plain["ends"]) == len(spelt["ends"])


def test_a_finer_axis_is_actually_finer(fs):
    coarse_ax, _ = _axes(fs, growth=4.0, panel_order=4, q=4)
    fine_ax, _ = _axes(fs, growth=2.0, panel_order=8, q=12)
    assert fine_ax["nodes"].shape[0] > coarse_ax["nodes"].shape[0]
    # the graded panels are on the interface-touching segments only, so the
    # extra nodes are not spread evenly
    assert fine_ax["w"].sum() == pytest.approx(coarse_ax["w"].sum(), rel=1e-12)


def _razor_reference(fs):
    """Razor's own chopped node row. Independent of the axis knobs, so it is
    built once -- rebuilding it per setting was this module's whole cost."""
    if "ref" in fs:
        return fs["ref"], fs["cols"], fs["A3"]
    rs, geom = fs["rs"], fs["geom"]
    jn = geom["n_basis_total"] - 1
    bo = np.asarray(geom["basis_offsets"])
    cols = np.arange(bo[0], bo[1])
    k, omega = rs.k, rs.omega
    Z = rs._assemble_Z_from_prepared(geom, rs._assemble_Z_prepare(geom), k, omega)
    gA, gB = dict(geom), dict(geom)
    gA["wing_sigma"] = geom["wing_sigma"].copy()
    gA["wing_sigma"][jn, 1] = 0.0
    gB["wing_sigma"] = geom["wing_sigma"].copy()
    gB["wing_sigma"][jn, 0] = 0.0
    Z_A = rs._assemble_Z_from_prepared(gA, rs._assemble_Z_prepare(gA), k, omega)
    Z_B = rs._assemble_Z_from_prepared(gB, rs._assemble_Z_prepare(gB), k, omega)
    T1_half = Z_B - (Z_A + Z_B - Z)
    seg_h, seg_t, seg_p0 = geom["seg_h"], geom["seg_t"], geom["seg_p0"]
    cent = seg_p0 + 0.5 * seg_h[:, None] * seg_t
    node = rs._knot_points(geom)[jn]
    s_half = int(geom["wing_seg"][jn, 1])
    M0 = rs._seg_moments_from_prepared(
        rs._seg_moments_prepare(
            np.array([node, cent[s_half]]), geom, rs._kernel_radius(geom)
        ),
        k,
        2,
        need_m1=False,
    )[0]
    dM0 = M0[1] - M0[0]
    prep = rs._assemble_Z_prepare(geom)
    T2h = dM0[prep["s_a"]] * prep["q_a"] + dM0[prep["s_b"]] * prep["q_b"]
    fs["ref"] = (T1_half[jn] - T2h / (1j * omega * rs.eps))[cols]
    fs["cols"] = cols
    fs["A3"] = CF.path_test_axis(
        geom["n_basis_total"], rs._path_test_rows(geom, [jn], halves="B")
    )
    return fs["ref"], fs["cols"], fs["A3"]


def _node_row_rel(fs, **kw):
    """Razor's junction row chopped at the node, trunk vs razor's own."""
    ref, cols, A3 = _razor_reference(fs)
    jn = fs["geom"]["n_basis_total"] - 1
    B = CF.axis_data(fs["ctx"], fs["seg_below"], **kw)
    got = -CF.cross_complete_block(fs["ctx"], A3, B, corner=False)[jn, cols]
    return float(np.abs(got - ref).max() / np.abs(ref).max())


def test_the_source_order_alone_never_moves_the_node_row(fs):
    """The measurement that corrects `BAR_ROW_HALF`'s recorded reason."""
    vals = [_node_row_rel(fs, q=q) for q in (4, 8, 16, 32)]
    assert all(4e-5 < v < 7e-5 for v in vals), vals
    assert max(vals) - min(vals) < 1e-6, vals


def test_the_panel_order_alone_stops_at_its_own_plateau(fs):
    # two growths, not three: the claim is that the plateau does not move
    # with growth, and the ends of the range say that as well as three points
    # do while keeping this module inside the 5 s ceiling.
    vals = [_node_row_rel(fs, growth=g, panel_order=16) for g in (4.0, 1.5)]
    assert all(1e-6 < v < 5e-6 for v in vals), vals
    assert max(vals) - min(vals) < 1e-9, vals


def test_both_together_reach_the_floor(fs):
    assert _node_row_rel(fs, growth=2.0, panel_order=8, q=8) < 1e-9
    assert _node_row_rel(fs, growth=2.0, panel_order=8, q=12) < 1e-11
