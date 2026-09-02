"""Razor's charge term evaluated AT the node — momwire#813 unit 2.

Razor's T2 differences the scalar potential between a testing path's two
CENTROIDS.  That is every row this formulation has ever filled, and it is
not enough for the crossing arc: a junction row chopped at the interface has
the KNOT as one endpoint, and the mixed-medium assembly cannot be written at
all until the fill can put Phi(node) there.

`_assemble_Z_prepare(geom, chop={row: "A" | "B"})` is that addition — one
more observation set, prepared exactly as the centroids are, against both
source sets when there is an image — and `_assemble_Z_source_block` reads it
where it already builds `dM0`.  The vocabulary is `_path_test_rows`' own:
``"A"`` is centroid(A) -> knot (the knot is the path's AFTER endpoint),
``"B"`` is knot -> centroid(B) (the knot is BEFORE).

What this module gates, in the order the claims matter:

  * **nothing else moved.**  `chop=None` is the fill that existed, byte for
    byte, on three anchors — free space, ground contact over a Sommerfeld
    ground, and an elevated Sommerfeld dipole — and a fill that DOES chop
    leaves every other row byte-identical.  The cross-revision half of the
    momwire#762 protocol (capture on main, rebuild, `array_equal`) was run by
    hand for this change; what is gated here is the standing claim, which is
    the one a later edit can break.
  * **the chopped row is razor's own chopped kernel.**  Built independently
    from the moments at the two endpoints — the construction
    `tests/test_razor_crossing_axis_813.py` uses as its reference and the
    momwire#651 probe used before it — and compared to the fill.  Both
    quadrature lanes.  This is the real gate: the rest is bookkeeping.
  * **Phi(node) is the kernel integrated to the knot.**  A 800-point
    Gauss-Legendre quadrature of ``exp(-jkR)/(4*pi*R)``, ``R = sqrt(d^2 +
    a^2)``, over each source segment.  Away from the knot the fill matches it
    to 8e-16, which is below the ORACLE's own self-convergence (5e-15); on
    the two segments that TOUCH the knot it is 1.2e-12, and there the oracle
    is the weaker party — that is the log singularity razor's closed-form
    static half exists for and plain Gauss-Legendre converges on only
    logarithmically.  Two bars, because they measure two different things.
  * **both kernel lanes agree** on a chopped fill, at momwire#796's 1e-10.
  * **two refusals rather than guesses**: the extended kernel (a knot
    observer sits ON the interface, where the eligibility scan has no
    answer), and a row that is already grounded (whose T2 the fill already
    rewrites, so chopping it would be two rules for one row's endpoints).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from momwire import razor as _razor
from momwire.razor import RazorSolver

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from test_crossing_serve_524 import crossing_deck  # noqa: E402

C0 = 299792458.0
LAM = C0 / 7.0e6

LANES = {"nec5": {"nec5_quadrature": True}, "gauss-legendre": {}}

BAR_ROW = 1e-14  # measured 5.0e-17: the same arithmetic in a different order
BAR_PHI_AWAY = 1e-14  # measured 7.9e-16, under the oracle's own 5.1e-15
BAR_PHI_TOUCHING = 1e-11  # measured 1.2e-12, and the ORACLE is the weak party
BAR_LANES = 1e-10  # momwire#796's bar, set by the macOS libm seam


def _free_space_crossing(**kw):
    """The crossing deck's geometry with the interface removed.

    At eps~ = 1 razor IS the truth for this geometry, which is why the whole
    unit is gated here before soil.
    """
    deck = crossing_deck(1)
    fs = {
        k: v
        for k, v in deck.items()
        if k not in ("ground_z", "ground_eps", "ground_model")
    }
    fs.update(kw)
    return fs


def _dipole(n, *, ground_model=None, contact=False, **kw):
    z0 = 0.0 if contact else 1.0
    deck = dict(
        wires=[np.array([[0.0, 0.0, z0], [0.0, 0.0, z0 + LAM / 4]])],
        n_per_edge_per_wire=[[n]],
        wire_radius=0.005,
        wavelength=LAM,
        feeds=[(0, 0.0 if contact else LAM / 8, 1 + 0j)],
    )
    if ground_model is not None:
        deck["ground_z"] = 0.0
        deck["ground_eps"] = (13.0, 0.005)
        deck["ground_model"] = ground_model
    deck.update(kw)
    return deck


def _fill(rs, geom, **kw):
    return rs._assemble_Z_from_prepared(
        geom, rs._assemble_Z_prepare(geom, **kw), rs.k, rs.omega
    )


# ---------------------------------------------------------------- nothing moved

ANCHORS = {
    "free space": lambda: _dipole(24),
    "contact / sommerfeld": lambda: _dipole(
        24, ground_model="sommerfeld", contact=True
    ),
    "elevated / sommerfeld": lambda: _dipole(24, ground_model="sommerfeld"),
    "crossing geometry, free space": _free_space_crossing,
}


@pytest.mark.parametrize("anchor", sorted(ANCHORS))
@pytest.mark.parametrize("lane", sorted(LANES))
def test_chop_none_is_the_fill_that_existed(anchor, lane):
    """The argument's absence and `chop=None` are one fill, byte for byte."""
    rs = RazorSolver(**ANCHORS[anchor](), **LANES[lane])
    geom = rs._build_geometry()
    assert np.array_equal(_fill(rs, geom), _fill(rs, geom, chop=None))


@pytest.mark.parametrize("lane", sorted(LANES))
def test_a_chop_moves_only_its_own_row(lane):
    rs = RazorSolver(**_free_space_crossing(n_qp_path=8), **LANES[lane])
    geom = rs._build_geometry()
    jn = geom["n_basis_total"] - 1
    plain, chopped = _fill(rs, geom), _fill(rs, geom, chop={jn: "B"})
    others = [m for m in range(geom["n_basis_total"]) if m != jn]
    assert np.array_equal(plain[others], chopped[others])
    # ... and the chop is not a no-op dressed as one.
    assert not np.array_equal(plain[jn], chopped[jn])


# ------------------------------------------------------- the row itself


def _reference_half_row(rs, geom, jn, halves):
    """Razor's junction row chopped at the node, built from razor's own
    kernel and NOT through the code under test: T1 of the surviving half by
    the wing-sigma decomposition, plus T2 between the node and that half's
    centroid, from the moments at exactly those two observers.

    The momwire#651 probe's construction, and the reference
    `test_razor_crossing_axis_813.py` already gates the trunk against.
    """
    k, omega = rs.k, rs.omega
    prep = rs._assemble_Z_prepare(geom)
    Z = rs._assemble_Z_from_prepared(geom, prep, k, omega)
    gA, gB = dict(geom), dict(geom)
    gA["wing_sigma"] = geom["wing_sigma"].copy()
    gA["wing_sigma"][jn, 1] = 0.0
    gB["wing_sigma"] = geom["wing_sigma"].copy()
    gB["wing_sigma"][jn, 0] = 0.0
    Z_A = rs._assemble_Z_from_prepared(gA, rs._assemble_Z_prepare(gA), k, omega)
    Z_B = rs._assemble_Z_from_prepared(gB, rs._assemble_Z_prepare(gB), k, omega)
    T1_half = (Z_B if halves == "B" else Z_A) - (Z_A + Z_B - Z)

    seg_h, seg_t, seg_p0 = geom["seg_h"], geom["seg_t"], geom["seg_p0"]
    cent = seg_p0 + 0.5 * seg_h[:, None] * seg_t
    node = rs._knot_points(geom)[jn]
    s_half = int(geom["wing_seg"][jn, 1 if halves == "B" else 0])
    before, after = (node, cent[s_half]) if halves == "B" else (cent[s_half], node)
    M0 = rs._seg_moments_from_prepared(
        rs._seg_moments_prepare(
            np.array([before, after]), geom, rs._kernel_radius(geom)
        ),
        k,
        2,
        need_m1=False,
    )[0]
    dM0 = M0[1] - M0[0]
    T2h = dM0[prep["s_a"]] * prep["q_a"] + dM0[prep["s_b"]] * prep["q_b"]
    return T1_half[jn] - T2h / (1j * omega * rs.eps)


@pytest.mark.parametrize("halves", ["A", "B"])
@pytest.mark.parametrize("lane", sorted(LANES))
def test_the_chopped_row_is_razors_own_chopped_kernel(halves, lane):
    """The fill's chopped row IS the independently-built one.

    Compared over the OTHER wire's columns: the junction column's own doublet
    moves with the wing sigma the reference zeroes to isolate T1, so that one
    entry is not the same object on the two sides and is not what this claim
    is about.
    """
    rs = RazorSolver(**_free_space_crossing(n_qp_path=8), **LANES[lane])
    geom = rs._build_geometry()
    jn = geom["n_basis_total"] - 1
    bas_off = np.asarray(geom["basis_offsets"])
    cols = np.arange(bas_off[0], bas_off[1])

    g = dict(geom)
    g["wing_sigma"] = geom["wing_sigma"].copy()
    g["wing_sigma"][jn, 1 if halves == "A" else 0] = 0.0
    got = _fill(rs, g, chop={jn: halves})[jn, cols]
    ref = _reference_half_row(rs, geom, jn, halves)[cols]
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    assert rel < BAR_ROW, rel


# ------------------------------------------------------- Phi at the node


def _phi_oracle(rs, geom, obs, q):
    """(n_seg,) — int_seg exp(-jkR)/(4 pi R) dl', R = sqrt(d^2 + a^2)."""
    seg_p0, seg_t, seg_h = geom["seg_p0"], geom["seg_t"], geom["seg_h"]
    a = float(np.ravel(rs._kernel_radius(geom))[0])
    x, w = leggauss(q)
    out = np.zeros(seg_h.size, dtype=complex)
    for s in range(seg_h.size):
        u = 0.5 * seg_h[s] * (x + 1.0)
        pts = seg_p0[s] + u[:, None] * seg_t[s]
        R = np.sqrt(((pts - obs) ** 2).sum(axis=1) + a * a)
        out[s] = 0.5 * seg_h[s] * np.sum(w * np.exp(-1j * rs.k * R) / R) / (4 * np.pi)
    return out


def test_phi_at_the_node_is_the_kernel_integrated_to_the_knot():
    rs = RazorSolver(**_free_space_crossing(n_qp_path=8))
    geom = rs._build_geometry()
    jn = geom["n_basis_total"] - 1
    prepared = rs._assemble_Z_prepare(geom, chop={jn: "B"})
    M0k, _ = rs._seg_moments_from_prepared(
        prepared["t2_chop_chunks"], rs.k, 1, need_m1=False
    )
    got = M0k[0]

    node = rs._knot_points(geom)[jn]
    ref = _phi_oracle(rs, geom, node, 800)
    # The oracle's own convergence, so the two bars below are readings of the
    # fill where the oracle is strong and of the ORACLE where it is not.
    coarse = _phi_oracle(rs, geom, node, 400)
    self_conv = np.abs(ref - coarse) / np.abs(ref)

    ends = np.stack(
        [geom["seg_p0"], geom["seg_p0"] + geom["seg_h"][:, None] * geom["seg_t"]]
    )
    touching = np.linalg.norm(ends - node, axis=2).min(axis=0) < 1e-9
    assert touching.sum() == 2, "the knot should touch exactly its two wings"

    rel = np.abs(got - ref) / np.abs(ref)
    assert rel[~touching].max() < BAR_PHI_AWAY, rel[~touching].max()
    assert rel[~touching].max() < self_conv[~touching].max() * 10
    assert rel[touching].max() < BAR_PHI_TOUCHING, rel[touching].max()


# ------------------------------------------------------- the two lanes


@pytest.mark.parametrize("lane", sorted(LANES))
def test_the_two_kernel_lanes_agree_on_a_chopped_fill(monkeypatch, lane):
    """The knot observers ride `_seg_moments_from_prepared` like every other
    observer, so they inherit its dispatch — asserted, not assumed."""
    deck = _free_space_crossing(n_qp_path=8)
    rs = RazorSolver(**deck, **LANES[lane])
    geom = rs._build_geometry()
    jn = geom["n_basis_total"] - 1
    fused = _fill(rs, geom, chop={jn: "B"})

    monkeypatch.setattr(_razor, "_FORCE_NUMPY", True)
    rs2 = RazorSolver(**deck, **LANES[lane])
    numpy_lane = _fill(rs2, rs2._build_geometry(), chop={jn: "B"})
    rel = np.abs(fused - numpy_lane).max() / np.abs(numpy_lane).max()
    assert rel < BAR_LANES, rel


# ------------------------------------------------------- the two refusals


def test_the_extended_kernel_is_refused_on_a_chopped_row():
    rs = RazorSolver(**_free_space_crossing(n_qp_path=8), extended_kernel=True)
    geom = rs._build_geometry()
    with pytest.raises(ValueError, match="do not take the extended kernel"):
        rs._assemble_Z_prepare(geom, chop={geom["n_basis_total"] - 1: "B"})


def test_a_grounded_row_may_not_be_chopped():
    rs = RazorSolver(**_dipole(12, ground_model="sommerfeld", contact=True))
    geom = rs._build_geometry()
    grounded = np.asarray(geom["grounded_bases"])
    assert grounded.size, "this deck should stand a wire end in the plane"
    with pytest.raises(ValueError, match="both GROUNDED and chopped"):
        rs._assemble_Z_prepare(geom, chop={int(grounded[0]): "B"})


def test_a_chop_side_must_be_a_or_b():
    rs = RazorSolver(**_free_space_crossing(n_qp_path=8))
    geom = rs._build_geometry()
    with pytest.raises(ValueError, match="must be 'A' or 'B'"):
        rs._assemble_Z_prepare(geom, chop={geom["n_basis_total"] - 1: "middle"})
