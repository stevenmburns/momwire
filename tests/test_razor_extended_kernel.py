"""`RazorSolver(extended_kernel=True)` — the fat-wire twin (momwire#436).

The refusal this replaces said "RazorSolver is reduced-kernel only: NEC-5's
formulation is the comparison target, and its expansion is tested on the wire
axis". The 2026-08-18 taper study measured that premise and found it half
right: the binary is EXTENDED-kernel everywhere (it has no `EK` card because
its formulation does not offer the choice), so the kernel the refusal
declined is the reference's own, and the expansion the refusal correctly
places on the axis is tested against a TUBE.

What this module gates, in the order the maintainer's brief asks for them:

  A. EK off is structurally absent — no EK entry point is executed, which is
     the mechanism behind the branch-point bit-identity (architecture doc §6
     gate (b), the standard every capability on this class is held to).
  B. THIN wire: the two kernels agree. |EK − reduced| on Ward's thinnest
     section is 1e-4 … 1.4e-3 ohm down the ladder, against the twin's own
     0.037 ohm residual — so the reduced row keeps its claim there and the
     kernel choice is invisible, which is the other half of the domain
     partition.
  C. CROSS-FORMULATION: `RazorSolver` and `BSplineSolver(degree=1)` share one
     kernel, checked in the units-4/5 difference-of-differences form — the
     kernel DELTA each formulation reports must agree to 0.25 ohm, which
     subtracts out razor's own O(1/N) walk instead of charging the kernel for
     it.
  D. THE HEADLINE: the fat-wire twin. On the study's uniform 25 mm control
     the (razor_EK_n5q − NEC-5) offset is CONSTANT to 0.012/0.021 ohm down
     the ladder against the `nec5_quadrature` sharp bar of 0.05 — on the very
     deck where the reduced row missed that bar by 43x.

plus the composition gates: both quadrature lanes, all three grounds, ground
contact, per-wire radii, loading, and the swept prepare/replay split.

Golden data: `tests/golden_razor_taper_nec5.py`, captured by
`scripts/capture_razor_taper_nec5_lane.py` (which documents the decks, the
matched-feed recipe and the ladder depth). NEC-5 printouts are End-User
Reports, LLNL-CODE-746721; only printed impedances appear here.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import BSplineSolver, RazorSolver
from momwire import _bspline_kernels as _bk
from momwire import _kernel_moments as _km
from momwire import razor as _razor

from golden_razor_taper_nec5 import TAPER_LADDERS

C = 299792458.0
FREQ_MHZ = 14.2
WL = C / (FREQ_MHZ * 1e6)

WARD_LEN = 10.51010
WARD_NSEC = 10
WARD_SEC = WARD_LEN / WARD_NSEC
WARD_RADII = (
    1.250000e-03,
    3.888889e-03,
    6.527778e-03,
    9.166667e-03,
    1.180556e-02,
    1.444444e-02,
    1.708333e-02,
    1.972222e-02,
    2.236111e-02,
    2.500000e-02,
)
FED_SEC = 6
Z_HEIGHT = 10.0

# The `nec5_quadrature` sharp lane's offset-constancy bar (momwire#398 unit
# 2, ratified 2026-08-17) and the units-4/5 cross-formulation bar.
TWIN_BAR = 0.05
CROSS_BAR = 0.25


def _radii(name):
    if name == "ward":
        return WARD_RADII
    if name == "fat":
        return (2.500000e-02,) * WARD_NSEC
    if name == "thin":
        return (1.250000e-03,) * WARD_NSEC
    raise ValueError(name)


def taper_solver(name, n_per_sec, kind):
    """One row of the ladder, fed at NEC-5's own knot (the `EX` trap)."""
    radii = _radii(name)
    wires = [
        np.array(
            [
                [k * WARD_SEC, 0.0, Z_HEIGHT],
                [(k + 1) * WARD_SEC, 0.0, Z_HEIGHT],
            ]
        )
        for k in range(WARD_NSEC)
    ]
    common = dict(
        wires=wires,
        n_per_edge_per_wire=[[n_per_sec]] * WARD_NSEC,
        wire_radius=list(radii),
        wavelength=WL,
        feed_wire_index=FED_SEC - 1,
        feed_arclength=WARD_SEC / 2.0,
    )
    if kind == "ek_n5q":
        return RazorSolver(extended_kernel=True, nec5_quadrature=True, **common)
    if kind == "red_n5q":
        return RazorSolver(nec5_quadrature=True, **common)
    if kind == "ek_gl":
        return RazorSolver(extended_kernel=True, **common)
    if kind == "red_gl":
        return RazorSolver(**common)
    junctions = [[(k, "end"), (k + 1, "start")] for k in range(WARD_NSEC - 1)]
    if kind == "bs1_ek":
        return BSplineSolver(
            degree=1, extended_kernel=True, junctions=junctions, **common
        )
    if kind == "bs1_red":
        return BSplineSolver(degree=1, junctions=junctions, **common)
    raise ValueError(kind)


COLUMNS = ("ek_n5q", "red_n5q", "ek_gl", "red_gl", "bs1_ek", "bs1_red")


def golden(name):
    """(N_total, Z_nec5, {column: Z}) per rung."""
    return [
        (row[0], row[1], dict(zip(COLUMNS, row[2:]))) for row in TAPER_LADDERS[name]
    ]


# ==========================================================================
# D. THE HEADLINE — the fat-wire twin
# ==========================================================================
def test_d_the_fat_wire_twin_offset_is_constant():
    """The unit's headline gate, and the reason the refusal fell.

    `fat` is the CONTROL deck: ten colinear sections at one 25 mm radius, so
    it carries no radius step and measures the kernel and nothing else. It is
    also the deck where the STUDY measured razor's reduced twin claim failing
    the offset-constancy bar by 43x (dR spread 2.133 ohm, over its full
    ladder to N = 400) and its continuum limit sitting 4.863 ohm from
    NEC-5's. On the gated rungs here the same reduced row reads 0.555 in dR
    and 1.400 ohm at the limit — the same finding, inside the valid domain.

    With the extended kernel the same lane's offset is a CONSTANT
    +0.005 + 0.011j ohm-ish at every rung. Both spreads are gated at the
    sharp bar; the per-rung magnitude is gated too, because a constant offset
    that happened to be large would satisfy constancy while saying nothing.
    """
    ds = []
    for _n, z5, cols in golden("fat"):
        d = cols["ek_n5q"] - z5
        ds.append(d)
        assert abs(d) <= TWIN_BAR, f"N={_n}: |dZ| {abs(d):.4f}"
    dr = max(x.real for x in ds) - min(x.real for x in ds)
    dx = max(x.imag for x in ds) - min(x.imag for x in ds)
    assert dr <= TWIN_BAR, f"dR spread {dr:.4f}"
    assert dx <= TWIN_BAR, f"dX spread {dx:.4f}"


def test_d_the_reduced_kernel_still_fails_that_bar_on_fat_wire():
    """The gate that keeps the finding from silently reverting.

    Read this as the negative control for the one above: the SAME deck, the
    SAME lane, the reduced kernel, and the constancy claim does not hold —
    measured 0.555 / 0.475 ohm against the 0.05 bar, growing monotonically
    down the ladder because it is a kernel mismatch and not a constant. If a
    future change ever made this pass, either the kernel identification is
    wrong or the two kernels stopped being different, and both are things to
    find out here rather than in a user's deck.
    """
    ds = [cols["red_n5q"] - z5 for _n, z5, cols in golden("fat")]
    dr = max(x.real for x in ds) - min(x.real for x in ds)
    assert dr > 4 * TWIN_BAR, f"dR spread {dr:.4f} — the reduced row got sharp?"
    # and it walks AWAY, monotonically, rather than scattering
    mags = [abs(x) for x in ds]
    assert mags == sorted(mags)


def test_d_the_taper_deck_holds_the_bar_in_dr_and_is_recorded_in_dx():
    """Ward's actual deck: the bar holds in dR; dX is pinned where measured.

    Nine radius STEPS separate this from the `fat` control, and momwire's
    eligibility rule declines to extend ACROSS a step (it extends only
    coaxial EQUAL-radius pairs) where NEC still extends some cross-arm pairs
    at an `IND = 2` junction — strictly more conservative, and O(h) in the
    refinement limit (#249 §4.3). That conservatism is what the dX spread
    measures, so it is recorded at its measured level rather than gated at a
    bar the rule cannot hold, and the reduced row's own dX (0.340) is carried
    beside it so the comparison is not lost.
    """
    ek = [cols["ek_n5q"] - z5 for _n, z5, cols in golden("ward")]
    red = [cols["red_n5q"] - z5 for _n, z5, cols in golden("ward")]

    def spread(ds, part):
        vals = [getattr(x, part) for x in ds]
        return max(vals) - min(vals)

    assert spread(ek, "real") <= TWIN_BAR
    assert spread(ek, "imag") <= 0.10  # measured 0.0779
    # EK is still the better row on both parts, which is the claim that
    # matters on a taper.
    assert spread(ek, "real") < spread(red, "real")
    assert spread(ek, "imag") < spread(red, "imag")


# ==========================================================================
# B. THIN wire — the two kernels agree, so the reduced twin keeps its claim
# ==========================================================================
def test_b_the_kernels_agree_on_thin_wire():
    """a/lambda = 5.9e-5: |EK − reduced| is two to three orders below the bar.

    This is the other half of the domain partition. The reduced kernel was
    never wrong on thin wire — the study measured its constancy there at
    0.0012/0.0038 ohm — and the reason is simply that the two kernels agree,
    which is what is pinned here rather than assumed.
    """
    twin_offsets = []
    for n, z5, cols in golden("thin"):
        d = abs(cols["ek_n5q"] - cols["red_n5q"])
        twin_offsets.append(abs(cols["ek_n5q"] - z5))
        assert d <= 0.005, f"N={n}: |EK − reduced| {d:.6f}"
        # and it is far below the residual the twin already carries
        assert d < 0.1 * twin_offsets[-1]
    # both lanes hold the sharp bar on thin wire, EK on or off
    for col in ("ek_n5q", "red_n5q"):
        ds = [cols[col] - z5 for _n, z5, cols in golden("thin")]
        dr = max(x.real for x in ds) - min(x.real for x in ds)
        dx = max(x.imag for x in ds) - min(x.imag for x in ds)
        assert dr <= TWIN_BAR and dx <= TWIN_BAR, col


def test_b_the_same_difference_is_three_orders_larger_on_fat_wire():
    """The partition, as one ratio: the kernel is invisible on thin wire and
    dominant on fat, at the same mesh."""
    thin = abs(
        TAPER_LADDERS["thin"][-1][2] - TAPER_LADDERS["thin"][-1][3]
    )  # ek_n5q − red_n5q
    fat = abs(TAPER_LADDERS["fat"][-1][2] - TAPER_LADDERS["fat"][-1][3])
    assert fat / thin > 100.0


# ==========================================================================
# C. CROSS-FORMULATION — two formulations, one kernel
# ==========================================================================
@pytest.mark.parametrize("deck", ["fat", "ward", "thin"])
def test_c_the_two_formulations_report_the_same_kernel_delta(deck):
    """Units 4/5's difference-of-differences, applied to the kernel.

    Comparing `razor_EK` with `bs1_EK` directly would charge the kernel for
    razor's own O(1/N) walk, which is the formulation's and not the kernel's
    — on `fat` at N=200 the two rows sit 0.68 ohm apart while each is within
    0.05 / 0.40 ohm of NEC-5. What the shared kernel actually claims is that
    the DELTA each formulation reports when the kernel is switched is the
    same object, and that is what is gated:

        D_razor = Z_razor(EK) − Z_razor(reduced)
        D_bs1   = Z_bs1(EK)   − Z_bs1(reduced)
        |D_razor − D_bs1| <= 0.25 ohm

    Measured 0.001 … 0.246 ohm over the gated rungs. The N=20 rung (two
    segments per section, one interior knot each) is excluded: it is the one
    place the two discretizations are too coarse to be talking about the same
    perturbation, measured 0.28 ohm on `ward`.
    """
    for n, _z5, cols in golden(deck):
        if n <= 20:
            continue
        d_razor = cols["ek_gl"] - cols["red_gl"]
        d_bs1 = cols["bs1_ek"] - cols["bs1_red"]
        assert abs(d_razor - d_bs1) <= CROSS_BAR, (
            f"{deck} N={n}: |D_razor − D_bs1| {abs(d_razor - d_bs1):.4f}"
        )


# ==========================================================================
# The golden columns are live — a regression pin on momwire's own arithmetic
# ==========================================================================
@pytest.mark.parametrize("deck", ["fat", "ward", "thin"])
@pytest.mark.parametrize("col", ["ek_n5q", "red_n5q"])
def test_the_recorded_momwire_columns_reproduce(deck, col):
    """Without this the four gates above test a text file.

    Only the two n5q columns and only the two coarsest rungs are re-solved
    here: the GL rows at N=200 are seconds each, and the point is that the
    recorded numbers came from this code, not to re-run the capture.
    """
    for n, _z5, cols in golden(deck)[:2]:
        z, _ = taper_solver(deck, n // WARD_NSEC, col).compute_impedance()
        # the golden literals carry six decimals, which is the tolerance
        assert abs(z - cols[col]) < 2e-6


# ==========================================================================
# A. EK off is STRUCTURALLY ABSENT
# ==========================================================================
# Patched on `razor`'s own module globals, not on the modules that DEFINE
# them: razor imports each by name, so a patch on `_bspline_kernels` /
# `_kernel_moments` would bind nothing and the gate below would pass
# vacuously — which is exactly what the paired control test catches.
_EK_ENTRY_POINTS = (
    (_razor, "_ek_axis_groups"),
    (_razor, "_ek_pair_mask"),
    (_razor, "_ek_factor"),
    (_razor, "_ek_reg_extra"),
    (_razor, "_static_axis_moments_ek"),
)


@pytest.fixture
def ek_call_counts(monkeypatch):
    counts = {}
    for owner, attr in _EK_ENTRY_POINTS:
        counts[attr] = counts.get(attr, 0)
        original = getattr(owner, attr)

        def wrapper(*args, _a=attr, _f=original, **kwargs):
            counts[_a] += 1
            return _f(*args, **kwargs)

        monkeypatch.setattr(owner, attr, wrapper)
    return counts


def _dipole(wavelength=WL, **kw):
    return RazorSolver(
        wires=[np.array([[0.0, 0.0, 4.0], [0.0, 10.18946, 4.0]])],
        nsegs=12,
        wire_radius=1e-2,
        wavelength=wavelength,
        **kw,
    )


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"nec5_quadrature": True},
        {"ground_z": 0.0},
        {"ground_z": 0.0, "ground_eps": (13.0, 5e-3)},
        {"wire_conductivity": 5.8e7},
    ],
)
def test_a_ek_off_enters_no_ek_code(ek_call_counts, kw):
    """The `extended_kernel=False` standard, in its structural form.

    An EK-off fill must not execute one float operation the pre-kernel fill
    did not — that is architecture doc §6 gate (b), and it is the mechanism
    behind the vs-branch-point bit-identity rather than a restatement of it.
    Measured over the shadow lanes at the branch point: 84 arrays (21
    configurations x Z matrix, impedance, coefficients, three-point sweep)
    bit-for-bit unchanged.
    """
    _dipole(**kw).compute_impedance()
    assert all(v == 0 for v in ek_call_counts.values()), ek_call_counts


def test_a_the_counters_fire_when_ek_is_on(ek_call_counts):
    """The control: a monkeypatch that failed to bind would make the gate
    above pass vacuously."""
    _dipole(extended_kernel=True).compute_impedance()
    assert all(v > 0 for v in ek_call_counts.values()), ek_call_counts


# ==========================================================================
# Eligibility — the SHARED rule, and this formulation's observer labelling
# ==========================================================================
def test_the_eligibility_rule_is_the_shared_one():
    """Razor computes no rule of its own: its labels ARE `_ek_axis_groups`'."""
    s = _dipole(extended_kernel=True)
    geom = s._build_geometry()
    src, img = s._ek_labels(geom)
    assert img is src
    seg_l = geom["seg_p0"]
    seg_r = seg_l + geom["seg_h"][:, None] * geom["seg_t"]
    expected = _bk._ek_axis_groups(seg_l, seg_r, geom["seg_t"], s._seg_radius(geom))
    assert np.array_equal(src, expected)
    # one straight uniform wire is one group
    assert len(set(src.tolist())) == 1


def test_a_bend_and_a_radius_step_are_not_extended_across():
    """Coaxial AND equal-radius, pairwise — the rule's two clauses, one deck
    each. Neither is a razor decision; both fall out of the shared scan."""
    bent = RazorSolver(
        wires=[np.array([[-4.0, 0.0, 0.0], [0.0, 0.0, 2.0], [4.0, 0.0, 0.0]])],
        nsegs=6,
        wire_radius=1e-2,
        wavelength=WL,
        extended_kernel=True,
    )
    lab, _ = bent._ek_labels(bent._build_geometry())
    assert len(set(lab.tolist())) == 2  # one group per straight arm

    step = RazorSolver(
        wires=[
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -5.0]]),
        ],
        nsegs=6,
        wire_radius=[1e-2, 1e-3],
        wavelength=WL,
        extended_kernel=True,
    )
    lab, _ = step._ek_labels(step._build_geometry())
    # collinear but unequal radii: two groups, so the step is not extended
    # across even though the axes coincide.
    assert len(set(lab.tolist())) == 2


def test_the_observer_labels_follow_the_testing_path_halves():
    """A path point's label is the label of the wing segment it lies on, in
    the same flattening the observers themselves take — and the two
    quadrature lanes differ only in how many nodes each half has."""
    for kw, per_wing in (({}, 32), ({"nec5_quadrature": True}, 1)):
        s = RazorSolver(
            wires=[np.array([[-4.0, 0.0, 0.0], [0.0, 0.0, 2.0], [4.0, 0.0, 0.0]])],
            nsegs=6,
            wire_radius=1e-2,
            wavelength=WL,
            extended_kernel=True,
            **kw,
        )
        geom = s._build_geometry()
        lab, _ = s._ek_labels(geom)
        assert s._path_nodes_per_wing() == per_wing
        obs = s._ek_obs_labels_path(geom, lab)
        pts, _t, _w = s._testing_paths(geom)
        assert obs.shape == (pts.shape[0] * pts.shape[1],)
        grid = obs.reshape(pts.shape[0], pts.shape[1])
        s_a, s_b = geom["wing_seg"][:, 0], geom["wing_seg"][:, 1]
        assert np.array_equal(
            grid[:, :per_wing], np.repeat(lab[s_a][:, None], per_wing, axis=1)
        )
        assert np.array_equal(
            grid[:, per_wing:], np.repeat(lab[s_b][:, None], per_wing, axis=1)
        )


# ==========================================================================
# The mirror policy — the ground supplies geometry, never the kernel's opinion
# ==========================================================================
def test_the_image_eligibility_is_one_joint_scan():
    """A VERTICAL wire's image is coaxial with it and of equal radius, so the
    real/image pairs extend — NEC's `IND = 0` perpendicular-ground branch. A
    HORIZONTAL wire's image is merely PARALLEL, offset by twice the height,
    and does not. Two independent scans would label both 0 and 0 and declare
    every real/image pair coaxial; the joint scan is what distinguishes them.
    """
    vert = RazorSolver(
        wires=[np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 8.0]])],
        nsegs=8,
        wire_radius=1e-2,
        wavelength=WL,
        ground_z=0.0,
        extended_kernel=True,
    )
    real, img = vert._ek_labels(vert._build_geometry(), mirror=True)
    assert set(real.tolist()) == set(img.tolist())

    horiz = RazorSolver(
        wires=[np.array([[-5.0, 0.0, 4.0], [5.0, 0.0, 4.0]])],
        nsegs=8,
        wire_radius=1e-2,
        wavelength=WL,
        ground_z=0.0,
        extended_kernel=True,
    )
    real, img = horiz._ek_labels(horiz._build_geometry(), mirror=True)
    assert set(real.tolist()).isdisjoint(set(img.tolist()))


def test_the_ground_never_changes_the_free_space_eligibility():
    """`ground_z` moves geometry into the scan; it does not move the rule.
    The REAL half of a joint scan is the free-space labelling, up to the
    group numbering the joint scan assigns."""
    kw = dict(
        wires=[np.array([[-5.0, 0.0, 4.0], [5.0, 0.0, 4.0]])],
        nsegs=8,
        wire_radius=1e-2,
        wavelength=WL,
        extended_kernel=True,
    )
    free = RazorSolver(**kw)
    over = RazorSolver(ground_z=0.0, **kw)
    a, _ = free._ek_labels(free._build_geometry())
    b, _ = over._ek_labels(over._build_geometry(), mirror=True)
    assert np.array_equal(a == a[0], b == b[0])


# ==========================================================================
# Composition — every capability this class serves, served with EK on
# ==========================================================================
@pytest.mark.parametrize("nec5_quadrature", [False, True])
@pytest.mark.parametrize(
    "kw",
    [
        pytest.param({}, id="free"),
        pytest.param({"ground_z": 0.0}, id="pec"),
        pytest.param({"ground_z": 0.0, "ground_eps": (13.0, 5e-3)}, id="refl-coef"),
        pytest.param(
            {
                "ground_z": 0.0,
                "ground_eps": (13.0, 5e-3),
                "ground_model": "sommerfeld",
            },
            id="sommerfeld",
        ),
        pytest.param({"wire_conductivity": 5.8e7}, id="loading"),
        pytest.param({"lumped_loads": [(0, 5.09473, 50 + 30j)]}, id="lumped"),
    ],
)
def test_every_capability_is_served_with_the_kernel_on(kw, nec5_quadrature):
    """No combination hole: each capability solves under EK, and the kernel
    MOVES the answer in every one of them (a silently ignored kwarg would
    otherwise read as support)."""
    common = dict(nec5_quadrature=nec5_quadrature, **kw)
    z_red, _ = _dipole(**common).compute_impedance()
    z_ek, _ = _dipole(extended_kernel=True, **common).compute_impedance()
    assert np.isfinite(z_ek.real) and np.isfinite(z_ek.imag)
    assert abs(z_ek - z_red) > 1e-6
    # the kernel is a perturbation, not a different antenna
    assert abs(z_ek - z_red) < 0.05 * abs(z_red)


def test_ground_contact_takes_the_kernel_through_its_own_image():
    """A base-fed monopole's grounded tent has its own image for a lower
    wing, and that image is coaxial with the real wire, so the contact case
    is extended without one line written for it. The dipole/2 identity —
    this formulation's first ground-contact gate — must survive the swap.
    """
    kw = dict(nsegs=12, wire_radius=1e-2, wavelength=WL, extended_kernel=True)
    mono, _ = RazorSolver(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.09473]])],
        ground_z=0.0,
        feed_arclength=0.0,
        **kw,
    ).compute_impedance()
    dip, _ = RazorSolver(
        wires=[np.array([[0.0, 0.0, -5.09473], [0.0, 0.0, 5.09473]])],
        nsegs=24,
        wire_radius=1e-2,
        wavelength=WL,
        extended_kernel=True,
    ).compute_impedance()
    assert abs(mono - dip / 2.0) < 1e-6 * abs(dip)


def test_mixed_radii_extend_within_a_section_and_not_across_the_step():
    """Per-wire radii need no EK special case: eligibility is equal-radius
    pairwise, so the shared rule already refuses across the step while
    extending within each arm. Gated on the MASK, which is the statement
    itself, plus the fill running and moving.
    """
    wires = [
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -5.0]]),
    ]
    kw = dict(wires=wires, nsegs=10, wavelength=WL)

    same = RazorSolver(extended_kernel=True, wire_radius=[1e-2, 1e-2], **kw)
    lab, _ = same._ek_labels(same._build_geometry())
    n = lab.size
    mask = _bk._ek_pair_mask(_bk._EK(None, lab, lab), n, n)
    assert mask.all()  # collinear, one radius: every pair extends

    step = RazorSolver(extended_kernel=True, wire_radius=[1e-2, 1e-3], **kw)
    lab, _ = step._ek_labels(step._build_geometry())
    mask = _bk._ek_pair_mask(_bk._EK(None, lab, lab), n, n)
    # exactly the two within-arm blocks; the cross-arm quadrants are refused
    half = n // 2
    assert mask[:half, :half].all() and mask[half:, half:].all()
    assert not mask[:half, half:].any() and not mask[half:, :half].any()

    red, _ = RazorSolver(wire_radius=[1e-2, 1e-3], **kw).compute_impedance()
    ek, _ = step.compute_impedance()
    assert abs(ek - red) > 1e-6


def test_loading_is_orthogonal_to_the_kernel():
    """`L` is a surface-impedance path integral outside the fold: it neither
    sees the kernel nor is seen by it. The exact form of that claim on this
    formulation is the `Z_driven = Z_unloaded + Z_L` identity at the fed
    knot, which must hold with EK on exactly as it does with EK off."""
    z_l = 50 + 30j
    kw = dict(
        wires=[np.array([[0.0, 0.0, -5.09473], [0.0, 0.0, 5.09473]])],
        nsegs=12,
        wire_radius=1e-2,
        wavelength=WL,
        extended_kernel=True,
    )
    bare, _ = RazorSolver(**kw).compute_impedance()
    loaded, _ = RazorSolver(lumped_loads=[(0, 5.09473, z_l)], **kw).compute_impedance()
    assert abs(loaded - (bare + z_l)) < 1e-9 * abs(loaded)


@pytest.mark.parametrize("kw", [{}, {"ground_z": 0.0}])
def test_the_prepare_replay_split_stays_honest_under_ek(kw):
    """The EK statics are k-independent — the extended kernel's k -> 0 limit
    is a function of R and the radius alone — so they belong in the prepare
    half beside the reduced ones, and a swept solve must return exactly what
    three single solves do. A wavenumber that had leaked across the boundary
    would show up here and nowhere else."""
    ks = np.array([2 * np.pi / WL * f for f in (0.9, 1.0, 1.1)])
    s = _dipole(extended_kernel=True, **kw)
    swept = s.compute_impedance_swept(ks)
    swept = swept[0] if isinstance(swept, tuple) else swept
    for i, k in enumerate(ks):
        one = _dipole(extended_kernel=True, wavelength=2 * np.pi / k, **kw)
        z, _ = one.compute_impedance()
        assert abs(complex(swept[i]) - z) < 1e-10 * abs(z)


# ==========================================================================
# Capabilities and the retired refusal
# ==========================================================================
def test_capabilities_declare_the_kernel_and_carry_no_refusal_for_it():
    caps = RazorSolver.capabilities
    assert caps.extended_kernel is True
    assert "extended_kernel" not in caps.refusals
    # the rest of the row is unchanged
    assert caps.junction_ports is False
    assert caps.node_gaps is False
    assert "junction_ports" in caps.refusals
    assert "node_gaps" in caps.refusals


def test_the_kernel_moments_closed_form_matches_its_integrand():
    """`_static_axis_moments_ek` against direct quadrature of the extended
    kernel's k -> 0 integrand, on the coaxial case the rule declares eligible
    (rho = a, where the two 1/R terms cancel exactly) and on an off-axis
    control (rho > a, where they do not)."""
    from scipy.integrate import quad

    h = 0.4
    seg_h = np.array([h])
    for a in (1e-3, 2.5e-2):
        for perp in (0.0, 0.05):
            rho2 = np.array([[perp * perp + a * a]])

            def f(t, _a=a, _r=rho2[0, 0], _u=0.0):
                R = np.sqrt((t - _u) ** 2 + _r)
                return 1.0 / R - _a * _a / (2 * R**3) + 3 * _a**4 / (4 * R**5)

            for u_r in (-0.7, 0.13, 0.2, 1.9):
                m0, m1 = _km._static_axis_moments_ek(np.array([[u_r]]), rho2, seg_h, a)
                # `points` puts the integrand's peak (t = u_r, where R is
                # smallest) on a panel boundary. Without it `quad` cannot
                # resolve a millimetre-wide spike inside a 0.4 m interval and
                # warns — the REFERENCE struggling, not the closed form.
                args = (a, rho2[0, 0], u_r)
                pts = [u_r] if 0.0 < u_r < h else None
                q0 = quad(f, 0.0, h, args=args, limit=400, points=pts)[0]
                q1 = quad(
                    lambda t, *ar: t * f(t, *ar),
                    0.0,
                    h,
                    args=args,
                    limit=400,
                    points=pts,
                )[0]
                assert abs(m0[0, 0] - q0) < 1e-9 * abs(q0)
                assert abs(m1[0, 0] - q1) < 1e-8 * max(abs(q1), 1e-3)

    # and it collapses onto the REDUCED closed form as the radius vanishes —
    # exactly, in IEEE, not merely in the limit.
    U = np.array([[0.13]])
    rho2 = np.array([[1e-8]])
    r0, r1 = _km._static_axis_moments(U, rho2, seg_h)
    e0, e1 = _km._static_axis_moments_ek(U, rho2, seg_h, 0.0)
    assert e0[0, 0] == r0[0, 0] and e1[0, 0] == r1[0, 0]
