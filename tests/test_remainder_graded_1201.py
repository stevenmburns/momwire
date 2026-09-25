"""momwire#1201: the Sommerfeld remainder's grazing pairs on graded panels.

#631 integrated a grazing pair with one Gauss rule of order
`ceil(c · len / R_min)`, capped at 192, and #1189 applied it per pair. One
rule over a spike of relative width R_min/len converges only algebraically,
so at the cap a 48-radial surface screen sat 1.1e-2 from its converged
impedance. `_remainder_graded` splits each pair at the near-image foot and at
the interpolation grid's seams and grades the panels toward them.

These gates read the mechanism as well as the answer: the seams it splits at
are complete, the listed-pair kernel is the table kernel, a pair's moments
converge to brute force, the deck-level answer lands on a converged value the
keyed rule misses, and a deck with nothing grazing never reaches any of it.
"""

import math
import warnings

import numpy as np
import pytest

from momwire import _ground_refl, _remainder_graded, _sommerfeld
from momwire import bspline as _bs
from momwire.bspline import BSplineSolver

from test_remainder_pair_order_1189 import surface_screen_deck
from test_sommerfeld_ground import _GRAZE_WL, _grazing_wire

_acc = _sommerfeld._acc
needs_acc = pytest.mark.skipif(
    _acc is None or not hasattr(_acc, "remainder_field_proj_owned"),
    reason="accelerator without remainder_field_proj_owned",
)

# Converged Z_in references, from the BRUTE-FORCE ladder: the keyed Gauss rule
# at c = 4, 8, 16 with no cap — independent of the graded rule. The ladder
# creeps rather than converges below ~1e-7, because a Gauss rule straddling the
# grid's theta-band seam converges like 1/n (see `_remainder_graded`); its
# spread is quoted with each value.
#
# The screens' ladders list pairs with the graded route's TOLERANT broadside
# guard (see `test_the_graded_listing_is_as_symmetric_as_the_deck`). With the
# strict guard of the shipped rule the hub's tie leaves some junction pairs at
# the base order in some rotated copies and not others, and a ladder built that
# way converges to a different number: 2.3e-3 away on the 12-radial screen
# (60.11171518674077 + 15.253388251951405j at c = 8), 6e-7 on the 48-radial —
# the issue's c = 4 reference is of that kind.
#
# #631's grazing wire, h/lambda = 1.09e-4, 16 segments, soil (13, 0.005) (one
# straight wire: no junction, the two guards agree):
#   c = 4  : 67.98234826663277 - 249.14358775566026j
#   c = 8  : 67.98234144432308 - 249.14358797174907j
#   c = 16 : REF_GRAZE                  (c = 4..16 spread 1.1e-7 of |Z|)
REF_GRAZE = complex(67.98236850760632, -249.1435846940572)
# The 12-radial surface screen (N6LF's vertical at h = 2a, momwire#1131):
#   c = 4  : 60.11668596818814 + 15.395931463777389j
#   c = 8  : REF_12                     (c = 4..8 spread 1.0e-7 of |Z|)
REF_12 = complex(60.11669137863695, 15.395928071490694)
# The 48-radial surface screen, the deck #1201 was filed on:
#   c = 4  : REF_48                     (1197 s; strict-guard c = 4, the issue's
#            37.87219022578941 + 5.691938854687347j, is 5.9e-7 away)
REF_48 = complex(37.87216549326275, 5.691954288750884)
# The keyed rule as shipped in 0.63.0 (c = 1, cap 192) is 1.5e-3 from
# REF_GRAZE, 3.3e-2 from REF_12 and 1.1e-2 from REF_48; the graded rule lands
# within 1e-6 of all three.
GRADED_BAR = 1e-6


def _solve(kw, **flags):
    saved = {k: getattr(_bs, k) for k in flags}
    try:
        for k, v in flags.items():
            setattr(_bs, k, v)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the near-ground advisory
            s = BSplineSolver(**kw)
            z, _ = s.compute_impedance()
        return complex(np.atleast_1d(z)[0]), s
    finally:
        for k, v in saved.items():
            setattr(_bs, k, v)


def _grid_for(s):
    g = s._build_geometry()
    eps_t = _ground_refl.eps_tilde(s.ground_eps, s.omega, s.eps)
    grid = s._somm_grid(
        eps_t, _sommerfeld.max_image_distance(g["seg_l"], g["seg_r"], s.ground_z)
    )
    return g, grid


def _graze(n_seg=16):
    kw = _grazing_wire(
        1.09e-4 * _GRAZE_WL,
        n_seg=n_seg,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = BSplineSolver(**kw)
    return s


# ---------------------------------------------------------------------------
# The listed-pair kernel
# ---------------------------------------------------------------------------


def _owned_sample(seed=1201):
    rng = np.random.default_rng(seed)
    obs = rng.uniform(-3.0, 3.0, (17, 3))
    obs[:, 2] = rng.uniform(1e-3, 0.5, 17)
    src = rng.uniform(-3.0, 3.0, (240, 3))
    src[:, 2] = rng.uniform(1e-3, 0.5, 240)
    t_obs = rng.standard_normal((17, 3))
    t_obs /= np.linalg.norm(t_obs, axis=1, keepdims=True)
    t_src = rng.standard_normal((240, 3))
    t_src /= np.linalg.norm(t_src, axis=1, keepdims=True)
    owner = rng.integers(0, 17, 240)
    return obs, t_obs, src, t_src, owner


@needs_acc
def test_the_owned_kernel_is_the_table_kernel_entry_for_entry(monkeypatch):
    """Each listed pair is the (owner, n) entry of the full table, from the
    C++ kernel and from the numpy fallback alike."""
    s = _graze()
    _g, grid = _grid_for(s)
    obs, t_obs, src, t_src, owner = _owned_sample()
    got = _sommerfeld.remainder_field_proj_owned(
        obs, t_obs, src, t_src, owner, 0.0, s.k, grid
    )
    ref = np.array(
        [
            _sommerfeld.remainder_field_proj(
                obs[owner[n] : owner[n] + 1],
                t_obs[owner[n] : owner[n] + 1],
                src[n : n + 1],
                t_src[n : n + 1],
                0.0,
                s.k,
                grid,
            )[0, 0]
            for n in range(src.shape[0])
        ]
    )
    scale = np.abs(ref).max()
    assert np.abs(got - ref).max() <= 1e-13 * scale
    monkeypatch.setattr(_sommerfeld, "_acc", None)
    slow = _sommerfeld.remainder_field_proj_owned(
        obs, t_obs, src, t_src, owner, 0.0, s.k, grid
    )
    assert np.abs(slow - ref).max() <= 1e-11 * scale


@needs_acc
def test_the_owned_kernel_cancels_as_solve_aborted():
    from momwire import CancelToken, SolveAborted

    s = _graze()
    _g, grid = _grid_for(s)
    obs, t_obs, src, t_src, owner = _owned_sample()
    token = CancelToken()
    token.cancel()
    with pytest.raises(SolveAborted):
        _sommerfeld.remainder_field_proj_owned(
            obs, t_obs, src, t_src, owner, 0.0, s.k, grid, token.ptr
        )


@needs_acc
def test_the_fused_inner_moments_are_the_numpy_ones(monkeypatch):
    """`remainder_graded_inner` (panels in, source-side moments out) against
    the same rule spelled in numpy over explicit nodes and the owned kernel."""
    s = _graze()
    g, grid = _grid_for(s)
    I = np.array([5, 5, 5, 0, 3])
    J = np.array([5, 6, 7, 1, 9])
    fused = s._remainder_pair_moments_graded(g, I, J, grid)
    monkeypatch.setattr(_sommerfeld, "_acc", None)
    ref = s._remainder_pair_moments_graded(g, I, J, grid)
    assert np.abs(fused - ref).max() <= 1e-11 * np.abs(ref).max()


@needs_acc
def test_the_fused_inner_cancels_as_solve_aborted():
    from momwire import CancelToken, SolveAborted

    s = _graze()
    g, grid = _grid_for(s)
    token = CancelToken()
    token.cancel()
    with pytest.raises(SolveAborted):
        _remainder_graded.pair_moments(
            g["seg_l"],
            g["seg_r"],
            g["tangents"],
            g["h_per_seg"],
            np.array([5]),
            np.array([5]),
            s.ground_z,
            s.k,
            grid,
            s.degree + 1,
            cancel_flag=token.ptr,
        )


# ---------------------------------------------------------------------------
# The seams
# ---------------------------------------------------------------------------


def _stencil_signature(grid, P, X):
    """(region, R₁ stencil origin) of the grid lookup at each pair (P, X[n]),
    exactly as `SommerfeldGrid.eval` / `proj_one` choose them."""
    rho = np.hypot(P[0] - X[:, 0], P[1] - X[:, 1])
    hh = P[2] + X[:, 2]
    r1 = np.minimum(np.hypot(rho, hh), grid.r1_max)
    th = np.arctan2(hh, rho)
    split = math.radians(_sommerfeld._SOMM_TH_SPLIT_DEG)
    band = np.where(th <= split, 0, 1)
    zone = np.where(r1 <= grid.r_break, 0, np.where(r1 <= grid.r_near, 1, 2))
    reg = 2 * zone + band
    i0 = np.empty(reg.size, dtype=np.int64)
    for idx, R in enumerate(grid._regions):
        m = reg == idx
        fr = (r1[m] - R["r0"]) / R["dr"]
        i0[m] = np.clip(np.floor(fr).astype(np.int64) - 1, 0, R["n_r"] - 4)
    return reg * 100000 + i0


def test_every_seam_a_segment_crosses_is_a_breakpoint():
    """Sample the lookup finely along random segments: wherever the region or
    the R₁ stencil changes, `_crossings` must have put a root in between. A
    missed seam is a jump or a kink inside a panel, and algebraic
    convergence again."""
    s = _graze()
    _g, grid = _grid_for(s)
    lattice = _remainder_graded._lattice(grid)
    rng = np.random.default_rng(7)
    n = 40
    L = 0.2 * _GRAZE_WL
    P = rng.uniform(-L, L, (n, 3))
    P[:, 2] = rng.uniform(1e-4, 0.1 * L, n)
    C = rng.uniform(-L, L, (n, 3))
    C[:, 2] = rng.uniform(1e-4, 0.1 * L, n)
    D = rng.normal(0.0, L, (n, 3))
    D[:, 2] = np.clip(D[:, 2], -C[:, 2] + 1e-4, None) * 0.1
    task, root = _remainder_graded._crossings(P, C, D, 0.0, lattice)
    u = np.linspace(0.0, 1.0, 40001)
    changes = 0
    for t in range(n):
        sig = _stencil_signature(grid, P[t], C[t][None] + u[:, None] * D[t][None])
        where = np.flatnonzero(np.diff(sig) != 0)
        changes += where.size
        roots = np.sort(root[task == t])
        for w in where:
            lo, hi = u[w], u[w + 1]
            assert ((roots >= lo) & (roots <= hi)).any(), (t, lo, hi)
    assert changes > 50, "the sample crossed almost no seams: vacuous"


def test_the_graded_listing_is_as_symmetric_as_the_deck():
    """A 12-radial screen is invariant under turning it one radial, and so
    must be the list of pairs that go on graded panels. Radials meet at the
    hub, where an observer's image foot lands EXACTLY on the other segment's
    end, and the strict broadside guard `0 < t < 1` decides that tie by the
    last bit of a cosine: the keyed route's own list is asymmetric here (the
    negative control), the graded route's tolerant guard is not."""
    s = BSplineSolver(**surface_screen_deck(12))
    g = s._build_geometry()
    nr = 12 * 10

    def turned(I, J):
        rot = lambda i: np.where(i < nr, (i + 10) % nr, i)  # noqa: E731
        return set(zip(rot(I).tolist(), rot(J).tolist(), strict=True))

    I, J, _ = s._remainder_qp_pairs(g["seg_l"], g["seg_r"], s.ground_z)
    listed = set(zip(I.tolist(), J.tolist(), strict=True))
    assert listed == turned(I, J)

    saved = _bs._REMAINDER_GRADED
    try:
        _bs._REMAINDER_GRADED = False
        I0, J0, _ = s._remainder_qp_pairs(
            g["seg_l"], g["seg_r"], s.ground_z, c=_bs._REMAINDER_GRADED_C
        )
    finally:
        _bs._REMAINDER_GRADED = saved
    strict = set(zip(I0.tolist(), J0.tolist(), strict=True))
    assert strict != turned(I0, J0), "the strict guard's tie did not show"


# ---------------------------------------------------------------------------
# Pair moments
# ---------------------------------------------------------------------------


@needs_acc
@pytest.mark.parametrize("pair", [(5, 5), (5, 6), (6, 5), (5, 7)])
def test_pair_moments_converge_to_brute_force(pair):
    """#631's grazing wire (h/lambda = 1.09e-4, len/R_min = 69): the graded
    moments against a Gauss rule of order 1500 on both sides.

    Adjudicated independently by nested adaptive quadrature (scipy `quad`,
    no breakpoints given): the graded moments sit 1.3e-9, 4.9e-10 and 4.6e-11
    from it on (5,5), (5,6), (5,7), where Gauss order 4000 sits 1.2e-8,
    6.6e-10 and 4.6e-11 and order 1500 ~4e-8 on the self pair. So the bar is
    the brute-force reference's own error; the keyed order this replaces
    (70 here, 192 at most) is 2.3e-6 off on the self pair even at 192."""
    s = _graze()
    g, grid = _grid_for(s)
    i, j = pair
    got = s._remainder_pair_moments_graded(g, np.array([i]), np.array([j]), grid)[0]
    ref = s._remainder_pair_moments(g, [i], [j], 1500, grid)[0, 0]
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    assert rel < 1e-7, f"{pair}: {rel:.2e}"


@needs_acc
def test_pair_moments_are_reciprocal_and_chunk_free(monkeypatch):
    """(j, i) is (i, j) transposed — exactly, so the filled Q stays symmetric
    as the one-order fill always was — and neither the pair chunk nor the
    observer block reaches the bits."""
    s = _graze()
    g, grid = _grid_for(s)
    I = np.array([5, 6, 5, 7, 0, 15, 3])
    J = np.array([6, 5, 5, 5, 0, 14, 9])
    a = s._remainder_pair_moments_graded(g, I, J, grid)
    assert np.array_equal(a[0], a[1].T)
    assert np.array_equal(
        a[3], s._remainder_pair_moments_graded(g, [5], [7], grid)[0].T
    )
    monkeypatch.setattr(_remainder_graded, "PAIR_CHUNK", 2)
    monkeypatch.setattr(_remainder_graded, "OUTER_BLOCK", 5)
    b = s._remainder_pair_moments_graded(g, I, J, grid)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# The deck
# ---------------------------------------------------------------------------


def _lands_and_misses(kw, ref, keyed_miss):
    z, s = _solve(kw)
    assert s._last_remainder_graded > 0, "the graded route did not run"
    rel = abs(z - ref) / abs(ref)
    assert rel < GRADED_BAR, f"graded {rel:.2e}"

    z_keyed, s_keyed = _solve(kw, _REMAINDER_GRADED=False)
    assert s_keyed._last_remainder_graded == 0
    missed = abs(z_keyed - ref) / abs(ref)
    assert missed > keyed_miss, f"keyed rule only {missed:.2e} away"


def test_the_graded_rule_lands_on_the_converged_wire_and_the_keyed_rule_misses():
    """The negative control the issue asked for, on #631's grazing wire:
    the graded rule inside 1e-6 of the brute-force converged value, and with
    `_REMAINDER_GRADED` off — the keyed order, as shipped — over 1e-3 away.
    That miss is NOT the cap (this wire's keyed order is 70): at c = 1 the
    keyed order is len/R_min itself, one point per spike width."""
    _lands_and_misses(_grazing_wire(
        1.09e-4 * _GRAZE_WL, ground_eps=(13.0, 0.005), ground_model="sommerfeld"
    ), REF_GRAZE, 1e3 * GRADED_BAR)  # fmt: skip


@pytest.mark.slow
def test_the_graded_rule_lands_on_the_converged_screen_and_the_keyed_rule_misses():
    """The same on the 12-radial surface screen, where the radial self pairs
    hit the cap of 192 and the hub pairs cross at every angle: 3.1e-2 keyed,
    inside 1e-6 graded."""
    _lands_and_misses(surface_screen_deck(12), REF_12, 1e4 * GRADED_BAR)


@pytest.mark.slow
def test_the_graded_rule_lands_on_the_converged_48_radial_screen():
    """The issue's own deck and bar: 1.1e-2 keyed (the cap of 192 on ~9k
    pairs), inside 1e-6 graded."""
    _lands_and_misses(surface_screen_deck(48), REF_48, 1e4 * GRADED_BAR)


def test_a_deck_with_nothing_grazing_never_reaches_the_graded_route():
    """The short-circuit is load-bearing: an elevated wire lists no pair on
    either route, so switching the graded rule off changes no bit."""
    kw = _grazing_wire(
        0.05 * _GRAZE_WL, ground_eps=(13.0, 0.005), ground_model="sommerfeld"
    )
    z_on, s_on = _solve(kw)
    assert s_on._last_remainder_graded == 0
    assert len(s_on._last_remainder_orders) == 1
    z_off, _ = _solve(kw, _REMAINDER_GRADED=False)
    assert z_on == z_off
