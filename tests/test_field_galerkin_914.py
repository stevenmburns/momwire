"""The field-form Galerkin assembly in C++ (momwire#914 unit 2).

`_field_galerkin_block` contracts a chunk's projected pair table against the
observer and source moment weights and scatters the result onto basis rows.
On the 48-radial screen the numpy spelling of those two contractions costs
10.4 s of a 30 s solve — a `(d+1, d+1, n_chunk, n_src)` `Jc` per chunk plus a
fancy-index gather per wing pair.

The C++ twin fuses both moment sums into one q-vector per wing (`Jc` never
exists) and threads the SOURCE axis, which is the decomposition that cannot
race: a basis row owns its whole `Q` row across every wing.

Gates:

- G-914-2   the C++ equals the numpy path to 1e-13 relative on the REAL
            chunks of three decks, captured from live solves rather than
            synthesised.
- G-914-2b  the accelerator's OTHER route (`fused=False`, `Jc` materialised)
            equals it too. That route is production-unreachable, so this is
            what keeps it from being untested code that still ships — and two
            independent index derivations agreeing is a stronger statement
            about the fused one than numpy agreement alone.
- G-914-2c  the three decks' Z, pinned to the digit AND compared across the
            seam. The pin alone would miss a dispatch that never fires; the
            seam comparison alone would miss a change that moves both paths.
- G-914-2d  the numpy path still answers when the accelerator is absent, and
            the attribute the other gates switch on actually exists — a
            renamed flag would otherwise let `monkeypatch.setattr` invent one
            and leave every "numpy" gate quietly running C++.
- G-914-2e  the kernel refuses a malformed call rather than reading past an
            array under a released GIL.
- G-1115    the accumulation TARGET, and `scale` (momwire#1115). A target the
            kernel cannot accumulate into is refused rather than answered with
            a copy, and `scale` multiplies each contribution — pinned across
            more than one chunk, where scaling the target instead would show.

The assembly is exercised across MORE THAN ONE chunk on the 12-radial deck
(asserted, not assumed): `i0` only matters at a chunk boundary, so a
single-chunk gate would say nothing about the offset arithmetic.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import momwire.bspline as _bs
from momwire import BSplineSolver

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_buried_serve_553 import SOIL_A, WL7, _radial  # noqa: E402
from test_crossing_serve_524 import crossing_deck, hub_deck  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:crossing node")

_acc = getattr(_bs, "_acc", None)
requires_accel = pytest.mark.skipif(
    not _bs._HAVE_FIELD_GALERKIN_ACCEL,
    reason="built without the #914 field-Galerkin accelerator",
)


def screen_deck(n_radials=12, depth=0.15, n_per_radial=20):
    """A 12-radial buried screen on the SERVED spelling: the radials meet at a
    buried hub, one rise carries that hub to the surface, and the monopole
    junction-joins it there.

    Written here rather than by widening `hub_deck`, whose radial directions
    are a 4-tuple it slices — asking that for 12 returns 4 and then declares
    junction members that do not exist.

    The detached-fan spelling is NOT an option: a monopole with an end in the
    plane over buried radials is a refused combination (`_medium_spec`), the
    ground-contact image being a fiction a buried observer would see through.

    `n_per_radial` is 20 so the observer axis spans several chunks; see
    G-914-2's count.
    """
    wires, npe = [], []
    for i in range(n_radials):
        th = 2.0 * np.pi * i / n_radials
        wires.append(_radial(depth=depth, direction=(np.cos(th), np.sin(th)))[::-1])
        npe.append([n_per_radial])
    rise_i = len(wires)
    wires.append(np.array([(0.0, 0.0, -depth), (0.0, 0.0, 0.0)]))
    npe.append([2])
    mono_i = rise_i + 1
    wires.append(np.array([(0.0, 0.0, 10.0), (0.0, 0.0, 0.0)]))
    npe.append([15])
    return dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=[
            [(i, "end") for i in range(n_radials)] + [(rise_i, "start")],
            [(rise_i, "end"), (mono_i, "end")],
        ],
        feeds=[(mono_i, 4.3333333333, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )


DECKS = {
    "hub": lambda: hub_deck(n_radials=4),
    "crossing": lambda: crossing_deck(),
    "screen12": lambda: screen_deck(12),
}

# Printed by these decks on the numpy path at f537080, before this edit. The
# hub and crossing values were #912's own G-912-5 pins, unchanged — until
# momwire#956 re-pinned all three: the crossing fill's ẑẑ kernel is k²V + ∂z′W
# and the test-end W term TW is emitted, which moves every deck with a
# vertical member below the plane (hub 141.017−43.425j → 148.634−30.463j,
# crossing 138.961−102.610j → 169.776−82.280j, screen12 98.787−67.076j →
# 105.070−54.251j; antennaknobs scratch/956-derivation is the record).
Z_PINNED = {
    "hub": 148.634045567 - 30.463037886j,
    "crossing": 169.775596898 - 82.280023587j,
    # Measured on the NUMPY path at f537080 and asserted equal on both here;
    # this deck is new with these tests, so the pin is a same-commit record of
    # the reference's answer rather than a value inherited from before it.
    "screen12": 105.070030090 - 54.251084913j,
}


def _capture(build):
    """Every argument tuple `_field_galerkin_block` is called with in one
    solve, plus the Z that solve produced."""
    seen = []
    real = BSplineSolver._field_galerkin_block

    def spy(self, *args, **kw):
        # `**kw` carries momwire#1115 part 3's `out=`/`scale=`. Recorded
        # separately from `args` because the replays below re-run the block
        # STANDALONE and want the returning form, not an accumulation into
        # somebody else's Z.
        seen.append((self, args))
        return real(self, *args, **kw)

    BSplineSolver._field_galerkin_block = spy
    try:
        z, _ = BSplineSolver(**build).compute_impedance()
    finally:
        BSplineSolver._field_galerkin_block = real
    return z, seen


@pytest.fixture(scope="module", params=sorted(DECKS))
def captured(request):
    z, calls = _capture(DECKS[request.param]())
    assert calls, f"{request.param} never entered the field-form assembly"
    return request.param, z, calls


def _reblock(self, args, *, accel, fused=True):
    real = BSplineSolver._field_galerkin_block
    keep = (_bs._HAVE_FIELD_GALERKIN_ACCEL, _bs._FIELD_GALERKIN_FUSED)
    _bs._HAVE_FIELD_GALERKIN_ACCEL = accel
    _bs._FIELD_GALERKIN_FUSED = fused
    try:
        return real(self, *args)
    finally:
        _bs._HAVE_FIELD_GALERKIN_ACCEL, _bs._FIELD_GALERKIN_FUSED = keep


def _rel(a, b):
    return float(np.abs(a - b).max() / max(float(np.abs(b).max()), 1e-300))


# --- G-914-2: the C++ is the numpy numbers on real chunks ------------------


@requires_accel
def test_g914_2_the_cpp_assembly_equals_the_numpy_one(captured):
    name, _z, calls = captured
    for i, (solver, args) in enumerate(calls):
        ref = _reblock(solver, args, accel=False)
        got = _reblock(solver, args, accel=True)
        assert _rel(got, ref) <= 1e-13, (name, i, _rel(got, ref))


@requires_accel
def test_g914_2b_the_unfused_route_equals_it_too(captured):
    """`fused=False` is unreachable from production, which is exactly why it
    is gated here: an untested branch still ships inside the .so."""
    name, _z, calls = captured
    for i, (solver, args) in enumerate(calls):
        ref = _reblock(solver, args, accel=False)
        got = _reblock(solver, args, accel=True, fused=False)
        assert _rel(got, ref) <= 1e-13, (name, i, _rel(got, ref))


@requires_accel
def test_g914_2_the_assembly_spans_more_than_one_chunk(captured):
    """`i0` is only load-bearing at a chunk boundary. A deck whose observer
    axis fits one chunk would gate the offset arithmetic not at all, so the
    count is asserted rather than hoped for."""
    name, _z, calls = captured
    if name != "screen12":
        pytest.skip("the small decks are single-chunk by construction")
    solver, args = calls[-1]
    q = solver._n_qp_buried_field()
    n_src = len(args[4])
    n_obs = len(args[3])
    chunk = max(1, (1 << 19) // max(n_src * q * q, 1))
    assert n_obs > chunk, f"{n_obs} observers fit one chunk of {chunk}"


# --- G-914-2c: the decks' Z ------------------------------------------------


@requires_accel
def test_g914_2c_the_deck_z_is_unchanged_across_the_seam(captured):
    name, z_fast, _calls = captured
    keep = _bs._HAVE_FIELD_GALERKIN_ACCEL
    _bs._HAVE_FIELD_GALERKIN_ACCEL = False
    try:
        z_slow, _ = BSplineSolver(**DECKS[name]()).compute_impedance()
    finally:
        _bs._HAVE_FIELD_GALERKIN_ACCEL = keep
    assert abs(z_fast - z_slow) <= 1e-12 * abs(z_slow), (name, z_fast, z_slow)


def test_g914_2c_the_deck_z_is_the_pinned_digit(captured):
    """The seam comparison above cannot see a change that moves BOTH paths;
    this can. Runs on either path, so a build without the accelerator still
    gates the number."""
    name, z, _calls = captured
    assert abs(z - Z_PINNED[name]) < 5e-7, (name, z, Z_PINNED[name])


# --- G-914-2d: the fallback is a real path, and the flag is a real flag ----


def test_g914_2d_the_dispatch_flags_exist_under_these_names():
    """Every gate above switches paths with `setattr`, which happily CREATES
    an attribute that no longer exists — renaming the flag would leave them
    green while running C++ throughout. Same hole #914 unit 1 closed in
    G-910-2."""
    assert hasattr(_bs, "_HAVE_FIELD_GALERKIN_ACCEL")
    assert hasattr(_bs, "_FIELD_GALERKIN_FUSED")
    assert _bs._FIELD_GALERKIN_FUSED is True, "production must take the fused route"


def test_g914_2d_the_numpy_path_still_answers(monkeypatch):
    monkeypatch.setattr(_bs, "_HAVE_FIELD_GALERKIN_ACCEL", False)
    z, _ = BSplineSolver(**hub_deck(n_radials=4)).compute_impedance()
    assert abs(z - Z_PINNED["hub"]) < 5e-7, z


# --- G-914-2e: the kernel refuses a malformed call -------------------------


def _good_args(n_basis=6, n_seg=8, nc=2, ns=8, q=6, P=3, A=3, seed=914):
    rng = np.random.default_rng(seed)
    return dict(
        proj=(
            rng.normal(size=(nc * q, ns * q)) + 1j * rng.normal(size=(nc * q, ns * q))
        ),
        W_obs=rng.normal(size=(P, nc, q)),
        W_src=rng.normal(size=(P, ns, q)),
        supp_seg=rng.integers(0, n_seg, size=(n_basis, A)).astype(np.int64),
        polys=rng.normal(size=(n_basis, A, P)),
        pos_o=np.arange(n_seg, dtype=np.int64),
        pos_s=(np.arange(n_seg, dtype=np.int64) % ns),
        i0=0,
        Q=np.zeros((n_basis, n_basis), dtype=np.complex128),
    )


@requires_accel
@pytest.mark.parametrize(
    "field,bad",
    [
        ("W_src", np.zeros((2, 8, 6))),  # P disagrees with W_obs
        ("proj", np.zeros((5, 48), dtype=np.complex128)),  # not n_chunk*q rows
        ("Q", np.zeros((5, 6), dtype=np.complex128)),  # not (n_basis, n_basis)
        ("polys", np.zeros((6, 3, 2))),  # last axis is not P
        ("pos_s", np.arange(3, dtype=np.int64)),  # shorter than pos_o
    ],
)
def test_g914_2e_a_malformed_call_raises(field, bad):
    args = _good_args()
    args[field] = bad
    with pytest.raises(ValueError):
        _acc.assemble_field_galerkin(**args)


@requires_accel
def test_g914_2e_a_segment_id_outside_pos_o_raises():
    """The Python indexes pos_o with supp_seg directly, so an out-of-range id
    is a caller bug — and one that would otherwise read off the end of the
    array with the GIL released."""
    args = _good_args()
    args["supp_seg"][0, 0] = 999
    with pytest.raises(ValueError):
        _acc.assemble_field_galerkin(**args)


# --- G-1115: the accumulation target, and `scale` --------------------------


requires_strided_1115 = pytest.mark.skipif(
    not (
        _bs._HAVE_FIELD_GALERKIN_ACCEL
        and getattr(_acc, "field_galerkin_strided_1115", False)
    ),
    reason="built before momwire#1115 part 3's strided accumulation target",
)

requires_target_1115 = pytest.mark.skipif(
    not (
        _bs._HAVE_FIELD_GALERKIN_ACCEL
        and getattr(_acc, "field_galerkin_target_1115", False)
    ),
    reason="built before momwire#1115's accumulation-target contract",
)


def _chunked_spec(n_basis=16, n_obs=500, ns=64, q=6, P=3, A=3, seed=1115):
    """One assembly big enough that the observer loop must SPLIT it.

    `chunk = (1 << 19) // (n_src * q**2)` is 227 here and `n_obs` is 500, so
    the kernel is entered three times on one `Q` — which is the only shape in
    which `scale` can be got wrong. A scale applied to the TARGET rather than
    to each contribution is exactly `scale * Q` on a single-chunk call and
    wrong from the second chunk on, so a single-chunk gate cannot see it.

    The wings are scattered over the whole observer axis rather than clustered,
    so basis rows accumulate on both sides of every boundary.
    """
    rng = np.random.default_rng(seed)
    n_seg = n_obs
    return dict(
        proj_full=(
            rng.normal(size=(n_obs * q, ns * q))
            + 1j * rng.normal(size=(n_obs * q, ns * q))
        ),
        W_obs_full=rng.normal(size=(P, n_obs, q)),
        W_src=rng.normal(size=(P, ns, q)),
        supp_seg=rng.integers(0, n_seg, size=(n_basis, A)).astype(np.int64),
        polys=rng.normal(size=(n_basis, A, P)),
        pos_o=np.arange(n_seg, dtype=np.int64),
        pos_s=(np.arange(n_seg, dtype=np.int64) % ns),
        n_basis=n_basis,
        n_obs=n_obs,
        ns=ns,
        q=q,
    )


def _run_chunks(spec, *, fused=True, **kw):
    """`spec` assembled the way `_field_galerkin_block` assembles it: one
    accelerator call per observer chunk, all of them into the same `Q`.
    Returns that `Q` and the number of chunks it took."""
    q, ns, n_obs = spec["q"], spec["ns"], spec["n_obs"]
    chunk = max(1, (1 << 19) // max(ns * q * q, 1))
    Q = np.zeros((spec["n_basis"], spec["n_basis"]), dtype=np.complex128)
    n_chunks = 0
    for i0 in range(0, n_obs, chunk):
        i1 = min(i0 + chunk, n_obs)
        _acc.assemble_field_galerkin(
            spec["proj_full"][i0 * q : i1 * q],
            np.ascontiguousarray(spec["W_obs_full"][:, i0:i1]),
            spec["W_src"],
            spec["supp_seg"],
            spec["polys"],
            spec["pos_o"],
            spec["pos_s"],
            i0,
            Q,
            fused,
            **kw,
        )
        n_chunks += 1
    return Q, n_chunks


@pytest.fixture(scope="module")
def chunked():
    return _chunked_spec()


@requires_strided_1115
@pytest.mark.parametrize(
    "why,target",
    [
        # The buried Z is column-major by momwire#136, so this is the target
        # the part 3 lever actually hands over.
        ("column-major", lambda n: np.zeros((n, n), dtype=np.complex128, order="F")),
        ("strided view", lambda n: np.zeros((n, 2 * n), dtype=np.complex128)[:, ::2]),
        ("column block", lambda n: np.zeros((n, n + 3), dtype=np.complex128)[:, :n]),
    ],
)
def test_g1115_a_strided_target_is_accumulated_into_where_it_lies(why, target):
    """Part 3 supersedes part 1's refusal for these three.

    Part 1 REFUSED them, because `py::array_t<..., c_style>` does not require a
    C-contiguous argument, it manufactures one: pybind11 handed the kernel a
    copy and every accumulation went into it, and refusal was the only outcome
    a caller could tell apart from success. Part 3 addresses the target through
    its own strides instead, so these are now accumulated into where they lie.

    Acceptance alone would be a weak pin -- it cannot tell a correct scatter
    from a scrambled one -- so this asserts the numbers equal the C-contiguous
    assembly EXACTLY. Only the addresses differ; the order of operations does
    not, so this is an equality and not a tolerance."""
    ref = _good_args()
    _acc.assemble_field_galerkin(**ref)

    args = _good_args()
    Q = target(args["Q"].shape[0])
    args["Q"] = Q
    _acc.assemble_field_galerkin(**args)
    assert np.count_nonzero(Q), f"{why}: the strided call wrote nothing"
    assert np.array_equal(Q, ref["Q"]), why


@requires_target_1115
@pytest.mark.parametrize(
    "why,target,match",
    [
        ("float64", lambda n: np.zeros((n, n)), "complex128"),
        (
            "read-only",
            lambda n: np.broadcast_to(np.zeros((n, n), dtype=np.complex128), (n, n)),
            "writeable",
        ),
    ],
)
def test_g1115_a_target_that_cannot_be_written_through_is_refused(why, target, match):
    """The same argument as the contiguity refusal: a dtype conversion is a
    copy too, and a read-only target cannot be an accumulator at all."""
    args = _good_args()
    args["Q"] = target(args["Q"].shape[0])
    with pytest.raises(ValueError, match=match):
        _acc.assemble_field_galerkin(**args)


@requires_target_1115
def test_g1115_a_c_contiguous_target_still_writes():
    """The refusals above are worthless if the accepted case stopped writing."""
    args = _good_args()
    _acc.assemble_field_galerkin(**args)
    assert np.count_nonzero(args["Q"])


@requires_target_1115
@pytest.mark.parametrize("fused", [True, False])
def test_g1115_the_default_scale_is_the_unscaled_assembly(fused):
    """`scale=1.0` has to be bit-identical to the assembly before #1115 gave
    it one, which is why it multiplies the contribution and not any factor
    inside the contraction."""
    a, b = _good_args(), _good_args()
    a["fused"] = b["fused"] = fused
    b["scale"] = 1.0
    _acc.assemble_field_galerkin(**a)
    _acc.assemble_field_galerkin(**b)
    assert np.array_equal(a["Q"], b["Q"])


@requires_target_1115
@pytest.mark.parametrize("fused", [True, False])
def test_g1115_scale_minus_one_negates_the_assembly_exactly(fused):
    """The lever this argument exists for is `Z -= Q`, so `scale=-1` has to be
    the exact negation and not merely a close one: IEEE addition is
    sign-symmetric, and the kernel's summation order does not depend on the
    sign of what it sums."""
    one, neg = _good_args(), _good_args()
    one["fused"] = neg["fused"] = fused
    one["scale"], neg["scale"] = 1.0, -1.0
    _acc.assemble_field_galerkin(**one)
    _acc.assemble_field_galerkin(**neg)
    assert np.array_equal(neg["Q"], -one["Q"])


@requires_target_1115
@pytest.mark.parametrize("fused", [True, False])
def test_g1115_scale_multiplies_each_contribution_across_chunks(chunked, fused):
    """THE gate on where `scale` lands. Applied to `Q` once per call instead
    of to each contribution, it would re-scale every chunk already in the
    target: with three chunks the first would come out `scale**3` too large.
    A power-of-two scale makes the comparison exact — scaling by one commutes
    with rounding — so this is an equality, not a tolerance."""
    ref, n_chunks = _run_chunks(chunked, fused=fused, scale=1.0)
    assert n_chunks > 1, f"{n_chunks} chunk(s): this gate needs more than one"
    for s in (-2.0, 0.25, -1.0):
        got, _ = _run_chunks(chunked, fused=fused, scale=s)
        assert np.array_equal(got, s * ref), (s, fused)


@requires_target_1115
@pytest.mark.parametrize("fused", [True, False])
def test_g1115_a_scale_that_is_not_a_power_of_two_is_still_a_scale(chunked, fused):
    """0.1 rounds, so `scale * (a + b)` and `scale * a + scale * b` part in the
    last bits; pinning it at the module's 1e-13 register says the factor is
    applied once per contribution without claiming an exactness that is not
    there."""
    ref, _ = _run_chunks(chunked, fused=fused, scale=1.0)
    got, _ = _run_chunks(chunked, fused=fused, scale=0.1)
    assert _rel(got, 0.1 * ref) <= 1e-13


@requires_target_1115
def test_g1115_both_routes_honour_scale_identically(chunked):
    """`fused=False` is the independent second derivation of these indices
    (G-914-2b). It is also an independent second place for `scale` to land,
    and the two are gated against each other here for the same reason."""
    for s in (1.0, -1.0, 0.1):
        a, _ = _run_chunks(chunked, fused=True, scale=s)
        b, _ = _run_chunks(chunked, fused=False, scale=s)
        assert _rel(a, b) <= 1e-13, s


@requires_accel
def test_g1115_the_contract_flag_and_the_contract_agree():
    """Each flag must mean what a caller gates on it for, in BOTH directions.

    Three builds exist in the wild and a column-major target tells them apart
    by itself: before part 1 it is silently copied and the writes are lost;
    after part 1 it is refused; after part 3 it is accumulated into. A caller
    handing over the buried Z gates on `field_galerkin_strided_1115`, and a
    build that carries the flag without the behaviour -- or the behaviour
    without the flag -- would send it into one of the other two worlds.
    """
    target_flag = getattr(_acc, "field_galerkin_target_1115", False)
    strided_flag = getattr(_acc, "field_galerkin_strided_1115", False)
    assert not (strided_flag and not target_flag), (
        "part 3 implies part 1's contract; a build cannot carry only the later flag"
    )

    args = _good_args()
    n = args["Q"].shape[0]
    args["Q"] = np.zeros((n, n), dtype=np.complex128, order="F")
    try:
        _acc.assemble_field_galerkin(**args)
    except ValueError:
        outcome = "refused"
    else:
        outcome = "written" if np.count_nonzero(args["Q"]) else "silently copied"

    expected = (
        "written" if strided_flag else ("refused" if target_flag else "silently copied")
    )
    assert outcome == expected, (outcome, expected, target_flag, strided_flag)
