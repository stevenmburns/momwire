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
# hub and crossing values are #912's own G-912-5 pins, unchanged.
Z_PINNED = {
    "hub": 141.016615417 - 43.425182328j,
    "crossing": 138.960862256 - 102.609718869j,
    # Measured on the NUMPY path at f537080 and asserted equal on both here;
    # this deck is new with these tests, so the pin is a same-commit record of
    # the reference's answer rather than a value inherited from before it.
    "screen12": 98.787052258 - 67.075970581j,
}


def _capture(build):
    """Every argument tuple `_field_galerkin_block` is called with in one
    solve, plus the Z that solve produced."""
    seen = []
    real = BSplineSolver._field_galerkin_block

    def spy(self, *args):
        seen.append((self, args))
        return real(self, *args)

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
