"""`FieldGround`: the field trunk's ground, as one object (momwire#397).

The gates the interface sketch (`docs/design/field-ground-interface.md`)
asks unit 2 to stand on: the factory's free-space `None`, the five-ground
table reproduced from the object rather than from prose, and a structural
row showing that the point-matched fill follows the OBJECT and not the
`ground_z` / `ground_eps` / `ground_model` strings it used to branch on.

Bit-identity of every grounded fill is not gated here — it is gated where
it always was (`test_sinusoidal_banded_assembly_is_bit_equal`,
`test_sinusoidal_sommerfeld_remainder_replay_is_bit_equal`, the refl-coef
golden rows, the collapse tests), which is the point: this migration is
supposed to be invisible to all of them.
"""

import numpy as np
import pytest

from momwire import SinusoidalSolver, _field_ground, _ground_refl

LAM = 20.0
EPS = (13.0, 0.005)

GROUND_KWARGS = {
    "free": {},
    "pec": {"ground_z": 0.0},
    "refl-coef": {"ground_z": 0.0, "ground_eps": EPS},
    "sommerfeld": {
        "ground_z": 0.0,
        "ground_eps": EPS,
        "ground_model": "sommerfeld",
    },
}


def _solver(ground, **extra):
    """An elevated half-wave dipole over `ground` — high enough that no wire
    end touches the plane, so the #282 contact-charge correction is a no-op
    and the fill's ground arithmetic is the only thing that varies."""
    wires = [[[0.0, -0.24 * LAM, 3.0], [0.0, 0.24 * LAM, 3.0]]]
    return SinusoidalSolver(
        wires=wires,
        nsegs=21,
        wavelength=LAM,
        wire_radius=0.002,
        **GROUND_KWARGS[ground],
        **extra,
    )


def _ground_of(sim):
    geom = sim._build_geometry()
    return geom, _field_ground.field_ground_for(sim, geom, sim.k, sim.omega)


def test_free_space_has_no_ground_object():
    """`field_ground_for` returns `None` in free space, and the fill's ground
    branch is then structurally absent rather than skipped.

    The `None` is load-bearing: a null object with `mode="fold"` and a
    coefficient of 1 would still cost the branch a test per band and would
    make "no ground" a special case of "a ground", which is exactly the
    shape the off-path armour standard (`extended_kernel=False`) forbids.
    """
    _geom, fg = _ground_of(_solver("free"))
    assert fg is None


@pytest.mark.parametrize(
    ("ground", "mode", "weighted", "has_remainder"),
    [
        ("pec", "fold", False, False),
        ("refl-coef", "fold", True, False),
        ("sommerfeld", "compose", False, True),
    ],
)
def test_the_five_ground_table_is_what_the_object_says(
    ground, mode, weighted, has_remainder
):
    """The sketch's five-ground table, read off the object.

    | ground     | mode    | pair_weights | coef  | remainder |
    |------------|---------|--------------|-------|-----------|
    | free       | *(None — no object)*                        |
    | PEC        | fold    | None         | 1     | None      |
    | refl-coef  | fold    | Fresnel dyad | 1     | None      |
    | sommerfeld | compose | None         | C2(e~)| prepare/replay |

    The fifth row (the radial-wire screen) is the one this table is
    DESIGNED against and not implemented for — it lands as a
    `weighted=True, standard_fresnel=False` row with a modified
    reflection-coefficient function one level down in `_ground_refl`, and
    the acceptance test for criterion 1 is that neither sinusoidal file is
    edited for it. Nothing here can gate a ground that does not exist yet;
    what it can gate is that the four that do exist are distinguished by
    these four members and by nothing else.
    """
    sim = _solver(ground)
    geom, fg = _ground_of(sim)
    assert fg is not None
    assert fg.mode == mode

    tabs = fg.pair_weights()
    assert (tabs is not None) is weighted
    if weighted:
        # The k-independent quintet, at the whole geometry when no observer
        # window is named, and at the window when one is.
        n = geom["n_segs"]
        assert tabs.cos_th.shape == (n, n)
        band = fg.pair_weights((2, 7))
        assert band.cos_th.shape == (5, n)
        assert band.tm_p.shape == (5, n)

    rem = fg.remainder()
    assert (rem is not None) is has_remainder
    if has_remainder:
        # One prepared state per source-shape set, memoized: a "compose"
        # consumer asking twice inside one fill must not rebuild the
        # remainder's source side (#357 item 1).
        assert fg.remainder() is rem
        assert fg.remainder("cos-1") is not rem

    if ground == "sommerfeld":
        eps_t = _ground_refl.eps_tilde(sim.ground_eps, sim.omega, sim.eps)
        assert fg.eps_tilde == eps_t
        assert fg.image_coefficient == (eps_t - 1.0) / (eps_t + 1.0)
    else:
        assert fg.image_coefficient == 1.0

    # Every ground momwire ships today has the stock Fresnel dyad, so every
    # one of them is eligible for the fused C++ kernels that compute
    # rho_v/rho_h in-kernel. The flag exists for the ground that will not
    # be (the screen), which bypasses them with zero solver edits.
    assert fg.standard_fresnel is True


def test_the_image_source_mirror_map_comes_from_the_object():
    """`image_sources` is the mirror map, and it is the solver's own — the
    ground supplies mirrored GEOMETRY and never the extended kernel's policy
    about it (EK eligibility is still scored solver-side against these
    sources via `mirror=True`)."""
    sim = _solver("refl-coef")
    geom, fg = _ground_of(sim)
    src_c, src_t = fg.image_sources()
    ref_c, ref_t = sim._image_source_centers_tangents(geom)
    assert np.array_equal(src_c, ref_c)
    assert np.array_equal(src_t, ref_t)


@pytest.mark.parametrize("extended_kernel", [False, True])
def test_a_non_standard_fresnel_ground_bypasses_the_fused_kernels(
    extended_kernel, monkeypatch
):
    """The unit-1 amendment, gated: the fused C++ refl kernels compute the
    Fresnel dyad IN-KERNEL — a third spelling of it, which the sketch's
    weight row did not anticipate — so they are an optimization keyed to
    standard-Fresnel grounds, not the general path.

    `standard_fresnel=False` therefore has to route the same ground to
    `PairWeights.project`, and bit-identically to what an unavailable
    accelerator would produce. That is what makes the radial-wire screen a
    coefficient change one level down rather than a kernel port: it will
    arrive with modified rho_v/rho_h, which no fused kernel knows how to
    compute, and the backend dispatch must already be out of its way.
    """
    import momwire.sinusoidal as sin_mod

    have_fused = (
        sin_mod._HAVE_FIELD_TENSOR_EK_REFL
        if extended_kernel
        else sin_mod._HAVE_FIELD_TENSOR_REFL
    )
    if not have_fused:
        pytest.skip("no fused refl kernel in this build — nothing to bypass")
    sim = _solver("refl-coef", extended_kernel=extended_kernel)
    geom, fg = _ground_of(sim)

    fused = sim._field_tensor_image_refl(geom, sim.k, ground=fg)

    fg.standard_fresnel = False
    bypassed = sim._field_tensor_image_refl(geom, sim.k, ground=fg)

    fg.standard_fresnel = True
    monkeypatch.setattr(sin_mod, "_HAVE_FIELD_TENSOR_REFL", False)
    monkeypatch.setattr(sin_mod, "_HAVE_FIELD_TENSOR_EK_REFL", False)
    numpy_path = sim._field_tensor_image_refl(geom, sim.k, ground=fg)

    for got, ref in zip(bypassed, numpy_path):
        assert np.array_equal(got, ref), (
            "a non-standard-Fresnel ground did not reach the numpy projector "
            "— the fused kernel would silently compute the stock dyad for it"
        )
    # The fused kernel is a different SPELLING of the same dyad — bit-close,
    # not bit-equal (its rho_v/rho_h never leave C++) — which is why routing
    # around it has to be a decision the object makes rather than something
    # a caller could be trusted to notice.
    np.testing.assert_allclose(fused[0], numpy_path[0], rtol=1e-9, atol=1e-30)


def test_the_point_matched_fill_follows_the_object_not_the_strings(monkeypatch):
    """Structural row for unit 2: `_assemble_Z` reads the `FieldGround`, so
    handing it a DIFFERENT ground than its own strings describe changes the
    matrix it fills.

    A solver configured for the refl-coef ground is given a PEC ground
    object instead. If the band loop still branched on `ground_eps` /
    `ground_model` the fill would be unmoved; because it branches on the
    object, the result is the PEC solver's matrix — bit for bit, since the
    two reach the identical evaluator by the identical schedule. The deck
    is elevated, so the one thing downstream of the loop that still reads
    `ground_eps` (the #282 contact-charge correction, which is a
    testing-scheme correction and deliberately out of the sketch's scope)
    has no contact to correct and cannot mask the swap.

    The positive half of the row is the call counts: exactly one ground per
    fill (the per-fill hoists are its construction), one `image_field` per
    observer band, and a remainder replay per band under the composing
    ground and never under a folding one.
    """
    calls = {"built": 0, "image": 0, "replay": 0}
    real_factory = _field_ground.field_ground_for
    real_image = _field_ground.FieldGround.image_field
    real_replay = _field_ground.Remainder.replay

    def spy_factory(*a, **kw):
        calls["built"] += 1
        return real_factory(*a, **kw)

    def spy_image(self, obs_rows=None):
        calls["image"] += 1
        return real_image(self, obs_rows=obs_rows)

    def spy_replay(self, *a, **kw):
        calls["replay"] += 1
        return real_replay(self, *a, **kw)

    monkeypatch.setattr(_field_ground, "field_ground_for", spy_factory)
    monkeypatch.setattr(_field_ground.FieldGround, "image_field", spy_image)
    monkeypatch.setattr(_field_ground.Remainder, "replay", spy_replay)

    # Reference matrices: what each ground's own solver fills.
    G_ref = {}
    for ground in ("pec", "refl-coef", "sommerfeld"):
        sim = _solver(ground)
        geom = sim._build_geometry()
        calls["built"] = calls["image"] = calls["replay"] = 0
        G_ref[ground], _ = sim._assemble_Z(geom, sim.k)
        n_bands = 1  # N = 21 is below the dense-assembly threshold
        assert calls["built"] == 1, "the fill built more than one ground"
        assert calls["image"] == n_bands
        assert calls["replay"] == (n_bands if ground == "sommerfeld" else 0)

    assert not np.array_equal(G_ref["pec"], G_ref["refl-coef"])

    # The swap: a refl-coef solver handed the PEC ground object.
    sim = _solver("refl-coef")
    geom = sim._build_geometry()
    pec_ground = _field_ground.FieldGround(
        sim,
        geom,
        sim.k,
        sim.omega,
        mode="fold",
        weighted=False,
        eps_tilde=None,
        image_coefficient=1.0,
    )
    monkeypatch.setattr(_field_ground, "field_ground_for", lambda *a, **kw: pec_ground)
    G_swapped, _ = sim._assemble_Z(geom, sim.k)
    assert np.array_equal(G_swapped, G_ref["pec"]), (
        "the fill ignored the ground object it was handed — the band loop is "
        "reading ground_eps/ground_model again"
    )
