"""`PotentialGround`: the mixed-potential trunk's ground, as one object
(momwire#398 unit 1).

The potential-trunk twin of `test_field_ground.py`, and the same three
things it gates: the factory's free-space `None`, the ground table read off
the object rather than off prose, and a structural row showing that the
bspline dense fill follows the OBJECT and not the `ground_z` / `ground_eps`
/ `ground_model` / `ground_phi_mode` strings it used to branch on.

Bit-identity of every grounded fill is not gated here — it is gated where
it always was (`test_momwire.py`'s bspline ground blocks, the refl-coef
golden rows, `test_sommerfeld_ground.py`'s route-equivalence tests,
`test_hmatrix.py` and `test_array_block.py`'s dense references), which is
the point: this migration is supposed to be invisible to all of them.
"""

import numpy as np
import pytest

from momwire import BSplineSolver, _ground_refl, _potential_ground

LAM = 20.0
EPS = (10.0, 0.002)

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
    end touches the plane, so the ground-junction basis (#151) is a no-op
    and the fill's ground arithmetic is the only thing that varies."""
    wires = [[[0.0, -0.24 * LAM, 4.0], [0.0, 0.24 * LAM, 4.0]]]
    return BSplineSolver(
        wires=wires,
        nsegs=21,
        wavelength=LAM,
        wire_radius=0.02,
        degree=2,
        **GROUND_KWARGS[ground],
        **extra,
    )


def _ground_for(sim):
    geom = sim._build_geometry()
    return geom, _potential_ground.potential_ground_for(sim, geom, sim.k, sim.omega)


def _Z(sim):
    geom = sim._build_geometry()
    supp_seg, polys, _kcl, _wk, _wbg = sim._build_basis_polynomials(geom)
    return sim._compute_Z_operator(geom, supp_seg, polys)


def test_free_space_has_no_ground_object():
    """`potential_ground_for` returns `None` in free space, and the fill's
    ground branch is then structurally absent rather than skipped — the
    same standard `extended_kernel=False` holds itself to, and the reason
    a free-space solve cannot pay one float operation for this migration.
    """
    sim = _solver("free")
    geom, pg = _ground_for(sim)
    assert pg is None
    assert sim.ground_z is None
    for ground in ("pec", "refl-coef", "sommerfeld"):
        assert _ground_for(_solver(ground))[1] is not None


@pytest.mark.parametrize(
    ("ground", "mode", "weighted", "phi_mode", "has_remainder"),
    [
        ("pec", "fold", False, None, False),
        ("refl-coef", "fold", True, "normal", False),
        ("sommerfeld", "compose", True, None, True),
    ],
)
def test_the_ground_table(ground, mode, weighted, phi_mode, has_remainder):
    """The factory's five-ground table (`potential_ground_for`'s docstring),
    read off the object.

    `weighted` here is "does `weight_tables` hand back a pair" — `None` is
    PEC alone, because PEC is the one ground the tensor route assembles
    through a genuinely different, unweighted kernel. The Φ-mode column is
    the row with no field-trunk analogue at all (architecture doc §2.2).
    """
    sim = _solver(ground)
    geom, pg = _ground_for(sim)

    assert pg.mode == mode
    assert pg.phi_mode == phi_mode
    assert (pg.remainder() is not None) == has_remainder
    # The remainder and the mode are the same fact asked two ways.
    assert (pg.remainder() is not None) == (pg.mode == "compose")

    tables = pg.weight_tables()
    assert (tables is not None) == weighted
    if tables is not None:
        n = geom["n_segs_total"]
        for w in tables:
            assert w.shape == (n, n)
            assert w.dtype == np.complex128

    # The image coefficient: 1 except for the composing ground's C2(ε̃).
    if ground == "sommerfeld":
        eps_t = _ground_refl.eps_tilde(sim.ground_eps, sim.omega, sim.eps)
        assert pg.eps_tilde == eps_t
        assert pg.image_coefficient == (eps_t - 1.0) / (eps_t + 1.0)
        # C2 enters through the CONSTANT tables, not through a multiply the
        # consumer performs — the potential trunk's own spelling of the
        # field trunk's `np.multiply(coef, block, out=block)` contract.
        w_A, w_Phi = tables
        td_img = sim._image_tangent_dot(geom["tangents"])
        assert np.array_equal(w_A, pg.image_coefficient * td_img.astype(np.complex128))
        assert np.array_equal(w_Phi, np.full_like(w_A, pg.image_coefficient))
    else:
        assert pg.image_coefficient == 1.0
        assert (pg.eps_tilde is None) == (ground == "pec")

    # Every ground momwire ships is standard-Fresnel; the screen is the row
    # that will not be. Nothing on the dense fill READS this — see the
    # attribute's docstring for the kernel that will.
    assert pg.standard_fresnel is True


def test_the_weight_windows_never_answer_none():
    """The documented asymmetry between the two weight accessors: the same
    PEC ground answers `None` from `weight_tables` and a real
    `(td_img, ones)` pair from `weight_windows`.

    It is not an inconsistency in the ground, it is the two kernels — the
    windowed C++ assembler speaks only weighted, so the chunked route has
    to spell "no weight" out as tables while the tensor route has a kernel
    that says it structurally. Gated because a future consumer that treats
    the two accessors as the same question would silently drop the PEC
    image on one route.
    """
    for ground in ("pec", "refl-coef", "sommerfeld"):
        sim = _solver(ground)
        geom, pg = _ground_for(sim)
        n = geom["n_segs_total"]
        w_A, w_Phi = pg.weight_windows()(0, 4)
        for w in (w_A, w_Phi):
            assert w.shape == (4, n)
            assert w.dtype == np.complex128
            assert w.flags.c_contiguous

    sim = _solver("pec")
    geom, pg = _ground_for(sim)
    assert pg.weight_tables() is None
    w_A, w_Phi = pg.weight_windows()(0, geom["n_segs_total"])
    assert np.array_equal(w_A, pg.image_geometry().tangent_dot().astype(np.complex128))
    assert np.array_equal(w_Phi, np.ones_like(w_A))


def test_the_mirror_map_is_the_grounds():
    """`image_geometry` is the mirror map: mirrored ENDPOINTS for the moment
    builder plus the fused image-sign table `td_img`.

    Two parts because this trunk splits the image in two where the field
    trunk has one mirrored source set — and `tangent_dot()` is built only
    when asked, so the moment builder (which wants endpoints alone) never
    pays for the O(N²) table. This is the surface razor consumes in unit 2.
    """
    sim = _solver("pec")
    geom, pg = _ground_for(sim)
    img = pg.image_geometry()
    assert np.array_equal(img.seg_l, sim._image_positions(geom["seg_l"]))
    assert np.array_equal(img.seg_r, sim._image_positions(geom["seg_r"]))
    assert np.array_equal(img.tangent_dot(), sim._image_tangent_dot(geom["tangents"]))
    # The mirror is an involution about the plane, and only z moves.
    assert np.array_equal(img.seg_l[:, :2], geom["seg_l"][:, :2])
    assert np.allclose(img.seg_l[:, 2], 2 * sim.ground_z - geom["seg_l"][:, 2])


def test_the_dense_fill_follows_the_object_not_the_strings(monkeypatch):
    """The structural row: `_compute_Z_operator` reads the
    `PotentialGround`, so handing it a DIFFERENT ground than its own
    strings describe changes the matrix it fills.

    A solver configured for the refl-coef ground is given a PEC ground
    object instead. If the fill still branched on `ground_eps` /
    `ground_model` it would be unmoved; because it branches on the object,
    the result is the PEC solver's matrix — bit for bit, since the two
    reach the identical kernels by the identical schedule. The deck is
    elevated, so nothing downstream (the #151 ground-junction basis) can
    mask the swap.

    Run on BOTH image routes: the chunked weighted-windowed accumulator a
    default solve takes, and the no-accelerator moment-tensor fallback,
    because they consume different members of the object
    (`weight_windows` / `remainder` against `weight_tables` /
    `image_geometry`).
    """
    from momwire import bspline as bspline_mod

    if not bspline_mod._HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL:
        pytest.skip("weighted windowed Z assembly accelerator not built")

    for tensor_route in (False, True):
        monkeypatch.setattr(
            bspline_mod,
            "_HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL",
            not tensor_route,
        )
        Z_ref = {g: _Z(_solver(g)) for g in ("pec", "refl-coef", "sommerfeld")}
        assert not np.array_equal(Z_ref["pec"], Z_ref["refl-coef"])
        assert not np.array_equal(Z_ref["pec"], Z_ref["sommerfeld"])

        sim = _solver("refl-coef")
        geom = sim._build_geometry()
        pec_ground = _potential_ground.PotentialGround(
            sim,
            geom,
            sim.k,
            sim.omega,
            mode="fold",
            eps_tilde=None,
            image_coefficient=1.0,
        )
        monkeypatch.setattr(
            _potential_ground,
            "potential_ground_for",
            lambda *a, **kw: pec_ground,
        )
        assert np.array_equal(_Z(sim), Z_ref["pec"]), (
            f"(tensor_route={tensor_route}) the fill ignored the ground "
            "object it was handed — it is reading ground_eps/ground_model "
            "again"
        )
        monkeypatch.undo()


def test_one_ground_per_fill_and_the_remainder_only_composes(monkeypatch):
    """The positive half of the structural row: the call counts.

    Exactly one ground object per dense fill (its construction IS the
    per-fill ε̃/C2 hoist), one weight-window producer, and a remainder
    evaluated exactly once under the composing ground and never under a
    folding one.
    """
    calls = {"built": 0, "windows": 0, "tables": 0, "remainder": 0}
    real_factory = _potential_ground.potential_ground_for
    real_windows = _potential_ground.PotentialGround.weight_windows
    real_tables = _potential_ground.PotentialGround.weight_tables
    real_eval = _potential_ground.Remainder.evaluate

    def spy_factory(*a, **kw):
        calls["built"] += 1
        return real_factory(*a, **kw)

    def spy_windows(self):
        calls["windows"] += 1
        return real_windows(self)

    def spy_tables(self):
        calls["tables"] += 1
        return real_tables(self)

    def spy_eval(self, *a, **kw):
        calls["remainder"] += 1
        return real_eval(self, *a, **kw)

    monkeypatch.setattr(_potential_ground, "potential_ground_for", spy_factory)
    monkeypatch.setattr(
        _potential_ground.PotentialGround, "weight_windows", spy_windows
    )
    monkeypatch.setattr(_potential_ground.PotentialGround, "weight_tables", spy_tables)
    monkeypatch.setattr(_potential_ground.Remainder, "evaluate", spy_eval)

    for ground in ("free", "pec", "refl-coef", "sommerfeld"):
        for key in calls:
            calls[key] = 0
        _Z(_solver(ground))
        assert calls["built"] == 1, f"{ground}: the fill built {calls['built']} grounds"
        if ground == "free":
            # No object exists, so no member of it can be read.
            assert calls["windows"] == calls["tables"] == calls["remainder"] == 0
            continue
        assert calls["windows"] == 1, f"{ground}: {calls['windows']} window producers"
        assert calls["tables"] == 0, (
            f"{ground}: the chunked route asked for whole-geometry weight "
            "tables — that is the (N, N) residency #323 retired"
        )
        assert calls["remainder"] == (1 if ground == "sommerfeld" else 0)
