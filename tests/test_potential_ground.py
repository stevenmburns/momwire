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


def _solver(ground, degree=2, **extra):
    """An elevated half-wave dipole over `ground` — high enough that no wire
    end touches the plane, so the ground-junction basis (#151) is a no-op
    and the fill's ground arithmetic is the only thing that varies."""
    wires = [[[0.0, -0.24 * LAM, 4.0], [0.0, 0.24 * LAM, 4.0]]]
    return BSplineSolver(
        wires=wires,
        nsegs=21,
        wavelength=LAM,
        wire_radius=0.02,
        degree=degree,
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


@pytest.mark.parametrize("ground", ["pec", "refl-coef", "sommerfeld"])
def test_the_weight_windows_default_to_the_geometrys_own_segments(ground):
    """`weight_windows()` is `weight_windows(observers=own, sources=own)`,
    bit for bit — the unit-4 anti-drift pin.

    Unit 4 (razor's refl-coef ground) generalised this producer the same
    way unit 2 generalised `ImageGeometry`: the OPERATION is "Fresnel
    weights between an observer set and a source set", and the B-spline
    fill's square geometry-shaped call is now a DEFAULT over it rather
    than the definition. Razor's observers are its testing-path quadrature
    points and its segment centroids — neither is the source set, and its
    `geom` carries none of the keys the default reads.

    A default that merely *agreed* with the explicit spelling would be a
    latent divergence: `array_equal` here, not `allclose`, is what keeps
    the B-spline fill's every grounded answer unmoved by the
    generalisation.
    """
    sim = _solver(ground)
    geom, pg = _ground_for(sim)
    n = geom["n_segs_total"]
    own = pg.own_segments()
    assert np.array_equal(own[0], 0.5 * (geom["seg_l"] + geom["seg_r"]))
    assert own[1] is geom["tangents"]

    for i0, i1 in ((0, n), (3, 9)):
        a = pg.weight_windows()(i0, i1)
        b = pg.weight_windows(observers=own, sources=own)(i0, i1)
        c = pg.weight_windows(sources=own)(i0, i1)
        for x, y, z in zip(a, b, c):
            assert np.array_equal(x, y)
            assert np.array_equal(x, z)


def test_the_weight_windows_take_an_observer_set_that_is_not_the_sources():
    """The operation itself: an arbitrary observer set, at a shape no
    geometry axis has.

    Razor's A-term observers are `(n_basis · n_path)` path quadrature
    points against `n_segs` sources — a rectangular block whose row count
    is unrelated to the source count — and each row carries the PATH's
    tangent there, not a segment's. The result must be exactly the rows
    the square call would give when those observers happen to BE the
    segments, which is what makes the razor fill's PEC limit land on the
    PEC image.
    """
    sim = _solver("refl-coef")
    geom, pg = _ground_for(sim)
    seg_c, tang = pg.own_segments()
    n = geom["n_segs_total"]

    # A deliberately odd observer set: three segment centres in reverse
    # order, so it is neither a slice of the source axis nor its length.
    picks = [n - 1, 0, n // 2]
    obs = (seg_c[picks], tang[picks])
    w_A, w_Phi = pg.weight_windows(observers=obs, sources=(seg_c, tang))(0, 3)
    assert w_A.shape == (3, n)
    ref_A, ref_Phi = pg.weight_windows()(0, n)
    for got, ref in ((w_A, ref_A), (w_Phi, ref_Phi)):
        for row, pick in enumerate(picks):
            assert np.array_equal(got[row], ref[pick])


def test_the_mirror_map_is_the_grounds():
    """`image_geometry` is the mirror map: two OPERATIONS, and the
    B-spline-shaped conveniences built on top of them.

    Unit 2 turned this class inside out. It used to be its first consumer's
    shape — mirrored segment ENDPOINTS plus a fused N² image-sign table,
    because that is what the B-spline moment builder eats — and razor,
    which quadratures in each segment's own arc frame and must not
    materialise anything N², needed a different shape from the same
    mirror. So `mirror_positions` / `mirror_tangents` are the object now,
    `seg_l` / `seg_r` / `tangent_dot()` are conveniences over them, and the
    conveniences are LAZY: razor's `geom` dict does not even carry the keys
    they read.
    """
    sim = _solver("pec")
    geom, pg = _ground_for(sim)
    img = pg.image_geometry()

    # the operations, on arbitrary arrays neither consumer's geom holds
    pts = np.array([[1.0, -2.0, 3.0], [0.0, 0.0, 0.0]])
    assert np.array_equal(
        img.mirror_positions(pts),
        np.array([[1.0, -2.0, 2 * sim.ground_z - 3.0], [0.0, 0.0, 2 * sim.ground_z]]),
    )
    assert np.array_equal(img.mirror_tangents(pts), pts * np.array([1.0, 1.0, -1.0]))
    # ...non-destructive, and an involution
    assert np.array_equal(img.mirror_positions(img.mirror_positions(pts)), pts)
    assert np.array_equal(pts[0], [1.0, -2.0, 3.0])

    # the conveniences, still bit-equal to the solver methods hmatrix's
    # un-migrated block paths keep calling — the anti-drift pin
    assert np.array_equal(img.seg_l, sim._image_positions(geom["seg_l"]))
    assert np.array_equal(img.seg_r, sim._image_positions(geom["seg_r"]))
    assert np.array_equal(img.tangent_dot(), sim._image_tangent_dot(geom["tangents"]))
    # The mirror is an involution about the plane, and only z moves.
    assert np.array_equal(img.seg_l[:, :2], geom["seg_l"][:, :2])
    assert np.allclose(img.seg_l[:, 2], 2 * sim.ground_z - geom["seg_l"][:, 2])


def test_the_conveniences_are_lazy_and_the_operations_are_not():
    """A `geom` without the B-spline keys still yields a working mirror.

    This is exactly razor's situation (its geometry is `seg_p0` / `seg_t` /
    `seg_h`, no `seg_l` / `seg_r` / `tangents`), and it is the property
    that lets one ground object serve two differently-shaped fills. Reading
    a convenience that the geometry cannot supply is a `KeyError` and not a
    silent wrong answer.
    """
    from momwire._potential_ground import ImageGeometry

    img = ImageGeometry(2.0, {"seg_p0": np.zeros((3, 3))})
    assert np.array_equal(
        img.mirror_positions(np.array([[0.0, 0.0, 5.0]])),
        np.array([[0.0, 0.0, -1.0]]),
    )
    with pytest.raises(KeyError):
        img.seg_l


def _galerkin_Q_from_the_operation(sim, geom, supp_seg, polys, pg):
    """`evaluate`'s Galerkin block, rebuilt from `field_windows` alone.

    The B-spline testing rule written out by hand on top of the shared
    operation: source arc moments come from the producer, the observer-side
    moment quadrature and the basis assembly are this caller's own — which
    is exactly the division razor and pulse need and the division
    `_Z_sommerfeld_remainder`'s fused kernel does internally.
    """
    from momwire._quadrature import leggauss

    rem = pg.remainder()
    d, q = sim.degree, sim.n_qp_sommerfeld
    seg_l, seg_r, tang = rem.source_segments()
    h = geom["h_per_seg"]
    n_seg = seg_l.shape[0]

    xg, wg = leggauss(q)
    tq = 0.5 * (xg + 1.0)
    nodes = seg_l[:, None, :] + tq[None, :, None] * (seg_r - seg_l)[:, None, :]
    u = h[:, None] * tq[None, :]
    W = (0.5 * h[:, None] * wg[None, :])[None] * u[None] ** np.arange(d + 1)[
        :, None, None
    ]

    obs = nodes.reshape(n_seg * q, 3)
    t_obs = np.repeat(tang, q, axis=0)
    fm = rem.field_windows((obs, t_obs), n_moment=d + 1, n_qp=q)(0, n_seg * q)
    Jc = np.einsum("piq,iqjP->pPij", W, fm.reshape(n_seg, q, n_seg, d + 1))

    Q = np.zeros((polys.shape[0], polys.shape[0]), dtype=np.complex128)
    for a in range(d + 1):
        sm = supp_seg[:, a]
        for b in range(d + 1):
            sn = supp_seg[:, b]
            J_blk = Jc[:, :, sm[:, None], sn[None, :]]
            Q += np.einsum("mp,pPmn,nP->mn", polys[:, a, :], J_blk, polys[:, b, :])
    return Q


@pytest.mark.parametrize("degree", [1, 2])
def test_the_remainder_exposes_the_field_and_evaluate_is_the_convenience(degree):
    """The unit-5 generalisation, pinned the only way it can be.

    `Remainder` had exactly one method, `evaluate(supp_seg, polys)`, and it
    is a B-spline signature twice over: the arguments are that basis's
    description and the return value is a finished Galerkin block. Two
    independent consumers — razor (path-integral rows) and pulse (point
    matching, momwire#416 §4.1) — reported the same blocker, so the class
    now exposes the OPERATION: the remainder FIELD of each source segment's
    arc moments, projected on the observer tangents, in observer windows.

    Unlike units 2 and 4, the two spellings are pinned at the quadrature's
    own agreement rather than at `array_equal`, and that is a property of
    the kernels rather than of the physics: `evaluate`'s hot path is the
    fused C++ `sommerfeld_remainder_bspline_Q`, which never forms the
    intermediate this operation returns. Measured worst relative residual
    over the two degrees: 2.9e-15 (degree 1; 2.0e-15 at degree 2) — the
    same class as the existing
    fused-vs-numpy gate in `tests/test_sommerfeld_ground.py`, which is what
    says the two are the same integral and not merely similar ones.
    """
    sim = _solver("sommerfeld", degree=degree)
    geom = sim._build_geometry()
    supp_seg, polys, _kcl, _wk, _wbg = sim._build_basis_polynomials(geom)
    _g, pg = _ground_for(sim)

    Q_ref = pg.remainder().evaluate(supp_seg, polys)
    Q_op = _galerkin_Q_from_the_operation(sim, geom, supp_seg, polys, pg)
    rel = np.abs(Q_op - Q_ref).max() / np.abs(Q_ref).max()
    assert rel < 1e-11, f"degree {degree}: the two spellings differ by {rel:.3e}"


def test_the_field_windows_are_rows_and_the_pulse_shape_is_the_first_moment():
    """The producer's contract: windows are row bands of one table, and
    `n_moment` is the only thing that separates the two consumers.

    Row-locality has to be exact (`array_equal`, not `allclose`) or a
    consumer's answer would depend on its chunk schedule — the same
    property `weight_windows` holds, and the same one razor's fill gates
    end-to-end. `n_moment = 1` is pulse's plain rectangular field and
    `n_moment = 2` razor's tent pair; the zeroth moment must be the
    identical array in both, or the two consumers are integrating
    different things.
    """
    sim = _solver("sommerfeld")
    geom, pg = _ground_for(sim)
    rem = pg.remainder()
    seg_c, tang = pg.own_segments()
    n = geom["n_segs_total"]
    obs = (seg_c, tang)

    full = rem.field_windows(obs, n_moment=2)(0, n)
    assert full.shape == (n, n, 2)
    assert full.dtype == np.complex128
    band = rem.field_windows(obs, n_moment=2)(3, 9)
    assert np.array_equal(band, full[3:9])

    pulse = rem.field_windows(obs, n_moment=1)(0, n)
    assert pulse.shape == (n, n, 1)
    assert np.array_equal(pulse[..., 0], full[..., 0])

    # ...and the source axis defaults to the geometry's own segments.
    explicit = rem.field_windows(obs, rem.source_segments(), n_moment=2)(0, n)
    assert np.array_equal(explicit, full)
    assert np.array_equal(rem.source_segments()[0], geom["seg_l"])


def test_the_remainder_grid_is_built_at_the_grounds_wavenumber(monkeypatch):
    """The unit-5 schedule rule, at the layer it is decided.

    The Sommerfeld grid is k-dependent through and through (its lattice is
    in wavelengths), so a consumer with a prepare/replay split must build
    it per SOLVED wavenumber. `BSplineSolver` cannot tell the difference —
    its factory call passes `self.k` and its swept path moves `self.k`
    itself — but razor sweeps by passing k to the factory while `self.k`
    stays put, so a grid keyed off the solver would freeze at the
    constructor's frequency and every swept point would be wrong.
    """
    from momwire import _sommerfeld

    seen = []
    real = _sommerfeld.get_grid
    monkeypatch.setattr(
        _sommerfeld,
        "get_grid",
        lambda eps_t, k2, r1_max, **kw: (
            seen.append((k2, kw["omega"])) or real(eps_t, k2, r1_max, **kw)
        ),
    )

    sim = _solver("sommerfeld")
    geom = sim._build_geometry()
    k, omega = 1.37 * sim.k, 1.37 * sim.omega
    pg = _potential_ground.potential_ground_for(sim, geom, k, omega)
    obs = pg.own_segments()
    pg.remainder().field_windows(obs)(0, 2)

    assert seen == [(k, omega)], (
        f"the remainder built its grid at {seen}, not at the ground's own "
        f"({k}, {omega}) — it is reading solver.k"
    )


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
