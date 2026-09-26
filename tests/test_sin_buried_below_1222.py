"""momwire#1222 (#1220 stage 1): wholly-buried and detached decks on the
point-matched `SinusoidalSolver`.

The lane is called CORRECT, not a twin: NEC-4.2 has not been run (Steve,
2026-09-26), so every gate here is secondary — an exact limit, a structural
identity, or convergence onto SG / bspline — and never cross-basis agreement
at a fixed mesh. Collocation and Galerkin differ at finite N by a gap that is
LEGITIMATE (0.9 % on a 1 m dipole at N = 41 in free space); what is gated is
that the buried route adds nothing to it.

Measured 2026-09-26, soil A (13, 0.005 S/m), 7 MHz, laptop:

    complex twin vs numpy, each component, both shapes     <= 1.0e-15
    eps~ = 1: buried vs free space                         3.2e-11 / 4.0e-10
    eps~ = 1: detached Y vs free space (diag / all)        1.5e-8 / 9.8e-6
    deep (1.5 m) vs the infinite medium  v / h             2.1e-4 / 3.0e-4
    shallow (0.15 m) vs the infinite medium  v / h         7.9e-3 / 4.4e-2
    SS-SG gap, infinite medium vs free space (N=41)        9.07e-3 vs 9.26e-3
    SS vs bspline(161), N = 11/21/41/81 (vertical)         9.1/5.4/3.1/1.6 e-2
    interface term Z(d)-Z_inf, SS vs SG (N=41)             1.7e-2 ; A_m flipped 2.0
    reciprocity |Y12-Y21|/|Y12|, bent buried m=1/2/4       1.2e-2/3.5e-3/2.0e-3
    reciprocity, detached m=1/2/4                          2.9e-3/1.1e-3/5.4e-4
    far-apart detached: blocks vs single-class solves      2.6e-15 / 0 ; cross 1e-7
    loading shift vs SG(segment feed), bent m=1/2/4/8      7.7/3.1/1.7/0.98 e-3
    jacket charge term alone vs SG, bent m=1/2/4/8          7.5/2.9/1.5/0.82 e-3
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from momwire import (
    BSplineSolver,
    SinusoidalGalerkinSolver,
    SinusoidalSolver,
    _field_ground,
    _ground_refl,
    _sommerfeld_below,
    _wire_loading,
)
from momwire import sinusoidal as sin_mod
from momwire._below_interface import BURIED_EXTENDED_KERNEL_REFUSAL
from momwire._medium_spec import BURIED_PEC_REFUSAL, BURIED_REFL_REFUSAL

from test_sinusoidal_complex_k_980 import bspline_infinite_medium

C0 = 299792458.0
WL7 = C0 / 7e6
SOIL_A = (13.0, 0.005)
EPS_ONE = (1.0, 0.0)
GROUND_KEYS = ("ground_z", "ground_eps", "ground_model")

needs_twin = pytest.mark.skipif(
    not sin_mod._HAVE_FIELD_TENSOR_CPLX,
    reason="no compiled sinusoidal accelerator in this build",
)


def ground(eps=SOIL_A):
    return dict(ground_z=0.0, ground_eps=eps, ground_model="sommerfeld")


def dipole(n=21, depth=0.15, vertical=True, eps=SOIL_A, free=False, length=1.0):
    """The phase-0 buried dipole (SG D1's `sg_dipole`), centre fed."""
    pts = (
        np.array([(0.0, 0.0, -(depth + length)), (0.0, 0.0, -depth)])
        if vertical
        else np.array([(-0.5 * length, 0.0, -depth), (0.5 * length, 0.0, -depth)])
    )
    arc = (((n + 1) // 2) - 0.5) / n * length
    return dict(
        wires=[pts],
        n_per_edge_per_wire=[[n]],
        feeds=[(0, arc, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        **({} if free else ground(eps)),
    )


def buried_bent(m=1, eps=SOIL_A):
    """A wholly-buried 2-port with a BEND, so reciprocity is not vacuous."""
    return dict(
        wires=[
            np.array([(-2.5, 0.0, -0.5), (0.0, 0.0, -0.5), (0.0, 2.0, -0.8)]),
            np.array([(-2.5, 1.0, -1.0), (2.5, 1.5, -1.0)]),
        ],
        n_per_edge_per_wire=[[5 * m + 1, 4 * m + 1], [10 * m + 1]],
        feeds=[(0, 1.3, 1 + 0j), (1, 2.6, 1 + 0j)],
        wavelength=WL7,
        wire_radius=1e-3,
        **ground(eps),
    )


def detached(m=1, eps=SOIL_A, sep=0.0):
    """A DETACHED 2-port: a buried wire under a bent wire 1 m up, no junction
    in the plane. `sep` slides the buried wire away along x."""
    return dict(
        wires=[
            np.array([(-2.5 + sep, 0.0, -0.5), (2.5 + sep, 0.0, -0.5)]),
            np.array([(-2.5, 0.0, 1.0), (0.0, 0.0, 1.0), (2.0, 1.5, 1.4)]),
        ],
        n_per_edge_per_wire=[[10 * m + 1], [5 * m + 1, 5 * m]],
        feeds=[(0, 2.5, 1 + 0j), (1, 1.3, 1 + 0j)],
        wavelength=WL7,
        wire_radius=1e-3,
        **ground(eps),
    )


def free_space(d):
    return {k: v for k, v in d.items() if k not in GROUND_KEYS}


def z_of(cls, d, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return complex(np.ravel(cls(**d, **kw).compute_impedance()[0])[0])


def y_of(cls, d, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls(**d, **kw).compute_port_solution().y


def rel(a, b):
    return abs(a - b) / abs(b)


def infinite_medium(cls, d):
    """The step-A seam: free-space geometry driven at k_m / eta_m."""
    s = cls(**free_space(d))
    eps_t = _ground_refl.eps_tilde(SOIL_A, s.omega, s.eps)
    s.k = _sommerfeld_below.k_medium(eps_t, s.k)
    s.eta = np.sqrt(s.mu / (s.eps * eps_t))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return complex(np.ravel(s.compute_impedance()[0])[0])


# ----------------------------------------------------------------------
# The complex twin
# ----------------------------------------------------------------------


@needs_twin
@pytest.mark.parametrize("cos_shape", ["cos", "cos-1"])
@pytest.mark.parametrize("deck", ["vertical", "bent", "mixed-radius"])
def test_the_complex_twin_is_the_numpy_fill(deck, cos_shape, monkeypatch):
    """Each of the three tensors against the numpy reference at k_m, eta_m,
    on the whole observer axis and on a band. Per component, so a wrong third
    shape cannot hide behind the other two."""
    d = {
        "vertical": dipole(),
        "bent": buried_bent(),
        "mixed-radius": dict(buried_bent(), wire_radius=[1e-3, 2.5e-3]),
    }[deck]
    s = SinusoidalSolver(**d)
    geom = s._build_geometry()
    eps_t = _ground_refl.eps_tilde(SOIL_A, s.omega, s.eps)
    k_m = _sommerfeld_below.k_medium(eps_t, s.k)
    eta_m = np.sqrt(s.mu / (s.eps * eps_t))
    for rows in (None, (3, 9)):
        twin = s._field_tensor(geom, k_m, obs_rows=rows, cos_shape=cos_shape, eta=eta_m)
        with monkeypatch.context() as mp:
            mp.setattr(sin_mod, "_HAVE_FIELD_TENSOR_CPLX", False)
            ref = s._field_tensor(
                geom, k_m, obs_rows=rows, cos_shape=cos_shape, eta=eta_m
            )
        for a, b in zip(twin, ref, strict=True):
            assert a.shape == b.shape
            assert np.abs(a - b).max() / np.abs(b).max() < 1e-12
    # The two shape sets are different operators — the third tensor moves.
    lit = s._field_tensor(geom, k_m, cos_shape="cos", eta=eta_m)[2]
    fold = s._field_tensor(geom, k_m, cos_shape="cos-1", eta=eta_m)[2]
    assert np.abs(lit - fold).max() > 1e-3 * np.abs(lit).max()


@needs_twin
def test_the_twin_is_what_a_buried_solve_runs(monkeypatch):
    """Counted: the buried fill reaches the complex entry point for the direct
    AND the image block, and never the real one."""
    seen = {"cplx": 0, "real": 0}
    real_c = sin_mod._acc.sinusoidal_field_tensor_cplx
    real_r = sin_mod._acc.sinusoidal_field_tensor

    class Spy:
        def __getattr__(self, name):
            return getattr(sin_mod._acc_real, name)

        def sinusoidal_field_tensor_cplx(self, *a, **kw):
            seen["cplx"] += 1
            return real_c(*a, **kw)

        def sinusoidal_field_tensor(self, *a, **kw):
            seen["real"] += 1
            return real_r(*a, **kw)

    monkeypatch.setattr(sin_mod, "_acc_real", sin_mod._acc, raising=False)
    monkeypatch.setattr(sin_mod, "_acc", Spy())
    z_of(SinusoidalSolver, dipole())
    assert seen["cplx"] >= 2, seen  # direct + image, at least one band each
    assert seen["real"] == 0, seen


def test_a_buried_fill_casts_nothing_complex_to_real():
    """The |k|Δ switch (not `float(k·Δ)`): with every ComplexWarning an error,
    a buried solve and the step-A seam both run."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        SinusoidalSolver(**dipole()).compute_impedance()
        infinite_medium(SinusoidalSolver, dipole())


def test_the_shape_switch_reads_the_modulus(monkeypatch):
    """At a complex k_m the switch must see |k_m|Δ. Shrink the segment until
    |k_m|Δ < `_WELL_SCALED_KD` while Re(k_m)Δ would say otherwise is not
    reachable (|k| >= Re k), so check the other way round: the fill asks for
    the folded shape exactly when |k_m|·min Δ is below the threshold."""
    asked = []
    real = SinusoidalSolver._field_tensor

    def spy(self, geom, k, *a, cos_shape="cos", **kw):
        asked.append((np.iscomplexobj(k), cos_shape))
        return real(self, geom, k, *a, cos_shape=cos_shape, **kw)

    monkeypatch.setattr(SinusoidalSolver, "_field_tensor", spy)
    s = SinusoidalSolver(**dipole(n=11))
    z_of(SinusoidalSolver, dipole(n=11))
    geom = s._build_geometry()
    with s._operating_medium(geom) as medium:
        kd = abs(medium.k_m) * float(np.min(geom["seg_h"]))
    want = "cos-1" if kd < sin_mod._WELL_SCALED_KD else "cos"
    assert asked and all(c for c, _s in asked)
    assert {shape for _c, shape in asked} == {want}


# ----------------------------------------------------------------------
# S1 — eps~ = 1 collapses onto the lane's own free-space solve
# ----------------------------------------------------------------------


@pytest.mark.parametrize("vertical", [True, False], ids=["v", "h"])
def test_eps_one_collapses_a_buried_deck_onto_free_space(vertical):
    z1 = z_of(SinusoidalSolver, dipole(depth=1.5, eps=EPS_ONE, vertical=vertical))
    zf = z_of(SinusoidalSolver, dipole(depth=1.5, free=True, vertical=vertical))
    assert rel(z1, zf) < 1e-8, f"{rel(z1, zf):.3e}"


@pytest.mark.slow
def test_eps_one_collapses_a_detached_deck_onto_free_space():
    """Every entry of Y. The off-diagonal floor is the transmitted GRID's
    accuracy (SG and bspline carry the same ~1e-5 there); the diagonal is
    rounding plus the remainder grids."""
    y1 = y_of(SinusoidalSolver, detached(2, EPS_ONE))
    yf = y_of(SinusoidalSolver, free_space(detached(2)))
    err = np.abs(y1 - yf) / np.abs(yf)
    assert err.max() < 1e-4, err
    assert np.diag(err).max() < 1e-6, err


# ----------------------------------------------------------------------
# S2 — deep burial IS the infinite medium, and the basis gap is free space's
# ----------------------------------------------------------------------


@pytest.mark.parametrize("vertical", [True, False], ids=["v", "h"])
def test_deep_burial_approaches_the_infinite_medium(vertical):
    z_inf = infinite_medium(SinusoidalSolver, dipole(n=41, vertical=vertical))
    deep = rel(
        z_of(SinusoidalSolver, dipole(n=41, depth=1.5, vertical=vertical)), z_inf
    )
    shallow = rel(
        z_of(SinusoidalSolver, dipole(n=41, depth=0.15, vertical=vertical)), z_inf
    )
    assert deep < 1e-3, f"{deep:.3e}"
    assert shallow / deep > 10.0, f"only {shallow / deep:.1f}x"


def test_the_in_medium_basis_gap_tracks_the_free_space_one():
    """The collocation-vs-Galerkin gap at k_m (the infinite medium, bspline's
    oracle) against the same gap in free space, same mesh: the medium must
    add nothing of its own to what point matching costs."""
    d = dipole(n=41)
    wires, npe, feeds = d["wires"], d["n_per_edge_per_wire"], d["feeds"]
    z_bs_inf, z_bs_free = bspline_infinite_medium(wires, npe, feeds)
    gap_med = rel(infinite_medium(SinusoidalSolver, d), z_bs_inf)
    gap_free = rel(z_of(SinusoidalSolver, free_space(d)), z_bs_free)
    assert 0.8 < gap_med / gap_free < 1.25, f"{gap_med:.3e} vs {gap_free:.3e}"
    # And SG's own in-medium gap to the same oracle is orders below it, so
    # the gap measured is the testing's, not the oracle's.
    assert rel(infinite_medium(SinusoidalGalerkinSolver, d), z_bs_inf) < 1e-5


# ----------------------------------------------------------------------
# S3 — convergence onto bspline / SG
# ----------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("vertical", [True, False], ids=["v", "h"])
def test_the_buried_ladder_converges_onto_bspline_and_sg(vertical, record_property):
    """First order in the mesh, as point matching is: the gap to bspline's
    N = 161 roughly halves per doubling, and so does the gap to SG at equal N."""
    ref = z_of(BSplineSolver, dipole(n=161, depth=0.15, vertical=vertical))
    to_bs, to_sg = [], []
    for n in (11, 21, 41, 81):
        d = dipole(n=n, depth=0.15, vertical=vertical)
        zs = z_of(SinusoidalSolver, d)
        to_bs.append(rel(zs, ref))
        to_sg.append(rel(zs, z_of(SinusoidalGalerkinSolver, d)))
    record_property("to_bspline", [float(x) for x in to_bs])
    record_property("to_sg", [float(x) for x in to_sg])
    for a, b in zip(to_bs, to_bs[1:]):
        assert b < 0.7 * a, to_bs
    for a, b in zip(to_sg, to_sg[1:]):
        assert b < 0.8 * a, to_sg


# ----------------------------------------------------------------------
# S4 — reciprocity decays under refinement
# ----------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("deck", ["bent", "detached"])
def test_reciprocity_decays_under_refinement(deck, record_property):
    """Collocation Z is not symmetric, so Y12 != Y21 at finite N — but the
    asymmetry must fall as the mesh refines. Bent / two-wire decks, because a
    straight uniform wire makes this vacuous."""
    mk = buried_bent if deck == "bent" else detached
    asym = []
    for m in (1, 2, 4):
        y = y_of(SinusoidalSolver, mk(m))
        asym.append(abs(y[0, 1] - y[1, 0]) / abs(y[0, 1]))
    record_property("asymmetry", [float(x) for x in asym])
    assert asym[0] > asym[1] > asym[2], asym
    assert asym[2] < asym[0] / 3.0, asym


# ----------------------------------------------------------------------
# S5 — a far-apart detached deck is the two single-class solves
# ----------------------------------------------------------------------


def test_far_apart_detached_blocks_are_the_single_class_solves():
    """The check that caught SG's D2 eta bug: 40 m apart, the diagonal blocks
    of the mixed matrix ARE the matrices of the two single-class solves (each
    class at its own k and eta), and the cross blocks are small."""
    d = detached(1, sep=40.0)
    s = SinusoidalSolver(**d)
    geom = s._build_geometry()
    G, _ = s._assemble_Z(geom, s.k, s.eta)
    nb = 11
    below = dict(
        d,
        wires=[d["wires"][0]],
        n_per_edge_per_wire=[d["n_per_edge_per_wire"][0]],
        feeds=[(0, 2.5, 1 + 0j)],
    )
    above = dict(
        d,
        wires=[d["wires"][1]],
        n_per_edge_per_wire=[d["n_per_edge_per_wire"][1]],
        feeds=[(0, 1.3, 1 + 0j)],
    )
    sb = SinusoidalSolver(**below)
    gb = sb._build_geometry()
    with sb._operating_medium(gb) as medium:
        assert medium is not None
        Gb, _ = sb._assemble_Z(gb, sb.k, sb._medium_eta(medium))
    sa = SinusoidalSolver(**above)
    ga = sa._build_geometry()
    Ga, _ = sa._assemble_Z(ga, sa.k, sa.eta)
    assert np.abs(G[:nb, :nb] - Gb).max() / np.abs(Gb).max() < 1e-12
    assert np.abs(G[nb:, nb:] - Ga).max() / np.abs(Ga).max() < 1e-12
    assert np.abs(G[:nb, nb:]).max() / np.abs(G).max() < 1e-5
    assert np.abs(G[nb:, :nb]).max() / np.abs(G).max() < 1e-5


# ----------------------------------------------------------------------
# S6 — the A_m sign, on the interface term
# ----------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("depth", [0.15, 0.5])
@pytest.mark.parametrize("vertical", [True, False], ids=["v", "h"])
def test_the_interface_term_has_sg_s_sign(depth, vertical, monkeypatch):
    """Z(d) − Z_inf is the whole interface term (image + remainder) on each
    lane's own ladder, so the testing gap cancels out of it. SS carries SG's
    to ~2 % at N = 41; with A_m flipped it lands at ~2x (an inversion), which
    is the thing a gate on Z alone cannot see — at N = 41 the 1 % basis gap
    is larger than the image term on the vertical."""
    d = dipole(n=41, depth=depth, vertical=vertical)
    i_ss = z_of(SinusoidalSolver, d) - infinite_medium(SinusoidalSolver, d)
    i_sg = z_of(SinusoidalGalerkinSolver, d) - infinite_medium(
        SinusoidalGalerkinSolver, d
    )
    assert rel(i_ss, i_sg) < 5e-2, f"{rel(i_ss, i_sg):.3e}"

    real_init = _field_ground.FieldGround.__init__

    def flipped(self, *a, **kw):
        real_init(self, *a, **kw)
        if self._medium is not None:
            self.image_coefficient = -self.image_coefficient

    monkeypatch.setattr(_field_ground.FieldGround, "__init__", flipped)
    i_flip = z_of(SinusoidalSolver, d) - infinite_medium(SinusoidalSolver, d)
    assert rel(i_flip, i_sg) > 1.5, f"{rel(i_flip, i_sg):.3e}"


# ----------------------------------------------------------------------
# S7 — passivity and power
# ----------------------------------------------------------------------


@pytest.mark.parametrize("deck", ["bent", "detached"])
def test_the_admittance_is_passive(deck):
    """No excitation extracts power: the Hermitian part of Y is positive
    definite (every eigenvalue > 0 — the soil is lossy)."""
    mk = buried_bent if deck == "bent" else detached
    y = y_of(SinusoidalSolver, mk(1))
    assert np.linalg.eigvalsh(0.5 * (y + y.conj().T)).min() > 0.0


JACKET = dict(
    wire_conductivity=[3.5e7] * 2,
    insulation_radius=[1.8e-3] * 2,
    insulation_eps_r=[3.5] * 2,
)


def test_a_loaded_detached_deck_dissipates_less_than_it_is_fed():
    s = SinusoidalSolver(**detached(1), **JACKET)
    ps = s.compute_port_solution()
    p_in = 0.5 * float(np.real(ps.y[0, 0]))  # 1 V at port 0
    p_wire, per_wire = s.wire_loss_power(ps.coeffs[:, 0])
    assert 0.0 < p_wire < p_in
    assert per_wire.shape == (2,) and np.all(per_wire > 0.0)


# ----------------------------------------------------------------------
# Loading, per class
# ----------------------------------------------------------------------


def _loading_shift(cls, d, load, **kw):
    return y_of(cls, dict(d, **load), **kw) - y_of(cls, d, **kw)


@pytest.mark.slow
@pytest.mark.parametrize("deck", ["bent", "detached"])
def test_the_loading_shift_converges_onto_sg(deck, record_property):
    """Loaded − unloaded Y against SG's, SG on the SAME segment gap (its
    default point gap is a different source, and that difference does not
    converge with the loading). Read at the real ω, each class's σ·AC."""
    mk = buried_bent if deck == "bent" else detached
    gaps = []
    for m in (1, 2, 4):
        ys = _loading_shift(SinusoidalSolver, mk(m), JACKET)
        yg = _loading_shift(
            SinusoidalGalerkinSolver, mk(m), JACKET, feed_model="segment"
        )
        gaps.append(np.abs(ys - yg).max() / np.abs(yg).max())
    record_property("gaps", [float(x) for x in gaps])
    assert gaps[0] > gaps[1] > gaps[2], gaps
    assert gaps[2] < 1e-2, gaps


@pytest.mark.slow
def test_the_buried_jacket_charge_term_converges_onto_sg(monkeypatch):
    """#1154's charge term alone (jacket on minus jacket with the term
    dropped), SS's point form −zq·k²·σC against SG's by-parts overlap."""
    jacket = dict(insulation_radius=[1.8e-3] * 2, insulation_eps_r=[3.5] * 2)

    def term(cls, d, **kw):
        on = y_of(cls, dict(d, **jacket), **kw)
        with monkeypatch.context() as mp:
            mp.setattr(_wire_loading, "buried_jacket_charge", lambda s, w: None)
            off = y_of(cls, dict(d, **jacket), **kw)
        return on - off

    gaps = []
    for m in (1, 2, 4):
        ts = term(SinusoidalSolver, buried_bent(m))
        tg = term(SinusoidalGalerkinSolver, buried_bent(m), feed_model="segment")
        # The term is a real share of Y, not a rounding: ~half of it here.
        assert (
            np.abs(tg).max()
            > 0.1 * np.abs(y_of(SinusoidalSolver, buried_bent(m))).max()
        )
        gaps.append(np.abs(ts - tg).max() / np.abs(tg).max())
    assert gaps[0] > gaps[1] > gaps[2] and gaps[2] < 3e-3, gaps


def test_loading_is_read_at_the_real_omega(monkeypatch):
    """`loading_for` refuses a complex omega by name, so a buried solve that
    handed it k_m·c would raise; count the reads and check they are real."""
    seen = []
    real = _wire_loading.loading_for

    def spy(solver, omega, geom=None):
        seen.append(omega)
        return real(solver, omega, geom)

    monkeypatch.setattr(_wire_loading, "loading_for", spy)
    z_of(SinusoidalSolver, dict(dipole(), wire_conductivity=3.5e7))
    assert seen and all(not np.iscomplexobj(w) for w in seen)


# ----------------------------------------------------------------------
# The readout, the sweep, and the refusals
# ----------------------------------------------------------------------


def test_buried_knot_currents_read_through_the_solve_s_view():
    """The currents come back in the basis the solve used (k_m entries), so
    they sit at SG's to the basis gap; read through AIR's view they would be a
    different function (the #1159 failure)."""
    d = dipole(n=41, depth=0.5, vertical=False)
    out = {}
    for cls in (SinusoidalSolver, SinusoidalGalerkinSolver):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = cls(**d)
            _z, alpha = s.compute_impedance()
            out[cls] = np.asarray(s.currents_at_knots(alpha)[0])
            if cls is SinusoidalSolver:
                view = s._readout_view(s._build_geometry())
                assert "k_entry" in view
                air = s._basis_coefs(s._build_geometry(), s.k)
                s._readout_view = lambda geom, _v=air: _v
                out["air"] = np.asarray(s.currents_at_knots(alpha)[0])
    ss, sg = out[SinusoidalSolver], out[SinusoidalGalerkinSolver]
    scale = np.abs(sg).max()
    assert np.abs(ss - sg).max() / scale < 3e-2
    assert np.abs(out["air"] - sg).max() / scale > 0.1


def test_a_buried_sweep_is_the_single_frequency_solves():
    d = dipole(n=11)
    s = SinusoidalSolver(**d)
    k0 = float(s.k)
    ks = [k0, 1.05 * k0]
    swept = s.compute_impedance_swept(ks)
    single = []
    for kk in ks:
        t = SinusoidalSolver(**dict(d, wavelength=2.0 * np.pi / kk))
        single.append(complex(np.ravel(t.compute_impedance()[0])[0]))
    for a, b in zip(swept, single):
        assert rel(a, b) < 1e-12


def test_a_crossing_junction_is_refused_by_name():
    d = dict(
        wires=[
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)]),
            np.array([(0.0, 0.0, 0.0), (2.0, 0.0, -0.5)]),
        ],
        n_per_edge_per_wire=[[9], [9]],
        feeds=[(0, 1.0, 1 + 0j)],
        junctions=[[(0, "start"), (1, "start")]],
        wavelength=WL7,
        wire_radius=1e-3,
        **ground(),
    )
    with pytest.raises(NotImplementedError) as err:
        SinusoidalSolver(**d).compute_impedance()
    assert sin_mod._CROSSING_JUNCTION_REFUSAL in str(err.value)
    assert (
        SinusoidalSolver.capabilities.refusal("buried", "crossing_junction")
        == sin_mod._CROSSING_JUNCTION_REFUSAL
    )
    # The same deck is served by the family that has the crossing fill.
    assert np.isfinite(z_of(SinusoidalGalerkinSolver, d))


@pytest.mark.parametrize(
    ("kw", "reason"),
    [
        (dict(ground_z=0.0), BURIED_PEC_REFUSAL),
        (dict(ground_z=0.0, ground_eps=SOIL_A), BURIED_REFL_REFUSAL),
    ],
    ids=["pec", "refl-coef"],
)
def test_a_ground_with_no_lower_medium_refuses_with_its_own_reason(kw, reason):
    d = dict(free_space(dipole()), **kw)
    with pytest.raises(ValueError) as err:
        SinusoidalSolver(**d).compute_impedance()
    assert reason in str(err.value)


def test_the_extended_kernel_is_refused_below_the_interface():
    with pytest.raises(NotImplementedError) as err:
        SinusoidalSolver(**dipole(), extended_kernel=True).compute_impedance()
    assert BURIED_EXTENDED_KERNEL_REFUSAL in str(err.value)


def test_a_fill_at_k_m_without_eta_is_refused():
    """#995 on this lane: the half-set operating point raises."""
    s = SinusoidalSolver(**dipole())
    geom = s._build_geometry()
    with s._operating_medium(geom):
        with pytest.raises(ValueError, match="momwire#995"):
            s._assemble_Z(geom, s.k)
