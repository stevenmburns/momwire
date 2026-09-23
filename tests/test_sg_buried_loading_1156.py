"""Wire loading on a BURIED deck, SinusoidalGalerkinSolver — momwire#1156.

**The defect.** On a fully-buried deck SG runs the whole solve at the soil's
complex wavenumber k_m (`_operating_medium`), and `_apply_loading` read the
loading spec at `k * self.c` — a complex "angular frequency" — so every
`wire_conductivity`, `distributed_rlc` and jacket on a buried deck died in
`series_impedance_per_wire` with an unnamed TypeError. The mixed route
(detached and crossing decks, momwire#980 D2/D3) refused loading by a
sentence that no capability cell declared, because the only ω on hand there
was the fill's one k.

**The fix.** A bare conductor's surface impedance and a distributed RLC are
local, medium-independent functions of the REAL ω (razor's #1149 U3
derivation); only the overlap's SHAPES carry the medium, through k. So the
spec is read at `medium.k_p * c` — the solve's own real ω, and exactly the
`k * c` every above-ground deck always read — and the overlap is written at
each segment's own k (`k_entry` on a mixed deck's stitched view, which the
crossing wings carry too). A jacketed buried wire also takes #1154's
charge-side term, tested the way bspline's `_charge_gram` tests it:

    G[i, j] −= zq_s · ∫_s f_i′ f_j′ dξ,   zq = ΔS′/(jω),
    ∫_s f_e′ f_f′ = k²·[Q_eQ_f·(h/2 + sin kh/2k) + R_eR_f·(h/2 − sin kh/2k)]

on the three-term shape f = P + Q·sin kξ + R·cos kξ. `wire_loss_power`
likewise rebuilds the view the solve used and integrates |I|² at complex k —
the inherited readout rebuilt the shapes at ω/c and read a buried dipole's
metal loss 340x low.

**The gates** (real constructors, reference-free where possible):

* loaded − unloaded is SG's own loading term to rounding, the term built
  here by QUADRATURE of the basis shapes at the real ω (independent of the
  closed forms), on a wholly-buried dipole, a mixed deck and a crossing deck;
* #1154's exact in-medium pair (lossless soil ε̃ = 4, jacket εr = 10): the
  jacket within 0.1 Ω; the charge term dropped misses by 25+ Ω;
* ε̃ = 1 adds nothing (the assembled matrix is unchanged by the term);
* swept == single-frequency (1e-9 Ω), reciprocity unchanged;
* the loading SHIFT converges onto bspline's under refinement (slow lane).

The crossing decks cannot carry the exact-pair gate on this trunk: the exact
pair puts the two media's wires at different radii, and SG's crossing serve
refuses per-wire radii (`crossing_junctions`, momwire#524 phase 2).

The mixed decks here are fed on ONE wire. SG's mixed-deck transmitted
coupling has the opposite sign to bspline's and razor's (probe7: at ε̃ = 1
SG's mixed Y12 is minus its own free-space Y12). A single-port Z is exactly
invariant under that inversion — it is D·G·D with D = ±1 per medium class,
and loading is block-diagonal per wire — so these gates cannot see it and do
not depend on it. See scratch/1156-sg-buried-loading/NOTES.md.
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire import _wire_loading  # noqa: E402
from momwire._wire_loading import DistributedRLC  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver  # noqa: E402
from test_jacket_in_soil_1154 import LOSSLESS, _pair_decks, dipole  # noqa: E402
from test_razor_crossing_loading_1149 import crossing  # noqa: E402

WL7 = 299792458.0 / 7.0e6
SOIL_A = (13.0, 0.005)
SIGMA = 3.5e7
JACKET_B, JACKET_ER = 1.8e-3, 3.5


def mixed(m=1, eps=SOIL_A, fed=0, **kw):
    """A buried centre-fed wire (5 m, depth 0.5 m) beside a wire 1 m above
    the plane, uniform n = 10m each: a MIXED deck refined everywhere. Fed on
    wire `fed` only (see the module docstring)."""
    d = dict(
        wires=[
            np.array([(-2.5, 0.0, -0.5), (2.5, 0.0, -0.5)]),
            np.array([(-2.5, 0.0, 1.0), (2.5, 0.0, 1.0)]),
        ],
        n_per_edge_per_wire=[[10 * m], [10 * m]],
        feeds=[(fed, 2.5, 1 + 0j)],
        wavelength=WL7,
        wire_radius=1e-3,
        ground_z=0.0,
        ground_eps=eps,
        ground_model="sommerfeld",
    )
    d.update(kw)
    return d


def xing(m=1, **kw):
    """The crossing deck of #1149/#1154 at a 1 mm radius, so a PVC-class
    jacket (b = 1.8 mm) fits on it."""
    return dict(crossing(m), wire_radius=1e-3, **kw)


def per_wire(d, **kw):
    """Loading kwargs spelled per wire for deck `d`."""
    nw = len(d["wires"])
    return dict(d, **{k: [v] * nw for k, v in kw.items()})


JACKET = dict(insulation_radius=JACKET_B, insulation_eps_r=JACKET_ER)
SIG_JACKET = dict(wire_conductivity=SIGMA, **JACKET)

DECKS = {
    "dipole": lambda m=1: dipole(n=20 * m),
    "mixed": mixed,
    "crossing": xing,
}


def sg(d):
    return SinusoidalGalerkinSolver(**d)


def z(cls, d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return complex(np.ravel(cls(**d).compute_impedance()[0])[0])


@pytest.fixture
def charge_term_dropped(monkeypatch):
    """#1154's red control: the buried jacket's charge term dropped."""

    def use():
        monkeypatch.setattr(_wire_loading, "buried_jacket_charge", lambda s, w: None)

    return use


def assembled(s, loading):
    """(G, seg_view, geom, k) from the production fill, loading on or off on
    the SAME instance, so the kernel radius (a jacket's a′) is shared."""
    geom = s._build_geometry()
    saved = s._loading_active
    s._loading_active = loading
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with s._operating_medium(geom):
                G, seg_view = s._assemble_Z(geom, s.k)
                k = s.k
    finally:
        s._loading_active = saved
    return np.asarray(G), seg_view, geom, k


def quadrature_loading(s, geom, seg_view, k, omega=None, n_gl=12):
    """(L, M): L = −Σ_s [z_s ∫ f_i f_j + zq_s ∫ f_i′ f_j′] by Gauss–Legendre
    on each segment, from the shapes alone and the spec at the REAL ω — none
    of `_apply_loading`'s closed forms — and M, the same sums over |shape
    coefficients|, i.e. the size of the terms BEFORE they cancel.

    M is what "to rounding" is measured against. On a graded deck the
    three-term shape is a near-cancellation (P ≈ −R, both ~1/(kh)² larger
    than f), and the shipped series overlap's closed form loses up to 5e-7
    relative on the crossing deck's node segments against a 40-digit
    evaluation of itself (probe13; the new charge Gram loses 1.5e-12)."""
    spec = _wire_loading.loading_for(s, s.omega if omega is None else omega, geom)
    zq = spec.zq_seg
    starts = np.asarray(seg_view["starts"])
    h = np.asarray(geom["seg_h"], dtype=float)
    k_entry = np.asarray(seg_view.get("k_entry", np.full(starts[-1], k)))
    n = int(geom["n_segs"]) + s._n_extra_cols()
    L = np.zeros((n, n), dtype=np.complex128)
    M = np.zeros((n, n))
    x, w = np.polynomial.legendre.leggauss(n_gl)
    for m in range(int(geom["n_segs"])):
        xi = 0.5 * h[m] * x
        wm = 0.5 * h[m] * w
        f, df, size, dsize, cols = [], [], [], [], []
        for e in range(starts[m], starts[m + 1]):
            kk = k_entry[e]
            P = seg_view["sigma"][e] * seg_view["A"][e]
            Q = seg_view["B"][e]
            R = seg_view["sigma"][e] * seg_view["C"][e]
            f.append(P + Q * np.sin(kk * xi) + R * np.cos(kk * xi))
            df.append(kk * (Q * np.cos(kk * xi) - R * np.sin(kk * xi)))
            size.append(abs(P) + abs(Q) + abs(R))
            dsize.append(abs(kk) * (abs(Q) + abs(R)))
            cols.append(int(seg_view["jbasis"][e]))
        f, df = np.array(f), np.array(df)
        size, dsize = np.array(size), np.array(dsize)
        block = spec.z_seg[m] * (f * wm) @ f.T
        mag = abs(spec.z_seg[m]) * h[m] * np.outer(size, size)
        if zq is not None:
            block = block + zq[m] * (df * wm) @ df.T
            mag = mag + abs(zq[m]) * h[m] * np.outer(dsize, dsize)
        L[np.ix_(cols, cols)] -= block
        M[np.ix_(cols, cols)] += mag
    return L, M


def off_by_more_than_rounding(G_on, G_off, L, M):
    """Entries where on − off misses L by more than rounding: 1e-12 of the
    terms' pre-cancellation size, plus the subtraction's own 16 ulp of G."""
    tol = 1e-12 * M + 16 * np.finfo(float).eps * np.abs(G_off)
    return np.abs((G_on - G_off) - L) > tol


# ----------------------------------------------------------------------
# the repro, served
# ----------------------------------------------------------------------

LOADS = {
    "distributed R": dict(distributed_rlc=DistributedRLC("series", r=20.0)),
    "distributed L": dict(distributed_rlc=DistributedRLC("series", l=1e-6)),
    "conductivity": dict(wire_conductivity=SIGMA),
    "jacket": dict(JACKET),
}


@pytest.mark.parametrize("load", sorted(LOADS))
def test_the_issue_repro_serves_every_loading_kind(load):
    """#1154 probe2's buried dipole: an unnamed TypeError before, a finite
    impedance that moves with the load now."""
    d = dipole(n=20)
    loaded = z(SinusoidalGalerkinSolver, dict(d, **LOADS[load]))
    assert np.isfinite(loaded)
    assert abs(loaded - z(SinusoidalGalerkinSolver, d)) > 0.1


def test_a_mixed_deck_serves_loading_it_used_to_refuse():
    """The D2 refusal ('applied at a single wavenumber') is retired: a
    mixed deck's overlap is written at each segment's own k."""
    d = mixed()
    loaded = z(SinusoidalGalerkinSolver, per_wire(d, **SIG_JACKET))
    assert np.isfinite(loaded)
    assert abs(loaded - z(SinusoidalGalerkinSolver, d)) > 1.0


# ----------------------------------------------------------------------
# loaded − unloaded is SG's own loading term
# ----------------------------------------------------------------------


@pytest.mark.parametrize("load", ["conductivity", "conductivity+jacket"])
@pytest.mark.parametrize("name", sorted(DECKS))
def test_loaded_minus_unloaded_is_the_loading_term(name, load):
    kw = dict(wire_conductivity=SIGMA) if load == "conductivity" else SIG_JACKET
    s = sg(per_wire(DECKS[name](), **kw))
    G_on, seg_view, geom, k = assembled(s, True)
    G_off, _sv, _g, _k = assembled(s, False)
    # wholly-buried dipole: one view at the one complex k_m; the mixed and
    # crossing decks: a stitched view whose entries (the crossing's wings
    # too) carry their own k.
    assert ("k_entry" in seg_view) == (name != "dipole")
    if load == "conductivity+jacket":
        assert _wire_loading.loading_for(s, s.omega, geom).zq_seg is not None
    L, M = quadrature_loading(s, geom, seg_view, k)
    assert np.abs(L).max() > 0.0
    assert not off_by_more_than_rounding(G_on, G_off, L, M).any()


@pytest.mark.parametrize("red", ["omega x 1.001", "charge term dropped"])
def test_the_loading_term_gate_has_teeth(red, charge_term_dropped):
    """The same comparison, against a reference that is wrong on purpose."""
    s = sg(per_wire(mixed(), **SIG_JACKET))
    G_on, seg_view, geom, k = assembled(s, True)
    G_off, _sv, _g, _k = assembled(s, False)
    if red == "omega x 1.001":
        L, M = quadrature_loading(s, geom, seg_view, k, omega=1.001 * s.omega)
    else:
        charge_term_dropped()
        L, M = quadrature_loading(s, geom, seg_view, k)
    assert off_by_more_than_rounding(G_on, G_off, L, M).any()


def test_the_loading_is_read_at_the_real_omega(monkeypatch):
    """Every loading read on a buried solve sees the solve's real ω."""
    seen = []
    real = _wire_loading.loading_for

    def spy(solver, omega, geom=None):
        seen.append(omega)
        return real(solver, omega, geom)

    monkeypatch.setattr(_wire_loading, "loading_for", spy)
    for name in sorted(DECKS):
        s = sg(per_wire(DECKS[name](), **SIG_JACKET))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s.compute_impedance()
        assert seen, name
        assert all(np.isrealobj(w) and float(w) == s.k * s.c for w in seen), seen
        seen.clear()


def test_the_loading_term_keeps_reciprocity():
    """The term is symmetric, so G's asymmetry is what the fill left."""
    s = sg(per_wire(mixed(), **SIG_JACKET))
    G_on, _sv, _g, _k = assembled(s, True)
    G_off, _sv, _g, _k = assembled(s, False)
    L = G_on - G_off
    assert np.abs(L - L.T).max() <= 1e-14 * np.abs(L).max()
    a_on = np.linalg.norm(G_on - G_on.T) / np.linalg.norm(G_on)
    a_off = np.linalg.norm(G_off - G_off.T) / np.linalg.norm(G_off)
    assert a_on <= 1.01 * a_off + 1e-15, (a_on, a_off)


# ----------------------------------------------------------------------
# the jacket: #1154's exact in-medium pair, and its red control
# ----------------------------------------------------------------------

PAIR_DECKS = {
    # (deck, bar on |jacket - exact|, floor on |charge dropped - exact|);
    # measured 0.0061 / 0.031 ohm, dropped 28.1 / 77.5 ohm (probe4).
    "dipole": (lambda: dipole(eps=LOSSLESS), 0.1, 20.0),
    "mixed": (lambda: mixed(eps=LOSSLESS), 0.1, 25.0),
}


@pytest.mark.parametrize("name", sorted(PAIR_DECKS))
def test_the_jacket_in_soil_is_the_in_medium_pair(name):
    mk, bar, _floor = PAIR_DECKS[name]
    exact, jacket = _pair_decks(mk())
    gap = abs(z(SinusoidalGalerkinSolver, jacket) - z(SinusoidalGalerkinSolver, exact))
    assert gap < bar, gap


@pytest.mark.parametrize("name", sorted(PAIR_DECKS))
def test_the_charge_term_dropped_misses_the_in_medium_pair(name, charge_term_dropped):
    mk, _bar, floor = PAIR_DECKS[name]
    exact, jacket = _pair_decks(mk())
    z_exact = z(SinusoidalGalerkinSolver, exact)
    charge_term_dropped()
    gap = abs(z(SinusoidalGalerkinSolver, jacket) - z_exact)
    assert gap > floor, gap


def test_the_collapse_eps_1_adds_nothing(charge_term_dropped):
    """At ε̃ = 1 the buried jacket's charge coefficient is zero, and the
    assembled matrix is the one with the term structurally absent."""
    s = sg(dipole(eps=1.0 + 0j, **JACKET))
    geom = s._build_geometry()
    zq = _wire_loading.loading_for(s, s.omega, geom).zq_seg
    assert zq is not None and np.all(zq == 0.0)
    G_with, *_ = assembled(s, True)
    charge_term_dropped()
    G_without, *_ = assembled(s, True)
    assert np.array_equal(G_with, G_without)


def test_only_the_buried_wire_gets_the_charge_term():
    """Per wire by medium: the mixed deck's wire 1 is above the plane."""
    s = sg(per_wire(mixed(), **JACKET))
    zq = _wire_loading.loading_for(s, s.omega).zq_wire
    assert zq[0] != 0.0 and zq[1] == 0.0


# ----------------------------------------------------------------------
# sweeps
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(DECKS))
def test_a_swept_solve_is_the_single_frequency_solves(name, charge_term_dropped):
    """eps~(w) and the loading per frequency: swept == single, and the sweep
    carries the charge term.

    The sweep is handed the k each single solve actually ran at. Handing it
    2 pi / lambda instead differs from omega / c by an ulp, and the loaded
    crossing deck moves 5.2e-7 ohm for ONE ulp of wavelength (probe11) — the
    shipped series overlap's closed-form cancellation on the graded node
    segments (probe13), not the loading read; the bare deck moves 5.9e-10."""
    d = per_wire(DECKS[name](), **SIG_JACKET)
    lams = (d["wavelength"], d["wavelength"] / 1.05)
    ks = [sg(dict(d, wavelength=lam)).k for lam in lams]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        swept = np.asarray(sg(d).compute_impedance_swept(ks)).ravel()
    single = np.array(
        [z(SinusoidalGalerkinSolver, dict(d, wavelength=lam)) for lam in lams]
    )
    assert np.max(np.abs(swept - single)) < 1e-9, swept - single
    charge_term_dropped()
    assert abs(swept[0] - z(SinusoidalGalerkinSolver, d)) > 1.0


# ----------------------------------------------------------------------
# the metal-loss readout
# ----------------------------------------------------------------------


def _loss(cls, d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = cls(**d)
        _z, coeffs = s.compute_impedance()
        return s, coeffs, s.wire_loss_power(coeffs)


def test_the_buried_loss_readout_integrates_the_solved_shapes():
    """∫|I|² at complex k, against quadrature of the same view."""
    s, coeffs, (total, per_wire_w) = _loss(
        SinusoidalGalerkinSolver, per_wire(xing(), wire_conductivity=SIGMA)
    )
    _G, seg_view, geom, _k = assembled(s, True)
    starts = np.asarray(seg_view["starts"])
    h = np.asarray(geom["seg_h"], dtype=float)
    x, w = np.polynomial.legendre.leggauss(16)
    r_w = np.real(_wire_loading.loading_for(s, s.omega).z_wire)
    wire_of = s._wire_of_seg(geom)
    want = np.zeros_like(per_wire_w)
    for m in range(int(geom["n_segs"])):
        xi = 0.5 * h[m] * x
        cur = np.zeros_like(xi, dtype=np.complex128)
        for e in range(starts[m], starts[m + 1]):
            kk = seg_view["k_entry"][e]
            cur += coeffs[seg_view["jbasis"][e]] * (
                seg_view["sigma"][e] * seg_view["A"][e]
                + seg_view["B"][e] * np.sin(kk * xi)
                + seg_view["sigma"][e] * seg_view["C"][e] * np.cos(kk * xi)
            )
        want[wire_of[m]] += (
            0.5 * r_w[wire_of[m]] * np.sum(0.5 * h[m] * w * abs(cur) ** 2)
        )
    assert np.allclose(per_wire_w, want, rtol=1e-12, atol=0.0)
    assert total == pytest.approx(want.sum(), rel=1e-12)


def test_the_buried_loss_readout_agrees_with_bspline():
    """The inherited readout (shapes at ω/c) read 2.8e-8 W here against
    bspline's 9.40e-6 W; this one agrees to 6e-6 (probe9)."""
    d = dipole(n=40, wire_conductivity=SIGMA)
    got = _loss(SinusoidalGalerkinSolver, d)[2][0]
    want = _loss(BSplineSolver, d)[2][0]
    assert got == pytest.approx(want, rel=1e-4)


# ----------------------------------------------------------------------
# the trunks onto each other (slow)
# ----------------------------------------------------------------------


def _shift(cls, d0, kw):
    return z(cls, per_wire(d0, **kw) if "distributed_rlc" not in kw else dict(d0, **kw))


TRUNK_CASES = {
    # measured SG - bspline shift gaps (probe3/probe8), m = 1, 2, 4:
    # dipole L' 2.57 -> 1.50 -> 0.87, jacket 0.78 -> 0.43 -> 0.25;
    # mixed L' 9.34 -> 4.66 -> 2.54, jacket 2.20 -> 1.20 -> 0.69;
    # crossing L' 0.018 -> 0.009 -> 0.003, jacket 0.0067 -> 0.0051 -> 0.0024
    # ohm on shifts of 463 / 19 ohm. The crossing jacket's first step is
    # 0.76, from a gap already 3.5e-4 of the shift, so its bar is 0.8.
    "dipole": (DECKS["dipole"], 0.75),
    "mixed": (mixed, 0.75),
    "crossing": (xing, 0.8),
}
TRUNK_LOADS = {
    "distributed L": dict(distributed_rlc=DistributedRLC("series", l=1e-6)),
    "jacket": dict(JACKET),
}


@pytest.mark.slow
@pytest.mark.parametrize("load", sorted(TRUNK_LOADS))
@pytest.mark.parametrize("name", sorted(TRUNK_CASES))
def test_the_loading_shift_converges_onto_bspline(name, load):
    (mk, ratio), kw = TRUNK_CASES[name], TRUNK_LOADS[load]
    gaps = []
    for m in (1, 2, 4):
        d0 = mk(m)
        shifts = [
            _shift(cls, d0, kw) - z(cls, d0)
            for cls in (SinusoidalGalerkinSolver, BSplineSolver)
        ]
        gaps.append(abs(shifts[0] - shifts[1]))
    assert all(y / x <= ratio for x, y in zip(gaps, gaps[1:])), gaps


def test_a_complex_omega_is_refused_by_name():
    """The shared read's contract, stated where the TypeError used to be
    anonymous: a complex 'omega' is k_m * c, not a frequency."""
    s = sg(dipole(n=20, wire_conductivity=SIGMA))
    with pytest.raises(TypeError, match="REAL angular frequency"):
        _wire_loading.loading_for(s, complex(s.omega, 1.0))
