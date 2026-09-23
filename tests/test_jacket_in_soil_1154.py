"""A dielectric jacket on a wire in SOIL — momwire#1154.

**The defect.** momwire serves a jacketed wire as a PAIR (momwire#865): the
kernel radius is the equivalent radius a' = a (b/a)^(1 - 1/eps_r) and the
series term is L' = mu0/2pi (1 - 1/eps_r) ln(b/a). Both are written against
a free-space exterior, and every buried route of both trunks served them to
wires in the soil.

**The derivation** (scratch/1154-jacket-in-soil/NOTES.md,
`_wire_loading.jacket_elastance`). The flux linkage of a coated wire is set
by the METAL, so L' is exact in any exterior: it only undoes a' back to a.
What a' gets wrong in an exterior eps~ is the CHARGE: the two shells'
elastance is [ln(b/a)/eps_r + ln(R/b)/eps~]/(2 pi eps0), and the kernel at a'
carries ln(R/a')/(2 pi eps0 eps~), short by

    dS' = ln(b/a) / (2 pi eps0 eps_r) * (1 - 1/eps~)

independent of R. It multiplies q = -(1/jw) dI/dl, so it is a charge-side
term (bspline: a derivative Gram, razor: a potential stencil), zero at
eps~ = 1 and zero on every above-ground wire. The issue's first reading,
(1 - eps~/eps_r) in L', is the right L' only beside the in-medium radius
a (b/a)^(1 - eps~/eps_r) — complex in a lossy soil, and no kernel takes one.

**The gates** (real constructors):

* the collapse: eps~ = 1 adds exactly nothing, on both trunks' terms;
* REFERENCE-FREE, the in-medium pair: in a LOSSLESS soil (eps~ = 4) under a
  jacket with eps_r = 10, the exact quasi-static pair is real — each wire
  served BARE at its own medium's equivalent radius with its own series L
  (`DistributedRLC`). The jacket kwargs agree with it to hundredths of an
  ohm on a buried dipole, a crossing deck and a buried-hub screen, on both
  trunks; the free-space pair (the charge term dropped — main's behaviour)
  misses it by 27 / 12 / 5.5 ohm. That is the red control: the old term
  and bspline<->razor agreement are NOT a red control for each other,
  because both trunks share the same wrong pair and converge onto each
  other anyway (probe3/probe4 measured it);
* bspline <-> razor on soil A with a PVC-class jacket converge onto each
  other under refinement (slow lane);
* swept == single-frequency (1e-9 ohm), with the term provably present;
* every above-ground or surface jacket gets no term at all (structurally),
  which is what probe5 measured bit-identical to main on four formulations.

Measured 2026-09-23 (scratch/1154-jacket-in-soil/probe3.out, probe4*.out).
"""

from __future__ import annotations

import math
import pathlib
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire import _medium_spec, _wire_loading  # noqa: E402
from momwire._wire_loading import MU0, DistributedRLC  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from test_razor_crossing_loading_1149 import crossing, hub  # noqa: E402

WL7 = 299792458.0 / 7.0e6
SOIL_A = (13.0, 0.005)
EPS0 = 8.854187817e-12
# The lossless in-medium-pair setting: eps~ < eps_r keeps L(eps~) >= 0, so
# the reference is servable through the public DistributedRLC kwarg.
ET, ER_REF = 4.0, 10.0
LOSSLESS = complex(ET, -1e-9)


def dipole(n=20, length=5.0, depth=0.5, eps=SOIL_A, **kw):
    """A horizontal centre-fed dipole `depth` below the interface."""
    kw.setdefault("wire_radius", 1e-3)
    pts = np.array([(-0.5 * length, 0.0, -depth), (0.5 * length, 0.0, -depth)])
    return dict(
        wires=[pts],
        n_per_edge_per_wire=[[n]],
        feeds=[(0, 0.5 * length, 1 + 0j)],
        wavelength=WL7,
        ground_z=0.0,
        ground_eps=eps,
        ground_model="sommerfeld",
        **kw,
    )


def solver(cls, d):
    d = dict(d)
    if cls is RazorSolver:
        d.pop("junctions", None)
        return RazorSolver(**d, nec5_quadrature=True)
    return cls(**d)


def z(cls, d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return complex(np.ravel(solver(cls, d).compute_impedance()[0])[0])


@pytest.fixture
def free_space_pair(monkeypatch):
    """main's behaviour: the buried jacket's charge term dropped."""

    def use():
        monkeypatch.setattr(_wire_loading, "buried_jacket_charge", lambda s, w: None)

    return use


# ----------------------------------------------------------------------
# the term itself
# ----------------------------------------------------------------------


def test_the_elastance_is_the_derived_closed_form():
    a, b, er = 1e-3, 1.8e-3, 3.5
    et = complex(13.0, -12.8)
    want = math.log(b / a) / (2 * math.pi * EPS0 * er) * (1 - 1 / et)
    got = complex(_wire_loading.jacket_elastance(a, b, er, et, EPS0))
    assert got == pytest.approx(want, rel=1e-15)
    # In ordinary soil the missing elastance is POSITIVE and lossy: the
    # buried jacketed wire is less capacitive than the free-space pair says.
    assert got.real > 0 and got.imag < 0


def test_the_collapse_eps_1_adds_exactly_nothing():
    assert _wire_loading.jacket_elastance(1e-3, 2e-3, 3.0, 1.0, EPS0) == 0.0
    d = dipole(eps=1.0 + 0j, insulation_radius=1.8e-3, insulation_eps_r=3.5)
    bs = BSplineSolver(**d)
    zq = _wire_loading.loading_for(bs, bs.omega).zq_wire
    assert zq is not None and np.all(zq == 0.0)
    n = bs._build_basis_polynomials(bs._build_geometry())[0].shape[0]
    with_term = bs._apply_loading(np.zeros((n, n), dtype=np.complex128))
    rows, cols, vals, wid = bs._loading_gram()
    series_only = np.zeros((n, n), dtype=np.complex128)
    np.add.at(
        series_only,
        (rows, cols),
        _wire_loading.loading_for(bs, bs.omega).z_wire[wid] * vals,
    )
    assert np.array_equal(with_term, series_only)

    rz = solver(RazorSolver, d)
    geom = rz._build_geometry()
    spec = _wire_loading.loading_for(rz, rz.omega, geom)
    stencil = rz._charge_stencil(geom)
    zero = np.zeros((geom["n_basis_total"],) * 2, dtype=np.complex128)
    assert np.array_equal(RazorSolver._apply_charge(zero.copy(), stencil, spec), zero)


def test_only_jacketed_wires_in_the_lower_medium_get_the_term():
    """Per wire by medium: the crossing deck's wire 0 is buried, wire 1 is
    the mast. A jacket on both loads only wire 0's charge side."""
    d = crossing(1, insulation_radius=[2e-3, 2e-3], insulation_eps_r=[3.0, 3.0])
    for cls in (BSplineSolver, RazorSolver):
        s = solver(cls, d)
        assert s._wire_media() == (_medium_spec.BELOW, _medium_spec.ABOVE)
        zq = _wire_loading.loading_for(s, s.omega).zq_wire
        assert zq[1] == 0.0 and zq[0] != 0.0


@pytest.mark.parametrize(
    "name",
    ["free", "pec", "refl-coef", "sommerfeld-above", "surface-at-b", "crossing-mast"],
)
@pytest.mark.parametrize("cls", [BSplineSolver, RazorSolver])
def test_an_above_ground_jacket_gets_no_charge_term_at_all(name, cls):
    """Structurally absent, not zero — the reason probe5 measured every
    above-ground jacketed deck bit-identical to main (four formulations,
    single and swept, with two buried-jacket CONTROLS that differ)."""
    jk = dict(insulation_radius=1.8e-3, insulation_eps_r=3.5)
    above = np.array([(-2.5, 0.0, 3.0), (2.5, 0.0, 3.0)])
    base = dict(
        n_per_edge_per_wire=[[20]],
        feeds=[(0, 2.5, 1 + 0j)],
        wavelength=WL7,
        wire_radius=1e-3,
        **jk,
    )
    som = dict(ground_z=0.0, ground_eps=SOIL_A, ground_model="sommerfeld")
    d = {
        "free": dict(base, wires=[above]),
        "pec": dict(base, wires=[above], ground_z=0.0),
        "refl-coef": dict(base, wires=[above], ground_z=0.0, ground_eps=SOIL_A),
        "sommerfeld-above": dict(base, wires=[above], **som),
        # A jacketed radial RESTING on the soil, h = b: above by the labels.
        "surface-at-b": dict(base, wires=[above - [0, 0, 3.0 - 1.8e-3]], **som),
        "crossing-mast": crossing(
            1, insulation_radius=[np.nan, 2e-3], insulation_eps_r=[np.nan, 3.0]
        ),
    }[name]
    s = solver(cls, d)
    geom = s._build_geometry()
    assert _wire_loading.loading_for(s, s.omega).zq_wire is None
    if cls is RazorSolver:
        assert s._assemble_Z_prepare(geom)["charge"] is None


# ----------------------------------------------------------------------
# the reference-free gate: the in-medium pair, and its red control
# ----------------------------------------------------------------------

A_REF, B_REF = 0.25e-3, 0.75e-3


def _pair_decks(d0):
    """(exact in-medium pair, jacket kwargs) for deck `d0` in LOSSLESS."""
    media = BSplineSolver(**d0)._wire_media()
    rad, dl = [], []
    for med in media:
        ext = ET if med == _medium_spec.BELOW else 1.0
        rad.append(A_REF * (B_REF / A_REF) ** (1 - ext / ER_REF))
        dl.append(
            DistributedRLC(
                "series",
                l=MU0 / (2 * np.pi) * (1 - ext / ER_REF) * np.log(B_REF / A_REF),
            )
        )
    nw = len(media)
    exact = dict(d0, wire_radius=rad, distributed_rlc=dl)
    jacket = dict(
        d0,
        wire_radius=A_REF,
        insulation_radius=[B_REF] * nw,
        insulation_eps_r=[ER_REF] * nw,
    )
    return exact, jacket


PAIR_DECKS = {
    # (deck, bar on |jacket - exact|, floor on |free-space pair - exact|);
    # measured fixed <= 0.035 / 0.026 / 0.014 ohm, old 27.2 / 11.6 / 5.4 ohm.
    "dipole": (lambda: dipole(eps=LOSSLESS), 0.1, 20.0),
    "crossing": (lambda: crossing(1, ground_eps=LOSSLESS), 0.1, 8.0),
    "hub4": (lambda: hub(1, ground_eps=LOSSLESS), 0.1, 4.0),
}


@pytest.mark.parametrize("cls", [BSplineSolver, RazorSolver])
@pytest.mark.parametrize("name", sorted(PAIR_DECKS))
def test_the_jacket_in_soil_is_the_in_medium_pair(name, cls):
    mk, bar, _floor = PAIR_DECKS[name]
    exact, jacket = _pair_decks(mk())
    gap = abs(z(cls, jacket) - z(cls, exact))
    assert gap < bar, gap


@pytest.mark.parametrize("cls", [BSplineSolver, RazorSolver])
@pytest.mark.parametrize("name", sorted(PAIR_DECKS))
def test_the_free_space_pair_misses_the_in_medium_pair(name, cls, free_space_pair):
    """The red control: main's term, on the same decks, the same constructor."""
    mk, _bar, floor = PAIR_DECKS[name]
    exact, jacket = _pair_decks(mk())
    z_exact = z(cls, exact)
    free_space_pair()
    gap = abs(z(cls, jacket) - z_exact)
    assert gap > floor, gap


# ----------------------------------------------------------------------
# sweeps
# ----------------------------------------------------------------------


SWEPT_DECKS = {
    "dipole": lambda: dipole(insulation_radius=1.8e-3, insulation_eps_r=3.5),
    "crossing": lambda: crossing(
        1, insulation_radius=[2e-3, 2e-3], insulation_eps_r=[3.0, 3.0]
    ),
}


@pytest.mark.parametrize(
    "name", ["dipole", pytest.param("crossing", marks=pytest.mark.slow)]
)
@pytest.mark.parametrize("cls", [BSplineSolver, RazorSolver])
def test_a_swept_solve_is_the_single_frequency_solves(name, cls, free_space_pair):
    """eps~(w) per frequency: swept == single, and the sweep carries the term
    (its first point is not the free-space pair's).

    To 1e-9 ohm, the repo's swept bar, not bit for bit: bspline's swept
    FIRST point differs from a fresh single solve by ~5e-11 ohm on the bare
    crossing deck too (measured on this branch with no jacket, and with the
    charge term dropped), so that residue predates this term. Razor's and
    every later point are bit-identical."""
    d = SWEPT_DECKS[name]()
    lams = (d["wavelength"], d["wavelength"] / 1.05)
    ks = [2 * math.pi / lam for lam in lams]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        swept = np.asarray(solver(cls, d).compute_impedance_swept(ks)).ravel()
    single = np.array([z(cls, dict(d, wavelength=lam)) for lam in lams])
    assert np.max(np.abs(swept - single)) < 1e-9, swept - single
    free_space_pair()
    assert abs(swept[0] - z(cls, d)) > 1.0


# ----------------------------------------------------------------------
# the trunks onto each other (slow)
# ----------------------------------------------------------------------


def _jacket_shift(cls, d0, b, er):
    nw = len(d0["wires"])
    dj = dict(d0, insulation_radius=[b] * nw, insulation_eps_r=[er] * nw)
    return z(cls, dj) - z(cls, d0)


TRUNK_DECKS = {
    # measured gaps (probe3/probe4b): dipole 2.51 -> 1.49 -> 0.93,
    # crossing 0.68 -> 0.30 -> 0.15, hub4 0.61 -> 0.44 -> 0.28 ohm, on
    # jacket shifts of 47 / 49 / 65 ohm.
    "dipole": (lambda m: dipole(n=20 * m), 1.8e-3, 3.5),
    "crossing": (lambda m: dict(crossing(m), wire_radius=A_REF), 0.45e-3, 3.5),
    "hub4": (lambda m: dict(hub(m), wire_radius=A_REF), 0.45e-3, 3.5),
}


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(TRUNK_DECKS))
def test_the_jacket_shift_converges_bspline_onto_razor(name):
    mk, b, er = TRUNK_DECKS[name]
    g = [
        abs(
            _jacket_shift(BSplineSolver, mk(m), b, er)
            - _jacket_shift(RazorSolver, mk(m), b, er)
        )
        for m in (1, 2, 4)
    ]
    assert all(y / x <= 0.75 for x, y in zip(g, g[1:])), g
