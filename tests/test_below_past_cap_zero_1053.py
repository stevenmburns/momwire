"""momwire#1053: past the below/below R1 cap the remainder is served as ZERO.

Until #1053 a buried structure reaching past `_SOMM_BELOW_R1_CAP_LAMBDA_M`
in-medium wavelengths refused by name. The record on the #1053 branch
(`scratch/u4-below-range/MEASUREMENTS.md`) measured what serving it as zero
costs before the refusal came out: on a synthetic deck whose buried screen
spans 4.76 lambda_m, zeroing every pair past 4 lambda_m moved Z by 3.1e-5 ohm
against the table's own ladder step. These gates hold that as a BOUND on each
trunk that consumes the projection, never as a literal: the fill's digits are
machine-dependent, and the ratio is not.

THE DECK is antennaknobs' own build of that record's `synthetic_u4.nec`: a
21.4 m monopole on a 0.15 m rise over four 38 m radials at 0.15 m depth, soil
(13, 0.005), 3.5 MHz. It is spelled here as the exact solver kwargs
antennaknobs' engine constructs, captured at construction with no fill
(`capture_kwargs.py` in the record), so a gate here solves the deck the record
solved.

THE BAR. `extended` is the same solve with the cap moved to 5 lambda_m, past
everything this deck reaches, so nothing is zeroed and the grid is tabulated
to the deck. The gate is |Z(shipped) - Z(extended)| <= 1e-2 of the extended
table's own r = 1 -> 3 ladder step: the zeroing's error is at most a
hundredth of the discretisation error the answer already carries at that
rung. Measured through antennaknobs' engine: 3.14e-5 against 0.1497 ohm,
2.1e-4 of the step.

THE GUARDS, read before delta, each one a way the bar could pass vacuously:
the shipped plan must reach past 4 lambda_m (else nothing past the cap existed
to zero); the shipped grid must stop AT the cap and the extended one must be
tabulated past it (else both spellings served the same pairs the same way and
agree trivially); and delta must not be bit-zero (else the zeroing never ran:
a change that moves nothing bit-for-bit is unplumbed).

`slow`: each gate fills two grids and solves three times.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

from momwire import _below_interface, _sommerfeld_below
from momwire.bspline import BSplineSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

MAST_M = 21.4
RADIAL_M = 38.0
DEPTH_M = 0.15
SHIPPED_CAP = 4.0
EXTENDED_CAP = 5.0
BAR_OF_LADDER_STEP = 1e-2

# The mast's lower vertices per refinement: antennaknobs splits the fed NEC
# segment out as its own edge so the feed lands at that segment's centre.
# Written as (length / n) * k, the product the captured kwargs hold bit-for-bit.
_MAST = {
    1: ([MAST_M / 11], [1, 10]),
    3: ([MAST_M / 33, (MAST_M / 33) * 2], [1, 1, 31]),
}


def synthetic_u4(refine):
    """The #1053 record's synthetic deck, as antennaknobs' engine builds it."""
    lower, mast_edges = _MAST[refine]
    mast = np.array(
        [(0.0, 0.0, 0.0)] + [(0.0, 0.0, z) for z in lower] + [(0.0, 0.0, MAST_M)]
    )
    rise = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, -DEPTH_M)])
    radials = [
        np.array([(0.0, 0.0, -DEPTH_M), (RADIAL_M * dx, RADIAL_M * dy, -DEPTH_M)])
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
    ]
    return dict(
        wires=[mast, rise, *radials],
        n_per_edge_per_wire=[mast_edges, [refine]] + [[19 * refine]] * 4,
        feeds=[(0, MAST_M / 22, 1 + 0j)],
        wavelength=85.654988,
        wire_radius=0.001,
        ground_z=0.0,
        junctions=[
            [(0, "start"), (1, "start")],
            [(1, "end"), (2, "start"), (3, "start"), (4, "start"), (5, "start")],
        ],
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
    )


@pytest.fixture
def cap_moves(monkeypatch):
    """The live cap is the one these gates were measured at, and every below
    grid tabulated under a MOVED cap is dropped once the gate ends. That drop
    is load-bearing, not tidiness: `get_grid_below` keys on the R1 BUCKET
    (1.25^n) and not on the cap, and 4.0 and this deck's 4.75 lambda_m share
    the 4.77 bucket, so a later shipped solve on this medium would otherwise
    be handed a grid tabulated past the cap that zeroes nothing."""
    assert _sommerfeld_below._SOMM_BELOW_R1_CAP_LAMBDA_M == SHIPPED_CAP, (
        "the cap moved: re-measure the bound before re-pointing this gate"
    )
    yield monkeypatch
    cache = _sommerfeld_below._GRID_CACHE
    for key in [
        k
        for k, g in cache.items()
        if getattr(g, "regime", None) == "below"
        and g.r1_cap > SHIPPED_CAP * g.lam_m * (1.0 + 1e-12)
    ]:
        del cache[key]


def _evict(grids):
    """Drop exactly these grid objects from the shared cache."""
    cache = _sommerfeld_below._GRID_CACHE
    for key in [k for k, g in cache.items() if any(g is x for x in grids)]:
        del cache[key]


def _spied_z(cls, refine, monkeypatch, seen):
    """Z at this rung, with the plan's R1 and every grid's tabulation
    recorded in in-medium wavelengths."""
    real_plan = _below_interface.serve_plan
    real_grid = _sommerfeld_below.get_grid_below
    plan_sig = inspect.signature(real_plan)

    def plan(*a, **kw):
        out = real_plan(*a, **kw)
        k_m = plan_sig.bind(*a, **kw).arguments["k_m"]
        seen.setdefault("plan_r1_wl", []).append(
            out["r1_below"] * abs(k_m) / (2.0 * np.pi)
        )
        return out

    def grid(*a, **kw):
        g = real_grid(*a, **kw)
        seen.setdefault("grid_r1_wl", []).append(g.r1_max / g.lam_m)
        seen.setdefault("grids", []).append(g)
        return g

    with monkeypatch.context() as m:
        m.setattr(_below_interface, "serve_plan", plan)
        m.setattr(_sommerfeld_below, "get_grid_below", grid)
        with warnings.catch_warnings():
            # The coarse-crossing-node advisory. The node is ungraded on
            # purpose: this is the record's deck, byte for byte.
            warnings.simplefilter("ignore")
            return complex(cls(**synthetic_u4(refine)).compute_impedance()[0])


def _the_bound_holds(cls, monkeypatch, record_property):
    shipped, extended = {}, {}
    z_ship = _spied_z(cls, 1, monkeypatch, shipped)
    # The same bucket collision from the other side: without this eviction the
    # extended solve is handed the SHIPPED grid (tabulated to 4, zeroing past
    # it) out of the cache, and the comparison is a bit-identical no-op. The
    # extended-grid guard below is what caught that.
    _evict(shipped["grids"])
    monkeypatch.setattr(_sommerfeld_below, "_SOMM_BELOW_R1_CAP_LAMBDA_M", EXTENDED_CAP)
    z_ext1 = _spied_z(cls, 1, monkeypatch, extended)
    z_ext3 = _spied_z(cls, 3, monkeypatch, {})
    delta = abs(z_ship - z_ext1)
    step = abs(z_ext3 - z_ext1)
    for name, value in (
        ("z_shipped_r1", z_ship),
        ("z_extended_r1", z_ext1),
        ("z_extended_r3", z_ext3),
        ("delta_ohm", delta),
        ("ladder_step_ohm", step),
        ("delta_over_step", delta / step),
    ):
        # Plain strings: record_property values cross the xdist wire.
        record_property(name, repr(value))

    assert max(shipped["plan_r1_wl"]) > SHIPPED_CAP, shipped
    assert max(shipped["grid_r1_wl"]) <= SHIPPED_CAP * (1.0 + 1e-12), shipped
    assert max(extended["grid_r1_wl"]) > SHIPPED_CAP, extended
    assert delta > 1e-9 * abs(z_ext1), (z_ship, z_ext1)
    assert delta <= BAR_OF_LADDER_STEP * step, (
        f"zeroing past the cap moved Z by {delta:.3e} ohm, more than "
        f"{BAR_OF_LADDER_STEP:g} of the extended table's own r = 1 -> 3 ladder "
        f"step ({step:.4g} ohm): shipped {z_ship!r}, extended {z_ext1!r}. The "
        "#1053 bound no longer holds on this trunk; do NOT widen the bar — "
        "re-measure, and refuse past the cap again if the zero is not licensed"
    )


@pytest.mark.slow
def test_bspline_zeroing_past_the_cap_is_a_hundredth_of_the_ladder_step(
    cap_moves, record_property
):
    _the_bound_holds(BSplineSolver, cap_moves, record_property)


@pytest.mark.slow
def test_sg_zeroing_past_the_cap_is_a_hundredth_of_the_ladder_step(
    cap_moves, record_property
):
    _the_bound_holds(SinusoidalGalerkinSolver, cap_moves, record_property)
