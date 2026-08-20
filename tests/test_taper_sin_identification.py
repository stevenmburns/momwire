"""The sin/nec2c identification gate (momwire#398 D3, momwire#435 restated).

The taper-readiness study measured `|Z_sin - Z_nec2c|` converging to
0.01-0.04 Ω on `ward`/`fat` and slower-but-still-converging on `step2`/
`thin` — `sin` (momwire's point-matched `SinusoidalSolver`, reduced kernel)
reproduces nec2c's answer on a radius step, INCLUDING nec2c's own stepped-
radius (Leeson) defect, because it shares NEC-2's per-segment end-condition
convention (the `a_seg = 1/(ln(2/(k·a_seg)) - gamma)` constant `tbf()`
computes per segment).

**The maintainer's decision (D3, 2026-08-18): this is a FEATURE, not a bug.**
momwire#435 closes as an IDENTIFICATION: the sinusoidal family exists to be
the NEC-2-closest rung of momwire's basis ladder, and inheriting NEC-2's
defect on a radius step is what "NEC-2-closest" means on this deck class.
Fixing it would mean building momwire's own Leeson correction and `sin`
would stop being NEC-2-identified — a different, unrequested project.

So this file gates the IDENTIFICATION as a protected property:
`|Z_sin - Z_nec2c| <= 0.1 ohm` at the finest captured rung, on `step2`
(#435's own deck class — a 10:1 radius step at a junction) and `thin`
(the uniform control: the property should hold there too, more easily,
confirming it is not an artifact of the step). `sin` is EXPECTED to
disagree with NEC-5 on these decks (`test_taper_agreement_floor.py`
measures that disagreement directly, ~19-33 Ω on `step2`/`ward` at the
continuum limit) and REQUIRED to agree with nec2c.

Both rows ride the SAME NEC-2 deck (odd per-section segment count, so
NEC-2's `EX` centre-segment feed lands exactly at the fed section's
midpoint) — feed-matched by construction, no EX-trap placement term.
`nec2c` (open source, /usr/bin/nec2c) is captured once by
`scripts/capture_taper_nec5_lane.py` and pinned as a golden literal, the
same house choice `tests/test_extended_kernel.py`'s "ORACLE (nec2c 1.3.1
...)" comment documents, rather than re-run at test time.
"""

from __future__ import annotations

import pytest

from momwire.deck import build_solver, parse

from golden_taper_nec5 import SIN_NEC2C_LADDERS

FREQ_MHZ = 14.2
Z_HEIGHT = 10.0

WARD_LEN = 10.51010
STEP_LEN = 10.18946
STEP_RADII = [1.0e-2, 1.0262e-3]
THIN_RADIUS = 1.250000e-03

# n_per_sec ladders this golden module's rows were captured at (odd counts —
# see the module docstring). Kept here only to reconstruct the deck text;
# the golden module is the source of truth for n_total and the two Z's.
_SEC_COUNT = {"step2": 2, "thin": 10}

# Anti-stale bar on the recorded Z_sin column — how far the live solve may sit
# from the literal before we call it drift rather than arithmetic. It is NOT
# the 5e-6 the rest of the golden-taper suite uses (momwire#483): that bar is
# sized for the literal's own 6-decimal rounding (at most sqrt(2)*5e-7 =
# 7.1e-7, see `scripts/capture_taper_nec5_lane.py`), and it holds only while
# the solve stays well conditioned. The rung THIS test gates is the ladder's
# worst — kappa(G) runs 2.0e3 -> 2.3e5 over step2's N=26..402 and 2.0e3 ->
# 1.7e5 over thin's N=30..410.
#
# The fill is not bit-reproducible across arithmetic paths, and not at the ulp
# level either: the collocation matrix moves by ||dG||_F/||G||_F ~ 9e-12 (~4e4
# ulp — the Eq 76-79 endpoint brackets cancel) between the three fill paths
# available on one box, all of them supported. The C++ accelerator's SIMD
# sincos binds `_ZGVdN4v_sin`, an IFUNC whose implementation glibc picks from
# the host's hwcaps; masking FMA (GLIBC_TUNABLES=glibc.cpu.hwcaps=-FMA) picks
# a different one for the same symbol; and `_accel.py` documents the pure-
# numpy fallback as a deliberate pure-Python install. The linear-solve bound
# turns that into
#
#   |dZ| <= kappa(G) * ||dG||/||G|| * |Z|
#         = 2.3e5 * 9.0e-12 * 116.8 = 2.4e-4 ohm   (step2, N=402)
#         = 1.7e5 * 8.2e-12 *  89.9 = 1.3e-4 ohm   (thin,  N=410)
#
# and the bar below is that bound, rounded up to cover both rows. The largest
# spread actually measured — worst pair over those three fill paths plus the
# recording box's own column — is 5.3e-5 (step2) / 2.3e-5 (thin), so the bound
# is conservative by ~5x and that is where the margin sits.
#
# This is arithmetic, not a stale literal: down the ladder the live-vs-
# recorded gap stays at or near the literal's rounding floor (<= 1.8e-6 at
# every rung up to N=282/290) and only the finest rung, where kappa peaks,
# exceeds 5e-6 (1.6e-5 step2 / 1.4e-5 thin on a Haswell box whose libmvec
# variant differs from CI's). Nor is the widened bar decorative: it is ~40x
# tighter than the claim it guards, since `thin` clears the 0.1 ohm D3 gate
# below with only 0.0104 ohm to spare.
_GOLDEN_DRIFT_BAR = 2.5e-4


def _gw(tag, n, x0, x1, z, rad):
    return (
        f"GW {tag} {n} {x0:.6E} 0.000000E+00 {z:.6E} "
        f"{x1:.6E} 0.000000E+00 {z:.6E} {rad:.6E}"
    )


def _sections(name):
    if name == "step2":
        h = STEP_LEN / 2.0
        return [(0.0, h, STEP_RADII[0]), (h, STEP_LEN, STEP_RADII[1])], 1
    if name == "thin":
        s = WARD_LEN / 10
        return [(k * s, (k + 1) * s, THIN_RADIUS) for k in range(10)], 6
    raise ValueError(name)


def _nec2_deck(name, n_per_sec):
    sections, fed_tag = _sections(name)
    lines = [f"CM taper sin-id study {name} n={n_per_sec}", "CE"]
    for k, (x0, x1, rad) in enumerate(sections):
        lines.append(_gw(k + 1, n_per_sec, x0, x1, Z_HEIGHT, rad))
    lines.append("GE 0")
    seg = (n_per_sec + 1) // 2
    lines.append(f"EX 0 {fed_tag} {seg} 0 1.000000E+00 0.000000E+00")
    lines.append(f"FR 0 1 0 0 {FREQ_MHZ:.6E} 0.000000E+00")
    lines.append("XQ 0")
    lines.append("EN")
    return "\n".join(lines) + "\n"


def _sin_z(name, n_per_sec):
    deck = _nec2_deck(name, n_per_sec)
    model = parse(deck, dialect="nec2")
    built = build_solver(model, basis="sinusoidal", extended_kernel=False)
    ps = built.solver.compute_port_solution()
    port = built.ports.feed_ports[0]
    return complex(1.0 / ps.y[port, port])


@pytest.mark.parametrize("name", sorted(SIN_NEC2C_LADDERS))
def test_sin_converges_onto_nec2c_at_the_finest_rung(name):
    """CLAIM (D3, protected): at the finest captured rung,
    |Z_sin - Z_nec2c| <= 0.1 ohm on both `step2` and `thin`.

    Measured (this deck's own ladder): `step2` decays 0.787 -> 0.058 ohm
    over N=26..402; `thin` decays 0.122 -> 0.090 ohm over N=30..410 — both
    CROSS the 0.1 ohm line only at the finest rungs, so this is a
    convergence identification, not a claim that holds at every mesh
    density (the coarsest-rung sanity test below confirms that).
    """
    n_total, z_sin_recorded, z_nec2c = SIN_NEC2C_LADDERS[name][-1]
    n_per_sec = n_total // _SEC_COUNT[name]
    z_sin = _sin_z(name, n_per_sec)
    assert abs(z_sin - z_sin_recorded) < _GOLDEN_DRIFT_BAR, (
        f"{name}: sin moved from golden literal by "
        f"{abs(z_sin - z_sin_recorded):.3e} > {_GOLDEN_DRIFT_BAR:g}"
    )
    assert abs(z_sin - z_nec2c) <= 0.1, (
        f"{name} N={n_total}: |Z_sin - Z_nec2c| = {abs(z_sin - z_nec2c):.4f} > 0.1 ohm"
    )


@pytest.mark.parametrize("name", sorted(SIN_NEC2C_LADDERS))
def test_sin_nec2c_residual_decays_down_the_ladder(name):
    """The identification strengthens under refinement (not required by D3,
    but it is what "convergence" in the claim means, and a decay gate is
    cheap insurance against the two rows silently drifting apart under a
    future change instead of together)."""
    res = [abs(z_sin - z_nec2c) for _n, z_sin, z_nec2c in SIN_NEC2C_LADDERS[name]]
    for prev, nxt in zip(res, res[1:]):
        assert nxt <= prev + 1e-9, f"{name}: sin/nec2c residual grew ({res})"


@pytest.mark.parametrize("name", sorted(SIN_NEC2C_LADDERS))
def test_sin_does_not_trivially_agree_with_nec2c_at_the_coarsest_rung(name):
    """Guard against the gate being vacuous: the coarsest captured rung
    should NOT already clear the 0.1 ohm bar — otherwise "converges onto"
    would be true of any two solvers agreeing to within meshing noise,
    proving nothing about the shared NEC-2 formulation."""
    _n, z_sin, z_nec2c = SIN_NEC2C_LADDERS[name][0]
    assert abs(z_sin - z_nec2c) > 0.1


def test_sin_disagrees_with_nec5_where_it_agrees_with_nec2c():
    """The other half of D3's identification: `sin` is ALLOWED to disagree
    with NEC-5 on a radius step (it is not required to, and is not gated to
    — `test_taper_agreement_floor.py` does not gate `sin` against NEC-5 at
    all, on purpose). This test only confirms the two claims are not
    accidentally the same claim: on `step2`, `sin`'s finest-rung agreement
    with nec2c (~0.06 ohm) coexists with a large disagreement against NEC-5
    (the taper-readiness study measured the continuum-limit gap at ~19 ohm
    on this deck class) — reusing the TAPER_LADDERS NEC-5 column at a
    nearby rung as the reference point.
    """
    from golden_taper_nec5 import TAPER_LADDERS

    _n_sin, z_sin, _z_nec2c = SIN_NEC2C_LADDERS["step2"][-1]
    z_nec5 = TAPER_LADDERS["step2"][-1][1]
    assert abs(z_sin - z_nec5) > 1.0
