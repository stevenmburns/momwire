"""momwire#510 — is the grazing floor the interpolation grid's θ resolution?

Where experiments 1 and 3 left it
---------------------------------
The floor is in the finite ground (``GN 1`` is exact at 1.09e-4 λ), it is
keyed to h/λ and not h/Δ, it gets WORSE under mesh refinement, and it is the
HORIZONTAL wire in the grazing zone rather than the grazing feedpoint — a lone
vertical whose bottom segment grazes is exact.  Read together that says
catastrophic cancellation in the *numerical* ground: the same cancellation
over a closed-form image is exact, so it is the evaluation that fails, not the
physics.

``probe_contact_direct_remainder.py`` split that evaluation into two axes at
CONTACT and found the integration tolerance SATURATED (rtol 1e-7 through 1e-13
all give the same residual) with the whole ~0.003 Ω shift belonging to the
INTERPOLATION.  Contact is a different regime, so the split has to be asked
again here — but the grid's own layout comment already names the premise this
deck breaks::

    # The grazing band (region 0) keeps the 0.01-lambda spacing: there the
    # layer variable h = R1 sin(theta) stretches it out in R1, no physical
    # deck queries small R1 at grazing (two points at grazing-small R1 means
    # both are ON the plane — radial screens, which are refused) ...

0033 **is** an elevated radial screen and it is **not** refused.  And the
premise's mechanism fails for it specifically: the layer variable
h = R₁·sinθ is what "stretches out in R₁" for a general pair, but for two
points that both sit at a fixed grazing height it is PINNED at z_s + z_o
however large R₁ grows.  It does not stretch.

What this measures
------------------
``--mode theta`` is pure geometry against the shipped grid — no solver, no
binary.  For 0033's own mesh it walks every source/observer pair, computes the
(R₁, θ) that pair hands the grid, and asks the grid ITSELF (constructed at
this deck's ε̃, k and extent, so the numbers are the shipped lattice's and not
a re-derivation) which θ cell each query lands in.  If the queries pile into
the first cell, the cubic stencil is interpolating a function of h across a
cell whose h-span is orders of magnitude wider than the h being asked about.

``--mode direct`` is the decisive one: swap the interpolated grid for direct
evaluation and re-run 0033 against the binary at several heights.  If the
error collapses, the floor is the interpolation lattice — a keying defect with
a fix, the same shape momwire#443 already fixed once for the contact boundary
layer — and #510 stops being a refuse-only formulation limit.  If it does not,
the integration or the formulation underneath is next.

``--mode reflcoef`` partitions what is left.  ``refl-coef`` and ``sommerfeld``
share the image fold and differ only by the remainder Q, so forcing the seam's
ground kwargs to refl-coef — one monkeypatch on ``_serve._ground_kwargs``,
everything else about the deck, drive, mesh and route held — asks whether the
error lives in the shared half.  The deck route has no card for it (``GN 0``
maps to sommerfeld and should), which is why it is patched rather than
spelled.  refl-coef is outside its own documented 0.1-0.5 λ validity window
down here, so its ABSOLUTE error is not evidence of anything; what is evidence
is whether it tracks sommerfeld.

WHAT IT MEASURED (2026-08-25)
-----------------------------
``--mode theta`` — 0033's own mesh, 325 pairs, against the shipped lattice:

  h/λ            1.09e-4    1e-3    3e-3    1e-2    3e-2     1e-1
  median θ (deg)   0.115   1.058   3.171   10.46   23.66       54
  in first cell    64.6 %  64.6 %  64.6 %  43.1 %     0 %      0 %
  razor err%      171.86   44.18    3.91    0.06    0.00     0.01

The grazing band's θ cell is **10°** (region 0, R₁ < 0.2 λ) and 5° beyond it,
and at the native height the MEDIAN query sits at 0.115° — 87× inside the
first cell.  The error tracks how deep into that first cell the queries fall,
which is exactly what an interpolation-resolution failure looks like.

``--mode direct`` — and it is NOT that.  With the lattice removed entirely and
the surfaces evaluated directly at rtol 1e-11:

  h/λ        grid err%   direct err%
  1.09e-4      435.18       435.23    bspline
  1.09e-4      171.86       172.08    razor-nec5
  1e-3          44.18        44.19    razor-nec5
  3e-3           3.91         3.91    razor-nec5
  1e-2           0.06         0.06    razor-nec5

Identical to three figures.  **The interpolation lattice is exonerated, and so
is the integration tolerance** (direct evaluation IS the fill function, swept
at a tolerance two decades past production).  A textbook-looking resolution
story measured out — the surfaces are evidently smooth enough in θ near zero
that a cubic across 10° carries them.

``--mode reflcoef`` — and refl-coef does NOT track sommerfeld here, which is
where this parts company with #624's contact finding.  They separate, and the
gap grows exactly as the error does (razor-nec5, |Z_somm − Z_refl|): 10.0 Ω at
1e-2 λ, 22.9 at 3e-3, 71.9 at 1e-3, 176.3 at 1.09e-4.  Since the two differ
only by the remainder Q, that gap IS Q's contribution.

The sharpest form, and the one to work from — the GROUND CORRECTION,
Z(``GN 0``) − Z(``GN 1``), which is what the finite ground is worth on top of
a perfect image.  razor-nec5 over ``GN 1`` reproduces the binary to 0.00 % at
every height, so the binary's own PEC answer is a common baseline both sides
share exactly:

  h/λ       true (binary)        momwire (razor)      overshoot
  1e-2      1.577  +0.670j       1.582  +0.739j        1.02x
  3e-3      3.084  +8.700j       3.391 +13.009j        1.46x
  1e-3      4.473 +20.397j       9.065 +63.307j        3.06x
  1.09e-4  13.900 +61.346j      82.971+144.743j        2.65x

**momwire over-computes the finite-ground correction by up to ~3× at grazing
and is exact at 1e-2 λ.**  Backing refl-coef's undershoot out, Q is worth
176 Ω at the native height where it should be worth 71 Ω.

So the defect is in **how Q is scaled or folded into the system, not in
computing it** — its evaluation is verified correct by ``--mode direct`` and
its contribution is 2.5× too large anyway.  That is a bounded, well-localized
target rather than a formulation dead end.  It also rhymes with #624's
row-halving suspicion (a model-independent scale factor on a near-plane row)
without contradicting experiment 3: different geometric trigger and different
symptom, possibly one defect family.

(bspline's overshoot column is not readable the same way — its own ~5 % basis
error rides on a correction that is only 1.7 Ω at 1e-2 λ, which is what makes
that row 4.77×.  razor's column is the clean one, and it is clean precisely
because razor over PEC is exact.)

``--mode soil`` — and the scale factor is NOT one number.  |Δ_mw / Δ_true| for
razor-nec5 across the five golden half-spaces:

  h/λ        sea    vgood     avg    poor    diel      arg spread
  1e-2      1.066   1.025   1.019   0.978   0.997      0.0 .. 2.0 deg
  3e-3      2.048   1.482   1.456   1.661   0.889     -8.8 .. 7.8
  1e-3      5.455   3.183   3.063   3.261   1.946    -77.7 .. 12.3
  1.09e-4   4.436   3.033   2.652   3.232  61.188    -45.8 .. 1.9

Two readings, and the first is a control: **at 1e-2 λ the correction is right
on all five half-spaces** (0.98-1.07, phase ≤ 2°), so the floor is a property
of HEIGHT and not of any one soil.

Below it the overshoot is **soil-DEPENDENT** — 2.65× to 4.44× across the four
real soils at the native height, with the phase wandering −16° to −57°.  A
single real constant does not describe that, so **row-halving is ruled out as
the mechanism** and the analogy to #624 does not carry.  It was worth testing
and it is dead.

What replaces it is a better lead: **the lossless dielectric is catastrophic**
— 61× at the native height, against 2.6-4.4× for soils that conduct.  It is
also the one half-space whose true correction has a NEGATIVE real part
(−5.288 + 7.670j: a lossless dielectric lowers the resistance relative to a
perfect image).  momwire answers 110.7 + 559.2j to that.

``--mode sigma`` / ``--mode epsr`` — the two ORTHOGONAL axes, because the five
golden soils confound them (``diel`` is the least conductive AND least dense,
``sea`` both the most).  Absolute error in the ground correction, |Δ_mw −
Δ_true| in Ω, razor-nec5 at the native height:

  sigma       0     1e-6    1e-5    1e-4    3e-4    1e-3    3e-3    1e-2    1e-1
  eps_r 5   300.2   300.3   300.7   279.8   205.8   136.3   120.4   117.1   116.3
  eps_r 20   26.6    26.6    26.9    29.3    35.0    54.1    86.7   111.2   116.2

  eps_r    1.05    1.5     2.5     3.0    3.10    3.6      5      13      20     81
  |err|   18.95  96.71  563.58 2480.3  3343.9  939.6  300.7   63.92  26.85  67.53

**Two separate things, and the controls separate them.**

*One:* a **shared, soil-independent grazing defect** — both σ sweeps converge
on the SAME ~116 Ω plateau once the ground conducts (tan δ ≳ 5), from opposite
directions.  Across the four lossy golden soils the ABSOLUTE error is 82-136 Ω
while the true correction varies 23-63 Ω, so it is an ADDITIVE error that
barely depends on the half-space — not a multiplicative one.  That is a
sharper statement than the soil mode's ratio column supports on its own, and
it is a second reason row-halving is the wrong suspect: a mis-scaled row would
give error ∝ Δ_true.

*Two:* a **razor-specific resonance in ε_r at ≈ 3.1**, at deep grazing with
low loss.  Resolved finely, and it is a pole, not a bump:

  eps_r    2.60    2.80    2.90    3.00    3.05    3.10    3.20    3.40    3.60
  razor   695.7  1171.9  1650.9  2480.3  3002.4  3343.9  2797.1  1442.3   939.6
   arg    -44.1   -46.5   -53.2   -69.3   -84.1  -104.1  -142.2  -170.5  -177.6
  bspline  99.3   109.6   114.3   118.7   120.7   122.8   126.7   133.8   140.3

A magnitude peak with a ~135° PHASE SWEEP through it, in razor alone; bspline
walks smoothly 99 → 140 Ω across the same window with no feature at all.  That
reads as a spurious pole in razor's ASSEMBLY, and ``--mode direct --eps-r 3.0``
confirms it is not the Sommerfeld evaluation: grid 2466.91 % against direct
2470.40 %, so the one-soil exoneration above survives its worst corner.

**Correcting the reading this probe reached one run earlier.**  The ε_r = 5
σ-sweep's monotone fall (300 → 116 Ω) looked like momwire#282 stage 2's "a
loss term must vanish with σ" shape.  It is not: ε_r = 5 sits on the
resonance's SHOULDER, and what σ damps there is the pole.  Clear of it at
ε_r = 20 the trend REVERSES (26.6 → 116 Ω).  So #282's σ resemblance is not
confirmed, and the earlier note claiming it was is superseded by the
orthogonal sweep.  ε_r ≈ 3 is also outside real ground (5-81), which is why
no captured deck meets the pole — it is a diagnostic pointer, not a user
symptom.

Runs the binary in ``--mode direct`` / ``--mode reflcoef``, so: antennaknobs
venv, ``NEC5_EXE`` set.

    NEC5_EXE=~/antennas/NEC5-downloads/nec5-linux/nec5cl \
        /home/smburns/antennas/antennaknobs/.venv/bin/python \
        scripts/probe_grazing_ground_path.py --mode theta
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from momwire import _sommerfeld
from probe_grazing_height_floor import (
    FREQ_MHZ,
    N_SEG,
    RADIAL_LEN,
    SOIL,
    TRUNKS,
    VERT_LEN,
    WL,
    deck,
    momwire_z,
)

EPS0 = 8.8541878128e-12

# The heights the two earlier probes bracket the floor with.
THETA_HEIGHTS = (1.09e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
DIRECT_HEIGHTS = (1.09e-4, 1e-3, 3e-3, 1e-2)

# The soil sweep's rungs: two deep inside the broken zone, one on the shoulder
# and one clean, so a soil-independent ratio can be told from a soil-dependent
# one at each depth rather than only at the worst.
SOIL_HEIGHTS = (1.09e-4, 1e-3, 3e-3, 1e-2)

# The two ORTHOGONAL half-space axes, momwire#282 stage 2's shape.  The five
# golden soils confound σ with ε_r — `diel` is both the least conductive and
# the least dense, `sea` both the most — so neither can be read off them.
# `sigma` holds ε_r at poor soil's 5.0 and walks the conductivity through five
# decades; `epsr` holds σ small enough that the loss tangent stays ≲ 0.1
# everywhere on the sweep and walks the permittivity.
SIGMA_SWEEP = (0.0, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
EPSR_SWEEP = (1.05, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0, 13.0, 20.0, 40.0, 81.0)
AXIS_HEIGHTS = (1.09e-4, 1e-3, 1e-2)


def medium_eps_t() -> complex:
    """``GN 0 ... 13. .005`` as the complex ε̃ the grid is keyed on."""
    omega = 2.0 * np.pi * FREQ_MHZ * 1e6
    return SOIL[0] - 1j * SOIL[1] / (omega * EPS0)


def node_points(h: float) -> np.ndarray:
    """Every mesh node of 0033 at radial height ``h``.

    Segment CENTRES would be the more faithful query set, but the pair
    geometry this mode is about — R₁ and θ of a source/observer pair — is set
    by the heights and the horizontal separation, and nodes and centres span
    the same ranges of both.  Nodes are what the deck states, so nodes are
    what is walked.
    """
    pts = [(0.0, 0.0, h + i * VERT_LEN / N_SEG) for i in range(N_SEG + 1)]
    for ux, uy in ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)):
        for i in range(1, N_SEG + 1):
            s = i * RADIAL_LEN / N_SEG
            pts.append((ux * s, uy * s, h))
    return np.array(pts)


def pair_queries(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R₁, θ) for every ordered pair, in the grid's own convention.

    R₁ is source to the IMAGE of the observer — the distance the half-space
    surfaces are tabulated against — and θ is that ray's angle above the
    interface, so a pair of points at a common grazing height gives
    R₁ ≈ horizontal separation and θ ≈ (z_s + z_o)/R₁.
    """
    d = pts[:, None, :2] - pts[None, :, :2]
    rho = np.hypot(d[..., 0], d[..., 1])
    zsum = pts[:, None, 2] + pts[None, :, 2]
    r1 = np.hypot(rho, zsum)
    theta = np.arcsin(np.clip(zsum / np.maximum(r1, 1e-300), 0.0, 1.0))
    iu = np.triu_indices(len(pts), k=1)
    return r1[iu], theta[iu]


def grid_for(h: float):
    """The shipped grid this deck's solve would build, so the θ lattice read
    below is the real one rather than a re-derivation of the layout rules."""
    k2 = 2.0 * np.pi / WL
    pts = node_points(h)
    r1, _ = pair_queries(pts)
    return _sommerfeld.get_grid(
        medium_eps_t(), k2, float(r1.max()), omega=k2 * 299792458.0
    )


def theta_cells(grid) -> list[tuple[float, float, float, float]]:
    """(r0, r1, θ0, Δθ) of each region, in metres and degrees."""
    out = []
    for reg in grid._regions:
        r_lo = float(reg["r0"])
        r_hi = r_lo + float(reg["dr"]) * (int(reg["n_r"]) - 1)
        out.append(
            (
                r_lo,
                r_hi,
                float(np.degrees(reg["th0"])),
                float(np.degrees(reg["dth"])),
            )
        )
    return out


def mode_theta() -> list[dict]:
    print("0033's mesh — every pair's (R1, theta) against the shipped lattice")
    print(f"lambda = {WL:.2f} m, soil eps_r={SOIL[0]} sigma={SOIL[1]}\n")

    rows = []
    for hw in THETA_HEIGHTS:
        h = hw * WL
        pts = node_points(h)
        r1, theta = pair_queries(pts)
        th_deg = np.degrees(theta)
        grid = grid_for(h)
        cells = theta_cells(grid)
        # The coarsest first-cell width any query actually lands in: each
        # region owns an R1 band, so a query's cell is its band's.
        first = []
        for rr, tt in zip(r1, th_deg):
            widths = [
                dth for (r0, r1e, th0, dth) in cells if r0 - 1e-9 <= rr <= r1e + 1e-9
            ]
            first.append(min(widths) if widths else float("nan"))
        first = np.array(first)
        inside = np.isfinite(first) & (th_deg < first)
        print(f"h/lambda = {hw:<9.3g}  h = {h:.5g} m   ({len(r1)} pairs)")
        print("   grid theta cells (deg), by R1 band:")
        for r0, r1e, th0, dth in cells:
            print(
                f"      R1 {r0 / WL:>7.4f} .. {r1e / WL:<7.4f} lambda   "
                f"theta0 {th0:>4.1f}  dtheta {dth:.3f}"
            )
        print(
            f"   queried theta: min {th_deg.min():.4g} deg, "
            f"median {np.median(th_deg):.4g}, max {th_deg.max():.4g}"
        )
        print(
            f"   pairs inside their band's FIRST theta cell: "
            f"{inside.sum()} / {len(r1)}  ({100.0 * inside.mean():.1f} %)\n"
        )
        rows.append(
            dict(
                h_over_wl=hw,
                h=h,
                n_pairs=int(len(r1)),
                theta_min_deg=float(th_deg.min()),
                theta_median_deg=float(np.median(th_deg)),
                theta_max_deg=float(th_deg.max()),
                frac_in_first_cell=float(inside.mean()),
                cells=[list(c) for c in cells],
            )
        )
    return rows


class DirectGrid:
    """:class:`SommerfeldGrid`'s query contract, answered without a lattice.

    Lifted from ``probe_contact_direct_remainder.py`` — same contract, same
    memoization — and carrying the attributes the remainder path reads.  The
    cache is exact, not an approximation: a deck's obs/src halves repeat their
    (R₁, θ) bit-for-bit.
    """

    def __init__(self, eps_t, k2, r1_max, omega, mu, rtol):
        self.eps_t = complex(eps_t)
        self.k2 = float(k2)
        self.r1_max = float(r1_max)
        self.r_near = float(r1_max)
        self.omega = float(omega)
        self.mu = float(mu)
        self.rtol = float(rtol)
        self._cache: dict[tuple[float, float], tuple[complex, ...]] = {}
        self.calls = 0
        self.misses = 0

    def eval(self, R1, theta):
        r_b, th_b = np.broadcast_arrays(
            np.asarray(R1, dtype=float), np.asarray(theta, dtype=float)
        )
        shape = r_b.shape
        r_f = r_b.ravel()
        th_f = np.clip(th_b.ravel(), 0.0, 0.5 * np.pi)
        self.calls += r_f.size
        keys = [(float(a), float(b)) for a, b in zip(r_f, th_f)]
        fresh = sorted({kk for kk in keys if kk not in self._cache})
        if fresh:
            self.misses += len(fresh)
            surf = _sommerfeld.iv_surfaces_direct(
                self.eps_t,
                self.k2,
                np.array([kk[0] for kk in fresh]),
                np.array([kk[1] for kk in fresh]),
                rtol=self.rtol,
                omega=self.omega,
                mu=self.mu,
            )
            for i, kk in enumerate(fresh):
                self._cache[kk] = tuple(surf[s][i] for s in _sommerfeld._SURF_KEYS)
        out = np.empty((len(_sommerfeld._SURF_KEYS), r_f.size), dtype=np.complex128)
        for i, kk in enumerate(keys):
            out[:, i] = self._cache[kk]
        return {
            key: out[s].reshape(shape) for s, key in enumerate(_sommerfeld._SURF_KEYS)
        }


class MaskedAcc:
    """The accelerator module with only its GRID-consuming kernels hidden.

    Both remainder kernels are selected by ``hasattr``, so masking those two
    names routes the remainder — and only the remainder — through the numpy
    body that calls ``grid.eval``, leaving the rest of the fill accelerated.
    """

    _HIDDEN = ("sommerfeld_remainder_bspline_Q", "remainder_field_proj_batch")

    def __init__(self, acc):
        self._acc = acc

    def __getattr__(self, name):
        if name in MaskedAcc._HIDDEN:
            raise AttributeError(name)
        return getattr(self._acc, name)


def with_direct_grid(fn, rtol: float):
    """Run ``fn`` with every solver's Sommerfeld grid replaced by direct
    evaluation.

    ``_sommerfeld.get_grid`` is the single funnel — bspline reaches it through
    ``_somm_grid`` and razor through ``_potential_ground`` — so one patch
    covers both trunks whatever route the seam takes to build them.
    """
    from momwire import bspline

    # `_potential_ground` — razor's route — holds no `_acc` of its own: it
    # reaches the kernels through `_sommerfeld.remainder_field_proj`, so
    # masking `_sommerfeld`'s covers both trunks.  It still picks up the
    # patched grid, because `get_grid` is looked up on the module per call.
    saved_get = _sommerfeld.get_grid
    saved = {m: m._acc for m in (bspline, _sommerfeld)}
    grids: list[DirectGrid] = []

    def _direct(eps_t, k2, r1_max, omega, mu=_sommerfeld._MU0, cancel_flag=0):
        g = DirectGrid(eps_t, k2, r1_max, omega, mu, rtol)
        grids.append(g)
        return g

    _sommerfeld.get_grid = _direct
    for mod, acc in saved.items():
        if acc is not None:
            mod._acc = MaskedAcc(acc)
    try:
        return fn(), grids
    finally:
        _sommerfeld.get_grid = saved_get
        for mod, acc in saved.items():
            mod._acc = acc


def mode_direct(rtol: float, soil: tuple[float, float] = SOIL) -> list[dict]:
    exe = Path(os.path.expanduser(os.environ.get("NEC5_EXE", "")))
    if not exe.is_file():
        raise SystemExit(f"NEC5_EXE not found: {exe}")
    sys.path.insert(0, str(Path.home() / "antennas/antennaknobs/scripts"))
    from antennaknobs.engines.nec5 import NEC5Engine
    from bench_nec5_walk_why import make_dipole

    captures = Path(
        os.environ.get("GRAZING_CAPTURES", "/tmp/claude-1000/510-grazing-captures")
    )
    captures.mkdir(parents=True, exist_ok=True)
    eng = NEC5Engine(make_dipole(20), ground=None, capture_dir=captures)

    print("0033 against the binary, shipped grid vs direct evaluation")
    print(f"direct rtol = {rtol:g}; lambda = {WL:.2f} m")
    print(f"soil: eps_r = {soil[0]}, sigma = {soil[1]}\n")
    hdr = (
        f"{'h/lambda':>10} {'trunk':>11} | {'nec5cl':>21} | "
        f"{'grid':>21} {'err%':>8} | {'direct':>21} {'err%':>8} | {'calls':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for hw in DIRECT_HEIGHTS:
        text = deck(hw * WL, soil=soil)
        z_ref = complex(eng.run_deck(text)[0][0][2])
        for trunk in TRUNKS:
            z_grid = momwire_z(text, trunk)
            (z_dir, grids) = with_direct_grid(lambda: momwire_z(text, trunk), rtol)
            calls = sum(g.misses for g in grids)
            e_g = 100.0 * abs(z_grid - z_ref) / abs(z_ref) if z_grid else float("nan")
            e_d = 100.0 * abs(z_dir - z_ref) / abs(z_ref) if z_dir else float("nan")
            print(
                f"{hw:>10.3g} {trunk:>11} | "
                f"{z_ref.real:>10.4f}{z_ref.imag:>+10.4f}j | "
                f"{z_grid.real:>10.4f}{z_grid.imag:>+10.4f}j {e_g:>8.2f} | "
                f"{z_dir.real:>10.4f}{z_dir.imag:>+10.4f}j {e_d:>8.2f} | {calls:>8}",
                flush=True,
            )
            rows.append(
                dict(
                    h_over_wl=hw,
                    trunk=trunk,
                    z_ref=[z_ref.real, z_ref.imag],
                    z_grid=[z_grid.real, z_grid.imag],
                    z_direct=[z_dir.real, z_dir.imag],
                    err_grid_pct=e_g,
                    err_direct_pct=e_d,
                    direct_evals=calls,
                )
            )
    return rows


def mode_reflcoef() -> list[dict]:
    """Does the error live in the half refl-coef and sommerfeld SHARE?

    The seam has no card for refl-coef and should not grow one — ``GN 0`` is
    a Sommerfeld ground and serving it as anything else would be a lie. So
    the ground kwargs are patched for the length of one solve, which holds
    the deck, the drive, the mesh and the whole route fixed and moves exactly
    one thing.
    """
    from momwire.eznec import _serve

    exe = Path(os.path.expanduser(os.environ.get("NEC5_EXE", "")))
    if not exe.is_file():
        raise SystemExit(f"NEC5_EXE not found: {exe}")
    sys.path.insert(0, str(Path.home() / "antennas/antennaknobs/scripts"))
    from antennaknobs.engines.nec5 import NEC5Engine
    from bench_nec5_walk_why import make_dipole

    captures = Path(
        os.environ.get("GRAZING_CAPTURES", "/tmp/claude-1000/510-grazing-captures")
    )
    captures.mkdir(parents=True, exist_ok=True)
    eng = NEC5Engine(make_dipole(20), ground=None, capture_dir=captures)

    saved = _serve._ground_kwargs

    def _as_refl(deck_, medium):
        kw = saved(deck_, medium)
        if kw.get("ground_model") == "sommerfeld":
            kw = dict(kw, ground_model="refl-coef")
        return kw

    print("0033 against the binary — sommerfeld vs refl-coef, everything")
    print("else held (same deck, drive, mesh, seam route)\n")
    print("refl-coef is outside its own 0.1-0.5 lambda window down here, so")
    print("its absolute error proves nothing; whether it TRACKS does.\n")
    hdr = (
        f"{'h/lambda':>10} {'trunk':>11} | {'nec5cl':>21} | "
        f"{'sommerfeld':>21} {'err%':>8} | {'refl-coef':>21} {'err%':>8} | "
        f"{'|somm-refl|':>11}"
    )
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for hw in DIRECT_HEIGHTS:
        text = deck(hw * WL)
        z_ref = complex(eng.run_deck(text)[0][0][2])
        z_pec = complex(eng.run_deck(deck(hw * WL, pec=True))[0][0][2])
        for trunk in TRUNKS:
            z_s = momwire_z(text, trunk)
            _serve._ground_kwargs = _as_refl
            try:
                z_r = momwire_z(text, trunk)
            finally:
                _serve._ground_kwargs = saved
            if z_s is None or z_r is None:
                continue
            e_s = 100.0 * abs(z_s - z_ref) / abs(z_ref)
            e_r = 100.0 * abs(z_r - z_ref) / abs(z_ref)
            print(
                f"{hw:>10.3g} {trunk:>11} | "
                f"{z_ref.real:>10.4f}{z_ref.imag:>+10.4f}j | "
                f"{z_s.real:>10.4f}{z_s.imag:>+10.4f}j {e_s:>8.2f} | "
                f"{z_r.real:>10.4f}{z_r.imag:>+10.4f}j {e_r:>8.2f} | "
                f"{abs(z_s - z_r):>11.4f}",
                flush=True,
            )
            rows.append(
                dict(
                    h_over_wl=hw,
                    trunk=trunk,
                    z_ref=[z_ref.real, z_ref.imag],
                    z_pec_ref=[z_pec.real, z_pec.imag],
                    z_sommerfeld=[z_s.real, z_s.imag],
                    z_reflcoef=[z_r.real, z_r.imag],
                    err_sommerfeld_pct=e_s,
                    err_reflcoef_pct=e_r,
                    gap=abs(z_s - z_r),
                    d_true=[(z_ref - z_pec).real, (z_ref - z_pec).imag],
                    d_momwire=[(z_s - z_pec).real, (z_s - z_pec).imag],
                    overshoot=abs(z_s - z_pec) / abs(z_ref - z_pec),
                )
            )

    # The sharpest form of the question. razor-nec5 over `GN 1` reproduces the
    # binary to 0.00 % at every height (experiment 1), so the binary's own PEC
    # answer is a legitimate common baseline and the GROUND CORRECTION — what
    # the finite ground is worth on top of a perfect image — can be compared
    # directly instead of through two absolute impedances.
    print("\n\nthe ground correction itself: Z(GN 0) - Z(GN 1)\n")
    hdr2 = (
        f"{'h/lambda':>10} {'trunk':>11} | {'true (binary)':>21} | "
        f"{'momwire':>21} | {'overshoot':>9}"
    )
    print(hdr2)
    print("-" * len(hdr2))
    for row in rows:
        dt = complex(*row["d_true"])
        dm = complex(*row["d_momwire"])
        print(
            f"{row['h_over_wl']:>10.3g} {row['trunk']:>11} | "
            f"{dt.real:>10.4f}{dt.imag:>+10.4f}j | "
            f"{dm.real:>10.4f}{dm.imag:>+10.4f}j | {row['overshoot']:>8.2f}x"
        )
    return rows


def mode_soil() -> list[dict]:
    """Is the overshoot a model-INDEPENDENT scale factor?

    The reflcoef mode measured the ground correction Z(``GN 0``) − Z(``GN 1``)
    overshooting by up to ~3x on average soil.  A scale factor that is the
    same across five very different half-spaces is a scaling defect; one that
    tracks ε̃ is a physics term with the wrong weight.  That is the question
    #624's spike asked from the contact side and answered "model-independent"
    — here it is asked at grazing, where the symptom is 100x bigger.

    The COMPLEX ratio is what is read, not its magnitude.  A pure scale factor
    is real: if Δ_momwire / Δ_true sits at (say) 2.6 + 0j on every soil, the
    correction is being multiplied by a constant.  If its phase wanders with
    the soil, it is not one number and row-halving is the wrong suspect.

    razor-nec5 only.  bspline's ~5 % basis error rides on the same correction
    and contaminates the ratio (its 1e-2 λ row came out 4.77x on a 1.7 Ω
    correction), and razor's column is exact over ``GN 1`` at every height,
    which is what makes the PEC baseline a shared one.
    """
    exe = Path(os.path.expanduser(os.environ.get("NEC5_EXE", "")))
    if not exe.is_file():
        raise SystemExit(f"NEC5_EXE not found: {exe}")
    sys.path.insert(0, str(Path.home() / "antennas/antennaknobs/scripts"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
    from antennaknobs.engines.nec5 import NEC5Engine
    from bench_nec5_walk_why import make_dipole
    from golden_contact_nec5 import GROUND_EPS

    captures = Path(
        os.environ.get("GRAZING_CAPTURES", "/tmp/claude-1000/510-grazing-captures")
    )
    captures.mkdir(parents=True, exist_ok=True)
    eng = NEC5Engine(make_dipole(20), ground=None, capture_dir=captures)

    print("the ground correction Z(GN 0) - Z(GN 1) across five half-spaces")
    print("razor-nec5; the PEC baseline is soil-independent and exact\n")
    print("a model-INDEPENDENT scale factor is a REAL ratio, constant in soil;")
    print("a mis-weighted physics term tracks eps~ and wanders in phase.\n")

    rows = []
    for hw in SOIL_HEIGHTS:
        z_pec = complex(eng.run_deck(deck(hw * WL, pec=True))[0][0][2])
        print(f"h/lambda = {hw:<9.3g}   Z(GN 1) = {z_pec.real:.4f}{z_pec.imag:+.4f}j")
        hdr = (
            f"   {'soil':>6} {'eps_r':>7} {'sigma':>8} | {'true':>20} | "
            f"{'momwire':>20} | {'ratio':>18} {'|.|':>7} {'arg':>8}"
        )
        print(hdr)
        print("   " + "-" * (len(hdr) - 3))
        for name, soil in GROUND_EPS.items():
            text = deck(hw * WL, soil=soil)
            z_ref = complex(eng.run_deck(text)[0][0][2])
            z_mw = momwire_z(text, "razor-nec5")
            if z_mw is None:
                continue
            d_true = z_ref - z_pec
            d_mw = z_mw - z_pec
            ratio = d_mw / d_true if abs(d_true) > 0 else complex("nan")
            print(
                f"   {name:>6} {soil[0]:>7.1f} {soil[1]:>8.5g} | "
                f"{d_true.real:>9.4f}{d_true.imag:>+9.4f}j | "
                f"{d_mw.real:>9.4f}{d_mw.imag:>+9.4f}j | "
                f"{ratio.real:>8.3f}{ratio.imag:>+8.3f}j "
                f"{abs(ratio):>7.3f} {np.degrees(np.angle(ratio)):>7.1f}d",
                flush=True,
            )
            rows.append(
                dict(
                    h_over_wl=hw,
                    soil=name,
                    eps_r=soil[0],
                    sigma=soil[1],
                    z_pec=[z_pec.real, z_pec.imag],
                    z_ref=[z_ref.real, z_ref.imag],
                    z_mw=[z_mw.real, z_mw.imag],
                    d_true=[d_true.real, d_true.imag],
                    d_momwire=[d_mw.real, d_mw.imag],
                    ratio=[ratio.real, ratio.imag],
                    ratio_mag=abs(ratio),
                    ratio_arg_deg=float(np.degrees(np.angle(ratio))),
                )
            )
        print()
    return rows


def mode_axis(
    which: str,
    *,
    values: tuple[float, ...] | None = None,
    heights: tuple[float, ...] | None = None,
    trunks: tuple[str, ...] = ("razor-nec5",),
    hold_eps: float = 5.0,
) -> list[dict]:
    """The overshoot along ONE half-space axis, with the other held.

    The soil mode found the ground correction overshooting 2.6-4.4x on the
    four conducting soils and 61x on the lossless dielectric, which resembles
    momwire#282 stage 2's contact discrepancy — single-peaked in ε_r, falling
    monotonically in σ.  But the five golden soils confound the two axes
    (``diel`` is the least conductive AND the least dense; ``sea`` is both the
    most), so the resemblance cannot be read off them.  These are #282's own
    two orthogonal sweeps, asked at grazing.

    A LOSS story has to vanish with σ.  If the overshoot is largest at σ = 0
    and falls monotonically as the ground becomes conductive, the resemblance
    is real and #282's stage-2 record is machinery rather than analogy.  If it
    is flat in σ and structured in ε_r alone, the two findings only look
    alike.

    The loss tangent is printed because it, not σ, is what says whether a
    half-space is "lossy" at a given frequency: at 1.832 MHz average soil's
    tan δ is 3.8 and the lossless dielectric's is 0.039, which is the gap the
    soil mode was actually straddling.
    """
    exe = Path(os.path.expanduser(os.environ.get("NEC5_EXE", "")))
    if not exe.is_file():
        raise SystemExit(f"NEC5_EXE not found: {exe}")
    sys.path.insert(0, str(Path.home() / "antennas/antennaknobs/scripts"))
    from antennaknobs.engines.nec5 import NEC5Engine
    from bench_nec5_walk_why import make_dipole

    captures = Path(
        os.environ.get("GRAZING_CAPTURES", "/tmp/claude-1000/510-grazing-captures")
    )
    captures.mkdir(parents=True, exist_ok=True)
    eng = NEC5Engine(make_dipole(20), ground=None, capture_dir=captures)

    omega = 2.0 * np.pi * FREQ_MHZ * 1e6
    if which == "sigma":
        cases = [(hold_eps, s) for s in (values or SIGMA_SWEEP)]
        print(f"sigma axis at eps_r = {hold_eps} — a LOSS term must vanish with sigma")
    else:
        cases = [(e, 1e-5) for e in (values or EPSR_SWEEP)]
        print("eps_r axis at sigma = 1e-5 — loss tangent <= 0.1 throughout")
    print(f"ground correction Z(GN 0) - Z(GN 1), {', '.join(trunks)}\n")

    rows = []
    for hw in heights or AXIS_HEIGHTS:
        z_pec = complex(eng.run_deck(deck(hw * WL, pec=True))[0][0][2])
        print(f"h/lambda = {hw:<9.3g}   Z(GN 1) = {z_pec.real:.4f}{z_pec.imag:+.4f}j")
        hdr = (
            f"   {'eps_r':>7} {'sigma':>9} {'tan_d':>9} {'trunk':>11} | "
            f"{'true':>20} | {'momwire':>20} | {'|ratio|':>9} {'arg':>7} | "
            f"{'|err|':>10}"
        )
        print(hdr)
        print("   " + "-" * (len(hdr) - 3))
        for eps_r, sigma in cases:
            text = deck(hw * WL, soil=(eps_r, sigma))
            z_ref = complex(eng.run_deck(text)[0][0][2])
            d_true = z_ref - z_pec
            tan_d = sigma / (omega * EPS0 * eps_r)
            for trunk in trunks:
                z_mw = momwire_z(text, trunk)
                if z_mw is None:
                    continue
                d_mw = z_mw - z_pec
                ratio = d_mw / d_true if abs(d_true) > 0 else complex("nan")
                print(
                    f"   {eps_r:>7.2f} {sigma:>9.3g} {tan_d:>9.3g} {trunk:>11} | "
                    f"{d_true.real:>9.4f}{d_true.imag:>+9.4f}j | "
                    f"{d_mw.real:>9.4f}{d_mw.imag:>+9.4f}j | "
                    f"{abs(ratio):>9.3f} {np.degrees(np.angle(ratio)):>6.1f}d | "
                    f"{abs(d_mw - d_true):>10.4f}",
                    flush=True,
                )
                rows.append(
                    dict(
                        axis=which,
                        h_over_wl=hw,
                        eps_r=eps_r,
                        sigma=sigma,
                        tan_delta=tan_d,
                        trunk=trunk,
                        z_pec=[z_pec.real, z_pec.imag],
                        d_true=[d_true.real, d_true.imag],
                        d_momwire=[d_mw.real, d_mw.imag],
                        ratio_mag=abs(ratio),
                        ratio_arg_deg=float(np.degrees(np.angle(ratio))),
                        abs_err=abs(d_mw - d_true),
                    )
                )
        print()
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=("theta", "direct", "reflcoef", "soil", "sigma", "epsr"),
        default="theta",
    )
    p.add_argument("--rtol", type=float, default=1e-11)
    # `--mode direct`'s first run was on average soil alone, and the eps_r
    # axis then found a 2480 ohm peak at eps_r = 3 with no loss in it. An
    # exoneration measured at one half-space does not cover that one, so the
    # soil is settable and the peak gets asked the same question.
    p.add_argument("--eps-r", type=float, default=None)
    p.add_argument("--sigma", type=float, default=1e-5)
    # Axis-mode overrides, for resolving a feature the default lattice only
    # samples: the eps_r sweep found a 2480 ohm spike between its 2.5 and 4.0
    # rungs and how SHARP that spike is decides whether it reads as a pole.
    p.add_argument("--values", type=float, nargs="+", default=None)
    p.add_argument("--heights", type=float, nargs="+", default=None)
    p.add_argument("--trunks", nargs="+", default=["razor-nec5"])
    # Which eps_r the sigma axis holds. Default 5.0 is poor soil's, which sits
    # on the SHOULDER of razor's eps_r ~ 3.1 spike -- so a second sigma sweep
    # well clear of it is what separates "loss damps a pole" from "loss damps
    # the shared grazing error".
    p.add_argument("--hold-eps", type=float, default=5.0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.mode == "theta":
        rows = mode_theta()
    elif args.mode == "direct":
        soil = SOIL if args.eps_r is None else (args.eps_r, args.sigma)
        rows = mode_direct(args.rtol, soil)
    elif args.mode == "reflcoef":
        rows = mode_reflcoef()
    elif args.mode == "soil":
        rows = mode_soil()
    else:
        rows = mode_axis(
            args.mode,
            values=tuple(args.values) if args.values else None,
            hold_eps=args.hold_eps,
            heights=tuple(args.heights) if args.heights else None,
            trunks=tuple(args.trunks),
        )
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
