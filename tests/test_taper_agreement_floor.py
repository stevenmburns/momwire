"""The taper agreement floor: momwire#398 unit 1 (taper-readiness study).

The study (`scratch/study-taper/taper-readiness.md` at the time this was
written; see `docs/design/solver-architecture.md` §6.12 for the pointer that
survives) measured momwire against the licensed NEC-5 binary on four decks
that together span a 20:1 range of Δ/a — the difficulty of tapered and
stepped-radius wire in one number. This file is what makes that measurement
a STANDING gate rather than a one-time study: `tests/golden_taper_nec5.py`
carries NEC-5's (and nec2c's) printed output, captured by
`scripts/capture_taper_nec5_lane.py`, and every bar here recomputes momwire's
side LIVE and checks it both against the recorded value (catches a stale
fixture) and against the bar (catches a regression).

Four decks (`golden_taper_nec5.py`'s docstring has the exact radii/lengths):
  `ward`   Ward Harriman AE6TY's 20:1 tapered dipole (antenna-problem-decks
           issue #1) — the flagship, ten sections, linear radius ramp.
  `step2`  momwire#435's two-wire 10:1 radius STEP at a junction.
  `fat`    Ward's fattest section's radius, uniform — the Δ/a floor control.
  `thin`   Ward's thinnest section's radius, uniform — where every kernel
           agrees.

Every momwire row here is built DIRECTLY with `feed_arclength` at the exact
knot NEC-5's `EX` field-4 end code addresses (not through the deck path),
because NEC-2's `EX` drives a segment CENTRE and NEC-5's drives a KNOT — a
half-element-apart mismatch that would charge momwire up to 2.6 Ω for a
feed-placement difference with nothing to do with physics (study §1.4). The
residual measured here is formulation against formulation.

**Gate depth (D4, maintainer decision 2026-08-18).** N=400 on `ward` puts the
fat end at Δ/a ≈ 1.05, past where any reduced-kernel row means anything
(momwire#248). The standing gate stops at `GATE_MAX_N` — Δ/a ≈ 2.1 on the
ward/fat/thin fat end, and the full ladder on `step2` (its worst Δ/a at
N=400 is still ≈ 4.4, comfortably inside the floor) — and the finer,
recorded-not-gated rungs stay in the golden module for the next unit
(momwire#398 D1, giving razor EK) to build on.

Every one of these is `slow` (a couple to a dozen seconds for the module's
fixtures): the ladders reach N=400 total segments with the extended kernel
on, which house convention marks `slow` rather than folding into the
couple-thousand-test default tier.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import BSplineSolver, RazorSolver

from golden_taper_nec5 import GATE_MAX_N, KERNEL_ID_PATH_A, TAPER_LADDERS

pytestmark = pytest.mark.slow

FREQ_MHZ = 14.2
_C = 299792458.0
WL = _C / (FREQ_MHZ * 1e6)
Z_HEIGHT = 10.0  # cosmetic — every deck is free-space

# --------------------------------------------------------------------------
# the four decks, re-spelled locally (the NEC decks and the capture recipe
# live in scripts/capture_taper_nec5_lane.py, which this file does not
# import — same separation `tests/test_razor_pec_ground.py` keeps from its
# capture script).
# --------------------------------------------------------------------------
WARD_LEN = 10.51010
WARD_NSEC = 10
WARD_SEC = WARD_LEN / WARD_NSEC
WARD_RADII = [
    1.250000e-03,
    3.888889e-03,
    6.527778e-03,
    9.166667e-03,
    1.180556e-02,
    1.444444e-02,
    1.708333e-02,
    1.972222e-02,
    2.236111e-02,
    2.500000e-02,
]
WARD_FED_SEC = 6

STEP_LEN = 10.18946
STEP_RADII = [1.0e-2, 1.0262e-3]

FAT_RADIUS = 2.500000e-02
THIN_RADIUS = 1.250000e-03


def _ward_sections():
    return [(k * WARD_SEC, (k + 1) * WARD_SEC, WARD_RADII[k]) for k in range(WARD_NSEC)]


def _step_sections():
    h = STEP_LEN / 2.0
    return [(0.0, h, STEP_RADII[0]), (h, STEP_LEN, STEP_RADII[1])]


def _uniform_sections(radius, n_sec=10, length=WARD_LEN):
    s = length / n_sec
    return [(k * s, (k + 1) * s, radius) for k in range(n_sec)]


GEOMS = {
    "ward": dict(sections=_ward_sections(), fed=WARD_FED_SEC),
    "step2": dict(sections=_step_sections(), fed=1),
    "fat": dict(sections=_uniform_sections(FAT_RADIUS), fed=6),
    "thin": dict(sections=_uniform_sections(THIN_RADIUS), fed=6),
}


def _wires_of(sections):
    wires = [
        np.array([[x0, 0.0, Z_HEIGHT], [x1, 0.0, Z_HEIGHT]]) for x0, x1, _r in sections
    ]
    radii = [r for _a, _b, r in sections]
    junctions = [[(k, "end"), (k + 1, "start")] for k in range(len(wires) - 1)]
    return wires, radii, junctions


def _build(sections, fed_tag, n_per_sec, kind, ek):
    wires, radii, junctions = _wires_of(sections)
    fed = fed_tag - 1
    sec_len = float(np.linalg.norm(wires[fed][1] - wires[fed][0]))
    common = dict(
        wires=wires,
        n_per_edge_per_wire=[[n_per_sec]] * len(wires),
        wire_radius=radii,
        wavelength=WL,
        feed_wire_index=fed,
        feed_arclength=sec_len / 2.0,
    )
    if kind == "bs1":
        return BSplineSolver(
            degree=1, extended_kernel=ek, junctions=junctions, **common
        )
    if kind == "bs2":
        return BSplineSolver(
            degree=2, extended_kernel=ek, junctions=junctions, **common
        )
    if kind == "razor":
        return RazorSolver(**common)
    if kind == "razor-n5q":
        return RazorSolver(nec5_quadrature=True, **common)
    raise ValueError(kind)


def _z(name, n_per_sec, kind, ek=False):
    g = GEOMS[name]
    z, _ = _build(g["sections"], g["fed"], n_per_sec, kind, ek).compute_impedance()
    return complex(z)


def _gated_rows(name):
    """The TAPER_LADDERS rows for `name` at or below its D4 gate depth."""
    return [row for row in TAPER_LADDERS[name] if row[0] <= GATE_MAX_N[name]]


def _n_per_sec(name, n_total):
    return n_total // len(GEOMS[name]["sections"])


# --------------------------------------------------------------------------
# fixtures: live momwire, recomputed once per module, exactly the columns
# each gate needs
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def thin_twin_lane():
    """`razor` and `razor-nec5` (nec5_quadrature=True) on `thin`, every gated
    rung — the deck where razor's kernel claim actually holds (a/λ ≲ 5e-4,
    Δ/a ≳ 4, per the study's requalified domain)."""
    rows = {}
    for row in _gated_rows("thin"):
        n = row[0]
        npersec = _n_per_sec("thin", n)
        rows[n] = {
            "razor": _z("thin", npersec, "razor"),
            "razor-n5q": _z("thin", npersec, "razor-n5q"),
        }
    return rows


@pytest.fixture(scope="module")
def ek_decay_lane():
    """`bs1-ek` and `bs2-ek` on `ward`, `step2`, `fat` — every gated rung."""
    out = {}
    for name in ("ward", "step2", "fat"):
        rows = {}
        for row in _gated_rows(name):
            n = row[0]
            npersec = _n_per_sec(name, n)
            rows[n] = {
                "bs1-ek": _z(name, npersec, "bs1", ek=True),
                "bs2-ek": _z(name, npersec, "bs2", ek=True),
            }
        out[name] = rows
    return out


@pytest.fixture(scope="module")
def fat_reduced_lane():
    """`razor` and `bs1` (both reduced-kernel) on `fat` — every gated rung."""
    rows = {}
    for row in _gated_rows("fat"):
        n = row[0]
        npersec = _n_per_sec("fat", n)
        rows[n] = {
            "razor": _z("fat", npersec, "razor"),
            "bs1": _z("fat", npersec, "bs1"),
        }
    return rows


@pytest.fixture(scope="module")
def kernel_id_lane():
    """`bs1` (reduced), `bs1-ek` and `razor-n5q` at the three
    KERNEL_ID_PATH_A rungs with Δ/a ≤ 1.0 (N=420, 600, 840) — the study's
    §5 kernel-identification instrument, gated (see
    `test_kernel_identification_alpha_and_b_over_d`)."""
    rows = {}
    for row in KERNEL_ID_PATH_A:
        n, delta_over_a = row[0], row[1]
        # The N=420 rung's Delta/a is 1.0010, not exactly 1.0 (WARD_SEC/42
        # over a=25mm rounds to 1.001 — the study's own printed table shows
        # the same rung as "1.00" for the same reason). 1.005 cleanly
        # separates it from the next rung up (Delta/a=1.5014, N=280).
        if delta_over_a > 1.005:
            continue
        npersec = n // WARD_NSEC
        sections = _uniform_sections(FAT_RADIUS)
        rows[n] = {
            "bs1": complex(
                _build(sections, 6, npersec, "bs1", False).compute_impedance()[0]
            ),
            "bs1-ek": complex(
                _build(sections, 6, npersec, "bs1", True).compute_impedance()[0]
            ),
            "razor-n5q": complex(
                _build(sections, 6, npersec, "razor-n5q", False).compute_impedance()[0]
            ),
        }
    return rows


# --------------------------------------------------------------------------
# 1. the twin-lane bar (razor-nec5 on thin only — where the twin claim holds)
# --------------------------------------------------------------------------
def test_thin_twin_offset_is_constant_to_0_05_ohm(thin_twin_lane):
    """CLAIM: over the range where the two kernels agree (a/λ ≲ 5e-4,
    Δ/a ≳ 4 — `thin`, per the study's requalified razor domain), the
    (razor_n5q − NEC-5) residual is a CONSTANT offset to within 0.05 Ω down
    the whole gated ladder — the same bar `test_razor_pec_ground.py` gates
    razor's PEC sharp lane against. A residual that is constant in N is a
    shared kernel/quadrature nuance; one that walks would be a real
    discretization disagreement.
    """
    res = []
    for row in _gated_rows("thin"):
        n, z5 = row[0], row[1]
        z = thin_twin_lane[n]["razor-n5q"]
        assert abs(z - row[7]) < 5e-6, (
            f"thin N={n}: razor-n5q moved from the golden literal"
        )
        res.append(z - z5)
    spread_r = max(r.real for r in res) - min(r.real for r in res)
    spread_x = max(r.imag for r in res) - min(r.imag for r in res)
    assert spread_r <= 0.05, f"thin: dR spread {spread_r:.4f}"
    assert spread_x <= 0.05, f"thin: dX spread {spread_x:.4f}"


def test_thin_twin_agrees_with_nec5_at_every_rung(thin_twin_lane):
    """CLAIM: on top of the constancy bar, the twin's absolute agreement with
    NEC-5 on `thin` is tight in absolute terms too — the sharp-lane bar
    `test_razor_pec_ground.py` uses, |Z − Z_nec5| ≤ max(0.20, 0.25% |Z_nec5|),
    reused here rather than re-derived: it is the same claim on a free-space
    uniform wire, just tapered decks' thin control instead of the PEC
    ladder's geometries.
    """
    for row in _gated_rows("thin"):
        n, z5 = row[0], row[1]
        z = thin_twin_lane[n]["razor-n5q"]
        bar = max(0.20, 0.0025 * abs(z5))
        assert abs(z - z5) <= bar, f"thin N={n}: |dZ|={abs(z - z5):.4f} > {bar:.4f}"


def test_free_space_thin_wire_would_not_pass_the_bar_by_accident():
    """Guard against the bar being vacuous: a razor solve with NO taper
    context at all (this exact wire, but at a height that makes free space
    NOT match NEC-5's printed value) still needs the ground/height right —
    cheap sanity that the fixture is measuring something, not a tautology
    from two codes agreeing about a trivial thin straight wire regardless of
    setup."""
    row = _gated_rows("thin")[0]
    n = row[0]
    npersec = _n_per_sec("thin", n)
    # A materially shorter wire at the same radius: if the bar held for ANY
    # dipole regardless of length, this would too. It should not.
    off_wires = [np.array([[0.0, 0.0, Z_HEIGHT], [1.0, 0.0, Z_HEIGHT]])]
    z, _ = RazorSolver(
        wires=off_wires,
        nsegs=npersec * WARD_NSEC,
        wire_radius=THIN_RADIUS,
        wavelength=WL,
        nec5_quadrature=True,
    ).compute_impedance()
    bar = max(0.20, 0.0025 * abs(row[1]))
    assert abs(complex(z) - row[1]) > bar


# --------------------------------------------------------------------------
# 2. decay bars: bs1-ek / bs2-ek on ward, step2, fat
# --------------------------------------------------------------------------
# Finest GATED rung's measured |residual|, pinned +25% (the shape momwire#398
# unit 6 already ratified for razor's production lane: a decay claim plus a
# floor at the tightest point actually measured, not a tuned number).
_FINEST_BAR = {
    ("ward", "bs1-ek"): 0.6255 * 1.25,
    ("ward", "bs2-ek"): 1.348 * 1.25,
    ("step2", "bs1-ek"): 0.552 * 1.25,
    ("step2", "bs2-ek"): 1.656 * 1.25,
    ("fat", "bs1-ek"): 0.7045 * 1.25,
    ("fat", "bs2-ek"): 1.6892 * 1.25,
}


@pytest.mark.parametrize("name", ["ward", "step2", "fat"])
@pytest.mark.parametrize("label,col", [("bs1-ek", 3), ("bs2-ek", 5)])
def test_ek_residual_decays_down_the_ladder(name, label, col, ek_decay_lane):
    """CLAIM: `bspline-d1 + EK` and `bspline-d2 + EK` are the study's
    identified fat-section twin (§5) — the binary is EXTENDED/tubular
    everywhere, so an EK Galerkin row is the two-axis (basis + kernel) match.
    Its residual against NEC-5 must STRICTLY SHRINK down the gated ladder,
    finishing at or under the finest measured rung's value + 25 %. A
    residual that stopped shrinking, or grew, would mean the kernel
    identification (or the EK fill itself) had regressed.
    """
    rows = _gated_rows(name)
    key = "bs1-ek" if label == "bs1-ek" else "bs2-ek"
    res = []
    for row in rows:
        n = row[0]
        z = ek_decay_lane[name][n][key]
        assert abs(z - row[col]) < 5e-6, (
            f"{name} {label} N={n}: moved from golden literal"
        )
        res.append(abs(z - row[1]))
    for prev, nxt in zip(res, res[1:]):
        assert nxt < prev, f"{name} {label}: residual did not shrink ({res})"
    bar = _FINEST_BAR[(name, label)]
    assert res[-1] <= bar, (
        f"{name} {label}: finest residual {res[-1]:.4f} > bar {bar:.4f}"
    )


# --------------------------------------------------------------------------
# 3. turnaround guard: razor and bs1 on fat — envelope only, NOT decay
# --------------------------------------------------------------------------
# CLAIM: below Δ/a ≈ 2 a reduced kernel is answering the wrong physics
# (momwire#248, and now §5's kernel identification explains why on `fat`
# specifically). `razor` and `bs1` (reduced) are recorded here as
# NON-MONOTONE — measured minimum around Δ/a ≈ 2-3, rising on both sides —
# and pinned to an ENVELOPE only. A decay gate here would assert something
# false: the residual does not shrink monotonically even inside the gated
# range (razor's worst point in range is its COARSEST rung, N=20).
_FAT_ENVELOPE = {"razor": 1.270 * 1.25, "bs1": 6.134 * 1.25}


@pytest.mark.parametrize("label,col", [("razor", 6), ("bs1", 2)])
def test_fat_reduced_row_stays_in_envelope(label, col, fat_reduced_lane):
    """The GATED half of the claim: live-recomputed, anti-stale-checked,
    bounded by the envelope."""
    rows = _gated_rows("fat")
    res = []
    for row in rows:
        n = row[0]
        z = fat_reduced_lane[n][label]
        assert abs(z - row[col]) < 5e-6, f"fat {label} N={n}: moved from golden literal"
        res.append(abs(z - row[1]))
    bar = _FAT_ENVELOPE[label]
    assert max(res) <= bar, (
        f"fat {label}: envelope exceeded ({max(res):.4f} > {bar:.4f})"
    )


@pytest.mark.parametrize("label,col", [("razor", 6), ("bs1", 2)])
def test_fat_reduced_row_is_non_monotone_over_the_full_recorded_ladder(label, col):
    """The reason this row gets an envelope and not a decay bar: OVER THE
    FULL RECORDED LADDER (including the ungated N=280/400 rungs — golden
    literals only, no live recompute needed for arithmetic on already-gated
    data), the residual is not monotone. `razor` dips then rises within the
    range this file's live gate covers (min at N=60, rising again by N=200);
    `bs1` keeps falling through N=200 and only turns at N=280 — so a decay
    claim restricted to N<=200 would look true today and go false the
    moment a user (or a wider gate) refines past it. That is exactly the
    shape momwire#248's Δ/a floor predicts and §5's kernel identification
    explains: a reduced kernel is answering the wrong physics on fat wire,
    and refining the mesh cannot fix wrong physics.

    If a future change makes this row decay monotonically over the FULL
    ladder too, that would be excellent news (an EK-for-razor unit,
    momwire#398 D1, or a new formulation) — replace this test with a decay
    gate deliberately at that point rather than silently outgrowing it.
    """
    res = [abs(row[col] - row[1]) for row in TAPER_LADDERS["fat"]]
    increases = sum(1 for a, b in zip(res, res[1:]) if b > a)
    decreases = sum(1 for a, b in zip(res, res[1:]) if b < a)
    assert increases >= 1 and decreases >= 1, (
        f"fat {label}: now monotone over the full ladder ({res}) — replace "
        "this envelope test with a decay gate deliberately, see momwire#398 D1"
    )


# --------------------------------------------------------------------------
# 4. the kernel-identification ladder (study §5), gated as an identification
# --------------------------------------------------------------------------
def test_kernel_identification_alpha_and_b_over_d(kernel_id_lane):
    """CLAIM: the NEC-5 binary is extended/tubular-kernel EVERYWHERE, not
    reduced — identified, not assumed, from printed output alone (study §5).
    `D = Z_bs1_ek - Z_bs1_reduced` is a pure kernel difference (same basis,
    testing, mesh, quadrature, feed on both sides); `alpha = Re(A·conj(D))/
    |D|²` with `A = Z_nec5 - Z_bs1_reduced` reads which row the binary
    tracks: alpha → 1 means extended, alpha → 0 means reduced.

    Gated at every rung with Δ/a ≤ 1.0 (N=420, 600, 840 on the fixed-a,
    refine-N path — the study's "path A"): `alpha ≥ 0.9` AND
    `|B| ≤ 0.35·|D|` where `B = Z_nec5 - Z_bs1_ek`. This is the ONE gate in
    the unit that protects a claim about the REFERENCE rather than about
    momwire — a regression here would mean either `extended_kernel` quietly
    changed, or the captured NEC-5 printouts did, and either is worth
    catching loudly rather than in a user's deck (see
    `docs/design/solver-architecture.md` §6.12).

    Path B (fix N, fatten a) is the study's second, independent path to the
    same conclusion; it is recorded not captured here (see
    `scripts/capture_taper_nec5_lane.py`'s docstring) and is a defensible
    next increment, not required to close this gate.
    """
    assert len(kernel_id_lane) == 3, "expected exactly the 3 rungs with Delta/a <= 1.0"
    for n, row in kernel_id_lane.items():
        golden = next(r for r in KERNEL_ID_PATH_A if r[0] == n)
        _, delta_over_a, z5, z_bs1, z_bs1_ek, z_n5q = golden
        assert abs(row["bs1"] - z_bs1) < 5e-6, f"kernA N={n}: bs1 moved"
        assert abs(row["bs1-ek"] - z_bs1_ek) < 5e-6, f"kernA N={n}: bs1-ek moved"
        assert abs(row["razor-n5q"] - z_n5q) < 5e-6, f"kernA N={n}: razor-n5q moved"

        d = row["bs1-ek"] - row["bs1"]
        a = z5 - row["bs1"]
        b = z5 - row["bs1-ek"]
        alpha = (a.real * d.real + a.imag * d.imag) / (abs(d) ** 2)
        assert alpha >= 0.9, (
            f"kernA N={n} (D/a={delta_over_a}): alpha={alpha:.3f} < 0.9"
        )
        assert abs(b) <= 0.35 * abs(d), (
            f"kernA N={n} (D/a={delta_over_a}): |B|/|D|={abs(b) / abs(d):.3f} > 0.35"
        )


def test_kernel_identification_gate_actually_discriminates():
    """Guard against the alpha/|B|/|D| gate being vacuous: at the same rung
    (N=840, Δ/a=0.5), does the REDUCED row (`bs1`, not `bs1-ek`) fail the
    bound the gate applies to the EK row? If reduced also cleared
    `|Z_nec5 - Z_bs1| ≤ 0.35·|D|`, the gate would not be discriminating
    between "the binary is reduced" and "the binary is extended" at all —
    both hypotheses would pass and the identification would be worthless.
    Measured: |A| = |Z_nec5 - Z_bs1_reduced| is 46.5 Ω at N=840, over 3× the
    35 % bound (14.97 Ω) — the reduced row fails hard where the EK row
    passes comfortably (§5.2's own point: NEC-5's column stays with the EK
    row while the reduced row diverges).
    """
    n840 = next(r for r in KERNEL_ID_PATH_A if r[0] == 840)
    _, _, z5, z_bs1, z_bs1_ek, _ = n840
    d = z_bs1_ek - z_bs1
    a_reduced = z5 - z_bs1  # the "binary tracks reduced" hypothesis's own B
    assert abs(a_reduced) > 0.35 * abs(d), (
        "the reduced row would ALSO clear the bound — the gate would not "
        "discriminate reduced from extended"
    )


# --------------------------------------------------------------------------
# G-449 — the fed-knot C0 spelling un-stalls bs2's deep tail (momwire#449)
# --------------------------------------------------------------------------


def _ward_split_fed(n_per_sec):
    """The ward deck with LOCAL C0 at the fed knot and nothing else
    changed: the fed section split AT the feed into two half-wires, the
    new 2-member junction declared, and the drive a series node gap —
    momwire#305's third spelling, which is also the shape NEC-5's own
    ``EX`` end code writes, so this row is feed-matched to the anchor.
    ``feeds=[]`` is explicit and load-bearing: omitting it engages the
    legacy default feed and silently double-drives the deck."""
    wires, radii, junctions = _wires_of(_ward_sections())
    fed = WARD_FED_SEC - 1
    a, b = wires[fed][0], wires[fed][1]
    mid = 0.5 * (a + b)
    new_wires = (
        list(wires[:fed])
        + [np.stack([a, mid]), np.stack([mid, b])]
        + list(wires[fed + 1 :])
    )
    new_radii = list(radii[:fed]) + [radii[fed]] * 2 + list(radii[fed + 1 :])
    new_junctions = []
    for j in junctions:
        grp = []
        for w, end in j:
            if w == fed:
                grp.append((fed, end) if end == "start" else (fed + 1, end))
            else:
                grp.append((w if w < fed else w + 1, end))
        new_junctions.append(grp)
    new_junctions.append([(fed, "end"), (fed + 1, "start")])
    half = n_per_sec // 2
    npe = (
        [[n_per_sec]] * fed
        + [[half], [half]]
        + [[n_per_sec]] * (len(wires) - fed - 1)
    )
    return BSplineSolver(
        degree=2,
        extended_kernel=True,
        wires=new_wires,
        n_per_edge_per_wire=npe,
        wire_radius=new_radii,
        wavelength=WL,
        feeds=[],
        junctions=new_junctions,
        node_gaps=[(fed, "end", 1.0 + 0j)],
    )


def test_g449_fed_knot_c0_unstalls_the_bs2_tail():
    """momwire#449's delivered attribution, held as a standing gate: bs2's
    ward-ladder stall at ~0.5 Ω was the C1 constraint at the FED KNOT —
    the deck's only C1-constrained non-smooth feature (the nine section
    boundaries are separate wires + junctions, C0 by construction, and
    every rung feeds at a mesh knot already). Giving the basis slope
    freedom at that one knot converges the tail through the stall:
    measured 0.047 Ω at N=200 against NEC-5's Richardson anchor, where
    the banked matched-feed bs2-ek row sits at 0.532 and drifting. The
    0.15 bar is 3× the measured value and 3.5× under the stall — a miss
    here means either the node-gap drive or the junction slope freedom
    regressed. Knot multiplicity (per-knot C0 inside one wire) stays the
    RECORDED escalation if a second consumer appears; this spelling is
    the served answer (2026-08-28 maintainer decision on #449)."""
    rows = {r[0]: r for r in TAPER_LADDERS["ward"]}
    ns = sorted(rows)
    z1, z2 = rows[ns[-2]][1], rows[ns[-1]][1]
    anchor = z2 + (z2 - z1) / (ns[-1] / ns[-2] - 1.0)
    z, _ = _ward_split_fed(200 // WARD_NSEC).compute_impedance()
    z = complex(np.atleast_1d(z)[0])
    d = abs(z - anchor)
    stalled = abs(rows[200][5] - anchor)
    assert d <= 0.15, (
        f"split-fed bs2-ek answers {z:.4f} at N=200, {d:.4f} ohm from the "
        f"NEC-5 anchor {anchor:.4f} (measured 0.047; the stalled banked row "
        f"sits at {stalled:.4f}) — the fed-knot C0 spelling has regressed"
    )
    assert d < 0.5 * stalled, (
        "the split-fed row no longer beats the stalled banked row — the "
        "un-stall itself has regressed"
    )
