"""Capture the taper agreement-floor lane against the licensed NEC-5 binary.

momwire#398's taper-readiness study, unit 1 ("the taper agreement floor").
Runs the study's four decks — `ward` (Ward Harriman AE6TY's 20:1 tapered
dipole, antenna-problem-decks issue #1), `step2` (momwire#435's two-wire
radius step), `fat` and `thin` (Ward's fattest/thinnest section, uniform) —
through the NEC-5 binary and through momwire's rows, then regenerates
`tests/golden_taper_nec5.py` as pure literals so the momwire test suite
needs neither the binary nor antennaknobs.

Must run under the antennaknobs venv (that is where the NEC-5 engine
wrapper lives), with the binary on `NEC5_EXE`:

    NEC5_EXE=~/antennas/NEC5-downloads/nec5-linux/nec5cl \
        /home/smburns/antennas/antennaknobs/.venv/bin/python \
        scripts/capture_taper_nec5_lane.py

Only the binary's PRINTED impedances are recorded; nothing about NEC-5's
internals is read, quoted or inferred here. NEC-5 printouts are End-User
Reports, LLNL-CODE-746721.

Unlike `capture_razor_pec_nec5_lane.py`, which builds decks the engine's own
`deck()` convention already spells, this lane needs NEC-5's `EX` field-4 END
CODE addressed at an exact knot (see "the EX trap" below) — the same raw-deck
escape hatch `NEC5Engine.run_deck()` offers, just with hand-built card text
instead of the engine's own emitter. `nec2c` (/usr/bin/nec2c, open source,
quoted freely) is invoked directly by subprocess, matching this tree's
existing house choice to PIN a one-time oracle run as golden literals rather
than depend on the binary at test time (see `tests/test_extended_kernel.py`'s
"ORACLE (nec2c 1.3.1 ...)" comment) — nec2c is cheap enough to shell out to
live, but doing so would still make `tests/golden_taper_nec5.py`'s import
alone require a binary that CI does not carry, so it is captured here too.

Three ladders, three methods — the taper-readiness study's own method
separation, preserved deliberately rather than collapsed into one table:

1. **The matched-feed taper ladder** (`ward`, `step2`, `fat`, `thin`).
   NEC-2's `EX` drives a SEGMENT CENTRE; NEC-5's `EX` field 4 is an END CODE
   and drives a KNOT. On a shared mesh those are different points, half an
   element apart — comparing deck-for-deck would charge momwire up to 2.6 Ω
   for a feed-placement difference with nothing to do with physics (study
   §1.4). So momwire's rows here are built DIRECTLY with `feed_arclength` at
   the exact knot NEC-5's `EX` addresses (the midpoint of the fed section,
   which is a mesh knot whenever the per-section segment count is even —
   every rung below is even for exactly that reason), and NEC-5's deck uses
   its own end-code spelling. The residual is formulation against
   formulation, not feed-placement against feed-placement.

2. **The sin/nec2c identification ladder** (`step2`, `thin` — momwire#435,
   maintainer decision D3). `sin` and `nec2c` ride the SAME NEC-2 deck, so
   this pairing is feed-matched by construction — an ODD per-section segment
   count puts NEC-2's `EX` centre-segment feed exactly at the section
   midpoint, and no placement term enters at all.

3. **The kernel-identification ladder** (study §5, path A only — a fixed
   a = 25 mm, uniform 10-section dipole matching `fat`'s radius, refined from
   N=40 to N=840 so Δ/a crosses from ~10 down to 0.5). Path B (fix N, fatten
   a) is the study's second, independent path to the same conclusion; it is
   not captured here — recorded not-gated is a defensible next increment,
   and path A alone already clears the D3/gate thresholds this unit ships
   (see the golden module's docstring for the reproduction recipe if path B
   is ever wanted).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "tests" / "golden_taper_nec5.py"
sys.path.insert(0, str(REPO / "src"))

FREQ_MHZ = 14.2
C = 299792458.0
WL = C / (FREQ_MHZ * 1e6)

CAPTURES = Path(
    __import__("os").environ.get("TAPER_NEC5_CAPTURES", "/tmp/taper-nec5-captures")
)

# --------------------------------------------------------------------------
# geometries — the study's four decks, verbatim (taper-readiness.md §2.1)
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
WARD_FED_SEC = 6  # 1-based tag of the fed section

STEP_LEN = 10.18946
STEP_RADII = [1.0e-2, 1.0262e-3]  # inner (fed) half fat, outer half thin

FAT_RADIUS = 2.500000e-02  # Ward's fattest section
THIN_RADIUS = 1.250000e-03  # Ward's thinnest section

Z_HEIGHT = 10.0  # all decks free-space; z is cosmetic (kept from the study)


def ward_sections():
    return [(k * WARD_SEC, (k + 1) * WARD_SEC, WARD_RADII[k]) for k in range(WARD_NSEC)]


def step_sections():
    h = STEP_LEN / 2.0
    return [(0.0, h, STEP_RADII[0]), (h, STEP_LEN, STEP_RADII[1])]


def uniform_sections(radius, n_sec=10, length=WARD_LEN):
    s = length / n_sec
    return [(k * s, (k + 1) * s, radius) for k in range(n_sec)]


GEOMS = {
    "ward": dict(sections=ward_sections(), fed=WARD_FED_SEC),
    "step2": dict(sections=step_sections(), fed=1),
    "fat": dict(sections=uniform_sections(FAT_RADIUS), fed=6),
    "thin": dict(sections=uniform_sections(THIN_RADIUS), fed=6),
}

# Ladders, in PER-SECTION segment count. Every entry is EVEN (matched-feed
# ladder needs a centre knot). N_total = n_per_sec * n_sections.
LADDER_PER_SEC = {
    "ward": (2, 4, 6, 8, 10, 14, 20, 28, 40),
    "fat": (2, 4, 6, 8, 10, 14, 20, 28, 40),
    "thin": (2, 4, 6, 8, 10, 14, 20, 28, 40),
    "step2": (12, 24, 36, 48, 72, 96, 140, 200),
}
# D4 (maintainer decision, 2026-08-18): gate the standing suite to N_total
# such that the ward fat end sits at Delta/a >= 2.1 (N_total <= 200); record
# the finer rungs too, ungated. GATE_MAX is a per-section count, so it reads
# directly against LADDER_PER_SEC.
GATE_MAX_PER_SEC = {"ward": 20, "fat": 20, "thin": 20, "step2": 200}

# The sin/nec2c identification ladder (D3): ODD per-section counts, deck-path
# only, on step2 and thin.
SIN_LADDER_PER_SEC = {
    "step2": (13, 25, 37, 49, 73, 97, 141, 201),
    "thin": (3, 5, 7, 9, 11, 15, 21, 29, 41),
}

# Kernel-identification ladder (study §5, path A): a = 25 mm (== `fat`'s
# radius), 10-section uniform dipole, N_total 40 -> 840.
KERNEL_A_RADIUS = FAT_RADIUS
KERNEL_A_PER_SEC = (4, 6, 10, 14, 20, 28, 42, 60, 84)


# --------------------------------------------------------------------------
# deck text, in both spellings
# --------------------------------------------------------------------------
def _gw(tag, n, x0, x1, z, rad):
    return (
        f"GW {tag} {n} {x0:.6E} 0.000000E+00 {z:.6E} "
        f"{x1:.6E} 0.000000E+00 {z:.6E} {rad:.6E}"
    )


def nec2_deck(sections, fed_tag, n_per_sec, *, ek: bool):
    """NEC-2 spelling. `EX` field 4 is NEC-2's flag (0), driving the CENTRE
    of the fed segment; an ODD `n_per_sec` puts that centre exactly at the
    fed section's midpoint."""
    lines = [f"CM taper study n={n_per_sec}", "CE"]
    for k, (x0, x1, rad) in enumerate(sections):
        lines.append(_gw(k + 1, n_per_sec, x0, x1, Z_HEIGHT, rad))
    lines.append("GE 0")
    if ek:
        lines.append("EK")
    seg = (n_per_sec + 1) // 2
    lines.append(f"EX 0 {fed_tag} {seg} 0 1.000000E+00 0.000000E+00")
    lines.append(f"FR 0 1 0 0 {FREQ_MHZ:.6E} 0.000000E+00")
    lines.append("XQ 0")
    lines.append("EN")
    return "\n".join(lines) + "\n"


def nec5_deck(sections, fed_tag, n_per_sec):
    """NEC-5 spelling: no `EK` card (the binary rejects it after the
    geometry section), and `EX` field 4 is the END CODE naming which KNOT
    hosts the source. An EVEN `n_per_sec` puts a knot at the fed section's
    exact midpoint."""
    if n_per_sec % 2 != 0:
        raise ValueError(
            f"n_per_sec must be even for a matched-feed knot, got {n_per_sec}"
        )
    lines = [f"CM taper study n={n_per_sec}", "CE"]
    for k, (x0, x1, rad) in enumerate(sections):
        lines.append(_gw(k + 1, n_per_sec, x0, x1, Z_HEIGHT, rad))
    lines.append("GE 0")
    seg, end = n_per_sec // 2, 2
    lines.append(f"EX 0 {fed_tag} {seg} {end} 1.000000E+00 0.000000E+00")
    lines.append(f"FR 0 1 0 0 {FREQ_MHZ:.6E} 0.000000E+00")
    lines.append("XQ 0")
    lines.append("EN")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# oracles
# --------------------------------------------------------------------------
_ENGINE = None


def _engine():
    global _ENGINE
    if _ENGINE is None:
        from antennaknobs.engines.nec5 import NEC5Engine

        sys.path.insert(0, str(Path.home() / "antennas/antennaknobs/scripts"))
        from bench_nec5_walk_why import make_dipole

        CAPTURES.mkdir(parents=True, exist_ok=True)
        _ENGINE = NEC5Engine(make_dipole(20), ground=None, capture_dir=CAPTURES)
    return _ENGINE


def nec5_z(deck: str) -> complex:
    rows = _engine().run_deck(deck)
    return complex(rows[0][0][2])


def _isnum(s):
    try:
        float(s)
    except ValueError:
        return False
    return True


def nec2c_z(deck: str) -> complex:
    with tempfile.TemporaryDirectory() as d:
        i, o = Path(d) / "in.nec", Path(d) / "out.txt"
        i.write_text(deck)
        subprocess.run(
            ["/usr/bin/nec2c", "-i", str(i), "-o", str(o)],
            capture_output=True,
            text=True,
            timeout=900,
        )
        text = o.read_text(errors="replace") if o.exists() else ""
    k = text.find("ANTENNA INPUT PARAMETERS")
    if k < 0:
        raise RuntimeError("nec2c printed no input parameters")
    for line in text[k:].splitlines()[1:8]:
        t = line.split()
        if len(t) >= 8 and all(_isnum(x) for x in t[:8]):
            return complex(float(t[6]), float(t[7]))
    raise RuntimeError("nec2c input-parameters block has no data row")


# --------------------------------------------------------------------------
# momwire rows — matched-feed lane (built directly, not through the deck path)
# --------------------------------------------------------------------------
def wires_of(sections):
    wires = [
        np.array([[x0, 0.0, Z_HEIGHT], [x1, 0.0, Z_HEIGHT]]) for x0, x1, _r in sections
    ]
    radii = [r for _a, _b, r in sections]
    junctions = [[(k, "end"), (k + 1, "start")] for k in range(len(wires) - 1)]
    return wires, radii, junctions


def build(name, sections, fed_tag, n_per_sec, kind, ek):
    from momwire import BSplineSolver, RazorSolver

    wires, radii, junctions = wires_of(sections)
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


MATCHED_ROWS = [
    ("bs1", "bs1", False),
    ("bs1-ek", "bs1", True),
    ("bs2", "bs2", False),
    ("bs2-ek", "bs2", True),
    ("razor", "razor", False),
    ("razor-n5q", "razor-n5q", False),
]


def matched_rung(name, n_per_sec):
    g = GEOMS[name]
    deck5 = nec5_deck(g["sections"], g["fed"], n_per_sec)
    z5 = nec5_z(deck5)
    out = {"z5": z5}
    for label, kind, ek in MATCHED_ROWS:
        z, _ = build(
            name, g["sections"], g["fed"], n_per_sec, kind, ek
        ).compute_impedance()
        out[label] = complex(z)
    return out


# --------------------------------------------------------------------------
# sin/nec2c identification lane — deck path, odd n_per_sec
# --------------------------------------------------------------------------
def sin_rung(name, n_per_sec):
    from momwire.deck import build_solver, parse

    g = GEOMS[name]
    d2 = nec2_deck(g["sections"], g["fed"], n_per_sec, ek=False)
    model = parse(d2, dialect="nec2")
    built = build_solver(model, basis="sinusoidal", extended_kernel=False)
    ps = built.solver.compute_port_solution()
    port = built.ports.feed_ports[0]
    z_sin = complex(1.0 / ps.y[port, port])
    z_n2c = nec2c_z(d2)
    return z_sin, z_n2c


# --------------------------------------------------------------------------
# kernel-identification lane — path A
# --------------------------------------------------------------------------
def kernel_a_rung(n_per_sec):
    sections = uniform_sections(KERNEL_A_RADIUS)
    fed_tag = 6
    delta = WARD_SEC / n_per_sec
    delta_over_a = delta / KERNEL_A_RADIUS
    deck5 = nec5_deck(sections, fed_tag, n_per_sec)
    z5 = nec5_z(deck5)
    z_bs1, _ = build(
        "kernA", sections, fed_tag, n_per_sec, "bs1", False
    ).compute_impedance()
    z_bs1_ek, _ = build(
        "kernA", sections, fed_tag, n_per_sec, "bs1", True
    ).compute_impedance()
    z_n5q, _ = build(
        "kernA", sections, fed_tag, n_per_sec, "razor-n5q", False
    ).compute_impedance()
    return dict(
        delta_over_a=delta_over_a,
        z5=z5,
        bs1=complex(z_bs1),
        bs1_ek=complex(z_bs1_ek),
        razor_n5q=complex(z_n5q),
    )


# --------------------------------------------------------------------------
def main() -> None:
    matched = {}
    for name in GEOMS:
        rows = []
        for n in LADDER_PER_SEC[name]:
            ntot = n * len(GEOMS[name]["sections"])
            t0 = time.time()
            rec = matched_rung(name, n)
            rows.append((ntot, rec))
            print(
                f"{name:>6} N={ntot:<4} nec5={rec['z5']:.4f} ({time.time() - t0:.1f}s)",
                flush=True,
            )
        matched[name] = rows

    sin_lane = {}
    for name, ladder in SIN_LADDER_PER_SEC.items():
        rows = []
        for n in ladder:
            ntot = n * len(GEOMS[name]["sections"])
            z_sin, z_n2c = sin_rung(name, n)
            rows.append((ntot, z_sin, z_n2c))
            print(
                f"sin/{name:>6} N={ntot:<4} sin={z_sin:.4f} nec2c={z_n2c:.4f}",
                flush=True,
            )
        sin_lane[name] = rows

    kernel_a = []
    for n in KERNEL_A_PER_SEC:
        ntot = n * WARD_NSEC
        rec = kernel_a_rung(n)
        kernel_a.append((ntot, rec))
        print(
            f"kernA N={ntot:<4} D/a={rec['delta_over_a']:.2f} nec5={rec['z5']:.4f}",
            flush=True,
        )

    _write_golden(matched, sin_lane, kernel_a)


def _lit(z, places=4):
    sign = "-" if z.imag < 0 else "+"
    return f"{z.real:.{places}f} {sign} {abs(z.imag):.{places}f}j"


def _write_golden(matched, sin_lane, kernel_a) -> None:
    lines = [
        '"""NEC-5 (and nec2c) printed impedances for the taper agreement floor.',
        "",
        "GENERATED by scripts/capture_taper_nec5_lane.py — do not edit by hand.",
        "momwire#398 (taper-readiness study), unit 1. See that script for the",
        "decks, the matched-feed method and the run recipe. Citations: NEC-5",
        "(LLNL-CODE-746721), printed output only; nec2c 1.3.1 (open source,",
        "/usr/bin/nec2c), quoted freely. Frequency 14.2 MHz, free space.",
        "",
        "TAPER_LADDERS[name] is a tuple of rows",
        "    (n_total, Z_nec5, Z_bs1, Z_bs1_ek, Z_bs2, Z_bs2_ek, Z_razor, Z_razor_n5q)",
        "for name in ('ward', 'step2', 'fat', 'thin'), momwire's rows built",
        "DIRECTLY with feed_arclength at NEC-5's exact fed knot (the EX-trap",
        "matched-feed construction, study Sec 1.4). GATE_MAX_N[name] is the",
        "largest n_total the standing suite gates (D4: Delta/a >= 2.1 on the",
        "ward fat end); rungs above it are recorded, not gated.",
        "",
        "SIN_NEC2C_LADDERS[name] (name in 'step2', 'thin') is a tuple of rows",
        "    (n_total, Z_sin, Z_nec2c)",
        "riding the SAME NEC-2 deck (odd n_per_sec, centre-segment feed) — the",
        "momwire#435 / D3 identification lane, feed-matched by construction.",
        "",
        "KERNEL_ID_PATH_A is a tuple of rows",
        "    (n_total, delta_over_a, Z_nec5, Z_bs1, Z_bs1_ek, Z_razor_n5q)",
        "the study's Sec 5 kernel-identification ladder: a = 25 mm fixed",
        "(Ward's fattest section), N refined 40 -> 840 so Delta/a crosses from",
        "~10 down to 0.5.",
        '"""',
        "",
    ]
    lines.append("GATE_MAX_N = {")
    for name in GEOMS:
        gate_ntot = GATE_MAX_PER_SEC[name] * len(GEOMS[name]["sections"])
        lines.append(f'    "{name}": {gate_ntot},')
    lines.append("}")
    lines.append("")
    # NEC-5/nec2c columns: 4 places, matching the binaries' own printed
    # precision (no spurious digits past what was actually printed).
    # momwire's own recomputed columns: 6 places, so the golden module's
    # anti-stale check (`abs(live - recorded) < 5e-6`, the same idiom
    # `golden_razor_pec_nec5.py` uses) has headroom against literal
    # rounding — 4 places alone would leave up to ~7e-5 of rounding noise,
    # which is bigger than the check.
    lines.append("TAPER_LADDERS = {")
    for name, rows in matched.items():
        lines.append(f'    "{name}": (')
        for ntot, rec in rows:
            lines.append(
                f"        ({ntot}, {_lit(rec['z5'], 4)}, {_lit(rec['bs1'], 6)}, "
                f"{_lit(rec['bs1-ek'], 6)}, {_lit(rec['bs2'], 6)}, "
                f"{_lit(rec['bs2-ek'], 6)}, {_lit(rec['razor'], 6)}, "
                f"{_lit(rec['razor-n5q'], 6)}),"
            )
        lines.append("    ),")
    lines.append("}")
    lines.append("")
    lines.append("SIN_NEC2C_LADDERS = {")
    for name, rows in sin_lane.items():
        lines.append(f'    "{name}": (')
        for ntot, z_sin, z_n2c in rows:
            lines.append(f"        ({ntot}, {_lit(z_sin, 6)}, {_lit(z_n2c, 4)}),")
        lines.append("    ),")
    lines.append("}")
    lines.append("")
    lines.append("KERNEL_ID_PATH_A = (")
    for ntot, rec in kernel_a:
        lines.append(
            f"    ({ntot}, {rec['delta_over_a']:.4f}, {_lit(rec['z5'], 4)}, "
            f"{_lit(rec['bs1'], 6)}, {_lit(rec['bs1_ek'], 6)}, "
            f"{_lit(rec['razor_n5q'], 6)}),"
        )
    lines.append(")")
    GOLDEN.write_text("\n".join(lines) + "\n")
    # Unlike capture_razor_pec_nec5_lane.py's short 4-column rows, these rows
    # (7-8 columns at 6-decimal-place precision) run past ruff's line length
    # even with spaced operators, so `ruff format` explodes them into one
    # entry per line. Run it here rather than hand-format the writer's
    # output, so the file `ruff format` produces IS what this script writes
    # — a re-capture shows up as a content diff only, never a formatting
    # one, and running the script twice is byte-identical (both runs invoke
    # the same formatter on the same content).
    subprocess.run(["ruff", "format", str(GOLDEN)], check=True)
    print(f"\nwrote {GOLDEN}")


if __name__ == "__main__":
    main()
