"""Capture golden PyNEC impedances for the reflection-coefficient ground work.

Runs the validation matrix from docs/refl-coef-ground-plan.md against PyNEC
and regenerates tests/golden_refl_coef_ground.py as pure literals, so the
momwire test suite needs no PyNEC / antennaknobs dependency.

Must run under the antennaknobs venv (PyNEC + antennaknobs installed):

    /home/smburns/antennas/antennaknobs/.venv/bin/python \
        scripts/capture_refl_coef_ground_golden.py

Cases:
  - flat half-wave dipole (antennaknobs dipoles.invvee variant "dipole",
    28.47 MHz) at heights 0.02 / 0.05 / 0.1 / 0.2 / 0.35 / 0.5 wavelength;
  - inverted-L (verticals.inverted_l defaults, 28.57 MHz) with its radial
    counterpoise base at the same height fractions;
  - 6-element flat yagi (beams.yagi, n_directors=4, 28.47 MHz) at 0.2
    wavelength only — its > 1.2-wavelength boom drives image-ray distances
    past the 1-wavelength edge of NEC's Sommerfeld interpolation grid
    (docs/sommerfeld-ground-plan.md Phase 0);
  - ground constants (eps_r, sigma): (10, 0.002), (13, 0.005), (3, 0.001).

The 0.02 heights and the yagi case exist for the Sommerfeld work: below
~0.1 wavelength gn 0 and gn 2 diverge hard, so gn 2 ("finite") is the
oracle there, not gn 0.

gn 2 ORACLE = nec2c, NOT PyNEC (measured 2026-07-06). Two independent
problems with PyNEC's (nec2++'s) Sommerfeld solve:
  1. Cross-solve state: the SAME gn 2 case solved twice in one process
     returns two different impedances (62.791-2.173j then 62.308-2.532j
     for dipole 0.05wl/(10, 0.002)); a preceding 0.02wl solve shifts it
     by 17 ohms. Only the first Sommerfeld solve of a process is
     meaningful.
  2. Low-height breakage: even first-in-process, PyNEC gn 2 at 0.02wl is
     erratic across similar grounds (572-227j / 216-783j / 78+43j for
     the three matrix grounds) where nec2c varies smoothly (95+32j /
     91+30j / 93+31j) and is segment-refinement stable (0.8 ohm at 2x).
Control experiments: identical exported decks give nec2c-vs-PyNEC
agreement to 0.02 ohm on gn 0 and PEC at all heights (deck translation
faithful), and to 0.02 ohm on gn 2 at 0.1-0.5wl (the two Sommerfeld
codes agree where both are healthy) — the split is below 0.1wl only,
and there nec2c (a direct C translation of the public-domain NEC-2
Fortran, whose SOMNEC explicitly supports wires close to the interface)
is the trustworthy one. So "finite" values are captured by exporting
each case as a NEC deck (antennaknobs.nec_export) and running the nec2c
CLI — one fresh process per case as a bonus. gn 0 and PEC values stay
on PyNEC (order-stable, reproduced bit-exactly across runs).

Per case we record three PyNEC grounds:
  - "finite-fast": gn 0, reflection-coefficient approximation — the oracle
    the momwire implementation is judged against;
  - "finite": gn 2, Sommerfeld-Norton — secondary sanity, confirms we
    inherit gn 0's known low-height divergence rather than inventing our own;
  - "pec": PEC ground — the model momwire's grounded solve implements today,
    i.e. the error being fixed.
"""

import os
import sys

C_LIGHT = 299.792458  # MHz * m

HEIGHT_FRACS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5)
YAGI_HEIGHT_FRACS = (0.2,)
GROUNDS = ((10.0, 0.002), (13.0, 0.005), (3.0, 0.001))
GROUND_MODELS = ("finite-fast", "finite", "pec")

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests",
    "golden_refl_coef_ground.py",
)


def _builder(name, frac):
    """(design_freq_MHz, builder) for a matrix geometry at height frac."""
    from antennaknobs import resolve_variant_params
    from antennaknobs.designs.beams.yagi import Builder as Yagi
    from antennaknobs.designs.dipoles.invvee import Builder as InvVee
    from antennaknobs.designs.verticals.inverted_l import Builder as InvertedL

    if name == "dipole":
        cls, params = InvVee, dict(resolve_variant_params(InvVee, "dipole"))
    elif name == "inverted_l":
        cls, params = InvertedL, dict(InvertedL.default_params)
    elif name == "yagi":
        cls, params = Yagi, dict(Yagi.default_params)
        params["n_directors"] = 4
    else:
        raise ValueError(name)
    lam = C_LIGHT / params["design_freq"]
    params["base"] = frac * lam
    return params["design_freq"], cls(params)


def _cases():
    for name, fracs in (
        ("dipole", HEIGHT_FRACS),
        ("inverted_l", HEIGHT_FRACS),
        ("yagi", YAGI_HEIGHT_FRACS),
    ):
        for frac in fracs:
            freq_mhz, builder = _builder(name, frac)
            yield (name, frac, freq_mhz, builder)


def _solve_finite_nec2c(builder, eps_r, sigma):
    """One gn 2 solve through the nec2c CLI (see module docstring for why
    this is the gn 2 oracle instead of PyNEC)."""
    import subprocess
    import tempfile
    from pathlib import Path

    from antennaknobs.nec_export import export_nec

    deck = export_nec(builder, ground=("finite", eps_r, sigma), include_rp=False)
    with tempfile.TemporaryDirectory() as d:
        nec = Path(d) / "deck.nec"
        out = Path(d) / "deck.out"
        nec.write_text(deck)
        subprocess.run(
            ["nec2c", "-i", str(nec), "-o", str(out)],
            check=True,
            capture_output=True,
        )
        lines = out.read_text().splitlines()
    for i, ln in enumerate(lines):
        if "ANTENNA INPUT PARAMETERS" in ln:
            j = i + 3
            while j < len(lines) and lines[j].strip():
                toks = lines[j].split()
                if len(toks) >= 8:
                    return complex(float(toks[6]), float(toks[7]))
                j += 1
    raise RuntimeError("no impedance in nec2c output")


def main():
    import shutil

    from antennaknobs.engines import PyNECEngine

    if shutil.which("nec2c") is None:
        sys.exit("nec2c CLI not found — required for the gn 2 oracle values")

    entries = []
    for name, frac, freq_mhz, builder in _cases():
        for eps_r, sigma in GROUNDS:
            row = {}
            for model in ("finite-fast", "pec"):
                ground = "pec" if model == "pec" else (model, eps_r, sigma)
                z = PyNECEngine(builder, ground=ground).impedance()[0]
                row[model] = complex(z)
            row["finite"] = _solve_finite_nec2c(builder, eps_r, sigma)
            entries.append((name, frac, eps_r, sigma, freq_mhz, row))
            print(
                f"{name:11s} h={frac:4.2f}wl eps={eps_r:4.1f} sig={sigma:5.3f}"
                f"  gn0={row['finite-fast']:.3f}  gn2={row['finite']:.3f}"
                f"  pec={row['pec']:.3f}"
            )

    lines = [
        '"""Golden PyNEC impedances for the reflection-coefficient ground work.',
        "",
        "GENERATED by scripts/capture_refl_coef_ground_golden.py — do not edit",
        "by hand; re-run that script (under the antennaknobs venv) to refresh.",
        "See docs/refl-coef-ground-plan.md for the validation matrix.",
        "",
        "GOLDEN maps (geometry, height_frac, eps_r, sigma) -> dict of ground",
        'model -> feed impedance in ohms. Ground models: "finite-fast" is NEC',
        "gn 0 (reflection-coefficient approximation, the oracle for momwire's",
        'refl-coef ground_eps solve), "finite" is NEC gn 2 (Sommerfeld-Norton,',
        "the oracle for the Sommerfeld work; captured via the nec2c CLI, NOT",
        "PyNEC, whose Sommerfeld solve is order-dependent and breaks below",
        '0.1 wavelength — see the capture script docstring), "pec" is the PEC',
        "ground. Geometries:",
        '"dipole" is the antennaknobs dipoles.invvee "dipole" variant at',
        '28.47 MHz with base = height_frac * wavelength; "inverted_l" is',
        "verticals.inverted_l defaults at 28.57 MHz, radial-counterpoise base",
        'at the same fractions; "yagi" is beams.yagi with n_directors=4 at',
        "28.47 MHz (> 1.2-wavelength boom — the large image-ray-distance",
        "case for the Sommerfeld grid, docs/sommerfeld-ground-plan.md).",
        '"""',
        "",
        "GOLDEN = {",
    ]
    for name, frac, eps_r, sigma, freq_mhz, row in entries:
        lines.append(f"    ({name!r}, {frac!r}, {eps_r!r}, {sigma!r}): {{")
        for model in GROUND_MODELS:
            lines.append(f"        {model!r}: {row[model]!r},")
        lines.append("    },")
    lines += [
        "}",
        "",
        "FREQ_MHZ = {'dipole': 28.47, 'inverted_l': 28.57, 'yagi': 28.47}",
        "",
    ]
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {OUT_PATH} ({len(entries)} cases)")


if __name__ == "__main__":
    sys.exit(main())
