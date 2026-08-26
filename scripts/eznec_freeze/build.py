"""Freeze the EZNEC drop-in with PyInstaller (one dir, one exe inside).

Run from the repo root, in an environment where momwire (this checkout) and
pyinstaller are installed::

    python scripts/eznec_freeze/build.py

Produces ``dist/momwire-eznec/`` containing ``momwire-eznec[.exe]`` and its
``_internal`` runtime.  The bundle directory must be kept together — EZNEC's
engine path points at the exe inside it.

One-dir on purpose: Windows sitting 4 (antennaknobs, 2026-08-21) measured the
one-file form at ~17 s per launch (the self-extract happens on EVERY launch,
and EZNEC launches once per frequency point) against ~1.3 s for one-dir.
One-file is disqualified, not merely slower.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "momwire-eznec"

# The bases that ship as their own executable (momwire#593).  The bundle is
# ONE PyInstaller build plus COPIES of its exe: the basis rides on the
# filename (`entry.py`), so a copy is a working engine and costs one stub
# rather than a second 117 MB runtime.
#
# Two, not the whole eight-name roster, and the reason is the zip.  Each copy
# adds ~9.5 MB uncompressed; all eight would roughly double a 52 MB download
# to ship the others, engines almost nobody picks from a file dialog,
# two of which (the sinusoidal family) refuse every deck in this dialect
# because their delta gap lands on a segment centre and every deck here
# drives a node (momwire#611 — it used to be `current_slopes`, until that
# landed and the real obstruction turned out to be one layer down).  So the
# bundle carries the PAIR the parity work was about — the default and the
# NEC-5 twin — and the README says how to make any other, which is a copy.
SHIPPED_VARIANTS = ("razor-nec5",)


def main() -> int:
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--console",
        "--name",
        NAME,
        # The C++ accelerator (momwire._accelerators) is imported inside a
        # try/except; collect the whole package so no lazily-reached module
        # is left out of the bundle.
        "--collect-submodules",
        "momwire",
        # Optional-dependency imports the seam never reaches; excluded so an
        # environment that happens to carry them doesn't fatten the bundle.
        "--exclude-module",
        "matplotlib",
        "--exclude-module",
        "tkinter",
        str(HERE / "entry.py"),
    ]
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        return result.returncode

    bundle = Path("dist") / NAME
    exe = next(bundle.glob(f"{NAME}*"), None)
    if exe is None or not exe.is_file():
        print(f"ERROR: no {NAME} executable in {bundle}", file=sys.stderr)
        return 1

    # One copy per shipped variant.  `shutil.copy2` and not a symlink or hard
    # link: the zip is unpacked on Windows, where neither survives the round
    # trip through Compress-Archive and Explorer, and a link that arrives as a
    # 0-byte stub is a broken engine that looks like a present one.
    for basis in SHIPPED_VARIANTS:
        variant = exe.with_name(f"{NAME}-{basis}{exe.suffix}")
        shutil.copy2(exe, variant)
        print(f"variant: {variant.name}")

    # A provenance note inside the bundle: which momwire this is, and where
    # the drop-in's serve/refuse contract is documented.
    labels = {NAME: "the default — degree-2 B-spline (bs2)"}
    labels.update(
        {f"{NAME}-{b}": "the NEC-5 formulation twin" for b in SHIPPED_VARIANTS}
    )
    width = max(len(n) for n in labels) + len(exe.suffix)
    table = "\n".join(
        f"  {n + exe.suffix:<{width}}  {label}" for n, label in labels.items()
    )
    (bundle / "README.txt").write_text(
        f"momwire-eznec {version('momwire')} — momwire standing in for the "
        "NEC-5 console engine EZNEC Pro+ v7 launches.\n"
        "\n"
        "Point EZNEC's external-engine path at one of these, in this folder.\n"
        "Keep the folder together: every exe needs the _internal runtime "
        "beside it.\n"
        "\n"
        f"{table}\n"
        "\n"
        "WHICH ONE.  They accept the same models and answer in different\n"
        "formulations.  The default is momwire's own degree-2 B-spline basis.\n"
        "momwire-eznec-razor-nec5 is NEC-5's formulation TWIN — the tent basis\n"
        "with razor-blade path testing NEC-5 itself uses.\n"
        "\n"
        "REPRODUCTION IS NOT ACCURACY.  The twin agrees with the licensed\n"
        "engine because it runs the same algorithm, not because it is more\n"
        "correct, and it inherits that engine's discretization error along\n"
        "with its answers.\n"
        "\n"
        "SEGMENT COUNTS: use EVEN ones for a centre-fed dipole.  NEC-5's\n"
        "basis is the tent, so its unknowns and its sources sit at KNOTS.\n"
        "An odd count leaves no knot at the centre and cannot feed there.\n"
        "(The 'odd segments' habit is NEC-2's, where sources sit at segment\n"
        "centres.)\n"
        "\n"
        "Measured on a 0.476-wavelength dipole, free space, even meshes, all\n"
        "fed at the centre knot:\n"
        "\n"
        "    segments   licensed NEC-5      B-spline (bs2)\n"
        "        4       56.12 - 108.59j     67.64 - 31.15j\n"
        "       20       66.67 -  35.88j     67.74 - 29.16j\n"
        "      160       67.67 -  29.28j     67.80 - 28.34j\n"
        "\n"
        "razor-nec5 tracks the licensed column to 0.003 - 0.007 ohm at every\n"
        "row -- flat, not improving, which is what a twin looks like.\n"
        "\n"
        "Which is nearer the truth is a different question, asked by scoring\n"
        "each basis against ITS OWN answer at the finest mesh above, N = 160,\n"
        "through this same engine:\n"
        "\n"
        "    segments   bs2 error   razor-nec5 error\n"
        "        4       2.81 ohm      80.14 ohm\n"
        "       20       0.82 ohm       6.67 ohm\n"
        "       60       0.25 ohm       1.43 ohm\n"
        "\n"
        "Both converge; at a matched mesh the B-spline basis is 5.8-28x nearer\n"
        "its own limit.  Neither is converged at a coarse mesh -- bs2 is\n"
        "still 2.8 ohm out at four segments -- the difference is how fast the\n"
        "error falls.\n"
        "\n"
        "Extrapolated to their limits, the two formulations MEET within\n"
        "0.08-0.21 ohm on this deck (parity_limits() in the probe script is\n"
        "the receipt).  So most of a twin-vs-default disagreement at a\n"
        "practical mesh is the twin still walking the O(1/N) path it shares\n"
        "with the licensed engine, not the two engines heading somewhere\n"
        "different.\n"
        "\n"
        "So pick the twin when you want what NEC-5 would have said — checking\n"
        "a published NEC-5 result, or comparing against a NEC-5 workflow.\n"
        "Pick the default when you want momwire's own best answer.  When they\n"
        "disagree at a practical mesh, neither is broken: most of the gap is\n"
        "the twin's inherited discretization, and what remains is a fraction\n"
        "of an ohm of formulation.\n"
        "\n"
        "MAKING ANOTHER.  The basis rides on the FILENAME: everything after\n"
        "'eznec-' selects it.  So copy an exe in this folder and rename the\n"
        "copy to momwire-eznec-<basis>.exe and that basis is what answers.\n"
        "Known bases:\n"
        "\n"
        "  bspline  bspline-d1  hmatrix  arrayblock  razor  razor-nec5\n"
        "  sinusoidal  sinusoidal-galerkin\n"
        "\n"
        "(The sinusoidal families cannot answer this dialect — every\n"
        "deck in it drives a node, and their delta gap lands on the nearest\n"
        "segment centre instead — and will say so, by name, in the\n"
        "printout.)\n"
        "\n"
        "A name that matches no basis is not a silent fallback: it refuses,\n"
        "names itself, and lists the bases that exist.\n"
        "\n"
        "What serves and what refuses (by name, in the printout):\n"
        "https://momwire.dev/reference/eznec-nec5/\n",
        newline="\r\n",
    )
    print(f"bundle ready: {bundle}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
