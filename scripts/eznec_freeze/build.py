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

import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "momwire-eznec"


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

    # A provenance note inside the bundle: which momwire this is, and where
    # the drop-in's serve/refuse contract is documented.
    (bundle / "README.txt").write_text(
        f"momwire-eznec {version('momwire')} — momwire standing in for the "
        "NEC-5 console engine EZNEC Pro+ v7 launches.\n"
        "\n"
        "Point EZNEC's external-engine path at momwire-eznec.exe in this "
        "folder.  Keep the folder together: the exe needs the _internal "
        "runtime beside it.\n"
        "\n"
        "What serves and what refuses (by name, in the printout):\n"
        "https://momwire.dev/reference/eznec-nec5/\n",
        newline="\r\n",
    )
    print(f"bundle ready: {bundle}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
