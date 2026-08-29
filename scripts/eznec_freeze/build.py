"""Build the EZNEC drop-in bundle: native launchers plus one frozen engine.

Run from the repo root, in an environment where momwire (this checkout) and
pyinstaller are installed::

    python scripts/eznec_freeze/build.py

Produces ``dist/momwire-eznec/`` containing the EZNEC-facing launchers
``momwire-eznec[-<basis>][.exe]``, the single frozen ``momwire-eznec-engine
[.exe]`` they run, and that engine's ``_internal`` runtime.  The bundle
directory must be kept together — a launcher spawns the engine BESIDE it, and
the engine needs its runtime beside it in turn.

Two programs and not one, since momwire#718 phase 3.  The launcher is
`scripts/eznec_client_c/momwire_eznec_client.c`, ~31 KB of C that forwards a
deck to a resident engine over the phase-2 wire protocol; the engine is
`entry.py` frozen, and answers both as that resident daemon (``--serve``) and
as the one-shot the launcher's fallback ladder runs.  The launch cost is the
whole reason: the frozen one-shot pays ~1.3 s of interpreter and NumPy import
on EVERY launch, EZNEC launches once per frequency point, and the licensed
engine it stands in for launches in 18-37 ms.

One-dir on purpose: Windows sitting 4 (antennaknobs, 2026-08-21) measured the
one-file form at ~17 s per launch (the self-extract happens on EVERY launch,
and EZNEC launches once per frequency point) against ~1.3 s for one-dir.
One-file is disqualified, not merely slower.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLIENT_SOURCE = HERE.parent / "eznec_client_c"

# The bundle directory and the DEFAULT launcher's name: what EZNEC's engine
# path points at, and what a user's muscle memory already spells.
NAME = "momwire-eznec"

# The one frozen program, under the name its own entry point consumes
# (`entry.py`'s ENGINE_SEGMENT, and the launcher's ENGINE_NAME).  PyInstaller
# builds `dist/<engine name>/`; the directory is renamed to NAME below so the
# bundle, the zip and the workflow keep the path they have always had.
ENGINE_NAME = f"{NAME}-engine"

# The bases that ship as their own executable (momwire#593).  The bundle is
# ONE frozen engine plus COPIES of the native launcher: the basis rides on the
# filename (`entry.py`, `momwire_eznec_client.c`), so a copy is a working
# launcher and costs ~31 KB rather than a second 117 MB runtime.
#
# Two, not the whole eight-name roster — and since the phase-3 flip the reason
# is no longer the zip, because eight ~31 KB copies would round to nothing in
# a 52 MB download.  It is curation: the rest are engines almost nobody picks
# from a file dialog, one of which (`sinusoidal`) refuses every deck in this
# dialect, because every deck here drives a node and point matching has no
# excitation for a source at one (momwire#611/#648 — it was first read as
# `current_slopes`, then as the feed grid, and is really the testing).  So the
# bundle carries the PAIR the parity work was about — the default and the
# NEC-5 twin — and the README says how to make any other, which is a copy.
SHIPPED_VARIANTS = ("razor-nec5",)


def _load_sign():
    """Import the sibling ``sign`` module by path.

    By path for the same reason smoke.py imports THIS file by path: scripts/
    is not a package.  Called from inside ``main()`` rather than imported at
    module level because smoke.py exec's this module to read
    SHIPPED_VARIANTS, and reading a list must not drag in signing.
    """
    spec = importlib.util.spec_from_file_location("eznec_freeze_sign", HERE / "sign.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_client(destination: Path) -> int:
    """Compile the native launcher straight into the bundle.

    Through the shipped build scripts rather than a compiler line spelt here:
    the flags and the version defines ARE part of what the launcher is (the
    momwire version is a hash input for the server key, and -Werror is the
    promise about buffers), and a second spelling of them would build a
    different program from the one the tests gate.

    ``PYTHON`` so both scripts read the version from the SAME install that is
    about to be frozen — the launcher and the engine must agree about the key
    or a bundle's two halves would never find each other's daemon.
    """
    env = {**os.environ, "PYTHON": sys.executable}
    if os.name == "nt":
        script = CLIENT_SOURCE / "build_msvc.bat"
        # Through the command interpreter explicitly: CreateProcess does not
        # execute a .bat itself, and a build that works locally and fails in
        # CI on that distinction is the worst place to learn it.  cl.exe
        # reaches PATH from the workflow's msvc-dev-cmd step.
        cmd = [
            os.environ.get("COMSPEC", "cmd.exe"),
            "/c",
            str(script),
            str(destination),
        ]
    else:
        cmd = [str(CLIENT_SOURCE / "build_cc.sh"), str(destination)]
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=env).returncode


def main() -> int:
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--console",
        "--name",
        ENGINE_NAME,
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

    # PyInstaller named the directory after the program; the BUNDLE keeps the
    # name it has always had, so the zip layout, the workflow's
    # `dist/momwire-eznec/...` paths and every published instruction survive
    # the flip untouched.  Renaming the outer directory is safe because a
    # one-dir exe finds `_internal` relative to ITSELF, never to a recorded
    # path.
    built = Path("dist") / ENGINE_NAME
    bundle = Path("dist") / NAME
    if bundle.exists():
        shutil.rmtree(bundle)
    built.rename(bundle)

    engine = next(bundle.glob(f"{ENGINE_NAME}*"), None)
    if engine is None or not engine.is_file():
        print(f"ERROR: no {ENGINE_NAME} executable in {bundle}", file=sys.stderr)
        return 1

    # The launcher, compiled into the bundle under the DEFAULT EZNEC-facing
    # name.  Its suffix is the engine's: one OS, one spelling of "executable".
    exe = bundle / f"{NAME}{engine.suffix}"
    returncode = _build_client(exe)
    if returncode != 0:
        print(
            f"ERROR: the native launcher did not build (exit {returncode})",
            file=sys.stderr,
        )
        return returncode
    if not exe.is_file():
        print(f"ERROR: no {exe.name} in {bundle}", file=sys.stderr)
        return 1

    # Sign BEFORE the copy loop, not after.  Authenticode covers PE contents
    # and NOT the filename, so a copy of a signed exe is itself validly
    # signed -- one signtool call therefore ships every variant signed, which
    # matters when a cloud-HSM CA meters signing operations.
    #
    # Both DISTINCT binaries and no more: the engine and the launcher are two
    # programs, and every variant is a byte-for-byte copy of the launcher.
    #
    # Scope, so the README's claim stays honest: in one-dir the Python payload
    # lives in _internal/ and is outside the signed bytes, so the engine's
    # signature attests who shipped the loader, not that the runtime beside it
    # is untouched.  One-file would cover both and is disqualified on the
    # 17 s vs 1.3 s launch measurement above.
    signer = _load_sign()
    signed = signer.sign_if_configured([engine, exe])

    # CI's post-condition: the workflow decides MOMWIRE_SIGN_MODE exactly
    # once, and this holds the build to it.  Without this check a
    # half-configured environment (one mis-edited step condition upstream)
    # would ship a green, canonically-named, UNSIGNED bundle — the fail-open
    # the momwire#711 retro-review flagged.  Local builds never set the
    # mode variable and stay legitimately unsigned.
    mode = os.environ.get("MOMWIRE_SIGN_MODE")
    if mode and not signed:
        print(
            f"ERROR: MOMWIRE_SIGN_MODE={mode} promises a signed build but no "
            "signing identity reached build.py (neither MOMWIRE_SIGN_METADATA "
            "nor MOMWIRE_SIGN_SHA1 is set)",
            file=sys.stderr,
        )
        return 1

    # One copy of the LAUNCHER per shipped variant — never of the engine: the
    # basis rides on the launcher's filename, and the engine it spawns is told
    # which basis by flag.  `shutil.copy2` and not a symlink or hard link: the
    # zip is unpacked on Windows, where neither survives the round trip
    # through Compress-Archive and Explorer, and a link that arrives as a
    # 0-byte stub is a broken engine that looks like a present one.
    variants = []
    for basis in SHIPPED_VARIANTS:
        variant = exe.with_name(f"{NAME}-{basis}{exe.suffix}")
        shutil.copy2(exe, variant)
        variants.append(variant)
        print(f"variant: {variant.name}")

    # Assert what the pre-copy ordering CLAIMS, rather than trusting it: a
    # variant that silently arrived unsigned is precisely the failure that
    # ordering exists to prevent, and nothing downstream would notice it.
    if signed:
        for variant in variants:
            if not signer.has_signature(variant):
                print(
                    f"ERROR: {variant.name} carries no signature; the copy "
                    "did not inherit it",
                    file=sys.stderr,
                )
                return 1
        # The engine and the default launcher were signed directly; the
        # variants are the copies just asserted.
        print(f"signature present on all {len(variants) + 2} executables")

    # A provenance note inside the bundle: which momwire this is, HOW it was
    # (or wasn't) signed, and where the drop-in's serve/refuse contract is
    # documented.  The signing line records what actually happened rather
    # than what any label claims: the zip's name does not survive unzipping,
    # so this is the only signing provenance that travels with the bytes
    # (#711 retro-review).
    if signed and os.environ.get("MOMWIRE_SIGN_METADATA"):
        signing_note = (
            "SIGNING: every executable in this folder is Authenticode-signed "
            "(Azure\nArtifact Signing).  The signature covers those "
            "executables; the Python\npayload in _internal/ is outside the "
            "signed bytes."
        )
    elif signed and os.environ.get("MOMWIRE_SIGN_ALLOW_UNTRUSTED") == "1":
        signing_note = (
            "SIGNING: SELF-SIGNED REHEARSAL signature (untrusted root) — "
            "this is a CI test\nbuild, NOT a release download."
        )
    elif signed:
        signing_note = (
            "SIGNING: every executable in this folder is Authenticode-signed "
            "(local\ncertificate store identity)."
        )
    else:
        signing_note = "SIGNING: this build is unsigned."

    labels = {NAME: "the default — degree-2 B-spline (bs2)"}
    labels.update(
        {f"{NAME}-{b}": "the NEC-5 formulation twin" for b in SHIPPED_VARIANTS}
    )
    labels[ENGINE_NAME] = "the compute engine the launchers run"
    width = max(len(n) for n in labels) + len(exe.suffix)
    table = "\n".join(
        f"  {n + exe.suffix:<{width}}  {label}" for n, label in labels.items()
    )
    (bundle / "README.txt").write_text(
        f"momwire-eznec {version('momwire')} — momwire standing in for the "
        "NEC-5 console engine EZNEC Pro+ v7 launches.\n"
        "\n"
        f"{signing_note}\n"
        "\n"
        "Point EZNEC's external-engine path at one of the LAUNCHERS below, in\n"
        "this folder.  Keep the folder together: a launcher runs the engine\n"
        "beside it, and the engine needs its _internal runtime beside IT.\n"
        "\n"
        f"{table}\n"
        "\n"
        "Point EZNEC at a launcher, never at momwire-eznec-engine: the engine\n"
        "is what the launchers run, and naming it directly gives up the warm\n"
        "start below for nothing.\n"
        "\n"
        "WHY IT IS FAST.  A launcher keeps a warm engine RESIDENT — started on\n"
        "the first calculation, retired after 15 idle minutes — so every launch\n"
        "after the first costs milliseconds instead of the engine's own\n"
        "start-up.  EZNEC launches the engine once per frequency point, so that\n"
        "is most of a sweep.  If anything about the resident path fails, the\n"
        "launcher runs the engine directly instead: the same answer, at the old\n"
        "speed.\n"
        "\n"
        "WHICH ONE.  The launchers accept the same models and answer in\n"
        "different formulations.  The default is momwire's own degree-2\n"
        "B-spline basis.  momwire-eznec-razor-nec5 is NEC-5's formulation\n"
        "TWIN — the tent basis with razor-blade path testing NEC-5 itself\n"
        "uses.\n"
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
        "'eznec-' selects it.  So copy a LAUNCHER in this folder — a few tens\n"
        "of kilobytes, not the engine — rename the copy to\n"
        "momwire-eznec-<basis>.exe, and that basis is what answers.\n"
        "Known bases:\n"
        "\n"
        "  bspline  bspline-d1  hmatrix  arrayblock  razor  razor-nec5\n"
        "  sinusoidal  sinusoidal-galerkin\n"
        "\n"
        "(`sinusoidal` cannot answer this dialect — every deck in it drives\n"
        "a node, and point matching has no excitation for a source at one —\n"
        "and will say so, by name, in the printout.)\n"
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
