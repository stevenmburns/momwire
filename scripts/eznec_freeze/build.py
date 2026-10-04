"""Build the EZNEC drop-in bundle: native launchers plus one frozen engine.

Run from the repo root, in an environment where momwire (this checkout) and
pyinstaller are installed::

    python scripts/eznec_freeze/build.py

Produces ``dist/momwire-eznec/`` containing the launchers EZNEC and SimNEC
point at -- ``momwire-nec5[-<basis>][.exe]``, ``momwire-nec4[-<basis>][.exe]``
and the deprecated ``momwire-eznec[-<basis>][.exe]`` -- the single frozen
``momwire-eznec-engine[.exe]`` they run, and that engine's ``_internal`` runtime.  The bundle
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

# The bundle directory, and the name the native launcher is COMPILED as before
# every other launcher is copied from it.  Since momwire#1295 phase 4 that
# launcher name is a deprecated one (`DEPRECATED_SUCCESSORS`), but it stays the
# compiled one: the zip, the workflow's `dist/momwire-eznec/...` paths and the
# smoke's argument all spell it, and every launcher is the same bytes anyway.
NAME = "momwire-eznec"

# The one frozen program, under the name its own entry point consumes
# (`entry.py`'s ENGINE_SEGMENT, and the launcher's ENGINE_NAME).  PyInstaller
# builds `dist/<engine name>/`; the directory is renamed to NAME below so the
# bundle, the zip and the workflow keep the path they have always had.
ENGINE_NAME = f"{NAME}-engine"

# The PRIMARY launcher names (momwire#1295 phase 4): one family per EZNEC
# slot, named for the slot rather than for the host.  `momwire-nec5[-<basis>]`
# answers EZNEC's External NEC-5 slot AND SimNEC's NEC-5 engine -- SimNEC
# reads the deck SYNTAX off a substring of the engine path (`nec2c`, then
# `nec5`, momwire#1239) and EZNEC does not read the name at all, so one name
# serves both hosts.  `momwire-nec4[-<basis>]` answers EZNEC's External
# NEC-4.2 slot: the launcher reads `nec4` in its own name as the dialect and
# the basis past `nec4-`.
#
# Every one is a COPY of the one native launcher (`entry.py`,
# `momwire_eznec_client.c`), so a copy is a working launcher and costs ~31 KB
# rather than a second 117 MB runtime.  Two per slot, not the whole roster:
# the default plus the one formulation each slot's parity work was about
# (razor-2p is NEC-5's own; point-matched sinusoidal is the family NEC-4.2
# belongs to), and the README says how to make any other, which is a copy.
#
# Keyed by the file stem, valued by the README table's label.  Kept SHORT: a
# table row in a plain-text README wrapped at ~70 columns; the longer story
# is under WHICH ONE.
NEC5_LAUNCHERS = {
    "momwire-nec5": "the default, degree-2 B-spline (bs2)",
    "momwire-nec5-razor-2p": "the tent basis, razor-blade testing",
}
NEC4_LAUNCHERS = {
    "momwire-nec4": "the default, degree-2 B-spline (bs2)",
    "momwire-nec4-sinusoidal": "the sinusoidal basis, point matched",
}

# The DEPRECATED names: the `momwire-eznec[-<basis>]` launchers every release
# before the NEC-4.2 slot shipped (momwire#593), still shipped and still
# answering exactly as they did.  Deprecated in the README and the docs ONLY,
# never in behaviour: an EZNEC engine path is a string a user typed once and
# forgot, and dropping or changing a name would break those installs
# silently -- a launcher is ~31 KB, the cheapest possible way not to.
#
# No printout note says so, deliberately.  The launcher forwards a deck to a
# warm server keyed by BASIS, not by name, so `momwire-eznec.exe` and
# `momwire-nec5.exe` share one server and the engine cannot tell which name
# launched it; the only channel that knows is the launcher, and teaching it to
# say so would change a deprecated name's printout -- the opposite of
# "answering exactly as they did" -- for a note EZNEC users never read.
#
# `SHIPPED_VARIANTS` are the bases copied as `momwire-eznec-<basis>` beside
# the default `momwire-eznec` the launcher is COMPILED as (`NAME`); the smoke
# reads this tuple to gate them.  TWO SPELLINGS OF ONE BASIS, not two bases
# (momwire#817): `razor-2p` is the roster's long-term name and `razor-nec5`
# its deprecated spelling (#785/#794); both resolve through
# `basis_from_program_name` to the same engine and answer identically.
SHIPPED_VARIANTS = ("razor-2p", "razor-nec5")

# What each deprecated name now IS, for the README's table: the primary name
# that answers identically.  Keyed by the SHIPPED_VARIANTS entry, so a variant
# added without one fails the build here rather than shipping an unlabelled
# row.
DEPRECATED_SUCCESSORS = {
    "": "momwire-nec5",
    "razor-2p": "momwire-nec5-razor-2p",
    "razor-nec5": "momwire-nec5-razor-2p",
}

# The README's lists of bases a copy can name, per slot.  Restated rather
# than read from `momwire.deck` because this script must not import the
# runtime it is about to freeze for a constant; `tests/test_eznec_bundle_
# readme.py` holds them to the roster.  The NEC-5 list leaves out the bases
# that refuse every deck in that dialect (`sinusoidal`, `pulse`: centre-fed
# only) and the deprecated `razor-nec5` spelling; the NEC-4.2 list is
# `NEC2_BASES`, the centre-feeding ones (razor refuses there by name).
NEC5_KNOWN_BASES = (
    "bspline",
    "bspline-d1",
    "hmatrix",
    "arrayblock",
    "razor-2p",
    "sinusoidal-galerkin",
)
NEC4_KNOWN_BASES = (
    "bspline",
    "bspline-d1",
    "hmatrix",
    "arrayblock",
    "sinusoidal",
    "sinusoidal-galerkin",
    "pulse",
)


def launcher_stems() -> tuple[str, ...]:
    """Every launcher the bundle ships, by file stem: the primary families
    first, then the deprecated names.  The engine is not a launcher."""
    deprecated = (NAME, *(f"{NAME}-{b}" for b in SHIPPED_VARIANTS))
    return (*NEC5_LAUNCHERS, *NEC4_LAUNCHERS, *deprecated)


# The version this freeze bakes into the bundle (momwire#1277), as a top-level
# module the engine imports by name (`momwire_serve_client.BUILD_VERSION_MODULE`,
# restated here because this script must not import the runtime it is about to
# freeze for a constant).  Written into PyInstaller's own scratch tree under
# `build/` — gitignored, and never inside `src/` — so a source checkout or a pip
# install has no such module to find, and the readers consult it only when
# `sys.frozen` says they are inside a bundle anyway.
BUILD_VERSION_MODULE = "_momwire_build_version"
BAKE_DIR = Path("build") / "eznec_freeze_bake"


def _bake_version(release: str) -> Path:
    """Write the baked-version module and return the directory holding it.

    Why baked at all: the printout stamp, the SimNEC probe and the clients
    used to read `importlib.metadata`, which answers from whichever
    `momwire-*.dist-info` sits first in `_internal`.  A release extracted over
    an older one keeps the older dist-info, and Windows lists it first — so
    0.68.0 code stamped itself 0.66.0.  A constant frozen with the code cannot
    be outvoted by a leftover folder.
    """
    BAKE_DIR.mkdir(parents=True, exist_ok=True)
    (BAKE_DIR / f"{BUILD_VERSION_MODULE}.py").write_text(
        '"""Generated by scripts/eznec_freeze/build.py at freeze time; never '
        'committed (momwire#1277)."""\n'
        "\n"
        f"VERSION = {release!r}\n",
        encoding="utf-8",
    )
    return BAKE_DIR


# LLVM's OpenMP runtime.  Both Windows extensions (`_accelerators` and
# `_near_interface_accel`) link it, because setup.py builds them with
# `/openmp:llvm` — not `vcomp140.dll`, which is what a reader expects from MSVC
# and is also a redistributable Windows knows about.  This one is neither a
# system DLL nor something PyInstaller collects, so momwire#737 shipped two
# phases of bundle in which the accelerator import simply failed and every
# solve was pure Python.
OPENMP_DLL = "libomp140.x86_64.dll"


def _openmp_runtime() -> Path | None:
    """The full path of the OpenMP runtime the built extensions actually load.

    By IMPORTING the extension first and then asking the loader where the DLL
    came from: that is the copy this build's `.pyd` resolves to, so what gets
    bundled is what the engine will look for rather than the first file of that
    name on PATH.  `GetModuleHandleW` does not load anything — it answers only
    for a module already mapped into this process, which the import just did.

    The PATH walk is the fallback for the case where the extension imported
    without pulling the runtime in (a future build that links it lazily); a
    build box with no copy at all gets None, and the caller refuses to ship.
    """
    import ctypes
    import ctypes.util

    try:
        import momwire._accelerators  # noqa: F401
    except ImportError as exc:
        print(f"WARNING: {exc}; looking for {OPENMP_DLL} on PATH", file=sys.stderr)

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # Explicit types: the default `c_int` restype truncates a 64-bit HMODULE,
    # and a truncated handle answers with a wrong path or none at all.
    kernel32.GetModuleHandleW.restype = ctypes.c_void_p
    kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    kernel32.GetModuleFileNameW.restype = ctypes.c_uint32
    kernel32.GetModuleFileNameW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
    ]
    handle = kernel32.GetModuleHandleW(OPENMP_DLL)
    if handle:
        buffer = ctypes.create_unicode_buffer(32768)
        if kernel32.GetModuleFileNameW(handle, buffer, len(buffer)):
            return Path(buffer.value)

    located = ctypes.util.find_library(OPENMP_DLL)
    if located and Path(located).is_file():
        return Path(located)
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry and (candidate := Path(entry) / OPENMP_DLL).is_file():
            return candidate
    return None


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


def launcher_table(suffix: str) -> str:
    """The README's launcher table: one block per family, primary first.

    Generated from the same constants the copy loop reads, so a launcher
    cannot ship without a row or get a row without shipping.  ``suffix`` is
    the executable suffix of the OS the bundle was built on.
    """
    deprecated = {NAME: DEPRECATED_SUCCESSORS[""]}
    deprecated.update(
        {f"{NAME}-{b}": DEPRECATED_SUCCESSORS[b] for b in SHIPPED_VARIANTS}
    )
    blocks = (
        (
            "EZNEC's External NEC-5 slot, and SimNEC's NEC-5 engine:",
            NEC5_LAUNCHERS,
        ),
        ("EZNEC's External NEC-4.2 slot:", NEC4_LAUNCHERS),
        (
            "The old names, DEPRECATED - still shipped, still identical:",
            {n: f"= {successor}{suffix}" for n, successor in deprecated.items()},
        ),
        (
            "And the engine itself, which is not a launcher:",
            {ENGINE_NAME: "the compute engine the launchers run"},
        ),
    )
    width = max(len(n) for _, rows in blocks for n in rows) + len(suffix)
    return "\n\n".join(
        heading
        + "\n\n"
        + "\n".join(f"  {n + suffix:<{width}}  {label}" for n, label in rows.items())
        for heading, rows in blocks
    )


def _wrap_bases(bases: tuple[str, ...], indent: str) -> str:
    """A list of basis names as README lines under ~70 columns."""
    lines, line = [], indent
    for basis in bases:
        if line.strip() and len(line) + 2 + len(basis) > 70:
            lines.append(line.rstrip())
            line = indent
        line += basis + "  "
    lines.append(line.rstrip())
    return "\n".join(lines)


def readme_text(release: str, signing_note: str, suffix: str) -> str:
    """The bundle's README.txt, LF line endings (the caller writes CRLF).

    A function of three strings and the module's constants, so it can be
    gated without a freeze (`tests/test_eznec_bundle_readme.py`).
    """
    nec5_known = _wrap_bases(NEC5_KNOWN_BASES, "  NEC-5 slot:    ")
    nec4_known = _wrap_bases(NEC4_KNOWN_BASES, "  NEC-4.2 slot:  ")
    return (
        f"momwire-eznec {release} — momwire standing in for the external\n"
        "NEC-5 and NEC-4.2 engines EZNEC Pro+ v7 launches, and for SimNEC's\n"
        "NEC-5 engine.\n"
        "\n"
        f"{signing_note}\n"
        "\n"
        "INSTALL INTO A FOLDER OF ITS OWN.  The zip holds a momwire-eznec\\\n"
        "folder; extract it somewhere dedicated, e.g. C:\\momwire-eznec\\ -\n"
        "NOT into EZNEC's Docs folder, and NOT over a previous release.\n"
        "\n"
        "FOR EVERY NEW RELEASE: delete that folder ENTIRELY, extract the new\n"
        "zip in its place, and leave EZNEC's engine path unchanged.  Never\n"
        "extract over an old install: that leaves the old release's files\n"
        "behind (libraries in _internal among them), mixed in with the new\n"
        "ones.  If it happens anyway, line 2 of the printout ends in\n"
        "'(stale: <old version>)' - delete the folder and extract again.\n"
        "\n"
        "If Windows says a file is in use while you delete the folder, that\n"
        "is the resident engine (see WHY IT IS FAST below).  Wait 15 minutes\n"
        "for it to retire, or end momwire-eznec-engine.exe in Task Manager.\n"
        "\n"
        "BEFORE EXTRACTING: right-click the downloaded zip, Properties, tick\n"
        '"Unblock", OK - then extract.  That clears the mark-of-the-web on every\n'
        "file inside at once, so the SmartScreen screen below never appears,\n"
        "and the files keep their original dates instead of the download's.\n"
        "(Thanks to Dan AC6LA for the tip.)\n"
        "\n"
        'FIRST RUN (if you extracted first): Windows will show "Windows protected your PC", and you\n'
        'have to click "More info" and then "Run anyway".  Nothing is wrong\n'
        "with the download.  That screen is SmartScreen saying this file has no\n"
        'REPUTATION yet — click "More info" and it names the publisher, because\n'
        "the signature above is valid and Windows is reading it.  Reputation is\n"
        "earned by download volume, so it fades with time and starts over with\n"
        'each release; the certificate that would skip the wait outright ("EV")\n'
        "is not one the signing service used here issues.\n"
        "\n"
        "In EZNEC, choose the external engine you want - External NEC-5 or\n"
        "External NEC-4.2 - and point its engine path at a LAUNCHER from the\n"
        "matching family below, in this folder.  The name picks the deck\n"
        "dialect, so a momwire-nec4 name belongs in the NEC-4.2 slot and a\n"
        "momwire-nec5 name in the NEC-5 slot.  Keep the folder together: a\n"
        "launcher runs the engine beside it, and the engine needs its\n"
        "_internal runtime beside IT.\n"
        "\n"
        f"{launcher_table(suffix)}\n"
        "\n"
        "Point EZNEC at a launcher, never at momwire-eznec-engine: the engine\n"
        "is what the launchers run, and naming it directly gives up the warm\n"
        "start below for nothing.\n"
        "\n"
        "THE OLD NAMES.  Earlier releases told you to point EZNEC's NEC-5\n"
        "slot at the momwire-eznec names.  Those names are DEPRECATED; they\n"
        "still ship and answer exactly as before: momwire-eznec.exe is\n"
        "momwire-nec5.exe, and momwire-eznec-razor-2p.exe and\n"
        "momwire-eznec-razor-nec5.exe are both momwire-nec5-razor-2p.exe.  An\n"
        "engine path you typed once and forgot keeps working.  New setups\n"
        "should use the momwire-nec5 names, which work in SimNEC as well.\n"
        "\n"
        "WHY IT IS FAST.  A launcher keeps a warm engine RESIDENT — started on\n"
        "the first calculation, retired after 15 idle minutes — so every launch\n"
        "after the first costs milliseconds instead of the engine's own\n"
        "start-up.  EZNEC launches the engine once per frequency point, so that\n"
        "is most of a sweep.  If anything about the resident path fails, the\n"
        "launcher runs the engine directly instead: the same answer, at the old\n"
        "speed.\n"
        "\n"
        "THE FIRST CALCULATION IS THE SLOW ONE, and EZNEC shows nothing while\n"
        "it happens: several seconds, once, to start that engine.  It is not a\n"
        "hang — wait for it and every calculation after it is the fast one.\n"
        "You meet it again after the engine retires, so a pause for thought\n"
        "longer than 15 minutes buys one more slow calculation.\n"
        "\n"
        "WHICH ONE.  Within a slot the launchers accept the same models and\n"
        "answer in different formulations.  Each slot's default is momwire's\n"
        "own degree-2 B-spline basis.\n"
        "\n"
        "In the NEC-5 slot, momwire-nec5-razor-2p is the tent basis with\n"
        "razor-blade path testing at NEC-5's two-point rule — the same\n"
        "formulation NEC-5 itself uses, which is why it agrees with the\n"
        "licensed engine to the fraction of an ohm measured below.\n"
        "\n"
        "In the NEC-4.2 slot, momwire-nec4-sinusoidal is the sinusoidal basis\n"
        "with point matching, the basis family NEC-4.2 itself belongs to.\n"
        "razor cannot serve this slot: NEC-4.2 decks put their sources at\n"
        "segment centres and razor places sources at knots, so a\n"
        "momwire-nec4-razor-2p copy refuses, by name, in the printout.\n"
        "\n"
        "SIMNEC.  Point SimNEC's NEC-5 engine setting at momwire-nec5.exe or\n"
        "momwire-nec5-razor-2p.exe.  SimNEC decides which deck syntax to write\n"
        "from the engine's NAME, which must contain nec5, so the old\n"
        "momwire-eznec names will not do there.  Same engine, same warm start;\n"
        "a path with no spaces in it is safest.\n"
        "\n"
        "REPRODUCTION IS NOT ACCURACY.  razor-2p agrees with the licensed\n"
        "engine because it runs the same algorithm, not because it is more\n"
        "correct, and it inherits that engine's discretization error along\n"
        "with its answers.\n"
        "\n"
        "SEGMENT COUNTS IN THE NEC-5 SLOT: use EVEN ones for a centre-fed\n"
        "dipole.  NEC-5's basis is the tent, so its unknowns and its sources\n"
        "sit at KNOTS.  An odd count leaves no knot at the centre and cannot\n"
        "feed there.  (The 'odd segments' habit is NEC-2's, and NEC-4.2's,\n"
        "where sources sit at segment centres.)\n"
        "\n"
        "Measured on a 0.476-wavelength dipole, free space, even meshes, all\n"
        "fed at the centre knot:\n"
        "\n"
        "    segments   licensed NEC-5      B-spline (bs2)\n"
        "        4       56.12 - 108.59j     67.64 - 31.15j\n"
        "       20       66.67 -  35.88j     67.74 - 29.16j\n"
        "      160       67.67 -  29.28j     67.80 - 28.34j\n"
        "\n"
        "razor-2p tracks the licensed column to 0.003 - 0.007 ohm at every\n"
        "row -- flat, not improving, which is what a twin looks like.\n"
        "\n"
        "Which is nearer the truth is a different question, asked by scoring\n"
        "each basis against ITS OWN answer at the finest mesh above, N = 160,\n"
        "through this same engine:\n"
        "\n"
        "    segments   bs2 error   razor-2p error\n"
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
        "'nec5-' or 'nec4-' selects it.  So copy a LAUNCHER in this folder —\n"
        "a few tens of kilobytes, not the engine — rename the copy to\n"
        "momwire-nec5-<basis>.exe or momwire-nec4-<basis>.exe, and that\n"
        "basis is what answers, in that slot.  Known bases:\n"
        "\n"
        f"{nec5_known}\n"
        f"{nec4_known}\n"
        "\n"
        "(In the NEC-5 slot, sinusoidal and pulse cannot answer — every deck\n"
        "there drives a node, and point matching has no excitation for a\n"
        "source at one.  In the NEC-4.2 slot, razor cannot, as above.  Either\n"
        "says so, by name, in the printout.)\n"
        "\n"
        "A name that matches no basis is not a silent fallback: it refuses,\n"
        "names itself, and lists the bases that exist.\n"
        "\n"
        "What serves and what refuses (by name, in the printout):\n"
        "https://momwire.dev/reference/eznec-nec5/\n"
    )


def main() -> int:
    # Read ONCE: the baked constant, the README's title line and (through the
    # launcher's own build script) the server key all name the same release.
    release = version("momwire")
    bake_dir = _bake_version(release)
    print(f"baked version: {release} ({bake_dir / BUILD_VERSION_MODULE}.py)")
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
        # The printout's line-2 engine stamp reads the installed version
        # through `importlib.metadata`, which answers from the package's
        # `.dist-info` — a directory `--collect-submodules` does NOT bring
        # along.  Without this the frozen engine stamps every printout
        # `unknown` and the one field a bug report needs most is the one it
        # cannot carry (`momwire.eznec._printout.engine_stamp`).
        "--copy-metadata",
        "momwire",
        # The baked version (momwire#1277), which the stamp reads BEFORE the
        # metadata above.  Hidden-imported because its only importer names it
        # through `importlib.import_module`, which PyInstaller cannot trace;
        # the metadata stays as the fallback and as what the overlay check
        # counts.
        "--paths",
        str(bake_dir.resolve()),
        "--hidden-import",
        BUILD_VERSION_MODULE,
        # Optional-dependency imports the seam never reaches; excluded so an
        # environment that happens to carry them doesn't fatten the bundle.
        "--exclude-module",
        "matplotlib",
        "--exclude-module",
        "tkinter",
        str(HERE / "entry.py"),
    ]

    # Windows only, and a hard failure there.  PyInstaller collects what the
    # `.pyd`s declare through its own analysis on POSIX, but on Windows the
    # OpenMP runtime is not a dependency it finds, and a build box that cannot
    # produce the file would ship exactly the bundle momwire#737 is about — one
    # that runs, answers correctly, and is 20x slow on the machines that lack a
    # Visual Studio toolchain.  Refusing beats shipping that again.
    #
    # POSIX takes nothing: those wheels link the SYSTEM OpenMP runtime on
    # purpose, so they share one runtime with pynec-accel rather than clashing
    # (`src/momwire/_accel.py`'s docstring), and bundling one here would undo
    # that for the frozen engine.
    if os.name == "nt":
        runtime = _openmp_runtime()
        if runtime is None:
            print(
                f"ERROR: {OPENMP_DLL} is nowhere this build can see it; the "
                "bundle would ship with a dead accelerator and every solve on "
                "the pure-Python path (momwire#737)",
                file=sys.stderr,
            )
            return 1
        # Destination `.`, which a one-dir build resolves to `_internal/` —
        # beside the collected `.pyd`s, which is where the loader looks first
        # for an extension's dependencies.
        cmd[-1:-1] = ["--add-binary", f"{runtime}{os.pathsep}."]
        print(f"openmp runtime: {runtime}")

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
    # TWO paths, BEFORE the variant copies below — so sign.py's own gates
    # count 2 (`timestamped: all 2 signature(s) countersigned`, momwire#755).
    # The "all N executables" line further down counts FILES, after the
    # copies, and is therefore a different and larger number by design. The
    # two do not disagree and must not be made to agree: the launcher
    # variants are byte-identical copies of `exe` (verified on the v0.44.0
    # release — momwire-eznec.exe and momwire-eznec-razor-nec5.exe share
    # SHA256 111B8A48…, 192,768 bytes), so there are two DISTINCT signed
    # artifacts across EIGHT shipped files since momwire#1295 phase 4: seven
    # launchers (`launcher_stems()`) plus the engine. The distinct
    # count stays 2 however many variants ship, which is the whole point of
    # copying a signed binary rather than signing each one.
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
    for stem in (*NEC5_LAUNCHERS, *NEC4_LAUNCHERS):
        variant = exe.with_name(f"{stem}{exe.suffix}")
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
        # variants are the copies just asserted. FILES, not distinct
        # binaries — this counts the far side of the variant copy, where
        # sign.py's gates counted the near side. See the note at the
        # `sign_if_configured` call above before reconciling the two.
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

    (bundle / "README.txt").write_text(
        readme_text(release, signing_note, exe.suffix), newline="\r\n"
    )
    print(f"bundle ready: {bundle}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
