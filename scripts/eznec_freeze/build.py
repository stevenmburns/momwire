"""Build the Windows drop-in bundle: native launchers plus one frozen engine.

Run from the repo root, in an environment where momwire (this checkout) and
pyinstaller are installed::

    python scripts/eznec_freeze/build.py

Produces ``dist/momwire/`` containing the launchers EZNEC, SimNEC and 4nec2
point at -- ``momwire-nec5[-<basis>][.exe]`` and ``momwire-nec42[-<basis>][.exe]``
-- the single frozen ``momwire-engine[.exe]`` they run, and that engine's
``_internal`` runtime.  The bundle directory must be kept together — a launcher spawns the engine BESIDE it, and
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

# The bundle directory: the zip's one top-level folder, and the workflow's
# `dist/momwire/...` paths.  `momwire-eznec` until momwire#1583 (0.80.0), when
# the bundle stopped naming one host of the three it serves.
NAME = "momwire"

# The one frozen program (the launcher's ENGINE_NAME, restated in smoke.py).
# PyInstaller builds `dist/<engine name>/`; the directory is renamed to NAME
# below.  `momwire-eznec-engine` until momwire#1583: its name carries no
# basis marker, so the plain name selects the default and `--basis` is the
# only thing that moves it.
ENGINE_NAME = "momwire-engine"

# The name the native launcher is COMPILED as before every other launcher is
# copied from it -- the first NEC-5 name, and the smoke's argument.  Every
# launcher is the same bytes, so which one is compiled is bookkeeping.
LAUNCHER = "momwire-nec5"

# The PRIMARY launcher names (momwire#1295 phase 4): one family per EZNEC
# slot, named for the slot rather than for the host.  `momwire-nec5[-<basis>]`
# answers EZNEC's External NEC-5 slot AND SimNEC's NEC-5 engine -- SimNEC
# reads the deck SYNTAX off a substring of the engine path (`nec2c`, then
# `nec5`, momwire#1239) and EZNEC does not read the name at all, so one name
# serves both hosts.  `momwire-nec42[-<basis>]` answers EZNEC's External
# NEC-4.2 slot: the launcher reads `nec4` in its own name as the dialect and
# the basis past `nec42-`.  `nec42` and not `nec4` (momwire#1443): SimNEC
# recognises an engine by `nec42` in its path, as it does `nec5`; a copy
# renamed to carry only `nec4` still answers, its basis read past `nec4-`.
#
# Every one is a COPY of the one native launcher (`entry.py`,
# `momwire_eznec_client.c`), so a copy is a working launcher and costs ~31 KB
# rather than a second 117 MB runtime.  Two per slot, not the whole roster:
# the default plus the one formulation each slot's parity work was about
# (razor-2p is NEC-5's own; point-matched sinusoidal is the family NEC-4.2
# belongs to), and the reference page says how to make any other, which is a
# copy (the README links it; momwire#1593 kept the README to install steps).
#
# Keyed by the file stem, valued by the README table's label.  Kept SHORT: a
# table row in a plain-text README wrapped at ~70 columns; the longer story
# is under WHICH ONE.
NEC5_LAUNCHERS = {
    "momwire-nec5": "the default, degree-2 B-spline (bs2)",
    "momwire-nec5-razor-2p": "the tent basis, razor-blade testing",
}
NEC4_LAUNCHERS = {
    "momwire-nec42": "the default, degree-2 B-spline (bs2)",
    "momwire-nec42-sinusoidal": "the sinusoidal basis, point matched",
}

# The bases copied as `<LAUNCHER>-<basis>` beside the compiled launcher,
# derived from the NEC-5 table so the two cannot disagree; the smoke reads
# this tuple to gate them (its gate 4).  The deprecated `momwire-eznec[-<basis>]`
# launchers that used to be these copies were dropped in momwire#1583 (0.80.0).
SHIPPED_VARIANTS = tuple(
    stem.removeprefix(f"{LAUNCHER}-")
    for stem in NEC5_LAUNCHERS
    if stem.startswith(f"{LAUNCHER}-")
)

# Which bundle this freeze is (momwire#1456).  `standard` is the release zip,
# frozen with python.org's 3.12; `win7` is the same code frozen with the
# community PythonWin7 interpreter, whose python312.dll starts on Windows 7.
# Read from MOMWIRE_BUNDLE_TARGET; only the README differs between them --
# what makes the bundle Win7-capable is the interpreter the workflow
# installs, and scan_win7_imports.py is the gate on that.
TARGETS = ("standard", "win7")


def win7_prerequisites(ucrt_bundled: bool) -> str:
    """The Windows 7 README's prerequisites block (momwire#1456).

    Whether the freeze collected Microsoft's app-local UCRT depends on the
    build machine (the CI runner's freeze does; a desktop's may not), so the
    UCRT line says which this bundle is rather than asserting either.
    """
    if ucrt_bundled:
        ucrt = (
            "  - The Universal C Runtime: this bundle carries its own copy\n"
            "    (_internal\\ucrtbase.dll), so nothing to install.  If it\n"
            "    still will not start, install KB2999226 or the Microsoft\n"
            "    Visual C++ 2015-2022 x64 redistributable.\n"
        )
    else:
        ucrt = (
            "  - The Universal C Runtime: KB2999226, or the Microsoft Visual C++\n"
            "    2015-2022 x64 redistributable, which installs it.  This bundle\n"
            "    does not carry its own copy.\n"
        )
    return (
        "WINDOWS 7 BUILD.  This bundle is frozen with the community PythonWin7\n"
        "interpreter so that it starts on Windows 7.  It needs, on the Windows 7\n"
        "PC:\n"
        "\n"
        "  - Windows 7 SP1, 64-bit.\n"
        "  - KB2533623 (adds AddDllDirectory; any SP1 PC kept up to date with\n"
        "    Windows Update has it).\n"
        f"{ucrt}"
        "\n"
        "On Windows 10 or 11, use the regular momwire-windows.zip instead.\n"
        "\n"
    )


def launcher_stems() -> tuple[str, ...]:
    """Every launcher the bundle ships, by file stem: the NEC-5 family, then
    the NEC-4.2 one.  The engine is not a launcher."""
    return (*NEC5_LAUNCHERS, *NEC4_LAUNCHERS)


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
    blocks = (
        (
            "NEC-5 (EZNEC's External NEC-5, SimNEC's NEC-5, 4nec2's NEC-5):",
            NEC5_LAUNCHERS,
        ),
        ("NEC-4.2 (EZNEC's External NEC-4.2):", NEC4_LAUNCHERS),
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


def readme_text(
    release: str,
    signing_note: str,
    suffix: str,
    target: str = "standard",
    ucrt_bundled: bool = False,
) -> str:
    """The bundle's README.txt, LF line endings (the caller writes CRLF).

    A function of its arguments and the module's constants, so it can be
    gated without a freeze (`tests/test_eznec_bundle_readme.py`).
    ``target`` is one of `TARGETS`; the Windows 7 bundle carries its
    prerequisites up front (momwire#1456), worded by ``ucrt_bundled``.
    """
    if target not in TARGETS:
        raise ValueError(f"unknown bundle target {target!r}; expected one of {TARGETS}")
    win7 = win7_prerequisites(ucrt_bundled) if target == "win7" else ""
    return (
        f"momwire {release} — momwire as a NEC-5 or NEC-4.2 engine for the\n"
        "antenna programs that run an external one: EZNEC Pro+ v7 (External\n"
        "NEC-5 and External NEC-4.2), SimNEC (its NEC-5 engine) and 4nec2\n"
        "(its NEC-5 package).  The program writes a deck, runs the engine,\n"
        "and reads back the printout; momwire answers in its place.\n"
        "\n"
        f"{signing_note}\n"
        "\n"
        f"{win7}"
        "NAMES CHANGED IN 0.80.0.  Earlier releases used names that said\n"
        '"eznec".  If your program\'s engine path points at an old name,\n'
        "change it to the new one:\n"
        "\n"
        "  zip     momwire-eznec-windows.zip       -> momwire-windows.zip\n"
        "          momwire-eznec-windows-win7.zip  -> momwire-windows-win7.zip\n"
        "  folder  momwire-eznec\\                  -> momwire\\\n"
        "  engine  momwire-eznec-engine.exe        -> momwire-engine.exe\n"
        "  REMOVED momwire-eznec.exe               -> use momwire-nec5.exe\n"
        "  REMOVED momwire-eznec-razor-2p.exe      -> use momwire-nec5-razor-2p.exe\n"
        "  REMOVED momwire-eznec-razor-nec5.exe    -> use momwire-nec5-razor-2p.exe\n"
        "\n"
        "The momwire-nec5 and momwire-nec42 launcher names are unchanged.\n"
        "\n"
        "INSTALL INTO A FOLDER OF ITS OWN.  The zip holds a momwire\\ folder;\n"
        "extract it somewhere dedicated, e.g. C:\\momwire\\ - NOT into your\n"
        "antenna program's own folder, and NOT over a previous release.  (For\n"
        "4nec2, see INSTALLING FOR 4NEC2 below, which is the exception.)\n"
        "\n"
        "FOR EVERY NEW RELEASE: delete that folder ENTIRELY, extract the new\n"
        "zip in its place, and leave your program's engine path unchanged.\n"
        "Never extract over an old install: that leaves the old release's files\n"
        "behind (libraries in _internal among them), mixed in with the new\n"
        "ones.  If it happens anyway, line 2 of the printout ends in\n"
        "'(stale: <old version>)' - delete the folder and extract again.\n"
        "\n"
        "If Windows says a file is in use while you delete the folder, that\n"
        "is the resident engine (see WHY IT IS FAST below).  Wait 15 minutes\n"
        "for it to retire, or end momwire-engine.exe in Task Manager.\n"
        "\n"
        "BEFORE EXTRACTING: right-click the downloaded zip, Properties, tick\n"
        '"Unblock", OK - then extract.  That clears the mark-of-the-web on every\n'
        "file inside at once, so the SmartScreen screen below never appears,\n"
        "and the files keep their original dates instead of the download's.\n"
        "(Thanks to Dan AC6LA for the tip.)\n"
        "\n"
        'FIRST RUN (if you did not "unblock" as described above):\n'
        'Windows will show "Windows protected your PC", and you\n'
        'have to click "More info" and then "Run anyway". Nothing is wrong\n'
        "with the download.  That screen is SmartScreen saying this file has no\n"
        'REPUTATION yet — click "More info" and it names the publisher, because\n'
        "the signature above is valid and Windows is reading it.  Reputation is\n"
        "earned by download volume, so it fades with time and starts over with\n"
        'each release; the certificate that would skip the wait outright ("EV")\n'
        "is not one the signing service used here issues.\n"
        "\n"
        "Point your program's engine setting at a LAUNCHER from the matching\n"
        "family below, in this folder.  The name picks the deck dialect: a\n"
        "momwire-nec5 name answers a NEC-5 engine setting and a momwire-nec42\n"
        "name a NEC-4.2 one.  In EZNEC that means choosing External NEC-5 or\n"
        "External NEC-4.2 and setting its engine path; SimNEC and 4nec2 are\n"
        "below.  Keep the folder together: a launcher runs the engine beside\n"
        "it, and the engine needs its _internal runtime beside IT.\n"
        "\n"
        f"{launcher_table(suffix)}\n"
        "\n"
        "Point your program at a launcher, never at momwire-engine: the engine\n"
        "is what the launchers run, and naming it directly gives up the warm\n"
        "start below for nothing.\n"
        "\n"
        "WHY IT IS FAST.  A launcher keeps a warm engine RESIDENT — started on\n"
        "the first calculation, retired after 15 idle minutes — so every launch\n"
        "after the first costs milliseconds instead of the engine's own\n"
        "start-up.  A program launches the engine once per frequency point, so\n"
        "that is most of a sweep.  If anything about the resident path fails, the\n"
        "launcher runs the engine directly instead: the same answer, at the old\n"
        "speed.\n"
        "\n"
        "THE FIRST CALCULATION IS THE SLOW ONE, and your program shows nothing\n"
        "while it happens: several seconds, once, to start that engine.  It is\n"
        "not a hang — wait for it and every calculation after it is the fast one.\n"
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
        "licensed engine to a few thousandths of an ohm.\n"
        "\n"
        "In the NEC-4.2 slot, momwire-nec42-sinusoidal is the sinusoidal\n"
        "basis with point matching, the basis family NEC-4.2 itself belongs\n"
        "to.  razor cannot serve this slot: NEC-4.2 decks put their sources\n"
        "at segment centres and razor places sources at knots, so a\n"
        "momwire-nec42-razor-2p copy refuses, by name, in the printout.\n"
        "\n"
        "SIMNEC.  Point SimNEC's NEC-5 engine setting at momwire-nec5.exe or\n"
        "momwire-nec5-razor-2p.exe.  SimNEC decides which deck syntax to write\n"
        "from the engine's NAME, which must contain nec5, as both of these\n"
        "do.  Same engine, same warm start; a path with no spaces in it is\n"
        "safest.  SimNEC recognises a NEC-4.2 engine by nec42 in its name,\n"
        "which is why the NEC-4.2 launchers carry it; a copy named with just\n"
        "nec4 still answers the NEC-4.2 slot in EZNEC.\n"
        "\n"
        "4NEC2.  The momwire-nec5 launchers also answer 4nec2's NEC-5 package.\n"
        "4nec2 runs its engine with the two file names on its input\n"
        "(ENGINE.exe < nec2d.tmp) rather than on the command line: given no\n"
        "arguments, a launcher reads the deck and printout names from there\n"
        "(quoted or not), accepts the deck without an EN card, and writes the\n"
        ".GMO geometry file beside the printout when the deck's GE card asks\n"
        "(GE 0 0 1, or any nonzero third field).\n"
        "\n"
        "INSTALLING FOR 4NEC2.  This assumes 4nec2's NEC-5 package (the\n"
        "NEC5CL_x13 setup with Edit_inp.exe and Edit_out.exe) is installed\n"
        "and working; install it first if it is not.  Then:\n"
        "  1. Copy the WHOLE CONTENTS of this momwire folder - every\n"
        "     .exe and the _internal folder, not the .zip itself - into\n"
        "     4nec2's exe folder (e.g. C:\\4nec2\\exe).  They share no names\n"
        "     with 4nec2's own files.  A launcher copied there ALONE does not\n"
        "     work: it runs the engine beside it, and answers 'NEC ERROR -\n"
        "     MOMWIRE ENGINE UNAVAILABLE - no engine at ...' instead.\n"
        "  2. In that folder, edit 4nec2.bat and change ONLY the engine line,\n"
        "     the one between Edit_inp.exe and Edit_out.exe, to:\n"
        "         momwire-nec5-razor-2p.exe < nec2d.tmp\n"
        "     (or momwire-nec5.exe < nec2d.tmp).  Keep the Edit_inp.exe and\n"
        "     Edit_out.exe lines: 4nec2 reads its results through them.\n"
        "  3. Leave 4nec2's Settings > NEC-Engine choices alone.  Naming a\n"
        "     launcher under Manual select does NOT work (reported by AC6LA,\n"
        "     momwire#1503); the .bat is how 4nec2's NEC-5 path runs an\n"
        "     engine.\n"
        "For a new release, delete the copied files and _internal first, as\n"
        "above; the edited 4nec2.bat line can stay.\n"
        "\n"
        "Everything else -- how the launchers compare, making a copy that\n"
        "names another basis, and what serves and what refuses (by name,\n"
        "in the printout):\n"
        "https://momwire.dev/reference/eznec-nec5/\n"
    )


def main() -> int:
    # Read ONCE: the baked constant, the README's title line and (through the
    # launcher's own build script) the server key all name the same release.
    release = version("momwire")
    # Checked before the freeze, so a mistyped target fails in seconds rather
    # than after PyInstaller has run.
    target = os.environ.get("MOMWIRE_BUNDLE_TARGET", "standard")
    if target not in TARGETS:
        print(f"ERROR: MOMWIRE_BUNDLE_TARGET={target!r} is not one of {TARGETS}")
        return 1
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
        # numpy's version, read from its `.dist-info` by
        # `momwire_serve_client.pin_cpu_env` (entry.py) to name numpy's
        # AVX-512 targets in that release's own vocabulary. Without it the
        # pin names both vocabularies and still works, at the cost of an
        # (ignored) ImportWarning for the names this numpy does not know.
        "--copy-metadata",
        "numpy",
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

    # PyInstaller named the directory after the program; the BUNDLE is named
    # NAME, which is the zip's top-level folder and the workflow's
    # `dist/momwire/...` paths.  Renaming the outer directory is safe because a
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

    # The launcher, compiled into the bundle under LAUNCHER, the default
    # NEC-5 name.  Its suffix is the engine's: one OS, one spelling of
    # "executable".
    exe = bundle / f"{LAUNCHER}{engine.suffix}"
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
    # release — two launcher names shared SHA256 111B8A48…, 192,768 bytes),
    # so there are two DISTINCT signed artifacts across FIVE shipped files
    # since momwire#1583 dropped the three deprecated names: four launchers
    # (`launcher_stems()`) plus the engine. The distinct
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
    for stem in launcher_stems():
        if stem == LAUNCHER:
            continue
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
        # The engine and the compiled launcher were signed directly; the
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

    ucrt_bundled = any(bundle.rglob("ucrtbase.dll"))
    (bundle / "README.txt").write_text(
        readme_text(release, signing_note, exe.suffix, target, ucrt_bundled),
        newline="\r\n",
    )
    print(f"bundle ready: {bundle}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
