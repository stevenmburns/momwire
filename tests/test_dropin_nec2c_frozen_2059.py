"""The NEC-2 engines in the frozen Windows drop-in (momwire#2059).

The bundle used to ship only the NEC-5 and NEC-4.2 launchers, and a copy
renamed ``momwire-nec2c-sinusoidal.exe`` answered as NEC-5: the frozen
engine's ``dialect_from_program_name`` knew ``nec4`` and ``nec5`` only, and
the native launcher read no ``nec2c`` either.  That is momwire#628's failure
reached by a rename.

Now a ``nec2c`` name selects the NEC-2 route at both ends.  The native
launcher (`scripts/eznec_client_c/momwire_eznec_client.c`) under such a name
is a pass-through: it runs the engine on its own stdin and stdout as
``momwire-engine --program <its name> <its argv>``, and the engine
(`scripts/eznec_freeze/entry.py`) hands a ``nec2c`` program name, whole, to
``momwire.dropin.nec2c.main`` -- the function the pip console scripts run.

The gates:

* the name rule, ``nec2c`` read before every other marker;
* **byte identity on the whole nec2c corpus** (``tests/fixtures/nec_portal``)
  in the default basis and in ``sinusoidal``: the engine's entry logic run as
  the frozen exe runs it (``run(sys.argv)``, its CPU and BLAS pins applied
  first) against the pip console script's ``main()`` under the same name,
  each side in its own process, with the solver's wall clock frozen in both
  so that every remaining byte is compared;
* the launcher chain end to end -- compiled client, engine shim running
  ``entry.py`` -- for the ``-version`` probe, a multi-deck NX stream, the
  file form and an unknown basis, each against the pip console script.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from momwire.deck._solver import NEC2C_MARKER, dialect_from_program_name

REPO = Path(__file__).resolve().parent.parent
ENTRY = REPO / "scripts" / "eznec_freeze" / "entry.py"
BUILD_CC = REPO / "scripts" / "eznec_client_c" / "build_cc.sh"
CORPUS_DIR = REPO / "tests" / "fixtures" / "nec_portal"
CORPUS = sorted(CORPUS_DIR.glob("*.deck"))

# The two NEC-2 launchers the bundle ships (build.py's NEC2_LAUNCHERS).
NEC2_STEMS = ("momwire-nec2c", "momwire-nec2c-sinusoidal")
ENGINE_NAME = "momwire-engine"

_CC = shutil.which(os.environ.get("CC", "")) or shutil.which("cc")


# --------------------------------------------------------------------------
# the name rule
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prog", "dialect"),
    [
        ("momwire-nec2c", "nec2"),
        ("momwire-nec2c-sinusoidal.exe", "nec2"),
        (r"C:\momwire\Momwire-NEC2C-Sinusoidal.EXE", "nec2"),
        ("/opt/bin/momwire-nec2c-bs1", "nec2"),
        ("momwire-nec2c-", "nec2"),
        # SimNEC's order: `nec2c` wins over every other marker in the name.
        ("momwire-nec2c-nec42", "nec2"),
        ("momwire-nec5-nec2c", "nec2"),
        # The other families are untouched.
        ("momwire-nec42-sinusoidal.exe", "nec4"),
        ("momwire-nec5-razor2p.exe", "nec5"),
        ("momwire-engine.exe", "nec5"),
        # `nec2` without the `c` is not the marker.
        ("momwire-nec2-sinusoidal", "nec5"),
    ],
)
def test_a_nec2c_name_is_the_nec2_dialect(prog, dialect):
    assert dialect_from_program_name(prog) == dialect


def test_the_c_launcher_spells_the_same_marker_and_flag():
    source = (
        REPO / "scripts" / "eznec_client_c" / "momwire_eznec_client.c"
    ).read_text()
    assert f'#define NEC2C_MARKER "{NEC2C_MARKER}"' in source
    entry = ENTRY.read_text()
    assert 'PROGRAM_FLAG = "--program"' in entry
    assert '#define PROGRAM_FLAG "--program"' in source


def _entry(script: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        timeout=300,
        stdin=subprocess.DEVNULL,
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("how", "args", "line"),
    [
        # A native NEC-2 launcher's forwarding.
        (
            "flag",
            ["--program", r"C:\momwire\momwire-nec2c-sinusoidal.exe", "-version"],
            "NEC2 momwire-nec2c-sinusoidal [sb] ",
        ),
        # `--selftest`'s children: `sys.executable -m momwire.dropin.nec2c`,
        # which in a bundle is the engine itself.
        (
            "module",
            ["-m", "momwire.dropin.nec2c", "-version"],
            "NEC2 momwire-nec2c [sb] ",
        ),
        # A copy of the engine under a NEC-2 name reads its own.
        ("copy", ["-version"], "NEC2 momwire-nec2c-sinusoidal [sb] "),
        # The engine's own name is still the NEC-5 engine: NEC-5's argument
        # error, where the NEC-2 route would read decks off stdin.
        ("plain", [], "ERROR getting input file from command line"),
    ],
)
def test_the_engine_reaches_the_nec2_route_by_each_spelling(how, args, line, tmp_path):
    script = ENTRY
    if how == "copy":
        script = tmp_path / "momwire-nec2c-sinusoidal.exe"
        shutil.copy2(ENTRY, script)
    elif how == "plain":
        script = tmp_path / f"{ENGINE_NAME}.exe"
        shutil.copy2(ENTRY, script)
    proc = _entry(script, args)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith(line), proc.stdout


# --------------------------------------------------------------------------
# the corpus gate: the engine's entry logic against the pip console script
# --------------------------------------------------------------------------

# One process per side and basis.  `entry` loads entry.py by path -- which
# applies the frozen exe's CPU and BLAS pins before NumPy is imported, as the
# bundle does -- and calls its `run(sys.argv)` exactly as the frozen exe's
# `__main__` does.  `pip` calls `momwire.dropin.nec2c.main()` with no
# arguments under the same `sys.argv`, as the console script does.  The
# solver's two clocks (the `FILL:`/`FACTOR:` msec: the portal's own
# `perf_counter` and the LU's `factor_seconds`) are frozen at 0 in BOTH, so
# the comparison is of every byte and needs no mask.
_DRIVER = r"""
import importlib.util, io, json, sys, types

side, prog, entry_path, decks = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
if side == "entry":
    spec = importlib.util.spec_from_file_location("frozen_entry", entry_path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
from momwire.dropin.nec2c import _portal, engine_scope, main

_portal.time = types.SimpleNamespace(perf_counter=lambda: 0.0)
_portal._lapack.factor_seconds = lambda: 0.0
real_out, real_err = sys.stdout, sys.stderr
answers = {}
for path in decks:
    with open(path, encoding="latin-1") as handle:
        text = handle.read()
    sys.argv = [prog]
    sys.stdin, sys.stdout, sys.stderr = io.StringIO(text), io.StringIO(), io.StringIO()
    with engine_scope():
        code = entry.run(sys.argv) if side == "entry" else main()
    answers[path] = [code, sys.stdout.getvalue(), sys.stderr.getvalue()]
    sys.stdout, sys.stderr = real_out, real_err
json.dump(answers, real_out)
"""


# The key under which `_drive` reports what its process left in its cwd.
_LEFTOVERS = "files left in the working directory"


def _drive(side: str, prog: str, decks: list[Path]) -> dict[str, list]:
    """One side's answers, by deck, from a process in an EMPTY directory of
    its own: a first line naming a file that exists there is read as 4nec2's
    names rather than as a card (`momwire_serve_client.is_names_line`), so a
    shared cwd would let one run's output files change the next run's
    reading of a deck."""
    room = tempfile.mkdtemp(prefix="mw-nec2c-drive-")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _DRIVER, side, prog, str(ENTRY), *map(str, decks)],
            capture_output=True,
            text=True,
            timeout=1800,
            stdin=subprocess.DEVNULL,
            cwd=room,
        )
        leftovers = os.listdir(room)
    finally:
        shutil.rmtree(room, ignore_errors=True)
    assert proc.returncode == 0, proc.stderr[-4000:]
    answers = json.loads(proc.stdout)
    answers[_LEFTOVERS] = leftovers
    return answers


def _corpus_answers(stem: str, decks: list[Path]) -> tuple[dict, dict]:
    """(the frozen entry's answers, the pip console script's), by deck."""
    frozen = _drive("entry", rf"C:\momwire\{stem}.exe", decks)
    pip = _drive("pip", f"/venv/bin/{stem}", decks)
    assert set(frozen) == set(pip) == {str(d) for d in decks} | {_LEFTOVERS}
    return frozen, pip


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("stem", NEC2_STEMS)
def test_the_frozen_entry_answers_the_whole_corpus_as_the_pip_launcher(stem):
    assert len(CORPUS) >= 70, "the nec_portal corpus went missing"
    frozen, pip = _corpus_answers(stem, CORPUS)
    decks = sorted(k for k in frozen if k != _LEFTOVERS)
    differ = [Path(k).stem for k in decks if frozen[k] != pip[k]]
    assert differ == [], f"{len(differ)} of {len(decks)} decks differ: {differ}"
    # Every deck was read as cards: no run wrote a file of its own.
    assert frozen.pop(_LEFTOVERS) == pip.pop(_LEFTOVERS) == []
    for key, (code, out, _err) in frozen.items():
        # Not a vacuous pass: every deck produced a printout under the
        # NEC-2 banner, in the basis the name asked for.
        assert code == 0, key
        assert "VERSION:nec2c.ae6ty.momwire" in out, key
        assert ("VERSION:nec2c.ae6ty.momwire.9.1+sin" in out) == stem.endswith(
            "-sinusoidal"
        ), key


# --------------------------------------------------------------------------
# the launcher chain: native client -> engine shim -> entry.py
# --------------------------------------------------------------------------

_ENGINE_SHIM = f'#!/bin/sh\nexec "{sys.executable}" "{ENTRY}" "$@"\n'


@pytest.fixture(scope="module")
def nec2c_bundle():
    """The compiled client copied under both NEC-2 names, beside an engine
    shim that runs entry.py -- the bundle's shape with the freeze left out."""
    if _CC is None:
        pytest.skip("no C compiler on this box; the native client cannot be built")
    room = Path(tempfile.mkdtemp(prefix="mw-nec2c-bundle-"))
    client = room / "momwire-nec5"
    proc = subprocess.run(
        [str(BUILD_CC), str(client)],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PYTHON": sys.executable},
    )
    assert proc.returncode == 0, proc.stderr
    for stem in NEC2_STEMS:
        shutil.copy2(client, room / stem)
    shim = room / ENGINE_NAME
    shim.write_text(_ENGINE_SHIM)
    shim.chmod(0o755)
    yield room
    shutil.rmtree(room, ignore_errors=True)


def _pip(
    stem: str, args: list[str], stdin: bytes, cwd: Path
) -> subprocess.CompletedProcess:
    """The pip console script: `main()` under `stem` as its program name."""
    code = (
        "import sys; from momwire.dropin.nec2c import main; "
        f"sys.argv[0] = {stem!r}; sys.exit(main())"
    )
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        input=stdin,
        capture_output=True,
        timeout=600,
        cwd=cwd,
    )


def _launcher(
    bundle: Path, stem: str, args: list[str], stdin: bytes, cwd: Path
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(bundle / stem), *args],
        input=stdin,
        capture_output=True,
        timeout=600,
        cwd=cwd,
    )


def _canon(data: bytes) -> str:
    """The wall-clock fields masked, as the portal's two-run gates mask them."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "nec_portal_capture_2059", REPO / "scripts" / "nec_portal_capture.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.canonicalize_timings(data.decode("latin-1"))


@pytest.mark.integration
@pytest.mark.parametrize("stem", NEC2_STEMS)
def test_the_launcher_answers_the_version_probe_as_the_pip_launcher(
    nec2c_bundle, stem, tmp_path
):
    """SimNEC's configure probe: the same bytes, the same exit status, and
    the line names the LAUNCHER (`NEC2 momwire-nec2c-sinusoidal [sb] ...`)."""
    got = _launcher(nec2c_bundle, stem, ["-version"], b"", tmp_path)
    want = _pip(stem, ["-version"], b"", tmp_path)
    assert got.returncode == want.returncode == 0
    assert got.stdout == want.stdout
    assert got.stdout.decode().startswith(f"NEC2 {stem} [sb] ")


@pytest.mark.integration
@pytest.mark.parametrize("stem", NEC2_STEMS)
def test_the_launcher_streams_nx_framed_decks_as_the_pip_launcher(
    nec2c_bundle, stem, tmp_path
):
    """SimNEC's resident protocol: three NX-framed decks on one stdin, one
    printout each on stdout, the stream left open until stdin ends."""
    names = (
        "catalog_dipoles_invvee",
        "catalog_beams_moxon",
        "catalog_verticals_vertical",
    )
    stream = b"".join((CORPUS_DIR / f"{n}.deck").read_bytes() for n in names)
    got = _launcher(nec2c_bundle, stem, [], stream, tmp_path)
    want = _pip(stem, [], stream, tmp_path)
    assert got.returncode == want.returncode
    assert _canon(got.stdout) == _canon(want.stdout)
    assert got.stdout.count(b"DATA CARD No:") > 0
    assert got.stdout.count(b"ANTENNA INPUT PARAMETERS") >= len(names)


@pytest.mark.integration
def test_the_launcher_serves_the_file_form_as_the_pip_launcher(nec2c_bundle, tmp_path):
    deck = CORPUS_DIR / "catalog_dipoles_invvee.deck"
    stem = "momwire-nec2c-sinusoidal"
    for side in ("launcher", "pip"):
        room = tmp_path / side
        room.mkdir()
        shutil.copy2(deck, room / "d.nec")
        if side == "launcher":
            proc = _launcher(nec2c_bundle, stem, ["d.nec", "d.out"], b"", room)
        else:
            proc = _pip(stem, ["d.nec", "d.out"], b"", room)
        assert proc.returncode == 0 and proc.stdout == b"", proc.stdout
    got = (tmp_path / "launcher" / "d.out").read_bytes()
    want = (tmp_path / "pip" / "d.out").read_bytes()
    assert b"\r\n" in got
    assert _canon(got) == _canon(want)
    assert b"+sin" in got


@pytest.mark.integration
def test_an_unknown_basis_is_refused_by_name_as_the_pip_launcher(
    nec2c_bundle, tmp_path
):
    """momwire#628: a name that asks for a basis nobody has is refused at the
    probe, naming it, with the pip launcher's exit status -- never served as
    the default."""
    stem = "momwire-nec2c-rzaor"
    shutil.copy2(nec2c_bundle / NEC2_STEMS[0], nec2c_bundle / stem)
    try:
        got = _launcher(nec2c_bundle, stem, ["-version"], b"", tmp_path)
        want = _pip(stem, ["-version"], b"", tmp_path)
    finally:
        (nec2c_bundle / stem).unlink()
    assert got.returncode == want.returncode != 0
    assert got.stdout == want.stdout
    assert b"'rzaor'" in got.stdout


@pytest.mark.integration
def test_a_missing_engine_is_named_on_stdout_with_a_failing_status(tmp_path):
    """stdout is the NEC-2 channel, so a launcher with nothing to run says so
    there and exits non-zero, as the streaming call does."""
    if _CC is None:
        pytest.skip("no C compiler on this box; the native client cannot be built")
    client = tmp_path / "momwire-nec2c"
    proc = subprocess.run(
        [str(BUILD_CC), str(client)],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PYTHON": sys.executable},
    )
    assert proc.returncode == 0, proc.stderr
    got = _launcher(tmp_path, "momwire-nec2c", ["-version"], b"", tmp_path)
    assert got.returncode != 0
    assert b"MOMWIRE ENGINE UNAVAILABLE" in got.stdout
