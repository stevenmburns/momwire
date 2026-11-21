"""momwire-nec2c-sinusoidal-standalone (standalone/, MW_EXE_BASIS=nec2c_sin,
momwire#2065): the NEC-2 portal's point-matched sinusoidal launcher,
`momwire-nec2c-sinusoidal`, as one Python-free executable.

Every byte is held to the launcher's (this environment's
`momwire-nec2c-sinusoidal`), except the one line each solve fills with
MEASURED time -- MATRIX TIMING's `FILL: <ms> msec  FACTOR: <ms> msec` --
whose two numbers are masked by exact line pattern (`_mask`). Covered:

- every deck of the nec2c corpus (`tests/fixtures/nec_portal`, the 71 decks
  the drop-in is held to against nec2c itself), sent on stdin as SimNEC sends
  it: what the C++ route answers prints the launcher's bytes; what it
  declines is a refusal frame saying "not served by this build" (skipped,
  and named, so the list of what this build does not serve yet is the skip
  list);
- nec2c's command line: `-version`, the banner and NX framing, EN ending a
  run, a deck left open at end of input, CRLF, the `in out` file form, 4nec2's
  two names on stdin, `- -`, the shell's argument errors, an unreadable deck,
  `--legacy-probe`, an unknown `--basis` and `--save-decks`;
- NEC-2's free-end cap (momwire#2033): the launcher's default (on, withdrawn
  with its line where it cannot act), `MOMWIRE_X_NEC2_END_CAP=0` (off) and
  `=1` (the refusing request).

The executable needs its OpenBLAS and a CMake build, which only the
`standalone` workflow's Linux job makes, so this module runs only when
MOMWIRE_NEC2C_SIN_STANDALONE_EXE names one.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from _standalone_version import assert_standalone_line

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "tests" / "fixtures" / "nec_portal"
CORPUS = sorted(CORPUS_DIR.glob("*.deck"))
EXE = os.environ.get("MOMWIRE_NEC2C_SIN_STANDALONE_EXE")
LAUNCHER = "momwire-nec2c-sinusoidal"

pytestmark = pytest.mark.skipif(
    not EXE, reason="MOMWIRE_NEC2C_SIN_STANDALONE_EXE names no built executable"
)

DIPOLE = CORPUS_DIR / "dipole_free_space.deck"
PATTERN = CORPUS_DIR / "dipole_rp_pattern.deck"
# A deck the dialect refuses by name (RP mode 1, the surface wave): the
# ERROR frame. (The corpus's GN 0 decks are not served by this build yet.)
REFUSED_TEXT = (
    b"CE refused by name\nGW 1 9 0 0 -2.5 0 0 2.5 .001\nGE 0\nEX 0 1 5 0 1 0\n"
    b"FR 0 1 0 0 30\nRP 1 1 1 1000 0 0 0 0 1000\nNX\n"
)

FILL = re.compile(rb"^( {31}FILL: )\d+( msec  FACTOR: )\d+( msec)\r?$", re.M)
NOT_SERVED = b"not served by this build"


def _mask(printout: bytes) -> bytes:
    return FILL.sub(rb"\1<t>\2<t>\3", printout)


def _launcher() -> str:
    found = shutil.which(LAUNCHER) or str(Path(sys.executable).parent / LAUNCHER)
    assert Path(found).exists(), found
    return found


def _env(**extra) -> dict:
    env = dict(os.environ, MOMWIRE_THREADS="1", **extra)
    for name, value in extra.items():
        if value is None:
            env.pop(name, None)
    return env


def _run(prog, argv, stdin: bytes, cwd: Path, env: dict):
    return subprocess.run(
        [prog, *argv],
        input=stdin,
        cwd=cwd,
        env=env,
        capture_output=True,
        check=False,
        timeout=600,
    )


def _both(argv, tmp_path, *, stdin=b"", env=None, prepare=None):
    """The executable's and the launcher's run of one command line, each in an
    empty working directory of its own (a first stdin line naming a file
    that exists reads as 4nec2's names)."""
    out = []
    for name, prog in (("exe", EXE), ("launcher", _launcher())):
        d = tmp_path / name
        d.mkdir(parents=True)
        if prepare is not None:
            prepare(d)
        out.append(_run(prog, argv, stdin, d, _env(**(env or {}))))
    return out[0], out[1], tmp_path / "exe", tmp_path / "launcher"


def _files(d: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(d)): _mask(p.read_bytes())
        for p in sorted(d.rglob("*"))
        if p.is_file()
    }


def _same(exe, ref, de=None, dl=None):
    assert exe.returncode == ref.returncode, (exe, ref)
    assert _mask(exe.stdout) == _mask(ref.stdout), (
        exe.stdout[-2000:],
        ref.stdout[-2000:],
    )
    assert exe.stderr == ref.stderr, (exe.stderr, ref.stderr)
    if de is not None:
        assert _files(de) == _files(dl)


def test_no_python_anywhere_in_the_executable():
    ldd = subprocess.run(["ldd", EXE], capture_output=True, text=True, check=True)
    assert "python" not in ldd.stdout.lower(), ldd.stdout
    blob = Path(EXE).read_bytes().lower()
    for word in (
        b"libpython",
        b"pyconfig",
        b"py_initialize",
        b"site-packages",
        b"python",
    ):
        assert word not in blob, word


def test_the_version_line_marks_the_executable():
    """The launcher's line, NAME suffixed `-standalone-<variant>` (momwire#1831)."""
    exe = subprocess.run([EXE, "-version"], capture_output=True, check=True)
    ref = subprocess.run([_launcher(), "-version"], capture_output=True, check=True)
    assert_standalone_line(exe.stdout, ref.stdout)
    assert exe.stdout.startswith(f"NEC2 {LAUNCHER}-standalone-".encode())


@pytest.mark.parametrize("deck", CORPUS, ids=lambda p: p.stem)
def test_a_corpus_deck_answers_as_the_launcher(deck, tmp_path):
    """Each nec2c corpus deck on stdin, as SimNEC sends it: the launcher's
    stdout under the time mask, or a named refusal frame. A deck the
    executable declines is skipped BEFORE the launcher runs."""
    text = deck.read_bytes()
    exe = _run(EXE, [], text, tmp_path, _env())
    assert exe.returncode == 0, exe
    if NOT_SERVED in exe.stdout:
        why = exe.stdout.split(NOT_SERVED + b": ")[1].split(b"\n")[0][:100]
        pytest.skip("not served by this build: " + why.decode())
    ref = _run(_launcher(), [], text, tmp_path, _env())
    _same(exe, ref)


def test_the_corpus_is_mostly_served(tmp_path):
    """Not vacuous: the corpus is answered by the C++ route, not refused."""
    served = 0
    for deck in CORPUS:
        r = _run(EXE, [], deck.read_bytes(), tmp_path, _env())
        served += NOT_SERVED not in r.stdout
    assert served >= 60, served


def _cards(deck: Path) -> bytes:
    """A deck's cards up to (not including) its terminator."""
    lines = deck.read_bytes().splitlines(keepends=True)
    head = [ln for ln in lines if ln.strip()[:2].upper() not in (b"NX", b"EN")]
    return b"".join(head)


_ONE = _cards(DIPOLE) + b"NX\n"
_TWO = _cards(PATTERN) + b"NX\n"
_STREAMS = {
    "empty": b"",
    "two decks, no EN": _ONE + _TWO,
    "EN ends the run": _ONE + _cards(DIPOLE) + b"EN\n" + _ONE,
    "a refused deck mid-stream": _ONE + REFUSED_TEXT + _TWO,
    "a parse refusal": _ONE
    + b"CE\nGW 1 9 0 0 -1 0 0 1 .001\nGE 0\nZZ 1\nXQ\nNX\n"
    + _ONE,
    "stdin closes mid-deck": _ONE + _cards(DIPOLE),
    "blank lines at the end": _ONE + b"\n\n  \n",
    "CRLF": (_ONE + _TWO).replace(b"\n", b"\r\n"),
    "no trailing newline": _ONE + _cards(DIPOLE) + b"NX",
}


@pytest.mark.parametrize("stream", list(_STREAMS.values()), ids=list(_STREAMS))
def test_the_resident_protocol(stream, tmp_path):
    """SimNEC's stdin protocol: the banner, each deck's frame as its NX
    arrives (the NX echo, the banner again), EN ending the run, an open deck
    at end of input closed as EN closes it."""
    _same(*_both([], tmp_path, stdin=stream))


@pytest.mark.parametrize(
    "stream", list(_STREAMS.values())[1:4], ids=list(_STREAMS)[1:4]
)
def test_the_streaming_call(stream, tmp_path):
    """`- -` is the resident protocol itself (momwire#1611)."""
    _same(*_both(["-", "-"], tmp_path, stdin=stream))


def _decks(d: Path) -> None:
    for f in (DIPOLE, PATTERN):
        shutil.copy(f, d / f.name)
    (d / "refused.deck").write_bytes(REFUSED_TEXT)


@pytest.mark.parametrize(
    "argv",
    [
        [DIPOLE.name, "d.out"],
        [PATTERN.name, "p.out"],
        ["refused.deck", "r.out"],
        ["missing.deck", "m.out"],
        ["sub", "s.out"],
        [DIPOLE.name, "nodir/x.out"],
        [DIPOLE.name],
        [DIPOLE.name, "a.out", "b.out"],
        [DIPOLE.name, "-"],
        ["--cache", DIPOLE.name, "c.out"],
    ],
    ids=[
        "file form",
        "a pattern",
        "a refused deck",
        "missing",
        "a directory",
        "unwritable printout",
        "one argument",
        "three arguments",
        "streaming out",
        "--cache is inert",
    ],
)
def test_the_file_form(argv, tmp_path):
    """`momwire-nec2c in out` (momwire#1611): the deck file's printout,
    CRLF, and the shell's argument errors."""

    def prepare(d):
        _decks(d)
        (d / "sub").mkdir()

    _same(*_both(argv, tmp_path, prepare=prepare))


@pytest.mark.parametrize(
    "names",
    [
        f"{DIPOLE.name}\nd.out\n",
        f'  "{DIPOLE.name}"  \r\n\t"d.out"\r\n',
        f"{DIPOLE.name}\n",
        "missing.deck\nm.out\n",
    ],
    ids=["plain", "quoted and padded", "one name", "missing deck"],
)
def test_4nec2s_names_on_stdin(names, tmp_path):
    """4nec2's `ENGINE < nec2d.tmp`: the two names on stdin, the deck then
    closed by EN where it has none."""
    _same(*_both([], tmp_path, stdin=names.encode(), prepare=_decks))


@pytest.mark.parametrize(
    "argv",
    [
        ["--legacy-probe", "-version"],
        ["--basis", "nonesuch", "-version"],
        ["--basis=nonesuch"],
        ["--cache-stats"],
        ["--save-decks"],
        ["--save-decks", "no-such-dir", "-version"],
    ],
)
def test_the_engine_flags(argv, tmp_path):
    _same(*_both(argv, tmp_path))


@pytest.mark.parametrize(
    "argv",
    [
        ["--version"],
        ["-VERSION"],
        ["--basis", "sinusoidal", "-version"],
        ["--cache", "x", "-version"],
    ],
)
def test_every_spelling_of_the_version_probe(argv, tmp_path):
    """`any(a.lstrip("-").lower() == "version")`, after the engine flags."""
    exe, ref, _, _ = _both(argv, tmp_path)
    assert exe.returncode == ref.returncode == 0
    assert_standalone_line(exe.stdout, ref.stdout)


def test_save_decks(tmp_path):
    """`--save-decks DIR` (momwire#1080): each deck received, its terminator
    EN, in a file of its own (named by the time; compared by content)."""
    runs = []
    for name, prog in (("exe", EXE), ("launcher", _launcher())):
        room = tmp_path / name
        (room / "saved").mkdir(parents=True)
        r = _run(prog, ["--save-decks", "saved"], _ONE + _TWO, room, _env())
        runs.append((r, sorted(p.read_bytes() for p in (room / "saved").iterdir())))
    (exe, exe_saved), (ref, ref_saved) = runs
    _same(exe, ref)
    assert exe_saved == ref_saved and len(exe_saved) == 2


@pytest.mark.parametrize("cap", [None, "0", "1"], ids=["default", "off", "on"])
@pytest.mark.parametrize(
    "deck",
    [
        DIPOLE,
        CORPUS_DIR / "dipole_pec_ground.deck",
        CORPUS_DIR / "catalog_wire_w8jk.deck",
    ],
    ids=lambda p: p.stem,
)
def test_the_nec2_end_cap(deck, cap, tmp_path):
    """NEC-2's free-end cap (momwire#2033): on by default in this launcher,
    MOMWIRE_X_NEC2_END_CAP=0 the uncapped answer, =1 the request; each the
    launcher's bytes, and the default not the uncapped answer."""
    env = {"MOMWIRE_X_NEC2_END_CAP": cap}
    exe, ref, _, _ = _both([], tmp_path, stdin=deck.read_bytes(), env=env)
    assert NOT_SERVED not in exe.stdout
    _same(exe, ref)
    if cap is None:
        off = _run(
            EXE, [], deck.read_bytes(), tmp_path, _env(MOMWIRE_X_NEC2_END_CAP="0")
        )
        assert _mask(off.stdout) != _mask(exe.stdout)


def test_the_end_cap_withdrawn_from_a_buried_deck(tmp_path):
    """A wire below a finite ground: the launcher's default runs without the
    cap and says so (`withdrawn_notice`); the request refuses."""
    deck = (
        b"CE buried counterpoise\n"
        b"GW 1 9 0 0 0.1 0 0 5 .001\n"
        b"GW 2 9 0 0 -0.1 3 0 -0.1 .001\n"
        b"GE -1\nGN 2 0 0 0 13 .005\nFR 0 1 0 0 14\nEX 0 1 1 0 1\nXQ\nNX\n"
    )
    for cap in (None, "1"):
        exe, ref, _, _ = _both(
            [], tmp_path / str(cap), stdin=deck, env={"MOMWIRE_X_NEC2_END_CAP": cap}
        )
        if NOT_SERVED in exe.stdout:
            pytest.skip(
                "this build declines the deck: "
                + exe.stdout.split(NOT_SERVED)[1][:80].decode()
            )
        _same(exe, ref)


def test_an_unknown_feature_is_refused():
    """Any other experimental feature turned on refuses before a deck is read
    (exit 2, one sentence), as every executable does."""
    r = subprocess.run(
        [EXE, "-version"],
        env=_env(MOMWIRE_X_NO_SUCH_FEATURE="1"),
        capture_output=True,
        check=False,
    )
    assert r.returncode == 2 and b"no_such_feature" in r.stderr


def test_a_scratch_directory_is_left_clean():
    room = tempfile.mkdtemp(prefix="mw-nec2c-exe-")
    try:
        _run(EXE, [], _ONE, Path(room), _env())
        assert os.listdir(room) == []
    finally:
        shutil.rmtree(room, ignore_errors=True)
