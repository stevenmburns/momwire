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
  `=1` (the refusing request);
- the reflection-coefficient ground (GN 0): the corpus's two GN 0 decks and
  the hand-written ones in `tests/fixtures/nec2c_gn0` (horizontal, vertical
  and sloping wires, a yagi, a moxon on the sparse route, a long wire filled
  in several bands, loads, grounds from poor soil to near-PEC, two radii,
  the extended kernel, the folded cos-1 shapes, RP 0 / 2 / 3, a frequency
  sweep and a re-armed GN, and the refusals: ground contact, NE), each
  served -- never declined -- at the launcher's bytes under every end-cap
  setting.

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
# ERROR frame.
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


GN0_CORPUS = [
    CORPUS_DIR / "dipole_reflection_ground.deck",
    CORPUS_DIR / "dipole_rp2_cliff_on_the_gn_card.deck",
]
GN0 = sorted((ROOT / "tests" / "fixtures" / "nec2c_gn0").glob("*.deck"))
# the GN 0 decks the launcher refuses: a wire end in the plane (momwire#282
# stage 1, at construction) and a near field over the ground
GN0_REFUSED = {"g13_contact_refused", "g14_contact_one_end", "g15_ne_refused"}
REFL_COEF = b"FINITE GROUND - REFLECTION COEFFICIENT APPROXIMATION"
# The long wire (N = 1801, past `_schedule.mb_rows`' one-band height) is
# about its banded fill, not the cap: it runs at the default cap only.
_CAPS = {None: "default", "0": "off", "1": "on"}
_GN0_CASES = [
    pytest.param(deck, cap, id=f"{deck.stem}-{name}")
    for deck in GN0_CORPUS + GN0
    for cap, name in _CAPS.items()
    if cap is None or deck.stem != "g12_long_wire_banded"
]


@pytest.mark.parametrize(("deck", "cap"), _GN0_CASES)
def test_a_reflection_coefficient_deck_answers_as_the_launcher(deck, cap, tmp_path):
    """GN 0: the Fresnel-weighted image block (`_field_tensor_image_refl`)
    in the C++ fill. Served, never declined -- a decline fails here rather
    than skipping -- and the launcher's bytes under each end-cap setting;
    a deck the launcher refuses, its refusal."""
    exe, ref, _, _ = _both(
        [], tmp_path, stdin=deck.read_bytes(), env={"MOMWIRE_X_NEC2_END_CAP": cap}
    )
    assert NOT_SERVED not in exe.stdout, exe.stdout[-600:]
    _same(exe, ref)
    if deck.stem in GN0_REFUSED:
        assert b"\nERROR: " in exe.stdout
    else:
        assert REFL_COEF in exe.stdout
        assert b"ANTENNA INPUT PARAMETERS" in exe.stdout


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
        # a retired spelling refused by name, as the launcher words it (#2091)
        ["--basis", "razor-nec5", "-version"],
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


# --- the one fill-reuse switch, MOMWIRE_X_PAIR_REUSE (momwire#1911) ----------
#
# Reach 11: this executable runs the point-matched fill's plan
# (`_sin_pair_reuse_core.h`, `EXE_PARTS`). The C++ plan serves the extended
# kernel only (`reuse_route`: the NEC-4.2 slot never builds a reduced-kernel
# fill), so a NEC-2 deck reuses under EK and runs the ordinary fill without
# it, where the launcher's Python plan serves both. The NEC-2 printout
# carries no line 2, so on prints off's bytes wherever the reuse moves Zin
# under a printed digit (its drift, 5e-10, is).
_REUSE_LINE = re.compile(
    rb"^cxx: reuse plans built (\d+), declined (\d+), found (\d+)\r?$", re.M
)
_REDUCED = b"cxx: reuse last fill the reduced-kernel fill"


def _reuse_curtain(ground: str, ek: bool = True) -> bytes:
    """Ten parallel wires of 41 segments with no junction (410 segments,
    past the free-space and PEC small-deck threshold): every band of rows
    repeats."""
    lam = 299.792458 / 28.0
    lines = ["CE pair reuse curtain"]
    for i in range(10):
        x = 0.1 * lam * i
        a, b = (x, 0.0, 1.0), (x + 0.3 * (i % 3), 0.0, 1.0 + 0.5 * lam)
        if i in (1, 4, 7):
            a, b = b, a
        lines.append(
            f"GW {i + 1} 41 {a[0]:.6f} {a[1]:.6f} {a[2]:.6f} "
            f"{b[0]:.6f} {b[1]:.6f} {b[2]:.6f} 0.002"
        )
    lines += ["GE 0"] if ground == "free" else ["GE 1", "GN 1"]
    lines += ["EK"] if ek else []
    lines += ["EX 0 6 21 0 1. 0", "FR 0 3 0 0 28. 0.1", "XQ", "NX"]
    return ("\n".join(lines) + "\n").encode()


def _reuse_counts(stderr: bytes) -> tuple[int, int, int]:
    m = _REUSE_LINE.search(stderr)
    assert m, stderr
    return int(m[1]), int(m[2]), int(m[3])


@pytest.mark.parametrize("ground", ["free", "pec"])
def test_pair_reuse_runs_and_answers_as_the_launcher(ground, tmp_path):
    """Asked on, both sides, an EK deck: the launcher's bytes under the time
    mask. The plan really ran (MW_EXE_TIMES's reuse line: a plan built that
    did not decline, and the last fill served), and on is off's bytes.
    Unset, no plan is built."""
    deck = _reuse_curtain(ground)
    on = {"MOMWIRE_X_PAIR_REUSE": "1"}
    exe, ref, _de, _dl = _both([], tmp_path / "both", stdin=deck, env=on)
    assert NOT_SERVED not in exe.stdout, exe.stdout[-600:]
    _same(exe, ref)
    timed = _run(EXE, [], deck, tmp_path, _env(**on, MW_EXE_TIMES="1"))
    built, declined, found = _reuse_counts(timed.stderr)
    assert built - declined >= 1, (built, declined, found)
    assert b"cxx: reuse last fill served" in timed.stderr, timed.stderr
    off = _run(
        EXE, [], deck, tmp_path, _env(MOMWIRE_X_PAIR_REUSE=None, MW_EXE_TIMES="1")
    )
    assert _reuse_counts(off.stderr)[0] == 0, off.stderr
    assert _mask(off.stdout) == _mask(exe.stdout)


def test_pair_reuse_steps_aside_on_the_reduced_kernel(tmp_path):
    """Without EK the C++ plan declines, by name, before building, and the
    fill is the one with it off; on this deck the launcher's reuse moves no
    printed digit either, so the bytes are the launcher's."""
    deck = _reuse_curtain("free", ek=False)
    on = {"MOMWIRE_X_PAIR_REUSE": "1"}
    exe, ref, _de, _dl = _both([], tmp_path / "both", stdin=deck, env=on)
    assert NOT_SERVED not in exe.stdout, exe.stdout[-600:]
    _same(exe, ref)
    timed = _run(EXE, [], deck, tmp_path, _env(**on, MW_EXE_TIMES="1"))
    assert _reuse_counts(timed.stderr)[0] == 0, timed.stderr
    assert _REDUCED in timed.stderr, timed.stderr
