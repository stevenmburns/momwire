"""momwire-nec42-sinusoidal-standalone (standalone/, MW_EXE_BASIS=sin42,
momwire#1607 B4b): the NEC-4.2 slot's sinusoidal launcher as one Python-free
executable; and the same for momwire-nec42-sinusoidal-galerkin-standalone
(MW_EXE_BASIS=sg42, momwire#1718 S4), the Sinusoidal-Galerkin launcher, and
for momwire-nec42-standalone (MW_EXE_BASIS=bs2_nec42, momwire#1607 Phase C2), the
slot's default launcher `momwire-nec42` (bs2).

Every printout is held to `momwire-nec42-sinusoidal`'s (this environment's
launcher) byte for byte, except the two lines the NEC-4.2 slot fills with
MEASURED time -- `FILL= ... SEC.,  FACTOR= ... SEC.` and `RUN TIME =` --
which are masked by exact line pattern, the same number on both sides
(`_mask`). Covered:
- every fixture deck, read as a NEC-4.2 deck: what the C++ route serves prints the
  launcher's bytes; what it declines is a refusal printout saying "not
  served by this build" (the list of what this build does not serve yet);
- the launcher layer: the argument errors, `-version`, a refused '-',
  4nec2's stdin form, an unreadable deck, an unwritable printout, no `.GMO`
  ever (the NEC-4.2 slot writes none), and a C++ exception as the INTERNAL
  ERROR printout in the NEC-4.2 frame;
- SimNEC's streaming call, `- -`: NX chains, a refusal and EN mid-stream,
  CRLF, and stdin closing mid-structure.

The executable needs its OpenBLAS and a CMake build, which only the
`standalone` workflow's Linux job makes, so this module runs only when
MOMWIRE_NEC42_SIN_STANDALONE_EXE (or MOMWIRE_NEC42_SG_STANDALONE_EXE, or
MOMWIRE_NEC42_STANDALONE_EXE) names one.
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

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
NEC42 = FIXTURES / "eznec_nec42"
# The same gates hold `momwire-nec42-sinusoidal-galerkin-standalone`
# (MW_EXE_BASIS=sg42, momwire#1718 S4) to `momwire-nec42-sinusoidal-galerkin`
# when MOMWIRE_NEC42_SG_STANDALONE_EXE names it instead.
# And `momwire-nec42-standalone` (MW_EXE_BASIS=bs2_nec42, momwire#1607 Phase C2)
# to `momwire-nec42`, the slot's default (bs2), under
# MOMWIRE_NEC42_STANDALONE_EXE.
_SG = os.environ.get("MOMWIRE_NEC42_SG_STANDALONE_EXE")
_BS = os.environ.get("MOMWIRE_NEC42_STANDALONE_EXE")
EXE = os.environ.get("MOMWIRE_NEC42_SIN_STANDALONE_EXE") or _SG or _BS
if EXE is not None and EXE == _SG:
    BASIS = "sinusoidal-galerkin"
elif EXE is not None and EXE == _BS:
    BASIS = "bspline"
else:
    BASIS = "sinusoidal"
LAUNCHER = "momwire-nec42" if BASIS == "bspline" else f"momwire-nec42-{BASIS}"

pytestmark = pytest.mark.skipif(
    not EXE,
    reason="none of MOMWIRE_NEC42_SIN_STANDALONE_EXE, "
    "MOMWIRE_NEC42_SG_STANDALONE_EXE, MOMWIRE_NEC42_STANDALONE_EXE names a "
    "built executable",
)

DIPOLE = NEC42 / "0224_dipole-ex0.nec"
CURRENT = NEC42 / "0223_dipole-ex6.nec"
# wires on both sides of the interface with no crossing junction: served by
# the point-matched copy (momwire#1718 S12), not yet by the Galerkin one
GROUND = NEC42 / "probes" / "p9_above_and_buried_gn2.nec"
REFUSED = NEC42 / "refused" / "r1_gd_after_a_run.nec"
DECKS = (DIPOLE, CURRENT, GROUND, REFUSED)

FILL = re.compile(rb"^( {24}FILL=)[ 0-9.]{9}( SEC\.,  FACTOR=)[ 0-9.]{9}( SEC\.)$")
RUN_TIME = re.compile(rb"^( RUN TIME =)[ 0-9.]{10}$")


def _mask(printout: bytes) -> tuple[bytes, int]:
    out, n = [], 0
    for line in re.split(rb"(\r\n)", printout):
        m = FILL.match(line)
        if m:
            line = m.group(1) + b"<t>" + m.group(2) + b"<t>" + m.group(3)
            n += 1
        else:
            m = RUN_TIME.match(line)
            if m:
                line = m.group(1) + b"<t>"
                n += 1
        out.append(line)
    return b"".join(out), n


def _launcher() -> str:
    found = shutil.which(LAUNCHER) or str(Path(sys.executable).parent / LAUNCHER)
    assert Path(found).exists(), found
    return found


# The launcher's resident server, private to this module's process: one
# started by another environment (another variant, say) is not this one.
_RUNTIME = tempfile.mkdtemp(prefix="mw-nec42-exe-")


def _env(**extra) -> dict:
    # MOMWIRE_BELOW_ANCHORS=off: a buried deck's below grid is the exact one
    # on both sides (its anchors follow the frequencies a process has seen)
    return dict(
        os.environ,
        MOMWIRE_THREADS="1",
        MOMWIRE_PORTAL_RUNTIME_DIR=_RUNTIME,
        MOMWIRE_BELOW_ANCHORS="off",
        **extra,
    )


def _both(argv_tail, tmp_path, *, stdin_text=None, prepare=None, env=None):
    """The executable's and the launcher's run of one command line, each in
    its own copy of the working directory."""
    out = []
    for name, prog in (("exe", EXE), ("launcher", _launcher())):
        d = tmp_path / name
        d.mkdir(parents=True)
        for f in DECKS:
            shutil.copy(f, d / f.name)
        if prepare is not None:
            prepare(d)
        stdin = subprocess.DEVNULL
        if stdin_text is not None:
            (d / "stdin.txt").write_bytes(stdin_text.encode())
            stdin = open(d / "stdin.txt", "rb")  # noqa: SIM115 - closed below
        try:
            out.append(
                subprocess.run(
                    [prog, *argv_tail],
                    cwd=d,
                    stdin=stdin,
                    env=_env(**(env or {})),
                    capture_output=True,
                    check=False,
                )
            )
        finally:
            if stdin_text is not None:
                stdin.close()
    return out[0], out[1], tmp_path / "exe", tmp_path / "launcher"


def _files(d: Path) -> dict[str, tuple[bytes, int]]:
    return {
        p.name: _mask(p.read_bytes())
        for p in sorted(d.iterdir())
        if p.is_file() and p.name != "stdin.txt"
    }


def _same(exe, ref, de, dl):
    assert (exe.returncode, exe.stdout) == (ref.returncode, ref.stdout), (
        exe.stdout,
        ref.stdout,
    )
    assert _files(de) == _files(dl)


def test_no_python_anywhere_in_the_executable():
    ldd = subprocess.run(["ldd", EXE], capture_output=True, text=True, check=True)
    assert "python" not in ldd.stdout.lower(), ldd.stdout
    blob = Path(EXE).read_bytes().lower()
    for word in (b"libpython", b"pyconfig", b"py_initialize", b"site-packages"):
        assert word not in blob, word


def test_the_version_line_is_the_launchers():
    exe = subprocess.run([EXE, "-version"], capture_output=True, check=True)
    ref = subprocess.run([_launcher(), "-version"], capture_output=True, check=True)
    assert exe.stdout == ref.stdout
    assert exe.stdout.startswith(f"NEC42 {LAUNCHER} [sb] ".encode())


def _decks(root: Path):
    return [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in (".nec", ".deck")
    ]


def _nec42_decks():
    return _decks(NEC42)


@pytest.mark.parametrize(
    "deck", _decks(FIXTURES), ids=lambda p: str(p.relative_to(FIXTURES))
)
def test_a_fixture_deck_answers_as_the_launcher(deck, tmp_path):
    """Each fixture deck read as a NEC-4.2 deck, the printout file: the
    launcher's bytes under the time mask when the C++ route answers (or the
    slot refuses), else a named refusal. A deck the executable declines is
    skipped BEFORE the launcher runs: a deck this build does not serve is
    never solved by Python here (some are far too large for a test)."""
    exe_out = tmp_path / "exe.out"
    r = subprocess.run([EXE, str(deck), str(exe_out)], env=_env(), capture_output=True)
    assert r.returncode == 0 and r.stdout == b"", r
    exe = exe_out.read_bytes()
    if b"NEC ERROR - not served by this build" in exe:
        why = exe.split(b"not served by this build: ")[1][:80].decode()
        pytest.skip("not served by this build: " + why)
    ref_out = tmp_path / "launcher.out"
    r = subprocess.run(
        [_launcher(), str(deck), str(ref_out)], env=_env(), capture_output=True
    )
    assert r.returncode == 0 and r.stdout == b"", r
    assert _mask(exe) == _mask(ref_out.read_bytes())
    assert not list(tmp_path.glob("*.GMO"))


def test_the_corpus_is_mostly_served(tmp_path):
    """Not vacuous: the free-space NEC-4.2 fixtures are answered by the C++
    route, not refused by name."""
    served = 0
    for deck in _nec42_decks():
        out = tmp_path / "x.out"
        subprocess.run([EXE, str(deck), str(out)], env=_env(), check=True)
        served += b"not served by this build" not in out.read_bytes()
    assert served >= 30, served


@pytest.mark.parametrize(
    "argv",
    [["only-one.nec"], ["a.nec", "b.out", "c"], ["x.nec", "-"], ["-version"]],
    ids=["one argument", "three arguments", "streaming out", "version"],
)
def test_the_command_line_edges(argv, tmp_path):
    exe, ref, de, dl = _both(argv, tmp_path)
    _same(exe, ref, de, dl)
    assert exe.stderr.strip() == ref.stderr.strip()


@pytest.mark.parametrize(
    "stdin_text",
    [
        f"{DIPOLE.name}\nd.out\n",
        f'  "{DIPOLE.name}"  \r\n\t"d.out"\r\n',
        f"{DIPOLE.name}\n",
        "",
        "missing.nec\nm.out\n",
    ],
    ids=["plain", "quoted and padded", "one name", "empty", "missing deck"],
)
def test_the_stdin_form(stdin_text, tmp_path):
    _same(*_both([], tmp_path, stdin_text=stdin_text))


@pytest.mark.parametrize(
    "deck",
    ["missing.nec", "sub/"],
    ids=["missing", "directory"],
)
def test_an_unreadable_deck(deck, tmp_path):
    _same(*_both([deck, "x.out"], tmp_path, prepare=lambda d: (d / "sub").mkdir()))


def test_a_refused_deck_and_an_unserved_one(tmp_path):
    _same(*_both([REFUSED.name, "x.out"], tmp_path))
    run = _both([GROUND.name, "y.out"], tmp_path / "g")
    if BASIS != "sinusoidal-galerkin":
        # the point-matched copy serves the mixed deck as the launcher does
        # (momwire#1718 S12: the transmitted blocks)
        _same(*run)
        assert b"NEC ERROR" not in (run[2] / "y.out").read_bytes()
        return
    # the Galerkin copy serves a wholly-buried deck and declines the mixed one
    # (momwire#1718 S11a), in its own words
    assert (
        b"NEC ERROR - not served by this build: wires on both sides of the interface"
        in (run[2] / "y.out").read_bytes()
    )


def test_an_unwritable_printout(tmp_path):
    _same(*_both([DIPOLE.name, "nodir/x.out"], tmp_path))


@pytest.mark.parametrize(
    "fail, line",
    [
        ("alloc", "INTERNAL ERROR IN MOMWIRE ENGINE - MemoryError: std::bad_alloc"),
        (
            "runtime",
            "INTERNAL ERROR IN MOMWIRE ENGINE - RuntimeError: injected by MW_EXE_FAIL",
        ),
    ],
)
def test_a_cxx_exception_is_the_internal_error_printout(fail, line, tmp_path):
    from momwire.eznec import _printout, _shell

    out = tmp_path / "x.out"
    r = subprocess.run(
        [EXE, str(DIPOLE), str(out)], env=_env(MW_EXE_FAIL=fail), capture_output=True
    )
    assert r.returncode == 0
    want = tmp_path / "want.out"
    text = DIPOLE.read_bytes().decode("latin-1")
    _shell.write_printout(
        want, _printout.render_refusal(text, line, basis=BASIS, dialect="nec4")
    )
    got = out.read_bytes()
    assert got.split(b"\r\n")[2:] == want.read_bytes().split(b"\r\n")[2:]
    assert line.encode() in got


def _printouts(stdout: bytes) -> list[bytes]:
    """A stream's stdout cut into its printouts (each opens "1", then the
    engine stamp)."""
    return [x for x in re.split(rb"(?=1\r\n momwire )", stdout) if x]


def _cards(deck: Path) -> bytes:
    data = deck.read_bytes()
    return data[: data.rindex(b"EN")]


_ONE = _cards(DIPOLE) + b"NX\n"
_TWO = _cards(CURRENT) + b"NX\n"
_STREAMS = {
    "empty": b"",
    "two structures, no EN": _ONE + _TWO,
    "EN closes it": _ONE + DIPOLE.read_bytes() + _ONE,
    "a refusal mid-stream": _ONE + _cards(REFUSED) + b"NX\n" + _ONE,
    "an unserved structure": _ONE + _cards(GROUND) + b"NX\n" + _ONE,
    "stdin closes mid-structure": _ONE + _cards(DIPOLE)[:120],
    "CRLF": (_ONE + _TWO).replace(b"\n", b"\r\n"),
}


@pytest.mark.parametrize("stream", list(_STREAMS.values()), ids=list(_STREAMS))
def test_the_streaming_call(stream, tmp_path):
    """SimNEC's `- -` (momwire#1592): the same printouts on stdout (under the
    time mask), exit 0, no file written. A structure this build does not
    serve is its named refusal, which ends the stream as a refusal does."""
    runs = []
    for name, prog in (("exe", EXE), ("launcher", _launcher())):
        d = tmp_path / name
        d.mkdir()
        runs.append(
            subprocess.run(
                [prog, "-", "-"], cwd=d, input=stream, env=_env(), capture_output=True
            )
        )
        assert list(d.iterdir()) == []
    exe, ref = runs
    assert exe.returncode == ref.returncode == 0
    if b"not served by this build" in exe.stdout:
        # the printouts before it are the launcher's; the unserved structure
        # is the named refusal, and the stream ends there
        mine, theirs = _printouts(exe.stdout), _printouts(ref.stdout)
        assert b"not served by this build" in mine[-1]
        assert [_mask(x) for x in mine[:-1]] == [
            _mask(x) for x in theirs[: len(mine) - 1]
        ]
        return
    assert _mask(exe.stdout) == _mask(ref.stdout)
    assert exe.stderr == b""


# --- the Sommerfeld store (the perf-1028 study's finding 3) --------------------

GN2 = NEC42 / "0231_dipole-10m-gn2.nec"
BURIED = NEC42 / "0239_dipole-buried-gn3.nec"


@pytest.mark.parametrize("deck", [GN2, BURIED], ids=["gn2 masters", "buried table"])
def test_the_ground_store_is_kept_and_moves_no_byte(deck, tmp_path):
    """Every NEC-4.2 executable keeps its Sommerfeld masters (GN 2 above the
    plane) and its below/below table (a buried deck) in the store between
    runs, as the NEC-5 ones do: run cold, the store holds the grids; a second
    run reads them (MOMWIRE_TIMES counts no build) and prints the cold run's
    bytes, which are the launcher's under the time mask. Until the sinusoidal
    route configured the store, its warm run refilled every grid."""
    sys.path.insert(0, str(ROOT / "tests"))
    import times_check as tc

    store = tmp_path / "store"
    env = _env(MOMWIRE_SOMM_CACHE_DIR=str(store), MOMWIRE_SOMM_CACHE="1")
    ref = subprocess.run(
        [_launcher(), str(deck), str(tmp_path / "ref.out")],
        env=env,
        capture_output=True,
        check=False,
    )
    assert ref.returncode == 0, ref.stderr
    want = _mask((tmp_path / "ref.out").read_bytes())

    def exe_run(name):
        r = subprocess.run(
            [EXE, str(deck), str(tmp_path / f"{name}.out")],
            env=dict(env, MOMWIRE_TIMES="1"),
            capture_output=True,
            check=False,
        )
        assert r.returncode == 0, r.stderr
        assert _mask((tmp_path / f"{name}.out").read_bytes()) == want, name
        (line,) = tc.times_lines(r.stderr)
        return line

    cold = exe_run("cold")
    assert sorted((store / "exe").glob("*/*.grid")), "nothing kept"
    warm = exe_run("warm")
    assert cold["ground_built"] > 0 and warm["ground_built"] == 0, (cold, warm)
    assert warm["ground_disk"] >= 1, warm
