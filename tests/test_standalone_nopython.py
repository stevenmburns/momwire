"""momwire-razor2p-standalone (standalone/, MW_EXE_NO_PYTHON, momwire#1512):
the launcher layer with no Python in the process.

The corpus gate (`scripts/core_split/exe_compare.py --standalone`) holds every
deck's printout to `momwire-nec5-razor-2p`'s. These tests hold the launcher
layer the executable now answers itself -- `momwire_eznec_client`'s argv
rules and `_shell`'s frame -- to the launcher's own bytes: the argument
errors, `-version`, SimNEC's streaming call, 4nec2's stdin form, an
unreadable deck, an unwritable printout or `.GMO`, a deck that refuses or
raises, and what the build does with a deck its C++ route declines (a named
refusal) or a C++ exception (the INTERNAL ERROR printout).

The executable needs its OpenBLAS and a CMake build, which only the
`standalone` workflow's Linux and Windows jobs make, so this module runs only
when MOMWIRE_RAZOR2P_STANDALONE_EXE names one. The launcher it is compared
with is this environment's. On Windows (momwire#1558) the build is MSVC's,
with SciPy's OpenBLAS DLL beside it, held to the same bytes: the launcher
layer is text, and these decks print the same digits on both sides.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
EXE = os.environ.get("MOMWIRE_RAZOR2P_STANDALONE_EXE")
# Which formulation the executable is (momwire#1517 Phase A): razor-2p (the
# default), or bspline for `momwire-bs2-standalone`, compared with
# `momwire-nec5-bspline`.
BASIS = os.environ.get("MOMWIRE_STANDALONE_BASIS", "razor-2p")

pytestmark = pytest.mark.skipif(
    not EXE,
    reason="MOMWIRE_RAZOR2P_STANDALONE_EXE does not name a built executable",
)

DIPOLE = FIXTURES / "razor_printout_3a" / "node_gap_two_sources.nec"
REFUSES = FIXTURES / "razor_printout_3a" / "chain_then_refusal.nec"
RAISES = FIXTURES / "razor_printout_3a" / "echo_raises_nxq.nec"
BURIED = FIXTURES / "razor_printout_somm" / "gn0_buried_radial.nec"
EXAMPLE3 = FIXTURES / "4nec2_example3" / "EXAMPLE3.inp"
DECKS = (DIPOLE, REFUSES, RAISES, BURIED, EXAMPLE3)
# What a build does not serve yet, by deck name (none since the Galerkin
# build's crossing decks, momwire#1718 S11b).
_NOT_YET: dict[str, str] = {}


def _skip_unserved(*decks):
    for d in decks:
        if Path(d).name in _NOT_YET:
            pytest.skip(f"not served by this build yet: {_NOT_YET[Path(d).name]}")


def _launcher() -> str:
    name = f"momwire-nec5-{BASIS}"
    found = shutil.which(name) or str(Path(sys.executable).parent / name)
    assert Path(found).exists(), found
    return found


def _both(argv_tail, tmp_path, *, stdin_text=None, prepare=None, env=None):
    """The executable's and the launcher's run of one command line, each in
    its own copy of the working directory: (exe, launcher, exe dir, launcher
    dir). `prepare(d)` shapes each directory first."""
    out = []
    for name, prog in (("exe", EXE), ("launcher", _launcher())):
        d = tmp_path / name
        d.mkdir()
        for f in DECKS:
            shutil.copy(f, d / f.name)
        (d / "EXAMPLE3_EN.nec").write_bytes(EXAMPLE3.read_bytes() + b"EN\n")
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
                    env=dict(os.environ, **(env or {})),
                    capture_output=True,
                    check=False,
                )
            )
        finally:
            if stdin_text is not None:
                stdin.close()
    return out[0], out[1], tmp_path / "exe", tmp_path / "launcher"


def _files(d: Path) -> dict[str, bytes]:
    return {
        p.name: p.read_bytes()
        for p in sorted(d.iterdir())
        if p.is_file() and p.name != "stdin.txt"
    }


def _same(exe, ref, de, dl):
    assert (exe.returncode, exe.stdout) == (ref.returncode, ref.stdout), (
        exe.stdout,
        ref.stdout,
    )
    assert _files(de) == _files(dl)


def _pe_imports(path: str) -> list[str]:
    """The DLLs a Windows executable names in its import and delay-import
    tables, read from the PE file itself (a bare runner has no dumpbin)."""
    import struct

    data = Path(path).read_bytes()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    n_sections = struct.unpack_from("<H", data, pe + 6)[0]
    opt = pe + 24
    sections = opt + struct.unpack_from("<H", data, pe + 20)[0]

    def offset(rva):
        for k in range(n_sections):
            s = sections + 40 * k
            vsize, va, _, raw = struct.unpack_from("<IIII", data, s + 8)
            if va <= rva < va + max(vsize, 1):
                return rva - va + raw
        raise ValueError(rva)

    names = []
    # (data directory, descriptor size, offset of its name RVA): the import
    # table and the delay-import table (the OpenBLAS DLL is delay-loaded)
    for entry, size, name_at in ((1, 20, 12), (13, 32, 4)):
        rva = struct.unpack_from("<I", data, opt + 112 + 8 * entry)[0]
        if not rva:
            continue
        o = offset(rva)
        while name_rva := struct.unpack_from("<I", data, o + name_at)[0]:
            n = offset(name_rva)
            names.append(data[n : data.index(b"\0", n)].decode("ascii"))
            o += size
    return names


def test_no_python_anywhere_in_the_executable():
    if sys.platform == "win32":
        dlls = _pe_imports(EXE)
        assert dlls, "no imports read"
        assert not any("python" in d.lower() for d in dlls), dlls
    else:
        # macOS (momwire#1559) has otool where Linux has ldd
        tool = ["otool", "-L"] if sys.platform == "darwin" else ["ldd"]
        ldd = subprocess.run([*tool, EXE], capture_output=True, text=True, check=True)
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


@pytest.mark.parametrize(
    "deck",
    [DIPOLE, REFUSES, RAISES, EXAMPLE3.with_name("EXAMPLE3_EN.nec"), BURIED],
    ids=["served", "refused", "raised", "gmo", "buried"],
)
def test_a_deck_answers_as_the_launcher(deck, tmp_path):
    _skip_unserved(deck)
    _same(*_both([deck.name, "x.out"], tmp_path))


@pytest.mark.parametrize(
    "argv",
    [
        ["only-one.nec"],
        ["a.nec", "b.out", "c"],
        ["-", "-"],
        ["x.nec", "-"],
        ["-version"],
    ],
    ids=["one argument", "three arguments", "streaming", "streaming out", "version"],
)
def test_the_command_line_edges(argv, tmp_path):
    exe, ref, de, dl = _both(argv, tmp_path)
    _same(exe, ref, de, dl)
    assert exe.stderr.strip() == ref.stderr.strip()


def _cards(deck: Path) -> bytes:
    """A deck's cards before its EN."""
    data = deck.read_bytes()
    return data[: data.rindex(b"EN")]


_ONE = _cards(DIPOLE) + b"NX\n"
_STREAMS = {
    "empty": b"",
    "two structures, no EN": _ONE + _ONE,
    "EN closes it": _ONE + DIPOLE.read_bytes() + _ONE,
    "a refusal mid-stream": REFUSES.read_bytes().replace(b"\nEN", b"\nNX") + _ONE,
    "an exception mid-stream": _ONE + _cards(RAISES) + b"NX\n" + _ONE,
    "stdin closes mid-structure": _ONE + _cards(DIPOLE)[:120],
    "blank lines after the last NX": _ONE + b"\n  \r\n",
    "CRLF": (_ONE + _ONE).replace(b"\n", b"\r\n"),
    "bare CR": (_ONE + _ONE).replace(b"\n", b"\r"),
    "starts on NX": b"NX\n" + _ONE,
    "latin-1 comment": b"CM \xb5 \xe9\n" + _ONE,
    "a buried structure": _cards(BURIED) + b"NX\n" + _ONE,
}


@pytest.mark.parametrize("stream", list(_STREAMS.values()), ids=list(_STREAMS))
def test_the_streaming_call(stream, tmp_path):
    """SimNEC's `- -` (momwire#1512): the same printouts on stdout, nothing
    on stderr, exit 0, no file written."""
    if stream == _STREAMS["a refusal mid-stream"]:
        _skip_unserved(REFUSES)
    runs = []
    for name, prog in (("exe", EXE), ("launcher", _launcher())):
        d = tmp_path / name
        d.mkdir()
        runs.append(
            subprocess.run(
                [prog, "-", "-"], cwd=d, input=stream, capture_output=True, check=False
            )
        )
        assert list(d.iterdir()) == []
    exe, ref = runs
    assert (
        (exe.returncode, exe.stdout) == (ref.returncode, ref.stdout) == (0, ref.stdout)
    )
    assert exe.stderr == b""


def test_the_streaming_call_answers_each_structure_before_the_next(tmp_path):
    """SimNEC's lockstep, through real pipes: one structure written, its
    printout read up to its NX echo before the next is written -- with the
    second line break of a CRLF arriving only with the next structure."""
    sys.path.insert(0, str(ROOT / "scripts" / "core_split"))
    import stream_compare as sc

    crlf = _ONE.replace(b"\n", b"\r\n")
    data = crlf + crlf
    segments = []
    for prog in (EXE, _launcher()):
        writes = [crlf[:-1], b"\n" + crlf]
        r = sc.lockstep([prog], data, dict(os.environ), tmp_path, 120, writes)
        assert r["states"] == ["nx", "nx", "close:eof"], r["states"]
        assert r["rc"] == 0
        segments.append(r["segments"])
    assert segments[0] == segments[1]
    whole = subprocess.run(
        [EXE, "-", "-"], input=data, capture_output=True, check=False
    ).stdout
    assert b"".join(segments[0]) == whole


@pytest.mark.parametrize(
    "stdin_text",
    [
        "EXAMPLE3.inp\nEXAMPLE3.out\n",
        '  "EXAMPLE3.inp"  \r\n\t"EXAMPLE3.out"\r\n',
        "EXAMPLE3.inp\r\nEXAMPLE3.out",  # no final newline
        "EXAMPLE3_EN.nec\nEXAMPLE3.out\nignored\n",
        "EXAMPLE3.inp\n",  # one name
        "EXAMPLE3.inp\n\nEXAMPLE3.out\n",  # a blank line ends it
        "",  # nothing
        "missing.inp\nm.out\n",
        "node_gap_two_sources.nec\nd.out\n",  # a deck with EN, no .GMO
    ],
    ids=[
        "plain",
        "quoted and padded",
        "no final newline",
        "a third line",
        "one name",
        "blank line",
        "empty",
        "missing deck",
        "deck with EN",
    ],
)
def test_the_stdin_form(stdin_text, tmp_path):
    _same(*_both([], tmp_path, stdin_text=stdin_text))


@pytest.mark.parametrize(
    "deck",
    ["missing.nec", "./sub//missing.nec", "sub/", "sub"],
    ids=["missing", "path spelling", "directory slash", "directory"],
)
def test_an_unreadable_deck(deck, tmp_path):
    _same(*_both([deck, "x.out"], tmp_path, prepare=lambda d: (d / "sub").mkdir()))


@pytest.mark.parametrize(
    "out",
    ["nodir/x.out", "sub"],
    ids=["no such directory", "a directory"],
)
def test_an_unwritable_printout(out, tmp_path):
    _same(*_both([DIPOLE.name, out], tmp_path, prepare=lambda d: (d / "sub").mkdir()))


def test_an_unwritable_gmo(tmp_path):
    """The `.GMO` file cannot be written (a directory holds its name): the
    shell's last line of defence writes the INTERNAL ERROR printout naming
    the OSError, with the deck's echo."""
    exe, ref, de, dl = _both(
        ["EXAMPLE3_EN.nec", "x.out"],
        tmp_path,
        prepare=lambda d: (d / "x.GMO").mkdir(),
    )
    _same(exe, ref, de, dl)
    # opening a directory to write is EISDIR on POSIX, EACCES on Windows
    kind = b"PermissionError" if sys.platform == "win32" else b"IsADirectoryError"
    assert kind in (de / "x.out").read_bytes()


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
    """An exception from the C++ side (memory, ...) is `_shell`'s last line of
    defence: the refusal printout with the deck's echo, never a crash."""
    from momwire.eznec import _printout, _shell

    out = tmp_path / "x.out"
    r = subprocess.run(
        [EXE, str(DIPOLE), str(out)],
        env=dict(os.environ, MW_EXE_FAIL=fail),
        capture_output=True,
        check=False,
    )
    assert r.returncode == 0
    want = tmp_path / "want.out"
    text = DIPOLE.read_bytes().decode("latin-1")
    _shell.write_printout(want, _printout.render_refusal(text, line, basis=BASIS))
    got = out.read_bytes()
    # the header's engine stamp is the executable's; the rest is the frame
    assert got.split(b"\r\n")[2:] == want.read_bytes().split(b"\r\n")[2:]
    assert line.encode() in got


# --- MOMWIRE_TIMES -------------------------------------------------------------

TIMED_BURIED = FIXTURES / "momwire_times" / "buried_radials.nec"


@pytest.mark.parametrize("deck", [DIPOLE, TIMED_BURIED], ids=["dipole", "buried"])
def test_momwire_times_is_honoured_as_the_launcher_honours_it(deck, tmp_path):
    """MOMWIRE_TIMES=1: the executable and the launcher each fill MATRIX
    TIMING and RUN TIME with measured seconds and write one `times:` line on
    stderr; every other byte is the untimed printout's. Off, they write the
    zeros (the corpus gates hold that)."""
    _skip_unserved(deck)
    sys.path.insert(0, str(ROOT / "tests"))
    import times_check as tc

    off = tmp_path / "off"
    off.mkdir()
    shutil.copy(deck, off / deck.name)
    subprocess.run([EXE, deck.name, "x.out"], cwd=off, check=True, capture_output=True)
    untimed = (off / "x.out").read_bytes().decode("latin-1")
    assert tc.printout_timing(untimed) == ([(0.0, 0.0)], [0.0])

    for name, prog in (("exe", EXE), ("launcher", _launcher())):
        d = tmp_path / name
        d.mkdir()
        shutil.copy(deck, d / deck.name)
        run = subprocess.run(
            [prog, deck.name, "x.out"],
            cwd=d,
            env=dict(os.environ, MOMWIRE_TIMES="1"),
            capture_output=True,
            check=False,
        )
        assert run.returncode == 0, run.stderr
        printout = (d / "x.out").read_bytes().decode("latin-1")
        assert tc.zero_timing(printout) == untimed, name
        (line,) = tc.times_lines(run.stderr)
        assert line["runs"] == 1 and line["fill_ms"] > 0.0, (name, line)
        ((fill, factor),), (run_time,) = tc.printout_timing(printout)
        assert fill > 0.0 or line["fill_ms"] < 0.5, (name, line)
        assert fill + factor <= run_time + 1e-3, name
        assert run_time * 1e3 <= line["total_ms"] + 1.0, (name, line)
        (env,), phases, wall = tc.timed_extras(printout)
        assert (phases, wall) == (5, 1), name
        assert env[3].startswith("THREADS = "), (name, env)
        if name == "exe":
            assert line["route"] == "cxx" and line["start_ms"] > 0.0, line
            assert abs(fill * 1e3 - line["fill_ms"]) <= 0.5 + 1e-9, line


def test_momwire_times_streams_a_line_per_structure(tmp_path):
    sys.path.insert(0, str(ROOT / "tests"))
    import times_check as tc

    run = subprocess.run(
        [EXE, "-", "-"],
        cwd=tmp_path,
        input=_ONE + _ONE,
        env=dict(os.environ, MOMWIRE_TIMES="1"),
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0
    lines = tc.times_lines(run.stderr)
    assert [c["route"] for c in lines] == ["stream", "stream"]
    assert lines[0]["start_ms"] > 0.0 and lines[1]["start_ms"] == 0.0
    fills, runs = tc.printout_timing(run.stdout)
    assert len(fills) == 2 and runs == []


# --- the transmitted grid's store (momwire#1752 T1) ----------------------------

MIXED = FIXTURES / "transmitted_mixed" / "m3_two_radii.nec"


@pytest.mark.skipif(
    BASIS != "bspline", reason="the transmitted blocks are bs2's (and sg5's) fill"
)
def test_the_transmitted_grid_is_stored_and_moves_no_byte(tmp_path):
    """A mixed deck (wires above and below, no crossing node) fills the
    below->above grid; the executable keeps it in its Sommerfeld store as
    Python keeps it in `_somm_disk_cache`. Run cold, the store holds one
    `trans-*.grid`; a second run reads it (MOMWIRE_TIMES counts no build) and
    leaves it as it was; a damaged file is a miss, refilled to the same bytes;
    every printout is the launcher's."""
    sys.path.insert(0, str(ROOT / "tests"))
    import times_check as tc

    store = tmp_path / "store"
    env = dict(os.environ, MOMWIRE_SOMM_CACHE_DIR=str(store), MOMWIRE_SOMM_CACHE="1")
    ref = subprocess.run(
        [_launcher(), str(MIXED), str(tmp_path / "ref.out")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        check=False,
    )
    assert ref.returncode == 0, ref.stderr
    want = (tmp_path / "ref.out").read_bytes()

    def exe_run(name):
        r = subprocess.run(
            [EXE, str(MIXED), str(tmp_path / f"{name}.out")],
            cwd=tmp_path,
            env=dict(env, MOMWIRE_TIMES="1"),
            capture_output=True,
            check=False,
        )
        assert r.returncode == 0, r.stderr
        got = (tmp_path / f"{name}.out").read_bytes().decode("latin-1")
        assert tc.zero_timing(got).encode("latin-1") == want, name
        (line,) = tc.times_lines(r.stderr)
        return line

    cold = exe_run("cold")
    (grid,) = sorted((store / "exe").glob("*/trans-*.grid"))
    stamp = grid.read_bytes()
    warm = exe_run("warm")
    # Nothing built: since the stores name their files by the key's hash
    # (momwire#1752) every grid of the deck is kept, Windows' MAX_PATH under
    # pytest's deep temporary directory included.
    assert cold["ground_built"] > 0 and warm["ground_built"] == 0, (cold, warm)
    assert warm["ground_disk"] >= 1, warm
    names = sorted(p.name for p in (store / "exe").glob("*/*.grid"))
    assert names and all(len(n) <= 32 for n in names), names
    assert grid.read_bytes() == stamp
    grid.write_bytes(stamp[:-9] + b"\0" * 9)
    exe_run("damaged")
    assert grid.read_bytes() == stamp


# --- the near-interface sheets' store (the perf-1028 study) -------------------

NEAR_BURIED = FIXTURES / "near_buried_1653" / "d1_buried_dipole.nec5.nec"


def test_the_near_field_sheets_are_stored_and_move_no_byte(tmp_path):
    """A buried deck's near field above and below the plane reads the
    designed near-interface tables, whose sheets Python keeps in process
    across renders; the executable keeps them in its Sommerfeld store. Run
    cold, the store holds `nisheet-*.grid`; a second run reads every one
    (MOMWIRE_TIMES counts no build) and leaves the files as they were; a
    damaged file is a miss, refilled to the same bytes; every printout is the
    launcher's."""
    sys.path.insert(0, str(ROOT / "tests"))
    import times_check as tc

    store = tmp_path / "store"
    env = dict(os.environ, MOMWIRE_SOMM_CACHE_DIR=str(store), MOMWIRE_SOMM_CACHE="1")
    ref = subprocess.run(
        [_launcher(), str(NEAR_BURIED), str(tmp_path / "ref.out")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        check=False,
    )
    assert ref.returncode == 0, ref.stderr
    want = (tmp_path / "ref.out").read_bytes()

    def exe_run(name):
        r = subprocess.run(
            [EXE, str(NEAR_BURIED), str(tmp_path / f"{name}.out")],
            cwd=tmp_path,
            env=dict(env, MOMWIRE_TIMES="1"),
            capture_output=True,
            check=False,
        )
        assert r.returncode == 0, r.stderr
        got = (tmp_path / f"{name}.out").read_bytes().decode("latin-1")
        assert tc.zero_timing(got).encode("latin-1") == want, name
        (line,) = tc.times_lines(r.stderr)
        return line

    cold = exe_run("cold")
    sheets = sorted((store / "exe").glob("*/nisheet-*.grid"))
    assert sheets, sorted(p.name for p in store.rglob("*"))
    stamps = [p.read_bytes() for p in sheets]
    warm = exe_run("warm")
    assert cold["ground_built"] > 0 and warm["ground_built"] == 0, (cold, warm)
    assert warm["ground_disk"] >= len(sheets), warm
    assert [p.read_bytes() for p in sheets] == stamps
    sheets[0].write_bytes(stamps[0][:-9] + b"\0" * 9)
    exe_run("damaged")
    assert [p.read_bytes() for p in sheets] == stamps
