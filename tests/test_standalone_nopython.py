"""momwire-nec5-razor-2p-standalone (standalone/, MW_EXE_NO_PYTHON, momwire#1512):
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

import ast
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from _standalone_version import assert_standalone_line

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
SPLIT = ROOT / "scripts" / "core_split"
EXE = os.environ.get("MOMWIRE_RAZOR2P_STANDALONE_EXE")
# Which formulation the executable is (momwire#1517 Phase A): razor-2p (the
# default), or bspline for `momwire-nec5-bspline-standalone`, compared with
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
# What a build does not serve yet, by deck name (none today).
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


# The platform rule the fixture-deck gates apply (standalone.yml's
# `exe_compare.py --standalone [--rtol win32|macos]`, momwire#1558 and #1559):
# byte for byte on Linux; on Windows (MSVC /fp:fast) and macOS a printout
# that is not the launcher's bytes may still pass within PRINTOUT_RTOL_WIN32
# or PRINTOUT_RTOL_MACOS by `printout_tolerance.ratio`. Both bars are read out
# of exe_compare.py's source, not imported (importing it pins this process's
# CPU environment and loads numpy), so this gate and that one cannot part.
_PLATFORM_RTOL = {"win32": "PRINTOUT_RTOL_WIN32", "darwin": "PRINTOUT_RTOL_MACOS"}


def _printout_rtol() -> float | None:
    """This platform's bar, or None where the rule is byte for byte."""
    name = _PLATFORM_RTOL.get(sys.platform)
    if name is None:
        return None
    for node in ast.parse((SPLIT / "exe_compare.py").read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return float(ast.literal_eval(node.value))
    raise AssertionError(f"exe_compare.py no longer defines {name}")


def _printout_tolerance():
    spec = importlib.util.spec_from_file_location(
        "core_split_printout_tolerance", SPLIT / "printout_tolerance.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _same_by_platform(exe, ref, de, dl):
    """`_same` under the fixture gates' platform rule: the exit status, the
    standard output and the file names exactly; each file the launcher's
    bytes on Linux, and elsewhere its bytes or within this platform's bar."""
    rtol = _printout_rtol()
    if rtol is None:
        _same(exe, ref, de, dl)
        return
    assert (exe.returncode, exe.stdout) == (ref.returncode, ref.stdout), (
        exe.stdout,
        ref.stdout,
    )
    got, want = _files(de), _files(dl)
    assert sorted(got) == sorted(want), (sorted(got), sorted(want))
    pt = _printout_tolerance()
    for name, ref_bytes in want.items():
        if got[name] == ref_bytes:
            continue
        q = pt.ratio(ref_bytes, got[name])
        assert q is not None and q <= rtol, (name, q, rtol)


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
    ],
    ids=["one argument", "three arguments", "streaming", "streaming out"],
)
def test_the_command_line_edges(argv, tmp_path):
    exe, ref, de, dl = _both(argv, tmp_path)
    _same(exe, ref, de, dl)
    assert exe.stderr.strip() == ref.stderr.strip()


def test_the_version_line_marks_the_executable(tmp_path):
    """The launcher's line, NAME suffixed `-standalone-<variant>` (momwire#1831):
    nothing else differs, and neither writes a file or a word on stderr."""
    exe, ref, de, dl = _both(["-version"], tmp_path)
    assert exe.returncode == ref.returncode == 0
    assert_standalone_line(exe.stdout, ref.stdout)
    assert exe.stderr.strip() == ref.stderr.strip()
    assert sorted(p.name for p in de.iterdir()) == sorted(p.name for p in dl.iterdir())


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
    from momwire.dropin.nec5 import _printout, _shell

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


@pytest.mark.skipif(BASIS != "bspline", reason="the transmitted blocks are bs2's fill")
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


MIXED_MORE = FIXTURES / "transmitted_store" / "m3_plus_short_vertical.nec"


@pytest.mark.skipif(BASIS != "bspline", reason="the transmitted blocks are bs2's fill")
def test_the_transmitted_grid_fills_only_what_its_reads_add(tmp_path):
    """The executable fills the below->above grid node by node, only where
    the deck's pairs read it. A second deck on the same grid key (the same
    buried wire, a second vertical) reads more of the lattice: it fills what
    it adds and writes the union back (still one `trans-*.grid`), and the
    first deck run again builds nothing. Every printout is the launcher's,
    which fills the whole grid."""
    sys.path.insert(0, str(ROOT / "tests"))
    import times_check as tc

    store = tmp_path / "store"
    env = dict(os.environ, MOMWIRE_SOMM_CACHE_DIR=str(store), MOMWIRE_SOMM_CACHE="1")
    want = {}
    for deck in (MIXED, MIXED_MORE):
        ref = subprocess.run(
            [_launcher(), str(deck), str(tmp_path / f"{deck.stem}.ref")],
            cwd=tmp_path,
            env=dict(env, MOMWIRE_SOMM_CACHE_DIR=str(tmp_path / "ref-store")),
            capture_output=True,
            check=False,
        )
        assert ref.returncode == 0, ref.stderr
        want[deck] = (tmp_path / f"{deck.stem}.ref").read_bytes()

    def exe_run(deck, name):
        r = subprocess.run(
            [EXE, str(deck), str(tmp_path / f"{name}.out")],
            cwd=tmp_path,
            env=dict(env, MOMWIRE_TIMES="1"),
            capture_output=True,
            check=False,
        )
        assert r.returncode == 0, r.stderr
        got = (tmp_path / f"{name}.out").read_bytes().decode("latin-1")
        assert tc.zero_timing(got).encode("latin-1") == want[deck], name
        (line,) = tc.times_lines(r.stderr)
        return line

    exe_run(MIXED, "first")
    (grid,) = sorted((store / "exe").glob("*/trans-*.grid"))
    first = grid.read_bytes()
    more = exe_run(MIXED_MORE, "more")
    assert more["ground_built"] > 0, more
    assert sorted((store / "exe").glob("*/trans-*.grid")) == [grid]
    assert len(grid.read_bytes()) == len(first) + 8 and grid.read_bytes() != first
    again = exe_run(MIXED, "again")
    assert again["ground_built"] == 0 and again["ground_disk"] >= 1, again


LAZY_ABOVE = FIXTURES / "lazy_above" / "gn2_dipole41.nec"
LAZY_ABOVE_READER = {
    "bspline": "bspline sommerfeld remainder",
}


@pytest.mark.skipif(
    BASIS not in LAZY_ABOVE_READER,
    reason="razor-2p's windows took their census already (momwire#1512)",
)
def test_the_above_remainder_fills_only_the_nodes_it_reads(tmp_path):
    """bs2's above-ground remainder reads a
    LAZY grid: a census of the block's and the graded pairs' reads names the
    nodes, and only those are filled (a node is its own contour, so its value
    is the whole grid's). Every read is checked finite (a node not filled
    reads NaN), the printout is the eager grid's (MOMWIRE_SOMM_LAZY=0) and the
    launcher's byte for byte."""
    sys.path.insert(0, str(ROOT / "tests"))
    import times_check as tc

    def run(prog, name, **extra):
        env = dict(
            os.environ,
            MOMWIRE_SOMM_CACHE_DIR=str(tmp_path / f"store-{name}"),
            MOMWIRE_SOMM_CACHE="1",
            MOMWIRE_TIMES="1",
            **extra,
        )
        r = subprocess.run(
            [prog, str(LAZY_ABOVE), str(tmp_path / f"{name}.out")],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            check=False,
        )
        assert r.returncode == 0, r.stderr
        text = (tmp_path / f"{name}.out").read_bytes().decode("latin-1")
        return tc.zero_timing(text), r.stderr.decode("latin-1")

    lazy, err = run(EXE, "lazy", MOMWIRE_SOMM_LAZY_CHECK="1")
    assert f"LAZY_CHECK ok: {LAZY_ABOVE_READER[BASIS]}" in err, err
    eager, _ = run(EXE, "eager", MOMWIRE_SOMM_LAZY="0")
    assert lazy == eager
    ref, _ = run(_launcher(), "ref")
    assert lazy == ref


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


# --- the huge-page allocator's bookkeeping (Linux) --------------------------------

# A preloaded shim: after every large anonymous mmap (a block of 4 MiB and up
# is one), the next four 48-byte mallocs -- a std::map node's size -- come
# back on a 2 MiB boundary, as glibc's malloc now and then places one. The
# executable's bookkeeping node for that block is then such a node.
_ALIGN_SHIM = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <sys/mman.h>
#include <sys/types.h>
extern void *__libc_malloc(size_t);
extern void *__libc_memalign(size_t, size_t);
static int pending;
void *mmap(void *a, size_t len, int prot, int flags, int fd, off_t off) {
    static void *(*real)(void *, size_t, int, int, int, off_t);
    if (!real) real = (void *(*)(void *, size_t, int, int, int, off_t))dlsym(RTLD_NEXT, "mmap");
    if (len >= ((size_t)6 << 20) && (flags & MAP_ANONYMOUS)) pending = 4;
    return real(a, len, prot, flags, fd, off);
}
void *malloc(size_t n) {
    if (n == 48 && pending > 0) {
        pending--;
        return __libc_memalign((size_t)2 << 20, n);
    }
    return __libc_malloc(n);
}
"""

# A free-space dipole big enough that its matrix is a block of 4 MiB and up.
_BIG_DIPOLE = (
    "CM free-space dipole, 601 segments\nCE\nGW 1 601 0 0 -5 0 0 5 0.001\nGE 0\n"
    "FR 0 1 0 0 14.0 0\nEX 0 1 301 0 1 0\nXQ\nEN\n"
)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="the allocator is Linux's"
)
def test_a_bookkeeping_node_on_a_huge_page_boundary_does_not_deadlock(tmp_path):
    """The executable maps each block of 4 MiB and up on its own and keeps a
    map of them under a mutex. Freeing a block erased its node under that
    mutex, the node's free went back through the executable's operator
    delete, and a node on a 2 MiB boundary passed the block test and locked
    the mutex again: the run hung for good, every thread in a futex wait.
    With the shim forcing that boundary the run must still answer, and
    answer as it does without it."""
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler for the shim")
    shim = tmp_path / "align_shim.so"
    src = tmp_path / "align_shim.c"
    src.write_text(_ALIGN_SHIM)
    subprocess.run(
        [cc, "-shared", "-fPIC", "-O2", "-o", str(shim), str(src), "-ldl"], check=True
    )
    (tmp_path / "big.nec").write_text(_BIG_DIPOLE, newline="")
    env = dict(os.environ, MOMWIRE_THREADS="1")
    plain = subprocess.run(
        [EXE, "big.nec", "plain.out"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=120,
    )
    assert plain.returncode == 0, plain.stderr
    shimmed = subprocess.run(
        [EXE, "big.nec", "shim.out"],
        cwd=tmp_path,
        env=dict(env, LD_PRELOAD=str(shim)),
        capture_output=True,
        timeout=120,
    )
    assert shimmed.returncode == 0, shimmed.stderr
    assert (tmp_path / "shim.out").read_bytes() == (tmp_path / "plain.out").read_bytes()


# --- momwire#1860's levers, on (reach 8: the executables run them) ----------------

_LEVERS_1860 = (
    "MOMWIRE_X_FACTORED_COLUMNS",
    "MOMWIRE_X_HEIGHT_SHEET_RECAL",
    "MOMWIRE_X_CROSSING_TABLE",
    "MOMWIRE_X_CROSSING_ABOVE_RULE",
)


@pytest.mark.parametrize(
    "on", [(), _LEVERS_1860, None], ids=["off", "all four on", "defaults"]
)
def test_the_1860_levers_answer_as_the_launcher(on, tmp_path):
    """The synthetic EFHW of momwire#1860 (a buried rod under a vertical, a
    counterpoise at 1 cm: every lever fires on it) with the levers off ("0"),
    on, and unset (L1 and L2 on by default, L5 and B1 off): the executable's
    printout is the launcher's, line 2 naming on both sides only the levers
    on whose default is off (crossing_table, crossing_above_rule)."""
    sys.path.insert(0, str(ROOT / "tests"))
    from test_crossing_table_1860 import DECKS as DECKS_1860

    def prepare(d):
        (d / "efhw_lo.nec").write_text(DECKS_1860["efhw_lo"], newline="")

    if on is None:
        env = {}
    else:
        env = {var: ("1" if var in on else "0") for var in _LEVERS_1860}
    exe, ref, de, dl = _both(
        ["efhw_lo.nec", "efhw_lo.out"], tmp_path, prepare=prepare, env=env
    )
    assert exe.returncode == 0, exe.stderr
    _same(exe, ref, de, dl)
    line2 = (de / "efhw_lo.out").read_bytes().split(b"\n")[1]
    assert b"+factored_columns" not in line2, line2
    assert (b"+crossing_table" in line2) == bool(on), line2
    assert (b"+crossing_above_rule" in line2) == bool(on), line2


# --- momwire#1956's crossing_sheet_retune (reach 7: the nec5 executables) -------

_RETUNE = "MOMWIRE_X_CROSSING_SHEET_RETUNE"


def _retune_decks():
    sys.path.insert(0, str(ROOT / "tests"))
    from test_crossing_sheet_retune_1956 import DECKS as DECKS_1956

    return DECKS_1956


@pytest.mark.reach(7)
@pytest.mark.parametrize("deck", ["hub16", "deep_hub"])
@pytest.mark.parametrize("on", [False, True], ids=["off", "on"])
def test_the_sheet_retune_answers_as_the_launcher(on, deck, tmp_path):
    """The retuned sheet weights off ("0") and on: the executable's printout
    is the launcher's under the fixture gates' platform rule (byte for byte
    on Linux, PRINTOUT_RTOL_WIN32 / _MACOS elsewhere: `_same_by_platform`),
    and line 2 names the feature on both sides only when it is on (it is off
    by default). On and off print the same digits, so this cannot see the
    retune; the route tests below and the retune module's
    `test_the_launchers_route_moves_its_plan_within_its_drift` do."""
    text = _retune_decks()[deck]

    def prepare(d):
        (d / f"{deck}.nec").write_text(text, newline="")

    exe, ref, de, dl = _both(
        [f"{deck}.nec", f"{deck}.out"],
        tmp_path,
        prepare=prepare,
        env={_RETUNE: "1" if on else "0"},
    )
    assert exe.returncode == 0, exe.stderr
    _same_by_platform(exe, ref, de, dl)
    line2 = (de / f"{deck}.out").read_bytes().split(b"\n")[1]
    assert (b"+crossing_sheet_retune" in line2) == on, line2
    ref_line2 = (dl / f"{deck}.out").read_bytes().split(b"\n")[1]
    assert (b"+crossing_sheet_retune" in ref_line2) == on, ref_line2
    assert b"uncertified" not in line2, line2


def _retune_run(deck, on, tmp_path):
    """One executable run of `deck` with the retune off or on, its
    Sommerfeld store and its arrays its own: (sheets stored, Zin)."""
    import numpy as np

    d = tmp_path / f"{deck}-{on}"
    d.mkdir()
    (d / f"{deck}.nec").write_text(_retune_decks()[deck], newline="")
    (d / "arrays").mkdir()
    env = {k: v for k, v in os.environ.items() if k != _RETUNE}
    env.update(
        {
            _RETUNE: "1" if on else "0",
            "MOMWIRE_SOMM_CACHE": "1",
            "MOMWIRE_SOMM_CACHE_DIR": str(d / "store"),
            "MW_EXE_ARRAYS": str(d / "arrays"),
        }
    )
    r = subprocess.run(
        [EXE, f"{deck}.nec", f"{deck}.out"],
        cwd=d,
        env=env,
        capture_output=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    sheets = len(list((d / "store").rglob("nisheet-*.grid")))
    (i_source,) = np.fromfile(d / "arrays" / "0_i_source.bin", dtype=np.complex128)
    return sheets, 1.0 / complex(i_source)


@pytest.mark.reach(7)
@pytest.mark.parametrize("deck", ["hub16", "deep_hub"])
def test_the_sheet_retune_moves_the_executables_plan_within_its_drift(deck, tmp_path):
    """The route proof the byte gate cannot give (on and off print the same
    digits): on, the executable stores more near-interface sheets than off,
    so the retuned weights reached its census; and its input impedance, read
    at full precision from its arrays (MW_EXE_ARRAYS), moves by no more than
    the declared drift. deep_hub is where the executables measured the
    largest drift. The launcher's half of this proof (its planner's plane
    count, on the same two decks) is the retune module's
    `test_the_launchers_route_moves_its_plan_within_its_drift`."""
    from momwire import _features

    off_sheets, z_off = _retune_run(deck, False, tmp_path)
    on_sheets, z_on = _retune_run(deck, True, tmp_path)
    assert on_sheets > off_sheets, (off_sheets, on_sheets)
    drift = abs(z_on - z_off) / abs(z_off)
    assert drift <= _features.FEATURES["crossing_sheet_retune"].drift, drift


CROSSING = FIXTURES / "razor_printout_crossing" / "slant_base_source.nec"


def test_a_small_crossing_deck_reports_its_pairs_to_the_thread_policy(tmp_path):
    # momwire#1956: the size policy (`threads_for`) gives every core to a
    # buried deck with CROSSING_PAIRS above x below segment pairs or more,
    # whatever its basis count. nec5-bspline reported none, so a crossing
    # deck under 128 basis ran on one thread. Both formulations report them
    # now, and the policy's printout is the one thread's, byte for byte.
    shutil.copy(CROSSING, tmp_path / CROSSING.name)
    env = {k: v for k, v in os.environ.items() if k != "MOMWIRE_THREADS"}
    r = subprocess.run(
        [EXE, CROSSING.name, "policy.out"],
        cwd=tmp_path,
        env=dict(env, MW_EXE_TIMES="1"),
        capture_output=True,
        text=True,
        check=True,
    )
    m = re.search(
        r"cxx: size n_basis (\d+), crossing pairs (\d+), below pairs", r.stderr
    )
    assert m, r.stderr
    assert int(m[1]) < 128 and int(m[2]) >= 64, m[0]
    subprocess.run(
        [EXE, CROSSING.name, "one.out"],
        cwd=tmp_path,
        env=dict(env, MOMWIRE_THREADS="1"),
        capture_output=True,
        check=True,
    )
    policy = (tmp_path / "policy.out").read_bytes()
    assert policy == (tmp_path / "one.out").read_bytes()


# --- momwire#1971: razor-2p's fill-pair reuse (reach 7: nec5-razor-2p runs it) ---

# Synthetic decks past the reuse's small-deck thresholds
# (`_razor_pair_reuse_core.h`: SMALL_DECK_SEGS_FREE 400 in free space and
# over the PEC image, SMALL_DECK_SEGS_GROUND 100 over a finite ground), below
# which the reuse steps aside and on would equal off by construction: two
# parallel 201-segment dipoles (402 segments) in free space and over a
# perfect ground, two 61-segment ones (122) over the composing Sommerfeld
# ground.
_REUSE_DECKS_1971 = {
    "free": (
        "CM momwire#1971: two parallel dipoles, free space, 402 segments\nCE\n"
        "GW 1 201 0 -5 10 0 5 10 0.001\nGW 2 201 3 -5 10 3 5 10 0.001\nGE 0\n"
        "EX 0 1 101 0 1 0\nFR 0 1 0 0 14.0 0\nXQ\nEN\n"
    ),
    "pec": (
        "CM momwire#1971: two parallel dipoles over a perfect ground, 402 segments\n"
        "CE\nGW 1 201 0 -5 8 0 5 8 0.001\nGW 2 201 3 -5 8 3 5 8 0.001\nGE 1\n"
        "GN 1\nEX 0 1 101 0 1 0\nFR 0 1 0 0 14.0 0\nXQ\nEN\n"
    ),
    "sommerfeld": (
        "CM momwire#1971: two parallel dipoles over Sommerfeld ground, 122 segments\n"
        "CE\nGW 1 61 0 -2.5 6 0 2.5 6 0.001\nGW 2 61 2 -2.5 6 2 2.5 6 0.001\n"
        "GE 0\nGN 0 0 0 0 13.0 -50.0\nEX 0 1 31 0 1 0\nFR 0 1 0 0 28.5 0\nXQ\nEN\n"
    ),
}

_REUSE_LINE = re.compile(
    r"^cxx: reuse plans built (\d+), declined (\d+), found (\d+)$", re.M
)


def _reuse_counts(stderr: bytes) -> tuple[int, int, int]:
    """MW_EXE_TIMES's reuse line: (built, declined, found), process totals."""
    m = _REUSE_LINE.search(stderr.decode("latin-1"))
    assert m, stderr
    return int(m[1]), int(m[2]), int(m[3])


def _reuse_ran(built: int, declined: int) -> bool:
    """The gate's proof that the reuse RAN: a plan built that did not come
    back declined. A declined plan fills as off, so counting it would let an
    executable whose every plan declines agree with the launcher by not
    running (`test_razor_pair_reuse_gate_does_not_count_a_declined_plan`)."""
    return built - declined >= 1


@pytest.mark.skipif(BASIS != "razor-2p", reason="razor-2p's feature")
@pytest.mark.parametrize("ground", sorted(_REUSE_DECKS_1971))
def test_razor_pair_reuse_runs_as_the_launcher(ground, tmp_path, monkeypatch):
    """Reach 7 covers the nec5 executables (levels 6 and 7). On ("1") and
    off ("0"), the executable's printout is the launcher's byte for byte,
    line 2 naming +razor_pair_reuse on both sides only when it is on. The
    executable and the launcher share the C++ core, so that alone cannot
    catch a fault in the core: the reuse is also shown to RUN (MW_EXE_TIMES's
    reuse line: a plan built that did not decline when on, none built when
    off) and on is held to off
    byte for byte except line 2 (the reuse moves Zin ~1e-13 here, far under
    a printed digit)."""
    monkeypatch.delenv("MOMWIRE_X_RAZOR_PAIR_REUSE", raising=False)
    name = f"reuse_{ground}.nec"

    def prepare(d):
        (d / name).write_text(_REUSE_DECKS_1971[ground], newline="")

    lines = {}
    for state in ("1", "0"):
        run = tmp_path / state
        run.mkdir()
        exe, ref, de, dl = _both(
            [name, "reuse.out"],
            run,
            prepare=prepare,
            env={"MOMWIRE_X_RAZOR_PAIR_REUSE": state, "MW_EXE_TIMES": "1"},
        )
        assert exe.returncode == 0, exe.stderr
        _same(exe, ref, de, dl)
        built, declined, found = _reuse_counts(exe.stderr)
        if state == "1":
            assert _reuse_ran(built, declined), (built, declined, found)
        else:
            assert built == 0 and found == 0, (built, declined, found)
        lines[state] = (de / "reuse.out").read_bytes().split(b"\n")
        assert (b"+razor_pair_reuse" in lines[state][1]) == (state == "1"), lines[
            state
        ][1]
    on, off = lines["1"], lines["0"]
    assert on[1].rstrip() == off[1].rstrip() + b" +razor_pair_reuse", (on[1], off[1])
    assert on[:1] + on[2:] == off[:1] + off[2:]


# Eight wires of random place and direction over Sommerfeld ground (the
# Python gates' `_random_wires`, 15 segments each): past the finite ground's
# small-deck threshold (100 segments), so the executable BUILDS a plan, but
# its pairs are nearly all distinct, so the plan comes back declined ("too
# few shared pairs to repay the plan") and the fill runs as off.
_DECLINING_DECK_1971 = (
    "CM momwire#1971: random wires over Sommerfeld ground, 120 segments: the\n"
    "CM reuse plan is built and declines\nCE\n"
    "GW 1 15 -3.3148 -2.1055 9.2051 -5.5321 -3.8733 8.3631 0.001\n"
    "GW 2 15 -2.7221 1.8766 6.4547 -1.3016 -0.3419 4.6850 0.001\n"
    "GW 3 15 1.9027 3.6501 7.1368 1.4282 5.5551 6.7394 0.001\n"
    "GW 4 15 3.7877 -1.6128 7.2559 3.4663 -0.6626 10.6577 0.001\n"
    "GW 5 15 -3.7572 1.6557 7.4970 -5.4830 1.1258 9.1001 0.001\n"
    "GW 6 15 1.0407 -1.6147 8.9670 2.5946 -3.0747 6.4283 0.001\n"
    "GW 7 15 1.4624 2.5606 7.7143 1.5311 2.4219 11.4106 0.001\n"
    "GW 8 15 -0.8486 -0.1625 6.5853 -2.7531 1.0388 7.7836 0.001\n"
    "GE 0\nGN 0 0 0 0 13.0 -50.0\nEX 0 1 8 0 1 0\nFR 0 1 0 0 28.5 0\nXQ\nEN\n"
)


@pytest.mark.skipif(BASIS != "razor-2p", reason="razor-2p's feature")
def test_razor_pair_reuse_gate_does_not_count_a_declined_plan(tmp_path, monkeypatch):
    """The adversarial case for `test_razor_pair_reuse_runs_as_the_launcher`:
    with the reuse on, a deck whose plan is built and declines prints the
    launcher's bytes (it fills as off on both sides) and builds a plan, so a
    gate that counted plans built would pass it without the reuse having
    run. The reuse line counts that plan as declined, and the gate's proof
    (`_reuse_ran`) is false on it."""
    monkeypatch.delenv("MOMWIRE_X_RAZOR_PAIR_REUSE", raising=False)
    name = "reuse_declines.nec"

    def prepare(d):
        (d / name).write_text(_DECLINING_DECK_1971, newline="")

    exe, ref, de, dl = _both(
        [name, "reuse.out"],
        tmp_path,
        prepare=prepare,
        env={"MOMWIRE_X_RAZOR_PAIR_REUSE": "1", "MW_EXE_TIMES": "1"},
    )
    assert exe.returncode == 0, exe.stderr
    _same(exe, ref, de, dl)
    built, declined, found = _reuse_counts(exe.stderr)
    assert built >= 1 and declined == built, (built, declined, found)
    assert not _reuse_ran(built, declined)


@pytest.mark.parametrize("deck", [DIPOLE], ids=lambda p: p.stem)
def test_razor_pair_reuse_unset_or_off_moves_no_byte(deck, tmp_path, monkeypatch):
    """Unset or off, either executable's printout is the launcher's, byte for
    byte, with no feature on line 2."""
    monkeypatch.delenv("MOMWIRE_X_RAZOR_PAIR_REUSE", raising=False)

    def prepare(d):
        shutil.copy(deck, d / deck.name)

    for i, env in enumerate((None, {"MOMWIRE_X_RAZOR_PAIR_REUSE": "0"})):
        run = tmp_path / str(i)
        run.mkdir()
        exe, ref, de, dl = _both(
            [deck.name, "reuse.out"], run, prepare=prepare, env=env
        )
        assert exe.returncode == 0, exe.stderr
        _same(exe, ref, de, dl)
        line2 = (de / "reuse.out").read_bytes().split(b"\n")[1]
        assert b"+razor_pair_reuse" not in line2, line2


@pytest.mark.skipif(BASIS != "bspline", reason="nec5-bspline's refusal")
def test_nec5_bspline_refuses_razor_pair_reuse_by_basis(tmp_path, monkeypatch):
    """nec5-bspline shares nec5-razor-2p's support levels (6 and 7) and its
    feature table, so the reach alone would let it accept
    MOMWIRE_X_RAZOR_PAIR_REUSE, name it on line 2 and fill as with it off.
    The feature is razor-2p's fill, so this executable refuses it, in the
    registry's sentence, before reading the deck, and writes nothing."""
    from momwire import _features

    monkeypatch.delenv("MOMWIRE_X_RAZOR_PAIR_REUSE", raising=False)
    shutil.copy(DIPOLE, tmp_path / DIPOLE.name)
    r = subprocess.run(
        [EXE, DIPOLE.name, "reuse.out"],
        cwd=tmp_path,
        env=dict(os.environ, MOMWIRE_X_RAZOR_PAIR_REUSE="1"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 2, (r.stdout, r.stderr)
    assert r.stderr == _features.EXE_REFUSAL.format(name="razor_pair_reuse") + "\n"
    assert not (tmp_path / "reuse.out").exists()


# --- the frequency-parallel sweep (razor-2p's buried stepping FR) ---

# Synthetic buried decks with a stepping FR, small enough for every lane:
# a crossing deck whose network cells are thread-sensitive (a parked wire
# pinned by a 1e10-ohm load, fed through TLs: its admittance reactance
# prints OpenBLAS's threaded digits at two or more BLAS threads and other
# ones at one, so a sweep that factored at one BLAS thread fails it), a
# wholly-buried dipole with near fields, and a vertical on a buried hub.
_SWEEP_DECKS = {
    "crossing_net_pin": (
        "CM two loaded verticals on buried stubs, each fed by an EX 4 through its\n"
        "CM own TL from a parked node pinned by a 1e10-ohm load\nCM NEC-5\nCE\n"
        "GW 1 3 0 0 -0.3 0 0 0 0.0015\nGW 4 12 0 0 0 0 0 2.5 0.0015\n"
        "GW 2 3 2 0 -0.3 2 0 0 0.0015\nGW 5 12 2 0 0 2 0 2.5 0.0015\n"
        "GW 3 3 0 2000 5 0 2000.05 5 0.001\nGE -1 0\n"
        "GN 0 0 0 0 13.0 0.005 1.0 0.0 NOFILE\n"
        "LD 0,4,1,0,18.,0.,0.\nLD 0,5,1,0,18.,0.,0.\nLD 4,3,2,0,1.E+10,0.\n"
        "TL 3,2,4,1,50.,2.6,0.,0.,0.,0.\nTL 3,2,5,1,50.,2.9,0.,0.,0.,0.\n"
        "EX 4 4 1 0 1.414214 0\nEX 4 5 1 0 0 1.414214\n"
        "FR 0 4 0 0 28.0 0.5\nXQ\nEN\n"
    ),
    "below_plane_near": (
        "CM a buried horizontal dipole 5 m at -0.3 m, near fields per frequency\n"
        "CM NEC-5\nCE\nGW 1 11 -2.5 0.0 -0.3 2.5 0.0 -0.3 0.001\nGE -1 0\n"
        "GN 0 0 0 0 13 0.005 1.0 0.0 NOFILE\nEX 0 1 6 0 1.0 0.0\n"
        "FR 0 4 0 0 7.0 0.25\nNE 0 5 1 1 0.5 0.0 -0.05 1.0 0.0 0.0\n"
        "NE 0 5 1 1 0.5 0.0 1.0 1.0 0.0 0.0\nXQ\nEN\n"
    ),
    "buried_hub": (
        "CM GN 2: a vertical over a buried hub, fed at its base\nCE\n"
        "GW 1 8 0 0 0 0 0 5.0 0.003\nGW 2 2 0 0 0 0 0 -0.1 0.003\n"
        "GW 3 8 0 0 -0.1 5.0 0 -0.1 0.003\nGW 4 8 0 0 -0.1 -5.0 0 -0.1 0.003\n"
        "GE -1\nGN 2 0 0 0 13.0 0.005\nFR 0 4 0 0 14.0 0.25\nEX 0 1 -1 0 1 0\nXQ\nEN\n"
    ),
}

_SWEEP_LINE = re.compile(r"^cxx: frequencies presolved (\d+)$", re.M)
_WORKERS_LINE = re.compile(r"^cxx: buried runs (\d+), frequency workers (\d+)$", re.M)


def _sweep_run(tmp_path, name, deck, tag, env):
    """One run of the executable on `deck` in its own directory, with its own
    fresh Sommerfeld store: (printout bytes, store files, presolved count,
    MW_EXE_TIMES's (buried runs, frequency workers) or None)."""
    d = tmp_path / tag
    d.mkdir()
    (d / name).write_text(deck, newline="")
    store = d / "store"
    store.mkdir()
    base = {
        k: v
        for k, v in os.environ.items()
        if k not in ("MOMWIRE_THREADS", "MW_EXE_FREQS", "MOMWIRE_SOMM_CACHE")
    }
    r = subprocess.run(
        [EXE, name, "sweep.out"],
        cwd=d,
        env=dict(
            base,
            MW_EXE_TIMES="1",
            MOMWIRE_SOMM_CACHE="1",
            MOMWIRE_SOMM_CACHE_DIR=str(store),
            **env,
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    m = _SWEEP_LINE.search(r.stderr)
    assert m, r.stderr
    w = _WORKERS_LINE.search(r.stderr)
    stored = {
        str(p.relative_to(store)): p.read_bytes()
        for p in sorted(store.rglob("*"))
        if p.is_file()
    }
    return (
        (d / "sweep.out").read_bytes(),
        stored,
        int(m[1]),
        (int(w[1]), int(w[2])) if w else None,
    )


@pytest.mark.skipif(BASIS != "razor-2p", reason="razor-2p's sweep")
@pytest.mark.parametrize("deck", sorted(_SWEEP_DECKS))
@pytest.mark.parametrize(
    "crossing_table", [False, True], ids=["plain", "crossing_table"]
)
def test_a_buried_sweep_prints_the_serial_bytes(deck, crossing_table, tmp_path):
    """The frequency-parallel sweep (`freq_workers_for`, `presolve`) taken
    (MW_EXE_FREQS=1) and turned off (=0) at the same four-thread budget, and
    the one-thread run: the sweep prints the flag-off run's bytes and writes
    the same store, file for file, and is shown to RUN (its presolved count),
    while the flag-off control presolves nothing. The flag-off run at four
    threads is the threaded copy with BLAS at four, so a sweep that factored
    at one BLAS thread would print the one-thread run's digits in the pinned
    network cells instead. With crossing_table on, the sweep runs too: each
    fill's feature state is its own thread's in the serial copy."""
    name = f"{deck}.nec"
    text = _SWEEP_DECKS[deck]
    env = {"MOMWIRE_X_CROSSING_TABLE": "1"} if crossing_table else {}
    off, off_store, off_n, _ = _sweep_run(
        tmp_path, name, text, "off", dict(env, MOMWIRE_THREADS="4", MW_EXE_FREQS="0")
    )
    on, on_store, on_n, on_w = _sweep_run(
        tmp_path, name, text, "on", dict(env, MOMWIRE_THREADS="4", MW_EXE_FREQS="1")
    )
    assert off_n == 0
    assert on_n == 4, on_n
    assert on_w == (4, 4), on_w
    assert on == off
    assert on_store == off_store
    assert on_store, "the store was never written: the store half went unexercised"
    if deck != "crossing_net_pin":
        # The pinned network cells are the one deck whose one-thread digits
        # are OpenBLAS's single-thread path's (7eb2830c); the others print
        # the same at any count.
        one, _, one_n, _ = _sweep_run(
            tmp_path, name, text, "one", dict(env, MOMWIRE_THREADS="1")
        )
        assert one_n == 0
        assert on == one


@pytest.mark.skipif(BASIS != "razor-2p", reason="razor-2p's sweep")
def test_the_sweep_policy_takes_full_rounds_and_declines_a_large_mesh(tmp_path):
    """Without the gate hook: a small buried deck whose four frequencies fill
    two-thread rounds exactly is swept, and a mesh past SWEEP_MAX_BASIS
    (1250 basis functions: here a vertical on four 320-segment radials) is
    not, at the same budget: past the largest mesh measured faster, the
    OpenMP team's fill is taken (`freq_workers_for`'s size term)."""
    small = _SWEEP_DECKS["buried_hub"]
    _, _, n, w = _sweep_run(
        tmp_path, "small.nec", small, "small", {"MOMWIRE_THREADS": "2"}
    )
    assert (n, w) == (4, (4, 2)), (n, w)
    large = (
        "CM a vertical on four 320-segment buried radials\nCM NEC-5\nCE\n"
        "GW 1 320 0 0 -0.15 6.3 0 -0.15 0.0005\nGW 2 320 0 0 -0.15 0 6.3 -0.15 0.0005\n"
        "GW 3 320 0 0 -0.15 -6.3 0 -0.15 0.0005\nGW 4 320 0 0 -0.15 0 -6.3 -0.15 0.0005\n"
        "GW 5 2 0 0 -0.15 0 0 0 0.0005\nGW 6 10 0 0 0 0 0 10.5 0.0005\nGE -1 0\n"
        "GN 0 0 0 0 13.0 0.005 1.0 0.0 NOFILE\nEX 0 6 1 0 1.0 0.0\n"
        "FR 0 2 0 0 7.0 0.1\nXQ\nEN\n"
    )
    _, _, n, w = _sweep_run(
        tmp_path, "large.nec", large, "large", {"MOMWIRE_THREADS": "2"}
    )
    assert (n, w) == (0, (2, 1)), (n, w)
