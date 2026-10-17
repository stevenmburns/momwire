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

The executable needs a static OpenBLAS and a CMake build, which only the
`standalone` workflow's Linux job makes, so this module runs only when
MOMWIRE_RAZOR2P_STANDALONE_EXE names one. The launcher it is compared with is
this environment's.
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

pytestmark = pytest.mark.skipif(
    not EXE or sys.platform == "win32",
    reason="MOMWIRE_RAZOR2P_STANDALONE_EXE does not name a built (Linux) executable",
)

DIPOLE = FIXTURES / "razor_printout_3a" / "node_gap_two_sources.nec"
REFUSES = FIXTURES / "razor_printout_3a" / "chain_then_refusal.nec"
RAISES = FIXTURES / "razor_printout_3a" / "echo_raises_inf_field.nec"
BURIED = FIXTURES / "razor_printout_somm" / "gn0_buried_radial.nec"
EXAMPLE3 = FIXTURES / "4nec2_example3" / "EXAMPLE3.inp"
DECKS = (DIPOLE, REFUSES, RAISES, BURIED, EXAMPLE3)


def _launcher() -> str:
    found = shutil.which("momwire-nec5-razor-2p") or str(
        Path(sys.executable).parent / "momwire-nec5-razor-2p"
    )
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


@pytest.mark.parametrize(
    "deck",
    [DIPOLE, REFUSES, RAISES, EXAMPLE3.with_name("EXAMPLE3_EN.nec"), BURIED],
    ids=["served", "refused", "raised", "gmo", "buried"],
)
def test_a_deck_answers_as_the_launcher(deck, tmp_path):
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
    assert b"IsADirectoryError" in (de / "x.out").read_bytes()


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
    _shell.write_printout(want, _printout.render_refusal(text, line, basis="razor-2p"))
    got = out.read_bytes()
    # the header's engine stamp is the executable's; the rest is the frame
    assert got.split(b"\r\n")[2:] == want.read_bytes().split(b"\r\n")[2:]
    assert line.encode() in got
