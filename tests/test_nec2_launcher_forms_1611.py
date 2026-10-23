"""The NEC-2 launchers' call forms through the shared launcher shell
(momwire#1611 stage 1).

``momwire-nec2c*`` answered one call form until now: SimNEC's, no arguments,
``NX``-chained decks on stdin and printouts on stdout.  ``momwire-nec2c deck
out`` printed a banner and wrote nothing.  They now answer the forms every
momwire launcher answers, through :func:`momwire.eznec._shell.main` with
``dialect="nec2"``:

* ``in out``: the deck file's printout written to ``out``;
* ``- -``: decks on stdin, printouts on stdout, which for NEC-2 is SimNEC's
  own protocol spelled out;
* no arguments and 4nec2's ``ENGINE < nec2d.tmp``: the two file names on
  stdin, one per line, told apart from a deck by the first line.

The printout itself is still the portal's (stage 2 moves it), so every form
is held to what the stdin form prints for the same deck: the file form to
the same text with CRLF endings, ``- -`` to the same bytes.
"""

import importlib.util
import io
import subprocess
import sys
from pathlib import Path

import pytest

import momwire_nec2c_client as nec2c_client
import momwire_serve_client as mech
from momwire.eznec import _shell
from momwire.portal import _portal as nec_portal
from momwire.portal import engine_scope, main

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "nec_portal"
CORPUS = sorted(p.stem for p in FIXTURE_DIR.glob("*.deck"))

DIPOLE = (
    "CE dipole 1611\n"
    "GW 1 11 0. -5. 10. 0. 5. 10. 0.001\n"
    "GE 0\nEX 0 1 6 0 1.\nFR 0 1 0 0 14.0 1\nXQ\n"
)


def _stdin_form(text: str, argv=()) -> str:
    """What the bare command prints for ``text`` on stdin (today's form)."""
    out = io.StringIO()
    with engine_scope():
        assert main(list(argv), stdin=io.StringIO(text), stdout=out) == 0
    return out.getvalue()


def _file_form(tmp_path: Path, text: str, argv=()) -> bytes:
    deck = tmp_path / "deck.nec"
    out = tmp_path / "deck.out"
    deck.write_bytes(text.encode("latin-1"))
    with engine_scope():
        assert main([*argv, str(deck), str(out)], stdin=io.StringIO("")) == 0
    return out.read_bytes()


def _load_capture():
    path = Path(__file__).resolve().parent.parent / "scripts" / "nec_portal_capture.py"
    spec = importlib.util.spec_from_file_location("nec_portal_capture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CAPTURE = _load_capture()


def _canon(text: str) -> str:
    """The wall-clock fields (``FILL: 2 msec``) are the only printout fields
    two runs of one deck may differ in: canonicalised as every other
    two-run equality in the portal's suite does (momwire#403)."""
    return _CAPTURE.canonicalize_timings(text)


class _Crlf:
    """``text`` as the shell writes a printout file: CRLF, latin-1.  Equal
    to the written bytes when they are CRLF throughout and say the same
    thing once the wall-clock fields are canonicalised."""

    def __init__(self, text: str):
        self.text = text

    def __eq__(self, served) -> bool:
        if not isinstance(served, bytes) or b"\n" in served.replace(b"\r\n", b""):
            return False
        decoded = served.decode("latin-1").replace("\r\n", "\n")
        return _canon(decoded) == _canon(self.text)

    def __repr__(self) -> str:
        return f"_Crlf({self.text[:60]!r}...)"


def _crlf(text: str) -> _Crlf:
    return _Crlf(text)


# --------------------------------------------------------------------------
# in out
# --------------------------------------------------------------------------


def test_the_file_form_writes_what_the_stdin_form_prints(tmp_path):
    for text in (DIPOLE, DIPOLE + "EN\n", DIPOLE + "NX\n" + DIPOLE + "EN\n"):
        assert _file_form(tmp_path, text) == _crlf(_stdin_form(text))


def test_the_file_form_reads_crlf_decks_as_the_stdin_form_does(tmp_path):
    assert _file_form(tmp_path, DIPOLE.replace("\n", "\r\n")) == _crlf(
        _stdin_form(DIPOLE)
    )


def test_the_file_form_rides_the_basis_flag(tmp_path):
    served = _file_form(tmp_path, DIPOLE, ["--basis", "sinusoidal"])
    assert served == _crlf(_stdin_form(DIPOLE, ["--basis", "sinusoidal"]))
    assert b"VERSION:nec2c.ae6ty.momwire.9.1+sin" in served


@pytest.mark.parametrize("name", CORPUS)
def test_the_file_form_answers_every_oracle_deck_as_the_stdin_form(tmp_path, name):
    """The 70 nec_portal oracle decks, each through ``in out``: the printout
    the portal's existing gates hold to the captures, byte for byte, with
    CRLF endings."""
    text = (FIXTURE_DIR / f"{name}.deck").read_text()
    assert _file_form(tmp_path, text) == _crlf(_stdin_form(text))


def test_an_unreadable_deck_still_gets_a_printout(tmp_path):
    out = tmp_path / "x.out"
    with engine_scope():
        assert main([str(tmp_path / "missing.nec"), str(out)]) == 0
    text = out.read_bytes().decode("latin-1")
    assert "VERSION:nec2c.ae6ty.momwire.9.1" in text
    assert "\r\nERROR: UNABLE TO READ INPUT FILE " in text
    assert "ERROR-NEC2C: UNABLE TO READ INPUT FILE " in text
    last = text.splitlines()[-1].split()
    assert last[:5] == ["DATA", "CARD", "No:", "1", "EN"], last


def test_one_path_is_the_argument_error_and_writes_nothing(tmp_path, capsys):
    with engine_scope():
        assert main([str(tmp_path / "deck.nec")]) == 0
    assert capsys.readouterr().out == _shell.ARGUMENT_ERROR_OUTPUT + "\n"
    assert list(tmp_path.iterdir()) == []


def test_a_lone_dash_is_refused_on_stderr(capsys):
    with engine_scope():
        assert main(["-", "deck.out"]) == 2
    assert _shell.STREAMING_REFUSAL in capsys.readouterr().err


# --------------------------------------------------------------------------
# - -
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        DIPOLE + "NX\n",
        DIPOLE + "NX\n" + DIPOLE.replace("14.0", "21.0") + "NX\n",
        DIPOLE + "EN\n",
        DIPOLE,
        (FIXTURE_DIR / "resident_two_decks.deck").read_text(),
    ],
)
def test_the_stream_form_is_the_stdin_form(text):
    assert _canon(_stdin_form(text, ["-", "-"])) == _canon(_stdin_form(text))


# --------------------------------------------------------------------------
# 4nec2's names on stdin
# --------------------------------------------------------------------------


def test_4nec2s_names_on_stdin_serve_the_deck_file(tmp_path, monkeypatch):
    """``ENGINE < nec2d.tmp``: the names relative to the working directory,
    the deck without ``EN`` (4nec2 writes none), the printout the file form
    writes."""
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "EXAMPLE3.inp").write_text(DIPOLE)
    monkeypatch.chdir(tmp_path)
    names = io.StringIO("out/EXAMPLE3.inp\r\nout/EXAMPLE3.out\r\n")
    with engine_scope():
        assert main([], stdin=names, stdout=io.StringIO()) == 0
    served = (tmp_path / "out" / "EXAMPLE3.out").read_bytes()
    assert served == _crlf(_stdin_form(DIPOLE))


def test_4nec2s_quoted_names_are_read_as_paths(tmp_path, monkeypatch):
    (tmp_path / "my dir").mkdir()
    (tmp_path / "my dir" / "x.inp").write_text(DIPOLE)
    monkeypatch.chdir(tmp_path)
    names = io.StringIO('"my dir/x.inp"\n"my dir/x.out"\n')
    with engine_scope():
        assert main([], stdin=names, stdout=io.StringIO()) == 0
    assert (tmp_path / "my dir" / "x.out").read_bytes() == _crlf(_stdin_form(DIPOLE))


def test_one_name_on_stdin_is_the_argument_error(tmp_path, capsys):
    (tmp_path / "x.inp").write_text(DIPOLE)
    with engine_scope():
        assert main([], stdin=io.StringIO(f"{tmp_path / 'x.inp'}\n")) == 0
    assert capsys.readouterr().out == _shell.ARGUMENT_ERROR_OUTPUT + "\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.inp"]


def test_a_missing_input_named_on_stdin_gets_a_refusal_printout(tmp_path):
    out = tmp_path / "x.out"
    names = io.StringIO(f"{tmp_path / 'nope.inp'}\n{out}\n")
    with engine_scope():
        assert main([], stdin=names, stdout=io.StringIO()) == 0
    assert b"ERROR: UNABLE TO READ INPUT FILE" in out.read_bytes()


def test_a_deck_on_stdin_is_still_simnecs_protocol():
    """The first line is a card, so this is SimNEC's call, answered exactly
    as it always was: the start-up banner, the deck, the ``NX`` echo and the
    trailing banner."""
    out = _stdin_form("CM FF 2\n" + DIPOLE + "NX\n")
    assert out.startswith("\n\n\n")
    assert out.count("VERSION:nec2c.ae6ty.momwire.9.1") == 2
    assert "ANTENNA INPUT PARAMETERS" in out
    assert " NX " in out


@pytest.mark.parametrize(
    ("line", "names"),
    [
        ("CM FF 2\n", False),
        ("CE\n", False),
        ("GW 1 11 0. -5. 10. 0. 5. 10. 0.001\n", False),
        ("GE1\n", False),
        ("GW,1,11\n", False),
        ("   \n", False),
        ("..\\out\\EXAMPLE3.inp\r\n", True),
        ("EXAMPLE3.inp\n", True),
        ('"C:\\4nec2\\out\\EX 3.inp"\n', True),
        ("/home/u/deck.nec\n", True),
        ("C:\\x.nec\n", True),
    ],
)
def test_names_or_a_deck_by_the_first_line(line, names):
    assert mech.is_names_line(line) is names


def test_a_file_that_exists_is_a_name_whatever_it_looks_like(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not mech.is_names_line("GW1\n")
    (tmp_path / "GW1").write_text(DIPOLE)
    assert mech.is_names_line("GW1\n")


# --------------------------------------------------------------------------
# the version line and the shared client
# --------------------------------------------------------------------------


def test_the_version_line_is_unchanged(capsys):
    with engine_scope():
        assert main(["-version"], prog="/usr/bin/momwire-nec2c-pulse") == 0
    assert capsys.readouterr().out == (
        f"NEC2 momwire-nec2c-pulse [sb] {mech.momwire_version() or '0.0.0'}\n"
    )
    assert nec_portal.PROBE_VERSION.split()[2] == f"[{mech.LAUNCHER_MODES}]"


@pytest.mark.parametrize(
    ("engine", "flags", "paths"),
    [
        ([], [], []),
        (
            ["--basis", "pulse", "d.nec", "d.out"],
            ["--basis", "pulse"],
            ["d.nec", "d.out"],
        ),
        (["-", "-", "--cache"], ["--cache"], ["-", "-"]),
        (
            ["--save-decks", "dir", "--cache"],
            ["--save-decks", "dir", "--cache"],
            [],
        ),
    ],
)
def test_the_shared_client_keeps_paths_out_of_the_engine_identity(engine, flags, paths):
    assert nec2c_client.split_positional(engine) == (flags, paths)


@pytest.mark.integration
def test_the_stock_command_serves_the_file_form_in_a_real_process(tmp_path):
    (tmp_path / "d.nec").write_text(DIPOLE + "EN\n")
    proc = subprocess.run(
        [sys.executable, "-m", "momwire.portal", "d.nec", "d.out"],
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=180,
    )
    assert proc.returncode == 0 and proc.stdout == b"" and proc.stderr == b""
    assert (tmp_path / "d.out").read_bytes() == _crlf(_stdin_form(DIPOLE + "EN\n"))


@pytest.mark.integration
def test_the_stock_command_reads_4nec2s_names_from_a_redirected_file(tmp_path):
    (tmp_path / "d.inp").write_text(DIPOLE)
    (tmp_path / "nec2d.tmp").write_text("d.inp\nd.out\n")
    with open(tmp_path / "nec2d.tmp", "rb") as names:
        proc = subprocess.run(
            [sys.executable, "-m", "momwire.portal"],
            cwd=tmp_path,
            stdin=names,
            capture_output=True,
            timeout=180,
        )
    assert proc.returncode == 0 and proc.stdout == b"", proc.stdout
    assert (tmp_path / "d.out").read_bytes() == _crlf(_stdin_form(DIPOLE))


@pytest.mark.integration
def test_the_stock_command_streams_with_two_dashes(tmp_path):
    text = DIPOLE + "NX\n" + DIPOLE + "NX\n"
    bare = subprocess.run(
        [sys.executable, "-m", "momwire.portal"],
        input=text.encode(),
        capture_output=True,
        timeout=180,
    )
    dashed = subprocess.run(
        [sys.executable, "-m", "momwire.portal", "-", "-"],
        input=text.encode(),
        capture_output=True,
        timeout=180,
    )
    assert bare.returncode == dashed.returncode == 0
    assert _canon(dashed.stdout.decode()) == _canon(bare.stdout.decode())
    assert dashed.stdout.count(b" NX ") == 2


@pytest.mark.integration
@pytest.mark.skipif(sys.platform == "win32", reason="the shared client is POSIX only")
def test_the_shared_client_hands_the_file_form_to_the_stock_engine(tmp_path):
    (tmp_path / "d.nec").write_text(DIPOLE + "EN\n")
    proc = subprocess.run(
        [sys.executable, "-m", "momwire_nec2c_client", "d.nec", "d.out"],
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=180,
    )
    assert proc.returncode == 0 and proc.stdout == b"", proc.stdout
    assert (tmp_path / "d.out").read_bytes() == _crlf(_stdin_form(DIPOLE + "EN\n"))
