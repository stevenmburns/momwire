"""The shared engine: one thin client per spawn, one resident server (#379).

``momwire-nec2c-shared`` exists for one measured reason — SimNEC spawns a
fresh engine per evaluation burst and throws it away, so every burst pays the
cold start and the cross-deck cache never survives to be hit. Splitting the
process is only worth anything if three things hold, and each of them is a
test here rather than a claim:

* **the answer does not change.** Every fixture deck streamed through
  client→server is the stock engine's printout, byte for byte, wall-clock
  timings aside (the one non-deterministic field, canonicalised by the same
  ``scripts/nec_portal_capture.py`` helper the fixtures were captured with).
  Including the three refusal decks: an ``ERROR:`` printout is part of the
  contract and travels the socket like any other.
* **the client is thin.** It imports neither momwire nor NumPy — the whole
  saving is that import not happening — and answers ``-version`` from
  distribution metadata without spawning anything.
* **the cache outlives the client.** A deck solved by one invocation is
  served from cache to the NEXT invocation, which is the case the issue is
  about and the one a per-burst process can never reach.

The rest is the failure modes a resident server has and a per-burst process
does not: a crew firing 16 clients at a directory with no server in it, a
client killed mid-solve, and a server nobody comes back to.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import tomllib

import momwire_nec2c_client as client
from momwire.portal import _portal as nec_portal
from momwire.portal import main as portal_main

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "nec_portal"
ALL_NAMES = tuple(sorted(p.stem for p in FIXTURE_DIR.glob("*.deck")))
# The corpus decks whose answer is a REFUSAL, which travels the socket like
# any other answer and is gated by the same byte oracle. All four are the
# MININEC-type ground idiom — a ``GD`` in force under a ``GN 1`` at an execute
# card that will not read it (#458; the three ``mininec_*`` decks captured by
# momwire#487).
#
# The three hand-authored network probes were here from #930 until
# momwire#456 phase C and they SOLVE now, alongside the four decks that unit
# added (`dipole_tl_zero_length`, `dipole_nt_all_zero`,
# `dipole_ld_nt_colocated`, `dipole_ex6_gyrator`).
REFUSED_NAMES = (
    "dipole_gd_second_medium",
    "mininec_gp80_seam",
    "mininec_vertical_gd2_rp0",
    "mininec_vertical_rp0",
)

NX_ECHO = re.compile(r"^\s*DATA CARD No:\s+\d+ NX\b", re.MULTILINE)
LISTENING = re.compile(r"^\S+ listening pid=(\d+) socket=(\S+)$", re.MULTILINE)

CLIENT_ARGV = [sys.executable, "-m", "momwire_nec2c_client"]


def _load_capture_module():
    """The capture script's timing canonicaliser — scripts/ is not a package."""
    path = REPO_ROOT / "scripts" / "nec_portal_capture.py"
    spec = importlib.util.spec_from_file_location("nec_portal_capture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAPTURE = _load_capture_module()


def fixture_deck(name: str) -> str:
    """A fixture deck verbatim. All 45 already carry their ``NX`` terminator,
    so appending one would frame a second, EMPTY deck — legal, and a silent way
    to double every count a test makes about how many decks were solved."""
    return (FIXTURE_DIR / f"{name}.deck").read_text()


def deck_body(name: str) -> str:
    """The deck with its terminator taken off — an unterminated body."""
    return re.sub(r"(?im)^\s*NX\b.*\n?\Z", "", fixture_deck(name))


def deck_ending_in_en(name: str) -> str:
    """The deck a standalone ``.nec`` file is: ``EN`` where ``NX`` was."""
    return f"{deck_body(name)}EN\n"


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


@pytest.fixture
def runtime():
    """A private runtime directory, and every server started in it reaped.

    Not ``tmp_path``: the socket name has to fit in ``sun_path``'s 107 bytes,
    and pytest's directory names are long enough (and the box's ``TMPDIR``
    unpredictable enough) that a test would fail for a reason having nothing
    to do with what it tests. A short ``mkdtemp`` prefix keeps the whole
    budget under 40 bytes.

    Teardown SIGTERMs by pid rather than by name so a developer's own resident
    server — the one this feature exists to leave running — is never killed by
    running the tests.
    """
    room = tempfile.mkdtemp(prefix="mw379-")
    try:
        yield room
    finally:
        for pid in server_pids(room):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and any(
            _alive(pid) for pid in server_pids(room)
        ):
            time.sleep(0.05)
        shutil.rmtree(room, ignore_errors=True)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def logs(room: str) -> str:
    return "".join(
        (Path(room) / name).read_text(errors="replace")
        for name in sorted(os.listdir(room))
        if name.endswith(".log")
    )


def server_pids(room: str) -> list[int]:
    """Every server that got as far as binding a socket in ``room``."""
    return [int(pid) for pid, _path in LISTENING.findall(logs(room))]


def sockets(room: str) -> list[str]:
    return sorted(n for n in os.listdir(room) if n.endswith(".sock"))


def run_client(room: str, decks: str = "", *args: str, timeout: float = 300.0):
    return subprocess.run(
        [*CLIENT_ARGV, *args],
        input=decks,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "MOMWIRE_PORTAL_RUNTIME_DIR": room},
    )


def stock(decks: str, argv: list[str] | None = None) -> str:
    """What a FRESH stock ``momwire-nec2c`` process answers to ``decks``.

    In-process rather than spawned: ``main`` takes its streams, and
    ``configure_engine`` resets the basis and the cache on entry, so one call
    is indistinguishable from one process — which is the whole reason the
    oracle can afford to be all 45 decks.
    """
    out = io.StringIO()
    assert portal_main(list(argv or []), io.StringIO(decks), out, io.StringIO()) == 0
    return out.getvalue()


def same_printout(left: str, right: str) -> bool:
    a = CAPTURE.canonicalize_timings(left)
    b = CAPTURE.canonicalize_timings(right)
    if a != b and os.environ.get("MOMWIRE_403_DUMP"):
        # Diagnostic tap, kept from the momwire#403 hunt: byte-oracle
        # mismatches only ever reproduced under a loaded full-suite xdist
        # run, where no debugger reaches. Set the env var to a directory and
        # every mismatch drops its canonicalized (served, stock) pair there
        # for offline diffing.
        import time as _time

        stamp = f"{os.getpid()}-{_time.monotonic_ns()}"
        root = Path(os.environ["MOMWIRE_403_DUMP"])
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{stamp}-served.txt").write_text(a)
        (root / f"{stamp}-stock.txt").write_text(b)
    return a == b


# --------------------------------------------------------------------------
# the client is thin — the entire premise
# --------------------------------------------------------------------------


def test_the_client_imports_neither_momwire_nor_numpy():
    """The saving IS this import not happening.

    ~680 ms of the measured ~1 s engine lifetime was Python plus NumPy plus
    momwire's first fill; a client that imported the package to find out its
    own version would hand all of it straight back, and would do so invisibly
    — everything else would still work. Checked in a fresh interpreter that
    runs the client's real entry point on its most import-hungry path (the
    version probe reads distribution metadata), then asks what got loaded.
    """
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys\n"
            "sys.argv = ['momwire-nec2c-shared', '-version']\n"
            "try:\n"
            "    runpy.run_module('momwire_nec2c_client', run_name='__main__')\n"
            "except SystemExit:\n"
            "    pass\n"
            "print(sorted(m for m in sys.modules "
            "if m.split('.')[0] in ('numpy', 'scipy', 'momwire')), file=sys.stderr)\n",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stderr.strip().endswith("[]"), probe.stderr
    assert probe.stdout.strip() == nec_portal.PROBE_VERSION


def test_no_line_of_the_client_imports_momwire():
    """The rule above, at the source, where a lazy import inside a function
    would hide from any check of what got loaded at start-up.

    The one sanctioned exception is ``--selftest``, which ``execv``s the real
    module: a smoke test is not a live session, and it is an exec rather than
    an import precisely so the fast path cannot reach it.
    """
    source = Path(client.__file__).read_text()
    offenders = [
        f"{number}: {line.strip()}"
        for number, line in enumerate(source.splitlines(), 1)
        if re.match(r"\s*(import|from)\s+(momwire|numpy|scipy)\b", line)
    ]
    assert not offenders, (
        "the client must import stdlib only; found:\n  " + "\n  ".join(offenders)
    )


def test_the_version_probe_is_the_stock_engines_answer():
    """Reconstructed, not imported — so it has to be pinned against the
    original. Both shapes: the honest ``versionNECd`` identity and the
    ``--legacy-probe`` masquerade."""
    assert client.probe_version() == nec_portal.PROBE_VERSION
    assert client.probe_version(legacy=True) == nec_portal.LEGACY_PROBE_VERSION


def test_the_version_probe_answers_without_a_server(runtime):
    """SimNEC probes at configure time and at every engine start. A probe that
    spawned a server would put the cold start back in the one place a user
    watches for it — and would fail on a box where the socket cannot be made
    at all."""
    probe = run_client(runtime, "", "-version")
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.splitlines() == [nec_portal.PROBE_VERSION]
    assert sockets(runtime) == [], "the probe started a server"

    legacy = run_client(runtime, "", "-version", "--legacy-probe")
    assert legacy.stdout.splitlines() == [nec_portal.LEGACY_PROBE_VERSION]
    assert sockets(runtime) == []


def test_a_platform_without_af_unix_refuses_on_one_line(monkeypatch, capsys):
    """POSIX only, said out loud. Windows CPython gained ``AF_UNIX`` in 3.12
    and none of the spawn/lock discipline here has ever run there, so the
    client refuses rather than half-working — on one line, naming the engine
    that does work, and never as a traceback SimNEC would show as garbage."""
    monkeypatch.delattr(socket, "AF_UNIX", raising=False)
    assert client.main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(captured.err.splitlines()) == 1
    assert "momwire-nec2c" in captured.err


# --------------------------------------------------------------------------
# the byte oracle
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_fixture_deck_answers_exactly_what_the_stock_engine_answers(
    runtime, name
):
    """The correctness claim, at full corpus width.

    The client forwards bytes and the server runs the stock frame loop, so the
    printout must be the stock printout — banner, sections, columns, the ``NX``
    echo, and since momwire#456 phase C the two NETWORK blocks of the seven
    decks that carry a ``TL`` or an ``NT``. Timings are canonicalised because
    they are wall clock; nothing else is touched.

    Two of those blocks' numbers are residuals and had to be made deterministic
    before they could be gated here rather than after: a pinned port's voltage
    read back out of the solve printed ``0.0000E+00`` or ``-0.0000E+00`` by
    BLAS thread count, and a lossless network's loss printed ``0.0000E+00`` or
    ``-8.6736E-19``. Both are snapped at the source in ``_portal``; this test
    is what would have caught them, one run in six.
    """
    deck = fixture_deck(name)
    served = run_client(runtime, deck)
    assert served.returncode == 0, served.stderr
    assert same_printout(served.stdout, stock(deck)), name
    if name in REFUSED_NAMES:
        assert "ERROR:" in served.stdout
    else:
        assert NX_ECHO.search(served.stdout), "no sentinel — SimNEC blocks forever"


@pytest.mark.parametrize(
    "name",
    (
        "dipole_rp2_linear_cliff",
        "dipole_rp3_circular_cliff",
        "dipole_rp3_cliff_sommerfeld",
    ),
)
def test_pattern_dust_prints_as_exact_zero(name):
    """momwire#403's regression gate: no E column may print the angle of zero.

    A linear deck's cross-pol lands ~16 orders under the co-pol — pure fill
    round-off — and its printed magnitude AND phase move with process history
    (heap layout steers vectorized-math lane splits at the ULP level). That
    made served==stock order-sensitive: the byte oracle above reddened under
    xdist load while every serial replay of the same tests passed. The engine
    now zeroes any component under NEC's own _FIELD_FLOOR2 before it prints
    (a deliberate, documented departure from nec2c, which prints its own
    dust raw), so the whole class is gone: this asserts the cliff decks'
    pattern rows carry no sub-floor magnitudes and that every zero magnitude
    prints with a zero phase.
    """
    rows = [
        ln
        for ln in stock(fixture_deck(name)).splitlines()
        if len(ln.split()) == 12 and "LINEAR" in ln
    ]
    assert rows, f"{name}: no pattern rows found"
    for ln in rows:
        tok = ln.split()
        for mag, phase in ((tok[8], tok[9]), (tok[10], tok[11])):
            assert float(mag) == 0.0 or float(mag) > 1e-15, (
                f"{name}: sub-floor dust printed raw: {ln}"
            )
            if float(mag) == 0.0:
                assert float(phase) == 0.0, f"{name}: the angle of zero: {ln}"


def _near_field_rows(text: str) -> list[list[str]]:
    rows = []
    for ln in text.splitlines():
        tok = ln.split()
        if len(tok) != 9:
            continue
        try:
            float(tok[0])
        except ValueError:
            continue  # the header ("X"/"METERS" rows are 9 tokens too)
        rows.append(tok)
    return rows


def test_ne_dust_prints_as_exact_zero():
    """momwire#464's regression gate, sibling to the one above for the
    NEAR ELECTRIC/MAGNETIC FIELDS table: no E/H column may print the angle
    of zero.

    A z-oriented centre-fed dipole's field on the x-axis at the mid-plane is
    analytically zero in EX; the engine used to print the fill's rounding
    dust there instead — ``EX = 9.1237E-17`` against ``EZ = 2.6506E-01`` at
    the same point (``dipole_ne_nearfield``, x = -1/+1, y = 0, z = 0) — the
    same "angle of zero" class momwire#403 fixed for the RP table, left
    unfixed on this print path until now. Checked on both fixtures that
    exercise ``NE``/``NH`` (``fmt_near_field_row`` cannot tell which card
    produced its row): every near-field row's magnitude/phase pair is either
    exactly zero or a real, above-floor reading, and the two known dust
    cells print exact zero with exact zero phase.
    """
    for name in ("dipole_ne_nearfield", "dipole_nh_nearfield"):
        rows = _near_field_rows(stock(fixture_deck(name)))
        assert rows, f"{name}: no near-field rows found"
        for tok in rows:
            for mag, phase in ((tok[3], tok[4]), (tok[5], tok[6]), (tok[7], tok[8])):
                # The bar IS the floor: _FIELD_FLOOR2 is applied to the
                # printed value on this path, so anything the engine let
                # through must be strictly above 1e-10. (The RP sibling's
                # 1e-15 is a loose "no dust" heuristic because its floor is
                # tested in the pre-rescale FFLD basis, not on the printout.)
                assert float(mag) == 0.0 or float(mag) > 1e-10, (
                    f"{name}: sub-floor dust printed raw: {tok}"
                )
                if float(mag) == 0.0:
                    assert float(phase) == 0.0, f"{name}: the angle of zero: {tok}"

    dust_rows = [
        tok
        for tok in _near_field_rows(stock(fixture_deck("dipole_ne_nearfield")))
        if tok[0] in ("-1.0000", "1.0000") and tok[2] == "0.0000"
    ]
    assert len(dust_rows) == 2, dust_rows
    for tok in dust_rows:
        assert (tok[3], tok[4]) == ("0.0000E+00", "0.00"), tok


def _excitation_rows(text: str) -> list[list[str]]:
    """The STRUCTURE EXCITATION DATA rows of a printout: eleven numeric
    tokens, scanned only between that banner and the next blank-blank gap so
    no other eleven-token table can drift into the gate."""
    rows = []
    in_block = False
    for ln in text.splitlines():
        if "STRUCTURE EXCITATION DATA AT NETWORK CONNECTION POINTS" in ln:
            in_block = True
            continue
        if in_block:
            tok = ln.split()
            if len(tok) != 11:
                if not tok:
                    in_block = False
                continue
            try:
                int(tok[0])
            except ValueError:
                continue
            rows.append(tok)
    return rows


def test_connection_point_dust_prints_as_exact_zero():
    """momwire#403's dust floor, third print path (the second was #477's
    near-field table): an OPEN network connection point has an analytically
    zero current, and the row derived from it — impedance V/I, admittance
    I/V, power — must print exact zeros, never ~1e-19 A residue whose sign
    moves with process history and ~1e16 Ω of noise over noise.  CI's byte
    oracle caught exactly that on ``dipole_nt_all_zero`` on the first run
    after the #476 dump tap was armed; nec2c prints its own dust there
    (1.5e-17 A, 4.3e16 Ω in the committed capture), a documented departure.

    Generic half: every excitation row's current magnitude is either exactly
    zero (with every derived column exactly zero alongside it) or a real
    reading strictly above the floor.  Specific half: the all-zero NT's open
    endpoint row, pinned column by column."""
    for name in (
        "dipole_nt_all_zero",
        "dipole_nt_network",
        "dipole_tl_network",
        "dipole_tl_shunt_crossed",
        "dipole_ex6_gyrator",
    ):
        rows = _excitation_rows(stock(fixture_deck(name)))
        assert rows, f"{name}: no excitation rows found"
        for tok in rows:
            i_mag = abs(complex(float(tok[4]), float(tok[5])))
            assert i_mag == 0.0 or i_mag > 1e-10, (
                f"{name}: sub-floor connection current printed raw: {tok}"
            )
            if i_mag == 0.0:
                assert tok[4:] == ["0.0000E+00"] * 7, (
                    f"{name}: the digits of zero: {tok}"
                )

    open_rows = [
        tok
        for tok in _excitation_rows(stock(fixture_deck("dipole_nt_all_zero")))
        if tok[0] == "2"
    ]
    assert len(open_rows) == 1, open_rows
    assert open_rows[0][4:] == ["0.0000E+00"] * 7, open_rows[0]


def test_two_decks_down_one_connection_frame_the_way_the_stock_loop_does(runtime):
    """The banner belongs to the CONNECTION here, not to the process — the
    server is already warm when SimNEC's engine starts. One connection must
    therefore look exactly like one stock process: banner once, then a frame
    per ``NX``."""
    decks = fixture_deck("dipole_free_space") + fixture_deck("dipole_rp_pattern")
    served = run_client(runtime, decks)
    assert served.returncode == 0, served.stderr
    assert same_printout(served.stdout, stock(decks))
    assert len(NX_ECHO.findall(served.stdout)) == 2
    assert served.stdout.count("ANTENNA INPUT PARAMETERS") == 2


def test_a_deck_arriving_by_shell_redirect_is_answered(runtime, tmp_path):
    """``momwire-nec2c-shared < dipole.nec`` — the documented command-line use,
    and the one case where stdin is a REGULAR FILE rather than a pipe.

    Regression, not hypothetical: ``epoll`` (what ``selectors`` picks on Linux)
    refuses a regular file with ``EPERM``, so the first cut of the pump died on
    every redirect while working perfectly for SimNEC. The client falls back to
    ``select``, which reports a file permanently ready — the truth about a file.
    """
    source = tmp_path / "dipole.nec"
    source.write_text(deck_ending_in_en("dipole_free_space"))
    with source.open() as handle:
        served = subprocess.run(
            CLIENT_ARGV,
            stdin=handle,
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "MOMWIRE_PORTAL_RUNTIME_DIR": runtime},
        )
    assert served.returncode == 0, served.stderr
    assert same_printout(served.stdout, stock(source.read_text()))


def test_en_ends_the_connection_and_the_server_serves_the_next_one(runtime):
    """``EN`` ends a run (#901), which for a shared engine must mean the
    CONNECTION and not the server: a standalone ``.nec`` file redirected in
    would otherwise take the whole crew's engine down with it."""
    deck = deck_ending_in_en("dipole_free_space")
    first = run_client(runtime, deck)
    assert first.returncode == 0, first.stderr
    assert same_printout(first.stdout, stock(deck))

    second = run_client(runtime, deck)
    assert second.returncode == 0, second.stderr
    assert same_printout(second.stdout, stock(deck))
    assert len(server_pids(runtime)) == 1, "EN restarted the server"


def test_a_bad_deck_costs_its_own_connection_nothing_and_the_server_nothing(runtime):
    """Residency, one level up. A deck that raises is reported and stepped
    over inside the connection, and the next connection is served by the same
    process."""
    broken = "CE bogus\nGW 1 9 0. 0. -2.5 0. 0. 2.5 0.001\nGE 0\nZZ 1\nNX\n"
    good = fixture_deck("dipole_free_space")
    served = run_client(runtime, broken + good)
    assert served.returncode == 0, served.stderr
    assert served.stdout.count("ERROR-NEC2C") == 1
    assert len(NX_ECHO.findall(served.stdout)) == 2
    assert same_printout(served.stdout, stock(broken + good))
    assert len(server_pids(runtime)) == 1


def test_deck_warnings_go_to_the_server_log_not_back_down_the_socket(runtime):
    """Per-connection stderr is the server's, and there is nowhere honest to
    send it: ``NEC2Daemon`` never drains stderr (a full pipe buffer deadlocks
    the UI). The refusal contract is untouched by that — ``ERROR:`` lines live
    in the PRINTOUT and the sentinel is always answered — so the only thing
    rerouted is advice nobody was reading."""
    served = run_client(runtime, fixture_deck("split_dipole_qq_daemon_framed"))
    assert served.returncode == 0
    assert served.stderr == "", "deck stderr reached the client"
    assert "reducedField:2" in logs(runtime)


def test_a_body_unterminated_at_eof_is_served_over_the_socket(runtime):
    """#458: the connection's end of input terminates the deck the way NEC's
    own reader does. Through the shared server the stream ends when the client
    closes, so a body with its NX taken off must answer exactly as the same
    deck ending in EN — a dropped deck here would reach the host as an empty
    printout with nothing on any channel it reads."""
    served = run_client(runtime, deck_body("dipole_free_space"))
    assert served.returncode == 0
    assert served.stderr == ""
    assert same_printout(served.stdout, stock(deck_ending_in_en("dipole_free_space")))


def test_the_basis_flag_reaches_the_server_and_stamps_the_banner(runtime):
    """``--basis`` rides the portal-dialog command line, and the client keys
    the socket on it — so it must reach the server it names, and two entries
    differing only in the basis must be two servers."""
    deck = fixture_deck("dipole_free_space")
    default = run_client(runtime, deck)
    alternate = run_client(runtime, deck, "--basis", "sinusoidal")
    assert alternate.returncode == 0, alternate.stderr
    assert "+sin" in alternate.stdout
    assert "+sin" not in default.stdout
    assert same_printout(alternate.stdout, stock(deck, ["--basis", "sinusoidal"]))
    assert len(sockets(runtime)) == 2, "two engines shared one server"


# --------------------------------------------------------------------------
# which server: the socket is the configuration
# --------------------------------------------------------------------------


def test_the_same_engine_spelt_two_ways_is_one_server():
    """``--basis=X`` and ``--basis X`` are one engine, and bare flags do not
    care what order the dialog string put them in. Otherwise a user who
    retyped their portal entry would silently get a second 90 MB process with
    a cold cache."""
    equals, *_rest = client.split_argv(["--basis=sinusoidal", "--cache"])
    spaced, *_rest = client.split_argv(["--cache", "--basis", "sinusoidal"])
    assert client.config_key(equals, 900.0) == client.config_key(spaced, 900.0)


@pytest.mark.parametrize(
    "argv",
    [
        ["--basis", "sinusoidal"],
        ["--cache"],
        ["--legacy-probe"],
        ["--cache-stats", "/tmp/one.json"],
    ],
)
def test_a_different_engine_flag_is_a_different_server(argv):
    """Two portal-dialog entries differing in any engine flag are two engines
    — the doctrine the flags already carried, now with a process behind it.
    A shared server keyed on less than the full command line would answer one
    entry's deck with the other's physics."""
    engine, *_rest = client.split_argv(argv)
    assert client.config_key(engine, 900.0) != client.config_key([], 900.0)


def test_the_key_carries_the_version_and_the_interpreter(monkeypatch):
    """An upgrade or a different venv is a different engine, and must not be
    answered by whatever is still resident from before — the one failure this
    design could produce that a user would read as a wrong answer rather than
    as a crash."""
    baseline = client.config_key([], 900.0)
    monkeypatch.setattr(client, "probe_version", lambda legacy=False: "NEC2momwire.9.9")
    assert client.config_key([], 900.0) != baseline
    monkeypatch.undo()
    monkeypatch.setattr(sys, "executable", "/nowhere/python3")
    assert client.config_key([], 900.0) != baseline


def test_a_runtime_directory_owned_by_somebody_else_is_refused(monkeypatch, tmp_path):
    """A socket in a directory this user does not own is a socket somebody
    else can pre-empt. The ``/tmp`` fallback is the exposed case (``$XDG_RUNTIME_DIR``
    is private by construction), so ownership is checked, not assumed."""
    monkeypatch.setenv("MOMWIRE_PORTAL_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(os, "getuid", lambda: os.stat(tmp_path).st_uid + 1)
    with pytest.raises(PermissionError):
        client.runtime_dir()


def test_an_over_long_socket_path_is_refused_by_the_client(monkeypatch, tmp_path):
    """``sun_path`` is 108 bytes, and a name that overruns it fails at
    ``bind()`` inside a DETACHED server — where the only symptom is a client
    waiting for a socket that will never appear. Caught in the client so the
    message names the cause."""
    deep = tmp_path / ("d" * 90)
    deep.mkdir()
    monkeypatch.setenv("MOMWIRE_PORTAL_RUNTIME_DIR", str(deep))
    with pytest.raises(ValueError, match="AF_UNIX limit"):
        client.socket_path([], 900.0)


# --------------------------------------------------------------------------
# the point of the issue: a cache that outlives the client
# --------------------------------------------------------------------------


def test_a_deck_solved_by_one_invocation_is_served_to_the_next(runtime, tmp_path):
    """**The issue.** ``--cache``'s main case is a value retyped minutes later,
    and a process that lives ~1 s can never see it: by the time the user
    retypes, the engine that solved it is gone and so is its cache.

    Two SEPARATE client invocations, one deck, one server. The instrument is
    the stats file, not the printout — the whole correctness claim is that the
    printout is identical either way, so the printout cannot be the evidence.
    """
    stats = tmp_path / "stats.json"
    deck = fixture_deck("dipole_free_space")
    flags = ("--cache", "--cache-stats", str(stats))

    first = run_client(runtime, deck, *flags)
    assert first.returncode == 0, first.stderr
    assert json.loads(stats.read_text())["hits"] == 0, "nothing to hit yet"

    second = run_client(runtime, deck, *flags)
    assert second.returncode == 0, second.stderr
    tally = json.loads(stats.read_text())
    assert tally["hits"] == 1, tally
    assert tally["decks_rendered"] == 2, tally
    assert len(server_pids(runtime)) == 1, "two servers is two cold caches"
    assert same_printout(second.stdout, stock(deck)), "a hit must answer fresh bytes"


def test_the_control_a_fresh_server_has_nothing_to_serve(runtime, tmp_path):
    """The control the claim above needs: the hit came from the SERVER
    surviving, not from anything the client did. Same deck, same flags, a
    runtime directory whose server has been taken away — no hit."""
    stats = tmp_path / "stats.json"
    deck = fixture_deck("dipole_free_space")
    flags = ("--cache", "--cache-stats", str(stats))

    assert run_client(runtime, deck, *flags).returncode == 0
    assert json.loads(stats.read_text())["decks_rendered"] == 1

    for pid in server_pids(runtime):
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and sockets(runtime):
        time.sleep(0.05)
    assert sockets(runtime) == [], "the server did not go"

    assert run_client(runtime, deck, *flags).returncode == 0
    tally = json.loads(stats.read_text())
    assert tally["hits"] == 0, tally
    assert tally["decks_rendered"] == 1, "a fresh server counts from zero"
    assert len(server_pids(runtime)) == 2, "the second server never started"


# --------------------------------------------------------------------------
# the failure modes a resident server has and a per-burst process does not
# --------------------------------------------------------------------------


def test_sixteen_clients_at_once_start_exactly_one_server(runtime):
    """The spawn race, at crew width. SimNEC fires a whole crew at the same
    instant, into a directory with no server in it — every one of them fails
    to connect, and if each then spawned, the crew would be back to N
    processes with N cold caches plus a pile of losers fighting over one
    socket path.

    The lock is what makes it one, and the gate is both halves: one server,
    and sixteen complete correct printouts. Sixteen answers from one server
    that started twice would still be wrong.
    """
    deck = fixture_deck("dipole_free_space")
    expected = stock(deck)
    crew = [
        subprocess.Popen(
            CLIENT_ARGV,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "MOMWIRE_PORTAL_RUNTIME_DIR": runtime},
        )
        for _ in range(16)
    ]
    results = [member.communicate(deck, timeout=600) for member in crew]

    for member, (out, err) in zip(crew, results):
        assert member.returncode == 0, err
        assert same_printout(out, expected), err

    assert len(sockets(runtime)) == 1
    assert len(server_pids(runtime)) == 1, logs(runtime)
    assert "another server already owns" not in logs(runtime)


def test_a_client_killed_mid_solve_costs_its_own_answer_and_nothing_else(runtime):
    """SimNEC ends a session with ``Process.destroy()`` — a kill, not an EOF —
    and with one engine behind a crew that kill must not be reachable from
    one member to the rest. The solve completes (and warms the cache); its
    output is discarded because nobody is reading it; the server keeps
    serving.
    """
    deck = fixture_deck("dipole_free_space")
    assert run_client(runtime, deck).returncode == 0, "server should be warm first"
    pid_before = server_pids(runtime)

    victim = subprocess.Popen(
        CLIENT_ARGV,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env={**os.environ, "MOMWIRE_PORTAL_RUNTIME_DIR": runtime},
    )
    victim.stdin.write(deck)
    victim.stdin.flush()
    time.sleep(0.5)  # long enough for the deck to reach the server
    victim.kill()
    victim.wait(timeout=60)
    for pipe in (victim.stdin, victim.stdout):
        pipe.close()

    survivor = run_client(runtime, deck)
    assert survivor.returncode == 0, survivor.stderr
    assert same_printout(survivor.stdout, stock(deck))
    assert server_pids(runtime) == pid_before, "the server died with its client"
    assert _alive(pid_before[0])


def test_the_server_exits_after_its_idle_timeout_and_removes_its_socket(runtime):
    """Nobody comes back. A server that outlived every session would leave a
    90 MB process per engine configuration on a box whose owner never asked
    for one, and a socket file to trip over next time.

    Two seconds here for the same reason the default is 900: the number is a
    parameter, and the behaviour is what is being tested.
    """
    deck = fixture_deck("dipole_free_space")
    served = run_client(runtime, deck, "--idle-timeout", "2")
    assert served.returncode == 0, served.stderr
    assert len(sockets(runtime)) == 1
    (pid,) = server_pids(runtime)

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline and (sockets(runtime) or _alive(pid)):
        time.sleep(0.1)
    assert sockets(runtime) == [], "the socket outlived its server"
    assert not _alive(pid), "the server outlived its idle timeout"
    assert "idle 2s; exiting" in logs(runtime)


def test_a_stale_socket_file_is_replaced_rather_than_waited_on(runtime):
    """A server killed with ``SIGKILL`` cannot unlink its own socket, so the
    next client meets a path that exists and answers nothing. Removing it is
    only safe under the spawn lock — which is where it happens — and the
    symptom of getting this wrong is an engine that never starts again until
    somebody clears a directory by hand."""
    deck = fixture_deck("dipole_free_space")
    assert run_client(runtime, deck).returncode == 0
    (pid,) = server_pids(runtime)
    os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and _alive(pid):
        time.sleep(0.05)
    assert len(sockets(runtime)) == 1, "a SIGKILLed server tidied up after itself"

    revived = run_client(runtime, deck)
    assert revived.returncode == 0, revived.stderr
    assert same_printout(revived.stdout, stock(deck))
    assert len(server_pids(runtime)) == 2


# --------------------------------------------------------------------------
# packaging: the second console script SimNEC may be pointed at
# --------------------------------------------------------------------------

SHARED_ENTRY_POINT = "momwire-nec2c-shared"


def _console_scripts() -> dict[str, str]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"][
        "scripts"
    ]


def test_the_shared_console_script_obeys_simnecs_filename_rules():
    """Same two traps as the stock name, and they are not softer for a sibling:
    SimNEC decides the engine — dialect, daemon class, printout offsets — off
    the FILENAME, and refuses outright any command path containing ``out``.
    A tidier name (``momwire-portal-client``) is refused with a message that
    names neither momwire nor the cause."""
    scripts = _console_scripts()
    assert scripts.get(SHARED_ENTRY_POINT) == "momwire_nec2c_client:main"
    assert "nec2c" in SHARED_ENTRY_POINT.lower()
    assert "out" not in SHARED_ENTRY_POINT.lower()
    assert "nec5" not in SHARED_ENTRY_POINT.lower()
    assert "nec42" not in SHARED_ENTRY_POINT.lower()


def test_the_stock_console_script_is_untouched():
    """The stock engine is the supported default and this issue does not move
    it. Pinned here as well as in ``test_portal.py`` because the risk being
    guarded is a future tidy-up that rewires BOTH entries at once."""
    assert _console_scripts()["momwire-nec2c"] == "momwire.portal:main"


def test_the_client_module_is_shipped_as_a_top_level_module():
    """``py_modules``, not a package member: the entry point must resolve
    without importing ``momwire/__init__.py``, and a wheel that shipped the
    console script without the module behind it would fail at the version
    probe on a machine with no checkout."""
    source = (REPO_ROOT / "setup.py").read_text()
    assert 'py_modules=["momwire_nec2c_client"]' in source
    assert (REPO_ROOT / "src" / "momwire_nec2c_client.py").is_file()


def test_the_client_entry_point_target_resolves():
    """A typo in ``module:attr`` is only discovered by a user whose SimNEC
    session dies at the version probe."""
    module_name, _, attr = _console_scripts()[SHARED_ENTRY_POINT].partition(":")
    assert attr
    entry = getattr(__import__(module_name), attr)
    assert entry is client.main
