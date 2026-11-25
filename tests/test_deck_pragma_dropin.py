"""Per-deck feature pragmas through the drop-ins (`_features.pragma_scope`).

A ``CM momwire: <name>`` card turns `pair_reuse` or `rotational_symmetry` on
for its own deck, as the feature's variable set on would. Held here, with
the route counted each time so a stamp alone cannot pass:

- the pragma alone turns each feature on, through the NEC-5 and NEC-4.2
  drop-ins and the nec2c portal: line 2 names it where the printout has a
  line 2, the reuse is SERVED (its fill report) or the sector route RAN
  (its solves counted), and the printout is the variable's to the byte;
- MOMWIRE_X_<NAME>=0 beats it;
- where the route cannot serve the deck it declines, as under the variable,
  and never refuses the deck;
- an ignored name is named on line 2 and moves nothing else;
- per deck, not per process: a stream, the portal's operator cache and ONE
  shared warm server each answer decks without, with, and again without the
  pragma, each as its own one-shot answer.
"""

from __future__ import annotations

import io
import re
import threading
import warnings

import pytest
from test_rotational_symmetry_dropin_1338 import Probe, screen

from momwire import _features as F
from momwire import _rotational_symmetry as rs
from momwire.dropin.nec2c import _portal
from momwire.dropin.nec5 import _resident, _shell

PAIR = "MOMWIRE_X_PAIR_REUSE"
ROT = "MOMWIRE_X_ROTATIONAL_SYMMETRY"
_CLOCK = re.compile(r"msec|SEC\.|TIME|Time", re.I)

# Two parallel dipoles, 402 segments: past every solver's small-deck
# threshold, and every band of pairs repeats (momwire#1971's free deck).
DIPOLES = (
    "CM two parallel dipoles, free space, 402 segments\nCE\n"
    "GW 1 201 0 -5 10 0 5 10 0.001\nGW 2 201 3 -5 10 3 5 10 0.001\nGE 0\n"
    "EX 0 1 101 0 1 0\nFR 0 1 0 0 14.0 0\nXQ\n"
)


def pragma(name, deck):
    return f"CM momwire: {name}\n" + deck


@pytest.fixture
def no_features(monkeypatch):
    for f in F.FEATURES.values():
        for var in (f.env, *f.aliases):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(F, "_PRAGMA_WITHHELD", frozenset())
    return monkeypatch


class Solvers:
    """Every solver the run bound, with what it bound (`_features.bind`)."""

    def __init__(self, monkeypatch):
        self.seen = []
        original = F.bind

        def recorded(solver, kind, **kw):
            out = original(solver, kind, **kw)
            self.seen.append(solver)
            return out

        monkeypatch.setattr(F, "bind", recorded)

    def reuse(self):
        """Each solver's last fill's reuse state (None: not asked)."""
        out = []
        for s in self.seen:
            rep = s.fill_report(wires=False)["reuse"]
            out.append(
                None if rep is None else (rep["state"], (rep["reused"] or 0) > 0)
            )
        return out


def unclocked(text):
    return [s for s in text.splitlines() if not _CLOCK.search(s)]


def nec5(deck, basis, dialect="nec5"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _shell.render(deck + "EN\n", basis=basis, dialect=dialect)


def portal(deck, basis="sinusoidal", argv=(), quiet=True):
    out = io.StringIO()
    with warnings.catch_warnings():
        if quiet:
            warnings.simplefilter("ignore")
        with _portal.engine_scope():
            rc = _portal.main(
                list(argv),
                stdin=io.StringIO(deck),
                stdout=out,
                stderr=io.StringIO(),
                prog=f"momwire-nec2c-{basis}",
            )
    assert rc == 0
    return out.getvalue()


# --- pair_reuse ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("dialect", "basis"),
    [("nec5", "razor2p"), ("nec5", "bs2"), ("nec4", "bs2")],
)
def test_pair_reuse_pragma_alone_turns_it_on(no_features, dialect, basis):
    deck = pragma("pair_reuse", DIPOLES)
    with no_features.context() as m:
        probe = Solvers(m)
        text = nec5(deck, basis, dialect)
    assert "ERROR" not in text, text[-600:]
    assert text.splitlines()[1].endswith(" +pair_reuse"), text.splitlines()[1]
    assert probe.reuse() == [("served", True)]
    # It is the variable, to the byte.
    no_features.setenv(PAIR, "1")
    assert unclocked(nec5(deck, basis, dialect)) == unclocked(text)


def test_pair_reuse_pragma_through_the_portal(no_features):
    deck = pragma("pair_reuse", DIPOLES) + "NX\n"
    with no_features.context() as m:
        probe = Solvers(m)
        text = portal(deck)
    assert probe.reuse() == [("served", True)]
    no_features.setenv(PAIR, "1")
    assert unclocked(portal(deck)) == unclocked(text)


@pytest.mark.parametrize("basis", ["razor2p", "bs2"])
def test_the_variable_off_beats_the_pair_reuse_pragma(no_features, basis):
    no_features.setenv(PAIR, "0")
    deck = pragma("pair_reuse", DIPOLES)
    with no_features.context() as m:
        probe = Solvers(m)
        text = nec5(deck, basis)
    assert probe.reuse() == [None]
    assert "+" not in text.splitlines()[1]
    no_features.delenv(PAIR)
    plain = nec5(
        deck.replace("CM momwire: pair_reuse", "CM momwire- pair_reuse"), basis
    )
    assert (
        unclocked(text)[2:]
        == [
            s.replace("momwire- pair_reuse", "momwire: pair_reuse")
            for s in unclocked(plain)
        ][2:]
    )


def test_the_variable_off_beats_it_through_the_portal(no_features):
    no_features.setenv(PAIR, "0")
    with no_features.context() as m:
        probe = Solvers(m)
        portal(pragma("pair_reuse", DIPOLES) + "NX\n")
    assert probe.reuse() == [None]


def test_pair_reuse_pragma_declines_a_small_deck(no_features):
    # One 41-segment dipole: under the sinusoidal plan's small-deck
    # threshold, so the plan steps aside by name and the fill runs as off,
    # as under the variable. Never a refusal.
    small = (
        "CM momwire: pair_reuse\nCE one dipole\n"
        "GW 1 41 0 -2.6 8 0 2.6 8 0.001\nGE 0\nEX 0 1 21 0 1 0\n"
        "FR 0 1 0 0 28.0 0\nXQ\nNX\n"
    )
    with no_features.context() as m:
        probe = Solvers(m)
        text = portal(small)
    assert "ERROR" not in text
    (solver,) = probe.seen
    assert isinstance(solver.pair_reuse_declined, str)
    assert probe.reuse()[0][0] == "declined"


# --- rotational_symmetry ------------------------------------------------------------


def _rot_run(monkeypatch, route, basis, deck):
    with monkeypatch.context() as m:
        probe = Probe(m)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if route == "nec2c":
                text = portal(deck, basis, quiet=False)
            else:
                text = _shell.render(deck, basis=basis, dialect=route)
    declined = [
        str(w.message)
        for w in caught
        if issubclass(w.category, rs.RotationalSymmetryDeclined)
    ]
    assert "ERROR" not in text, text[-600:]
    return text, probe, declined


def _rot(dialect, **kw):
    return pragma("rotational_symmetry", screen(dialect, **kw))


@pytest.mark.parametrize(
    ("route", "basis", "dialect"),
    [
        ("nec5", "bs2", "nec5"),
        ("nec4", "bs2", "nec4"),
        ("nec2c", "bs2", "nec2"),
        ("nec2c", "sinusoidal", "nec2"),
    ],
)
def test_rotational_symmetry_pragma_alone_turns_it_on(
    no_features, route, basis, dialect
):
    deck = _rot(dialect)
    text, probe, declined = _rot_run(no_features, route, basis, deck)
    assert probe.sector_solves > 0 and any(probe.mapped)
    assert declined == []
    if route != "nec2c":  # the NEC-2 printout has no line 2
        assert text.splitlines()[1].endswith(" +rotational_symmetry")
    no_features.setenv(ROT, "1")
    by_var, _probe, _ = _rot_run(no_features, route, basis, deck)
    assert unclocked(by_var) == unclocked(text)


@pytest.mark.parametrize(("route", "dialect"), [("nec5", "nec5"), ("nec2c", "nec2")])
def test_the_variable_off_beats_the_rotational_symmetry_pragma(
    no_features, route, dialect
):
    no_features.setenv(ROT, "0")
    text, probe, declined = _rot_run(no_features, route, "bs2", _rot(dialect))
    assert probe.sector_solves == 0 and not any(probe.mapped)
    assert declined == []
    if route != "nec2c":
        assert "+" not in text.splitlines()[1]


@pytest.mark.parametrize(
    ("route", "dialect", "basis"),
    [("nec5", "nec5", "bs2"), ("nec2c", "nec2", "sinusoidal")],
)
def test_rotational_symmetry_pragma_declines_a_deck_it_cannot_serve(
    no_features, route, dialect, basis
):
    # Radial 0 a metre longer: not N copies of one sector. Under the
    # variable this declines (dense, warned, `sector.declined`); the pragma
    # is the variable, so the same, and never the keyword's refusal.
    deck = _rot(dialect, long_radial=True)
    with no_features.context() as m:
        solvers = Solvers(m)
        text, probe, declined = _rot_run(m, route, basis, deck)
    assert probe.sector_solves == 0
    assert len(declined) >= 1, declined
    assert all(s.rotational_symmetry_declined for s in solvers.seen)
    assert all(s.fill_report()["sector"]["declined"] for s in solvers.seen)
    no_features.setenv(ROT, "1")
    by_var, _probe, declined_var = _rot_run(no_features, route, basis, deck)
    assert unclocked(by_var) == unclocked(text)
    assert declined_var == declined


def test_razor2p_ignores_the_rotational_symmetry_pragma_as_the_variable(no_features):
    # razor2p does not honour it: not run and not named, and not "ignored"
    # either (it is a pragma feature here; the solver is what ignores it).
    text, probe, _ = _rot_run(no_features, "nec5", "razor2p", _rot("nec5"))
    assert probe.sector_solves == 0
    line2 = text.splitlines()[1]
    assert "+" not in line2 and "pragma" not in line2


# --- names it does not take --------------------------------------------------------


@pytest.mark.parametrize(
    ("dialect", "basis"), [("nec5", "bs2"), ("nec5", "razor2p"), ("nec4", "sinusoidal")]
)
def test_an_ignored_name_is_named_on_line_2_and_moves_nothing_else(
    no_features, dialect, basis
):
    base = screen(dialect)
    deck = "CM momwire: mixed_lu\nCM momwire: Bogus\n" + base
    other = "CM momwire- mixed_lu\nCM momwire- Bogus\n" + base
    with no_features.context() as m:
        probe = Solvers(m)
        text = _shell.render(deck, basis=basis, dialect=dialect)
    lines, plain = text.splitlines(), _shell.render(other, basis=basis, dialect=dialect)
    assert lines[1] == plain.splitlines()[1] + " (pragma ignored: mixed_lu, bogus)"
    assert (
        unclocked(text)[2:]
        == [s.replace("momwire-", "momwire:") for s in unclocked(plain)][2:]
    )
    assert all(not getattr(s, "_mixed_lu", False) for s in probe.seen)


def test_the_note_follows_the_features_on_line_2(no_features):
    deck = "CM momwire: pair_reuse\nCM momwire: nope\n" + DIPOLES
    line2 = nec5(deck, "razor2p").splitlines()[1]
    assert line2.endswith(" +pair_reuse (pragma ignored: nope)"), line2


def test_the_nec2d_printout_names_both_on_line_2(no_features, tmp_path):
    # 4nec2's NEC-2D layout carries momwire's line 2 too.
    from momwire.dropin.nec2c import engine_scope, main

    deck, out = tmp_path / "d.nec", tmp_path / "d.out"
    deck.write_bytes(("CM momwire: pair_reuse\nCM momwire: zzz\n" + DIPOLES).encode())
    with no_features.context() as m:
        probe = Solvers(m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with engine_scope():
                rc = main(
                    [str(deck), str(out)],
                    stdin=io.StringIO(""),
                    prog="momwire-nec2d-bs2",
                )
    assert rc == 0
    line2 = out.read_bytes().decode("latin-1").splitlines()[1]
    assert line2.endswith(" +pair_reuse (pragma ignored: zzz)"), line2
    assert probe.reuse() == [("served", True)]


def test_a_long_ignored_name_keeps_line_2_in_its_columns(no_features):
    deck = f"CM momwire: {'x' * 120}\n" + DIPOLES
    line2 = nec5(deck, "razor2p").splitlines()[1]
    assert len(line2) <= 79 and line2.startswith(" momwire "), line2


# --- per deck, not per process ----------------------------------------------------------


def test_a_stream_scopes_each_structure(no_features):
    # SimNEC's stream: one process, structure after structure. The pragma
    # holds for its own structure only.
    pieces = [
        DIPOLES + "NX\n",
        pragma("pair_reuse", DIPOLES) + "NX\n",
        DIPOLES + "NX\n",
    ]
    data = [p.encode("latin-1") for p in pieces]
    chunks = [*data, b""]
    outs = []
    with no_features.context() as m:
        probe = Solvers(m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert _shell.stream(lambda: chunks.pop(0), outs.append, basis="bs2") == 0
    assert probe.reuse() == [None, ("served", True), None]
    line2s = [o.decode("latin-1").split("\n")[1].rstrip("\r") for o in outs]
    assert line2s[1] == line2s[0] + " +pair_reuse" and line2s[2] == line2s[0]


def test_the_portal_cache_keys_its_entries_on_the_pragmas(no_features):
    # `--cache`: two decks alike but for the pragma describe one operator,
    # and the solver reads the pragma at construction, so they must not
    # share an entry (`_operator_key`).
    stream = (
        DIPOLES + "NX\n" + pragma("pair_reuse", DIPOLES) + "NX\n" + DIPOLES + "NX\n"
    )
    with no_features.context() as m:
        probe = Solvers(m)
        portal(stream, argv=["--cache"])
    # three decks, two operators: the third deck is the first one's hit
    assert probe.reuse() == [None, ("served", True)]


@pytest.fixture
def warm_server(monkeypatch):
    """ONE live in-process eznec server (`test_serve_resident`'s shape)."""
    import shutil
    import tempfile
    from pathlib import Path

    import momwire_serve_client as mech
    from momwire.serve._server import Server

    monkeypatch.setenv(mech.TRANSPORT_ENV, mech.TCP)
    room = Path(tempfile.mkdtemp(prefix="mwp-"))
    path = str(room / f"srv{mech.address_suffix(mech.TCP)}")
    log = io.StringIO()

    def connection(conn, number, log_stream, solve_lock):
        _resident._connection(conn, number, log_stream, solve_lock, basis="bs2")

    server = Server(path, idle_timeout=3600.0, log=log, connection=connection)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    assert server.listening.wait(10), "the server never started listening"
    try:
        yield path
    finally:
        server.stop()
        thread.join(timeout=10)
        shutil.rmtree(room, ignore_errors=True)


def _ask(path, deck_bytes):
    import socket

    import momwire_serve_client as mech

    conn = mech.connect(path)
    assert conn is not None
    try:
        conn.sendall(deck_bytes)
        conn.shutdown(socket.SHUT_WR)
        chunks = []
        while chunk := conn.recv(65536):
            chunks.append(chunk)
    finally:
        conn.close()
    return b"".join(chunks)


def test_one_warm_server_answers_each_deck_under_its_own_pragmas(
    no_features, warm_server
):
    """Design item 6: the pragma is the solve's, not the server's. One
    server started with nothing set answers a deck without the pragma, the
    same deck with it, and the first again: each is its own one-shot answer
    (the bytes, line 2 included), and the reuse is served for the second
    alone."""
    plain = DIPOLES + "EN\n"
    asked = pragma("pair_reuse", DIPOLES) + "EN\n"
    with no_features.context() as m:
        probe = Solvers(m)
        answers = [
            _ask(warm_server, d.encode("latin-1")) for d in (plain, asked, plain)
        ]
    assert probe.reuse() == [None, ("served", True), None]
    texts = [a.decode("latin-1") for a in answers]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shots = [_shell.render(d, basis="bs2") for d in (plain, asked, plain)]
    for text, shot in zip(texts, shots, strict=True):
        assert unclocked(text.replace("\r\n", "\n")) == unclocked(shot)
    assert texts[1].split("\r\n")[1].endswith(" +pair_reuse")
    assert "+" not in texts[0].split("\r\n")[1]
    assert texts[0] == texts[2] or unclocked(texts[0]) == unclocked(texts[2])


def test_one_spawned_server_answers_each_deck_under_its_own_pragmas(tmp_path):
    """The same through the real launcher (`momwire_eznec_client`) and the
    server it spawns, worker processes and all: three calls, ONE server,
    each printout the one-shot's."""
    import os
    import shutil
    import signal
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    import momwire_serve_client as mech

    room = Path(tempfile.mkdtemp(prefix="mwp-"))
    env = {k: v for k, v in os.environ.items() if not k.startswith(F.ENV_PREFIX)} | {
        "MOMWIRE_PORTAL_RUNTIME_DIR": str(room),
        mech.TRANSPORT_ENV: mech.TCP,
    }
    decks = [DIPOLES + "EN\n", pragma("pair_reuse", DIPOLES) + "EN\n", DIPOLES + "EN\n"]
    try:
        outs = []
        for i, text in enumerate(decks):
            deck = tmp_path / f"d{i}.nec"
            deck.write_text(text, encoding="latin-1", newline="")
            out = tmp_path / f"d{i}.out"
            proc = subprocess.run(
                [sys.executable, "-m", "momwire_eznec_client", str(deck), str(out)],
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
            )
            assert proc.returncode == 0, proc.stderr
            outs.append(out.read_bytes().decode("latin-1"))
            sockets = sorted(room.glob(f"*{mech.address_suffix(mech.TCP)}"))
            assert len(sockets) == 1, sockets
        listening = [
            line
            for log in room.glob("*.log")
            for line in log.read_text(errors="replace").splitlines()
            if "listening pid=" in line
        ]
        assert len(listening) == 1, listening
    finally:
        for log in room.glob("*.log"):
            for word in log.read_text(errors="replace").split():
                if word.startswith("pid="):
                    try:
                        os.kill(int(word.split("=", 1)[1]), signal.SIGTERM)
                    except (OSError, ValueError):
                        pass
        shutil.rmtree(room, ignore_errors=True)
    line2s = [o.split("\r\n")[1] for o in outs]
    assert line2s[1] == line2s[0] + " +pair_reuse", line2s
    assert line2s[2] == line2s[0] and "+" not in line2s[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shots = [_shell.render(d, basis=_shell._serve.BASIS) for d in decks]
    for text, shot in zip(outs, shots, strict=True):
        assert unclocked(text.replace("\r\n", "\n")) == unclocked(shot)
