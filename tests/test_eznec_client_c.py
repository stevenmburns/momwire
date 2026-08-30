"""The native client's gates (#718 phase 3, momwire#532).

The claim under test is `scripts/eznec_client_c/momwire_eznec_client.c`'s
whole reason to exist: a plain-C exe IS the EZNEC engine contract — argv
``<deck> <printout>``, exit 0 always, a printout on every path — while
speaking the phase-2 wire protocol to the warm daemon at a launch cost the
frozen Python stub cannot reach.

The shape mirrors `test_serve_resident.py`'s e2e gate deliberately: same
deck, same one-shot oracle, same "exactly one ``listening pid=`` line"
counting, same short-runtime-dir rule. What is new here is that BOTH ends are
processes the tests build — a compiled client and a bundle whose engine is a
shim standing in for the frozen exe — so the gates below are about the
bundle's SHAPE (which name spawns which daemon, which rung catches which
failure) as much as about the bytes.

The fallback ladder is what makes those gates load-bearing rather than
decorative: a byte-identical printout proves nothing on its own, because rung
3 produces one by construction. Every gate that means to test residency
therefore also asserts the address file existed and the daemon logged
exactly the spawns it should have.

The Windows half of that source compiles only under cl.exe, so what runs here
is the POSIX half plus the source-level gates that keep the ``#ifdef _WIN32``
branch from rotting into a stub between canary runs.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import momwire_serve_client as mech
from momwire.eznec import _shell

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "scripts" / "eznec_client_c" / "momwire_eznec_client.c"
BUILD_CC = REPO / "scripts" / "eznec_client_c" / "build_cc.sh"
BUILD_MSVC = REPO / "scripts" / "eznec_client_c" / "build_msvc.bat"
ENTRY = REPO / "scripts" / "eznec_freeze" / "entry.py"
DECKS = Path(__file__).resolve().parent / "fixtures" / "eznec" / "decks"

# The names the phase-3 bundle ships: the C client takes the EZNEC-facing
# names and the ONE frozen exe behind them is the engine. `engine` is a
# consumed segment for that exe exactly as `client` is for the client, which
# is why the client must never be called that — it would select nothing and
# spawn itself.
CLIENT_NAME = "momwire-eznec"
TWIN_NAME = "momwire-eznec-razor-nec5"
TWIN_BASIS = "razor-nec5"
ENGINE_NAME = "momwire-eznec-engine"

# A name that asked for a basis and spelt none: the marker with nothing after
# it. momwire#733's divergence case — this is a NAMED (empty) basis, distinct
# from CLIENT_NAME's "no marker at all", and must key its own daemon.
EMPTY_NAME = "momwire-eznec-"

# smoke.py's gate-4 deck: a free-space dipole every basis hosts and on which
# the two SHIPPED bases disagree — bspline answers 85.073+45.369j where
# razor-nec5 answers 79.948+29.919j, the licensed engine's own number. A twin
# that ignored its filename and served the default would miss by ~16 ohm.
BASIS_DECK = "0010"

_CC = shutil.which(os.environ.get("CC", "")) or shutil.which("cc")

pytestmark = pytest.mark.skipif(
    _CC is None, reason="no C compiler on this box; the native client cannot be built"
)


def _deck(prefix: str) -> Path:
    return sorted(DECKS.glob(f"{prefix}*.nec"))[0]


def _oracle(deck_path: Path, tmp_path: Path, basis: str | None = None) -> bytes:
    """What the one-shot shell would write for this deck — the oracle."""
    out = tmp_path / f"oracle-{basis or 'default'}.out"
    if basis is None:
        _shell.run(deck_path, out)
    else:
        _shell.run(deck_path, out, basis=basis)
    return out.read_bytes()


# The stand-in for the frozen exe. It resolves its own basis from ITS argv[0]
# — the shim's path, which carries no marker — so the twin's choice reaches
# it only through `--basis`, which is exactly the deployed shape: one engine,
# named by the flag its spawner passes.
_ENGINE_SHIM = f'#!/bin/sh\nexec "{sys.executable}" "{ENTRY}" "$@"\n'

# The rung-3 bundle: an engine that cannot be a daemon but is a perfectly
# good one-shot. This is the failure the ladder exists for — a box where the
# resident path is impossible must still get a correct printout.
_REFUSING_SHIM = (
    "#!/bin/sh\n"
    'for arg in "$@"; do\n'
    '    if [ "$arg" = "--serve" ]; then exit 1; fi\n'
    "done\n"
    f'exec "{sys.executable}" "{ENTRY}" "$@"\n'
)


@pytest.fixture(scope="session")
def client_exe() -> Path:
    """The compiled client, built once by the script that ships it.

    Through `build_cc.sh` rather than a bare cc line here: the flags and the
    version defines ARE part of what is being gated (the momwire version is a
    hash input, and -Werror is the promise about buffers), and a second
    spelling of them in the tests would certify the wrong build.
    """
    room = Path(tempfile.mkdtemp(prefix="mw-cc-build-"))
    out = room / CLIENT_NAME
    proc = subprocess.run(
        [str(BUILD_CC), str(out)],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PYTHON": sys.executable},
    )
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()
    yield out
    shutil.rmtree(room, ignore_errors=True)


def _bundle(client_exe: Path, engine: str | None, names=(CLIENT_NAME,)) -> Path:
    """A bundle directory: client copies under `names`, plus an engine shim.

    ``mkdtemp`` and not ``tmp_path``: the runtime directory rule below is
    about sun_path, but the ENGINE path is a hash input, and a bundle nested
    as deeply as pytest's tmp would make the key differ from a deployed one
    for no reason worth the confusion.
    """
    room = Path(tempfile.mkdtemp(prefix="mw-cc-bundle-"))
    for name in names:
        shutil.copy2(client_exe, room / name)
    if engine is not None:
        shim = room / ENGINE_NAME
        shim.write_text(engine)
        shim.chmod(0o755)
    return room


def _room() -> Path:
    """A short private runtime directory.

    NOT under tmp_path: pytest's nested tmp directories can push the socket
    name past sun_path, the client then falls back to the one-shot, and the
    test would pass without testing residency at all (test_serve_resident's
    lesson, and the reason every gate here also counts address files).
    """
    return Path(tempfile.mkdtemp(prefix="mw-cc-room-"))


def _run(
    bundle: Path,
    name: str,
    args: list[str],
    room: Path,
    transport: str | None = None,
) -> subprocess.CompletedProcess:
    env = {**os.environ, "MOMWIRE_PORTAL_RUNTIME_DIR": str(room)}
    if transport is None:
        env.pop(mech.TRANSPORT_ENV, None)
    else:
        env[mech.TRANSPORT_ENV] = transport
    return subprocess.run(
        [str(bundle / name), *args],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )


def _listening(room: Path) -> list[str]:
    """Every ``listening pid=`` line a daemon in this room ever wrote.

    One line per spawned daemon: none means the resident path never ran (the
    ladder carried the test), two means a second spawn where one server was
    the whole claim.
    """
    return [
        line
        for log in room.glob("*.log")
        for line in log.read_text(errors="replace").splitlines()
        if "listening pid=" in line
    ]


def _stop(room: Path) -> None:
    """Ask every daemon this room started to go, by pid out of its own log."""
    for line in _listening(room):
        for word in line.split():
            if word.startswith("pid="):
                try:
                    os.kill(int(word.split("=", 1)[1]), signal.SIGTERM)
                except (OSError, ValueError):
                    pass
    shutil.rmtree(room, ignore_errors=True)


# --------------------------------------------------------------------------
# the source itself
# --------------------------------------------------------------------------


def test_the_build_script_makes_every_warning_an_error():
    """-Werror is the contract, not a preference: every warning this program
    can raise is about a buffer or a conversion, and both are the failure
    modes of a process that runs as an engine on somebody else's machine."""
    text = BUILD_CC.read_text()
    for flag in ("-std=c11", "-Wall", "-Wextra", "-Werror"):
        assert flag in text, flag


def test_the_windows_half_is_written_rather_than_stubbed():
    """The ``#ifdef _WIN32`` branch is compiled by cl.exe on the canary and
    by nothing here, so the only thing standing between it and rot is this:
    it must be complete Win32 — its own socket library, its own lock, its own
    detached spawn — with no placeholder anywhere in the file."""
    source = SOURCE.read_text()
    for placeholder in ("#warning", "TODO", "FIXME", "XXX"):
        assert placeholder not in source, placeholder
    for symbol in (
        "winsock2.h",
        "afunix.h",
        "WSAStartup",
        "closesocket",
        "SOCKET",
        "LockFileEx",
        "CreateProcessA",
        "DETACHED_PROCESS",
        "CREATE_NEW_PROCESS_GROUP",
    ):
        assert symbol in source, symbol
    # The static CRT is what makes the exe a single file a user can unzip
    # beside EZNEC without a redistributable.
    assert "/MT" in BUILD_MSVC.read_text()


def test_the_client_is_never_named_for_the_engine():
    """``momwire-eznec-engine`` as a CLIENT name would consume nothing — the
    consumed segment is ``client``, not ``engine`` — so such a client would
    select a basis called "engine" and, worse, name itself as its own spawn
    target. The bundle's names are a fact, so they are pinned here."""
    assert CLIENT_NAME != ENGINE_NAME
    assert TWIN_NAME != ENGINE_NAME
    assert f'"{ENGINE_NAME}"' in SOURCE.read_text()


# --------------------------------------------------------------------------
# the argv contract
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_argument_errors_print_and_write_nothing(client_exe, tmp_path):
    """EZNEC's own fault strings, exit 0, and no file — because on these two
    paths there is no output path to write one to."""
    bundle = _bundle(client_exe, _ENGINE_SHIM)
    room = _room()
    try:
        proc = _run(bundle, CLIENT_NAME, [], room)
        assert proc.returncode == 0
        assert "UNABLE TO OPEN FILE" in proc.stdout

        out = tmp_path / "never.out"
        proc = _run(
            bundle, CLIENT_NAME, [str(_deck(BASIS_DECK)), str(out), "extra"], room
        )
        assert proc.returncode == 0
        assert "UNABLE TO OPEN SECOND FILE" in proc.stdout
        assert not out.exists()
        assert _listening(room) == []
    finally:
        _stop(room)
        shutil.rmtree(bundle, ignore_errors=True)


# --------------------------------------------------------------------------
# the resident path
# --------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("transport", [mech.UNIX, mech.TCP])
def test_the_c_client_end_to_end_spawns_one_server_and_reuses_it(
    client_exe, tmp_path, transport
):
    """The deployment claim as a real process tree: two invocations of the
    compiled client, ONE spawned daemon, both printouts byte-identical to the
    one-shot's.

    Run once per transport with the override in the ENVIRONMENT rather than
    monkeypatched, because the client and the daemon it spawns are separate
    processes here and the point is that they agree about the transport
    without exchanging a word about it. The client's spawn PIN is what makes
    that true: it names the address file the transport it pinned would
    publish, and follows the override wherever the override points.
    """
    bundle = _bundle(client_exe, _ENGINE_SHIM)
    room = _room()
    deck = _deck(BASIS_DECK)
    expected = _oracle(deck, tmp_path)
    suffix = mech.address_suffix(transport)

    try:
        outs = []
        for index in range(2):
            out = tmp_path / f"c-e2e-{index}.out"
            proc = _run(bundle, CLIENT_NAME, [str(deck), str(out)], room, transport)
            assert proc.returncode == 0, proc.stderr
            outs.append(out.read_bytes())
            # The address file EXISTED — without this the fallback ladder
            # could carry the whole test at one-shot speed.
            addresses = sorted(p.name for p in room.glob(f"*{suffix}"))
            assert len(addresses) == 1, addresses

        assert outs[0] == expected
        assert outs[1] == expected
        assert len(_listening(room)) == 1, _listening(room)
    finally:
        _stop(room)
        shutil.rmtree(bundle, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.slow
def test_a_variant_named_copy_answers_in_the_basis_its_name_claims(
    client_exe, tmp_path
):
    """momwire#593's gate on the native client: the basis rides on the
    filename, so a copy named for one formulation must not serve another.

    Byte identity against the twin's own one-shot is half of it; the other
    half is the anti-coincidence check, because a printout is internally
    CONSISTENT whichever engine wrote it — the banner names whatever actually
    ran, and nothing in the file can reveal the substitution.
    """
    bundle = _bundle(client_exe, _ENGINE_SHIM, names=(CLIENT_NAME, TWIN_NAME))
    room = _room()
    deck = _deck(BASIS_DECK)
    twin_expected = _oracle(deck, tmp_path, basis=TWIN_BASIS)
    default_expected = _oracle(deck, tmp_path)
    assert twin_expected != default_expected, "the gate's deck stopped disagreeing"

    try:
        out = tmp_path / "twin.out"
        proc = _run(bundle, TWIN_NAME, [str(deck), str(out)], room)
        assert proc.returncode == 0, proc.stderr
        assert out.read_bytes() == twin_expected
        assert out.read_bytes() != default_expected
        assert len(_listening(room)) == 1, _listening(room)
    finally:
        _stop(room)
        shutil.rmtree(bundle, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.slow
def test_each_formulation_gets_its_own_warm_server(client_exe, tmp_path):
    """One warm server per formulation: the basis is part of the key, so the
    twin cannot be answered by whatever is resident for the default. Two
    names, two daemons, in ONE runtime directory — the same rule that makes
    two runs of one name share a single server (the e2e gate above)."""
    bundle = _bundle(client_exe, _ENGINE_SHIM, names=(CLIENT_NAME, TWIN_NAME))
    room = _room()
    deck = _deck(BASIS_DECK)

    try:
        for index, name in enumerate((CLIENT_NAME, TWIN_NAME)):
            out = tmp_path / f"per-basis-{index}.out"
            proc = _run(bundle, name, [str(deck), str(out)], room)
            assert proc.returncode == 0, proc.stderr
            assert out.is_file()
        assert len(_listening(room)) == 2, _listening(room)
        assert len(sorted(room.glob("*.sock"))) == 2
    finally:
        _stop(room)
        shutil.rmtree(bundle, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.slow
def test_a_named_empty_basis_never_shares_the_defaults_server(client_exe, tmp_path):
    """momwire#733's divergence: the default (no marker) and a name that asked
    for a basis and spelt none (the marker alone, ``momwire-eznec-``) are two
    DIFFERENT engines — the first spawns the default formulation, the second
    spawns a daemon that refuses every deck by name. Before the fix both
    reduced to the same ``config_key`` element (the bare, empty suffix) and
    shared ONE socket: whichever spawned first poisoned the other, silently
    answering the empty-named copy or silently refusing the default.

    The default runs FIRST so its daemon is the one that could have won the
    shared socket under the old key; the empty-named copy runs second and
    must still refuse under its OWN daemon rather than inherit the answer
    already resident."""
    bundle = _bundle(client_exe, _ENGINE_SHIM, names=(CLIENT_NAME, EMPTY_NAME))
    room = _room()
    deck = _deck(BASIS_DECK)
    default_expected = _oracle(deck, tmp_path)

    try:
        default_out = tmp_path / "divergence-default.out"
        proc = _run(bundle, CLIENT_NAME, [str(deck), str(default_out)], room)
        assert proc.returncode == 0, proc.stderr
        default_text = default_out.read_bytes().decode("latin-1")
        assert "ANTENNA INPUT PARAMETERS" in default_text
        assert default_out.read_bytes() == default_expected

        empty_out = tmp_path / "divergence-empty.out"
        proc = _run(bundle, EMPTY_NAME, [str(deck), str(empty_out)], room)
        assert proc.returncode == 0, proc.stderr
        empty_text = empty_out.read_bytes().decode("latin-1")
        assert "NEC ERROR" in empty_text
        assert "ANTENNA INPUT PARAMETERS" not in empty_text

        # One warm server per formulation, the empty suffix being a
        # formulation that refuses — never one shared socket poisoning
        # either printout.
        assert len(_listening(room)) == 2, _listening(room)
        assert len(sorted(room.glob("*.sock"))) == 2
    finally:
        _stop(room)
        shutil.rmtree(bundle, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(
    "name, basis_element",
    [
        pytest.param(TWIN_NAME, f"basis={TWIN_BASIS}", id="named"),
        pytest.param(CLIENT_NAME, "default", id="unnamed"),
    ],
)
def test_the_key_is_the_d2_digest_and_not_a_private_hash(
    client_exe, tmp_path, name, basis_element
):
    """#718 D2 pins the key's SHAPE: sha256 first-16-hex over
    ``eznec.<major>.<minor>``, the engine exe's resolved path, ``repr(900.0)``
    and the basis element, NUL-joined. Every other gate here only needs the
    key to agree with itself, so a drifted vendored SHA-256 or a reordered
    hash input would pass them all — but the key is observable as the address
    FILENAME, so the recorded shape is checkable against the Python spelling
    of the same digest.

    The basis element is ``"default"`` for a name that selects nothing and
    ``f"basis={suffix}"`` for a name that selects one — momwire#733's rule,
    checked here for BOTH spellings so the digest test pins the element, not
    just its presence."""
    bundle = _bundle(client_exe, _ENGINE_SHIM, names=(name,))
    room = _room()
    deck = _deck(BASIS_DECK)
    major, minor = mech.dist_version()
    expected_key = mech.digest(
        [
            f"eznec.{major}.{minor}",
            os.path.realpath(str(bundle / ENGINE_NAME)),
            repr(900.0),
            basis_element,
        ]
    )
    # The suffix the client's spawn pin chooses on this platform (unix on
    # POSIX, tcp on Windows) — the same rule `choose_pin` spells in C.
    suffix = mech.address_suffix(mech.TCP if os.name == "nt" else mech.UNIX)

    try:
        out = tmp_path / "key.out"
        proc = _run(bundle, name, [str(deck), str(out)], room)
        assert proc.returncode == 0, proc.stderr
        addresses = sorted(p.name for p in room.glob(f"*{suffix}"))
        assert addresses == [f"{expected_key}{suffix}"], addresses
    finally:
        _stop(room)
        shutil.rmtree(bundle, ignore_errors=True)


# --------------------------------------------------------------------------
# the ladder
# --------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
def test_rung_three_answers_when_no_daemon_can_start(client_exe, tmp_path):
    """The resident path failing anywhere must degrade to today's behavior —
    a byte-identical printout at one-shot speed — and must leave nothing
    behind, because a half-started daemon is worse than none."""
    bundle = _bundle(client_exe, _REFUSING_SHIM)
    room = _room()
    deck = _deck(BASIS_DECK)
    expected = _oracle(deck, tmp_path)

    try:
        out = tmp_path / "rung3.out"
        proc = _run(bundle, CLIENT_NAME, [str(deck), str(out)], room)
        assert proc.returncode == 0, proc.stderr
        assert out.read_bytes() == expected
        assert _listening(room) == []
        assert sorted(room.glob("*.sock")) == []
        assert sorted(room.glob("*.port")) == []
    finally:
        _stop(room)
        shutil.rmtree(bundle, ignore_errors=True)


@pytest.mark.integration
def test_rung_four_writes_a_named_refusal_and_still_exits_zero(client_exe, tmp_path):
    """No engine at all. The file is the only channel EZNEC reads, so an
    absent printout reads as a broken install; a named refusal at least
    reaches the user. Exit 0 either way — the code is not a channel."""
    bundle = _bundle(client_exe, None)
    room = _room()

    try:
        out = tmp_path / "rung4.out"
        proc = _run(bundle, CLIENT_NAME, [str(_deck(BASIS_DECK)), str(out)], room)
        assert proc.returncode == 0, proc.stderr
        text = out.read_bytes().decode("latin-1")
        assert "NEC ERROR" in text
        assert "MOMWIRE ENGINE UNAVAILABLE" in text
        assert text.endswith("\r\n")
    finally:
        _stop(room)
        shutil.rmtree(bundle, ignore_errors=True)
