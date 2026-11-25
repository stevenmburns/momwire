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
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import momwire_serve_client as mech
from momwire.dropin.nec5 import _shell

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "scripts" / "eznec_client_c" / "momwire_eznec_client.c"
BUILD_CC = REPO / "scripts" / "eznec_client_c" / "build_cc.sh"
BUILD_MSVC = REPO / "scripts" / "eznec_client_c" / "build_msvc.bat"
ENTRY = REPO / "scripts" / "eznec_freeze" / "entry.py"
DECKS = Path(__file__).resolve().parent / "fixtures" / "eznec" / "decks"

# The `eznec-` filename rule, which the C client still reads for a renamed
# copy although the bundle stopped shipping `momwire-eznec*` names in
# momwire#1583; the shipped names are the NEC5_* and NEC4_* ones below. The
# ONE frozen exe behind every launcher is the engine, `momwire-engine` since
# momwire#1583, and the client must never be called that — it would spawn
# itself.
CLIENT_NAME = "momwire-eznec"
TWIN_NAME = "momwire-eznec-razor2p"
TWIN_BASIS = "razor2p"
ENGINE_NAME = "momwire-engine"

# A name that asked for a basis and spelt none: the marker with nothing after
# it. momwire#733's divergence case — this is a NAMED (empty) basis, distinct
# from CLIENT_NAME's "no marker at all", and must key its own daemon.
EMPTY_NAME = "momwire-eznec-"

# SimNEC's NEC-5 slot (momwire#1239): SimNEC reads the deck syntax off `nec5`
# in the engine path, so the same launcher ships again under these names and
# reads the basis past `nec5-` when `eznec-` is absent.
NEC5_NAME = "momwire-nec5"
NEC5_TWIN_NAME = "momwire-nec5-razor2p"

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


def _is_identity_var(name: str) -> bool:
    """A variable the server key reads from the environment (momwire#1979):
    a junction rule, a feature (or an alias of one), a thread count."""
    return (
        name == mech.JUNCTION_RULE_ENV
        or name.startswith(mech.FEATURE_ENV_PREFIX)
        or name in mech.FEATURE_ALIASES
        or name in mech.THREAD_ENVS
    )


def _python_env_elements() -> list[str]:
    """What the Python client appends to its key for `os.environ`."""
    return (
        mech.junction_rule_identity() + mech.feature_identity() + mech.thread_identity()
    )


def _run(
    bundle: Path,
    name: str,
    args: list[str],
    room: Path,
    transport: str | None = None,
    *,
    stdin: bytes | None = None,
    cwd: Path | None = None,
    identity: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """The launcher as a host runs it. ``stdin`` is what arrives there --
    nothing (``/dev/null``) unless a test is speaking 4nec2's form, so a
    test runner's own stdin never reaches the launcher (momwire#1488).
    ``identity``, when given, replaces every server-identity variable of
    this environment (`_is_identity_var`) with exactly these."""
    env = {**os.environ, "MOMWIRE_PORTAL_RUNTIME_DIR": str(room)}
    if identity is not None:
        env = {k: v for k, v in env.items() if not _is_identity_var(k)}
        env.update(identity)
    if transport is None:
        env.pop(mech.TRANSPORT_ENV, None)
    else:
        env[mech.TRANSPORT_ENV] = transport
    proc = subprocess.run(
        [str(bundle / name), *args],
        capture_output=True,
        timeout=300,
        env=env,
        cwd=cwd,
        **({"stdin": subprocess.DEVNULL} if stdin is None else {"input": stdin}),
    )
    return subprocess.CompletedProcess(
        proc.args,
        proc.returncode,
        proc.stdout.decode(errors="replace"),
        proc.stderr.decode(errors="replace"),
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
    """A client named for the engine would name itself as its own spawn
    target. The bundle's names are a fact, so they are pinned here."""
    for name in (CLIENT_NAME, TWIN_NAME, NEC5_NAME, NEC5_TWIN_NAME):
        assert name != ENGINE_NAME
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


@pytest.mark.integration
@pytest.mark.parametrize("name", [NEC5_NAME, NEC5_TWIN_NAME, CLIENT_NAME])
def test_the_version_probe_answers_without_an_engine(client_exe, name):
    """SimNEC's configure step: ``"<path>" -version``, exit 0, first line
    ``NEC5 <name> [sb] <version>`` — the Python client's line (momwire#1606),
    from the version compiled in, with no daemon spawned and no file
    written."""
    bundle = _bundle(client_exe, None, names=(name,))
    room = _room()
    try:
        proc = _run(bundle, name, ["-version"], room)
        import momwire_eznec_client

        assert proc.returncode == 0
        assert proc.stdout.splitlines()[0] == momwire_eznec_client.probe_version(name)
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
def test_a_nec5_named_copy_answers_in_the_basis_past_the_marker(client_exe, tmp_path):
    """momwire#1239: ``momwire-nec5-razor2p`` serves razor2p, not the
    default — the same anti-coincidence pair as the EZNEC twin's gate."""
    bundle = _bundle(client_exe, _ENGINE_SHIM, names=(NEC5_TWIN_NAME,))
    room = _room()
    deck = _deck(BASIS_DECK)
    expected = _oracle(deck, tmp_path, basis="razor2p")
    assert expected != _oracle(deck, tmp_path), "the gate's deck stopped disagreeing"
    try:
        out = tmp_path / "nec5-twin.out"
        proc = _run(bundle, NEC5_TWIN_NAME, [str(deck), str(out)], room)
        assert proc.returncode == 0, proc.stderr
        assert out.read_bytes() == expected
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
        # The nec5 names key the SAME servers as their EZNEC twins: the
        # element is the basis, never the spelling that selected it.
        pytest.param(NEC5_TWIN_NAME, "basis=razor2p", id="nec5-named"),
        pytest.param(NEC5_NAME, "default", id="nec5-unnamed"),
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
            mech.WIRE,
            mech.ADVISORIES_WIRE,
            basis_element,
            # momwire#1979: the environment this run inherits (an xdist
            # worker pins the thread counts), appended as Python appends it.
            *_python_env_elements(),
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


# --------------------------------------------------------------------------
# EZNEC's NEC-4.2 slot (momwire#1295)

NEC4_TWIN_NAME = "momwire-nec42-sinusoidal"
NEC42_DECK = (
    Path(__file__).resolve().parent / "fixtures" / "eznec_nec42" / "0224_dipole-ex0.nec"
)


def _unclocked(printout: bytes) -> bytes:
    """The NEC-4.2 printout carries this engine's own timings — the fill in
    MATRIX TIMING and the wall time in RUN TIME — so two runs of one deck
    differ there only."""
    import re

    printout = re.sub(rb"FILL= *\d+\.\d+ SEC\.", b"FILL= - SEC.", printout)
    # FACTOR= is measured too since momwire#1896 (the LU apart from the fill)
    printout = re.sub(rb"FACTOR= *\d+\.\d+ SEC\.", b"FACTOR= - SEC.", printout)
    return re.sub(rb"RUN TIME = *\d+\.\d+", b"RUN TIME = -", printout)


def _nec4_oracle(tmp_path: Path) -> bytes:
    out = tmp_path / "oracle-nec4-sinusoidal.out"
    _shell.run(NEC42_DECK, out, basis="sinusoidal", dialect="nec4")
    return _unclocked(out.read_bytes())


@pytest.mark.integration
@pytest.mark.slow
def test_a_nec4_named_copy_answers_the_nec42_slot_on_its_own_server(
    client_exe, tmp_path
):
    """``momwire-nec42-sinusoidal`` reads the deck in the nec4 dialect and
    solves it point-matched, under a key no NEC-5 name can share: the
    ``dialect=nec4`` element precedes the basis element in the D2 digest."""
    bundle = _bundle(client_exe, _ENGINE_SHIM, names=(NEC4_TWIN_NAME,))
    room = _room()
    expected = _nec4_oracle(tmp_path)
    assert b"ANTENNA INPUT PARAMETERS" in expected
    major, minor = mech.dist_version()
    expected_key = mech.digest(
        [
            f"eznec.{major}.{minor}",
            os.path.realpath(str(bundle / ENGINE_NAME)),
            repr(900.0),
            mech.WIRE,
            mech.ADVISORIES_WIRE,
            "dialect=nec4",
            "basis=sinusoidal",
            *_python_env_elements(),
        ]
    )
    suffix = mech.address_suffix(mech.TCP if os.name == "nt" else mech.UNIX)
    try:
        out = tmp_path / "nec4.out"
        proc = _run(bundle, NEC4_TWIN_NAME, [str(NEC42_DECK), str(out)], room)
        assert proc.returncode == 0, proc.stderr
        assert _unclocked(out.read_bytes()) == expected
        assert len(_listening(room)) == 1, _listening(room)
        addresses = sorted(p.name for p in room.glob(f"*{suffix}"))
        assert addresses == [f"{expected_key}{suffix}"], addresses
    finally:
        _stop(room)
        shutil.rmtree(bundle, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.slow
def test_a_nec4_named_copy_one_shots_in_the_nec42_slot(client_exe, tmp_path):
    """Rung 3 carries the slot as well as the basis: the engine's one-shot is
    launched with ``--dialect nec4``, or it would read the deck as NEC-5."""
    bundle = _bundle(client_exe, _REFUSING_SHIM, names=(NEC4_TWIN_NAME,))
    room = _room()
    expected = _nec4_oracle(tmp_path)
    try:
        out = tmp_path / "nec4-rung3.out"
        proc = _run(bundle, NEC4_TWIN_NAME, [str(NEC42_DECK), str(out)], room)
        assert proc.returncode == 0, proc.stderr
        assert _unclocked(out.read_bytes()) == expected
        assert _listening(room) == []
    finally:
        _stop(room)
        shutil.rmtree(bundle, ignore_errors=True)


# The C copy of the name rule, held to the stdlib one (momwire#1443): an
# engine that records the argv the launcher derived from ITS OWN name, then
# refuses both rungs, so nothing solves and the record is the whole result.
_ARGV_SHIM = (
    "#!/bin/sh\n"
    'for arg in "$@"; do printf "%s\\n" "$arg" >> "$MW_ARGV_LOG"; done\n'
    'printf "%s\\n" "--END--" >> "$MW_ARGV_LOG"\n'
    "exit 1\n"
)


def _flag_value(args: list[str], flag: str) -> str | None:
    return args[args.index(flag) + 1] if flag in args else None


@pytest.mark.integration
@pytest.mark.parametrize(
    "name",
    [
        "momwire-nec42",
        "momwire-nec42-sinusoidal",
        "Momwire-NEC42-Bspline-D1.EXE",
        "momwire-nec42-",
        "momwire-nec42x",
        # A renamed copy carrying `nec4` but not `nec42` reads past `nec4`.
        "momwire-nec4",
        "momwire-nec4-sinusoidal",
        "momwire-nec4-",
        "momwire-nec43",
        "momwire-nec5-razor2p",
    ],
)
def test_the_c_name_rule_is_the_stdlib_one(client_exe, tmp_path, monkeypatch, name):
    is_nec4, basis = mech.filename_nec4(name)
    bundle = _bundle(client_exe, _ARGV_SHIM, names=(name,))
    room = _room()
    log = tmp_path / "argv.log"
    monkeypatch.setenv("MW_ARGV_LOG", str(log))
    try:
        out = tmp_path / "argv.out"
        proc = _run(bundle, name, [str(NEC42_DECK), str(out)], room)
        assert proc.returncode == 0, proc.stderr
        calls = [
            c.splitlines() for c in log.read_text().split("--END--\n") if c.strip()
        ]
        # Rung 2's --serve spawn and rung 3's one-shot both carry the name's
        # reading of itself.
        assert len(calls) == 2, calls
        for args in calls:
            assert (_flag_value(args, "--dialect") == "nec4") == is_nec4, args
            if is_nec4:
                assert _flag_value(args, "--basis") == basis, args
    finally:
        _stop(room)
        shutil.rmtree(bundle, ignore_errors=True)


# --------------------------------------------------------------------------
# the 4nec2 front end (momwire#1488)
# --------------------------------------------------------------------------

EXAMPLE3 = Path(__file__).resolve().parent / "fixtures" / "4nec2_example3"
# 4nec2's two names, as POSIX spells the same relative paths.
_NAMES_4NEC2 = b"../out/EXAMPLE3.inp\r\n../out/EXAMPLE3.out\r\n"


def _4nec2_room(tmp_path: Path) -> tuple[Path, Path]:
    """The engine runs in ``bin``; 4nec2's files sit in ``out`` beside it."""
    engine_dir = tmp_path / "bin"
    out_dir = tmp_path / "out"
    engine_dir.mkdir()
    out_dir.mkdir()
    (out_dir / "EXAMPLE3.inp").write_bytes(_4nec2_deck())
    return engine_dir, out_dir


def _4nec2_deck() -> bytes:
    """Example 3 over no ground: Dan's geometry (so Ward's ``.GMO``) without
    the Sommerfeld fill, which costs ~13 s a solve. The smoke and the slow
    test in test_eznec_4nec2_1488.py run the deck as sent."""
    raw = (EXAMPLE3 / "EXAMPLE3.inp").read_bytes()
    return raw.replace(b"GN 2 0 0 0 14 .006", b"GN -1")


def _4nec2_oracle(tmp_path: Path, basis: str | None) -> bytes:
    deck = tmp_path / "oracle-EXAMPLE3.inp"
    deck.write_bytes(_4nec2_deck())
    out = tmp_path / f"oracle-4nec2-{basis or 'default'}.out"
    kwargs = {} if basis is None else {"basis": basis}
    _shell.run(deck, out, en_optional=True, **kwargs)
    return out.read_bytes()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(
    ("name", "basis"), [(NEC5_NAME, None), (NEC5_TWIN_NAME, "razor2p")]
)
def test_the_c_client_reads_4nec2s_names_from_stdin(client_exe, tmp_path, name, basis):
    """``ENGINE < nec2d.tmp`` against a real daemon: the names arrive on
    stdin with CRLF, relative to the working directory; the deck has no EN;
    the ``.GMO`` comes back framed and lands beside the printout as Ward's
    sample, and the printout is the one-shot's to the byte."""
    bundle = _bundle(client_exe, _ENGINE_SHIM, names=(name,))
    room = _room()
    engine_dir, out_dir = _4nec2_room(tmp_path)
    expected = _4nec2_oracle(tmp_path, basis)
    try:
        proc = _run(bundle, name, [], room, stdin=_NAMES_4NEC2, cwd=engine_dir)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == ""
        assert (out_dir / "EXAMPLE3.out").read_bytes() == expected
        assert (out_dir / "EXAMPLE3.GMO").read_bytes() == (
            EXAMPLE3 / "EXAMPLE3.GMO"
        ).read_bytes()
        assert len(_listening(room)) == 1, _listening(room)
        assert list(engine_dir.iterdir()) == []
    finally:
        _stop(room)
        shutil.rmtree(bundle, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.slow
def test_rung_three_carries_the_stdin_allowance_and_writes_the_gmo(
    client_exe, tmp_path
):
    """No daemon can start, so the one-shot engine answers: the paths reach
    it on its command line, and ``--en-optional`` carries what came with
    them. The engine itself writes the ``.GMO``."""
    bundle = _bundle(client_exe, _REFUSING_SHIM, names=(NEC5_NAME,))
    room = _room()
    engine_dir, out_dir = _4nec2_room(tmp_path)
    expected = _4nec2_oracle(tmp_path, None)
    try:
        proc = _run(bundle, NEC5_NAME, [], room, stdin=_NAMES_4NEC2, cwd=engine_dir)
        assert proc.returncode == 0, proc.stderr
        assert (out_dir / "EXAMPLE3.out").read_bytes() == expected
        assert (out_dir / "EXAMPLE3.GMO").read_bytes() == (
            EXAMPLE3 / "EXAMPLE3.GMO"
        ).read_bytes()
        assert _listening(room) == []
    finally:
        _stop(room)
        shutil.rmtree(bundle, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("stdin", "fault"),
    [
        (b"", "UNABLE TO OPEN FILE"),
        (b"\r\n", "UNABLE TO OPEN FILE"),
        (b"deck.inp\r\n", "UNABLE TO OPEN SECOND FILE"),
    ],
)
def test_the_c_clients_stdin_form_faults_like_its_command_line(
    client_exe, tmp_path, stdin, fault
):
    bundle = _bundle(client_exe, _ENGINE_SHIM, names=(NEC5_NAME,))
    room = _room()
    try:
        proc = _run(bundle, NEC5_NAME, [], room, stdin=stdin, cwd=tmp_path)
        assert proc.returncode == 0
        assert proc.stdout.splitlines() == [fault]
        assert list(tmp_path.iterdir()) == []
        assert _listening(room) == []
    finally:
        _stop(room)
        shutil.rmtree(bundle, ignore_errors=True)


@pytest.mark.integration
def test_a_nec4_named_copy_reads_4nec2s_names_from_stdin(client_exe, tmp_path):
    """4nec2's NEC-4.2 engine form (momwire#1592): the names on stdin, a deck
    with no EN, against a real daemon of the NEC-4.2 slot. No ``.GMO``."""
    bundle = _bundle(client_exe, _ENGINE_SHIM, names=(NEC4_TWIN_NAME,))
    room = _room()
    engine_dir = tmp_path / "bin"
    out_dir = tmp_path / "out"
    engine_dir.mkdir()
    out_dir.mkdir()
    raw = NEC42_DECK.read_bytes()
    (out_dir / "EXAMPLE3.inp").write_bytes(raw[: raw.rindex(b"EN")])
    try:
        proc = _run(
            bundle, NEC4_TWIN_NAME, [], room, stdin=_NAMES_4NEC2, cwd=engine_dir
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == ""
        printout = (out_dir / "EXAMPLE3.out").read_bytes()
        assert _unclocked(printout) == _nec4_oracle(tmp_path)
        assert sorted(p.name for p in out_dir.iterdir()) == [
            "EXAMPLE3.inp",
            "EXAMPLE3.out",
        ]
        assert len(_listening(room)) == 1, _listening(room)
    finally:
        _stop(room)
        shutil.rmtree(bundle, ignore_errors=True)


# A stand-in daemon that answers every deck with one canned framed answer, so
# the C copy of the stem rule can be held to the stdlib one without a solve.
_CANNED_GMO = b"     0\r\n     0\r\n"
_CANNED_PRINTOUT = b" canned printout\r\n"
_FRAMING_DAEMON = """
import sys
import momwire_serve_client as mech
from momwire.serve._server import serve_forever, take_value

argv = [a for a in sys.argv[1:] if a != "--serve"]
path, argv = take_value(argv, "--socket")
log_path, argv = take_value(argv, "--log")
answer = mech.GMO_FRAME_START + {gmo!r} + mech.GMO_FRAME_END + {printout!r}

def connection(conn, number, log, solve_lock, pool=None):
    reader = conn.makefile("rb")
    mech.read_deck_frame(reader)
    reader.close()
    conn.sendall(answer)

sys.exit(serve_forever(path, "60", log_path, connection))
"""


@pytest.mark.integration
@pytest.mark.parametrize(
    "printout",
    [
        "EX3.out",
        "EX3.OUT",
        "a.b.c",
        "noext",
        "dir.d/noext",
        ".out",
        "..out",
        "x/...a.b",
    ],
)
def test_the_c_gmo_path_is_the_stdlib_one(client_exe, tmp_path, printout):
    daemon = tmp_path / "framing_daemon.py"
    daemon.write_text(
        _FRAMING_DAEMON.format(gmo=_CANNED_GMO, printout=_CANNED_PRINTOUT)
    )
    shim = f'#!/bin/sh\nexec "{sys.executable}" "{daemon}" "$@"\n'
    bundle = _bundle(client_exe, shim, names=(NEC5_NAME,))
    room = _room()
    work = tmp_path / "work"
    (work / "x").mkdir(parents=True)
    (work / "dir.d").mkdir()
    (work / "deck.nec").write_bytes(b"CM\r\nCE\r\nEN\r\n")
    try:
        proc = _run(bundle, NEC5_NAME, ["deck.nec", printout], room, cwd=work)
        assert proc.returncode == 0, proc.stderr
        assert len(_listening(room)) == 1, _listening(room)
        assert (work / printout).read_bytes() == _CANNED_PRINTOUT
        assert (work / mech.gmo_path(printout)).read_bytes() == _CANNED_GMO
    finally:
        _stop(room)
        shutil.rmtree(bundle, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.slow
def test_the_c_client_reads_quoted_names_with_spaces(client_exe, tmp_path):
    """Ward's port strips surrounding quotes from a ``.tmp`` line; before
    momwire#1488's fix a quoted name made this launcher write nothing."""
    bundle = _bundle(client_exe, _ENGINE_SHIM, names=(NEC5_NAME,))
    room = _room()
    engine_dir = tmp_path / "bin"
    out_dir = tmp_path / "my out"
    engine_dir.mkdir()
    out_dir.mkdir()
    (out_dir / "EX 3.inp").write_bytes(_4nec2_deck())
    names = b'"../my out/EX 3.inp"\r\n  "../my out/EX 3.out" \r\n'
    try:
        proc = _run(bundle, NEC5_NAME, [], room, stdin=names, cwd=engine_dir)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == ""
        assert b"NEC ERROR" not in (out_dir / "EX 3.out").read_bytes()
        assert (out_dir / "EX 3.GMO").read_bytes() == (
            EXAMPLE3 / "EXAMPLE3.GMO"
        ).read_bytes()
    finally:
        _stop(room)
        shutil.rmtree(bundle, ignore_errors=True)


@pytest.mark.integration
@pytest.mark.slow
def test_an_unwritable_printout_is_named_on_stdout_never_silent(client_exe, tmp_path):
    """A printout path that cannot be opened (a stray quote names a folder
    that is not there): every rung fails to write, and the failure reaches
    stdout -- from the one-shot engine, whose stdout this launcher shares --
    rather than an exit 0 that said nothing anywhere."""
    bundle = _bundle(client_exe, _ENGINE_SHIM, names=(NEC5_NAME,))
    room = _room()
    (tmp_path / "deck.inp").write_bytes(_4nec2_deck())
    names = b'deck.inp\r\n"no such dir/deck.out\r\n'
    try:
        proc = _run(bundle, NEC5_NAME, [], room, stdin=names, cwd=tmp_path)
        assert proc.returncode == 0
        assert '"no such dir/deck.out' in proc.stdout
        assert "UNABLE TO OPEN FILE" in proc.stdout
    finally:
        _stop(room)
        shutil.rmtree(bundle, ignore_errors=True)


@pytest.mark.integration
def test_rung_four_names_an_unwritable_printout_on_stdout(client_exe, tmp_path):
    bundle = _bundle(client_exe, None, names=(NEC5_NAME,))
    room = _room()
    bad = str(tmp_path / "no such dir" / "x.out")
    try:
        proc = _run(bundle, NEC5_NAME, [str(_deck(BASIS_DECK)), bad], room)
        assert proc.returncode == 0
        assert proc.stdout.splitlines()[0] == f"UNABLE TO OPEN FILE {bad}"
        assert "MOMWIRE ENGINE UNAVAILABLE" in proc.stdout
    finally:
        _stop(room)
        shutil.rmtree(bundle, ignore_errors=True)


STREAM_DECK = REPO / "scripts" / "core_split" / "nx_decks" / "simnec_stream_2281566.nec"


@pytest.mark.integration
@pytest.mark.parametrize("name", [NEC5_NAME, CLIENT_NAME])
def test_the_streaming_call_streams_through_the_engine(client_exe, tmp_path, name):
    """SimNEC's ``- -`` (momwire#1395, #1512): decks on stdin, printouts on
    stdout, the engine's own answer byte for byte. 0.79.1's launcher read the
    two dashes as file names, wrote a printout to a file named ``-`` and
    exited 0 with stdout empty (momwire 0.79.2)."""
    bundle = _bundle(client_exe, _ENGINE_SHIM, names=(name,))
    room = _room()
    deck = STREAM_DECK.read_bytes()
    try:
        oracle = subprocess.run(
            [sys.executable, str(ENTRY), "-", "-"],
            input=deck,
            capture_output=True,
            timeout=300,
        )
        proc = _run(bundle, name, ["-", "-"], room, stdin=deck, cwd=tmp_path)
        assert proc.returncode == oracle.returncode, proc.stderr
        assert proc.stdout == oracle.stdout.decode(errors="replace")
        assert "INPUT LINE" in proc.stdout and len(proc.stdout) > 10_000
        assert not (tmp_path / "-").exists()
        assert _listening(room) == []
    finally:
        _stop(room)
        shutil.rmtree(bundle, ignore_errors=True)


@pytest.mark.integration
def test_the_streaming_call_without_an_engine_says_so_and_fails(client_exe, tmp_path):
    """No engine: named on stdout (the stream's channel) with a non-zero exit,
    never a silent exit 0."""
    bundle = _bundle(client_exe, None, names=(NEC5_NAME,))
    room = _room()
    try:
        proc = _run(bundle, NEC5_NAME, ["-", "-"], room, stdin=b"", cwd=tmp_path)
        assert proc.returncode != 0
        assert "MOMWIRE ENGINE UNAVAILABLE" in proc.stdout
        assert not (tmp_path / "-").exists()
    finally:
        _stop(room)
        shutil.rmtree(bundle, ignore_errors=True)


# --------------------------------------------------------------------------
# the environment a server inherits (momwire#1979)
# --------------------------------------------------------------------------


def _c_string(name: str) -> str:
    match = re.search(rf'#define {name} "([^"]*)"', SOURCE.read_text())
    assert match is not None, name
    return match[1]


def _c_list(name: str) -> list[str]:
    match = re.search(
        rf"static const char \*const {name}\[\] = \{{(.*?)\n\}};",
        SOURCE.read_text(),
        re.S,
    )
    assert match is not None, name
    return re.findall(r'"([^"]*)"', match[1])


def test_the_c_identity_tables_are_the_python_ones():
    """The C client cannot import `momwire_serve_client`, so it carries
    copies of the tables the identity is built from; this holds each copy
    equal to its original, which tests/test_features_registry.py in turn
    holds equal to the registry. A feature promoted to default-on, a new
    alias or a new thread variable then fails HERE, naming the table, rather
    than silently letting the Windows drop-in share a server across it."""
    assert _c_string("JUNCTION_RULE_ENV") == mech.JUNCTION_RULE_ENV
    assert _c_string("FEATURE_ENV_PREFIX") == mech.FEATURE_ENV_PREFIX
    table = re.search(
        r"FEATURE_ALIASES\[\]\[2\] = \{(.*?)\n\};", SOURCE.read_text(), re.S
    )
    assert table is not None
    pairs = re.findall(r'\{"([^"]*)", "([^"]*)"\}', table[1])
    assert dict(pairs) == mech.FEATURE_ALIASES
    assert len(pairs) == len(mech.FEATURE_ALIASES)
    default_on = _c_list("FEATURE_DEFAULT_ON")
    assert set(default_on) == mech.FEATURE_DEFAULT_ON
    assert len(default_on) == len(mech.FEATURE_DEFAULT_ON)
    parts = _c_list("FEATURE_PARTS")
    assert set(parts) == mech.FEATURE_PARTS and len(parts) == len(mech.FEATURE_PARTS)
    on = _c_list("FEATURE_ON")
    assert set(on) == mech._FEATURE_ON and len(on) == len(mech._FEATURE_ON)
    # The C compare lowercases the value only, so the table must be lower.
    assert all(v == v.lower() for v in on)
    # Order is part of the key: threads are appended in THREAD_ENVS order.
    assert tuple(_c_list("THREAD_ENVS")) == mech.THREAD_ENVS


def _identity_cases() -> list:
    feature = "MOMWIRE_X_CROSSING_TABLE"
    cases = [
        ("nothing", {}),
        ("junction", {mech.JUNCTION_RULE_ENV: "kcl"}),
        # Python hashes the junction value unstripped; so must C.
        ("junction-spaced", {mech.JUNCTION_RULE_ENV: " kcl "}),
        ("junction-empty", {mech.JUNCTION_RULE_ENV: ""}),
        ("feature-1", {feature: "1"}),
        ("feature-odd-case", {feature: " On "}),
        ("feature-TRUE", {feature: "TRUE"}),
        ("feature-yes-tab", {feature: "yes\t"}),
        ("feature-unit-sep", {feature: "\x1f1\x1c"}),
        ("feature-0", {feature: "0"}),
        ("feature-empty", {feature: ""}),
        ("feature-maybe", {feature: "maybe"}),
        ("feature-name-case", {"MOMWIRE_X_Crossing_Table": "1"}),
        ("feature-unregistered", {"MOMWIRE_X_NOT_YET_REGISTERED": "on"}),
        ("feature-bare-prefix", {mech.FEATURE_ENV_PREFIX: "1"}),
        (
            "features-sorted",
            {"MOMWIRE_X_ZZ_LAST": "1", feature: "1", "MOMWIRE_X_AA_FIRST": "1"},
        ),
        ("threads-empty", {"OMP_NUM_THREADS": ""}),
        ("threads-blank", {"OMP_NUM_THREADS": "  "}),
        ("threads-spaced", {"OMP_NUM_THREADS": " 8 "}),
        ("threads-all", {v: str(i + 1) for i, v in enumerate(mech.THREAD_ENVS)}),
        (
            "everything",
            {
                mech.JUNCTION_RULE_ENV: "kcl",
                feature: "on",
                "MOMWIRE_X_FACTORED_COLUMNS": "0",
                "MOMWIRE_RAZOR_CXX": "1",
                "MOMWIRE_X_RAZOR_CXX": "yes",
                "OPENBLAS_NUM_THREADS": "2",
                "MOMWIRE_THREADS": "4",
            },
        ),
    ]
    # Every alias, every default-on feature and every part, so a new one is exercised by
    # construction rather than by remembering to add a row.
    for alias, name in sorted(mech.FEATURE_ALIASES.items()):
        cases.append((f"alias-{name}", {alias: "1"}))
        cases.append((f"alias-and-own-{name}", {alias: "1", f"MOMWIRE_X_{name}": "1"}))
    for name in sorted(mech.FEATURE_DEFAULT_ON):
        var = f"MOMWIRE_X_{name.upper()}"
        for value in ("0", "off", "false", "1", " ON ", "", " "):
            cases.append((f"default-on-{name}-{value!r}", {var: value}))
    for name in sorted(mech.FEATURE_PARTS):
        var = f"MOMWIRE_X_{name.upper()}"
        for value in ("0", "off", "1", "", " "):
            cases.append(
                (f"part-{name}-{value!r}", {"MOMWIRE_X_PAIR_REUSE": "1", var: value})
            )
    for var in mech.THREAD_ENVS:
        cases.append((f"threads-{var}", {var: "3"}))
    return [pytest.param(env, id=case) for case, env in cases]


def _key_elements(bundle: Path, name: str, identity: dict[str, str]) -> list[str]:
    """The C client's key elements, from its test-only probe."""
    room = _room()
    try:
        env = {
            k: v
            for k, v in os.environ.items()
            if not _is_identity_var(k) and k != "MOMWIRE_PORTAL_RUNTIME_DIR"
        }
        env.update(identity)
        env["MOMWIRE_PORTAL_RUNTIME_DIR"] = str(room)
        proc = subprocess.run(
            [str(bundle / name), "-key-elements"],
            capture_output=True,
            timeout=60,
            env=env,
            stdin=subprocess.DEVNULL,
        )
        assert proc.returncode == 0, proc.stderr
        # The probe spawns nothing and makes no address file.
        assert list(room.iterdir()) == []
    finally:
        shutil.rmtree(room, ignore_errors=True)
    raw = proc.stdout.decode("utf-8")
    assert raw.endswith("\0"), raw
    return raw[:-1].split("\0")


@pytest.fixture(scope="module")
def key_bundle(client_exe):
    bundle = _bundle(client_exe, _ENGINE_SHIM, names=(CLIENT_NAME, NEC4_TWIN_NAME))
    yield bundle
    shutil.rmtree(bundle, ignore_errors=True)


def _fixed_elements(bundle: Path, name: str) -> list[str]:
    major, minor = mech.dist_version()
    head = [
        f"eznec.{major}.{minor}",
        os.path.realpath(str(bundle / ENGINE_NAME)),
        repr(900.0),
        mech.WIRE,
        mech.ADVISORIES_WIRE,
    ]
    if name == NEC4_TWIN_NAME:
        return [*head, "dialect=nec4", "basis=sinusoidal"]
    return [*head, "default"]


@pytest.mark.integration
@pytest.mark.parametrize("name", [CLIENT_NAME, NEC4_TWIN_NAME])
@pytest.mark.parametrize("identity", _identity_cases())
def test_the_c_key_appends_the_pythons_environment_elements(
    key_bundle, monkeypatch, name, identity
):
    """momwire#1979: the C client appends to its key exactly the elements
    the Python client appends -- junction rule, features, threads, in that
    order and spelling -- for the same environment. The digests differ by
    design (different engines); the elements must not."""
    for var in [v for v in os.environ if _is_identity_var(v)]:
        monkeypatch.delenv(var)
    for var, value in identity.items():
        monkeypatch.setenv(var, value)
    fixed = _fixed_elements(key_bundle, name)
    elements = _key_elements(key_bundle, name, identity)
    assert elements[: len(fixed)] == fixed
    assert elements[len(fixed) :] == _python_env_elements()


@pytest.mark.integration
@pytest.mark.parametrize("name", [CLIENT_NAME, NEC4_TWIN_NAME])
def test_with_nothing_set_the_c_key_is_the_key_it_was(key_bundle, name):
    """No identity variable set: exactly the pre-#1979 elements, so every
    warm server a bundle already has keeps its name across the upgrade."""
    assert _key_elements(key_bundle, name, {}) == _fixed_elements(key_bundle, name)


@pytest.mark.integration
@pytest.mark.slow
def test_a_warm_server_never_answers_a_call_under_another_feature(client_exe, tmp_path):
    """The fault momwire#1979 names, end to end: a server spawned with no
    feature set must not answer a call that sets one -- that call spawns its
    own, at its own address -- and each server IS reused by a call identical
    to the one that spawned it."""
    bundle = _bundle(client_exe, _ENGINE_SHIM)
    room = _room()
    deck = _deck(BASIS_DECK)
    plain: dict[str, str] = {}
    featured = {"MOMWIRE_X_CROSSING_TABLE": "1"}
    threaded = {"OMP_NUM_THREADS": "1"}
    suffix = mech.address_suffix(mech.TCP if os.name == "nt" else mech.UNIX)

    def call(index: int, identity: dict[str, str]) -> None:
        out = tmp_path / f"identity-{index}.out"
        proc = _run(bundle, CLIENT_NAME, [str(deck), str(out)], room, identity=identity)
        assert proc.returncode == 0, proc.stderr
        assert b"ANTENNA INPUT PARAMETERS" in out.read_bytes()

    try:
        call(0, plain)
        assert len(_listening(room)) == 1, _listening(room)
        call(1, plain)
        assert len(_listening(room)) == 1, _listening(room)
        call(2, featured)
        assert len(_listening(room)) == 2, _listening(room)
        call(3, featured)
        assert len(_listening(room)) == 2, _listening(room)
        call(4, threaded)
        assert len(_listening(room)) == 3, _listening(room)
        assert len(sorted(room.glob(f"*{suffix}"))) == 3
    finally:
        _stop(room)
        shutil.rmtree(bundle, ignore_errors=True)


# --------------------------------------------------------------------------
# advisories on stderr (momwire#1673 reversed)
# --------------------------------------------------------------------------


def _c_bytes(name: str) -> bytes:
    """A ``static const char NAME[] = "...";`` constant's bytes."""
    match = re.search(
        rf'static const char {name}\[\] = "((?:[^"\\]|\\.)*)";', SOURCE.read_text()
    )
    assert match is not None, name
    return match[1].encode("latin-1").decode("unicode_escape").encode("latin-1")


def _c_concat(name: str) -> bytes:
    """A ``static const char NAME[] =`` constant spelt as adjacent literals."""
    literal = r'"(?:[^"\\]|\\.)*"'
    match = re.search(
        rf"static const char {name}\[\] =\s*((?:{literal}\s*)+);", SOURCE.read_text()
    )
    assert match is not None, name
    raw = "".join(part[1:-1] for part in re.findall(literal, match[1]))
    return raw.encode("latin-1").decode("unicode_escape").encode("latin-1")


def test_the_c_advisories_constants_are_the_python_ones():
    assert _c_string("ADVISORIES_ENV") == mech.ADVISORIES_ENV
    assert _c_string("ADVISORIES_KEY_ELEMENT") == mech.ADVISORIES_WIRE
    assert _c_bytes("ADVISORIES_REQUEST") == mech.ADVISORIES_REQUEST
    assert _c_bytes("ADVISORIES_FRAME_START") == mech.ADVISORIES_FRAME_START
    assert _c_bytes("ADVISORIES_FRAME_END") == mech.ADVISORIES_FRAME_END
    assert _c_string("ADVISORIES_DIVIDER_HEAD") == mech.ADVISORIES_DIVIDER_HEAD
    room = re.search(r"#define ADVISORIES_LABEL_ROOM (\d+)\n", SOURCE.read_text())
    assert room is not None and int(room[1]) == mech.ADVISORIES_LABEL_ROOM
    capped = mech.ADVISORIES_CAPPED.replace("\n", "\r\n").encode()
    assert _c_concat("ADVISORIES_CAPPED") == capped  # as the frame carries it
    # The mode tables (momwire#2119): each writing mode's words, and off is
    # everything else on both sides.
    for mode, table in (
        ("deck", "ADVISORIES_DECK"),
        ("all", "ADVISORIES_ALL"),
    ):
        words = _c_list(table)
        assert set(words) == mech._ADVISORIES_WORDS[mode], mode
        assert len(words) == len(set(words)) and all(w == w.lower() for w in words)
    assert set(mech._ADVISORIES_WORDS) == set(mech.ADVISORIES_MODES) - {"off"}


_DUP_DECK = DECKS.parent.parent / "razor_printout_small" / "dup_reversed_rp_ne_pq.nec"

# A stand-in daemon that records each request's bytes and answers one canned
# frame (when asked) and printout: the C launcher and the Python one, given
# the same answer, must send the same bytes and write the same stderr.
_RECORDING_DAEMON = """
import sys
import momwire_serve_client as mech
from momwire.serve._server import serve_forever, take_value

argv = [a for a in sys.argv[1:] if a != "--serve"]
path, argv = take_value(argv, "--socket")
log_path, argv = take_value(argv, "--log")
record = {record!r}
frame = {frame!r}

def connection(conn, number, log, solve_lock, pool=None):
    reader = conn.makefile("rb")
    body = mech.read_deck_frame(reader)
    reader.close()
    with open(record, "ab") as handle:
        handle.write(len(body).to_bytes(8, "big") + body)
    answer = b" canned printout\\r\\n"
    if body.startswith(mech.ADVISORIES_REQUEST):
        answer = mech.ADVISORIES_FRAME_START + frame + mech.ADVISORIES_FRAME_END + answer
    conn.sendall(answer)

sys.exit(serve_forever(path, "60", log_path, connection))
"""

_CANNED_ADVISORY = b"x.py:1: DuplicateWire: canned\r\n  line\r\n"


def _recorded(path: Path) -> list[bytes]:
    data = path.read_bytes() if path.exists() else b""
    out = []
    while data:
        n = int.from_bytes(data[:8], "big")
        out.append(data[8 : 8 + n])
        data = data[8 + n :]
    return out


_SETTINGS = (None, "", "off", "nonsense", "stream", "deck", " DECK ", "stderr")
_SETTINGS += ("on", "1", "yes", "true", "all")


@pytest.mark.integration
@pytest.mark.parametrize(
    "frame",
    [
        _CANNED_ADVISORY,
        b"",
        mech.ADVISORIES_CAPPED.replace("\n", "\r\n").encode(),
    ],
    ids=["advisory", "empty", "note-alone"],
)
def test_the_c_and_python_launchers_send_and_write_the_same_bytes(
    client_exe, tmp_path, monkeypatch, capsys, frame
):
    """For every mode spelling, against the same answer: the request bytes
    the C launcher sends are the Python one's, and so is its stderr -- the
    divider naming the deck only when the frame holds an advisory, the
    deck's label cut the same way when it is long."""
    import momwire_eznec_client as eznec_client

    record = tmp_path / "c-requests.bin"
    daemon = tmp_path / "recording_daemon.py"
    daemon.write_text(_RECORDING_DAEMON.format(record=str(record), frame=frame))
    shim = f'#!/bin/sh\nexec "{sys.executable}" "{daemon}" "$@"\n'
    bundle = _bundle(client_exe, shim, names=(NEC5_NAME,))
    room = _room()
    deep = tmp_path / ("d" * 120) / ("e" * 120)
    deep.mkdir(parents=True)
    decks = [tmp_path / "d.nec", deep / "d.nec"]
    for deck in decks:
        deck.write_bytes(_DUP_DECK.read_bytes())
    py_sent: list[bytes] = []

    def served(deck_bytes, _basis, _idle, _dialect="nec5"):
        py_sent.append(deck_bytes)
        answer = b" canned printout\r\n"
        if deck_bytes.startswith(mech.ADVISORIES_REQUEST):
            return (
                mech.ADVISORIES_FRAME_START + frame + mech.ADVISORIES_FRAME_END + answer
            )
        return answer

    monkeypatch.setattr(eznec_client, "_served_bytes", served)
    monkeypatch.delenv("PYTHONWARNINGS", raising=False)
    try:
        for setting in _SETTINGS:
            for deck in decks:
                if setting is None:
                    monkeypatch.delenv("MOMWIRE_ADVISORIES", raising=False)
                else:
                    monkeypatch.setenv("MOMWIRE_ADVISORIES", setting)
                out = tmp_path / "c.out"
                proc = _run(bundle, NEC5_NAME, [str(deck), str(out)], room)
                assert proc.returncode == 0, proc.stderr
                assert out.read_bytes() == b" canned printout\r\n"
                capsys.readouterr()
                py_out = tmp_path / "py.out"
                monkeypatch.setattr("sys.argv", [NEC5_NAME, str(deck), str(py_out)])
                assert eznec_client.main(None) == 0
                py_err = capsys.readouterr().err
                assert proc.stderr == py_err, (setting, deck, proc.stderr, py_err)
                assert _recorded(record)[-1] == py_sent[-1], setting
                mode = mech.advisories_mode({"MOMWIRE_ADVISORIES": setting or ""})
                if mode == "off" or not frame:
                    assert py_err == ""
                elif frame == _CANNED_ADVISORY:
                    assert py_err.startswith(mech.advisories_divider(str(deck)))
                else:
                    assert py_err == mech.ADVISORIES_CAPPED
        assert len(_listening(room)) == 1
    finally:
        _stop(room)
        shutil.rmtree(bundle, ignore_errors=True)


@pytest.mark.integration
def test_a_served_deck_carries_its_advisories_to_the_c_clients_stderr(
    client_exe, tmp_path, monkeypatch
):
    """The Windows drop-in's launcher against the real warm server: a deck
    the server answers raised its advisories in the server's process, and
    they come back framed to THIS process's stderr under the divider naming
    the deck, on every launch in ``deck`` and ``all`` mode, and never by
    default. The printout is the one-shot's every time."""
    bundle = _bundle(client_exe, _ENGINE_SHIM, names=(NEC5_TWIN_NAME,))
    room = _room()
    expected = _oracle(_DUP_DECK, tmp_path, basis="razor2p")
    plan = (None, "deck", "deck", "stream", "off", "all")
    shown = []
    try:
        for index, setting in enumerate(plan):
            out = tmp_path / f"advised-{index}.out"
            # `_run` passes this process's environment on
            monkeypatch.delenv("PYTHONWARNINGS", raising=False)
            if setting is None:
                monkeypatch.delenv("MOMWIRE_ADVISORIES", raising=False)
            else:
                monkeypatch.setenv("MOMWIRE_ADVISORIES", setting)
            proc = _run(bundle, NEC5_TWIN_NAME, [str(_DUP_DECK), str(out)], room)
            assert proc.returncode == 0
            assert len(_listening(room)) == 1, "the deck was not served warm"
            assert out.read_bytes() == expected
            shown.append(proc.stderr.count("DuplicateWire"))
            if setting in (None, "off", "stream"):
                assert proc.stderr == "", (setting, proc.stderr[:300])
            if proc.stderr:
                head = mech.advisories_divider(str(_DUP_DECK))
                assert proc.stderr.startswith(head), proc.stderr[:300]
                assert proc.stderr.count(mech.ADVISORIES_DIVIDER_HEAD) == 1
    finally:
        _stop(room)
        shutil.rmtree(bundle, ignore_errors=True)
    assert shown == [0, 1, 1, 0, 0, 1]
