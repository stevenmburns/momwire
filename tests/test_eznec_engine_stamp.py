"""Line 2 of every printout says which engine answered it.

In the captured printouts line 2 is the licensed engine's own build tag, which
makes it the one line in the header where an engine identity already lives.
This engine writes its own there — ``momwire <version> <basis> <variant>`` —
and nothing else in the printout moves: the departure is one line wide, and
the gate below measures exactly that.

Why those three fields and not fewer: three things move a printed number, and
a tester mailing back a ``NEC5.OUT`` cannot be asked to remember any of them.
The RELEASE is the version.  The BASIS is which formulation answered, which on
a deployed bundle is chosen by the launcher's filename (momwire#593) and is
therefore invisible in the file otherwise.  The VARIANT is which compiled
accelerator loaded, and ``none`` there is the whole of momwire#737 read off a
printout instead of off a daemon log.

Two things this file is careful NOT to claim:

* **The stamp is not evidence about the solver.**  It is threaded from the
  same string the solve is, so a route that resolved the wrong basis stamps
  the wrong basis too and reads perfectly consistent.  What catches that is
  the byte comparison against a module run in the named basis
  (``scripts/eznec_freeze/smoke.py`` gate 4, ``tests/test_eznec_client_c.py``).
* **The byte gates elsewhere mask line 2, so they assert nothing about it.**
  That is what makes this file load-bearing rather than duplicative, and it
  is why :func:`test_the_stamp_gate_rejects_the_licensed_build_tag` exists:
  a gate that could not fail on the OLD line 2 would be measuring nothing.
"""

from __future__ import annotations

import importlib.metadata
import io
import socket
import threading

import pytest

import momwire
from momwire import _accel
from momwire.eznec import _printout, _resident, _serve, _shell
from momwire.eznec._printout import engine_stamp, render_header
from test_eznec_printout import (
    _ENGINE_STAMP_INDEX,
    FIXTURE_DIR,
    GATED_IDS,
    capture,
    deck_text,
)
from test_eznec_shell import header_prefix

# The closed set of variant labels, taken from the module that owns them plus
# the stamp's own word for "no extension loaded".  Read out of `_accel` rather
# than transcribed: a fourth build variant would otherwise pass this file
# while the stamp printed a label nothing here had ever seen.
VARIANT_LABELS = frozenset(
    {label for label, _ in (_accel._AVX2, _accel._SSE2, _accel._LEGACY)}
    | {_printout._STAMP_NO_VARIANT}
)

# The refusal every gate here uses, and it is a REAL one: 0022 asks for a near
# field over a `GN 0` earth, which this seam refuses by name (the standing
# refusal `scripts/eznec_freeze/smoke.py` uses too).  A refusal is the printout
# most likely to be the one a user writes in about, so it has to be stamped.
REFUSAL_ID = "0022"

# A deck the DIALECT front end refuses, which is a different path to the same
# frame (`_shell.render`'s gate 1 against its gate 2) and reaches
# `render_refusal` from a DeckError rather than a ServeRefusal.
NOT_NEC5_DECK = (
    "CM Stamp probe\nCM\nCE\n"
    "SP 0,1,0.,0.,0.,0.,0.,1.\n"
    "GW 1,10,0.,0.,0.,0.,0.,10.3,.02\n"
    "GE 1,-1\nFR 0,1,0,0,7.\nEX 4,1,-1,0,1.414214,0.\nEN\n"
)


def stamp_line(text: str) -> str:
    """Line 2 of ``text``, by position, CR tolerated.

    Positional and never searched for: a stamp that vanished and a stamp that
    landed on the wrong line are both regressions, and a search would call the
    second one a pass.  The CR strip is so a caller can hand over the bytes a
    printout was WRITTEN as (CRLF, `_shell.write_printout`) without first
    normalizing them — the endings are gated in ``test_eznec_shell.py``.
    """
    lines = text.split("\n")
    return lines[_ENGINE_STAMP_INDEX].rstrip("\r") if len(lines) > 1 else ""


def assert_stamped(text: str, *, basis: str) -> None:
    """The assertion the frozen bundle's smoke gate makes, in Python.

    Kept as a helper with one owner because four gates below and the smoke
    script have to agree on what a good stamp IS; the smoke script's copy is
    separate on purpose (it may not import momwire's own answer to a question
    it is gating), and ``scripts/eznec_freeze/smoke.py::_gate_stamp`` says so.
    """
    line = stamp_line(text)
    assert line.startswith(_printout._STAMP_PREFIX), line
    fields = line.split()
    assert len(fields) == 4, line
    _, version, stamped, variant = fields
    assert version != _printout._STAMP_UNKNOWN_VERSION, line
    assert stamped == basis, line
    assert variant in VARIANT_LABELS, line
    assert len(line) < 80, line


# --------------------------------------------------------------------------
# the stamp text itself
# --------------------------------------------------------------------------


def test_the_default_stamp_is_the_four_fields_in_order():
    line = engine_stamp(_serve.BASIS)
    assert line == (
        f" momwire {importlib.metadata.version('momwire')} "
        f"{_serve.BASIS} {momwire.accelerator_variant or 'none'}"
    )
    assert_stamped("1\n" + line + "\n", basis=_serve.BASIS)


def test_the_basis_field_is_the_name_that_was_threaded():
    """Verbatim, deprecated spellings included: what a bug report needs is
    which exe answered, and ``razor-nec5`` is a name a shipped launcher
    carries (``build.py``'s ``SHIPPED_VARIANTS``) even though it resolves to
    the same solver ``razor-2p`` does."""
    for basis in ("bspline", "razor-2p", "razor-nec5", "hmatrix"):
        assert engine_stamp(basis).split()[2] == basis


def test_a_missing_package_version_stamps_unknown_rather_than_raising(monkeypatch):
    """The metadata probe is the one field that can fail, and a printout is
    worth more than a stamp: a source tree with no ``.dist-info``, or a frozen
    bundle built without ``--copy-metadata``, still gets a printout."""

    def boom(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", boom)
    assert engine_stamp("bspline") == (
        f" momwire unknown bspline {momwire.accelerator_variant or 'none'}"
    )


def test_any_metadata_failure_is_caught_not_just_the_documented_one(monkeypatch):
    """A half-written ``dist-info`` raises from inside the reader rather than
    as ``PackageNotFoundError``, so the handler is deliberately blanket."""
    monkeypatch.setattr(
        importlib.metadata, "version", lambda name: (_ for _ in ()).throw(OSError("x"))
    )
    assert engine_stamp("bspline").split()[1] == "unknown"


@pytest.mark.parametrize("label", ["avx2", "sse2", "legacy"])
def test_the_variant_field_is_whatever_this_process_loaded(monkeypatch, label):
    monkeypatch.setattr(momwire, "accelerator_variant", label)
    assert engine_stamp("bspline").split()[3] == label


def test_no_extension_loaded_stamps_none(monkeypatch):
    """``momwire.accelerator_variant`` is ``None`` when nothing imported, and
    ``none`` is the printable spelling of it.  This is momwire#737's failure
    made visible in the file EZNEC shows the user: every solve in such a
    process is the pure-Python one, and the fallback warning goes to a stream
    EZNEC never displays."""
    monkeypatch.setattr(momwire, "accelerator_variant", None)
    assert engine_stamp("bspline").split()[3] == "none"


def test_an_unnamed_basis_still_stamps_four_fields():
    """The empty launcher suffix is reachable — ``momwire-eznec-.exe`` is a
    copy a user can make, and it travels verbatim so the deck refuses by name
    rather than being rounded off to the default (``entry.py::basis_for``).
    A blank field there would silently make the stamp a three-field line."""
    assert engine_stamp("") == engine_stamp(" \t ") != engine_stamp("bspline")
    assert engine_stamp("").split()[2] == "-"


def test_the_stamp_stays_under_eighty_columns_whatever_the_filename_said():
    """The basis is USER text: it is the launcher filename's suffix, so its
    length and its characters are not this engine's to assume.  A long name
    loses its tail; it never wraps one printout line into two."""
    for basis in ("x" * 200, "razor 2p", "a b c d e f " * 20):
        line = engine_stamp(basis)
        assert len(line) < 80, line
        assert len(line.split()) == 4, line
    assert engine_stamp("razor 2p").split()[2] == "razor_2p"


# --------------------------------------------------------------------------
# the departure is one line wide
# --------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("cid", GATED_IDS)
def test_the_stamp_is_the_only_header_line_that_left_the_captures(cid):
    """The in-test form of the diff proof, on all 75 printout-carrying
    captures: the rendered header and the captured one have the same line
    count and differ at exactly one index, and that index is the stamp's.

    This is what the masks in ``test_eznec_printout.py`` /
    ``test_eznec_serve.py`` / ``test_eznec_shell.py`` are allowed to assume,
    and it is asserted here rather than there so a second departure cannot
    hide behind a mask that was widened to accommodate it.
    """
    ours = render_header(deck_text(cid)).split("\n")
    theirs = header_prefix(cid).split("\n")
    assert len(ours) == len(theirs)
    differ = [i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b]
    assert differ == [_ENGINE_STAMP_INDEX]
    assert_stamped(
        header_prefix(cid).replace(theirs[1], ours[1], 1), basis=_serve.BASIS
    )


@pytest.mark.integration
@pytest.mark.parametrize("cid", GATED_IDS)
def test_the_stamp_gate_rejects_the_licensed_build_tag(cid):
    """The gate is not vacuous: the captured line 2 FAILS :func:`assert_stamped`.

    Taken off the fixture rather than typed in, so nothing here transcribes
    the licensed engine's build tag — and so this stays true if a future
    capture session ships a printout from a different build.
    """
    with pytest.raises(AssertionError):
        assert_stamped(header_prefix(cid), basis=_serve.BASIS)


# --------------------------------------------------------------------------
# every route stamps, and stamps the basis IT was given
# --------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("basis", [_serve.BASIS, "razor-2p"])
def test_the_one_shot_route_stamps_the_basis_it_was_launched_as(tmp_path, basis):
    """``_shell.main(..., basis=...)`` — the frozen exe's own path, where the
    basis came from the launcher's filename before argv was read."""
    out = tmp_path / "NEC5.OUT"
    assert (
        _shell.main([str(FIXTURE_DIR / capture("0010")["deck"]), str(out)], basis=basis)
        == 0
    )
    written = out.read_bytes().decode("latin-1")
    assert "ANTENNA INPUT PARAMETERS" in written
    assert_stamped(written, basis=basis)


@pytest.mark.integration
@pytest.mark.parametrize("basis", [_serve.BASIS, "razor-2p"])
def test_the_resident_route_stamps_the_same_basis_the_flag_named(monkeypatch, basis):
    """``--serve --basis <name>``, from the flag down to the bytes on the wire.

    Driven through :func:`momwire.eznec._resident.serve_main` rather than
    around it, because the flag's own parse is half the claim: the socket name
    hashes the engine identity client-side, so ``--basis`` is the ONLY thing
    that tells a warm server which formulation it is (``_resident``'s module
    docstring), and a route that stamped the default while serving the twin
    would be invisible in the answer.
    """
    captured = {}

    def fake_serve_forever(path, idle_raw, log_path, connection, *, configure):
        captured["connection"] = connection
        return 0

    monkeypatch.setattr(_resident, "serve_forever", fake_serve_forever)
    assert (
        _resident.serve_main(["--serve", "--socket", "unused.sock", "--basis", basis])
        == 0
    )

    server, client = socket.socketpair()
    log = io.StringIO()

    def work():
        try:
            captured["connection"](server, 1, log, None)
        finally:
            # The connection's own file objects hold dups; closing the socket
            # too is what ends the client's read, and without it the recv
            # below waits for a write end that is never going to close.
            server.close()

    worker = threading.Thread(target=work)
    worker.start()
    try:
        client.settimeout(120)
        client.sendall(deck_text("0010").encode("latin-1"))
        client.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        worker.join(timeout=120)
        client.close()

    answer = b"".join(chunks).decode("latin-1")
    assert "ANTENNA INPUT PARAMETERS" in answer
    assert_stamped(answer, basis=basis)


@pytest.mark.integration
@pytest.mark.parametrize("basis", [_serve.BASIS, "razor-2p"])
def test_a_refused_deck_is_stamped_too_and_the_frame_is_untouched(basis):
    """A refusal has to say which engine refused.  The frame is the contract
    and this checks it survived the extra field: the comment echo intact (or
    EZNEC discards the file as stale and no message reaches anyone), then the
    ``NEC ERROR`` line AFTER it."""
    text = _shell.render(deck_text(REFUSAL_ID), basis=basis)
    assert_stamped(text, basis=basis)
    # The separator eats the ONE blank line `render_refusal` puts between the
    # header and the error, so what is left is the header exactly — trailing
    # newline and all.
    header, _, tail = text.partition("\n ***** NEC ERROR - ")
    assert tail, text[-200:]
    assert "Vertical over real ground" in header
    assert header == render_header(deck_text(REFUSAL_ID), basis=basis)


@pytest.mark.integration
def test_the_dialect_refusal_path_is_stamped_as_well():
    """``_shell.render``'s first gate, which reaches ``render_refusal`` from a
    ``DeckError`` rather than from a ``ServeRefusal`` — a different call site
    with the same obligation."""
    text = _shell.render(NOT_NEC5_DECK, basis="razor-2p")
    assert "NEC ERROR" in text
    assert_stamped(text, basis="razor-2p")


@pytest.mark.integration
def test_an_unreadable_deck_is_stamped_although_it_has_no_echo(tmp_path):
    """The one printout with no comment box at all (``render_header(None)``).
    It cannot pass EZNEC's belongs-to-this-run check whatever it says, but it
    is still written, and it still names the engine that could not read."""
    out = tmp_path / "NEC5.OUT"
    _shell.run(tmp_path / "does-not-exist.nec", out, basis="razor-2p")
    written = out.read_bytes().decode("latin-1")
    assert "UNABLE TO READ INPUT FILE" in written
    assert_stamped(written, basis="razor-2p")


@pytest.mark.integration
def test_the_last_ditch_internal_error_printout_is_stamped(monkeypatch, tmp_path):
    """``main``'s process-level catch re-reads the deck to keep the echo; the
    basis has to reach it too, or the one printout written by a failure
    nobody foresaw is the one that cannot say what produced it."""
    out = tmp_path / "NEC5.OUT"

    def boom(deck_path, printout_path, **kwargs):
        raise RuntimeError("injected")

    monkeypatch.setattr(_shell, "run", boom)
    deck = FIXTURE_DIR / capture("0010")["deck"]
    assert _shell.main([str(deck), str(out)], basis="razor-2p") == 0
    written = out.read_bytes().decode("latin-1")
    assert "INTERNAL ERROR IN MOMWIRE ENGINE" in written
    assert_stamped(written, basis="razor-2p")


@pytest.mark.integration
def test_the_seams_own_last_line_of_defence_is_stamped(monkeypatch):
    """The resident transport's twin of the gate above: ``seam().answer`` must
    never raise, and the printout it substitutes carries the stamp."""

    def boom(text, *, basis):
        raise RuntimeError("injected")

    monkeypatch.setattr(_shell, "render", boom)
    text, err = _shell.seam(basis="razor-2p").answer(deck_text("0010"), "")
    assert err == ""
    assert "INTERNAL ERROR IN MOMWIRE ENGINE" in text
    assert_stamped(text, basis="razor-2p")
