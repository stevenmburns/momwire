"""The one-shot process shell EZNEC launches: argv in, printout out, exit 0.

EZNEC Pro+ v7 runs its NEC-5 console engine once per calculation, as

    NEC5CL_x13.exe "EZN5.NEC" "NEC5.OUT"      cwd = the engine's directory

— two quoted positional arguments (input deck, output printout), resolved
against the current directory, stdin never read, stderr empty, exit 0.  This
module is that protocol and nothing else; :mod:`momwire.eznec._printout` owns
the bytes it writes, and U2 onward own what goes below the header.

Everything here is derived from CAPTURED INPUT/OUTPUT ONLY — the launch lines,
the fault-injection table, and the engine's own argument-error paths recorded
in antennaknobs ``docs/status/2026-08-16-eznec-nec5-dialect-capture.md``, with
the deck/printout corpus in ``docs/status/2026-08-20-eznec-nec5-scored-matrix.md``
and ``tests/fixtures/eznec/``.  No NEC-5 internals are described or relied on.

Three rules, and they are the whole unit
----------------------------------------
1. **Exit 0, always.**  EZNEC never reads the exit status: a byte-perfect
   printout returned with exit 1 produced normal results and no complaint.
   Success, refusal, unreadable deck, wrong argument count, an unexpected
   traceback — all exit 0.  The exit code is not a channel in either
   direction, so a drop-in gains nothing by using it and loses nothing by
   returning zero.

2. **Always write the printout.**  Deleting it (or never creating it) sends
   EZNEC down the "this may be due to the location of the external NEC
   program, try a different location" path, which blames the user's install
   for what is actually a refusal.  The file is the only channel there is.

3. **Never read stdin.**  The real engine, given no arguments, prompts for a
   filename — but it reads that from the console device, not stdin, so a
   piped answer is ignored and it dies at end-of-file.  EZNEC never takes
   that path, so this engine does not implement the prompt loop at all: the
   argv form is the whole interface, and stdin stays untouched whether it is
   a terminal, a pipe carrying bytes, or /dev/null.
"""

from __future__ import annotations

import codecs
import sys
import unicodedata
from dataclasses import replace
from pathlib import Path

from ..deck import DeckError
from ..deck._nec5 import Nec5Deck, Nec5Structure, parse_nec5, split_structures
from ..serve import Seam
from . import _nec4, _printout, _serve

__all__ = [
    "ARGUMENT_ERROR_INPUT",
    "ARGUMENT_ERROR_OUTPUT",
    "DIALECTS",
    "main",
    "run",
    "seam",
]

# Observed verbatim: `NEC5CL_x13.exe deck.nec` with no second argument prints
# exactly this to stdout and exits 0 (capture doc, "Invocation protocol").
ARGUMENT_ERROR_OUTPUT = "ERROR getting output file from command line"

# NOT observed — the real engine answers a bare command line with its console
# prompt loop, which is deliberately not implemented (see rule 3 above).  The
# same shape and the same exit 0 is the honest stand-in for a path EZNEC never
# takes; it is spelled symmetrically so a human reading a terminal sees which
# half of the command line went missing.
ARGUMENT_ERROR_INPUT = "ERROR getting input file from command line"

# The two EZNEC slots this shell answers (momwire#1295): ``nec5`` is the
# External NEC-5 slot every release has served, ``nec4`` the External NEC-4.2
# one (`momwire.eznec._nec4`).  The process protocol is identical for both;
# the program's NAME picks which deck dialect and printout answer it.
DIALECTS = ("nec5", "nec4")

# Decks and printouts are handled as bytes wearing a lossless 1-byte codec:
# whatever a comment card holds has to survive into the echo unchanged, and a
# strict decode would turn one stray byte into a crash on a deck that is
# otherwise perfectly readable.
_CODEC = "latin-1"

# What a printout character OUTSIDE latin-1 becomes on the way out.  The
# deck's own text arrives through `_CODEC` and so always fits; what does not
# is momwire's prose — a refusal sentence the solver or a capability row wrote
# with an em dash, an omega or a minus sign.  momwire#1100 swept the seam's
# OWN constants clean, and the sinusoidal family's knot-feed refusal (read off
# its capability row, a module that sweep never saw) still crashed the writer
# into "INTERNAL ERROR ... UnicodeEncodeError" on Dan AC6LA's working deck
# (momwire#1282).  A refusal is the one thing the printout exists to carry,
# so the codec boundary answers it: any text, from any module, reaches the
# file — transliterated rather than raised over or replaced with "?".
_TRANSLITERATIONS = {
    "—": "-",  # em dash; spaced to " - " where the prose left no space
    "–": "-",
    "−": "-",
    "‐": "-",
    "‑": "-",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "…": "...",
    "→": "->",
    "←": "<-",
    "≤": "<=",
    "≥": ">=",
    "≠": "!=",
    "≈": "~",
    "≡": "==",
    "√": "sqrt",
    "∞": "inf",
    "Ω": "ohm",
    "λ": "lambda",
    "μ": "u",
    "ε": "eps",
    "σ": "sigma",
    "ρ": "rho",
    "π": "pi",
    "ω": "omega",
}
_PRINTOUT_ERRORS = "momwire-eznec-printout"


def _spell(text: str, i: int) -> str:
    """``text[i]``, a character outside latin-1, spelled inside it."""
    char = text[i]
    spelled = _TRANSLITERATIONS.get(char)
    if spelled is None:
        spelled = "".join(
            c
            for c in unicodedata.normalize("NFKD", char)
            if ord(c) < 0x100 and not unicodedata.combining(c)
        )
        return spelled or "?"
    if char == "\u2014":
        # A separator: padded to " - " on whichever side the prose left
        # unspaced, so "a — b" and "a—b" both print as "a - b".
        before = text[i - 1] if i > 0 else " "
        after = text[i + 1] if i + 1 < len(text) else " "
        spelled = (
            ("" if before.isspace() else " ")
            + spelled
            + ("" if after.isspace() else " ")
        )
    return spelled


def transliterate(text: str) -> str:
    """``text`` with every character outside latin-1 spelled inside it.

    The table above first; anything else by its compatibility decomposition
    (``unicodedata.normalize("NFKD")``) where that lands in latin-1, and
    ``?`` only when nothing does.
    """
    if text.isascii():
        return text
    return "".join(
        char if ord(char) < 0x100 else _spell(text, i) for i, char in enumerate(text)
    )


def _transliterate_error(exc: UnicodeError) -> tuple[str, int]:
    """The codec error handler both transports encode the printout with: the
    one-shot shell's file here, the resident server's socket wrapper there."""
    if not isinstance(exc, UnicodeEncodeError):
        raise exc
    text = exc.object
    return "".join(_spell(text, i) for i in range(exc.start, exc.end)), exc.end


codecs.register_error(_PRINTOUT_ERRORS, _transliterate_error)


def read_deck(deck_path: Path) -> str | None:
    """The deck's text, or ``None`` when it cannot be read at all.

    ``None`` is not an exception because an unreadable input is not an
    exceptional outcome at this seam — it is one more thing the printout has
    to say out loud (rule 2).
    """
    try:
        return deck_path.read_bytes().decode(_CODEC)
    except OSError:
        return None


def write_printout(printout_path: Path, text: str) -> None:
    """Write the printout with CRLF endings, whatever the platform prefers.

    Downstream DOES care, and momwire#512 is the measurement: EZNEC refuses
    an engine whose printout is LF-only, behind the misleading popup "Output
    file NEC.OUT is present, but was written earlier from another
    calculation" — naming a file that never existed in any run.  Proven by
    controlled substitution on the Windows box (capture sitting 4,
    antennaknobs PR #970): a wrapper changing NOTHING but the line endings
    made EZNEC render this engine's results first try.  The captured
    printouts are the Windows engine's own and arrived CRLF; this seam
    writes what the engine writes, and the byte-gates that compare the
    RENDERED STRING keep normalizing per the fixture manifest while the
    shell gate compares the written bytes — the layer this defect hid in.

    And whatever the text holds, it is written: a character outside latin-1
    is transliterated at the codec (:func:`transliterate`, momwire#1282)
    rather than raised over, so a refusal never turns into an internal error
    on its way to the file.
    """
    with printout_path.open(
        "w", encoding=_CODEC, errors=_PRINTOUT_ERRORS, newline="\r\n"
    ) as handle:
        handle.write(text)


def render(text: str, *, basis: str = _serve.BASIS, dialect: str = "nec5") -> str:
    """One deck's text as a printout: the results, or a refusal that says why.

    ``dialect`` is the EZNEC slot (:data:`DIALECTS`).  ``"nec4"`` hands the
    whole deck to :func:`momwire.eznec._nec4.render`; everything below this
    paragraph describes the ``"nec5"`` path, which it leaves untouched.  An
    unknown dialect is a refusal naming it, like an unknown basis.

    ``basis`` picks which momwire formulation answers it (momwire#603 U3) and
    is passed straight to :func:`~momwire.eznec._serve.serve`, whose default
    is the one every committed printout is gated against.  A name the roster
    does not know, and a basis this deck's own geometry cannot be hosted by,
    both come back through gate 3 below as a refusal naming the basis.

    Three gates in order, and each of them names the thing that stopped it:

    1. the dialect front end (U2) refuses a card that is not in the observed
       vocabulary, or a field form no capture has shown;
    2. :func:`~momwire.eznec._serve.refusal` refuses a deck that parses but
       is above the served rungs of the scored ladder — a near field, an
       interleaved network table, a multi-source drive in one of the shapes
       no capture writes;
    3. the translation itself refuses an address the mesh cannot host.

    The stub refusal U1 shipped stays behind all three, and only for a case
    none of them foresaw: a deck that parses, passes the ladder check, builds
    a mesh, and still fails somewhere the seam has no sentence for.  That
    surfaces through :func:`main`'s last line of defence rather than here.
    """
    if dialect == "nec4":
        return _nec4.render(text, basis=basis)
    if dialect != "nec5":
        known = ", ".join(repr(name) for name in DIALECTS)
        return _printout.render_refusal(
            text, f"unknown EZNEC dialect {dialect!r}; known: {known}", basis=basis
        )
    structures = split_structures(text)
    if structures is not None:
        return _render_chain(structures, basis=basis)
    answered = _answer(text, basis=basis)
    if isinstance(answered, str):
        return answered
    deck, answers, multi = answered
    if multi:
        return _printout.render_multi_printout(deck, answers, basis=basis)
    return _printout.render_printout(deck, answers[0], basis=basis)


def _answer(
    text: str, *, basis: str, echo_text: str | None = None
) -> tuple[Nec5Deck, list[_printout.RunData], bool] | str:
    """One structure's deck parsed and solved: ``(deck, answers, multi)``,
    or the refusal printout naming why it could not be.

    ``multi`` is momwire#1237's multi-run deck — one ``XQ`` block per run,
    each served as its own single-run deck, printed in the licensed
    multi-run layout. ``echo_text`` is what the printout echoes when it is
    not the deck that was solved: a structure an ``NX`` closed is SOLVED as
    the standalone deck its body makes (closed by ``EN``) and ECHOED with the
    ``NX`` it arrived with.
    """
    echo = text if echo_text is None else echo_text
    try:
        deck = parse_nec5(text)
    except DeckError as exc:
        return _printout.render_refusal(echo, str(exc), basis=basis)
    if echo_text is not None:
        deck = replace(deck, source_text=echo_text)
    try:
        runs = _serve.split_runs(text)
    except _serve.ServeRefusal as exc:
        return _printout.render_refusal(echo, str(exc), basis=basis)
    if runs is not None:
        try:
            answers = _serve.serve_runs([parse_nec5(t) for t in runs], basis=basis)
        except (DeckError, _serve.ServeRefusal) as exc:
            return _printout.render_refusal(echo, str(exc), basis=basis)
        return deck, answers, True
    try:
        data = _serve.serve(deck, basis=basis)
    except _serve.ServeRefusal as exc:
        return _printout.render_refusal(echo, str(exc), basis=basis)
    return deck, [data], False


def _render_chain(structures: list[Nec5Structure], *, basis: str) -> str:
    """An ``NX``-chained stream's printout: each structure's own printout in
    turn, as the licensed NEC-5 lays one out (measured black-box on a
    two-structure deck).

    Every structure starts from scratch — the engine carries no frequency,
    ground or load across ``NX`` — so each is solved exactly as the
    standalone deck its body makes. Each prints in full, header and comment
    box included; a structure that ``NX`` closes ends on its ``NX`` echo
    (numbered on from its own cards, where a lone deck's ``EN`` echo stands)
    and is followed straight away by the next structure's header, and the
    stream's last structure ends on the ``EN`` echo and the one ``RUN TIME``,
    the whole stream's.

    SimNEC is the caller that sends this: with enough runs queued it batches
    several into one engine call, ``NX`` between them, and reads each run's
    printout up to the ``NX`` echo. A structure that refuses ends the stream
    there with its own refusal printout, naming the card, after the
    structures that solved.
    """
    printouts: list[str] = []
    seconds = 0.0
    for k, structure in enumerate(structures):
        last = k == len(structures) - 1
        answered = _answer(
            structure.deck_text, basis=basis, echo_text=structure.echo_text
        )
        if isinstance(answered, str):
            printouts.append(answered)
            break
        deck, answers, multi = answered
        seconds += sum(a.run_seconds for a in answers)
        if multi:
            printout = _printout.render_multi_printout(
                deck, answers, basis=basis, closing=last, run_seconds=seconds
            )
        else:
            printout = _printout.render_printout(
                deck, answers[0], basis=basis, closing=last, run_seconds=seconds
            )
        printouts.append(printout)
    return "".join(printouts)


def run(
    deck_path: str | Path,
    printout_path: str | Path,
    *,
    basis: str = _serve.BASIS,
    dialect: str = "nec5",
) -> str:
    """Serve one deck: read it, render a printout, write the printout.

    Returns the text written, which is the whole of what this process
    communicates.  A rung-1 deck comes back as a full printout; everything
    else comes back as the same header with one ``NEC ERROR`` line under it,
    naming the card or the reason.

    Paths are used as given, so they resolve against the current directory the
    way EZNEC's cwd-relative ``"EZN5.NEC"`` does.
    """
    deck = Path(deck_path)
    out = Path(printout_path)

    text = read_deck(deck)
    if text is None:
        printout = _printout.render_refusal(
            None, f"UNABLE TO READ INPUT FILE {deck}", basis=basis, dialect=dialect
        )
    else:
        printout = render(text, basis=basis, dialect=dialect)

    write_printout(out, printout)
    return printout


def main(
    argv: list[str] | None = None,
    *,
    basis: str = _serve.BASIS,
    dialect: str = "nec5",
) -> int:
    """The process entry point.  Returns 0.  Always returns 0.

    ``basis`` is not a command line flag and must not become one: the real
    engine takes two positional paths and nothing else, and EZNEC sends
    exactly those.  It is here so that ONE frozen executable can be built per
    formulation (momwire#593) — the entry script names the basis, the process
    answers every deck in it, and EZNEC's engine-path setting is what
    chooses.  ``dialect`` is the same kind of choice and rides the same way
    (momwire#1295): the launcher's name picks the EZNEC slot.

    Argument errors go to stdout and write no printout — that is what the real
    engine does, and it is the one case where writing a file would be wrong:
    there is no output path to write it to.  Every other failure, including an
    unexpected exception from anywhere below, still leaves a printout behind
    with a ``NEC ERROR`` line in it.
    """
    args = list(sys.argv[1:] if argv is None else argv)

    if len(args) < 1:
        print(ARGUMENT_ERROR_INPUT)
        return 0
    if len(args) != 2:
        # One argument is the observed case.  More than two is not observed —
        # EZNEC always sends exactly two, quoted — and is treated the same
        # way: the command line did not resolve to an output file.
        print(ARGUMENT_ERROR_OUTPUT)
        return 0

    deck_path, printout_path = args
    try:
        run(deck_path, printout_path, basis=basis, dialect=dialect)
    except Exception as exc:  # noqa: BLE001 - the last line of defence
        # A traceback on stderr would be invisible (EZNEC captures neither
        # stderr nor the exit code), and a missing file would be read as a
        # broken installation.  So the crash is reported the only way that
        # reaches a human: as a refusal, in the printout, exit 0.
        _report_internal_error(
            deck_path, printout_path, exc, basis=basis, dialect=dialect
        )
    return 0


def _report_internal_error(
    deck_path: str | Path,
    printout_path: str | Path,
    exc: BaseException,
    *,
    basis: str = _serve.BASIS,
    dialect: str = "nec5",
) -> None:
    """Last-ditch printout for a failure the shell did not anticipate.

    The deck is re-read so the refusal still carries the comment echo:
    without the stamp, EZNEC rejects the file as stale and the message
    never reaches anyone (capture doc, "Error convention").  Re-reading is
    the simplest way to have the text here without threading it through
    every failure path, and the deck is still on disk — EZNEC wrote it
    moments ago.
    """
    reason = _internal_error_reason(exc)
    try:
        deck_text = read_deck(Path(deck_path))
        write_printout(
            Path(printout_path),
            _printout.render_refusal(deck_text, reason, basis=basis, dialect=dialect),
        )
    except OSError:
        # The output path itself is unwritable; there is no channel left, and
        # a non-zero exit would only be discarded. Stay quiet, stay zero.
        pass


def _internal_error_reason(exc: BaseException) -> str:
    """The one spelling of the last-ditch refusal line — :func:`main`'s
    process-level catch and :func:`seam`'s answer-level catch must frame an
    unforeseen failure identically, whichever transport carried the deck."""
    return f"INTERNAL ERROR IN MOMWIRE ENGINE - {type(exc).__name__}: {exc}"


def seam(*, basis: str = _serve.BASIS, dialect: str = "nec5") -> Seam:
    """This dialect as a :class:`momwire.serve.Seam` (momwire#719 U4).

    The eznec contract is the session loop's degenerate case: no greeting, no
    frame vocabulary, the whole input is one body answered once — under
    :func:`momwire.serve.run_session` that is an ``eof_terminator`` frame and
    nothing else, a session of exactly one deck. The resident server the
    #718 arc builds (momwire#532) hosts this seam one connection per deck and
    returns ``answer``'s bytes; the frozen one-shot shell above is the same
    contract spoken over argv and files, and both funnel through
    :func:`render`, so the two transports cannot drift apart.

    ``basis`` is the per-engine formulation exactly as :func:`main` takes it
    (momwire#593: one frozen executable per formulation, the filename names
    the basis) — a resident server hosting several engine variants holds one
    seam per basis.  ``dialect`` is the EZNEC slot, per server like the
    basis (momwire#1295).

    The :class:`~momwire.serve.Seam` invariant — ``answer`` never raises —
    is kept the same way :func:`main` keeps exit 0: anything
    :func:`render`'s three gates did not foresee comes back as the
    ``NEC ERROR`` printout :func:`_internal_error_reason` frames, never as
    an exception the transport has no channel for.
    """

    def answer(body: str, terminator: str) -> tuple[str, str]:
        try:
            return render(body, basis=basis, dialect=dialect), ""
        except Exception as exc:  # noqa: BLE001 - the seam's last line of defence
            return (
                _printout.render_refusal(
                    body, _internal_error_reason(exc), basis=basis, dialect=dialect
                ),
                "",
            )

    return Seam(
        name="eznec",
        greeting=lambda: [],
        terminators=frozenset(),
        closing=frozenset(),
        eof_terminator="",
        answer=answer,
    )
