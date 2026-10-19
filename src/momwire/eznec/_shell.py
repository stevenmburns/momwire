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

3. **Stdin is read only when the command line is empty, or is ``- -``.**
   EZNEC and SimNEC pass the two paths, and with paths on the command line
   stdin stays untouched whether it is a terminal, a pipe carrying bytes, or
   /dev/null.  ``- -`` is SimNEC's streaming call (momwire#1395, served since
   momwire#1512, :func:`stream`): decks on stdin, printouts on stdout.
   With NO arguments, the NEC-5 slot reads the two paths from stdin, one per
   line (momwire#1488): that is how 4nec2 runs its engine, as
   ``ENGINE.exe < nec2d.tmp`` with ``..\\out\\EXAMPLE3.inp`` and
   ``..\\out\\EXAMPLE3.out`` in the file, resolved against the working
   directory like any relative path.  A terminal on stdin is not read (a
   person who types the bare command gets the argument error at once), and
   the NEC-4.2 slot never reads stdin at all.

The 4nec2 front end
-------------------
A deck that arrives through stdin may end without ``EN`` (4nec2 writes none),
and only such a deck: the two-argument form still refuses a deck with no
``EN`` as truncated, because every deck EZNEC and SimNEC write carries one.
The engine closes the deck the way NEC's own card reader does at end of
file, with an ``EN`` the printout echoes.  And any deck whose ``GE`` card
carries a third field of 1 (``GE 0 0 1``, momwire#1488) gets a
``<printout stem>.GMO`` geometry file beside its printout
(:mod:`momwire.eznec._gmo`), which 4nec2's NEC-5 package rebuilds its
SEGMENTATION DATA table from; no EZNEC or SimNEC deck has a third ``GE``
field, so their runs write exactly what they wrote before.
"""

from __future__ import annotations

import codecs
import io
import os
import sys
import unicodedata
from dataclasses import replace
from pathlib import Path

import momwire_serve_client as _mech

from .. import _times
from ..deck import DeckError
from ..deck._nec5 import (
    Nec5Deck,
    Nec5Structure,
    ends_on_nx,
    next_structure,
    parse_nec5,
    parse_nec5_shared,
    split_structures,
)
from ..serve import Seam
from . import _gmo, _nec4, _printout, _serve

__all__ = [
    "ARGUMENT_ERROR_INPUT",
    "ARGUMENT_ERROR_OUTPUT",
    "DIALECTS",
    "main",
    "run",
    "seam",
    "stream",
]

# Observed verbatim: `NEC5CL_x13.exe deck.nec` with no second argument prints
# exactly this to stdout and exits 0 (capture doc, "Invocation protocol").
ARGUMENT_ERROR_OUTPUT = "ERROR getting output file from command line"

# Observed: the real engine names a file it cannot open this way on stdout
# ("UNABLE TO OPEN FILE -i", capture doc).  Printed, followed by the path,
# when the PRINTOUT cannot be written, which is the one failure the printout
# itself cannot carry.
UNABLE_TO_OPEN = "UNABLE TO OPEN FILE"

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


def printout_bytes(text: str) -> bytes:
    """The bytes :func:`write_printout` writes for ``text``: the same text
    layer (CRLF, latin-1, transliterated), for a printout that goes to a
    pipe rather than a file (:func:`stream`)."""
    raw = io.BytesIO()
    handle = io.TextIOWrapper(
        raw, encoding=_CODEC, errors=_PRINTOUT_ERRORS, newline="\r\n"
    )
    handle.write(text)
    handle.flush()
    data = raw.getvalue()
    handle.close()
    return data


def write_gmo(gmo_path: Path, text: str) -> None:
    """Write a ``.GMO`` file with CRLF endings, as the sample has them and as
    every printout this engine writes has them (:func:`write_printout`)."""
    with gmo_path.open("w", encoding=_CODEC, newline="\r\n") as handle:
        handle.write(text)


def close_deck(text: str) -> str:
    """``text`` closed by ``EN`` when no line of it is one (momwire#1488).

    For a deck whose file names came from stdin: 4nec2 writes no ``EN``, and
    NEC's own card reader closes a deck at end of file.  A deck that has an
    ``EN`` anywhere is returned unchanged, which keeps everything after it out
    of the deck exactly as before.
    """
    for line in text.splitlines():
        tokens = line.strip().replace(",", " ").split()
        if tokens and tokens[0][:2].upper() == "EN":
            return text
    if text and not text.endswith(("\n", "\r")):
        text += "\n"
    return text + "EN\n"


def render_files(
    text: str,
    *,
    basis: str = _serve.BASIS,
    dialect: str = "nec5",
    en_optional: bool = False,
) -> tuple[str, str | None]:
    """``(printout, gmo)``: :func:`render`'s printout, and the ``.GMO``
    geometry file the deck asks for, or ``None`` when it asks for none
    (momwire#1488).  The NEC-4.2 slot never writes one.

    ``en_optional`` is the stdin form's: the deck is closed by
    :func:`close_deck` before anything reads it.
    """
    if en_optional and dialect == "nec5":
        text = close_deck(text)
    printout = render(text, basis=basis, dialect=dialect)
    gmo = _gmo.gmo_text(text) if dialect == "nec5" else None
    return printout, gmo


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
    deck, answers, layout = answered
    return _print(deck, answers, layout, basis=basis)


def _print(
    deck: Nec5Deck,
    answers: list[_printout.RunData],
    layout: str,
    *,
    basis: str,
    closing: bool = True,
    run_seconds: float | None = None,
) -> str:
    """One structure's printout in the layout its answers came in:
    ``"single"``, ``"multi"`` (momwire#1237's ``XQ`` blocks) or
    ``"stepped"`` (momwire#1508's stepping ``FR``).

    Under MOMWIRE_TIMES its ``RUN TIME`` is the answer's measured seconds."""
    times = _times.current()
    if times is not None and closing:
        run_seconds = times.run_seconds()
    with _times.stage("printout"):
        return _print_layout(
            deck, answers, layout, basis=basis, closing=closing, run_seconds=run_seconds
        )


def _print_layout(
    deck: Nec5Deck,
    answers: list[_printout.RunData],
    layout: str,
    *,
    basis: str,
    closing: bool,
    run_seconds: float | None,
) -> str:
    if layout == "sequence":
        (sequence,) = answers
        return _printout.render_sequence_printout(
            deck,
            sequence.runs,
            sequence.tail,
            basis=basis,
            closing=closing,
            run_seconds=run_seconds,
        )
    if layout == "multi":
        return _printout.render_multi_printout(
            deck, answers, basis=basis, closing=closing, run_seconds=run_seconds
        )
    if layout == "stepped":
        return _printout.render_stepped_printout(
            deck, answers, basis=basis, closing=closing, run_seconds=run_seconds
        )
    return _printout.render_printout(
        deck, answers[0], basis=basis, closing=closing, run_seconds=run_seconds
    )


def _answer(
    text: str, *, basis: str, echo_text: str | None = None
) -> tuple[Nec5Deck, list[_printout.RunData], str] | str:
    """One structure's deck parsed and solved: ``(deck, answers, layout)``,
    or the refusal printout naming why it could not be.

    ``layout`` is ``"single"``; ``"multi"`` for momwire#1237's multi-run
    deck — one ``XQ`` block per run, each served as its own single-run deck,
    printed in the licensed multi-run layout; or ``"stepped"`` for
    momwire#1508's stepping ``FR`` — the one run solved at each of its
    frequencies in turn, in this process, so every cache a solve fills
    serves the next frequency. ``echo_text`` is what the printout echoes when it is
    not the deck that was solved: a structure an ``NX`` closed is SOLVED as
    the standalone deck its body makes (closed by ``EN``) and ECHOED with the
    ``NX`` it arrived with.
    """
    echo = text if echo_text is None else echo_text
    # momwire#1420: the deck and its runs share their leading cards, so they
    # are parsed together (`parse_nec5_shared`) — which needs the runs'
    # texts first. Splitting is pure text work; its refusal, like each
    # parse's, is held and raised exactly where the one-at-a-time spelling
    # raised it: the deck's own parse first, then the split, then the runs.
    try:
        runs = _serve.split_runs(text)
        split_failure = None
    except Exception as exc:  # noqa: BLE001 — held, raised after the parse
        runs, split_failure = None, exc
    parsed = parse_nec5_shared([text, *(runs or ())])
    try:
        deck = _raised(parsed[0])
    except DeckError as exc:
        return _printout.render_refusal(echo, str(exc), basis=basis)
    if echo_text is not None:
        deck = replace(deck, source_text=echo_text)
    # momwire 0.79.2: a deck whose frequency changes between its runs, or
    # whose first run steps it and another run follows, read as the licensed
    # NEC-5 reads it (`_serve.split_sequence`), before the one-run and
    # multi-run readings below, which it replaces for exactly those decks.
    try:
        sequence = _serve.split_sequence(text)
    except _serve.ServeRefusal as exc:
        return _printout.render_refusal(echo, str(exc), basis=basis)
    if sequence is not None:
        return _answer_sequence(deck, *sequence, echo=echo, basis=basis)
    try:
        if split_failure is not None:
            raise split_failure
    except _serve.ServeRefusal as exc:
        return _printout.render_refusal(echo, str(exc), basis=basis)
    if deck.frequencies_mhz:
        reason = _stepping_refusal(deck, multi_run=runs is not None)
        if reason is not None:
            return _printout.render_refusal(echo, reason, basis=basis)
        # One sweep state for every frequency: the fill's
        # wavenumber-independent half is built once (`_serve._port_solution`).
        # The whole frequency list is known here, before the first solve.
        shared: dict = {}
        try:
            answers = [
                _serve.serve(replace(deck, frequency_mhz=f), basis=basis, shared=shared)
                for f in deck.frequencies_mhz
            ]
        except _serve.ServeRefusal as exc:
            return _printout.render_refusal(echo, str(exc), basis=basis)
        return deck, answers, "stepped"
    if runs is not None:
        try:
            answers = _serve.serve_runs(
                [_raised(run) for run in parsed[1:]], basis=basis
            )
        except (DeckError, _serve.ServeRefusal) as exc:
            return _printout.render_refusal(echo, str(exc), basis=basis)
        return deck, answers, "multi"
    try:
        data = _serve.serve(deck, basis=basis)
    except _serve.ServeRefusal as exc:
        return _printout.render_refusal(echo, str(exc), basis=basis)
    return deck, [data], "single"


class _Sequence:
    """A sequence deck's answers (`_answer_sequence`): its runs and the
    cards echoed after the last, with the seconds the chain sums."""

    def __init__(self, runs: list[_printout.SequenceRun], tail: list[int]):
        self.runs = runs
        self.tail = tail
        self.run_seconds = sum(
            getattr(d, "run_seconds", 0.0) for run in runs for d in run.answers
        )


def _answer_sequence(deck, groups, tail, *, echo: str, basis: str):
    """Each run of a sequence deck (`_serve.split_sequence`) solved at the
    frequency in force: a run that carries an ``FR`` (and the first) over
    its whole list, a run that does not at the last frequency the run
    before it solved, as the licensed NEC-5 leaves it after a loop."""
    runs: list[_printout.SequenceRun] = []
    last: float | None = None
    try:
        for group in groups:
            try:
                deck_g = parse_nec5(group.text)
            except DeckError as exc:
                return _printout.render_refusal(echo, str(exc), basis=basis)
            if group.full:
                freqs = deck_g.frequencies_mhz or (deck_g.frequency_mhz,)
            else:
                freqs = (last,)
            # one sweep state per run: the runs' decks differ
            shared: dict = {}
            answers = tuple(
                _serve.serve(
                    replace(deck_g, frequency_mhz=f), basis=basis, shared=shared
                )
                for f in freqs
            )
            last = freqs[-1]
            runs.append(
                _printout.SequenceRun(
                    cards=group.cards,
                    answers=answers,
                    full=group.full,
                    requests_only=not group.full and not group.sources,
                )
            )
    except _serve.ServeRefusal as exc:
        return _printout.render_refusal(echo, str(exc), basis=basis)
    return deck, [_Sequence(runs, tail)], "sequence"


def _stepping_refusal(deck: Nec5Deck, *, multi_run: bool) -> str | None:
    """Why a stepping-``FR`` deck is not served, or ``None`` when it is.

    Served: ONE run over the frequencies, executed by an ``XQ`` (near-field
    requests before it print per frequency) or by an ``RP`` (likewise, and
    an ``XQ`` after the ``RP`` runs nothing), which is every shape the
    licensed NEC-5 prints a full frequency loop for (momwire#1508's probes).
    Refused, each by name: a deck with nothing that executes the loop, which
    NEC-5 runs no frequency for; and a request or block after the loop,
    which NEC-5 answers at the last frequency only, a layout this seam does
    not reproduce yet.
    """
    if multi_run:
        return _STEPPING_MULTI_RUN
    cards = [c.mnemonic for c in _printout.post_ge_cards(deck.source_text)]
    executes = [i for i, m in enumerate(cards) if m in ("RP", "XQ")]
    if not executes:
        return _STEPPING_NOTHING_EXECUTES
    after = [m for m in cards[executes[0] + 1 :] if m not in ("XQ", "EN", "NX")]
    if after:
        return _STEPPING_AFTER_THE_LOOP.format(card=after[0])
    return None


_STEPPING_MULTI_RUN = (
    "a stepping FR is served in a deck with one run; this deck has several "
    "XQ blocks, and the licensed NEC-5 answers every block after the first at "
    "the sweep's last frequency only, which this engine does not reproduce. "
    "Give each run its own deck, or step one frequency per run"
)
_STEPPING_NOTHING_EXECUTES = (
    "a stepping FR needs an XQ or RP after it to run the sweep; without one "
    "the licensed NEC-5 solves no frequency at all (a near-field request "
    "alone does not start the loop), so add XQ after the last request"
)
_STEPPING_AFTER_THE_LOOP = (
    "{card} follows the card that runs the stepping FR's sweep; the licensed "
    "NEC-5 answers it at the sweep's last frequency only, which this engine "
    "does not reproduce. Put every request before the first RP or XQ"
)


def _raised(outcome: Nec5Deck | Exception) -> Nec5Deck:
    """A :func:`parse_nec5_shared` outcome as ``parse_nec5`` would have
    returned it: the deck, or its exception raised here."""
    if isinstance(outcome, Exception):
        raise outcome
    return outcome


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
        # A structure NX closes never closes the run, the last of an open
        # stream's included (momwire#1512).
        last = k == len(structures) - 1 and not ends_on_nx(structure)
        answered = _answer(
            structure.deck_text, basis=basis, echo_text=structure.echo_text
        )
        if isinstance(answered, str):
            printouts.append(answered)
            break
        deck, answers, layout = answered
        seconds += sum(a.run_seconds for a in answers)
        printouts.append(
            _print(
                deck, answers, layout, basis=basis, closing=last, run_seconds=seconds
            )
        )
    return "".join(printouts)


def run(
    deck_path: str | Path,
    printout_path: str | Path,
    *,
    basis: str = _serve.BASIS,
    dialect: str = "nec5",
    en_optional: bool = False,
) -> str:
    """Serve one deck: read it, render a printout, write the printout.

    Returns the text written, which is the whole of what this process
    communicates.  A rung-1 deck comes back as a full printout; everything
    else comes back as the same header with one ``NEC ERROR`` line under it,
    naming the card or the reason.

    Paths are used as given, so they resolve against the current directory the
    way EZNEC's cwd-relative ``"EZN5.NEC"`` does.

    A deck that asks for a ``.GMO`` file (``GE`` third field 1, momwire#1488)
    gets it written beside the printout, before it, at
    ``<printout stem>.GMO``.  ``en_optional`` is the stdin form's: a deck
    with no ``EN`` is closed rather than refused.
    """
    deck = Path(deck_path)
    out = Path(printout_path)

    text = read_deck(deck)
    gmo = None
    if text is None:
        printout = _printout.render_refusal(
            None, f"UNABLE TO READ INPUT FILE {deck}", basis=basis, dialect=dialect
        )
    else:
        printout, gmo = render_files(
            text, basis=basis, dialect=dialect, en_optional=en_optional
        )

    with _times.stage("write"):
        if gmo is not None:
            write_gmo(Path(_gmo.gmo_path(str(printout_path))), gmo)
        write_printout(out, printout)
    return printout


STREAMING_REFUSAL = (
    "momwire serves the streaming mode (decks on stdin, printouts on stdout) as '- -' only, "
    "and only in its NEC-5 slot; give it '- -', or two file paths (momwire#1395, momwire#1512)"
)

_STREAM_CHUNK = 65536


def _stream_piece(text: str, *, basis: str) -> tuple[str, bool]:
    """One piece of a stream as its printout: ``(printout, go_on)``.

    A piece :func:`next_structure` cut at an ``NX`` is that structure,
    solved as the standalone deck its body makes and printed up to its
    ``NX`` echo with no ``RUN TIME`` (another structure follows), exactly as
    :func:`_render_chain` prints it inside a file. Any other piece (one cut
    at ``EN``, or what was left when stdin closed) is a whole deck, printed
    as :func:`render` prints a file holding it. ``go_on`` is false once the
    stream is over: after ``EN``, after the last piece, and after a
    structure that refuses, which ends a stream as it ends a chain. An
    exception is the INTERNAL ERROR printout :func:`main` would write for
    a file holding the piece, and ends the stream too.
    """
    try:
        structures = split_structures(text, open_end=True)
        if structures is None:
            return render(text, basis=basis), False
        (structure,) = structures
        answered = _answer(
            structure.deck_text, basis=basis, echo_text=structure.echo_text
        )
        if isinstance(answered, str):
            return answered, False
        deck, answers, layout = answered
        return _print(deck, answers, layout, basis=basis, closing=False), True
    except Exception as exc:  # noqa: BLE001 - the stream's last line of defence
        return (
            _printout.render_refusal(text, _internal_error_reason(exc), basis=basis),
            False,
        )


def _timed_piece(piece: str, basis: str, write) -> bool:
    """One stream piece answered and written (``go_on``); under
    MOMWIRE_TIMES with its ``times:`` line on stderr."""
    if not _times.env_on():
        printout, go_on = _stream_piece(piece, basis=basis)
        write(printout_bytes(printout))
        return go_on
    with _times.answering("stream", **_times_pre()) as times:
        printout, go_on = _stream_piece(piece, basis=basis)
        with _times.stage("write"):
            write(printout_bytes(printout))
    _emit_times(times)
    return go_on


def _times_pre() -> dict:
    """The ``times:`` line's process start and load for this process's FIRST
    answer: the standalone executable's (its fallback hands them over as
    ``sys._mw_times_pre``) plus the momwire import, else the import alone.
    Taken once: a stream's later pieces start from nothing."""
    global _TIMES_PRE_TAKEN
    if _TIMES_PRE_TAKEN:
        return {}
    _TIMES_PRE_TAKEN = True
    pre = getattr(sys, "_mw_times_pre", None) or {}
    out = {"load_ms": float(pre.get("load_ms", 0.0)) + _times.import_ms()}
    if "start_ms" in pre:
        out["start_ms"] = float(pre["start_ms"])
    return out


_TIMES_PRE_TAKEN = False


def _emit_times(times) -> None:
    """``times``' line on stderr (MOMWIRE_TIMES), flushed: a stream's reader
    may hold the process open."""
    pre = getattr(sys, "_mw_times_pre", None)
    if pre and pre.get("route") and times.route == "python":
        times.route = pre["route"]
    print(times.line(), file=sys.stderr, flush=True)


def stream(read=None, write=None, *, basis: str = _serve.BASIS) -> int:
    """SimNEC's streaming call, ``- -`` (momwire#1395, momwire#1512).

    SimNEC starts the engine once and holds it: it writes one structure's
    deck to stdin, ending on ``NX``, reads that structure's printout from
    stdout up to its ``NX`` echo, then writes the next, and closes stdin
    after the last ``NX`` with no ``EN`` (the A4 captures). So each
    structure is answered as soon as its ``NX`` line is complete
    (:func:`next_structure`) and its printout flushed before anything more
    is read; stdin closing ends the stream quietly, and nothing is printed
    for it. The printouts are the ones a file holding the whole stream gets
    (:func:`_render_chain`), structure by structure, without the truncated
    empty structure a file ending on ``NX`` is refused for.

    What else can arrive is answered as a file holding it would be: an
    ``EN`` closes the stream with that structure's closing printout and
    nothing after it is read; cards left when stdin closes without an
    ``NX`` after them are a deck of their own (with no ``EN`` it is refused
    as truncated); a refusal, or an exception, ends the stream after its
    printout. Exit 0, as for a file.

    ``read`` returns the next bytes of stdin (``b""`` at its end) and
    ``write`` takes each printout's bytes and flushes them; both default to
    the process's own stdin and stdout.
    """
    if read is None:
        fd = sys.stdin.fileno()

        def read() -> bytes:
            return os.read(fd, _STREAM_CHUNK)

    if write is None:
        out = sys.stdout.buffer

        def write(data: bytes) -> None:
            out.write(data)
            out.flush()

    pending = ""
    swallow_lf = False  # an NX line cut at a bare CR whose LF had not arrived
    final = False
    while not final:
        data = read()
        final = not data
        chunk = data.decode(_CODEC)
        if swallow_lf and chunk:
            if chunk.startswith("\n"):
                chunk = chunk[1:]
            swallow_lf = False
        pending += chunk
        while True:
            cut = next_structure(pending, final=final)
            if cut is None:
                if final and pending.strip():
                    _timed_piece(pending, basis, write)
                break
            end = cut[0]
            piece, pending = pending[:end], pending[end:]
            swallow_lf = piece.endswith("\r") and not pending and not final
            if not _timed_piece(piece, basis, write):
                return 0
    return 0


def main(
    argv: list[str] | None = None,
    *,
    basis: str = _serve.BASIS,
    dialect: str = "nec5",
    en_optional: bool = False,
) -> int:
    """The process entry point.  Returns 0, except for a ``-`` that is not
    SimNEC's streaming call (``- -``, :func:`stream`), which is refused on
    stderr with 2 (momwire#1395).

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

    With no arguments the NEC-5 slot takes the two paths from stdin, one per
    line (4nec2's form, momwire#1488), and the deck may then end without
    ``EN``.  ``en_optional`` is that same allowance passed by a launcher whose
    OWN paths came from stdin and which runs this as its one-shot rung with
    the paths on the command line; like ``basis`` it is not a flag.
    """
    args = list(sys.argv[1:] if argv is None else argv)

    if not args and dialect == "nec5":
        from_stdin = _mech.stdin_paths()
        if from_stdin is not None:
            # One name is the one-argument fault, none the empty command line.
            args = from_stdin
            en_optional = True

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
    if "-" in (deck_path, printout_path):
        if args == ["-", "-"] and dialect == "nec5":
            # SimNEC's streaming call (momwire#1512): decks on stdin,
            # printouts on stdout.
            try:
                return stream(basis=basis)
            except BrokenPipeError:
                # The caller stopped reading: nobody is left to tell.
                return 0
        # Any other '-' (momwire#1395). There is no printout file to refuse
        # in, and answering 0 with nothing on stdout leaves the caller
        # waiting forever, so this one refusal goes to stderr with a failing
        # status, which SimNEC reports as a NEC failure.
        print(STREAMING_REFUSAL, file=sys.stderr)
        return 2
    times_on = _times.env_on()
    try:
        if times_on:
            with _times.answering("python", **_times_pre()) as times:
                run(
                    deck_path,
                    printout_path,
                    basis=basis,
                    dialect=dialect,
                    en_optional=en_optional,
                )
            _emit_times(times)
        else:
            run(
                deck_path,
                printout_path,
                basis=basis,
                dialect=dialect,
                en_optional=en_optional,
            )
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
        # The output path itself is unwritable, so the printout channel is
        # gone.  Say so on stdout, the channel the argument errors use, rather
        # than exit 0 having written nothing anywhere (momwire#1488: a quoted
        # path did exactly that).  Still exit 0: the code is not a channel.
        print(f"{UNABLE_TO_OPEN} {printout_path}")


def _internal_error_reason(exc: BaseException) -> str:
    """The one spelling of the last-ditch refusal line — :func:`main`'s
    process-level catch and :func:`seam`'s answer-level catch must frame an
    unforeseen failure identically, whichever transport carried the deck."""
    return f"INTERNAL ERROR IN MOMWIRE ENGINE - {type(exc).__name__}: {exc}"


# The resident engine's side of the 4nec2 front end's wire additions
# (momwire#1488), as text: the socket wrapper is latin-1 with CRLF
# translation, so these are `momwire_serve_client`'s byte constants exactly.
_EN_OPTIONAL_REQUEST = _mech.EN_OPTIONAL_REQUEST.decode("latin-1")
_GMO_FRAME_START = _mech.GMO_FRAME_START.decode("latin-1").replace("\r\n", "\n")
_GMO_FRAME_END = _mech.GMO_FRAME_END.decode("latin-1").replace("\r\n", "\n")
_TIMES_REQUEST = _mech.TIMES_REQUEST.decode("latin-1")
_TIMES_FRAME_START = _mech.TIMES_FRAME_START.decode("latin-1").replace("\r\n", "\n")
_TIMES_FRAME_END = _mech.TIMES_FRAME_END.decode("latin-1").replace("\r\n", "\n")


def _request(body: str) -> tuple[str, bool]:
    """``(deck text, en_optional)`` from a resident request's body: a launcher
    whose paths came from stdin sends :data:`momwire_serve_client.
    EN_OPTIONAL_REQUEST` ahead of the deck (after MOMWIRE_TIMES's request
    line, :data:`momwire_serve_client.TIMES_REQUEST`, when it sends one)."""
    if body.startswith(_TIMES_REQUEST):
        body = body[len(_TIMES_REQUEST) :]
    if body.startswith(_EN_OPTIONAL_REQUEST):
        return body[len(_EN_OPTIONAL_REQUEST) :], True
    return body, False


def framed(printout: str, gmo: str | None) -> str:
    """The resident answer: the printout alone, or the ``.GMO`` file framed
    ahead of it (:func:`momwire_serve_client.split_answer` is the reader)."""
    if gmo is None:
        return printout
    return f"{_GMO_FRAME_START}{gmo}{_GMO_FRAME_END}{printout}"


def seam_failure(
    body: str, exc: BaseException, *, basis: str = _serve.BASIS, dialect: str = "nec5"
) -> tuple[str, str]:
    """The seam's answer for a deck whose solve failed unforeseen: the
    ``NEC ERROR`` printout :func:`_internal_error_reason` frames.

    One owner because two callers must frame it identically: :func:`seam`'s
    own catch, and the resident server's worker pool when the worker solving
    the deck dies under it (momwire#1418) -- the caller reads the same
    printout either way."""
    body, _ = _request(body)
    return (
        _printout.render_refusal(
            body, _internal_error_reason(exc), basis=basis, dialect=dialect
        ),
        "",
    )


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

    Two additions ride the same bytes for the 4nec2 front end (momwire#1488),
    and neither changes a byte for a deck that does not use them: a body that
    starts with the stdin form's request line may end without ``EN``, and a
    deck that asks for a ``.GMO`` file gets it framed ahead of the printout
    (:func:`framed`), which the launcher writes beside the printout.

    The :class:`~momwire.serve.Seam` invariant — ``answer`` never raises —
    is kept the same way :func:`main` keeps exit 0: anything
    :func:`render`'s three gates did not foresee comes back as the
    ``NEC ERROR`` printout :func:`_internal_error_reason` frames, never as
    an exception the transport has no channel for.
    """

    def answer(body: str, terminator: str) -> tuple[str, str]:
        timed = body.startswith(_TIMES_REQUEST)
        if timed:
            # MOMWIRE_TIMES on the launcher's side: this answer timed, its
            # `times:` line framed ahead of the rest
            body = body[len(_TIMES_REQUEST) :]
            with _times.answering("resident") as times:
                answered = _answer_body(body)
            return f"{_TIMES_FRAME_START}{times.line()}{_TIMES_FRAME_END}{answered}", ""
        return _answer_body(body), ""

    def _answer_body(body: str) -> str:
        try:
            text, en_optional = _request(body)
            printout, gmo = render_files(
                text, basis=basis, dialect=dialect, en_optional=en_optional
            )
            return framed(printout, gmo)
        except Exception as exc:  # noqa: BLE001 - the seam's last line of defence
            return seam_failure(body, exc, basis=basis, dialect=dialect)[0]

    return Seam(
        name="eznec",
        greeting=lambda: [],
        terminators=frozenset(),
        closing=frozenset(),
        eof_terminator="",
        answer=answer,
    )
