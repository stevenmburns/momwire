"""The dialect-generic session loop — one executor for both serve seams.

momwire#719 U4 (phase 1 of the #718 consolidation): the portal and the eznec
front ends are two *seams* over one execution structure — read a deck, solve
it, write a report. The portal's is the GENERAL flow: a streamed session whose
``NX`` terminator returns a report mid-stream without closing the input, and
whose ``EN`` (or EOF) ends the run. The eznec contract — one deck in, one
printout out, connection closes — is the degenerate case: a session of
exactly one deck. This module owns the session shape; everything
dialect-specific (grammar, framing bytes, refusal prose, banners) lives
behind a :class:`Seam`.

The loop here is :func:`momwire.portal._portal.resident_loop`'s, moved
verbatim and parameterized — that function now delegates here, because the
only way to guarantee a served printout is byte-identical to the stock
daemon's is for it to be the same loop rather than a second copy of it
(the #379 rule, now applied across dialects rather than across processes).

The resident server the #718 arc builds (momwire#532: thin native client,
warm daemon, idle-timeout self-stop) hosts a :class:`Seam` per engine: a
line-stream connection runs :func:`run_session`; a one-shot connection (the
EZNEC leg) calls :meth:`Seam.answer` once and returns the bytes. Either way
the seam's contract, not the transport, decides what the host reads.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Seam:
    """One dialect front end, as the session loop sees it.

    The invariant every seam MUST keep: :attr:`answer` never raises.
    Refusals, parse failures, even internal errors ride the returned output
    in the dialect's own error frame (the portal's ``ERROR:`` lines under an
    unconditional sentinel echo; eznec's ``NEC ERROR`` line under a real
    header) — a host reads only that output, and an exception here is an
    answer the host never receives (SimNEC blocks in ``readLine()``; EZNEC
    blames the user's install).
    """

    # The dialect's name, as the resident server's registry will key it.
    name: str
    # Lines written once at session start, before any frame (the portal's
    # banner; empty for a seam whose output is a bare printout).
    greeting: Callable[[], list[str]]
    # First-token spellings that end a frame on a line stream. Empty for a
    # seam that is only ever served one-shot (the whole input is one body).
    terminators: frozenset[str]
    # Terminators that also end the SESSION after their frame is answered.
    closing: frozenset[str]
    # The terminator synthesized when the stream ends mid-body, or None to
    # discard an unterminated body. NEC's own card reader synthesizes EN at
    # EOF and runs what it has (#458); a dialect without that convention
    # passes None.
    eof_terminator: str | None
    # (body, terminator) -> (stdout chunk, stderr chunk). The whole answer
    # for one deck, error frames included; chunks are written verbatim.
    answer: Callable[[str, str], tuple[str, str]]


def run_session(seam: Seam, stdin, stdout, stderr, solve_lock=None) -> int:
    """The seam's greeting, then frames off ``stdin`` until it ends.

    Frames are delimited by a line whose first token is one of
    ``seam.terminators`` — no length prefix, no sentinel of our own — and the
    stream is never restarted between them. A terminator in ``seam.closing``
    answers its frame and then ends the run. A non-empty body still open when
    the stream ends is answered under ``seam.eof_terminator`` when the seam
    has one; a body that is empty or whitespace-only ends the run silently.

    ``solve_lock`` is a context manager held across each answer, and is how
    a resident server serialises frames arriving on different connections
    into one thread-pool budget; a single-stream caller passes nothing.
    """
    lock = contextlib.nullcontext() if solve_lock is None else solve_lock

    greeting = seam.greeting()
    if greeting:
        stdout.write("\n".join(greeting) + "\n")
        stdout.flush()

    def emit(body: str, terminator: str) -> None:
        with lock:
            out, err = seam.answer(body, terminator)
        stdout.write(out)
        stdout.flush()
        if err:
            stderr.write(err)
            stderr.flush()

    body: list[str] = []
    for line in stdin:
        head = line.strip().upper().split()[:1]
        if not head or head[0] not in seam.terminators:
            body.append(line.rstrip("\n"))
            continue
        emit("\n".join(body), head[0])
        if head[0] in seam.closing:
            return 0
        body = []
    if seam.eof_terminator is not None and any(ln.strip() for ln in body):
        emit("\n".join(body), seam.eof_terminator)
    return 0
