"""The ``nec4`` dialect front-end: the decks EZNEC's External NEC-4.2 slot writes.

momwire#1295.  EZNEC Pro+ v7, with its engine option set to External NEC-4.2,
writes a NEC-4.2-dialect deck and launches the engine as
``"<engine>.exe" "EZ.NEC" "NEC.OUT"`` with the engine's own folder as cwd.
:mod:`momwire.eznec` serves that slot when its executable's NAME carries
``nec4`` (:func:`~momwire.deck._solver.dialect_from_program_name`); this module
is the deck reader behind it.

Derived from CAPTURED INPUT ONLY: the seventeen decks EZNEC wrote in the
2026-10-03 capture sitting (antennaknobs ``scratch/eznec-capture/0223``-``0239``,
committed here under ``tests/fixtures/eznec_nec42/``) and the black-box solve
notes taken from them (prose and numbers, no printout).  No NEC-4.2 code,
manual text or printout is described or relied on, and none may be added: its
licence is own use at home.

NEC-4.2 reads NEC-2's cards and addresses a source by SEGMENT, whose centre is
the gap, so the dialect is the ``nec2`` one (:mod:`._nec2`) with the deltas the
captures show, and nothing else:

* ``EX 6`` is a CURRENT source, the card's F1/F2 its set current in amperes.
  Measured, not inferred from the card: on the captures carrying it the solved
  current comes back pinned at the card's 1.414214 A while the voltage floats,
  and 0223/0224 — one antenna, ``EX 6`` against ``EX 0`` — give the same
  impedance to every printed digit.  A sole current source is served as a
  voltage drive of the same port rescaled after the solve
  (:attr:`DeckModel.current_feeds`): the response is linear in the drive, so
  the impedance is the voltage drive's own and only the scale moves.  Several
  sources with any of them a current source — EZNEC's stock Cardioid in this
  slot writes two ``EX 6`` cards — is a PHASED drive and no scale serves it:
  the portal solves for the port voltages that deliver every set current
  while every ``EX 0`` keeps its set voltage (``DeckSolver._phased_drive``, the
  NEC-5 slot's ``EX 4`` algebra on the portal's port space).  One segment is
  still one port for the whole deck, so a segment driven both ways refuses.
* ``GE`` I1 reads -1 / 0 / 1, and -1 is NEC-4's "structure below ground"
  (0239 is a wholly buried wire under ``GE -1,-1``).  The ``nec2`` reading of
  the same sign — a ground plane WITHOUT the contact current expansion — is the
  part that carries over: a free wire end standing in the plane under ``GE -1``
  refuses exactly as it does there, and a buried or crossing structure is
  served by the solver's own below-interface machinery.  I2 is -1 on every
  capture; 0 (the omitted field) is accepted beside it, anything else refuses
  because no capture says what it means.
* ``GN 3`` is the Sommerfeld half-space, the same PHYSICAL ground as ``GN 2``:
  EZNEC offers the two as NEC-4.2's two ground methods, and they are two
  evaluations of one model rather than two models (the licensed engine gave
  0231/0232 the same impedance to every printed digit; antennaknobs measured
  them about an ohm apart on buried designs).  momwire has one evaluation of
  that half-space, so both cards map to ``("finite", eps_r, sigma)``.  The
  card as written still reaches the printout echo; only the solve is shared.
* ``GD`` accompanies ``GN 1`` (0230, 0234, 0235), EZNEC's MININEC-type ground.
  It is read exactly as the ``nec2`` dialect reads it — a second medium that
  only a cliff pattern (``RP 2``/``RP 3``) consults — so the currents are the
  perfect ground's, which is what the licensed engine answered (0230's
  impedance equals 0229's, the same ground with no ``GD``, to every digit).
  Its leading integer (2 on every capture) moves nothing here.
* A ``GN 2`` card's trailing non-numeric token is a Sommerfeld-table FILE
  name, ``NOFILE`` included (momwire#1317).  No EZNEC capture writes one; the
  licensed engine, given ``GN 2 0 0 0 13. .005 0 0 0 0 NOFILE`` as a black
  box, reported it could not open the file, computed the tables from the GN
  fields and solved.  The name is a cache, not physics, so it is accepted and
  ignored: the solve is the card's without it, and the printout echoes the
  name and says the file was not read.  On any other GN type a trailer still
  refuses, because no run shows what the licensed engine does with one there.
* ``EK`` is read and IGNORED, and every run solves with the extended kernel
  (momwire#1326, Steve's decision 2026-10-04).  NEC-4.2 has one thin-wire
  model and no extended-kernel switch: the binary, given ``EK 0`` as a black
  box (2026-10-04), echoes the card and prints ``THE EK AND KH COMMANDS HAVE
  NO EFFECT IN NEC-4``.  Against momwire's two kernels that model behaves
  as the extended one where it matters (a 28 MHz dipole ladder: at Delta/a
  1.5-2 the extended kernel lands up to 5 ohm nearer NEC-4.2, and on thin
  wire it costs 0.5-2 ohm, inside the model gap either way), so it is this
  dialect's DEFAULT (`DeckModel.extended_kernel_default`), and a basis or a
  deck that cannot take it falls back to the reduced kernel with an
  advisory rather than refusing.

Everything else the captures carry is a ``nec2`` card with the ``nec2``
meaning: ``GW``, ``LD 4`` and ``LD 5``, ``FR 0``, ``GN -1``/``1``/``2`` (``GN 1``
in both spellings EZNEC writes), ``TL``/``NT`` addressed by ``(tag, segment)``,
``RP`` (the mode-1001 3D grid and the negative theta step included), ``XQ`` and
``PQ``.  The ``nec2`` refusals stand for every card and form no capture shows.
"""

from __future__ import annotations

from dataclasses import replace

from ._cards import Card, DeckError
from ._nec2 import _Nec2Parser
from .model import DeckModel

__all__ = ["parse_nec4"]

# EX types this dialect drives.  NEC-4's other source types (plane waves, the
# current-slope-discontinuity voltage source) appear on no capture.
_EX_VOLTAGE = 0
_EX_CURRENT = 6

# GE I2 values a capture shows (-1), plus the omitted field.
_GE_I2 = frozenset({0, -1})


class _Nec4Parser(_Nec2Parser):
    """The ``nec2`` state machine with the NEC-4.2 deltas in the module
    docstring; every other card is read by the parent unchanged."""

    def __init__(self) -> None:
        super().__init__()
        # The extended kernel by default, and no card turns it off (module
        # docstring, momwire#1326).
        self._extended_kernel = True
        # (tag, seg) -> the source kind that card address was driven with.
        # A port's kind is fixed for the deck because the model's feed list is
        # one port set across every execute group.
        self._source_kind: dict[tuple[int, int], int] = {}

    def _geometry(self, card: Card) -> None:
        if card.mnemonic == "GE":
            if card.i(0) not in (-1, 0, 1):
                raise DeckError(
                    f"GE {card.i(0)} is not a NEC-4.2 ground flag; this engine's "
                    f"nec4 dialect reads GE -1 (structure below ground), GE 0 "
                    f"(no ground) and GE 1 (structure above ground)"
                )
            if card.i(1) not in _GE_I2:
                raise DeckError(
                    f"GE with a second field of {card.i(1)} is not served by "
                    f"this engine's nec4 dialect; EZNEC's NEC-4.2 slot writes -1 "
                    f"there, and no capture shows what any other value means"
                )
        super()._geometry(card)

    def _ek(self, card: Card) -> None:
        """Ignored: NEC-4.2's ``EK`` has no effect (module docstring), so
        the kernel stays the dialect's default whatever the card says.  The
        printout echoes the card and NEC-4.2's own line about it."""

    def _ld(self, card: Card) -> None:
        if card.i(0) in (2, 3) and card.f(6) != 0.0:
            # The nec2 reading folds this field with the segment length
            # (momwire#1091), measured on nec2c.  NEC-4.2 was not measured on
            # it and no capture carries one, so this dialect keeps the
            # refusal rather than assume NEC-2's scaling.
            raise DeckError(
                f"LD {card.i(0)} asks for a capacitance of {card.f(6):g} in its "
                f"per-unit-length RLC, which this engine's nec4 dialect does "
                f"not serve: NEC-2 scales that field by the segment length, and "
                f"no NEC-4.2 run shows whether NEC-4.2 does the same"
            )
        super()._ld(card)

    def _gn(self, card: Card) -> None:
        if card.trailer is not None:
            if card.i(0) != 2:
                raise DeckError(
                    f"GN {card.i(0)} carries a trailing token {card.trailer!r}; "
                    f"this engine's nec4 dialect reads a Sommerfeld-table file "
                    f"name on GN 2 only, the one ground type the licensed "
                    f"NEC-4.2 was observed reading it on"
                )
            # The Sommerfeld-table FILE (module docstring): a cache, so the
            # card is read without it.  The printout reads the name back off
            # the deck text (``eznec._nec4._ground_file``).
            card = Card(card.mnemonic, card.values, card.raw)
        if card.i(0) == 3:
            if card.i(1) != 0:
                raise DeckError(
                    f"GN 3 with a {card.i(1)}-wire radial ground screen is not "
                    f"supported by this engine"
                )
            # One physical ground, two NEC-4.2 evaluations of it (module
            # docstring): read as the GN 2 it shares a model with.
            card = Card(card.mnemonic, (2.0, *card.values[1:]), card.raw, card.trailer)
        super()._gn(card)

    def _ex(self, card: Card) -> None:
        ex_type = card.i(0)
        if ex_type not in (_EX_VOLTAGE, _EX_CURRENT):
            raise DeckError(
                f"EX type {ex_type} is not served by this engine's nec4 dialect, "
                f"which drives EX 0 (voltage source) and EX 6 (current source) "
                f"only"
            )
        address = (card.i(1), card.i(2))
        kind = self._source_kind.setdefault(address, ex_type)
        if kind != ex_type:
            raise DeckError(
                f"EX {ex_type} on tag {address[0]} segment {address[1]}, which an "
                f"earlier EX {kind} drove, is not supported by this engine: one "
                f"segment is one port for the whole deck, and a port is either a "
                f"voltage source or a current source"
            )
        if ex_type == _EX_VOLTAGE:
            super()._ex(card)
            return
        amps = complex(card.f(4), card.f(5))
        if amps == 0:
            raise DeckError(
                f"EX 6 on tag {address[0]} segment {address[1]} sets a current of "
                f"zero, which is an open circuit rather than a source; this "
                f"engine refuses it rather than guess a drive"
            )
        # The parent's retention rule, unchanged: the first EX after an
        # execute card replaces the list, every further one adds to it.
        self._saw_ex = True
        if self._sources_stale:
            self._sources = []
            self._sources_stale = False
        self._sources.append((address[0], address[1], amps))

    def model(self) -> DeckModel:
        model = super().model()
        # The parent's union port order (`_feeds_and_groups`): every group's
        # sources, first appearance first.
        order: list[tuple[int, int]] = []
        for pending in self.groups:
            if pending is None:
                continue
            for tag, seg, _v in pending.sources:
                if (tag, seg) not in order:
                    order.append((tag, seg))
        current = tuple(
            i
            for i, address in enumerate(order)
            if self._source_kind[address] == _EX_CURRENT
        )
        return replace(
            model,
            current_feeds=current if current else model.current_feeds,
            extended_kernel_default=True,
        )


def parse_nec4(text: str) -> DeckModel:
    """Parse one ``nec4`` deck body into a :class:`DeckModel`."""
    parser = _Nec4Parser()
    parser.feed(text)
    return parser.model()
