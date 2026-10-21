"""EZNEC's External NEC-4.2 slot: the ``nec4`` dialect behind the same shell.

momwire#1295.  A program NAME carrying ``nec4``
(:func:`~momwire.deck._solver.dialect_from_program_name`) makes the one-shot
shell answer EZNEC's NEC-4.2 slot instead of its NEC-5 one.  The protocol is
the NEC-5 slot's exactly, as captured in 0223-0239: ``"<engine>.exe" "EZ.NEC"
"NEC.OUT"``, two quoted cwd-relative paths, cwd the engine's folder, no stdin,
exit status unread.  Only the deck dialect and the printout differ, and both
are decided here.

The deck is read by :func:`momwire.deck._nec4.parse_nec4`, which addresses a
source by segment CENTRE — so the slot's roster is
:data:`~momwire.deck.NEC2_BASES`, the families whose ``centre_feeds``
capability holds.  A basis outside it (razor-2p, razor-nec5) is refused BY
NAME in the printout before any deck is read: razor places a gap only on a
knot, and moving a centre feed half a cell silently is momwire#821's failure.

The solve is the portal's :class:`~momwire.portal._portal.DeckSolver`, the
one that already serves centre-addressed decks; :func:`solve` turns its
records into a :class:`~momwire.eznec._nec4_printout.Nec4Printout` and
:mod:`._nec4_printout` lays that out in the licensed NEC-4.2's columns.

What the printout serves is what the captured NEC-4.2 printouts show.  A deck
shape none of them shows is refused BY NAME rather than printed in an
invented layout (:func:`_served_shape`): EZNEC finds sections by heading, so
a guessed heading is a silent misread where a refusal is a sentence on
screen.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace

import numpy as np

from ..deck import BASES, NEC2_BASES
from ..deck._nec5 import Nec5Structure, ends_on_nx, split_structures
from . import _printout
from ._nec4_printout import (
    FarFieldGround,
    Nec4Printout,
    Nec4Report,
    Nec4Run,
    Preamble,
    SegmentRow,
    WireRow,
    render_nec4_printout,
    value_row,
)
from ._printout import (
    ENVIRONMENT_FREE_SPACE,
    ENVIRONMENT_PERFECT_GROUND,
    GroundMedium,
    LineRow,
    LoadRow,
    NearFieldBlock,
    NetworkRow,
    PortRow,
    PowerBudget,
)
from .._vec3 import norm3

__all__ = ["Nec4Refusal", "basis_refusal", "render", "solve"]

# NEC's metre-megahertz product, which NEC-4.2 prints its WAVELENGTH with:
# 7.15 MHz prints 4.1930E+01 (299.8/7.15 = 41.9301; the SI c gives 41.9290),
# the NEC-5 seam's measurement repeated on this engine (0234, 0235).  The
# wavelength-normalised columns of the current and charge tables use it too.
_C_MHZ_M = 299.8

# NEC-4.2's COMPLEX DIELECTRIC CONSTANT is eps_r - j*sigma/(omega*eps0) with
# omega = 2*pi*f at the deck's frequency and eps0 = 8.854E-12: 13 and
# 0.005 S/m at 299.7925 MHz print 1.30000E+01-2.99799E-01 (0231, 0232, 0239),
# where NEC-5's 59.96*lambda*sigma would print -2.99808E-01 and the
# full-precision eps0 -2.99792E-01; at 7.15 MHz the same reading prints
# -1.25703E+01, which is what p7 prints.
_EPS0_PRINTED = 8.854e-12

# The cards a captured or probed NEC-4.2 printout shows, after GE.  Everything else the
# nec4 dialect reads (it inherits nec2's vocabulary) has no measured layout.
_SERVED_CARDS = frozenset(
    {"GN", "GD", "FR", "EX", "LD", "TL", "NT", "PQ", "RP", "NE", "NH", "XQ", "EN", "EK"}
)
_SERVED_GEOMETRY = frozenset({"GW", "GE"})

# The pattern request forms captured: XNDA 1000 (a cut) and 1001 (the 3-D
# grid, with its average-gain trailer), RP mode 0, at no range or at one
# (the range form, momwire#1463: `_at_range`).  1002 (the trailer and no
# rows), 1003 (1001's printout), 1010 and 1011 (DIRECTIVE gains in the rows,
# the trailer still a power gain) print as NEC-5's do for the same XNDA, in
# the same row builder (probes p33-p36, momwire#1592).
_SERVED_XNDA = frozenset({1000, 1001, 1002, 1003, 1010, 1011})

# A GD card's leading integer, as the one captured value prints it.
_GD_KIND = {2: "CIRCLE CLIFF"}
# The far-field readout's code for that cliff: the circular one, whose edge
# is a radius about the origin (`_far_readout._cliff_medium_2`).
_CIRCULAR_CLIFF = 3


class Nec4Refusal(Exception):
    """A deck this slot does not print: the message is the reason."""


def basis_refusal(basis: str) -> str | None:
    """The sentence refusing ``basis`` in this slot, or ``None`` if it serves.

    Two cases, told apart so the user can act on the right one: a name the
    roster does not know is a typo and the sentence lists the choices; a
    roster basis that cannot place a centre feed is refused in its own
    family's declared words.
    """
    if basis in NEC2_BASES:
        return None
    if basis in BASES:
        cls, _kwargs = BASES[basis]
        return (
            f"basis {basis!r} does not serve EZNEC's NEC-4.2 slot, which "
            f"addresses sources at segment centres: "
            f"{cls.capabilities.refusal('centre_feeds')}"
        )
    known = ", ".join(repr(name) for name in NEC2_BASES)
    return f"unknown basis {basis!r} for EZNEC's NEC-4.2 slot; known bases: {known}"


def render(text: str, *, basis: str) -> str:
    """One NEC-4.2-dialect deck as a printout: the answer, or a named refusal.

    A refusal is the NEC-5 slot's convention under the NEC-4.2 banner: the
    header with its comment box, so EZNEC accepts the file as this run's, and
    one ``NEC ERROR`` line naming the reason where the results would be.  One
    refused step refuses the deck: a partial printout would read as a
    complete one.
    """
    refused = basis_refusal(basis)
    if refused is not None:
        return _printout.render_refusal(text, refused, basis=basis, dialect="nec4")
    structures = split_structures(text)
    if structures is not None:
        return _render_chain(structures, basis=basis)
    printout, _seconds = render_structure(text, basis=basis)
    return printout


def render_structure(
    text: str,
    *,
    basis: str,
    echo_text: str | None = None,
    closing: bool = True,
    run_seconds: float = 0.0,
) -> tuple[str, float | None]:
    """One structure's printout and the seconds its solve took, ``None`` for
    a refusal (which ends a chain).

    ``text`` is SOLVED; ``echo_text`` is what the printout echoes when it is
    not the same (a structure an ``NX`` closed is solved as the standalone
    deck its body makes and echoed with the ``NX`` it arrived with, as the
    NEC-5 slot's chain is, momwire#1592).  ``closing=False`` leaves out the
    ``RUN TIME`` trailer; with it, ``run_seconds`` is added to this
    structure's own, so the last structure of a chain carries the whole
    chain's.
    """
    echo = text if echo_text is None else echo_text
    refused = basis_refusal(basis)
    if refused is not None:
        return _printout.render_refusal(
            echo, refused, basis=basis, dialect="nec4"
        ), None
    try:
        data = solve(text, basis=basis)
    except Nec4Refusal as exc:
        return _printout.render_refusal(
            echo, str(exc), basis=basis, dialect="nec4"
        ), None
    seconds = data.run_seconds
    data = replace(data, run_seconds=seconds + run_seconds)
    return render_nec4_printout(echo, data, basis=basis, closing=closing), seconds


def _render_chain(structures: list[Nec5Structure], *, basis: str) -> str:
    """An ``NX``-chained deck's printout (momwire#1592): each structure's own
    printout in turn, as the NEC-5 slot lays a chain out
    (:func:`momwire.eznec._shell._render_chain`).

    SimNEC's batch and stream modes queue several runs into one engine call
    with ``NX`` between them, and read each run's printout up to its ``NX``
    echo.  Every structure starts from scratch, so each is solved as the
    standalone deck its body makes; each prints in full, header included,
    ends on its ``NX`` echo, and only the last carries ``RUN TIME`` (the
    chain's).  A structure that refuses ends the chain there with its own
    refusal printout, after the structures that solved.

    No NEC-4.2 printout of a chain has been captured: the layout between
    structures is the NEC-5 one, and inside each structure it is this
    slot's own.  Before momwire#1592 the whole chain was handed to the
    one-deck path, whose reader stops at the first ``NX`` while its echo
    counted every execute card up to ``EN``, and the deck refused as
    ``N execute cards against 1 execute runs``.
    """
    printouts: list[str] = []
    seconds = 0.0
    for k, structure in enumerate(structures):
        last = k == len(structures) - 1 and not ends_on_nx(structure)
        printout, took = render_structure(
            structure.deck_text,
            basis=basis,
            echo_text=structure.echo_text,
            closing=last,
            run_seconds=seconds,
        )
        printouts.append(printout)
        if took is None:
            break
        seconds += took
    return "".join(printouts)


def _unmeasured(what: str) -> str:
    return (
        f"{what} is not served by this engine's NEC-4.2 printout: no captured "
        f"NEC-4.2 printout shows how it prints"
    )


_EXECUTE = frozenset({"RP", "XQ", "NE", "NH"})
_REPORTS = frozenset({"RP", "NE", "NH"})


def _further_tables(text: str, deck) -> tuple[list[int | None], dict[int, list]]:
    """Where each run's cards sit in the echo, and the tables asked of a run
    after it ran.

    Returns ``(echo_at, further)``.  ``echo_at`` has one entry per entry of
    ``deck.groups``: the echo number of its execute card, ``None`` for an
    idle one.  ``further`` maps a run's group index to the ``(echo number,
    card)`` of every ``RP``/``NE``/``NH`` card after it that the deck reader
    found nothing new for (its ``None`` groups): the licensed engine prints
    one more table of the same run for each, in card order (momwire#1463,
    p20-p28).  An idle ``XQ`` is not among them: it prints nothing (p27), so
    its echo joins the cards echoed in front of whatever follows.

    Until momwire#1463 every idle card was skipped, an ``RP`` or ``NE``
    included, and its table was missing from the printout with nothing said.
    """
    numbered = [
        (i, card)
        for i, card in enumerate(_printout._post_ge_cards(text), start=1)
        if card.mnemonic in _EXECUTE
    ]
    if len(numbered) != len(deck.groups):
        raise Nec4Refusal(
            f"{len(numbered)} execute cards against {len(deck.groups)} execute "
            f"runs: this engine cannot place the card echo"
        )
    echo_at: list[int | None] = []
    further: dict[int, list] = {}
    run = None
    for index, ((number, card), group) in enumerate(
        zip(numbered, deck.groups, strict=True)
    ):
        if group is not None:
            run = index
            echo_at.append(number)
            continue
        echo_at.append(None)
        if card.mnemonic in _REPORTS:
            # The first execute card of a deck always runs, so a run stands
            # in front of every idle card.
            further.setdefault(run, []).append((number, card))
    return echo_at, further


def _served_table(card) -> None:
    """:class:`Nec4Refusal` for an ``RP`` form no NEC-4.2 printout shows.

    Asked of every ``RP`` the printout answers, the card that ran and any
    after it alike.  ``NE``/``NH`` pass: rectangular and spherical grids,
    free space, a perfect ground and the Sommerfeld one (GN 2 and GN 3
    alike) each print in one layout (probes p4 and p11-p15, momwire#1352).
    The dialect refuses every other coordinate system before this is asked;
    the cells a finite ground cannot answer refuse in :func:`_near_field`,
    which owns the grid.
    """
    if card.mnemonic != "RP":
        return
    if card.i(0) != 0:
        raise Nec4Refusal(_unmeasured(f"RP mode {card.i(0)}"))
    if card.i(3) not in _SERVED_XNDA:
        raise Nec4Refusal(_unmeasured(f"RP with XNDA {card.i(3)}"))
    if card.f(8) < 0.0:
        raise Nec4Refusal(_unmeasured("RP at a negative range"))


def _served_shape(text: str, deck, echo_at, further) -> list[tuple[int, object]]:
    """``(group index, group)`` for every run the deck executes, or
    :class:`Nec4Refusal` for a shape no NEC-4.2 printout shows.

    Two multi-run shapes are measured: one execute card over an ``FR`` sweep
    (p2) and several execute cards at one frequency, each after the first
    changing only the sources (p8).  A later run that refills the matrix (a
    new ``FR``, ``GN`` or load between execute cards), or a sweep with more
    than one execute card, is unmeasured.

    A further ``RP``/``NE``/``NH`` card of a run (:func:`_further_tables`) is
    measured at one frequency (p20-p28).  After a sweep it is not, and
    neither is one that follows a ``GD`` card read since the run: the
    licensed engine prints a FREQUENCY and an ANTENNA ENVIRONMENT block
    there and no input parameters (p32's second pattern, kept out of the
    fixtures), a layout this printout does not lay out.  Both refuse by
    name: a table left out with nothing said is momwire#1463's fault.
    """
    for card in deck.geometry:
        if card.mnemonic not in _SERVED_GEOMETRY:
            raise Nec4Refusal(_unmeasured(f"the {card.mnemonic} geometry card"))
    for card in deck.data_cards:
        if card.mnemonic not in _SERVED_CARDS:
            raise Nec4Refusal(_unmeasured(f"the {card.mnemonic} card"))
        if card.mnemonic == "GD" and card.i(0) not in _GD_KIND:
            raise Nec4Refusal(_unmeasured(f"GD {card.i(0)}"))
    runs = [(i, g) for i, g in enumerate(deck.groups) if g is not None]
    if not runs:
        raise Nec4Refusal(_unmeasured("a deck with no execute card"))
    if len(runs) > 1:
        if any(len(g.freqs_mhz) != 1 for _i, g in runs):
            raise Nec4Refusal(
                _unmeasured("several execute cards over a frequency sweep")
            )
        if any(g.refilled or g.refilled_partial for _i, g in runs[1:]):
            raise Nec4Refusal(
                _unmeasured("a later execute card after a card that refills the matrix")
            )
    gd_at = [
        i
        for i, card in enumerate(_printout._post_ge_cards(text), start=1)
        if card.mnemonic == "GD"
    ]
    for index, group in runs:
        if group.ground.kind == "refl":
            raise Nec4Refusal(_unmeasured("the GN 0 reflection-coefficient ground"))
        if group.pq is not None and group.pq.restricted:
            raise Nec4Refusal(_unmeasured("a PQ card restricted to a segment range"))
        if group.report is not None:
            _served_table(group.report)
        for number, card in further.get(index, ()):
            if len(group.freqs_mhz) != 1:
                raise Nec4Refusal(
                    _unmeasured(
                        f"a further {card.mnemonic} card after a frequency sweep's "
                        f"execute card"
                    )
                )
            if any(echo_at[index] < at < number for at in gd_at):
                raise Nec4Refusal(
                    _unmeasured(
                        f"a further {card.mnemonic} card after a GD card read "
                        f"since the run it follows"
                    )
                )
            _served_table(card)
    return runs


def solve(text: str, *, basis: str) -> Nec4Printout:
    """Solve one deck and return every number its NEC-4.2 printout prints.

    Every refusal on the way — the dialect's, the solver's, a basis that
    cannot host this deck, a shape no printout shows — is a
    :class:`Nec4Refusal` carrying its sentence.
    """
    # Imported on use: the NEC-5 slot's one-shot never pays for the portal.
    from ..portal._portal import (
        _DECK_REFUSALS,
        DeckSolver,
        PortalError,
        _run_records,
        _segment_end_nodes,
        parse_deck,
    )

    started = time.perf_counter()
    try:
        deck = parse_deck(text, dialect="nec4")
    except _DECK_REFUSALS as exc:
        raise Nec4Refusal(str(exc)) from exc
    echo_at, further = _further_tables(text, deck)
    plan = _served_shape(text, deck, echo_at, further)
    runs: list[Nec4Run | Nec4Report] = []
    try:
        solver = DeckSolver(deck, basis=basis)
        connections = _connections(solver, deck)
        for count, (index, group) in enumerate(plan):
            for step, freq in enumerate(group.freqs_mhz):
                records = _run_records(solver, group, freq, index)
                runs.append(
                    _run(
                        deck,
                        solver,
                        group,
                        index,
                        freq,
                        records,
                        connections,
                        echo_through=echo_at[index] if step == 0 else None,
                        preamble=count == 0,
                    )
                )
            # One frequency here (`_served_shape`), so `records` is the run
            # every further table is read from.
            for number, card in further.get(index, ()):
                runs.append(_report(deck, solver, group, freq, records, card, number))
    except (
        PortalError,
        ValueError,
        NotImplementedError,
        np.linalg.LinAlgError,
    ) as exc:
        raise Nec4Refusal(str(exc)) from exc

    ends, order = _segment_end_nodes(solver.wires)
    return Nec4Printout(
        wires=_wire_rows(deck),
        ge_flag=_ge_flag(deck),
        total_segments=solver.n_segments,
        junctions=tuple(
            tuple(sorted(ends[key], key=abs)) for key in order if len(ends[key]) >= 3
        ),
        segments=tuple(
            _segment_row(seg, i_minus, i_plus)
            for seg, (i_minus, i_plus) in zip(solver.segments, connections, strict=True)
        ),
        runs=tuple(runs),
        run_seconds=time.perf_counter() - started,
    )


def _run(
    deck,
    solver,
    group,
    index: int,
    freq: float,
    records,
    connections,
    *,
    echo_through: int | None,
    preamble: bool,
) -> Nec4Run:
    """One run's numbers: one frequency of one execute card."""
    result = records.result
    wavelength = _C_MHZ_M / freq
    medium = _medium(group.ground, freq)
    local = _local_wavelength(wavelength, medium, result.ground_z)
    networks, excitation = _networks(solver, result, index)
    gd = _far_ground_card(deck)
    second = group.second_medium
    patterns, near = _tables(group.report, result, solver, freq, gd, medium)
    charges = (
        _charge_rows(_charges(solver, result, connections, freq), solver, local)
        if group.pq is not None and group.pq.prints
        else None
    )
    # At a port the printed current is the STRUCTURE current at the gap,
    # as NEC-4.2 prints it: exactly the set current on an EX 6 feed (0223's
    # segment 6 reads 1.4142E+00 where the basis's own mid-element value is
    # 1.411-0.018j), the source current on an EX 0 feed (0224), and the
    # current INTO the structure at a network point, not the source's (0234's
    # segment 61 reads 4.4E-09 A under a 1.414 A source).
    gap_current = {segment: port for port, segment, _v in result.driven}
    gap_current.update({segment: port for _t, segment, port in result.network_points})
    currents = []
    for seg in solver.segments:
        port = gap_current.get(seg.number)
        value = (
            complex(result.i_port[port])
            if port is not None
            else complex(result.segment_currents[seg.number - 1])
        )
        currents.append(
            value_row(
                seg.number,
                seg.tag,
                tuple(seg.centre / local(seg.centre)),
                float(norm3(seg.direction)) / local(seg.centre),
                value,
            )
        )
    return Nec4Run(
        echo_through=echo_through,
        preamble=(
            Preamble(
                frequency_mhz=freq,
                wavelength_m=wavelength,
                environment=_environment_label(deck, group.ground),
                ground=medium,
                ground_file=_ground_file(deck) if medium is not None else None,
                # The cards in force at THIS execute card (momwire#1069), not
                # the deck's last: an `LD -1` after it does not unload it.
                loads=tuple(_load_row(card) for card in group.loads),
                fill_seconds=result.fill_ms / 1000.0,
                factor_seconds=0.0,
            )
            if preamble
            else None
        ),
        networks=networks if preamble else (),
        network_excitation=excitation,
        sources=tuple(
            PortRow(
                row.tag,
                row.segment,
                0,
                row.volts,
                row.current,
                row.impedance,
                row.admittance,
                row.power,
            )
            for row in records.aip_rows
        ),
        currents=tuple(currents),
        charges=charges,
        power=PowerBudget(
            input_power=result.p_in,
            radiated_power=result.p_radiated,
            wire_loss=result.p_structure,
            efficiency_percent=result.efficiency,
            # Printed whenever the deck carries a network: 0234 and 0235 both
            # print it, 0235's at 1.5987E-14 W, so NEC-4.2 does not gate the
            # line on the loss's size or sign the way NEC-5 does.
            network_loss=result.p_network if excitation else None,
        ),
        near_fields=near,
        far_field_ground=_far_field_ground(gd, second),
        patterns=patterns,
    )


def _tables(card, result, solver, freq: float, gd, medium):
    """``(patterns, near fields)`` one ``RP``/``NE``/``NH`` card asks of a
    solved run; both empty for an ``XQ`` (``card`` is ``None``)."""
    if card is None:
        return (), ()
    if card.mnemonic == "RP":
        return (_pattern(card, result, freq, gd, medium),), ()
    return (), (_near_field(card, result, solver),)


def _far_field_ground(gd, second) -> FarFieldGround | None:
    if gd is None or second is None:
        return None
    return FarFieldGround(
        _GD_KIND[gd.i(0)],
        second.edge_distance,
        second.height,
        second.eps_r2,
        second.sigma2,
    )


def _report(deck, solver, group, freq: float, records, card, number: int) -> Nec4Report:
    """One further table of a run already solved (:func:`_further_tables`):
    nothing is solved again, the card reads the run's own currents."""
    result = records.result
    gd = _far_ground_card(deck)
    patterns, near = _tables(
        card, result, solver, freq, gd, _medium(group.ground, freq)
    )
    return Nec4Report(
        echo_through=number,
        near_fields=near,
        far_field_ground=_far_field_ground(gd, group.second_medium),
        patterns=patterns,
    )


def _connections(solver, deck) -> list[tuple[int, int]]:
    """The segmentation table's ``(I-, I+)`` columns.

    The portal's reading (the neighbour across each end, the segment's own
    number at a ``GE 1`` plane) with NEC-4.2's rule at a JUNCTION: each end
    names the NEXT member of the junction after itself, cyclically, in the
    junction row's order (p1: junction ``6 -7 -13`` prints 7 after 6, -13
    before 7 and 6 before 13), where the portal names the first other one.
    """
    from ..portal._portal import _connection_data, _segment_end_nodes

    connections = _connection_data(solver.wires, deck.ground_plane_interpolates)
    ends, _order = _segment_end_nodes(solver.wires)
    for members in ends.values():
        if len(members) < 3:
            continue
        ring = sorted(members, key=abs)
        for k, member in enumerate(ring):
            following = ring[(k + 1) % len(ring)]
            number = abs(member)
            i_minus, i_plus = connections[number - 1]
            if member < 0:
                # This node is the segment's END 1: the I- column.
                i_minus = following
            else:
                i_plus = -following
            connections[number - 1] = (i_minus, i_plus)
    return connections


# --------------------------------------------------------------------------
# the structure
# --------------------------------------------------------------------------


def _wire_rows(deck) -> tuple[WireRow, ...]:
    rows = []
    first = 1
    gw = [c for c in deck.geometry if c.mnemonic == "GW"]
    for number, card in enumerate(gw, start=1):
        n_seg = card.i(1)
        rows.append(
            WireRow(
                number,
                (card.f(2), card.f(3), card.f(4)),
                (card.f(5), card.f(6), card.f(7)),
                card.f(8),
                n_seg,
                first,
                first + n_seg - 1,
                card.i(0),
            )
        )
        first += n_seg
    return tuple(rows)


def _ge_flag(deck) -> int:
    for card in deck.geometry:
        if card.mnemonic == "GE":
            return card.i(0)
    return 0


def _segment_row(seg, i_minus: int, i_plus: int) -> SegmentRow:
    d = seg.direction
    return SegmentRow(
        seg.number,
        tuple(float(c) for c in seg.centre),
        float(norm3(d)),
        math.degrees(math.atan2(d[2], math.hypot(d[0], d[1]))),
        math.degrees(math.atan2(d[1], d[0])),
        seg.radius,
        i_minus,
        i_plus,
        seg.tag,
    )


# --------------------------------------------------------------------------
# the environment and the loads
# --------------------------------------------------------------------------


def _environment_label(deck, ground) -> str:
    """The environment line.  A Sommerfeld ground names the ``GN`` type the
    deck wrote, ``GN2`` or ``GN3`` (0231 and 0232 differ in that word alone),
    although the nec4 dialect solves both as one half-space."""
    if ground.kind == "free":
        return ENVIRONMENT_FREE_SPACE
    if ground.kind == "pec":
        return ENVIRONMENT_PERFECT_GROUND
    written = [c.i(0) for c in deck.data_cards if c.mnemonic == "GN"]
    return f"FINITE GROUND.  SOMMERFELD SOLUTION GN{written[-1]}"


def _ground_file(deck) -> str | None:
    """The Sommerfeld-table file name the deck's ``GN`` card carries, or
    ``None``.  The nec4 dialect reads the card without it, on every GN type
    (momwire#1317, momwire#1443); the caller asks only under a Sommerfeld
    ground (``GN 2``/``GN 3``), the one place a table file means anything, and
    the printout says the file was not read, as the licensed engine does on
    ``GN 2``.  Under any other ground the name is only echoed on the card's
    continuation line (``_printout._card_echo``)."""
    written = [c for c in deck.data_cards if c.mnemonic == "GN"]
    return written[-1].trailer if written else None


def _medium(ground, freq_mhz: float) -> GroundMedium | None:
    if ground.kind != "sommerfeld":
        return None
    omega = 2.0 * math.pi * freq_mhz * 1e6
    eps_c = complex(ground.eps_r, -ground.sigma / (omega * _EPS0_PRINTED))
    return GroundMedium(ground.eps_r, ground.sigma, eps_c)


def _local_wavelength(wavelength: float, medium: GroundMedium | None, ground_z):
    """``point -> the wavelength its row is normalised by``.

    Below a finite ground's interface NEC-4.2 normalises by the MEDIUM's
    wavelength, 2*pi/|k|, as the table note says: 0239's wire at z = -1 m in
    13/0.005 soil prints Z = -3.6059 and a 0.04545 m segment 0.16391, the
    free-space values times |sqrt(eps_c)| = 3.6060 of the printed eps_c.
    The choice is per row, by which side of the interface the point is on:
    p9's vertical above a GN 2 interface prints free-space lengths and its
    buried wire medium ones, in one table.
    """
    if medium is None:
        return lambda _point: wavelength
    below = wavelength / abs(np.sqrt(medium.eps_c))
    interface = ground_z or 0.0
    return lambda point: below if float(point[2]) < interface else wavelength


def _load_row(card) -> LoadRow:
    """One ``LD`` card's row.  An omitted THRU is FROM (``LD 4,1,3,0`` prints
    ``1 3 3``, 0225); a zero FROM is the whole tag and stays zero."""
    kind = card.i(0)
    tag, first, last = card.i(1), card.i(2), card.i(3)
    if last == 0:
        last = first
    if kind == 4:
        reactance = card.f(5)
        return LoadRow(
            tag,
            first,
            last,
            resistance=card.f(4),
            reactance=reactance if reactance != 0.0 else None,
        )
    if kind == 5:
        return LoadRow(tag, first, last, conductivity=card.f(4), kind="WIRE")
    names = {
        0: "SERIES",
        1: "PARALLEL",
        2: "SERIES (PER METER)",
        3: "PARALLEL (PER METER)",
    }
    return LoadRow(
        tag,
        first,
        last,
        resistance=card.f(4) or None,
        inductance=card.f(5) or None,
        capacitance=card.f(6) or None,
        kind=names[kind],
    )


# --------------------------------------------------------------------------
# networks
# --------------------------------------------------------------------------


def _networks(solver, result, index: int):
    """``(NETWORK DATA rows, excitation rows)`` for the run.

    The excitation rows are in DISCOVERY order — each card's end A, then its
    end B, cards in deck order, a segment named twice printed once — except
    that a point driven by a voltage source moves to the end.  The portal's
    nec2c order moves EVERY driven point to the end, which 0234's EX 6 point,
    printed first, contradicts.
    """
    from ..deck._networks import card_branches, live_cards
    from ..portal._portal import _PRINTED_DUST_FLOOR2

    rows: list[NetworkRow | LineRow] = []
    seen: list[int] = []
    for card, _pair in live_cards(solver.model, solver.plan, index):
        ends = []
        for tag, seg in (card.address_a, card.address_b):
            segment = solver.global_segment(*solver.structure.locate(tag, seg))
            ends += [solver.segments[segment - 1].tag, segment]
            if segment not in seen:
                seen.append(segment)
        f1, f2, f3, f4, f5, f6 = card.payload
        if card.kind == "NT":
            rows.append(
                NetworkRow(*ends, complex(f1, f2), complex(f3, f4), complex(f5, f6))
            )
            continue
        line = card_branches(card, 0, 1, solver.model.wires)[0]
        rows.append(
            LineRow(
                *ends,
                z0=line.z0,
                length_m=line.length,
                shunt_a=complex(f3, f4),
                shunt_b=complex(f5, f6),
                crossed=line.transposed,
            )
        )
    by_segment = {segment: (tag, port) for tag, segment, port in result.network_points}
    # A point a VOLTAGE source drives prints after the rest (p6: the EX 0
    # point last, though its cards name it first); a current source's point
    # keeps its place (0234 prints its EX 6 point first).
    voltage_driven = {
        segment
        for port, segment, _v in result.driven
        if port not in solver._current_ports
    }
    seen = [g for g in seen if g not in voltage_driven] + [
        g for g in seen if g in voltage_driven
    ]
    excitation = []
    for segment in seen:
        if segment not in by_segment:
            continue
        tag, port = by_segment[segment]
        volts = complex(result.v_applied[port])
        current = complex(result.i_port[port])
        # An open point's analytically-zero current prints as zero, not as
        # the solve's rounding residue (the portal's momwire#403 floor).
        if current.real**2 + current.imag**2 <= _PRINTED_DUST_FLOOR2:
            current = 0j
        excitation.append(
            PortRow(
                tag,
                segment,
                0,
                volts,
                current,
                volts / current if current != 0 else 0j,
                current / volts if volts != 0 else 0j,
                0.5 * (volts * current.conjugate()).real,
            )
        )
    return tuple(rows), tuple(excitation)


# --------------------------------------------------------------------------
# charges
# --------------------------------------------------------------------------


def _charges(solver, result, connections, freq_mhz: float):
    """``(per-segment charges, {(wire index, at start): end charge})``.

    The end charge is ``q = -(1/jw)*dI/ds`` read AT a free wire end, the
    centre rows' own formula moved to the end point, for the charge table's
    ``E`` rows (:class:`~._nec4_printout.SegmentValueRow`).  A wire end is
    free when its connection column is 0: no other segment and no ``GE 1``
    plane meets it.  The end is found on momwire's own polylines by
    position, since chaining may merge or reverse the deck's wires; a charge
    is a scalar, so a reversed walk does not change it.
    """
    from ..portal._portal import _wire_arc_at_knot

    momwire = result.solver
    omega = 2.0 * math.pi * freq_mhz * 1e6
    polyline_ends = [
        (
            np.asarray(polyline[0], float),
            np.asarray(polyline[-1], float),
            float(_wire_arc_at_knot(momwire, w_idx)[-1]),
        )
        for w_idx, polyline in enumerate(momwire.wires_polylines)
    ]
    positions: list[list[float]] = [[] for _ in polyline_ends]
    slots: dict[tuple[int, bool], tuple[int, int]] = {}
    first = 0
    for wi, wire in enumerate(solver.wires):
        last = first + wire.n_seg - 1
        for at_start, point, free in (
            (True, np.asarray(wire.p1, float), connections[first][0] == 0),
            (False, np.asarray(wire.p2, float), connections[last][1] == 0),
        ):
            if not free:
                continue
            tol = max(float(norm3(point)), 1.0) * 1e-9
            for w_idx, (start, end, length) in enumerate(polyline_ends):
                if norm3(start - point) <= tol:
                    s = 0.0
                elif norm3(end - point) <= tol:
                    s = length
                else:
                    continue
                slots[(wi, at_start)] = (w_idx, len(positions[w_idx]))
                positions[w_idx].append(s)
                break
        first = last + 1
    slopes = (
        momwire.current_slopes(result.coeffs, positions, _solved=True)  # #1042
        if slots
        else []
    )
    end_charges = {
        key: complex(-slopes[w_idx][k] / (1j * omega))
        for key, (w_idx, k) in slots.items()
    }
    return result.segment_charges, end_charges


def _charge_rows(charges, solver, local):
    centre_charges, end_charges = charges
    rows = []
    first = 0
    for wi, wire in enumerate(solver.wires):
        segs = solver.segments[first : first + wire.n_seg]
        length = float(norm3(segs[0].direction))
        start_end = (
            (True, segs[0], wire.p1),
            (False, segs[-1], wire.p2),
        )
        if (wi, True) in end_charges:
            _at, seg, point = start_end[0]
            rows.append(
                value_row(
                    seg.number,
                    seg.tag,
                    tuple(np.asarray(point, float) / local(point)),
                    length / local(point),
                    end_charges[(wi, True)],
                    end=True,
                )
            )
        rows += [
            value_row(
                seg.number,
                seg.tag,
                tuple(seg.centre / local(seg.centre)),
                length / local(seg.centre),
                complex(centre_charges[seg.number - 1]),
            )
            for seg in segs
        ]
        if (wi, False) in end_charges:
            _at, seg, point = start_end[1]
            rows.append(
                value_row(
                    seg.number,
                    seg.tag,
                    tuple(np.asarray(point, float) / local(point)),
                    length / local(point),
                    end_charges[(wi, False)],
                    end=True,
                )
            )
        first += wire.n_seg
    return tuple(rows)


# --------------------------------------------------------------------------
# the pattern
# --------------------------------------------------------------------------


def _far_ground_card(deck):
    gd = [c for c in deck.data_cards if c.mnemonic == "GD"]
    return gd[-1] if gd else None


def _pattern(card, result, freq_mhz: float, gd, medium: GroundMedium | None):
    """One ``RP 0`` card's answer, through the NEC-5 seam's row builder.

    Under a ``GD`` card the reflected far field takes the card's second
    medium: 0230 (``GN 1`` + ``GD``) prints 0231's finite-ground pattern
    (2.05 dB at theta 75, against 0229's perfect-ground 2.60 dB) while its
    impedance stays 0229's to every digit.  So ``GD`` is a far-field ground
    on EVERY ``RP`` here, not only on a cliff mode as in NEC-2, and its edge
    distance of 0 puts every specular point on the second medium.
    """
    from ..deck._nec5 import Nec5FarFieldRequest
    from ._serve import _pattern as pattern_block

    request = Nec5FarFieldRequest(
        n_theta=max(card.i(1), 1),
        n_phi=max(card.i(2), 1),
        xnda=card.i(3),
        theta0_deg=card.f(4),
        phi0_deg=card.f(5),
        d_theta_deg=card.f(6),
        d_phi_deg=card.f(7),
    )
    second = result.second_medium
    cliff = None
    if gd is not None and second is not None and result.ground.kind != "free":
        cliff = (_CIRCULAR_CLIFF, second)
    block = pattern_block(
        request,
        result.solver,
        result.coeffs,
        result.ground,
        freq_mhz,
        result.wavelength,
        result.p_in,
        ground_z=result.ground_z or 0.0,
        cliff=cliff,
        p_radiated=result.p_radiated,
    )
    range_m = card.f(8)
    if range_m == 0.0:
        return block
    return _at_range(block, range_m, freq_mhz, medium, result.ground.kind != "free")


def _at_range(
    block, range_m: float, freq_mhz: float, medium: GroundMedium | None, grounded: bool
):
    """A pattern as NEC-4.2 prints it for an ``RP`` with a range field
    (momwire#1463: SimNEC writes ``RP 0 19 73 1001 0 0 5 5 1000``).

    Read off the licensed binary's printouts of one deck at range 0 and at
    1000 m (p17 against p16 over GN 3, p19 against p18 in free space, and
    p29, p30, p31 for the other grounds), as black-box output:

    * the heading gains ``RANGE=`` and ``EXP(-JKR)/R=`` lines and the field
      columns read ``VOLTS/M``, the NEC-5 printout's range form to the byte;
    * every gain, the axial ratio, the tilt and the sense are the range-0
      row's, and so is the average-gain trailer;
    * each E magnitude is the range-0 one divided by the range, and each E
      phase is the range-0 one plus the heading's phase, in degrees and not
      wrapped again (p18 prints 293.42 for 179.10 + 114.32), so a component
      that is zero prints the bare shift;
    * the heading's phase is that of ``exp(-jkR)`` at the wavelength the
      printout itself prints, 299.8/f (7.1 MHz at 1000 m prints 114.32; the
      SI speed of light gives 114.10);
    * over a Sommerfeld ground, GN 2 and GN 3 alike, one more line gives the
      LOWER MEDIUM's ``exp(-jkR)/R`` at ``k*sqrt(eps_c)``, ``eps_c`` being
      the complex constant the environment block prints (2.23389-107 at
      -164.12 degrees for 13 / 0.005 S/m, every digit).  A perfect ground
      prints no such line, with or without ``GD`` (p29, p28);
    * a row below the horizon over a ground is all zeros as at range 0, and
      its two phases are the LOWER MEDIUM's phase over a Sommerfeld ground
      (p31) and 0.00 over a perfect one (p32).

    The row builder is asked for the range-0 block and the range is applied
    here rather than through its own ``range_m``: its wavenumber is the
    solve's, and the two phases printed here follow the printed wavelength.
    """
    k = 2.0 * math.pi * freq_mhz / _C_MHZ_M
    shift = _phase_of(-k * range_m)
    lower = {}
    below_shift = 0.0
    if medium is not None:
        k_lower = k * np.sqrt(medium.eps_c)
        below_shift = _phase_of(-k_lower.real * range_m)
        lower = {
            "lower_range_mag": math.exp(k_lower.imag * range_m) / range_m,
            "lower_range_phase_deg": below_shift,
        }
    rows = []
    for row in block.rows:
        below = grounded and row.theta_deg > 90.0 + 1e-9
        rows.append(
            row._replace(
                e_theta_magnitude=row.e_theta_magnitude / range_m,
                e_phi_magnitude=row.e_phi_magnitude / range_m,
                e_theta_phase_deg=below_shift
                if below
                else row.e_theta_phase_deg + shift,
                e_phi_phase_deg=below_shift if below else row.e_phi_phase_deg + shift,
            )
        )
    return replace(
        block, rows=tuple(rows), range_m=range_m, range_phase_deg=shift, **lower
    )


def _phase_of(radians: float) -> float:
    """The phase of ``exp(j*radians)`` in degrees, in (-180, 180]."""
    return math.degrees(math.atan2(math.sin(radians), math.cos(radians)))


# --------------------------------------------------------------------------
# the near field
# --------------------------------------------------------------------------


def _near_field(card, result, deck_solver) -> NearFieldBlock:
    """One ``NE``/``NH`` table over free space, a perfect ground or the
    Sommerfeld one.

    The table is NEC-5's layout to the byte (p4), and the readout is every
    seam's (:mod:`momwire._near_readout`): the element sum in mixed-potential
    form, plus the PEC image with its horizontal moments flipped and its
    charge negated, plus ``C₂·image + remainder`` over a finite ground.  The
    grid is :func:`~momwire._near_readout.grid_points`': X fastest, then Y,
    then Z (p4 steps Z alone), or for ``NE 1`` R fastest, then θ, then φ,
    each point printed at its Cartesian place under the same headings
    (p11-p15, momwire#1352).

    Over a finite ground the cells no composition answers — a point in the
    soil, a point on a wire's ground contact, any point of a deck with a
    buried wire — refuse with the nec2 portal's sentences
    (``_refuse_finite_ground_near_field``), the same geometry asked the same
    way.
    """
    from .._near_readout import (
        NEAR_FIELD_SUBDIV,
        grid_points,
        near_field_at,
        near_ground,
    )
    from ..portal._portal import _refuse_finite_ground_near_field
    from ._serve import near_field_block

    magnetic = card.mnemonic == "NH"
    points = grid_points(
        card.i(0),
        tuple(max(card.i(k), 1) for k in (1, 2, 3)),
        (card.f(4), card.f(5), card.f(6)),
        (card.f(7), card.f(8), card.f(9)),
    )
    solver = result.solver
    ground = near_ground(solver)
    if ground.kind == "compose":
        _refuse_finite_ground_near_field(card, deck_solver, points, ground.ground_z)
    k = 2.0 * math.pi / result.wavelength
    elements = solver.element_currents(result.coeffs, subdiv=NEAR_FIELD_SUBDIV)
    field = near_field_at(
        points,
        elements,
        k,
        deck_solver._smallest_radius,
        magnetic,
        ground,
        solver.omega,
    )
    return near_field_block(points, field, magnetic)
