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

import numpy as np

from ..deck import BASES, NEC2_BASES
from . import _printout
from ._nec4_printout import (
    FarFieldGround,
    Nec4Printout,
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
    NearFieldRow,
    NetworkRow,
    PortRow,
    PowerBudget,
)

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
# grid, with its average-gain trailer), RP mode 0, no range.
_SERVED_XNDA = frozenset({1000, 1001})

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
    try:
        data = solve(text, basis=basis)
    except Nec4Refusal as exc:
        return _printout.render_refusal(text, str(exc), basis=basis, dialect="nec4")
    return render_nec4_printout(text, data, basis=basis)


def _unmeasured(what: str) -> str:
    return (
        f"{what} is not served by this engine's NEC-4.2 printout: no captured "
        f"NEC-4.2 printout shows how it prints"
    )


_EXECUTE = frozenset({"RP", "XQ", "NE", "NH"})


def _served_shape(deck) -> list[tuple[int, object]]:
    """``(group index, group)`` for every run the deck executes, or
    :class:`Nec4Refusal` for a shape no NEC-4.2 printout shows.

    Two multi-run shapes are measured: one execute card over an ``FR`` sweep
    (p2) and several execute cards at one frequency, each after the first
    changing only the sources (p8).  A later run that refills the matrix (a
    new ``FR``, ``GN`` or load between execute cards), or a sweep with more
    than one execute card, is unmeasured.
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
    for _index, group in runs:
        if group.ground.kind == "refl":
            raise Nec4Refusal(_unmeasured("the GN 0 reflection-coefficient ground"))
        if group.pq is not None and group.pq.restricted:
            raise Nec4Refusal(_unmeasured("a PQ card restricted to a segment range"))
        report = group.report
        if report is None:
            continue
        if report.mnemonic in ("NE", "NH"):
            if report.i(0) != 0:
                raise Nec4Refusal(_unmeasured(f"{report.mnemonic} {report.i(0)}"))
            if group.ground.kind not in ("free", "pec"):
                # The near field over a finite ground is a Sommerfeld
                # evaluation the portal's readout does not make.
                raise Nec4Refusal(
                    f"{report.mnemonic} over a finite ground is not served by this "
                    f"engine's NEC-4.2 slot"
                )
            continue
        if report.i(0) != 0:
            raise Nec4Refusal(_unmeasured(f"RP mode {report.i(0)}"))
        if report.i(3) not in _SERVED_XNDA:
            raise Nec4Refusal(_unmeasured(f"RP with XNDA {report.i(3)}"))
        if report.f(8) != 0.0:
            raise Nec4Refusal(_unmeasured("RP at a nonzero range"))
    return runs


def _execute_card_numbers(text: str, deck) -> list[int | None]:
    """Per entry of ``deck.groups``, the echo number of its execute card
    (``None`` for an idle one), numbered as the card echo numbers them."""
    numbers = [
        i
        for i, card in enumerate(_printout._post_ge_cards(text), start=1)
        if card.mnemonic in _EXECUTE
    ]
    if len(numbers) != len(deck.groups):
        raise Nec4Refusal(
            f"{len(numbers)} execute cards against {len(deck.groups)} execute "
            f"runs: this engine cannot place the card echo"
        )
    return [n if g is not None else None for n, g in zip(numbers, deck.groups)]


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
    plan = _served_shape(deck)
    echo_at = _execute_card_numbers(text, deck)
    runs = []
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
    report = group.report
    patterns = ()
    near = ()
    if report is not None and report.mnemonic == "RP":
        patterns = (_pattern(report, result, freq, gd),)
    elif report is not None:
        near = (_near_field(report, result, solver),)
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
                float(np.linalg.norm(seg.direction)) / local(seg.centre),
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
        far_field_ground=(
            None
            if gd is None or second is None
            else FarFieldGround(
                _GD_KIND[gd.i(0)],
                second.edge_distance,
                second.height,
                second.eps_r2,
                second.sigma2,
            )
        ),
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
        float(np.linalg.norm(d)),
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
    """The Sommerfeld-table file name the deck's ``GN 2`` card carries, or
    ``None``.  The nec4 dialect reads the card without it (momwire#1317); the
    printout says the file was not read, as the licensed engine does."""
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
            tol = max(float(np.linalg.norm(point)), 1.0) * 1e-9
            for w_idx, (start, end, length) in enumerate(polyline_ends):
                if np.linalg.norm(start - point) <= tol:
                    s = 0.0
                elif np.linalg.norm(end - point) <= tol:
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
        length = float(np.linalg.norm(segs[0].direction))
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


def _pattern(card, result, freq_mhz: float, gd):
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
    return pattern_block(
        request,
        result.solver,
        result.coeffs,
        result.ground,
        freq_mhz,
        result.wavelength,
        result.p_in,
        ground_z=result.ground_z or 0.0,
        cliff=cliff,
    )


# --------------------------------------------------------------------------
# the near field
# --------------------------------------------------------------------------


def _near_field(card, result, deck_solver) -> NearFieldBlock:
    """One ``NE``/``NH`` table over free space or a perfect ground.

    The table is NEC-5's layout to the byte (p4), and the readout is the
    portal's: the element sum in mixed-potential form, plus the PEC image
    with its horizontal moments flipped and its charge negated.  The grid
    is the rectangular one, X fastest, then Y, then Z (p4 steps Z alone).
    """
    from .._far_readout import _FIELD_FLOOR2, _element_fields, _image_moments
    from ..portal._portal import _NEAR_FIELD_SUBDIV

    magnetic = card.mnemonic == "NH"
    n_x, n_y, n_z = (max(card.i(k), 1) for k in (1, 2, 3))
    start = np.array([card.f(4), card.f(5), card.f(6)])
    step = np.array([card.f(7), card.f(8), card.f(9)])
    points = np.array(
        [
            start + np.array([ix, iy, iz]) * step
            for iz in range(n_z)
            for iy in range(n_y)
            for ix in range(n_x)
        ]
    )
    solver = result.solver
    k = 2.0 * math.pi / result.wavelength
    mid, moment, nodes, delta = solver.element_currents(
        result.coeffs, subdiv=_NEAR_FIELD_SUBDIV
    )
    radius = deck_solver._smallest_radius
    field = _element_fields(points, (mid, moment, nodes, delta), k, radius, magnetic)
    if result.ground.kind == "pec":
        ground_z = result.ground_z or 0.0
        mid_img, moment_img = _image_moments(mid, moment, ground_z)
        nodes_img = nodes.copy()
        nodes_img[:, 2] = 2.0 * ground_z - nodes[:, 2]
        field = field + _element_fields(
            points, (mid_img, moment_img, nodes_img, -delta), k, radius, magnetic
        )
    rows = []
    for point, value in zip(points, field, strict=True):
        cells = [complex(c) for c in value]
        cells = [
            0j if c.real * c.real + c.imag * c.imag <= _FIELD_FLOOR2 else c
            for c in cells
        ]
        rows.append(
            NearFieldRow(
                point=(float(point[0]), float(point[1]), float(point[2])),
                magnitudes=tuple(abs(c) for c in cells),
                phases_deg=tuple(
                    math.degrees(math.atan2(c.imag, c.real)) for c in cells
                ),
            )
        )
    return NearFieldBlock(rows=tuple(rows), magnetic=magnetic)
