"""Card semantics for ``TL`` and ``NT``: six reals become circuit branches.

The dialect front end (``_nec2``) resolved WHERE a network card attaches and
kept its six fields verbatim (see :class:`~momwire.deck.model.NetworkCard`).
This module is the other half — what those six numbers MEAN — and it is the
one place they mean it (design doc ``networks-move-into-the-engine.md``, "The
dialect": *parse per-reader, semantics once*).  The rules are ported from
``antennaknobs.nec_import._translate_network_cards`` / ``_end_shunt``, which
is wild-corpus validated; the two deliberate simplifications are named below.

Nothing here knows about a solver.  It takes a model, a
:class:`~momwire.deck._solver.PortPlan` that already says which solver port
each endpoint landed on, and returns a flat
:class:`~momwire.networks.Network` plus the ``port_to_idx`` map its reducer
takes.  Composing that with an antenna admittance is
:class:`~momwire.portal.DeckSolver`'s job.

The translation, card by card
-----------------------------

``TL`` — ``F1`` characteristic impedance, ``F2`` length, ``F3+jF4`` and
``F5+jF6`` the two end shunt admittances:

* ``z0 = abs(F1)`` and ``transposed = F1 < 0``.  A crossed line is a POLARITY
  flag, never a negated impedance: negating ``z0`` would flip the chain
  matrix's diagonal self terms too, which is a different circuit.
* ``length = F2``, or — when ``F2`` is exactly zero — the straight-line
  distance between the two endpoint segment CENTRES.  That is NEC's own
  zero-length rule, and it is byte-observable: the oracle prints the RESOLVED
  length in the ``NETWORK DATA`` table's ``LENGTH`` column, so a wrong
  distance shows up in the printout before it shows up in the physics.
* each nonzero end admittance becomes a 1-port
  :class:`~momwire.networks.Admittance` from that endpoint to the common
  return.  **Simplification 1**: the importer splits a conductance-only end
  into a frequency-dependent ``Shunt(1/G)`` and keeps ``Admittance`` for the
  reactive case.  Both are the same physics; the split exists so the app's
  schematic can draw a resistor.  NEC's shunt admittances are FIXED complex
  constants at every frequency, which is exactly what ``Admittance``
  expresses, so this path uses it for both and has no branch to get wrong.
* ``vf = 1`` and ``k1 = k2 = 0``: an ideal lossless line at free-space
  velocity is the whole of what a ``TL`` card can say.

``NT`` — ``Y11 = F1+jF2``, ``Y12 = F3+jF4``, ``Y22 = F5+jF6``:

* always the full 2×2 :class:`~momwire.networks.Admittance` with ``Y21 = Y12``
  forced, because NEC's card is reciprocal by construction (it gives three
  entries, not four).  **Simplification 2**: the importer reduces an
  ALL-REAL ``Y`` to an exact resistive π (series ``−Y12``, shunts ``Y11+Y12``
  and ``Y22+Y12``).  That is the same physics through three branches instead
  of one, and it exists for the same schematic reason; here the general
  matrix is stamped directly.
* an all-zero card is NOT special-cased.  It still creates its two ports and
  stamps a zero admittance, which leaves both gaps OPEN — and open is what
  NEC measures for it (the oracle answers such a deck with a segment current
  at the far endpoint of ~1e-17 A).  Skipping the card would leave the gaps
  SHORTED instead, which is a different antenna.

Addressing, symmetry and retention are the reader's business and are already
settled by the time anything here runs.
"""

from __future__ import annotations

import math

from ..networks import TL, Admittance, Driven, Network, NetworkReducer, PortOnWire
from .model import DeckModel, DeckWire, NetworkCard


def port_name(index: int) -> str:
    """The reducer's name for solver port ``index``.

    The deck side owns the naming (the reducer only ever looks names up), so
    it is the shortest thing that cannot collide with the dotted sub-node
    names ``NetworkReducer`` synthesises for floating ports.
    """
    return f"p{index}"


def network_endpoints(model: DeckModel) -> tuple[tuple[int, float], ...]:
    """Every distinct ``(wire, arclength)`` a network card attaches to.

    In card order, each card's end A before its end B, deduplicated on the
    model's own float the way :func:`~momwire.deck._solver._sites` dedupes a
    feed against a load: both came through the same addressing, so equality
    is the honest test and there is no tolerance to justify.

    A network endpoint is a PORT — a delta gap at that segment, exactly like a
    feed's.  That is what NEC's own network solve wants: ``netwk`` reads the
    structure's admittance at the connection segments, which is the admittance
    seen across a gap there.
    """
    seen: dict[tuple[int, float], None] = {}
    for card in model.networks:
        for end in (card.end_a, card.end_b):
            seen.setdefault(end, None)
    return tuple(seen)


def live_cards(model: DeckModel, group: int) -> tuple[NetworkCard, ...]:
    """The cards in force at execute group ``group``, in card order.

    A network card is RETAINED from where it is read to the end of the deck
    (spec ``#network-retention``), so the scoping is one comparison: a group
    that ran BEFORE the card was read is answered with no network at all,
    which is what the oracle prints.
    """
    return tuple(card for card in model.networks if group >= card.first_group)


def _point_at(wire: DeckWire, arclength: float) -> tuple[float, ...]:
    """The 3-D point ``arclength`` metres along ``wire``'s polyline.

    Used only for the zero-length rule.  The endpoint arclengths the reader
    resolved ARE segment centres, so this walk returns the segment centre NEC
    measures its automatic length between.
    """
    remaining = float(arclength)
    edges = list(zip(wire.vertices, wire.vertices[1:]))
    for index, (a, b) in enumerate(edges):
        length = math.dist(a, b)
        if remaining <= length or index == len(edges) - 1:
            t = remaining / length if length else 0.0
            return tuple(pa + (pb - pa) * t for pa, pb in zip(a, b))
        remaining -= length
    raise AssertionError("a wire with no edges reached _point_at")


def card_branches(
    card: NetworkCard,
    port_a: int,
    port_b: int,
    wires: tuple[DeckWire, ...],
) -> list[object]:
    """One card's branches, per the rules in the module docstring.

    Split out from :func:`build_network` because it is the whole of the card
    semantics and nothing else in it needs a solver, a plan or a group: it
    takes the record the reader wrote, the two port indices its endpoints
    landed on, and the geometry the zero-length rule measures against.
    """
    a, b = port_name(port_a), port_name(port_b)
    f1, f2, f3, f4, f5, f6 = card.payload

    if card.kind == "NT":
        y11, y12, y22 = complex(f1, f2), complex(f3, f4), complex(f5, f6)
        # Y21 = Y12: the card states three entries and NEC reads the fourth
        # off reciprocity.
        return [Admittance(ports=(a, b), y=((y11, y12), (y12, y22)))]

    if card.kind != "TL":
        raise ValueError(f"unknown network card kind {card.kind!r}")

    length = f2
    if length == 0.0:
        length = math.dist(
            _point_at(wires[card.end_a[0]], card.end_a[1]),
            _point_at(wires[card.end_b[0]], card.end_b[1]),
        )
    branches: list[object] = [
        TL(
            a=a,
            b=b,
            z0=abs(f1),
            length=length,
            transposed=f1 < 0.0,
            vf=1.0,
            k1=0.0,
            k2=0.0,
        )
    ]
    for port, admittance in ((a, complex(f3, f4)), (b, complex(f5, f6))):
        if admittance != 0:
            branches.append(Admittance(ports=(port,), y=((admittance,),)))
    return branches


def build_network(
    model: DeckModel,
    plan,
    *,
    group: int,
    voltages,
) -> tuple[Network, dict[str, int]] | None:
    """The flat network for one execute group, or ``None`` if it has none.

    ``plan`` is the group's :class:`~momwire.deck._solver.PortPlan` and
    ``voltages`` its per-port applied voltage vector — the SAME vector the
    no-network path solves ``V_gap = (E + Z·Y)⁻¹·V_source`` with, in the same
    port order.

    Every port that is NOT a network endpoint is pinned with a ``Driven``
    source at its applied voltage, zero included.  That is not a formality:
    the reducer floats an untouched port at ``I_ext = 0``, which is an OPEN
    gap, and the deck pipeline's undriven port is a SHORTED one ("present in
    the matrix, invisible to the physics" — :class:`DeckSolver`).  A zero-volt
    ``Driven`` is the reducer's own hard ``V = 0`` pin, so the two
    conventions meet exactly where the network stops.

    A network endpoint is never pinned unless the deck drives it: its voltage
    is whatever the circuit and the structure agree on.  An endpoint that IS
    driven gets its source anyway, and the reducer's termination branch then
    carries the SOURCE current — antenna plus network — which is the number
    the ``ANTENNA INPUT PARAMETERS`` row prints.
    """
    cards = live_cards(model, group)
    if not cards:
        return None

    n = plan.n_ports
    port_to_idx = {port_name(k): k for k in range(n)}
    # Every deck port is a delta gap on the structure, which is what
    # `PortOnWire` names.  There are no `PortVirtual` nodes here and there
    # cannot be: a NEC network's every node is a segment.
    ports = {port_name(k): PortOnWire(name=port_name(k)) for k in range(n)}

    branches: list[object] = []
    endpoints: set[int] = set()
    for card, (port_a, port_b) in zip(cards, _live_port_pairs(model, plan, group)):
        branches += card_branches(card, port_a, port_b, model.wires)
        endpoints.add(port_a)
        endpoints.add(port_b)

    sources = [
        Driven(port_name(k), complex(voltages[k]))
        for k in range(n)
        if k not in endpoints or voltages[k] != 0
    ]
    network = Network(ports=ports, branches=branches, sources=sources)
    return network, port_to_idx


def _live_port_pairs(model: DeckModel, plan, group: int):
    """``(port a, port b)`` per live card, in the same order as
    :func:`live_cards`."""
    return [
        pair
        for card, pair in zip(model.networks, plan.network_ports)
        if group >= card.first_group
    ]


def build_reducer(model: DeckModel, plan, *, group: int, voltages):
    """:class:`~momwire.networks.NetworkReducer` for one group, or ``None``.

    Real ports occupy indices ``0..n_ports-1`` in exactly the order the
    antenna's admittance matrix is in, which is the reducer's whole input
    contract, and there are no virtual ports: every node of a NEC network is a
    segment of the structure.
    """
    built = build_network(model, plan, group=group, voltages=voltages)
    if built is None:
        return None
    network, port_to_idx = built
    return NetworkReducer(network, port_to_idx, plan.n_ports)
