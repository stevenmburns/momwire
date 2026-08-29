"""The model's wires, chained into the polylines a solver constructor takes.

A :class:`~momwire.deck.model.DeckModel` describes conductors the way a deck
does: one entry per span, connected implicitly by sharing an endpoint.  Every
momwire solver wants the opposite — each electrical wire as ONE polyline, with
the shared nodes named explicitly in ``junctions=`` so the fill can put a KCL
row there.  This module is the translation, and it is the last geometry step
of the deck front end: after it, nothing downstream speaks in deck terms.

Four things happen here, in this order:

  1. **The port spans.**  A port — a feed or a load — is stamped in ONE
     element, and that element becomes an edge of its own, so the port's
     position is half of one edge length rather than a running sum over the
     mesh.  The one exception is a port already sitting at its edge's
     midpoint (the only port on an odd-element edge), where a split would add
     knots and move nothing.
  2. **The node graph.**  Span endpoints are quantized onto a ``1e-6`` m grid
     and fused, which is what turns "these two spans touch" into "these two
     spans share a node".  The tolerance is deliberately coarse compared with
     the dialect front end's own connection pass: by the time a model exists
     its coincident ends are already exactly equal, and the grid is here to
     absorb the last few ulps of a transform, not to decide connectivity.
  3. **The chaining.**  Nodes of degree 2 are interior to a polyline; every
     other node ends one.  A change of radius or material at a degree-2 node
     ends one too, because a solver takes one radius and one material per
     wire.  Pure cycles have no node of any other degree, so they are cut —
     at a port edge when the loop has one, since that edge has to be its own
     polyline anyway, and at the lowest-numbered edge when it does not.
  4. **The remap.**  Every port's arclength is recomputed along the polyline
     it ended up on, from that polyline's own edge lengths.

Steps 2-4 are transcribed from antennaknobs' ``geometry.flat_wires_to_
polylines``, which is the reference this arc measures against: the two must
build the same mesh out of the same deck, down to the polyline ORDER and the
float arithmetic, or the impedance a deck reports would move when the portal
changes hands.  ``tests/test_deck_nec2_corpus.py`` is that measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import DeckModel, DeckWire, WireMaterial

__all__ = ["Mesh", "to_polylines"]


# Node quantization, metres.  antennaknobs' own; see the module docstring for
# why it is coarse and why that is safe.
_NODE_EPS = 1e-6


@dataclass(frozen=True)
class Mesh:
    """What a solver constructor needs, in the solver's own vocabulary.

    ``polylines`` / ``edge_elements`` are its ``wires=`` and
    ``n_per_edge_per_wire=``; ``junctions`` is its ``junctions=``; ``radii``
    and ``materials`` are one entry per polyline, and the port lists carry
    every gap the model asked for, positioned along the polyline it landed
    on.
    """

    polylines: tuple[np.ndarray, ...]
    edge_elements: tuple[tuple[int, ...], ...]
    junctions: tuple[tuple[tuple[int, str], ...], ...]
    radii: tuple[float, ...]
    materials: tuple[WireMaterial | None, ...]

    # (polyline, arclength) per SOLVER port index — the order the ports were
    # met walking the structure, which is the order a solver's feed list and
    # therefore its Y matrix is in.
    ports: tuple[tuple[int, float], ...]
    # Solver port index i carries the model port ``port_order[i]`` — the
    # index into the port list :func:`to_polylines` was given.
    port_order: tuple[int, ...]
    # (polyline, "start"|"end") per node gap, in the model's own order.
    node_gap_members: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class _Span:
    """One straight piece of a model wire, as the chaining sees it.

    A model wire's edge is one span when it carries no port and several when
    it does.  ``port`` is the index (into the port list) of the port stamped
    in this span, or ``None``.
    """

    p0: tuple[float, float, float]
    p1: tuple[float, float, float]
    n: int
    port: int | None
    radius: float
    material: WireMaterial | None
    # Where this span came from, for the node-gap lookup: (model wire, model
    # edge) and whether it touches the edge's own start / end.
    wire: int
    edge: int
    at_edge_start: bool
    at_edge_end: bool


def _element_of(wire: DeckWire, arclength: float) -> tuple[int, int]:
    """``(edge, 0-based element)`` for a point on a wire.

    Every port sits at an element CENTRE — half an element from the nearest
    boundary — so a plain division locates it with an enormous margin, and
    the clamp only catches an arclength that walked off the end of the wire
    through rounding.
    """
    remaining = arclength
    lengths = wire.edge_lengths
    for k, edge_length in enumerate(lengths):
        n = wire.edge_elements[k]
        if remaining <= edge_length or k == len(lengths) - 1:
            step = edge_length / n
            element = int(remaining // step) if step > 0.0 else 0
            return k, max(0, min(n - 1, element))
        remaining -= edge_length
    raise AssertionError("unreachable: a wire always has one edge")


def _spans(model: DeckModel, ports: tuple[tuple[int, float], ...]) -> list[_Span]:
    """The model's wires cut into the spans the chaining walks.

    One span per edge, except where a port forces the split described in the
    module docstring.  The split points are evaluated as ``a + (b - a) * k/n``
    — the same expression the dialect front end's own boundaries use — so two
    spans that adjoin land on bitwise-equal coordinates and fuse into one
    node without relying on the quantization to save them.
    """
    sites: dict[tuple[int, int, int], int] = {}
    for index, (wire_index, arclength) in enumerate(ports):
        edge, element = _element_of(model.wires[wire_index], arclength)
        sites.setdefault((wire_index, edge, element), index)

    spans: list[_Span] = []
    for wire_index, wire in enumerate(model.wires):
        for edge in range(len(wire.edge_elements)):
            n = wire.edge_elements[edge]
            a, b = wire.vertices[edge], wire.vertices[edge + 1]
            on_edge = {
                element: port
                for (w, e, element), port in sites.items()
                if w == wire_index and e == edge
            }

            # `wire`/`wire_index`/`edge` are bound as defaults for the same
            # reason `a`/`b`/`n` already are: `emit` closes over the enclosing
            # loops' variables, and binding them at definition makes the
            # capture explicit rather than late (B023). Every call site is
            # inside this iteration, so this is a statement of intent, not a
            # bug fix -- and it keeps B023 live for a closure that one day
            # ISN'T called immediately.
            def emit(
                first: int,
                last: int,
                port: int | None,
                a=a,
                b=b,
                n=n,
                wire=wire,
                wire_index=wire_index,
                edge=edge,
            ) -> None:
                # An UNCUT edge keeps the model's own vertices; a cut one
                # evaluates every boundary through `_point`, its own two
                # ends included.  The distinction is not cosmetic:
                # `a + (b - a) * 1.0` is not always bitwise `b`, so a split
                # edge's last vertex is a hair off the vertex an unsplit one
                # would have kept — and the reference draws the line in
                # exactly the same place.
                whole = first == 0 and last == n
                p0 = a if whole else _point(a, b, first, n)
                p1 = b if whole else _point(a, b, last, n)
                spans.append(
                    _Span(
                        p0=p0,
                        p1=p1,
                        n=last - first,
                        port=port,
                        radius=wire.radius,
                        material=wire.material,
                        wire=wire_index,
                        edge=edge,
                        at_edge_start=first == 0,
                        at_edge_end=last == n,
                    )
                )

            if not on_edge:
                emit(0, n, None)
                continue
            if len(on_edge) == 1:
                ((element, port),) = on_edge.items()
                if n % 2 == 1 and element == (n - 1) // 2:
                    # Already at the edge's midpoint: the split would put the
                    # port at half of a one-element edge instead of half of
                    # this one, which is the same point and two more knots.
                    emit(0, n, port)
                    continue
            bounds = sorted(
                {k for element in on_edge for k in (element, element + 1)} - {0, n}
            )
            previous = 0
            for boundary in [*bounds, n]:
                count = boundary - previous
                emit(
                    previous,
                    boundary,
                    on_edge.get(boundary - 1) if count == 1 else None,
                )
                previous = boundary
    return spans


def _point(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    k: int,
    n: int,
) -> tuple[float, float, float]:
    """The point after ``k`` of an edge's ``n`` elements."""
    t = k / n
    return tuple(x + (y - x) * t for x, y in zip(a, b))  # type: ignore[return-value]


def to_polylines(model: DeckModel, ports: tuple[tuple[int, float], ...]) -> Mesh:
    """Chain a model's wires into polylines and place its ports on them.

    ``ports`` is ``(model wire, arclength)`` per port, in the caller's own
    order; :attr:`Mesh.port_order` says where each one ended up once the
    structure decided the solver's port ordering.
    """
    spans = _spans(model, ports)
    if not spans:
        raise ValueError("the model has no wires")

    # -- the node graph ----------------------------------------------------
    node_of: dict[tuple[int, int, int], int] = {}
    nodes: list[tuple[float, float, float]] = []

    def node_id(p: tuple[float, float, float]) -> int:
        key = tuple(round(c / _NODE_EPS) for c in p)
        if key not in node_of:
            node_of[key] = len(nodes)
            nodes.append(p)
        return node_of[key]

    edges: list[tuple[int, int, int]] = []  # (node a, node b, span index)
    for index, span in enumerate(spans):
        a, b = node_id(span.p0), node_id(span.p1)
        if a == b:
            raise ValueError(
                f"wire {span.wire} has a span whose two ends fall on one node "
                f"— a conductor shorter than the {_NODE_EPS} m node tolerance"
            )
        edges.append((a, b, index))

    adjacency: list[list[tuple[int, int]]] = [[] for _ in nodes]
    for edge_index, (a, b, _) in enumerate(edges):
        adjacency[a].append((b, edge_index))
        adjacency[b].append((a, edge_index))

    # -- polyline boundaries -----------------------------------------------
    boundary = [len(neighbours) != 2 for neighbours in adjacency]
    for node, neighbours in enumerate(adjacency):
        if boundary[node]:
            continue
        first, second = spans[neighbours[0][1]], spans[neighbours[1][1]]
        if (first.radius, first.material) != (second.radius, second.material):
            # A solver takes ONE radius and one material per wire, so a
            # change of either has to end a polyline.  The node is registered
            # as a two-member junction below, exactly like a cycle cut, so
            # the current still flows through it.
            boundary[node] = True

    node_gap_nodes = [
        node_id(model.wires[wire].vertices[vertex])
        for wire, vertex, _volts in model.node_gaps
    ]
    for node in node_gap_nodes:
        # A series gap at a knot needs that knot to be a wire END, which is
        # the same forcing a junction port needs (issue #579 in the
        # reference, momwire#305 for the gap itself).
        boundary[node] = True

    # -- the walk ----------------------------------------------------------
    walked = [False] * len(edges)
    polylines: list[list[int]] = []  # node paths
    edge_elements: list[list[int]] = []
    radii: list[float] = []
    materials: list[WireMaterial | None] = []
    span_place: dict[int, tuple[int, int]] = {}  # span -> (polyline, position)
    span_direction: dict[int, int] = {}  # +1 when walked p0 -> p1
    junction_ends: dict[int, list[tuple[int, str]]] = {
        node: [] for node in range(len(nodes)) if boundary[node]
    }

    def walk(start: int, first_edge: int) -> tuple[list[int], list[int]]:
        path_nodes = [start]
        path_edges: list[int] = []
        previous_edge = None
        current = start
        next_edge = first_edge
        while True:
            walked[next_edge] = True
            path_edges.append(next_edge)
            a, b, _ = edges[next_edge]
            current = b if a == current else a
            path_nodes.append(current)
            if boundary[current]:
                return path_nodes, path_edges
            previous_edge = next_edge
            next_edge = next(
                edge for _n, edge in adjacency[current] if edge != previous_edge
            )

    def record(path_nodes: list[int], path_edges: list[int]) -> int:
        index = len(polylines)
        polylines.append(path_nodes)
        edge_elements.append([spans[edges[e][2]].n for e in path_edges])
        head = spans[edges[path_edges[0]][2]]
        radii.append(head.radius)
        materials.append(head.material)
        for position, edge in enumerate(path_edges):
            span = edges[edge][2]
            span_place[span] = (index, position)
            span_direction[span] = 1 if edges[edge][0] == path_nodes[position] else -1
        return index

    for start in range(len(nodes)):
        if not boundary[start]:
            continue
        for _neighbour, edge in adjacency[start]:
            if walked[edge]:
                continue
            path_nodes, path_edges = walk(start, edge)
            index = record(path_nodes, path_edges)
            junction_ends[path_nodes[0]].append((index, "start"))
            junction_ends[path_nodes[-1]].append((index, "end"))

    # -- pure cycles -------------------------------------------------------
    while not all(walked):
        seed = next(i for i, done in enumerate(walked) if not done)
        component: list[int] = []
        stack = [seed]
        seen = {seed}
        while stack:
            edge = stack.pop()
            component.append(edge)
            walked[edge] = True
            for endpoint in edges[edge][:2]:
                for _n, other in adjacency[endpoint]:
                    if other not in seen and not walked[other]:
                        seen.add(other)
                        stack.append(other)

        # Cut at a port edge when the loop has one: that edge becomes its own
        # polyline either way, so cutting there costs the loop nothing.  A
        # parasitic loop — one that radiates only through coupling — has no
        # port edge, and then any edge breaks the cycle equally well.
        with_ports = [e for e in component if spans[edges[e][2]].port is not None]
        cut = with_ports[0] if with_ports else min(component)
        cut_a, cut_b, cut_span = edges[cut]

        cut_index = record([cut_a, cut_b], [cut])
        boundary[cut_a] = True
        boundary[cut_b] = True
        junction_ends.setdefault(cut_a, [])
        junction_ends.setdefault(cut_b, [])
        for edge in component:
            if edge != cut:
                walked[edge] = False

        first = next(
            edge for _n, edge in adjacency[cut_b] if edge != cut and not walked[edge]
        )
        path_nodes, path_edges = walk(cut_b, first)
        loop_index = record(path_nodes, path_edges)

        # The cut polyline was recorded as [A, B] and the long way walked
        # B -> A, so each cut node carries one end of each.
        junction_ends[cut_a].append((cut_index, "start"))
        junction_ends[cut_a].append((loop_index, "end"))
        junction_ends[cut_b].append((cut_index, "end"))
        junction_ends[cut_b].append((loop_index, "start"))
        # A one-edge polyline is walked in its own authored direction.
        span_direction[cut_span] = 1

    # -- the result --------------------------------------------------------
    arrays = tuple(
        np.stack([np.asarray(nodes[n], dtype=float) for n in path], axis=0)
        for path in polylines
    )
    lengths = [np.linalg.norm(np.diff(a, axis=0), axis=1) for a in arrays]

    forced = set(node_gap_nodes)
    junctions: list[tuple[tuple[int, str], ...]] = []
    junction_of_node: dict[int, int] = {}
    for node, ends in junction_ends.items():
        if len(ends) >= 2 or (node in forced and ends):
            junction_of_node[node] = len(junctions)
            junctions.append(tuple(ends))

    # Solver port order is STRUCTURE order — the order the port spans appear
    # in the wire list, not the order the chaining walk reached them.  The
    # two differ whenever a walk runs a polyline against the wire numbering
    # (a mid-structure feed on a chain assembled from both ends), and it is
    # the wire order that fixes a solver's feed list and therefore its Y
    # matrix.  The model's own port order is recorded beside it rather than
    # imposed on the mesh.
    port_spans = [index for index, span in enumerate(spans) if span.port is not None]
    placed: list[tuple[int, float]] = []
    order: list[int] = []
    for span_index in port_spans:
        polyline, position = span_place[span_index]
        edge_lengths = lengths[polyline]
        arclength = float(edge_lengths[:position].sum() + 0.5 * edge_lengths[position])
        placed.append((polyline, arclength))
        order.append(spans[span_index].port)  # type: ignore[arg-type]

    return Mesh(
        polylines=arrays,
        edge_elements=tuple(tuple(counts) for counts in edge_elements),
        junctions=tuple(junctions),
        radii=tuple(radii),
        materials=tuple(materials),
        ports=tuple(placed),
        port_order=tuple(order),
        node_gap_members=tuple(
            _node_gap_member(model, spans, span_place, span_direction, edge_elements, k)
            for k in range(len(model.node_gaps))
        ),
    )


def _node_gap_member(
    model: DeckModel,
    spans: list[_Span],
    span_place: dict[int, tuple[int, int]],
    span_direction: dict[int, int],
    edge_elements: list[list[int]],
    index: int,
) -> tuple[int, str]:
    """``(polyline, "start"|"end")`` for one node gap.

    A gap at knot 0 rides the span that STARTS there; every other knot rides
    the span that ENDS there, which is the convention a dialect emitting edge
    sources addresses them by.
    """
    wire, vertex, _volts = model.node_gaps[index]
    at_start = vertex == 0
    edge = 0 if at_start else vertex - 1
    candidates = [
        i
        for i, span in enumerate(spans)
        if (span.wire, span.edge) == (wire, edge)
        and (span.at_edge_start if at_start else span.at_edge_end)
    ]
    if not candidates:
        raise ValueError(f"node gap {index} names a knot no wire span touches")
    polyline, position = span_place[candidates[0]]
    direction = span_direction[candidates[0]]
    at = position + (0 if at_start == (direction == 1) else 1)
    if at not in (0, len(edge_elements[polyline])):
        raise ValueError(
            f"node gap {index} lands inside polyline {polyline} rather than "
            f"on one of its ends"
        )
    return polyline, "start" if at == 0 else "end"
