"""Solver capability declarations (momwire#396).

Stage 0 of `docs/design/solver-architecture.md` §1.4: the axes a solver
serves — grounds, wire loading, the extended kernel, junction ports, node
gaps, knot feeds, per-wire radii, singular enrichment — had leaked out of
momwire as antennaknobs' hand-curated `_GROUND_EPS_SOLVERS` / `_WIRE_LOADING_SOLVERS`
tuples. Each solver now carries one `capabilities` class attribute so a
consumer reads the declaration instead of maintaining its own copy. The
declaration DESCRIBES; it enforces nothing — every constructor / solve-time
refusal this module points at stays the authoritative check, unchanged.

Contract (design doc §0.2, the throwaway-tier test): this module stays ONE
NamedTuple plus ONE method — no validation machinery, no capability
objects, no plugin system. A ~200-line free-space prototype solver that
declares only what it serves must still run end-to-end in a consumer with
graceful refusals, written in hours; if extending this registry ever needs
more than a class attribute and this one helper, the extension has failed
that test even if it technically works.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple

# The boolean axes, in declared field order: gives a combination key ("a+b")
# one canonical spelling regardless of call order. A cell that names a
# combination-only condition rather than a real axis (e.g. "finite_ground",
# "mixed_radii") is not one of these, and sorts after every real axis.
_AXES = (
    "wire_loading",
    "extended_kernel",
    "junction_ports",
    "node_gaps",
    "knot_feeds",
    "per_wire_radius",
    "singular_enrichment",
)


def _combo_key(cells) -> str:
    return "+".join(
        sorted(
            cells,
            key=lambda c: (c not in _AXES, _AXES.index(c) if c in _AXES else 0, c),
        )
    )


class Capabilities(NamedTuple):
    """One solver's declared row of the capability matrix.

    `grounds` is the subset of ``{"pec", "refl-coef", "sommerfeld"}`` the
    solver serves — free space is universal and not listed. The seven
    booleans are the other axes. `refusals` maps a cell name, or an "a+b"
    combination key, to the reason prose already carried by that solver's
    own refusal (a constructor or solve-time raise) — referenced from
    there, not duplicated here.

    `knot_feeds` is the odd one, because it is the only axis a solver can
    fail SILENTLY (momwire#611). Every family resolves a ``feeds``
    arclength onto a grid of its own: the B-spline family integrates the
    delta at the arclength itself, `RazorSolver._snap_to_knot` moves it to
    the nearest knot, and the sinusoidal and pulse families move it to the
    nearest segment CENTRE. A consumer that addresses NODES — the NEC-5
    dialect is the one in tree — asks for a knot, and the last of those
    three answers half a segment away without raising anything. So the cell
    is True when a gap lands on the knot grid the caller named and False
    when it lands on the segment-centre grid instead; a family that snaps
    is not broken, it is answering a different question, and this is the
    axis that says so out loud. `test_capabilities.py` measures the
    declaration against a symmetry probe rather than trusting it.
    """

    grounds: frozenset[str]
    wire_loading: bool
    extended_kernel: bool
    junction_ports: bool
    node_gaps: bool
    knot_feeds: bool
    per_wire_radius: bool
    singular_enrichment: bool
    refusals: Mapping[str, str]

    def _served(self, cell: str) -> bool:
        if cell in ("pec", "refl-coef", "sommerfeld"):
            return cell in self.grounds
        if cell in _AXES:
            return bool(getattr(self, cell))
        # A condition token ("finite_ground", "mixed_radii", …) is not a
        # capability — it carries meaning only through a declared "a+b"
        # combination key. Treating an unknown token as refused would make
        # every solver WITHOUT that combination key spuriously refuse it
        # (BSplineSolver serves junction_ports over a finite ground and
        # declares no such key), so an unmatched condition is served.
        return True

    def refusal(self, *cells: str) -> str | None:
        """The declared reason `cells` is refused, or None if it is served.

        `cells` is one cell name, or two naming a combination. Checks the
        exact "a+b" key first (`_combo_key`'s canonical order), then each
        single cell in the order given: the first one this solver does not
        serve returns its `refusals` entry when one exists, else a
        generated one-line default. Nothing refused returns None — and a
        condition token that matches no combination key is served, per
        `_served`.
        """
        if len(cells) > 1:
            hit = self.refusals.get(_combo_key(cells))
            if hit is not None:
                return hit
        for cell in cells:
            if not self._served(cell):
                return self.refusals.get(
                    cell, f"{cell} is not supported by this solver"
                )
        return None
