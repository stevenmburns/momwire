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

from types import MappingProxyType
from typing import Mapping, NamedTuple

# The boolean axes, in LOGICAL order (not literal field order — a NamedTuple
# default has to come last, so the three cells added after the row shipped are
# declared at the end of the class and sit here where they belong): gives a
# combination key ("a+b") one canonical spelling regardless of call order. A
# cell that names a combination-only condition rather than a real axis (e.g.
# "finite_ground", "mixed_radii") is not one of these, and sorts after every
# real axis.
#
# The two GEOMETRY axes lead, and that is load-bearing rather than tidy: every
# combination key already written down pairs one of them with something else
# ("buried+contact", "buried+extended_kernel", "buried+crossing",
# "contact+refl-coef"), and those spellings are what the rows and the tests
# say. Putting them first is what keeps `_combo_key` producing them.
_AXES = (
    "buried",
    "contact",
    "wire_loading",
    "extended_kernel",
    "junction_ports",
    "node_gaps",
    "knot_feeds",
    "per_wire_radius",
    "singular_enrichment",
    # APPENDED rather than slotted beside `knot_feeds`, though that is where it
    # belongs by meaning (momwire#673).  `_combo_key` orders a combination key
    # by `_AXES.index`, so inserting in the middle would re-spell any existing
    # "a+b" key containing a later axis.  None exists today, which makes the
    # insert look safe and would make the next one a silent rename.
    "centre_feeds",
)


# --------------------------------------------------------------------------
# THE COMPOSITIONAL AXES (antennaknobs#1006 G2-1).
#
# The booleans above say what a solver SERVES. These say what it is MADE OF,
# which is a different question and the one a panel needs: a user choosing
# between `bspline` and `hmatrix` cannot see from a name that the physics is
# identical and only the solve strategy differs, and a user choosing
# `sinusoidal-galerkin` over `sinusoidal` cannot see that the basis is the
# same and only the testing changed. The roster names are, in effect, saved
# presets over the product space below, which nobody had written down.
#
# VALUES ARE A SET PER SOLVER, NOT ONE VALUE. Several of these axes are
# constructor kwargs rather than separate classes — `degree=`,
# `nec5_quadrature=`, `extended_kernel=`, `feed_model=` — so a class row
# declares WHICH VALUES IT CAN BE CONFIGURED TO, and a preset name picks one
# point. Declaring a single value would have to lie about `BSplineSolver`,
# which is both bspline-1 and bspline-2.
#
# This is DATA, deliberately: the module contract at the top of this file says
# one NamedTuple plus one method, no validation machinery. Nothing here
# validates a row against this vocabulary; it is the written-down spelling so
# three consumers stop inventing their own, and a generated matrix can carry a
# column per axis. A row that declares a value not listed here is a row that
# found a value this comment has not caught up with.
AXIS_VALUES: Mapping[str, tuple[str, ...]] = {
    "basis": ("pulse", "tent", "bspline-1", "bspline-2", "sinusoidal-3term"),
    "testing": ("point-matching", "galerkin", "path"),
    "charge_support": ("point", "dual-cell", "spline", "basis-implied"),
    "kernel": ("reduced", "extended"),
    "quadrature": ("converged", "nec5"),
    "solve_strategy": ("dense", "aca", "element-block"),
    "feed_model": ("segment-gap", "point-gap", "node-port"),
}

# DERIVED AXES — in the vocabulary so it is complete, and deliberately NOT
# fields a row restates. `ground_model` is `grounds` (plus the universal free
# space); `wire_position` is the `buried` / `contact` pair. Both are already
# declared, already served through to consumers, and already gated, so a row
# repeating them would be a second source of truth for a fact this module
# already holds — the drift shape momwire#568 records in another costume.
#
# Read them through `axes_for()` below, never by reaching for the booleans:
# a recipe written in prose here would be that same second source of truth
# the moment a consumer implemented it slightly differently.
DERIVED_AXES: tuple[str, ...] = ("ground_model", "wire_position")


def axes_for(row) -> dict[str, frozenset[str]]:
    """Every axis of one row — declared union derived — as axis -> values.

    THE single place the derived pair is computed. The generated matrix, its
    drift test and antennaknobs' `/capabilities` all call this rather than
    each deriving `ground_model` from `grounds` and `wire_position` from
    `buried`/`contact` in their own way.

    "free" and "above" are universal, exactly as free space is universal in
    `grounds`: a solver that fills no wire below the interface and stands no
    end in the plane still serves a wire in the air, and there is no cell to
    declare for it. So both derived axes always carry at least one value even
    on the empty prototype row, while the DECLARED axes of that row are simply
    absent — which is the distinction a consumer needs, and why this returns
    only the keys a row actually has rather than padding the vocabulary out
    with empty sets.

    A module-level function rather than a method, so the top-of-file contract
    ("one NamedTuple plus ONE method") still reads true: `refusal` remains the
    tuple's only public method, and this sits beside `_combo_key` as a helper
    over the row.
    """
    out = {axis: frozenset(values) for axis, values in row.axes.items()}
    out["ground_model"] = frozenset(("free",)) | frozenset(row.grounds)
    out["wire_position"] = (
        frozenset(("above",))
        | (frozenset(("contact",)) if row.contact else frozenset())
        | (frozenset(("buried",)) if row.buried else frozenset())
    )
    return out


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
    solver serves — free space is universal and not listed. The ten
    booleans are the other axes. `refusals` maps a cell name, or an "a+b"
    combination key, to the reason prose already carried by that solver's
    own refusal (a constructor or solve-time raise) — referenced from
    there, not duplicated here.

    THE ONE RULE for what is an axis and what is a condition token: a
    question about the DECK that a solver answers the same way for every
    deck is an axis and gets a declared boolean; a question whose answer
    depends on some other cell is a condition token and appears only inside
    an "a+b" key. `buried` and `contact` are axes under that rule — "does
    this family fill a wire below the interface", "does it stand a wire end
    in the plane" — and they had to become ones, because `_served` reads an
    undeclared token as SERVED and so a row could not say "refused, full
    stop" about either no matter what it put in `refusals`: the entry was
    unreachable from `refusal("buried")`. The alternative was a combination
    key per ground on every row, which still could not answer the
    single-cell question and would have spelled one refusal three times.
    "crossing", "finite_ground", "mixed_radii" and "stepped_radius_junction"
    stay condition tokens: each is a shape a served deck can take, not
    something a solver is made of.

    `knot_feeds` and `centre_feeds` are the odd pair, because they are the
    only axes a solver can fail SILENTLY (momwire#611, #673). Every family resolves a ``feeds``
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

    `centre_feeds` is that question asked the other way (momwire#673), and
    the two are mirrors rather than opposites: it asks whether a gap lands
    on the segment-CENTRE grid the caller named. The `nec2` dialect is the
    consumer that addresses centres — `Nec2Structure.resolve` returns
    ``(element + 0.5) * length / n_seg`` — so a family that snaps to knots
    answers half a cell away there, silently, exactly as the centre-snapping
    families do on the node-addressing side.

    Both cells are True on most rows, which is not a contradiction: a family
    that never snaps at all lands wherever it was named and satisfies both.
    `RazorSolver` is the only False here (`_snap_to_knot`), and `bspline-d1`
    is deliberately NOT in its class though both are tent bases — the axis is
    WHERE THE PORT ENDS UP, not how accurate it is once there. d=1 places the
    gap at the arclength it was given and pays for it in accuracy; razor puts
    it somewhere else. Mesh quality has its own mechanism a layer up and it
    is not a refusal.
    """

    grounds: frozenset[str]
    wire_loading: bool
    extended_kernel: bool
    junction_ports: bool
    node_gaps: bool
    per_wire_radius: bool
    singular_enrichment: bool
    refusals: Mapping[str, str]
    # Out of order, and after `refusals`, because a NamedTuple default has to
    # come last — and it needs a default. Every other field is required, which
    # is the right discipline for a row that shipped complete; `knot_feeds`
    # arrived afterwards (momwire#611), and a required field would have broken
    # every declaration already written, including the ~200-line prototype the
    # design doc's §0.2 definition-of-done builds with an EMPTY row. That test
    # is the one this module is measured against, so the field bends to it.
    #
    # False is the safe default and the honest one: the empty row means "every
    # axis refused", and a family that does place a gap where it was named
    # loses nothing by saying so. The other direction would let a family that
    # silently snaps be served by a node-addressing consumer, which is the
    # exact failure momwire#611 existed to close.
    knot_feeds: bool = False
    # The mirror (momwire#673), declared beside the axis it mirrors, and
    # defaulting False for the same reason: the empty row means "every axis
    # refused", and a family that does place a gap where it was named loses
    # nothing by saying so. The other direction would let a family that
    # silently snaps be served by a CENTRE-addressing consumer, which is the
    # `nec2` half of the failure momwire#611 closed on the node half.
    centre_feeds: bool = False
    # The two GEOMETRY axes (momwire#792), on `knot_feeds`' precedent
    # and for the same reason: they arrived after every row was written, so
    # they need a default, so they come last. See THE ONE RULE above for why
    # they are axes at all.
    #
    # False both, and again the safe direction is the honest one. `buried` is
    # served by exactly one family (`BSplineSolver`, momwire#553) and refused
    # by the other five plus the two accelerators, so False is also the
    # common case; `contact` is the reverse, served by five and refused by
    # the pulse family — but a row that forgets to declare either is a row
    # nobody has checked against a deck, and defaulting to "served" would
    # publish that omission as a capability.
    buried: bool = False
    contact: bool = False
    # The compositional row (antennaknobs#1006 G2-1): axis name -> the values
    # THIS solver class can be configured to, from `AXIS_VALUES` above. Last
    # and defaulted for the same reason every field since `knot_feeds` is —
    # a NamedTuple default has to come last, and the ~200-line prototype the
    # design doc's §0.2 builds with an EMPTY row must keep working.
    #
    # An empty mapping means "this row has not been described compositionally
    # yet", which is honestly different from every other default here: the
    # booleans default to the SAFE direction (refused), because publishing an
    # undeclared capability is the failure they guard. There is no unsafe
    # direction for a description — a missing axis renders as unknown and
    # nothing is claimed — so this one defaults to absent rather than to a
    # guess, and a consumer must handle a row that says nothing.
    # `MappingProxyType({})` rather than `{}`: a NamedTuple default is one
    # object shared by every row that omits the field, so a bare `{}` is a
    # shared mutable. Nothing mutates it and nothing is likely to, but the
    # proxy makes that a guarantee instead of a convention — and this row is
    # handed to consumers in two other repos.
    axes: Mapping[str, tuple[str, ...]] = MappingProxyType({})

    def _served(self, cell: str) -> bool:
        if cell in ("pec", "refl-coef", "sommerfeld"):
            return cell in self.grounds
        if cell in _AXES:
            return bool(getattr(self, cell))
        # A condition token ("crossing", "finite_ground", "mixed_radii", …)
        # is not a capability — it carries meaning only through a declared
        # "a+b" combination key. Treating an unknown token as refused would
        # make every solver WITHOUT that combination key spuriously refuse it
        # (BSplineSolver serves junction_ports over a finite ground and
        # declares no such key), so an unmatched condition is served.
        #
        # This is the rule "buried" and "contact" had to be promoted OUT of:
        # both name something a family either does or does not do, so reading
        # them as served-unless-paired made a row unable to refuse either.
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
