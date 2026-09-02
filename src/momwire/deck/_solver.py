"""``build_solver`` — a parsed deck, on a momwire solver.

This is the deck front end's last step and its only public verb besides
:func:`~momwire.deck.parse`.  It takes a :class:`~momwire.deck.model.DeckModel`
and one choice — which basis to solve in — and returns a constructed solver
plus the :class:`PortPlan` that says what its ports MEAN.

**Why a plan and not just a solver.**  A solver's ports are its own: gap feeds
in the order the structure met them, then junction ports, then node ports.  A
deck's ports are the deck's: ``EX`` cards in discovery order, then ``LD``
cards.  Neither ordering can be imposed on the other without lying to
somebody — the solver's comes out of the mesh, the deck's out of the card
sequence, and a load stamped on a driven segment is ONE solver port answering
to two deck cards.  So the plan is the bridge, and it carries the three things
a consumer cannot recover from the solver alone: which port is which model
feed or load, what each load's :class:`~momwire.deck.model.LoadSpec` is, and
the drive vector each execute group applies over the port set.

**What stays out.**  Stamping a load is port algebra — ``V_gap = (1 + Z·Y)⁻¹
V_source``, an impedance in the port's own current path — and it belongs to
whoever owns the answer, not to the deck reader.  ``momwire.deck`` puts the
load's gap in the matrix (a zero-voltage, zero-impedance port is a shorted gap
and costs nothing) and hands the ``LoadSpec`` over.  A consumer that wants
NEC's loaded impedance stamps it; a consumer that wants the bare structure
ignores it; neither has to re-derive where the gap went.

**One exception: razor.**  ``RazorSolver`` (momwire#432) refuses the
port-algebra route's zero-volt gap and serves a load through its own
``lumped_loads`` kwarg instead, so for that one family ``build_solver`` bakes
the ``LoadSpec`` into the fill itself rather than leaving it on the plan for
a consumer to stamp — :attr:`PortPlan.loaded_ports` still names the site (the
plan does not change shape by basis), but nothing further needs doing with
it when the basis is ``"razor-2p"`` or ``"razor-nec5"``.

**One geometry, every group.**  The port set is the union over every execute
group, so a deck with two ``XQ`` cards under two different ``EX`` sets is one
fill and two drive vectors — :attr:`PortPlan.voltages` is those vectors.  Only
the frequency, the extended-kernel flag and the environment can force a second
solver, which is why all three are :func:`build_solver` arguments rather than
plan entries.

**Translate once, fill many times.**  Nothing above depends on the operating
point: the port set, the chaining into polylines and the plan's numbering are
the STRUCTURE's, and a sweep rebuilds them at every step for nothing.
:func:`prepare_mesh` freezes them into a :class:`PreparedMesh` that
``build_solver(model, mesh=…)`` replays, handing every solver the same
polyline arrays rather than a fresh rounding of the same walk.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from .._constants import C_LIGHT
from ..array_block import ArrayBlockSolver
from ..bspline import BSplineSolver
from ..harrington import HarringtonSolver
from ..hmatrix import HMatrixSolver
from ..razor import RazorSolver
from ..sinusoidal import SinusoidalSolver
from ..sinusoidal_galerkin import SinusoidalGalerkinSolver
from ._networks import network_endpoints
from ._polylines import Mesh, to_polylines
from .model import DeckModel, Environment, LoadSpec

__all__ = [
    "BuiltSolver",
    "PortPlan",
    "PortSite",
    "PreparedMesh",
    "build_solver",
    "prepare_mesh",
    "BASES",
]

_C_LIGHT = C_LIGHT  # momwire#456: one owner, in `momwire._constants`

# The basis roster, spelled as antennaknobs' ``--basis`` names so one string
# selects the same physics on either side of the portal.  Six solver
# FAMILIES, eight entries: the extra two are a degree and a quadrature rule,
# not new code paths.  It was nine until momwire#654 collapsed the two
# Galerkin spellings into one — the third extra was a FEED MODEL, and a menu
# with two spellings of one dish is what that issue was about.  "bspline" is
# the degree-2 B-spline — the default here as it is there.
#
# `razor-2p` (momwire#432) is RazorSolver — the tent basis with razor-blade
# testing, the formulation NEC-5 identifies with; the measurements retreated
# from calling it a twin (momwire#785) — see ``docs/razor-solver.md``.
# "razor-nec5" is its deprecated spelling
# (momwire#316): same class, `nec5_quadrature=True`, the same "one class,
# one extra kwarg" shape `bspline-d1` set for a degree axis. Plain `razor` —
# the Gauss-Legendre testing-path lane of the same class — retired from this
# roster at momwire#753; it is still reached by constructing
# `RazorSolver(...)` directly, whose `nec5_quadrature` kwarg defaults to
# `False`. RazorSolver's translation differs from every sibling's in two ways
# both dialect front ends have to know about (`port_kwargs`): it is given no
# `junctions` spec, and it serves a lumped load as its own `lumped_loads`
# kwarg rather than as a deck-level port-algebra site, which is why it is
# keyed by CLASS in `_NATIVE_LOADING` rather than threaded through
# `basis_kwargs` here — a future razor variant inherits the translation for
# free.
BASES = MappingProxyType(
    {
        "bspline": (BSplineSolver, MappingProxyType({})),
        "bspline-d1": (BSplineSolver, MappingProxyType({"degree": 1})),
        "hmatrix": (HMatrixSolver, MappingProxyType({})),
        "arrayblock": (ArrayBlockSolver, MappingProxyType({})),
        "sinusoidal": (SinusoidalSolver, MappingProxyType({})),
        # ONE Galerkin name (momwire#654). It used to be two, differing only
        # in `feed_model`, which is a choice a portal user has no way to make
        # and no reason to be handed: the point gap is better on every axis
        # that was measured, and it is now this class's default, so the plain
        # name binds nothing and means it. The flag survives for the callers
        # who genuinely choose — antennaknobs' web panel renders it, and the
        # payoff gates name `"segment"` as their matched control — but a
        # roster entry is a menu item, and this menu had two spellings of one
        # dish with no way to tell them apart from the outside.
        "sinusoidal-galerkin": (SinusoidalGalerkinSolver, MappingProxyType({})),
        # `razor-2p` takes NEC-5's identified two-point trapezoid at the wing
        # centroids (momwire#316); `_path_nodes_per_wing` names the rule. The
        # other lane `RazorSolver` can take — `n_qp_path` Gauss-Legendre
        # nodes per wing along the same testing path — was this roster's
        # plain `razor` entry until momwire#753 retired it: the SAME class
        # differing only in that sampling choice, measured at 20x the wall
        # time (20.2 s vs 0.97 s at N=1600 free space) for a 0.001 ohm
        # difference from `razor-2p` — not worth ordering, on momwire#654's
        # rule that a roster entry is a menu item. It stays reachable for
        # convergence and certification work by constructing
        # `RazorSolver(...)` directly, whose `nec5_quadrature` kwarg
        # defaults to `False`.
        "razor-2p": (RazorSolver, MappingProxyType({"nec5_quadrature": True})),
        # DEPRECATED spelling of `razor-2p`, kept because it shipped in the
        # named entry points and in antennaknobs' CLI roster.
        #
        # The rename is about the CLAIM the name makes, not about any host
        # mis-reading it: `razor-2p` describes the quadrature rule, where
        # `razor-nec5` names another vendor's product and implies a special
        # relationship the measurements do not support. The two lanes share a
        # limit with bspline rather than tracking NEC-5 in particular — on the
        # ByDipole1 ladder razor-2p and bs2 are 0.124 ohm apart at N=3072 and
        # closing as N^-0.73 — so this lane is SLOWER than bspline, not
        # specially agreeing with anything.
        #
        # No host is confused by the old spelling and none was claimed to be:
        # EZNEC takes its engine class and path as explicit fields and infers
        # nothing from the name, and SimNEC sniffs the path in the order nec2c,
        # nec5, nec42 — `momwire-nec2c-razor-nec5` carries both tokens and
        # `nec2c` wins ties, so it has always classified correctly.
        "razor-nec5": (RazorSolver, MappingProxyType({"nec5_quadrature": True})),
        # The pulse family, and ONE name for it (momwire#564), on momwire#654's
        # rule: a roster entry is a menu item, so it should be the dish worth
        # ordering. `HarringtonSolver` and `PulseSolver` are the same basis and
        # the same testing differing in one ingredient — where the charge
        # lives — and that ingredient is the whole difference between O(1/N)
        # convergence and an error governed by Delta/a. `PulseSolver`'s own
        # docstring says "reach for HarringtonSolver for anything but that
        # study"; a portal user picking from a dialog is not doing the study,
        # and has no way to say which they meant. The pair stays reachable as
        # a library import, which is where the study lives.
        "pulse": (HarringtonSolver, MappingProxyType({})),
    }
)

# Solver classes that take a lumped load as their own kwarg instead of the
# deck-level port-algebra route every other family shares (momwire#432).
# `build_solver` reads this to decide how to spell `feeds` / a load's
# translation; nothing else in the ROSTER keys off it.
#
# One consumer outside this module: the portal's power budget (momwire#433).
# "the fill already carries Z_L" and "the port algebra must not stamp Z_L" are
# the same fact asked from two directions, so the budget reads this tuple
# rather than keeping its own list of which bases load natively — a second
# list is a second thing to forget when a family is added here.
_NATIVE_LOADING = (RazorSolver,)


def basis_entry(basis: str) -> tuple[type, Mapping[str, Any]]:
    """:data:`BASES`\\ ``[basis]``, or a ``ValueError`` that lists what is known.

    Both dialect front ends resolve a ``--basis`` name and both owe the same
    sentence when it is not one; this is that one place (momwire#603 U3).
    """
    try:
        return BASES[basis]
    except KeyError:
        known = ", ".join(repr(name) for name in BASES)
        raise ValueError(f"unknown basis {basis!r}; known bases: {known}") from None


def basis_from_program_name(
    prog: str, marker: str, *, consumed: str | None = None
) -> str | None:
    """The basis a copy/symlink NAME selects, or ``None`` when it names none.

    ``<anything><marker><basis>`` — everything after ``marker`` in the
    program's basename, with a Windows ``.exe`` stripped first.  ``None`` —
    "this name asks for nothing, serve the default" — is returned for a name
    without the marker at all: ``python -m momwire.portal``, a pytest runner,
    a bare ``momwire-nec2c``.

    ``consumed`` is a program's own trailing segment (``"engine"`` for the
    frozen ``momwire-eznec-engine``), swallowed when it is the whole suffix
    and stripped when it leads one, so the program's plain name selects
    nothing and a renamed copy still names its basis.  The thin clients'
    stdlib-only copy (`momwire_serve_client.filename_basis`) spelt this
    first, for ``momwire-nec2c-shared``; one rule, restated at the owner.

    CASEFOLDED, marker and suffix together, because the names this reads are
    Windows FILENAMES: ``Momwire-Eznec-Razor-Nec5.exe`` is the same file as
    the lowercase spelling there, both front ends invite the user to copy and
    rename, and a marker that only matched one casing would hand that copy
    ``None`` and serve the default under a name that asked for the twin —
    momwire#628's failure reached by a rename.  ``marker`` must therefore be
    given casefolded; every basis in :data:`BASES` is lowercase already.

    A name ending AT the marker (``momwire-eznec-``) returns the empty string,
    not ``None``: it asked for a basis and supplied none, which is a typo to
    be refused by name, not a request for the default.  ``basis_entry`` is
    what says so, at each front end's own refusal channel.

    Validation is deliberately NOT done here.  The two front ends fail an
    unknown suffix in different places — the nec2c side at its `-version`
    probe, the EZNEC side as a printed refusal, because EZNEC reads the file
    and nothing else — and both want the sentence :func:`basis_entry` gives.

    One owner because there are now two markers (momwire#528 spelled
    ``nec2c-``, momwire#593 adds ``eznec-``) and the rule is not "what does
    each front end feel like doing", it is one rule about executable names.
    A second copy is the thing that drifts: momwire#628 was a basis resolved
    by one route and keyed by another, and the symptom was an engine silently
    answering as a formulation nobody asked for.
    """
    name = prog.replace("\\", "/").rsplit("/", 1)[-1].casefold()
    if name.endswith(".exe"):
        name = name[:-4]
    if marker not in name:
        return None
    suffix = name.split(marker, 1)[1]
    if consumed is not None:
        if suffix == consumed:
            return None
        if suffix.startswith(f"{consumed}-"):
            suffix = suffix[len(consumed) + 1 :]
    return suffix


def port_kwargs(
    solver_class: type,
    *,
    junctions: Any = None,
    node_gaps: Any = None,
    lumped_loads: Any = None,
) -> dict[str, Any]:
    """The port and topology kwargs ``solver_class`` takes, and only those.

    One owner for a rule the nec2 front end and the NEC-5 one both need and
    neither should keep a copy of (momwire#603 U3).  Every entry is a refusal
    by PRESENCE and not by value — ``RazorSolver`` raises on a ``node_gaps=``
    it was handed even when what it was handed is ``None`` — so what this
    decides is which KEYS exist, never what they hold.

    ``junctions`` is withheld from a :data:`_NATIVE_LOADING` family, and the
    reason is not that the constructor cannot take one: razor and harrington
    have both accepted a spec since momwire#590 step 3b — razor as a declared
    parameter, harrington by intercepting it out of ``**kwargs``.  It is that
    a DECLARED spec replaces geometric detection wholesale, and neither front
    end's list is the whole truth about the geometry.  Both filter to groups
    of two or more ends — ``_polylines`` at its ``len(ends) >= 2``, the NEC-5
    seam at its own — so both leave out the one-member group at a LONE
    GROUNDED wire end, which detection keeps as that end's own contact tent.
    Hand either list over and a base-fed monopole silently loses its ground.
    """
    kwargs: dict[str, Any] = {}
    if node_gaps:
        kwargs["node_gaps"] = node_gaps
    if lumped_loads:
        kwargs["lumped_loads"] = lumped_loads
    if not issubclass(solver_class, _NATIVE_LOADING):
        kwargs["junctions"] = junctions or None
    return kwargs


@dataclass(frozen=True)
class PortSite:
    """One gap in the structure, and what the deck put there.

    A site is a POSITION, not a card: a load stamped on a driven segment
    shares its feed's site, so ``feed`` and ``load`` are both set and the
    solver has one port for the two cards.  ``wire``/``arclength`` are in the
    MODEL's terms (an index into :attr:`DeckModel.wires`), not the mesh's —
    the mesh's polyline numbering is an implementation detail of the fill.
    """

    wire: int
    arclength: float
    feed: int | None = None
    load: int | None = None
    load_spec: LoadSpec | None = None
    # Whether a ``TL``/``NT`` endpoint landed here.  A network attaches to a
    # SEGMENT, and the admittance NEC's own network solve reads at that
    # segment is the one seen across a gap in it — so an endpoint cuts a gap
    # exactly the way a feed does, and shares the site when a feed or a load
    # already cut one there.  Last in the field order because the class is
    # public API.
    network: bool = False


@dataclass(frozen=True)
class PortPlan:
    """The map between a deck's ports and a solver's.

    Indices into :attr:`sites` ARE solver port indices: the same order
    ``compute_port_solution().y`` is in, and the same order the constructed
    solver's ``feeds`` list is in.  Node gaps follow the sites, which is
    momwire's own port order ([gap feeds…, junction ports…, node ports…]);
    :attr:`node_gap_ports` names those rows so a consumer never has to
    reconstruct the offset.

    That first sentence is a CLAIM, and :func:`prepare_mesh` cannot make it
    true on its own: it builds one plan per structure, before a basis has been
    chosen, and a :data:`_NATIVE_LOADING` basis gives a load-only site no row
    of its own.  :func:`in_solver_ports` is what makes it true — every plan
    reachable from a :class:`BuiltSolver` has been through it.  The plan
    :attr:`PreparedMesh.ports` hands out is the DECK's, where a site is a site
    whatever the basis turns out to be.
    """

    sites: tuple[PortSite, ...]
    # Solver port index per model feed / load / node gap, parallel to
    # DeckModel.feeds, .loads and .node_gaps.  A load entry is None when the
    # basis carries that load on the fill instead of at a port, which is the
    # one case a model card has no row behind it at all (momwire#588).
    feed_ports: tuple[int, ...]
    load_ports: tuple[int | None, ...]
    node_gap_ports: tuple[int, ...]
    # One drive vector over the full port set per execute group, None where
    # the model's group is None (an execute card that ran nothing).
    voltages: tuple[tuple[complex, ...] | None, ...]
    # ``(port a, port b)`` per DeckModel.networks entry, parallel to it: the
    # two solver ports one ``TL``/``NT`` card spans.  Empty for the ordinary
    # antenna-only deck, and defaulted (and last) so a positional construction
    # written against an earlier release keeps meaning what it meant.
    network_ports: tuple[tuple[int, int], ...] = ()

    @property
    def n_ports(self) -> int:
        """Rows in ``compute_port_solution().y``."""
        return len(self.sites) + len(self.node_gap_ports)

    def loaded_ports(self) -> tuple[tuple[int, LoadSpec], ...]:
        """``(solver port, spec)`` for every port carrying a load.

        The stamping consumer's whole input: everything else it needs is the
        admittance matrix those port indices address.
        """
        return tuple(
            (index, site.load_spec)
            for index, site in enumerate(self.sites)
            if site.load_spec is not None
        )


def in_solver_ports(plan: PortPlan, has_port: Sequence[bool]) -> PortPlan:
    """``plan`` renumbered onto the rows the SOLVER actually built.

    ``has_port[k]`` says whether plan site ``k`` became a row of the solver's
    own matrix.  It is False in exactly one case — a load-only site on a
    :data:`_NATIVE_LOADING` basis, whose ``LD`` the fill carries through
    ``lumped_loads`` instead — and True everywhere else, where this returns
    ``plan`` UNTOUCHED and costs nothing but the ``all()``.

    Renumbering rather than leaving the caller to translate is deliberate.
    A deck port index and a solver port index are the same integer for eight
    of the nine families in :data:`BASES` and differ for the ninth, so a
    consumer that mixes the two spaces is right on every deck it is likely to
    be tested against and wrong on the one that matters (momwire#439's
    ``IndexError``, momwire#588's dimension mismatch).  There is no way to
    read the difference off an index, so the fix is to stop having two kinds
    of index in circulation: past this function there is one space, the
    solver's, and :class:`PortPlan`'s docstring is true by construction.

    Node gaps keep their place after the sites — that is momwire's port order,
    not a choice made here — but the offset shrinks with the sites ahead of
    them.  A feed and a network endpoint always have a row (they are what
    ``has_port`` is built from), so only :attr:`PortPlan.load_ports` can come
    back carrying ``None``.
    """
    if all(has_port):
        return plan
    bridge: list[int | None] = []
    kept: list[int] = []
    for index, flag in enumerate(has_port):
        if flag:
            bridge.append(len(kept))
            kept.append(index)
        else:
            bridge.append(None)
    n_gaps = len(plan.node_gap_ports)

    def _feed(port: int) -> int:
        mapped = bridge[port]
        if mapped is None:  # pragma: no cover - has_port is built from these
            raise AssertionError(f"port {port} drives but has no solver row")
        return mapped

    return replace(
        plan,
        sites=tuple(plan.sites[index] for index in kept),
        feed_ports=tuple(_feed(port) for port in plan.feed_ports),
        load_ports=tuple(bridge[port] for port in plan.load_ports),
        node_gap_ports=tuple(len(kept) + k for k in range(n_gaps)),
        voltages=tuple(
            None
            if drive is None
            else tuple([drive[index] for index in kept] + list(drive[len(bridge) :]))
            for drive in plan.voltages
        ),
        network_ports=tuple(
            (_feed(port_a), _feed(port_b)) for port_a, port_b in plan.network_ports
        ),
    )


@dataclass(frozen=True)
class BuiltSolver:
    """A constructed solver and the plan that reads its ports.

    Returned rather than a bare solver because the ports are useless without
    the plan and the plan is meaningless without the solver: handing back one
    object keeps a consumer from pairing a plan with a solver built for a
    different frequency or basis, which would be silent and wrong.  The
    remaining fields are the choices this instance froze, recorded so a
    caller can key a cache on them without re-deriving them from the model.
    """

    solver: Any
    # The plan in the SOLVER's port space — `ports.n_ports` is the number of
    # rows `compute_port_solution().y` has, and every index in it addresses
    # one. This is `deck_ports` put through `in_solver_ports`, and it is the
    # plan every consumer of a built solver wants.
    ports: PortPlan
    # The plan in the DECK's, where a site is a site whatever basis reads it:
    # one entry per position the cards cut a gap at, which is what a message
    # about the deck has to count. The two are the same object for every
    # family but the natively-loading one (momwire#588).
    deck_ports: PortPlan
    # Solver port index per DECK site, or None where the site has no port in
    # the solver's own matrix — the bridge between the two plans above, built
    # beside the `feeds` comprehension it has to agree with rather than
    # re-derived by a consumer.
    site_to_solver_port: tuple[int | None, ...]
    basis: str
    frequency_mhz: float
    wavelength: float
    group: int | None
    extended_kernel: bool


def _first_armed_group(model: DeckModel) -> int | None:
    for index, group in enumerate(model.groups):
        if group is not None:
            return index
    return None


def _sites(model: DeckModel) -> tuple[list[PortSite], list[int], list[int]]:
    """The union port set: every feed, every load not already on one, and
    every network endpoint not already on either.

    Coincidence is decided on the model's own ``(wire, arclength)`` — the
    dialect front end resolved all three cards through the same addressing, so
    a load on a driven segment produces the SAME float, not a nearby one, and
    an equality test is the honest comparison rather than a tolerance whose
    width nothing would justify.  A network endpoint that lands on a driven or
    loaded segment SHARES that gap, first claimant naming it, which is the
    composition NEC itself performs: an ``LD`` sits inside the segment and an
    ``NT`` hangs off the same connection point.
    """
    sites: list[PortSite] = []
    at: dict[tuple[int, float], int] = {}
    feed_ports: list[int] = []
    load_ports: list[int] = []

    for index, (wire, arclength, _volts) in enumerate(model.feeds):
        key = (wire, arclength)
        if key in at:
            raise ValueError(
                f"feed {index} shares its position with feed "
                f"{sites[at[key]].feed} — one gap cannot carry two sources"
            )
        at[key] = len(sites)
        feed_ports.append(len(sites))
        sites.append(PortSite(wire=wire, arclength=arclength, feed=index))

    for index, (wire, arclength, spec) in enumerate(model.loads):
        key = (wire, arclength)
        existing = at.get(key)
        if existing is None:
            at[key] = len(sites)
            load_ports.append(len(sites))
            sites.append(
                PortSite(wire=wire, arclength=arclength, load=index, load_spec=spec)
            )
            continue
        site = sites[existing]
        if site.load is not None:
            raise ValueError(
                f"load {index} shares its position with load {site.load} — "
                f"one gap cannot carry two loads"
            )
        load_ports.append(existing)
        sites[existing] = PortSite(
            wire=site.wire,
            arclength=site.arclength,
            feed=site.feed,
            load=index,
            load_spec=spec,
        )

    for key in network_endpoints(model):
        existing = at.get(key)
        if existing is None:
            at[key] = len(sites)
            sites.append(PortSite(wire=key[0], arclength=key[1], network=True))
            continue
        site = sites[existing]
        sites[existing] = PortSite(
            wire=site.wire,
            arclength=site.arclength,
            feed=site.feed,
            load=site.load,
            load_spec=site.load_spec,
            network=True,
        )
    return sites, feed_ports, load_ports


def _wire_loading(materials) -> dict[str, np.ndarray]:
    """The per-wire conductivity / insulation arrays a solver takes.

    momwire's per-wire convention is one entry per wire with ``NaN`` for "not
    this one", and an array is only passed when SOMETHING in it is finite —
    an all-NaN array and no argument at all describe the same bare wire, and
    the second is what every solver was built against.
    """
    nan = float("nan")
    conductivity = np.array(
        [
            m.conductivity if m is not None and m.conductivity is not None else nan
            for m in materials
        ]
    )
    radius = np.array(
        [
            m.insulation_radius
            if m is not None and m.insulation_radius is not None
            else nan
            for m in materials
        ]
    )
    eps_r = np.array(
        [
            m.insulation_eps_r
            if m is not None and m.insulation_radius is not None
            else nan
            for m in materials
        ]
    )
    kwargs: dict[str, np.ndarray] = {}
    if np.isfinite(conductivity).any():
        kwargs["wire_conductivity"] = conductivity
    if np.isfinite(radius).any():
        kwargs["insulation_radius"] = radius
        kwargs["insulation_eps_r"] = eps_r
    return kwargs


def _lumped_loads(
    sites: tuple[PortSite, ...],
    ports: tuple[tuple[int, float], ...],
    freq_hz: float,
) -> list[tuple[int, float, complex]]:
    """``lumped_loads=[(wire, arclength, Z)]`` for a solver in
    :data:`_NATIVE_LOADING` (momwire#432).

    ``sites`` and ``ports`` are :attr:`PortPlan.sites` and
    ``Mesh.ports`` — parallel, both in SOLVER port order — so a load's
    position is read off the exact knot the mesh already placed for it,
    the same one the port-algebra siblings would load: ``to_polylines``
    forced a vertex there for every site (feed or load) before any basis
    saw the geometry, so there is no separate snapping to get wrong.  ``Z``
    is evaluated at ``freq_hz`` here rather than left symbolic because
    ``lumped_loads`` takes one impedance and a sweep already calls
    :func:`build_solver` once per step (see the module docstring's
    "translate once, fill many times" — an OPERATING POINT, unlike the
    geometry, is not reused across steps), so evaluating per call already
    is the swept behaviour, not an approximation of it.
    """
    return [
        (polyline, arclength, site.load_spec.impedance(freq_hz))
        for site, (polyline, arclength) in zip(sites, ports)
        if site.load_spec is not None
    ]


def _ground(environment: Environment) -> dict[str, Any]:
    """``ground_z`` / ``ground_eps`` / ``ground_model`` for one environment.

    Free space passes ``ground_z=None`` rather than omitting it: a solver
    reads "no plane" off that argument, and ``0.0`` would be a plane at the
    origin.  ``"finite-fast"`` passes no ``ground_model`` at all — the
    reflection-coefficient model is every solver's default, and a solve that
    names it must stay bit-identical to one that does not.

    The environment's second medium is not read here and never will be: it
    reaches an answer through a far-field request's cliff modes alone, and
    the moment method never sees it.
    """
    ground = environment.ground
    if ground is None:
        return {"ground_z": None}
    kwargs: dict[str, Any] = {"ground_z": float(environment.ground_z)}
    if ground == "pec":
        return kwargs
    if (
        isinstance(ground, tuple)
        and len(ground) == 3
        and ground[0]
        in (
            "finite",
            "finite-fast",
        )
    ):
        kwargs["ground_eps"] = (float(ground[1]), float(ground[2]))
        if ground[0] == "finite":
            kwargs["ground_model"] = "sommerfeld"
        return kwargs
    raise ValueError(f"unrecognised ground spec: {ground!r}")


@dataclass(frozen=True)
class PreparedMesh:
    """A model's geometry, translated once and replayable.

    Everything :func:`build_solver` does that depends on the GEOMETRY alone —
    the union port set, the chaining into polylines, the renumbering of the
    plan onto the mesh's port order — and nothing that depends on the
    operating point.  A sweep translates once and fills many times:
    ``build_solver(model, mesh=prepare_mesh(model), frequency_mhz=f)`` per
    step, where the unprepared call redoes the whole walk at every frequency.

    The reuse is by IDENTITY, not by equality: the polyline arrays a prepared
    mesh hands a solver are the same ``ndarray`` objects every time (no
    solver writes to them), so two solvers built from one handle see bitwise
    the same coordinates rather than two roundings of one computation.

    :attr:`model` is the model the handle was built from, kept so
    :func:`build_solver` can refuse a handle prepared for a different
    structure instead of silently answering the wrong deck.
    """

    model: DeckModel
    mesh: Mesh
    ports: PortPlan


def prepare_mesh(model: DeckModel) -> PreparedMesh:
    """Translate ``model``'s geometry once, for repeated :func:`build_solver`.

    The frequency, the kernel and the environment are all operating-point
    choices and none of them appears here; what the handle freezes is the
    structure, which no operating point can move.
    """
    sites, feed_ports, load_ports = _sites(model)
    mesh = to_polylines(model, tuple((site.wire, site.arclength) for site in sites))

    # The mesh decided the solver's port order; renumber the plan onto it so
    # a plan index and a Y row are the same integer.
    ordered = [sites[model_port] for model_port in mesh.port_order]
    solver_of_model = {model_port: i for i, model_port in enumerate(mesh.port_order)}
    feed_ports = [solver_of_model[p] for p in feed_ports]
    load_ports = [solver_of_model[p] for p in load_ports]
    node_gap_ports = [len(ordered) + k for k in range(len(model.node_gaps))]

    voltages: list[tuple[complex, ...] | None] = []
    for entry in model.groups:
        if entry is None:
            voltages.append(None)
            continue
        drive = [0j] * len(ordered)
        for feed_index, volts in enumerate(entry.voltages):
            drive[feed_ports[feed_index]] = volts
        drive += [complex(v) for _w, _v0, v in model.node_gaps]
        voltages.append(tuple(drive))

    at = {(site.wire, site.arclength): index for index, site in enumerate(ordered)}
    network_ports = tuple((at[card.end_a], at[card.end_b]) for card in model.networks)

    return PreparedMesh(
        model=model,
        mesh=mesh,
        ports=PortPlan(
            sites=tuple(ordered),
            feed_ports=tuple(feed_ports),
            load_ports=tuple(load_ports),
            node_gap_ports=tuple(node_gap_ports),
            voltages=tuple(voltages),
            network_ports=network_ports,
        ),
    )


def build_solver(
    model: DeckModel,
    *,
    basis: str = "bspline",
    group: int | None = None,
    frequency_mhz: float | None = None,
    extended_kernel: bool | None = None,
    environment: Environment | None = None,
    mesh: PreparedMesh | None = None,
    cancel: Any = None,
) -> BuiltSolver:
    """Construct a solver for ``model`` and return it with its port plan.

    ``basis`` is one of :data:`BASES` — the same seven names antennaknobs'
    ``--basis`` takes, so a deck solved through either front end can be asked
    for the same physics by the same word.

    ``group`` selects which execute group's frequency, extended-kernel
    setting and ENVIRONMENT the solver is built for; the default is the
    deck's first group that ran, and a model with no group at all falls back
    to :attr:`DeckModel.environment`.  ``frequency_mhz``, ``extended_kernel``
    and ``environment`` override that group's, which is what a sweep does —
    one plan, one geometry, a solver per frequency.

    ``mesh`` is a :func:`prepare_mesh` handle: pass one and the geometry
    translation is not repeated, which is the whole of the per-operating-point
    cost that does not belong to the fill.  Passing a handle prepared for a
    different model raises.

    Raises ``ValueError`` for an unknown basis, a model with no wires, and a
    port set the geometry cannot host.

    It does NOT refuse a basis whose `capabilities.centre_feeds` is False,
    though this function is the `nec2` seam and that dialect addresses
    segment CENTRES. `RazorSolver` is that basis, and the refusal is filed
    separately (from momwire#673) rather than landed here: every parsed nec2
    deck carries feeds -- the parser refuses a deck with no EX card -- so the
    gate would refuse razor for EVERY deck and retire the two shipped
    `momwire-nec2c-razor-*` console scripts with it. That is a user-facing
    decision, not a side effect of declaring a matrix cell. Until it is
    taken, `_snap_to_knot` still moves a centre-named gap half a cell here,
    and `RazorSolver.capabilities.centre_feeds` is where that is now written
    down.
    """
    solver_class, basis_kwargs = basis_entry(basis)

    if group is None:
        group = _first_armed_group(model)
    if group is not None:
        armed = model.groups[group]
        if armed is None:
            raise ValueError(f"execute group {group} ran nothing")
    else:
        armed = None

    if frequency_mhz is None:
        frequency_mhz = armed.frequencies[0] if armed and armed.frequencies else 0.0
    if frequency_mhz <= 0.0:
        raise ValueError(f"frequency must be > 0 MHz, got {frequency_mhz}")
    if extended_kernel is None:
        extended_kernel = bool(armed.extended_kernel) if armed else False
    if environment is None:
        environment = armed.environment if armed else model.environment

    if mesh is None:
        prepared = prepare_mesh(model)
    elif mesh.model is model or mesh.model == model:
        prepared = mesh
    else:
        raise ValueError(
            "the prepared mesh was built for a different model — a handle "
            "describes one structure and cannot be replayed onto another"
        )
    plan = prepared.ports
    built_mesh = prepared.mesh

    # The gap voltages the solver is CONSTRUCTED with are the selected
    # group's: a Y-matrix readout ignores them entirely (it enumerates ports),
    # and a direct solve wants exactly this group's drive.
    drive = plan.voltages[group] if group is not None else None

    def _voltage(index: int) -> complex:
        return complex(drive[index]) if drive is not None else 0j

    radii = list(built_mesh.radii)
    wire_radius = radii[0] if len(set(radii)) == 1 else radii

    node_gaps = [
        (polyline, end, complex(volts))
        for (polyline, end), (_w, _v, volts) in zip(
            built_mesh.node_gap_members, model.node_gaps
        )
    ]
    loads = None
    if issubclass(solver_class, _NATIVE_LOADING):
        # A load-only site needs no port of its own here: the fill carries
        # the Z_s bump directly through `lumped_loads`, at the exact knot
        # `_lumped_loads` reads off the mesh, so only genuine EX sites
        # become `feeds` — a load-only site would otherwise be a spurious
        # zero-volt source with no row to add.
        _has_port = [
            plan.sites[index].feed is not None or plan.sites[index].network
            for index in range(len(built_mesh.ports))
        ]
        loads = _lumped_loads(plan.sites, built_mesh.ports, frequency_mhz * 1e6)
    else:
        _has_port = [True] * len(built_mesh.ports)
    feeds = [
        (polyline, arclength, _voltage(index))
        for index, (polyline, arclength) in enumerate(built_mesh.ports)
        if _has_port[index]
    ]

    kwargs = port_kwargs(
        solver_class,
        junctions=[list(entry) for entry in built_mesh.junctions],
        node_gaps=node_gaps,
        lumped_loads=loads,
    )
    if extended_kernel:
        kwargs["extended_kernel"] = True

    solver = solver_class(
        wires=list(built_mesh.polylines),
        n_per_edge_per_wire=[list(counts) for counts in built_mesh.edge_elements],
        feeds=feeds,
        wavelength=_C_LIGHT / (frequency_mhz * 1e6),
        wire_radius=wire_radius,
        cancel=cancel,
        **kwargs,
        **_wire_loading(built_mesh.materials),
        **_ground(environment),
        **basis_kwargs,
    )
    # Renumber onto the rows that were just built, from the same flags the
    # `feeds` list was filtered by, so the plan cannot drift from the matrix
    # it describes.
    solver_plan = in_solver_ports(plan, _has_port)
    bridge: list[int | None] = []
    next_port = 0
    for flag in _has_port:
        bridge.append(next_port if flag else None)
        next_port += int(flag)

    return BuiltSolver(
        solver=solver,
        ports=solver_plan,
        deck_ports=plan,
        site_to_solver_port=tuple(bridge),
        basis=basis,
        frequency_mhz=float(frequency_mhz),
        wavelength=_C_LIGHT / (frequency_mhz * 1e6),
        group=group,
        extended_kernel=bool(extended_kernel),
    )
