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
it when the basis is ``"razor"`` or ``"razor-nec5"``.

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

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from ..array_block import ArrayBlockSolver
from ..bspline import BSplineSolver
from ..hmatrix import HMatrixSolver
from ..razor import RazorSolver
from ..sinusoidal import SinusoidalSolver
from ..sinusoidal_galerkin import SinusoidalGalerkinSolver
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

_C_LIGHT = 299_792_458.0

# The basis roster, spelled as antennaknobs' ``--basis`` names so one string
# selects the same physics on either side of the portal.  Six solver
# FAMILIES, nine entries: the extra three are a degree, a feed model and a
# quadrature rule, not new code paths.  "bspline" is the degree-2 B-spline —
# the default here as it is there.
#
# "razor" (momwire#432) is the NEC-5 formulation twin — see
# ``docs/razor-solver.md``. "razor-nec5" is the quadrature-lane variant
# (momwire#316): same class, `nec5_quadrature=True`, the same "one class,
# one extra kwarg" shape `bspline-d1` set for a degree axis. RazorSolver's
# translation differs from every sibling's in two ways `build_solver` has to
# know about (see below): it takes no `junctions` spec (detected from the
# geometry) and it serves a lumped load as its own `lumped_loads` kwarg
# rather than as a deck-level port-algebra site, which is why it is keyed by
# CLASS in `_NATIVE_LOADING` rather than threaded through `basis_kwargs`
# here — a future razor variant inherits the translation for free.
BASES = MappingProxyType(
    {
        "bspline": (BSplineSolver, MappingProxyType({})),
        "bspline-d1": (BSplineSolver, MappingProxyType({"degree": 1})),
        "hmatrix": (HMatrixSolver, MappingProxyType({})),
        "arrayblock": (ArrayBlockSolver, MappingProxyType({})),
        "sinusoidal": (SinusoidalSolver, MappingProxyType({})),
        "sinusoidal-galerkin": (SinusoidalGalerkinSolver, MappingProxyType({})),
        "sinusoidal-galerkin-converged": (
            SinusoidalGalerkinSolver,
            MappingProxyType({"feed_model": "point"}),
        ),
        "razor": (RazorSolver, MappingProxyType({})),
        "razor-nec5": (RazorSolver, MappingProxyType({"nec5_quadrature": True})),
    }
)

# Solver classes that take a lumped load as their own kwarg instead of the
# deck-level port-algebra route every other family shares (momwire#432).
# `build_solver` reads this to decide how to spell `feeds` / a load's
# translation; nothing else in the roster keys off it.
_NATIVE_LOADING = (RazorSolver,)


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


@dataclass(frozen=True)
class PortPlan:
    """The map between a deck's ports and a solver's.

    Indices into :attr:`sites` ARE solver port indices: the same order
    ``compute_port_solution().y`` is in, and the same order the constructed
    solver's ``feeds`` list is in.  Node gaps follow the sites, which is
    momwire's own port order ([gap feeds…, junction ports…, node ports…]);
    :attr:`node_gap_ports` names those rows so a consumer never has to
    reconstruct the offset.
    """

    sites: tuple[PortSite, ...]
    # Solver port index per model feed / load / node gap, parallel to
    # DeckModel.feeds, .loads and .node_gaps.
    feed_ports: tuple[int, ...]
    load_ports: tuple[int, ...]
    node_gap_ports: tuple[int, ...]
    # One drive vector over the full port set per execute group, None where
    # the model's group is None (an execute card that ran nothing).
    voltages: tuple[tuple[complex, ...] | None, ...]

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
    ports: PortPlan
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
    """The union port set: every feed, then every load not already on one.

    Coincidence is decided on the model's own ``(wire, arclength)`` — the
    dialect front end resolved both cards through the same addressing, so a
    load on a driven segment produces the SAME float, not a nearby one, and
    an equality test is the honest comparison rather than a tolerance whose
    width nothing would justify.
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

    return PreparedMesh(
        model=model,
        mesh=mesh,
        ports=PortPlan(
            sites=tuple(ordered),
            feed_ports=tuple(feed_ports),
            load_ports=tuple(load_ports),
            node_gap_ports=tuple(node_gap_ports),
            voltages=tuple(voltages),
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
    """
    try:
        solver_class, basis_kwargs = BASES[basis]
    except KeyError:
        known = ", ".join(repr(name) for name in BASES)
        raise ValueError(f"unknown basis {basis!r}; known bases: {known}") from None

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

    kwargs: dict[str, Any] = {}
    if built_mesh.node_gap_members:
        kwargs["node_gaps"] = [
            (polyline, end, complex(volts))
            for (polyline, end), (_w, _v, volts) in zip(
                built_mesh.node_gap_members, model.node_gaps
            )
        ]
    if extended_kernel:
        kwargs["extended_kernel"] = True

    if issubclass(solver_class, _NATIVE_LOADING):
        # A load-only site needs no port of its own here: the fill carries
        # the Z_s bump directly through `lumped_loads`, at the exact knot
        # `_lumped_loads` reads off the mesh, so only genuine EX sites
        # become `feeds` — a load-only site would otherwise be a spurious
        # zero-volt source with no row to add.  And this formulation takes
        # no `junctions` spec at all (not even `None`: it is a bare kwarg
        # name this constructor never declared), so the key stays out of
        # `kwargs` entirely rather than being set and refused.
        feeds = [
            (polyline, arclength, _voltage(index))
            for index, (polyline, arclength) in enumerate(built_mesh.ports)
            if plan.sites[index].feed is not None
        ]
        loads = _lumped_loads(plan.sites, built_mesh.ports, frequency_mhz * 1e6)
        if loads:
            kwargs["lumped_loads"] = loads
    else:
        feeds = [
            (polyline, arclength, _voltage(index))
            for index, (polyline, arclength) in enumerate(built_mesh.ports)
        ]
        kwargs["junctions"] = [list(entry) for entry in built_mesh.junctions] or None

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
    return BuiltSolver(
        solver=solver,
        ports=plan,
        basis=basis,
        frequency_mhz=float(frequency_mhz),
        wavelength=_C_LIGHT / (frequency_mhz * 1e6),
        group=group,
        extended_kernel=bool(extended_kernel),
    )
