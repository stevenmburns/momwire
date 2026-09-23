"""Distributed series wire impedance: conductor loss + insulation loading.

Real antenna wire is not PEC. Two per-unit-length series effects matter at
HF (stevenmburns/momwire#131):

* **Conductor loss** — the internal impedance of a round conductor,
  Z'_int(ω) [Ω/m]. Exact closed form for a solid cylinder of radius a and
  conductivity σ:

      k_c    = sqrt(jωμσ)                    (complex wavenumber in the metal)
      Z'_int = k_c / (2π a σ) · I₀(k_c a) / I₁(k_c a)

  which recovers both limits: DC (|k_c a| → 0) → 1/(σ π a²), and strong
  skin effect (|k_c a| ≫ 1) → (1+j)/(2π a σ δ) with δ = sqrt(2/(ωμσ)).
  The I₀/I₁ ratio is evaluated with scipy's exponentially *scaled* Bessels
  (`ive`), whose common scale factor cancels in the ratio — the unscaled
  I's overflow for a/δ ≳ 350.

* **Insulation loading** — a dielectric jacket (inner radius a, outer
  radius b, relative permittivity εr) stores extra E-field energy near the
  wire, which acts as a distributed series inductance (King's insulated-
  antenna theory, quasi-static limit):

      L'_ins = μ₀/(2π) · (1 − 1/εr) · ln(b/a)      [H/m]

  This slows the guided wave on the wire — the familiar few-percent
  "insulated wire tunes long" velocity-factor effect. Dielectric loss
  (tan δ) is deliberately out of scope for now.

  It is one half of a PAIR: the kernel sees the jacket through the
  equivalent radius a′ (`equivalent_radius`, momwire#865), which is where
  the jacket's effect on the CHARGE lives. Both halves are written against
  a free-space exterior. In a lossy exterior medium ε̃ (a BURIED wire) the
  inductance half is still exact — it only undoes a′ back to the metal's a,
  and no permittivity enters a flux linkage — but a′ is not: the wire
  misses a local ELASTANCE (momwire#1154)

      ΔS′ = ln(b/a) / (2π ε₀ εr) · (1 − 1/ε̃)      [m/F]

  which `jacket_elastance` returns, and `loading_for` serves as the
  charge-side coefficient ``zq = ΔS′/(jω)`` on every jacketed segment in
  the lower medium (zero, and structurally absent, everywhere else). A
  formulation tests it as it tests its own scalar-potential term: the
  Galerkin rows as ``zq · ∫ f_m′ f_n′ dl`` (`BSplineSolver._charge_gram`,
  and `SinusoidalGalerkinSolver._apply_loading` in closed form on its
  three-term shapes, momwire#1156), razor as the potential ``zq · q``
  differenced across the testing path (`RazorSolver._charge_stencil`).

Both enter the MoM as one per-wire series impedance Z'(ω) = Z'_int +
jωL'_ins, applied through each solver family's own testing scheme: the
Galerkin BSpline family loads Z over same-wire basis overlaps (see
`BSplineSolver._loading_gram`), while the point-matched SinusoidalSolver
applies NEC's impedance boundary condition at the segment-centre match
points (see `SinusoidalSolver._apply_loading`, momwire#134).

`wire_internal_impedance` and `insulation_inductance` are public momwire
exports (#133).

The spec layer (momwire#428)
----------------------------
Loading cannot share its MATRIX assembly across the formulations — the term
is a testing-side object (SG overlaps, sinusoidal collocates, razor
path-integrates, bspline spline-overlaps), the same reason the ground layer
is two objects. Everything UPSTREAM of the term is formulation-independent,
and since momwire#427 made `RazorSolver` the fourth consumer it lives here
once rather than four times:

* `configure_loading(solver, n_wires, ...)` — the kwarg normalisation, the
  unit conventions and the fail-fast validation, run from every
  constructor;
* `normalize_lumped_loads(value, n_wires)` — the same for the lumped-load
  sequence, for the formulations that serve one directly (razor today; the
  siblings reach the same physics as deck-level port algebra over a
  `node_gaps` port);
* `loading_for(solver, omega, geom=None)` — the ω-dependent read: Z'_w(ω)
  per wire, per SEGMENT when a geometry is given, and the lumped loads
  resolved to whatever index the formulation names a site by.

  The spec also carries the buried jacket's charge-side coefficient
  (``zq_wire`` / ``zq_seg``, momwire#1154), ω-dependent twice over: through
  1/(jω) and through the soil's ε̃(ω).

`loading_for` is on the correct side of every prepare/replay boundary by
construction: it takes ω as an argument and caches nothing, so a solver
whose k-independent half builds a stencil (razor) or a Gram (bspline) calls
it once per solved wavenumber and nothing about the skin effect leaks into
the geometry.

What each solver keeps is only its testing-idiom term assembly:
`BSplineSolver._loading_gram` / `_apply_loading` (basis overlaps),
`SinusoidalSolver._apply_loading` (the match-point boundary condition),
`SinusoidalGalerkinSolver._apply_loading` (the closed-form shape overlaps)
and `RazorSolver._loading_stencil` / `_apply_loading` (the testing-path
integral).
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import ive

from . import _ground_refl, _medium_spec, _wire_spec

MU0 = 1.25663706127e-6


def wire_internal_impedance(omega, radius, sigma):
    """Per-unit-length internal impedance of a round solid conductor [Ω/m].

    Exact for all skin depths (DC through strong skin effect). `omega` may
    be a scalar or an array; the result broadcasts accordingly.
    """
    omega = np.asarray(omega, dtype=float)
    if radius <= 0.0:
        raise ValueError(f"wire radius must be > 0, got {radius}")
    if sigma <= 0.0:
        raise ValueError(f"conductivity must be > 0 S/m, got {sigma}")
    kc = np.sqrt(1j * omega * MU0 * sigma)
    z = kc * radius
    # ive(v, z) = iv(v, z)·exp(-|Re z|): the scale factor is common to both
    # orders, so the ratio equals I0/I1 exactly without overflow.
    ratio = ive(0, z) / ive(1, z)
    return kc / (2.0 * np.pi * radius * sigma) * ratio


def insulation_inductance(radius, ins_radius, eps_r):
    """Distributed series inductance of a dielectric jacket [H/m]."""
    if ins_radius <= radius:
        raise ValueError(
            f"insulation_radius ({ins_radius}) must exceed the conductor "
            f"radius ({radius})"
        )
    if eps_r < 1.0:
        raise ValueError(f"insulation_eps_r must be >= 1, got {eps_r}")
    return MU0 / (2.0 * np.pi) * (1.0 - 1.0 / eps_r) * np.log(ins_radius / radius)


def jacket_elastance(radius, ins_radius, eps_r, eps_ext, eps0):
    """The local ELASTANCE a jacketed wire in an exterior medium is missing
    [m/F] (momwire#1154).

        ΔS′ = ln(b/a) / (2π ε₀ εr) · (1 − 1/ε̃)

    ``eps_ext`` is the exterior's relative permittivity ε̃ (complex for a
    lossy soil; scalar or array, it broadcasts), ``eps0`` the absolute
    permittivity it is relative to — the solver's own upper medium. Zero at
    ε̃ = 1, which is the statement that the free-space pair is exact in free
    space.

    Derivation (scratch/1154-jacket-in-soil/NOTES.md). Per unit length the
    coated wire's true inductance is set by the metal, μ₀/2π·ln(R/a), and
    its true elastance by the two dielectric shells in series,
    [ln(b/a)/εr + ln(R/b)/ε̃]/(2π ε₀). A bare wire of kernel radius a′ in the
    exterior has μ₀/2π·ln(R/a′) and ln(R/a′)/(2π ε₀ ε̃). With momwire's a′
    (`equivalent_radius`, the free-space choice) the inductance gap is
    exactly `insulation_inductance` — so that term is already right in any
    exterior — and the elastance gap is the expression above, independent of
    the far radius R. It is a CHARGE term, not a series one: ΔS′ multiplies
    q = −(1/jω)·dI/dl, so it cannot be folded into L′ (the issue's first
    reading, (1 − ε̃/εr) in L′, is the right L′ only for the in-medium
    radius a·(b/a)^(1 − ε̃/εr), which is complex in a lossy soil).
    """
    if ins_radius <= radius:
        raise ValueError(
            f"insulation_radius ({ins_radius}) must exceed the conductor "
            f"radius ({radius})"
        )
    if eps_r < 1.0:
        raise ValueError(f"insulation_eps_r must be >= 1, got {eps_r}")
    return (
        np.log(ins_radius / radius)
        / (2.0 * np.pi * eps0 * eps_r)
        * (1.0 - 1.0 / np.asarray(eps_ext, dtype=np.complex128))
    )


def equivalent_radius(radius, ins_radius, eps_r):
    """Popovic-Nesic EQUIVALENT RADIUS of a dielectric-coated wire [m].

        a' = a * (b/a) ** ((eps_r - 1) / eps_r),      a < a' < b

    Popovic & Nesic, "Generalisation of the Concept of Equivalent Radius of
    Thin Cylindrical Antennas", IEE Proc. 131 pt. H no. 3, 153-158 (1984).
    The quasi-static reading of a coated wire is a PAIR: the kernel sees a
    larger radius a' (which is where the coating's effect on the CHARGE, and
    so on the capacitance, comes from), and a distributed series inductance

        L = (mu0 / 2pi) * ln(a'/a)

    puts back the inductance that enlarging the radius would otherwise
    remove. `insulation_inductance` above is already exactly that L -- the
    identity `ln(a'/a) = ((eps_r - 1)/eps_r) * ln(b/a)` makes King's
    `(1 - 1/eps_r) * ln(b/a)` form and this one the same number -- so momwire
    has carried the inductance half of the pair since momwire#131 and not the
    radius half.

    THE TWO HALVES DO DIFFERENT JOBS, measured on the Severns surface deck
    (momwire#865) at N = 16, h = 1.6 mm, against 56.1 + 6.2j:

        bare a          46.62 + 17.96j     dR -9.48   dX +11.76
        a + L (today)   49.33 + 18.83j     dR -6.77   dX +12.63
        a' + L          49.38 + 14.93j     dR -6.72   dX  +8.73

    L raises R the right way and pushes X the WRONG way; the equivalent
    radius leaves R alone and pulls X down. The surface-radial residual is
    R-low / X-high, so the missing half is the one that fixes the half the
    inductance cannot.

    BOTH HALVES ASSUME A FREE-SPACE EXTERIOR. On a buried wire the pair
    keeps this radius and `insulation_inductance` (the inductance half stays
    exact) and adds the charge-side correction `jacket_elastance`
    (momwire#1154) — keeping a′ real and medium-independent is what lets the
    kernels, and every above-ground wire, stay untouched.

    The RADIUS this returns is for the KERNEL only. Everything about the
    metal -- skin-effect internal impedance, and any refusal that means "the
    conductor" -- keeps the real `radius`, which is why the spec layer stores
    both (`_conductor_radius_per_wire` beside `_radius_per_wire`).
    """
    if ins_radius <= radius:
        raise ValueError(
            f"insulation_radius ({ins_radius}) must exceed the conductor "
            f"radius ({radius})"
        )
    if eps_r < 1.0:
        raise ValueError(f"insulation_eps_r must be >= 1, got {eps_r}")
    return float(radius) * (float(ins_radius) / float(radius)) ** (
        (float(eps_r) - 1.0) / float(eps_r)
    )


def rlc_impedance(kind, r, l, c, omega):  # noqa: E741 — NEC's own field name
    """The impedance of an R/L/C triple at ``omega`` — NEC's own arithmetic.

    ``kind`` is ``"series"`` (R + jwL + 1/jwC) or ``"parallel"`` (their
    parallel combination).  UNITS ARE THE CALLER'S: the same three numbers
    spell a LUMPED impedance in ohms/henries/farads (``LD 0``/``LD 1``, via
    :meth:`momwire.deck.model.LoadSpec.impedance`) and a PER-METRE one in
    ohms/m, henries/m and farads·m (``LD 2``/``LD 3``, via
    :class:`DistributedRLC`).  One evaluator for both, because the zero
    convention below is the part that is easy to get subtly different and a
    second copy of it would eventually disagree with this one (momwire#1088).

    **A zero element drops out of the branch it is in** rather than dividing
    by zero: NEC reads an absent element as ABSENT, not as a short (series)
    or an open (parallel).  So a series triple with ``c == 0`` is R + jwL,
    and a parallel triple with ``l == 0`` is R ‖ (1/jwC).

    ``omega`` may be a scalar or an array; the result broadcasts.
    """
    if kind == "series":
        z = np.asarray(r, dtype=np.complex128) + 0j * np.asarray(omega, dtype=float)
        if l:
            z = z + 1j * omega * l
        if c:
            z = z + 1.0 / (1j * omega * c)
        return z
    if kind == "parallel":
        y = np.zeros(np.shape(omega), dtype=np.complex128)
        if r:
            y = y + 1.0 / r
        if l:
            y = y + 1.0 / (1j * omega * l)
        if c:
            y = y + 1j * omega * c
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / y
    raise ValueError(f"unknown RLC kind {kind!r}")


@dataclass(frozen=True)
class DistributedRLC:
    """A wire's DISTRIBUTED series impedance, spelled as an R/L/C triple
    per unit length (momwire#1088).

    Z'(w) [Ohm/m] is :func:`rlc_impedance` read in per-metre units, and it
    enters :func:`series_impedance_per_wire` beside the conductor's own
    internal impedance and the jacket's inductance — the three are added,
    because all three are the same kind of object: a per-metre series term
    the fill integrates along the wire.

    :attr:`kind` is ``"series"`` (NEC's ``LD 2``: Z' = R' + jwL' + 1/jwC')
    or ``"parallel"`` (``LD 3``: Y' = 1/R' + jwC' + 1/jwL', Z' = 1/Y').
    :attr:`r` is Ohm/m, :attr:`l` H/m; :attr:`c` is the per-metre
    capacitance IN THE SERIES-CHAIN SENSE, farad-metres, so that 1/(jwC')
    is Ohm/m like the rest of the triple.

    THAT LAST UNIT IS NOT WHAT NEC'S ``LD 2``/``LD 3`` CAPACITANCE FIELD
    MEANS, which is why both dialect readers refuse a nonzero one rather
    than converting it — see :meth:`momwire.deck._nec2._Nec2Parser._ld23`.
    The seam serves the general triple because a per-metre series
    capacitance is a well-defined distributed quantity; it is NEC's CARD
    that is not.
    """

    kind: str
    r: float = 0.0
    l: float = 0.0  # noqa: E741 — NEC's own field name for the inductance
    c: float = 0.0

    def __post_init__(self):
        if self.kind not in ("series", "parallel"):
            raise ValueError(
                f"distributed_rlc kind must be 'series' or 'parallel', "
                f"got {self.kind!r}"
            )
        if self.r < 0.0 or self.l < 0.0 or self.c < 0.0:
            raise ValueError(
                f"distributed_rlc R'/L'/C' must be >= 0, got "
                f"({self.r}, {self.l}, {self.c})"
            )
        if self.kind == "parallel" and self.r == 0.0 and self.l == 0.0:
            # 1/R' and 1/jwL' are the only two branches that can carry the
            # DC/low-frequency admittance; with both absent the parallel
            # combination is a pure capacitor and 1/Y' is finite, but with
            # all three absent Y' is 0 and Z' is infinite.  Caught here
            # rather than at the divide.
            if self.c == 0.0:
                raise ValueError(
                    "distributed_rlc 'parallel' with R' = L' = C' = 0 is an "
                    "open circuit (Y' = 0), not a load"
                )

    def impedance_per_metre(self, omega):
        """Z'(omega) [Ohm/m]; ``omega`` scalar or array."""
        return rlc_impedance(self.kind, self.r, self.l, self.c, omega)


def normalize_distributed_rlc(value, n_wires):
    """None | one spec (every wire) | length-n_wires sequence of None-or-spec.

    Returns None (off everywhere) or a length-``n_wires`` tuple whose entries
    are :class:`DistributedRLC` or None.  The scalar-or-sequence shape is
    :func:`normalize_per_wire`'s, and ``None`` in a sequence is what ``NaN``
    is there: this wire carries no distributed RLC.
    """
    if value is None:
        return None
    if isinstance(value, DistributedRLC):
        return tuple([value] * n_wires)
    entries = list(value)
    if len(entries) != n_wires:
        raise ValueError(
            f"distributed_rlc: expected one spec or a length-{n_wires} "
            f"sequence (one entry per wire), got {len(entries)} entries"
        )
    out = []
    for i, entry in enumerate(entries):
        if entry is None:
            out.append(None)
            continue
        if not isinstance(entry, DistributedRLC):
            raise ValueError(
                f"distributed_rlc[{i}]: expected a DistributedRLC or None, "
                f"got {entry!r}"
            )
        out.append(entry)
    if all(entry is None for entry in out):
        return None
    return tuple(out)


def normalize_per_wire(value, n_wires, name):
    """None | scalar | length-n_wires sequence → None | (n_wires,) float array.

    A scalar applies to every wire; None disables the effect. Entries of a
    sequence may be NaN to disable the effect on individual wires.
    """
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n_wires, float(arr))
    if arr.shape != (n_wires,):
        raise ValueError(
            f"{name}: expected a scalar or a length-{n_wires} sequence "
            f"(one entry per wire), got shape {arr.shape}"
        )
    return arr


def series_impedance_per_wire(
    omega,
    wire_radius,
    conductivity,
    insulation_radius,
    insulation_eps_r,
    distributed_rlc=None,
):
    """Per-wire distributed series impedance Z'(ω) [Ω/m].

    `wire_radius` is a scalar (every wire) or a length-n_wires sequence
    (each wire's own conductor radius — stevenmburns/momwire#147): the
    skin-loss internal impedance and the insulation-jacket inductance are
    both evaluated at wire w's own radius, through the SAME normaliser the
    constructors use (momwire#425 — this was the third, weaker copy of it:
    same message, no positivity check). `conductivity` /
    `insulation_radius` / `insulation_eps_r` are the normalized (n_wires,)
    arrays (or None) from `normalize_per_wire`; NaN entries switch the
    effect off for that wire. `distributed_rlc` is the normalized
    (n_wires,) tuple (or None) from `normalize_distributed_rlc`, whose
    None entries are the same "not this wire" — an `LD 2` / `LD 3`
    per-metre RLC (momwire#1088), EVALUATED HERE rather than folded in
    upstream because Z'(w) is frequency-dependent and a swept fill changes
    omega under it. `omega` may be scalar or (n_k,); the result is
    (n_wires,) or (n_wires, n_k) complex.

    The three terms ADD. They are three different per-metre series
    impedances on one conductor — the metal's own, the jacket's, and
    whatever the deck loaded the wire with — and nothing about them
    interacts, which is what lets `LD 2` and `LD 5` on the same wire
    compose rather than one overwriting the other.
    """
    omega = np.asarray(omega, dtype=float)
    n_w = (
        conductivity.shape[0]
        if conductivity is not None
        else insulation_radius.shape[0]
        if insulation_radius is not None
        else len(distributed_rlc)
    )
    radius, _uniform = _wire_spec.normalize_wire_radius(wire_radius, n_w)
    out = np.zeros((n_w,) + omega.shape, dtype=np.complex128)
    for w in range(n_w):
        if conductivity is not None and np.isfinite(conductivity[w]):
            out[w] += wire_internal_impedance(omega, radius[w], conductivity[w])
        if insulation_radius is not None and np.isfinite(insulation_radius[w]):
            L = insulation_inductance(
                radius[w], insulation_radius[w], insulation_eps_r[w]
            )
            out[w] += 1j * omega * L
        if distributed_rlc is not None and distributed_rlc[w] is not None:
            out[w] += distributed_rlc[w].impedance_per_metre(omega)
    return out


# --------------------------------------------------------------------------
# the spec layer (momwire#428): everything upstream of the testing rule
# --------------------------------------------------------------------------


def configure_loading(
    solver,
    n_wires,
    wire_conductivity,
    insulation_radius,
    insulation_eps_r,
    distributed_rlc=None,
):
    """Normalise and validate the house loading kwargs onto `solver`.

    Sets `wire_conductivity`, `insulation_radius`, `insulation_eps_r` — each
    None (off) or a normalized (n_wires,) float array with NaN switching a
    wire off — plus `distributed_rlc` (None, or a length-n_wires tuple of
    `DistributedRLC`-or-None, momwire#1088), and the `_loading_active`
    predicate every fill branches on.
    Reads `solver._radius_per_wire`, so the constructor must have normalized
    the radius first.

    Every check is a fail-fast: the same values are re-validated inside
    `insulation_inductance` at solve time, which is too late to name the
    caller's mistake. The order of the checks is part of the contract —
    `tests/test_wire_loading.py::test_validation_errors` pins the messages.
    """
    solver.wire_conductivity = normalize_per_wire(
        wire_conductivity, n_wires, "wire_conductivity"
    )
    solver.insulation_radius = normalize_per_wire(
        insulation_radius, n_wires, "insulation_radius"
    )
    solver.insulation_eps_r = normalize_per_wire(
        insulation_eps_r, n_wires, "insulation_eps_r"
    )
    solver.distributed_rlc = normalize_distributed_rlc(distributed_rlc, n_wires)
    if (solver.insulation_radius is None) != (solver.insulation_eps_r is None):
        raise ValueError(
            "insulation_radius and insulation_eps_r must be given together"
        )
    if solver.insulation_radius is not None:
        finite_b = np.isfinite(solver.insulation_radius)
        if not np.array_equal(finite_b, np.isfinite(solver.insulation_eps_r)):
            raise ValueError(
                "insulation_radius and insulation_eps_r must be finite "
                "on the same wires (NaN switches a wire off in both)"
            )
        for w in np.nonzero(finite_b)[0]:
            insulation_inductance(
                solver._radius_per_wire[w],
                solver.insulation_radius[w],
                solver.insulation_eps_r[w],
            )
    if solver.wire_conductivity is not None:
        for w in np.nonzero(np.isfinite(solver.wire_conductivity))[0]:
            if solver.wire_conductivity[w] <= 0.0:
                raise ValueError(
                    f"wire_conductivity[{w}] must be > 0 S/m, "
                    f"got {solver.wire_conductivity[w]}"
                )
    solver._loading_active = (
        solver.wire_conductivity is not None
        or solver.insulation_radius is not None
        or solver.distributed_rlc is not None
    )

    # THE OTHER HALF OF THE COATED-WIRE PAIR (momwire#865).
    #
    # `_radius_per_wire` becomes the KERNEL radius, a' on every jacketed wire.
    # `_conductor_radius_per_wire` keeps the metal's own a and is what the
    # loading (skin effect, and the jacket's own L) is evaluated at — see
    # `equivalent_radius` for why the pair is a pair.
    #
    # Done HERE rather than in each constructor so the four formulations
    # cannot disagree about what a coated wire's radius means; it is the same
    # argument that put the loading normalisation in this layer (momwire#428).
    solver._conductor_radius_per_wire = np.array(solver._radius_per_wire, dtype=float)
    if solver.insulation_radius is not None:
        kernel_radius = np.array(solver._radius_per_wire, dtype=float)
        for w in np.nonzero(np.isfinite(solver.insulation_radius))[0]:
            kernel_radius[w] = equivalent_radius(
                solver._conductor_radius_per_wire[w],
                solver.insulation_radius[w],
                solver.insulation_eps_r[w],
            )
        solver._radius_per_wire = kernel_radius
        # The scalar fast path is keyed on every wire sharing one radius, and
        # a jacket on SOME wires breaks that even when the conductors agreed.
        # Leaving a stale `_uniform_radius` would hand the single-`a` C++
        # kernels one wire's radius for the whole deck.
        if getattr(solver, "_uniform_radius", None) is not None:
            first = float(kernel_radius[0])
            solver._uniform_radius = (
                first if bool(np.all(kernel_radius == first)) else None
            )


def normalize_lumped_loads(value, n_wires):
    """`[(wire_index, arclength_or_None, impedance), ...]` → checked tuples.

    None is the empty list. The arc length is left as the caller wrote it —
    resolving a site to a basis index is the one part of loading that no
    shared layer can do, because what a formulation calls a site is a
    property of its basis (`loading_for` asks the solver for it).
    """
    if value is None:
        return []
    out = []
    for i, entry in enumerate(value):
        if len(entry) != 3:
            raise ValueError(
                f"lumped_loads[{i}]: expected "
                f"(wire_index, arclength, impedance), got {entry!r}"
            )
        w_i, arc_i, z_i = entry
        if not (0 <= w_i < n_wires):
            raise ValueError(
                f"lumped_loads[{i}]: wire_index {w_i} out of range [0, {n_wires})"
            )
        z_i = complex(z_i)
        if not np.isfinite(z_i.real) or not np.isfinite(z_i.imag):
            raise ValueError(f"lumped_loads[{i}]: impedance must be finite, got {z_i}")
        out.append((int(w_i), None if arc_i is None else float(arc_i), z_i))
    return out


class LoadingSpec:
    """What a fill needs to know about loading at one ω, naming no testing
    rule (momwire#428).

    * ``z_wire`` — ``(n_wires,)`` or ``(n_wires, n_k)`` complex Z'_w(ω)
      [Ω/m], or None when no distributed loading is configured. The form a
      term keyed by WIRE consumes (`BSplineSolver`'s Gram carries a wire id
      per entry).
    * ``z_seg`` — the same, gathered to ``(n_segs,)`` (or ``(n_segs, n_k)``)
      through the solver's segment→wire map; None when distributed loading
      is off or no geometry was given. The form a term keyed by SEGMENT
      consumes, which is the granularity every point-tested and
      path-tested fill indexes.
    * ``lumped`` — ``(indices, Z_L)`` for the configured lumped loads, or
      None. The index is whatever the formulation names a site by (razor: a
      basis index at a knot).
    * ``zq_wire`` / ``zq_seg`` — the buried jacket's CHARGE-side coefficient
      ``ΔS′/(jω)`` [Ω·m] (`jacket_elastance`, momwire#1154), per wire and
      gathered per segment, same shapes as ``z_wire`` / ``z_seg``. Nonzero
      only on jacketed wires in the lower medium, and None — structurally
      absent, not zero — whenever the deck has no such wire, so every
      above-ground and free-space fill is untouched. A consumer that serves
      a buried deck with a jacket must apply it (or refuse): it is not a
      series term and ``z_wire`` does not contain it.
    """

    __slots__ = ("z_wire", "z_seg", "lumped", "zq_wire", "zq_seg")

    def __init__(self, z_wire, z_seg, lumped, zq_wire=None, zq_seg=None):
        self.z_wire = z_wire
        self.z_seg = z_seg
        self.lumped = lumped
        self.zq_wire = zq_wire
        self.zq_seg = zq_seg


def _eps_tilde_at(ground_eps, omega, eps0):
    """`_ground_refl.eps_tilde` at a scalar ω or elementwise over an (n_k,)
    array — the same call per frequency, so a swept read equals the
    single-frequency reads bit for bit."""
    if np.ndim(omega) == 0:
        return _ground_refl.eps_tilde(ground_eps, float(omega), eps0)
    return np.array(
        [_ground_refl.eps_tilde(ground_eps, float(w), eps0) for w in omega],
        dtype=np.complex128,
    )


def buried_jacket_charge(solver, omega):
    """``ΔS′_w/(jω)`` per wire — ``(n_wires,)`` or ``(n_wires, n_k)`` — or
    None when no jacketed wire lies in the lower medium (momwire#1154).

    The exterior of a wire is decided by `solver._wire_media()`, the label
    every buried fill routes by: a wire is BELOW only when it lies strictly
    under the interface, so one RESTING on the plane (a jacketed surface
    radial at h ≥ b) is ABOVE, its exterior is air, and it gets nothing
    here. ε̃ is the fills' own: `_ground_refl.eps_tilde(ground_eps, ω,
    solver.eps)` at each solved ω.
    """
    if solver.insulation_radius is None:
        return None
    jacketed = np.isfinite(solver.insulation_radius)
    if not jacketed.any() or getattr(solver, "ground_eps", None) is None:
        return None
    # A formulation with no `_wire_media` has no buried fill at all (the
    # point-matched SinusoidalSolver refuses the `buried` cell at
    # construction), so every wire it solves is above the interface.
    media_fn = getattr(solver, "_wire_media", None)
    if media_fn is None:
        return None
    media = media_fn()
    below = [
        w for w in range(len(media)) if jacketed[w] and media[w] == _medium_spec.BELOW
    ]
    if not below:
        return None
    omega = np.asarray(omega, dtype=float)
    eps_t = _eps_tilde_at(solver.ground_eps, omega, solver.eps)
    out = np.zeros((len(media),) + omega.shape, dtype=np.complex128)
    a = solver._conductor_radius_per_wire
    for w in below:
        out[w] = jacket_elastance(
            a[w],
            solver.insulation_radius[w],
            solver.insulation_eps_r[w],
            eps_t,
            solver.eps,
        ) / (1j * omega)
    return out


def loading_for(solver, omega, geom=None):
    """The loading spec at `omega` — the shared read, once for four rows.

    `omega` may be a scalar or an (n_k,) array (a swept fill), and nothing
    is cached: this is the ω-dependent half of loading by construction, so
    a solver calls it inside the per-wavenumber fill and its k-independent
    half never sees a skin effect.

    Pass `geom` to get `z_seg` as well; it is gathered through
    `solver._wire_of_seg(geom)`. Lumped loads (`solver.lumped_loads`, absent
    on the formulations that refuse them) are resolved through
    `solver._lumped_site_index(geom, i, wire, arclength)` — the one step
    that cannot be shared, since a site index is a basis-layer fact.

    `omega` is the solve's REAL angular frequency. A buried fill's complex
    wavenumber times c is not one — loading is local and medium-independent
    — and handing one in is refused by name here rather than failing inside
    a float conversion (momwire#1156).
    """
    if np.iscomplexobj(omega):
        raise TypeError(
            f"loading_for takes the solve's REAL angular frequency, got a "
            f"complex omega ({omega!r}): wire loading is local and "
            f"medium-independent, so a buried fill reads it at the real omega, "
            f"never at its complex wavenumber times c (momwire#1156)"
        )
    z_wire = None
    if solver._loading_active:
        z_wire = series_impedance_per_wire(
            omega,
            # The CONDUCTOR radius, never the kernel one: skin effect is about
            # the metal, and the jacket's L is defined against a rather than
            # a' (momwire#865). `configure_loading` guarantees the attribute.
            solver._conductor_radius_per_wire,
            solver.wire_conductivity,
            solver.insulation_radius,
            solver.insulation_eps_r,
            solver.distributed_rlc,
        )
    z_seg = None
    if z_wire is not None and geom is not None:
        z_seg = z_wire[solver._wire_of_seg(geom)]
    lumped = None
    entries = getattr(solver, "lumped_loads", ())
    if entries:
        idx = [
            solver._lumped_site_index(geom, i, w, arc)
            for i, (w, arc, _z) in enumerate(entries)
        ]
        lumped = (
            np.asarray(idx, dtype=np.int64),
            np.asarray([z for _w, _a, z in entries], dtype=np.complex128),
        )
    zq_wire = buried_jacket_charge(solver, omega) if solver._loading_active else None
    zq_seg = None
    if zq_wire is not None and geom is not None:
        zq_seg = zq_wire[solver._wire_of_seg(geom)]
    return LoadingSpec(z_wire, z_seg, lumped, zq_wire, zq_seg)
