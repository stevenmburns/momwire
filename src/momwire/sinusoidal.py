"""NEC2-style sinusoidal-basis MoM for wires (Section III of the NEC2
Theory Manual, Burke & Poggio 1981 — see `docs/sinusoidal_basis_design.md`).

This is an OPTIONAL solver alongside `BSplineSolver` (the default). The
sinusoidal basis is what NEC2 / PyNEC / nec2c use; reproducing it in momwire
lets us isolate which parts of NEC's pulse-basis convergence behaviour are
intrinsic to the basis itself versus its kernel / source / junction
treatment.

Scope (deliberately narrow):
  * Free space, PEC-image ground (`ground_z`), NEC-style reflection-
    coefficient finite ground (`ground_z` + `ground_eps`, matching NEC's
    IPERF=0 approximation), or Sommerfeld/Norton finite ground
    (`ground_model="sommerfeld"`, the same C2-image + interpolated
    smooth-remainder decomposition BSplineSolver uses — see
    docs/sommerfeld-everywhere-plan.md Phase 2).
  * Thin-wire kernel (Eqs 73-79), no extended thin-wire / current-element.
  * Delta-gap "applied-E" source (Eq 187) on a single basis function.
  * Per-wire radius (stevenmburns/momwire#147): `wire_radius` is a scalar
    or a length-n_wires sequence. Mixed radii follow NEC2's per-segment
    convention — the basis end-condition constants use each segment's own
    radius (nec2c TBF), and the thin-wire kernel keeps the source current
    a filament on the source axis while enforcing the boundary condition
    on the OBSERVER segment's surface (EFLD's rh = sqrt(rho² + a_obs²)).
    The scalar-radius C++ field-tensor kernels serve mixed radii one
    constant-radius observer-row run at a time (`_radius_runs`).
  * Free wire ends use the X_i = 0 zero-current condition (the more
    physical J_1/J_0 end-cap condition is negligible for thin wires).
    A wire end lying IN an active ground plane is instead ground-
    connected (#151): NEC's tbf ground path — the plane side's P-sum
    gains the segment's own atom and the image-side extension folds
    back onto the segment with the sin term mirrored, so basis + image
    carry current continuously through the plane. Junctions AT the
    plane follow NEC's conect(): each member is ground-connected
    independently, with no inter-wire junction entries.
"""

import numpy as np
import scipy.linalg
import scipy.sparse

from . import _ground_refl, _sommerfeld, _wire_loading
from ._accel import acc as _acc
from ._cancel import _Cancelable

_HAVE_FIELD_TENSOR = _acc is not None and hasattr(_acc, "sinusoidal_field_tensor")
_HAVE_FIELD_TENSOR_REFL = _acc is not None and hasattr(
    _acc, "sinusoidal_field_tensor_refl"
)

_EULER_GAMMA = 0.5772156649015329

# Threshold for dense vs sparse assembly in `_assemble_Z`. Below this N
# the BLAS overhead on a tiny matrix loses to dense matmul; above it the
# O(N³) zgemm cost on a mostly-zero matrix loses to CSC sparse matmul.
# Measured crossover on Kaby Lake R / OpenBLAS-pthreads ≈ 60.
_DENSE_ASSEMBLY_THRESHOLD = 60


def _sin_minus_arg(u):
    """sin(u) − u to full relative precision.

    The literal difference loses everything below u²/6 relative, which is the
    whole answer for the segment half-angles this solver works at (u = kΔ/2 ≈
    1.9e-3 at N=801 on the half-wave dipole → ε·6/u² = 3.6e-10 relative). The
    Taylor series has no cancellation at all — every term is smaller than the
    last — so it is exact where the subtraction is worst; above u ≈ 0.1 the
    subtraction costs at most 6ε/u² = 1.3e-13 and the series would need more
    terms, so that is where the two swap.
    """
    u = np.asarray(u, dtype=float)
    u2 = u * u
    series = (
        -(u * u2)
        / 6.0
        * (1.0 - u2 / 20.0 * (1.0 - u2 / 42.0 * (1.0 - u2 / 72.0 * (1.0 - u2 / 110.0))))
    )
    return np.where(np.abs(u) < 0.1, series, np.sin(u) - u)


def _asinh_minus_arg(x):
    """asinh(x) − x to full relative precision (momwire#205).

    Substituting x = sinh t turns the answer into −(sinh t − t), whose series
    converges factorially — nine terms cover |t| ≤ 1 to below ε, where the
    asinh series itself would need ~28 — and t = asinh(x) is exactly what
    libm gives to a relative ε. Perturbing t by ε·t moves sinh t − t ≈ t³/6 by
    (t²/2)·εt = 3ε of itself, so routing through t costs nothing.

    Above |t| = 1 the plain subtraction is already accurate (the answer is
    within a factor 6 of sinh t there) and is used instead.
    """
    t = np.asarray(np.arcsinh(x), dtype=float)
    t2 = t * t
    series = (
        (t * t2)
        / 6.0
        * (
            1.0
            + t2
            / 20.0
            * (
                1.0
                + t2
                / 42.0
                * (1.0 + t2 / 72.0 * (1.0 + t2 / 110.0 * (1.0 + t2 / 156.0)))
            )
        )
    )
    return np.where(np.abs(t) < 1.0, -series, -(np.sinh(t) - t))


def _expm1_neg_j(w):
    """e^{−jw} − 1, spelled so neither part cancels: the real part is
    cos w − 1 = −2sin²(w/2), which the literal subtraction would compute to
    an absolute ε rather than a relative one."""
    half = np.sin(0.5 * w)
    return (-2.0 * half * half) - 1j * np.sin(w)


class SinusoidalSolver(_Cancelable):
    """NEC2's three-term (const + sin + cos) basis on each segment, with
    end-condition coefficients closed-form per Eqs 25-64.

    Constructor takes the same `wires` / `n_per_edge_per_wire` / `junctions`
    interface as `BSplineSolver` for drop-in comparison.

    `feed_model`
        Which SOURCE a delta-gap feed applies. Accepted for signature parity
        with `SinusoidalGalerkinSolver` (momwire#192), but `"segment"` — NEC's
        E_app = V/Δ_m over the whole feed segment, Eq 187 — is the only value
        this solver can carry, and `"point"` is refused with the derivation in
        `_reject_point_feed_model`. The short form: the feed-model axis is a
        property of the TESTING, not of the basis, because a zero-width gap
        has no collocation RHS at all, and under point matching the segment
        gap already IS the zero-width gap at the mesh's own resolution
        (momwire#212, report §17).
    """

    eps = 8.8541878188e-12
    mu = 1.25663706127e-6

    def __init__(
        self,
        *,
        wires,
        n_per_edge_per_wire=None,
        feed_wire_index=0,
        feed_arclength=None,
        feeds=None,
        feed_model="segment",
        wavelength=22,
        halfdriver_factor=0.962,
        wire_radius=0.0005,
        nsegs=101,
        ground_z=None,
        ground_eps=None,
        ground_model="refl-coef",
        n_qp_sommerfeld=3,
        junctions=None,
        junction_ports=None,
        n_qp_const=8,
        extended_kernel=False,
        wire_conductivity=None,
        insulation_radius=None,
        insulation_eps_r=None,
        cancel=None,
    ):
        if junction_ports:
            self._reject_junction_ports()
        if feed_model not in ("segment", "point"):
            raise ValueError(
                f"feed_model must be 'segment' or 'point', got {feed_model!r}"
            )
        if feed_model == "point":
            self._reject_point_feed_model()
        self.feed_model = feed_model
        # NEC's EK card (momwire#233). False — the default, and what every
        # momwire solver did before #233 — is NEC's reduced ("thin-wire")
        # kernel: the source current is a filament on the wire axis and the
        # only trace of the conductor's girth is the a² regularization of ρ
        # (`rho_eval` below). True is NEC's EXTENDED thin-wire kernel, Eqs
        # 84-98 of the theory manual: the current is a uniform tube of
        # surface current at ρ' = a and the kernel is its circumferential
        # average, expanded to O(a²). The two agree to O((a/R)²), which is
        # invisible at ordinary Δ/a and worth tens of percent below Δ/a ≈ 1.
        # NEC's `EK`/`EK 0` maps to True, `EK -1` to False.
        #
        # Cost: True routes the fill through the numpy reference kernel,
        # because the C++ accelerators transcribe EKSC only. Measured ~8.5×
        # the fill time at N=801 against the OpenMP path, and the numpy fill's
        # (N, N, n_qp_const) temporaries make large meshes memory-bound.
        self.extended_kernel = bool(extended_kernel)
        self._cached_ek_gating: tuple | None = None
        self._cancel = cancel
        self.wavelength = wavelength
        self.halfdriver_factor = halfdriver_factor
        self.wire_radius = wire_radius
        self.nsegs = nsegs
        self.ground_z = ground_z
        # Finite ground via NEC-style reflection-coefficient weighting of
        # the image field (docs/refl-coef-ground-plan.md Phase 6). None →
        # PEC image (today's behavior). A complex ε̃ or (eps_r, sigma)
        # tuple → Fresnel-weighted image; needs ground_z. Unlike
        # BSplineSolver there is deliberately NO `ground_phi_mode` knob:
        # the sinusoidal solver is field-based — Eqs 76-79 give the TOTAL
        # E-field of each source shape, vector- and scalar-potential
        # contributions already merged — so NEC's field dyad applies
        # exactly and no image-charge (Φ-term) weighting split exists to
        # approximate.
        if ground_eps is not None and ground_z is None:
            raise ValueError("ground_eps requires ground_z to be set")
        self.ground_eps = ground_eps
        # Finite-ground physics model, mirroring BSplineSolver: the
        # default "refl-coef" keeps every existing result bit-exact;
        # "sommerfeld" swaps the Fresnel field dyad for NEC's exact
        # decomposition — constant C2 = (eps-1)/(eps+1) on the PEC image
        # tensor (the C++ field-tensor kernel keeps serving it, a scalar
        # commutes with the projection) plus the smooth interpolated
        # remainder (`_field_tensor_sommerfeld_remainder`).
        if ground_model not in ("refl-coef", "sommerfeld"):
            raise ValueError(
                "ground_model must be 'refl-coef' or 'sommerfeld', "
                f"got {ground_model!r}"
            )
        if ground_model == "sommerfeld" and ground_eps is None:
            raise ValueError("ground_model='sommerfeld' requires ground_eps")
        self.ground_model = ground_model
        self.n_qp_sommerfeld = n_qp_sommerfeld

        self.c = 1 / np.sqrt(self.eps * self.mu)
        self.freq = self.c / self.wavelength
        self.omega = 2 * np.pi * self.freq
        self.k = self.omega / self.c
        self.eta = float(np.sqrt(self.mu / self.eps))
        self.halfdriver = self.halfdriver_factor * self.wavelength / 4
        # Gauss-Legendre nodes for the const-source self-integral are
        # k-independent; cache by n_qp so sweep loops don't pay for
        # repeated leggauss() calls.
        self._leggauss_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # `compute_impedance(...)` and `currents_at_knots(alpha)` are almost
        # always called as a pair from the UI; both internally rebuild geom
        # and basis-coefs from scratch (~5 ms/step on N=21 hentenna). The
        # geometry is purely a function of (wires, n_per_edge, junctions),
        # which are immutable after __init__; the basis-coefs are a function
        # of (geom, k, wire_radius). Cache both so the second call reuses
        # the work the first call already did. _basis_coefs validates the
        # cache by identity-checking the geom dict + value-comparing k and
        # a radius key (the scalar radius, or the per-wire array's bytes)
        # — so `compute_impedance_swept` (which mutates self.k in a loop)
        # still rebuilds basis-coefs per k, but reuses geom.
        self._cached_geometry: dict | None = None
        self._cached_basis: tuple[dict, float, float | bytes, list] | None = None
        # k-independent specular-ray tables for the `ground_eps` weighted
        # image (cos θ, p̂ components, tangent·p̂ projections), cached per
        # geometry object — same identity-check pattern as _cached_basis /
        # bspline's `_image_refl_prep`. ρ_v/ρ_h are NOT cached: they depend
        # on ε̃(ω) and are recomputed per frequency.
        self._cached_image_refl_prep: tuple | None = None

        if not wires:
            raise ValueError("wires must be non-empty")
        self.wires_polylines = [np.asarray(w, dtype=float) for w in wires]
        for i, pl in enumerate(self.wires_polylines):
            if pl.ndim != 2 or pl.shape[0] < 2 or pl.shape[1] != 3:
                raise ValueError(f"wire {i}: polyline must be (M, 3) with M >= 2")

        n_w = len(self.wires_polylines)

        # Per-wire conductor radius (stevenmburns/momwire#147): a scalar
        # applies to every wire; a length-n_wires sequence gives each wire
        # (polyline) its own radius, mapped to segments via _wire_of_seg.
        # `_uniform_radius` is the scalar fast path — it keeps the
        # historical scalar code paths (and the single-`a` C++ kernels)
        # bit-identical whenever all wires share one radius, including
        # when that radius arrived as a uniform array.
        radius = np.asarray(wire_radius, dtype=float)
        if radius.ndim == 0:
            radius = np.full(n_w, float(radius))
        elif radius.shape != (n_w,):
            raise ValueError(
                f"wire_radius: expected a scalar or a length-{n_w} sequence "
                f"(one entry per wire), got shape {radius.shape}"
            )
        if not np.all(np.isfinite(radius)) or np.any(radius <= 0.0):
            raise ValueError(
                f"wire_radius entries must be positive and finite, got {radius}"
            )
        self._radius_per_wire = radius
        self._uniform_radius = float(radius[0]) if np.all(radius == radius[0]) else None

        # Distributed series wire impedance (stevenmburns/momwire#131,
        # sinusoidal support #134): finite conductivity and/or a dielectric
        # jacket, same normalize/validate contract as BSplineSolver. This
        # solver is point-matched (NEC collocation), so the loading enters
        # as NEC's impedance boundary condition at the match points —
        # E_scat(n) − Z'_w·I(n) = −E_app(n) — a sparse subtraction of
        # Z'·(basis current at segment centre) in `_assemble_Z`, NOT the
        # Galerkin overlap the BSpline family uses. `wire_loss_power` DOES
        # use the closed-form ∫|I|² overlaps (a physical integral is basis-
        # scheme-independent).
        self.wire_conductivity = _wire_loading.normalize_per_wire(
            wire_conductivity, n_w, "wire_conductivity"
        )
        self.insulation_radius = _wire_loading.normalize_per_wire(
            insulation_radius, n_w, "insulation_radius"
        )
        self.insulation_eps_r = _wire_loading.normalize_per_wire(
            insulation_eps_r, n_w, "insulation_eps_r"
        )
        if (self.insulation_radius is None) != (self.insulation_eps_r is None):
            raise ValueError(
                "insulation_radius and insulation_eps_r must be given together"
            )
        if self.insulation_radius is not None:
            finite_b = np.isfinite(self.insulation_radius)
            if not np.array_equal(finite_b, np.isfinite(self.insulation_eps_r)):
                raise ValueError(
                    "insulation_radius and insulation_eps_r must be finite "
                    "on the same wires (NaN switches a wire off in both)"
                )
            for w in np.nonzero(finite_b)[0]:
                _wire_loading.insulation_inductance(
                    self._radius_per_wire[w],
                    self.insulation_radius[w],
                    self.insulation_eps_r[w],
                )
        if self.wire_conductivity is not None:
            for w in np.nonzero(np.isfinite(self.wire_conductivity))[0]:
                if self.wire_conductivity[w] <= 0.0:
                    raise ValueError(
                        f"wire_conductivity[{w}] must be > 0 S/m, "
                        f"got {self.wire_conductivity[w]}"
                    )
        self._loading_active = self.wire_conductivity is not None or (
            self.insulation_radius is not None
        )

        if n_per_edge_per_wire is None:
            n_per_edge_per_wire = [None] * n_w
        if len(n_per_edge_per_wire) != n_w:
            raise ValueError(
                f"n_per_edge_per_wire length {len(n_per_edge_per_wire)} "
                f"!= number of wires {n_w}"
            )

        self.n_per_edge_per_wire = []
        for i, (pl, npe) in enumerate(zip(self.wires_polylines, n_per_edge_per_wire)):
            n_edges_w = pl.shape[0] - 1
            if npe is None:
                npe = self.nsegs
            if np.isscalar(npe):
                npe = [int(npe)] * n_edges_w
            npe = list(npe)
            if len(npe) != n_edges_w:
                raise ValueError(
                    f"wire {i}: n_per_edge length {len(npe)} "
                    f"!= number of edges {n_edges_w}"
                )
            self.n_per_edge_per_wire.append(npe)

        if feeds is None:
            if not (0 <= feed_wire_index < n_w):
                raise ValueError(f"feed_wire_index {feed_wire_index} out of range")
            self.feeds = [(int(feed_wire_index), feed_arclength, 1.0 + 0.0j)]
        else:
            # feeds=[] is legal only when the model is driven entirely through
            # junction ports (issue #172's rule, mirrored from BSplineSolver).
            if len(feeds) == 0 and not junction_ports:
                raise ValueError("feeds must contain at least one entry")
            norm = []
            for i, f in enumerate(feeds):
                if len(f) != 3:
                    raise ValueError(
                        f"feeds[{i}]: expected (wire_index, arclength, voltage), got {f!r}"
                    )
                w_i, arc_i, v_i = f
                if not (0 <= w_i < n_w):
                    raise ValueError(
                        f"feeds[{i}]: wire_index {w_i} out of range [0, {n_w})"
                    )
                arc_i = None if arc_i is None else float(arc_i)
                norm.append((int(w_i), arc_i, complex(v_i)))
            self.feeds = norm

        self.feed_wire_index = self.feeds[0][0] if self.feeds else None
        self.feed_arclength = self.feeds[0][1] if self.feeds else None
        self.n_qp_const = n_qp_const

        self.junctions = []
        if junctions is not None:
            for j, jw in enumerate(junctions):
                # 1-entry groups are legal (issue #172's scope item 2): as a
                # non-port they emit no neighbour entries, so the member end
                # keeps the free-end branch and the solve is unchanged; as a
                # port they are the natural lone-conductor-end terminal.
                if len(jw) < 1:
                    raise ValueError(f"junction {j}: need >= 1 wire-end, got 0")
                normalized = []
                for w, end in jw:
                    if not (0 <= w < n_w):
                        raise ValueError(
                            f"junction {j}: wire_idx {w} out of range [0, {n_w})"
                        )
                    if end not in ("start", "end"):
                        raise ValueError(
                            f"junction {j}: end must be 'start' or 'end', got {end!r}"
                        )
                    normalized.append((int(w), end))
                self.junctions.append(normalized)

        self.junction_ports = self._normalize_junction_ports(junction_ports)

    def _reject_junction_ports(self):
        """Refuse `junction_ports=` on the point-matched solver.

        Not a plumbing gap — the basis excludes the port (issue #177, where
        the full derivation lives so it isn't redone). Junction continuity
        here is enforced INSIDE the sinusoidal basis via the N⁻/N⁺ neighbour
        tables, so every basis function satisfies KCL at every junction as an
        algebraic identity (measured residuals ~1e-13 on a 3-way star) — the
        span simply contains no current with nonzero net inflow at a node, at
        any mesh density. Relaxing a junction's neighbour entries yields I = 0
        free ends, and a KCL row added on top would enforce 0 = 0. A real port
        needs Galerkin (segment-integrated) test rows the entire
        point-collocation field stack doesn't provide — NEC-2 has the same
        limitation for the same reason (its EX/NT/TL port is a segment-centre
        delta gap; a node-localized EMF samples to a zero RHS).

        `SinusoidalGalerkinSolver` overrides this to a no-op so the port
        basis column CAN be built and measured there — but its solves refuse
        too, for a second and stronger reason #177 did not anticipate
        (momwire#182 M5: the port basis's node charge is priced at the
        wire radius by this family's field kernel). See that class.
        """
        raise NotImplementedError(
            "junction_ports are not supported on SinusoidalSolver: the "
            "sinusoidal basis enforces KCL identically, so a node-current "
            "port is outside its span (momwire#177; same limitation as "
            "NEC-2). Use BSplineSolver, or do what NEC requires: mesh a "
            "short bridge wire across the gap and gap-feed it"
        )

    def _reject_point_feed_model(self):
        """Refuse `feed_model="point"` on the point-matched solver.

        Not a plumbing gap and not an RHS formula nobody has written down yet
        — the pairing that RHS *is* does not exist. Collocation tests with
        δ(s − s_m), so row m's drive is ⟨δ_m, E_app⟩ = E_app(s_m), defined
        only where E_app is a FUNCTION. The zero-width gap's applied field is
        the distribution E_app = V·δ(s − s0), and δ·δ is not one: there is no
        number to put in the feed row, and 0 in every other row is the
        unexcited problem. `SinusoidalGalerkinSolver` carries the same source
        only because its pairing ⟨f_i, E_app⟩ tests against a continuous f_i,
        on which the delta collapses to the drive column −V·f_i(s0)
        (momwire#192). The feed-model axis is a property of the TESTING.

        Every regularization of the delta collapses into one of two dead ends.
        Replace δ by a nascent delta g_w of unit integral and width w, and
        sample: v_m = −V·g_w(s_m − s0). Measured on the canonical 0.962 λ/2
        dipole over N = 41…641 (momwire#212, report §17):

          **w ≪ h.** Every match point but the feed's sees ~0 while the feed
          row sees −V·g_w(0) ≈ −V/w, so the RHS is the segment-gap RHS scaled
          by h/w, and Z — homogeneous of degree −1 in the RHS — comes out
          (w/h)·Z_segment. Collocation reads the source's WIDTH off the feed
          row's amplitude and cannot tell a narrow gap from a small voltage,
          so a sub-mesh width is returned as a fraction of a volt. The only
          width this model already owns is the wire radius: the b → a limit of
          the magnetic frill, i.e. a delta ring of magnetic current at ρ = a,
          whose on-axis field E_z = (V a²/2)(1 + jkR)e^{−jkR}/R³ with
          R = √(d² + a²) is smooth, finite everywhere, integrates to V, and
          has no free parameter left (the textbook frill's entire content is
          its b/a — exactly the modeling parameter a zero-width gap may not
          introduce). Evaluated in this solver's own kernel convention — ring
          on the surface, observation on the axis, which reproduces the
          source-filament / BC-on-surface separation √(d² + a²) the fill
          already uses — it has w = 2a, and a/h ≤ 3.0e-2 across the whole
          ladder. Measured: the sampled RHS is the segment-gap RHS times
          h/2a to ≤2.8e-5 relative, and Z DOUBLES at every mesh doubling
          (0.270 → 0.533 → 1.059 → 2.112 → 4.218 Ω at N = 41…641) rather than
          stabilizing. Not the segment gap's log walk — a linear one.

          **w ≳ h.** The source is mesh-resolved and collocation is consistent
          again, but w is then a modeling parameter, and the smallest one the
          mesh resolves is w = h — which IS NEC's segment gap. Confirmed from
          the other side: the exact cell average of the zero-width field above
          reproduces the segment gap's −V/h to 6.0e-8…1.4e-6 of |Z| on the
          same ladder, four decades inside the (a/h)² bound.

        So under collocation the segment gap is not a rival source model to
        the point gap — it is the point gap, rendered at the only resolution
        collocation has. There is nothing left to opt into, which is why this
        is a refusal rather than a second default.

        A Hallén-style particular-solution forcing term is the one remaining
        escape and is rejected on scope: it needs the infinite-wire kernel's
        Fourier inversion to get J_p, J_p satisfies neither this basis's end
        conditions nor its junction tables, and moving the unknown changes
        what "the basis" means — which would break the one-axis-at-a-time
        discipline the whole instrument exists to keep.
        """
        raise NotImplementedError(
            "feed_model='point' is not supported on SinusoidalSolver: a "
            "zero-width gap has no collocation RHS — the drive is E_app "
            "sampled AT a match point and the source is a delta there, so "
            "the pairing is undefined (momwire#212). Under point matching "
            "NEC's segment gap already is the zero-width gap at the mesh's "
            "own resolution. For the zero-width source use "
            "SinusoidalGalerkinSolver(feed_model='point'), whose test "
            "integral is what makes a delta source admissible"
        )

    def _normalize_junction_ports(self, junction_ports):
        """Validate `junction_ports=` into a list of (junction_index, voltage).

        Same rules as `BSplineSolver` (issue #172): entries may be a plain
        int (voltage 0) or a (index, voltage) pair; indices must be in range
        and may not repeat. Returns [] when there are none, which is the only
        outcome the point-matched solver ever reaches — it raises in
        `_reject_junction_ports` first.
        """
        if not junction_ports:
            return []
        out = []
        seen = set()
        for p in junction_ports:
            j_idx, volt = (p, 0.0 + 0.0j) if np.isscalar(p) else p
            j_idx = int(j_idx)
            if not (0 <= j_idx < len(self.junctions)):
                raise ValueError(
                    f"junction_ports: junction index {j_idx} out of range "
                    f"[0, {len(self.junctions)})"
                )
            if j_idx in seen:
                raise ValueError(f"junction_ports: junction {j_idx} listed twice")
            seen.add(j_idx)
            out.append((j_idx, complex(volt)))
        return out

    def _leggauss_cached(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._leggauss_cache.get(n)
        if cached is not None:
            return cached
        gx, gw = np.polynomial.legendre.leggauss(n)
        gx = np.ascontiguousarray(gx, dtype=np.float64)
        gw = np.ascontiguousarray(gw, dtype=np.float64)
        self._leggauss_cache[n] = (gx, gw)
        return gx, gw

    # ------------------------------------------------------------------
    # Geometry build
    # ------------------------------------------------------------------

    def _build_geometry(self):
        """Discretize wires into segments and build the N^-/N^+ neighbor
        tables for every segment, with arc-flip σ signs.

        For a segment n at the head of a wire-segment sequence whose
        natural arc direction matches NEC's convention (segment's `end-2` =
        `seg_r` is at the junction node when treated as an N^- neighbour
        of another basis), σ = +1. When the natural tangent is reversed
        relative to NEC's expected arc, σ = -1.
        """
        if self._cached_geometry is not None:
            return self._cached_geometry
        # Build per-edge chunks (n_e segments per edge) in vectorized form,
        # then concatenate. Inner k_seg Python loop (171 iters × 5 list
        # appends + 4 numpy temporaries) was ~12% of py-spy samples at N=21.
        seg_l_chunks: list[np.ndarray] = []
        seg_r_chunks: list[np.ndarray] = []
        seg_c_chunks: list[np.ndarray] = []
        seg_t_chunks: list[np.ndarray] = []
        seg_h_chunks: list[np.ndarray] = []
        wire_first_seg: list[int] = []
        wire_last_seg: list[int] = []
        running_count = 0

        for w_idx, (pl, npe_list) in enumerate(
            zip(self.wires_polylines, self.n_per_edge_per_wire)
        ):
            wire_first = running_count
            for e_idx in range(pl.shape[0] - 1):
                p0 = pl[e_idx]
                p1 = pl[e_idx + 1]
                vec = p1 - p0
                edge_len = float(np.linalg.norm(vec))
                if edge_len < 1e-15:
                    raise ValueError(f"wire {w_idx} edge {e_idx} has zero length")
                tan = vec / edge_len
                n_e = npe_list[e_idx]
                h_e = edge_len / n_e
                # frac in [0, 1] sampled at n_e+1 points; consecutive points
                # bound each segment.
                frac = np.linspace(0.0, 1.0, n_e + 1)
                pts = p0 + frac[:, None] * vec  # (n_e+1, 3)
                pl_l_arr = pts[:-1]  # (n_e, 3)
                pl_r_arr = pts[1:]  # (n_e, 3)
                seg_l_chunks.append(pl_l_arr)
                seg_r_chunks.append(pl_r_arr)
                seg_c_chunks.append(0.5 * (pl_l_arr + pl_r_arr))
                seg_t_chunks.append(np.broadcast_to(tan, (n_e, 3)).copy())
                seg_h_chunks.append(np.full(n_e, h_e, dtype=np.float64))
                running_count += n_e
            wire_last_seg.append(running_count - 1)
            wire_first_seg.append(wire_first)

        seg_l = np.concatenate(seg_l_chunks, axis=0)
        seg_r = np.concatenate(seg_r_chunks, axis=0)
        seg_c = np.concatenate(seg_c_chunks, axis=0)
        seg_t = np.concatenate(seg_t_chunks, axis=0)
        seg_h = np.concatenate(seg_h_chunks)
        n_segs = seg_l.shape[0]

        # Per-segment N^- / N^+ neighbours — built directly as flat arrays
        # (nm_basis, nm_seg, nm_sigma and the np_ trio), where each entry
        # k is a single (basis i, neighbour seg j, σ) triple. No
        # list-of-lists intermediate, no per-seg Python append loop.
        #
        # nm_basis[k] = i  → basis i has a N⁻ neighbour at seg j
        #   nm_seg[k] = j     ("j's NEC end-2 coincides with i's end-1")
        # nm_sigma[k] = σ.   In-wire connections use σ = +1; junction
        # connections use ±1 per the L/R side rule below.
        nm_basis_chunks: list[np.ndarray] = []
        nm_seg_chunks: list[np.ndarray] = []
        nm_sigma_chunks: list[np.ndarray] = []
        np_basis_chunks: list[np.ndarray] = []
        np_seg_chunks: list[np.ndarray] = []
        np_sigma_chunks: list[np.ndarray] = []

        # In-wire neighbours: per wire, segs [first+1..last] each get nm
        # = (i-1, +1) and segs [first..last-1] each get np = (i+1, +1).
        # Two np.arange pairs per wire instead of an n_segs Python loop.
        for w_idx in range(len(self.wires_polylines)):
            first = wire_first_seg[w_idx]
            last = wire_last_seg[w_idx]
            if last > first:
                m = last - first
                nm_basis_chunks.append(np.arange(first + 1, last + 1, dtype=np.int64))
                nm_seg_chunks.append(np.arange(first, last, dtype=np.int64))
                nm_sigma_chunks.append(np.ones(m, dtype=np.int8))
                np_basis_chunks.append(np.arange(first, last, dtype=np.int64))
                np_seg_chunks.append(np.arange(first + 1, last + 1, dtype=np.int64))
                np_sigma_chunks.append(np.ones(m, dtype=np.int8))

        # Junctions AT the ground plane (#151): NEC's conect() connects a
        # ground-touching end to ground ONLY — it never searches for real
        # partners there (`icon1[i] = i; jump = TRUE` in nec2c). So a
        # grounded junction emits NO inter-wire neighbour entries: each
        # member is independently ground-connected (self-image basis), and
        # the members couple through the plane via their images.
        grounded_junctions = set()
        gz = self.ground_z
        if gz is not None:
            for j_i, jn in enumerate(self.junctions):
                w0, end0 = jn[0]
                pl0 = np.asarray(self.wires_polylines[w0], dtype=np.float64)
                pt = pl0[0] if end0 == "start" else pl0[-1]
                length0 = float(np.sum(np.linalg.norm(np.diff(pl0, axis=0), axis=1)))
                if abs(pt[2] - gz) <= 1e-6 * max(length0, 1e-30):
                    grounded_junctions.add(j_i)

        # Junction neighbours: small Python loop (junctions count is O(1)
        # in geometry size — 2-4 junctions on typical antennas, with K=2-6
        # members each producing K(K-1) edges). Append to per-junction
        # Python lists, then convert to numpy once.
        junc_nm_basis: list[int] = []
        junc_nm_seg: list[int] = []
        junc_nm_sigma: list[int] = []
        junc_np_basis: list[int] = []
        junc_np_seg: list[int] = []
        junc_np_sigma: list[int] = []
        for j_i, jn in enumerate(self.junctions):
            if j_i in grounded_junctions:
                continue
            # (segment_idx, which_end_of_segment_is_at_node) for every
            # wire-end at this junction.
            members = []
            for w, end in jn:
                if end == "start":
                    seg_idx = wire_first_seg[w]
                    end_side = "L"  # seg_l of this segment is at node
                else:
                    seg_idx = wire_last_seg[w]
                    end_side = "R"  # seg_r of this segment is at node
                members.append((seg_idx, end_side))
            # Every (i, j) pair with i != j contributes one edge from i's
            # perspective — N^- or N^+ depending on i's side at the node.
            for a_idx in range(len(members)):
                i_seg, i_side = members[a_idx]
                for b_idx in range(len(members)):
                    if b_idx == a_idx:
                        continue
                    j_seg, j_side = members[b_idx]
                    if i_side == "L":
                        # j is in N^-(i). σ = +1 if j's natural seg_r is at
                        # the node (j_side == "R"), else −1.
                        sigma = +1 if j_side == "R" else -1
                        junc_nm_basis.append(i_seg)
                        junc_nm_seg.append(j_seg)
                        junc_nm_sigma.append(sigma)
                    else:
                        # j is in N^+(i). σ = +1 if j's natural seg_l is at
                        # the node (j_side == "L"), else −1.
                        sigma = +1 if j_side == "L" else -1
                        junc_np_basis.append(i_seg)
                        junc_np_seg.append(j_seg)
                        junc_np_sigma.append(sigma)
        if junc_nm_basis:
            nm_basis_chunks.append(np.asarray(junc_nm_basis, dtype=np.int64))
            nm_seg_chunks.append(np.asarray(junc_nm_seg, dtype=np.int64))
            nm_sigma_chunks.append(np.asarray(junc_nm_sigma, dtype=np.int8))
        if junc_np_basis:
            np_basis_chunks.append(np.asarray(junc_np_basis, dtype=np.int64))
            np_seg_chunks.append(np.asarray(junc_np_seg, dtype=np.int64))
            np_sigma_chunks.append(np.asarray(junc_np_sigma, dtype=np.int8))

        # Per-feed segment index: for each feed, the segment on its wire
        # whose center is closest to the requested arclength (default:
        # midpoint of the wire). Primary feed kept as `feed_seg` for
        # back-compat with single-feed callers / fields.
        feed_segs = []
        for w_f, arc_req, _v in self.feeds:
            first = wire_first_seg[w_f]
            last = wire_last_seg[w_f]
            feed_h_w = seg_h[first : last + 1]
            feed_arc_centers = np.cumsum(feed_h_w) - 0.5 * feed_h_w
            total_arc = float(np.sum(feed_h_w))
            feed_arc = arc_req if arc_req is not None else 0.5 * total_arc
            feed_segs.append(
                first + int(np.argmin(np.abs(feed_arc_centers - feed_arc)))
            )
        # None with feeds=[] — legal only under junction-port drive (#172).
        feed_seg = feed_segs[0] if feed_segs else None

        # Concatenate the in-wire + junction chunks into the final flat
        # neighbour arrays. Empty geometries (e.g. a single-segment wire
        # with no junctions) get length-zero arrays of the right dtype.
        def _cat(chunks: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
            return np.concatenate(chunks) if chunks else np.empty(0, dtype=dtype)

        nm_basis = _cat(nm_basis_chunks, np.int64)
        nm_seg = _cat(nm_seg_chunks, np.int64)
        nm_sigma = _cat(nm_sigma_chunks, np.int8)
        np_basis = _cat(np_basis_chunks, np.int64)
        np_seg = _cat(np_seg_chunks, np.int64)
        np_sigma = _cat(np_sigma_chunks, np.int8)
        # Per-seg neighbour counts via bincount — no Python loop.
        nm_count = np.bincount(nm_basis, minlength=n_segs).astype(np.int64)
        np_count = np.bincount(np_basis, minlength=n_segs).astype(np.int64)

        # Ground-junction flags (#151): a wire endpoint lying in an active
        # ground plane is electrically connected to its own image — NEC2's
        # "connected to ground" condition. Mark the end segment's plane
        # side so _basis_coefs swaps the free-end X=0 branch for the
        # interior branch with a self-image P-sum atom. end-1 (seg_l at
        # the plane) is the N⁻ side, end-2 (seg_r) the N⁺ side.
        ground_minus = np.zeros(n_segs, dtype=bool)
        ground_plus = np.zeros(n_segs, dtype=bool)
        gz = self.ground_z
        if gz is not None:
            junctioned = set()
            for jn in self.junctions:
                for w, end in jn:
                    junctioned.add((w, end))
            for w_idx, pl in enumerate(self.wires_polylines):
                pl_arr = np.asarray(pl, dtype=np.float64)
                length = float(np.sum(np.linalg.norm(np.diff(pl_arr, axis=0), axis=1)))
                tol = 1e-6 * max(length, 1e-30)
                if float(pl_arr[:, 2].min()) < gz - tol:
                    raise ValueError(
                        f"wire {w_idx} dips below the ground plane "
                        f"(min z = {pl_arr[:, 2].min():.6g} < ground_z = {gz:g})"
                    )
                start_touch = abs(pl_arr[0, 2] - gz) <= tol
                end_touch = abs(pl_arr[-1, 2] - gz) <= tol
                if start_touch and (w_idx, "start") not in junctioned:
                    ground_minus[wire_first_seg[w_idx]] = True
                if end_touch and (w_idx, "end") not in junctioned:
                    ground_plus[wire_last_seg[w_idx]] = True
            # A segment lying IN the plane (both ends at gz) is degenerate
            # over a conducting ground — its image cancels it.
            in_plane = (np.abs(seg_l[:, 2] - gz) + np.abs(seg_r[:, 2] - gz)) <= (
                2e-6 * np.maximum(seg_h, 1e-30)
            )
            if in_plane.any():
                raise ValueError(
                    "segment(s) lying in the ground plane (both endpoints at "
                    "ground_z) — degenerate over a conducting ground"
                )
            # Grounded junction: per NEC's conect(), each member wire-end
            # is ground-connected INSTEAD of inter-connected (its junction
            # entries were skipped above) — the members couple through
            # their images.
            for j_i in grounded_junctions:
                for w, end in self.junctions[j_i]:
                    if end == "start":
                        ground_minus[wire_first_seg[w]] = True
                    else:
                        ground_plus[wire_last_seg[w]] = True

        self._cached_geometry = {
            "seg_l": seg_l,
            "seg_r": seg_r,
            "seg_centers": seg_c,
            "seg_tangents": seg_t,
            "seg_h": seg_h,
            "n_segs": n_segs,
            "nm_basis": nm_basis,
            "nm_seg": nm_seg,
            "nm_sigma": nm_sigma,
            "np_basis": np_basis,
            "np_seg": np_seg,
            "np_sigma": np_sigma,
            "nm_count": nm_count,
            "np_count": np_count,
            "wire_first": wire_first_seg,
            "wire_last": wire_last_seg,
            "feed_seg": feed_seg,
            "feed_segs": feed_segs,
            "ground_minus": ground_minus,
            "ground_plus": ground_plus,
            # Junctions whose node lies in the ground plane: their members are
            # ground-connected INSTEAD of inter-connected, so they emit no
            # neighbour entries. Published because a junction port cannot live
            # on one (its node voltage is pinned by the image) — see
            # `SinusoidalGalerkinSolver._junction_port_view`.
            "grounded_junctions": frozenset(grounded_junctions),
        }
        return self._cached_geometry

    # ------------------------------------------------------------------
    # Basis-function coefficient computation
    # ------------------------------------------------------------------

    def _basis_coefs(self, geom, k):
        """Per-basis closed-form (A, B, C, σ) coefficients on every
        supporting segment, following Eqs 43-64 of the NEC2 Theory Manual.

        Returns a CSR-by-segment `seg_view` dict:

            seg_view["starts"][s:s+2] → range of entries for segment s in
                the flat per-segment arrays below.
            seg_view["jbasis"][k]     → which basis contributes entry k.
            seg_view["A"/"B"/"C"][k]  → that basis's coefficient on seg.
            seg_view["AC"][k]         → A + C, pre-summed (see below).
            seg_view["sigma"][k]      → σ sign relative to NEC arc.

        `AC` is A + C — the entry's current at the segment CENTRE, and the
        coefficient the well-scaled shape set {1, sin kξ, cos kξ − 1} puts on
        the constant shape. It is published rather than left to each consumer
        because A ≈ −C to O((kΔ)²) on every entry type (self: A = −1 against
        C → 1; N± neighbour: C/A = −cos(kΔ/2)), so `σA + B·sin kξ + σC·cos kξ`
        evaluates a quantity of size ~(kΔ)²/8 from terms of size 1 — a
        relative error of ~ε·8/(kΔ)², which is 1.2e-10 at N=801 on the
        half-wave dipole and grows like N² (stevenmburns/momwire#203). Adding
        the two float64 coefficients here is correctly rounded *to the sum*,
        so every consumer that evaluates the basis gets the cancellation done
        once, in the one place where it is exact.

        Writing the entries directly into seg-major position during the
        main per-basis loop (instead of building a list-of-lists `basis`
        and then transposing it) avoids a second Python pass over ~1700
        entries — the path that was costing ~1.2 ms/step at N=21 hentenna.
        It also lets `_assemble_Z`'s flat-array fill become vectorized
        numpy (np.repeat + element-wise multiply) rather than a Python
        scatter loop.

        Reciprocity lets us compute `starts[]` upfront from the geometry
        alone, without a first pass to count: a segment s appears as a
        support entry of basis s itself (the self entry) plus once per
        basis i in `nm[s] ∪ np_[s]` — i.e. once for every neighbour of s.
        So entries_per_seg[s] = 1 + len(nm[s]) + len(np_[s]). Each basis
        i contributes 1 + len(nm[i]) + len(np_[i]) entries on its own
        support, so the *basis-major* count is the same value indexed
        differently. We only need the seg-major layout for downstream
        consumers, so that's all we materialise.
        """
        # Uniform radius keeps the historical scalar path (bit-exact);
        # mixed radii promote `a` to a per-segment array — each segment
        # inherits its wire's radius. The cache key is the scalar for the
        # uniform case, the radius array's bytes otherwise.
        if self._uniform_radius is not None:
            a = a_key = self._uniform_radius
        else:
            a = self._seg_radius(geom)
            a_key = self._radius_per_wire.tobytes()
        cached = self._cached_basis
        if (
            cached is not None
            and cached[0] is geom
            and cached[1] == k
            and cached[2] == a_key
        ):
            return cached[3]
        seg_h = geom["seg_h"]
        n_segs = geom["n_segs"]
        nm_basis = geom["nm_basis"]
        nm_seg = geom["nm_seg"]
        nm_sigma = geom["nm_sigma"]
        np_basis = geom["np_basis"]
        np_seg = geom["np_seg"]
        np_sigma = geom["np_sigma"]
        nm_count = geom["nm_count"]
        np_count = geom["np_count"]

        ka = k * a
        # a_i± from Eq 25: 1/(ln(2/(k·a)) − γ) per segment (scalar when the
        # radius is uniform). Mixed radii follow NEC2's TBF convention —
        # each segment's log-constant uses that segment's OWN radius: the
        # self formulas below (D, Q±, B_i0, C_i0 and the end branches) take
        # a_const at the basis's segment, while the P sums and the N±
        # neighbour coefficient entries take it at the NEIGHBOUR's segment
        # (nec2c computes aj from bi[jcox] for every connected segment and
        # resets aj = ap = the self constant before the Q/D formulas).
        a_const = 1.0 / (np.log(2.0 / ka) - _EULER_GAMMA)

        # Pre-compute every per-segment trig in one vectorized pass.
        kd_arr = k * np.asarray(seg_h, dtype=np.float64)
        sin_kd = np.sin(kd_arr)
        cos_kd = np.cos(kd_arr)
        sin_kd_2 = np.sin(0.5 * kd_arr)
        cos_kd_2 = np.cos(0.5 * kd_arr)
        # P-sum atoms: (1-cos(kd_j))/sin(kd_j) * a_const for N⁻; flip sign
        # for N⁺.
        P_minus_atom = (1.0 - cos_kd) / sin_kd * a_const

        # Per-basis P_minus[i] = Σ_{j ∈ N⁻(i)} atom[j], via scatter-sum on
        # the flat nm arrays. Same for P_plus[i] over N⁺.
        P_minus_arr = np.zeros(n_segs, dtype=np.float64)
        np.add.at(P_minus_arr, nm_basis, P_minus_atom[nm_seg])
        P_plus_arr = np.zeros(n_segs, dtype=np.float64)
        np.add.at(P_plus_arr, np_basis, -P_minus_atom[np_seg])

        # Ground junction (#151): an end at the ground plane is connected
        # to its own image — same length, same radius — so the plane side's
        # P-sum gains the segment's own atom and the segment takes the
        # interior (connected) branch instead of the free-end X=0 one. No
        # entry lands in the nm/np extension tables: the image side's
        # current is supplied by the ground image blocks, which mirror the
        # whole real expansion.
        ground_minus = geom["ground_minus"]
        ground_plus = geom["ground_plus"]
        if ground_minus.any():
            P_minus_arr = P_minus_arr + np.where(ground_minus, P_minus_atom, 0.0)
        if ground_plus.any():
            P_plus_arr = P_plus_arr - np.where(ground_plus, P_minus_atom, 0.0)

        # Per-basis (A_i0, B_i0, C_i0, Q_minus, Q_plus) as N-vectors,
        # following Eqs 43-64. The 4-way branch on (has_minus, has_plus)
        # is masked: compute the interior formula everywhere, then patch
        # the rare end / isolated branches via boolean masks. For a
        # hentenna (closed loop) every segment is interior; for a dipole
        # the wire-tip segments hit only_minus / only_plus.
        has_minus = (nm_count > 0) | ground_minus
        has_plus = (np_count > 0) | ground_plus
        both = has_minus & has_plus

        # Interior branch (Eqs 49-53). Compute everywhere using a_minus =
        # a_plus = a_const; mask out below where the branch doesn't apply.
        D = (P_minus_arr * P_plus_arr + a_const * a_const) * sin_kd + (
            P_minus_arr - P_plus_arr
        ) * a_const * cos_kd
        # Guard the denominator: replace 0 with 1 so the (masked-away)
        # bogus result is finite. Real divide-by-zero in the interior
        # branch would be a degenerate geometry (kd_i = nπ) we don't
        # support anyway.
        D_safe = np.where(D != 0, D, 1.0)
        sin_kd_safe = np.where(sin_kd != 0, sin_kd, 1.0)
        Q_minus_arr = (a_const * (1.0 - cos_kd) - P_plus_arr * sin_kd) / D_safe
        Q_plus_arr = (a_const * (cos_kd - 1.0) - P_minus_arr * sin_kd) / D_safe
        A_i0_arr = np.full(n_segs, -1.0)
        B_i0_arr = a_const * (Q_minus_arr + Q_plus_arr) * sin_kd_2 / sin_kd_safe
        C_i0_arr = a_const * (Q_minus_arr - Q_plus_arr) * cos_kd_2 / sin_kd_safe

        # Only-plus / only-minus / isolated branches: skip the work
        # entirely when none of them apply (the common case).
        only_plus = has_plus & ~has_minus
        only_minus = has_minus & ~has_plus
        iso = ~has_minus & ~has_plus
        if only_plus.any() or only_minus.any() or iso.any():
            # End segment with free end at end-1 (Eqs 54-57). X = 0.
            denom_x = cos_kd
            denom_x_safe = np.where(denom_x != 0, denom_x, 1.0)
            qpe1_denom = a_const * sin_kd - P_plus_arr * cos_kd
            qpe1_denom_safe = np.where(qpe1_denom != 0, qpe1_denom, 1.0)
            Q_plus_e1 = (cos_kd - 1.0) / qpe1_denom_safe
            B_e1 = (sin_kd_2 + a_const * Q_plus_e1 * cos_kd_2) / denom_x_safe
            C_e1 = (cos_kd_2 + a_const * Q_plus_e1 * sin_kd_2) / denom_x_safe

            # End segment with free end at end-2 (Eqs 58-61). X = 0.
            qme2_denom = a_const * sin_kd + P_minus_arr * cos_kd
            qme2_denom_safe = np.where(qme2_denom != 0, qme2_denom, 1.0)
            Q_minus_e2 = (1.0 - cos_kd) / qme2_denom_safe
            B_e2 = (-sin_kd_2 + a_const * Q_minus_e2 * cos_kd_2) / denom_x_safe
            C_e2 = (cos_kd_2 - a_const * Q_minus_e2 * sin_kd_2) / denom_x_safe

            # Isolated single segment (Eq 64). X = 0 → A = -1, B = 0,
            # C = 1/cos(kΔ/2).
            cos_kd_2_safe = np.where(cos_kd_2 != 0, cos_kd_2, 1.0)
            C_iso = 1.0 / cos_kd_2_safe

            Q_minus_arr = np.where(only_plus | iso, 0.0, Q_minus_arr)
            Q_minus_arr = np.where(only_minus, Q_minus_e2, Q_minus_arr)
            Q_plus_arr = np.where(only_minus | iso, 0.0, Q_plus_arr)
            Q_plus_arr = np.where(only_plus, Q_plus_e1, Q_plus_arr)
            B_i0_arr = np.where(only_plus, B_e1, B_i0_arr)
            B_i0_arr = np.where(only_minus, B_e2, B_i0_arr)
            B_i0_arr = np.where(iso, 0.0, B_i0_arr)
            C_i0_arr = np.where(only_plus, C_e1, C_i0_arr)
            C_i0_arr = np.where(only_minus, C_e2, C_i0_arr)
            C_i0_arr = np.where(iso, C_iso, C_i0_arr)
            # A_i0 = -1 in every branch — no patch needed.
        # `both` mask is unused: the interior formula already lives there.
        del both

        # Ground junction (#151), part 2: the connected-side EXTENSION.
        # For a real neighbour the Q-scaled Eqs 43-48 shape lands on the
        # neighbour segment; for a ground connection the "neighbour" is the
        # segment's own image, and nec2c's tbf folds that extension back
        # onto the segment itself with the sin term mirrored (s → −s, so
        # B → −B; `segj.bx[jsnox] = -segj.bx[jsnox]` in tbf). The folded
        # entry targets the same (basis, seg) pair as the self entry — and
        # downstream scatter is assignment, not add — so merge it into the
        # self coefficients. Basis + its image then forms the intended
        # continuous through-plane current.
        # (the ground "neighbour" is the segment itself, so the neighbour
        # a-constant is its own — works for scalar and per-segment alike)
        if ground_minus.any():
            Qg = np.where(ground_minus, Q_minus_arr, 0.0)
            A_i0_arr = A_i0_arr + a_const * Qg / sin_kd_safe
            B_i0_arr = B_i0_arr - a_const * Qg / (2.0 * cos_kd_2)
            C_i0_arr = C_i0_arr - a_const * Qg / (2.0 * sin_kd_2)
        if ground_plus.any():
            Qg = np.where(ground_plus, Q_plus_arr, 0.0)
            A_i0_arr = A_i0_arr - a_const * Qg / sin_kd_safe
            B_i0_arr = B_i0_arr - a_const * Qg / (2.0 * cos_kd_2)
            C_i0_arr = C_i0_arr + a_const * Qg / (2.0 * sin_kd_2)

        # Build the flat entry arrays. Three blocks (self, N⁻, N⁺), each
        # produced with one set of vectorized array ops:
        #
        # Self entries: basis = arange(n_segs), seg = arange(n_segs).
        self_seg = np.arange(n_segs, dtype=np.int64)

        # N⁻ neighbour entries (Eqs 43-45). Basis i contributes at seg j
        # (the j-side neighbour) using Q_minus[i] for the magnitude and the
        # NEIGHBOUR segment's a-constant (a_const[nm_seg], per TBF).
        a_nm = a_const if np.ndim(a_const) == 0 else a_const[nm_seg]
        a_np = a_const if np.ndim(a_const) == 0 else a_const[np_seg]
        nm_Q = Q_minus_arr[nm_basis]
        nm_A = a_nm * nm_Q / sin_kd[nm_seg]
        nm_B = a_nm * nm_Q / (2.0 * cos_kd_2[nm_seg])
        nm_C = -a_nm * nm_Q / (2.0 * sin_kd_2[nm_seg])

        # N⁺ neighbour entries (Eqs 46-48).
        np_Q = Q_plus_arr[np_basis]
        np_A = -a_np * np_Q / sin_kd[np_seg]
        np_B = a_np * np_Q / (2.0 * cos_kd_2[np_seg])
        np_C = a_np * np_Q / (2.0 * sin_kd_2[np_seg])

        # Concatenate the three blocks and sort by seg-target to get CSR.
        # Within-seg ordering is unconstrained — every downstream consumer
        # (M_{A,B,C} scatter assignment, I_feed reduction,
        # _evaluate_basis_at_points scatter-add) is order-invariant
        # because each (basis, seg) coordinate pair is unique (a basis
        # contributes to a given seg at most once: as self, as N⁻
        # neighbour, or as N⁺ neighbour, never two of those).
        all_seg = np.concatenate([self_seg, nm_seg, np_seg])
        all_basis = np.concatenate([self_seg, nm_basis, np_basis])
        all_A = np.concatenate([A_i0_arr, nm_A, np_A]).astype(np.complex128)
        all_B = np.concatenate([B_i0_arr, nm_B, np_B]).astype(np.complex128)
        all_C = np.concatenate([C_i0_arr, nm_C, np_C]).astype(np.complex128)
        # Self entries have σ=+1; neighbour σ comes from geometry.
        self_sigma = np.ones(n_segs, dtype=np.int8)
        all_sigma = np.concatenate([self_sigma, nm_sigma, np_sigma])

        order = np.argsort(all_seg, kind="stable")
        counts = np.ones(n_segs, dtype=np.int64) + nm_count + np_count
        starts = np.empty(n_segs + 1, dtype=np.int64)
        starts[0] = 0
        np.cumsum(counts, out=starts[1:])

        A_ord, C_ord = all_A[order], all_C[order]
        seg_view = {
            "starts": starts,
            "jbasis": all_basis[order],
            "A": A_ord,
            "B": all_B[order],
            "C": C_ord,
            "AC": A_ord + C_ord,
            "sigma": all_sigma[order],
        }
        self._cached_basis = (geom, k, a_key, seg_view)
        return seg_view

    def _evaluate_basis_at_points(self, seg_view, eval_seg, eval_s, alpha):
        """Vectorized evaluation of Σ_j α_j · f_{j, seg}(s_local) at an
        array of (segment, s_local) pairs.

        Replaces the per-knot `eval_at` Python closure: 342 individual
        calls per N=21 hentenna step (1.5 ms/step of Python frame +
        per-segment list iteration) collapse to one sin/cos call over
        n_eval points, one ragged gather, and one scatter-add.

        Parameters
        ----------
        seg_view : dict
            CSR-format inverse index from `_basis_coefs`.
        eval_seg : (n_eval,) int64
            Segment index per evaluation point.
        eval_s : (n_eval,) float64
            Local arc from each segment's centre.
        alpha : (n_basis,) complex128
            Basis amplitudes (from the EFIE solve).

        Returns
        -------
        (n_eval,) complex128
        """
        n_eval = eval_seg.shape[0]
        if n_eval == 0:
            return np.zeros(0, dtype=np.complex128)
        starts = seg_view["starts"]
        starts_at = starts[eval_seg]  # (n_eval,)
        lengths = starts[eval_seg + 1] - starts_at  # entries per eval
        n_entries = int(lengths.sum())
        if n_entries == 0:
            return np.zeros(n_eval, dtype=np.complex128)
        # Precompute trig at each eval point.
        sin_ks = np.sin(self.k * eval_s)
        cos_ks = np.cos(self.k * eval_s)
        # Ragged-gather expansion: for each eval i with `lengths[i]` entries,
        # produce that many entry-level rows. `entry_eval_idx` maps each row
        # back to its source eval; `entry_global` gathers from `seg_view`.
        entry_eval_idx = np.repeat(np.arange(n_eval, dtype=np.int64), lengths)
        # within-segment offset of each entry within its eval's block:
        # arange(n_entries) - cumulative-start-per-eval-block
        cum_starts = np.empty(n_eval, dtype=np.int64)
        cum_starts[0] = 0
        if n_eval > 1:
            np.cumsum(lengths[:-1], out=cum_starts[1:])
        within = np.arange(n_entries, dtype=np.int64) - np.repeat(cum_starts, lengths)
        entry_global = np.repeat(starts_at, lengths) + within
        jb = seg_view["jbasis"][entry_global]
        A_e = seg_view["A"][entry_global]
        B_e = seg_view["B"][entry_global]
        C_e = seg_view["C"][entry_global]
        sigma_e = seg_view["sigma"][entry_global]
        sin_e = sin_ks[entry_eval_idx]
        cos_e = cos_ks[entry_eval_idx]
        contrib = alpha[jb] * (sigma_e * A_e + B_e * sin_e + sigma_e * C_e * cos_e)
        I_out = np.zeros(n_eval, dtype=np.complex128)
        np.add.at(I_out, entry_eval_idx, contrib)
        return I_out

    # ------------------------------------------------------------------
    # Field of elementary current segments (Eqs 76-79)
    # ------------------------------------------------------------------

    def _field_tensor(self, geom, k, src_centers=None, src_tangents=None):
        """Tangential-field tensor Φ of shape (3, N, N) where
        Φ[0, m, n] = ŝ_m · E^const_n(at center of m's surface),
        Φ[1, m, n] = ŝ_m · E^sin_n(at center of m's surface),
        Φ[2, m, n] = ŝ_m · E^cos_n(at center of m's surface).

        The source's local frame is centered on segment n with z-axis
        along n's natural tangent. The "sin"/"cos" sources are
        sin(k·z'_local)/cos(k·z'_local) with z'_local measured from n's
        center along n's natural tangent. σ accounting is the caller's
        job — the tensor is in NATURAL-arc convention.

        `src_centers` / `src_tangents` default to the geometry's segment
        centers and tangents (free-space build). The PEC image build
        passes mirrored versions so the same tensor formula computes
        the image-source field at the original observer points.

        Hot path uses the C++ accelerator `sinusoidal_field_tensor` (the
        70% bottleneck of single-k solves at N≳80); the pure-numpy
        formulation (`_field_components` + the tangential projection
        below) is kept as a reference / fallback when the accelerator
        isn't available. The C++ kernel takes a single scalar radius —
        the OBSERVER segment's (necpp EFLD) — so mixed per-wire radii
        dispatch one call per constant-radius observer-row run
        (stevenmburns/momwire#147). The
        finite-ground image block bypasses this method — see
        `_field_tensor_image_refl`, which applies the Fresnel field dyad
        pre-projection through its own kernel
        (`sinusoidal_field_tensor_refl`).
        """
        seg_c = geom["seg_centers"]  # (N, 3) — observer centers
        seg_t = geom["seg_tangents"]  # (N, 3) — observer tangents
        seg_h = geom["seg_h"]  # (N,) full lengths

        src_c = src_centers if src_centers is not None else seg_c
        src_t = src_tangents if src_tangents is not None else seg_t

        # The C++ kernels transcribe EKSC only, and they are handed neither
        # the source radius nor any connectivity, so `extended_kernel=True`
        # takes the numpy reference path (momwire#233). Nothing about the
        # EK-OFF dispatch moves, which is what keeps the default bit-exact.
        if _HAVE_FIELD_TENSOR and not self.extended_kernel:
            gx, gw = self._leggauss_cached(self.n_qp_const)

            def _call(rows, a):
                return _acc.sinusoidal_field_tensor(
                    np.ascontiguousarray(seg_c[rows], dtype=np.float64),
                    np.ascontiguousarray(seg_t[rows], dtype=np.float64),
                    np.ascontiguousarray(src_c, dtype=np.float64),
                    np.ascontiguousarray(src_t, dtype=np.float64),
                    np.ascontiguousarray(seg_h, dtype=np.float64),
                    float(a),
                    float(k),
                    float(self.eta),
                    np.ascontiguousarray(gx, dtype=np.float64),
                    np.ascontiguousarray(gw, dtype=np.float64),
                    self._cancel_flag,
                )

            if self._uniform_radius is not None:
                return _call(slice(None), self._uniform_radius)
            # Mixed per-wire radii: the radius is the OBSERVER segment's
            # (necpp EFLD convention) and the C++ kernel takes one scalar,
            # so dispatch one call per contiguous constant-radius run of
            # observer rows and stitch — segments are wire-contiguous, so
            # runs are few (same pattern as the bspline kernels, #147).
            parts = [_call(slice(s, e), a) for s, e, a in self._radius_runs(geom)]
            return tuple(
                np.concatenate([p[i] for p in parts], axis=0) for i in range(3)
            )

        # numpy fallback: unprojected per-shape (E_z, E_ρ) components,
        # then project tangentially onto the observer:
        #   E_t = td · E_z + rho_proj · E_ρ  (NEC's ρ-projection rule).
        cm = self._field_components(geom, k, src_centers=src_c, src_tangents=src_t)
        td = cm["td"]
        rho_proj_factor = cm["rho_proj_factor"]
        Phi_const = td * cm["Ez_const"] + rho_proj_factor * cm["Erho_const"]
        Phi_sin = td * cm["Ez_sin"] + rho_proj_factor * cm["Erho_sin"]
        Phi_cos = td * cm["Ez_cos"] + rho_proj_factor * cm["Erho_cos"]
        return Phi_const, Phi_sin, Phi_cos

    # ---- NEC's extended thin-wire kernel (EK card), momwire#233 -----------
    #
    # Theory manual Eqs 84-98 (Burke & Poggio Part I, pp. 30-34). The reduced
    # kernel treats the segment current as a filament on the wire axis and
    # keeps the observer on the wire surface, so the source-observer distance
    # never falls below a; that is Eq 84's R = sqrt((z-z')² + a²), and it is
    # what every momwire solver computes. The EXTENDED kernel instead keeps
    # the current as a uniform tube of surface current at ρ' = a and averages
    # the free-space Green's function over the circumference (Eq 85). Eqs
    # 86-88 expand that average in a Taylor series about the axial filament
    # and truncate at second order, which is exact to O(a²/R²) and — crucially
    # — reintroduces the ρ' ≠ 0 terms the reduced kernel drops. Eq 89 is the
    # resulting scalar kernel and Eqs 90-98 are its z- and ρ-derivatives, the
    # six per-end quantities the field expressions consume.
    #
    # NEC's implementation of all of this is `GXX` (nec2-1.2.1.2.f:4857-4897),
    # which stands in for the reduced-kernel `GX` (f.4842-4852) at ONE END of
    # ONE SOURCE SEGMENT at a time, and `EKSCX` (f.3170-3234), which is
    # `EKSC` (f.3124-3166) with GX swapped for GXX per end plus a correction
    # to the constant-current term. momwire's `_field_components_bcast` is
    # algebraically EKSC rearranged into per-endpoint brackets (verified term
    # by term — see `test_extended_kernel_zero_radius_limit_is_the_reduced
    # _kernel`), so the extended kernel drops into the same slots.

    def _ek_gating(self, geom):
        """Per-source-segment extended-kernel end gating — NEC's IND1/IND2.

        NEC does NOT apply the extended kernel unconditionally. Once per
        SOURCE segment it asks, separately for each end, whether the segment's
        sinusoidal current really continues straight through that end into an
        identical conductor; only then is the O(a²) surface-current expansion
        legitimate there. The decision lives in the matrix fill, not in the
        kernel: nec2-1.2.1.2.f lines 2019-2053 (repeated verbatim at
        6197-6224 and 7217-7244 for the other two fill routines), guarded by
        `IF( IEXK.EQ.0) GOTO 16` at f.2019. The three codes it produces are
        consumed by EKSCX at f.3193 (`IF( INX1.EQ.2) GOTO 3`) and f.3199,
        which route the end to the reduced GX for code 2 and to the extended
        GXX for codes 0 and 1:

          IND = 1  Free end — ICONn(J) == 0 (f.2032 / f.2049). Nothing is
                   attached, so nothing can violate the expansion; NEC
                   extends.
          IND = 0  Either (a) a simple two-segment junction whose partner is
                   collinear to |t·t'| >= 0.999999 (f.2040-2041) AND of equal
                   radius to |a'/a - 1| <= 1e-6 (f.2042-2043), or (b) a
                   ground-plane contact, ICONn(J) == J, by a segment
                   perpendicular to the plane — NEC's CABJ²+SABJ² <= 1e-8
                   test at f.2026-2027 / f.2043-2044 — where the image
                   continues the wire straight through.
          IND = 2  Everything else: a bend, a radius step, a non-perpendicular
                   ground contact, or a multi-way junction. The last is
                   implicit: NEC threads 3+ segments meeting at a node onto a
                   circular ICON chain, so the reciprocal test at f.2025
                   (`IF(-ICON1(IPR).NE.J)`) / f.2031 (`IF(ICON2(IPR).NE.J)`)
                   fails for every member as soon as K >= 3.

        momwire's N⁻/N⁺ neighbour tables carry exactly this information. End 1
        is the N⁻ side and end 2 the N⁺ side (`nm_seg[k]` is documented as
        "j's NEC end-2 coincides with i's end-1", so the arc orientation
        matches NEC's ICON1/ICON2 convention). A neighbour count of exactly 1
        is precisely NEC's reciprocal-ICON test: momwire emits K(K-1) edges at
        a K-member junction, so K == 2 leaves count 1 on both members and
        K >= 3 leaves count >= 2 — the case NEC's chain rejects.

        Returns `(ind1, ind2)`, int8 arrays of shape (N,) holding NEC's codes.
        """
        geom_key = self._cached_ek_gating
        if geom_key is not None and geom_key[0] is geom:
            return geom_key[1], geom_key[2]

        n = geom["n_segs"]
        t = geom["seg_tangents"]  # (N, 3) unit tangents, NEC arc order
        rad = self._seg_radius(geom)  # (N,)
        # NEC's CABJ² + SABJ² is the squared HORIZONTAL projection of the
        # segment tangent (f.2026): the ground plane is z = const, so a
        # segment is perpendicular to it exactly when t_x² + t_y² vanishes.
        horiz = t[:, 0] * t[:, 0] + t[:, 1] * t[:, 1]
        perpendicular_to_plane = horiz <= 1e-8

        inds = []
        for count, basis, seg, ground in (
            (geom["nm_count"], geom["nm_basis"], geom["nm_seg"], geom["ground_minus"]),
            (geom["np_count"], geom["np_basis"], geom["np_seg"], geom["ground_plus"]),
        ):
            # Default IND = 2: reduced kernel at this end unless proven safe.
            ind = np.full(n, 2, dtype=np.int8)
            # Free end (f.2032): no neighbour entry AND no ground contact.
            ind[(count == 0) & ~ground] = 1
            # Ground contact (f.2026-2029): NEC's ICON == J self-connection.
            # momwire records it as a `ground_minus`/`ground_plus` flag and
            # emits no neighbour edge, so it never collides with the branch
            # below.
            ind[ground & perpendicular_to_plane] = 0
            # Simple two-segment junction: collinear AND equal radius.
            single = np.flatnonzero(count == 1)
            if single.size:
                # count == 1 means exactly one table entry names this basis,
                # so a last-write-wins scatter resolves the neighbour.
                nbr = np.empty(n, dtype=np.int64)
                nbr[basis] = seg
                j = nbr[single]
                # ABS() at f.2040: σ = ±1 records whether the neighbour's
                # natural tangent is arc-flipped, and NEC likewise compares
                # magnitudes, so an antiparallel collinear pair still counts.
                dot = np.abs(np.einsum("id,id->i", t[single], t[j]))
                ok = (dot >= 0.999999) & (np.abs(rad[j] / rad[single] - 1.0) <= 1e-6)
                ind[single[ok]] = 0
            inds.append(ind)

        self._cached_ek_gating = (geom, inds[0], inds[1])
        return inds[0], inds[1]

    @staticmethod
    def _ek_end_gxx(k, zz, rh, b, want_swapped):
        """NEC's GXX (f.4857-4897): extended-kernel segment-end contributions.

        `zz` is NEC's ZZ — the axial distance from the observer to this END of
        the source segment, signed as NEC signs it (Z2 = H - z, Z1 = -(H + z)).
        `rh` is the radial distance and `b` the radius entering the O(a²)
        terms; EKSCX has already ordered them so that `rh >= b` (see
        `_extended_kernel_fields`). `want_swapped` selects EKSCX's IRA == 1
        arm — the case where the observation point lies INSIDE the source
        conductor's radius and the roles of "distance" and "radius" trade
        places (f.4886-4896).

        Returns the six quantities EKSCX consumes, in NEC's argument order:
        `(G1, G1P, G2, G2P, G3, GZP)` — respectively the extended scalar
        kernel (Eq 89), its z-derivative, the ρ-kernel and its ρ- and
        z-derivatives (Eqs 90-96), and the reduced-kernel z-derivative that
        only the constant-current correction term uses.
        """
        # Multi-step spelling throughout (momwire#205): a one-expression
        # complex product with a dead operand changes rounding above numpy's
        # temporary-elision threshold, making the fill depend on block size.
        r2 = zz * zz + rh * rh
        r = np.sqrt(r2)
        kr = k * r
        kr2 = kr * kr
        # C1, C2, C3 of f.4869-4871 — the polynomial coefficients of the
        # second-order Taylor terms of Eqs 86-88.
        c1 = 1.0 + 1j * kr
        c2 = 3.0 * c1 - kr2
        c3 = (6.0 + 1j * kr) * kr2 - 15.0 * c1
        a2 = b * b
        # T1, T2 of f.4867-4868: the two O(a²) shape factors of Eq 89.
        rh2 = rh * rh
        r4 = r2 * r2
        t1 = 0.25 * a2 * rh2 / r4
        t2 = 0.5 * a2 / r2
        gz = np.exp(-1j * kr) / r  # reduced kernel e^{-jkR}/R
        # Eq 89: the circumferentially averaged kernel, ρ-flavoured (G2) and
        # z-flavoured (G1, which carries the extra -T2·C1 term).
        g2 = gz * (1.0 + t1 * c2)
        g1 = g2 - t2 * c1 * gz
        gzr = gz / r2
        g2p = gzr * (t1 * c3 - c1)
        gzp_t = t2 * c2 * gzr
        g3 = g2p + gzp_t
        g1p = g3 * zz
        # GZP: the plain reduced-kernel z-derivative. It is a-independent and
        # only enters EKSCX's constant-current term, scaled by (ka/2)².
        gzp_out = -zz * c1 * gzr
        if want_swapped:
            # IRA == 1 (f.4886-4896): observation point inside the conductor.
            t2b = 0.5 * b
            g2_out = -t2b * c1 * gzr
            g2p_s = t2b * gzr * c2 / r2
            g3_out = rh2 * g2p_s - b * gzr * c1
            g2p_out = g2p_s * zz
        else:
            # IRA == 0 (f.4879-4885), the ordinary case. `rh` is never zero in
            # momwire — it is at least the observer radius — so NEC's
            # RH < 1e-10 guard at f.4881 is unreachable here.
            g3_out = (g3 + gzp_t) * rh
            g2_out = g2 / rh
            g2p_out = g2p * zz / rh
        return g1, g1p, g2_out, g2p_out, g3_out, gzp_out

    @staticmethod
    def _ek_end_gx(k, zz, rhx):
        """NEC's GX (f.4842-4852) repackaged into GXX's six-value contract.

        EKSCX's `INX == 2` arm (f.3195-3201, f.3201-3207) calls the REDUCED
        end routine and then rescales its two outputs into the same six slots
        GXX fills. Note it passes RHX — the ORIGINAL radial distance — not the
        possibly-swapped RH, and it zeroes GZP so the end contributes nothing
        to the constant-current correction.
        """
        r2 = zz * zz + rhx * rhx
        r = np.sqrt(r2)
        kr = k * r
        c1 = 1.0 + 1j * kr
        gz = np.exp(-1j * kr) / r
        gp = -c1 * gz / r2
        g1p = gp * zz
        return gz, g1p, gz / rhx, g1p / rhx, gp * rhx, np.zeros_like(gz)

    def _extended_kernel_fields(self, k, H, z_eval, rho_eval, src_a, ind1, ind2):
        """NEC's EKSCX (f.3170-3234) — the extended-kernel field tables.

        Drop-in replacement for the reduced-kernel body of
        `_field_components_bcast`, returning the same six per-shape tables.
        `H` is the source HALF length, `z_eval` the observer's axial
        coordinate in the source frame, `rho_eval` the a-regularized radial
        distance (NEC's RHX out of EFLD), `src_a` the SOURCE segment's radius
        (NEC's BX = BI(J) — not the observer radius the reduced path
        regularizes with), and `ind1`/`ind2` the per-end gating codes from
        `_ek_gating`, broadcastable against the source axis.
        """
        # f.3186-3192: order the two lengths so RH is the larger. When the
        # observation point falls inside the source conductor the roles trade
        # and IRA is set; every downstream use of RH — including the INTX
        # integration radius and the (kB/2)² correction factor — sees the
        # ordered pair, not the raw one.
        rhx = rho_eval
        swap = rhx < src_a
        any_swap = bool(np.any(swap))
        rh = (
            np.where(swap, src_a, rhx) if any_swap else np.broadcast_to(rhx, swap.shape)
        )
        b = (
            np.where(swap, rhx, src_a)
            if any_swap
            else np.broadcast_to(src_a, swap.shape)
        )

        # NEC's Z2 = SH - Z and Z1 = -(SH + Z); momwire's dz_i is -Z_i.
        z2 = H - z_eval
        z1 = -(H + z_eval)
        ss = np.sin(k * H)
        cs = np.cos(k * H)

        ends = []
        for zz, ind in ((z1, ind1), (z2, ind2)):
            zz = np.broadcast_to(zz, swap.shape)
            ext = np.broadcast_to(ind != 2, swap.shape)
            q_ext = self._ek_end_gxx(k, zz, rh, b, any_swap)
            if ext.all():
                ends.append(q_ext)
                continue
            q_red = self._ek_end_gx(k, zz, rhx)
            ends.append(tuple(np.where(ext, e, r) for e, r in zip(q_ext, q_red)))
        (g1_1, g1p_1, g2_1, g2p_1, g3_1, gzz_1) = ends[0]
        (g1_2, g1p_2, g2_2, g2p_2, g3_2, gzz_2) = ends[1]

        # CON of f.3181 — NEC's DATA CONX/0., 4.771341189/ is jη/(4πk) with
        # NEC's k = 2π. Identical to `_field_components_bcast`'s `pref_z`.
        con = 1j * self.eta / (4.0 * np.pi * k)

        # f.3213-3222, verbatim. These are algebraically the same brackets the
        # reduced path spells per endpoint; only the G-quantities changed.
        ez_sin = con * ((g1_2 - g1_1) * cs * k - (g1p_2 + g1p_1) * ss)
        ez_cos = -con * ((g1_2 + g1_1) * ss * k + (g1p_2 - g1p_1) * cs)
        erho_sin = -con * (
            (z2 * g2p_2 + z1 * g2p_1 + g2_2 + g2_1) * ss
            - (z2 * g2_2 - z1 * g2_1) * cs * k
        )
        erho_cos = -con * (
            (z2 * g2p_2 - z1 * g2p_1 + g2_2 - g2_1) * cs
            + (z2 * g2_2 + z1 * g2_1) * ss * k
        )
        erho_const = con * (g3_2 - g3_1)

        # f.3223-3227: the constant-current term. INTX integrates e^{-jkR}/R
        # along the segment at the ORDERED radius RH, and the extended kernel
        # scales that integral by (1 - (kB/2)²) while adding back a (kB/2)²
        # weighted reduced-kernel end difference — the O(a²) piece of Eq 98
        # that the ρ-expansion of the axial term leaves behind. Unlike the
        # per-end substitutions this factor is NOT gated: it applies whenever
        # EK is on, even when both ends fell back to GX.
        u2 = (H - z_eval) / rh
        u1 = (-H - z_eval) / rh
        int_inv_r0 = np.arcsinh(u2) - np.arcsinh(u1)
        gx, gw = self._leggauss_cached(self.n_qp_const)
        z_qp = H[..., None] * gx
        dz_qp = z_eval[..., None] - z_qp
        r0_qp = np.sqrt(rh[..., None] ** 2 + dz_qp**2)
        G0_qp = np.exp(-1j * k * r0_qp) / r0_qp
        reg_qp = G0_qp - 1.0 / r0_qp
        int_G0 = int_inv_r0 + np.einsum("...q,q->...", reg_qp, gw) * H
        bk = k * b
        bk2 = 0.25 * bk * bk
        ez_const = -con * (
            g1p_2 - g1p_1 + k * k * (1.0 - bk2) * int_G0 - bk2 * (gzz_2 - gzz_1)
        )
        return {
            "Erho_const": erho_const,
            "Ez_const": ez_const,
            "Erho_sin": erho_sin,
            "Ez_sin": ez_sin,
            "Erho_cos": erho_cos,
            "Ez_cos": ez_cos,
        }

    def _field_components(
        self,
        geom,
        k,
        src_centers=None,
        src_tangents=None,
        obs_centers=None,
        obs_tangents=None,
        obs_radius=None,
    ):
        """Pure-numpy unprojected field tables behind `_field_tensor`'s
        fallback path (Eqs 76-79 of the NEC2 Theory Manual).

        Observers default to the geometry's segment centres (collocation).
        A Galerkin test caller (`SinusoidalGalerkinSolver`) overrides
        `obs_centers` / `obs_tangents` / `obs_radius` to evaluate the same
        closed-form source fields at Gauss quadrature points ALONG each test
        segment; the source side is unaffected (still the geometry's segments
        unless `src_*` is overridden by an image build). With M observers and
        N sources every returned table is (M, N).

        For each (observer m, source n) pair and each of the three source
        current shapes (const / sin / cos), computes the field scalars in
        the SOURCE's local cylindrical frame: E = E_z·t_src + E_ρ·ρ̂ with
        ρ̂ = rho_vec/rho_eval. Returns a dict of (M, N) tables:

            Erho_const, Ez_const, Erho_sin, Ez_sin, Erho_cos, Ez_cos —
                per-shape field scalars;
            td              : t_obs · t_src (E_z projection factor);
            rho_proj_factor : (rho_vec · t_obs)/rho_eval (E_ρ projection);
            rho_vec         : (M, N, 3) perpendicular from source axis to
                              the observer point;
            rho_eval        : radius-regularized |rho_vec|, ≥ a.

        `_field_tensor` consumes td/rho_proj_factor for the plain
        tangential projection; `_field_tensor_image_refl` also needs
        rho_vec/rho_eval to resolve E·p̂ for the Fresnel field dyad.
        """
        # Thin-wire surface offset: the OBSERVER segment's radius. NEC
        # keeps the source current a filament on the source axis and
        # enforces the boundary condition on the observing segment's
        # surface — nec2c/necpp pass ai = segment_radius[i] (the segment
        # the field is evaluated on) into EFLD, where it regularizes
        # rh = sqrt(rho² + ai²). Self terms therefore use the wire's own
        # radius, and mutual terms between wires of different radii use
        # the OBSERVER wire's radius. (The opposite source-radius
        # convention was tried first and refuted by the PyNEC oracle: the
        # mixed-radius delta grew with mesh refinement near junctions.)
        # Scalar on the uniform path; an (M, 1) per-observer column when
        # mixed. Observers are always the real segments — image builds
        # only mirror the SOURCE side — so this indexes geom directly.
        if obs_radius is not None:
            a = obs_radius
        elif self._uniform_radius is not None:
            a = self._uniform_radius
        else:
            a = self._seg_radius(geom)[:, None]
        seg_c = geom["seg_centers"] if obs_centers is None else obs_centers  # (M,3)
        seg_t = geom["seg_tangents"] if obs_tangents is None else obs_tangents  # (M,3)
        seg_h = geom["seg_h"]  # (N,) SOURCE full lengths
        h_n = 0.5 * seg_h  # (N,) source half-lengths

        # Sources default to the geometry's segments — NOT the observer set,
        # which may be quadrature points; image builds override src_* directly.
        src_c = src_centers if src_centers is not None else geom["seg_centers"]
        src_t = src_tangents if src_tangents is not None else geom["seg_tangents"]

        # Extended thin-wire kernel (#233): the SOURCE segment's radius and
        # the per-end gating codes, both indexed by source segment. Image
        # builds mirror the source geometry without reordering it, so the same
        # (N,) tables apply there — which is also what NEC does, EFLD passing
        # one IND1/IND2 pair through both passes of its KSYMP image loop
        # (nec2-1.2.1.2.f:2914-2971). (N,) broadcasts against (M, N).
        ek = None
        if self.extended_kernel:
            ind1, ind2 = self._ek_gating(geom)
            src_a = (
                self._uniform_radius
                if self._uniform_radius is not None
                else self._seg_radius(geom)
            )
            ek = (src_a, ind1, ind2)

        # Every-observer × every-source: insert the singleton axes that make
        # the shared core broadcast to the (M, N) outer product.
        return self._field_components_bcast(
            k,
            obs_c=seg_c[:, None, :],  # (M, 1, 3)
            obs_t=seg_t[:, None, :],  # (M, 1, 3)
            a=a,  # scalar or (M, 1)
            src_c=src_c[None, :, :],  # (1, N, 3)
            src_t=src_t[None, :, :],  # (1, N, 3)
            src_hh=h_n[None, :],  # (1, N)
            ek=ek,
        )

    def _field_components_bcast(
        self, k, obs_c, obs_t, a, src_c, src_t, src_hh, cos_shape="cos", ek=None
    ):
        """Shape-agnostic core of `_field_components` (Eqs 76-79).

        Every argument is broadcast against the others, so the caller — not
        this method — decides the pairing. `_field_components` passes
        (M,1,·) observers against (1,N,·) sources for the full outer product;
        `SinusoidalGalerkinSolver`'s near-singular correction passes (P,G,·)
        quadrature points against (P,1,·) sources to get just the P near
        pairs evaluated, without ever forming the N² table those pairs live
        in. Returns the same dict of field tables, each of the common
        broadcast shape S (`rho_vec` is S + (3,)).

        `obs_c`/`obs_t`/`src_c`/`src_t` are position/tangent arrays whose last
        axis is the 3 spatial components; `src_hh` is the source segment's
        HALF length; `a` is the observer-side thin-wire radius.

        `cos_shape` selects which third SOURCE shape the `*_cos` tables carry
        (momwire#205):

        `"cos"`
            I = cos kξ, the literal NEC shape. The point-matched solver's
            contract, and the default.
        `"cos-1"`
            I = cos kξ − 1, the shape the Galerkin fill's coefficient product
            actually pairs with σC once #203's fold is applied. Its field is
            O((kΔ)²) of the const shape's, so taking the difference of the two
            closed forms in float64 leaves ε·‖T_const‖ in it — which is the
            reciprocity floor #205 exists to remove. The branch below gets the
            same quantity out of a spelling in which no term is larger than
            the answer.
        """
        if cos_shape not in ("cos", "cos-1"):
            raise ValueError(f"cos_shape must be 'cos' or 'cos-1', got {cos_shape!r}")
        # Pairwise separation obs - src; shape is whatever the two broadcast to.
        rvec = obs_c - src_c
        t_src = src_t
        t_obs = obs_t

        z_eval = np.einsum("...d,...d->...", rvec, t_src)
        # Perpendicular component:
        rho_vec = rvec - z_eval[..., None] * t_src  # (M, N, 3)
        rho_axis = np.linalg.norm(rho_vec, axis=-1)  # (M, N)
        rho_eval = np.sqrt(rho_axis * rho_axis + a * a)  # (M, N), >= a

        # Tangent dot products; t_obs is shape (M, 1, 3), broadcasting
        # handles the singletons.
        td = (t_obs * t_src).sum(axis=-1)  # (M, N)
        # rho_vec · t_obs at the observer (rho_vec is the perpendicular
        # vector from source axis to obs-axis center)
        rho_dot_tobs = (rho_vec * t_obs).sum(axis=-1)  # (M, N)
        # NEC's prescription: tangential E_ρ component is (ρ·ŝ)/ρ' · E_ρ.
        rho_proj_factor = rho_dot_tobs / rho_eval  # (M, N)

        # Source half-length, materialized at the full broadcast shape so the
        # `H[..., None]` source-quadrature axis below has somewhere to land.
        # Under the outer product that shape is (M, N), M being the observer
        # count: N segment centres (collocation) or N·n_qp Gauss points
        # (Galerkin test integration).
        H = np.broadcast_to(src_hh, z_eval.shape)

        # Extended thin-wire kernel (#233). The geometry above is shared —
        # rho_eval, td and rho_proj_factor are EFLD's, computed before it
        # chooses between EKSC and EKSCX (nec2-1.2.1.2.f:2919-2962), and the
        # ρ̂ projection uses the UNSWAPPED radial distance either way. From
        # here down the two kernels part company, so the extended branch is a
        # clean early return: with `ek=None` (the default, and everything
        # `extended_kernel=False` can reach) not one floating-point operation
        # below changes, which is what makes the option a true no-op when off.
        if ek is not None:
            src_a, ind1, ind2 = ek
            tables = self._extended_kernel_fields(
                k, H, z_eval, rho_eval, src_a, ind1, ind2
            )
            tables["td"] = td
            tables["rho_proj_factor"] = rho_proj_factor
            tables["rho_vec"] = rho_vec
            tables["rho_eval"] = rho_eval
            return tables

        # z values at source ends: z' = +H (z2) and z' = -H (z1).
        # Δz = z_eval - z' at the two endpoints.
        dz2 = z_eval - H  # at z' = +H
        dz1 = z_eval + H  # at z' = -H
        r0_2 = np.sqrt(rho_eval * rho_eval + dz2 * dz2)
        r0_1 = np.sqrt(rho_eval * rho_eval + dz1 * dz1)
        G0_2 = np.exp(-1j * k * r0_2) / r0_2
        G0_1 = np.exp(-1j * k * r0_1) / r0_1

        # Common scalar prefactors. λ = 2π/k → k²λ = 2πk → jη/(2k²λ) = jη/(4πk).
        # Eqs 76-79 carry a "-I_0/λ · jη/(2k²ρ)" (E_ρ) or "I_0/λ · jη/(2k²)"
        # (E_z) prefactor. For unit I_0 = 1, factor pulled out:
        #   E_ρ prefactor = -jη/(4πk·ρ_eval)
        #   E_z prefactor = +jη/(4πk)
        pref_rho = -1j * self.eta / (4.0 * np.pi * k * rho_eval)
        pref_z = 1j * self.eta / (4.0 * np.pi * k)

        # ---- Constant source (I = 1): Eqs 78, 79 ----
        # E_ρ^f = -I/λ · jη/(2k²) · [(1+jkr_0) ρ G_0 / r_0²]_{z1}^{z2}
        #       = pref_rho · ρ_eval · [(1+jkr_0) ρ_eval G_0 / r_0²]_{z1}^{z2}
        #         (pref_rho already has 1/ρ_eval; multiply back by ρ_eval to
        #          recover the form -jη·ρ_eval/(4πk·ρ_eval)= -jη/(4πk))
        # Reorganize for clarity:
        pref_rho_const = -1j * self.eta / (4.0 * np.pi * k)
        Erho_const = pref_rho_const * (
            (1.0 + 1j * k * r0_2) * rho_eval * G0_2 / (r0_2 * r0_2)
            - (1.0 + 1j * k * r0_1) * rho_eval * G0_1 / (r0_1 * r0_1)
        )
        # E_z^f = -I/λ · jη/(2k²) · { [(1+jkr_0)(z-z') G_0 / r_0²]_{z1}^{z2}
        #                              + k² ∫_{z1}^{z2} G_0 dz' }
        # Note sign: -I/λ · jη/(2k²) = -jη/(4πk) for our prefactor convention.
        # We have pref_z = +jη/(4πk); so we use -pref_z.
        # ∫G_0 dz' via singularity extraction. The plain integrand has a
        # 1/r_0 spike at z' = z_eval when ρ_eval is small (self / near-self
        # pairs in thin wires), and Gauss-Legendre with a small node count
        # under-resolves it. Split into closed-form singular + smooth regular:
        #   ∫G_0 dz' = ∫ 1/r_0 dz'              (closed form, arcsinh)
        #            + ∫ (G_0 - 1/r_0) dz'      (regular: tends to -jk as r_0→0)
        # 1/r_0 closed form: ∫_{-H}^{+H} 1/√(ρ²+(z-z')²) dz' = arcsinh((H-z)/ρ) - arcsinh((-H-z)/ρ).
        u2 = (H - z_eval) / rho_eval  # (M, N)
        u1 = (-H - z_eval) / rho_eval
        int_inv_r0 = np.arcsinh(u2) - np.arcsinh(u1)
        gx, gw = self._leggauss_cached(self.n_qp_const)
        z_qp = H[..., None] * gx  # S + (n_qp,)
        dz_qp = z_eval[..., None] - z_qp
        r0_qp = np.sqrt(rho_eval[..., None] ** 2 + dz_qp**2)
        G0_qp = np.exp(-1j * k * r0_qp) / r0_qp
        reg_qp = G0_qp - 1.0 / r0_qp
        int_reg = np.einsum("...q,q->...", reg_qp, gw) * H
        int_G0 = int_inv_r0 + int_reg  # (M, N)
        Ez_const_boundary = (1.0 + 1j * k * r0_2) * dz2 * G0_2 / (r0_2 * r0_2) - (
            1.0 + 1j * k * r0_1
        ) * dz1 * G0_1 / (r0_1 * r0_1)
        Ez_const = -pref_z * (Ez_const_boundary + k * k * int_G0)

        # ---- Sine source (I = sin(k·z'_local)): Eq 76, Eq 77 ----
        # Eq 71 (sinusoidal general) factored: E_ρ^f = pref_rho · G_0 ·
        # {k(z-z') cos(kz') + [1 - (z-z')²(1+jkr_0)/r_0²] sin(kz')} |_{z1}^{z2}
        # — note G_0 is an overall factor on the WHOLE bracket, evaluated
        # at each endpoint with its own r_0.
        sin2 = np.sin(k * H)  # sin(k · z'_2) where z'_2 = +H, so sin(kH)
        cos2 = np.cos(k * H)
        sin1 = np.sin(-k * H)  # at z'_1 = -H
        cos1 = np.cos(-k * H)
        bracket_sin_2 = G0_2 * (
            k * dz2 * cos2
            + (1.0 - dz2 * dz2 * (1.0 + 1j * k * r0_2) / (r0_2 * r0_2)) * sin2
        )
        bracket_sin_1 = G0_1 * (
            k * dz1 * cos1
            + (1.0 - dz1 * dz1 * (1.0 + 1j * k * r0_1) / (r0_1 * r0_1)) * sin1
        )
        Erho_sin = pref_rho * (bracket_sin_2 - bracket_sin_1)
        # E_z^f = pref_z · G_0 · {k cos(kz') - (1+jkr_0)(z-z')/r_0² sin(kz')}_{z1}^{z2}
        bracket_sin_z_2 = G0_2 * (
            k * cos2 - (1.0 + 1j * k * r0_2) * dz2 / (r0_2 * r0_2) * sin2
        )
        bracket_sin_z_1 = G0_1 * (
            k * cos1 - (1.0 + 1j * k * r0_1) * dz1 / (r0_1 * r0_1) * sin1
        )
        Ez_sin = pref_z * (bracket_sin_z_2 - bracket_sin_z_1)

        if cos_shape == "cos":
            # ---- Cosine source (I = cos(k·z'_local)): same as Eqs 76, 77 with
            #      the "(cos kz'/-sin kz')" toggle picking the lower row, i.e.
            #      swap sin↔cos and negate the (sin-row → -sin) term.
            bracket_cos_2 = G0_2 * (
                -k * dz2 * sin2
                + (1.0 - dz2 * dz2 * (1.0 + 1j * k * r0_2) / (r0_2 * r0_2)) * cos2
            )
            bracket_cos_1 = G0_1 * (
                -k * dz1 * sin1
                + (1.0 - dz1 * dz1 * (1.0 + 1j * k * r0_1) / (r0_1 * r0_1)) * cos1
            )
            Erho_cos = pref_rho * (bracket_cos_2 - bracket_cos_1)
            bracket_cos_z_2 = G0_2 * (
                -k * sin2 - (1.0 + 1j * k * r0_2) * dz2 / (r0_2 * r0_2) * cos2
            )
            bracket_cos_z_1 = G0_1 * (
                -k * sin1 - (1.0 + 1j * k * r0_1) * dz1 / (r0_1 * r0_1) * cos1
            )
            Ez_cos = pref_z * (bracket_cos_z_2 - bracket_cos_z_1)
        else:
            Erho_cos, Ez_cos = self._folded_cos_fields(
                k,
                H,
                z_eval,
                rho_eval,
                dz1,
                dz2,
                r0_1,
                r0_2,
                G0_1,
                G0_2,
                sin2,
                dz_qp,
                r0_qp,
                gx,
                gw,
                pref_z,
                pref_rho,
            )

        return {
            "Erho_const": Erho_const,
            "Ez_const": Ez_const,
            "Erho_sin": Erho_sin,
            "Ez_sin": Ez_sin,
            "Erho_cos": Erho_cos,
            "Ez_cos": Ez_cos,
            "td": td,
            "rho_proj_factor": rho_proj_factor,
            "rho_vec": rho_vec,
            "rho_eval": rho_eval,
        }

    @staticmethod
    def _folded_cos_fields(
        k,
        H,
        z,
        rho,
        dz1,
        dz2,
        r1,
        r2,
        G1,
        G2,
        sH,
        dz_qp,
        r_qp,
        gx,
        gw,
        pref_z,
        pref_rho,
    ):
        """(E_ρ, E_z) of the source shape I = cos kξ − 1 (momwire#205).

        Same quantity as `Ez_cos − Ez_const` on the same source quadrature —
        exactly, not approximately: nothing here changes the rule, only which
        differences are taken before which sums. What changes is the error.
        The literal subtraction leaves ε·|const-shape field| in an answer that
        is O((kH)²) of it, so it is ε·6/(kH)² relative and it lands on G undiluted
        — 3.9e-12 of the const scale at N=2401 on the half-wave dipole, which
        is exactly the reciprocity floor #203 measured and could not reach.

        Taking the two closed forms apart term by term (the endpoint pairs
        cancel against each other, and the const shape's ∫G₀ against the cos
        shape's endpoint terms) leaves three quantities, each ALREADY the size
        of the answer:

            E_z:  pref_z·[ k²·(∫G₀ − (sin kH/k)(G₁+G₂)) − (cos kH − 1)(F₂−F₁) ]
            E_ρ:  pref_ρ·[ −k·W + ρ²(cos kH − 1)(T₂−T₁) ]

        with F_e = Δz_e·T_e, T_e = (1+jkr_e)G_e/r_e², and W the closed form of
        kρ²∫sin(kξ)T(ξ)dξ. The three cancellations that spelling still hides
        are removed one at a time:

        * **∫G₀ against its endpoint pair.** Split at the singularity
          extraction the const shape already does. The 1/r half is the exact
          trapezoid remainder ∫(1/r) − H(1/r₁+1/r₂), which rationalizes to
          `asinh(X) − X` plus a manifestly positive term (`X` is the argument
          of the arcsinh difference, itself rationalized so its numerator does
          not cancel). The regular half is a quadrature sum whose nodes are
          taken relative to the NEAREST endpoint, through r-differences that
          are exact by construction (Δz_q − Δz_e = ±H(1 ∓ t_q): the observer
          drops out) and an `expm1` of the resulting phase.
        * **H − sin(kH)/k**, from `_sin_minus_arg`.
        * **W**, whose two endpoint groups cancel to O((kH)³). Referring both
          endpoints to the pair's MEAN radius turns them into one even and one
          odd combination, and the even one telescopes to
          A·(sin B − B) − B·(sin A − A) with A, B = kH ± k(r₁−r₂)/2 — two more
          `_sin_minus_arg`s — plus a term in H(c₁+c₂) − (r₁−r₂), which has its
          own cancellation-free closed form (`d_lin` below).

        What is left is the regular quadrature sum, where the node terms are
        O(kH) and their sum is O((kH)²): one power of kH, not two, so the
        residual is ε·(kH)·|const| rather than ε·|const| — measured 1.6e-15 of
        the const scale at N=2401 against a 50-digit reference, a factor 2400.
        Removing the last power needs the SECOND difference of the regular
        integrand (node pair against endpoint pair) in closed form; at the
        meshes this solver runs, that residual is already below the fill's
        structural asymmetry, so it is left to #205's follow-up.
        """
        kH = k * H
        cm1 = -2.0 * np.sin(0.5 * kH) ** 2  # cos kH − 1, without the subtraction
        T2 = (1.0 + 1j * k * r2) * G2 / (r2 * r2)
        T1 = (1.0 + 1j * k * r1) * G1 / (r1 * r1)

        # X = sinh of the arcsinh difference ∫(1/r) dξ, i.e. asinh(X) = that
        # integral. Its numerator Δz₁r₂ − Δz₂r₁ cancels when the observer is
        # off the segment's span and Δz₁r₂ + Δz₂r₁ cancels when it is inside,
        # so each branch takes the sum that does not: rationalizing
        # (Δz₁r₂)² − (Δz₂r₁)² = ρ²(Δz₁² − Δz₂²) swaps one for the other.
        outside = dz1 * dz2 >= 0.0
        s_den = np.where(outside, dz1 * r2 + dz2 * r1, 1.0)
        X = np.where(
            outside,
            2.0 * H * (dz1 + dz2) / s_den,
            (dz1 * r2 - dz2 * r1) / (rho * rho),
        )
        # ∫(1/r)dξ − H(1/r₁ + 1/r₂). The rationalized second term is
        # X − H(r₁+r₂)/(r₁r₂) = Hρ²X²/((r₁+r₂)r₁r₂) — all positive, no
        # cancellation — leaving asinh(X) − X for `_asinh_minus_arg`. Above
        # |X| = 1 the pair no longer cancels at all (this is the self and
        # near-neighbour band, where the answer IS the arcsinh), and the two
        # rationalized terms would instead cancel against each other, so the
        # literal difference is the accurate spelling there.
        t_sing = np.where(
            np.abs(X) < 1.0,
            _asinh_minus_arg(X) + H * rho * rho * X * X / ((r1 + r2) * r1 * r2),
            np.arcsinh(X) - H * (1.0 / r1 + 1.0 / r2),
        )

        # Σ_q gw_q·g(ξ_q) − (g₁ + g₂) for the regular part g = G₀ − 1/r, with
        # every node referred to the endpoint on its own side of the segment.
        g1 = _expm1_neg_j(k * r1) / r1
        g2 = _expm1_neg_j(k * r2) / r2
        near2 = gx >= 0.0
        r_ref = np.where(near2, r2[..., None], r1[..., None])
        dz_ref = np.where(near2, dz2[..., None], dz1[..., None])
        g_ref = np.where(near2, g2[..., None], g1[..., None])
        # e^{-jkr} at the reference endpoint, from the Green's value already in
        # hand rather than from a second exp.
        e_ref = np.where(near2, (G2 * r2)[..., None], (G1 * r1)[..., None])
        # Δz_q − Δz_ref is ±H(1 ∓ t_q) exactly: z_eval cancels, so the node
        # radius difference below is exact however far away the observer is.
        d_step = np.where(near2, H[..., None] * (1.0 - gx), -H[..., None] * (1.0 + gx))
        delta = d_step * (dz_qp + dz_ref) / (r_qp + r_ref)
        # g(r_ref + δ) − g(r_ref), rearranged so both terms are O(kδ). Spelled
        # in named steps, not as one expression: above numpy's 256 KB temporary
        # threshold a complex product with a dead operand is evaluated in place
        # by a different loop, which would make this (…, n_qp)-sized array's
        # rounding depend on the fill's BLOCK size — the one thing
        # `test_far_fill_blocking_is_bit_exact` may not tolerate.
        dg_phase = _expm1_neg_j(k * delta)
        dg_a = e_ref * dg_phase
        dg_b = g_ref * delta
        dg_num = dg_a - dg_b
        dg = dg_num / r_qp
        # The rule's weights sum to 2 and its two halves to 1 each only to
        # within ε, and those ε's are part of the value the const shape
        # already carries, so they are kept rather than idealized away.
        m_reg = (
            np.einsum("...q,q->...", dg, gw)
            + (gw[near2].sum() - 1.0) * g2
            + (gw[~near2].sum() - 1.0) * g1
        )

        d_int = t_sing + H * m_reg - (_sin_minus_arg(kH) / k) * (G1 + G2)
        Ez = pref_z * (k * k * d_int - cm1 * (dz2 * T2 - dz1 * T1))

        # E_ρ: W = kρ²∫sin(kξ)·(1+jkr)G₀/r² dξ in closed form, with both
        # endpoint terms referred to the pair's mean radius (r₁+r₂)/2 — which
        # costs no new exponential, since e^{-jk(r₁+r₂)/2} = e^{-jkr₂}·e^{-jkφ}
        # and φ = k(r₁−r₂)/2 is needed anyway.
        dr = 4.0 * H * z / (r1 + r2)  # r₁ − r₂, exactly
        phi = 0.5 * k * dr
        A = kH + phi
        B = kH - phi
        # H(c₁+c₂) − (r₁−r₂), with c_e = Δz_e/r_e: the same rationalization
        # twice over, ending on
        # ρ² + Δz₁Δz₂ − r₁r₂ = −4ρ²H²/(ρ² + Δz₁Δz₂ + r₁r₂).
        d_lin = (
            -8.0
            * H**3
            * z
            * rho
            * rho
            / ((rho * rho + dz1 * dz2 + r1 * r2) * (r1 + r2) * r1 * r2)
        )
        cph = np.cos(phi)
        sph = np.sin(phi)
        w_even = (A * _sin_minus_arg(B) - B * _sin_minus_arg(A)) / kH + (
            d_lin / H
        ) * sH * cph
        # c₂ − c₁ = −ρ²X/(r₁r₂), from the same rationalized numerator as X.
        w_odd = 1j * sH * (-(rho * rho * X) / (r1 * r2)) * sph
        W = (G2 * r2) * (cph - 1j * sph) * (w_even + w_odd)
        Erho = pref_rho * (-k * W + rho * rho * cm1 * (T2 - T1))
        return Erho, Ez

    # ------------------------------------------------------------------
    # Matrix assembly and solve
    # ------------------------------------------------------------------

    def _image_source_centers_tangents(self, geom):
        """Mirror source segments across z = ground_z and flip their tangent
        z-components, mirroring the convention the Galerkin solvers use for the
        PEC image build. Same shape ((N, 3), (N, 3)) as the originals.
        """
        seg_c = geom["seg_centers"]
        seg_t = geom["seg_tangents"]
        src_c_img = seg_c * np.array([1.0, 1.0, -1.0]) + np.array(
            [0.0, 0.0, 2.0 * self.ground_z]
        )
        src_t_img = seg_t * np.array([1.0, 1.0, -1.0])
        return src_c_img, src_t_img

    def _field_tensor_image(self, geom, k):
        """Field tensor for image sources at PEC ground. The image keeps the
        same per-segment half-length and basis shape; only the source center
        is mirrored and the source tangent z-component is flipped.
        """
        src_c_img, src_t_img = self._image_source_centers_tangents(geom)
        return self._field_tensor(
            geom, k, src_centers=src_c_img, src_tangents=src_t_img
        )

    def _image_refl_prep(self, geom):
        """k-independent per-pair specular tables for the `ground_eps`
        weighted image: incidence cosine cos θ, p̂ components (px, py),
        and the tangent projections tm_p = t_m·p̂, tn_p = t_n·p̂. Cached
        per geometry object (identity check, same pattern as
        `_cached_basis`) so swept callers pay the O(N²) build once, not
        per frequency; ρ_v/ρ_h depend on ε̃(ω) and are NOT cached.
        """
        cached = self._cached_image_refl_prep
        if cached is not None and cached[0] is geom:
            return cached[1]
        seg_c = geom["seg_centers"]
        seg_t = geom["seg_tangents"]
        # Square case: sources = observers; specular geometry from the
        # REAL (unmirrored) source centers — specular_ray_tables does the
        # mirroring internally (dz = z_m + z_n − 2·ground_z).
        cos_th, px, py = _ground_refl.specular_ray_tables(seg_c, self.ground_z)
        # t·p̂ tables. p̂ has zero z-component and the image tangent only
        # flips z, so t_img·p̂ = t_src·p̂ — the REAL source tangents serve
        # for the image-source projection too.
        tm_p = seg_t[:, 0][:, None] * px + seg_t[:, 1][:, None] * py
        tn_p = seg_t[:, 0][None, :] * px + seg_t[:, 1][None, :] * py
        prep = (cos_th, px, py, tm_p, tn_p)
        self._cached_image_refl_prep = (geom, prep)
        return prep

    def _field_tensor_image_refl(self, geom, k):
        """Fresnel-weighted image field tensor for the `ground_eps` finite
        ground (NEC IPERF=0 reflection-coefficient approximation).

        Applies NEC's field dyad D = ρ_v·(I − p̂p̂) − ρ_h·p̂p̂ to the IMAGE
        source's field vector at each observer before the tangential
        projection, with p̂ the horizontal unit normal to the plane of
        incidence of the specular ray (image midpoint → observer midpoint)
        and ρ_v/ρ_h the Fresnel coefficients at that ray's incidence angle
        — per-pair constants, NEC's approximation. Expanding the
        projection (t_m has unit norm, p̂·p̂ = 1):

            t_m · D · E = ρ_v·(t_m·E) − (ρ_v + ρ_h)·(t_m·p̂)·(E·p̂)

        with both scalars available from the unprojected component tables:
            t_m·E = td·E_z + rho_proj·E_ρ    (the plain projection)
            E·p̂  = (t_n·p̂)·E_z + (ρ̂·p̂)·E_ρ  (t_img·p̂ = t_n·p̂; ρ̂ = rho_vec
                                              /rho_eval as in the E_ρ rule)

        PEC limit ε̃ → ∞: ρ_v → +1, ρ_h → −1, so the p̂ correction vanishes
        and this reduces exactly to `_field_tensor_image` — the ε̃=1e16
        collapse test rides on that. The returned tensors are SUBTRACTED
        in `_assemble_Z` with the same single global minus sign as the PEC
        image. Hot path is the fused C++ `sinusoidal_field_tensor_refl`
        kernel (Eqs 76-79 + the dyad projection in one pass; ρ_v/ρ_h
        computed in-kernel per pair); the numpy formulation below is the
        bit-close reference / fallback.
        """
        src_c_img, src_t_img = self._image_source_centers_tangents(geom)
        cos_th, px, py, tm_p, tn_p = self._image_refl_prep(geom)
        # ε̃(ω) — per-frequency (the swept loops update self.omega
        # alongside k before assembling).
        eps_t = _ground_refl.eps_tilde(self.ground_eps, self.omega, self.eps)

        # As in `_field_tensor`: the fused C++ kernel is reduced-kernel-only,
        # so EK-on falls to the numpy reference (momwire#233).
        if _HAVE_FIELD_TENSOR_REFL and not self.extended_kernel:
            # Fused C++ path: Eqs 76-79 field components + the Fresnel
            # dyad projection in one pass, with rho_v/rho_h computed
            # in-kernel per pair from eps_t and cos_th (same principal-
            # branch sqrt as _ground_refl.fresnel_rho). The numpy path
            # below is the bit-close reference / fallback.
            seg_c = geom["seg_centers"]
            seg_t = geom["seg_tangents"]
            seg_h = geom["seg_h"]
            gx, gw = self._leggauss_cached(self.n_qp_const)

            def _call(rows, a):
                # Observer-side slicing: the (M, N) specular tables slice
                # on their observer axis alongside the obs arrays.
                return _acc.sinusoidal_field_tensor_refl(
                    np.ascontiguousarray(seg_c[rows], dtype=np.float64),
                    np.ascontiguousarray(seg_t[rows], dtype=np.float64),
                    np.ascontiguousarray(src_c_img, dtype=np.float64),
                    np.ascontiguousarray(src_t_img, dtype=np.float64),
                    np.ascontiguousarray(seg_h, dtype=np.float64),
                    float(a),
                    float(k),
                    float(self.eta),
                    np.ascontiguousarray(gx, dtype=np.float64),
                    np.ascontiguousarray(gw, dtype=np.float64),
                    np.ascontiguousarray(cos_th[rows], dtype=np.float64),
                    np.ascontiguousarray(px[rows], dtype=np.float64),
                    np.ascontiguousarray(py[rows], dtype=np.float64),
                    np.ascontiguousarray(tm_p[rows], dtype=np.float64),
                    np.ascontiguousarray(tn_p[rows], dtype=np.float64),
                    complex(eps_t),
                    self._cancel_flag,
                )

            if self._uniform_radius is not None:
                return _call(slice(None), self._uniform_radius)
            # Mixed per-wire radii: one call per constant-radius run of
            # observer rows (see `_field_tensor` / `_radius_runs`).
            parts = [_call(slice(s, e), a) for s, e, a in self._radius_runs(geom)]
            return tuple(
                np.concatenate([p[i] for p in parts], axis=0) for i in range(3)
            )

        cm = self._field_components(
            geom, k, src_centers=src_c_img, src_tangents=src_t_img
        )
        rho_v, rho_h = _ground_refl.fresnel_rho(eps_t, cos_th)

        # ρ̂·p̂ from the image-build rho_vec (p̂ is horizontal, so only the
        # x/y components contribute). Same radius-regularized rho_eval
        # denominator as the E_ρ projection rule, so the two E_ρ pickups
        # stay mutually consistent; near-vertical rays have rho_vec → 0,
        # which kills this term along with the p̂ ambiguity.
        rho_p = (cm["rho_vec"][..., 0] * px + cm["rho_vec"][..., 1] * py) / cm[
            "rho_eval"
        ]

        td = cm["td"]
        rho_proj_factor = cm["rho_proj_factor"]
        rvh = rho_v + rho_h  # → 0 in the PEC limit

        def _project_weighted(Ez, Erho):
            tm_E = td * Ez + rho_proj_factor * Erho  # t_m · E
            E_p = tn_p * Ez + rho_p * Erho  # E · p̂
            return rho_v * tm_E - rvh * tm_p * E_p

        Phi_const = _project_weighted(cm["Ez_const"], cm["Erho_const"])
        Phi_sin = _project_weighted(cm["Ez_sin"], cm["Erho_sin"])
        Phi_cos = _project_weighted(cm["Ez_cos"], cm["Erho_cos"])
        return Phi_const, Phi_sin, Phi_cos

    def _field_tensor_sommerfeld_remainder(
        self, geom, k, eps_t, obs_centers=None, obs_tangents=None, cos_shape="cos"
    ):
        """Smooth Sommerfeld-remainder tensor S[3, M, N]: the tangential
        remainder field of segment n's three source shapes (`cos_shape`) at the
        center of segment m, from the interpolated SommerfeldGrid F dyad
        (theory manual eqs 143-147 azimuth combination — the same algebra
        as `BSplineSolver._Z_sommerfeld_remainder`, minus the observer
        quadrature: this solver point-matches at segment centers, so only
        the SOURCE side integrates, GL with `n_qp_sommerfeld` nodes).

        Observers default to the geometry's segment centres/tangents
        (collocation). `SinusoidalGalerkinSolver` overrides them with the
        Gauss points along each test segment — the same reuse-the-evaluator
        pattern as `_field_components`'s `obs_*` overrides — so M is then
        the number of quadrature points rather than the number of segments.
        The grid extent below is derived from segment ENDPOINTS, which
        bounds any interior quadrature point, so it serves both callers.

        The grid's surfaces are the E-field of a unit current MOMENT
        (Il = 1, eq 123 normalization), so integrating shape(z')·F dz'
        along the segment gives the remainder field of the full source
        distribution — endpoint-charge contributions are inherent in the
        element superposition, matching the field-form Eqs 76-79 blocks.
        ADDED in `_assemble_Z` (`Phi_free - C2·Phi_img + S`): the sign is
        pinned by the cross-solver test against bspline-sommerfeld on a
        0.05-wl dipole (where the remainder is ~20 ohm, so a sign error
        is a ~2x miss), not derived here — same discipline as the
        ground-plan's PEC-limit sign pinning.
        """
        gz = self.ground_z
        seg_c = geom["seg_centers"]
        seg_t = geom["seg_tangents"]
        seg_l = geom["seg_l"]
        seg_r = geom["seg_r"]
        h_half = 0.5 * geom["seg_h"]  # (N,)
        N = geom["n_segs"]
        zmin = min(seg_l[:, 2].min(), seg_r[:, 2].min()) - gz
        # Touching (zmin == 0) is allowed since #151: the ground-junction
        # basis handles contact, and the remainder quadrature samples
        # Gauss nodes strictly interior to segments, so z+z' > 0 holds
        # even for a wire ending in the plane. Only genuinely submerged
        # geometry is rejected (already caught at geometry build too).
        if zmin < -1e-12:
            raise ValueError(
                "ground_model='sommerfeld' requires every wire at or "
                f"above ground_z (min height above plane: {zmin:.3g})"
            )

        q = self.n_qp_sommerfeld
        gx, gw = self._leggauss_cached(q)
        zloc = h_half[:, None] * gx[None, :]  # (N, q) local arc offsets
        src = seg_c[:, None, :] + zloc[..., None] * seg_t[:, None, :]
        w_node = h_half[:, None] * gw[None, :]  # (N, q) dz' weights
        # Source shapes at the nodes, NATURAL-arc convention like the
        # free-space tensor (sigma accounting stays the caller's job). The
        # third shape follows the free-space tensor's `cos_shape` contract
        # (momwire#205) — this block is SUBTRACTED from one built there, so
        # both have to be on the same shape set. Here the fold is only a
        # better-scaled weight inside a quadrature (no cancelling closed
        # forms), and −2sin²(kξ/2) keeps it accurate at its own size.
        shp_cos = (
            np.cos(k * zloc)
            if cos_shape == "cos"
            else -2.0 * np.sin(0.5 * k * zloc) ** 2
        )
        shp = np.stack([np.ones_like(zloc), np.sin(k * zloc), shp_cos])  # (3, N, q)

        # Grid extent: obs-to-image-point distance is convex in the two
        # segment parameters, so its max is attained at endpoint pairs.
        ex = np.concatenate([seg_l, seg_r])
        dxe = ex[:, 0][:, None] - ex[:, 0][None, :]
        dye = ex[:, 1][:, None] - ex[:, 1][None, :]
        hze = (ex[:, 2][:, None] - gz) + (ex[:, 2][None, :] - gz)
        r1_max = float(np.sqrt(dxe * dxe + dye * dye + hze * hze).max()) * 1.001
        grid = _sommerfeld.get_grid(
            eps_t,
            k,
            r1_max,
            omega=self.omega,
            mu=self.mu,
            cancel_flag=self._cancel_flag,
        )

        n_src = N * q
        srcf = src.reshape(n_src, 3)
        t_src = np.repeat(seg_t, q, axis=0)

        obs_c = seg_c if obs_centers is None else np.asarray(obs_centers, dtype=float)
        obs_t = seg_t if obs_tangents is None else np.asarray(obs_tangents, dtype=float)
        M = obs_c.shape[0]

        S = np.empty((3, M, N), dtype=np.complex128)
        chunk = max(1, (1 << 19) // max(n_src, 1))
        for i0 in range(0, M, chunk):
            self._checkpoint()  # per observer chunk of the eval block
            i1 = min(i0 + chunk, M)
            proj = _sommerfeld.remainder_field_proj(
                obs_c[i0:i1], obs_t[i0:i1], srcf, t_src, gz, k, grid
            )
            fq = proj.reshape(i1 - i0, N, q)
            S[:, i0:i1, :] = np.einsum("snq,mnq->smn", shp * w_node[None], fq)
        return S

    def _assemble_Z(self, geom, k):
        Phi_c, Phi_s, Phi_co = self._field_tensor(geom, k)
        if self.ground_z is not None:
            # Image ground: subtract the sub-assembly built from the image
            # field tensor. The image source's mirrored geometry + flipped
            # z-tangent already encode both the anti-parallel horizontal
            # image current and the parallel vertical image current; the
            # combined image-current + image-charge sign flip reduces to
            # a single minus sign on the image-Z block (same as BSpline).
            # With `ground_eps` set, the image field is additionally
            # Fresnel-dyad weighted (NEC IPERF=0); the subtraction sign is
            # unchanged because the weighted tensor reduces to the PEC one
            # in the ε̃ → ∞ limit.
            if self.ground_eps is not None:
                if self.ground_model == "sommerfeld":
                    # NEC's decomposition (theory manual eqs 136-147):
                    # exact image scaled by the constant C2, which absorbs
                    # all the singular behavior — a plain scalar on the
                    # projected PEC-image tensor, so the C++ kernel keeps
                    # serving it — plus the smooth interpolated remainder,
                    # which ADDS (see the remainder method's docstring for
                    # the sign pinning). eps->inf: C2 -> 1, S -> 0, PEC
                    # image exactly; eps -> 1: both vanish, free space.
                    eps_t = _ground_refl.eps_tilde(
                        self.ground_eps, self.omega, self.eps
                    )
                    c2 = (eps_t - 1.0) / (eps_t + 1.0)
                    Phi_c_i, Phi_s_i, Phi_co_i = self._field_tensor_image(geom, k)
                    S_c, S_s, S_co = self._field_tensor_sommerfeld_remainder(
                        geom, k, eps_t
                    )
                    Phi_c_i = c2 * Phi_c_i - S_c
                    Phi_s_i = c2 * Phi_s_i - S_s
                    Phi_co_i = c2 * Phi_co_i - S_co
                else:
                    Phi_c_i, Phi_s_i, Phi_co_i = self._field_tensor_image_refl(geom, k)
            else:
                Phi_c_i, Phi_s_i, Phi_co_i = self._field_tensor_image(geom, k)
            Phi_c = Phi_c - Phi_c_i
            Phi_s = Phi_s - Phi_s_i
            Phi_co = Phi_co - Phi_co_i

        seg_view = self._basis_coefs(geom, k)
        N = geom["n_segs"]
        # Build (N, N) coefficient matrices M_{A,B,C}[n, j] = effective
        # coefficient that basis j contributes at source segment n. With
        #   A_eff = σ * A, B_eff = B, C_eff = σ * C
        # (see docs/sinusoidal_basis_design.md), the per-basis loop reduces
        # to three N×N matmuls G = Phi_c @ M_A + Phi_s @ M_B + Phi_co @ M_C.
        #
        # seg_view is already CSR-by-segment, so the row coordinate n_idx
        # is just np.repeat(arange(N), counts) and σA / σC are element-wise
        # numpy products. No Python scatter loop here — the per-basis fill
        # already happened directly into seg_view inside _basis_coefs.
        #
        # Each basis has only ~3 entries (self + N⁻ + N⁺ neighbour), so M
        # is very sparse (~3N nonzeros). Two regimes:
        #   N < _DENSE_ASSEMBLY_THRESHOLD: dense matmul wins because the
        #     scipy.sparse constructor overhead dominates the BLAS call.
        #   N ≥ threshold: sparse matmul wins because BLAS zgemm pays the
        #     full O(N³) cost on a mostly-zero matrix, while CSC matmul
        #     pays O(N · 3N) = O(N²).
        # Crossover measured ≈ N=60 on Kaby Lake R / OpenBLAS-pthreads.
        starts = seg_view["starts"]
        n_idx_arr = np.repeat(np.arange(N, dtype=np.int64), starts[1:] - starts[:-1])
        j_idx_arr = seg_view["jbasis"]
        sigma_arr = seg_view["sigma"]
        A_eff = sigma_arr * seg_view["A"]
        B_eff = seg_view["B"]
        C_eff = sigma_arr * seg_view["C"]
        if N < _DENSE_ASSEMBLY_THRESHOLD:
            M_A = np.zeros((N, N), dtype=np.complex128)
            M_B = np.zeros((N, N), dtype=np.complex128)
            M_C = np.zeros((N, N), dtype=np.complex128)
            M_A[n_idx_arr, j_idx_arr] = A_eff
            M_B[n_idx_arr, j_idx_arr] = B_eff
            M_C[n_idx_arr, j_idx_arr] = C_eff
            G = Phi_c @ M_A + Phi_s @ M_B + Phi_co @ M_C
        else:
            M_A = scipy.sparse.csc_matrix((A_eff, (n_idx_arr, j_idx_arr)), shape=(N, N))
            M_B = scipy.sparse.csc_matrix((B_eff, (n_idx_arr, j_idx_arr)), shape=(N, N))
            M_C = scipy.sparse.csc_matrix((C_eff, (n_idx_arr, j_idx_arr)), shape=(N, N))
            G = (Phi_c @ M_A) + (Phi_s @ M_B) + (Phi_co @ M_C)
        self._apply_loading(G, geom, seg_view, k)
        return G, seg_view

    def _loading_zw(self, omega):
        """Per-wire series impedance Z'_w(ω) [Ω/m]; zeros when a wire's
        loading is switched off (NaN entries)."""
        return _wire_loading.series_impedance_per_wire(
            omega,
            self._radius_per_wire,
            self.wire_conductivity,
            self.insulation_radius,
            self.insulation_eps_r,
        )

    @staticmethod
    def _wire_of_seg(geom):
        """(n_segs,) int array mapping segment index → wire index."""
        firsts = np.asarray(geom["wire_first"], dtype=np.int64)
        lasts = np.asarray(geom["wire_last"], dtype=np.int64)
        return np.repeat(np.arange(firsts.shape[0]), lasts - firsts + 1)

    def _seg_radius(self, geom):
        """(n_segs,) per-segment radius — each segment inherits its wire's."""
        return self._radius_per_wire[self._wire_of_seg(geom)]

    def _radius_runs(self, geom):
        """Contiguous constant-radius observer-row runs, as (start, stop,
        radius) triples — the per-run dispatch unit that serves mixed
        per-wire radii through the scalar-radius C++ field kernels
        (stevenmburns/momwire#147). Segments are wire-contiguous, so the
        number of runs is at most the number of wires."""
        a_seg = self._seg_radius(geom)
        bounds = np.flatnonzero(np.diff(a_seg)) + 1
        starts = np.concatenate(([0], bounds))
        stops = np.concatenate((bounds, [a_seg.shape[0]]))
        return [(int(s), int(e), float(a_seg[s])) for s, e in zip(starts, stops)]

    def _apply_loading(self, G, geom, seg_view, k):
        """NEC's impedance boundary condition, in place; no-op when loading
        is off. The system point-matches E_scat(n) = −E_app(n) at segment
        centres; a distributed series impedance changes the wire surface
        condition to E_scat(n) − Z'_w(n)·I(n) = −E_app(n), so subtract
        Z'·(current of basis j at segment n's centre) = Z'·σ(A+C) from
        G[n, j] over the basis-support entries. (Contrast the Galerkin
        overlap loading in `BSplineSolver._apply_loading` — same physics,
        each entering through its own testing scheme.)"""
        if not self._loading_active:
            return G
        omega = k * self.c
        zw = self._loading_zw(omega)  # (n_w,)
        starts = seg_view["starts"]
        n_segs = geom["n_segs"]
        rows = np.repeat(np.arange(n_segs, dtype=np.int64), starts[1:] - starts[:-1])
        cols = seg_view["jbasis"]
        i_center = seg_view["sigma"] * (seg_view["A"] + seg_view["C"])
        # Each (seg, basis) pair is unique in seg_view (see _basis_coefs),
        # so plain fancy-index subtraction is exact.
        G[rows, cols] -= zw[self._wire_of_seg(geom)[rows]] * i_center
        return G

    def wire_loss_power(self, coeffs, omega=None):
        """Ohmic power dissipated in the wire metal, from a solve's coeffs.

        P_wire = ½ Σ_w Re[Z'_w(ω)] · Σ_{s∈w} ∫|I_s(ξ)|² dξ — the physical
        ∫R'(l)·|I(l)|² dl readout, same contract as the BSpline family's.
        On each segment the current is the single aggregated three-term
        shape I_s(ξ) = P + Q·sin(kξ) + R·cos(kξ) (α-weighted sums of the
        per-basis σA/B/σC entries), so the integral over ξ ∈ [−h/2, h/2]
        is closed-form:

            ∫|I|² = |P|²h + 2Re(P·R̄)·(2/k)sin(kh/2)
                    + |Q|²(h/2 − sin(kh)/2k) + |R|²(h/2 + sin(kh)/2k)

        (P–Q and Q–R cross terms vanish by parity). Insulation loading is
        purely reactive and contributes nothing.

        Returns (total_watts, per_wire_watts ndarray (n_wires,)).
        """
        n_w = len(self.wires_polylines)
        per_wire = np.zeros(n_w, dtype=np.float64)
        if not self._loading_active:
            return 0.0, per_wire
        if omega is None:
            omega = self.omega
        k = omega / self.c
        geom = self._build_geometry()
        seg_view = self._basis_coefs(geom, k)
        n_segs = geom["n_segs"]
        h = np.asarray(geom["seg_h"], dtype=np.float64)

        starts = seg_view["starts"]
        rows = np.repeat(np.arange(n_segs, dtype=np.int64), starts[1:] - starts[:-1])
        alpha_e = np.asarray(coeffs)[seg_view["jbasis"]]
        P = np.zeros(n_segs, dtype=np.complex128)
        Q = np.zeros(n_segs, dtype=np.complex128)
        R = np.zeros(n_segs, dtype=np.complex128)
        np.add.at(P, rows, alpha_e * seg_view["sigma"] * seg_view["A"])
        np.add.at(Q, rows, alpha_e * seg_view["B"])
        np.add.at(R, rows, alpha_e * seg_view["sigma"] * seg_view["C"])

        sin_kh = np.sin(k * h)
        sin_kh_2 = np.sin(0.5 * k * h)
        int_abs_i2 = (
            np.abs(P) ** 2 * h
            + 2.0 * np.real(P * np.conj(R)) * (2.0 / k) * sin_kh_2
            + np.abs(Q) ** 2 * (0.5 * h - sin_kh / (2.0 * k))
            + np.abs(R) ** 2 * (0.5 * h + sin_kh / (2.0 * k))
        )
        r_w = np.real(self._loading_zw(omega))  # zeros where switched off
        wire_of = self._wire_of_seg(geom)
        np.add.at(per_wire, wire_of, 0.5 * r_w[wire_of] * int_abs_i2)
        return float(per_wire.sum()), per_wire

    def _feed_segment_current(self, alpha, seg_view, feed_seg):
        """Current at centre of a feed segment. I(s_local=0) = Σ_j α_j · σ_j ·
        (A_jn + C_jn) over bases j whose support includes `feed_seg`
        (sin(k·0)=0 so B drops out).
        """
        s = seg_view["starts"][feed_seg]
        e = seg_view["starts"][feed_seg + 1]
        return complex(
            (
                alpha[seg_view["jbasis"][s:e]]
                * seg_view["sigma"][s:e]
                * (seg_view["A"][s:e] + seg_view["C"][s:e])
            ).sum()
        )

    def compute_impedance(self):
        """Return (Z_drive, alpha). With a single feed, Z_drive is a scalar
        V/I (back-compat). With N feeds, Z_drive is a length-N complex
        array of per-feed driving-point impedances V_i / I_i — the RHS is
        built as Σ_i V_i · (-1/h_i) · e_{feed_i} (linear superposition of
        Eq 187 delta-gap sources).
        """
        geom = self._build_geometry()
        self._checkpoint()  # after geometry, before the field-tensor fill
        G, seg_view = self._assemble_Z(geom, self.k)
        feed_segs = geom["feed_segs"]
        voltages = np.array([v for _, _, v in self.feeds], dtype=np.complex128)

        v = np.zeros(geom["n_segs"], dtype=np.complex128)
        for fi, V_i in zip(feed_segs, voltages):
            v[fi] += -V_i / geom["seg_h"][fi]
        self._checkpoint()  # after assembly, before the dense LU solve
        # Factor Gᵀ in place: G.T is an F-ordered view of the C-ordered G,
        # so LAPACK's getrf overwrites it with zero copies (factoring G
        # directly would silently F-copy the whole matrix). trans=1 on the
        # factors then solves G·α = v. The factors — not the raw matrix —
        # are what gets stashed: the collocation G is not symmetric, so the
        # wire-loading adjoint oracle needs Gᵀw = r, which is the same
        # factorization at trans=0. One getrf serves both.
        lu_piv = scipy.linalg.lu_factor(G.T, overwrite_a=True)
        alpha = scipy.linalg.lu_solve(lu_piv, v, trans=1)

        feed_currents = np.array(
            [self._feed_segment_current(alpha, seg_view, fi) for fi in feed_segs],
            dtype=np.complex128,
        )
        z_per_feed = voltages / feed_currents
        Z_drive = z_per_feed[0] if len(self.feeds) == 1 else z_per_feed
        self.Z_factors = lu_piv  # LU of Gᵀ; adjoint solve = lu_solve(·, trans=0)
        return Z_drive, alpha

    def compute_y_matrix(self) -> np.ndarray:
        """Short-circuit admittance matrix [Y_sc] at the configured feeds.

        Y_sc[i, j] is the current flowing out of port i when port j is
        driven with V_j = 1 and every other port held at V_k = 0; invert
        to recover the open-circuit Z matrix for network analysis.
        Implementation mirrors `compute_impedance`: build G once, but
        solve with an N-column RHS where column j has unit excitation
        at port j's feed segment and zeros elsewhere. SinusoidalSolver
        uses Eq 187's delta-gap source which scales by `-1/h_i`; the
        Y matrix readout via `_feed_segment_current` already accounts
        for the basis arithmetic.
        """
        geom = self._build_geometry()
        G, seg_view = self._assemble_Z(geom, self.k)
        feed_segs = geom["feed_segs"]
        n_ports = len(feed_segs)
        n_segs = geom["n_segs"]

        # Column j: drive port j with V = 1 (RHS = -1/h_j at port j's
        # segment, 0 elsewhere). Stack all N columns and let LAPACK
        # do one LU + N back-subs.
        B = np.zeros((n_segs, n_ports), dtype=np.complex128)
        for j, fi in enumerate(feed_segs):
            B[fi, j] = -1.0 / geom["seg_h"][fi]
        alphas = scipy.linalg.solve(G, B)  # (n_segs, n_ports)

        Y = np.zeros((n_ports, n_ports), dtype=np.complex128)
        for j in range(n_ports):
            for i, fi in enumerate(feed_segs):
                Y[i, j] = self._feed_segment_current(alphas[:, j], seg_view, fi)
        return Y

    def compute_y_matrix_swept(self, k_array) -> np.ndarray:
        """Per-frequency Y matrices. Loops over k like
        `compute_impedance_swept` (no batched assembly here yet); returns
        an (n_k, n_ports, n_ports) array."""
        k_array = np.asarray(k_array, dtype=float)
        k_save = self.k
        wl_save = self.wavelength
        omega_save = self.omega
        geom = self._build_geometry()
        feed_segs = geom["feed_segs"]
        n_ports = len(feed_segs)
        n_segs = geom["n_segs"]
        # RHS columns are k-independent (depend only on segment lengths).
        B = np.zeros((n_segs, n_ports), dtype=np.complex128)
        for j, fi in enumerate(feed_segs):
            B[fi, j] = -1.0 / geom["seg_h"][fi]

        out = np.zeros((k_array.shape[0], n_ports, n_ports), dtype=np.complex128)
        for ki, kk in enumerate(k_array):
            self.k = float(kk)
            self.omega = self.k * self.c
            self.wavelength = self.c / (self.omega / (2 * np.pi))
            G, seg_view = self._assemble_Z(geom, self.k)
            alphas = scipy.linalg.solve(G, B)
            for j in range(n_ports):
                for i, fi in enumerate(feed_segs):
                    out[ki, i, j] = self._feed_segment_current(
                        alphas[:, j], seg_view, fi
                    )
        self.k = k_save
        self.wavelength = wl_save
        self.omega = omega_save
        return out

    def compute_impedance_swept(self, k_array):
        """Loop over wavenumbers. Per-call work that doesn't depend on k
        (geometry build, source-vector index, the set of bases that touch
        the feed segment) is lifted out of the loop so the per-k cost
        reduces to field-tensor + basis-coefs + assembly + solve. Together
        with the assemble_Z vectorization and the C++ field-tensor
        accelerator, this brings the n=21 sweep from ~70 ms to ~30 ms.
        """
        k_array = np.asarray(k_array, dtype=float)
        n_feeds = len(self.feeds)
        if n_feeds == 1:
            z_out = np.zeros(k_array.shape[0], dtype=np.complex128)
        else:
            z_out = np.zeros((k_array.shape[0], n_feeds), dtype=np.complex128)
        k_save = self.k
        wl_save = self.wavelength
        omega_save = self.omega
        geom = self._build_geometry()
        feed_segs = geom["feed_segs"]
        voltages = np.array([v for _, _, v in self.feeds], dtype=np.complex128)
        n_segs = geom["n_segs"]
        v = np.zeros(n_segs, dtype=np.complex128)
        for fi, V_i in zip(feed_segs, voltages):
            v[fi] += -V_i / geom["seg_h"][fi]
        for i, kk in enumerate(k_array):
            self._checkpoint()  # top of each frequency iteration
            self.k = float(kk)
            self.omega = self.k * self.c
            self.wavelength = self.c / (self.omega / (2 * np.pi))
            G, seg_view = self._assemble_Z(geom, self.k)
            alpha = scipy.linalg.solve(G, v)
            feed_currents = np.array(
                [self._feed_segment_current(alpha, seg_view, fi) for fi in feed_segs],
                dtype=np.complex128,
            )
            z_per_feed = voltages / feed_currents
            if n_feeds == 1:
                z_out[i] = z_per_feed[0]
            else:
                z_out[i] = z_per_feed
        self.k = k_save
        self.wavelength = wl_save
        self.omega = omega_save
        return z_out

    def currents_at_knots(self, alpha, s_array=None):
        """Per-wire complex current sampled at every mesh knot.

        Each basis j contributes (A_jn + B_jn sin(k·s_local) +
        C_jn cos(k·s_local))·σ_jn on every segment n in its support, with
        s_local measured from segment n's centre. The current at a knot
        between adjacent segments is the average of the right-edge value
        of the segment to its left and the left-edge value of the segment
        to its right (continuity makes them equal up to round-off; the
        average is the symmetric pick). Wire-endpoint knots use only the
        adjacent segment.

        When `s_array` is provided as a list of 1D arc-length arrays (one per
        wire), evaluates the basis sum at those arc positions instead of the
        mesh knots. Arc is measured from the wire's start (s=0) to its end
        (s=Σ h_seg). Samples that fall exactly on an interior knot return the
        symmetric average of the two adjacent segments (same as the default
        knot path); samples in the interior of a segment evaluate the basis
        directly on that segment.
        """
        alpha = np.asarray(alpha)
        geom = self._build_geometry()
        seg_view = self._basis_coefs(geom, self.k)
        seg_h = geom["seg_h"]
        n_wires = len(self.wires_polylines)

        # Pattern shared between both paths: each evaluation produces a
        # (segment, s_local) pair, a destination index in the flat output,
        # and a weight (1.0 for endpoints / interior samples, 0.5 each side
        # for interior-knot symmetric-average pairs). We batch them all up,
        # call `_evaluate_basis_at_points` once, then scatter-add into the
        # output. In the natural-tangent frame I = σA + B·sin(ks) + σC·cos(ks);
        # this lives inside `_evaluate_basis_at_points`. The earlier `σ` bug
        # (fixed 2026-06: multiplying the whole bracket by σ added a spurious
        # 2·B·sin(ks) at σ=−1 junction neighbours) is gone by construction.

        if s_array is None:
            # Per wire of M segs: M+1 knots. Each segment contributes two
            # edge evaluations — its left edge to the adjacent left knot,
            # its right edge to the adjacent right knot. The first seg's
            # left-edge and the last seg's right-edge hit wire endpoints
            # (weight 1.0); every other edge hits an interior knot shared
            # with another segment's edge, so each contributes weight 0.5.
            eval_seg_parts = []
            eval_s_parts = []
            eval_target_parts = []
            eval_weight_parts = []
            wire_knot_offsets = [0]
            for w_idx in range(n_wires):
                first = geom["wire_first"][w_idx]
                last = geom["wire_last"][w_idx]
                n_w_segs = last - first + 1
                h_w = seg_h[first : last + 1]
                base = wire_knot_offsets[-1]
                seg_arr = np.arange(first, last + 1, dtype=np.int64)
                # Left edges (s = -h/2) feed knot 0..M-1
                left_target = base + np.arange(n_w_segs, dtype=np.int64)
                left_weight = np.full(n_w_segs, 0.5)
                left_weight[0] = 1.0
                # Right edges (s = +h/2) feed knot 1..M
                right_target = base + np.arange(1, n_w_segs + 1, dtype=np.int64)
                right_weight = np.full(n_w_segs, 0.5)
                right_weight[-1] = 1.0
                eval_seg_parts.append(np.concatenate([seg_arr, seg_arr]))
                eval_s_parts.append(np.concatenate([-0.5 * h_w, +0.5 * h_w]))
                eval_target_parts.append(np.concatenate([left_target, right_target]))
                eval_weight_parts.append(np.concatenate([left_weight, right_weight]))
                wire_knot_offsets.append(base + n_w_segs + 1)

            eval_seg = np.concatenate(eval_seg_parts)
            eval_s = np.concatenate(eval_s_parts)
            eval_target = np.concatenate(eval_target_parts)
            eval_weight = np.concatenate(eval_weight_parts)

            I_evals = self._evaluate_basis_at_points(seg_view, eval_seg, eval_s, alpha)
            I_flat = np.zeros(wire_knot_offsets[-1], dtype=np.complex128)
            np.add.at(I_flat, eval_target, eval_weight * I_evals)
            return [
                I_flat[wire_knot_offsets[i] : wire_knot_offsets[i + 1]]
                for i in range(n_wires)
            ]

        # s_array path: per-wire, vectorized over the sample positions.
        # Each sample is either right on an interior knot (→ two evals,
        # weight 0.5 each, for the continuity-average) or interior to a
        # segment (→ one eval, weight 1.0).
        sampled = []
        for w_idx, sv in enumerate(s_array):
            sv = np.asarray(sv, dtype=np.float64)
            first = geom["wire_first"][w_idx]
            last = geom["wire_last"][w_idx]
            n_w_segs = last - first + 1
            wire_h = seg_h[first : last + 1]
            arc_at_knot = np.concatenate([[0.0], np.cumsum(wire_h)])
            wire_arc = float(arc_at_knot[-1])
            n_s = sv.shape[0]
            if n_s == 0:
                sampled.append(np.zeros(0, dtype=np.complex128))
                continue
            s_clip = np.clip(sv, 0.0, wire_arc)
            eps = 1e-12 * max(wire_arc, 1.0)
            knot_hit = np.searchsorted(arc_at_knot, s_clip)
            on_interior_knot = (
                (knot_hit > 0)
                & (knot_hit < n_w_segs)
                & (np.abs(s_clip - arc_at_knot[np.clip(knot_hit, 0, n_w_segs)]) <= eps)
            )
            non_knot_mask = ~on_interior_knot

            # Containing-segment evaluation for non-knot samples.
            seg_in_wire = np.searchsorted(arc_at_knot, s_clip, side="right") - 1
            seg_in_wire = np.clip(seg_in_wire, 0, n_w_segs - 1)
            nk_target = np.nonzero(non_knot_mask)[0]
            nk_seg_in_wire = seg_in_wire[non_knot_mask]
            nk_seg = first + nk_seg_in_wire
            nk_s = (s_clip[non_knot_mask] - arc_at_knot[nk_seg_in_wire]) - 0.5 * wire_h[
                nk_seg_in_wire
            ]
            nk_weight = np.ones(nk_seg.shape[0])

            # Two-sided evaluation for samples on an interior knot.
            k_target = np.nonzero(on_interior_knot)[0]
            k_idx = knot_hit[on_interior_knot]
            k_left_seg = first + k_idx - 1
            k_right_seg = first + k_idx
            k_left_s = +0.5 * seg_h[k_left_seg]
            k_right_s = -0.5 * seg_h[k_right_seg]
            k_weight = np.full(2 * k_target.shape[0], 0.5)

            all_seg = np.concatenate([nk_seg, k_left_seg, k_right_seg])
            all_s = np.concatenate([nk_s, k_left_s, k_right_s])
            all_target = np.concatenate([nk_target, k_target, k_target])
            all_weight = np.concatenate([nk_weight, k_weight])

            I_evals = self._evaluate_basis_at_points(seg_view, all_seg, all_s, alpha)
            I_out = np.zeros(n_s, dtype=np.complex128)
            np.add.at(I_out, all_target, all_weight * I_evals)
            sampled.append(I_out)
        return sampled
