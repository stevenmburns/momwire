"""Richmond-style piecewise-sinusoidal **Galerkin** solver (momwire#182).

`SinusoidalGalerkinSolver` shares everything geometric and basis-related with
`SinusoidalSolver` — the same three-term sinusoidal basis, the same N⁻/N⁺
neighbour tables that bake junction continuity / KCL into the basis-function
shapes — and differs in ONE place: how the EFIE is tested.

`SinusoidalSolver` collocates: it point-samples ŝ·E at each segment centre
(one test row per segment). This solver instead forms each test row by
**integrating** the field against the corresponding basis function, by Gauss
quadrature along the test segment(s). That makes it the missing
basis × testing cell (sinusoidal basis + Galerkin testing) beside the
point-matched sinusoidal solver and the B-spline Galerkin family.

Fill scheme (issue #182, chosen empirically in M1 — "let G1 decide"): reuse
the existing closed-form Eqs 76-79 field evaluators for the SOURCE integral
(analytic over each source segment) and Gauss-quadrature only the TEST
integral. M1 established that this is a structurally correct Galerkin fill:
‖Z − Zᵀ‖/‖Z‖ is exact on the diagonal and at roundoff for well-separated
pairs.

**M2 — the near-singular test integral.** M1 used one uniform Gauss rule for
every pair, which is the wrong tool for the pairs that touch. When the test
segment shares a node with the source segment (or IS the source segment), the
integrand carries a spike of width ~`a` (the thin-wire radius) located ~`a`
inside the shared endpoint: the Eqs 78-79 endpoint terms go as
Δz·(1+jkr₀)·G₀/r₀² with r₀ = √(a² + Δz²), which peaks at |Δz| ≈ a/√2 with
height ~1/a². A uniform rule cannot see a feature of relative width a/h until
it has ~h/a nodes — which is exactly why M1 needed `n_qp_test` ≈ 128 to reach
the G1 gate, and paid that cost on every one of the N² pairs.

M2 splits the quadrature instead of refining it globally:

* **Far pairs** keep one shared uniform rule (`n_qp_test`, default 8). Their
  integrands vary on the scale of the pair separation, so 8 nodes is already
  converged to roundoff.
* **Near pairs** — self, node-sharing neighbours, and any pair whose segments
  approach within `near_factor` of their combined half-lengths — are re-integrated
  on a composite rule whose panels shrink dyadically toward BOTH endpoints of
  the test segment, down to relative width `a/h` (`_graded_endpoint_rule`).
  Geometric grading plus a fixed per-panel Gauss order is the standard
  exponentially-convergent treatment for an endpoint-localized feature, and
  the panel count is *derived* from the geometry's own a/h — not tuned.

Because the near set is O(N) pairs rather than O(N²), the correction is
evaluated through `_field_components_bcast`, which pairs (P, G) quadrature
points against (P, 1) sources directly instead of forming the N² table those
pairs live in. Net effect: the G1 symmetry gate is met at the DEFAULT
quadrature, at ~1/16 the fill cost of M1's converged setting.

**M3 — the variational payoff, measured.** At 11-21 segments this solver's
driving-point impedance is closer to the fine-mesh answer than the
point-matched solver's, by 1.01× (k3_star) to 2.97× (k2_junction) worst-case
over a family of six references. Two things worth knowing when reading those
numbers, both pinned by tests in `tests/test_sinusoidal_galerkin.py`:

* The win is a **reactance** win. In R the point-matched solver actually
  converges faster on three of the four validation geometries; the net |Z|
  verdict is a reactance gain partly given back on resistance. That is the
  useful direction — the sin↔bs2 questions this solver exists to arbitrate are
  reactance-dominated.
* There is no "converged impedance" to measure against in an absolute sense:
  the delta-gap width IS the segment length, so refining the mesh shrinks the
  source, and X drifts logarithmically without limit while R settles. Every
  error figure above is relative to a *chosen* fine-mesh reference, and the
  gate is the verdict that survives the whole spread of defensible choices.

**M4 — ground models, by reuse.** All three of the point-matched solver's
grounds are wired through the same test quadrature, with NO new field
kernels: each one is the existing per-ground source-field evaluator called
with the Gauss points along the test segment as its observers instead of the
segment centres.

* **PEC image** — the mirrored-source build (`_image_source_centers_tangents`)
  through the same Eqs 76-79 evaluator, plain tangential projection.
* **Reflection-coefficient** (`ground_eps`) — the same mirrored build with
  NEC's Fresnel field dyad applied before the projection. The dyad's specular
  tables (cos θ, p̂) are per (observer, source) *pair*, so they are rebuilt at
  the quadrature points via `_ground_refl.specular_ray_tables_bcast` rather
  than reused from the segment-centre cache.
* **Sommerfeld** (`ground_model="sommerfeld"`) — NEC's C2·(PEC image) plus the
  smooth interpolated remainder, the latter from the point-matched solver's
  `_field_tensor_sommerfeld_remainder` with its observer set overridden to the
  quadrature points.

The image blocks get the SAME graded near-pair treatment as the free-space
block, selected against the mirrored source geometry — which matters exactly
when a wire touches or nearly touches the plane, because then a segment and
its own image share the endpoint that carries M2's width-`a` spike.

**M5 — junction ports: built, measured, and REFUSED.** #177 derived why the
point-matched solver cannot have them (every sinusoidal basis function
satisfies KCL at every junction as an algebraic identity, so a current with
nonzero net inflow at a node is outside the span at any mesh density) and
named what a real implementation would need: the port basis COLUMN is cheap,
the port ROW needs the segment-integrated field testing that point
collocation cannot provide. This solver HAS those rows, so the construction
was built here and taken all the way to the oracle:

* `_junction_port_view` appends #177's `g_p = (1/P_J)·Σ_m ext_m` to the CSR
  basis table — unit net inflow at the node, vanishing at every member's far
  end, verified to 1e-15 against the ordinary bases' identically-zero inflow;
* the port row is that column tested against the field, which costs no new
  kernel: G grows from (N, N) to (N+P, N+P) through the same quadrature and
  the same M2 near-pair correction, and stays symmetric to ~1e-11;
* drive and readout are exactly dual (`b_p = -V_p`, readout `α_p`), so the
  port block of Y is symmetric to 1.7e-14.

Structurally everything #177 asked for is there. **The physics is not**, and
the reason is one #177 did not anticipate. A current with unit net inflow at
node A terminates there, so it deposits a point charge q = I/(jω) AT the
node — and the Eqs 78-79 endpoint terms (the same width-`a` feature M2's
graded rule exists to resolve) charge that point charge its own self-energy,
regularized at the thin-wire radius rather than at the mesh scale:

    Z_pp ≈ 1/(jω · 4πε · a)   — measured to 2-8 % over a decade of `a`

That term is set by `a`, not by `h`, so it does not converge away; and it
cannot be cancelled, because every ordinary basis has zero net charge at a
junction by the very KCL identity #177 identified, so no combination of them
carries a matching charge. The whole span removes 2 % of it. The result is a
port that reads as the self-capacitance of the node instead of the antenna:
on `tests/test_junction_ports.py`'s paired-tip oracle the Zdiff misses the
bridged-gap reference by ~2400×, unchanged under mesh refinement.

`BSplineSolver` is not subject to this because its ports are a
MIXED-POTENTIAL construct: the node potential is an explicit Lagrange
multiplier on a KCL constraint row, i.e. the integration-by-parts BOUNDARY
term, held apart from the reaction integral. This solver is deliberately
field-based (Eqs 76-79 give the total E of each source shape, vector and
scalar potential already merged), so the boundary term is inside the kernel
and cannot be separated from it. That is #177's "node-voltage row" bullet,
now with a number attached.

So `junction_ports=` is accepted and validated here — the construction is
kept because the measurement that kills it is the only thing that stops it
being re-proposed, and it is pinned by tests — but every solve entry point
refuses.

**M5b — `node_ports=`, the port M5's mechanism does NOT forbid.** M5's
obstruction is constructive: it forbids exactly one thing, a basis that
TERMINATES current at a node. Formulation (a) terminates nothing. Join the
port's two terminals into one junction, so current flows THROUGH the node
under the ordinary KCL-identical span, and drive it with a ZERO-width
delta-gap EMF sitting exactly at the node, across a declared bipartition of
the junction's members:

    b_i = -V·f_i(node),     I_port = -b·α / V

`f_i(node)` is the cut vector — basis i's current crossing the gap, summed
over the + side's members. Two facts carry the whole formulation. Summed over
ALL the members it is the identically-zero KCL residual (#177's identity,
4e-13 here), so nothing accumulates at the node and M5's
Z_pp ≈ 1/(jω·4πε·a) has nothing to attach to. Summed over half of them it is
O(1), so the source has a well-defined excitation — even though the same
source point-samples to an identically ZERO RHS in the point-matched solver,
because a node is never a segment centre. That was #177's argument for why
NEC has no node port; read forwards it is a statement about the TESTING, and
this is the one capability the fourth basis × testing cell adds that
collocation cannot have.

No basis column is added, drive and readout are the same vector (so the node
ports' block of Y is symmetric to 1.3e-13 under either `feed_readout`, with
none of the gap feed's centre-vs-dual tradeoff), and a 0 V node port leaves
the solve bit-identical. Against the `_bridged_z` oracle — the same structure
re-joined by a real bridge wire of length delta carrying a real delta-gap
feed, extrapolated to delta → 0 — the node port lands 0.13-0.56 % away over
n_half 10-40 at K=2 and K=4, against G5b's 1.5 %, decaying with mesh; the
reference family's own spread is 0.013-0.041 %, so the delta → 0 idealization
is absorbed by the extrapolation rather than assumed away.

**What `node_ports` is not.** It is a two-terminal object. A ONE-terminal
net-inflow port at a lone conductor end — antennaknobs' `PortAtEnd`, which
resolves every one of `wire.sterba_bl`'s 16 ports to a ONE-member junction
whose return path is at a different node — has nothing to bipartition and
genuinely must accept net current at the node. That is `junction_ports`'
topology, and `junction_ports` is still refused here. `node_ports` refuses
a one-member junction at construction rather than silently modelling an open.

For the one-terminal case, use `BSplineSolver`; for a gap that happens to sit
at a node, `node_ports` now beats the workaround NEC-2 requires (mesh a short
bridge wire across the gap and gap-feed it), which is still available and is
what the oracle above uses as its reference.

One thing M5 did land: `compute_y_matrix` / `compute_y_matrix_swept` /
`compute_impedance_swept` are now honestly Galerkin. They were inherited
verbatim from the point-matched solver, which paired THIS solver's Galerkin
matrix with collocation's point RHS — #182 M4's finding 2 — so they did not
even agree with `compute_impedance`. They now share its drive and readout.
The delta-gap feed's readout is still not its drive's dual (`feed_readout`,
below), which is why a two-gap-feed Y is symmetric only to O(h).

Still deferred: wire loading and the C++ accelerator — both raise rather than
return plausible wrong numbers.
"""

import numpy as np
import scipy.linalg
import scipy.spatial.distance

from . import _ground_refl
from .sinusoidal import _EULER_GAMMA, SinusoidalSolver

# Pairs are corrected in blocks so the (P, G, n_qp_const) source-quadrature
# scratch inside the field kernel stays bounded regardless of model size.
_PAIR_BLOCK = 512


def _graded_endpoint_rule(eps, n_per_panel, leggauss):
    """Composite Gauss rule on [-1, 1] with panels graded toward BOTH ends.

    Panel widths double away from each endpoint — eps, 2·eps, 4·eps, … — so a
    feature of relative width `eps` sitting at an endpoint is resolved by a
    panel of its own size, and the smooth interior is covered by a handful of
    large panels. Returns (nodes, weights) with Σw = 2.

    `eps` is the feature width relative to the segment half-length (a/h for
    the thin-wire endpoint spike); `leggauss` is the solver's cached
    Gauss-Legendre factory.
    """
    eps = float(np.clip(eps, 1e-9, 0.5))
    # Distances from the endpoint at which panels break. Stop at the midpoint
    # so the largest panel is [-0.5, 0] and the grading ratio stays ~2.
    d = eps * 2.0 ** np.arange(64)
    d = d[d <= 0.5]
    left = np.concatenate(([-1.0], -1.0 + d, [0.0]))
    edges = np.unique(np.concatenate([left, -left[::-1]]))

    gx, gw = leggauss(n_per_panel)
    lo, hi = edges[:-1], edges[1:]
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    x = (mid[:, None] + half[:, None] * gx[None, :]).ravel()
    w = (half[:, None] * gw[None, :]).ravel()
    return x, w


def _plain_projection(cm, m_idx, n_idx):
    """Tangential projection of the Eqs 76-79 component tables — NEC's rule
    E_t = (t_obs·t_src)·E_z + ((ρ⃗·t_obs)/ρ')·E_ρ, the same expression
    `SinusoidalSolver._field_tensor`'s numpy path uses.

    Serves the free-space block and the PEC image block; the (test-segment,
    source-segment) index arrays are the projector-protocol signature (see
    `SinusoidalGalerkinSolver._tested_contribs`) and are unused here — only
    the Fresnel-weighted projector needs to look anything up per pair.
    """
    td = cm["td"]
    rp = cm["rho_proj_factor"]
    return (
        td * cm["Ez_const"] + rp * cm["Erho_const"],
        td * cm["Ez_sin"] + rp * cm["Erho_sin"],
        td * cm["Ez_cos"] + rp * cm["Erho_cos"],
    )


class SinusoidalGalerkinSolver(SinusoidalSolver):
    """Piecewise-sinusoidal Galerkin MoM. Same constructor surface as
    `SinusoidalSolver`, plus the test-quadrature controls:

    `n_qp_test`
        Gauss nodes per test segment for the shared FAR-pair rule.
    `n_qp_near`
        Gauss nodes per panel of the graded near-pair rule (M2). The default 8
        is where the G1 residual stops being near-pair quadrature error and
        hits the source-side `n_qp_const` floor (~2e-11) on every validation
        geometry; 4 leaves the bent/junction cases short of the gate.
    `near_factor`
        A pair is "near" when its segments approach within
        `near_factor · (h_m + h_n)/2`. The default 0.5 selects the self and
        node-sharing pairs (both at distance 0) while excluding the
        next-nearest collinear neighbour (at distance h), and additionally
        catches genuinely close non-touching wires.
    `near_correction`
        Set False to fall back to M1 behaviour (one uniform rule everywhere)
        — the contrast the M2 tests use to show the correction is what buys
        the symmetry gate.
    `feed_readout`
        Which functional reads the current at a DELTA-GAP feed (junction
        ports are unaffected — theirs is dual either way).

        ``"centre"`` (default) is the point-matched solver's and NEC's: the
        current AT the feed segment's centre. ``"variational"`` is the exact
        dual of the Galerkin drive: the gap-averaged current
        (1/h)∫_gap J ds, i.e. Y = −Uᵀ G⁻¹ U on the drive columns U.

        They differ because the delta-gap source this solver inherits is
        NEC's — E_app = V/Δ spread over the whole feed segment — whose dual
        readout is the gap AVERAGE, not the gap centre. The default keeps
        the point-matched sibling's readout so that the two sinusoidal cells
        differ in exactly one thing (the testing), which is what the whole
        basis × testing instrument is for; the price is that a multiport Y
        with a gap feed in it is symmetric only to the O(h) difference
        between the two functionals (6.7e-5 at N=21, 6.5e-6 at N=81 on the
        two-feed dipole — better than the point-matched solver's own 9.1e-5
        / 7.3e-6 there, and decaying). ``"variational"`` makes that Y
        symmetric to ~1e-12 instead, and costs the M3 payoff gate: on
        k3_star the worst-case errColl/errGal falls from 1.014 to 0.797, so
        it is offered, measured and documented rather than made the default.

    All three ground models are wired (M4): `ground_z` alone gives the PEC
    image, `+ ground_eps` NEC's reflection-coefficient ground, and
    `+ ground_model="sommerfeld"` the Sommerfeld/Norton ground — each by
    reusing that ground's existing source-field evaluator under this test
    quadrature. One inherited caveat, shared with the point-matched solver and
    pinned by `test_finite_ground_at_a_ground_contact_is_an_inherited_defect`:
    a wire END LYING IN the plane is only sound under the PEC ground, because
    #151's ground-connected basis completes the end current with an exact
    mirror image that a finite ground does not provide.

    `node_ports`
        Two-terminal ports located AT a junction node (M5b formulation (a)):
        a zero-width delta-gap EMF across a declared bipartition of the
        junction's members, so current flows THROUGH the node and nothing
        terminates there. Entries are `(junction_index, side_a)` (voltage 0)
        or `(junction_index, side_a, voltage)`, where `side_a` indexes into
        `junctions[junction_index]` and must be a nonempty PROPER subset —
        both sides of the gap need a conductor.

        Ports are ordered [gap feeds…, junction ports…, node ports…]. Drive
        and readout are the same vector, so a node port's Y block is
        machine-symmetric under either `feed_readout` and costs none of the
        M3 payoff. Which side is called "+" is a convention the impedance
        does not see (2.5e-16). Grounded junctions are rejected (#151: the
        node's current closes through its image, not on its partners), as is
        a ONE-member junction — that is a one-terminal net-inflow port, i.e.
        `junction_ports`, which this solver refuses. The point-matched
        sibling has no equivalent and takes no such keyword: a node source
        samples to an identically zero collocation RHS.

    `junction_ports=` is ACCEPTED and validated here (with `BSplineSolver`'s
    rules: a junction index or an (index, voltage) pair, in range, no
    repeats, no grounded junction, and `feeds=[]` legal when there is at
    least one port) so that #177's port basis can be built and measured —
    but every solve entry point refuses, because the measurement says the
    construction is wrong by ~2400× for a reason no amount of mesh fixes.
    See the module docstring for the mechanism and
    `tests/test_junction_ports.py` for the pinned numbers. Use a meshed
    bridge wire with a gap feed instead, which is what NEC-2 requires too.

    Wire loading and the C++ accelerator are deliberately not wired yet and
    raise where reached.
    """

    def __init__(
        self,
        *,
        n_qp_test=8,
        n_qp_near=8,
        near_factor=0.5,
        near_correction=True,
        feed_readout="centre",
        node_ports=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_qp_test = int(n_qp_test)
        self.n_qp_near = int(n_qp_near)
        self.near_factor = float(near_factor)
        self.near_correction = bool(near_correction)
        if feed_readout not in ("centre", "variational"):
            raise ValueError(
                f"feed_readout must be 'centre' or 'variational', got {feed_readout!r}"
            )
        self.feed_readout = feed_readout
        self.node_ports = self._validate_node_ports(node_ports)
        # (base seg_view, k) → port-augmented seg_view. Keyed by identity on
        # the inherited view, which `_basis_coefs` already caches per (geom,
        # k, radius), so this rides that cache's validation.
        self._port_basis_cache = None

    # ------------------------------------------------------------------
    # Node ports (M5b formulation (a)) — a delta-gap EMF AT a junction node
    # ------------------------------------------------------------------

    def _validate_node_ports(self, node_ports):
        """Normalize `node_ports=` to [(j_idx, (member indices…), voltage), …].

        Accepted entry forms are `(j_idx, side_a)` (voltage 0) and
        `(j_idx, side_a, voltage)`, with `side_a` a sequence of indices into
        `self.junctions[j_idx]` naming the members on the port's + terminal.
        """
        if not node_ports:
            return []
        out = []
        seen = set()
        junction_port_idx = {j for j, _v in self.junction_ports}
        for entry in node_ports:
            if not isinstance(entry, (tuple, list)) or len(entry) not in (2, 3):
                raise ValueError(
                    "node_ports entries must be (junction_index, side_a) or "
                    f"(junction_index, side_a, voltage), got {entry!r}"
                )
            j_idx = int(entry[0])
            side_a = tuple(int(m) for m in entry[1])
            volts = complex(entry[2]) if len(entry) == 3 else 0j
            if not 0 <= j_idx < len(self.junctions):
                raise ValueError(
                    f"node_ports: junction index {j_idx} out of range "
                    f"(have {len(self.junctions)} junctions)"
                )
            if j_idx in seen:
                raise ValueError(f"node_ports: junction {j_idx} listed twice")
            if j_idx in junction_port_idx:
                raise ValueError(
                    f"junction {j_idx} is declared as both a junction_port and "
                    "a node_port — they are different port objects (net inflow "
                    "at the node vs an EMF across the node) and cannot share a "
                    "junction"
                )
            seen.add(j_idx)
            members = self.junctions[j_idx]
            if len(members) < 2:
                raise ValueError(
                    f"node_ports: junction {j_idx} has {len(members)} member(s); "
                    "a node port is an EMF ACROSS the node, so it needs at "
                    "least two wire-ends to put on opposite sides of the gap. "
                    "A one-terminal net-inflow port at a lone conductor end is "
                    "a junction_port, which this solver refuses (momwire#182 M5)"
                )
            if len(set(side_a)) != len(side_a):
                raise ValueError(f"node_ports: repeated member index in {side_a!r}")
            if any(not 0 <= m < len(members) for m in side_a):
                raise ValueError(
                    f"node_ports: member index out of range in {side_a!r} "
                    f"(junction {j_idx} has {len(members)} members)"
                )
            if not 0 < len(side_a) < len(members):
                raise ValueError(
                    f"node_ports: side_a {side_a!r} must be a nonempty PROPER "
                    f"subset of junction {j_idx}'s {len(members)} members — "
                    "both sides of the gap need at least one conductor"
                )
            out.append((j_idx, side_a, volts))
        return out

    def _reject_junction_ports(self):
        """No-op at CONSTRUCTION: #177's stated blocker was the missing
        segment-integrated test row and this solver's rows are segment
        integrals, so the port basis is buildable here and gets built. The
        refusal moved to `_refuse_junction_port_solve`, which is where the
        measurement that actually kills it belongs."""

    def _refuse_junction_port_solve(self):
        """Refuse any solve that would return a junction-port number.

        The construction is sound and the matrix is symmetric; what fails is
        the physics, quantified in the module docstring and pinned by
        `tests/test_junction_ports.py`: the port basis's node charge is
        priced at the thin-wire radius (Z_pp ≈ 1/(jω·4πε·a)), the KCL-clean
        span removes 2 % of that, and the paired-tip oracle lands ~2400×
        from its bridged-gap reference at every mesh.
        """
        if not self.junction_ports:
            return
        raise NotImplementedError(
            "junction_ports are built but not solvable on "
            "SinusoidalGalerkinSolver: the port basis necessarily deposits a "
            "point charge at the node, and this family's field kernel prices "
            "that charge at the wire radius (Z_pp ~ 1/(jw*4*pi*eps*a)), which "
            "no mesh refinement removes and the KCL-clean span cannot cancel "
            "— the paired-tip oracle misses its bridged-gap reference by "
            "~2400x (momwire#182 M5, extending #177). Use BSplineSolver; if "
            "the port's two terminals meet at ONE node, use node_ports= "
            "instead (momwire#182 M5b), or do what NEC requires: mesh a short "
            "bridge wire across the gap and gap-feed it"
        )

    def _node_cut_vectors(self, geom, seg_view, k):
        """(n_basis, P_node) — per node port, each basis's current THROUGH the
        node from the port's + side to its − side.

        Member m of the junction contributes σ_m·I_{i,m}(node) where σ_m is
        `_junction_members`' sign (+1 when the node is the member segment's
        natural end-2, −1 when it is end-1) and I_{i,m} is basis i's current
        shape σA + B·sin(kξ) + σC·cos(kξ) evaluated at ξ = σ_m·h_m/2. That
        product is basis i's current flowing INTO the node along member m, so
        summing it over the + side gives the current crossing the gap.

        Two facts make this the whole of formulation (a):

        * The sum over ALL members is identically zero — #177's KCL identity,
          the same property that makes a net-inflow junction port impossible
          here. So the cut vector is antisymmetric under swapping the sides
          and the port carries no net charge into the node: nothing terminates
          there, and M5's Z_pp ≈ 1/(jω·4πε·a) node self-energy never appears.
        * It is not identically zero, because current flows THROUGH the node.
          A point EMF sitting exactly at the node therefore has a well-defined
          Galerkin excitation, U_i = f_i(node)·V, even though it point-samples
          to nothing at any collocation point — which is #177's observation
          that a node source has an identically zero RHS in the point-matched
          solver, read as a statement about the TESTING rather than the basis.
        """
        N = geom["n_segs"]
        n_basis = N + len(self.junction_ports)
        grounded = geom["grounded_junctions"]
        out = np.zeros((n_basis, len(self.node_ports)), dtype=np.complex128)
        starts = seg_view["starts"]
        seg_h = np.asarray(geom["seg_h"], dtype=float)
        for p, (j_idx, side_a, _v) in enumerate(self.node_ports):
            if j_idx in grounded:
                raise ValueError(
                    f"junction {j_idx} is both grounded and a node port — a "
                    "node in the ground plane carries current into its own "
                    "image (#151), so the members' currents do not close on "
                    "each other and there is no through-current to drive"
                )
            members = self._junction_members(geom, j_idx)
            for mi in side_a:
                m, sgn = members[mi]
                half = 0.5 * float(seg_h[m])
                s, e = starts[m], starts[m + 1]
                sig = seg_view["sigma"][s:e]
                val = (
                    sig * seg_view["A"][s:e]
                    + seg_view["B"][s:e] * np.sin(k * half * sgn)
                    + sig * seg_view["C"][s:e] * np.cos(k * half)
                )
                np.add.at(out[:, p], seg_view["jbasis"][s:e], sgn * val)
        return out

    @property
    def n_ports(self):
        """Network ports, in order: gap feeds, junction ports, node ports."""
        return len(self.feeds) + len(self.junction_ports) + len(self.node_ports)

    # ------------------------------------------------------------------
    # Near-pair selection
    # ------------------------------------------------------------------

    def _near_pairs(self, geom, src_c=None, src_t=None, n_samples=5):
        """Ordered (test, source) segment pairs whose test integral needs the
        graded rule, as two index arrays.

        Selection is geometric rather than topological, so it covers the self
        pair, node-sharing neighbours, AND close-approach pairs that share no
        node (closely spaced parallel runs) — all of which put a sub-segment
        length scale into the test integrand.

        `src_c` / `src_t` default to the geometry's own segments (the
        free-space block). The M4 image blocks pass the MIRRORED source
        geometry, which is what makes the correction fire for a wire touching
        the ground plane: there the end segment and its own image share the
        in-plane endpoint, so the image block carries the same width-`a`
        endpoint spike the free-space self pair does. Image half-lengths are
        unchanged by mirroring, so `reach` is still the geometry's own.

        Exact segment-segment distance is only computed for pairs that survive
        a cheap centre-distance prefilter. That prefilter is a strict superset:
        |c_m − c_n| ≤ d_min + h_m + h_n always, so any pair meeting the
        `near_factor` criterion also meets the prefilter.
        """
        c = geom["seg_centers"]
        t = geom["seg_tangents"]
        cs = c if src_c is None else src_c
        ts = t if src_t is None else src_t
        hh = 0.5 * np.asarray(geom["seg_h"], dtype=float)
        reach = hh[:, None] + hh[None, :]

        # cdist rather than an explicit (N, N, 3) difference: the prefilter
        # should not cost 3× the memory of the matrix it is protecting.
        dc = scipy.spatial.distance.cdist(c, cs)
        cand = np.argwhere(dc <= (1.0 + self.near_factor) * reach)
        m, n = cand[:, 0], cand[:, 1]

        s = np.linspace(-1.0, 1.0, n_samples)

        def _gap(ai, bi, ca, ta, cb, tb):
            """min over sample points of segment `ai` (geometry ca/ta) of the
            distance to the clamped axis of segment `bi` (geometry cb/tb)."""
            pts = (
                ca[ai][:, None, :]
                + (hh[ai][:, None] * s[None, :])[:, :, None] * ta[ai][:, None, :]
            )  # (P, n_samples, 3)
            d = pts - cb[bi][:, None, :]
            u = np.einsum("pgd,pd->pg", d, tb[bi])
            u = np.clip(u, -hh[bi][:, None], hh[bi][:, None])
            perp = d - u[..., None] * tb[bi][:, None, :]
            return np.linalg.norm(perp, axis=-1).min(axis=1)

        gap = np.minimum(
            _gap(m, n, c, t, cs, ts),  # test segment sampled against the source
            _gap(n, m, cs, ts, c, t),  # and the other way round
        )
        keep = gap <= self.near_factor * reach[m, n]
        return m[keep], n[keep]

    # ------------------------------------------------------------------
    # Junction ports (M5) — the port basis column
    # ------------------------------------------------------------------

    def _junction_members(self, geom, j_idx):
        """(segment index, σ) for every wire-end at junction `j_idx`.

        σ is the sign the extension shape below is stamped with: +1 when the
        junction node is at the member segment's natural end-2 (`seg_r`), −1
        when it is at end-1 (`seg_l`) — the same L/R rule `_build_geometry`
        applies to a real N⁻ neighbour. Under the current-shape convention
        I(s) = σA + B·sin(ks) + σC·cos(ks) that sign is exactly what makes
        the shape's current flow INTO the node in both orientations.
        """
        members = []
        for w, end in self.junctions[j_idx]:
            if end == "start":
                members.append((geom["wire_first"][w], -1))
            else:
                members.append((geom["wire_last"][w], +1))
        return members

    def _junction_port_view(self, geom, k, base_view):
        """Append one basis column per junction port to the CSR seg-view.

        Port p's basis is #177's `g_p = (1/P_J)·Σ_m ext_m`, where `ext_m` is
        the three-term extension a real N⁻ neighbour would carry on member
        segment m at unit Q,

            A_m = a_m/sin(kΔ_m),  B_m = a_m/(2cos(kΔ_m/2)),
            C_m = −a_m/(2sin(kΔ_m/2)),

        with `a_m` the member's own Eq-25 log constant. That shape is zero at
        the member's FAR end and carries current `a_m(1−cos kΔ_m)/sin kΔ_m`
        — the same `P_minus_atom` the basis coefficients are built from —
        into the node. Summing over the members and dividing by
        `P_J = Σ_m P_minus_atom[m]` therefore gives a current distribution
        with **unit net inflow at the node**, KCL-clean everywhere else, and
        vanishing outside the K member segments: precisely the vector #177
        proved the ordinary span cannot contain.

        The entries are merged into the same segment-major CSR the inherited
        `_basis_coefs` returns, with basis indices N…N+P−1. Everything
        downstream — the test quadrature, the M2 near-pair correction, the
        source-side coefficient matrices, `_feed_segment_current`,
        `currents_at_knots` — then treats them as ordinary bases, which is
        why the port row is a Galerkin row and not a bolted-on constraint.
        """
        N = geom["n_segs"]
        grounded = geom["grounded_junctions"]
        for j_idx, _v in self.junction_ports:
            if j_idx in grounded:
                raise ValueError(
                    f"junction {j_idx} is both grounded and a junction port — "
                    "a node in the ground plane is connected to its own image "
                    "instead of to its partners (#151), so its voltage is "
                    "pinned and it cannot also be a driven port"
                )

        seg_h = np.asarray(geom["seg_h"], dtype=float)
        a = (
            self._uniform_radius
            if self._uniform_radius is not None
            else self._seg_radius(geom)
        )
        a_const = 1.0 / (np.log(2.0 / (k * a)) - _EULER_GAMMA)
        a_const = np.broadcast_to(np.asarray(a_const, dtype=float), (N,))
        kd = k * seg_h
        atom = (1.0 - np.cos(kd)) / np.sin(kd) * a_const

        segs, bases, A, B, C, sig = [], [], [], [], [], []
        for p, (j_idx, _v) in enumerate(self.junction_ports):
            members = self._junction_members(geom, j_idx)
            m_seg = np.array([m for m, _s in members], dtype=np.int64)
            m_sig = np.array([s for _m, s in members], dtype=np.int8)
            P_J = float(atom[m_seg].sum())
            if P_J == 0.0:
                raise ValueError(
                    f"junction_ports: junction {j_idx} has a degenerate port "
                    "normalization (Σ P⁻ atoms = 0) at this mesh"
                )
            q = a_const[m_seg] / P_J
            segs.append(m_seg)
            bases.append(np.full(m_seg.shape, N + p, dtype=np.int64))
            A.append(q / np.sin(kd[m_seg]))
            B.append(q / (2.0 * np.cos(0.5 * kd[m_seg])))
            C.append(-q / (2.0 * np.sin(0.5 * kd[m_seg])))
            sig.append(m_sig)

        starts = base_view["starts"]
        base_seg = np.repeat(np.arange(N, dtype=np.int64), np.diff(starts))
        all_seg = np.concatenate([base_seg] + segs)
        all_basis = np.concatenate([base_view["jbasis"]] + bases)
        all_A = np.concatenate([base_view["A"]] + A).astype(np.complex128)
        all_B = np.concatenate([base_view["B"]] + B).astype(np.complex128)
        all_C = np.concatenate([base_view["C"]] + C).astype(np.complex128)
        all_sigma = np.concatenate([base_view["sigma"]] + sig)

        order = np.argsort(all_seg, kind="stable")
        new_starts = np.zeros(N + 1, dtype=np.int64)
        np.cumsum(np.bincount(all_seg, minlength=N), out=new_starts[1:])
        return {
            "starts": new_starts,
            "jbasis": all_basis[order],
            "A": all_A[order],
            "B": all_B[order],
            "C": all_C[order],
            "sigma": all_sigma[order],
        }

    def _basis_coefs(self, geom, k):
        """The inherited CSR basis view, extended with the junction-port
        basis columns. Verbatim passthrough when there are no ports."""
        base_view = super()._basis_coefs(geom, k)
        if not self.junction_ports:
            return base_view
        cached = self._port_basis_cache
        if cached is not None and cached[0] is base_view and cached[1] == k:
            return cached[2]
        view = self._junction_port_view(geom, k, base_view)
        self._port_basis_cache = (base_view, k, view)
        return view

    # ------------------------------------------------------------------
    # Galerkin matrix assembly
    # ------------------------------------------------------------------

    def _test_context(self, geom, seg_view, k):
        """Everything the TEST side of the Galerkin integral needs, built once
        per assembly and shared by the free-space block and every ground block.

        Holds the Gauss points along each test segment (as a flat (N·nq, 3)
        observer list), the per-observer thin-wire radius, and `w_entry` —
        the (nnz, nq) table of test-function value × arc weight for every
        (segment, basis) support entry. For each such entry the test
        function's value at quad node q is
        f_{i,m}(ξ_q) = σA + B·sin(kξ_q) + σC·cos(kξ_q) (the current-shape
        convention pinned in `_evaluate_basis_at_points`) and the arc weight
        is gw_q·(h_m/2).
        """
        N = geom["n_segs"]
        seg_c = geom["seg_centers"]  # (N, 3)
        seg_t = geom["seg_tangents"]  # (N, 3)
        hh = 0.5 * np.asarray(geom["seg_h"], dtype=float)  # (N,) half-lengths

        gx, gw = self._leggauss_cached(self.n_qp_test)  # nodes on [-1, 1]
        nq = gx.shape[0]
        xi = hh[:, None] * gx[None, :]  # (N, nq) local arc from each centre
        obs_pts = seg_c[:, None, :] + xi[:, :, None] * seg_t[:, None, :]  # (N,nq,3)
        obs_c = np.ascontiguousarray(obs_pts.reshape(N * nq, 3))
        obs_t = np.ascontiguousarray(
            np.broadcast_to(seg_t[:, None, :], (N, nq, 3)).reshape(N * nq, 3)
        )
        a_seg = None if self._uniform_radius is not None else self._seg_radius(geom)
        # Scalar on the uniform path, (N·nq, 1) per-observer column when mixed
        # — the same shape contract `_field_components_bcast` takes for `a`.
        a_obs = self._uniform_radius if a_seg is None else np.repeat(a_seg, nq)[:, None]

        starts = seg_view["starts"]
        counts = np.diff(starts)
        m_of_entry = np.repeat(np.arange(N, dtype=np.int64), counts)  # (nnz,)
        sig = seg_view["sigma"].astype(np.complex128)
        sigA = sig * seg_view["A"]
        B = seg_view["B"]
        sigC = sig * seg_view["C"]
        fval = (
            sigA[:, None]
            + B[:, None] * np.sin(k * xi)[m_of_entry]
            + sigC[:, None] * np.cos(k * xi)[m_of_entry]
        )  # (nnz, nq)
        w_entry = (gw[None, :] * hh[m_of_entry][:, None]) * fval  # (nnz, nq)

        return {
            "N": N,
            "nq": nq,
            "hh": hh,
            "obs_c": obs_c,
            "obs_t": obs_t,
            # which test segment each flat observer belongs to — the row
            # coordinate a per-segment-pair projector table indexes with.
            "m_of_obs": np.repeat(np.arange(N, dtype=np.int64), nq),
            "a_obs": a_obs,
            "a_seg": a_seg,
            "starts": starts,
            "counts": counts,
            "m_of_entry": m_of_entry,
            "i_of_entry": seg_view["jbasis"],
            "sigA": sigA,
            "B": B,
            "sigC": sigC,
            "w_entry": w_entry,
        }

    def _tested_contrib(self, ctx, Phi):
        """contrib[entry, n] = Σ_q w_entry[entry, q] · Phi[m(entry), q, n].

        Accumulated one quadrature node at a time: the equivalent einsum over
        `Phi[m_of_entry]` would first materialize an (nnz, nq, N) gather, nq×
        the peak memory for the same arithmetic.
        """
        w_entry = ctx["w_entry"]
        m_of_entry = ctx["m_of_entry"]
        out = np.zeros((w_entry.shape[0], ctx["N"]), dtype=np.complex128)
        for q in range(ctx["nq"]):
            out += w_entry[:, q, None] * Phi[m_of_entry, q, :]
        return out

    def _tested_contribs(self, geom, k, ctx, projector, src_c=None, src_t=None):
        """Test-integrate one source block: (contrib_const, sin, cos), each
        (nnz, N).

        `src_c` / `src_t` are the source geometry the field evaluator sees —
        the geometry's own segments for the free-space block, the mirrored
        ones for a ground image block.

        `projector(cm, m_idx, n_idx)` turns the unprojected Eqs 76-79
        component tables into the three tangential field tables. `m_idx` /
        `n_idx` are broadcastable index arrays naming the (test segment,
        source segment) each entry of `cm` belongs to, so a projector can
        look up per-segment-pair tables regardless of how the caller paired
        things: (M, 1) against (1, N) for the full block here, (P, 1) against
        (P, 1) for the near-pair path's one-to-one pairing.

        Two-tier quadrature (M2): a shared uniform rule for the far pairs,
        overwritten per near pair by the endpoint-graded rule.
        """
        N = ctx["N"]
        nq = ctx["nq"]
        src_c = geom["seg_centers"] if src_c is None else src_c
        src_t = geom["seg_tangents"] if src_t is None else src_t

        cm = self._field_components_bcast(
            k,
            obs_c=ctx["obs_c"][:, None, :],  # (M, 1, 3)
            obs_t=ctx["obs_t"][:, None, :],  # (M, 1, 3)
            a=ctx["a_obs"],
            src_c=src_c[None, :, :],
            src_t=src_t[None, :, :],
            src_hh=ctx["hh"][None, :],
        )
        Phi = projector(cm, ctx["m_of_obs"][:, None], np.arange(N)[None, :])
        contribs = tuple(self._tested_contrib(ctx, P.reshape(N, nq, N)) for P in Phi)

        if self.near_correction:
            self._apply_near_correction(geom, k, ctx, contribs, projector, src_c, src_t)
        return contribs

    def _refl_projection(self, geom, eps_t):
        """Projector applying NEC's Fresnel field dyad to the image-source
        field before the tangential projection — the `ground_eps`
        reflection-coefficient ground (IPERF=0).

        Identical algebra to `SinusoidalSolver._field_tensor_image_refl`'s
        numpy path,

            t_m · D · E = ρ_v·(t_m·E) − (ρ_v + ρ_h)·(t_m·p̂)·(E·p̂),

        reading the specular tables straight out of `_image_refl_prep` — the
        same per-SEGMENT-PAIR cache the point-matched solver uses. That is
        deliberate and it is what NEC's IPERF=0 model actually says: ρ_v/ρ_h
        are evaluated once at the midpoint-to-image-midpoint specular angle
        and held **constant over the segment pair**. The test quadrature
        integrates the field, not the reflection coefficient.

        The alternative — re-deriving the specular ray at each test
        quadrature point — was implemented and rejected on measurement: it
        makes the weight a function of where on the test segment you are,
        which no longer has a partner term in the transposed entry, and the
        Galerkin matrix loses reciprocity (‖G−Gᵀ‖/‖G‖ = 3.0e-9 on the
        h=0.1λ dipole, 2.3e-9 on the L-shape, and *insensitive to
        quadrature refinement* — 3.00e-9 at n_qp_test 8, 16 and 32, i.e.
        structural). The pair-constant form below sits at the free-space
        floor instead. Since the observers are points ON test segment m,
        `tm_p = t_m·p̂` from the segment tangents is exact for them, not an
        approximation.
        """
        cos_th, px, py, tm_p, tn_p = self._image_refl_prep(geom)
        rho_v_t, rho_h_t = _ground_refl.fresnel_rho(eps_t, cos_th)

        def _project(cm, m_idx, n_idx):
            rho_v = rho_v_t[m_idx, n_idx]
            rvh = rho_v + rho_h_t[m_idx, n_idx]  # → 0 in the PEC limit
            tmp = tm_p[m_idx, n_idx]
            tnp = tn_p[m_idx, n_idx]
            # ρ̂·p̂ from the image-build rho_vec (p̂ is horizontal, so only the
            # x/y components contribute), same radius-regularized denominator
            # as the E_ρ projection rule.
            rho_p = (
                cm["rho_vec"][..., 0] * px[m_idx, n_idx]
                + cm["rho_vec"][..., 1] * py[m_idx, n_idx]
            ) / cm["rho_eval"]
            td = cm["td"]
            rp = cm["rho_proj_factor"]

            def _weighted(Ez, Erho):
                tm_E = td * Ez + rp * Erho  # t_m · E
                E_p = tnp * Ez + rho_p * Erho  # E · p̂
                return rho_v * tm_E - rvh * tmp * E_p

            return (
                _weighted(cm["Ez_const"], cm["Erho_const"]),
                _weighted(cm["Ez_sin"], cm["Erho_sin"]),
                _weighted(cm["Ez_cos"], cm["Erho_cos"]),
            )

        return _project

    def _tested_ground_block(self, geom, k, ctx):
        """The ground sub-assembly, tested exactly like the free-space block
        and SUBTRACTED from it by the caller — the same single global minus
        sign the point-matched `_assemble_Z` uses (the image current + image
        charge sign flips reduce to it).

        Reuse, per ground, of the evaluator the point-matched solver already
        validated against its own references:

        * PEC image — the mirrored-source Eqs 76-79 build, plain projection;
        * `ground_eps` refl-coef — the same build with the Fresnel dyad
          (`_refl_projection`);
        * `ground_model="sommerfeld"` — NEC's decomposition, C2·(PEC image)
          minus the smooth interpolated remainder, so that the caller's
          subtraction reproduces `Phi_free − C2·Phi_img + S`.
        """
        src_c_img, src_t_img = self._image_source_centers_tangents(geom)
        if self.ground_eps is None:
            return self._tested_contribs(
                geom, k, ctx, _plain_projection, src_c_img, src_t_img
            )

        eps_t = _ground_refl.eps_tilde(self.ground_eps, self.omega, self.eps)
        if self.ground_model == "sommerfeld":
            c2 = (eps_t - 1.0) / (eps_t + 1.0)
            img = self._tested_contribs(
                geom, k, ctx, _plain_projection, src_c_img, src_t_img
            )
            rem = self._tested_sommerfeld_remainder(geom, k, ctx, eps_t)
            return tuple(c2 * a - b for a, b in zip(img, rem))

        return self._tested_contribs(
            geom, k, ctx, self._refl_projection(geom, eps_t), src_c_img, src_t_img
        )

    def _tested_sommerfeld_remainder(self, geom, k, ctx, eps_t):
        """Test-integrate the smooth Sommerfeld remainder tensor.

        The remainder evaluator is the point-matched solver's, called with the
        test-quadrature points as its observer set. No near-pair correction:
        the remainder kernel lives on the distance to the IMAGE point, which
        stays smooth even where a wire touches the plane (r₁ → 0 has a finite
        limit the grid carries), so the width-`a` endpoint spike the graded
        rule exists for is absent here. The residual that IS left on a
        ground-touching wire is this block's own SOURCE-side rule
        (`n_qp_sommerfeld`), which
        `test_g4_sommerfeld_symmetry_near_the_plane_is_source_quadrature_limited`
        identifies by showing that refining the test rule does not move it
        while refining the source rule drives it to the free-space floor.
        """
        N = ctx["N"]
        nq = ctx["nq"]
        S = self._field_tensor_sommerfeld_remainder(
            geom, k, eps_t, obs_centers=ctx["obs_c"], obs_tangents=ctx["obs_t"]
        )
        return tuple(self._tested_contrib(ctx, s.reshape(N, nq, N)) for s in S)

    def _assemble_Z(self, geom, k):
        """Galerkin system matrix G (basis i tested against source basis j).

        G[i, j] = ∫ f_i(s) · ŝ·E_j(s) ds
                = Σ_shape ( T[shape] @ M[shape] )[i, j]

        where T[shape, i, n] integrates the closed-form field of unit
        `shape` (const/sin/cos) current on source segment n against test
        basis i (quadrature along i's support), and M[shape][n, j] is the
        effective coefficient source basis j puts on segment n — the SAME
        source-side coefficient matrices the collocation path builds.

        With `ground_z` set, the ground's tested sub-assembly is subtracted
        from the free-space one before the scatter — one more source block
        through the same test quadrature, not a second scheme.

        With junction ports the index `i`/`j` runs over N+P bases rather than
        N (the port columns appended by `_junction_port_view`), so G grows to
        (N+P, N+P). The source index `n` still runs over the N segments —
        a port basis is a current distribution on real segments like any
        other, so it needs no new field kernel and gets none.
        """
        if self._loading_active:
            raise NotImplementedError(
                "SinusoidalGalerkinSolver wire loading (Galerkin overlap form) "
                "is not implemented yet (momwire#182)"
            )

        seg_view = self._basis_coefs(geom, k)
        ctx = self._test_context(geom, seg_view, k)
        N = ctx["N"]

        contribs = self._tested_contribs(geom, k, ctx, _plain_projection)
        if self.ground_z is not None:
            gnd = self._tested_ground_block(geom, k, ctx)
            contribs = tuple(c - g for c, g in zip(contribs, gnd))

        i_of_entry = ctx["i_of_entry"]
        n_basis = N + len(self.junction_ports)

        def _scatter(contrib):
            T = np.zeros((n_basis, N), dtype=np.complex128)
            np.add.at(T, i_of_entry, contrib)
            return T

        T_c, T_s, T_co = (_scatter(c) for c in contribs)

        # --- Source-side coefficient matrices (identical to collocation) --
        m_of_entry = ctx["m_of_entry"]
        M_A = np.zeros((N, n_basis), dtype=np.complex128)
        M_B = np.zeros((N, n_basis), dtype=np.complex128)
        M_C = np.zeros((N, n_basis), dtype=np.complex128)
        M_A[m_of_entry, i_of_entry] = ctx["sigA"]
        M_B[m_of_entry, i_of_entry] = ctx["B"]
        M_C[m_of_entry, i_of_entry] = ctx["sigC"]

        G = T_c @ M_A + T_s @ M_B + T_co @ M_C
        return G, seg_view

    def _apply_near_correction(self, geom, k, ctx, contribs, projector, src_c, src_t):
        """Recompute the near-pair test integrals on the endpoint-graded rule,
        overwriting the uniform-rule values in `contribs` (M2).

        Each (entry, source-segment) cell is owned by exactly one near pair —
        the entry fixes the test segment — so the overwrite is an assignment,
        not an accumulation.

        Runs per source block (M4): the free-space block selects its near
        pairs against the real segments, an image block against the MIRRORED
        ones, so a wire touching the plane gets its segment↔own-image spike
        corrected too. The blocks are separate arrays, so the assignments
        never collide.
        """
        mm, nn = self._near_pairs(geom, src_c=src_c, src_t=src_t)
        if mm.size == 0:
            return

        seg_c = geom["seg_centers"]
        seg_t = geom["seg_tangents"]
        hh = ctx["hh"]
        a_seg = ctx["a_seg"]
        starts, counts = ctx["starts"], ctx["counts"]
        sigA, B, sigC = ctx["sigA"], ctx["B"], ctx["sigC"]

        # Feature width relative to the half-length, taken at the tightest
        # segment in the model so one shared template resolves them all.
        a_all = self._seg_radius(geom)
        xg, wg = _graded_endpoint_rule(
            float(np.min(a_all / hh)), self.n_qp_near, self._leggauss_cached
        )

        # Flatten (pair, support-entry-of-its-test-segment) to a run-length
        # layout: entries of pair p occupy cum[p]:cum[p+1].
        cnt = counts[mm]
        cum = np.concatenate(([0], np.cumsum(cnt)))
        pair_of = np.repeat(np.arange(mm.size), cnt)
        entry_of = (
            np.arange(cum[-1]) - np.repeat(cum[:-1], cnt) + np.repeat(starts[mm], cnt)
        )

        contrib_c, contrib_s, contrib_co = contribs
        for p0 in range(0, mm.size, _PAIR_BLOCK):
            p1 = min(p0 + _PAIR_BLOCK, mm.size)
            mi, ni = mm[p0:p1], nn[p0:p1]

            # (P, G, 3) observer points along each test segment.
            xi = hh[mi][:, None] * xg[None, :]  # (P, G)
            obs = seg_c[mi][:, None, :] + xi[:, :, None] * seg_t[mi][:, None, :]
            cm = self._field_components_bcast(
                k,
                obs_c=obs,
                obs_t=seg_t[mi][:, None, :],
                a=self._uniform_radius if a_seg is None else a_seg[mi][:, None],
                src_c=src_c[ni][:, None, :],
                src_t=src_t[ni][:, None, :],
                src_hh=hh[ni][:, None],
            )
            # (P, 1) pair indices broadcast against the (P, G) field tables.
            Phi = projector(cm, mi[:, None], ni[:, None])  # each (P, G)

            e0, e1 = cum[p0], cum[p1]
            ei = entry_of[e0:e1]  # into the flat seg_view arrays
            lp = pair_of[e0:e1] - p0  # into this block's pair axis
            xi_e = xi[lp]  # (E, G)
            fval = (
                sigA[ei][:, None]
                + B[ei][:, None] * np.sin(k * xi_e)
                + sigC[ei][:, None] * np.cos(k * xi_e)
            )
            w = (wg[None, :] * hh[mi][lp][:, None]) * fval  # (E, G)

            col = ni[lp]
            for contrib, Ph in zip((contrib_c, contrib_s, contrib_co), Phi):
                contrib[ei, col] = np.einsum("eg,eg->e", w, Ph[lp])

    # ------------------------------------------------------------------
    # Galerkin-tested source vector + solve
    # ------------------------------------------------------------------

    def _drive_columns(self, geom, seg_view, k):
        """Unit-voltage Galerkin excitation column per port, (N+P, n_ports),
        ordered [gap feeds…, junction ports…, node ports…].

        Gap feed j: b_i = -∫ f_i(s)·ŝ·E^app(s) ds with the delta-gap applied
        field E^app = 1/Δ_m along +ŝ_m on feed segment m and zero elsewhere,
        so only that segment's support entries contribute:

            ∫_{seg m} f_{i,m}(ξ) dξ = σA·Δ_m + σC·(2/k)·sin(kΔ_m/2)

        (the sin term integrates to zero by parity). Contrast the collocation
        RHS, which point-samples -1/Δ_m at the feed segment centre. This
        integral is exact, so M2's quadrature work does not touch it.

        Junction port p: the source is an EMF in the infinitesimal lead
        between the port terminal and the node, so testing it against basis
        w gives -1 × (w's net current THROUGH that lead) = -1 × (w's net
        inflow at the node). Every ordinary basis has zero net inflow there
        — that is #177's KCL identity, the very property that makes the port
        impossible in the point-matched solver — and `g_q` has δ_pq by
        construction. So the whole column is a single -1 at row N+p: the
        port's excitation touches exactly the port's own row, exactly.

        Node port p (M5b formulation (a)): the source is a ZERO-width delta
        gap sitting exactly at the junction node, E^app = V·δ(s − s_node)
        along the arc that runs from the port's + side to its − side. The
        Galerkin test integral collapses on the delta, so

            b_i = -∫ f_i(s)·ŝ·E^app(s) ds = -V·f_i(node)

        with f_i(node) the through-current `_node_cut_vectors` builds. No
        basis column, no node charge — the whole port is one RHS column plus
        its exact dual readout.
        """
        N = geom["n_segs"]
        h = np.asarray(geom["seg_h"], dtype=float)
        starts = seg_view["starts"]
        n_basis = N + len(self.junction_ports)
        U = np.zeros((n_basis, self.n_ports), dtype=np.complex128)
        for j, fseg in enumerate(geom["feed_segs"]):
            s, e = starts[fseg], starts[fseg + 1]
            hm = float(h[fseg])
            sig = seg_view["sigma"][s:e]
            int_f = sig * seg_view["A"][s:e] * hm + sig * seg_view["C"][s:e] * (
                2.0 / k
            ) * np.sin(0.5 * k * hm)
            np.add.at(U[:, j], seg_view["jbasis"][s:e], -int_f / hm)
        for p in range(len(self.junction_ports)):
            U[N + p, len(self.feeds) + p] = -1.0
        if self.node_ports:
            n0 = len(self.feeds) + len(self.junction_ports)
            U[:, n0:] = -self._node_cut_vectors(geom, seg_view, k)
        return U

    def _tested_source_vector(self, geom, seg_view, k):
        """Galerkin RHS for the configured port voltages: Σ_ports V·U[:, port].
        See `_drive_columns` for what each column is."""
        U = self._drive_columns(geom, seg_view, k)
        return U @ self._port_voltages()

    def _port_voltages(self):
        return np.array(
            [v for _, _, v in self.feeds]
            + [v for _j, v in self.junction_ports]
            + [v for _j, _s, v in self.node_ports],
            dtype=np.complex128,
        )

    def _port_currents(self, alpha, geom, seg_view, U):
        """Per-port current readout, ordered [gap feeds…, junction ports…,
        node ports…].

        A junction port's readout is `α_{N+p}` — its own basis amplitude,
        which by construction IS the current injected at the node — and that
        equals -U[:, port]·α exactly, so the port block of Y is symmetric to
        machine precision whichever branch below runs.

        A NODE port's readout is the current through the node, -U[:, port]·α,
        which is likewise the exact dual of its drive: the drive column IS the
        through-current functional, so there is no centre-vs-average choice to
        make here and no payoff to trade (contrast the gap feed below). Its
        block of Y is symmetric to machine precision in both branches.

        A gap feed's readout depends on `feed_readout`: the segment-CENTRE
        current (default, the point-matched solver's and NEC's) or the
        gap-averaged current -U[:, j]·α, which is the exact dual of the
        Galerkin drive. See the class docstring for why the non-dual one is
        the default and what it costs.
        """
        N = geom["n_segs"]
        if self.feed_readout == "variational":
            return -(U.T @ alpha)
        n0 = len(self.feeds) + len(self.junction_ports)
        return np.array(
            [
                self._feed_segment_current(alpha, seg_view, fi)
                for fi in geom["feed_segs"]
            ]
            + [alpha[N + p] for p in range(len(self.junction_ports))]
            + [-(U[:, n0 + p] @ alpha) for p in range(len(self.node_ports))],
            dtype=np.complex128,
        )

    def compute_impedance(self):
        """Return (Z_drive, alpha). Mirrors `SinusoidalSolver.compute_impedance`
        but assembles the Galerkin matrix and the Galerkin-tested RHS.

        With junction ports, `alpha` is length N+P and `Z_drive` covers
        [gap feeds…, junction ports…]; it stays a bare scalar only when the
        model has exactly one port of any kind.
        """
        self._refuse_junction_port_solve()
        geom = self._build_geometry()
        self._checkpoint()  # after geometry, before the field fill
        G, seg_view = self._assemble_Z(geom, self.k)
        U = self._drive_columns(geom, seg_view, self.k)
        voltages = self._port_voltages()
        self._checkpoint()  # after assembly, before the dense solve

        alpha = scipy.linalg.solve(G, U @ voltages)

        currents = self._port_currents(alpha, geom, seg_view, U)
        z_per_port = voltages / currents
        Z_drive = z_per_port[0] if self.n_ports == 1 else z_per_port
        return Z_drive, alpha

    def compute_y_matrix(self) -> np.ndarray:
        """Short-circuit admittance matrix over [gap feeds…, junction ports…].

        Honestly Galerkin throughout, which the inherited implementation was
        not: it paired this solver's Galerkin matrix with the point-matched
        solver's RHS (a bare -1/h at the feed segment's row), so its Y did
        not even agree with `compute_impedance`. Here the columns are the
        same `_drive_columns` the impedance solve uses and the readout is the
        same `_port_currents`, so 1/Y⁻¹ reproduces `compute_impedance` to
        solver precision by construction.
        """
        self._refuse_junction_port_solve()
        geom = self._build_geometry()
        G, seg_view = self._assemble_Z(geom, self.k)
        U = self._drive_columns(geom, seg_view, self.k)
        alphas = scipy.linalg.solve(G, U)
        return np.stack(
            [
                self._port_currents(alphas[:, j], geom, seg_view, U)
                for j in range(self.n_ports)
            ],
            axis=1,
        )

    def compute_y_matrix_swept(self, k_array) -> np.ndarray:
        """Per-frequency Y matrices; loops over k like the inherited version
        but with the Galerkin drive/readout of `compute_y_matrix`. The drive
        columns are k-DEPENDENT here (the gap column integrates the basis
        shapes, which move with k), so they are rebuilt per step."""
        self._refuse_junction_port_solve()
        k_array = np.asarray(k_array, dtype=float)
        geom = self._build_geometry()
        out = np.zeros(
            (k_array.shape[0], self.n_ports, self.n_ports), dtype=np.complex128
        )
        saved = (self.k, self.wavelength, self.omega)
        try:
            for ki, kk in enumerate(k_array):
                self._checkpoint()
                self._set_k(float(kk))
                G, seg_view = self._assemble_Z(geom, self.k)
                U = self._drive_columns(geom, seg_view, self.k)
                alphas = scipy.linalg.solve(G, U)
                out[ki] = np.stack(
                    [
                        self._port_currents(alphas[:, j], geom, seg_view, U)
                        for j in range(self.n_ports)
                    ],
                    axis=1,
                )
        finally:
            self.k, self.wavelength, self.omega = saved
        return out

    def compute_impedance_swept(self, k_array):
        """Driving-point impedance over a batch of wavenumbers — a per-k loop
        of exactly `compute_impedance`'s algebra (the inherited version used
        the collocation RHS, so it disagreed with `compute_impedance`)."""
        self._refuse_junction_port_solve()
        k_array = np.asarray(k_array, dtype=float)
        geom = self._build_geometry()
        voltages = self._port_voltages()
        n_p = self.n_ports
        z_out = np.zeros(
            k_array.shape[0] if n_p == 1 else (k_array.shape[0], n_p),
            dtype=np.complex128,
        )
        saved = (self.k, self.wavelength, self.omega)
        try:
            for i, kk in enumerate(k_array):
                self._checkpoint()
                self._set_k(float(kk))
                G, seg_view = self._assemble_Z(geom, self.k)
                U = self._drive_columns(geom, seg_view, self.k)
                alpha = scipy.linalg.solve(G, U @ voltages)
                z_per = voltages / self._port_currents(alpha, geom, seg_view, U)
                z_out[i] = z_per[0] if n_p == 1 else z_per
        finally:
            self.k, self.wavelength, self.omega = saved
        return z_out

    def _set_k(self, kk):
        """Rebind the frequency triple the sweeps mutate in lockstep."""
        self.k = float(kk)
        self.omega = self.k * self.c
        self.wavelength = self.c / (self.omega / (2 * np.pi))
