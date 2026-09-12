"""momwire#980 D3: `SinusoidalGalerkinSolver` serves a deck whose junction
CROSSES the interface.

What D3 adds over D2 is a node, and the node is the whole difficulty. The
segment bases end there as FREE ENDS and the node carries its own dofs — one
`_crossing_wing_view` wing per member wire-end — because with one dof per
segment a segment's own basis cannot carry a free value AND a free slope at
the node, and all three end conditions it *can* carry are wrong: I = 0 (a free
end), dI/ds = 0 (the contact/image shape, which the C1-kink lesson rules out
since the charge must be free to jump), and a neighbour extension (there is no
neighbour in this medium). That count — K wings, no KCL row — is bspline's:
`_build_basis_polynomials` keeps every member's value-1 directional basis and
drops only the KCL row at a grounded junction. The KCL-closed through-tent is
one combination inside that span rather than a convention chosen in the basis,
which is why `test_the_node_kcl_emerges_from_the_fill` is a measurement and
not a tautology.

Two of the gates below exist because they caught something the answer could
not see, and both are recorded here so a later edit cannot quietly undo them:

* `test_the_cross_block_sign_is_fixed_by_kcl_not_by_the_answer` — both signs
  of the cross block give the same Z to seven digits. Only the node's KCL
  separates them.
* `test_the_below_axis_is_sampled_at_k_m` — the crossing fill samples both
  axes through ONE basis object, so a below entry built at k_m was being
  evaluated with sin(k_p·ξ). Invisible at ε̃ = 1 (k_m == k_p), 170 % wrong on
  soil A.
"""

import warnings

import numpy as np
import pytest

from momwire import BSplineSolver, SinusoidalGalerkinSolver, _crossing_fill
from momwire.sinusoidal_galerkin import SinusoidalBasisSampler

C0 = 299792458.0
WL7 = C0 / 7e6
SOIL_A = (13.0, 0.005)
EPS_ONE = (1.0, 0.0)

# One above wire rising from the interface, one below wire leaving it, joined
# at a node IN the plane. The smallest deck that is a crossing deck at all.
_WIRES = [
    np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)]),
    np.array([(0.0, 0.0, 0.0), (2.0, 0.0, -0.5)]),
]
_JUNCTIONS = [[(0, "start"), (1, "start")]]


def crossing_deck(ground_eps, n=15):
    return SinusoidalGalerkinSolver(
        wires=_WIRES,
        n_per_edge_per_wire=[[n], [n]],
        feeds=[(0, 1.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        junctions=_JUNCTIONS,
        ground_z=0.0,
        ground_eps=ground_eps,
        ground_model="sommerfeld",
    )


def free_deck(n=15):
    """The same deck with no ground: an ORDINARY junction, the collapse target."""
    return SinusoidalGalerkinSolver(
        wires=_WIRES,
        n_per_edge_per_wire=[[n], [n]],
        feeds=[(0, 1.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        junctions=_JUNCTIONS,
    )


def _mixed_view(s):
    geom = s._build_geometry()
    below = s._below_segments(geom)
    medium = s._fill_medium(geom)
    seg_view = s._stitch_basis_coefs(geom, below, medium.k_p, medium.k_m)
    seg_view = s._crossing_wing_view(geom, seg_view, below, medium)
    return geom, below, medium, seg_view


def _sampler(s, geom, seg_view, medium):
    n_basis = int(geom["n_segs"]) + s._n_extra_cols()
    return SinusoidalBasisSampler(seg_view, medium.k_p, geom["seg_h"], n_basis)


def _node_currents(s, alpha, geom, seg_view, medium):
    """(I, inflow) at the crossing node along each member, from the solution."""
    samp = _sampler(s, geom, seg_view, medium)
    out = []
    for m, sgn in s._junction_members(geom, 0):
        h = float(geom["seg_h"][m])
        fv = samp.end_values(m, h if sgn > 0 else 0.0)
        current = complex(np.dot(alpha[: fv.size], fv))
        out.append((current, sgn * current))
    return out


@pytest.fixture(scope="module")
def eps_one():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = crossing_deck(EPS_ONE)
        z, alpha = s.compute_impedance()
        geom, below, medium, seg_view = _mixed_view(s)
        return {
            "s": s,
            "z": complex(z),
            "alpha": alpha,
            "geom": geom,
            "below": below,
            "medium": medium,
            "seg_view": seg_view,
        }


# ----------------------------------------------------------------------
# The basis
# ----------------------------------------------------------------------


def test_a_crossing_deck_gets_one_node_wing_per_member(eps_one):
    s, geom = eps_one["s"], eps_one["geom"]
    assert sorted(s._crossing_junction_indices()) == [0]
    assert s._n_crossing_wings() == 2
    assert s._n_extra_cols() == 2
    # The wings are basis columns, not a constraint block: G grows with them.
    assert eps_one["alpha"].shape == (int(geom["n_segs"]) + 2,)


def test_the_node_wing_is_unit_inflow_at_the_node_and_zero_at_the_far_end(eps_one):
    s, geom, medium = eps_one["s"], eps_one["geom"], eps_one["medium"]
    samp = _sampler(s, geom, eps_one["seg_view"], medium)
    N = int(geom["n_segs"])
    for w, (m, sgn) in enumerate(s._junction_members(geom, 0)):
        h = float(geom["seg_h"][m])
        near = samp.end_values(m, h if sgn > 0 else 0.0)[N + w]
        far = samp.end_values(m, 0.0 if sgn > 0 else h)[N + w]
        # σ·f(node) = 1 is the normalisation; f(far) = 0 is the shape.
        assert sgn * near == pytest.approx(1.0, abs=1e-12)
        assert abs(far) < 1e-12
        # ONE segment, so ONE medium: no basis spans the interface, which is
        # what keeps D2's per-entry stitch honest.
        starts = np.asarray(eps_one["seg_view"]["starts"])
        jb = np.asarray(eps_one["seg_view"]["jbasis"])
        seg_of_entry = np.repeat(np.arange(N), np.diff(starts))
        support = np.unique(seg_of_entry[jb == N + w])
        assert support.tolist() == [m]


def test_the_segment_bases_end_at_the_node_as_free_ends(eps_one):
    """(b), and it is now a statement about the SEGMENT bases alone."""
    s, geom, medium = eps_one["s"], eps_one["geom"], eps_one["medium"]
    samp = _sampler(s, geom, eps_one["seg_view"], medium)
    N = int(geom["n_segs"])
    for m, sgn in s._junction_members(geom, 0):
        h = float(geom["seg_h"][m])
        fv = samp.end_values(m, h if sgn > 0 else 0.0)
        assert np.abs(fv[:N]).max() < 1e-15


def test_shipped_paths_carry_no_wings(eps_one):
    """Every deck this family serves today answers zero, structurally."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = free_deck()
        assert s._crossing_junction_indices() == frozenset()
        assert s._n_crossing_wings() == 0
        assert s._n_extra_cols() == 0
        assert "k_entry" not in s._basis_coefs(s._build_geometry(), s.k)


# ----------------------------------------------------------------------
# The gates
# ----------------------------------------------------------------------


def test_the_eps_tilde_one_collapse(eps_one):
    """At ε̃ → 1 the interface is not there and the crossing solve must
    reproduce the same deck's free-space ORDINARY junction.

    1.4e-06, and it is a FLOOR rather than a converging sequence: 1.385e-06 /
    1.663e-06 / 1.472e-06 at 15 / 31 / 61 segments per wire. The floor is D2's,
    not the node's — the same collapse on a crossing-FREE mixed deck whose
    wires merely approach the plane reads 2.3e-10 at a 0.5 m gap, 5.8e-09 at
    0.1 m, 1.6e-07 at 0.02 m and 1.9e-07 at 0.005 m, so a deck whose wires MEET
    in the plane landing at 1.4e-06 is that sequence continued to zero gap.

    And it is not this family's floor alone. `BSplineSolver`'s OWN collapse on
    the same deck and the same rungs reads 1.47e-07 / 2.73e-06 / 1.94e-06 —
    the same order as ours for n >= 31, and LARGER than ours at those two
    rungs. (The 1.8e-07 this comment used to cite for bspline was its n = 15
    reading taken as if it were a floor; it is not one.) Both families sitting
    at ~1e-06 once the mesh is refined is what makes the near-plane
    transmitted block, which they share, the suspect rather than either basis.

    The bar is set to catch a formulation change, not to pin the floor: the
    spelling this replaced read 4.45e-01 with no node dof at all, and
    4.58e-01 with the node dofs present but the block signs uncorrected.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z_free = complex(free_deck().compute_impedance()[0])
    rel = abs(eps_one["z"] - z_free) / abs(z_free)
    assert rel < 5e-06, rel


def test_the_node_kcl_emerges_from_the_fill(eps_one):
    """No constraint row and no merged dof: the K wings are independent, so
    KCL at the node is something the fill has to produce."""
    pairs = _node_currents(
        eps_one["s"],
        eps_one["alpha"],
        eps_one["geom"],
        eps_one["seg_view"],
        eps_one["medium"],
    )
    scale = max(abs(i) for i, _ in pairs)
    residual = abs(sum(inflow for _, inflow in pairs))
    assert scale > 1e-5, scale
    assert residual / scale < 1e-3, (residual, scale)


def test_the_node_carries_the_free_space_current(eps_one):
    """Not merely KCL-clean — the right amount of current, both sides."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sf = free_deck()
        _z, alpha_f = sf.compute_impedance()
        gf = sf._build_geometry()
        samp = SinusoidalBasisSampler(
            sf._basis_coefs(gf, sf.k), sf.k, gf["seg_h"], int(gf["n_segs"])
        )
        m, sgn = sf._junction_members(gf, 0)[0]
        h = float(gf["seg_h"][m])
        fv = samp.end_values(m, h if sgn > 0 else 0.0)
        i_free = abs(complex(np.dot(alpha_f, fv)))
    pairs = _node_currents(
        eps_one["s"],
        eps_one["alpha"],
        eps_one["geom"],
        eps_one["seg_view"],
        eps_one["medium"],
    )
    for current, _inflow in pairs:
        assert abs(current) == pytest.approx(i_free, rel=1e-3)


def test_the_cross_block_sign_is_fixed_by_kcl_not_by_the_answer():
    """The finding this gate exists to hold: flipping the cross block's sign
    moves Z by less than 1e-7 relative and the node's KCL residual by four
    orders of magnitude. An answer-only gate would have shipped either sign.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = crossing_deck(EPS_ONE)
        geom, below, medium, seg_view = _mixed_view(s)
        ctx = s._stitch_test_context(geom, seg_view, below, medium.k_p, medium.k_m)
        plan = s._mixed_serve_plan(geom, below, medium, ctx, True)
        G0 = s._scatter_coef_product(
            ctx, s._assemble_mixed_contribs(geom, ctx, below, medium, plan, True)
        )
        ctx_x = s._crossing_context(geom, seg_view, medium)
        a_idx = np.nonzero(~np.asarray(below))[0]
        b_idx = np.nonzero(np.asarray(below))[0]
        t_ab = _crossing_fill.cross_complete_block_split(
            ctx_x,
            a_idx,
            b_idx,
            _crossing_fill.axis_data(ctx_x, a_idx),
            _crossing_fill.axis_data(ctx_x, b_idx),
        )
        U = s._drive_columns(geom, seg_view, s.k)
        rhs = U @ s._port_voltages()

        out = {}
        for sign in (+1.0, -1.0):
            alpha = np.linalg.solve(G0 + sign * (t_ab + t_ab.T), rhs)
            pairs = _node_currents(s, alpha, geom, seg_view, medium)
            out[sign] = (
                abs(sum(inflow for _, inflow in pairs)) / max(abs(i) for i, _ in pairs),
                complex(s._port_currents(alpha, geom, seg_view, U)[0]),
            )
    kcl_plus, i_plus = out[+1.0]
    kcl_minus, i_minus = out[-1.0]
    # The answer cannot tell them apart...
    assert abs(i_plus - i_minus) / abs(i_plus) < 1e-6
    # ...and the node can, by four orders of magnitude.
    assert kcl_plus < 1e-3
    assert kcl_minus > 1e-1
    assert kcl_minus / kcl_plus > 1e3


def test_the_node_block_tracks_the_arbiters(eps_one):
    """The node's 2x2 block against `BSplineSolver`'s on the same deck.

    Recorded as RATIOS because the two families normalise both the basis and
    the operator differently — this trunk's G carries the opposite overall
    sign from bspline's Z, which is exactly why `self_completions` (written in
    bspline's convention) cancelled this fill's own boundary content instead of
    completing it. What must agree is the SHAPE of the block: off-diagonal
    within a few percent of the diagonal, so that the net node charge is
    expensive and the through current is not.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = eps_one["s"]
        geom = eps_one["geom"]
        with s._operating_medium(geom):
            G, _sv = s._assemble_Z_ported(geom, s.k)
        N = int(geom["n_segs"])
        ours = np.array([[G[N, N], G[N, N + 1]], [G[N + 1, N], G[N + 1, N + 1]]])

        bs = BSplineSolver(
            wires=_WIRES,
            n_per_edge_per_wire=[[15], [15]],
            feeds=[(0, 1.0, 1 + 0j)],
            wavelength=WL7,
            wire_radius=0.001,
            degree=1,
            junctions=_JUNCTIONS,
            ground_z=0.0,
            ground_eps=EPS_ONE,
            ground_model="sommerfeld",
        )
        gb = bs._build_geometry()
        supp_seg, polys, _kcl, _wk, wbg = bs._build_basis_polynomials(gb)
        Z = bs._compute_Z_operator_buried(gb, supp_seg, polys)
        cols = [
            l2g[ki]
            for _kept, l2g in wbg
            for ki, (_j, kind, _ji, _ep) in enumerate(_kept)
            if kind in ("dir", "gnd")
        ]
        theirs = Z[np.ix_(cols, cols)]

    for M, tag in ((ours, "sg"), (theirs, "bspline")):
        offd = abs(M[0, 1]) / abs(M[0, 0])
        assert 0.85 < offd < 1.05, (tag, offd)
    # Opposite overall sign between the families, on a term whose sign is not
    # ambiguous: the node's charge self-energy.
    assert np.sign(ours[0, 0].imag) == -np.sign(theirs[0, 0].imag)


def test_the_free_end_choice_no_longer_moves_the_answer():
    """With the node dof present, giving the crossing ends the grounded
    junction's self-image atom instead moves Z by ~1e-14. The (a)/(b) question
    that the C0 formulation turned on is answered by the node dof, not by the
    end condition."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z_b = complex(crossing_deck(EPS_ONE).compute_impedance()[0])
        s = crossing_deck(EPS_ONE)
        geom = s._build_geometry()
        geom["ground_minus"] = geom["ground_minus"] | geom["crossing_minus"]
        geom["ground_plus"] = geom["ground_plus"] | geom["crossing_plus"]
        geom["crossing_minus"] = np.zeros_like(geom["crossing_minus"])
        geom["crossing_plus"] = np.zeros_like(geom["crossing_plus"])
        z_a = complex(s.compute_impedance()[0])
    assert abs(z_a - z_b) / abs(z_b) < 1e-12


# ----------------------------------------------------------------------
# Per-axis k
# ----------------------------------------------------------------------


def test_the_below_axis_is_sampled_at_k_m():
    """`axis_data` samples BOTH axes through ONE basis object, so a below
    entry built at k_m was evaluated with sin(k_p·ξ) — three different
    functions where `ends`, `F` and `Fd` have to describe one. Carrying k
    per ENTRY, where the coefficients are already carried, is the fix.

    Invisible at ε̃ = 1 by construction (k_m == k_p there), which is why it
    survived gate 1; on soil A |k_m/k_p| = 4.27 and it is 170 % of F.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = crossing_deck(SOIL_A)
        geom, below, medium, seg_view = _mixed_view(s)
        assert "k_entry" in seg_view
        entry_below = np.asarray(seg_view["k_entry"]) == medium.k_m
        assert entry_below.any() and not entry_below.all()

        shared = {k: v for k, v in seg_view.items() if k != "k_entry"}
        ctx_fixed = s._crossing_context(geom, seg_view, medium)
        ctx_shared = s._crossing_context(geom, shared, medium)
        a_idx = np.nonzero(~np.asarray(below))[0]
        b_idx = np.nonzero(np.asarray(below))[0]

        # The ABOVE axis is bit-identical: k_p was always right there.
        fa = _crossing_fill.axis_data(ctx_fixed, a_idx)
        sa = _crossing_fill.axis_data(ctx_shared, a_idx)
        assert np.array_equal(fa["F"], sa["F"])
        assert np.array_equal(fa["Fd"], sa["Fd"])

        # The BELOW axis moves, by far more than any quadrature noise.
        fb = _crossing_fill.axis_data(ctx_fixed, b_idx)
        sb = _crossing_fill.axis_data(ctx_shared, b_idx)
        rel_f = np.abs(fb["F"] - sb["F"]).max() / np.abs(sb["F"]).max()
        rel_fd = np.abs(fb["Fd"] - sb["Fd"]).max() / np.abs(sb["Fd"]).max()
    assert rel_f > 1.0, rel_f
    assert rel_fd > 1.0, rel_fd


def test_per_entry_k_is_absent_and_inert_on_a_single_medium_view():
    """The scalar-k sampler is the shipped one and must stay untouched."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = free_deck()
        geom = s._build_geometry()
        view = s._basis_coefs(geom, s.k)
        samp = SinusoidalBasisSampler(view, s.k, geom["seg_h"], int(geom["n_segs"]))
        assert samp._k_per_entry is False
        assert samp.end_values(0, 0.0).shape == (int(geom["n_segs"]),)


# ----------------------------------------------------------------------
# The screen: what D3 exists to serve
# ----------------------------------------------------------------------

DEPTH = 0.5
MAST = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 10.0)])


def _dirs(n):
    return [(np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n)) for i in range(n)]


def direct_fan(n_radials):
    """N buried radials running straight from their far end to the node."""
    wires = [
        np.array([(5.0 * dx, 5.0 * dy, -DEPTH), (0.0, 0.0, 0.0)])
        for dx, dy in _dirs(n_radials)
    ]
    npe = [[12] for _ in wires]
    mono = len(wires)
    wires.append(MAST)
    npe.append([15])
    return dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=[[(i, "end") for i in range(n_radials)] + [(mono, "start")]],
        feeds=[(mono, 0.25, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )


def buried_hub(n_radials):
    """ONE rise to the node, N radials joined to it AT DEPTH — the screen's
    other spelling, named in `_below_interface.crossing_junctions`' scope."""
    wires = [np.array([(0.0, 0.0, -DEPTH), (0.0, 0.0, 0.0)])]
    npe = [[2]]
    for dx, dy in _dirs(n_radials):
        wires.append(np.array([(5.0 * dx, 5.0 * dy, -DEPTH), (0.0, 0.0, -DEPTH)]))
        npe.append([10])
    mono = len(wires)
    wires.append(MAST)
    npe.append([15])
    return dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=[
            [(0, "end"), (mono, "start")],
            [(0, "start")] + [(i + 1, "end") for i in range(n_radials)],
        ],
        feeds=[(mono, 0.25, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )


def _both(kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zs = complex(SinusoidalGalerkinSolver(**kw).compute_impedance()[0])
        zb = complex(BSplineSolver(**kw, degree=1).compute_impedance()[0])
    return zs, zb


@pytest.mark.parametrize("n_radials", [1, 2, 4])
def test_the_direct_fan_agrees_with_the_arbiter(n_radials):
    """One above wire over N below wires — `crossing_junctions`' fan widening,
    and the shape D3 was built for.

    Before D3 this deck did not fail; it answered. `SinusoidalGalerkinSolver`
    on momwire main returns 13284.8 − 15216.6j at N = 1 and 13258.3 − 15204.6j
    at N = 4 — an answer that does not move with the size of the screen,
    because the node carried no basis, the mast was open at its base and the
    screen was simply absent. bspline reads 150.4 and 57.0.
    """
    zs, zb = _both(direct_fan(n_radials))
    assert abs(zs - zb) / abs(zb) < 1e-2, (n_radials, zs, zb)


# 5.2 s at 4 radials, 4.6 s at 12: two Sommerfeld solves apiece, and the
# hub deck carries N+2 wires.
@pytest.mark.slow
@pytest.mark.parametrize("n_radials", [4, 12])
def test_the_buried_hub_agrees_with_the_arbiter(n_radials):
    """The screen's hub spelling, up to the 12-radial screen of #983."""
    zs, zb = _both(buried_hub(n_radials))
    assert abs(zs - zb) / abs(zb) < 1e-2, (n_radials, zs, zb)


def test_coincident_members_are_refused_by_name():
    """#926's `crossing_deck` writes each radial as far-end -> hub -> RISE, so
    every radial carries the SAME rise. This family cannot solve that: the
    thin-wire kernel regularizes a zero-separation pair to the wire radius —
    what it does for a segment against itself — so coincident members give
    near-identical rows and the matrix is ill-conditioned.

    It is a property of the family and not of the crossing serve: in FREE
    SPACE, no ground and no junction, N coincident wires each carrying its own
    distinct arm give max|alpha| 2.3 / 41 / 184 at N = 1 / 2 / 4 against
    bspline's 2.0e-3 / 4.0e-3 / 9.0e-3. Over the ground the same deck reaches
    8.4e32. Refused here because a crossing junction is where that spelling is
    natural, and refused with the two respellings that work.
    """
    wires = [
        np.array([(5.0 * dx, 5.0 * dy, -DEPTH), (0.0, 0.0, -DEPTH), (0.0, 0.0, 0.0)])
        for dx, dy in _dirs(4)
    ]
    npe = [[10, 2] for _ in wires]
    mono = len(wires)
    wires.append(MAST)
    npe.append([15])
    kw = dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=[[(i, "end") for i in range(4)] + [(mono, "start")]],
        feeds=[(mono, 0.25, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NotImplementedError) as exc:
            SinusoidalGalerkinSolver(**kw).compute_impedance()
        # bspline serves it, which is why the refusal is this family's and not
        # `_below_interface.crossing_junctions`'.
        BSplineSolver(**kw, degree=1).compute_impedance()
    msg = str(exc.value)
    assert "COINCIDENT" in msg
    assert "buried-hub spelling" in msg


# 44.6 s, and all of it is the DETACHED deck: two wires 1 cm off the plane
# put the transmitted grid at its finest, which is the same near-plane
# cost the collapse floor above is made of. The crossing solve is 0.8 s.
@pytest.mark.slow
def test_connected_is_not_detached():
    """The gate the D2 bug would have failed: a crossing junction must be
    doing something. Read on RESISTANCE — the reactance of this deck is set by
    the mast and barely moves, which is exactly how an absent screen hides."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        connected = complex(crossing_deck(SOIL_A).compute_impedance()[0])
        # A contact end above a buried wire is itself refused (#151/#524), so
        # the legal detached comparison lifts both ends clear of the plane.
        detached = complex(
            SinusoidalGalerkinSolver(
                wires=[
                    np.array([(0.0, 0.0, 0.01), (0.0, 0.0, 2.0)]),
                    np.array([(0.0, 0.0, -0.01), (2.0, 0.0, -0.5)]),
                ],
                n_per_edge_per_wire=[[15], [15]],
                feeds=[(0, 1.0, 1 + 0j)],
                wavelength=WL7,
                wire_radius=0.001,
                ground_z=0.0,
                ground_eps=SOIL_A,
                ground_model="sommerfeld",
            ).compute_impedance()[0]
        )
    # 58.04 until momwire#956 (the crossing fill's ẑẑ kernel is k²V + ∂z′W
    # with the TW end term, which moves every crossing deck's R up by the
    # W-term excess: +3.1 Ω here, the same class as the #524 / #674 anchors
    # re-pinned in #1043). The detached deck has no crossing and does not move.
    assert connected.real == pytest.approx(61.12, rel=0.05)
    assert detached.real == pytest.approx(10.46, rel=0.05)
    assert connected.real / detached.real > 3.0


# ----------------------------------------------------------------------
# The two pieces D3 was supposed to bring into use
# ----------------------------------------------------------------------


def test_d3_exercises_the_crossing_trunk_and_the_complex_k_twin():
    """A D3 solve must actually route through both things #980 built for it:
    `_crossing_fill`'s designed DIRECT evaluation of the cross pair (step C's
    trunk) and the complex-k C++ far fill (step E's twin).

    Counted by CALL, not inferred from a flag. The #977 lesson is that a flag
    saying a path is available says nothing about whether it ran — two rounds
    of instrumentation there recorded zero calls because the deck never took
    the path at all.
    """
    import momwire.sinusoidal_galerkin as sgm

    calls = {"trunk": 0, "twin": 0}
    real_trunk = _crossing_fill.cross_complete_block_split
    real_acc = sgm._acc

    class _CountingAcc:
        def __getattr__(self, name):
            attr = getattr(real_acc, name)
            if name != "sinusoidal_galerkin_far_fill_cplx":
                return attr

            def counted(*a, **kw):
                calls["twin"] += 1
                return attr(*a, **kw)

            return counted

    def counting_trunk(*a, **kw):
        calls["trunk"] += 1
        return real_trunk(*a, **kw)

    assert sgm._HAVE_GALERKIN_FAR_FILL_CPLX, "the complex-k twin is not built"
    _crossing_fill.cross_complete_block_split = counting_trunk
    sgm._acc = _CountingAcc()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Soil A, so k_m is complex and the below class needs the twin.
            crossing_deck(SOIL_A).compute_impedance()
    finally:
        _crossing_fill.cross_complete_block_split = real_trunk
        sgm._acc = real_acc
    assert calls["trunk"] > 0, calls
    assert calls["twin"] > 0, calls
