"""The crossing fill takes a `CrossingContext`, not a solver (momwire#801).

`_crossing_fill` was written against `BSplineSolver` by address. The G1-3
probe (momwire#651, 2026-09-02) read every consumer and found that the knot
structure never reaches the fill: the ONE bspline-shaped read is
`axis_data`'s view of the basis as per-segment polynomials, and everything
after the axis dict — the tables, the five-term sandwich, the by-parts ends,
the corner, the #688 split, the self completions — consumes the dict and
physical scalars. #801 makes that structural. Two gates:

  * **the module has forgotten the solver.** No attribute of `BSplineSolver`
    is named in the module's source; the context is the whole interface.
    (The numerical gate — bit-identical `axis_data`, `t_ab` and
    `self_completions` on `crossing_deck(level=1)` and the two-radial fan,
    92/92 arrays `array_equal` across the change — is the #762 protocol,
    run by hand for the PR; a test cannot compare against a commit.)

  * **a tent basis goes through `axis_data`.** A degree-1 basis written by
    hand in razor's spelling — `c0 + c1·u` per support segment, value 1 at
    the shared knot, a value-1 tent at the in-plane wire end — samples to the
    analytic tents and lands its in-plane end in the `ends` table with the
    derivation's sign. This is the corroboration of the G1-3 answer, and the
    starting point of any razor test-side probe.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np

from momwire import _crossing_fill as CF
from momwire import bspline as _bspline


def test_the_module_names_no_solver_attribute():
    """Code, not prose: every attribute access in the module is collected
    from the AST, and none of them is a `BSplineSolver` method or private
    field. Docstrings may (and do) name the history."""
    tree = ast.parse(inspect.getsource(CF))
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for name in (
        "_build_basis_polynomials",
        "_radius_per_wire",
        "_buried_medium",
        "wires_polylines",
        "junctions",
    ):
        assert name not in attrs, name
    imported = {
        alias.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module in (None, "bspline")
        for alias in n.names
    }
    assert "bspline" not in imported and "BSplineSolver" not in imported


def test_bspline_builds_the_context_it_used_to_be():
    """The one caller hands the fill the same six numbers `_buried_medium`
    returned and the five geometry columns, through the record."""
    assert _bspline.BSplineSolver._crossing_context is not None
    m = CF.buried_medium((13.0, 0.005), 2 * np.pi * 7.0e6, 8.8541878128e-12, 0.1467)
    assert isinstance(m, CF.Medium)
    assert m.k_m.imag <= 0.0
    assert m.a_m == -m.c2 or np.isclose(m.a_m, -m.c2)


def _tent_context(h=0.4, gz=0.0):
    """A vertical two-segment wire standing on the plane, with a degree-1
    tent basis in razor's spelling: basis 0 peaks at the shared knot, basis 1
    is the value-1 tent at the in-plane end (the grounded-end / junction
    tent). `supp_seg` rows are zero-padded and the padding slot's polynomial
    is all-zero, which is the trap `axis_data` guards."""
    seg_l = np.array([[0.0, 0.0, gz], [0.0, 0.0, gz + h]])
    seg_r = np.array([[0.0, 0.0, gz + h], [0.0, 0.0, gz + 2 * h]])
    geom = CF.AxisGeometry(
        seg_l=seg_l,
        seg_r=seg_r,
        h=np.array([h, h]),
        tangents=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        seg_offsets=np.array([0, 2]),
    )
    supp_seg = np.array([[0, 1], [0, 0]])
    polys = np.array(
        [
            [[0.0, 1.0 / h], [1.0, -1.0 / h]],  # u/h on seg 0, 1 - u/h on seg 1
            [[1.0, -1.0 / h], [0.0, 0.0]],  # 1 - u/h on seg 0; padding slot
        ]
    )
    basis = CF.BasisPolynomials(supp_seg=supp_seg, polys=polys, degree=1)
    medium = CF.buried_medium(
        (13.0, 0.005), 2 * np.pi * 7.0e6, 8.8541878128e-12, 0.1467
    )
    return CF.CrossingContext(
        basis=basis,
        geom=geom,
        medium=medium,
        ground_z=gz,
        a_wire=1e-3,
        omega=2 * np.pi * 7.0e6,
        mu=4e-7 * np.pi,
        eps=8.8541878128e-12,
    )


def test_a_tent_basis_samples_to_the_analytic_tents():
    ctx = _tent_context()
    ax = CF.axis_data(ctx, [0, 1])
    h = ctx.geom.h[0]
    assert ax["n_basis"] == 2
    u = ax["nodes"][:, 2] - ctx.geom.seg_l[ax["segof"], 2]  # local arc per node
    on0, on1 = ax["segof"] == 0, ax["segof"] == 1
    assert on0.any() and on1.any()
    # basis 0: the knot tent, rising on seg 0 and falling on seg 1
    assert np.allclose(ax["F"][0, on0], u[on0] / h)
    assert np.allclose(ax["F"][0, on1], 1.0 - u[on1] / h)
    assert np.allclose(ax["Fd"][0, on0], 1.0 / h)
    assert np.allclose(ax["Fd"][0, on1], -1.0 / h)
    # basis 1: the in-plane end tent, falling on seg 0 and absent on seg 1
    assert np.allclose(ax["F"][1, on0], 1.0 - u[on0] / h)
    assert np.all(ax["F"][1, on1] == 0.0) and np.all(ax["Fd"][1, on1] == 0.0)
    # the in-plane segment is graded toward the plane; the other is plain
    # Gauss at the fill's density
    assert on0.sum() > on1.sum() == CF._NEAR_Q
    # weights integrate the tents exactly (linear on each segment)
    assert np.isclose((ax["F"][0] * ax["w"]).sum(), h)  # ∫ tent = h
    assert np.isclose((ax["F"][1] * ax["w"]).sum(), h / 2)


def test_the_in_plane_end_lands_in_the_ends_table_with_the_derivations_sign():
    ctx = _tent_context()
    ax = CF.axis_data(ctx, [0, 1])
    # One end only: the free end at 2h carries no basis value and drops out.
    assert len(ax["ends"]) == 1
    pt, sign, fv = ax["ends"][0]
    assert np.allclose(pt, [0.0, 0.0, 0.0])
    assert sign == -1.0  # σ = −1 at a wire's first segment's u = 0 end
    assert np.allclose(fv, [0.0, 1.0])  # only the end tent stands there
