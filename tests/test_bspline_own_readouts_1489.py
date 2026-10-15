"""momwire#1489: bs2's B-spline evaluation without scipy.interpolate.

`import momwire` used to load scipy.interpolate (and, through it,
scipy.optimize) for `BSplineSolver`'s basis build, source vector and two
current readouts. They now evaluate momwire's own code. scipy's `BSpline`
stays the REFERENCE here, imported inside each test only, so the gates
below compare against the implementation the solver used before.

Two contracts, deliberately different:

* `_bspline_basis_rows` is scipy's design matrix reproduced to the BIT. The
  basis polynomials every bs2 fill is built from come out of it, so any
  roundoff difference there would move every impedance; it does not.
* The readouts (`currents_at_knots`, `current_slopes`) evaluate the per-span
  basis polynomials instead of scipy's de Boor recurrence, so they agree to
  a measured tolerance rather than to the bit (the eval-order policy).
"""

import platform
from pathlib import Path

import numpy as np
import pytest

from momwire.bspline import (
    BSplineSolver,
    _bspline_basis_rows,
    _bspline_design_matrix_dense,
)


_EPS = np.finfo(float).eps


def _clamped_knots(rng, k, n_seg, uniform):
    if uniform:
        h = np.full(n_seg, rng.uniform(0.01, 30.0) / n_seg)
    else:
        h = rng.uniform(0.001, 2.0, n_seg)
    arc = np.concatenate([[0.0], np.cumsum(h)])
    return arc, h, np.concatenate([np.zeros(k), arc, np.full(k, arc[-1])])


# scipy's compiled de Boor is fused into FMAs where the compiler may
# contract a*b + c (clang and gcc on arm64; x86-64 wheels target a baseline
# without FMA), and momwire's numpy evaluation never fuses -- the stance its
# own C++ takes with -ffp-contract=off. So the two are equal to the bit on
# x86-64 and only to a few ulps elsewhere; the BIT gate is where it can hold.
_SCIPY_UNFUSED = platform.machine().lower() in ("x86_64", "amd64")

# Worst |momwire - scipy| on a fusing platform, in eps (basis values lie in
# [0, 1]), measured over this test's own samples on macOS arm64 (GitHub's
# macos-14 runner, scipy 1.18.1, numpy 2.5.4): 0 / 0.5 / 1.5 at degree
# 1 / 2 / 3 (degree 1 has no a*b + c to fuse). The tolerance is the next
# power of two at or above twice the worst, the rule the readouts use below.
_FUSED_TOL_EPS = {1: 0.0, 2: 1.0, 3: 4.0}


@pytest.mark.parametrize("k", [1, 2, 3])
def test_basis_rows_are_scipys_design_matrix_to_the_bit(k):
    """Values AND first-basis offsets, bit for bit on x86-64 (a uint64 view,
    so even a zero's sign counts), on clamped knots like the solver's:
    uniform and non-uniform meshes, one to many segments, and the samples the
    solver takes (d+1 per segment, so every interior knot is hit exactly)
    plus every knot and random interior points. Where scipy's build fuses,
    the offsets stay exact and the values agree to the measured tolerance."""
    from scipy.interpolate import BSpline

    rng = np.random.default_rng(1489 + k)
    worst = 0.0
    for trial in range(300):
        n_seg = int(rng.integers(1, 40))
        arc, h, t = _clamped_knots(rng, k, n_seg, uniform=trial % 3 == 0)
        unit = np.linspace(0.0, 1.0, k + 1)
        x = np.concatenate(
            [
                (arc[:-1, None] + h[:, None] * unit[None, :]).ravel(),
                arc,
                rng.uniform(0.0, arc[-1], 40),
            ]
        )
        x = np.clip(x, 0.0, arc[-1])
        dm = BSpline.design_matrix(x, t, k)
        ref_vals = dm.data.reshape(-1, k + 1)
        ref_first = dm.indices.reshape(-1, k + 1)[:, 0]
        vals, first = _bspline_basis_rows(x, t, k)
        assert np.array_equal(first, ref_first)
        dense = _bspline_design_matrix_dense(x[:7], t, k)
        ref_dense = BSpline.design_matrix(x[:7], t, k).toarray()
        if _SCIPY_UNFUSED:
            assert np.array_equal(vals.view(np.uint64), ref_vals.view(np.uint64))
            assert np.array_equal(dense.view(np.uint64), ref_dense.view(np.uint64))
        worst = max(worst, float(np.abs(vals - ref_vals).max()) / _EPS)
        worst = max(worst, float(np.abs(dense - ref_dense).max()) / _EPS)
    if not _SCIPY_UNFUSED:
        assert worst <= _FUSED_TOL_EPS[k], f"degree {k}: {worst} eps"


def test_basis_rows_take_the_right_span_on_a_knot_and_the_last_at_the_end():
    """The span rule, spelled out at degree 1 where it is visible: on an
    interior knot the row starts at the basis whose support OPENS there; at
    the final knot it closes the last span."""
    t = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 3.0])
    vals, first = _bspline_basis_rows(np.array([0.0, 1.0, 2.0, 3.0]), t, 1)
    assert first.tolist() == [0, 1, 2, 2]
    assert vals.tolist() == [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]


def test_basis_rows_refuse_what_scipys_design_matrix_refused():
    t = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0])
    for bad in (2.0 + 1e-12, -1e-18):
        with pytest.raises(ValueError, match="outside the knot range"):
            _bspline_basis_rows(np.array([0.5, bad]), t, 2)
    with pytest.raises(ValueError, match="not a finite number"):
        _bspline_basis_rows(np.array([np.nan]), t, 2)
    vals, first = _bspline_basis_rows(np.zeros(0), t, 2)
    assert vals.shape == (0, 3)
    assert first.shape == (0,)


# ---------------------------------------------------------------------------
# The readouts: per-span polynomials against scipy's de Boor
# ---------------------------------------------------------------------------


# The tolerance is DERIVED from a measurement (the eval-order policy), in
# units of eps times the evaluation's own scale: max |c| for the current
# and max |c| / h_min for the slope (h_min the wire's shortest span), the
# scales the per-span sums work at, since the basis is a partition of unity
# and its slope goes as 1/h. Pointwise relative error is the wrong ruler:
# the slope crosses zero.
#
# The survey (Skylake, scipy 1.18.1): degrees 1-3 on straight, unevenly
# meshed and bent single wires of 1 to 301 segments, a three-wire junction
# and a grounded stand, read at knots, centres and random points (3,720
# readout calls); and every readout call the bs2 NEC-5 render makes over
# the 401-deck step-0 corpus plus momwire's fixture decks (434 unique decks,
# 19,382 calls, degree 2). Worst |new - scipy|:
#
#     degree             1       2       3
#     current         1.47    3.92   12.99   (corpus, d=2: 3.24)
#     slope           1.30    7.35   33.98   (corpus, d=2: 3.37)
#
# It does not grow with the mesh (each value is a d+1-term sum on one span)
# and it does grow with degree, as the conversion to monomials does. The
# tolerance is the next power of two at or above TWICE the worst: the
# factor of two is one binary order of headroom for the reference itself,
# which is scipy's compiled de Boor and may round differently on another
# platform (an FMA-contracting build).
_TOL_EPS = {
    "current": {1: 4.0, 2: 8.0, 3: 32.0},
    "slope": {1: 4.0, 2: 16.0, 3: 128.0},
}


def _scipy_readouts(solver, coeffs, s_array):
    """The pre-#1489 readouts, verbatim in substance: scatter the kept
    coefficients onto the wire's full basis and let scipy evaluate."""
    from scipy.interpolate import BSpline

    geom = solver._build_geometry()
    _s, _p, _k, wire_knots, wbg = solver._build_basis_polynomials(geom)
    d = solver.degree
    cur, slope, scales = [], [], []
    for w in range(len(solver.wires_polylines)):
        arc = geom["per_wire"][w]["arc_at_knot"]
        kv = wire_knots[w]
        s = arc if s_array is None else np.asarray(s_array[w], dtype=np.float64)
        s = np.clip(s, kv[0], kv[-1])
        kept, l2g = wbg[w]
        c = np.zeros(len(kv) - d - 1, dtype=np.complex128)
        for i, (j, *_rest) in enumerate(kept):
            c[j] = coeffs[l2g[i]]
        dm = BSpline.design_matrix(s, kv, d).toarray()
        cur.append(dm @ c)
        slope.append(BSpline(kv, c, d, extrapolate=False).derivative(1)(s))
        cmax = float(np.abs(c).max())
        scales.append((cmax, cmax / float(np.diff(arc).min())))
    return cur, slope, scales


def _geometries():
    bent = np.array([[0, 0, 0.0], [3.0, 0, 4.0], [6.0, 1.0, 4.5], [7, 5, 2.0]])
    zs = np.linspace(-5.0, 5.0, 5)
    uneven = np.column_stack([np.zeros(5), np.zeros(5), zs])
    yield (
        "straight",
        {
            "wires": [np.array([[0, 0, -5.0], [0, 0, 5.0]])],
            "n_per_edge_per_wire": [[21]],
        },
    )
    yield "uneven", {"wires": [uneven], "n_per_edge_per_wire": [[2, 9, 3, 6]]}
    yield "bent", {"wires": [bent], "n_per_edge_per_wire": [[4, 7, 5]]}
    yield (
        "junction",
        {
            "wires": [
                np.array([[0, -5.0, 3.0], [0, 0, 3.0]]),
                np.array([[0, 0, 3.0], [4.5, 0, 3.0]]),
                np.array([[0, 0, 3.0], [0, 0, 8.1]]),
            ],
            "n_per_edge_per_wire": [[7], [8], [5]],
            "junctions": [[(0, "end"), (1, "start"), (2, "start")]],
        },
    )
    yield (
        "ground",
        {
            "wires": [
                np.array([[0, 0, 0.0], [0, 0, 5.0]]),
                np.array([[0, 0, 5.0], [4.0, 0, 5.0]]),
            ],
            "n_per_edge_per_wire": [[9], [11]],
            "junctions": [[(0, "end"), (1, "start")]],
            "ground_z": 0.0,
        },
    )


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("case", [name for name, _kw in _geometries()])
def test_readouts_agree_with_scipy_to_the_measured_tolerance(degree, case):
    kw = dict(_geometries())[case]
    solver = BSplineSolver(
        degree=degree,
        wavelength=20.0,
        wire_radius=1e-3,
        feeds=[(0, None, 1 + 0j)],
        **kw,
    )
    _z, coeffs = solver.compute_impedance()
    geom = solver._build_geometry()
    arcs = [pw["arc_at_knot"] for pw in geom["per_wire"]]
    rng = np.random.default_rng(degree)
    samples = {
        "knots": None,
        "centres": [0.5 * (a[:-1] + a[1:]) for a in arcs],
        "random": [rng.uniform(a[0], a[-1], 40) for a in arcs],
        # A hair outside each end is clipped onto it, as before.
        "ends": [np.array([a[0] - 1e-12, a[0], a[-1], a[-1] + 1e-12]) for a in arcs],
    }
    tol_i = _TOL_EPS["current"][degree] * _EPS
    tol_s = _TOL_EPS["slope"][degree] * _EPS
    for where, s_array in samples.items():
        cur = solver.currents_at_knots(coeffs, s_array)
        slope = solver.current_slopes(coeffs, s_array)
        ref_cur, ref_slope, scales = _scipy_readouts(solver, coeffs, s_array)
        for w, (sc_i, sc_s) in enumerate(scales):
            assert cur[w].dtype == np.complex128
            assert slope[w].dtype == np.complex128
            d_i = float(np.abs(cur[w] - ref_cur[w]).max()) / sc_i
            d_s = float(np.abs(slope[w] - ref_slope[w]).max()) / sc_s
            assert d_i <= tol_i, (where, w, d_i / _EPS)
            assert d_s <= tol_s, (where, w, d_s / _EPS)


def _tent_solver():
    return BSplineSolver(
        degree=1,
        wavelength=20.0,
        wire_radius=1e-3,
        feeds=[(0, None, 1 + 0j)],
        wires=[np.array([[0, 0, -5.0], [0, 0, 5.0]])],
        n_per_edge_per_wire=[[9]],
    )


def test_a_degree_one_slope_on_a_knot_reads_the_span_to_its_right():
    """At degree 1 the slope is constant on each span and jumps at a knot, so
    a sample ON a knot has to pick a side: the span to its right, and the
    last span at the wire's far end -- the side scipy's spline picked."""
    solver = _tent_solver()
    _z, coeffs = solver.compute_impedance()
    arc = solver._build_geometry()["per_wire"][0]["arc_at_knot"]
    centres = 0.5 * (arc[:-1] + arc[1:])
    (at_knots,) = solver.current_slopes(coeffs)
    (at_centres,) = solver.current_slopes(coeffs, [centres])
    # Bit for bit: on a span the degree-1 slope does not depend on u.
    assert np.array_equal(at_knots[:-1], at_centres)
    assert at_knots[-1] == at_centres[-1]
    # ... and the sides really differ, so the assertion above has teeth.
    assert np.abs(np.diff(at_centres)).max() > 0.0
    _rc, ref_slope, scales = _scipy_readouts(solver, coeffs, None)
    tol = _TOL_EPS["slope"][1] * _EPS * scales[0][1]
    assert np.abs(at_knots - ref_slope[0]).max() <= tol


def test_readouts_keep_their_edge_cases():
    solver = _tent_solver()
    _z, coeffs = solver.compute_impedance()
    empty = [np.zeros(0)]
    (cur,) = solver.currents_at_knots(coeffs, empty)
    (slope,) = solver.current_slopes(coeffs, empty)
    assert cur.shape == slope.shape == (0,)
    assert cur.dtype == slope.dtype == np.complex128
    # Not a number: the slope reads as not a number (scipy's spline answered
    # NaN), the current is refused (scipy's design matrix refused it).
    (slope,) = solver.current_slopes(coeffs, [np.array([1.0, np.nan])])
    assert np.isfinite(slope[0])
    assert np.isnan(slope[1].real) and np.isnan(slope[1].imag)
    with pytest.raises(ValueError, match="not a finite number"):
        solver.currents_at_knots(coeffs, [np.array([1.0, np.nan])])


def test_current_slopes_still_refuse_singular_enrichment():
    solver = BSplineSolver(
        degree=2,
        wavelength=20.0,
        wire_radius=1e-3,
        feeds=[(0, None, 1 + 0j)],
        wires=[
            np.array([[0, -5.0, 3.0], [0, 0, 3.0]]),
            np.array([[0, 0, 3.0], [4.5, 0, 3.0]]),
        ],
        n_per_edge_per_wire=[[6], [6]],
        junctions=[[(0, "end"), (1, "start")]],
        use_singular_enrichment=True,
    )
    with pytest.raises(NotImplementedError, match="use_singular_enrichment"):
        solver.current_slopes(np.zeros(1))


# A free-space dipole whose deck asks for charges (PQ), so the render reads
# both the knot currents and the slopes.
_SMALL_DECK = (
    Path(__file__).parent
    / "fixtures"
    / "eznec"
    / "decks"
    / "0042_dipole-in-free-space.nec"
)


def test_import_and_a_bs2_render_load_no_scipy_interpolate(tmp_path):
    """The point of #1489: neither `import momwire` nor a bs2 NEC-5 render
    loads scipy.interpolate or, through it, scipy.optimize. In a fresh
    interpreter, since this one has imported both for the tests above."""
    import subprocess
    import sys

    deck = _SMALL_DECK
    out = tmp_path / "dipole.out"
    probe = (
        "import sys\n"
        "import momwire, momwire.eznec._shell\n"
        "def heavy():\n"
        "    return sorted(m for m in sys.modules if m.split('.')[:2] in\n"
        "                  (['scipy', 'interpolate'], ['scipy', 'optimize']))\n"
        "print('import', heavy())\n"
        "from momwire.eznec._shell import main\n"
        "rc = main([sys.argv[1], sys.argv[2]], basis='bspline')\n"
        "print('render', rc, heavy())\n"
    )
    p = subprocess.run(
        [sys.executable, "-c", probe, str(deck), str(out)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert p.returncode == 0, p.stderr
    assert "import []" in p.stdout, p.stdout
    assert "render 0 []" in p.stdout, p.stdout
    text = out.read_text()
    assert "NEC ERROR" not in text
    assert "Wire Currents" in text
    assert "Wire Charge Densities" in text
