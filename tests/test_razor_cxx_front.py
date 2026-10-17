"""razor-2p's FRONT half in C++ (`_razor_front_core.cpp`) against the Python
front half it hardens, byte for byte.

The core takes `RazorSolver`'s own constructor arguments and builds what the
Python front half builds -- the geometry (`_build_geometry`'s dict), every
table `_assemble_Z_prepare` hands the fill on razor-2p's centroid-sharing
branch, and the port columns -- then fills and factors
(`RazorSolveContext.front_port_solution`). Each case below holds:

* every array (and the constructor's normalised state) equal to the Python
  front half's in bytes, dtype, shape and Python type;
* the filled Z, the coefficients and Y equal to the Python route's;
* the C++ branch having RUN, read off `razor._CXX_ROUTE` and the context's
  counters before any agreement is believed;
* every refusal the Python front half raises on the covered class raised by
  the core with the same type and message, and every route the core does
  not cover declined BY NAME.
"""

import importlib.util
import pathlib
import warnings

import numpy as np
import pytest
from _lane_gate import WIN32, lanes_match

from momwire import RazorSolver, razor
from momwire._accel import acc
from momwire._feed_snap import AmbiguousSite

pytestmark = pytest.mark.skipif(
    not razor._HAVE_RAZOR_FRONT_CXX, reason="accelerator without razor_front_proto"
)

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _compare_module():
    spec = importlib.util.spec_from_file_location(
        "core_split_front_compare", ROOT / "scripts" / "core_split" / "front_compare.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fc = _compare_module()

DIPOLE = [[(0.0, -5.0, 6.0), (0.0, 5.0, 6.0)]]
BASE = dict(
    wire_radius=0.002, wavelength=22.0, nec5_quadrature=True, exact_kernel="auto"
)

CASES = {
    "free": dict(wires=DIPOLE, nsegs=21, feed_arclength=5.0),
    "free_midpoint": dict(wires=DIPOLE, nsegs=21),
    "free_ek": dict(wires=DIPOLE, nsegs=21, feed_arclength=5.0, extended_kernel=True),
    "pec": dict(wires=DIPOLE, nsegs=21, feed_arclength=5.0, ground_z=0.0),
    "pec_ek": dict(
        wires=DIPOLE, nsegs=21, feed_arclength=5.0, ground_z=0.0, extended_kernel=True
    ),
    "pec_grounded": dict(
        wires=[[(0.0, 0.0, 0.0), (0.0, 0.0, 5.3)]],
        nsegs=15,
        feeds=[(0, 0.4, 1.0 + 0j)],
        ground_z=0.0,
    ),
    # A monopole fed AT its grounded end (the grounded tent is the port).
    "pec_base_fed": dict(
        wires=[[(0.0, 0.0, 0.0), (0.0, 0.0, 5.3)]],
        nsegs=15,
        feeds=[(0, 0.0, 1.0 + 0j)],
        ground_z=0.0,
    ),
    "junction_two_ports": dict(
        wires=[
            [(0.0, -5.0, 6.0), (0.0, 0.0, 6.0)],
            [(0.0, 0.0, 6.0), (0.0, 5.0, 6.0)],
            [(0.0, 0.0, 6.0), (0.0, 0.0, 9.0)],
        ],
        n_per_edge_per_wire=[[9], [9], [7]],
        feeds=[(0, 5.0 * 4 / 9, 1.0 + 0j), (2, 3.0 * 3 / 7, 0.5 + 0.25j)],
    ),
    "kinked_pec": dict(
        wires=[[(0.0, -5.0, 4.0), (0.0, 0.0, 6.0), (0.0, 5.0, 4.0)]],
        n_per_edge_per_wire=[[11, 11]],
        feeds=[(0, 5.4, 1.0 + 0j)],
        ground_z=0.0,
    ),
    # Two wires joined end to start: a split-wire feed at the junction knot.
    "split_feed": dict(
        wires=[[(0.0, -5.0, 6.0), (0.0, 0.0, 6.0)], [(0.0, 0.0, 6.0), (0.0, 5.0, 6.0)]],
        n_per_edge_per_wire=[[10], [10]],
        feeds=[(0, 5.0, 1.0 + 0j)],
    ),
    # A series EMF at a three-wire node: one through-current port (a dense
    # KCL column when it is named through the first member).
    "node_gap_first": dict(
        wires=[
            [(0.0, 0.0, 6.0), (0.0, -5.0, 6.0)],
            [(0.0, 0.0, 6.0), (0.0, 5.0, 6.0)],
            [(0.0, 0.0, 6.0), (0.0, 0.0, 9.0)],
        ],
        n_per_edge_per_wire=[[9], [9], [7]],
        feeds=[],
        node_gaps=[(0, "start", 1.0 + 0j)],
    ),
    "node_gap_second": dict(
        wires=[
            [(0.0, 0.0, 6.0), (0.0, -5.0, 6.0)],
            [(0.0, 0.0, 6.0), (0.0, 5.0, 6.0)],
            [(0.0, 0.0, 6.0), (0.0, 0.0, 9.0)],
        ],
        n_per_edge_per_wire=[[9], [9], [7]],
        feeds=[(2, 1.5, 1.0 + 0j)],
        node_gaps=[(1, "start", 0.5 + 0j), (2, "start", 0.25j)],
    ),
    # Per-wire radii, with and without the extended kernel's two-radius tube.
    "radius_step": dict(
        wires=[[(0.0, -5.0, 6.0), (0.0, 0.0, 6.0)], [(0.0, 0.0, 6.0), (0.0, 5.0, 6.0)]],
        n_per_edge_per_wire=[[10], [10]],
        wire_radius=[0.002, 0.004],
        feeds=[(0, 2.5, 1.0 + 0j)],
    ),
    "radius_step_ek_pec": dict(
        wires=[[(0.0, -5.0, 6.0), (0.0, 0.0, 6.0)], [(0.0, 0.0, 6.0), (0.0, 5.0, 6.0)]],
        n_per_edge_per_wire=[[10], [10]],
        wire_radius=[0.002, 0.004],
        feeds=[(0, 2.5, 1.0 + 0j)],
        ground_z=0.0,
        extended_kernel=True,
    ),
    # The exact ring kernel (momwire#1433), engaged by "auto" (segments under
    # three radii): the plan and the correction in C++, between the free
    # block and the image; with the extended kernel's labels and a grounded
    # base row; and two lines of two radii (one radius per line, the EK tube
    # per observer).
    "exact_fat": dict(wires=DIPOLE, nsegs=21, feed_arclength=5.0, wire_radius=0.4),
    "exact_fat_pec_ek": dict(
        wires=[[(0.0, 0.0, 0.0), (0.0, 0.0, 5.3)]],
        nsegs=15,
        feeds=[(0, 0.4, 1.0 + 0j)],
        ground_z=0.0,
        wire_radius=0.2,
        extended_kernel=True,
    ),
    # A radius step on one line: the groups, then the line's cross-radius
    # pairs on the two-radius moments (momwire#1438).
    "exact_radius_step": dict(
        wires=[
            [(0.0, -5.0, 6.0), (0.0, 0.0, 6.0)],
            [(0.0, 0.0, 6.0), (0.0, 5.0, 6.0)],
        ],
        n_per_edge_per_wire=[[10], [10]],
        wire_radius=[0.3, 0.4],
        feeds=[(0, 2.5, 1.0 + 0j)],
    ),
    "exact_radius_step_ek_pec": dict(
        wires=[
            [(0.0, -5.0, 6.0), (0.0, 0.0, 6.0)],
            [(0.0, 0.0, 6.0), (0.0, 5.0, 6.0)],
        ],
        n_per_edge_per_wire=[[10], [10]],
        wire_radius=[0.3, 0.4],
        feeds=[(0, 2.5, 1.0 + 0j)],
        ground_z=0.0,
        extended_kernel=True,
    ),
    "exact_two_lines_ek": dict(
        wires=[DIPOLE[0], [(1.5, -5.0, 6.0), (1.5, 5.0, 6.0)]],
        n_per_edge_per_wire=[[21], [17]],
        wire_radius=[0.4, 0.25],
        feeds=[(0, 5.0, 1.0 + 0j), (1, None, 0.5j)],
        extended_kernel=True,
    ),
    # A closed loop (a wire whose two ends coincide), off-axis and tilted.
    "loop": dict(
        wires=[
            [
                (1.0, 0.3, 4.0),
                (0.2, 1.1, 4.4),
                (-0.9, 0.1, 4.2),
                (0.1, -1.0, 3.9),
                (1.0, 0.3, 4.0),
            ]
        ],
        n_per_edge_per_wire=[[5, 6, 5, 7]],
        feeds=[(0, 0.9, 1.0 + 0j)],
        ground_z=0.5,
    ),
    # Several wires, irrational coordinates, a ground plane not at z = 0,
    # and many anchors on one polyline (numpy's pairwise sum of edge lengths).
    "yagi_irrational": dict(
        wires=[
            [(-0.1 * np.pi, -2.5 * np.e, 7.3), (0.1 * np.pi, 2.5 * np.e, 7.3)],
            [(1.7, -2.4 * np.sqrt(2), 7.3), (1.7, 2.4 * np.sqrt(2), 7.3)],
            [(-1.9, -2.7, 7.3), (-1.9, 2.7, 7.3)],
        ],
        n_per_edge_per_wire=[[23], [21], [19]],
        feeds=[(0, None, 1.0 + 0j)],
        ground_z=-0.25,
    ),
    "many_anchors": dict(
        wires=[[(0.0, 0.1 * i, 3.0 + 0.05 * np.sin(i)) for i in range(12)]],
        n_per_edge_per_wire=[[2] * 11],
        feeds=[(0, 0.33, 1.0 + 0j)],
        ground_z=0.0,
    ),
    # The composing Sommerfeld ground (momwire#1479 S3): an elevated dipole,
    # a base-fed vertical's grounded row, a dipole low enough to raise the
    # remainder's source order (momwire#510), and a radius step under the
    # extended kernel.
    "sommerfeld": dict(
        wires=DIPOLE,
        nsegs=21,
        feed_arclength=5.0,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
    ),
    "sommerfeld_contact": dict(
        wires=[[(0.0, 0.0, 0.0), (0.0, 0.0, 5.2)]],
        nsegs=12,
        feeds=[(0, 0.0, 1.0 + 0j)],
        ground_z=0.0,
        ground_eps=(5.0, 0.001),
        ground_model="sommerfeld",
    ),
    "sommerfeld_grazing": dict(
        wires=[[(0.0, -5.0, 0.05), (0.0, 5.0, 0.05)]],
        nsegs=21,
        feed_arclength=5.0,
        ground_z=0.0,
        ground_eps=(5.0, 0.001),
        ground_model="sommerfeld",
    ),
    "sommerfeld_radius_step_ek": dict(
        wires=[[(0.0, -5.0, 6.0), (0.0, 0.0, 6.0)], [(0.0, 0.0, 6.0), (0.0, 5.0, 6.0)]],
        n_per_edge_per_wire=[[10], [10]],
        wire_radius=[0.002, 0.004],
        feeds=[(0, 2.5, 1.0 + 0j)],
        ground_z=0.0,
        ground_eps=(15.0, 0.008),
        ground_model="sommerfeld",
        extended_kernel=True,
    ),
    # One-segment wires joined at one end (momwire#608), and a scalar count.
    "one_segment_wires": dict(
        wires=[
            [(0.0, -5.0, 6.0), (0.0, 0.0, 6.0)],
            [(0.0, 0.0, 6.0), (0.0, 0.0, 6.7)],
            [(0.0, 0.0, 6.7), (0.0, 0.0, 7.1)],
        ],
        n_per_edge_per_wire=[9, [1], 1],
        feeds=[(0, 5.0 * 4 / 9, 1.0 + 0j)],
    ),
}


@pytest.fixture
def capture(monkeypatch):
    """The Python route's Z as `_lu_solve` receives it; the core's tables."""
    seen = {}
    orig = razor._bspline._lu_solve

    def lu(Z, rhs, **kw):
        seen["Z"] = np.array(Z, order="F", copy=True)
        return orig(Z, rhs, **kw)

    monkeypatch.setattr(razor._bspline, "_lu_solve", lu)
    monkeypatch.setattr(razor, "_CXX_WANT_Z", True)
    monkeypatch.setattr(razor, "_CXX_WANT_FRONT", True)
    monkeypatch.setattr(razor, "_CXX_ROUTE", {})
    monkeypatch.setattr(razor, "_CXX_FRONT_ROUTE", {})
    return seen


def _both(monkeypatch, seen, kw):
    kw = {**BASE, **kw}
    monkeypatch.setattr(razor, "_RAZOR_CXX", False)
    seen.clear()
    ref = RazorSolver(**kw)
    py = ref.compute_port_solution()
    z_py = seen["Z"]
    monkeypatch.setattr(razor, "_RAZOR_CXX", True)
    solver = RazorSolver(**kw)
    cx = solver.compute_port_solution()
    return ref, py, z_py, solver, cx


def _all_lanes_match(got, want) -> bool:
    """`lanes_match` through a readout's nesting (one array per wire)."""
    if isinstance(want, (list, tuple)):
        return len(got) == len(want) and all(
            _all_lanes_match(g, w) for g, w in zip(got, want, strict=True)
        )
    return lanes_match(got, want)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_front_half_is_the_python_front_half_to_the_bit(monkeypatch, capture, name):
    ctx = razor._cxx_context()
    before = ctx.counters
    ref, py, z_py, solver, cx = _both(monkeypatch, capture, CASES[name])
    after = razor._cxx_context().counters
    # The C++ branch RAN: the route map, and the context's own counters.
    assert razor._CXX_ROUTE == {"cxx-front": 1}
    assert razor._CXX_FRONT_ROUTE == {"front": 1}
    assert after["solves"] == before["solves"] + 1
    assert after["lu_calls"] == before["lu_calls"] + 1
    if CASES[name].get("ground_z") is not None:
        assert after["image_windows"] > before["image_windows"]
    if CASES[name].get("ground_model") == "sommerfeld":
        assert after["compose_windows"] > before["compose_windows"]
    geom, front, cols = fc.python_front(ref)
    assert fc.differences(geom, solver._cxx_geom) == []
    assert fc.differences(front, solver._cxx_front) == []
    assert fc.differences(cols, solver._cxx_cols) == []
    # The geometry the readouts read: the core's, unless `exact_kernel="auto"`
    # already had the constructor build one (it resolves on the mesh).
    assert fc.differences(solver._cached_geometry, solver._cxx_geom) == []

    if WIN32:
        # Windows needs no bit-compatibility: under MSVC's /fp:fast the core's
        # fill and the extension's (the same text, other TUs) may round
        # differently -- the Sommerfeld cases since the cores' Sommerfeld
        # port, the exact-kernel radius steps since the buried port -- so
        # there the derived lane tolerance applies.
        assert lanes_match(solver._cxx_z, z_py)
        assert lanes_match(cx.coeffs, py.coeffs)
        assert lanes_match(cx.y, py.y)
        assert _all_lanes_match(
            solver.currents_at_knots(cx.coeffs[:, 0]),
            ref.currents_at_knots(py.coeffs[:, 0]),
        )
        return
    assert solver._cxx_z.tobytes() == z_py.tobytes()
    assert cx.coeffs.tobytes() == py.coeffs.tobytes()
    assert cx.y.tobytes() == py.y.tobytes()
    # The readout over the core's geometry is the readout over Python's.
    assert (
        fc.differences(
            solver.currents_at_knots(cx.coeffs[:, 0]),
            ref.currents_at_knots(py.coeffs[:, 0]),
        )
        == []
    )


def test_without_auto_no_python_geometry_is_built(monkeypatch, capture):
    # `exact_kernel=False`: the constructor never builds the geometry, so on
    # the C++ route nothing of the Python front half runs and the readouts
    # read the core's own dict.
    kw = dict(CASES["junction_two_ports"], exact_kernel=False)
    ref, py, z_py, solver, cx = _both(monkeypatch, capture, kw)
    assert razor._CXX_ROUTE == {"cxx-front": 1}
    assert solver._cached_geometry is solver._cxx_geom
    assert cx.coeffs.tobytes() == py.coeffs.tobytes()
    assert (
        fc.differences(
            solver.currents_at_knots(cx.coeffs[:, 1]),
            ref.currents_at_knots(py.coeffs[:, 1]),
        )
        == []
    )


def test_many_windows_are_the_one_window_to_the_bit(monkeypatch, capture):
    monkeypatch.setattr(razor, "_CHUNK_ELEMS", 1)
    before = razor._cxx_context().counters["windows"]
    ref, py, z_py, solver, cx = _both(monkeypatch, capture, CASES["pec"])
    assert razor._cxx_context().counters["windows"] - before == 20
    assert razor._CXX_ROUTE == {"cxx-front": 1}
    assert solver._cxx_front["window_rows"] == 1
    assert solver._cxx_z.tobytes() == z_py.tobytes()
    assert cx.coeffs.tobytes() == py.coeffs.tobytes()


def test_a_prepared_sweep_keeps_the_fill_only_route(monkeypatch, capture):
    # `prepared` handed in (a sweep's hoisted prepare): the front half is the
    # caller's already, so only the fill and the LU move to C++.
    monkeypatch.setattr(razor, "_RAZOR_CXX", True)
    solver = RazorSolver(**BASE, **CASES["free"])
    prepared = solver._assemble_Z_prepare(solver._build_geometry())
    solver.compute_port_solution(prepared)
    assert razor._CXX_ROUTE == {"cxx": 1}
    assert razor._CXX_FRONT_ROUTE == {}


def test_the_default_is_the_python_route(monkeypatch):
    monkeypatch.setattr(razor, "_RAZOR_CXX", None)
    monkeypatch.delenv("MOMWIRE_RAZOR_CXX", raising=False)
    monkeypatch.setattr(razor, "_CXX_ROUTE", {})
    monkeypatch.setattr(razor, "_CXX_FRONT_ROUTE", {})
    RazorSolver(**BASE, **CASES["free"]).compute_port_solution()
    assert razor._CXX_ROUTE == {} and razor._CXX_FRONT_ROUTE == {}


def test_the_environment_switch_routes(monkeypatch):
    monkeypatch.setattr(razor, "_RAZOR_CXX", None)
    monkeypatch.setenv("MOMWIRE_RAZOR_CXX", "1")
    monkeypatch.setattr(razor, "_CXX_ROUTE", {})
    monkeypatch.setattr(razor, "_CXX_FRONT_ROUTE", {})
    RazorSolver(**BASE, **CASES["free"]).compute_port_solution()
    assert razor._CXX_ROUTE == {"cxx-front": 1}


def test_a_tripped_token_aborts_inside_the_native_call(monkeypatch):
    from momwire import CancelToken, SolveAborted

    monkeypatch.setattr(razor, "_RAZOR_CXX", True)
    monkeypatch.setattr(razor, "_CXX_FRONT_ROUTE", {})
    token = CancelToken()
    solver = RazorSolver(**BASE, **CASES["pec"], cancel=token)
    # The Python checkpoints are silenced, so only the C++ poll can see it.
    monkeypatch.setattr(solver, "_checkpoint", lambda: None)
    token.cancel()
    with pytest.raises(SolveAborted):
        solver.compute_port_solution()
    assert razor._CXX_FRONT_ROUTE == {}  # aborted inside, never counted


# --- declines, each by name, each still answered by the Python route -------

DECLINES = [
    # Read off the constructed solver (`_cxx_front_refusal`).
    (dict(lumped_loads=[(0, 5.0, 50 + 0j)]), "loading"),
    (dict(ground_z=0.0, ground_eps=13.0 - 1.2j), "finite-ground"),
    # The composing ground spelled with a complex eps~ rather than the seam's
    # (eps_r, sigma): the core reads only the seam's spelling.
    (
        dict(ground_z=0.0, ground_eps=13.0 - 1.2j, ground_model="sommerfeld"),
        "malformed-args",
    ),
    (dict(nec5_quadrature=False, n_qp_path=2), "path-rule"),
    (dict(exact_kernel=True), "exact-kernel"),
    (
        dict(
            wires=[DIPOLE[0], DIPOLE[0][::-1], [(1.0, -5.0, 6.0), (1.0, 5.0, 6.0)]],
            n_per_edge_per_wire=[[21], [21], [21]],
            feeds=[(2, 5.0, 1.0 + 0j)],
        ),
        "duplicated-wire",
    ),
    (
        dict(
            wires=[
                [(0.0, -5.0, 6.0), (0.0, 0.0, 6.0)],
                [(0.0, 0.0, 6.0), (0.0, 5.0, 6.0)],
            ],
            n_per_edge_per_wire=[[10], [10]],
            junctions=[[(0, "end"), (1, "start")]],
            feeds=[(0, 2.5, 1.0 + 0j)],
        ),
        "declared-junctions",
    ),
]


@pytest.mark.parametrize("extra, reason", DECLINES, ids=[r for _, r in DECLINES])
def test_an_uncovered_solve_is_declined_by_name(monkeypatch, capture, extra, reason):
    kw = {**CASES["free"], **extra}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref, py, _, solver, cx = _both(monkeypatch, capture, kw)
    assert razor._CXX_FRONT_ROUTE == {reason: 1}
    assert "cxx-front" not in razor._CXX_ROUTE
    assert cx.coeffs.tobytes() == py.coeffs.tobytes()


def test_the_core_declines_what_the_attributes_cannot_see():
    # The core's own declines, asked directly of `razor_front_build`.
    kw = dict(BASE, **CASES["free"])
    assert (
        acc.razor_front_build(dict(kw, exact_kernel=True))["decline"] == "exact-kernel"
    )
    assert acc.razor_front_build(dict(kw, wire_radius=0.4))["decline"] is None
    # A wire below a perfect ground raises Python's ValueError since
    # momwire#1512 B1, in its words, rather than declining.
    with pytest.raises(ValueError, match="runs below the ground plane"):
        acc.razor_front_build(
            dict(kw, wires=[[(0.0, 0.0, -1.0), (0.0, 0.0, -3.0)]], ground_z=0.0)
        )
    # A duplicated wire is served since momwire#1512 step 4: the front drops
    # the copy (`drop_duplicated_wires`) as Python does.
    assert (
        acc.razor_front_build(dict(kw, wires=[DIPOLE[0], DIPOLE[0]], nsegs=21))[
            "decline"
        ]
        is None
    )
    assert acc.razor_front_build(dict(kw, feeds=[(0.5, 1.0, 1)]))["decline"] == (
        "malformed-args"
    )
    # Beyond the frozen table Python takes numpy's eigensolve, not replayed.
    assert acc.razor_front_build(dict(kw, n_qp_source=193))["decline"] == (
        "quadrature-order"
    )
    assert acc.razor_front_build(dict(kw, n_qp_source=8))["decline"] is None
    assert acc.razor_front_build(kw)["decline"] is None


# --- refusals: the same exception, the same words ------------------------

_THREE = [
    [(0.0, -5.0, 6.0), (0.0, 0.0, 6.0)],
    [(0.0, 0.0, 6.0), (0.0, 5.0, 6.0)],
    [(0.0, 0.0, 6.0), (0.0, 0.0, 9.0)],
]

REFUSALS = {
    "no_wires": dict(wires=[]),
    "one_anchor": dict(wires=[[(0.0, 0.0, 1.0)]]),
    "ground_not_finite": dict(ground_z=float("nan")),
    "n_qp_source": dict(n_qp_source=0),
    "n_qp_path": dict(n_qp_path=0),
    "in_plane_edge": dict(wires=[[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]], ground_z=0.0),
    "interior_touch": dict(
        wires=[[(0.0, 0.0, 2.0), (1.0, 0.0, 0.0), (2.0, 0.0, 2.0)]],
        n_per_edge_per_wire=[[5, 5]],
        ground_z=0.0,
    ),
    "radius_shape": dict(wire_radius=[0.001, 0.002]),
    "radius_zero": dict(wire_radius=0.0),
    "radius_negative_list": dict(
        wires=_THREE,
        n_per_edge_per_wire=[[9], [9], [7]],
        wire_radius=[0.001, -2.0, 0.0],
    ),
    "radius_nan": dict(wire_radius=float("nan")),
    "counts_length": dict(n_per_edge_per_wire=[[5], [5]]),
    "edge_counts_length": dict(n_per_edge_per_wire=[[5, 5]]),
    "edge_count_zero": dict(n_per_edge_per_wire=[[0]]),
    "one_segment_unjoined": dict(
        wires=[DIPOLE[0], [(3.0, 0.0, 6.0), (3.0, 0.5, 6.0)]],
        n_per_edge_per_wire=[[21], [1]],
    ),
    "node_gap_range": dict(feeds=[], node_gaps=[(7, "start", 1.0)]),
    "node_gap_not_member": dict(feeds=[], node_gaps=[(0, "start", 1.0)]),
    "node_gap_single_member": dict(
        wires=[[(0.0, 0.0, 0.0), (0.0, 0.0, 5.0)]],
        ground_z=0.0,
        feeds=[],
        node_gaps=[(0, "start", 1.0)],
    ),
    "node_gap_twice": dict(
        wires=_THREE,
        n_per_edge_per_wire=[[9], [9], [7]],
        feeds=[],
        node_gaps=[(1, "start", 1.0), (1, "start", 1.0)],
    ),
    "node_gap_two_wire_cut": dict(
        wires=_THREE[:2],
        n_per_edge_per_wire=[[9], [9]],
        feeds=[],
        node_gaps=[(0, "end", 1.0), (1, "start", 1.0)],
    ),
    "node_gap_grounded": dict(
        wires=[
            [(0.0, 0.0, 0.0), (0.0, 0.0, 5.0)],
            [(0.0, 0.0, 0.0), (3.0, 0.0, 2.0)],
        ],
        n_per_edge_per_wire=[[9], [9]],
        ground_z=0.0,
        feeds=[],
        node_gaps=[(1, "start", 1.0)],
    ),
    "feed_range": dict(feeds=[(3, 1.0, 1.0)]),
    "feed_wire_index_range": dict(feed_wire_index=4),
    "no_ports": dict(feeds=[]),
    "zero_length_edge": dict(
        wires=[[(0.0, 0.0, 6.0), (0.0, 0.0, 6.0), (0.0, 1.0, 6.0)]],
        n_per_edge_per_wire=[[3, 3]],
        exact_kernel=False,
    ),
    "zero_length_edge_auto": dict(
        wires=[[(0.0, 0.0, 6.0), (0.0, 0.0, 6.0), (0.0, 1.0, 6.0)]],
        n_per_edge_per_wire=[[3, 3]],
    ),
    "bundle": dict(
        wires=[
            [(0.0, -5.0, 6.0), (0.0, 5.0, 6.0)],
            [(0.0, -5.0, 6.0), (0.0, 0.0, 6.0), (0.0, 5.0, 6.0)],
        ],
        n_per_edge_per_wire=[[20], [10, 10]],
        feeds=[(0, 5.0, 1.0)],
    ),
    "feed_at_three_wire_node": dict(
        wires=_THREE, n_per_edge_per_wire=[[9], [9], [7]], feeds=[(0, 5.0, 1.0)]
    ),
}


def _python_refusal(kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            RazorSolver(**kw).compute_port_solution()
        except (ValueError, NotImplementedError) as exc:
            return type(exc), str(exc)
    return None


@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_a_refusal_is_the_python_refusal_word_for_word(monkeypatch, name):
    monkeypatch.setattr(razor, "_RAZOR_CXX", False)
    kw = {**BASE, **CASES["free"], **REFUSALS[name]}
    expected = _python_refusal(kw)
    assert expected is not None, "the case is meant to be refused"
    with pytest.raises(expected[0]) as got:
        out = acc.razor_front_build(kw)
        pytest.fail(f"the core did not refuse: {out.get('decline')!r}")
    assert type(got.value) is expected[0]
    assert str(got.value) == expected[1]


# --- ties (momwire#1262) ---------------------------------------------------


def test_an_ambiguous_feed_warns_on_both_routes(monkeypatch, capture):
    # A monopole fed between two knots, with a hub below and a top wire
    # above, so the deck is not symmetric about the tie.
    kw = dict(
        wires=[
            [(0.0, 0.0, 1.0), (0.0, 0.0, 5.0)],
            [(0.0, 0.0, 5.0), (1.0, 0.0, 5.0)],
            [(0.0, 0.0, 1.0), (0.7, 0.4, 0.6)],
        ],
        n_per_edge_per_wire=[[8], [3], [3]],
        feeds=[(0, 1.25, 1.0 + 0j)],
    )

    def run(on):
        monkeypatch.setattr(razor, "_RAZOR_CXX", on)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sol = RazorSolver(**BASE, **kw).compute_port_solution()
        return sol, [str(x.message) for x in w if issubclass(x.category, AmbiguousSite)]

    py, w_py = run(False)
    cx, w_cx = run(True)
    assert razor._CXX_ROUTE == {"cxx-front": 1}
    assert len(w_py) == 1 and w_cx == w_py
    assert cx.coeffs.tobytes() == py.coeffs.tobytes()


# --- the constants the core carries ----------------------------------------


def test_the_table_is_the_frozen_one_for_every_tabled_order():
    """The core's Gauss-Legendre rules are `_quadrature.leggauss`'s, bit for
    bit, on every platform: both read the one frozen table (momwire#1484),
    the core through the header generated from it."""
    from momwire import _leggauss_table, _quadrature

    for n in range(1, _leggauss_table.N_MAX + 1):
        x, w = acc.razor_front_leggauss(n)
        rx, rw = _quadrature.leggauss(n)
        assert x.tobytes() == rx.tobytes() and w.tobytes() == rw.tobytes(), n
    assert acc.razor_front_leggauss(_leggauss_table.N_MAX + 1) is None
    assert acc.razor_front_leggauss(0) is None


def test_the_built_in_constants_are_razors():
    k = acc.razor_front_constants()
    assert k["eps"] == RazorSolver.eps and k["mu"] == RazorSolver.mu
    from momwire import _exact_kernel_seam, _feed_snap, _junction_rule

    assert k["junction_tol"] == _junction_rule.JUNCTION_TOL
    assert k["ambiguity_tol_frac"] == _feed_snap.AMBIGUITY_TOL_FRAC
    assert (
        k["exact_kernel_auto_h_over_a"] == _exact_kernel_seam.EXACT_KERNEL_AUTO_H_OVER_A
    )
    assert k["chunk_elems"] == razor._CHUNK_ELEMS
    assert k["n_qp_source"] == 12  # RazorSolver's default, the table's order


@pytest.mark.parametrize(
    "values",
    [
        [0.0005],
        [0.001, 0.0, 0.002],
        [0.001, -2.0, 0.0],
        [float("nan"), 0.001],
        [float("inf"), -float("inf"), 1.5],
        [1e-5, 2.5],
        [1e-5, 1e5],
        [123456789.0, 1.0],
        [-0.0, 0.25],
        [0.1 * i for i in range(40)],
        [1.0 / 3.0, 2.0 / 3.0, 1e-3],
        [3.14159265358979, 2.718281828459045e-7],
        [float("nan")],
        list(np.random.default_rng(5).standard_normal(30) * 1e-3),
        list(np.random.default_rng(6).standard_normal(1200)),
    ],
)
def test_the_radius_refusal_spells_numbers_as_wire_spec_does(values):
    from momwire import _wire_spec

    assert acc.razor_front_format_numbers(values) == _wire_spec.format_numbers(values)
