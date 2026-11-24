"""`fill_report()` (momwire#2016): what a fill served, read back.

The counts here are checked against something other than the code that keeps
them: a Python re-derivation of razor2p's key, a spy on the sinusoidal kernel
and on bspline's moment fill, and the sector map's own row count.
"""

from __future__ import annotations

import copy
import json
import math
import pickle

import numpy as np
import pytest

import momwire
from momwire import _accel, _lapack, _sin_pair_reuse as pr
from momwire import bspline as bspline_mod
from momwire import razor as razor_mod
from momwire._rotational_symmetry import RotationalSymmetryRefused
from momwire.bspline import BSplineSolver
from momwire.hmatrix import HMatrixSolver
from momwire.razor import RazorSolver
from momwire.sinusoidal import SinusoidalSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

acc = _accel.acc
pytestmark = pytest.mark.skipif(acc is None, reason="needs the C++ accelerator")

LAM = 21.0
RADIUS = 0.002
SOMMERFELD = {
    "ground_z": 0.0,
    "ground_eps": (13.0, 0.005),
    "ground_model": "sommerfeld",
}
PEC = {"ground_z": 0.0}


def yagi(n_el=6, n_seg=21, height=0.0, length=10.0, spacing=4.0):
    """Equal parallel elements on one segment length: every family repeats."""
    wires = [
        np.array(
            [[i * spacing, -length / 2, height], [i * spacing, length / 2, height]]
        )
        for i in range(n_el)
    ]
    return {
        "wires": wires,
        "n_per_edge_per_wire": [[n_seg]] * n_el,
        "wavelength": LAM,
        "wire_radius": RADIUS,
        "feed_wire_index": 1,
    }


def radial_screen(n_radials=8, n_seg=7):
    """A vertical on the axis over `n_radials` equally spaced radials."""
    wires = [np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 5.5]])]
    per_edge = [[11]]
    for i in range(n_radials):
        a = 2 * math.pi * i / n_radials
        wires.append(
            np.array([[0.0, 0.0, 0.5], [5.0 * math.cos(a), 5.0 * math.sin(a), 0.5]])
        )
        per_edge.append([n_seg])
    return {
        "wires": wires,
        "n_per_edge_per_wire": per_edge,
        "wavelength": LAM,
        "wire_radius": RADIUS,
        "feed_wire_index": 0,
    }


@pytest.fixture(autouse=True)
def _small_decks_and_fresh_plans(monkeypatch):
    """Serve the reuse on these small decks, from no plan left by another
    test."""
    acc.razor_reuse_set_small_deck_segs(0, 0)
    acc.bspline_reuse_set_small_deck_segs(0)
    monkeypatch.setattr(pr, "SMALL_DECK_SEGS_FREE", 0)
    monkeypatch.setattr(pr, "SMALL_DECK_SEGS_GROUND", 0)
    acc.razor_reuse_cache_clear()
    acc.bspline_reuse_cache_clear()
    pr.clear_shared_plans()
    yield
    acc.razor_reuse_set_small_deck_segs(-1, -1)
    acc.bspline_reuse_set_small_deck_segs(-1)
    acc.razor_reuse_cache_clear()
    acc.bspline_reuse_cache_clear()
    pr.clear_shared_plans()


def razor(deck=None, **kw):
    return RazorSolver(nec5_quadrature=True, **(deck or yagi()), **kw)


def reconciles(reuse):
    """computed + reused + near == pairs, per family, per wire block, in all."""
    for name, f in reuse["families"].items():
        assert f["computed"] + f["reused"] + f["near"] == f["pairs"], name
    assert reuse["computed"] + reuse["reused"] + reuse["near"] == reuse["pairs"]
    wires = reuse["wires"]
    p, c, r, n = (np.array(wires[k]) for k in ("pairs", "computed", "reused", "near"))
    assert np.array_equal(c + r + n, p)
    assert (int(p.sum()), int(c.sum()), int(r.sum()), int(n.sum())) == (
        reuse["pairs"],
        reuse["computed"],
        reuse["reused"],
        reuse["near"],
    )


# ---------------------------------------------------------------- the shape


@pytest.mark.parametrize(
    "make",
    [
        lambda: BSplineSolver(**yagi()),
        lambda: razor(),
        lambda: SinusoidalSolver(**yagi()),
        lambda: SinusoidalGalerkinSolver(**yagi()),
        lambda: HMatrixSolver(**yagi()),
    ],
    ids=["bs2", "razor", "sinusoidal", "galerkin", "hmatrix"],
)
def test_before_a_solve_it_is_empty_and_serialisable(make):
    report = make().fill_report()
    assert json.loads(json.dumps(report)) == report
    assert report["fills"] == 0 and report["route"] is None
    assert report["times"] == {"fill_s": 0.0, "factor_s": 0.0, "census_s": 0.0}
    assert report["reuse"] is None and report["sector"] is None


@pytest.mark.parametrize(
    "make, route",
    [
        (lambda: BSplineSolver(**yagi()), "dense"),
        (lambda: razor(), "dense"),
        (lambda: SinusoidalSolver(**yagi()), "dense"),
        (lambda: SinusoidalGalerkinSolver(**yagi()), "dense"),
    ],
    ids=["bs2", "razor", "sinusoidal", "galerkin"],
)
def test_stage_times_are_read_on_every_solve(make, route):
    s = make()
    s.compute_impedance()
    report = s.fill_report()
    json.dumps(report)
    assert report["route"] == route and report["fills"] == 1
    assert report["times"]["fill_s"] > 0.0 and report["times"]["factor_s"] > 0.0
    assert report["times"]["census_s"] == 0.0
    assert report["reuse"] is None


def test_the_factor_clock_belongs_to_the_solver_that_filled():
    a, b = BSplineSolver(**yagi(n_el=4)), BSplineSolver(**yagi(n_el=4))
    a.compute_impedance()
    before = a.fill_report()["times"]["factor_s"]
    b.compute_impedance()
    assert a.fill_report()["times"]["factor_s"] == before
    assert b.fill_report()["times"]["factor_s"] > 0.0


def test_a_solver_survives_a_pickle_with_its_totals():
    s = SinusoidalSolver(**yagi(n_el=3), pair_reuse=True)
    s.compute_impedance()
    fills = s.fill_report()["fills"]
    assert pickle.loads(pickle.dumps(s._fill_ledger)).fills == fills
    assert copy.deepcopy(s._fill_ledger).fills == fills


def test_sweep_times_are_totals_and_the_rest_is_the_last_fill():
    for s in (razor(pair_reuse=True), SinusoidalSolver(**yagi(), pair_reuse=True)):
        k = 2 * math.pi / LAM
        s.compute_impedance_swept(np.array([0.98, 1.0, 1.02]) * k)
        report = s.fill_report()
        assert report["fills"] == 3 and report["frequencies"] == 3
        assert report["reuse"]["state"] == "served"
        # The census ran once, on the first frequency.
        assert 0.0 < report["times"]["census_s"] < report["times"]["fill_s"]


def test_a_batched_sweep_counts_its_frequencies():
    s = BSplineSolver(**yagi(n_el=4))
    k = 2 * math.pi / LAM
    s.compute_impedance_swept(np.linspace(0.98, 1.02, 5) * k)
    report = s.fill_report()
    assert report["frequencies"] == 5
    assert report["fills"] <= 5


# ----------------------------------------------------- nothing runs unless on


def test_no_census_without_the_option(monkeypatch):
    calls = []
    real_razor, real_plan = acc.razor_reuse_plan, acc.sin_pair_reuse_plan
    monkeypatch.setattr(
        acc,
        "razor_reuse_plan",
        lambda *a, **k: calls.append("razor") or real_razor(*a, **k),
    )
    monkeypatch.setattr(
        acc,
        "sin_pair_reuse_plan",
        lambda *a, **k: calls.append("sin") or real_plan(*a, **k),
    )
    real_bs2 = acc.bspline_reuse_plan
    monkeypatch.setattr(
        acc,
        "bspline_reuse_plan",
        lambda *a, **k: calls.append("bs2") or real_bs2(*a, **k),
    )
    for s in (razor(), SinusoidalSolver(**yagi()), BSplineSolver(**yagi())):
        s.compute_impedance()
        assert s.fill_report()["reuse"] is None
    assert calls == []


def test_no_symmetry_probe_without_the_option(monkeypatch):
    def probe(*_a, **_k):
        raise AssertionError("the sector map ran with the option off")

    monkeypatch.setattr("momwire._rotational_symmetry.sector_map", probe, raising=True)
    for s in (BSplineSolver(**radial_screen()), SinusoidalSolver(**radial_screen())):
        s.compute_impedance()
        assert s.fill_report()["sector"] is None


# --------------------------------------------------------------- reconciling


@pytest.mark.parametrize(
    "ground, names",
    [
        ({}, {"direct"}),
        (PEC, {"direct", "image"}),
        (SOMMERFELD, {"direct", "image", "remainder"}),
    ],
    ids=["free", "pec", "sommerfeld"],
)
def test_razor_counts_reconcile(ground, names):
    s = razor(yagi(height=6.0 if ground else 0.0), pair_reuse=True, **ground)
    s.compute_impedance()
    reuse = s.fill_report()["reuse"]
    assert reuse["state"] == "served" and reuse["declined"] is None
    assert set(reuse["families"]) == names
    reconciles(reuse)
    assert reuse["reused"] > 0 and reuse["near"] > 0


@pytest.mark.parametrize(
    "ground, names",
    [
        ({}, {"direct"}),
        (PEC, {"direct", "image"}),
        (SOMMERFELD, {"direct", "reflected"}),
    ],
    ids=["free", "pec", "sommerfeld"],
)
def test_sinusoidal_counts_reconcile(ground, names):
    s = SinusoidalSolver(
        **yagi(height=6.0 if ground else 0.0), pair_reuse=True, **ground
    )
    s.compute_impedance()
    reuse = s.fill_report()["reuse"]
    assert reuse["state"] == "served"
    assert set(reuse["families"]) == names
    reconciles(reuse)


def test_bspline_counts_reconcile_on_the_tensor_route():
    deck = yagi()
    s = BSplineSolver(**deck, pair_reuse=True, offset_reuse=False)
    s.compute_impedance()
    served = s.pair_reuse_served
    assert served["route"] == "tensor"
    reuse = s.fill_report()["reuse"]
    assert reuse["state"] == "served"
    assert reuse["reused"] == served["copied_pairs"]
    assert reuse["near"] == served["near_pairs"]
    same = sum(n * n for (n,) in deck["n_per_edge_per_wire"])
    assert reuse["computed"] == served["classes"] + same
    reconciles(reuse)


def test_bspline_counts_reconcile_on_the_chunked_route():
    deck = yagi()
    s = BSplineSolver(**deck, pair_reuse=True)
    s.compute_impedance()
    served = s.pair_reuse_served
    assert served["route"] == "chunked"
    reuse = s.fill_report()["reuse"]
    assert reuse["reused"] == (
        served["copied_pairs"] + served["offset_pairs"] - served["offset_distinct"]
    )
    reconciles(reuse)


def test_bspline_offset_edges_alone_are_served():
    s = BSplineSolver(**yagi(), offset_reuse=True)
    s.compute_impedance()
    assert s.pair_reuse_served is None
    reuse = s.fill_report()["reuse"]
    assert reuse["state"] == "served"
    assert reuse["reused"] > 0
    reconciles(reuse)


def test_the_families_on_bspline_count_the_ground_image():
    # momwire#2068: the image is its own family (it used to be uncounted).
    deck = yagi(height=6.0)
    s = BSplineSolver(**deck, pair_reuse=True, ground_z=0.0)
    s.compute_impedance()
    reuse = s.fill_report()["reuse"]
    assert reuse["uncounted"] == []
    assert set(reuse["families"]) == {"direct", "image"}
    reconciles(reuse)


# ------------------------------------------- reporting what was served


def test_a_small_deck_reports_the_decline():
    acc.razor_reuse_set_small_deck_segs(-1, -1)
    s = razor(yagi(n_el=3), pair_reuse=True)
    s.compute_impedance()
    reuse = s.fill_report()["reuse"]
    assert reuse["state"] == "declined"
    assert "too few segments" in reuse["declined"]
    assert reuse["pairs"] is None and reuse["wires"] is None


def test_a_plan_past_its_allowance_reports_the_decline(monkeypatch):
    monkeypatch.setattr(pr, "ALLOWANCE_BYTES", 1024)
    s = SinusoidalSolver(**yagi(), pair_reuse=True)
    s.compute_impedance()
    reuse = s.fill_report()["reuse"]
    assert reuse["state"] == "declined"
    assert reuse["declined"] == pr.allowance_reason(1024)
    assert s.fill_report()["times"]["census_s"] >= 0.0


def test_a_later_unserved_fill_does_not_read_as_served():
    """The batched sweep reads no plan, whatever the fill before it served."""
    deck = yagi()
    s = BSplineSolver(**deck, pair_reuse=True, offset_reuse=False)
    s.compute_impedance()
    assert s.fill_report()["reuse"]["state"] == "served"
    s.compute_impedance_swept(np.array([0.99, 1.01]) * 2 * math.pi / LAM)
    reuse = s.fill_report()["reuse"]
    assert reuse["state"] == "declined"
    assert "batched sweep" in reuse["declined"]


def test_the_accelerated_operator_declines_by_name():
    deck = yagi()
    s = HMatrixSolver(**deck, pair_reuse=True)
    s.compute_impedance()
    report = s.fill_report()
    assert report["route"] == "hmatrix"
    assert report["reuse"]["state"] == "declined"


# ------------------------------------------------------------------ the map


def test_an_evicted_razor_plan_reads_unavailable_and_is_not_rebuilt():
    s = razor(pair_reuse=True)
    s.compute_impedance()
    assert "labels" in s.fill_report()["reuse"]["wires"]
    built = acc.razor_reuse_cache()["built"]
    acc.razor_reuse_cache_clear()
    wires = s.fill_report()["reuse"]["wires"]
    assert set(wires) == {"unavailable"}
    assert acc.razor_reuse_cache()["built"] == built


def test_an_evicted_sinusoidal_plan_reads_unavailable_and_is_not_rebuilt(monkeypatch):
    s = SinusoidalSolver(**yagi(), pair_reuse=True)
    s.compute_impedance()
    assert "labels" in s.fill_report()["reuse"]["wires"]

    def build(*_a, **_k):
        raise AssertionError("reading the report ran a census")

    monkeypatch.setattr(acc, "sin_pair_reuse_plan", build)
    pr.clear_shared_plans()
    s._sin_reuse_plans.clear()
    assert set(s.fill_report()["reuse"]["wires"]) == {"unavailable"}


def test_the_razor_cxx_route_reports_the_plan_its_core_read(monkeypatch):
    monkeypatch.setattr(razor_mod, "_RAZOR_CXX", True)
    before = razor_mod._CXX_FRONT_ROUTE.get("front", 0)
    s = razor(pair_reuse=True)
    s.compute_port_solution()
    assert razor_mod._CXX_FRONT_ROUTE.get("front", 0) == before + 1
    report = s.fill_report()
    assert report["route"] == "cxx" and report["fills"] == 1
    assert report["times"]["fill_s"] > 0.0 and report["times"]["factor_s"] > 0.0
    assert report["times"]["census_s"] > 0.0
    assert report["reuse"]["state"] == "served"
    reconciles(report["reuse"])


def test_the_map_is_skipped_on_request():
    s = razor(pair_reuse=True)
    s.compute_impedance()
    assert s.fill_report(wires=False)["reuse"]["wires"] is None


# ---------------------------------------------------- independent censuses


def razor_classes_by_brute_force(s):
    """razor2p's direct classes, re-derived in Python from the header's
    description of the key (free space, reduced kernel)."""
    geom = s._build_geometry()
    p0 = np.asarray(geom["seg_p0"])
    t = np.asarray(geom["seg_t"])
    h = np.asarray(geom["seg_h"])
    cent = p0 + 0.5 * h[:, None] * t
    a = RADIUS
    seen, near = set(), 0
    for o in range(len(h)):
        d = cent[o] - p0
        u = np.einsum("ij,ij->i", d, t)
        perp = d - u[:, None] * t
        rho = np.sqrt(np.einsum("ij,ij->i", perp, perp))
        v = u - 0.5 * h
        av = np.abs(v)
        for s_i in range(len(h)):
            if not max(av[s_i], rho[s_i]) >= 4.0 * max(h[s_i], a):
                near += 1
                continue
            D = max(av[s_i] - 0.5 * h[s_i], rho[s_i], a)
            e = math.frexp(D)[1] - 1
            inv = math.ldexp(1.0, 40 - e)
            seen.add(
                (
                    int(np.rint(av[s_i] * inv)),
                    int(np.rint(rho[s_i] * inv)),
                    float(h[s_i]),
                    e,
                )
            )
    return len(seen), near


def test_razor_counts_match_a_brute_force_census():
    s = razor(yagi(n_el=4, n_seg=15), pair_reuse=True, extended_kernel=False)
    s.compute_impedance()
    direct = s.fill_report()["reuse"]["families"]["direct"]
    computed, near = razor_classes_by_brute_force(s)
    assert direct["near"] == near
    assert direct["computed"] == computed


def test_sinusoidal_kernel_evals_match_a_spy_on_the_kernel(monkeypatch):
    pairs = []
    real = acc.sinusoidal_field_tensor_ek

    def spy(obs_c, obs_t, src_c, *rest, **kw):
        pairs.append(obs_c.shape[0] * src_c.shape[0])
        return real(obs_c, obs_t, src_c, *rest, **kw)

    monkeypatch.setattr(acc, "sinusoidal_field_tensor_ek", spy)
    s = SinusoidalSolver(
        **yagi(n_el=4, n_seg=15), pair_reuse=True, extended_kernel=True
    )
    s.compute_impedance()
    direct = s.fill_report()["reuse"]["families"]["direct"]
    assert sum(pairs) == direct["kernel_evals"]
    assert direct["kernel_evals"] < direct["pairs"]


def test_bspline_computed_pairs_match_a_spy_on_the_moment_fill(monkeypatch):
    pairs = []
    real = bspline_mod._seg_seg_full_moments_offedge

    def spy(seg_l_a, seg_r_a, seg_l_b, *rest, **kw):
        pairs.append(seg_l_a.shape[0] * seg_l_b.shape[0])
        return real(seg_l_a, seg_r_a, seg_l_b, *rest, **kw)

    monkeypatch.setattr(bspline_mod, "_seg_seg_full_moments_offedge", spy)
    deck = yagi(n_el=4, n_seg=15)
    s = BSplineSolver(**deck, pair_reuse=True)
    s.compute_impedance()
    served = s.pair_reuse_served
    assert served["route"] == "chunked"
    assert sum(pairs) == served["kernel_evals"]
    reuse = s.fill_report()["reuse"]
    assert reuse["families"]["direct"]["kernel_evals"] >= reuse["computed"]
    assert reuse["computed"] == (
        reuse["pairs"]
        - served["copied_pairs"]
        - served["near_pairs"]
        - served["offset_pairs"]
        + served["offset_distinct"]
    )


# ---------------------------------------------------------- bits are untouched


def solved(make):
    s = make()
    z, coeffs = s.compute_impedance()
    return s, np.asarray(z).tobytes() + np.asarray(coeffs).tobytes()


@pytest.mark.parametrize(
    "make",
    [
        lambda: razor(pair_reuse=True),
        lambda: SinusoidalSolver(**yagi(), pair_reuse=True),
        lambda: BSplineSolver(**yagi(), pair_reuse=True),
    ],
    ids=["razor", "sinusoidal", "bs2"],
)
def test_reading_the_report_moves_no_later_solve(make):
    s, first = solved(make)
    s.fill_report()
    z, coeffs = s.compute_impedance()
    assert np.asarray(z).tobytes() + np.asarray(coeffs).tobytes() == first
    _, again = solved(make)
    assert again == first


# ---------------------------------------------------------------- the sector


@pytest.mark.parametrize(
    "cls", [BSplineSolver, SinusoidalSolver], ids=["bs2", "sinusoidal"]
)
def test_the_sector_route_reports_the_map(cls):
    deck = radial_screen(n_radials=8)
    s = cls(**deck, rotational_symmetry=True)
    s.compute_impedance()
    report = s.fill_report()
    json.dumps(report)
    assert report["route"] == "sector"
    sector = report["sector"]
    smap = s._rotational_map
    assert sector["n_sectors"] == smap.n_sectors == 8
    assert sector["axial"] == len(smap.axial)
    assert sector["declined"] is None
    n_segs = sum(sum(p) for p in deck["n_per_edge_per_wire"])
    rows = sum(
        sum(deck["n_per_edge_per_wire"][w]) for w in (*smap.sectors[0], *smap.axial)
    )
    assert sector["pairs"] == n_segs**2
    assert sector["computed_pairs"] == rows * n_segs
    assert len(sector["blocks"]) == 1
    assert sector["wire_sector"].count(-1) == len(smap.axial)
    assert sorted(set(sector["wire_sector"]) - {-1}) == list(range(8))


def test_the_sector_route_lists_every_harmonic_it_factored():
    deck = radial_screen(n_radials=4)
    s = BSplineSolver(**deck, rotational_symmetry=True)
    s.compute_port_solution()
    blocks = s.fill_report()["sector"]["blocks"]
    assert len(blocks) >= 1 and blocks[0] == max(blocks)


def test_pair_reuse_on_the_sector_route_says_why_it_steps_aside():
    deck = radial_screen(n_radials=8)
    s = SinusoidalSolver(**deck, rotational_symmetry=True, pair_reuse=True)
    s.compute_impedance()
    report = s.fill_report()
    assert report["route"] == "sector"
    assert report["reuse"]["state"] == "declined"
    assert "sector route" in report["reuse"]["declined"]


def test_a_refused_deck_reports_the_refusal_sentence():
    deck = yagi(n_el=3)
    with pytest.raises(RotationalSymmetryRefused) as caught:
        BSplineSolver(**deck, rotational_symmetry=True)
    report = momwire.refused_fill_report(caught.value)
    json.dumps(report)
    assert report["sector"]["declined"] == str(caught.value)
    assert report["sector"]["declined"]
    assert report["fills"] == 0 and report["reuse"] is None


def test_the_factor_sink_follows_the_last_filler():
    s = BSplineSolver(**yagi(n_el=3))
    s.compute_impedance()
    ledger = s._fill_ledger
    before = ledger.factor_s
    _lapack.lu_factor(np.eye(3, dtype=np.complex128))
    assert ledger.factor_s > before
    _lapack.set_factor_sink(None)
    before = ledger.factor_s
    _lapack.lu_factor(np.eye(3, dtype=np.complex128))
    assert ledger.factor_s == before
