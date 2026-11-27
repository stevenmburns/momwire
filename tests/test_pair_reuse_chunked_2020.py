"""bspline's chunked fill serves `pair_reuse` (momwire#2020; the census's plan
since momwire#2063).

`offset_reuse`, a part of `pair_reuse`, sends any deck with a uniform edge of 12
or more segments down `_compute_Z_dense_chunked`, and that route used to read no
plan. Now the two compose:

- each observer window is gathered from the plan's class values, its own
  edge's columns the sweep's own fill, counted where they are served and
  held against what the kernel was actually asked for;
- congruent offset edges (the same arcs and radius, to the bit) share one
  table of distinct moments;
- the answer stays inside the two features' declared drift;
- every way a plan goes unread is named in `pair_reuse_declined`, on the
  routes that read no plan (a fill of chosen rows, the buried fill, the
  H-matrix and array-block operators) as well: those keep the fill's bits
  without reuse. A sweep with a served plan leaves the batched route to read
  it (momwire#2172).
"""

from __future__ import annotations

import numpy as np
import pytest

import momwire.bspline as bs
from momwire import _features
from momwire.array_block import ArrayBlockSolver
from momwire.bspline import BSplineSolver
from momwire.hmatrix import HMatrixSolver

# An experimental feature (`momwire._features`, reach 4: through the drop-in
# on Linux and Windows).
pytestmark = pytest.mark.reach(4)

LAM = 21.26

#: |dZin|/|Zin| the two features declare, added: each moves the answer alone.
DRIFT = (
    _features.FEATURES["pair_reuse"].drift + _features.FEATURES["offset_reuse"].drift
)
#: Z itself moves at the key's rounding (2^-40 of a pair's scale), far under
#: the answer's drift.
Z_DRIFT = 1e-11


@pytest.fixture(autouse=True)
def _small_decks():
    """These decks are below the small-deck threshold; the census is what is
    under test, so it runs on them."""
    acc = bs._acc
    if acc is None or not hasattr(acc, "bspline_reuse_plan"):
        pytest.skip("bs2's pair-reuse census is not built")
    acc.bspline_reuse_set_small_deck_segs(0)
    yield
    acc.bspline_reuse_set_small_deck_segs(-1)


def _deck(count=8, n=21):
    """`count` identical parallel dipoles at one spacing."""
    L = 0.47 * LAM
    wires = [
        np.array([[3.0 * i, -L / 2, 10.0], [3.0 * i, L / 2, 10.0]])
        for i in range(count)
    ]
    return wires, [[n] for _ in wires]


def _solver(count=8, n=21, **kw):
    wires, segs = _deck(count, n)
    kw.setdefault("pair_reuse", True)
    return BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=segs,
        feed_wire_index=0,
        wavelength=LAM,
        wire_radius=0.002,
        **kw,
    )


def _dense(count=8, n=21, **kw):
    return _solver(count, n, pair_reuse=False, offset_reuse=False, **kw)


def _fill(s, **kw):
    geom = s._build_geometry()
    supp, polys, *_ = s._build_basis_polynomials(geom)
    return np.array(s._compute_Z_operator(geom, supp, polys, **kw))


def _zin(s):
    return 1 / complex(np.asarray(s.compute_port_solution().y)[0, 0])


def _moved(Z, ref):
    return float(np.max(np.abs(Z - ref)) / np.max(np.abs(ref)))


@pytest.fixture
def kernel_pairs(monkeypatch):
    """The segment pairs the off-edge kernel is really asked for."""
    asked = []
    real = bs._seg_seg_full_moments_offedge

    def spy(seg_l_i, seg_r_i, seg_l_j, seg_r_j, *a, **kw):
        asked.append(len(seg_l_i) * len(seg_l_j))
        return real(seg_l_i, seg_r_i, seg_l_j, seg_r_j, *a, **kw)

    monkeypatch.setattr(bs, "_seg_seg_full_moments_offedge", spy)
    return asked


@pytest.fixture
def plan_reads(monkeypatch):
    """How many censuses a fill ran (or asked the process's cache for)."""
    reads = []
    real = BSplineSolver._reuse_plan_built

    def spy(self, geom):
        reads.append(1)
        return real(self, geom)

    monkeypatch.setattr(BSplineSolver, "_reuse_plan_built", spy)
    return reads


def test_the_chunked_route_gathers_from_the_plan(kernel_pairs):
    """Eight dipoles at one spacing: the kernel is asked for the plan's
    representatives' groups only (every own block is an offset edge's), and
    every other pair between two dipoles is copied."""
    s = _solver()
    assert s._offset_reuse, "offset_reuse follows pair_reuse"
    _fill(s)
    served = s.pair_reuse_served
    assert served["route"] == "chunked"
    assert s.pair_reuse_declined is None
    # The kernel's own count, not the fill's report of itself.
    assert sum(kernel_pairs) == served["kernel_evals"]
    assert served["copied_pairs"] > 10 * served["classes"] > 0


def test_the_counts_reconcile_with_the_matrix():
    s = _solver()
    _fill(s)
    c = s.pair_reuse_served
    assert (
        c["copied_pairs"] + c["classes"] + c["near_pairs"] + c["offset_pairs"]
        == c["pairs"]
    )
    assert c["pairs"] == (8 * 21) ** 2
    assert c["offset_pairs"] == 8 * 21 * 21


def test_congruent_offset_edges_share_one_table():
    s = _solver()
    _fill(s)
    c = s.pair_reuse_served
    assert c["tables_shared"] == 7
    assert c["offset_distinct"] == 2 * 21 - 1


def test_offset_reuse_alone_shares_no_table_and_reports_as_before():
    """`offset_reuse` without the whole is its own fill, as before the census:
    each edge its own table, nothing served to `pair_reuse_served`."""
    s = _solver(pair_reuse=False, offset_reuse=True)
    _fill(s)
    assert s.pair_reuse_served is None
    assert s.fill_report()["reuse"]["computed"] == (8 * 21) ** 2 - 8 * (21 * 21 - 41)


def test_the_values_reach_the_matrix(monkeypatch):
    """Spoil the class values: a route that did not read them would not
    notice."""
    ref = _fill(_dense())
    real = BSplineSolver._reuse_values

    def spoiled(self, *a, **kw):
        return 2.0 * real(self, *a, **kw)

    monkeypatch.setattr(BSplineSolver, "_reuse_values", spoiled)
    assert _moved(_fill(_solver()), ref) > 1e3 * Z_DRIFT


@pytest.mark.parametrize("ek", [False, True])
def test_the_answer_stays_inside_the_declared_drift(ek):
    dense, both = _dense(extended_kernel=ek), _solver(extended_kernel=ek)
    assert _moved(_fill(both), _fill(_dense(extended_kernel=ek))) <= Z_DRIFT
    z0, z1 = _zin(dense), _zin(both)
    assert abs(z1 - z0) / abs(z0) <= DRIFT
    assert both.pair_reuse_served["copied_pairs"] > 0


def test_over_a_perfect_ground_the_free_space_term_still_copies():
    ground = {"ground_z": 0.0}
    both = _solver(**ground)
    assert _moved(_fill(both), _fill(_dense(**ground))) <= Z_DRIFT
    assert both.pair_reuse_served["copied_pairs"] > 0


def test_the_plan_alone_is_served_when_the_tensor_does_not_fit():
    """No offset edge in play: the memory budget alone picks the chunked route."""
    s = _solver(offset_reuse=False, swept_mem_mb=2)
    assert _moved(_fill(s), _fill(_dense())) <= Z_DRIFT
    assert s.pair_reuse_served["route"] == "chunked"
    assert s.pair_reuse_served["offset_pairs"] == 0
    assert s.pair_reuse_served["copied_pairs"] > 0


def test_the_tensor_route_reports_what_it_served():
    s = _solver(offset_reuse=False)
    _fill(s)
    assert s.pair_reuse_served["route"] == "tensor"
    assert s.pair_reuse_served["copied_pairs"] > 0


@pytest.mark.parametrize("swept_mem_mb", [2, 256], ids=["chunked", "tensor"])
def test_a_swept_fill_is_served_from_its_own_prep(swept_mem_mb):
    s = _solver(offset_reuse=False, swept_mem_mb=swept_mem_mb)
    geom = s._build_geometry()
    prep = s._same_edge_prep(geom)
    _ki, _k, same_edge_k = next(iter(s._same_edge_prep_swept_chunks(prep, [s.k])))
    Z = _fill(s, same_edge_prep=same_edge_k)
    assert _moved(Z, _fill(_dense())) <= Z_DRIFT
    assert s.pair_reuse_served["copied_pairs"] > 0


def test_a_fill_of_chosen_rows_says_it_read_no_plan(plan_reads):
    s = _solver()
    _fill(s, rows=np.arange(21))
    assert s.pair_reuse_declined == "a fill of chosen rows does not read the hint"
    assert s.pair_reuse_served is None
    assert not plan_reads


def test_reuse_off_reads_no_plan_and_reports_nothing(plan_reads):
    s = _solver(pair_reuse=False, offset_reuse=True)
    _fill(s)
    assert s.pair_reuse_served is None
    assert s.pair_reuse_declined is None
    assert not plan_reads


SOMMERFELD = {
    "ground_z": 0.0,
    "ground_eps": (13.0, 0.005),
    "ground_model": "sommerfeld",
}


def _three(reuse, z=10.0, n=13, **kw):
    """Three identical parallel wires at height `z`. `offset_reuse` is held on,
    so on and off differ by the census alone."""
    wires = [np.array([[3.0 * i, -4.0, z], [3.0 * i, 4.0, z]]) for i in range(3)]
    kw.setdefault("offset_reuse", True)
    return BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=[[n] for _ in wires],
        feed_wire_index=0,
        wavelength=LAM,
        wire_radius=0.002,
        pair_reuse=reuse,
        **kw,
    )


def _sweep_k():
    return 2 * np.pi / LAM * np.array([1.0, 1.02, 1.04])


def test_the_buried_fill_names_the_plan_it_does_not_read(plan_reads):
    s = _three(True, z=-0.5, **SOMMERFELD)
    Z = _fill(s)
    assert s.pair_reuse_declined == "the buried fill does not read the hint"
    assert s.pair_reuse_served is None
    assert not plan_reads
    assert np.array_equal(Z, _fill(_three(False, z=-0.5, **SOMMERFELD)))


def test_the_buried_fill_says_nothing_with_reuse_off():
    s = _three(False, z=-0.5, **SOMMERFELD)
    _fill(s)
    assert s.pair_reuse_declined is None


def test_a_sweep_with_a_served_plan_leaves_the_batched_route(plan_reads):
    """The batched sweep reads no plan, so with one served the sweep runs per
    k and reads it (momwire#2172, `tests/test_bs2_sweep_reuse_2172.py`)."""
    s = _three(True)
    assert s._swept_batched_available()
    z = s.compute_impedance_swept(_sweep_k())
    assert s.pair_reuse_declined is None
    assert s.pair_reuse_served["copied_pairs"] > 0
    assert len(plan_reads) == 1
    ref = _three(False).compute_impedance_swept(_sweep_k())
    assert np.max(np.abs(z - ref) / np.abs(ref)) <= DRIFT


def test_the_port_sweep_leaves_it_too():
    s = _three(True)
    y = [np.asarray(p.y) for p in s.compute_port_solution_swept(_sweep_k())]
    assert s.pair_reuse_declined is None
    assert s.pair_reuse_served["copied_pairs"] > 0
    ref = _three(False).compute_port_solution_swept(_sweep_k())
    for a, b in zip(y, ref, strict=True):
        assert np.max(np.abs(a - np.asarray(b.y))) <= DRIFT * np.max(np.abs(b.y))


def test_the_batched_sweep_says_nothing_with_reuse_off():
    s = _three(False)
    s.compute_impedance_swept(_sweep_k())
    assert s.pair_reuse_declined is None


def test_a_sweep_frequency_by_frequency_reads_the_plan_once(plan_reads):
    """Over a finite ground the sweep is per-k fills: each serves the plan,
    and the census runs once for the sweep (the plan is k-free)."""
    s = _three(True, **SOMMERFELD)
    assert not s._swept_batched_available()
    s.compute_impedance_swept(_sweep_k())
    assert s.pair_reuse_declined is None
    assert s.pair_reuse_served["copied_pairs"] > 0
    assert len(plan_reads) == 1


H_REASON = "the H-matrix operator does not read the hint"
ARRAY_REASON = "the array-block operator does not read the hint"

#: Every way a caller reaches an accelerated operator, each returning arrays.
ACCELERATED = {
    "impedance": lambda s: s.compute_impedance()[0],
    "port": lambda s: s.compute_port_solution().y,
    "y": lambda s: s.compute_y_matrix(),
    "impedance_sweep": lambda s: s.compute_impedance_swept(_sweep_k()),
    "port_sweep": lambda s: np.stack(
        [np.asarray(p.y) for p in s.compute_port_solution_swept(_sweep_k())]
    ),
    "y_sweep": lambda s: s.compute_y_matrix_swept(_sweep_k()),
}


def _accelerated(cls, reuse):
    """A deck each class serves on its own operator. The size rule that hands
    a small H-matrix sweep to the dense route is switched off; `offset_reuse`
    is held on, so on and off differ by the census alone."""
    count, n, kw = (
        (4, 41, {"swept_dense_max_bases": 0}) if cls is HMatrixSolver else (6, 21, {})
    )
    wires, segs = _deck(count, n)
    return cls(
        wires=wires,
        n_per_edge_per_wire=segs,
        feed_wire_index=0,
        wavelength=LAM,
        wire_radius=0.002,
        pair_reuse=reuse,
        offset_reuse=True,
        **kw,
    )


@pytest.mark.parametrize("entry", sorted(ACCELERATED))
@pytest.mark.parametrize(
    ("cls", "reason"),
    [(HMatrixSolver, H_REASON), (ArrayBlockSolver, ARRAY_REASON)],
    ids=["hmatrix", "array_block"],
)
def test_an_accelerated_operator_names_the_plan_it_does_not_read(
    cls, reason, entry, plan_reads
):
    s = _accelerated(cls, True)
    out = ACCELERATED[entry](s)
    assert s._hmatrix is not None, "the accelerated operator did not run"
    assert s.pair_reuse_declined == reason
    assert s.pair_reuse_served is None
    assert not plan_reads
    assert np.array_equal(out, ACCELERATED[entry](_accelerated(cls, False)))


@pytest.mark.parametrize("entry", sorted(ACCELERATED))
@pytest.mark.parametrize("cls", [HMatrixSolver, ArrayBlockSolver])
def test_an_accelerated_operator_says_nothing_with_reuse_off(cls, entry):
    s = _accelerated(cls, False)
    ACCELERATED[entry](s)
    assert s._hmatrix is not None
    assert s.pair_reuse_declined is None


def test_a_cached_array_operator_still_names_the_plan():
    """The second solver finds the operator built: the build's own decline
    would never run."""
    _accelerated(ArrayBlockSolver, False).compute_port_solution()
    s = _accelerated(ArrayBlockSolver, True)
    s.compute_port_solution()
    assert s.pair_reuse_declined == ARRAY_REASON


def test_an_array_with_no_structure_names_the_operator_it_fell_back_to():
    """No two elements alike: the array-block solver runs the H-matrix."""
    wires = [
        np.array([[3.0 * i, -(3 + 0.4 * i), 10.0], [3.0 * i, 3 + 0.4 * i, 10.0]])
        for i in range(4)
    ]
    s = ArrayBlockSolver(
        wires=wires,
        n_per_edge_per_wire=[[21] for _ in wires],
        feed_wire_index=0,
        wavelength=LAM,
        wire_radius=0.002,
        pair_reuse=True,
    )
    s.compute_port_solution()
    assert s.pair_reuse_declined == H_REASON
