"""bspline's chunked fill serves the `pair_reuse` hint (momwire#2020).

`offset_reuse`, a part of `pair_reuse`, sends any deck with a uniform edge of 12
or more segments down `_compute_Z_dense_chunked`, and that route used to read no
plan: the hint's copies never ran and nothing said so. Now the two compose:

- the blocks between edges are copied from held source blocks, counted where
  they are served and held against what the kernel was actually asked for;
- a congruent offset edge shares its source's table of distinct moments;
- the answer stays inside the two features' declared drift;
- the held blocks stay inside the fill's memory budget, and every way a hint
  goes unread is named in `pair_reuse_declined`, on the routes that read no
  plan (the buried fill, the batched sweep, the H-matrix and array-block
  operators) as well: those keep the no-hint fill's bits.
"""

from __future__ import annotations

import numpy as np
import pytest

import momwire.bspline as bs
from momwire import _features
from momwire.array_block import ArrayBlockSolver
from momwire.bspline import BSplineSolver
from momwire.hmatrix import HMatrixSolver

# An experimental feature (`momwire._features`, reach 1).
pytestmark = pytest.mark.reach(1)

LAM = 21.26

#: |dZin|/|Zin| the two features declare, added: each moves the answer alone.
DRIFT = (
    _features.FEATURES["pair_reuse"].drift + _features.FEATURES["offset_reuse"].drift
)
#: Both routes sum the same moments, so Z itself moves at the offset route's
#: declared roundoff.
Z_DRIFT = _features.FEATURES["offset_reuse"].drift


def _deck(count=8, n=21):
    """`count` identical parallel dipoles at one spacing."""
    L = 0.47 * LAM
    wires = [
        np.array([[3.0 * i, -L / 2, 10.0], [3.0 * i, L / 2, 10.0]])
        for i in range(count)
    ]
    return wires, [[n] for _ in wires]


def _hint(count):
    """The front end's shape: per class, ``((wire, edge), (wire, edge))`` with
    the representative first."""
    own = [((i, 0), (i, 0)) for i in range(count)]
    gaps = [
        [((i, 0), (i + gap, 0)) for i in range(count - gap)]
        for gap in range(1, count - 1)
    ]
    return [own, *gaps]


def _solver(count=8, n=21, hint=True, **kw):
    wires, segs = _deck(count, n)
    if hint:
        kw.setdefault("pair_reuse", _hint(count))
    return BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=segs,
        feed_wire_index=0,
        wavelength=LAM,
        wire_radius=0.002,
        **kw,
    )


def _dense(count=8, n=21, **kw):
    return _solver(count, n, hint=False, pair_reuse=False, offset_reuse=False, **kw)


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


def test_the_chunked_route_copies_what_the_hint_offers(kernel_pairs):
    """Eight dipoles, seven spacings: of the 56 ordered off-edge blocks, the
    14 of the first pair at each spacing are computed and the other 42 copied
    (the widest spacing has one member, so it is computed in both orders)."""
    s = _solver()
    assert s._offset_reuse, "offset_reuse follows pair_reuse"
    _fill(s)
    served = s.pair_reuse_served
    assert served["route"] == "chunked"
    assert s.pair_reuse_declined is None
    block = 21 * 21
    assert served["copied_blocks"] == 42
    assert served["copied_pairs"] == 42 * block
    assert served["computed_pairs"] == 14 * block
    # The kernel's own count, not the fill's report of itself.
    assert sum(kernel_pairs) == served["computed_pairs"]


def test_the_counts_reconcile_with_the_matrix():
    s = _solver()
    _fill(s)
    c = s.pair_reuse_served
    assert c["copied_pairs"] + c["computed_pairs"] + c["offset_pairs"] == c["pairs"]
    assert c["pairs"] == (8 * 21) ** 2


def test_a_congruent_offset_edge_shares_its_sources_table():
    s = _solver()
    _fill(s)
    c = s.pair_reuse_served
    assert c["tables_shared"] == 7
    assert c["offset_distinct"] == 2 * 21 - 1
    assert c["offset_pairs"] == 8 * 21 * 21


def test_a_hint_of_own_blocks_alone_still_reconciles():
    """Only the tables are shared: nothing is held, and every other pair is
    computed."""
    s = _solver(pair_reuse=_hint(8)[:1])
    _fill(s)
    c = s.pair_reuse_served
    assert c["copied_blocks"] == 0
    assert c["tables_shared"] == 7
    assert c["computed_pairs"] + c["offset_pairs"] == c["pairs"]


def test_the_copies_reach_the_matrix(monkeypatch):
    """Spoil the held blocks: a route that did not read them would not notice."""
    ref = _fill(_dense())
    real = bs._ChunkedReuse.fill

    def spoiled(self, moments, checkpoint):
        out = real(self, moments, checkpoint)
        for J in self.held.values():
            J *= 2.0
        return out

    monkeypatch.setattr(bs._ChunkedReuse, "fill", spoiled)
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


def test_the_hint_alone_is_served_when_the_tensor_does_not_fit():
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


def test_a_swept_fill_is_served_from_its_own_prep():
    s = _solver(offset_reuse=False, swept_mem_mb=2)
    geom = s._build_geometry()
    prep = s._same_edge_prep(geom)
    _ki, _k, same_edge_k = next(iter(s._same_edge_prep_swept_chunks(prep, [s.k])))
    Z = _fill(s, same_edge_prep=same_edge_k)
    assert _moved(Z, _fill(_dense())) <= Z_DRIFT
    assert s.pair_reuse_served["copied_pairs"] > 0


def test_held_blocks_are_cut_to_the_budget_and_counted():
    """1 MB gives the held blocks 768 kB, and each of the six repeated spacings
    holds 180 kB: four fit."""
    s = _solver(n=25, swept_mem_mb=1)
    assert _moved(_fill(s), _fill(_dense(n=25))) <= Z_DRIFT
    c = s.pair_reuse_served
    assert c["cut_blocks"] > 0
    assert c["copied_blocks"] > 0
    assert s.pair_reuse_declined is None
    assert c["copied_pairs"] + c["computed_pairs"] + c["offset_pairs"] == c["pairs"]


def test_held_blocks_never_pass_the_budget(monkeypatch):
    held = []
    real = bs._ChunkedReuse.fill

    def spy(self, moments, checkpoint):
        out = real(self, moments, checkpoint)
        held.append(sum(J.nbytes for J in self.held.values()))
        return out

    monkeypatch.setattr(bs._ChunkedReuse, "fill", spy)
    _fill(_solver(n=25, swept_mem_mb=1))
    assert held
    assert 0 < held[0] <= 0.75 * (1 << 20)


def test_a_source_too_big_to_hold_is_declined_by_name():
    """One 61-segment block pair is over 1 MB, past the 768 kB held budget."""
    s = _solver(count=3, n=61, swept_mem_mb=1)
    assert _moved(_fill(s), _fill(_dense(count=3, n=61))) <= Z_DRIFT
    assert s.pair_reuse_declined == (
        "the source blocks do not fit the fill's memory budget"
    )


def test_a_fill_of_chosen_rows_says_it_read_no_hint():
    s = _solver()
    _fill(s, rows=np.arange(21))
    assert s.pair_reuse_declined == "a fill of chosen rows does not read the hint"
    assert s.pair_reuse_served is None


def test_no_hint_reads_no_plan_and_reports_nothing(monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("a fill with no hint built a held-block plan")

    monkeypatch.setattr(bs, "_ChunkedReuse", refuse)
    s = _solver(hint=False, offset_reuse=True)
    _fill(s)
    assert s.pair_reuse_served is None
    assert s.pair_reuse_declined is None


SOMMERFELD = {
    "ground_z": 0.0,
    "ground_eps": (13.0, 0.005),
    "ground_model": "sommerfeld",
}
THREE = [
    [((i, 0), (i, 0)) for i in range(3)],
    [((0, 0), (1, 0)), ((1, 0), (2, 0))],
]


def _three(reuse, z=10.0, n=13, **kw):
    """Three identical parallel wires at height `z`."""
    wires = [np.array([[3.0 * i, -4.0, z], [3.0 * i, 4.0, z]]) for i in range(3)]
    return BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=[[n] for _ in wires],
        feed_wire_index=0,
        wavelength=LAM,
        wire_radius=0.002,
        pair_reuse=reuse,
        **kw,
    )


@pytest.fixture
def plan_reads(monkeypatch):
    """How many times a fill asked for the hint's plan."""
    reads = []
    real = BSplineSolver._pair_reuse_plan

    def spy(self, geom):
        reads.append(1)
        return real(self, geom)

    monkeypatch.setattr(BSplineSolver, "_pair_reuse_plan", spy)
    return reads


def _sweep_k():
    return 2 * np.pi / LAM * np.array([1.0, 1.02, 1.04])


def test_the_buried_fill_names_the_hint_it_does_not_read(plan_reads):
    s = _three(THREE, z=-0.5, **SOMMERFELD)
    Z = _fill(s)
    assert s.pair_reuse_declined == "the buried fill does not read the hint"
    assert s.pair_reuse_served is None
    assert not plan_reads
    # `True` is the same switches with no hint: the fill's own bits.
    assert np.array_equal(Z, _fill(_three(True, z=-0.5, **SOMMERFELD)))


def test_the_buried_fill_says_nothing_without_a_hint():
    s = _three(True, z=-0.5, **SOMMERFELD)
    _fill(s)
    assert s.pair_reuse_declined is None


def test_the_batched_sweep_names_the_hint_it_does_not_read(plan_reads):
    s = _three(THREE)
    assert s._swept_batched_available()
    z = s.compute_impedance_swept(_sweep_k())
    assert s.pair_reuse_declined == "the batched sweep does not read the hint"
    assert s.pair_reuse_served is None
    assert not plan_reads
    assert np.array_equal(z, _three(True).compute_impedance_swept(_sweep_k()))


def test_the_batched_port_sweep_names_it_too():
    s = _three(THREE)
    y = [np.asarray(p.y) for p in s.compute_port_solution_swept(_sweep_k())]
    assert s.pair_reuse_declined == "the batched sweep does not read the hint"
    ref = _three(True).compute_port_solution_swept(_sweep_k())
    assert all(np.array_equal(a, np.asarray(b.y)) for a, b in zip(y, ref, strict=True))


def test_the_batched_sweep_says_nothing_without_a_hint():
    s = _three(True)
    s.compute_impedance_swept(_sweep_k())
    assert s.pair_reuse_declined is None


def test_a_sweep_frequency_by_frequency_still_reads_the_hint():
    """Over a finite ground the sweep is per-k fills, which do serve it."""
    s = _three(THREE, **SOMMERFELD)
    assert not s._swept_batched_available()
    s.compute_impedance_swept(_sweep_k())
    assert s.pair_reuse_declined is None
    assert s.pair_reuse_served["copied_blocks"] == 2


def test_a_chunked_fill_that_copies_nothing_says_so():
    """Own blocks of edges too short for the offset route, on a deck whose
    tensor does not fit: the plan is read, and none of it can be served."""
    s = _solver(count=8, n=11, pair_reuse=_hint(8)[:1], swept_mem_mb=1)
    Z = _fill(s)
    assert s.pair_reuse_declined == "the chunked fill copies nothing this hint offers"
    assert s.pair_reuse_served is None
    assert np.array_equal(
        Z, _fill(_solver(count=8, n=11, hint=False, pair_reuse=True, swept_mem_mb=1))
    )


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
    a small H-matrix sweep to the dense route is switched off."""
    count, n, kw = (
        (4, 41, {"swept_dense_max_bases": 0}) if cls is HMatrixSolver else (6, 21, {})
    )
    wires, segs = _deck(count, n)
    hint = _hint(count) if reuse == "hint" else reuse
    return cls(
        wires=wires,
        n_per_edge_per_wire=segs,
        feed_wire_index=0,
        wavelength=LAM,
        wire_radius=0.002,
        pair_reuse=hint,
        **kw,
    )


@pytest.mark.parametrize("entry", sorted(ACCELERATED))
@pytest.mark.parametrize(
    ("cls", "reason"),
    [(HMatrixSolver, H_REASON), (ArrayBlockSolver, ARRAY_REASON)],
    ids=["hmatrix", "array_block"],
)
def test_an_accelerated_operator_names_the_hint_it_does_not_read(
    cls, reason, entry, plan_reads
):
    s = _accelerated(cls, "hint")
    out = ACCELERATED[entry](s)
    assert s._hmatrix is not None, "the accelerated operator did not run"
    assert s.pair_reuse_declined == reason
    assert s.pair_reuse_served is None
    assert not plan_reads
    assert np.array_equal(out, ACCELERATED[entry](_accelerated(cls, True)))


@pytest.mark.parametrize("entry", sorted(ACCELERATED))
@pytest.mark.parametrize("cls", [HMatrixSolver, ArrayBlockSolver])
def test_an_accelerated_operator_says_nothing_without_a_hint(cls, entry):
    s = _accelerated(cls, True)
    ACCELERATED[entry](s)
    assert s._hmatrix is not None
    assert s.pair_reuse_declined is None


def test_a_cached_array_operator_still_names_the_hint():
    """The second solver finds the operator built: the build's own decline
    would never run."""
    _accelerated(ArrayBlockSolver, True).compute_port_solution()
    s = _accelerated(ArrayBlockSolver, "hint")
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
        pair_reuse=[[((0, 0), (1, 0)), ((1, 0), (2, 0))]],
    )
    s.compute_port_solution()
    assert s.pair_reuse_declined == H_REASON
