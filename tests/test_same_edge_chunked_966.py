"""The same-edge correction on the chunked route (momwire#966).

`_compute_Z_dense_chunked` row-chunks its main sweep to `swept_mem_mb`. The
SAME-EDGE CORRECTION under it was bounded only by the EDGE — which is the same
thing on every deck whose edges are short, and a different thing entirely on
one 4,001-segment wire, where a single edge IS the whole mesh. Measured there:
7,026 MB peak against a 0.26 GB answer, 27x, and degree 2 dead on an 8 GB
budget.

Two independent residencies, and the second was only found by RE-PROFILING
AFTER FIXING THE FIRST:

  * `_seg_seg_reg_geometry` built its (N_e·n_qp, N_e·n_qp) distance table
    through three live float64 arrays (`diff`, the squared temporary, the
    result). Now one buffer, in place. Bit-identical — same operations, same
    order, different destination.
  * The correction then held `A_st`, `A_reg`, the whole-edge `J_edge` and two
    `corr` temporaries, all (d+1, d+1, N_e, N_e) complex. `J_edge` and `corr`
    are now produced one observer window at a time, and R is dropped as soon
    as `A_reg` exists.

Fixing only the first buys 44 % at N=2000 and NOTHING at N=4001, because the
peak simply moves to the second. That is why both are here.

`A_st` and `A_reg` are still whole-edge: the C++ same-edge reg kernel checks
`R.shape == (N·n_qp, N·n_qp)` and refuses a rectangular window, so chunking
those needs a rectangular twin of that kernel rather than a call-site change.
That is #966's remaining half and is not smuggled in here.

Gates:

- G-966-1  chunked == dense BIT-IDENTICALLY on every deck whose edge fits one
           chunk — which is every ordinary deck. Chunking is not an
           approximation anyone pays for at normal sizes.
- G-966-2  where chunking really engages, chunked == dense to 1e-11 relative
           (measured 3.8e-12), with the engagement asserted so the bound is
           not being met by a route that never chunked.
- G-966-3  the dense route exceeds a memory cap the chunked route clears, one
           subprocess per arm — delete-the-line by construction, since the
           dense route IS the pre-#966 code.
- G-966-4  the in-place R build equals the readable three-array spelling
           exactly.
- G-966-5  a caller-supplied `same_edge_prep` survives being used twice; the
           R drop must never touch a prep this solver does not own.
- G-966-6  chunking is the DEFAULT, and a default solve on a long edge really
           does produce more than one window.

G-966-6 exists because the first five did not catch flipping the module
default to False: every one of them sets the switch explicitly, so the whole
file passed against a build where the fix was off. A gate on a behaviour
nobody exercises by default is a gate on nothing.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import momwire.bspline as _bs
from momwire._bspline_kernels import _seg_seg_reg_geometry
from momwire._quadrature import leggauss
from momwire.bspline import BSplineSolver

C0 = 299792458.0
LONG_WIRE = [np.array([(0.0, 0.0, 0.0), (0.0, 4000.0, 0.0)])]
LONG_LAM = C0 / 15e6


def _solve(wires, nspec, lam, radius, feeds, degree, *, chunked=True, mem=None):
    kw = {} if mem is None else {"swept_mem_mb": mem}
    old = _bs._SAME_EDGE_CORR_CHUNKED
    _bs._SAME_EDGE_CORR_CHUNKED = chunked
    try:
        s = BSplineSolver(
            wires=wires,
            n_per_edge_per_wire=nspec,
            wavelength=lam,
            wire_radius=radius,
            feeds=feeds,
            degree=degree,
            **kw,
        )
        return complex(s.compute_impedance()[0])
    finally:
        _bs._SAME_EDGE_CORR_CHUNKED = old


# Ordinary decks: short edges, so one chunk covers each edge and the chunked
# route runs the identical arithmetic.
ORDINARY = {
    "dipole": (
        [np.array([(0.0, 0.0, -2.0), (0.0, 0.0, 2.0)])],
        [[40]],
        8.0,
        0.01,
        [(0, 2.0, 1 + 0j)],
    ),
    "invvee": (
        [np.array([(-1.6, 0.0, 0.6), (0.0, 0.0, 1.9), (1.6, 0.0, 0.6)])],
        [[20, 20]],
        8.0,
        0.01,
        [(0, 2.06, 1 + 0j)],
    ),
    "two-wire": (
        [
            np.array([(0.0, 0.0, -2.0), (0.0, 0.0, 2.0)]),
            np.array([(1.2, 0.0, -2.0), (1.2, 0.0, 2.0)]),
        ],
        [[30], [30]],
        8.0,
        0.01,
        [(0, 2.0, 1 + 0j)],
    ),
}


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("name", sorted(ORDINARY))
def test_chunked_is_bit_identical_where_the_edge_fits_one_chunk(name, degree):
    """G-966-1. Not "close": the SAME answer, bit for bit.

    A deck whose every edge fits inside one window takes the same single
    producer call it always did, so anything but equality here means the
    rewrite changed the arithmetic rather than its residency.
    """
    wires, nspec, lam, rad, feeds = ORDINARY[name]
    zc = _solve(wires, nspec, lam, rad, feeds, degree, chunked=True)
    zd = _solve(wires, nspec, lam, rad, feeds, degree, chunked=False)
    assert zc == zd, f"{name} d={degree}: {zc!r} != {zd!r}"


@pytest.mark.parametrize("degree", [1, 2])
def test_chunked_matches_dense_where_chunking_actually_engages(degree):
    """G-966-2. A long single wire under a small budget: many windows.

    The tolerance covers ONE thing — the order in which window contributions
    are summed into Z. Measured worst 3.8e-12 relative on the 4,001-segment
    deck; pinned an order looser so a different BLAS can breathe, and tight
    enough that any real algebra change goes straight through it. Same
    two-tier armor `test_port_solution.py` documents for the batched routes.
    """
    n, mem = 1200, 4
    # The precondition. Without it this passes on a route that never chunked,
    # which is exactly the shape of gate this file exists because of.
    row_bytes = (degree + 1) ** 2 * n * 16
    n_windows = -(-n // max(1, int(mem * 1024 * 1024 // row_bytes)))
    assert n_windows > 1, (
        f"d={degree}: budget {mem} MB still fits the whole {n}-segment edge "
        f"in one window, so this test would compare the dense route to itself"
    )
    zc = _solve(
        LONG_WIRE,
        [[n]],
        LONG_LAM,
        5e-6,
        [(0, 2000.0, 1 + 0j)],
        degree,
        chunked=True,
        mem=mem,
    )
    zd = _solve(
        LONG_WIRE,
        [[n]],
        LONG_LAM,
        5e-6,
        [(0, 2000.0, 1 + 0j)],
        degree,
        chunked=False,
        mem=mem,
    )
    assert zc == pytest.approx(zd, rel=1e-11), f"{zc!r} vs {zd!r}"


# --- G-966-3: the memory gate -------------------------------------------
#
# One SUBPROCESS PER ARM, and peak RSS rather than `RLIMIT_AS`. Both choices
# are measurements, not taste:
#
#   * a process high-water mark is inherited twice over: two arms in ONE
#     interpreter make the second inherit the first's peak (measured — it
#     reported "no change" for a change worth 3.1 GB), and a forked child
#     inherits its PARENT's peak (measured — 1,610 MB in the parallel lane
#     for an arm that takes 900 MB alone). Hence a subprocess AND
#     `tracemalloc` rather than `ru_maxrss`.
#   * `RLIMIT_AS` caps VIRTUAL address space, which BLAS reserves lavishly:
#     at N=2000 both arms died at a 1.6 GB cap whose real peaks were 1.0 and
#     1.3 GB. It cannot separate these two arms on this box at all.
#
# The deck is chosen from the measured ladder so the margin is real in both
# directions: N=1500, d=2, 4 MB budget gives 1,589 MB dense against 904 MB
# chunked. The cap sits between them with ~25 % headroom on the passing side.
# TRACEMALLOC PEAK, NOT `ru_maxrss`, and RATIOS, NOT AN ABSOLUTE CAP. Both
# choices are repairs to earlier spellings of this gate that measured the box
# instead of the code:
#
#   1. `shipped < 1050 MB` on `ru_maxrss` passed standalone and FAILED in the
#      parallel slow lane at 1,620 MB against 900 MB alone. `subprocess`
#      forks, and a forked child INHERITS the parent's RSS high-water mark —
#      an xdist worker that has already run memory-heavy tests hands its own
#      peak to every arm. Pinning BLAS threads did not fix it, because the
#      thread count was never the cause.
#   2. `tracemalloc` counts this process's own Python/numpy allocations from
#      a standing start, so it cannot inherit anything. It tracks RSS closely
#      here (6,946 MB traced against 7,009 MB RSS on the 4,001-segment deck),
#      and it is exactly the allocation class #966 is about.
#
# Measured on this box, three arms of the same deck:
#
#   chunked + R dropped   810.9 MB    ships
#   chunked, R held      1011.8 MB    ratio 1.25 to shipped
#   dense whole edge     1494.8 MB    ratio 1.84 to shipped
_RSS_N, _RSS_DEGREE, _RSS_MEM = 1500, 2, 64
# Measured 1.84 and 1.25; pinned well under both so a different allocator has
# room, and well over 1.0 so neither can be met by two identical routes.
_DENSE_MIN_RATIO = 1.40
_HELD_R_MIN_RATIO = 1.12
# A loose absolute backstop: not the gate, but a catastrophic regression that
# somehow kept the ratios would still be caught.
_SHIPPED_MAX_MB = 1200.0

# Kept anyway: it costs nothing and makes the three arms identical in every
# respect but the switch under test.
_PIN = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

_ARM = textwrap.dedent(
    """
    import sys, tracemalloc
    import numpy as np
    import momwire.bspline as bs
    bs._SAME_EDGE_CORR_CHUNKED = sys.argv[1] == "1"
    bs._SAME_EDGE_DROP_R = sys.argv[2] == "1"
    tracemalloc.start()
    s = bs.BSplineSolver(
        wires=[np.array([(0.0, 0.0, 0.0), (0.0, 4000.0, 0.0)])],
        n_per_edge_per_wire=[[%d]],
        wavelength=%r,
        wire_radius=5e-6,
        feeds=[(0, 2000.0, 1 + 0j)],
        degree=%d,
        swept_mem_mb=%d,
    )
    s.compute_impedance()
    print(tracemalloc.get_traced_memory()[1] / 1e6)
    """
) % (_RSS_N, LONG_LAM, _RSS_DEGREE, _RSS_MEM)


def _arm_peak_mb(chunked, drop_r=True):
    done = subprocess.run(
        [sys.executable, "-c", _ARM, "1" if chunked else "0", "1" if drop_r else "0"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return float(done.stdout.strip().splitlines()[-1])


@pytest.mark.slow
def test_the_dense_route_blows_a_cap_the_chunked_route_clears():
    """G-966-3. Delete-the-line by construction, in BOTH directions: each
    arm is the shipped code with one of the two #966 changes reverted through
    its own switch, so neither can be removed without this failing.

    The held-R arm is here because reverting that alone passed every other
    gate in this file — it is a memory change and only a memory gate can see
    it."""
    chunked = _arm_peak_mb(True)
    assert chunked < _SHIPPED_MAX_MB, (
        f"the shipped route took {chunked:.0f} MB — far over anything measured "
        f"for it, so something regressed outside this gate's ratios"
    )
    for label, peak, floor in (
        ("dense whole-edge correction", _arm_peak_mb(False), _DENSE_MIN_RATIO),
        (
            "R held to the end of the loop",
            _arm_peak_mb(True, drop_r=False),
            _HELD_R_MIN_RATIO,
        ),
    ):
        ratio = peak / chunked
        assert ratio > floor, (
            f"reverting '{label}' cost {peak:.0f} MB against the shipped "
            f"route's {chunked:.0f} MB — a ratio of {ratio:.2f}, under the "
            f"{floor:.2f} this gate needs to tell the two apart. The deck or "
            f"budget needs re-picking from the ladder in the issue."
        )


def test_the_in_place_distance_table_equals_the_readable_spelling():
    """G-966-4. The rewrite is a destination change, not an algebra change."""
    seg = np.linspace(0.0, 40.0, 51)
    a, n_qp = 5e-6, 4
    geo = _seg_seg_reg_geometry(seg, a, 2, n_qp)

    gl_xi, gl_w = leggauss(n_qp)
    t01 = 0.5 * (gl_xi + 1.0)
    sl = seg[:-1]
    h = seg[1:] - sl
    s_flat = (sl[:, None] + t01[None, :] * h[:, None]).ravel()
    diff = s_flat[:, None] - s_flat[None, :]
    want = np.sqrt(diff * diff + a * a)
    assert np.array_equal(geo["R"], want)


def test_both_changes_are_the_default():
    """G-966-6, first half. The switches exist for gates, not for shipping."""
    assert _bs._SAME_EDGE_CORR_CHUNKED is True
    assert _bs._SAME_EDGE_DROP_R is True


def test_a_default_solve_on_a_long_edge_really_windows_the_correction(monkeypatch):
    """G-966-6, second half. Counts the producer calls rather than trusting
    the flag: the correction must ask for more than one window on an edge
    that does not fit the budget, with nothing in the test touching the
    switch."""
    calls = []
    real = _bs._seg_seg_full_moments_offedge

    def spy(seg_l_i, seg_r_i, seg_l_j, seg_r_j, *a, **kw):
        calls.append((len(seg_l_i), len(seg_l_j)))
        return real(seg_l_i, seg_r_i, seg_l_j, seg_r_j, *a, **kw)

    monkeypatch.setattr(_bs, "_seg_seg_full_moments_offedge", spy)
    n, mem = 1200, 4
    s = BSplineSolver(
        wires=LONG_WIRE,
        n_per_edge_per_wire=[[n]],
        wavelength=LONG_LAM,
        wire_radius=5e-6,
        feeds=[(0, 2000.0, 1 + 0j)],
        degree=2,
        swept_mem_mb=mem,
    )
    s.compute_impedance()
    # Same-edge calls are the square ones: columns span exactly the edge.
    same_edge = [c for c in calls if c[1] == n]
    assert len(same_edge) > 1, (
        f"the correction ran in {len(same_edge)} window(s) on a {n}-segment "
        f"edge at a {mem} MB budget — it is not chunking by default"
    )
    assert all(rows < n for rows, _ in same_edge), (
        f"a correction window covered the whole edge: {same_edge}"
    )


def test_a_caller_supplied_prep_is_not_consumed():
    """G-966-5. R is dropped only from a prep this call BUILT.

    A swept caller hands the same prep to every k in its sweep, so clearing
    it would make the second k rebuild the geometry — or fail outright. Two
    solves off one prep, and the second must equal the first.
    """
    s = BSplineSolver(
        wires=[np.array([(0.0, 0.0, -2.0), (0.0, 0.0, 2.0)])],
        n_per_edge_per_wire=[[24]],
        wavelength=8.0,
        wire_radius=0.01,
        feeds=[(0, 2.0, 1 + 0j)],
        degree=2,
    )
    geom = s._build_geometry()
    prep = s._same_edge_prep(geom)
    # `_same_edge_prep` hands back the k-independent ingredients; the per-k
    # list the solver actually consumes comes from the swept chunker, which
    # is the shape a sweep passes in.
    per_k = [chunk for chunk in s._same_edge_prep_swept_chunks(prep, np.array([s.k]))]
    _ki, _k, same_edge_k = per_k[0]
    first = complex(s.compute_impedance(same_edge_prep=same_edge_k)[0])
    second = complex(s.compute_impedance(same_edge_prep=same_edge_k)[0])
    assert first == second, f"{first!r} then {second!r}"
