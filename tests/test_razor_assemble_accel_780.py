"""The razor fill's C++ T1 assembly (momwire#780) against its numpy reference.

`#742` moved the segment MOMENTS into C++ and left the ASSEMBLY around them in
numpy: gather both wings' columns out of (M0, M1), apply the falling-wing
correction, contract each testing-path point's tangent with the source
tangents, weight, and sum over the path points. numpy spells that with an
`(n_obs, n_basis)` complex intermediate that is built and immediately reduced —
~32 MB per chunk at `n_path = 64` on a 200-segment deck. Measured before this
kernel, that assembly was **52-76%** of razor's wall time, on both quadrature
lanes and both grounds, and proportionally worst on `nec5_quadrature` (72.6% at
N=400 over ground), which is the interactive lane.

ONE KERNEL SERVES BOTH LANES, which is the structural claim worth gating.
`_path_nodes_per_wing` is, in its own words, "the one place the two lanes
differ"; `_testing_paths` hands the fill `(pts, tans, wts)` shape-agnostically;
and `_assemble_Z_source_block` contains no branch on `nec5_quadrature` at all.
So `n_path` is a loop bound inside the kernel — 2 for the wing-centroid
trapezoid, `2*n_qp_path` for Gauss-Legendre — and the two lanes cannot drift
apart because they are one implementation. `test_both_lanes_...` drives both
through it in one process.

What this module gates:

  * **the paths agree** on every configuration the kernel serves (both lanes,
    free space and the PEC fold, two mesh sizes). NOT bitwise, deliberately:
    the repo's standard is never to pin cross-build bit equality — the same
    standard `test_razor_fill_accel_742` states, and the standard momwire#781
    was the cost of ignoring. It happens that this kernel and numpy currently
    agree to 0.0e+00 on this box, because numpy's `reshape(...).sum(axis=1)`
    reduces a STRIDED axis and so does not take its pairwise path either. That
    is an observation about one build, not a contract: a different FMA
    contraction or a vectorized reduction on other hardware may reassociate,
    and the bars below sit far above the measured agreement to leave room for
    exactly that.

  * **the kernel actually runs.** The #822 lesson, and momwire#781's:
    a gate that cannot tell the two paths apart gates nothing. The agreement
    tests below would pass just as happily if the dispatch silently stopped
    dispatching, so `test_the_kernel_actually_runs` counts entries during a
    real solve and asserts the forced-off twin counts zero.

WHICH SIDE OF `solve` AN EXACT ASSERTION BELONGS ON (momwire#809)
----------------------------------------------------------------
The standard above is about builds. There is a second one, about the solve,
and it is the reason this docstring is where the next person looks:

    An `==` on a FILL is a claim. An `==` downstream of `solve` is a
    lottery ticket.

The razor fill is byte-repeatable -- same build, same process, same matrix at
any thread count. The solved impedance is not: at N=199 a dipole's Zin moves
7.5e-12 relative, 37616 ulp, between `OMP_NUM_THREADS=1` and `8`, with the
fill hash identical at both. BLAS picks blocking and kernel by thread count,
the reassociation follows, and the condition number does the rest. Nothing
there is a bug; it is a property of every dense direct solve.

So an assertion past `solve` holds only while two BLAS calls choose the same
path -- until a runner has a different core count, a different library
(OpenBLAS on Linux, Accelerate on macOS), or a different xdist pin. It can
also pass by LUCK when the two sides' fills differ slightly and the LU rounds
the difference away, which is what hid momwire#807 for months.

The test is not "is this `==` passing" but "do the two sides build a
bit-identical fill". If yes, the two solves run identical instructions on
identical data and `==` is structural -- keep it exact, and say so next to it.
If no, move the assertion onto the fill, which is the claim it was really
making, and keep the impedance under a measured tolerance beneath it.

momwire#809 swept all 33 such sites in this suite and every one measured
bit-identical, so none moved; each carries a one-line verdict where it sits.
Measure a new one the same way -- hash the matrix handed to
`scipy.linalg.solve` / `lu_factor` (BEFORE the call: `overwrite_a=True`
destroys it) on both sides -- rather than assuming, in either direction.

  * **the scope boundary holds.** The kernel serves the unweighted integrand
    only, i.e. `w_A_fn is None`. That boundary does not fall where a first
    reading suggests: `_assemble_Z` calls the block twice and only the IMAGE
    call carries the ground, so the real-source block is unweighted even over
    soil and legitimately runs the kernel there. What must never happen is the
    WEIGHTED half reaching it — the Fresnel weights would be dropped silently,
    a wrong answer rather than a slow one — so
    `test_a_finite_ground_answer_is_unchanged` pins it by the answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import razor as _razor
from momwire.razor import RazorSolver

C0 = 299792458.0
LAM = C0 / 7.0e6

LANES = {"nec5": {"nec5_quadrature": True}, "gauss-legendre": {}}


def _dipole(n, ground=False, **kw):
    deck = dict(
        wires=[np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0 + LAM / 2]])],
        n_per_edge_per_wire=[[n]],
        wire_radius=0.005,
        wavelength=LAM,
        feeds=[(0, LAM / 4, 1 + 0j)],
    )
    if ground:
        deck["ground_z"] = 0.0
    deck.update(kw)
    return deck


def _solve(monkeypatch, deck, *, accel, **kw):
    """One solve with the assembly kernel on or off.

    Only `_HAVE_RAZOR_ASSEMBLE_ACCEL` is flipped, never `_FORCE_NUMPY`: that
    one also disables the #742 MOMENTS kernel, so a comparison against it
    measures both halves and attributes the whole difference to this one. The
    first draft of this module's benchmark made exactly that mistake and
    reported speedups up to 27x that were not this kernel's.
    """
    monkeypatch.setattr(_razor, "_HAVE_RAZOR_ASSEMBLE_ACCEL", accel)
    z, _ = RazorSolver(**deck, **kw).compute_impedance()
    return complex(z)


@pytest.mark.parametrize("lane", sorted(LANES))
@pytest.mark.parametrize("ground", [False, True], ids=["free", "pec"])
@pytest.mark.parametrize("n", [120, 240])
def test_both_lanes_agree_with_the_numpy_assembly(monkeypatch, lane, ground, n):
    """Both quadrature lanes, both grounds, through one kernel."""
    deck = _dipole(n, ground)
    kw = LANES[lane]
    z_np = _solve(monkeypatch, deck, accel=False, **kw)
    z_acc = _solve(monkeypatch, deck, accel=True, **kw)
    rel = abs(z_acc - z_np) / abs(z_np)
    assert rel < 1e-11, f"{lane}/{ground}/N={n}: Zin {z_acc} vs {z_np} (rel {rel:.3e})"


def test_the_kernel_actually_runs(monkeypatch):
    """The agreement tests above would pass if the dispatch died. This one
    fails instead — two tells, a nonzero count on and a zero count off."""
    if not _razor._HAVE_RAZOR_ASSEMBLE_ACCEL:
        pytest.skip("build carries no razor_assemble_780 kernel")
    from momwire import _accelerators as _acc

    calls = {"n": 0}
    real = _acc.razor_assemble_t1

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(_acc, "razor_assemble_t1", counting)
    monkeypatch.setattr(_razor._acc, "razor_assemble_t1", counting, raising=False)

    calls["n"] = 0
    _solve(monkeypatch, _dipole(120), accel=True)
    on = calls["n"]

    calls["n"] = 0
    _solve(monkeypatch, _dipole(120), accel=False)
    off = calls["n"]

    assert on > 0, "the assembly kernel never ran with the flag on"
    assert off == 0, f"the kernel ran {off} times with the flag off"


@pytest.mark.parametrize("model", ["refl-coef", "sommerfeld"])
def test_a_finite_ground_answer_is_unchanged(monkeypatch, model):
    """The scope boundary, gated by the answer rather than by a call count.

    The kernel serves the UNWEIGHTED integrand only. On a finite ground that
    boundary does not fall where a first reading suggests: `_assemble_Z` calls
    the block twice, and only the IMAGE call carries the ground
    (`razor.py:2705` passes none, `:2715` passes it). So the real-source block
    is unweighted even over soil and legitimately runs the kernel, while the
    image block's `w_A_fn` keeps numpy.

    An earlier version of this test asserted the kernel never runs at all here
    and failed for that reason — the premise was wrong, not the code. What
    actually needs pinning is that the weighted half never reaches the kernel,
    and the sharpest way to say that is the answer: if it did, the Fresnel
    weights would be silently dropped and Zin would move far more than 1e-11.
    """
    deck = _dipole(80, ground=True, ground_eps=(13.0, 0.005), ground_model=model)
    z_np = _solve(monkeypatch, deck, accel=False)
    z_acc = _solve(monkeypatch, deck, accel=True)
    rel = abs(z_acc - z_np) / abs(z_np)
    assert rel < 1e-11, f"{model}: Zin {z_acc} vs {z_np} (rel {rel:.3e})"
