"""The razor fill's C++ moment kernel (momwire#742) against its own reference.

`RazorSolver`'s segment-moment fill was serial pure NumPy and retained the
whole `(n_obs, n_seg, n_qp_source)` distance table — 52x the size of the
matrix it was building at 3201 segments, and flat serial. `_accel_razor.cpp`
fuses the prepare/replay halves into one tiled OpenMP kernel that forms R a
scalar at a time; `razor._FusedMoments` is what `_seg_moments_prepare` returns
in its place and `_seg_moments_from_prepared`'s `isinstance` is the whole
dispatch.

What this module gates:

  * **the paths agree**, on every branch of the fill that funnels through
    those two methods — both quadrature lanes, both kernels, all three
    grounds, ground contact, loading, mixed radii, junctions, and the swept
    prepare/replay split. NOT bitwise: the kernel's per-pair reduction is not
    `np.einsum`'s, and the repo's standard is never to pin cross-build bit
    equality. Measured on this box across the ten configurations below, the
    worst deviations are

        max|dZ| / max|Z|      4.8e-16   (about two ulps of the matrix scale)
        entrywise |dZ|/|Z|    1.5e-12   (near-null entries, |Z| ~ 1e-3 ohm)
        solved |dZin|/|Zin|   6.3e-14

    and the bars below sit a decade or more above those, which is headroom
    for a different libm and a different FMA contraction on CI hardware
    rather than slack in the claim.

  * **the kernel actually runs.** The #822 lesson: a disabled-path probe that
    cannot tell the paths apart gates nothing. `test_the_kernel_runs_...`
    counts kernel entries during a real solve and the forced-off twin asserts
    the count is zero AND that the prepared object changed type — two
    independent tells, so a dispatch that silently stopped dispatching fails
    here instead of passing quietly at 5x the wall time.

  * **the transient is gone**, structurally rather than by RSS: the prepared
    token's arrays are O(n_obs + n_seg) where the chunk list's are
    O(n_obs * n_seg * n_qp_source). Measured end to end at 1601 segments over
    a reflection-coefficient ground, peak RSS falls 2,102 MB -> 438 MB.

  * **cancellation reaches Python as `SolveAborted`**, from inside the kernel
    and with the Python checkpoints neutralised, the same way
    `test_sommerfeld_accel.py` proves it for the grid fill.

There are deliberately NO timing assertions here. The wall numbers live in
the issue; an allocator-variance perf gate in the PR lane does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import CancelToken, RazorSolver, SolveAborted
from momwire import razor as _razor

WL = 22.0

# Read ONCE at import, before any fixture can move it: `_FORCE_NUMPY` is
# seeded from `MOMWIRE_RAZOR_FORCE_NUMPY`, and a whole run under that switch
# is the fallback-equivalence lane — where the kernel-specific gates below
# have nothing to say and must skip rather than fail.
needs_accel = pytest.mark.skipif(
    not _razor._use_razor_fill_accel(),
    reason="razor fill accelerator not built, or forced off for this run",
)

# Ten configurations, chosen to enter every branch that reaches
# `_seg_moments_prepare` / `_seg_moments_from_prepared`: the two observer sets
# (path points and centroids), the two quadrature lanes, the reduced and
# extended kernels, the real and mirrored source sets, the unweighted /
# weighted / composing image blocks, a grounded (contact) row, a loaded fill
# and a multi-wire junction with a radius step.
_FREE = dict(
    wires=[[(0.0, 0.0, -5.0), (0.0, 0.0, 5.0)]],
    nsegs=41,
    wire_radius=0.005,
    wavelength=WL,
    feed_arclength=5.0,
)
_HIGH = dict(_FREE, wires=[[(0.0, -5.0, 6.0), (0.0, 5.0, 6.0)]])

CONFIGS = {
    "free-space": _FREE,
    "nec5-lane": dict(_FREE, nec5_quadrature=True),
    "extended-kernel": dict(_FREE, extended_kernel=True),
    "extended-kernel-nec5": dict(_FREE, extended_kernel=True, nec5_quadrature=True),
    "pec-ground": dict(_HIGH, ground_z=0.0),
    "refl-coef-ground": dict(_HIGH, ground_z=0.0, ground_eps=(13.0, 5e-3)),
    "sommerfeld-ground": dict(
        _HIGH, ground_z=0.0, ground_eps=(13.0, 5e-3), ground_model="sommerfeld"
    ),
    "ground-contact": dict(
        _FREE,
        wires=[[(0.0, 0.0, 0.0), (0.0, 0.0, 5.0)]],
        feed_arclength=0.5,
        ground_z=0.0,
    ),
    "wire-loading": dict(_FREE, wire_conductivity=5.8e7),
    "junction-radius-step": dict(
        _FREE,
        wires=[
            [(0.0, 0.0, -5.0), (0.0, 0.0, 0.0)],
            [(0.0, 0.0, 0.0), (2.0, 0.0, 4.0)],
        ],
        wire_radius=[0.005, 0.02],
        feed_wire_index=0,
        feed_arclength=5.0,
    ),
}

# See the module docstring for the measured deviations these sit above.
Z_SCALE_BAR = 1e-14  # max|dZ| / max|Z|, measured worst 4.8e-16
ZIN_REL_BAR = 1e-12  # |dZin| / |Zin|,   measured worst 6.3e-14


def _both_ways(monkeypatch, kw, fn):
    """Run `fn(solver)` once on each machine, returning (numpy, accel)."""
    out = []
    for force in (True, False):
        monkeypatch.setattr(_razor, "_FORCE_NUMPY", force)
        out.append(fn(RazorSolver(**kw)))
    return out


def _z_deviation(z_ref, z_new):
    return float(np.abs(z_ref - z_new).max() / np.abs(z_ref).max())


# ==========================================================================
# The kernel is present, and it is the one that ran
# ==========================================================================
def test_this_build_carries_the_kernel():
    """An adversarial gate on the BUILD, not on the code path.

    `_HAVE_RAZOR_FILL_ACCEL` reads the kernel's own capability symbol, never
    a shared one: a .so from an earlier arc exports every other section's
    entries, and a shared flag would advertise a contract it cannot serve.
    Skipped only where the extension genuinely is not built.
    """
    if _razor._acc is None:
        pytest.skip("accelerator extension not built")
    assert _razor._HAVE_RAZOR_FILL_ACCEL


@needs_accel
def test_the_kernel_runs_and_forcing_numpy_stops_it(monkeypatch):
    """The disabled-path probe, with two independent tells.

    A counter alone would pass on a build whose dispatch had silently
    reverted (the count would be zero both ways and the assertion would be
    on the wrong side); a type check alone would pass on a dispatch that
    built the token and then never called the kernel. Both are asserted, in
    both directions.
    """
    calls = []
    original = _razor._acc.razor_seg_moments

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_razor._acc, "razor_seg_moments", spy)

    s = RazorSolver(**CONFIGS["refl-coef-ground"])
    geom = s._build_geometry()
    prepared = s._assemble_Z_prepare(geom)
    assert isinstance(prepared["t2_chunks"], _razor._FusedMoments)
    s._assemble_Z_from_prepared(geom, prepared, s.k, s.omega)
    assert calls, "the accelerated dispatch never reached the kernel"
    n_accel = len(calls)

    calls.clear()
    monkeypatch.setattr(_razor, "_FORCE_NUMPY", True)
    s = RazorSolver(**CONFIGS["refl-coef-ground"])
    geom = s._build_geometry()
    prepared = s._assemble_Z_prepare(geom)
    assert isinstance(prepared["t2_chunks"], list)
    s._assemble_Z_from_prepared(geom, prepared, s.k, s.omega)
    assert not calls, f"the forced-off fill still entered the kernel {calls} times"
    assert n_accel > 0


@needs_accel
def test_the_environment_switch_is_read_at_import(monkeypatch):
    """`MOMWIRE_RAZOR_FORCE_NUMPY` is the whole-run switch; the module global
    it seeds is the per-test one. Spelled the way `_near_interface` and
    `_sommerfeld_below` spell theirs, so a bisect has one habit and not
    three."""
    assert _razor._use_razor_fill_accel()
    monkeypatch.setattr(_razor, "_FORCE_NUMPY", True)
    assert not _razor._use_razor_fill_accel()


# ==========================================================================
# Agreement, branch by branch
# ==========================================================================
@needs_accel
@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_the_two_fills_agree_on_the_matrix(monkeypatch, name):
    kw = CONFIGS[name]
    z_ref, z_new = _both_ways(
        monkeypatch, kw, lambda s: s._assemble_Z(s._build_geometry(), s.k)
    )
    assert z_ref.shape == z_new.shape
    assert z_ref.dtype == z_new.dtype
    dev = _z_deviation(z_ref, z_new)
    assert dev < Z_SCALE_BAR, f"{name}: max|dZ|/max|Z| = {dev:.3e}"


@needs_accel
@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_the_two_fills_agree_on_the_solved_impedance(monkeypatch, name):
    kw = CONFIGS[name]
    ref, new = _both_ways(monkeypatch, kw, lambda s: s.compute_impedance())
    z_ref, i_ref = ref
    z_new, i_new = new
    rel = abs(z_new - z_ref) / abs(z_ref)
    assert rel < ZIN_REL_BAR, f"{name}: |dZin|/|Zin| = {rel:.3e} ({z_ref} vs {z_new})"
    # The current column too: a fill defect that cancels in the driving-point
    # ratio would still move the distribution.
    cur = float(np.abs(i_new - i_ref).max() / np.abs(i_ref).max())
    assert cur < ZIN_REL_BAR, f"{name}: max|dI|/max|I| = {cur:.3e}"


@needs_accel
def test_the_swept_replay_agrees(monkeypatch):
    """The prepare/replay split is where the kernel changes SCHEDULE, not just
    speed: the reference path caches R across the sweep and the kernel
    rebuilds the statics at every k. The answers must not know that."""
    ks = 2.0 * np.pi / np.array([21.0, 22.0, 23.0])
    ref, new = _both_ways(
        monkeypatch,
        CONFIGS["refl-coef-ground"],
        lambda s: np.array(s.compute_impedance_swept(ks)[0]),
    )
    rel = float(np.abs(new - ref).max() / np.abs(ref).max())
    assert rel < ZIN_REL_BAR, f"swept |dZin|/|Zin| = {rel:.3e}"


@needs_accel
def test_need_m1_false_returns_none_on_both_paths(monkeypatch):
    """The scalar-potential call asks for M0 only. The kernel spells "no M1"
    as a (0, 0) array across the seam and `_FusedMoments.evaluate` turns it
    back into the `None` the reference path returns, so the two callers in
    `_assemble_Z_source_block` need no branch."""
    kw = CONFIGS["free-space"]
    out = []
    for force in (True, False):
        monkeypatch.setattr(_razor, "_FORCE_NUMPY", force)
        s = RazorSolver(**kw)
        geom = s._build_geometry()
        cent = geom["seg_p0"] + 0.5 * geom["seg_h"][:, None] * geom["seg_t"]
        out.append(s._seg_moments(cent, geom, s.k, need_m1=False))
    (m0_ref, m1_ref), (m0_new, m1_new) = out
    assert m1_ref is None and m1_new is None
    assert _z_deviation(m0_ref, m0_new) < Z_SCALE_BAR


# ==========================================================================
# The transient — the point of the exercise
# ==========================================================================
@needs_accel
def test_the_prepared_token_holds_no_n_squared_table(monkeypatch):
    """Structural memory gate.

    The reference path's chunk list retains R, m0s and m1s for EVERY chunk at
    once, so its residency is O(n_obs * n_seg * n_qp_source) whatever the
    chunk budget is — that is the 52x-the-matrix peak the issue measured, and
    a chunk-size tweak cannot touch it. The fused token holds the fill's
    ARGUMENTS instead, which is O(n_obs + n_seg). Compared as bytes at one
    mesh so the gate is deterministic; the end-to-end RSS number is in the
    module docstring.
    """
    kw = dict(CONFIGS["free-space"], nsegs=101)

    def prepared_bytes(s):
        geom = s._build_geometry()
        cent = geom["seg_p0"] + 0.5 * geom["seg_h"][:, None] * geom["seg_t"]
        return s._seg_moments_prepare(cent, geom, s._kernel_radius(geom))

    monkeypatch.setattr(_razor, "_FORCE_NUMPY", True)
    chunks = prepared_bytes(RazorSolver(**kw))
    ref_bytes = sum(
        a.nbytes for lo, hi, R, m0s, m1s, ekc in chunks for a in (R, m0s, m1s)
    )
    monkeypatch.setattr(_razor, "_FORCE_NUMPY", False)
    token = prepared_bytes(RazorSolver(**kw))
    new_bytes = sum(
        getattr(token, slot).nbytes for slot in _razor._FusedMoments.__slots__
    )
    assert new_bytes * 20 < ref_bytes, (new_bytes, ref_bytes)


# ==========================================================================
# Cancellation
# ==========================================================================
@needs_accel
def test_a_tripped_flag_raises_solve_aborted_from_the_kernel():
    """Straight at the seam: a pre-tripped flag, one kernel call, and the
    `AcceleratorAborted` the drain pattern throws must arrive as
    `SolveAborted` — which is what listing the kernel in
    `_accel._CANCELLABLE_KERNELS` buys."""
    token = CancelToken()
    token.cancel()
    s = RazorSolver(**CONFIGS["free-space"])
    geom = s._build_geometry()
    cent = geom["seg_p0"] + 0.5 * geom["seg_h"][:, None] * geom["seg_t"]
    n_seg = geom["seg_h"].size
    a = np.full(n_seg, 0.005)
    empty_i = np.empty(0, dtype=np.int64)
    empty_f = np.empty(0)
    xg, wg = _razor.leggauss(s.n_qp_source)
    with pytest.raises(SolveAborted):
        _razor._acc.razor_seg_moments(
            cent,
            geom["seg_p0"],
            geom["seg_t"],
            geom["seg_h"],
            a,
            xg,
            wg,
            s.k,
            True,
            empty_i,
            empty_i,
            empty_f,
            token.ptr,
        )


@needs_accel
def test_a_razor_solve_cancels_through_the_cpp_poll_alone():
    """Neutralise the Python checkpoints so the only thing that can observe
    the tripped token is the kernel's own between-tiles poll — the razor twin
    of `test_cancel.py::test_cpp_polling_aborts_without_python_checkpoints`.
    With the fill fused, this is the ONLY poll left in a razor fill, which is
    why it is gated rather than assumed."""
    token = CancelToken()
    # Tripped AFTER construction: `RazorSolver.__init__` takes a Python
    # checkpoint of its own, which would raise before a fill ever started and
    # prove nothing about the kernel.
    s = RazorSolver(**dict(CONFIGS["free-space"], cancel=token))
    token.cancel()
    s._checkpoint = lambda: None
    with pytest.raises(SolveAborted):
        s.compute_impedance()
