"""The shared reduced-kernel segment moments (momwire#425).

`_axis_frame` / `_static_axis_moments` are the closed-form arithmetic any
axis-expanded row wants once it has chosen the reduced kernel — not a
formulation — so they live in `_kernel_moments` rather than inside one
solver's module, where `PulseSolver` had been reaching for them through
`RazorSolver`'s module-level privates.

Two claims are gated here:

1. **the move is a re-export, not a copy** — every consumer names the same
   function object, so a change to the arithmetic cannot reach one row and
   miss another;
2. **the radius may be per-segment or per-observer, and the scalar case is
   untouched** — `a` enters only as `a * a` added to ρ², so a uniform
   array in either spelling is BIT-IDENTICAL to the scalar it came from.
   That is the free bit gate the sharing audit predicted, and it is what
   lets `RazorSolver` carry per-wire radii into the same kernel.
"""

import numpy as np

from momwire import _kernel_moments, pulse, razor

RNG = np.random.default_rng(20260818)


def _frame_inputs(n_obs=7, n_seg=5):
    obs = RNG.normal(size=(n_obs, 3))
    seg_p0 = RNG.normal(size=(n_seg, 3))
    seg_t = RNG.normal(size=(n_seg, 3))
    seg_t /= np.linalg.norm(seg_t, axis=1)[:, None]
    seg_h = RNG.uniform(0.2, 1.5, size=n_seg)
    return obs, seg_p0, seg_t, seg_h


def test_every_consumer_names_the_same_function():
    assert razor._axis_frame is _kernel_moments._axis_frame
    assert razor._static_axis_moments is _kernel_moments._static_axis_moments
    # pulse.py still imports FROM razor (momwire#419); the re-export is what
    # keeps that working across the move, and migrating its import line is a
    # one-liner on pulse's own branch.
    assert pulse._axis_frame is _kernel_moments._axis_frame
    assert pulse._static_axis_moments is _kernel_moments._static_axis_moments


def test_uniform_radius_array_is_bit_identical_to_the_scalar():
    obs, seg_p0, seg_t, seg_h = _frame_inputs()
    a = 1.0262e-3
    u_ref, rho2_ref = _kernel_moments._axis_frame(obs, seg_p0, seg_t, a)

    per_source = np.full(seg_h.shape[0], a)  # (n_seg,)
    per_obs = np.full((obs.shape[0], 1), a)  # (n_obs, 1)
    for spelling in (per_source, per_obs):
        u, rho2 = _kernel_moments._axis_frame(obs, seg_p0, seg_t, spelling)
        assert np.array_equal(u, u_ref)
        assert np.array_equal(rho2, rho2_ref)
        m0, m1 = _kernel_moments._static_axis_moments(u, rho2, seg_h)
        m0_ref, m1_ref = _kernel_moments._static_axis_moments(u_ref, rho2_ref, seg_h)
        assert np.array_equal(m0, m0_ref)
        assert np.array_equal(m1, m1_ref)


def test_a_per_segment_radius_lands_on_the_source_axis():
    """A (n_seg,) radius must vary down the SOURCE axis and be constant
    across observers — the broadcast is the whole per-wire mechanism, and a
    transposed one would still have the right shape."""
    obs, seg_p0, seg_t, seg_h = _frame_inputs()
    a = np.array([1e-3, 2e-3, 3e-3, 4e-3, 5e-3])
    _u, rho2 = _kernel_moments._axis_frame(obs, seg_p0, seg_t, a)
    _u0, rho2_0 = _kernel_moments._axis_frame(obs, seg_p0, seg_t, 0.0)
    delta = rho2 - rho2_0
    for s in range(seg_h.shape[0]):
        assert np.allclose(delta[:, s], a[s] ** 2)


def test_a_per_observer_radius_lands_on_the_observer_axis():
    obs, seg_p0, seg_t, seg_h = _frame_inputs()
    a = np.linspace(1e-3, 7e-3, obs.shape[0])[:, None]
    _u, rho2 = _kernel_moments._axis_frame(obs, seg_p0, seg_t, a)
    _u0, rho2_0 = _kernel_moments._axis_frame(obs, seg_p0, seg_t, 0.0)
    delta = rho2 - rho2_0
    for p in range(obs.shape[0]):
        assert np.allclose(delta[p, :], a[p, 0] ** 2)


def test_static_moments_match_brute_force_quadrature():
    """The asinh/sqrt closed forms against a dense numerical ∫dτ/R and
    ∫τ dτ/R — the oracle that makes the move's bit-gate meaningful rather
    than merely self-consistent."""
    obs, seg_p0, seg_t, seg_h = _frame_inputs(n_obs=3, n_seg=4)
    a = 2.0e-3
    u_r, rho2 = _kernel_moments._axis_frame(obs, seg_p0, seg_t, a)
    m0, m1 = _kernel_moments._static_axis_moments(u_r, rho2, seg_h)
    for p in range(obs.shape[0]):
        for s in range(seg_h.shape[0]):
            tau = np.linspace(0.0, seg_h[s], 20001)
            r = np.sqrt((tau - u_r[p, s]) ** 2 + rho2[p, s])
            np.testing.assert_allclose(m0[p, s], np.trapezoid(1.0 / r, tau), rtol=1e-8)
            np.testing.assert_allclose(m1[p, s], np.trapezoid(tau / r, tau), rtol=1e-8)
