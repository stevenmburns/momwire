"""C++ Sommerfeld kernel vs the pure-Python engine (perf plan Phase 3).

The cross-check below is also a Bessel-function validation: the Python
path evaluates the same contours through scipy's AMOS jv/hankel2, so
agreement at quadrature tolerance means the clean-room A&S series /
asymptotic implementations in _accelerators.cpp match scipy over the
domain the contours actually visit (|x| to ~110, arg in [-100, +45]
degrees). The golden gn 2 gates in test_sommerfeld_ground.py run through
the accelerated path automatically and are the end-to-end guard.
"""

import numpy as np
import pytest

import momwire._sommerfeld as sm
from momwire._accel import acc as _acc

K2 = 2.0 * np.pi / 20.0  # 20 m wavelength
LAM = 20.0
EPS_AVG = 10.0 - 2.4j
EPS_LIST = (EPS_AVG, 3.0 - 1.2j, 81.0 - 9000.0j, 1e16 + 0.0j)

needs_acc = pytest.mark.skipif(
    _acc is None or not hasattr(_acc, "somm_six_integrals_batch"),
    reason="C++ accelerator with somm_six_integrals_batch not available",
)


def _sample_nodes():
    rhos, hs = [], []
    for r1_wl in (0.001, 0.01, 0.1, 0.5, 1.0, 5.0):
        for th_deg in (0.5, 10.0, 45.0, 89.0, 90.0):
            r1 = r1_wl * LAM
            rhos.append(r1 * np.cos(np.deg2rad(th_deg)))
            hs.append(r1 * np.sin(np.deg2rad(th_deg)))
    return np.maximum(np.array(rhos), 0.0), np.array(hs)


@needs_acc
@pytest.mark.parametrize("eps_t", EPS_LIST)
def test_batch_matches_python_six_integrals(eps_t):
    rho, h = _sample_nodes()
    cxx = _acc.somm_six_integrals_batch(complex(eps_t), K2, rho, h, 1e-9, 0)
    py = np.array(
        [sm._six_integrals(complex(eps_t), K2, r, hh, 1e-9) for r, hh in zip(rho, h)]
    )
    scale = np.abs(py).max(axis=1, keepdims=True) + 1e-300
    # per-node relative agreement (measured 3e-12 physical / 3e-9 PEC-limit)
    assert (np.abs(cxx - py) / scale).max() < 1e-7


@needs_acc
def test_batch_form_override_cross_form_agreement():
    """J and H forms must agree where both converge (rho ~ h band) —
    the C++ twin of the Python cross-form test."""
    r1 = 0.3 * LAM
    th = np.deg2rad(np.array([30.0, 40.0, 50.0]))
    rho, h = r1 * np.cos(th), r1 * np.sin(th)
    zj = _acc.somm_six_integrals_batch(EPS_AVG, K2, rho, h, 1e-9, 1)
    zh = _acc.somm_six_integrals_batch(EPS_AVG, K2, rho, h, 1e-9, 2)
    scale = np.abs(zj).max()
    assert np.abs(zj - zh).max() / scale < 1e-6


@needs_acc
def test_batch_free_space_is_exactly_zero():
    out = _acc.somm_six_integrals_batch(1.0 + 0.0j, K2, np.r_[1.0], np.r_[1.0], 1e-9, 0)
    assert np.all(out == 0.0)


@needs_acc
def test_batch_input_validation():
    with pytest.raises(ValueError):
        _acc.somm_six_integrals_batch(EPS_AVG, K2, np.r_[1.0, 2.0], np.r_[1.0], 1e-9, 0)
    with pytest.raises(ValueError):
        _acc.somm_six_integrals_batch(EPS_AVG, K2, np.r_[-1.0], np.r_[1.0], 1e-9, 0)
    with pytest.raises(ValueError):
        _acc.somm_six_integrals_batch(EPS_AVG, K2, np.r_[0.0], np.r_[0.0], 1e-9, 0)
    with pytest.raises(ValueError):
        _acc.somm_six_integrals_batch(EPS_AVG, K2, np.r_[1.0], np.r_[1.0], 1e-9, 3)


def test_python_fallback_matches_accelerated_wrapper(monkeypatch):
    """_six_integrals_batch must give the same answer with and without
    the accelerator — the honest-fallback guarantee."""
    rho = np.array([0.1 * LAM, 0.02 * LAM])
    h = np.array([0.05 * LAM, 0.08 * LAM])
    fast = sm._six_integrals_batch(EPS_AVG, K2, rho, h, rtol=1e-9)
    monkeypatch.setattr(sm, "_acc", None)
    slow = sm._six_integrals_batch(EPS_AVG, K2, rho, h, rtol=1e-9)
    scale = np.abs(slow).max()
    assert np.abs(fast - slow).max() / scale < 1e-7


def test_grid_fill_fallback_equivalence(monkeypatch):
    """A small SommerfeldGrid filled through the accelerator agrees with
    one filled through the pure-Python loop."""
    g_fast = sm.SommerfeldGrid(EPS_AVG, K2, 0.4 * LAM)
    monkeypatch.setattr(sm, "_acc", None)
    g_slow = sm.SommerfeldGrid(EPS_AVG, K2, 0.4 * LAM)
    r1 = np.array([0.05, 0.15, 0.3]) * LAM
    th = np.deg2rad(np.array([5.0, 45.0, 85.0]))
    vf = g_fast.eval(r1, th)
    vs = g_slow.eval(r1, th)
    for kk in ("IrhoV", "IzV", "IrhoH", "IphiH"):
        scale = np.abs(vs[kk]).max() + 1e-300
        assert np.abs(vf[kk] - vs[kk]).max() / scale < 1e-6


# ---------------------------------------------------------------------------
# Cooperative cancellation (same drain pattern as the other kernels)
# ---------------------------------------------------------------------------


@needs_acc
def test_batch_tripped_flag_raises_solve_aborted():
    from momwire import CancelToken, SolveAborted

    token = CancelToken()
    token.cancel()
    with pytest.raises(SolveAborted):
        _acc.somm_six_integrals_batch(
            EPS_AVG, K2, np.r_[1.0], np.r_[1.0], 1e-9, 0, token.ptr
        )


def test_python_fallback_tripped_flag_raises(monkeypatch):
    from momwire import CancelToken, SolveAborted

    monkeypatch.setattr(sm, "_acc", None)
    token = CancelToken()
    token.cancel()
    with pytest.raises(SolveAborted):
        sm._six_integrals_batch(
            EPS_AVG, K2, np.r_[1.0], np.r_[1.0], cancel_flag=token.ptr
        )


@needs_acc
def test_sommerfeld_solve_cancels_via_cpp_poll_only():
    """Neutralize the Python checkpoints so only the C++ grid-fill poll
    can observe the tripped token (the sommerfeld twin of
    test_cancel.py::test_cpp_polling_aborts_without_python_checkpoints)."""
    from fixtures_refl_coef_geoms import GEOMS
    from momwire import BSplineSolver, CancelToken, SolveAborted
    from momwire import _sommerfeld as sm_mod

    sm_mod._GRID_CACHE.clear()  # a cached grid would skip the fill
    token = CancelToken()
    token.cancel()
    kw = dict(GEOMS[("dipole", 0.05)])
    s = BSplineSolver(
        **kw,
        ground_z=0.0,
        ground_eps=(10.0, 0.002),
        ground_model="sommerfeld",
        cancel=token,
    )
    s._checkpoint = lambda: None
    with pytest.raises(SolveAborted):
        s.compute_impedance()
    assert not sm_mod._GRID_CACHE  # no partial grid was cached


def test_untripped_token_result_unchanged():
    from momwire import CancelToken

    rho = np.r_[0.5, 3.0]
    h = np.r_[1.0, 0.2]
    ref = sm._six_integrals_batch(EPS_AVG, K2, rho, h)
    tok = sm._six_integrals_batch(EPS_AVG, K2, rho, h, cancel_flag=CancelToken().ptr)
    assert np.array_equal(ref, tok)
