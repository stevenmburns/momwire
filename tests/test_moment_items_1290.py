"""Razor's segment-moment kernels over (observer tile, segment tile) items
(momwire#1290).

`razor_seg_moments` and its complex-k twin were parallel over observer
tiles alone, so a call with few observers left threads idle: the below-plane
windows at buried x16 ask 34 observers of 2562 segments, five items for four
threads. The items are now (observer tile, segment tile) pairs. An entry is
a function of its own observer, its own segment and the quadrature, formed
by one thread in one order, so the tiling cannot reach it. Gated here the
way that claim is stated: any block of a call's answer is the answer of the
call on that block's observers and segments alone (which tiles them
differently), to the bit, on both kernels and with the extended kernel's
groups in play.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import razor as _razor

pytestmark = pytest.mark.skipif(
    not (_razor._acc is not None and hasattr(_razor._acc, "razor_seg_moments_cplx")),
    reason="needs the razor moment kernels",
)


def _geometry(n_seg, seed):
    rng = np.random.default_rng(seed)
    p0 = rng.uniform(-3.0, 3.0, (n_seg, 3))
    p0[:, 2] = -rng.uniform(0.05, 1.0, n_seg)
    t = rng.standard_normal((n_seg, 3))
    t /= np.linalg.norm(t, axis=1)[:, None]
    h = rng.uniform(0.05, 0.3, n_seg)
    obs = p0 + 0.5 * h[:, None] * t  # centroids: the kernels' own observers
    return obs, p0, t, h


def _call(kernel, k, obs, p0, t, h, gi, gj, aek):
    xg, wg = _razor.leggauss(12)
    a = np.full(p0.shape[0], 1e-3)
    return kernel(obs, p0, t, h, a, xg, wg, k, True, gi, gj, aek)


@pytest.mark.parametrize("cplx", [False, True], ids=["real-k", "complex-k"])
@pytest.mark.parametrize("ek", [False, True], ids=["reduced", "ek"])
def test_a_block_is_its_own_call(cplx, ek):
    n_obs, n_seg = 37, 300  # several tiles each way, ragged at both ends
    obs, p0, t, h = _geometry(n_seg, seed=5)
    obs = obs[:n_obs]
    if ek:
        gi = np.arange(n_obs, dtype=np.int64) % 7
        gj = np.arange(n_seg, dtype=np.int64) % 7
        aek = np.full(n_seg, 1e-3)
    else:
        gi = gj = np.empty(0, dtype=np.int64)
        aek = np.empty(0)
    kernel = (
        _razor._acc.razor_seg_moments_cplx if cplx else _razor._acc.razor_seg_moments
    )
    k = complex(0.58, -0.24) if cplx else 0.146
    M0, M1 = _call(kernel, k, obs, p0, t, h, gi, gj, aek)
    assert np.isfinite(M0).all() and np.isfinite(M1).all()
    for (o0, o1), (s0, s1) in [
        ((0, 1), (0, n_seg)),
        ((3, 20), (5, 141)),
        ((30, 37), (129, 300)),
    ]:
        sub = _call(
            kernel,
            k,
            obs[o0:o1],
            p0[s0:s1],
            t[s0:s1],
            h[s0:s1],
            gi[o0:o1] if ek else gi,
            gj[s0:s1] if ek else gj,
            aek[s0:s1] if ek else aek,
        )
        for whole, part in zip((M0, M1), sub):
            want = np.ascontiguousarray(whole[o0:o1, s0:s1])
            assert np.array_equal(part.view(np.uint64), want.view(np.uint64))
