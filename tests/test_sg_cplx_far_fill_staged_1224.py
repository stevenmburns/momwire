"""The complex-k Galerkin far fill's rows, staged for the vector unit — momwire#1224.

After the sweep (test_sg_cplx_far_fill_simd_1224) the fill's time sat in
scalar code: per (observer, source) pair, the geometry (sqrt and divide bound)
and the assembly (scalar std::complex arithmetic, ~33 divisions a pair).
`sg_cplx_rows_staged` runs both as loops over the sources that the compiler
vectorises, and keeps the three geometry transcendentals (log1p, asinh, sinh)
scalar through the same libm calls.

It is the reference body's IEEE operations, one for one and in the same order
(-ffp-contract=off, no reduction crosses lanes, std::complex's operators
spelled out as libstdc++ evaluates them), so the claim is EQUALITY OF BYTES
with the scalar body, which stays callable as `reference=True`. The tolerance
is therefore zero, not derived: any difference at all is a transcription
error. What this file pins:

  * bytes, staged vs reference, on every fill call a buried, a crossing and a
    hub deck make through the production seam, with and without the ground
    fold's `out`/`scale`;
  * bytes on synthetic rows that drive each of the assembly's three
    instantiations — every lane on sin_minus_arg's series arm, every lane on
    its sin(kd) - u arm, and the two mixed — since the decks above mostly
    exercise the first;
  * that production takes the staged body: its bytes equal the reference's,
    so only the kernel's own per-body call counters can say which ran.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_crossing_serve_524 import hub_deck
from test_sg_cplx_far_fill_simd_1224 import DECKS

import momwire._accel as _accel
from momwire import sinusoidal_galerkin as sg


def _direct_sg(*args, **kw):
    """This module's subject is the DIRECT-field fill's own machinery, so
    every solver here names it (`SinusoidalGalerkinSolver`'s default fill is
    the mixed-potential one since momwire#1354, gated in
    `test_sinusoidal_mp_1354.py`)."""
    from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver as _cls

    kw.setdefault("fill", "direct")
    return _cls(*args, **kw)


pytestmark = pytest.mark.skipif(
    not sg._HAVE_GALERKIN_FAR_FILL_CPLX
    or not hasattr(_accel.acc, "sg_cplx_far_fill_calls"),
    reason="no compiled complex far fill with the staged rows in this build",
)

DECK_PARAMS = [
    *sorted(DECKS),
    # The production deck's geometry at x1 (N = 177); the timed deck is x16.
    pytest.param("hub16", marks=pytest.mark.slow),
]
ALL_DECKS = dict(DECKS, hub16=lambda: hub_deck(n_radials=16))


def _record(make):
    """Every complex fill call one solve makes, as argument copies, and how
    far the kernel's per-body counters moved during it. The spy only records
    and forwards; the replay happens after it is removed."""
    fill = _accel.acc.sinusoidal_galerkin_far_fill_cplx
    seen = []

    def spy(*a, **kw):
        seen.append(
            [np.array(x, copy=True) if isinstance(x, np.ndarray) else x for x in a]
        )
        return fill(*a, **kw)

    before = _accel.acc.sg_cplx_far_fill_calls()
    _accel.acc.sinusoidal_galerkin_far_fill_cplx = spy
    try:
        _direct_sg(**make()).compute_impedance()
    finally:
        _accel.acc.sinusoidal_galerkin_far_fill_cplx = fill
    after = _accel.acc.sg_cplx_far_fill_calls()
    return seen, (after[0] - before[0], after[1] - before[1])


def _same_bytes(a, b):
    return all(x.tobytes() == y.tobytes() for x, y in zip(a, b))


def _both(args, **kw):
    fill = _accel.acc.sinusoidal_galerkin_far_fill_cplx
    return fill(*args, **kw), fill(*args, reference=True, **kw)


@pytest.mark.parametrize("name", DECK_PARAMS)
def test_staged_rows_equal_the_reference_body_to_the_byte(name):
    """One solve per deck, three claims:

    * production took the staged body: the kernel counts its calls per body,
      and the solve moved the staged count by exactly the number of complex
      fill calls it made and the reference count not at all (the bodies'
      bytes are equal, so nothing else can tell them apart);
    * every one of those calls, replayed, gives the reference body's bytes;
    * with the ground fold's `out`/`scale` (the kernel adds scale*band into a
      caller's triple) the triple's bytes match too.
    """
    calls, (staged_n, ref_n) = _record(ALL_DECKS[name])
    assert calls, f"{name}: the complex far fill never ran"
    assert staged_n == len(calls), (staged_n, len(calls))
    assert ref_n == 0, f"{name}: a production fill took the reference body"
    for i, args in enumerate(calls):
        staged, ref = _both(args)
        assert _same_bytes(staged, ref), f"{name}: call {i} differs"

    rng = np.random.default_rng(1224)
    base = [
        rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape) for x in staged
    ]
    fill = _accel.acc.sinusoidal_galerkin_far_fill_cplx
    outs = []
    for reference in (False, True):
        out = tuple(np.array(b, copy=True) for b in base)
        fill(*calls[-1], out=out, scale=complex(-1.0, 0.3), reference=reference)
        outs.append(out)
    assert _same_bytes(*outs), f"{name}: the folded triple differs"


def _synthetic(H, n_obs_seg=3, seed=0):
    """A direct call: parallel sources of half-lengths H along z, observers
    on the z = 0 plane beside them. There z_eval = 0, so P = 0 and the two
    sin_minus_arg arguments are both u = kH: |kH| < 0.1 puts a lane on the
    series arm, anything larger on sin(kd) - u. The test segments are
    observers only (any w_entry will do for a byte comparison)."""
    rng = np.random.default_rng(seed)
    H = np.asarray(H, dtype=np.float64)
    N = H.size
    src_c = np.column_stack([rng.uniform(-3, 3, N), rng.uniform(-3, 3, N), np.zeros(N)])
    src_t = np.tile([0.0, 0.0, 1.0], (N, 1))
    nq = 4
    M = n_obs_seg
    obs_c = np.column_stack(
        [rng.uniform(4, 9, M * nq), rng.uniform(-3, 3, M * nq), np.zeros(M * nq)]
    )
    obs_t = np.tile([0.0, 0.0, 1.0], (M * nq, 1))
    a = np.full(M * nq, 1e-3)
    gx, gw = np.polynomial.legendre.leggauss(8)
    nnz = 2 * M
    w = rng.standard_normal((nnz, nq)) + 1j * rng.standard_normal((nnz, nq))
    starts = np.arange(0, nnz + 1, 2, dtype=np.int64)
    k = complex(0.58, -0.238)
    eta = complex(98.0, 12.0)
    return [obs_c, obs_t, a, src_c, src_t, H, k, eta, gx, gw, w, starts]


@pytest.mark.parametrize(
    "what, H",
    [
        # |k| = 0.627: |kH| < 0.1 below H = 0.16.
        ("every lane on the series", np.full(37, 0.02)),
        ("every lane on sin(kd) - u", np.full(37, 0.9)),
        ("lanes split", np.where(np.arange(37) % 3 == 0, 0.9, 0.02)),
        ("one far lane", np.where(np.arange(37) == 36, 0.9, 0.02)),
    ],
)
def test_each_arm_instantiation_matches_the_reference(what, H):
    staged, ref = _both(_synthetic(H))
    assert np.all(np.isfinite(staged[0])), what
    assert _same_bytes(staged, ref), what
