"""`pair_reuse`'s evaluate and its gather (momwire#1916 part B).

Two changes, neither of which may move a bit of the reused fill:

- `evaluate` runs the kernel on slices of the plan's canonical sources
  (classes renumbered so each observer radius is one run), at any chunk;
- the fused `pair_reuse_reduce_band` gathers each band's Φ by class inside
  the reduction: it is `scatter` then `reduce_phi_band`, bit for bit, the
  near pairs and the PEC image included.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse

from momwire import _sin_pair_reuse as pr
from momwire import sinusoidal as sin_mod
from momwire._accel import acc
from momwire.sinusoidal import SinusoidalSolver

pytestmark = [
    pytest.mark.reach(1),
    pytest.mark.skipif(
        acc is None or not getattr(acc, "pair_reuse_reduce_1916", False),
        reason="accelerator None or without pair_reuse_reduce_1916",
    ),
]

LAM = 21.1


def _deck(h=None):
    """Reflector, driver and six directors plus a bent wire of another
    radius-free shape: ~190 segments, above the dense-M threshold."""
    Ld = 0.47 * LAM
    z = 8.0 if h is None else h
    wires = [
        np.array([[0.2 * LAM * i, -L / 2, z], [0.2 * LAM * i, L / 2, z]])
        for i, L in enumerate([1.05 * Ld, Ld] + [0.95 * Ld] * 6)
    ]
    wires.append(np.array([[-2.0, 0.0, z - 3.0], [-2.0, 3.0, z - 1.0], [-2.0, 5.0, z]]))
    n = [[21] for _ in wires[:-1]] + [[9, 7]]
    return dict(
        wires=wires,
        n_per_edge_per_wire=n,
        feed_wire_index=1,
        wavelength=LAM,
        wire_radius=1e-3,
    )


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("MOMWIRE_X_PAIR_REUSE", raising=False)


def _fill(h=None, ek=False, **kw):
    deck = _deck(h)
    extra = {"ground_z": 0.0} if h is not None else {}
    s = SinusoidalSolver(**deck, **extra, extended_kernel=ek, pair_reuse=True, **kw)
    geom = s._build_geometry()
    G, _ = s._assemble_Z(geom, s.k)
    return G, s


@pytest.mark.parametrize("ek", [False, True])
@pytest.mark.parametrize("h", [None, 6.0])
@pytest.mark.parametrize("budget", [None, 1])
def test_the_fused_gather_is_scatter_then_reduce_bit_for_bit(
    ek, h, budget, monkeypatch
):
    kw = {} if budget is None else {"swept_mem_mb": budget}
    calls = []
    real = pr.reduce_band

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(pr, "reduce_band", counting)
    fused, s = _fill(h, ek, **kw)
    assert calls, "the fused path did not run"
    assert s.pair_reuse_declined is None
    if h is not None:
        assert set(s.pair_reuse_census) == {"direct", "image"}
    monkeypatch.setattr(pr, "_HAVE_REDUCE", False)
    calls.clear()
    banded, _ = _fill(h, ek, **kw)
    assert calls == []
    assert fused.tobytes() == banded.tobytes()


def test_evaluate_at_any_chunk(monkeypatch):
    """The chunk is a speed knob, but not a bit-free one: the kernel's
    batched sincos runs its vector loop over the call's phase buffer and a
    scalar tail past it, and the two libm spellings differ in the last ulp,
    so where a source falls in its call can move its Φ by about one ulp
    (measured: ~1e-16 relative on Z). The renumbered runs are the classes
    in the order the per-radius index lists had them, so at the old chunk
    the calls are the old calls."""
    base, s = _fill()
    plan = s._sin_reuse_plans["direct"][1]
    # The renumbering made each observer radius one run of classes.
    assert plan.runs[0][0] == 0 and plan.runs[-1][1] == plan.n_classes
    assert all(a[1] == b[0] for a, b in zip(plan.runs, plan.runs[1:]))
    monkeypatch.setattr(pr, "EVAL_CHUNK", 7)
    small, _ = _fill()
    assert np.max(np.abs(small - base)) <= 1e-14 * np.max(np.abs(base))


def test_the_kernel_against_numpy_on_a_synthetic_band():
    rng = np.random.default_rng(1916)
    rows, n_src, n_cls = 37, 90, 400

    def cplx(*shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    def block():
        cls = rng.integers(-3, n_cls + 3, size=(rows, n_src)).astype(np.int32)
        rep = cplx(n_cls, 3)
        near = np.unique(rng.integers(0, rows * n_src, size=300)).astype(np.int64)
        return cls, rep, near, cplx(near.size, 3)

    def numpy_phi(cls, rep, near, vals):
        phi = [np.take(rep[:, s], cls, mode="clip") for s in range(3)]
        for s in range(3):
            phi[s].reshape(-1)[near] = vals[:, s]
        return phi

    Ms = []
    for _ in range(3):
        n_idx = rng.integers(0, n_src, size=3 * n_src)
        j_idx = rng.integers(0, n_src, size=3 * n_src)
        Ms.append(
            scipy.sparse.csc_matrix(
                (cplx(3 * n_src), (n_idx, j_idx)), shape=(n_src, n_src)
            )
        )
    m_args = []
    for M in Ms:
        m_args += [M.indptr, M.indices, M.data]
    d, im = block(), block()
    for image in (None, im):
        phi = numpy_phi(*d)
        if image is not None:
            phi_i = numpy_phi(*image)
            for s in range(3):
                phi[s] -= phi_i[s]
        want = np.empty((rows, n_src), dtype=np.complex128)
        sin_mod._reduce_phi_band(phi, Ms, want)
        for nt in (1, 3):
            got = np.full((rows, n_src), np.nan, dtype=np.complex128)
            acc.pair_reuse_reduce_band(got, *d, image, *m_args, n_threads=nt)
            assert got.tobytes() == want.tobytes(), (image is None, nt)


def test_the_kernel_refuses_what_it_cannot_take():
    out = np.zeros((2, 4), dtype=np.complex128)
    cls = np.zeros((2, 4), dtype=np.int32)
    rep = np.ones((3, 3), dtype=np.complex128)
    M = scipy.sparse.csc_matrix(np.eye(4, dtype=np.complex128))
    m_args = [M.indptr, M.indices, M.data] * 3
    near = np.array([5, 3], dtype=np.int64)  # unsorted
    with pytest.raises(RuntimeError, match="sorted"):
        acc.pair_reuse_reduce_band(
            out, cls, rep, near, np.zeros((2, 3), np.complex128), None, *m_args
        )
    with pytest.raises(RuntimeError, match="outside the band"):
        acc.pair_reuse_reduce_band(
            out,
            cls,
            rep,
            np.array([8], np.int64),
            np.zeros((1, 3), np.complex128),
            None,
            *m_args,
        )
