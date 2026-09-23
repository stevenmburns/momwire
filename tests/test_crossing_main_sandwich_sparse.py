"""The whole-axis main sandwich is SPARSE, and razor takes it.

`_main_sandwich` is the product razor's buried fill runs for both cross
blocks (`cross_complete_block` / `_reversed`, through the real constructor).
It used to densify the four weight matrices to (n_basis, n_nodes) and run six
full GEMMs — 15 s of GEMM and a 346 MB complex temporary per product at
hub_deck(16) x8 — while the split route bspline serves already contracted
sparse @ dense @ sparse (momwire#1109). Both now go through ONE contraction,
`_sandwich_dense`.

Pinned by counting: every razor main sandwich hands its tables to
`_sandwich_dense`, and no sparse weight matrix is densified while it runs, so
a regression to the dense spelling fails here rather than surfacing as a
timing drift nobody reads.

The arithmetic is not pinned to the bit against the dense spelling — BLAS
sums its GEMMs in its own blocked order, the sparse product sequentially —
but the difference is rounding: at most 3.2 ulp of each entry's summand
magnitude eps * |c1| sum_i |P_i| |K| |Q_i|^T over razor's gate decks (hub_deck(16)
x1/x2, crossing_deck(1/2), the two-radius rod and deck_1140, the two-node and
hub_fan decks, the detached pair), which is the gate below.

The kernel tables it contracts are evaluated in column chunks over the below
axis under a memory budget (`_MAIN_CHUNK_BYTES`), and THAT is bit-identical
to the one-call evaluation — pinned below with a budget small enough to cut
every block into dozens of chunks. A naive per-chunk `_tables` call is not
(it regroups the column route's fresh triples; 1e-19 to 3e-17 of max|Z| on
the gate decks), which is why `_chunked_tables` evaluates once.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire import _crossing_fill as CF  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from test_crossing_serve_524 import crossing_deck  # noqa: E402
from test_razor_detached_1149 import detached  # noqa: E402


def _razor_fill(deck):
    d = {k: v for k, v in deck.items() if k != "junctions"}
    s = RazorSolver(**d, nec5_quadrature=True)
    s.compute_impedance()
    return np.asarray(s.z)


def _dense_reference(A, B, K, k2sq, c1):
    """The retired dense spelling of the five terms, kept here as the
    rounding reference: every weight densified, every product a GEMM."""
    wA, wB = A["w"], B["w"]
    txA, tyA, tzA = A["t"].T
    txB, tyB, tzB = B["t"].T
    FA_w, FB_w = A["F_csr"].toarray() * wA, B["F_csr"].toarray() * wB
    FdA_w, FdB_w = A["Fd_csr"].toarray() * wA, B["Fd_csr"].toarray() * wB
    U, V, W, dzpW = K["U"], K["V"], K["W"], K["dzpW"]
    return c1 * (
        (FA_w * txA) @ U @ (FB_w * txB).T
        + (FA_w * tyA) @ U @ (FB_w * tyB).T
        + (FA_w * tzA) @ (k2sq * V + dzpW) @ (FB_w * tzB).T
        + (FA_w * tzA) @ W @ FdB_w.T
        + FdA_w @ W @ (FB_w * tzB).T
        - FdA_w @ V @ FdB_w.T
    )


def _abs_bound(A, B, K, k2sq, c1):
    """|c1| sum_i |P_i| |K_x| |Q_i|^T — the magnitude of the summands each
    entry of the sandwich is built from (the forward-error scale)."""
    iA, iB = np.arange(A["nodes"].shape[0]), np.arange(B["nodes"].shape[0])
    P = [abs(p) for p in CF._row_weights(A, iA)]
    Q = [abs(q) for q in CF._row_weights(B, iB)]
    a = {k: np.abs(v) for k, v in K.items()}
    return abs(c1) * (
        P[0] @ a["U"] @ Q[0].T
        + P[1] @ a["U"] @ Q[1].T
        + P[2] @ (abs(k2sq) * a["V"] + a["dzpW"]) @ Q[2].T
        + P[2] @ a["W"] @ Q[3].T
        + P[3] @ a["W"] @ Q[2].T
        + P[3] @ a["V"] @ Q[3].T
    )


def test_razor_main_sandwich_takes_the_sparse_path(monkeypatch):
    """Both of razor's cross blocks reach `_main_sandwich`, each hands its
    product to `_sandwich_dense`, and nothing is densified inside it."""
    real_main, real_sparse = CF._main_sandwich, CF._sandwich_dense
    inside, n = [], {"main": 0, "sparse": 0, "densified": 0}

    def main(*a, **k):
        n["main"] += 1
        inside.append(True)
        try:
            return real_main(*a, **k)
        finally:
            inside.pop()

    def sparse(*a, **k):
        n["sparse"] += bool(inside)
        return real_sparse(*a, **k)

    def forbid(real):
        def guarded(self, *a, **k):
            n["densified"] += bool(inside)
            return real(self, *a, **k)

        return guarded

    monkeypatch.setattr(CF, "_main_sandwich", main)
    monkeypatch.setattr(CF, "_sandwich_dense", sparse)
    for cls in (sp.csr_array, sp.csc_array, sp.csr_matrix, sp.csc_matrix):
        monkeypatch.setattr(cls, "toarray", forbid(cls.toarray))
        monkeypatch.setattr(cls, "todense", forbid(cls.todense))
    _razor_fill(crossing_deck(1))
    assert n["main"] == 2, n  # forward + reversed, one radius
    assert n["sparse"] == n["main"], n
    assert n["densified"] == 0, n


def test_the_sparse_sandwich_is_the_dense_one_at_rounding(monkeypatch):
    """Against the retired dense spelling, entry by entry, on the arguments a
    real razor fill hands `_main_sandwich`: within a few ulp of each entry's
    own summand magnitude (measured ≤ 3.2 on the gate decks; 8 is the bar),
    and exactly zero wherever every summand is."""
    real_main = CF._main_sandwich
    got = []

    def main(ctx, A, B, eps_t, k_p, c1, gz, memo=None):
        out = real_main(ctx, A, B, eps_t, k_p, c1, gz, memo=memo)
        # A copy: the caller accumulates the end terms into `out` in place.
        got.append((ctx, A, B, eps_t, k_p, c1, gz, out.copy()))
        return out

    monkeypatch.setattr(CF, "_main_sandwich", main)
    _razor_fill(crossing_deck(1))
    assert len(got) == 2
    eps = np.finfo(float).eps
    for ctx, A, B, eps_t, k_p, c1, gz, out in got:
        rho = np.hypot(
            A["nodes"][:, 0][:, None] - B["nodes"][:, 0][None, :],
            A["nodes"][:, 1][:, None] - B["nodes"][:, 1][None, :],
        )
        z = np.broadcast_to((A["nodes"][:, 2] - gz)[:, None], rho.shape)
        zp = np.broadcast_to((B["nodes"][:, 2] - gz)[None, :], rho.shape)
        K = CF._tables(ctx, eps_t, k_p, rho, z, zp, CF._CROSS_RTOL, memo=None)
        ref = _dense_reference(A, B, K, k_p * k_p, c1)
        bound = _abs_bound(A, B, K, k_p * k_p, c1)
        d = np.abs(out - ref)
        assert not np.any(d[bound == 0]), "a structural zero picked up a value"
        live = bound > 0
        assert live.any()
        assert (d[live] / (eps * bound[live])).max() <= 8.0


def test_chunked_tables_are_the_one_call_bit_for_bit(monkeypatch):
    """The same Z to the bit whether the main sandwich's tables come from one
    `_tables` call or from `_chunked_tables` in many column chunks — on the
    crossing deck (every pair at one ρ, so a chunk boundary cuts its single
    column) and the detached deck (ρ varies across the grid)."""
    real, budget = CF._chunked_tables, CF._MAIN_CHUNK_BYTES
    chunks = []

    def spy(ctx, eps_t, k_p, rho, zA, zB, cols, memo):
        chunks.append(len(cols))
        yield from real(ctx, eps_t, k_p, rho, zA, zB, cols, memo)

    monkeypatch.setattr(CF, "_chunked_tables", spy)
    for deck in (crossing_deck(1), detached()):
        chunks.clear()
        whole = _razor_fill(deck)
        assert chunks == []  # these decks fit the default budget in one call
        monkeypatch.setattr(CF, "_MAIN_CHUNK_BYTES", 20_000)
        cut = _razor_fill(deck)
        monkeypatch.setattr(CF, "_MAIN_CHUNK_BYTES", budget)
        assert len(chunks) == 2 and min(chunks) > 10, chunks
        assert np.array_equal(whole, cut)
