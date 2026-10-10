"""The Sommerfeld remainder's per-chunk test fold, overlapped with the next
chunk's projection — perf item 9 (momwire#1290).

`_tested_sommerfeld_remainder` streams the remainder in observer chunks and
folds each chunk into the image triple (`_tested_contrib_rows` and a
subtraction, numpy). `_OverlappedFold` runs each fold on one worker thread
while the replay loop projects the next chunk; `_REMAINDER_OVERLAP = False`
folds inline. A fold runs the same expressions on the same arrays either way,
so numpy takes the same loop for every product — including the in-place
multiply it elides a temporary into past 256 KB, which rounds differently
from the out-of-place one (momwire#392) — and two chunks' folds write
disjoint rows. The claim is equality of bytes of the matrix handed to the
solve, Z and the currents: on above-ground decks (the plain fold), one large
enough that its folds' gathers cross the elision threshold, and on buried and
mixed decks (the masked fold, with and without a class's own source columns),
the overlapped folds counted.

Also pinned: a fold's exception surfaces from the solve, and the red
control — each entry's nq products summed in the reverse node order, the same
value in a different association — is seen by the gate.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_sg_cplx_far_fill_simd_1224 import _buried_dipole, _hub4
from test_sg_real_far_fill_staged_1290 import _array

from momwire import sinusoidal as sn
from momwire import sinusoidal_galerkin as sg
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver


def _direct_sg(*args, **kw):
    """This module's subject is the DIRECT-field fill's own machinery, so
    every solver here names it (`SinusoidalGalerkinSolver`'s default fill is
    the mixed-potential one since momwire#1354, gated in
    `test_sinusoidal_mp_1354.py`)."""
    from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver as _cls

    kw.setdefault("fill", "direct")
    return _cls(*args, **kw)


# numpy's NPY_MIN_ELIDE_BYTES.
ELIDE_BYTES = 256 * 1024


def _array_low(n=15):
    """The above-ground array at 0.3 m, where the remainder is large against
    the image triple it is subtracted from, so a last-bit change in it
    reaches G rather than rounding away."""
    d = _array(True, n=n)
    d["wires"] = [w - np.array([0.0, 0.0, 9.7]) for w in d["wires"]]
    return d


DECKS = {
    "array-low": _array_low,
    "array-low-251": lambda: _array_low(n=251),
    "buried-dipole": _buried_dipole,
    "hub4": _hub4,
}

_SOLVE = sg._solve_in_place
_WRAP = sg._OverlappedFold.wrap


def _capture(make, monkeypatch, seen=None):
    G = {}

    def spy_solve(A, rhs):
        G.setdefault("G", np.array(A, copy=True))
        return _SOLVE(A, rhs)

    monkeypatch.setattr(sg, "_solve_in_place", spy_solve)
    if seen is not None:

        def spy_wrap(self, consume):
            def counted(i0, i1, block):
                seen["folds"] += 1
                seen["overlapped"] += int(sg._REMAINDER_OVERLAP)
                seen["max_bytes"] = max(seen["max_bytes"], max(b.nbytes for b in block))
                return consume(i0, i1, block)

            return _WRAP(self, counted)

        monkeypatch.setattr(sg._OverlappedFold, "wrap", spy_wrap)
    Z, cur = _direct_sg(**make()).compute_impedance()
    return G["G"], np.atleast_1d(Z), np.asarray(cur)


@pytest.mark.parametrize(
    "name",
    [
        # The N = 251 array is the slow deck (median 8.5 s on Linux CI); the
        # other three keep the overlapped fold on every PR.
        pytest.param(n, marks=[pytest.mark.slow] if n == "array-low-251" else [])
        for n in sorted(DECKS)
    ],
)
def test_overlapped_fold_equals_the_inline_fold_to_the_byte(name, monkeypatch):
    if name != "array-low-251":
        # Small decks are one chunk at the shipped size; a smaller chunk gives
        # them several, so a fold actually runs beside a projection. Chunk
        # boundaries move no float (`_replay_remainder`).
        monkeypatch.setattr(sn, "_REMAINDER_CHUNK_ELEMS", 1 << 14)
    seen = {"folds": 0, "overlapped": 0, "max_bytes": 0}
    G1, Z1, I1 = _capture(DECKS[name], monkeypatch, seen)
    assert seen["overlapped"] > 1, f"{name}: no fold ran overlapped ({seen})"
    if name == "array-low-251":
        # A chunk's per-shape block is nq x each node's gather; past 8 x the
        # threshold, the gathers are past it too.
        assert seen["max_bytes"] > 8 * ELIDE_BYTES, seen
    monkeypatch.setattr(sg, "_REMAINDER_OVERLAP", False)
    G0, Z0, I0 = _capture(DECKS[name], monkeypatch)
    assert G1.tobytes() == G0.tobytes()
    assert Z1.tobytes() == Z0.tobytes()
    assert I1.tobytes() == I0.tobytes()


def test_a_fold_error_surfaces_from_the_solve(monkeypatch):
    class Boom(RuntimeError):
        pass

    def failing(w_entry, m_local, nq, Phi):
        raise Boom("fold")

    monkeypatch.setattr(
        SinusoidalGalerkinSolver, "_tested_contrib_rows", staticmethod(failing)
    )
    with pytest.raises(Boom):
        _direct_sg(**_array_low()).compute_impedance()


@pytest.mark.parametrize("name", ["array-low", "buried-dipole", "hub4"])
def test_red_control_a_reassociated_reduction_is_seen(name, monkeypatch):
    G0, _, _ = _capture(DECKS[name], monkeypatch)

    def reversed_rows(w_entry, m_local, nq, Phi):
        out = np.zeros((w_entry.shape[0], Phi.shape[-1]), dtype=np.complex128)
        for q in reversed(range(nq)):
            out += w_entry[:, q, None] * Phi[m_local, q, :]
        return out

    monkeypatch.setattr(
        SinusoidalGalerkinSolver, "_tested_contrib_rows", staticmethod(reversed_rows)
    )
    G1, _, _ = _capture(DECKS[name], monkeypatch)
    assert G1.tobytes() != G0.tobytes()
    assert np.allclose(G1, G0, rtol=1e-12, atol=1e-12 * np.abs(G0).max())
