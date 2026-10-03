"""Bit-identity gate for the point-matched fill's tiled Φ reduction — momwire#1290.

`_reduce_phi_band` used to spell a band's reduction as scipy's
`ndarray @ csc` three times — `out = Φ₀@M₀; out += Φ₁@M₁; out += Φ₂@M₂` —
which is `(Mᵀ @ Φᵀ)ᵀ` per product: a Φ-sized transposed copy, a Φ-sized
result, then a strided write into `out`. It now calls `Mᵀ @ Φᵀ` itself per
tile of `_REDUCE_TILE` observer rows, sums `(A + B) + C` in the tile's
transposed space and writes the sum back once. The products are the same
scipy routine (`csr_matvecs`) on the same operands, so the claim is
bit-identity by construction, on every platform.

Gates:
  * on the real Φ and M of two decks above the dense threshold — the
    8-dipole free-space array (`_assemble_Z`'s own route) and the
    inverted-L over 16 buried radials (the mixed fill's) — the tiled
    reduction equals the whole-band scipy spelling (written out below, not
    the solver's helper) BYTE FOR BYTE, at band heights that leave ragged
    and one-row tails and at tile widths other than the default (the tile
    is a speed knob that must move no bits);
  * the tiles ran: a spy on scipy's sparse products records each
    product's operand width, which must be exactly the tile cut — and
    never one column wide unless the band is one row, because scipy routes
    a one-column operand to a different routine (`csr_matvec`);
  * `_assemble_Z` on the free array equals the pre-#1290 fill (whole Φ,
    whole products) byte for byte.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse
from test_crossing_serve_524 import invl_deck
from test_mixed_fill_fused_1224 import whole_phi_tables

from momwire import sinusoidal as sn
from momwire.sinusoidal import SinusoidalSolver


def _free_array_deck(segs=22):
    """The momwire#1290 free-space array: 8 parallel 20 m dipoles 8.5 m apart
    at 7 MHz, `segs` segments each (N = 8 · segs)."""
    wires = [
        np.array([(-10.0, 8.5 * j, 10.0), (10.0, 8.5 * j, 10.0)]) for j in range(8)
    ]
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[segs] for _ in wires],
        feeds=[(0, 10.0, 1 + 0j)],
        wavelength=299792458.0 / 7e6,
        wire_radius=1e-3,
    )


def _free_tables(s, geom):
    """The free fill's whole Φ triple and its Ms, as `_assemble_Z` builds
    them (one band over every row)."""
    k = s.k
    view = s._basis_coefs(geom, k)
    N = int(geom["n_segs"])
    starts = view["starts"]
    n_idx = np.repeat(np.arange(N, dtype=np.int64), starts[1:] - starts[:-1])
    kd_min = float(np.min(np.abs(k) * np.asarray(geom["seg_h"], dtype=np.float64)))
    cos_shape = "cos-1" if kd_min < sn._WELL_SCALED_KD else "cos"
    sigma = view["sigma"]
    coefs = (
        sigma * (view["A"] if cos_shape == "cos" else view["AC"]),
        view["B"],
        sigma * view["C"],
    )
    Ms = [
        scipy.sparse.csc_matrix((c, (n_idx, view["jbasis"])), shape=(N, N))
        for c in coefs
    ]
    Phi = list(s._field_tensor(geom, k, cos_shape=cos_shape))
    return Phi, Ms, view


@pytest.fixture(scope="module")
def tables():
    out = {}
    s = SinusoidalSolver(**_free_array_deck())
    geom = s._build_geometry()
    Phi, Ms, _view = _free_tables(s, geom)
    out["free"] = (Phi, Ms)
    s = SinusoidalSolver(**invl_deck(x=4))
    Phi, Ms, _rest = whole_phi_tables(s, s._build_geometry())
    out["invl"] = (Phi, Ms)
    return out


def _bytes_equal(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def _scipy_band(Phi, Ms):
    """The pre-#1290 band: scipy's `ndarray @ csc`, whole band, `(A + B) + C`."""
    out = Phi[0] @ Ms[0]
    out += Phi[1] @ Ms[1]
    out += Phi[2] @ Ms[2]
    return out


@pytest.fixture
def widths(monkeypatch):
    """Operand width of every sparse product scipy runs, by routine."""
    seen = []
    cls = scipy.sparse.csr_matrix
    vec, multi = cls._matmul_vector, cls._matmul_multivector

    def spy_vec(self, other):
        seen.append(1)
        return vec(self, other)

    def spy_multi(self, other):
        seen.append(other.shape[1])
        return multi(self, other)

    monkeypatch.setattr(cls, "_matmul_vector", spy_vec)
    monkeypatch.setattr(cls, "_matmul_multivector", spy_multi)
    return seen


def _expected_tiles(n, tile):
    starts = list(range(0, n, max(2, tile)))
    if n > 1 and n - starts[-1] == 1:
        starts.pop()
    return [b - a for a, b in zip(starts, starts[1:] + [n])]


@pytest.mark.parametrize("deck", ["free", "invl"])
@pytest.mark.parametrize("tile", [None, 1, 5, 64])
@pytest.mark.parametrize("rows", [1, 2, 16, 17, 33, 100, "all"])
def test_tiled_band_is_the_scipy_band(tables, deck, tile, rows, widths, monkeypatch):
    Phi, Ms = tables[deck]
    N = Phi[0].shape[0]
    assert N >= sn._DENSE_ASSEMBLY_THRESHOLD
    assert all(isinstance(M, scipy.sparse.csc_matrix) for M in Ms)
    if tile is not None:
        monkeypatch.setattr(sn, "_REDUCE_TILE", tile)
    n = N if rows == "all" else rows
    # An interior band, so a row offset error cannot hide at row 0.
    r0 = (N - n) // 2
    band = [P[r0 : r0 + n] for P in Phi]
    ref = _scipy_band(band, Ms)
    widths.clear()
    out = np.full((n, N), np.nan + 0j)
    got = sn._reduce_phi_band(band, Ms, out)
    assert got is out
    assert widths == [w for w in _expected_tiles(n, sn._REDUCE_TILE) for _ in range(3)]
    assert (1 in widths) == (n == 1)
    assert _bytes_equal(out, ref)


def test_assemble_z_is_the_pre_1290_fill(widths):
    s = SinusoidalSolver(**_free_array_deck())
    geom = s._build_geometry()
    N = int(geom["n_segs"])
    Phi, Ms, view = _free_tables(s, geom)
    ref = _scipy_band(Phi, Ms)
    s._contact_charge_correction(ref, geom, s.k, view, eta=s._fill_eta(s.k, None))
    s._apply_loading(ref, geom, view, s.k)
    widths.clear()
    G, _view = s._assemble_Z(geom, s.k)
    # The production fill ran the tiles: every product is a tile wide.
    assert widths and max(widths) == sn._REDUCE_TILE
    assert sum(widths) == 3 * N
    assert _bytes_equal(G, ref)
