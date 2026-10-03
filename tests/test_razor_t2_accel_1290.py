"""Razor's fused T2 rows (momwire#1290) against the numpy spelling.

`_source_block_rows` finishes each row window as `c_A * T1 - T2 / c_Phi`,
with T2 gathered out of the centroid moments. numpy spelled that with two
(rows, n_seg) gathers and seven (rows, n_basis) temporaries, then copied the
result into the window of a column-major Z: ~0.28 s of a 1.27 s free-space
solve at N = 2816. `razor_t2_rows` reads each entry where it lies and writes
it once.

What this module gates:

  * **the kernel is the numpy spelling**, entry by entry, under a bound
    derived from its operands: every output is a fixed short sequence of
    roundings (the kernel's header derives it), so a build that contracts
    differently can move an entry by a few units in the last place of the
    terms that make it, and by no more. Both modes: fused (both prefactors
    purely imaginary) and T2-only (the lossy medium's complex c_Phi, whose
    division stays numpy's). Not `==`: the numpy side's bits belong to
    numpy's build, not to this one (the #1289 lesson). On Skylake avx2 they
    measured identical, and so did the solver's Z, impedance and currents
    on the free, buried and inverted-L benchmark decks.
  * **the routing is the numpy route's**: whole fills with the kernel on and
    off, on decks that reach every row case `_t2_tables` encodes -- plain
    rows, grounded rows (contact over PEC and over Sommerfeld), chopped rows
    on both sides, and the buried crossing deck whose below-medium blocks
    take the T2-only mode. A mis-routed row reference moves an entry by
    O(|Z|); the bar is 1e-13 of max|Z|, the kernel bound's scale.
  * **the kernel actually runs**, in both modes, and the off switch reaches
    zero calls -- the green-gate-measuring-nothing check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg

from momwire import razor as _razor
from momwire.razor import RazorSolver

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_crossing_serve_524 import crossing_deck  # noqa: E402

pytestmark = pytest.mark.skipif(
    not _razor._HAVE_RAZOR_T2_ACCEL, reason="accelerator without razor_t2_1290"
)

C0 = 299792458.0
LAM = C0 / 7.0e6
EPS = np.finfo(float).eps
BAR_ROUTE = 1e-13  # measured 0.0 on Skylake avx2 for every deck below


def _numpy_spelling(t1, tab, plus, minus, s_a, s_b, q_a, q_b, c_A, c_Phi):
    """The numpy route's arithmetic for one window, and the derived bound.

    The bound is 8 units of roundoff (4 eps) on the magnitudes of the terms
    each entry is built from -- the subtraction's two operands, both wings'
    products, and both prefactor products -- which covers any contraction
    of the few roundings per entry, and nothing a wrong index could hide in.
    """
    P = tab[plus]
    M = np.where((minus >= 0)[:, None], tab[np.maximum(minus, 0)], 0)
    dM0 = P - M
    T2 = dM0[:, s_a] * q_a[None, :] + dM0[:, s_b] * q_b[None, :]
    mag = (np.abs(P) + np.abs(M))[:, s_a] * np.abs(q_a)[None, :] + (
        np.abs(P) + np.abs(M)
    )[:, s_b] * np.abs(q_b)[None, :]
    if t1 is None:
        return T2, 4 * EPS * mag
    ref = c_A * t1 - T2 / c_Phi
    return ref, 4 * EPS * (abs(c_A) * np.abs(t1) + mag / abs(c_Phi))


def _window(rng, n_rows=37, n_seg=61, n_k=3):
    # One plain wire's wings (s_b = s_a + 1) and a few junction-like ones
    # reaching far segments; rows drawn from both tables, a few with no
    # subtraction (grounded) and a few reading the knot table (chopped).
    n_basis = n_seg - 1
    s_a = np.arange(n_basis, dtype=np.int64)
    s_b = s_a + 1
    s_b[[5, 17]] = [40, 3]
    q_a = rng.standard_normal(n_basis) * 50
    q_b = rng.standard_normal(n_basis) * 50
    n_w = n_rows + 2
    tab = rng.standard_normal((n_w + n_k, n_seg)) + 1j * rng.standard_normal(
        (n_w + n_k, n_seg)
    )
    plus = rng.integers(0, n_w, n_rows)
    minus = rng.integers(0, n_w, n_rows)
    minus[[2, 9]] = -1
    plus[4], minus[11] = n_w, n_w + 2
    t1 = rng.standard_normal((n_rows, n_basis)) + 1j * rng.standard_normal(
        (n_rows, n_basis)
    )
    return tab[:n_w], tab[n_w:], plus, minus, s_a, s_b, q_a, q_b, t1


@pytest.mark.parametrize("layout", ["C", "F", "aliased"])
def test_the_fused_kernel_is_the_numpy_spelling(layout):
    rng = np.random.default_rng(1290)
    M0w, M0k, plus, minus, s_a, s_b, q_a, q_b, t1 = _window(rng)
    omega = 2 * np.pi * 7e6
    c_A = 1j * omega * 4e-7 * np.pi
    c_Phi = 1j * omega * 8.854187817e-12
    assert c_A.real == 0.0 and c_Phi.real == 0.0
    ref, bound = _numpy_spelling(
        t1, np.vstack([M0w, M0k]), plus, minus, s_a, s_b, q_a, q_b, c_A, c_Phi
    )
    if layout == "aliased":
        # The numpy T1 branch builds t1 where the rows land.
        out = t1.copy()
        t1_in = out
    else:
        # "F" is a row window of a column-major Z, the block's own layout.
        Z = np.zeros((t1.shape[0] + 9, t1.shape[1]), np.complex128, order=layout)
        out = Z[4 : 4 + t1.shape[0]]
        t1_in = t1
    _razor._acc.razor_t2_rows(
        out, t1_in, M0w, M0k, plus, minus, s_a, s_b, q_a, q_b, c_A.imag, c_Phi.imag
    )
    assert np.all(np.abs(out.real - ref.real) <= bound)
    assert np.all(np.abs(out.imag - ref.imag) <= bound)
    if layout != "aliased":
        # Written through its strides, in place, and nowhere else.
        assert not Z[:4].any() and not Z[4 + t1.shape[0] :].any()


def test_the_t2_only_kernel_is_the_numpy_spelling():
    rng = np.random.default_rng(1291)
    M0w, M0k, plus, minus, s_a, s_b, q_a, q_b, _t1 = _window(rng)
    ref, bound = _numpy_spelling(
        None, np.vstack([M0w, M0k]), plus, minus, s_a, s_b, q_a, q_b, None, None
    )
    out = np.empty_like(ref)
    _razor._acc.razor_t2_rows(
        out, None, M0w, M0k, plus, minus, s_a, s_b, q_a, q_b, 0.0, 0.0
    )
    assert np.all(np.abs(out.real - ref.real) <= bound)
    assert np.all(np.abs(out.imag - ref.imag) <= bound)


def test_the_kernel_refuses_what_it_cannot_write_in_place():
    rng = np.random.default_rng(3)
    M0w, M0k, plus, minus, s_a, s_b, q_a, q_b, t1 = _window(rng)
    args = (M0w, M0k, plus, minus, s_a, s_b, q_a, q_b, 1.0, 1.0)
    with pytest.raises(ValueError, match="complex128"):
        _razor._acc.razor_t2_rows(t1.astype(np.complex64), t1, *args)
    ro = t1.copy()
    ro.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        _razor._acc.razor_t2_rows(ro, t1, *args)
    bad = plus.copy()
    bad[0] = M0w.shape[0] + M0k.shape[0]
    with pytest.raises(ValueError, match="out of range"):
        _razor._acc.razor_t2_rows(
            t1.copy(), t1, M0w, M0k, bad, minus, s_a, s_b, q_a, q_b, 1.0, 1.0
        )


# ------------------------------------------------------------ whole fills


def _dipole(n, *, ground_model=None, contact=False):
    z0 = 0.0 if contact else 1.0
    deck = dict(
        wires=[np.array([[0.0, 0.0, z0], [0.0, 0.0, z0 + LAM / 4]])],
        n_per_edge_per_wire=[[n]],
        wire_radius=0.005,
        wavelength=LAM,
        feeds=[(0, 0.0 if contact else LAM / 8, 1 + 0j)],
    )
    if ground_model == "pec":
        deck.update(ground_z=0.0)
    elif ground_model is not None:
        deck.update(ground_z=0.0, ground_eps=(13.0, 0.005), ground_model=ground_model)
    return deck


def _free_space_crossing():
    deck = crossing_deck(1)
    for key in ("ground_z", "ground_eps", "ground_model"):
        deck.pop(key)
    return deck


DECKS = {
    "free space": lambda: _dipole(24),
    "contact / PEC": lambda: _dipole(24, ground_model="pec", contact=True),
    "contact / sommerfeld": lambda: _dipole(
        24, ground_model="sommerfeld", contact=True
    ),
    "elevated / refl-coef": lambda: _dipole(24, ground_model="refl-coef"),
    "buried crossing": lambda: crossing_deck(1),
}


class _Counter:
    """The accelerator module, with `razor_t2_rows` counted by mode."""

    def __init__(self, acc):
        self._acc = acc
        self.calls = {"fused": 0, "t2-only": 0}

    def __getattr__(self, name):
        return getattr(self._acc, name)

    def razor_t2_rows(self, out, t1, *rest):
        self.calls["fused" if t1 is not None else "t2-only"] += 1
        return self._acc.razor_t2_rows(out, t1, *rest)


def _solved_fill(monkeypatch, deck, *, accel, lane):
    """The matrix the solve factors (copied before `overwrite_a`), and the
    kernel's call counts, with the kernel on or off."""
    monkeypatch.setattr(_razor, "_HAVE_RAZOR_T2_ACCEL", accel)
    counter = _Counter(_razor._acc)
    monkeypatch.setattr(_razor, "_acc", counter)
    seen = []
    lu_factor = scipy.linalg.lu_factor

    def spy(a, *args, **kw):
        seen.append(np.array(a, copy=True))
        return lu_factor(a, *args, **kw)

    monkeypatch.setattr(scipy.linalg, "lu_factor", spy)
    z, _ = RazorSolver(**deck, **lane).compute_impedance()
    monkeypatch.undo()
    return seen, complex(np.ravel(z)[0]), counter.calls


LANES = {"nec5": {"nec5_quadrature": True}, "gauss-legendre": {}}


@pytest.mark.parametrize("lane", sorted(LANES))
@pytest.mark.parametrize("name", sorted(DECKS))
def test_the_fill_is_the_numpy_routes(monkeypatch, name, lane):
    deck = DECKS[name]()
    ref, z_ref, off = _solved_fill(monkeypatch, deck, accel=False, lane=LANES[lane])
    got, z_got, on = _solved_fill(monkeypatch, deck, accel=True, lane=LANES[lane])
    assert off == {"fused": 0, "t2-only": 0}
    assert on["fused"] > 0
    if name == "buried crossing":
        assert on["t2-only"] > 0  # the below medium's complex eps
    assert len(ref) == len(got) >= 1
    for a, b in zip(ref, got, strict=True):
        scale = np.abs(a).max()
        assert np.abs(a - b).max() <= BAR_ROUTE * scale
    assert abs(z_got - z_ref) <= 1e-9 * abs(z_ref)


def test_a_chopped_fill_is_the_numpy_routes(monkeypatch):
    """`_t2_tables`' chopped rows, both sides, through the prepared seam the
    crossing assembly uses (momwire#813's `_fill`)."""

    def fill(accel):
        monkeypatch.setattr(_razor, "_HAVE_RAZOR_T2_ACCEL", accel)
        counter = _Counter(_razor._acc)
        monkeypatch.setattr(_razor, "_acc", counter)
        rs = RazorSolver(**_free_space_crossing(), n_qp_path=8)
        geom = rs._build_geometry()
        jn = geom["n_basis_total"] - 1
        out = [
            rs._assemble_Z_from_prepared(
                geom, rs._assemble_Z_prepare(geom, chop={jn: side}), rs.k, rs.omega
            )
            for side in ("A", "B")
        ]
        monkeypatch.undo()
        return out, counter.calls

    ref, off = fill(False)
    got, on = fill(True)
    assert off["fused"] == 0 and on["fused"] > 0
    for a, b in zip(ref, got, strict=True):
        assert np.abs(a - b).max() <= BAR_ROUTE * np.abs(a).max()
