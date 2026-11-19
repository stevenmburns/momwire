"""The C++ route's transmitted anchors (momwire#1847, the opt-in feature
`transmitted_anchors`) against Python's, which stays the reference.

`somm_trans::anchored_grid` (`_somm_transmitted_core.cpp`) ports
`_transmitted_anchors.get_grid`: the lattice fixed in metres, the eight
anchors at F_REF * 1.1**j, their degree-7 blend in ln f, and that blend
resampled (ten Lagrange points per axis) onto the nodes the exact grid has at
the request (momwire#2022). bs2's buried fill
and the point-matched sinusoidal lane read it when the feature is requested,
so the executables can carry it once its reach is raised; today no
executable requests it (no `EXE_FEATURES` row).

Bit for bit, or within the declared drift, and why. Every transcendental on
the lattice is libm's in C++, and Python's goes through numpy in two places
that matter: `np.exp` (the z' ladder, the fill's R nodes) and `np.log` (the
served ladder's resampling targets), and `np.sin` / `np.cos` of the fill's
theta nodes. And the resampling's sums go through numpy's BLAS. The copies are
the same bits exactly when all three of these hold on the box, and the gates
probe each:
- numpy's float64 exp and log are libm's (every x86-64 dispatch below
  AVX-512; numpy's AVX-512 exp and log differ in the last bit on a few per
  cent of arguments);
- the sine and cosine the C++ fill takes are numpy's. gcc -O3 fuses the
  fill's `sin(th)` and `cos(th)` of one argument into one `sincos` call on
  Linux, and glibc's `sincos` is not always `sin` and `cos`: on glibc 2.35
  (Haswell) its sine differs from `sin` in the last bit on ~0.1 % of
  arguments in [0, 1.6), on glibc 2.39 (Rocketlake) on none of them. The
  shared exact-grid fill (`fill_nodes`) has the same property, so this is not
  the anchors' own; un-fusing it would move the exact grid's default-path
  bits on such a box, which is not this change's call;
- numpy's einsum sums each resampled target in index order, each product
  fused into the running sum from +0, as the C++ does (`EINSUM_IN_ORDER`).
  It hands each axis to its BLAS's `zgemm`, and x86-64 OpenBLAS's kernels sum
  that way; another BLAS (macOS's numpy) may block or reorder the sum. The
  probe runs a small einsum against that sum spelled in exact arithmetic.
Where the first two fail the two blends agree to 2e-13 of scale (1.8e-13 measured
with numpy's AVX-512 on Rocketlake, 2.6e-16 on Haswell), and the gates hold
the feature's declared drift, 3e-8. The test
session turns numpy's AVX-512 off (tests/conftest.py), so on x86-64 the first
holds under pytest and the `sincos` probe is the one that decides: the gates
run bit for bit on Rocketlake (glibc 2.39) and within the drift on Haswell.
Where only the third fails, the same sums in another order: the resampling's
own gates then hold the rounding bound of a sum in any order,
2 gamma_n (|W_z| x |W_r| x |W_t|) |V| per entry with n the stencil points
summed plus one per axis (`_resampling_bound`), and everything downstream
holds the drift. On the anchored grids here that bound is 1.1e-14 to 5e-14
of scale, so the documented 2e-13 holds and is checked; on the random tables
it reaches 4.1e-13, since a clamped stencil extrapolates and its |W| grows,
so there the per-entry bound alone is the gate. Every sum taken in reverse
order (a stand-in for another BLAS) moved 1.3e-15 of scale at most, a tenth
of the bound or less.

What is held:
- the served grid, the blend it was resampled from, the blend's lattice and
  each of its anchors, single- and multi-depth;
- the blend's own arithmetic, numpy's over the C++ anchors, bit for bit
  everywhere (no transcendental in it);
- the resampling's arithmetic, `_on_exact_nodes` over the C++ blend, bit for
  bit wherever numpy's einsum sums in order and its log is libm's (the served
  ladder's targets are its only transcendental), and on random tables over
  both contraction orders; the matrix's zeros outside a stencil add nothing;
- the declines, where Python returns None, the wide outer radius's included;
- a sweep sharing its anchors;
- a full anchor store that evicts an anchor of the stencil being summed;
- bs2's buried Z and the sinusoidal printout with the feature on, against
  Python's with it on, and different from the C++ route with it off (so the
  anchored route is what ran).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib.util
import math
import sys
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from momwire import _accel
from momwire import _sommerfeld as som
from momwire import _sommerfeld_transmitted as trans
from momwire import _transmitted_anchors as anchors
from momwire._constants import C_LIGHT
from momwire._ground_refl import eps_tilde

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"

acc = _accel.acc
pytestmark = [
    pytest.mark.reach(1),
    pytest.mark.skipif(
        acc is None or not hasattr(acc, "transmitted_anchors_grid_core"),
        reason="needs the extension's anchored-grid hook",
    ),
]

EPS0 = 8.8541878128e-12
SOIL = (13.0, 0.005)
F0 = 7.1e6
R_MIN, R_MAX, DEPTH = 0.2, 2.0, 0.15
DRIFT = 3e-8  # the registry's `drift` for transmitted_anchors


def _numpy_is_libm() -> bool:
    """Whether numpy's float64 exp and log are libm's here (every x86-64
    dispatch below AVX-512): then the two copies agree bit for bit."""
    x = np.random.default_rng(1847).uniform(-30.0, 30.0, 4096)
    y = np.exp(x)
    if not np.array_equal(y, np.array([math.exp(v) for v in x])):
        return False
    return np.array_equal(np.log(y), np.array([math.log(v) for v in y]))


def _fill_trig_is_numpys() -> bool:
    """Whether the sine and cosine the C++ fill takes of its theta nodes are
    numpy's here. On Linux gcc fuses the fill's two calls into libm's
    `sincos`, which glibc 2.35 does not keep equal to `sin` (glibc 2.39
    does); elsewhere the C++ makes two libm calls (`_libm_complex_inline.h`
    keeps them apart on Apple), held against numpy's here too. Measured over
    the theta range [0, pi/2], densely: a box that fails anywhere there is
    gated by the drift."""
    x = np.linspace(0.0, 0.5 * math.pi, 100001)
    ns, nc = np.sin(x), np.cos(x)
    if not sys.platform.startswith("linux"):
        return np.array_equal(ns, [math.sin(v) for v in x]) and np.array_equal(
            nc, [math.cos(v) for v in x]
        )
    try:
        libm = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
        sincos = libm.sincos
    except (OSError, AttributeError):
        return False
    dp = ctypes.POINTER(ctypes.c_double)
    sincos.argtypes = [ctypes.c_double, dp, dp]
    sincos.restype = None
    s, c = ctypes.c_double(), ctypes.c_double()
    for i, v in enumerate(x.tolist()):
        sincos(v, ctypes.byref(s), ctypes.byref(c))
        if s.value != ns[i] or c.value != nc[i]:
            return False
    return True


def _bits(a):
    return np.ascontiguousarray(a, dtype=np.complex128).view(np.uint64)


def _fma(a, b, c):
    """a * b + c rounded once (float of an exact Fraction is correctly
    rounded)."""
    return float(Fraction(a) * Fraction(b) + Fraction(c))


def _in_order(v, w, axis):
    """Contract `axis` of complex `v` with the weight matrix `w` (targets,
    n): each target's nonzero products in index order, fused into the running
    sum from +0, real and imaginary parts apart."""
    v = np.moveaxis(v, axis, -1)
    out = np.empty(v.shape[:-1] + (w.shape[0],), dtype=np.complex128)
    for idx in np.ndindex(v.shape[:-1]):
        for t in range(w.shape[0]):
            re = im = 0.0
            for k in np.nonzero(w[t])[0]:
                x = complex(v[idx + (k,)])
                re = _fma(x.real, float(w[t, k]), re)
                im = _fma(x.imag, float(w[t, k]), im)
            out[idx + (t,)] = complex(re, im)
    return np.moveaxis(out, -1, axis)


def _einsum_is_in_order() -> bool:
    """Whether `_on_exact_nodes`' einsums sum here as the C++ resampling does:
    both in-plane orders (R first, theta first) and the z' step, on small
    random tables through `_resample_matrix`'s ten-point stencils."""
    rng = np.random.default_rng(2022)
    for n_r, n_t, a_r, a_t in ((14, 12, 6, 12), (14, 12, 14, 3)):
        v = rng.standard_normal((1, 5, n_r, n_t)) + 1j * rng.standard_normal(
            (1, 5, n_r, n_t)
        )
        w_r = anchors._resample_matrix(0.0, 1.0, n_r, rng.uniform(0, n_r - 1, a_r))
        w_t = anchors._resample_matrix(0.0, 1.0, n_t, rng.uniform(0, n_t - 1, a_t))
        got = np.einsum("zsrt,ar,bt->zsab", v, w_r, w_t, optimize=True)
        first = np.einsum_path("zsrt,ar,bt->zsab", v, w_r, w_t, optimize=True)[0][1]
        if first == (0, 1):
            want = _in_order(_in_order(v, w_r, 2), w_t, 3)
        else:
            want = _in_order(_in_order(v, w_t, 3), w_r, 2)
        if not np.array_equal(_bits(got), _bits(want)):
            return False
    v = rng.standard_normal((12, 5, 2, 2)) + 1j * rng.standard_normal((12, 5, 2, 2))
    w_z = anchors._resample_matrix(0.0, 1.0, 12, rng.uniform(0, 11, 9))
    got = np.einsum("zsab,cz->csab", v, w_z, optimize=True)
    return np.array_equal(_bits(got), _bits(_in_order(v, w_z, 0)))


NUMPY_IS_LIBM = _numpy_is_libm()
EINSUM_IN_ORDER = _einsum_is_in_order()
# The anchors' (and so the blends') values are the same bits on both sides,
# and so the served grid's where the resampling sums in the same order.
SAME_BLEND_BITS = NUMPY_IS_LIBM and _fill_trig_is_numpys()
SAME_BITS = SAME_BLEND_BITS and EINSUM_IN_ORDER


def _agree(a, b, what, same=None):
    """Bit for bit where the box agrees (`same`, `SAME_BITS` unless given),
    else within the declared drift of each surface's scale."""
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    assert a.shape == b.shape, what
    if SAME_BITS if same is None else same:
        assert np.array_equal(_bits(a), _bits(b)), what
        return
    scale = np.abs(b).reshape(b.shape[:2] + (-1,)).max(-1)
    err = np.abs(a - b).reshape(scale.shape + (-1,)).max(-1)
    assert float(np.max(err / scale)) < DRIFT, what


def _req(f_hz, r_max=R_MAX, zp=(DEPTH, DEPTH), r_min=R_MIN, soil=SOIL):
    om = 2.0 * math.pi * f_hz
    return dict(
        eps_t=eps_tilde(soil, om, EPS0),
        k2=om / C_LIGHT,
        r_max=r_max,
        zp_min=zp[0],
        zp_max=zp[1],
        omega=om,
        r_min=r_min,
    )


def _cxx(r, parts=False):
    return acc.transmitted_anchors_grid_core(
        r["eps_t"],
        r["k2"],
        r["r_max"],
        r["zp_min"],
        r["zp_max"],
        r["omega"],
        som._MU0,
        r_min=r["r_min"],
        parts=parts,
    )


@pytest.fixture
def fresh():
    anchors.reset()
    acc.transmitted_anchors_stats(True)
    yield
    anchors.reset()
    acc.transmitted_anchors_stats(True)


_STATE = (
    "r_min",
    "r_max",
    "lnr0",
    "dlnr",
    "n_r",
    "th_min",
    "th0",
    "dth",
    "n_th",
    "lam_p",
    "lam_m",
    "r_cap",
    "k2",
    "omega",
    "eps_t",
    "k_m",
)


def _same_state(g, c, what):
    """The axes and the request's own constants: the same floats."""
    for k in _STATE:
        assert getattr(g, k) == c[k], (what, k)
    if NUMPY_IS_LIBM:  # the ladder is exp's alone
        assert np.array_equal(g.zp_nodes, c["zp_nodes"]), what
        assert (g.zp_min, g.zp_max, g.lnz0, g.dlnz) == (
            c["zp_min"],
            c["zp_max"],
            c["lnz0"],
            c["dlnz"],
        ), what
    else:
        assert np.allclose(g.zp_nodes, c["zp_nodes"], rtol=1e-15, atol=0.0), what


def _as_blend(c, r):
    """The C++ blend as `_on_exact_nodes` reads a Python one."""
    b = object.__new__(trans.TransmittedGrid)
    lat = anchors._Lattice(c["r_min"], c["r_max"], c["zp_nodes"])
    anchors._set_lattice(b, lat, r["eps_t"], r["k2"], r["omega"], som._MU0)
    b._vals = c["vals"]
    return b


# u, the unit roundoff of binary64
_U = 2.0**-53


def _resampling_bound(resample):
    """Each served entry's bound on |a - b| for two evaluations of the same
    resampling summed in any orders: `resample(vals_abs, abs_weights)` runs
    `_on_exact_nodes` with |V| and |W|. A sum of n products, each rounded
    or fused, in any order, is within gamma_n = n u / (1 - n u) of the sum of
    their magnitudes; the axes nest, so their n add (gamma_m + gamma_n +
    gamma_m gamma_n <= gamma_(m + n)), and two evaluations err apart by twice
    that."""
    orig = anchors._resample_matrix
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(anchors, "_resample_matrix", lambda *a: np.abs(orig(*a)))
        mag = resample()
    n = 3 * (anchors._RESAMPLE_POINTS + 1)
    gamma = n * _U / (1.0 - n * _U)
    # 1.01: |V| x |W| itself rounds, a relative u per step
    return 2.02 * gamma * mag


def _close_by_the_bound(got, want, bound, what, cap=None):
    """|got - want| within `bound` per entry, real and imaginary parts apart;
    with `cap`, the bound itself under that fraction of each surface's
    scale."""
    got = np.asarray(got, dtype=np.complex128)
    want = np.asarray(want, dtype=np.complex128)
    assert got.shape == want.shape, what
    assert np.all(np.abs(got.real - want.real) <= bound), what
    assert np.all(np.abs(got.imag - want.imag) <= bound), what
    if cap is not None:
        scale = np.abs(want).reshape(want.shape[:2] + (-1,)).max(-1)
        worst = np.asarray(bound).reshape(scale.shape + (-1,)).max(-1) / scale
        assert float(np.max(worst)) < cap, (what, float(np.max(worst)))


def _check_resampling(c, r):
    """`_on_exact_nodes` over the C++ blend against the C++ served grid, and
    the contraction order einsum's greedy path takes against the C++ one."""
    py = anchors._on_exact_nodes(
        _as_blend(c["blend"], r),
        r["k2"],
        r["r_max"],
        r["zp_min"],
        r["zp_max"],
        r["r_min"],
    )
    _same_state(py, c, "served nodes")
    z, _, n_r, n_th = c["blend"]["vals"].shape
    path = np.einsum_path(
        "zsrt,ar,bt->zsab",
        np.empty((z, 5, n_r, n_th)),
        np.empty((c["n_r"], n_r)),
        np.empty((c["n_th"], n_th)),
        optimize=True,
    )[0]
    assert c["r_first"] == (path[1] == (0, 1)), path
    if EINSUM_IN_ORDER and (NUMPY_IS_LIBM or z == 1):
        assert np.array_equal(_bits(py._vals), _bits(c["vals"]))
    elif NUMPY_IS_LIBM or z == 1:
        mag = _as_blend(c["blend"], r)
        mag._vals = np.abs(mag._vals)
        bound = _resampling_bound(
            lambda: (
                anchors._on_exact_nodes(
                    mag, r["k2"], r["r_max"], r["zp_min"], r["zp_max"], r["r_min"]
                )._vals
            )
        )
        _close_by_the_bound(c["vals"], py._vals, bound, "resampling", cap=2e-13)
    else:
        _agree(c["vals"], py._vals, "resampling")


def _check_blend(f_hz, r):
    g = anchors.get_grid(mu=som._MU0, **r)
    c = _cxx(r, parts=True)
    assert g is not None and "declined" not in c, c
    _same_state(g, c, "served")
    # the blend's lattice: the family's, fixed in metres
    js, u = anchors._stencil(f_hz)
    eps_r, q = anchors._soil(r["eps_t"], r["omega"])
    lat = anchors._lattice(
        f_hz, js, r["r_max"], r["zp_min"], r["zp_max"], r["r_min"], eps_r, q
    )
    blend = anchors._blend(
        lat, js, u, r["eps_t"], r["k2"], r["omega"], som._MU0, eps_r, q, 1e-9, 0
    )
    _same_state(blend, c["blend"], "blend")
    # the anchors, each the exact fill at its own frequency on the lattice
    lat = anchors._Lattice(blend.r_min, blend.r_max, blend.zp_nodes)
    assert len(c["anchors"]) == anchors.ANCHOR_POINTS
    for j, a in zip(js, c["anchors"], strict=True):
        key = ("below-above-anchor", eps_r, q, int(j), lat.key(), float(som._MU0))
        pa = anchors._ANCHORS[key]
        assert (a["eps_t"], a["k2"], a["omega"]) == (pa.eps_t, pa.k2, pa.omega), j
        _agree(a["vals"], pa._vals, f"anchor {j}", SAME_BLEND_BITS)
    # the blend's arithmetic alone, everywhere: numpy's over the C++ anchors
    v = None
    for a, w in zip(c["anchors"], anchors._weights(u, js), strict=True):
        v = a["vals"] * w if v is None else v + a["vals"] * w
    assert np.array_equal(_bits(v), _bits(c["blend"]["vals"]))
    _agree(c["blend"]["vals"], blend._vals, "blend", SAME_BLEND_BITS)
    _check_resampling(c, r)
    _agree(c["vals"], g._vals, "served")


def test_the_cxx_blend_is_pythons_on_one_rung(fresh):
    _check_blend(F0, _req(F0))


@pytest.mark.slow  # eight 24-rung anchor fills on each side, ~20 s each
def test_the_cxx_blend_is_pythons_on_a_ladder(fresh):
    f = 7.3e6
    _check_blend(f, _req(f, r_max=3.0, zp=(0.1, 0.6)))


@pytest.mark.parametrize(
    "case",
    [
        # 2 lambda_p is 84 m at 7.1 MHz but 58 m at the stencil's top anchor
        dict(r_max=70.0),
        # under the radial floor at the stencil's anchors
        dict(r_min=1e-9),
        # 5 cm deep and 15 m out: the lattice's outer radius, three rows past
        # the request, buckets to 22.7 m, whose grazing floor (0.027 deg) is
        # above the exact grid's (0 at its 18.2 m), so the exact grid serves;
        # before momwire#2022 the lattice stopped at 18.2 m and served
        dict(r_max=15.0, zp=(0.05, 0.05)),
    ],
    ids=["r_cap", "r_min_floor", "outer_rows_floor"],
)
def test_the_cxx_route_declines_where_python_does(fresh, case):
    r = _req(F0, **case)
    assert anchors.get_grid(mu=som._MU0, **r) is None
    c = _cxx(r)
    assert set(c) == {"declined"}, sorted(c)
    assert acc.transmitted_anchors_stats()["declined"] == 1


def test_a_soil_with_a_growing_loss_is_declined(fresh):
    r = _req(F0)
    r["eps_t"] = complex(r["eps_t"].real, abs(r["eps_t"].imag))
    assert anchors.get_grid(mu=som._MU0, **r) is None
    assert set(_cxx(r)) == {"declined"}


def test_a_sweep_shares_its_cxx_anchors(fresh):
    for f in (F0, F0 * 1.02, F0 * 1.05):
        assert "declined" not in _cxx(_req(f))
    st = acc.transmitted_anchors_stats()
    # 7.1 -> 7.46 MHz crosses at most one anchor step
    assert st["blends"] == 3 and st["anchor_fills"] <= anchors.ANCHOR_POINTS + 1, st
    nodes = st["nodes_filled"]
    assert "declined" not in _cxx(_req(F0))
    st = acc.transmitted_anchors_stats()
    assert st["blends"] == 3 and st["nodes_filled"] == nodes, st


def test_a_full_anchor_store_never_frees_a_stencil_anchor(fresh):
    """The anchor store is a FIFO (Python's `_ANCHORS_MAX`, 96 here capped
    to 16 so two soils fill it): a request whose stencil finds anchors J..J+6
    and must add J+7 into a full store evicts the oldest entry, which is J,
    the first anchor it already took. The blend must still sum J's values
    (Python sums each anchor as it fetches it), not the freed table's."""
    other = (5.0, 0.001)
    acc.transmitted_anchors_stats(True, anchors_max=16)
    for r in (
        _req(F0 / 1.1),  # SOIL's J-1 .. J+6
        _req(F0, soil=other),  # the other soil's J .. J+7: the store is full
        _req(F0 * 1.1, soil=other),  # its J+8 evicts SOIL's J-1
    ):
        assert "declined" not in _cxx(r)
    # SOIL at F0 finds J .. J+6; adding J+7 evicts J, the stencil's first
    c = _cxx(_req(F0), parts=True)
    st = acc.transmitted_anchors_stats()
    assert st["anchor_fills"] == 8 + 8 + 1 + 1 and st["blends"] == 4, st
    g = anchors.get_grid(mu=som._MU0, **_req(F0))
    js, u = anchors._stencil(F0)
    v = None
    for a, w in zip(c["anchors"], anchors._weights(u, js), strict=True):
        v = a["vals"] * w if v is None else v + a["vals"] * w
    assert np.array_equal(_bits(v), _bits(c["blend"]["vals"]))
    _check_resampling(c, _req(F0))
    _agree(c["vals"], g._vals, "served over an evicted anchor")


class _Served:
    """`_Nodes`' shape, on axes the test spells."""

    def __init__(self, lnr0, n_r, th_min, dth, n_th, zp_nodes):
        self.lnr0, self.n_r = lnr0, n_r
        self.th_min, self.dth, self.n_th = th_min, dth, n_th
        self.zp_nodes = np.asarray(zp_nodes, dtype=float)
        self.r_min = math.exp(lnr0)
        self.r_max = math.exp(lnr0 + trans._R_STEP_LOG * (n_r - 1))


@pytest.mark.parametrize(
    "case",
    [
        # (lattice r_max, rungs, served row offset in rows, served rows,
        # served theta points, served rungs): the served theta axis decides
        # which axis einsum contracts first (few points: theta)
        (10.0, 1, 2.37, 24, 31, None),
        (10.0, 1, 4.81, 28, 5, None),
        (6.0, 9, 1.5, 12, 30, 12),
        (6.0, 24, -0.6, 22, 6, 30),
    ],
    ids=["r_first", "theta_first", "nine_rungs", "theta_first_rungs_clamped"],
)
def test_the_cxx_resampling_is_pythons_on_random_tables(monkeypatch, case):
    """`_on_exact_nodes` (Python's `_resample_matrix` weights and einsum)
    against the C++ resampling alone, on the same random table: no
    transcendental but the served ladder's `np.log` targets."""
    r_max, rungs, offset, n_r, n_th, served_rungs = case
    r = _req(F0)
    zp = (
        np.exp(np.linspace(math.log(0.1), math.log(0.6), rungs))
        if rungs > 1
        else [DEPTH]
    )
    lat = anchors._Lattice(0.2, r_max, zp)
    rng = np.random.default_rng(2022 + rungs)
    shape = (lat.zp_nodes.size, 5, lat.n_r, lat.n_th)
    vals = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    blend = object.__new__(trans.TransmittedGrid)
    anchors._set_lattice(blend, lat, r["eps_t"], r["k2"], r["omega"], som._MU0)
    blend._vals = vals
    th_min = lat.th_min + 0.3 * lat.dth
    dth = (0.5 * math.pi - th_min) / (n_th - 1)
    served_zp = (
        np.exp(np.linspace(math.log(0.11), math.log(0.62), served_rungs))
        if served_rungs
        else [DEPTH]
    )
    nodes = _Served(
        lat.lnr0 + trans._R_STEP_LOG * offset, n_r, th_min, dth, n_th, served_zp
    )
    monkeypatch.setattr(anchors, "_Nodes", lambda *_a, **_k: nodes)
    py = anchors._on_exact_nodes(blend, r["k2"], r_max, 0.1, 0.6, None)
    got, r_first = acc.transmitted_anchors_resample(
        vals,
        blend.lnr0,
        blend.th0,
        blend.dth,
        list(blend.zp_nodes),
        blend.lnz0,
        blend.dlnz,
        nodes.lnr0,
        nodes.n_r,
        nodes.th_min,
        nodes.dth,
        nodes.n_th,
        list(nodes.zp_nodes),
    )
    path = np.einsum_path(
        "zsrt,ar,bt->zsab",
        vals,
        np.empty((n_r, lat.n_r)),
        np.empty((n_th, lat.n_th)),
        optimize=True,
    )[0]
    assert r_first == (path[1] == (0, 1)), path
    assert r_first == (case[4] > 10)
    if EINSUM_IN_ORDER and (NUMPY_IS_LIBM or not served_rungs):
        assert np.array_equal(_bits(py._vals), _bits(got))
    elif NUMPY_IS_LIBM or not served_rungs:
        blend._vals = np.abs(vals)
        bound = _resampling_bound(
            lambda: anchors._on_exact_nodes(blend, r["k2"], r_max, 0.1, 0.6, None)._vals
        )
        _close_by_the_bound(got, py._vals, bound, "resampling")
    else:
        _agree(got, py._vals, "resampling")


# ---- the fills that read it ---------------------------------------------------


FEATURE = ["transmitted_anchors"]  # requested per call: `ni_eval::FeatureRequest`


@pytest.mark.parametrize(
    "deck", ["m2_dipole_over_buried.nec", "m3_two_radii.nec"], ids=lambda s: s[:2]
)
def test_the_bs2_buried_fill_with_anchors_is_pythons(fresh, deck):
    """`bspline_buried_stages` (the executables' `fill_buried`) with the
    feature requested, against `BSplineSolver(transmitted_anchors=True)`'s
    buried operator; and the C++ route's Z with it off differs, by the
    anchors' interpolation (~1e-10), so the anchored grid is what ran."""
    if not hasattr(acc, "bspline_buried_stages"):
        pytest.skip("needs the extension's buried stages hook")
    sys.path.insert(0, str(ROOT / "tests"))
    try:
        from test_bspline_buried_stages_1517 import _blas
    finally:
        sys.path.pop(0)
    from momwire import BSplineSolver

    blas = _blas()
    raw = (FIX / "transmitted_mixed" / deck).read_bytes()
    entry = acc.bspline_front_geometry(raw)[0]
    on = acc.bspline_buried_stages(raw, blas, features=FEATURE)[0]
    assert "stopped" not in on, on.get("stopped")
    st = acc.transmitted_anchors_stats()
    assert st["blends"] == 1 and st["anchor_fills"] == anchors.ANCHOR_POINTS, st
    off = acc.bspline_buried_stages(raw, blas)[0]
    assert acc.transmitted_anchors_stats()["blends"] == 1

    kw = dict(entry["kwargs"], transmitted_anchors=True)
    s = BSplineSolver(**kw)
    s._apply_loading = lambda Z, **_k: Z
    g = s._build_geometry()
    supp, polys, *_ = s._build_basis_polynomials(g)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Zp = np.ascontiguousarray(s._compute_Z_operator(g, supp, polys))
    assert anchors.STATS["blends"] == 1
    Zc = np.asarray(on["Z"], dtype=np.complex128).reshape(Zp.shape)
    Zx = np.asarray(off["Z"], dtype=np.complex128).reshape(Zp.shape)
    if SAME_BITS:
        assert np.array_equal(_bits(Zc), _bits(Zp))
    else:
        assert float(np.max(np.abs(Zc - Zp))) < DRIFT * float(np.max(np.abs(Zp)))
    assert not np.array_equal(_bits(Zc), _bits(Zx))
    assert float(np.max(np.abs(Zc - Zx))) < DRIFT * float(np.max(np.abs(Zx)))


def _sin_printout(rc, text, blas, features=()):
    """`rc.cxx_printout` with `features` requested for this run only."""
    version, variant, marker = rc.pc.stamp()
    return acc.sinusoidal_nec4_printout(
        text.encode("latin-1"), blas, version, variant, marker, list(features)
    )


def _nec4_compare():
    spec = importlib.util.spec_from_file_location(
        "core_split_nec4_run_compare",
        ROOT / "scripts" / "core_split" / "nec4_run_compare.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fill_compare():
    spec = importlib.util.spec_from_file_location(
        "core_split_sinusoidal_fill_compare",
        ROOT / "scripts" / "core_split" / "sinusoidal_fill_compare.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# A vertical over a separate buried wire (test_nec4_run_core's mixed case),
# swept over three frequencies so the three blends share eight anchors.
MIXED_SWEEP = (
    "CM run\nCE\nGW 1 10 0 0 .5 0 0 10 .001\nGW 2 10 -5 1 -.1 5 1 -.1 .001\n"
    "GE -1\nGN 2 0 0 0 13 .005\nFR 0 3 0 0 7.0 0.1\nEX 0 1 1 0 1\nXQ\nEN\n"
)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="the NEC-4.2 reader's rules")
def test_the_sinusoidal_printout_with_anchors_is_pythons(fresh, monkeypatch):
    """The point-matched lane's C++ printout with the feature requested
    against the Python slot's with it on, past line 2 (Python stamps the
    feature there; the hook requests it unstamped). Served on the exact
    grid's nodes (momwire#2022) the feature moves this deck's Z by ~1e-10, so
    its five-figure printout reads the same on and off; the fill dump (G, the
    port coefficients and Y, bit for bit) is what shows the anchored route
    ran: on is Python's on, and is not off."""
    if not hasattr(acc, "sinusoidal_nec4_printout"):
        pytest.skip("no compiled accelerator with the NEC-4.2 run half")
    rc = _nec4_compare()
    fc = _fill_compare()
    try:
        blas = rc.blas_addresses("sinusoidal")
        fill_blas = fc.blas_addresses()
    except (IndexError, AttributeError, OSError):
        pytest.skip("no bundled OpenBLAS beside numpy / SciPy (not a wheel install)")
    from threadpoolctl import threadpool_limits

    monkeypatch.setenv("MOMWIRE_BELOW_ANCHORS", "off")
    monkeypatch.delenv("MOMWIRE_X_TRANSMITTED_ANCHORS", raising=False)
    with threadpool_limits(limits=1):
        ok, off = _sin_printout(rc, MIXED_SWEEP, blas)
        assert ok, off
        assert acc.transmitted_anchors_stats()["blends"] == 0
        fill_off = fc.cxx_fill(MIXED_SWEEP, acc, fill_blas)
        assert acc.transmitted_anchors_stats()["blends"] == 0
        ok, on = _sin_printout(rc, MIXED_SWEEP, blas, FEATURE)
        assert ok, on
        st = acc.transmitted_anchors_stats()
        assert st["blends"] == 3 and st["anchor_fills"] == anchors.ANCHOR_POINTS, st
        fill_on = acc.sinusoidal_fill_dump(
            MIXED_SWEEP.encode("latin-1"), fill_blas, fc.WHOLE, FEATURE
        )
        # the request was for that run only
        assert fc.cxx_fill(MIXED_SWEEP, acc, fill_blas) == fill_off
        st = acc.transmitted_anchors_stats()
        assert st["blends"] == 3 and st["anchor_fills"] == anchors.ANCHOR_POINTS, st
        monkeypatch.setenv("MOMWIRE_X_TRANSMITTED_ANCHORS", "1")
        som._GRID_CACHE.clear()
        py = rc.py_printout(MIXED_SWEEP, "sinusoidal")
        assert anchors.STATS["blends"] == 3
        som._GRID_CACHE.clear()
        fill_py = fc.py_fill(MIXED_SWEEP)
    body = [rc.mask(t)[0].split("\n")[2:] for t in (py, on)]
    assert body[0] == body[1], rc.first_difference(
        "\n".join(body[0]), "\n".join(body[1])
    )
    if SAME_BITS:
        assert fill_on == fill_py, rc.first_difference(fill_py, fill_on)
    assert fill_on != fill_off
