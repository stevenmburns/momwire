"""The C++ route's transmitted anchors (momwire#1847, the opt-in feature
`transmitted_anchors`) against Python's, which stays the reference.

`somm_trans::anchored_grid` (`_somm_transmitted_core.cpp`) ports
`_transmitted_anchors.get_grid`: the lattice fixed in metres, the eight
anchors at F_REF * 1.1**j and their degree-7 blend in ln f. bs2's buried fill
and the point-matched sinusoidal lane read it when the feature is requested,
so the executables can carry it once its reach is raised; today no
executable requests it (no `EXE_FEATURES` row).

Bit for bit, or within the declared drift, and why. Every transcendental on
the lattice is libm's in C++, and Python's goes through numpy in two places
that matter: `np.exp` (the z' ladder, the fill's R nodes) and `np.sin` /
`np.cos` of the fill's theta nodes. The copies are the same bits exactly
when both of these hold on the box, and the gates probe both:
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
  bits on such a box, which is not this change's call.
Where either fails the two blends agree to ~1e-16 of scale (2.6e-16 measured
on Haswell), and the gates hold the feature's declared drift, 1e-8. The test
session turns numpy's AVX-512 off (tests/conftest.py), so on x86-64 the first
holds under pytest and the `sincos` probe is the one that decides: the gates
run bit for bit on Rocketlake (glibc 2.39) and within the drift on Haswell.

What is held:
- the blend, its lattice and each of its anchors, single- and multi-depth;
- the blend's own arithmetic, numpy's over the C++ anchors, bit for bit
  everywhere (no transcendental in it);
- the declines, where Python returns None;
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
from pathlib import Path

import numpy as np
import pytest

from momwire import _accel
from momwire import _sommerfeld as som
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
DRIFT = 1e-8  # the registry's `drift` for transmitted_anchors


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


NUMPY_IS_LIBM = _numpy_is_libm()
# The anchors' (and so the blends') values are the same bits on both sides.
SAME_BITS = NUMPY_IS_LIBM and _fill_trig_is_numpys()


def _bits(a):
    return np.ascontiguousarray(a, dtype=np.complex128).view(np.uint64)


def _agree(a, b, what):
    """Bit for bit where the box's transcendentals agree (`SAME_BITS`),
    else within the declared drift of each surface's scale."""
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    assert a.shape == b.shape, what
    if SAME_BITS:
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


def _check_blend(f_hz, r):
    g = anchors.get_grid(mu=som._MU0, **r)
    c = _cxx(r, parts=True)
    assert g is not None and "declined" not in c, c
    # the lattice and the request's own constants: the same floats
    for k in (
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
    ):
        assert getattr(g, k) == c[k], k
    if NUMPY_IS_LIBM:  # the ladder is exp's alone
        assert np.array_equal(g.zp_nodes, c["zp_nodes"])
        assert (g.zp_min, g.zp_max, g.lnz0, g.dlnz) == (
            c["zp_min"],
            c["zp_max"],
            c["lnz0"],
            c["dlnz"],
        )
    else:
        assert np.allclose(g.zp_nodes, c["zp_nodes"], rtol=1e-15, atol=0.0)
    # the anchors, each the exact fill at its own frequency on the lattice
    js, u = anchors._stencil(f_hz)
    eps_r, q = anchors._soil(r["eps_t"], r["omega"])
    lat = anchors._Lattice(g.r_min, g.r_max, g.zp_nodes)
    assert len(c["anchors"]) == anchors.ANCHOR_POINTS
    for j, a in zip(js, c["anchors"], strict=True):
        key = ("below-above-anchor", eps_r, q, int(j), lat.key(), float(som._MU0))
        pa = anchors._ANCHORS[key]
        assert (a["eps_t"], a["k2"], a["omega"]) == (pa.eps_t, pa.k2, pa.omega), j
        _agree(a["vals"], pa._vals, f"anchor {j}")
    # the blend's arithmetic alone, everywhere: numpy's over the C++ anchors
    v = None
    for a, w in zip(c["anchors"], anchors._weights(u, js), strict=True):
        v = a["vals"] * w if v is None else v + a["vals"] * w
    assert np.array_equal(_bits(v), _bits(c["vals"]))
    _agree(c["vals"], g._vals, "blend")


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
    ],
    ids=["r_cap", "r_min_floor"],
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
    assert np.array_equal(_bits(v), _bits(c["vals"]))
    _agree(c["vals"], g._vals, "blend over an evicted anchor")


# ---- the fills that read it ---------------------------------------------------


def _requested(on):
    acc.cxx_request_feature("transmitted_anchors", on)


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
    _requested(True)
    try:
        on = acc.bspline_buried_stages(raw, blas)[0]
    finally:
        _requested(False)
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


def _nec4_compare():
    spec = importlib.util.spec_from_file_location(
        "core_split_nec4_run_compare",
        ROOT / "scripts" / "core_split" / "nec4_run_compare.py",
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
    feature there; the hook requests it unstamped); and the C++ printout
    with it off moves digits in the currents, so the comparison is not
    blind to the feature."""
    if not hasattr(acc, "sinusoidal_nec4_printout"):
        pytest.skip("no compiled accelerator with the NEC-4.2 run half")
    rc = _nec4_compare()
    try:
        blas = rc.blas_addresses("sinusoidal")
    except (IndexError, AttributeError, OSError):
        pytest.skip("no bundled OpenBLAS beside numpy / SciPy (not a wheel install)")
    from threadpoolctl import threadpool_limits

    monkeypatch.setenv("MOMWIRE_BELOW_ANCHORS", "off")
    monkeypatch.delenv("MOMWIRE_X_TRANSMITTED_ANCHORS", raising=False)
    with threadpool_limits(limits=1):
        ok, off = rc.cxx_printout(MIXED_SWEEP, acc=acc, blas=blas, basis="sinusoidal")
        assert ok, off
        _requested(True)
        try:
            ok, on = rc.cxx_printout(
                MIXED_SWEEP, acc=acc, blas=blas, basis="sinusoidal"
            )
        finally:
            _requested(False)
        assert ok, on
        st = acc.transmitted_anchors_stats()
        assert st["blends"] == 3 and st["anchor_fills"] == anchors.ANCHOR_POINTS, st
        monkeypatch.setenv("MOMWIRE_X_TRANSMITTED_ANCHORS", "1")
        som._GRID_CACHE.clear()
        py = rc.py_printout(MIXED_SWEEP, "sinusoidal")
    assert anchors.STATS["blends"] == 3
    body = [rc.mask(t)[0].split("\n")[2:] for t in (py, on, off)]
    assert body[0] == body[1], rc.first_difference(
        "\n".join(body[0]), "\n".join(body[1])
    )
    assert body[1] != body[2]
