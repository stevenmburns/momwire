"""The point-matched sinusoidal fill's complex-k field tensor on the complex
Galerkin fill's staged rows (momwire#1849, the opt-in feature
`point_cplx_staged`).

What is held:

- off, nothing changes: the solver defaults to `sinusoidal_field_tensor_cplx`;
- the staged tensor is the scalar one to a few ulp of scale, all three shapes;
- a row is its observer's alone: any observer subset gives the full call's
  rows bit for bit (the sweep is padded per source tile);
- on, a buried crossing deck takes the staged kernel and its Z agrees with
  the flag off;
- the C++ route (the NEC-4.2 slot's sinusoidal run half, through the
  in-process hook with the feature requested) takes the staged kernel too,
  and prints the Python slot's bytes with the flag on, on a wholly-buried
  screen and a buried crossing deck; off, it never takes it.

Both shape sets: the folded cos k(xi) - 1 the staged rows assemble, and the
literal cosine the solver takes on any mesh with |k|Delta >= 1.5e-3.
"""

from __future__ import annotations

import importlib.util
import pathlib
import platform
import sys

import numpy as np
import pytest

from momwire import _features
from momwire._accel import acc
from momwire.sinusoidal import SinusoidalSolver
from test_crossing_serve_524 import hub_deck

# An experimental feature (`momwire._features`, reach 1).
pytestmark = [
    pytest.mark.reach(1),
    pytest.mark.skipif(
        acc is None or not hasattr(acc, "point_field_tensor_cplx_staged"),
        reason="the accelerator is not built",
    ),
]

K_M = 0.5872 - 0.2387j


def _screen(n_radials=8, n_seg=12, length=6.0, depth=0.15):
    c, t, h = [], [], []
    for r in range(n_radials):
        th = 2 * np.pi * r / n_radials
        u = np.array([np.cos(th), np.sin(th), 0.0])
        edges = np.linspace(0.05, length, n_seg + 1)
        for i in range(n_seg):
            c.append(u * 0.5 * (edges[i] + edges[i + 1]) + [0.0, 0.0, -depth])
            t.append(u)
            h.append(edges[i + 1] - edges[i])
    return np.ascontiguousarray(c), np.ascontiguousarray(t), np.asarray(h)


def _args(rows=slice(None)):
    c, t, h = _screen()
    gx, gw = np.polynomial.legendre.leggauss(8)
    a = np.full(len(c), 0.001)
    eta = 376.730313668 / (K_M / 0.1488)
    return (c[rows], t[rows], a[rows], c, t, h, K_M, eta, gx, gw)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("MOMWIRE_X_POINT_CPLX_STAGED", raising=False)
    assert not _features.enabled("point_cplx_staged")
    assert not SinusoidalSolver(**hub_deck())._point_cplx_staged


@pytest.mark.parametrize("folded", [True, False])
def test_the_staged_tensor_is_the_scalar_one_to_a_few_ulp_of_scale(folded):
    old = acc.sinusoidal_field_tensor_cplx(*_args(), folded)
    new = acc.point_field_tensor_cplx_staged(*_args(), folded)
    for o, n in zip(old, new, strict=True):
        assert np.max(np.abs(n - o)) <= 4e-15 * np.max(np.abs(o))


@pytest.mark.parametrize("folded", [True, False])
def test_a_row_is_its_observers_alone(folded):
    full = acc.point_field_tensor_cplx_staged(*_args(), folded)
    part = acc.point_field_tensor_cplx_staged(*_args(slice(37, 61)), folded)
    for f, p in zip(full, part, strict=True):
        assert np.array_equal(f[37:61], p)


def _z(flag):
    s = SinusoidalSolver(**hub_deck(), point_cplx_staged=flag)
    return 1.0 / complex(s.compute_port_solution().y.ravel()[0])


# The solver asks `_features.enabled`, which refuses a native feature outside
# its reach (macOS, sse2), so only this test needs the platform's level; the
# kernel tests above call the binding directly.
@pytest.mark.skipif(
    not _features.certified("point_cplx_staged"),
    reason="point_cplx_staged is not certified on this platform",
)
def test_on_a_buried_deck_takes_the_staged_kernel_and_z_agrees(monkeypatch):
    calls = []
    real = acc.point_field_tensor_cplx_staged

    def counted(*a):
        calls.append(len(a[0]))
        return real(*a)

    monkeypatch.setattr(acc, "point_field_tensor_cplx_staged", counted)
    z_off = _z(False)
    assert not calls
    z_on = _z(True)
    assert calls
    assert abs(z_on - z_off) < 1e-11 * abs(z_off)


# --- the C++ route (the NEC-4.2 slot's sinusoidal run half) ----------------
#
# `_sinusoidal_fill_core.cpp` takes the staged rows at both of the
# sinusoidal fill's in-medium kernel calls (a wholly-buried deck's band, a
# crossing deck's buried class) when the run requests `point_cplx_staged`.
# No executable carries the feature (`EXE_FEATURES`, reach 1), so the route
# is reached through the in-process hook, `sinusoidal_nec4_printout(...,
# features=[...])`, and held to the Python slot with the flag on, printout
# byte for byte with the two measured-time lines masked. The staged kernel's
# libmvec sweep is x86 Linux's: byte equality is held there only (the
# feature's reach is the hosted app's level anyway).

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "core_split_nec4_run_compare_1849",
    _ROOT / "scripts" / "core_split" / "nec4_run_compare.py",
)
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)
_spec = importlib.util.spec_from_file_location(
    "core_split_sinusoidal_fill_compare_1849",
    _ROOT / "scripts" / "core_split" / "sinusoidal_fill_compare.py",
)
fc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fc)

_ENV = "MOMWIRE_X_POINT_CPLX_STAGED"


def _deck(geometry: str, body: str) -> str:
    return "CM run\nCE\n" + geometry + "GE -1\n" + body + "EN\n"


def _screen_deck(n_radials=6, n_seg=10, length=3.0, depth=0.3):
    wires = "".join(
        f"GW {r + 1} {n_seg} 0 0 {-depth} "
        f"{length * np.cos(2 * np.pi * r / n_radials):.6f} "
        f"{length * np.sin(2 * np.pi * r / n_radials):.6f} {-depth} .001\n"
        for r in range(n_radials)
    )
    return _deck(wires, "GN 2 0 0 0 13 .005\nFR 0 1 0 0 7.15\nEX 0 1 1 0 1\nXQ\n")


# a vertical rising from a rod to a buried hub of two radials: the crossing
# fill, whose buried class takes the in-medium kernel
_CROSSING = _deck(
    "GW 1 10 0 0 0 0 0 5 .001\nGW 2 3 0 0 0 0 0 -.3 .001\n"
    "GW 3 10 0 0 -.3 3 0 -.3 .001\nGW 4 10 0 0 -.3 -3 0 -.3 .001\n",
    "GN 2 0 0 0 13 .005\nFR 0 1 0 0 7.15\nEX 0 1 1 0 1\nXQ\n",
)
_DECKS = {"wholly_buried_screen": _screen_deck(), "buried_crossing": _CROSSING}

_x86_linux = sys.platform.startswith("linux") and platform.machine() in (
    "x86_64",
    "AMD64",
)


@pytest.fixture
def run_half(_one_blas_thread):
    if not hasattr(acc, "sinusoidal_nec4_printout") or not hasattr(
        acc, "point_cplx_staged_calls"
    ):
        pytest.skip("no compiled accelerator with the NEC-4.2 run half")
    try:
        blas = rc.blas_addresses()
    except (IndexError, AttributeError, OSError):
        pytest.skip("no bundled OpenBLAS beside numpy / SciPy (not a wheel install)")
    return blas


@pytest.fixture(autouse=True)
def _below_anchors_off(monkeypatch):
    # the below grid's anchors follow the frequencies a process has seen; both
    # copies are held on the exact grid (as tests/test_nec4_run_core.py)
    monkeypatch.setenv("MOMWIRE_BELOW_ANCHORS", "off")


@pytest.fixture
def _one_blas_thread():
    # both copies call the same OpenBLAS, one thread (as test_nec4_run_core)
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        yield


def _cxx(text, blas, features=()):
    # line 2's tail as Python's slot spells it: the overlay marker, then the
    # features the environment turned on (`_features.stamp()`, which an
    # executable writes from `ni_eval::requested_stamp`)
    version, variant, overlay = rc.pc.stamp()
    marker = " ".join(t for t in (overlay, _features.stamp()) if t)
    return acc.sinusoidal_nec4_printout(
        text.encode("latin-1"), blas, version, variant, marker, list(features)
    )


def test_the_run_half_refuses_a_feature_it_does_not_carry(run_half):
    with pytest.raises(ValueError, match="not a C\\+\\+ feature: no_such_feature"):
        _cxx(_DECKS["buried_crossing"], run_half, ["no_such_feature"])


@pytest.mark.parametrize("name", sorted(_DECKS))
def test_off_the_run_half_never_takes_the_staged_kernel(name, run_half, monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    before = acc.point_cplx_staged_calls()
    ok, cxx = _cxx(_DECKS[name], run_half)
    assert ok, cxx[:400]
    assert acc.point_cplx_staged_calls() == before
    a, na = rc.mask(rc.py_printout(_DECKS[name]))
    b, nb = rc.mask(cxx)
    assert acc.point_cplx_staged_calls() == before
    assert rc.first_difference(a, b) is None, rc.first_difference(a, b)
    assert na == nb >= 2, (na, nb)


@pytest.mark.skipif(
    not _features.certified("point_cplx_staged"),
    reason="point_cplx_staged is not certified on this platform",
)
@pytest.mark.skipif(
    not _x86_linux, reason="the staged sweep's bits are x86 Linux libmvec's"
)
@pytest.mark.parametrize("name", sorted(_DECKS))
def test_on_the_run_half_prints_the_python_slots_bytes(name, run_half, monkeypatch):
    text = _DECKS[name]
    monkeypatch.setenv(_ENV, "1")
    c0 = acc.point_cplx_staged_calls()
    a, na = rc.mask(rc.py_printout(text))
    c1 = acc.point_cplx_staged_calls()
    assert c1 > c0, "the Python slot did not take the staged kernel"
    ok, cxx = _cxx(text, run_half, ["point_cplx_staged"])
    assert ok, cxx[:400]
    c2 = acc.point_cplx_staged_calls()
    assert c2 > c1, "the C++ run half did not take the staged kernel"
    b, nb = rc.mask(cxx)
    assert rc.first_difference(a, b) is None, rc.first_difference(a, b)
    assert na == nb >= 2, (na, nb)
    # the request was for that run only
    ok, _ = _cxx(text, run_half)
    assert ok and acc.point_cplx_staged_calls() == c2


# The printout's five figures read the same on and off on these decks, so the
# bytes above cannot tell the kernels apart. The fill dump can: G, the port
# coefficients and Y bit for bit (`scripts/core_split/sinusoidal_fill_compare.py`).
@pytest.mark.skipif(
    not _features.certified("point_cplx_staged"),
    reason="point_cplx_staged is not certified on this platform",
)
@pytest.mark.skipif(
    not _x86_linux, reason="the staged sweep's bits are x86 Linux libmvec's"
)
@pytest.mark.parametrize("name", sorted(_DECKS))
def test_on_the_cxx_fill_is_the_python_fill_bit_for_bit(name, run_half, monkeypatch):
    text = _DECKS[name]
    blas = fc.blas_addresses()
    monkeypatch.delenv(_ENV, raising=False)
    off = fc.cxx_fill(text, acc, blas)
    assert fc.classify(off) == "fills" and " ! " not in off, off[:400]
    monkeypatch.setenv(_ENV, "1")
    c0 = acc.point_cplx_staged_calls()
    py_on = fc.py_fill(text)
    c1 = acc.point_cplx_staged_calls()
    on = acc.sinusoidal_fill_dump(
        text.encode("latin-1"), blas, fc.WHOLE, ["point_cplx_staged"]
    )
    c2 = acc.point_cplx_staged_calls()
    assert c1 > c0 and c2 > c1, (c0, c1, c2)
    assert on == py_on, rc.first_difference(py_on, on)
    # and the gate can see the kernel: on is not off's bits
    assert on != off
