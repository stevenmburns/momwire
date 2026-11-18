"""The C++ bspline port solve's twin of momwire#1898 (`kcl_nullspace`):
`bspline::solve_ports` (`_bspline_fill_core.cpp`), the solve the
nec5-bspline and nec42-bspline executables run, eliminates the junction
rows before the factor exactly as `BSplineSolver._solve_reduced` does.

The feature keeps reach 1: no executable requests it (`EXE_FEATURES`
leaves it out, so its variable still makes one exit 2), and the port is
reached here only through the in-process test hooks' `features=` argument.

What is held, on synthetic junction decks (a T junction, a three-wire node,
a small cage) and on crafted matrices:

- `bspline_front_geometry(..., features=["kcl_nullspace"])`: the C++ X and Y
  equal to the bit (uint64 views) to `BSplineSolver(kcl_nullspace=True)`'s
  port solve on the same Z, and the route is "reduced" on both sides;
- `bspline_solve_ports` on random Z with constraint rows Python's `plan`
  refuses (a non-unit entry, a shared column, an empty row): both take the
  Schur route, say why in the same words, and agree to the bit (or are both
  singular);
- `bspline_nec4_printout(..., features=["kcl_nullspace"])` byte for byte
  against `_nec4.render(text, basis="bspline")` under
  MOMWIRE_X_KCL_NULLSPACE=1, the time lines masked, every solve's arrays to
  the bit, and the C++ route counter moved by one "reduced" per solve;
- with the flag off, the C++ never enters the reduction and both copies
  print what they printed before.
"""

from __future__ import annotations

import ctypes
import glob
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from momwire import _accel, _kcl_nullspace, _lapack
from momwire.bspline import BSplineSolver

ROOT = Path(__file__).resolve().parents[1]
ENV = "MOMWIRE_X_KCL_NULLSPACE"
FEATURES = ["kcl_nullspace"]

acc = _accel.acc
pytestmark = [
    pytest.mark.reach(1),
    pytest.mark.skipif(
        acc is None or not hasattr(acc, "bspline_solve_ports"),
        reason="needs the extension's momwire#1898 hooks",
    ),
    pytest.mark.skipif(
        sys.version_info < (3, 12),
        reason="the reader copies CPython 3.12's float rules",
    ),
]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rc = _load(
    "core_split_nec4_run_compare_1898",
    ROOT / "scripts" / "core_split" / "nec4_run_compare.py",
)


@pytest.fixture(autouse=True)
def _one_blas_thread(monkeypatch):
    from threadpoolctl import threadpool_limits

    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv("MOMWIRE_BELOW_ANCHORS", "off")
    with threadpool_limits(limits=1):
        yield


def _numpy_blas():
    """numpy's ILP64 dsyrk, then cblas_zgemm / zgemv / zdotu_sub and zgesv."""
    found = sorted(
        glob.glob(
            os.path.join(
                os.path.dirname(np.__file__),
                "..",
                "numpy.libs",
                "libscipy_openblas64_*",
            )
        )
    )
    if not found:
        pytest.skip("no bundled OpenBLAS beside numpy (not a wheel install)")
    lib = ctypes.CDLL(found[0])

    def sym(name):
        return ctypes.cast(getattr(lib, name), ctypes.c_void_p).value

    syrk = sym("scipy_cblas_dsyrk64_")
    four = [
        sym(n)
        for n in (
            "scipy_cblas_zgemm64_",
            "scipy_cblas_zgemv64_",
            "scipy_cblas_zdotu_sub64_",
            "scipy_zgesv_64_",
        )
    ]
    return syrk, four


def _bits(a):
    return np.ascontiguousarray(a, dtype=np.complex128).view(np.uint64)


def _same_bits(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and np.array_equal(_bits(a), _bits(b))


# ---- the synthetic decks ------------------------------------------------------
# Free space, lambda = 1 m (299.7925 MHz), endpoint junctions only.

FR = "FR 0 1 0 0 299.7925\n"


def _deck(geometry, ex, runs=1):
    body = "".join(ex + "XQ\n" for _ in range(runs))
    return "CM synthetic junction deck\nCE\n" + geometry + "GE 0\n" + FR + body + "EN\n"


# A T junction: two collinear arms and a perpendicular stub meet at the
# origin, fed on one arm (a three-entry row, both signs).
T_JUNCTION = _deck(
    "GW 1 9 -.25 0 0 0 0 0 .001\nGW 2 9 0 0 0 .25 0 0 .001\nGW 3 7 0 0 0 0 0 .2 .001\n",
    "EX 0 1 5 0 1\n",
)

# A fed dipole with a three-wire node at its top and a two-wire bend at its
# foot (rows with 3 and 2 entries).
THREE_WIRE_NODE = _deck(
    "GW 1 11 0 0 -.25 0 0 .25 .001\n"
    "GW 2 5 0 0 .25 .15 0 .25 .001\n"
    "GW 3 5 0 0 .25 -.1 .1 .25 .001\n"
    "GW 4 5 .12 0 -.25 0 0 -.25 .001\n",
    "EX 0 1 6 0 1\n",
)


def _cage():
    """A three-wire cage: a fed stub, three spokes out, three verticals, three
    spokes back to a tip: a four-wire node, six bends, a three-wire tip."""
    r, lines, tag = 0.05, ["GW 1 5 0 0 -.1 0 0 .1 .001\n"], 2
    for i in range(3):
        t = 2 * np.pi * i / 3
        x, y = r * np.cos(t), r * np.sin(t)
        lines.append(f"GW {tag} 2 0 0 .1 {x:.6f} {y:.6f} .15 .001\n")
        lines.append(f"GW {tag + 1} 7 {x:.6f} {y:.6f} .15 {x:.6f} {y:.6f} .45 .001\n")
        lines.append(f"GW {tag + 2} 2 {x:.6f} {y:.6f} .45 0 0 .5 .001\n")
        tag += 3
    return _deck("".join(lines), "EX 0 1 3 0 1\n")


CAGE = _cage()

DECKS = {"t_junction": T_JUNCTION, "three_wire_node": THREE_WIRE_NODE, "cage": CAGE}


def _counts():
    return dict(acc.bspline_kcl_nullspace_counts())


def _delta(before, after):
    return {
        k: after.get(k, 0) - before.get(k, 0)
        for k in after
        if after.get(k, 0) != before.get(k, 0)
    }


# ---- the front hook: X and Y against BSplineSolver's reduced solve ------------


def _front(text, features):
    syrk, four = _numpy_blas()
    entries = acc.bspline_front_geometry(
        text.encode("latin-1"), syrk, four, features=features
    )
    assert len(entries) == 1, len(entries)
    entry = entries[0]
    assert "basis_declined" not in entry and "basis_raised" not in entry, entry
    assert entry["solve_info"] == 0
    return entry


def _python_solve(entry, on):
    """Python's port solve on the entry's own deck, from Python's own fill:
    (Z, X, Y, the route) as `compute_port_solution`'s dense route has them."""
    s = BSplineSolver(**entry["kwargs"], kcl_nullspace=on)
    geom = s._build_geometry()
    supp, polys, kcl, knots, wbg = s._build_basis_polynomials(geom)
    J = s._build_J_blocks(geom, s.k)
    Z = s._assemble_Z(J, supp, polys, geom)
    B, kcon = s._port_columns(geom, knots, wbg, supp.shape[0], kcl)
    X = s._solve_with_kcl_ports(Z.copy(), B, kcon)
    Y = _lapack.matmul(B.T, X)
    return Z, kcon, X, Y, getattr(s, "kcl_nullspace_route", None)


@pytest.mark.parametrize("name", list(DECKS))
def test_the_cxx_reduced_solve_is_pythons_to_the_bit(name):
    text = DECKS[name]
    before, py_before = _counts(), dict(_kcl_nullspace.COUNTS)
    entry = _front(text, FEATURES)
    after = _counts()
    assert entry["kcl_route"] == "reduced"
    assert _delta(before, after) == {"reduced": 1}
    Z, kcon, X, Y, route = _python_solve(entry, True)
    assert route == "reduced"
    assert _kcl_nullspace.COUNTS["reduced"] == py_before.get("reduced", 0) + 1
    # not vacuous: rows were eliminated, and the fill under both is one Z
    assert kcon.shape[0] >= 1
    assert _same_bits(Z, entry["Z"])
    assert _same_bits(X, entry["X"]), np.max(np.abs(X - entry["X"]))
    assert _same_bits(Y, entry["Y"]), np.max(np.abs(Y - entry["Y"]))


@pytest.mark.parametrize("name", list(DECKS))
def test_flag_off_the_cxx_takes_the_schur_route_as_before(name):
    text = DECKS[name]
    before = _counts()
    off = _front(text, [])
    assert _counts() == before  # off never enters the plan
    assert off["kcl_route"] == ""
    _Z, _kcon, X, Y, route = _python_solve(off, False)
    assert route is None
    assert _same_bits(X, off["X"])
    assert _same_bits(Y, off["Y"])
    # the two routes are different arithmetic: the reduced bits moved
    on = _front(text, FEATURES)
    assert not _same_bits(on["X"], off["X"])
    rel = np.max(np.abs(on["X"] - off["X"])) / np.max(np.abs(off["X"]))
    assert rel < 1e-12, rel  # the registry's declared drift


# ---- crafted constraint rows: the Schur fallback -----------------------------


def _random_system(rng, n, n_p):
    Z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Z += n * np.eye(n)
    B = rng.standard_normal((n, n_p))
    return np.ascontiguousarray(Z), B


def _host_solver():
    """A BSplineSolver with the flag on, to run its port solve on a crafted
    (Z, B, A) (the solve reads only the flag from the solver)."""
    return BSplineSolver(
        wires=[np.array([(0.0, 0.0, -0.25), (0.0, 0.0, 0.25)])],
        n_per_edge_per_wire=[[11]],
        wavelength=1.0,
        wire_radius=1e-3,
        feeds=[(0, None, 1.0 + 0j)],
        kcl_nullspace=True,
    )


def _rows(n, entries):
    A = np.zeros((max(r for r, _c, _v in entries) + 1, n))
    for r, c, v in entries:
        A[r, c] = v
    return A


N = 14
CRAFTED = {
    "reducible": (
        _rows(N, [(0, 3, 1.0), (0, 1, -1.0), (0, 9, 1.0), (1, 2, -1.0), (1, 12, -1.0)]),
        "reduced",
    ),
    "non_unit_entry": (
        _rows(N, [(0, 1, 1.0), (0, 4, -1.0), (1, 2, 2.0), (1, 7, -1.0)]),
        "schur: an entry other than +-1",
    ),
    "shared_column": (
        _rows(N, [(0, 1, 1.0), (0, 4, -1.0), (1, 4, 1.0), (1, 8, 1.0)]),
        "schur: a column in two constraint rows",
    ),
}


@pytest.mark.parametrize("name", list(CRAFTED))
def test_crafted_rows_take_pythons_route(name):
    A, want = CRAFTED[name]
    rng = np.random.default_rng(1898)
    Z, B = _random_system(rng, N, 2)
    _syrk, four = _numpy_blas()
    before = _counts()
    info, Xc, Yc, route = acc.bspline_solve_ports(Z, B, A, four, FEATURES)
    assert route == want
    assert _delta(before, _counts()) == {want: 1}
    s = _host_solver()
    Xp = s._solve_with_kcl_ports(Z.copy(), B, A)
    assert s.kcl_nullspace_route == want
    assert info == 0
    assert _same_bits(Xp, Xc), np.max(np.abs(Xp - Xc))
    assert _same_bits(_lapack.matmul(B.T, Xp), Yc)


def test_an_empty_row_falls_back_and_both_are_singular():
    # the middle row empty: Python's plan refuses it, and the Schur system
    # A X then has a zero row, singular in both copies (a general one: the
    # outer rows keep it from reading as triangular)
    A = _rows(N, [(0, 1, 1.0), (0, 4, -1.0), (2, 6, 1.0), (2, 9, 1.0)])
    rng = np.random.default_rng(18981)
    Z, B = _random_system(rng, N, 1)
    _syrk, four = _numpy_blas()
    info, _X, _Y, route = acc.bspline_solve_ports(Z, B, A, four, FEATURES)
    assert route == "schur: an empty constraint row"
    assert info != 0
    s = _host_solver()
    with pytest.raises(np.linalg.LinAlgError):
        s._solve_with_kcl_ports(Z.copy(), B, A)
    assert s.kcl_nullspace_route == route


def test_an_unknown_feature_is_refused_and_nothing_stays_requested():
    rng = np.random.default_rng(3)
    Z, B = _random_system(rng, N, 1)
    A, _want = CRAFTED["reducible"]
    _syrk, four = _numpy_blas()
    with pytest.raises(ValueError, match="not a C\\+\\+ feature"):
        acc.bspline_solve_ports(Z, B, A, four, ["kcl_nullspace", "no_such_feature"])
    before = _counts()
    *_r, route = acc.bspline_solve_ports(Z, B, A, four, [])
    assert route == ""
    assert _counts() == before


# ---- expand's scatter-add order -------------------------------------------------


def test_expand_adds_in_np_add_at_order():
    """`expand`'s np.add.at(x, e, x[k] * a) is one pass in index order, and
    a row of many entries makes that order show in the bits: its eliminated
    end gains one contribution per other entry, and a floating-point sum of
    eight terms depends on the order it is taken in. The C++ expand must
    take Python's order, not merely one that agrees on short rows (a
    two-wire bend adds one term, a three-wire node two, and two terms
    commute exactly)."""
    n = 24
    big_row = [2, 3, 5, 7, 8, 11, 13, 16, 19]
    signs = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0]
    A = _rows(
        n,
        [(0, c, v) for c, v in zip(big_row, signs, strict=True)]
        + [(1, 4, 1.0), (1, 9, 1.0), (1, 21, -1.0)],
    )
    rng = np.random.default_rng(18982)
    Z, B = _random_system(rng, n, 3)
    p = _kcl_nullspace.plan(A)
    assert not isinstance(p, str), p
    # the long row's end: one e, repeated once per other entry
    assert np.count_nonzero(p.e == big_row[0]) == len(big_row) - 1

    s = _host_solver()
    Xp = s._solve_with_kcl_ports(Z.copy(), B, A)
    assert s.kcl_nullspace_route == "reduced"
    y = Xp[p.keep]  # x[keep] = y, untouched by the scatter onto e

    def scatter(order):
        x = np.zeros((n, y.shape[1]), dtype=np.complex128)
        x[p.keep] = y
        contrib = x[p.k] * p.a[:, None]
        for i in order:
            x[p.e[i]] += contrib[i]
        return x

    forward = scatter(range(p.e.size))
    backward = scatter(reversed(range(p.e.size)))
    # np.add.at is the forward pass, to the bit
    assert _same_bits(forward, Xp)
    # and the order is visible: the backward pass moves the long row's end
    assert not _same_bits(backward, Xp)
    assert _same_bits(backward[p.keep], Xp[p.keep])

    _syrk, four = _numpy_blas()
    info, Xc, _Yc, route = acc.bspline_solve_ports(Z, B, A, four, FEATURES)
    assert (info, route) == (0, "reduced")
    assert _same_bits(Xc, Xp), np.max(np.abs(Xc - Xp))


# ---- the NEC-4.2 slot printout ------------------------------------------------


def _cxx_printout(text, features, marker):
    version, variant, _overlay = rc.pc.stamp()
    return acc.bspline_nec4_printout(
        text.encode("latin-1"),
        rc.blas_addresses("bspline"),
        version,
        variant,
        marker,
        arrays=True,
        features=features,
    )


def _printouts_equal(py, cxx):
    a, na = rc.mask(py)
    b, nb = rc.mask(cxx)
    assert rc.first_difference(a, b) is None, rc.first_difference(a, b)
    assert na == nb, (na, nb)


@pytest.mark.parametrize("name", list(DECKS))
def test_the_nec42_printout_on_is_pythons_byte_for_byte(name, monkeypatch):
    from momwire import _features

    try:
        rc.blas_addresses("bspline")
    except (IndexError, AttributeError, OSError):
        pytest.skip("no bundled OpenBLAS beside numpy / SciPy (not a wheel install)")
    text = DECKS[name]
    monkeypatch.setenv(ENV, "1")
    stamp = _features.stamp()
    assert stamp.startswith("+kcl_nullspace"), stamp
    py_before = _kcl_nullspace.COUNTS["reduced"]
    py, py_solves = rc.py_arrays(text)
    assert _kcl_nullspace.COUNTS["reduced"] - py_before == len(py_solves) >= 1
    before = _counts()
    ok, cxx, cxx_solves = _cxx_printout(text, FEATURES, stamp)
    assert ok, cxx
    assert _delta(before, _counts()) == {"reduced": len(cxx_solves)}
    assert stamp in cxx.split("\n")[1]
    _printouts_equal(py, cxx)
    diff = rc.arrays_difference(py_solves, cxx_solves)
    assert diff is None, diff


@pytest.mark.parametrize("name", list(DECKS))
def test_the_nec42_printout_off_is_unchanged(name):
    try:
        rc.blas_addresses("bspline")
    except (IndexError, AttributeError, OSError):
        pytest.skip("no bundled OpenBLAS beside numpy / SciPy (not a wheel install)")
    text = DECKS[name]
    py_before = dict(_kcl_nullspace.COUNTS)
    py, py_solves = rc.py_arrays(text)
    assert dict(_kcl_nullspace.COUNTS) == py_before
    before = _counts()
    ok, cxx, cxx_solves = _cxx_printout(text, [], "")
    assert ok, cxx
    assert _counts() == before
    _printouts_equal(py, cxx)
    assert rc.arrays_difference(py_solves, cxx_solves) is None


# ---- FeatureRequest and the GIL ----------------------------------------------


def test_feature_scopes_on_two_threads_neither_leak_nor_cross():
    """`ni_eval::FeatureRequest` saves the process-global requested set,
    adds the hook's features, and restores it, while the hook's run holds
    the GIL released. Two Python threads in two hooks at once would then
    interleave their save / restore pairs (A saves S and asks kcl_nullspace,
    B saves S+kcl, A restores S, B restores S+kcl: the feature left on for
    the process) unless the scopes serialize, which they do on one
    process-wide lock taken with the GIL released (`_near_interface_eval.h`).

    One thread prints with the feature on, one with it off, at once: every
    printout's solves are its own serial run's to the bit, and afterwards a
    flag-off solve takes no reduced route."""
    import threading

    try:
        rc.blas_addresses("bspline")
    except (IndexError, AttributeError, OSError):
        pytest.skip("no bundled OpenBLAS beside numpy / SciPy (not a wheel install)")
    text = DECKS["t_junction"]
    want = {}
    for key, features in (("on", FEATURES), ("off", [])):
        ok, _out, solves = _cxx_printout(text, features, "")
        assert ok, _out
        want[key] = solves
    # the two are told apart: the reduced route moves the solve's bits
    assert rc.arrays_difference(want["on"], want["off"]) is not None

    rounds = 12
    start = threading.Barrier(2)
    got = {"on": [], "off": []}
    errors = []

    def run(key, features):
        try:
            start.wait()
            for _ in range(rounds):
                ok, out, solves = _cxx_printout(text, features, "")
                assert ok, out
                got[key].append(solves)
        except BaseException as exc:  # noqa: BLE001 — re-raised on the main thread below
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=(key, features))
        for key, features in (("on", FEATURES), ("off", []))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    for key in ("on", "off"):
        assert len(got[key]) == rounds
        for i, solves in enumerate(got[key]):
            diff = rc.arrays_difference(want[key], solves)
            assert diff is None, (key, i, diff)
    # nothing left requested: a flag-off solve outside any printout
    rng = np.random.default_rng(5)
    Z, B = _random_system(rng, N, 1)
    A, _want = CRAFTED["reducible"]
    _syrk, four = _numpy_blas()
    before = _counts()
    *_r, route = acc.bspline_solve_ports(Z, B, A, four, [])
    assert route == ""
    assert _counts() == before
