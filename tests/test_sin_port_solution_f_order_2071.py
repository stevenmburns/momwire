"""The point-matched port solution factors Z in the order the fill wrote it
(momwire#2071).

`compute_port_solution` asks `_assemble_Z` for LAPACK's (Fortran) order, the
fused band (momwire#2069) writes Z so, and `_lapack.solve(overwrite_a=True)`
factors it in its own storage: no Fortran copy. The finiteness check and the
1-norm gecon reads are taken in place (`finite_1norm_f`). Nothing here
decides a value -- the fill computes each element as before and only stores
it elsewhere, and the factor reads the bytes the copy would have held -- so:

  * the in-place norm is the copying pass's (and lange's), bit for bit, at
    every thread count;
  * an in-place solve is the copying solve, bit for bit: the answer, the
    error, the warning, and gecon's anorm and rcond;
  * `compute_port_solution` is the C-order route's, byte for byte, on every
    deck the fused band serves, and a counter proves the copy-free route
    ran; a deck the fused band declines keeps the copy and its bits.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.linalg

from momwire import _lapack
from momwire import sinusoidal as sn
from momwire._accel import acc
from momwire.sinusoidal import SinusoidalSolver

HAVE_KERNEL = acc is not None and getattr(acc, "solve_prep_2071", False)
HAVE_FUSED = sn._HAVE_FILL_ROWS and sn._HAVE_PHI_REDUCE_ACCEL


def _rand(rng, shape, dtype):
    a = rng.standard_normal(shape)
    if np.issubdtype(dtype, np.complexfloating):
        a = a + 1j * rng.standard_normal(shape)
    return a.astype(dtype)


# ---------------------------------------------------------------------------
# the in-place norm
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAVE_KERNEL, reason="no finite_1norm_f in the accelerator")
@pytest.mark.parametrize("dtype", [np.complex128, np.float64])
@pytest.mark.parametrize("shape", [(1, 1), (64, 64), (130, 77), (77, 130), (300, 300)])
def test_in_place_norm_is_the_copying_norm(dtype, shape):
    rng = np.random.default_rng(2071)
    a = _rand(rng, shape, dtype)
    # A wide dynamic range, so the sums' order would show if it moved.
    a *= np.exp(rng.uniform(-30, 30, shape))
    f = np.asfortranarray(a)
    lange = scipy.linalg.get_lapack_funcs("lange", (f,))
    want = lange("1", f)
    _, _, copied = acc.fortran_copy_1norm(np.ascontiguousarray(a))
    assert copied == want
    before = f.tobytes(order="A")
    for nt in (1, 3, 0):
        finite, anorm = acc.finite_1norm_f(f, nt)
        assert finite
        assert anorm == want, (nt, anorm, want)
    assert f.tobytes(order="A") == before  # read, not written


@pytest.mark.skipif(not HAVE_KERNEL, reason="no finite_1norm_f in the accelerator")
@pytest.mark.parametrize("where", ["re-nan", "im-inf", "re-ninf"])
def test_in_place_norm_sees_a_non_finite_entry(where):
    rng = np.random.default_rng(2)
    a = np.asfortranarray(_rand(rng, (100, 100), np.complex128))
    v = {"re-nan": complex(np.nan, 0.0), "im-inf": complex(0.0, np.inf)}.get(
        where, complex(-np.inf, 1.0)
    )
    a[97, 3] = v
    finite, _ = acc.finite_1norm_f(a)
    assert not finite
    with pytest.raises(RuntimeError, match="F-contiguous"):
        acc.finite_1norm_f(np.ascontiguousarray(a))


# ---------------------------------------------------------------------------
# `_lapack.solve(overwrite_a=True)`
# ---------------------------------------------------------------------------


@pytest.fixture
def gecon_seen(monkeypatch):
    """Every gecon `_lapack.solve` runs, as (anorm, rcond)."""
    seen = []
    real = scipy.linalg.get_lapack_funcs

    def get(names, arrays=(), *a, **kw):
        fns = real(names, arrays, *a, **kw)
        single = isinstance(names, str)
        out = []
        for name, fn in zip([names] if single else names, [fns] if single else fns):
            if name == "gecon":

                def gecon(lu, anorm, *ga, _fn=fn, **gk):
                    r = _fn(lu, anorm, *ga, **gk)
                    seen.append((float(anorm), float(r[0])))
                    return r

                out.append(gecon)
            else:
                out.append(fn)
        return out[0] if single else tuple(out)

    monkeypatch.setattr(scipy.linalg, "get_lapack_funcs", get)
    return seen


@pytest.fixture(params=["default", "host"])
def route(request, monkeypatch):
    if request.param == "default":
        monkeypatch.setattr(_lapack, "_host_for", lambda dtype: None)
    else:
        # scipy's own getrf / getrs, through the host route's code path.
        monkeypatch.setattr(_lapack, "_host_for", lambda dtype: "scipy")

        def getrf(fns, a):
            f = scipy.linalg.get_lapack_funcs("getrf", (a,))
            lu, piv, info = f(a, overwrite_a=True)
            assert lu is a or np.shares_memory(lu, a)
            return piv + 1, info

        def getrs(fns, lu, ipiv, b, trans=0):
            f = scipy.linalg.get_lapack_funcs("getrs", (lu,))
            x, info = f(lu, ipiv - 1, b, trans=trans, overwrite_b=True)
            b[...] = x
            return info

        monkeypatch.setattr(_lapack, "_getrf", getrf)
        monkeypatch.setattr(_lapack, "_getrs", getrs)
    return request.param


def _outcome(fn, *args, **kw):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        try:
            x = fn(*args, **kw)
            exc = None
        except Exception as e:  # noqa: BLE001 - the type IS the observation
            x, exc = None, type(e)
    return x, exc, sorted({w.category.__name__ for w in seen})


def _ill(rng, n, dtype):
    q1, _ = np.linalg.qr(_rand(rng, (n, n), dtype))
    q2, _ = np.linalg.qr(_rand(rng, (n, n), dtype))
    return np.ascontiguousarray((q1 * np.logspace(0, -19, n)) @ q2)


def _cases(dtype, n=200):
    rng = np.random.default_rng(3)
    fine = _rand(rng, (n, n), dtype) + n * np.eye(n, dtype=dtype)
    nan = fine.copy()
    nan[150, 7] = np.nan
    b = _rand(rng, (n,), dtype)
    return {
        "fine": (fine, b),
        "fine, many rhs": (fine, _rand(rng, (n, 3), dtype)),
        "ill": (_ill(rng, n, dtype), b),
        "nan": (nan, b),
        "nan rhs": (fine, np.where(np.arange(n) == 5, np.nan, b)),
    }


@pytest.mark.parametrize("accel", [True, False])
@pytest.mark.parametrize("check", ["on", "off"])
@pytest.mark.parametrize("dtype", [np.complex128, np.float64])
def test_solve_in_place_is_the_copying_solve(
    route, dtype, check, accel, gecon_seen, monkeypatch
):
    monkeypatch.setenv(_lapack.CONDITION_CHECK_ENV, "1" if check == "on" else "0")
    monkeypatch.setattr(_lapack, "_SOLVE_PREP_ACCEL", accel)
    for name, (a, b) in _cases(dtype).items():
        gecon_seen.clear()
        want = _outcome(_lapack.solve, a, b)
        want_gecon = list(gecon_seen)
        gecon_seen.clear()
        f = np.asfortranarray(a)
        n0 = _lapack.IN_PLACE_SOLVES["count"]
        got = _outcome(_lapack.solve, f, b, overwrite_a=True)
        assert _lapack.IN_PLACE_SOLVES["count"] == n0 + 1, name  # no copy taken
        assert got[1:] == want[1:], name
        if want[0] is not None:
            assert got[0].tobytes() == want[0].tobytes(), name
            # The factors went into f's own storage.
            assert not np.array_equal(f, a), name
        assert gecon_seen == want_gecon, name
    # The cases mean something: the estimate, its warning and the error occur.
    cases = _cases(dtype)
    gecon_seen.clear()
    assert _outcome(
        _lapack.solve,
        np.asfortranarray(cases["ill"][0]),
        cases["ill"][1],
        overwrite_a=True,
    )[2] == ["LinAlgWarning"]
    assert gecon_seen  # the screen (or the switch) ran gecon
    gecon_seen.clear()
    _outcome(
        _lapack.solve,
        np.asfortranarray(cases["fine"][0]),
        cases["fine"][1],
        overwrite_a=True,
    )
    assert bool(gecon_seen) == (check == "on")


def test_what_cannot_be_factored_in_place_is_copied():
    rng = np.random.default_rng(4)
    a = _rand(rng, (100, 100), np.complex128) + 100 * np.eye(100)
    b = _rand(rng, (100,), np.complex128)
    want = _lapack.solve(a, b)
    ro = np.asfortranarray(a)
    ro.flags.writeable = False
    for arg, rhs in [
        (a.copy(), b),  # C order
        (ro, b),  # read-only
        (np.asfortranarray(a.real), b),  # cast by a complex rhs
    ]:
        before = arg.copy(order="A")
        n0 = _lapack.IN_PLACE_SOLVES["count"]
        _lapack.solve(arg, rhs, overwrite_a=True)
        assert _lapack.IN_PLACE_SOLVES["count"] == n0
        assert np.array_equal(arg, before)  # left alone
    # Without the flag an F-ordered A is left alone too.
    f = np.asfortranarray(a)
    assert _lapack.solve(f, b).tobytes() == want.tobytes()
    assert np.array_equal(f, a)


# ---------------------------------------------------------------------------
# `compute_port_solution`
# ---------------------------------------------------------------------------


def _array_deck(segs=22, **extra):
    """momwire#1290's free-space array: 8 parallel 20 m dipoles 8.5 m apart
    at 7 MHz, `segs` segments each (N = 8 * segs)."""
    wires = [
        np.array([(-10.0, 8.5 * j, 10.0), (10.0, 8.5 * j, 10.0)]) for j in range(8)
    ]
    return dict(
        dict(
            wires=wires,
            n_per_edge_per_wire=[[segs] for _ in wires],
            feeds=[(0, 10.0, 1 + 0j)],
            wavelength=299792458.0 / 7e6,
            wire_radius=1e-3,
        ),
        **extra,
    )


def _decks():
    stepped = _array_deck(segs=23)
    stepped["wire_radius"] = [1e-3 if j % 2 else 2.5e-3 for j in range(8)]
    return {
        "free": _array_deck(),
        "free, ek": _array_deck(extended_kernel=True),
        "pec": _array_deck(ground_z=0.0),
        "pec, ek": _array_deck(ground_z=0.0, extended_kernel=True),
        "stepped": stepped,
        "banded": _array_deck(ground_z=0.0, swept_mem_mb=1),
        "two ports": _array_deck(feeds=[(0, 9.6, 1 + 0j), (5, 7.0, 1 + 0j)]),
        "loaded": _array_deck(wire_conductivity=3.5e7),
    }


def _port_solution(kw, order_f):
    """`compute_port_solution`, and the in-place solves it took. `order_f`
    False runs the C-order fill and the copying solve: the route before
    momwire#2071."""
    s = SinusoidalSolver(**kw)
    real = SinusoidalSolver._assemble_Z_filled
    n0 = _lapack.IN_PLACE_SOLVES["count"]
    if order_f:
        ps = s.compute_port_solution()
    else:

        def c_order(self, geom, k, eta=None, obs_rows=None, order="C"):
            return real(self, geom, k, eta, obs_rows=obs_rows)

        with pytest.MonkeyPatch.context() as m:
            m.setattr(SinusoidalSolver, "_assemble_Z_filled", c_order)
            ps = s.compute_port_solution()
    return ps, _lapack.IN_PLACE_SOLVES["count"] - n0


@pytest.mark.skipif(not HAVE_FUSED, reason="no fused sinusoidal fill")
@pytest.mark.parametrize("check", ["on", "off"])
@pytest.mark.parametrize("name", list(_decks()))
def test_port_solution_is_the_c_order_route(name, check, gecon_seen, monkeypatch):
    monkeypatch.setenv(_lapack.CONDITION_CHECK_ENV, "1" if check == "on" else "0")
    kw = _decks()[name]
    ref, ref_in_place = _port_solution(kw, False)
    ref_gecon = list(gecon_seen)
    gecon_seen.clear()
    got, in_place = _port_solution(kw, True)
    assert (ref_in_place, in_place) == (0, 1), name  # the copy-free route ran
    assert got.coeffs.tobytes() == ref.coeffs.tobytes(), name
    assert got.y.tobytes() == ref.y.tobytes(), name
    assert gecon_seen == ref_gecon, name
    assert bool(gecon_seen) == (check == "on"), name


@pytest.mark.parametrize(
    "name, kw",
    [
        ("refl-coef", _array_deck(ground_z=0.0, ground_eps=(13.0, 0.005))),
        ("dense regime", _array_deck(segs=5)),
        # N = 416, over `SMALL_DECK_SEGS_FREE` (PEC counts as free), so the
        # reuse serves.
        ("pair reuse", _array_deck(segs=52, ground_z=0.0, pair_reuse=True)),
    ],
)
def test_a_deck_the_fused_band_declines_keeps_the_copy(name, kw):
    """The two-pass band and `pair_reuse` reduce into contiguous rows only, so
    they fill C order and the solve copies, as before."""
    ref, _ = _port_solution(kw, False)
    got, in_place = _port_solution(kw, True)
    assert in_place == 0, name
    assert got.coeffs.tobytes() == ref.coeffs.tobytes(), name
    assert got.y.tobytes() == ref.y.tobytes(), name


@pytest.mark.skipif(not HAVE_FUSED, reason="no fused sinusoidal fill")
def test_the_order_is_asked_for_only_by_the_port_solution():
    """`compute_impedance` factors Zᵀ of the C-order fill in place; an F-order
    fill there would change its pivots. Its Z stays C-ordered."""
    s = SinusoidalSolver(**_array_deck(ground_z=0.0))
    geom = s._build_geometry()
    G, _ = s._assemble_Z(geom, s.k)
    assert G.flags.c_contiguous
    F, _ = s._assemble_Z(geom, s.k, order="F")
    assert F.flags.f_contiguous and not F.flags.c_contiguous
    assert np.ascontiguousarray(F).tobytes() == G.tobytes()
    n0 = _lapack.IN_PLACE_SOLVES["count"]
    s.compute_impedance()
    assert _lapack.IN_PLACE_SOLVES["count"] == n0
