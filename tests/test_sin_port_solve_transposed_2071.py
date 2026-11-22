"""The point-matched port solution factors Zᵀ in Z's own storage
(momwire#2071).

Every sinusoidal fill writes Z row-major, which LAPACK reads as Zᵀ. So
`_lapack.solve_transposed` hands getrf ``G.T`` (F-contiguous, G's bytes) and
solves with getrs trans='T': one dense matrix through the factor, where
`_lapack.solve` held G and its Fortran copy. What is held here:

  * the in-place norm (`finite_1norm_c`, and lange("I") of the F view
    without the accelerator) is the copying pass's ‖Z‖₁, bit for bit, at
    every thread count: gecon is handed exactly the anorm `solve` hands it;
  * `solve_transposed` IS ``lu_factor(a.T, overwrite_a=True)`` then
    ``lu_solve(trans=1)``, bit for bit, on the default and the host route;
    the factors land in ``a``'s own storage; anything else is copied once
    and gives the same bits;
  * `solve`'s contract: ValueError on a non-finite input, LinAlgError on an
    exactly singular matrix, the ill-conditioning LinAlgWarning, gecon only
    under the switch or the pivot screen, with norm 'I' over Zᵀ's factors
    (the quantity rcond₁(Z) `solve` estimates);
  * `compute_port_solution` and `compute_impedance_swept` take the in-place
    route on every fill route (fused, two-pass, pair_reuse, a finite
    ground), and their answers are that reference's bits; against the
    copying solve they move by no more than the LU's backward-error bound.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.linalg

from momwire import _lapack
from momwire._accel import acc
from momwire.sinusoidal import SinusoidalSolver

HAVE_KERNEL = acc is not None and getattr(acc, "solve_prep_2071", False)
EPS = np.finfo(np.float64).eps


def _rand(rng, shape, dtype):
    a = rng.standard_normal(shape)
    if np.issubdtype(dtype, np.complexfloating):
        a = a + 1j * rng.standard_normal(shape)
    return a.astype(dtype)


def _copy_norm(a):
    """`solve`'s anorm: lange("1") of the Fortran copy."""
    f = np.asfortranarray(a)
    return float(scipy.linalg.get_lapack_funcs("lange", (f,))("1", f))


# ---------------------------------------------------------------------------
# the in-place norm
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAVE_KERNEL, reason="no finite_1norm_c in the accelerator")
@pytest.mark.parametrize("dtype", [np.complex128, np.float64])
@pytest.mark.parametrize("shape", [(1, 1), (64, 64), (130, 77), (77, 130), (300, 300)])
def test_in_place_norm_is_the_copying_norm(dtype, shape):
    rng = np.random.default_rng(2071)
    a = _rand(rng, shape, dtype)
    # A wide dynamic range, so the sums' order would show if it moved.
    a *= np.exp(rng.uniform(-30, 30, shape))
    a = np.ascontiguousarray(a)
    want = _copy_norm(a)
    _, _, copied = acc.fortran_copy_1norm(a)
    assert copied == want
    before = a.tobytes()
    for nt in (1, 3, 0):
        finite, anorm = acc.finite_1norm_c(a, nt)
        assert finite
        assert anorm == want, (nt, anorm, want)
    assert a.tobytes() == before  # read, not written


@pytest.mark.skipif(not HAVE_KERNEL, reason="no finite_1norm_c in the accelerator")
@pytest.mark.parametrize("where", ["re-nan", "im-inf", "re-ninf"])
def test_in_place_norm_sees_a_non_finite_entry(where):
    rng = np.random.default_rng(2)
    a = _rand(rng, (100, 100), np.complex128)
    v = {"re-nan": complex(np.nan, 0.0), "im-inf": complex(0.0, np.inf)}.get(
        where, complex(-np.inf, 1.0)
    )
    a[97, 3] = v
    finite, _ = acc.finite_1norm_c(a)
    assert not finite
    with pytest.raises(RuntimeError, match="C-contiguous"):
        acc.finite_1norm_c(np.asfortranarray(a))


@pytest.mark.parametrize("dtype", [np.complex128, np.float64])
def test_the_norm_without_the_accelerator_is_the_same(dtype, monkeypatch):
    """lange("I") of the F view sums each column of Z in row order: the copy's
    lange("1") chains."""
    monkeypatch.setattr(_lapack, "_SOLVE_PREP_ACCEL", False)
    rng = np.random.default_rng(5)
    a = np.ascontiguousarray(
        _rand(rng, (150, 150), dtype) * np.exp(rng.uniform(-30, 30, (150, 150)))
    )
    assert _lapack._finite_1norm_c(a) == (True, _copy_norm(a))
    a[3, 140] = np.inf
    assert _lapack._finite_1norm_c(a)[0] is False


# ---------------------------------------------------------------------------
# `_lapack.solve_transposed`
# ---------------------------------------------------------------------------


@pytest.fixture
def gecon_seen(monkeypatch):
    """Every gecon `_lapack` runs, as (anorm, norm, rcond)."""
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
                    seen.append((float(anorm), gk.get("norm"), float(r[0])))
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
            assert np.shares_memory(lu, a)
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


def _reference(a, b):
    """The algorithm, spelled with scipy's public pair: getrf on aᵀ, getrs 'T'."""
    lu_piv = scipy.linalg.lu_factor(np.array(a).T, overwrite_a=True)
    return scipy.linalg.lu_solve(lu_piv, b, trans=1)


def _ill(rng, n, dtype):
    q1, _ = np.linalg.qr(_rand(rng, (n, n), dtype))
    q2, _ = np.linalg.qr(_rand(rng, (n, n), dtype))
    return np.ascontiguousarray((q1 * np.logspace(0, -19, n)) @ q2)


def _cases(dtype, n=200):
    rng = np.random.default_rng(3)
    fine = _rand(rng, (n, n), dtype) + n * np.eye(n, dtype=dtype)
    nan = fine.copy()
    nan[150, 7] = np.nan
    singular = fine.copy()
    singular[:, 11] = 0.0
    b = _rand(rng, (n,), dtype)
    return {
        "fine": (fine, b),
        "fine, many rhs": (fine, _rand(rng, (n, 3), dtype)),
        "symmetric": (np.ascontiguousarray(fine + fine.T), b),
        "ill": (_ill(rng, n, dtype), b),
        "nan": (nan, b),
        "nan rhs": (fine, np.where(np.arange(n) == 5, np.nan, b)),
        "singular": (singular, b),
    }


_EXPECT = {
    "fine": (None, []),
    "fine, many rhs": (None, []),
    "symmetric": (None, []),
    "ill": (None, ["LinAlgWarning"]),
    "nan": (ValueError, []),
    "nan rhs": (ValueError, []),
    "singular": (np.linalg.LinAlgError, []),
}


@pytest.mark.parametrize("accel", [True, False])
@pytest.mark.parametrize("check", ["on", "off"])
@pytest.mark.parametrize("dtype", [np.complex128, np.float64])
def test_solve_transposed_is_the_transposed_factor(
    route, dtype, check, accel, gecon_seen, monkeypatch
):
    monkeypatch.setenv(_lapack.CONDITION_CHECK_ENV, "1" if check == "on" else "0")
    monkeypatch.setattr(_lapack, "_SOLVE_PREP_ACCEL", accel)
    for name, (a, b) in _cases(dtype).items():
        gecon_seen.clear()
        work = a.copy()
        n0 = dict(_lapack.TRANSPOSED_SOLVES)
        x, exc, warned = _outcome(_lapack.solve_transposed, work, b)
        assert (exc, warned) == _EXPECT[name], name
        assert _lapack.TRANSPOSED_SOLVES["in_place"] == n0["in_place"] + 1, name
        assert _lapack.TRANSPOSED_SOLVES["copied"] == n0["copied"], name
        if exc is not None:
            continue
        assert x.tobytes() == _reference(a, b).tobytes(), name
        # The factors are in a's own storage: aᵀ's LU, read column-major.
        lu, _ = scipy.linalg.lu_factor(a.T)
        assert work.T.tobytes(order="F") == lu.tobytes(order="F"), name
        # The estimate: only under the switch or the screen, norm 'I' on the
        # factors of aᵀ, anorm ‖a‖₁ exactly as `solve` takes it.
        if check == "on" or name == "ill":
            assert [(g[0], g[1]) for g in gecon_seen] == [(_copy_norm(a), "I")], name
        else:
            assert gecon_seen == [], name


@pytest.mark.parametrize("dtype", [np.complex128, np.float64])
def test_the_estimate_is_rcond_one_of_z(dtype, gecon_seen, monkeypatch):
    """gecon('I') over Zᵀ's factors estimates rcond₁(Z), the quantity `solve`'s
    gecon('1') over Z's factors estimates: both against the exact value."""
    monkeypatch.setenv(_lapack.CONDITION_CHECK_ENV, "1")
    rng = np.random.default_rng(11)
    for scale in (0.0, 3.0, 8.0):
        a = np.ascontiguousarray(
            _rand(rng, (120, 120), dtype) * np.exp(rng.uniform(-scale, scale, 120))
        )
        exact = 1.0 / (np.linalg.norm(a, 1) * np.linalg.norm(np.linalg.inv(a), 1))
        b = _rand(rng, (120,), dtype)
        gecon_seen.clear()
        _lapack.solve(a, b)
        _lapack.solve_transposed(a.copy(), b)
        (old_anorm, old_norm, old), (new_anorm, new_norm, new) = gecon_seen
        assert (old_norm, new_norm) == ("1", "I")
        assert old_anorm == new_anorm
        # Hager/Higham's estimate never exceeds ‖A⁻¹‖, so rcond is never
        # under the exact one, and in practice within a factor of 10.
        for est in (old, new):
            assert exact * (1 - 1e-12) <= est <= 10 * exact, (scale, est, exact)


def test_what_is_not_writeable_c_order_is_copied_with_the_same_bits(route):
    rng = np.random.default_rng(4)
    a = _rand(rng, (100, 100), np.complex128) + 100 * np.eye(100)
    b = _rand(rng, (100, 2), np.complex128)
    want = _reference(a, b)
    ro = a.copy()
    ro.flags.writeable = False
    for arg, rhs in [
        (np.asfortranarray(a), b),  # F order
        (ro, b),  # read-only
        (a.real.copy(), b),  # cast by a complex rhs
        (np.hstack([a, a])[:, :100], b),  # a strided view
    ]:
        before = arg.copy(order="A")
        n0 = dict(_lapack.TRANSPOSED_SOLVES)
        x = _lapack.solve_transposed(arg, rhs)
        assert _lapack.TRANSPOSED_SOLVES["copied"] == n0["copied"] + 1
        assert _lapack.TRANSPOSED_SOLVES["in_place"] == n0["in_place"]
        assert np.array_equal(arg, before)  # left alone
        if arg.dtype == a.dtype:
            assert x.tobytes() == want.tobytes()


def test_a_one_by_one_or_rectangular_system_is_scipys():
    one = np.array([[2.0 - 1j]])
    b = np.array([1.0 + 0j])
    assert _lapack.solve_transposed(one, b).tobytes() == (
        scipy.linalg.solve(one, b).tobytes()
    )
    with pytest.raises(ValueError):
        _lapack.solve_transposed(np.ones((3, 2)), np.ones(3))


# ---------------------------------------------------------------------------
# the sinusoidal solves
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
        "stepped": stepped,
        "banded": _array_deck(ground_z=0.0, swept_mem_mb=1),
        "dense regime": _array_deck(segs=5),
        "two ports": _array_deck(feeds=[(0, 9.6, 1 + 0j), (5, 7.0, 1 + 0j)]),
        "loaded": _array_deck(wire_conductivity=3.5e7),
        "refl-coef": _array_deck(ground_z=0.0, ground_eps=(13.0, 0.005)),
        "pair reuse": _array_deck(segs=52, ground_z=0.0, pair_reuse=True),
    }


def _captured(solver):
    """Each Z the solver fills, copied before the solve can overwrite it."""
    seen = []
    real = solver._assemble_Z

    def capture(*a, **k):
        G, view = real(*a, **k)
        seen.append(np.array(G, copy=True))
        return G, view

    solver._assemble_Z = capture
    return seen


@pytest.mark.parametrize("name", list(_decks()))
def test_port_solution_factors_z_transposed_in_place(name):
    s = SinusoidalSolver(**_decks()[name])
    seen = _captured(s)
    n0 = dict(_lapack.TRANSPOSED_SOLVES)
    ps = s.compute_port_solution()
    # The route ran, on the fill's own storage, once.
    assert _lapack.TRANSPOSED_SOLVES["in_place"] == n0["in_place"] + 1, name
    assert _lapack.TRANSPOSED_SOLVES["copied"] == n0["copied"], name
    (G,) = seen
    geom = s._build_geometry()
    B = np.zeros((geom["n_segs"], len(geom["feed_segs"])), dtype=np.complex128)
    for j, fi in enumerate(geom["feed_segs"]):
        B[fi, j] = -1.0 / geom["seg_h"][fi]
    assert ps.coeffs.tobytes() == _reference(G, B).tobytes(), name
    # Against the copying solve it replaced: the LU's backward error, about
    # cond(Z) * eps, with a generous constant (both are backward stable; the
    # bound itself is derived in momwire#2071's PR).
    old = scipy.linalg.solve(G, B)
    cond = np.linalg.cond(G, 1)
    rel = np.max(np.abs(ps.coeffs - old)) / np.max(np.abs(old))
    assert rel <= 4 * G.shape[0] * cond * EPS, (name, rel, cond)


def test_the_sweep_factors_z_transposed_in_place():
    kw = _decks()["two ports"]
    s = SinusoidalSolver(**kw)
    k0 = 2 * np.pi / kw["wavelength"]
    ks = k0 * np.array([0.9, 1.0, 1.1])
    n0 = dict(_lapack.TRANSPOSED_SOLVES)
    z = s.compute_impedance_swept(ks)
    assert _lapack.TRANSPOSED_SOLVES["in_place"] == n0["in_place"] + 3
    assert _lapack.TRANSPOSED_SOLVES["copied"] == n0["copied"]
    # Each k is `compute_impedance`'s Z at that k to the bit: it has always
    # factored Zᵀ in place (getrf on G.T, getrs 'T'), the algorithm the sweep
    # now shares.
    for i, k in enumerate(ks):
        one = SinusoidalSolver(**kw)
        one._set_k(k)
        zi, _alpha = one.compute_impedance()
        assert np.asarray(z[i]).tobytes() == np.asarray(zi).tobytes(), i


def test_compute_impedance_keeps_its_own_route():
    """`compute_impedance` already factored Zᵀ in place (`lu_factor_strict`)
    and stashes the factors for the adjoint oracle: not this route."""
    s = SinusoidalSolver(**_decks()["pec"])
    n0 = dict(_lapack.TRANSPOSED_SOLVES)
    s.compute_impedance()
    assert _lapack.TRANSPOSED_SOLVES == n0
    assert s.Z_factors is not None
