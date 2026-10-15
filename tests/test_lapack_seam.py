"""`momwire._lapack`: the one seam every Z-path dense LU goes through, and
every dense complex product on the NEC-5 run path.

Four gates:

1. DEFAULT (nothing published) is the replaced call, bit for bit -- on wide
   random matrices and on moment matrices captured from real decks, with the
   same exceptions and warnings.
2. A FAKE HOST -- function pointers to scipy's own `cython_lapack` zgetrf /
   zgetrs, which are the library `scipy.linalg.lu_factor` calls -- gives the
   default's bits wherever the default is scipy's getrf + getrs, the
   per-slice getrf + getrs answer where it is numpy's gesv (another library),
   and a counter proving the host route ran. End to end, NEC-5 and NEC-2
   printouts over five bases are byte-identical with and without it.
3. A SOURCE SCAN: no module under `src/momwire` calls a dense LU / solve /
   inverse directly any more, apart from an explicit allowlist that says why.
4. THE PRODUCTS (`matmul`, ABI 2): the default and an ABI 1 table are `a @ b`
   bit for bit; an ABI 2 fake host (scipy's own `cython_blas` zgemm / zgemv)
   gives exactly the documented call's bits on that library and a counter
   proving it ran; a source scan holds the run path's product sites to the
   seam; end to end, the printouts come out of the host's products.
"""

import ast
import ctypes
import io
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg
from scipy.linalg import cython_blas, cython_lapack

from momwire import _lapack

SRC = Path(__file__).resolve().parents[1] / "src" / "momwire"
DECKS = Path(__file__).resolve().parent / "fixtures" / "eznec" / "decks"

SIZES = (1, 2, 3, 7, 16, 33, 100, 257)


def _rand(rng, *shape, real=False):
    x = rng.standard_normal(shape)
    if real:
        return x
    return x + 1j * rng.standard_normal(shape)


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _capsule_address(name):
    cap = cython_lapack.__pyx_capi__[name]
    api = ctypes.pythonapi
    api.PyCapsule_GetName.restype = ctypes.c_char_p
    api.PyCapsule_GetName.argtypes = [ctypes.py_object]
    api.PyCapsule_GetPointer.restype = ctypes.c_void_p
    api.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return api.PyCapsule_GetPointer(cap, api.PyCapsule_GetName(cap))


def _table(real=True):
    names = ("zgetrf", "zgetrs") + (("dgetrf", "dgetrs") if real else ())
    return {"abi": 1, **{n: _capsule_address(n) for n in names}}


def _blas_address(name):
    cap = cython_blas.__pyx_capi__[name]
    api = ctypes.pythonapi
    api.PyCapsule_GetName.restype = ctypes.c_char_p
    api.PyCapsule_GetName.argtypes = [ctypes.py_object]
    api.PyCapsule_GetPointer.restype = ctypes.c_void_p
    api.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return api.PyCapsule_GetPointer(cap, api.PyCapsule_GetName(cap))


def _table2():
    """ABI 2: scipy's LU pair and its BLAS zgemm / zgemv."""
    return {
        **_table(),
        "abi": 2,
        "zgemm": _blas_address("zgemm"),
        "zgemv": _blas_address("zgemv"),
    }


@pytest.fixture
def fake_host(monkeypatch):
    """scipy's own LAPACK, published the way a host executable publishes its."""
    monkeypatch.setattr(sys, "_mw_host_lapack", _table(), raising=False)
    yield _lapack.HOST_CALLS


@pytest.fixture
def no_host(monkeypatch):
    monkeypatch.delattr(sys, "_mw_host_lapack", raising=False)


def _calls():
    return dict(_lapack.HOST_CALLS)


def _ran(before, getrf=1, getrs=1):
    now = _lapack.HOST_CALLS
    return (
        now["getrf"] - before["getrf"] >= getrf
        and now["getrs"] - before["getrs"] >= getrs
    )


# ---------------------------------------------------------------------------
# 1. the default route IS the replaced call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("real", [False, True])
def test_default_route_is_the_replaced_call_bit_for_bit(no_host, real):
    rng = np.random.default_rng(1290)
    before = _calls()
    for n in SIZES:
        a = _rand(rng, n, n, real=real)
        b1 = _rand(rng, n, real=real)
        b2 = _rand(rng, n, 5, real=real)
        lu, piv = _lapack.lu_factor(a)
        lu_ref, piv_ref = scipy.linalg.lu_factor(a)
        assert _same(lu, lu_ref) and _same(piv, piv_ref)
        for trans in (0, 1, 2):
            for b in (b1, b2):
                assert _same(
                    _lapack.lu_solve((lu, piv), b, trans=trans),
                    scipy.linalg.lu_solve((lu_ref, piv_ref), b, trans=trans),
                )
        for b in (b1, b2):
            assert _same(_lapack.solve(a, b), scipy.linalg.solve(a, b))
            assert _same(_lapack.solve_stacked(a, b), np.linalg.solve(a, b))
        stack = _rand(rng, 3, n, n, real=real)
        rhs = _rand(rng, 3, n, 2, real=real)
        assert _same(_lapack.solve_stacked(stack, rhs), np.linalg.solve(stack, rhs))
        assert _same(_lapack.inv_stacked(stack), np.linalg.inv(stack))
    assert _calls() == before


def test_default_route_overwrites_as_the_replaced_call_does(no_host):
    rng = np.random.default_rng(7)
    a = np.asfortranarray(_rand(rng, 40, 40))
    a_ref = a.copy(order="F")
    lu, _ = _lapack.lu_factor(a, overwrite_a=True)
    lu_ref, _ = scipy.linalg.lu_factor(a_ref, overwrite_a=True)
    assert np.shares_memory(lu, a) and np.shares_memory(lu_ref, a_ref)
    assert _same(a, a_ref)


def _singular():
    return np.array([[1, 2, 3], [2, 4, 6], [0, 1, 1]], dtype=np.complex128)


def _ill_conditioned(rng, n=12):
    q1, _ = np.linalg.qr(_rand(rng, n, n))
    q2, _ = np.linalg.qr(_rand(rng, n, n))
    s = np.logspace(0, -19, n)
    return (q1 * s) @ q2


def _outcome(fn, *args, **kw):
    """(exception type or None, warning categories) of one call."""
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        try:
            fn(*args, **kw)
            exc = None
        except Exception as e:  # noqa: BLE001 - the type IS the observation
            exc = type(e)
    return exc, sorted({w.category.__name__ for w in seen})


def _contract_cases(rng):
    nan = _rand(rng, 6, 6)
    nan[2, 3] = np.nan
    b = _rand(rng, 12)
    return [
        (
            "lu_factor singular",
            _lapack.lu_factor,
            scipy.linalg.lu_factor,
            (_singular(),),
        ),
        ("lu_factor nan", _lapack.lu_factor, scipy.linalg.lu_factor, (nan,)),
        ("solve singular", _lapack.solve, scipy.linalg.solve, (_singular(), b[:3])),
        ("solve nan", _lapack.solve, scipy.linalg.solve, (nan, b[:6])),
        ("solve ill", _lapack.solve, scipy.linalg.solve, (_ill_conditioned(rng), b)),
        ("solve fine", _lapack.solve, scipy.linalg.solve, (_rand(rng, 12, 12), b)),
        (
            "solve_stacked singular",
            _lapack.solve_stacked,
            np.linalg.solve,
            (_singular(), b[:3]),
        ),
        (
            "inv_stacked singular",
            _lapack.inv_stacked,
            np.linalg.inv,
            (_singular()[None],),
        ),
    ]


def test_default_route_raises_and_warns_as_the_replaced_call(no_host):
    for name, seam, ref, args in _contract_cases(np.random.default_rng(3)):
        assert _outcome(seam, *args) == _outcome(ref, *args), name


@pytest.mark.integration
def test_default_route_on_moment_matrices_from_real_decks(no_host, monkeypatch):
    """Z matrices captured at the seam during real solves, then re-factored
    through the seam and through scipy directly: the same bits."""
    from momwire.eznec import _shell

    captured = []
    real_factor = _lapack.lu_factor

    def spy(a, *args, **kw):
        captured.append(np.array(a, copy=True))
        return real_factor(a, *args, **kw)

    monkeypatch.setattr(_lapack, "lu_factor", spy)
    for deck, basis in (("0033", "bspline"), ("0035", "sinusoidal-galerkin")):
        (path,) = DECKS.glob(f"{deck}_*.nec")
        _shell.render(path.read_text(), basis=basis)
    monkeypatch.setattr(_lapack, "lu_factor", real_factor)
    assert len(captured) >= 2
    rng = np.random.default_rng(11)
    for z in captured:
        rhs = _rand(rng, z.shape[0], 3)
        f = _lapack.lu_factor(z)
        f_ref = scipy.linalg.lu_factor(z)
        assert _same(f[0], f_ref[0]) and _same(f[1], f_ref[1])
        assert _same(_lapack.lu_solve(f, rhs), scipy.linalg.lu_solve(f_ref, rhs))


# ---------------------------------------------------------------------------
# 2. a fake host: the same library, through function pointers
# ---------------------------------------------------------------------------


def _ref_slices(a, b):
    """Per-slice scipy getrf + getrs: the host route's answer on this library."""
    a = np.asarray(a)
    out = np.empty(
        np.broadcast_shapes(a.shape[:-2], b.shape[:-2]) + b.shape[-2:],
        dtype=np.result_type(a, b),
    )
    a = np.broadcast_to(a, out.shape[:-2] + a.shape[-2:])
    b = np.broadcast_to(b, out.shape)
    for idx in np.ndindex(*out.shape[:-2]):
        out[idx] = scipy.linalg.lu_solve(scipy.linalg.lu_factor(a[idx]), b[idx])
    return out


@pytest.mark.parametrize("real", [False, True])
def test_fake_host_gives_scipys_bits_and_runs(fake_host, real):
    rng = np.random.default_rng(1479)
    before = _calls()
    for n in SIZES:
        a = _rand(rng, n, n, real=real)
        b1 = _rand(rng, n, real=real)
        b2 = _rand(rng, n, 4, real=real)
        lu, piv = _lapack.lu_factor(a)
        lu_ref, piv_ref = scipy.linalg.lu_factor(a)
        assert _same(lu, lu_ref) and _same(piv, piv_ref), n
        for trans in (0, 1, 2):
            for b in (b1, b2):
                x = _lapack.lu_solve((lu, piv), b, trans=trans)
                assert _same(
                    x, scipy.linalg.lu_solve((lu_ref, piv_ref), b, trans=trans)
                )
        for b in (b1, b2):
            assert _same(_lapack.solve(a, b), scipy.linalg.solve(a, b)), n
            got = _lapack.solve_stacked(a, b)
            ref = _ref_slices(a, b[:, None] if b.ndim == 1 else b)
            assert _same(got, ref[..., 0] if b.ndim == 1 else ref)
            np.testing.assert_allclose(
                got, np.linalg.solve(a, b), rtol=1e-8, atol=1e-10
            )
        stack = _rand(rng, 3, n, n, real=real)
        rhs = _rand(rng, 3, n, 2, real=real)
        assert _same(_lapack.solve_stacked(stack, rhs), _ref_slices(stack, rhs))
        eye = np.broadcast_to(np.eye(n, dtype=stack.dtype), stack.shape)
        assert _same(_lapack.inv_stacked(stack), _ref_slices(stack, eye))
    assert _ran(before, getrf=6 * len(SIZES), getrs=6 * len(SIZES))


def test_fake_host_complex_lu_takes_a_real_rhs(fake_host):
    rng = np.random.default_rng(5)
    a = _rand(rng, 9, 9)
    b = _rand(rng, 9, real=True)
    f = scipy.linalg.lu_factor(a)
    before = _calls()
    assert _same(_lapack.lu_solve(f, b), scipy.linalg.lu_solve(f, b))
    assert _ran(before, getrf=0)


def test_fake_host_overwrites_in_place_as_scipy_does(fake_host):
    rng = np.random.default_rng(8)
    a = np.asfortranarray(_rand(rng, 30, 30))
    a_ref = a.copy(order="F")
    before = _calls()
    lu, _ = _lapack.lu_factor(a, overwrite_a=True)
    scipy.linalg.lu_factor(a_ref, overwrite_a=True)
    assert np.shares_memory(lu, a) and _same(a, a_ref)
    assert _ran(before, getrs=0)
    c = np.ascontiguousarray(_rand(rng, 30, 30))
    c_copy = c.copy()
    lu, _ = _lapack.lu_factor(c, overwrite_a=True)
    assert not np.shares_memory(lu, c) and _same(c, c_copy)
    b = np.asfortranarray(_rand(rng, 30, 2))
    x = _lapack.lu_solve((lu, _), b, overwrite_b=True)
    assert np.shares_memory(x, b)


def test_fake_host_raises_and_warns_as_the_replaced_call(fake_host):
    before = _calls()
    for name, seam, ref, args in _contract_cases(np.random.default_rng(3)):
        assert _outcome(seam, *args) == _outcome(ref, *args), name
    assert _ran(before)


def test_fake_host_check_finite_off_skips_the_check(fake_host):
    rng = np.random.default_rng(9)
    f = _lapack.lu_factor(_rand(rng, 4, 4))
    b = _rand(rng, 4)
    b[1] = np.inf
    with pytest.raises(ValueError):
        _lapack.lu_solve(f, b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = _lapack.lu_solve(f, b, check_finite=False)
    assert not np.isfinite(x).all()


def test_what_the_host_cannot_take_goes_the_default_way(fake_host, monkeypatch):
    monkeypatch.setattr(sys, "_mw_host_lapack", _table(real=False))
    rng = np.random.default_rng(2)
    before = _calls()
    a = _rand(rng, 6, 6, real=True)
    assert _same(_lapack.lu_factor(a)[0], scipy.linalg.lu_factor(a)[0])
    a32 = _rand(rng, 6, 6).astype(np.complex64)
    assert _same(_lapack.lu_factor(a32)[0], scipy.linalg.lu_factor(a32)[0])
    one = _rand(rng, 1, 1)
    b = _rand(rng, 1)
    assert _same(_lapack.solve(one, b), scipy.linalg.solve(one, b))
    assert _calls() == before


def test_a_bad_table_is_refused_in_plain_words(monkeypatch):
    a = np.eye(3, dtype=np.complex128)
    monkeypatch.setattr(sys, "_mw_host_lapack", {**_table(), "abi": 3}, raising=False)
    with pytest.raises(RuntimeError, match="ABI 3; momwire reads ABI 1 and ABI 2"):
        _lapack.lu_factor(a)
    monkeypatch.setattr(sys, "_mw_host_lapack", {**_table(), "abi": 2})
    with pytest.raises(RuntimeError, match="ABI 2, which must give both zgemm"):
        _lapack.lu_factor(a)
    table = _table2()
    del table["zgemv"]
    monkeypatch.setattr(sys, "_mw_host_lapack", table)
    with pytest.raises(RuntimeError, match="must give both zgemm and zgemv"):
        _lapack.matmul(a, a)
    table = _table(real=False)
    del table["zgetrs"]
    monkeypatch.setattr(sys, "_mw_host_lapack", table)
    with pytest.raises(RuntimeError, match="both zgetrf and zgetrs"):
        _lapack.lu_factor(a)


_TIME = re.compile(r"TIME|[Tt]ime|ELAPSED|[Ee]lapsed|\bm?sec\b|seconds")


def _masked(text):
    return [line for line in text.splitlines() if not _TIME.search(line)]


# The NEC-5 render serves the knot bases only: every deck in that dialect
# names a node, which the point-matched families (sinusoidal, pulse) refuse
# by design. They answer through the NEC-2 portal instead. Decks chosen for
# the sites they reach: a dipole (the plain solve), junctions (the KCL Schur
# solve), networks and two sources (the port-space solves of both seams).
NEC5_DECKS = ("0010", "0033", "0012", "0035")
NEC5_BASES = ("bspline", "razor-2p", "sinusoidal-galerkin")
NEC2_DECKS = (
    "dipole_free_space",
    "catalog_dipoles_invvee",
    "dipole_nt_network",
    "two_source_sensor_lines",
)
NEC2_BASES = ("bspline", "sinusoidal", "sinusoidal-galerkin", "pulse")
NEC2_DIR = Path(__file__).resolve().parent / "fixtures" / "nec_portal"


def _printouts():
    from momwire.eznec import _shell
    from momwire.portal import _portal

    out = {}
    for d in NEC5_DECKS:
        text = next(DECKS.glob(f"{d}_*.nec")).read_text()
        for b in NEC5_BASES:
            out[("nec5", d, b)] = _shell.render(text, basis=b)
    for d in NEC2_DECKS:
        text = (NEC2_DIR / f"{d}.deck").read_text()
        for b in NEC2_BASES:
            buf = io.StringIO()
            with _portal.engine_scope():
                rc = _portal.main(
                    ["--basis", b],
                    stdin=io.StringIO(text),
                    stdout=buf,
                    stderr=io.StringIO(),
                )
            assert rc == 0, (d, b)
            out[("nec2", d, b)] = buf.getvalue()
    return out


@pytest.mark.integration
def test_fake_host_printouts_are_byte_identical_end_to_end(monkeypatch):
    callers = _spy_callers(monkeypatch)
    monkeypatch.delattr(sys, "_mw_host_lapack", raising=False)
    default = _printouts()
    before = _calls()
    monkeypatch.setattr(sys, "_mw_host_lapack", _table(), raising=False)
    host = _printouts()
    assert _ran(before, getrf=len(default), getrs=len(default))
    for key, text in default.items():
        assert "ERROR" not in text, key
        assert _masked(host[key]) == _masked(text), key
    for module in (
        "momwire.bspline",
        "momwire.sinusoidal",
        "momwire.sinusoidal_galerkin",
        "momwire.pulse",
        "momwire.eznec._serve",
        "momwire.portal._portal",
    ):
        assert module in callers, module


def _spy_callers(monkeypatch):
    callers = set()

    def spying(fn):
        def wrapped(*args, **kw):
            callers.add(sys._getframe(1).f_globals["__name__"])
            return fn(*args, **kw)

        return wrapped

    for name in ("lu_factor", "lu_solve", "solve", "solve_stacked", "inv_stacked"):
        monkeypatch.setattr(_lapack, name, spying(getattr(_lapack, name)))
    return callers


def test_fake_host_sector_route_and_array_blocks_are_byte_identical(monkeypatch):
    """The two routes no deck above reaches: the sector (block-circulant)
    solve, and the array-block preconditioner's per-shape LU."""
    from test_rotational_symmetry_1029 import solver as sector_solver

    from momwire import ArrayBlockSolver

    def array():
        wires = [
            np.array([[0.0, y, -5.0], [0.0, y, 5.0]]) for y in (0.0, 12.0, 24.0, 36.0)
        ]
        return ArrayBlockSolver(
            wires=wires,
            degree=2,
            n_per_edge_per_wire=[[12]] * len(wires),
            nsegs=12,
            wavelength=22.0,
            feeds=[(i, None, 1.0 + 0.0j) for i in range(len(wires))],
        )

    callers = _spy_callers(monkeypatch)

    def answers():
        z_sector, c_sector = sector_solver().compute_impedance()
        z_array, c_array = array().compute_impedance()
        return [np.asarray(z_sector), c_sector, np.asarray(z_array), c_array]

    monkeypatch.delattr(sys, "_mw_host_lapack", raising=False)
    default = answers()
    before = _calls()
    monkeypatch.setattr(sys, "_mw_host_lapack", _table(), raising=False)
    host = answers()
    assert _ran(before)
    for got, ref in zip(host, default):
        assert _same(got, ref)
    assert "momwire._rotational_symmetry" in callers
    assert "momwire.array_block" in callers


# ---------------------------------------------------------------------------
# 3. no Z-path module factors behind the seam's back
# ---------------------------------------------------------------------------

_LINALG = {"solve", "inv", "lu_factor", "lu_solve", "lu"}
_ROOTS = {"np", "numpy", "scipy", "la", "sla"}

# (module path under src/momwire, call) -> how many may stay, and why.
ALLOWED = {
    # A 2..5 real Gram matrix of the polynomial bubble basis, solved once per
    # degree at import-time scale: geometry, not a moment matrix.
    ("bspline.py", "np.linalg.solve"): 1,
    # A 1x1 or 2x2 real normal-equation solve recovering a lattice's integer
    # coordinates: geometry, not a moment matrix.
    ("array_block.py", "np.linalg.solve"): 1,
    # The constraint elimination T = -C_p^-1 C_r on the KCL rows (m x m, m =
    # the junction count): never touches G, whose reduced system still goes
    # through the seam.
    ("sinusoidal_galerkin.py", "np.linalg.solve"): 1,
    # The MNA network solve: LAPACK's equilibrating expert driver zgesvx for
    # its scaled condition estimate, a routine the host table does not carry.
    ("networks/_reduce.py", "scipy.linalg.lapack.zgesvx"): 1,
}


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _direct_linalg(tree):
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            name = _dotted(node)
            if name is None:
                continue
            parts = name.split(".")
            if parts[0] in _ROOTS and "linalg" in parts[:-1] and parts[-1] in _LINALG:
                found.append(name)
            if "cython_lapack" in parts or re.fullmatch(
                r"[zdcs](getr[fsi]|gesvx?)", parts[-1]
            ):
                found.append(name)
        elif (
            isinstance(node, ast.ImportFrom) and node.module and "linalg" in node.module
        ):
            for alias in node.names:
                if (
                    alias.name in _LINALG
                    or alias.name == "cython_lapack"
                    or re.fullmatch(r"[zdcs](getr[fsi]|gesvx?)", alias.name)
                ):
                    found.append(f"{node.module}.{alias.name}")
    return found


def test_no_z_path_module_calls_a_dense_lu_directly():
    seen = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel == "_lapack.py":
            continue
        for name in _direct_linalg(ast.parse(path.read_text(), filename=rel)):
            seen[(rel, name)] = seen.get((rel, name), 0) + 1
    unexpected = {k: v for k, v in seen.items() if v > ALLOWED.get(k, 0)}
    assert not unexpected, (
        "a dense LU / solve / inverse outside momwire._lapack; route it through "
        f"the seam or allowlist it with the reason: {unexpected}"
    )
    stale = {k: v for k, v in ALLOWED.items() if seen.get(k, 0) < v}
    assert not stale, f"allowlist entries that no longer match anything: {stale}"


# ---------------------------------------------------------------------------
# 4. the products: `matmul` (ABI 2)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_host2(monkeypatch):
    """scipy's LAPACK and BLAS, published as an ABI 2 table."""
    monkeypatch.setattr(sys, "_mw_host_lapack", _table2(), raising=False)
    yield _lapack.HOST_CALLS


def _operands(rng):
    """(a, b) pairs of every shape and layout the run path hands `matmul`:
    complex and real operands, C- and F-ordered, transposed and sliced
    views, matrix @ matrix / vector, vector @ matrix / vector, 1 x 1 and
    single-row / -column matrices."""
    out = []
    for m, k, n in ((1, 1, 1), (1, 5, 1), (3, 1, 4), (7, 16, 1), (33, 257, 5)):
        a, b = _rand(rng, m, k), _rand(rng, k, n)
        ar, br = _rand(rng, m, k, real=True), _rand(rng, k, n, real=True)
        out += [
            (a, b),
            (np.asfortranarray(a), b),
            (a, np.asfortranarray(b)),
            (ar, b),  # T.T @ y / cols.T @ X: real @ complex
            (a, br),  # (T.T @ y) @ T
            (np.ascontiguousarray(ar.T).T, b),  # a transposed (F) view
            (a, b[:, 0]),  # matrix @ vector
            (ar, b[:, 0]),  # _transform(mesh) @ v_gap
            (a[0], b),  # vector @ matrix
            (a[0], b[:, 0]),  # vector @ vector: U[:, p] @ alpha
            (ar[0], b[:, 0]),
            (a[:, ::-1], b[::-1]),  # negative strides
        ]
    big = _rand(rng, 300, 4)
    out.append((big.T, big))  # numpy's syrk case: a is b's own transpose
    return out


def test_default_matmul_is_the_operator_bit_for_bit(no_host):
    rng = np.random.default_rng(1500)
    before = _calls()
    for a, b in _operands(rng):
        assert _same(_lapack.matmul(a, b), a @ b), (a.shape, b.shape)
    # what the host could never take still answers as `@` does
    i = np.arange(6).reshape(2, 3)
    assert _same(_lapack.matmul(i, i.T), i @ i.T)
    stack = _rand(rng, 2, 3, 3)
    assert _same(_lapack.matmul(stack, stack), stack @ stack)
    assert _calls() == before


def test_an_abi_1_table_leaves_every_product_to_numpy(fake_host):
    rng = np.random.default_rng(1501)
    before = _calls()
    for a, b in _operands(rng):
        assert _same(_lapack.matmul(a, b), a @ b)
    assert _calls() == before


def _host_reference(a, b):
    """The module docstring's call per shape, made through scipy's own f2py
    BLAS wrappers on the same library: the C-contiguous complex128 operands
    read column-major are their transposes, so no wrapper copies them."""
    from scipy.linalg import blas

    a = np.ascontiguousarray(a, dtype=np.complex128)
    b = np.ascontiguousarray(b, dtype=np.complex128)
    if a.ndim == 2 and b.ndim == 2:
        return np.ascontiguousarray(blas.zgemm(1.0, b.T, a.T).T)
    if a.ndim == 2:
        return blas.zgemv(1.0, a.T, b, trans=1)
    if b.ndim == 2:
        return blas.zgemv(1.0, b.T, a, trans=0)
    return blas.zgemv(1.0, a[:, None], b, trans=1)[0]


def test_fake_host_products_are_the_hosts_call_and_run(fake_host2):
    rng = np.random.default_rng(1502)
    for a, b in _operands(rng):
        before = _calls()
        got = _lapack.matmul(a, b)
        ran = {k: _lapack.HOST_CALLS[k] - before[k] for k in ("gemm", "gemv")}
        assert ran == (
            {"gemm": 1, "gemv": 0} if a.ndim == b.ndim == 2 else {"gemm": 0, "gemv": 1}
        ), (a.shape, b.shape)
        ref = _host_reference(a, b)
        assert _same(got, ref), (a.shape, b.shape)
        assert got.flags.c_contiguous or got.ndim == 0
        np.testing.assert_allclose(got, a @ b, rtol=1e-12, atol=1e-12)


def test_fake_host_products_skip_what_the_host_cannot_take(fake_host2):
    rng = np.random.default_rng(1503)
    before = _calls()
    real = _rand(rng, 4, 4, real=True)
    single = _rand(rng, 4, 4).astype(np.complex64)
    stack = _rand(rng, 2, 4, 4)
    empty = np.zeros((4, 0), dtype=np.complex128)
    for a, b in ((real, real), (single, single), (stack, stack), (empty, empty.T)):
        assert _same(_lapack.matmul(a, b), a @ b)
    with pytest.raises(ValueError):
        _lapack.matmul(_rand(rng, 3, 4), _rand(rng, 3, 4))  # numpy's own error
    assert _calls() == before


# Where the NEC-5 run path forms its dense complex products, by module and
# function; each must reach them through `_lapack.matmul` only.
PRODUCT_SITES = {
    "eznec/_serve.py": ("_port_state", "_multi_drive_state", "_finish"),
    "razor.py": ("RazorSolver.compute_port_solution",),
    "bspline.py": ("BSplineSolver.compute_port_solution",),
    "sinusoidal_galerkin.py": (
        "_solve_constrained",
        "SinusoidalGalerkinSolver._port_currents",
    ),
    "deck/_networks.py": ("reduced_ports",),
    "networks/_reduce.py": ("MNASystem.branch_power",),
}


def _functions(tree):
    """{qualified name: node} for module-level functions and class methods."""
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[f"{node.name}.{item.name}"] = item
    return out


def test_the_run_path_products_go_through_the_seam():
    for rel, names in PRODUCT_SITES.items():
        fns = _functions(ast.parse((SRC / rel).read_text(), filename=rel))
        for name in names:
            assert name in fns, (rel, name)
            direct = [
                n.lineno
                for n in ast.walk(fns[name])
                if (isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult))
                or (
                    isinstance(n, ast.Call)
                    and (_dotted(n.func) or "").split(".")[-1]
                    in ("dot", "matmul", "einsum", "vdot", "inner")
                    and _dotted(n.func) != "_lapack.matmul"
                )
            ]
            assert not direct, f"{rel} {name}: a product at lines {direct}"


def _printed_alike(a, b):
    """Two printouts that differ at most in a printed number's last digit
    (a product's last bits, read through a 5-digit cell)."""
    la, lb = _masked(a), _masked(b)
    if len(la) != len(lb):
        return False
    for x, y in zip(la, lb):
        if x == y:
            continue
        tx, ty = x.split(), y.split()
        if len(tx) != len(ty):
            return False
        for u, v in zip(tx, ty):
            if u == v:
                continue
            try:
                fu, fv = float(u), float(v)
            except ValueError:
                return False
            if abs(fu - fv) > 2e-4 * max(abs(fu), abs(fv)) + 1e-30:
                return False
    return True


@pytest.mark.integration
def test_fake_host_products_run_end_to_end(monkeypatch):
    """With an ABI 2 table the NEC-5 and NEC-2 printouts come out of the
    host's products: every routed module made its product through the seam
    and the host ran it. scipy's BLAS is not numpy's, so the products' last
    bits may move against the default (measured on Haswell: 312 of 4,645
    products on razor-2p, rel <= 2e-15) -- and no printout moved there, over
    2,011. The gate asks for no more than a printed last digit."""
    callers = set()
    real = _lapack.matmul

    def spy(a, b):
        callers.add(sys._getframe(1).f_globals["__name__"])
        return real(a, b)

    monkeypatch.setattr(_lapack, "matmul", spy)
    monkeypatch.delattr(sys, "_mw_host_lapack", raising=False)
    default = _printouts()
    before = _calls()
    monkeypatch.setattr(sys, "_mw_host_lapack", _table2(), raising=False)
    host = _printouts()
    assert _lapack.HOST_CALLS["gemm"] - before["gemm"] >= len(NEC5_DECKS) * 2
    assert _lapack.HOST_CALLS["gemv"] - before["gemv"] >= len(NEC5_DECKS) * 2
    for module in (
        "momwire.eznec._serve",
        "momwire.razor",
        "momwire.bspline",
        "momwire.deck._networks",
    ):
        assert module in callers, module
    for key, text in default.items():
        assert "ERROR" not in host[key], key
        assert _printed_alike(host[key], text), key
