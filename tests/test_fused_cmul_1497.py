"""`momwire._cmul.cmul`: the fused elementwise complex multiply (momwire#1497).

numpy's elementwise complex `*` fuses each part into one multiply-add on its
x86-64-v3 / AVX-512 / ASIMD loops and rounds twice on its baseline x86-64
loop. `cmul` always fuses, with numpy's formula:

    re = fma(a.re, b.re, -(a.im * b.im))
    im = fma(a.re, b.im,   a.im * b.re)

Gates:

1. Bit for bit against that formula computed EXACTLY per element (rational
   arithmetic, rounded once), on adversarial cancellation cases where the
   fused and the twice-rounded answers differ -- through the accelerator,
   through its x86-64 baseline variant (libm's / the UCRT's `fma`), and
   through `math.fma` where Python has it.
2. Bit for bit against numpy's `*` on the layouts the run path uses, wherever
   numpy fuses that layout on this CPU; skipped, saying so, where it does not.
3. Broadcasting, dtype, layout and the non-complex128 fallback are numpy's.
4. A source scan holds `_serve._port_state` and `_serve._multi_drive_state`
   to the helper, and an end-to-end render shows the accelerated route ran.
"""

import ast
import math
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from momwire import _accel, _cmul

SRC = Path(__file__).resolve().parents[1] / "src" / "momwire"
DECKS = Path(__file__).resolve().parent / "fixtures" / "eznec" / "decks"

ACCEL = _cmul._fused_cmul is not None
needs_accel = pytest.mark.skipif(
    not ACCEL, reason="the accelerator is not loaded (or carries no fused_cmul)"
)


def _fma_exact(x, y, z):
    """fma(x, y, z): x*y + z exactly, rounded once to the nearest double.

    A rational has no signed zero, so an exact zero takes IEEE's sign rule:
    -0 only when the product and the addend are both negative zeros."""
    exact = Fraction(x) * Fraction(y) + Fraction(z)
    if exact != 0:
        return float(exact)
    product_negative = math.copysign(1.0, x) * math.copysign(1.0, y) < 0
    if x * y == 0 and z == 0 and product_negative and math.copysign(1.0, z) < 0:
        return -0.0
    return 0.0


def _fused_ref(a, b):
    """numpy's fused formula per element, exactly, on flat complex arrays."""
    out = np.empty(a.shape, dtype=np.complex128)
    for i, (p, q) in enumerate(zip(a.tolist(), b.tolist())):
        out[i] = complex(
            _fma_exact(p.real, q.real, -(p.imag * q.imag)),
            _fma_exact(p.real, q.imag, p.imag * q.real),
        )
    return out


def _twice_rounded(a, b):
    """The baseline loop's formula: each product and the sum rounded."""
    return (a.real * b.real - a.imag * b.imag) + 1j * (
        a.real * b.imag + a.imag * b.real
    )


def _same(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


def _adversarial(rng, n=400):
    """Pairs whose real or imaginary part cancels to a few bits of what the
    products hold: there, rounding the products first loses the answer."""
    e = 2.0**-30
    fixed_a = [
        complex(1 + e, 1),
        complex(1 + e, 1),
        complex(3.0, 1 + e),
        complex(-1 - e, 2),
    ]
    fixed_b = [
        complex(1 - e, 1),
        complex(1, -(1 - e)),
        complex(1 - e, 3.0),
        complex(1 - e, -0.5),
    ]
    ar = rng.uniform(0.5, 2.0, n) * rng.choice([-1.0, 1.0], n)
    ai = rng.uniform(0.5, 2.0, n) * rng.choice([-1.0, 1.0], n)
    br = rng.uniform(0.5, 2.0, n) * rng.choice([-1.0, 1.0], n)
    # Half cancel in the real part (ai*bi ~ ar*br), half in the imaginary
    # part (ar*bi ~ -ai*br), each with the last bits nudged.
    nudge = 1 + rng.integers(-4, 5, n) * 2.0**-52
    bi_re = ar * br / ai * nudge
    bi_im = -ai * br / ar * nudge
    bi = np.where(np.arange(n) % 2 == 0, bi_re, bi_im)
    # A sprinkle of magnitudes far from 1, still cancelling.
    scale = 2.0 ** rng.integers(-40, 41, n)
    a = np.concatenate([fixed_a, (ar + 1j * ai) * scale])
    b = np.concatenate([fixed_b, (br + 1j * bi) / np.sqrt(scale)])
    return a, b


@pytest.fixture(scope="module")
def adversarial():
    a, b = _adversarial(np.random.default_rng(1497))
    ref = _fused_ref(a, b)
    # The inputs have to be able to tell fused from twice-rounded, or every
    # comparison below is vacuous.
    differs = np.count_nonzero(ref != _twice_rounded(a, b))
    assert differs > len(a) // 4, differs
    return a, b, ref


# 1. bit for bit against the exact fused formula ------------------------------


@needs_accel
def test_accelerator_is_the_fused_formula_bit_for_bit(adversarial):
    a, b, ref = adversarial
    assert _same(_cmul._fused_cmul(a, b), ref)
    before = _cmul.CALLS["accel"]
    assert _same(_cmul.cmul(a, b), ref)
    assert _cmul.CALLS["accel"] == before + 1


def test_baseline_variant_is_the_fused_formula_bit_for_bit(adversarial, tmp_path):
    """The x86-64 baseline build has no hardware FMA: its `std::fma` is
    libm's (the UCRT's on Windows), which must still round once. The two
    variants cannot share a process (pybind11 registers their types once), so
    the baseline one runs in a child forced onto it."""
    if _accel.VARIANT != "avx2":
        pytest.skip(f"no separate baseline variant beside this one ({_accel.VARIANT})")
    if not _accel._extension_built("_sse2"):
        pytest.skip("the x86-64 baseline variant is not built")
    a, b, ref = adversarial
    np.save(tmp_path / "a.npy", a)
    np.save(tmp_path / "b.npy", b)
    child = (
        "import sys, numpy as np\n"
        "from momwire import _accel, _cmul\n"
        "assert _accel.VARIANT == 'sse2', _accel.VARIANT\n"
        "d = sys.argv[1]\n"
        "a, b = np.load(d + '/a.npy'), np.load(d + '/b.npy')\n"
        "np.save(d + '/out.npy', _cmul.cmul(a, b))\n"
        "assert _cmul.CALLS['accel'] == 1, _cmul.CALLS\n"
    )
    env = dict(os.environ, MOMWIRE_FORCE_VARIANT="sse2")
    done = subprocess.run(
        [sys.executable, "-c", child, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert _same(np.load(tmp_path / "out.npy"), ref)


@pytest.mark.skipif(not hasattr(math, "fma"), reason="math.fma needs Python 3.13")
def test_math_fma_route_is_the_fused_formula_bit_for_bit(adversarial, monkeypatch):
    monkeypatch.setattr(_cmul, "_fused_cmul", None)
    a, b, ref = adversarial
    before = _cmul.CALLS["math_fma"]
    assert _same(_cmul.cmul(a, b), ref)
    assert _cmul.CALLS["math_fma"] == before + 1
    # An element `math.fma` refuses (inf * 0) leaves the whole call to numpy.
    x = np.array([complex(math.inf, 0.0)])
    y = np.array([complex(0.0, 1.0)])
    with np.errstate(invalid="ignore"):
        assert _same(_cmul.cmul(x, y), x * y)


def test_without_accelerator_or_math_fma_it_is_numpy(adversarial, monkeypatch):
    monkeypatch.setattr(_cmul, "_fused_cmul", None)
    monkeypatch.setattr(_cmul, "_math_fma", None)
    a, b, _ref = adversarial
    before = _cmul.CALLS["numpy"]
    assert _same(_cmul.cmul(a, b), a * b)
    assert _cmul.CALLS["numpy"] == before + 1


# 2. bit for bit against numpy where numpy fuses ------------------------------

# The layouts of the eight run-path sites: `z[:, None] * y` (a column
# broadcast against a matrix), `z * i` (two vectors) and `v * scale` (a vector
# by a Python scalar, complex or real).
LAYOUTS = (
    "column_by_matrix",
    "vector_by_vector",
    "vector_by_complex",
    "vector_by_float",
)
LENGTHS = (1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 33)


def _layout(name, a, b, n):
    """Operands of `name` built from the flat pairs (a, b), size about n."""
    if name == "column_by_matrix":
        col = a[:n][:, None]
        mat = np.resize(b, (n, n))
        return col, mat
    if name == "vector_by_vector":
        return a[:n], b[:n]
    if name == "vector_by_complex":
        return a[:n], complex(b[0])
    return a[:n], float(b[0].real)


def _flat_ref(x, y):
    x, y = np.broadcast_arrays(
        np.asarray(x, np.complex128), np.asarray(y, np.complex128)
    )
    return _fused_ref(x.reshape(-1), y.reshape(-1)).reshape(x.shape)


def _numpy_fuses(name):
    """Does numpy's `*` give the fused bits for this layout on this CPU? Asked
    with the cancelling pair at every position, at every probe length, so a
    SIMD body that fuses beside a scalar tail that does not reads as 'no'."""
    e = 2.0**-30
    if name == "vector_by_float":
        # A real factor makes fused and twice-rounded agree (b.im = 0), so
        # there is nothing to probe; it follows the complex-scalar loop.
        name = "vector_by_complex"
    for n in LENGTHS:
        a = np.full(64, complex(1 + e, 1))
        b = np.full(64, complex(1 - e, 1))
        x, y = _layout(name, a, b, n)
        got = x * y
        if not _same(got, _flat_ref(x, y)) or _same(
            got, _twice_rounded(*np.broadcast_arrays(x, y))
        ):
            return False
    return True


@pytest.mark.parametrize("name", LAYOUTS)
def test_equals_numpy_where_numpy_fuses(name, adversarial):
    if not _numpy_fuses(name):
        pytest.skip(f"numpy's `*` does not fuse the {name} layout on this CPU")
    a, b, _ref = adversarial
    rng = np.random.default_rng(7)
    plain_a = rng.standard_normal(64) + 1j * rng.standard_normal(64)
    plain_b = rng.standard_normal(64) + 1j * rng.standard_normal(64)
    for pa, pb in ((a, b), (plain_a, plain_b)):
        for n in LENGTHS:
            x, y = _layout(name, pa, pb, n)
            assert _same(_cmul.cmul(x, y), x * y), (name, n)


# 3. numpy's broadcasting, dtype and layout -----------------------------------


@pytest.mark.parametrize(
    "x, y",
    [
        (
            np.arange(3, dtype=np.complex128)[:, None] + 1j,
            np.ones((3, 3), complex) * (2 - 1j),
        ),
        (np.arange(4.0) + 1j, np.arange(4.0) - 2j),
        (np.arange(4.0) + 1j, 0.5 - 0.25j),
        (np.arange(4.0) + 1j, 3.0),
        (np.arange(4.0), np.arange(4.0) + 1j),  # float64 by complex128
        (np.ones((2, 3), complex)[:, ::2], np.array([1j, 2.0])),  # strided view
        (np.array(1 + 2j), np.array(3 - 1j)),  # 0-d
        (np.zeros((0, 3), complex), np.ones(3, complex)),  # empty
    ],
)
def test_shape_dtype_and_layout_are_numpys(x, y):
    got, want = _cmul.cmul(x, y), x * y
    assert type(got) is type(want)
    assert np.asarray(got).dtype == np.complex128
    assert np.shape(got) == np.shape(want)
    if isinstance(got, np.ndarray):
        assert got.flags.c_contiguous
    assert _same(got, _flat_ref(x, y) if np.size(want) else want)


def test_operands_are_not_written():
    x = np.array([1 + 2j, 3 - 1j])
    y = np.array([0.5j, 2 + 0j])
    xs, ys = x.copy(), y.copy()
    _cmul.cmul(x, y)
    assert _same(x, xs) and _same(y, ys)


@pytest.mark.parametrize(
    "x, y",
    [
        (np.arange(3.0), np.arange(3.0)),  # real: nothing to fuse
        (np.arange(3, dtype=np.complex64) + 1j, 2 - 1j),  # complex64 stays complex64
        (np.arange(3, dtype=np.complex64) + 1j, np.float32(2)),
    ],
)
def test_non_complex128_products_are_numpys_own(x, y):
    before = _cmul.CALLS["numpy"]
    got = _cmul.cmul(x, y)
    assert _same(got, x * y)
    assert _cmul.CALLS["numpy"] == before + 1


# 4. the run path is held to the helper, and it runs --------------------------

# The run path's complex elementwise products, by function, with how many
# `cmul` calls each spells (`_port_state`'s `z_series * i_port` is written
# once in each branch, so #1497's eight products are nine calls). A `*` in these functions is a complex product by
# construction (every operand is a port quantity), so none may remain.
CMUL_SITES = {"eznec/_serve.py": {"_port_state": 7, "_multi_drive_state": 2}}


def _functions(tree):
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
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


def test_the_run_path_complex_products_go_through_cmul():
    for rel, sites in CMUL_SITES.items():
        fns = _functions(ast.parse((SRC / rel).read_text(), filename=rel))
        for name, count in sites.items():
            assert name in fns, (rel, name)
            nodes = list(ast.walk(fns[name]))
            direct = [
                n.lineno
                for n in nodes
                if (
                    isinstance(n, (ast.BinOp, ast.AugAssign))
                    and isinstance(n.op, ast.Mult)
                )
                or (
                    isinstance(n, ast.Call)
                    and (_dotted(n.func) or "").split(".")[-1] == "multiply"
                )
            ]
            assert not direct, f"{rel} {name}: an elementwise product at lines {direct}"
            calls = sum(
                1
                for n in nodes
                if isinstance(n, ast.Call) and _dotted(n.func) == "_cmul.cmul"
            )
            assert calls == count, (rel, name, calls)


@needs_accel
@pytest.mark.parametrize(
    "deck, at_least",
    [
        ("0010", 6),  # one source: `_port_state`'s six
        ("0031", 2),  # four sources: `_multi_drive_state`'s two
    ],
)
def test_a_render_runs_the_accelerated_route(deck, at_least):
    from momwire.eznec import _shell

    text = next(DECKS.glob(f"{deck}_*.nec")).read_text()
    before = dict(_cmul.CALLS)
    out = _shell.render(text, basis="razor-2p")
    assert "ERROR" not in out
    assert _cmul.CALLS["accel"] - before["accel"] >= at_least
    assert _cmul.CALLS["numpy"] == before["numpy"]
    assert _cmul.CALLS["math_fma"] == before["math_fma"]
