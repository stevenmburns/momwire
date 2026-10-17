"""`mw_fma::exact` (_fma_inline.h): fma(a, b, c) correctly rounded on every build.

The explicit fused sites (the numpy-shaped complex product `np_cmul` in the
razor, Sommerfeld and crossing cores, `_cmul.cmul`) spell `mw_fma::exact`.
On a build with FMA hardware it is `std::fma`; on the x86-64 baseline `_sse2`
variant it is `soft_exact`, an inline emulation (Dekker product, TwoSum, the
low part rounded to odd: Boldo & Melquiond 2008) that replaces a call into
libm's software fma, ~57 ns each on a CPU without the instruction. It is a
speed-up only if it returns the same double, so the gates are bit equality:

1. Random triples (uniform, wide exponents, near-total cancellation, ties,
   subnormal results), compared as uint64 with libm's `fma` -- only where the
   CPU has FMA, so the reference is the instruction itself.
2. Every edge class `soft_exact` routes (zero, subnormal, inf, NaN operands;
   each boundary of its fast path, from both sides; overflowing and
   underflowing results) against fma computed in exact rational arithmetic
   and rounded once, on any CPU.
3. Both routes: `exact` as this build compiles it, and `soft_exact` itself,
   which GCC/clang compile on the FMA builds too; and the same in a child
   forced onto the baseline variant, where `exact` IS the emulation.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest

from momwire import _accel

ACC = _accel.acc if _accel.LOADED else None
HOOK = getattr(ACC, "fma_route", None)

pytestmark = pytest.mark.skipif(
    HOOK is None, reason="the accelerator is not loaded (or carries no fma_route)"
)

# The avx2 variant only loads on a CPU with AVX2 and FMA (_accel's CPU check),
# and arm64 has FMA in every core: there libm's fma is the instruction.
CPU_HAS_FMA = _accel.VARIANT == "avx2" or (
    _accel.VARIANT == "legacy" and getattr(ACC, "fma_exact_soft", True) is False
)


def _bits(x):
    return np.ascontiguousarray(x, dtype=np.float64).view(np.uint64)


def _assert_same(got, want, a, b, c, what):
    """uint64 equality, any NaN equal to any NaN (fma's NaN payload is not
    part of the contract)."""
    got, want = np.asarray(got), np.asarray(want)
    both_nan = np.isnan(got) & np.isnan(want)
    bad = np.flatnonzero((_bits(got) != _bits(want)) & ~both_nan)
    if bad.size:
        i = bad[0]
        pytest.fail(
            f"{what}: {bad.size} of {got.size} differ; first fma({a[i].hex()}, "
            f"{b[i].hex()}, {c[i].hex()}) = {got[i].hex()}, want {want[i].hex()}"
        )


def _fma_rational(x, y, z):
    """fma(x, y, z) by exact rational arithmetic, rounded once to nearest."""
    if not (math.isfinite(x) and math.isfinite(y)):
        return x * y + z  # an inf or NaN product is exact, so this is fma's
    if not math.isfinite(z):
        return z  # a finite product never overflows inside fma
    exact = Fraction(x) * Fraction(y) + Fraction(z)
    if exact == 0:
        # A rational has no signed zero: IEEE's rule for an exact zero sum.
        if x * y == 0 and z == 0:
            product_negative = math.copysign(1.0, x) * math.copysign(1.0, y) < 0
            if product_negative and math.copysign(1.0, z) < 0:
                return -0.0
        return 0.0
    try:
        return float(exact)
    except OverflowError:
        return math.inf if exact > 0 else -math.inf


def _random_triples(rng, n):
    """The classes, concatenated: each is n triples."""
    sign = lambda: rng.choice([-1.0, 1.0], n)  # noqa: E731
    mant = lambda: 1.0 + rng.random(n)  # noqa: E731
    out = []
    # Uniform, the shape of the physics.
    out.append(tuple(rng.uniform(-4.0, 4.0, n) for _ in range(3)))
    # Exponents across the whole range, fast path and fallback both.
    out.append(
        tuple(np.ldexp(mant() * sign(), rng.integers(-1074, 1024, n)) for _ in range(3))
    )
    # Near-total cancellation: c within a few ulps of -(a*b).
    a = np.ldexp(mant() * sign(), rng.integers(-100, 100, n))
    b = np.ldexp(mant(), rng.integers(-100, 100, n))
    c = (-(a * b)).view(np.int64) + rng.integers(-4, 5, n)
    out.append((a, b, c.view(np.float64)))
    # Ties: c about half an ulp of a*b, so a*b + c sits on a rounding boundary.
    a, b = mant(), mant()
    half_ulp = np.ldexp(1.0, np.frexp(a * b)[1] - 54)
    c = half_ulp * sign() * (1.0 + rng.integers(0, 4, n) * 2.0**-52)
    out.append((a, b, c))
    # Results in the subnormal range: a*b near 2^-960 .. 2^-980, c ~ -(a*b).
    a = np.ldexp(mant(), -480 - rng.integers(0, 10, n))
    b = np.ldexp(mant() * sign(), -480 - rng.integers(0, 10, n))
    c = (-(a * b)).view(np.int64) + rng.integers(-2, 3, n)
    out.append((a, b, c.view(np.float64)))
    # Double-rounding traps, where the low part's rounding to odd is what
    # decides: a*b = 2^k + e with p = RN(a*b) = 2^k exactly and e a few bits
    # below p's ulp, and c whose half-ulp is 2^k, so c + p is a tie and
    # only e's sign (which a nearest-rounded e + tl loses) breaks it.
    b = mant() * sign()
    k = rng.integers(-60, 60, n)
    a = np.ldexp(1.0, k) / b
    c = np.ldexp((2.0**52 + rng.integers(0, 2**52, n)) * sign(), k + 1)
    out.append((a, b, c))
    # Raw bit patterns: NaN, inf, subnormal, everything.
    out.append(
        tuple(np.frombuffer(rng.bytes(8 * n), dtype=np.float64) for _ in range(3))
    )
    return tuple(np.concatenate([t[k] for t in out]) for k in range(3))


def _edge_triples():
    """Every class `soft_exact` routes, and each fast-path boundary from both
    sides (|a|, |b| at 2^995 / 2^996; e_a + e_b at -960 / -961 and 1000 /
    1001; |c| at 2^1000 / 2^1001)."""
    inf, nan = math.inf, math.nan
    tiny = 5e-324
    sub = 2.0**-1030 * 1.25
    big = 1.7976931348623157e308
    m = 1.0 + 2.0**-30
    specials = [0.0, -0.0, 1.0, -1.0, inf, -inf, nan, tiny, -tiny, sub, big, -big]
    rows = []
    for x in specials:
        for y in specials:
            for z in specials:
                rows.append((x, y, z))
    for ea in (994, 995, 996, 1000, 1023):
        for eb in (-1022, -980, -961, -960, -40, 0, 4, 5, 6, 28, 29):
            for zc in (
                0.0,
                1.0,
                -math.ldexp(1.0, min(ea + eb, 1023)),
                2.0**1000 * 1.5,
                2.0**1001,
            ):
                rows.append((math.ldexp(m, ea), math.ldexp(-m, eb), zc))
    for es in (-962, -961, -960, -959, 999, 1000, 1001, 1002):
        ea = es // 2
        a, b = math.ldexp(m, ea), math.ldexp(m * 1.5, es - ea)
        p = a * b
        for zc in (0.0, -p, -p * (1 + 2.0**-52), math.ldexp(1.0, -1074), -big, big):
            rows.append((a, b, zc))
            rows.append((-a, b, zc))
    for ec in (998, 999, 1000, 1001, 1023):
        for zc in (math.ldexp(1.0, ec), math.ldexp(-1.75, ec)):
            rows += [(1.5, m, zc), (big, 2.0, zc), (2.0**600, 2.0**423, zc)]
    # Overflow by the product alone, and by the sum.
    rows += [(2.0**600, 2.0**600, 1.0), (big, 1.0 + 2.0**-52, 0.0), (big, 1.0, big)]
    rows += [(big, 0.5, big * 0.75), (2.0**1000, 2.0**23, -(2.0**1023))]
    # Double-rounding traps (see _random_triples): a*b = 128 - 2^-53, which
    # rounds to 128, against c = 2^60 + 256 (odd significand, so c + 128 is a
    # tie that goes to even) and c = 2^60 (even).
    a, b = 1.0 + 2.0**-30, 128.0 * (1.0 - 2.0**-30)
    rows += [(a, b, 2.0**60 + 256.0), (a, b, 2.0**60), (-a, b, -(2.0**60 + 256.0))]
    # Exact zero sums: the sign of the zero.
    rows += [(m, m, -(m * m)), (-m, m, m * m), (1.0, 0.5, -0.5), (0.5, -1.0, 0.5)]
    # Subnormal operands with a large partner (raw exponent field 0).
    rows += [(sub, 2.0**200, 1.0), (tiny, 2.0**1000, 0.0), (-sub, -(2.0**60), sub)]
    a, b, c = (np.array(col, dtype=np.float64) for col in zip(*rows))
    return a, b, c


@pytest.fixture(scope="module")
def triples():
    return _random_triples(np.random.default_rng(20261017), 200_000)


@pytest.fixture(scope="module")
def edges():
    a, b, c = _edge_triples()
    ref = np.array(
        [_fma_rational(x, y, z) for x, y, z in zip(a.tolist(), b.tolist(), c.tolist())]
    )
    return a, b, c, ref


def _routes():
    out = ["exact"]
    if getattr(ACC, "fma_have_soft_exact", False):
        out.append("soft")
    return out


def test_this_build_routes_as_documented():
    """`exact` is the emulation exactly where the build has no FMA."""
    if _accel.VARIANT == "avx2":
        assert ACC.fma_exact_soft is False
        assert ACC.fma_have_soft_exact is (sys.platform != "win32")
    elif _accel.VARIANT == "sse2":
        assert ACC.fma_exact_soft is True
        assert ACC.fma_have_soft_exact is True


@pytest.mark.skipif(
    not CPU_HAS_FMA, reason="libm's fma is the reference only on FMA hardware"
)
@pytest.mark.parametrize("route", _routes())
def test_random_triples_are_libm_fma_to_the_bit(route, triples):
    a, b, c = triples
    want = HOOK("libm", a, b, c)
    _assert_same(HOOK(route, a, b, c), want, a, b, c, route)


@pytest.mark.parametrize("route", _routes())
def test_edges_are_the_rational_fma_to_the_bit(route, edges):
    a, b, c, ref = edges
    with np.errstate(all="ignore"):
        _assert_same(HOOK(route, a, b, c), ref, a, b, c, route)
        if CPU_HAS_FMA:
            _assert_same(HOOK("libm", a, b, c), ref, a, b, c, "libm")


def test_the_edges_reach_every_branch(edges):
    """The fixture holds triples on both sides of each fast-path test, or the
    gate above says nothing about the boundaries."""
    a, b, c, _ref = edges
    ea = (a.view(np.uint64) >> 52) & 0x7FF
    eb = (b.view(np.uint64) >> 52) & 0x7FF
    ec = (c.view(np.uint64) >> 52) & 0x7FF
    eab = ea + eb
    fast = (ea >= 1) & (eb >= 1) & (ea <= 2018) & (eb <= 2018)
    fast &= (eab >= 1086) & (eab <= 3046) & (ec <= 2023)
    assert fast.sum() > 50 and (~fast).sum() > 50
    for edge in (2018, 2019):
        assert np.any(ea == edge)
    for edge in (1085, 1086, 3046, 3047):
        assert np.any(eab == edge), edge
    for edge in (2023, 2024):
        assert np.any(ec == edge), edge


def test_the_baseline_variant_is_libm_fma_to_the_bit(triples, edges, tmp_path):
    """In the x86-64 baseline build `exact` IS the emulation. The two variants
    cannot share a process (pybind11 registers their types once), so it runs
    in a child forced onto the baseline variant, against this process's
    references (libm's fma on FMA hardware, and the rational one)."""
    if _accel.VARIANT != "avx2":
        pytest.skip(f"no separate baseline variant beside this one ({_accel.VARIANT})")
    if not _accel._extension_built("_sse2"):
        pytest.skip("the x86-64 baseline variant is not built")
    a, b, c = (np.concatenate([t, e]) for t, e in zip(triples, edges[:3]))
    want = HOOK("libm", a, b, c)
    for k, v in {"a": a, "b": b, "c": c}.items():
        np.save(tmp_path / f"{k}.npy", v)
    child = (
        "import sys, numpy as np\n"
        "from momwire import _accel\n"
        "assert _accel.VARIANT == 'sse2', _accel.VARIANT\n"
        "assert _accel.acc.fma_exact_soft is True\n"
        "d = sys.argv[1]\n"
        "a, b, c = (np.load(d + f'/{k}.npy') for k in 'abc')\n"
        "with np.errstate(all='ignore'):\n"
        "    np.save(d + '/out.npy', _accel.acc.fma_route('exact', a, b, c))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", child, str(tmp_path)],
        env=dict(os.environ, MOMWIRE_FORCE_VARIANT="sse2"),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    got = np.load(tmp_path / "out.npy")
    _assert_same(got, want, a, b, c, "sse2 exact")
    n = edges[0].size
    _assert_same(got[-n:], edges[3], a[-n:], b[-n:], c[-n:], "sse2 exact vs rational")
