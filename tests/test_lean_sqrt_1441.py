"""`mw_branch::lean_sqrt` (src/momwire/_branch_cut_inline.h) against
std::sqrt (momwire#1441 step 2).

gamma_cut's two-sqrt spelling IS the branch choice (momwire#714), so the
lean root must keep std::sqrt's cut exactly: Re >= 0 always, Im with the
sign of the operand's Im, signed zero included, so the negative real axis
lands on +j or -j by the sign of its zero. A purely imaginary operand must
come back as (t, +-t) from one sqrt, glibc's own branch (the general formula
would form the second part as y / (2t), which can differ from t in the last
bit: the numpy-vs-cmath trap). Everywhere else it is glibc's algorithm with
hypot replaced by sqrt(x*x + y*y), so it may differ from std::sqrt by a few
ulps and never by more; outside its guard it IS std::sqrt.

What the few-ulp difference does downstream is measured, not pinned here:
it flips a handful of the adaptive quadrature's rtol = 1e-9 accept/split
decisions, so table nodes move by up to ~3e-9 of their region's scale and
the feed Z of the buried corpus by up to ~6e-12 relative (|Z| > 1e-6 ohm).
The tolerance for that, 1e-11 relative on Z, was approved by Steve on
momwire#1511 as an exception to the eval-order policy's ~1e-12.

A small C++ program does the comparing, built with the extension's own
floating-point flags. The exact-match edge cases are glibc's (Linux); the
ulp bound holds against any library's correctly rounded-ish csqrt.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "momwire"


def _cxx():
    for name in (os.environ.get("CXX", ""), "g++", "c++"):
        path = shutil.which(name) if name else None
        if path is not None:
            return path
    return None


_CXX = _cxx()

pytestmark = [
    pytest.mark.skipif(_CXX is None, reason="no C++ compiler on this box"),
    pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason="the exact edge cases are glibc's csqrt",
    ),
]

_PROGRAM = r"""
#include "_branch_cut_inline.h"
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
typedef std::complex<double> cd;
static bool same(const cd &p, const cd &q) {
    return std::memcmp(&p, &q, sizeof(cd)) == 0;
}
static int64_t ulps(double a, double b) {
    if (a == b) return 0;
    int64_t ia, ib;
    std::memcpy(&ia, &a, 8);
    std::memcpy(&ib, &b, 8);
    if ((ia < 0) != (ib < 0)) return INT64_MAX;
    return ia > ib ? ia - ib : ib - ia;
}
int main() {
    long edge_bad = 0, branch_bad = 0, n = 0, differ = 0;
    int64_t worst = 0;
    // Exact: zeros, the axes, signed zeros, the negative real axis, purely
    // imaginary operands, and the non-finite / extreme values the guard hands
    // to std::sqrt.
    const double ev[] = {0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.25, -0.25, 3.0,
                         -7.5, 1e-200, -1e-200, 1e200, -1e200, DBL_MIN,
                         DBL_MAX, -DBL_MAX, INFINITY, -INFINITY, NAN};
    for (double x : ev)
        for (double y : ev) {
            const cd z(x, y), p = mw_branch::lean_sqrt(z), q = std::sqrt(z);
            const bool nan_both = std::isnan(p.real()) == std::isnan(q.real()) &&
                                  std::isnan(p.imag()) == std::isnan(q.imag()) &&
                                  (std::isnan(p.real()) || std::isnan(p.imag()));
            if (!same(p, q) && !nan_both) {
                ++edge_bad;
                std::fprintf(stderr, "edge (%a,%a): lean (%a,%a) std (%a,%a)\n",
                             x, y, p.real(), p.imag(), q.real(), q.imag());
            }
        }
    std::mt19937_64 g(1441);
    std::uniform_real_distribution<double> u(-1.0, 1.0), e(-140.0, 140.0);
    for (long i = 0; i < 2000000; ++i) {
        double x = u(g) * std::pow(10.0, e(g));
        double y = u(g) * std::pow(10.0, (i % 3 == 0) ? e(g) : std::log10(std::fabs(x) + 1e-300) + 4.0 * u(g));
        if (i % 11 == 0) x = 0.0;          // purely imaginary: must be exact
        if (i % 13 == 0) y = (i % 26 == 0) ? 0.0 : -0.0;  // real axis, both zeros
        const cd z(x, y), p = mw_branch::lean_sqrt(z), q = std::sqrt(z);
        ++n;
        // The cut: Re >= 0 (not -0.0 unless std says so) and Im's sign is
        // std::sqrt's, bit for bit, on every operand.
        if (std::signbit(p.real()) != std::signbit(q.real()) ||
            std::signbit(p.imag()) != std::signbit(q.imag()))
            ++branch_bad;
        if ((x == 0.0 || y == 0.0) && !same(p, q)) ++edge_bad;
        if (!same(p, q)) {
            ++differ;
            const int64_t w = std::max(ulps(p.real(), q.real()), ulps(p.imag(), q.imag()));
            if (w > worst) worst = w;
        }
    }
    std::printf("%ld %ld %ld %ld %lld\n", n, edge_bad, branch_bad, differ,
                (long long)worst);
    return 0;
}
"""


def test_the_lean_root_keeps_the_cut_and_stays_within_ulps():
    room = Path(tempfile.mkdtemp(prefix="mw-leansqrt-"))
    try:
        (room / "check.cpp").write_text(_PROGRAM)
        exe = room / "check"
        build = subprocess.run(
            [
                _CXX,
                "-O3",
                "-fno-math-errno",
                "-ffp-contract=off",
                "-std=gnu++17",
                f"-I{SRC}",
                str(room / "check.cpp"),
                "-o",
                str(exe),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert build.returncode == 0, build.stderr
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300)
        assert run.returncode == 0, run.stderr
        n, edge_bad, branch_bad, differ, worst = (int(v) for v in run.stdout.split())
        assert n == 2_000_000
        assert edge_bad == 0, run.stderr
        assert branch_bad == 0
        # hypot vs the plain sum: an ulp in |z|, a few in the root.
        assert worst <= 4, (differ, worst)
    finally:
        shutil.rmtree(room, ignore_errors=True)
