"""`mw_cdiv::div` (src/momwire/_cdiv_inline.h) is libgcc's complex division
bit for bit (momwire#1441 step 1).

The Sommerfeld integrands divide through it instead of calling `__divdc3`,
and the claim that moved nothing is this one: on GCC, every quotient it
returns is the operator's to the last bit, over random operands spanning the
double range, real and imaginary divisors, signed zeros and the non-finite
edges (which take the operator itself). A small C++ program does the
comparing, built with the extension's own floating-point flags.

GCC only: under clang and MSVC the helper IS the operator (their division
rounds differently from libgcc's, so the identity is not theirs to keep).
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "momwire"


def _cxx():
    for name in (os.environ.get("CXX", ""), "g++", "c++"):
        path = shutil.which(name) if name else None
        if path is None:
            continue
        version = subprocess.run(
            [path, "--version"], capture_output=True, text=True
        ).stdout
        if "clang" not in version.lower():  # macOS's g++ is clang
            return path
    return None


_CXX = _cxx()

pytestmark = pytest.mark.skipif(
    _CXX is None, reason="no GCC C++ compiler on this box (the identity is GCC's)"
)

_PROGRAM = r"""
#include "_cdiv_inline.h"
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <random>
typedef std::complex<double> cd;
static bool same(const cd &p, const cd &q) {
    return std::memcmp(&p, &q, sizeof(cd)) == 0;
}
static bool both_nan(const cd &p, const cd &q) {
    return std::isnan(p.real()) == std::isnan(q.real()) &&
           std::isnan(p.imag()) == std::isnan(q.imag()) &&
           (std::isnan(p.real()) || std::isnan(p.imag()));
}
int main() {
    std::mt19937_64 g(1441);
    std::uniform_real_distribution<double> u(-1.0, 1.0), e(-160.0, 160.0),
        wide(-320.0, 320.0);
    auto num = [&](double ex) { return u(g) * std::pow(10.0, ex); };
    long n = 0, bad = 0;
    for (long i = 0; i < 2000000; ++i) {
        const double sc = (i % 4 == 0) ? wide(g) : e(g);
        cd x(num(sc), num(e(g))), y(num(e(g)), num((i % 3 == 0) ? e(g) : sc));
        if (i % 17 == 0) x = cd(x.real(), 0.0);
        if (i % 19 == 0) x = cd(0.0, x.imag());
        if (i % 31 == 0) x = cd(-0.0, x.imag());
        if (i % 23 == 0) y = cd(y.real(), 0.0);
        if (i % 29 == 0) y = cd(0.0, y.imag());
        if (i % 41 == 0) y = cd(y.real(), -0.0);
        if (i % 37 == 0) y = cd(y.real(), y.real() * (1.0 + 1e-16 * u(g)));
        ++n;
        if (!same(mw_cdiv::div(x, y), x / y)) ++bad;
        const double s = num(e(g));
        ++n;
        if (!same(mw_cdiv::div(s, y), s / y)) ++bad;
    }
    const double edge[] = {0.0,     -0.0,     1.0,      -1.0,      DBL_MIN,
                           DBL_MIN / 4, DBL_MAX, INFINITY, -INFINITY, NAN,
                           1e-200,  1e200,    3.0549363634996e-151,
                           3.2733906078961e150};
    for (double a : edge)
        for (double b : edge)
            for (double c : edge)
                for (double d : edge) {
                    const cd x(a, b), y(c, d);
                    const cd p = mw_cdiv::div(x, y), q = x / y;
                    ++n;
                    if (!same(p, q) && !both_nan(p, q)) ++bad;
                }
    std::printf("%ld %ld\n", n, bad);
    return 0;
}
"""


def test_the_divide_is_libgccs_bit_for_bit():
    room = Path(tempfile.mkdtemp(prefix="mw-cdiv-"))
    try:
        (room / "check.cpp").write_text(_PROGRAM)
        exe = room / "check"
        # The extension's own floating-point flags (setup.py's Linux branch).
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
        n, bad = (int(v) for v in run.stdout.split())
        assert n > 4_000_000
        assert bad == 0, f"{bad} of {n} quotients differ from the operator's"
    finally:
        shutil.rmtree(room, ignore_errors=True)
