"""mw_vmath (`_vmath_inline.h`, momwire#1441): the vector exp and sincos the
near-interface column kernel uses on the builds libmvec cannot reach.

A small C++ harness is compiled against the header twice -- the x86-64
baseline (SSE2 lanes) and, where this CPU runs it, -mavx2 -mfma (AVX2 lanes)
-- with the extension's own floating-point flags, and run over seeded inputs
shaped like the column exponents (Re in [-746, 0], |Im| up to 1e5) plus the
edges: subnormal results, -0, NaN, -inf, angles past the fallback bound.

Gated:
- exp within 1 ulp of libm wherever libm's result is normal; sin and cos
  within 2 ulp;
- e^{ar} (cos, sin) within 2^-50 of libm's, absolutely, relative to e^{ar}
  where e^{ar} is normal (a subnormal has fewer bits to be relative to);
- a lane's bits depend on its own inputs only: every block length 1..9 and
  every offset gives the same bits as the full array;
- SSE2 and AVX2 lanes bit-identical;
- the fallback (an angle past MW_VMATH_TRIG_MAX, or not finite) is libm's
  answer exactly, and NaN propagates.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HEADER_DIR = REPO / "src" / "momwire"
_CXX = (
    shutil.which(os.environ.get("CXX", ""))
    or shutil.which("c++")
    or shutil.which("g++")
)

pytestmark = pytest.mark.skipif(
    _CXX is None or platform.machine().lower() not in ("x86_64", "amd64"),
    reason="needs a C++ compiler on x86-64",
)

# The extension's TU flags that matter for these bits (setup.py's core flags).
FLAGS = ["-O3", "-std=c++17", "-fno-math-errno", "-ffp-contract=off"]

HARNESS = r"""
#include "_vmath_inline.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#if defined(__AVX2__)
typedef mw_vmath::avx2 Bk;
#else
typedef mw_vmath::sse2 Bk;
#endif
static double ulps(double a, double b) {
    if (a == b) return 0;
    if (std::isnan(a) || std::isnan(b)) return 1e300;
    const double u = std::nextafter(std::fabs(b), INFINITY) - std::fabs(b);
    return std::fabs(a - b) / u;
}
static uint64_t bits(double x) { uint64_t u; std::memcpy(&u, &x, 8); return u; }
int main(int argc, char **argv) {
    const int N = 1 << 18;
    std::mt19937_64 g(1441);
    std::uniform_real_distribution<double> u(0, 1);
    std::vector<double> ar, ai;
    for (int i = 0; i < N; ++i) {
        const double d = u(g), e = u(g), sg = u(g) < 0.5 ? -1.0 : 1.0;
        ar.push_back(-(d < 0.3 ? u(g) : d < 0.6 ? 1 + 9 * u(g) : d < 0.9 ? 10 + 90 * u(g) : 100 + 646 * u(g)));
        ai.push_back(sg * (e < 0.5 ? u(g) : e < 0.8 ? 1 + 9 * u(g) : e < 0.95 ? 10 + 990 * u(g) : 1e3 + 9.9e4 * u(g)));
    }
    // edges
    const double ea[] = {0.0, -0.0, -708.0, -708.5, -744.0, -745.1, -746.0, -800.0, -INFINITY, NAN, -1e-300, -5e-324};
    const double eb[] = {0.0, -0.0, 1e-300, M_PI / 4, M_PI / 2, M_PI, 1e5, -1e5, 1.00001e5, INFINITY, NAN, 1e300};
    for (double a : ea) for (double b : eb) { ar.push_back(a); ai.push_back(b); }
    const int M = (int)ar.size();
    std::vector<double> er(M), ei(M);
    mw_vmath::exp_cis_lanes<Bk>(ar.data(), ai.data(), er.data(), ei.data(), M);
    double mexp = 0, msin = 0, mcos = 0, mabs = 0;
    long bad_fallback = 0, bad_nan = 0, bad_block = 0;
    // exp and sincos alone, through the step with a zero angle / a zero exponent
    std::vector<double> z(M, 0.0), e1(M), e2(M), s1(M), c1(M);
    mw_vmath::exp_cis_lanes<Bk>(ar.data(), z.data(), e1.data(), e2.data(), M);
    mw_vmath::exp_cis_lanes<Bk>(z.data(), ai.data(), c1.data(), s1.data(), M);
    for (int i = 0; i < M; ++i) {
        const double ex = std::exp(ar[i]);
        if (std::isfinite(ar[i]) && ex >= 2.2250738585072014e-308) mexp = std::max(mexp, ulps(e1[i], ex));
        const bool fb = !(std::fabs(ai[i]) <= mw_vmath::MW_VMATH_TRIG_MAX);
        if (!fb) {
            msin = std::max(msin, ulps(s1[i], std::sin(ai[i])));
            mcos = std::max(mcos, ulps(c1[i], std::cos(ai[i])));
        }
        const double rr = ex * std::cos(ai[i]), ri = ex * std::sin(ai[i]);
        if (fb) {
            if (!(bits(er[i]) == bits(rr) && bits(ei[i]) == bits(ri)) &&
                !(std::isnan(rr) && std::isnan(er[i]) && std::isnan(ri) && std::isnan(ei[i])))
                bad_fallback++;
        } else if (std::isnan(ar[i])) {
            if (!(std::isnan(er[i]) && std::isnan(ei[i]))) bad_nan++;
        } else if (ex >= 2.2250738585072014e-308) {  // relative to a normal e^{ar}
            mabs = std::max(mabs, std::max(std::fabs(er[i] - rr), std::fabs(ei[i] - ri)) / ex);
        }
    }
    // position / length independence
    for (int len = 1; len <= 9; ++len)
        for (int off = 0; off + len <= 64; off += 7) {
            double xr[16], xi[16];
            mw_vmath::exp_cis_lanes<Bk>(ar.data() + off, ai.data() + off, xr, xi, len);
            for (int l = 0; l < len; ++l)
                if (bits(xr[l]) != bits(er[off + l]) || bits(xi[l]) != bits(ei[off + l])) bad_block++;
        }
    std::printf("W %d\nexp_ulp %.3f\nsin_ulp %.3f\ncos_ulp %.3f\ncis_abs %.3e\n", Bk::W, mexp, msin, mcos, mabs);
    std::printf("bad_fallback %ld\nbad_nan %ld\nbad_block %ld\n", bad_fallback, bad_nan, bad_block);
    if (argc > 1) {
        FILE *f = std::fopen(argv[1], "wb");
        std::fwrite(er.data(), 8, M, f);
        std::fwrite(ei.data(), 8, M, f);
        std::fclose(f);
    }
}
"""


def _cpu_has_avx2() -> bool:
    try:
        flags = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    return " avx2 " in flags and " fma " in flags


def _run(tmp_path: Path, extra: list[str], tag: str) -> tuple[dict, bytes]:
    src = tmp_path / "harness.cpp"
    src.write_text(HARNESS)
    exe = tmp_path / f"harness-{tag}"
    subprocess.run(
        [_CXX, *FLAGS, *extra, f"-I{HEADER_DIR}", "-o", str(exe), str(src)], check=True
    )
    dump = tmp_path / f"{tag}.bin"
    out = subprocess.run(
        [str(exe), str(dump)], check=True, capture_output=True, text=True
    ).stdout
    vals = {k: float(v) for k, v in (ln.split() for ln in out.splitlines())}
    return vals, dump.read_bytes()


@pytest.fixture(scope="module")
def sse2(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("vmath"), [], "sse2")


def _check(vals: dict) -> None:
    assert vals["exp_ulp"] <= 1.0, vals
    assert vals["sin_ulp"] <= 2.0, vals
    assert vals["cos_ulp"] <= 2.0, vals
    assert vals["cis_abs"] <= 2.0**-50, vals
    assert vals["bad_fallback"] == 0, vals
    assert vals["bad_nan"] == 0, vals
    assert vals["bad_block"] == 0, vals


def test_sse2_lanes(sse2):
    vals, _ = sse2
    assert vals["W"] == 2
    _check(vals)


@pytest.mark.skipif(not _cpu_has_avx2(), reason="this CPU does not run AVX2 + FMA")
def test_avx2_lanes_match_sse2(sse2, tmp_path):
    vals, blob = _run(tmp_path, ["-mavx2", "-mfma"], "avx2")
    assert vals["W"] == 4
    _check(vals)
    assert blob == sse2[1], "the SSE2 and AVX2 lanes must be bit-identical"
