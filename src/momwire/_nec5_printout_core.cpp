// The NEC-5 seam's run half and printout, for any basis -- see
// `_nec5_printout_core.h`. Each function names the Python it is a copy of
// (`eznec/_serve.py`, `eznec/_printout.py`, `eznec/_shell.py`) or the numpy /
// CPython routine whose arithmetic it replays; the text it writes is that
// Python's, byte for byte, and so is the order in which it computes (an
// exception escapes where Python's does). What the solver answers is its
// `PrintoutEngine`'s (razor-2p: `_nec5_razor_printout_core.cpp`).
#include "_nec5_printout_core.h"
#include "_fma_inline.h"
// `max_image_distance` (`_sommerfeld.max_image_distance`, which every basis'
// near field sizes its grid with) lives with razor's front half.
#include "_razor_front_core.h"
#include "_somm_field_core.h"
#include "_somm_grid_core.h"
#include "_mw_times.h"
#include "_somm_below_fill_core.h"
#include "_near_buried_core.h"
#include "_accel_somm_proj_inline.h"

// MSVC builds every TU /fp:fast (setup.py); this one replays numpy's and
// CPython's arithmetic one IEEE operation at a time, so it asks for precise
// semantics. GCC / clang build it with -ffp-contract=off.
#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include <algorithm>
#include <cfloat>
#include <cctype>
#include <charconv>
#include <cmath>
#include <complex>
#include <new>
#include <tuple>
#include <functional>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace mw {
namespace nec5 {

namespace {


// An exception Python lets escape: `_shell.main`'s last line of defence
// reports it as "INTERNAL ERROR IN MOMWIRE ENGINE - <kind>: <message>".
using PyRaise = PrintoutRaise;

// A run this unit does not print yet.
using Decline = PrintoutDecline;

// ============================================================================
// numbers as Python prints them
// ============================================================================

std::string rjust(const std::string &s, std::size_t width) {
    return s.size() >= width ? s : std::string(width - s.size(), ' ') + s;
}

std::string ljust(const std::string &s, std::size_t width) {
    return s.size() >= width ? s : s + std::string(width - s.size(), ' ');
}

// `str.center(width)` (CPython's `pad`: the odd blank goes left only when
// both the padding and the width are odd).
std::string center(const std::string &s, std::size_t width) {
    if (s.size() >= width) return s;
    const std::size_t marg = width - s.size();
    const std::size_t left = marg / 2 + (marg & width & 1);
    return std::string(left, ' ') + s + std::string(marg - left, ' ');
}

std::string rstrip_blanks(std::string s) {
    while (!s.empty() && s.back() == ' ') s.pop_back();
    return s;
}

std::string fmt_double(const char *conv, double x, int width, int digits) {
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
    // `std::to_chars` at a precision is the correctly rounded decimal of the
    // binary value, as glibc's / the CRT's printf is, and spells the
    // exponent as printf does (sign and at least two digits); `%E` only
    // upper-cases it. Several times faster than snprintf, which a long
    // current table pays per cell (momwire#1517).
    char buf[400];
    const bool sci = conv[0] == 'E';
    const auto r = std::to_chars(buf, buf + sizeof buf, x, sci ? std::chars_format::scientific : std::chars_format::fixed,
                                 digits);
    if (r.ec == std::errc()) {
        if (sci)
            for (char *c = buf; c != r.ptr; c++) *c = static_cast<char>(std::toupper(static_cast<unsigned char>(*c)));
        const std::size_t n = static_cast<std::size_t>(r.ptr - buf);
        if (n >= static_cast<std::size_t>(width)) return std::string(buf, n);
        std::string out(static_cast<std::size_t>(width) - n, ' ');
        out.append(buf, n);
        return out;
    }
#endif
    char fmt[32];
    std::snprintf(fmt, sizeof fmt, "%%%d.%d%s", width, digits, conv);
    return format_c_locale(fmt, x);
}

std::string d(long long v, std::size_t width) { return rjust(std::to_string(v), width); }
std::string d(const Int &v, std::size_t width) { return rjust(v.str(), width); }

}  // namespace

std::string py_format_e(double x, int width, int digits) {
    if (std::isnan(x)) return rjust("NAN", static_cast<std::size_t>(width));
    return fmt_double("E", x, width, digits);
}

std::string py_format_f(double x, int width, int digits) {
    if (std::isnan(x)) return rjust("nan", static_cast<std::size_t>(width));
    return fmt_double("f", x, width, digits);
}

namespace {

std::string e(double x, int width, int digits) { return py_format_e(x, width, digits); }
std::string f(double x, int width, int digits) { return py_format_f(x, width, digits); }

// `_e_fortran`: a three-digit exponent drops its `E` to keep the width. A
// non-finite value has no `E` to split on, and Python raises there.
std::string e_fortran(double x, int width, int digits) {
    if (!std::isfinite(x)) throw Decline{"a non-finite lower-medium range cell (the reference raises)"};
    std::string text = py_format_e(x, 0, digits);
    const std::size_t at = text.find('E');
    const int exponent = std::atoi(text.c_str() + at + 1);
    if (std::abs(exponent) >= 100) text = text.substr(0, at) + text.substr(at + 1);
    if (text.size() < static_cast<std::size_t>(width)) text = std::string(width - text.size(), ' ') + text;
    return text;
}

// ============================================================================
// CPython 3.12's arithmetic
// ============================================================================

// `_Py_c_prod`
cd py_cmul(cd a, cd b) {
    return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}

// `_Py_c_quot` (3.12: Smith's method, no Annex G recovery). A zero divisor
// is Python's ZeroDivisionError, whose text is the interpreter's: it is
// DECLINED here rather than spelled in C++ (an EX 4 whose port carries no
// current; an LD capacitance at zero frequency), and Python answers it.
cd py_cdiv(cd a, cd b) {
    const double abs_breal = b.real() < 0 ? -b.real() : b.real();
    const double abs_bimag = b.imag() < 0 ? -b.imag() : b.imag();
    if (abs_breal >= abs_bimag) {
        if (abs_breal == 0.0) throw Decline{"a complex division by zero (the reference raises)"};
        const double ratio = b.imag() / b.real();
        const double denom = b.real() + b.imag() * ratio;
        return cd((a.real() + a.imag() * ratio) / denom, (a.imag() - a.real() * ratio) / denom);
    }
    if (abs_bimag >= abs_breal) {
        const double ratio = b.real() / b.imag();
        const double denom = b.real() * ratio + b.imag();
        return cd((a.real() * ratio + a.imag()) / denom, (a.imag() * ratio - a.real()) / denom);
    }
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return cd(nan, nan);
}

bool py_ctruth(cd z) { return z.real() != 0.0 || z.imag() != 0.0; }

// `sum(floats)` from the int 0: `0 + x0` is float's `0.0 + x0`, then
// Neumaier's compensated sum over the rest (CPython 3.12 `builtin_sum_impl`).
double py_sum(const std::vector<double> &xs) {
    if (xs.empty()) return 0.0;  // `float(0)`
    double result = 0.0 + xs[0];
    double c = 0.0;
    for (std::size_t k = 1; k < xs.size(); k++) {
        const double x = xs[k];
        const double t = result + x;
        if (std::fabs(result) >= std::fabs(x))
            c += (result - t) + x;
        else
            c += (x - t) + result;
        result = t;
    }
    if (c != 0.0 && std::isfinite(c)) result += c;
    return result;
}

// `math.atan2` (`m_atan2`: its own special cases, then libm).
double py_atan2(double y, double x) {
    const double pi = 3.141592653589793;
    if (std::isnan(x) || std::isnan(y)) return std::numeric_limits<double>::quiet_NaN();
    if (std::isinf(y)) {
        if (std::isinf(x)) {
            if (std::copysign(1., x) == 1.) return std::copysign(0.25 * pi, y);
            return std::copysign(0.75 * pi, y);
        }
        return std::copysign(0.5 * pi, y);
    }
    if (std::isinf(x) || y == 0.) {
        if (std::copysign(1., x) == 1.) return std::copysign(0., y);
        return std::copysign(pi, y);
    }
    return std::atan2(y, x);
}

// `math.degrees`: `x * (180 / pi)`.
double py_degrees(double x) { return (180.0 / 3.141592653589793) * x; }

// ============================================================================
// numpy's arithmetic
// ============================================================================

// The elementwise complex multiply as the run path spells it (`_cmul.cmul`,
// numpy's fused formula, on every CPU): each part one fused multiply-add.
cd np_cmul(cd a, cd b) {
    return cd(mw_fma::exact(a.real(), b.real(), -(a.imag() * b.imag())),
              mw_fma::exact(a.real(), b.imag(), a.imag() * b.real()));
}

// numpy's complex add / subtract and the fused complex multiply of two
// arrays; a REAL operand promoted to complex (x, 0).
inline cd c_add(cd a, cd b) { return cd(a.real() + b.real(), a.imag() + b.imag()); }
inline cd c_sub(cd a, cd b) { return cd(a.real() - b.real(), a.imag() - b.imag()); }
inline cd c_mul_real(cd a, double x) { return np_cmul(a, cd(x, 0.0)); }

// Scalar math (`cdouble_ctype_multiply`): never fused.
cd np_scalar_cmul(cd a, cd b) {
    return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}

// `CDOUBLE_divide` (the array loop, and the scalar divide that calls it).
cd np_cdiv(cd a, cd b) {
    const double in1r = a.real(), in1i = a.imag(), in2r = b.real(), in2i = b.imag();
    const double in2r_abs = std::fabs(in2r), in2i_abs = std::fabs(in2i);
    if (in2r_abs >= in2i_abs) {
        if (in2r_abs == 0 && in2i_abs == 0) return cd(in1r / in2r_abs, in1i / in2i_abs);
        const double rat = in2i / in2r;
        const double scl = 1.0 / (in2r + in2i * rat);
        return cd((in1r + in1i * rat) * scl, (in1i - in1r * rat) * scl);
    }
    const double rat = in2r / in2i;
    const double scl = 1.0 / (in2i + in2r * rat);
    return cd((in1r * rat + in1i) * scl, (in1i * rat - in1r) * scl);
}

// `DOUBLE_pairwise_sum` over `n` contiguous float64 values (the front core's
// `pairwise_sum`, measured against numpy 2.5 there).
double np_pairwise(const double *a, int64_t n) {
    if (n < 8) {
        double res = 0.0;
        for (int64_t i = 0; i < n; i++) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; j++) r[j] = a[j];
        int64_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; j++) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i];
        return res;
    }
    int64_t n2 = n / 2;
    n2 -= n2 % 8;
    return np_pairwise(a, n2) + np_pairwise(a + n2, n - n2);
}

// `CDOUBLE_pairwise_sum` over `n` complex values (2n reals), stride 1.
void np_cpairwise(const cd *a, int64_t n, double &rr, double &ri) {
    const int64_t nr = 2 * n;  // numpy counts reals
    if (nr < 8) {
        rr = -0.0;
        ri = -0.0;
        for (int64_t i = 0; i < n; i++) {
            rr += a[i].real();
            ri += a[i].imag();
        }
        return;
    }
    if (nr <= 128) {
        double r[8];
        r[0] = a[0].real();
        r[1] = a[0].imag();
        r[2] = a[1].real();
        r[3] = a[1].imag();
        r[4] = a[2].real();
        r[5] = a[2].imag();
        r[6] = a[3].real();
        r[7] = a[3].imag();
        int64_t i = 8;
        for (; i < nr - (nr % 8); i += 8) {
            const cd *q = a + i / 2;
            r[0] += q[0].real();
            r[1] += q[0].imag();
            r[2] += q[1].real();
            r[3] += q[1].imag();
            r[4] += q[2].real();
            r[5] += q[2].imag();
            r[6] += q[3].real();
            r[7] += q[3].imag();
        }
        rr = ((r[0] + r[2]) + (r[4] + r[6]));
        ri = ((r[1] + r[3]) + (r[5] + r[7]));
        for (; i < nr; i += 2) {
            rr += a[i / 2].real();
            ri += a[i / 2].imag();
        }
        return;
    }
    int64_t n2 = nr / 2;
    n2 -= n2 % 8;
    double rr1, ri1, rr2, ri2;
    np_cpairwise(a, n2 / 2, rr1, ri1);
    np_cpairwise(a + n2 / 2, n - n2 / 2, rr2, ri2);
    rr = rr1 + rr2;
    ri = ri1 + ri2;
}

// `a.sum()` of a 1-D complex array: the identity, plus the pairwise sum.
cd np_csum(const std::vector<cd> &a) {
    double rr = 0.0, ri = 0.0;
    if (!a.empty()) np_cpairwise(a.data(), static_cast<int64_t>(a.size()), rr, ri);
    return cd(0.0 + rr, 0.0 + ri);
}

// --- matmul (`numpy/_core/src/umath/matmul.c.src`) ----------------------------

const cd ONE(1.0, 0.0), ZERO(0.0, 0.0);
constexpr int ROW_MAJOR = 101, COL_MAJOR = 102, NO_TRANS = 111, TRANS = 112;

// `is_blasable2d` in element units.
bool blasable(int64_t s1, int64_t s2, int64_t d1, int64_t d2) {
    (void)d1;
    return s2 == 1 && s1 >= d2;
}

void noblas(const Op &a, const Op &b, cd *out, int64_t os_m, int64_t os_p) {
    for (int64_t m = 0; m < a.d0; m++)
        for (int64_t p = 0; p < b.d1; p++) {
            double r = 0.0, i = 0.0;
            for (int64_t n = 0; n < a.d1; n++) {
                const cd v1 = a.p[m * a.s0 + n * a.s1], v2 = b.p[n * b.s0 + p * b.s1];
                r = r + ((v1.real() * v2.real()) - (v1.imag() * v2.imag()));
                i = i + ((v1.real() * v2.imag()) + (v1.imag() * v2.real()));
            }
            out[m * os_m + p * os_p] = cd(r, i);
        }
}

// `CDOUBLE_dot`
void np_dot(const NumpyLinalg &np, const cd *ip1, int64_t is1, const cd *ip2, int64_t is2,
            cd *op, int64_t n) {
    if (is1 > 0 && is2 > 0) {
        double sum[2] = {0., 0.};
        cd tmp;
        np.zdotu_sub(n, ip1, is1, ip2, is2, &tmp);
        sum[0] += tmp.real();
        sum[1] += tmp.imag();
        *op = cd(sum[0], sum[1]);
        return;
    }
    double sumr = 0.0, sumi = 0.0;
    for (int64_t i = 0; i < n; i++) {
        const cd x = ip1[i * is1], y = ip2[i * is2];
        sumr += x.real() * y.real() - x.imag() * y.imag();
        sumi += x.real() * y.imag() + x.imag() * y.real();
    }
    *op = cd(sumr, sumi);
}

// `CDOUBLE_gemv`
void np_gemv(const NumpyLinalg &np, const cd *ip1, int64_t is1_m, int64_t is1_n, const cd *ip2,
             int64_t is2_n, cd *op, int64_t op_m, int64_t m, int64_t n) {
    int order;
    int64_t lda;
    if (blasable(is1_m, is1_n, m, n)) {
        order = COL_MAJOR;
        lda = is1_m;
    } else {
        order = ROW_MAJOR;
        lda = is1_n;
    }
    np.zgemv(order, TRANS, n, m, &ONE, ip1, lda, ip2, is2_n, &ZERO, op, op_m);
}

// `CDOUBLE_matmul_matrixmatrix`
void np_matrixmatrix(const NumpyLinalg &np, const cd *ip1, int64_t is1_m, int64_t is1_n,
                     const cd *ip2, int64_t is2_n, int64_t is2_p, cd *op, int64_t os_m,
                     int64_t os_p, int64_t m, int64_t n, int64_t p) {
    (void)os_p;
    const int64_t ldc = os_m;
    int trans1, trans2;
    int64_t lda, ldb;
    if (blasable(is1_m, is1_n, m, n)) {
        trans1 = NO_TRANS;
        lda = is1_m;
    } else {
        trans1 = TRANS;
        lda = is1_n;
    }
    if (blasable(is2_n, is2_p, n, p)) {
        trans2 = NO_TRANS;
        ldb = is2_n;
    } else {
        trans2 = TRANS;
        ldb = is2_p;
    }
    if (ip1 == ip2 && m == p && is1_m == is2_p && is1_n == is2_n && trans1 != trans2)
        throw std::logic_error("matmul: the syrk case is not replayed");
    np.zgemm(ROW_MAJOR, trans1, trans2, m, p, n, &ONE, ip1, lda, ip2, ldb, &ZERO, op, ldc);
}

// The C-contiguous complex128 copy `_lapack.matmul` hands the host
// (`np.ascontiguousarray(x, dtype=complex128)`).
std::vector<cd> c_copy(const Op &a) {
    std::vector<cd> out(static_cast<std::size_t>(a.d0 * a.d1));
    for (int64_t i = 0; i < a.d0; i++)
        for (int64_t j = 0; j < a.d1; j++)
            out[static_cast<std::size_t>(i * a.d1 + j)] = a.p[i * a.s0 + j * a.s1];
    return out;
}

// `_lapack.matmul` with an ABI 2 table: ONE fixed call per shape, on
// row-major copies read column-major (a 2-D @ 2-D is `zgemm('N', 'N', n, m,
// k, 1, b, n, a, k, 0, c, n)`; a 2-D @ 1-D is `zgemv('T', k, m, 1, a, k, x,
// 1, 0, y, 1)`). A 1-D `b` is `vec_op`'s (n, 1) with a zero second stride.
void host_matmul(const HostLinalg &h, const Op &a, const Op &b, cd *out, int64_t os_m,
                 int64_t os_p) {
    const int64_t m = a.d0, k = a.d1;
    const bool b_vector = b.d1 == 1 && b.s1 == 0;
    if (std::max({m, k, b.d1}) > std::numeric_limits<int>::max())
        throw Decline{"a product past the LP64 host's int range"};
    const std::vector<cd> a2 = c_copy(a), b2 = c_copy(b);
    const int mm_ = static_cast<int>(m), kk = static_cast<int>(k), one = 1;
    if (b_vector) {
        std::vector<cd> y(static_cast<std::size_t>(m), cd(0.0, 0.0));
        h.zgemv("T", &kk, &mm_, &ONE, a2.data(), &kk, b2.data(), &one, &ZERO, y.data(), &one);
        for (int64_t i = 0; i < m; i++) out[i * os_m] = y[static_cast<std::size_t>(i)];
        return;
    }
    const int nn = static_cast<int>(b.d1);
    std::vector<cd> c(static_cast<std::size_t>(m * b.d1), cd(0.0, 0.0));
    h.zgemm("N", "N", &nn, &mm_, &kk, &ONE, b2.data(), &nn, a2.data(), &kk, &ZERO, c.data(), &nn);
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < b.d1; j++)
            out[i * os_m + j * os_p] = c[static_cast<std::size_t>(i * b.d1 + j)];
}

// `CDOUBLE_matmul` for one (m, n) @ (n, p) product into `out` (strides in
// elements; a 1-D result has `os_p` 0). With a host table, `_lapack.matmul`.
void np_matmul(const NumpyLinalg &np, const Op &a, const Op &b, cd *out, int64_t os_m,
               int64_t os_p) {
    if (np.host.zgemm != nullptr && a.d0 > 0 && a.d1 > 0 && b.d1 > 0) {
        host_matmul(np.host, a, b, out, os_m, os_p);
        return;
    }
    const int64_t dm = a.d0, dn = a.d1, dp = b.d1;
    const int64_t is1_m = a.s0, is1_n = a.s1, is2_n = b.s0, is2_p = b.s1;
    const bool special = dm == 1 || dn == 1 || dp == 1;
    const bool any_zero = dm == 0 || dn == 0 || dp == 0;
    const bool scalar_out = dm == 1 && dp == 1;
    const bool scalar_vec = dn == 1 && (dp == 1 || dm == 1);
    const bool i1b = blasable(is1_m, is1_n, dm, dn) || blasable(is1_n, is1_m, dn, dm);
    const bool i2b = blasable(is2_n, is2_p, dn, dp) || blasable(is2_p, is2_n, dp, dn);
    const bool ob = blasable(os_m, os_p, dm, dp) || blasable(os_p, os_m, dp, dm);
    const bool vector_matrix = dm == 1 && i2b && blasable(is1_n, 1, dn, 1);
    const bool matrix_vector = dp == 1 && i1b && blasable(is2_n, 1, dn, 1);
    if (any_zero) {
        noblas(a, b, out, os_m, os_p);
    } else if (special) {
        if (scalar_out)
            np_dot(np, a.p, is1_n, b.p, is2_n, out, dn);
        else if (scalar_vec)
            noblas(a, b, out, os_m, os_p);
        else if (vector_matrix)
            np_gemv(np, b.p, is2_p, is2_n, a.p, is1_n, out, os_p, dp, dn);
        else if (matrix_vector)
            np_gemv(np, a.p, is1_m, is1_n, b.p, is2_n, out, os_m, dm, dn);
        else
            noblas(a, b, out, os_m, os_p);
    } else {
        if (!i1b || !i2b || !ob) throw std::logic_error("matmul: a copied operand is not replayed");
        const bool o_transpose = std::llabs(os_m) < std::llabs(os_p);
        if (o_transpose)
            np_matrixmatrix(np, b.p, is2_p, is2_n, a.p, is1_n, is1_m, out, os_p, os_m, dp, dn, dm);
        else
            np_matrixmatrix(np, a.p, is1_m, is1_n, b.p, is2_n, is2_p, out, os_m, os_p, dm, dn, dp);
    }
}

Op vec_op(const std::vector<cd> &x) { return Op{x.data(), static_cast<int64_t>(x.size()), 1, 1, 0}; }

// C-matrix @ C-matrix, and matrix @ vector, as numpy's `@`.
CMat mm(const NumpyLinalg &np, const Op &a, const Op &b) {
    CMat out(a.d0, b.d1);
    np_matmul(np, a, b, out.v.data(), out.cols, 1);
    return out;
}

std::vector<cd> mv(const NumpyLinalg &np, const Op &a, const std::vector<cd> &x) {
    std::vector<cd> out(static_cast<std::size_t>(a.d0));
    np_matmul(np, a, vec_op(x), out.data(), 1, 0);
    return out;
}

// `numpy.linalg.solve(a, b)` for a square C-ordered `a` and a vector `b`:
// numpy copies both into Fortran order and calls zgesv. A singular block is
// numpy's LinAlgError, in numpy's words: declined, and Python answers it.
// With a host table, `_lapack.solve_stacked`: getrf on the Fortran copy,
// then getrs ('N', one right-hand side), LP64 on the host's library.
std::vector<cd> np_solve(const NumpyLinalg &np, const CMat &a, const std::vector<cd> &b) {
    const np_int n = a.rows, nrhs = 1;
    std::vector<cd> af(static_cast<std::size_t>(n * n));
    for (np_int j = 0; j < n; j++)
        for (np_int i = 0; i < n; i++) af[static_cast<std::size_t>(i + j * n)] = a.at(i, j);
    std::vector<cd> x(b);
    if (np.host.zgemm != nullptr && n > 0) {
        if (n > std::numeric_limits<int>::max())
            throw Decline{"a solve past the LP64 host's int range"};
        int ni = static_cast<int>(n), one = 1, info = 0;
        std::vector<int> ipiv(static_cast<std::size_t>(n));
        np.host.getrf(&ni, &ni, af.data(), &ni, ipiv.data(), &info);
        if (info > 0) throw Decline{"a singular drive block (the host route raises)"};
        if (info < 0) throw Decline{"the host's getrf refused an argument"};
        char trans = 'N';
        np.host.getrs(&trans, &ni, &one, af.data(), &ni, ipiv.data(), x.data(), &ni, &info);
        if (info != 0) throw Decline{"the host's getrs refused an argument"};
        return x;
    }
    std::vector<np_int> ipiv(static_cast<std::size_t>(n > 0 ? n : 1));
    const np_int lda = std::max<np_int>(n, 1), ldb = lda;
    np_int info = 0;
    np.zgesv(&n, &nrhs, af.data(), &lda, ipiv.data(), x.data(), &ldb, &info);
    if (info > 0) throw Decline{"a singular drive block (the reference raises)"};
    return x;
}

// `numpy.linalg.solve(a, b)` / `_lapack.solve_stacked(a, b)` for a square
// C-ordered `a` and a C-ordered (n, k) `b`: both copied to Fortran order, one
// zgesv with k right-hand sides (numpy), or getrf + getrs (the host). The
// solution comes back C-ordered, as both return it.
CMat np_solve_matrix(const NumpyLinalg &np, const CMat &a, const CMat &b) {
    const int64_t n = a.rows, k = b.cols;
    std::vector<cd> af(static_cast<std::size_t>(n * n)), bf(static_cast<std::size_t>(n * k));
    for (int64_t j = 0; j < n; j++)
        for (int64_t i = 0; i < n; i++) af[static_cast<std::size_t>(i + j * n)] = a.at(i, j);
    for (int64_t j = 0; j < k; j++)
        for (int64_t i = 0; i < n; i++) bf[static_cast<std::size_t>(i + j * n)] = b.at(i, j);
    if (np.host.zgemm != nullptr && n > 0 && k > 0) {
        if (n > std::numeric_limits<int>::max() || k > std::numeric_limits<int>::max())
            throw Decline{"a solve past the LP64 host's int range"};
        int ni = static_cast<int>(n), ki = static_cast<int>(k), info = 0;
        std::vector<int> ipiv(static_cast<std::size_t>(n));
        np.host.getrf(&ni, &ni, af.data(), &ni, ipiv.data(), &info);
        if (info > 0) throw Decline{"a singular loaded admittance (the host route raises)"};
        if (info < 0) throw Decline{"the host's getrf refused an argument"};
        char trans = 'N';
        np.host.getrs(&trans, &ni, &ki, af.data(), &ni, ipiv.data(), bf.data(), &ni, &info);
        if (info != 0) throw Decline{"the host's getrs refused an argument"};
    } else if (n > 0 && k > 0) {
        const np_int nn = n, nrhs = k, lda = n, ldb = n;
        std::vector<np_int> ipiv(static_cast<std::size_t>(n));
        np_int info = 0;
        np.zgesv(&nn, &nrhs, af.data(), &lda, ipiv.data(), bf.data(), &ldb, &info);
        if (info > 0) throw Decline{"a singular loaded admittance (the reference raises)"};
    }
    CMat x(n, k);
    for (int64_t j = 0; j < k; j++)
        for (int64_t i = 0; i < n; i++) x.at(i, j) = bf[static_cast<std::size_t>(i + j * n)];
    return x;
}

// ============================================================================
// the run half (`_serve._finish`)
// ============================================================================

struct PortRowC {  // `PortRow`
    Int tag, segment;
    int end_index = 1;
    cd voltage, current, impedance, admittance;
    double power = 0;
};

struct ElementRowC {  // `WireCurrentRow` / `ChargeRow`
    long long element = 0;
    Int tag;
    double center[3] = {0, 0, 0};
    double length = 0;
    cd value;
    double magnitude = 0, phase_deg = 0;
};

struct LoadRowC {  // `LoadRow` (`_loading_row`): every LD card's row
    Int tag, node_from, node_thru;
    // "FIXED IMPEDANCE", "SERIES", "PARALLEL" (LD 4 / 0 / 1), "WIRE" (LD 5),
    // "SERIES (PER METER)" / "PARALLEL (PER METER)" (LD 2 / LD 3)
    std::string kind;
    double conductivity = 0;
    double resistance = 0, reactance = 0, inductance = 0, capacitance = 0;
    bool has_resistance = false, has_reactance = false, has_inductance = false,
         has_capacitance = false;
};

struct NearRowC;
struct NearBlockC;

struct PatternRowC {  // `PatternRow`
    double theta = 0, phi = 0, vert = 0, hor = 0, total = 0, axial = 0, tilt = 0;
    std::string sense;
    double et_mag = 0, et_deg = 0, ep_mag = 0, ep_deg = 0;
};

struct PatternBlockC {  // `PatternBlock`
    std::vector<PatternRowC> rows;
    bool has_average = false;
    double average_power_gain = 0, solid_angle_pi = 0, power_radiated_4pi = 0;
    bool has_range = false;
    double range_m = 0, range_phase_deg = 0;
    bool has_lower = false;
    double lower_range_mag = 0, lower_range_phase_deg = 0;
    // XNDA's D digit 1 (momwire#1503): the rows are DIRECTIVE gains.
    bool directive = false;
};

struct NetworkRowC {  // `NetworkRow` / `LineRow`
    bool line = false, crossed = false;
    Int tag_from, tag_to;
    long long segment_from = 0, segment_to = 0;
    double cells[6] = {0, 0, 0, 0, 0, 0};  // NT: Y11, Y12, Y22; TL: |Z0|, length, shunts
};

struct RunDataC {  // `RunData`, for the decks this unit prints
    long long node_count = 0, wire_element_count = 0, unknown_count = 0;
    double frequency_mhz = 0, wavelength_m = 0;
    enum Env { free_space, perfect_ground, finite_ground } environment = free_space;
    double eps_r = 0, sigma = 0, eps_c[2] = {0, 0};
    std::vector<LoadRowC> loads;
    std::vector<PortRowC> sources;
    std::vector<ElementRowC> currents, charges;
    double input_power = 0, radiated_power = 0, wire_loss = 0, efficiency = 0;
    std::vector<PatternBlockC> patterns;
    std::vector<std::shared_ptr<NearBlockC>> near_fields;
    std::vector<NetworkRowC> networks;
    std::vector<PortRowC> connections;  // STRUCTURE EXCITATION DATA AT NETWORK CONNECTION POINTS
    bool has_network_loss = false;
    double network_loss = 0;
    // MATRIX TIMING's two cells: 0 unless MOMWIRE_TIMES is on (`_mw_times.h`)
    double fill_seconds = 0, factor_seconds = 0;
    // MOMWIRE_TIMES only: the run's phase lines under MATRIX TIMING
    bool phases = false;
    double ground_seconds = 0, solution_seconds = 0, rp_seconds = 0, near_seconds = 0;
    unsigned long long ground_built = 0, ground_disk = 0, ground_blended = 0;
    int fill_threads = 0, factor_threads = 0;
};

// `_ratio`
cd ratio(cd numerator, cd denominator) {
    if (denominator.real() == 0.0 && denominator.imag() == 0.0) {
        if (py_ctruth(numerator)) {
            const double inf = std::numeric_limits<double>::infinity();
            return cd(inf, inf);
        }
        return cd(0.0, 0.0);
    }
    return py_cdiv(numerator, denominator);
}

bool same_node(const Node &a, const Node &b) { return a.tag == b.tag && a.node == b.node; }

// `Structure.first_element(tag)`
long long first_element(const Deck &d, const Int &tag) {
    long long first = 1;
    for (const Wire &w : d.wires) {
        if (w.tag == tag) return first;
        first += w.segment_count.i64();
    }
    throw std::logic_error("no GW card declares tag " + tag.str());  // the seam checked
}

// The cast `t` (`_transform(mesh)`, float64 (n_columns, n_sites)) as the
// complex128 copy numpy's matmul is handed: C order, (x, 0).
std::vector<cd> transform_c(const Prepared &p) {
    const int64_t nc = p.mesh.n_columns(), ns = static_cast<int64_t>(p.mesh.sites.size());
    std::vector<cd> t(static_cast<std::size_t>(nc * ns), cd(0.0, 0.0));
    for (const SeamSite &s : p.mesh.sites) {
        if (s.column < 0) continue;
        t[static_cast<std::size_t>(s.column * ns + s.index)] = cd(s.weight, 0.0);
    }
    return t;
}

struct PortState {  // `_PortState`
    std::vector<cd> v_applied, v_gap, i_port, i_source, z_load;
};

// `LoadSpec.impedance(freq_hz)` for LD 0 / 1 / 4, with
// `_wire_loading.rlc_impedance`'s arithmetic: its numpy 0-d values (a sum is
// numpy's add, `1.0 / y` numpy's complex divide) and its CPython complex
// scalars (`1j * omega * l` is `_Py_c_prod` twice, `1.0 / (...)` is
// `_Py_c_quot`). A zero element drops out of its branch; all three zero is
// 0 (series) or inf (parallel).
cd load_impedance(const LoadSpec &spec, double freq_hz) {
    const double omega = 2.0 * 3.141592653589793 * freq_hz;
    if (spec.kind == "fixed") return cd(spec.r, spec.x);
    const auto truthy = [](double v) { return v != 0.0 || std::isnan(v); };
    const auto j_omega = [&](double v) { return py_cmul(py_cmul(cd(0.0, 1.0), cd(omega, 0.0)), cd(v, 0.0)); };
    if (spec.kind != "series" && spec.kind != "parallel")
        throw Decline{"an unknown load kind"};
    if (spec.r == 0.0 && spec.l == 0.0 && spec.c == 0.0)
        return spec.kind == "parallel" ? cd(std::numeric_limits<double>::infinity(), 0.0)
                                       : cd(0.0, 0.0);
    if (spec.kind == "series") {
        // `np.asarray(r, complex128) + 0j * np.asarray(omega, float)`
        const cd zero_omega = np_cmul(cd(0.0, 0.0), cd(omega, 0.0));
        cd z(spec.r + zero_omega.real(), 0.0 + zero_omega.imag());
        if (truthy(spec.l)) {
            const cd t = j_omega(spec.l);
            z = cd(z.real() + t.real(), z.imag() + t.imag());
        }
        if (truthy(spec.c)) {
            const cd t = py_cdiv(cd(1.0, 0.0), j_omega(spec.c));
            z = cd(z.real() + t.real(), z.imag() + t.imag());
        }
        return z;
    }
    cd y(0.0, 0.0);  // `np.zeros(np.shape(omega), dtype=complex128)`
    if (truthy(spec.r)) y = cd(y.real() + 1.0 / spec.r, y.imag() + 0.0);
    if (truthy(spec.l)) {
        const cd t = py_cdiv(cd(1.0, 0.0), j_omega(spec.l));
        y = cd(y.real() + t.real(), y.imag() + t.imag());
    }
    if (truthy(spec.c)) {
        const cd t = j_omega(spec.c);
        y = cd(y.real() + t.real(), y.imag() + t.imag());
    }
    return np_cdiv(cd(1.0, 0.0), y);
}

// `site.load` for every site (`_prepare`'s loop: `by_address[load.at].load
// += load.spec.impedance(frequency_hz)`, Python complex adds from 0j).
std::vector<cd> site_loads(const Deck &deck, const Prepared &p) {
    std::vector<cd> z(p.mesh.sites.size(), cd(0.0, 0.0));
    const double freq_hz = p.frequency_mhz * 1e6;
    for (const Load &load : deck.loads) {
        const cd v = load_impedance(load.spec, freq_hz);
        bool found = false;
        for (const SeamSite &site : p.mesh.sites)
            if (site.at.tag == load.at.tag && site.at.node == load.at.node) {
                cd &acc = z[static_cast<std::size_t>(site.index)];
                acc = cd(acc.real() + v.real(), acc.imag() + v.imag());
                found = true;
                break;
            }
        if (!found) throw std::logic_error("a load with no site");  // the seam built one per load
    }
    return z;
}

// `_port_state`, network-free (the decks this unit prints), with lumped loads
// in series inside their ports (no virtual site reaches here: the seam
// declines phantom wires, so `_series_loads` leaves `z_load` as it is):
// `y = t.T @ y_solver @ t`, then one scaled unit probe, or the constrained
// multi-source drive (`_multi_drive_state`).
// ============================================================================
// the networks (`_serve._reduced_state`: `deck._networks.flat_network`,
// `networks.NetworkReducer.apply_branches`, `MNASystem`, `reduced_ports`)
// ============================================================================

// numpy scalar negation and the scalar product (scalarmath: never fused)
inline cd np_neg(cd a) { return cd(-a.real(), -a.imag()); }

// One Group-2 element (`_Group2Element`) as the NEC-5 seam builds them: a
// chain-matrix row (`terms`) or a termination (`b` = the port, c_v = 1).
struct G2 {
    std::vector<std::tuple<int64_t, cd, cd>> terms;  // (node, k, c_v)
    bool termination = false;
    int64_t b = -1;
    cd c_j, e;
};

// `tl_abcd(z0, length, wavelength)` (vf 1, no loss): (A, B, C, D).
void tl_abcd(double z0, double length, double wavelength, cd abcd[4]) {
    const double pi = 3.141592653589793;
    const double beta = 2.0 * pi / (1.0 * wavelength);
    const double f_mhz = 299792458.0 / wavelength / 1e6;
    const double loss = 0.0 * std::sqrt(f_mhz) + 0.0 * f_mhz;  // k1*sqrt(f) + k2*f
    const double neper_per_db = 1.0 / (20.0 / std::log(10.0));
    const double feet_per_m = 1.0 / 0.3048;
    const double alpha = loss * neper_per_db * feet_per_m / 100.0;
    const cd jb = py_cmul(cd(0.0, 1.0), cd(beta, 0.0));               // `1j * beta`
    const cd sum(alpha + jb.real(), 0.0 + jb.imag());                  // np.float64 + complex
    const cd gl = np_scalar_cmul(sum, cd(length, 0.0));                // `* length`
    const cd sh = std::sinh(gl), ch = std::cosh(gl);                   // glibc csinh / ccosh
    abcd[0] = ch;
    abcd[1] = np_scalar_cmul(cd(z0, 0.0), sh);
    abcd[2] = np_cdiv(sh, cd(z0, 0.0));
    abcd[3] = ch;
}

struct NetSolve {
    std::vector<cd> v_applied, i_port, i_source;
};

// `_reduced_state(cards, n, voltages, driven, y_eff, wavelength)`:
// `flat_network(..., held=held)` and `reduced_ports(..., driven)` (the seam
// holds exactly the ports it reads back; the NEC-4.2 slot may not).
NetSolve reduced_state(const std::vector<SeamCard> &cards, int64_t n, const std::vector<cd> &voltages,
                       const std::vector<int64_t> &held, const std::vector<int64_t> &driven, const Op &yeff_op,
                       const std::function<cd(int64_t, int64_t)> &y_eff_at, double wavelength,
                       const NumpyLinalg &np) {
    if (np.zgesvx == nullptr) throw Decline{"no zgesvx for the network solve"};
    // `flat_network`: every port that is not an endpoint (or is held, or
    // carries a voltage) is a Driven source at its voltage
    std::set<int64_t> endpoints;
    for (const SeamCard &c : cards) {
        endpoints.insert(c.site_a);
        endpoints.insert(c.site_b);
    }
    std::vector<int64_t> src_port;
    std::vector<cd> src_volts;
    for (int64_t k = 0; k < n; k++) {
        const cd v = voltages[static_cast<std::size_t>(k)];
        const bool is_held = std::find(held.begin(), held.end(), k) != held.end();
        if (!endpoints.count(k) || is_held || v.real() != 0.0 || v.imag() != 0.0) {
            src_port.push_back(k);
            src_volts.push_back(v);
        }
    }
    // `apply_branches`: G = Y_real, the cards' branches, the terminations
    std::vector<cd> g(static_cast<std::size_t>(n * n));
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++) g[static_cast<std::size_t>(i * n + j)] = y_eff_at(i, j);
    auto G = [&](int64_t i, int64_t j) -> cd & { return g[static_cast<std::size_t>(i * n + j)]; };
    std::vector<G2> elements;
    std::vector<std::tuple<int64_t, int64_t, cd>> couplings;
    for (const SeamCard &c : cards) {
        const int64_t a = c.site_a, b = c.site_b;
        if (c.kind == SeamCard::nt) {
            if (a == b) throw Decline{"an NT card with both ends on one site"};
            const cd y11(c.payload[0], c.payload[1]), y12(c.payload[2], c.payload[3]),
                y22(c.payload[4], c.payload[5]);
            G(a, a) = c_add(G(a, a), y11);
            G(a, b) = c_add(G(a, b), y12);
            G(b, a) = c_add(G(b, a), y12);
            G(b, b) = c_add(G(b, b), y22);
            continue;
        }
        // `card_branches`' TL: z0 = |F1|, crossed when F1 < 0; then the end shunts
        const double z0 = std::fabs(c.payload[0]);
        const bool transposed = c.payload[0] < 0.0;
        cd abcd[4];
        tl_abcd(z0, c.payload[1], wavelength, abcd);
        const cd w_a(1.0, 0.0), w_b = transposed ? cd(-1.0, 0.0) : cd(1.0, 0.0);
        const int64_t i1 = static_cast<int64_t>(elements.size()), i2 = i1 + 1;
        G2 e1;  // v_a with its own current's incidence; v_b sensed
        e1.terms.emplace_back(a, w_a, w_a);
        e1.terms.emplace_back(b, cd(0.0, 0.0), np_scalar_cmul(np_neg(abcd[0]), w_b));
        e1.c_j = cd(0.0, 0.0);
        e1.e = cd(0.0, 0.0);
        G2 e2;
        e2.terms.emplace_back(b, cd(-w_b.real(), -w_b.imag()), np_scalar_cmul(np_neg(abcd[2]), w_b));
        e2.c_j = np_neg(abcd[3]);
        e2.e = cd(0.0, 0.0);
        elements.push_back(e1);
        elements.push_back(e2);
        couplings.emplace_back(i1, i2, np_neg(abcd[1]));
        couplings.emplace_back(i2, i1, cd(1.0, 0.0));
        const cd sa(c.payload[2], c.payload[3]), sb(c.payload[4], c.payload[5]);
        if (py_ctruth(sa) || std::isnan(sa.real()) || std::isnan(sa.imag())) G(a, a) = c_add(G(a, a), sa);
        if (py_ctruth(sb) || std::isnan(sb.real()) || std::isnan(sb.imag())) G(b, b) = c_add(G(b, b), sb);
    }
    // the terminations, port by port (sorted): `v_k = E`
    std::map<int64_t, int64_t> termination_of;
    {
        std::map<int64_t, cd> emf;
        for (std::size_t k = 0; k < src_port.size(); k++) emf[src_port[k]] = src_volts[k];
        for (const auto &kv : emf) {
            G2 t;
            t.termination = true;
            t.b = kv.first;
            t.c_j = cd(0.0, 0.0);  // `sum([], 0j)`
            t.e = kv.second;
            termination_of[kv.first] = static_cast<int64_t>(elements.size());
            elements.push_back(t);
        }
    }
    // `MNASystem`: A = [[G, B], [C^T, D]], rhs = [0; e]
    const int64_t m = static_cast<int64_t>(elements.size()), N = n + m;
    if (N > std::numeric_limits<int>::max()) throw Decline{"a network past the LP64 int range"};
    std::vector<cd> A(static_cast<std::size_t>(N * N), cd(0.0, 0.0)), rhs(static_cast<std::size_t>(N), cd(0.0, 0.0));
    auto AT = [&](int64_t i, int64_t j) -> cd & { return A[static_cast<std::size_t>(i * N + j)]; };
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++) AT(i, j) = G(i, j);
    for (int64_t x = 0; x < m; x++) {
        const G2 &el = elements[static_cast<std::size_t>(x)];
        const int64_t col = n + x;
        if (!el.termination) {
            for (const auto &t : el.terms) {
                const int64_t node = std::get<0>(t);
                AT(node, col) = c_add(AT(node, col), std::get<1>(t));
                AT(col, node) = c_add(AT(col, node), std::get<2>(t));
            }
        } else {
            AT(el.b, col) = c_sub(AT(el.b, col), cd(1.0, 0.0));
            AT(col, el.b) = c_add(AT(col, el.b), cd(1.0, 0.0));
        }
        AT(col, col) = el.c_j;
        rhs[static_cast<std::size_t>(col)] = el.e;
    }
    for (const auto &c : couplings) {
        const int64_t i = n + std::get<0>(c), k = n + std::get<1>(c);
        AT(i, k) = c_add(AT(i, k), std::get<2>(c));
    }
    // `MNASystem.solve`: scipy's `zgesvx(A, rhs)` (fact 'E', trans 'N'), on
    // Fortran copies, x the refined solution
    std::vector<cd> af(static_cast<std::size_t>(N * N)), a_f(static_cast<std::size_t>(N * N)),
        x(static_cast<std::size_t>(N)), b = rhs, work(static_cast<std::size_t>(2 * N));
    for (int64_t j = 0; j < N; j++)
        for (int64_t i = 0; i < N; i++) a_f[static_cast<std::size_t>(i + j * N)] = AT(i, j);
    std::vector<int> ipiv(static_cast<std::size_t>(N));
    std::vector<double> r(static_cast<std::size_t>(N)), cc(static_cast<std::size_t>(N)),
        rwork(static_cast<std::size_t>(2 * N));
    int nn = static_cast<int>(N), nrhs = 1, info = 0;
    double rcond = 0.0, ferr = 0.0, berr = 0.0;
    char fact = 'E', trans = 'N', equed = 'B';
    np.zgesvx(&fact, &trans, &nn, &nrhs, a_f.data(), &nn, af.data(), &nn, ipiv.data(), &equed,
              r.data(), cc.data(), b.data(), &nn, x.data(), &nn, &rcond, &ferr, &berr,
              work.data(), rwork.data(), &info, 1, 1, 1);
    if ((0 < info && info <= nn) || !std::isfinite(rcond) || rcond < 1e-12)
        throw Decline{"a singular network (the reference raises SingularNetworkError)"};
    // `reduced_ports`
    NetSolve out;
    out.v_applied.assign(x.begin(), x.begin() + n);
    for (std::size_t k = 0; k < src_port.size(); k++)
        out.v_applied[static_cast<std::size_t>(src_port[k])] = src_volts[k];
    out.i_port = mv(np, yeff_op, out.v_applied);
    out.i_source = out.i_port;
    for (int64_t port : driven) {
        const auto t = termination_of.find(port);
        if (t == termination_of.end()) throw Decline{"a driven network port with no source branch (the reference raises)"};
        out.i_source[static_cast<std::size_t>(port)] = x[static_cast<std::size_t>(n + t->second)];
    }
    return out;
}

PortState port_state(const Deck &deck, const Prepared &p, const Solved &s, const NumpyLinalg &np) {
    const int64_t nc = p.mesh.n_columns(), n = static_cast<int64_t>(p.mesh.sites.size());
    const std::vector<cd> tc = transform_c(p);
    // `t.T @ y_solver`: t.T is the cast copy in its own (Fortran) order
    const CMat r1 = mm(np, Op{tc.data(), n, nc, 1, n}, s.y.op());
    CMat y = mm(np, r1.op(), Op{tc.data(), nc, n, n, 1});
    const std::vector<cd> z_load = site_loads(deck, p);
    // `_series_loads`: a load on a VIRTUAL site pins `y` there (its admittance
    // to the return) and leaves the series path
    std::vector<cd> z_series = z_load;
    for (const SeamSite &site : p.mesh.sites) {
        const cd z = z_load[static_cast<std::size_t>(site.index)];
        if (site.spelling != SeamSite::virtual_ || !(py_ctruth(z) || std::isnan(z.real()) || std::isnan(z.imag())))
            continue;
        y.at(site.index, site.index) = np_cdiv(cd(1.0, 0.0), z);
        z_series[static_cast<std::size_t>(site.index)] = cd(0.0, 0.0);
    }
    const bool cards = !p.cards.empty();
    bool any_series = false;
    for (const cd &z : z_series)
        if (py_ctruth(z) || std::isnan(z.real()) || std::isnan(z.imag())) any_series = true;
    PortState st;
    st.z_load = z_load;
    if (deck.sources.size() > 1) {
        // `_multi_drive_state` with no cards. With a series load, `y_eff =
        // solve(loaded.T, y.T).T`, loaded = `eye + z_series[:, None] * y`.
        // `y_eff` is the TRANSPOSED view of the C-ordered solution (numpy's
        // `.T`): its products see that layout (numpy's matmul picks its gemv
        // by strides), so it is kept as `xt` and read through `yeff_op`.
        CMat xt;
        Op yeff_op = y.op();
        if (any_series) {
            CMat lt(n, n), yt(n, n);
            for (int64_t i = 0; i < n; i++)
                for (int64_t j = 0; j < n; j++) {
                    const cd prod = np_cmul(z_series[i], y.at(i, j));
                    lt.at(j, i) = cd((i == j ? 1.0 : 0.0) + prod.real(), 0.0 + prod.imag());
                    yt.at(i, j) = y.at(j, i);
                }
            xt = np_solve_matrix(np, lt, yt);
            yeff_op = Op{xt.v.data(), n, n, 1, n};
        }
        const std::function<cd(int64_t, int64_t)> y_eff_at = [&](int64_t i, int64_t j) {
            return any_series ? xt.at(j, i) : y.at(i, j);
        };
        std::vector<int64_t> driven;
        for (const SeamSite &site : p.mesh.sites) {
            bool is_driven = false;
            for (const Source &src : deck.sources)
                if (same_node(src.at, site.at)) is_driven = true;
            if (is_driven) driven.push_back(site.index);
        }
        std::map<int64_t, int64_t> row_of;
        for (std::size_t k = 0; k < driven.size(); k++) row_of[driven[k]] = static_cast<int64_t>(k);
        std::vector<cd> spec(driven.size(), cd(0.0, 0.0));
        std::map<int64_t, bool> current_kind;
        for (const Source &src : deck.sources) {
            int64_t index = -1;
            for (const SeamSite &site : p.mesh.sites)
                if (same_node(site.at, src.at)) index = site.index;
            spec[static_cast<std::size_t>(row_of[index])] = cd(src.drive_re, src.drive_im);
            current_kind[index] = !src.kind.is_zero();
        }
        std::vector<int64_t> i_sites, v_sites, i_rows, v_rows;
        for (int64_t index : driven) (current_kind[index] ? i_sites : v_sites).push_back(index);
        for (int64_t index : i_sites) i_rows.push_back(row_of[index]);
        for (int64_t index : v_sites) v_rows.push_back(row_of[index]);
        std::vector<cd> volts(static_cast<std::size_t>(n), cd(0.0, 0.0));
        for (std::size_t k = 0; k < v_sites.size(); k++)
            volts[static_cast<std::size_t>(v_sites[k])] = spec[static_cast<std::size_t>(v_rows[k])];
        if (!i_sites.empty()) {
            const int64_t nd = static_cast<int64_t>(driven.size());
            CMat m(nd, nd);
            if (!cards) {
                for (int64_t i = 0; i < nd; i++)
                    for (int64_t j = 0; j < nd; j++) m.at(i, j) = y_eff_at(driven[i], driven[j]);
            } else {
                // the map measured a column at a time through the SAME reducer
                for (int64_t column = 0; column < nd; column++) {
                    std::vector<cd> probe(static_cast<std::size_t>(n), cd(0.0, 0.0));
                    probe[static_cast<std::size_t>(driven[column])] = cd(1.0, 0.0);
                    const NetSolve r = reduced_state(p.cards, n, probe, driven, driven, yeff_op, y_eff_at,
                                                     p.wavelength, np);
                    for (int64_t i = 0; i < nd; i++)
                        m.at(i, column) = r.i_source[static_cast<std::size_t>(driven[i])];
                }
            }
            std::vector<cd> rhs;
            for (int64_t r : i_rows) rhs.push_back(spec[static_cast<std::size_t>(r)]);
            if (!v_sites.empty()) {
                CMat miv(static_cast<int64_t>(i_rows.size()), static_cast<int64_t>(v_rows.size()));
                for (std::size_t i = 0; i < i_rows.size(); i++)
                    for (std::size_t j = 0; j < v_rows.size(); j++)
                        miv.at(static_cast<int64_t>(i), static_cast<int64_t>(j)) = m.at(i_rows[i], v_rows[j]);
                std::vector<cd> sv;
                for (int64_t r : v_rows) sv.push_back(spec[static_cast<std::size_t>(r)]);
                const std::vector<cd> prod = mv(np, miv.op(), sv);
                for (std::size_t k = 0; k < rhs.size(); k++) rhs[k] = rhs[k] - prod[k];
            }
            CMat mii(static_cast<int64_t>(i_rows.size()), static_cast<int64_t>(i_rows.size()));
            for (std::size_t i = 0; i < i_rows.size(); i++)
                for (std::size_t j = 0; j < i_rows.size(); j++)
                    mii.at(static_cast<int64_t>(i), static_cast<int64_t>(j)) = m.at(i_rows[i], i_rows[j]);
            const std::vector<cd> vi = np_solve(np, mii, rhs);
            for (std::size_t k = 0; k < i_sites.size(); k++) volts[static_cast<std::size_t>(i_sites[k])] = vi[k];
        }
        if (!cards) {
            st.v_applied = volts;
            st.i_port = mv(np, yeff_op, st.v_applied);
            st.i_source = st.i_port;
        } else {
            const NetSolve r = reduced_state(p.cards, n, volts, driven, driven, yeff_op, y_eff_at, p.wavelength, np);
            st.v_applied = r.v_applied;
            st.i_port = r.i_port;
            st.i_source = r.i_source;
        }
        if (!i_sites.empty()) {
            for (std::size_t k = 0; k < i_sites.size(); k++)
                st.i_source[static_cast<std::size_t>(i_sites[k])] = spec[static_cast<std::size_t>(i_rows[k])];
        } else if (cards) {
            for (std::size_t k = 0; k < v_sites.size(); k++)
                st.v_applied[static_cast<std::size_t>(v_sites[k])] = spec[static_cast<std::size_t>(v_rows[k])];
        }
        st.v_gap.resize(static_cast<std::size_t>(n));
        for (int64_t k = 0; k < n; k++)
            st.v_gap[k] = st.v_applied[k] - np_cmul(z_series[k], st.i_port[k]);
        return st;
    }
    const Source &source = deck.sources[0];
    int64_t driven = -1;
    for (const SeamSite &site : p.mesh.sites)
        if (same_node(site.at, source.at)) driven = site.index;
    std::vector<cd> probe(static_cast<std::size_t>(n), cd(0.0, 0.0));
    probe[static_cast<std::size_t>(driven)] = cd(1.0, 0.0);
    // `np.eye(n) + z_series[:, None] * y`
    CMat loaded(n, n);
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++) {
            const cd prod = np_cmul(z_series[i], y.at(i, j));
            loaded.at(i, j) = cd((i == j ? 1.0 : 0.0) + prod.real(), 0.0 + prod.imag());
        }
    std::vector<cd> v_gap, i_port, v_applied(static_cast<std::size_t>(n)), i_source;
    if (!cards) {
        v_gap = np_solve(np, loaded, probe);
        i_port = mv(np, y.op(), v_gap);
        for (int64_t k = 0; k < n; k++) v_applied[k] = v_gap[k] + np_cmul(z_series[k], i_port[k]);
        i_source = i_port;
    } else {
        // `y_eff = y`, or `solve(loaded.T, y.T).T` (a transposed view)
        CMat xt;
        Op yeff_op = y.op();
        if (any_series) {
            CMat lt(n, n), yt(n, n);
            for (int64_t i = 0; i < n; i++)
                for (int64_t j = 0; j < n; j++) {
                    lt.at(j, i) = loaded.at(i, j);
                    yt.at(i, j) = y.at(j, i);
                }
            xt = np_solve_matrix(np, lt, yt);
            yeff_op = Op{xt.v.data(), n, n, 1, n};
        }
        const std::function<cd(int64_t, int64_t)> y_eff_at = [&](int64_t i, int64_t j) {
            return any_series ? xt.at(j, i) : y.at(i, j);
        };
        const NetSolve r = reduced_state(p.cards, n, probe, {driven}, {driven}, yeff_op, y_eff_at, p.wavelength, np);
        v_applied = r.v_applied;
        i_port = r.i_port;
        i_source = r.i_source;
        v_gap.resize(static_cast<std::size_t>(n));
        for (int64_t k = 0; k < n; k++) v_gap[k] = v_applied[k] - np_cmul(z_series[k], i_port[k]);
    }
    const cd drive(source.drive_re, source.drive_im);
    const cd scale = source.kind.is_zero()
                         ? drive
                         : py_cdiv(drive, i_source[static_cast<std::size_t>(driven)]);
    for (int64_t k = 0; k < n; k++) {
        v_applied[k] = np_cmul(v_applied[k], scale);
        v_gap[k] = np_cmul(v_gap[k], scale);
        i_port[k] = np_cmul(i_port[k], scale);
        i_source[k] = np_cmul(i_source[k], scale);
    }
    st.v_applied = v_applied;
    st.v_gap = v_gap;
    st.i_source = i_source;
    st.i_port = i_port;
    return st;
}

// ============================================================================
// the radiation pattern (`_serve._pattern`, `_far_readout.pattern_fields`)
// ============================================================================

// CPython's `cmath.sqrt` for a finite value (numpy's complex sqrt gives the
// numpy's complex sqrt (`np.sqrt` on a complex array or scalar): libm's
// `csqrt`, which is NOT CPython's `cmath.sqrt` on every argument (a purely
// imaginary one differs on about half, measured), so `_image_coeffs`' np.sqrt
// is this and `cmath.sqrt` (the lower medium's k) is `py_csqrt` below.
cd np_csqrt(cd z) {
#if defined(__GNUC__)
    __complex__ double c;
    __real__ c = z.real();
    __imag__ c = z.imag();
    c = __builtin_csqrt(c);
    return cd(__real__ c, __imag__ c);
#else
    return std::sqrt(z);
#endif
}

// same bits on the machines measured, so `_image_coeffs`' np.sqrt is this).
cd py_csqrt(cd z) {
    if (!std::isfinite(z.real()) || !std::isfinite(z.imag()))
        throw Decline{"a non-finite complex square root"};
    if (z.real() == 0.0 && z.imag() == 0.0) return cd(0.0, z.imag());
    double ax = std::fabs(z.real());
    const double ay = std::fabs(z.imag());
    double sq;
    if (ax < DBL_MIN && ay < DBL_MIN) {
        ax = std::ldexp(ax, 53);
        sq = std::ldexp(std::sqrt(ax + std::hypot(ax, std::ldexp(ay, 53))), -27);
    } else {
        ax /= 8.0;
        sq = 2.0 * std::sqrt(ax + std::hypot(ax, ay / 8.0));
    }
    const double d = ay / (2.0 * sq);
    if (z.real() >= 0.0) return cd(sq, std::copysign(d, z.imag()));
    return cd(d, std::copysign(sq, z.imag()));
}


// `np.linspace(a, b, num)` for 3-vectors (`num >= 2`): numpy 2.5's form,
// including the step-zero branch, which applies to EVERY coordinate as soon
// as one coordinate's step is zero.
void linspace3(const double *a, const double *b, int64_t num, std::vector<double> &out) {
    const double div = static_cast<double>(num - 1);
    double delta[3], step[3];
    bool any_zero = false;
    for (int c = 0; c < 3; c++) {
        delta[c] = b[c] - a[c];
        step[c] = delta[c] / div;
        if (step[c] == 0.0) any_zero = true;
    }
    for (int64_t i = 0; i < num; i++)
        for (int c = 0; c < 3; c++) {
            double y = static_cast<double>(i);
            y = any_zero ? (y / div) * delta[c] : y * step[c];
            out.push_back(i == num - 1 ? b[c] : y + a[c]);
        }
}

// The direction grid's unit vectors (`_far_moments`), C order (theta, phi).
struct Grid {
    int64_t nt = 0, np_ = 0;
    std::vector<double> rx, ry, rz, sin_t, cos_t, sin_p, cos_p;
};

// `moments_of(centres, weights)`: Σ_n exp(j k r̂·r_n) w_n, the phase an einsum
// over c summed (c0 + c2) + c1, the moment an einsum over n summed in order
// from 0 with numpy's plain complex products (both measured: numpy 2.5,
// AVX2 and AVX-512 alike).
std::vector<cd> moments_of(const Grid &g, double k, const std::vector<double> &centres,
                           const std::vector<cd> &weights) {
    const std::size_t n = centres.size() / 3;
    std::vector<cd> out(static_cast<std::size_t>(g.nt * g.np_ * 3), cd(0.0, 0.0));
    for (int64_t i = 0; i < g.nt * g.np_; i++) {
        const double x = g.rx[i], y = g.ry[i], z = g.rz[i];
        cd acc[3] = {cd(0.0, 0.0), cd(0.0, 0.0), cd(0.0, 0.0)};
        for (std::size_t m = 0; m < n; m++) {
            const double *c = &centres[3 * m];
            const double phase = k * ((x * c[0] + z * c[2]) + y * c[1]);
            // `np.exp(1j * phase)`: (±0, phase) -> (cos, sin), exp(±0) = 1
            const double re0 = mw_fma::exact(0.0, phase, -(1.0 * 0.0));
            const double ex = std::exp(re0);
            const cd f(ex * std::cos(phase), ex * std::sin(phase));
            for (int cc = 0; cc < 3; cc++) {
                const cd w = weights[3 * m + cc];
                acc[cc] = cd(acc[cc].real() + (f.real() * w.real() - f.imag() * w.imag()),
                             acc[cc].imag() + (f.real() * w.imag() + f.imag() * w.real()));
            }
        }
        for (int cc = 0; cc < 3; cc++) out[static_cast<std::size_t>(3 * i + cc)] = acc[cc];
    }
    return out;
}

// `np.sum(a * b, axis=-1)` for a complex (.., 3) by a real (.., 3).
inline cd dot3(const cd *a, const double *b) {
    return c_add(c_add(c_mul_real(a[0], b[0]), c_mul_real(a[1], b[1])), c_mul_real(a[2], b[2]));
}

enum class FarGround { free, pec, refl };

// `_far_moments` for elements above the plane (the seam's covered decks):
// free space, the PEC image, or the image weighted by `_image_coeffs`.
// `_eps_complex(eps_r, sigma, freq_hz)`: eps_r - 1j * sigma / (omega * EPS0),
// CPython's complex arithmetic.
cd eps_complex(double eps_r, double sigma, double freq_hz) {
    const double omega = 2.0 * 3.141592653589793 * freq_hz;
    const cd js = py_cmul(cd(0.0, 1.0), cd(sigma, 0.0));
    const cd frac = py_cdiv(js, cd(omega * 8.854187817e-12, 0.0));
    return cd(eps_r - frac.real(), 0.0 - frac.imag());
}

// `_sommerfeld_below.k_medium(eps_t, k2)`: k2 * numpy's scalar csqrt, the
// scalar product unfused, on the Im <= 0 branch; a Python complex.
cd k_medium_of(cd eps_t, double k2) {
    const cd s = np_csqrt(eps_t);
    cd km(k2 * s.real() - 0.0 * s.imag(), k2 * s.imag() + 0.0 * s.real());
    if (km.imag() > 0) km = cd(km.real(), -km.imag());
    return km;
}

// numpy's complex exp (`npy_cexp`: libm's cexp).
cd np_cexp(cd z) {
#if defined(__GNUC__)
    __complex__ double c;
    __real__ c = z.real();
    __imag__ c = z.imag();
    c = __builtin_cexp(c);
    return cd(__real__ c, __imag__ c);
#else
    return std::exp(z);
#endif
}

// `transmitted_moments(mid, moment, k_p, k_m, theta, phi, ground_z)` over
// the BELOW elements, with `transmitted_factors` and `_k_mz` (momwire#570):
// numpy's array products fused (`np_cmul`), CPython's on the scalars, each
// row's sum over the elements numpy's identity plus pairwise sum.
void transmitted_moments(const Grid &g, double k_p, cd k_m, const std::vector<double> &mid,
                         const std::vector<cd> &moment, double ground_z, std::vector<cd> &t_theta,
                         std::vector<cd> &t_phi) {
    const std::size_t n = mid.size() / 3;
    const int64_t nt = g.nt, np_ = g.np_;
    const cd km2 = py_cmul(k_m, k_m);
    const double kp2 = k_p * k_p;
    const cd kp_over_km = py_cdiv(cd(k_p, 0.0), k_m);
    std::vector<cd> k_mz(static_cast<std::size_t>(nt)), t_e(k_mz.size()), t_h(k_mz.size()),
        t_v(k_mz.size());
    for (int64_t it = 0; it < nt; it++) {
        const double st = g.sin_t[it], ct = g.cos_t[it];
        // `_k_mz`: sqrt(k_m^2 - (k_p sin)^2), the decaying root
        const double ks = k_p * st;
        cd q = np_csqrt(c_sub(km2, cd(ks * ks, 0.0)));
        if (q.imag() > 0.0) q = cd(-q.real(), -q.imag());
        k_mz[it] = q;
        // `transmitted_factors`
        const double kpc = k_p * ct;
        const cd k_pz(kpc + 0.0, 0.0 + 0.0);
        const cd two_kpz = np_cmul(cd(2.0, 0.0), k_pz);
        const cd ts = np_cdiv(two_kpz, c_add(k_pz, q));
        const cd num = np_cmul(np_cmul(two_kpz, cd(k_p, 0.0)), k_m);
        const cd den = c_add(np_cmul(km2, k_pz), np_cmul(cd(kp2, 0.0), q));
        const cd tp = np_cdiv(num, den);
        t_e[it] = ts;
        t_h[it] = np_cmul(tp, np_cdiv(q, k_m));
        t_v[it] = np_cmul(cd(-tp.real(), -tp.imag()), np_cmul(kp_over_km, cd(st, 0.0)));
    }
    const cd jkp = py_cmul(cd(0.0, 1.0), cd(k_p, 0.0));  // `1j * k_p`
    t_theta.assign(static_cast<std::size_t>(nt * np_), cd(0.0, 0.0));
    t_phi.assign(t_theta.size(), cd(0.0, 0.0));
    std::vector<cd> a(n), b(n);
    for (int64_t it = 0; it < nt; it++) {
        const cd ref = np_cmul(jkp, cd(g.cos_t[it] * ground_z, 0.0));
        for (int64_t ip = 0; ip < np_; ip++) {
            const int64_t i = it * np_ + ip;
            const double cp = g.cos_p[ip], sp = g.sin_p[ip];
            for (std::size_t m = 0; m < n; m++) {
                const double *c = &mid[3 * m];
                const cd *w = &moment[3 * m];
                const double lateral = g.rx[i] * c[0] + g.ry[i] * c[1];
                const cd depth = np_cmul(k_mz[it], cd(ground_z - c[2], 0.0));
                const cd arg = c_add(c_sub(np_cmul(jkp, cd(lateral, 0.0)), np_cmul(cd(0.0, 1.0), depth)), ref);
                const cd carrier = np_cexp(arg);
                const cd m_par = c_add(np_cmul(cd(cp, 0.0), w[0]), np_cmul(cd(sp, 0.0), w[1]));
                const cd s_x = np_cmul(cd(sp, 0.0), w[0]);
                const cd m_perp = c_add(cd(-s_x.real(), -s_x.imag()), np_cmul(cd(cp, 0.0), w[1]));
                a[m] = np_cmul(carrier, c_add(np_cmul(t_h[it], m_par), np_cmul(t_v[it], w[2])));
                b[m] = np_cmul(np_cmul(carrier, t_e[it]), m_perp);
            }
            t_theta[static_cast<std::size_t>(i)] = np_csum(a);
            t_phi[static_cast<std::size_t>(i)] = np_csum(b);
        }
    }
}

void far_moments(const Grid &g, double k, const std::vector<double> &mid_all,
                 const std::vector<cd> &moment_all, FarGround ground, double eps_r, double sigma,
                 double freq_hz, std::vector<cd> &m_theta, std::vector<cd> &m_phi) {
    // `_below_elements`: the elements more than a hair below the plane radiate
    // through the interface, the rest direct + image (`mid[~below]`).
    std::vector<double> mid, mid_b;
    std::vector<cd> moment, moment_b;
    for (std::size_t m = 0; m < mid_all.size() / 3; m++) {
        const bool below = ground != FarGround::free && mid_all[3 * m + 2] < 0.0 - 1e-9;
        (below ? mid_b : mid).insert((below ? mid_b : mid).end(), &mid_all[3 * m], &mid_all[3 * m] + 3);
        (below ? moment_b : moment)
            .insert((below ? moment_b : moment).end(), &moment_all[3 * m], &moment_all[3 * m] + 3);
    }
    if (!mid_b.empty() && ground == FarGround::pec)
        throw Decline{"far field of a below element over a perfect ground"};
    const std::vector<cd> direct = moments_of(g, k, mid, moment);
    std::vector<cd> total = direct;
    if (ground != FarGround::free) {
        std::vector<double> mid_img = mid;
        std::vector<cd> mom_img(moment.size());
        for (std::size_t m = 0; m < mid.size() / 3; m++) {
            mid_img[3 * m + 2] = 2.0 * 0.0 - mid[3 * m + 2];
            for (int c = 0; c < 3; c++)
                mom_img[3 * m + c] = c_mul_real(moment[3 * m + c], c == 2 ? 1.0 : -1.0);
        }
        const std::vector<cd> img = moments_of(g, k, mid_img, mom_img);
        if (ground == FarGround::pec) {
            for (std::size_t q = 0; q < total.size(); q++) total[q] = c_add(direct[q], img[q]);
        } else {
            const cd eps_c = eps_complex(eps_r, sigma, freq_hz);
            for (int64_t i = 0; i < g.nt * g.np_; i++) {
                const double rx = g.rx[i], ry = g.ry[i], rz = g.rz[i];
                const int64_t it = i / g.np_, ip = i % g.np_;
                const double cp = g.cos_p[ip], sp = g.sin_p[ip], ct = g.cos_t[it], st = g.sin_t[it];
                const double h_hat[3] = {-sp, cp, 0.0};
                const double v_hat[3] = {(-cp) * ct, (-sp) * ct, st};
                const cd *mi = &img[static_cast<std::size_t>(3 * i)];
                const cd m_h = dot3(mi, h_hat), m_v = dot3(mi, v_hat);
                // `_image_coeffs`
                const cd under(eps_c.real() - rx * rx, eps_c.imag() - 0.0);
                const cd qq = np_csqrt(cd(under.real() - ry * ry, under.imag() - 0.0));
                const cd rho_h = np_cdiv(cd(rz - qq.real(), 0.0 - qq.imag()), cd(rz + qq.real(), 0.0 + qq.imag()));
                const cd e_rz = np_cmul(eps_c, cd(rz, 0.0));
                const cd rho_v = np_cdiv(c_sub(e_rz, qq), c_add(e_rz, qq));
                const cd a = np_cmul(rho_v, m_v), b = np_cmul(rho_h, m_h);
                for (int c = 0; c < 3; c++) {
                    const cd refl = c_sub(c_mul_real(a, v_hat[c]), c_mul_real(b, h_hat[c]));
                    total[static_cast<std::size_t>(3 * i + c)] = c_add(direct[static_cast<std::size_t>(3 * i + c)], refl);
                }
            }
        }
    }
    m_theta.resize(static_cast<std::size_t>(g.nt * g.np_));
    m_phi.resize(static_cast<std::size_t>(g.nt * g.np_));
    for (int64_t i = 0; i < g.nt * g.np_; i++) {
        const int64_t it = i / g.np_, ip = i % g.np_;
        const double cp = g.cos_p[ip], sp = g.sin_p[ip], ct = g.cos_t[it], st = g.sin_t[it];
        const double theta_hat[3] = {ct * cp, ct * sp, -st};
        const double phi_hat[3] = {-sp, cp, 0.0};
        m_theta[i] = dot3(&total[static_cast<std::size_t>(3 * i)], theta_hat);
        m_phi[i] = dot3(&total[static_cast<std::size_t>(3 * i)], phi_hat);
    }
    if (!mid_b.empty()) {
        std::vector<cd> t_theta, t_phi;
        transmitted_moments(g, k, k_medium_of(eps_complex(eps_r, sigma, freq_hz), k), mid_b, moment_b,
                            0.0, t_theta, t_phi);
        for (std::size_t i = 0; i < m_theta.size(); i++) {
            m_theta[i] = c_add(m_theta[i], t_theta[i]);
            m_phi[i] = c_add(m_phi[i], t_phi[i]);
        }
    }
}

constexpr double RAD_TO_DEG = 180.0 / 3.141592653589793;
constexpr double DEG_TO_RAD = 3.141592653589793 / 180.0;
constexpr double ETA0 = 376.730313668;

// `_gain_db_grid` for one value: DB10's floor, then 10 log10 (CPython's).
double gain_db(double g) {
    if (g < 1.0e-20) return -999.99;
    if (std::isnan(g)) return g;
    return 10.0 * std::log10(g);
}

// `_serve._pattern` + `_pattern_rows` for one RP card.
PatternBlockC pattern_block(const Request &rq, const std::vector<double> &mid,
                            const std::vector<cd> &moment, FarGround ground, double eps_r,
                            double sigma, double frequency_mhz, double wavelength, double p_in,
                            bool lower, cd eps_c_medium, double p_radiated) {
    if (!rq.n_theta.fits_i64() || !rq.n_phi.fits_i64() || !rq.xnda.fits_i64())
        throw Decline{"a pattern count past 64 bits"};
    const int64_t nt = rq.n_theta.i64(), nph = rq.n_phi.i64();
    if (nt < 1 || nph < 1) throw Decline{"an empty pattern grid"};
    if (!rq.mode.is_zero()) throw Decline{"a pattern mode other than 0"};
    std::vector<double> thetas(static_cast<std::size_t>(nt)), phis(static_cast<std::size_t>(nph));
    for (int64_t i = 0; i < nt; i++) thetas[i] = rq.theta0_deg + rq.d_theta_deg * static_cast<double>(i);
    for (int64_t i = 0; i < nph; i++) phis[i] = rq.phi0_deg + rq.d_phi_deg * static_cast<double>(i);
    const double k = 2.0 * 3.141592653589793 / wavelength;
    Grid g;
    g.nt = nt;
    g.np_ = nph;
    for (double t : thetas) {
        const double r = t * DEG_TO_RAD;
        g.sin_t.push_back(std::sin(r));
        g.cos_t.push_back(std::cos(r));
    }
    for (double ph : phis) {
        const double r = ph * DEG_TO_RAD;
        g.sin_p.push_back(std::sin(r));
        g.cos_p.push_back(std::cos(r));
    }
    for (int64_t it = 0; it < nt; it++)
        for (int64_t ip = 0; ip < nph; ip++) {
            g.rx.push_back(g.sin_t[it] * g.cos_p[ip]);
            g.ry.push_back(g.sin_t[it] * g.sin_p[ip]);
            g.rz.push_back(g.cos_t[it]);
        }
    std::vector<cd> m_theta, m_phi;
    far_moments(g, k, mid, moment, ground, eps_r, sigma, frequency_mhz * 1e6, m_theta, m_phi);
    // `pattern_fields`
    const cd scale = py_cdiv(py_cmul(py_cmul(cd(-0.0, -1.0), cd(ETA0, 0.0)), cd(k, 0.0)),
                             cd(4.0 * 3.141592653589793, 0.0));
    const double norm = p_in > 0 ? ((ETA0 * k) * k) / ((8.0 * 3.141592653589793) * p_in) : 0.0;
    const std::size_t n = m_theta.size();
    std::vector<cd> e_t(n), e_p(n);
    std::vector<double> g_v(n), g_h(n);
    for (std::size_t q = 0; q < n; q++) {
        e_t[q] = np_cmul(scale, m_theta[q]);
        e_p[q] = np_cmul(scale, m_phi[q]);
        const double at = std::hypot(m_theta[q].real(), m_theta[q].imag());
        const double ap = std::hypot(m_phi[q].real(), m_phi[q].imag());
        g_v[q] = norm * (at * at);
        g_h[q] = norm * (ap * ap);
    }
    std::vector<bool> below(static_cast<std::size_t>(nt), false);
    bool any_below = false;
    if (ground != FarGround::free)
        for (int64_t it = 0; it < nt; it++)
            if (thetas[it] > 90.0 + 1e-9) below[it] = any_below = true;
    if (any_below)
        for (int64_t it = 0; it < nt; it++)
            if (below[it])
                for (int64_t ip = 0; ip < nph; ip++) g_v[it * nph + ip] = g_h[it * nph + ip] = 0.0;
    const double floor_scale = 1.0 / wavelength;
    PatternBlockC block;
    // XNDA's D digit (momwire#1503): 1 asks for DIRECTIVE gains in the rows,
    // the pattern normalised to the radiated power (`g_v * scale`, numpy's
    // plain product); the average-gain trailer stays a power gain.
    const bool directive = rq.xnda.i64() / 10 % 10 == 1;
    std::vector<double> g_v_rows = g_v, g_h_rows = g_h;
    if (directive) {
        const double sc = p_radiated > 0 ? p_in / p_radiated : 0.0;
        for (std::size_t q = 0; q < n; q++) {
            g_v_rows[q] = g_v[q] * sc;
            g_h_rows[q] = g_h[q] * sc;
        }
        block.directive = true;
    }
    const double range_m = rq.range_m;
    double range_phase = 0.0;
    if (range_m != 0.0) {
        range_phase = py_degrees(py_atan2(-std::sin(k * range_m), std::cos(k * range_m)));
        block.has_range = true;
        block.range_m = range_m;
        block.range_phase_deg = range_phase;
        if (lower) {
            const cd root = py_csqrt(eps_c_medium);
            const double kk = 2.0 * 3.141592653589793 / wavelength;
            const cd lk = py_cmul(cd(kk, 0.0), root);
            const double kr = lk.real() * range_m;
            block.has_lower = true;
            block.lower_range_mag = std::exp(lk.imag() * range_m) / range_m;
            block.lower_range_phase_deg = py_degrees(py_atan2(-std::sin(kr), std::cos(kr)));
        }
    }
    // `_pattern_rows`, phi-major
    const double dust = 1.0e-14, field_floor = 1.0e-20, raw2 = floor_scale * floor_scale;
    for (int64_t ip = 0; ip < nph; ip++)
        for (int64_t it = 0; it < nt; it++) {
            const std::size_t q = static_cast<std::size_t>(it * nph + ip);
            cd et = e_t[q], ep = e_p[q];
            if (et.real() * et.real() + et.imag() * et.imag() <= dust) et = cd(0.0, 0.0);
            if (ep.real() * ep.real() + ep.imag() * ep.imag() <= dust) ep = cd(0.0, 0.0);
            const bool bl = any_below && below[static_cast<std::size_t>(it)];
            if (bl) et = ep = cd(0.0, 0.0);
            double a = std::hypot(et.real(), et.imag()), b = std::hypot(ep.real(), ep.imag());
            double ad = py_atan2(et.imag(), et.real()) * RAD_TO_DEG;
            double bd = py_atan2(ep.imag(), ep.real()) * RAD_TO_DEG;
            // `_polarisation_grid`
            const bool blank = (a * a * raw2 <= field_floor) && (b * b * raw2 <= field_floor);
            double wrapped = std::fmod((bd - ad) + 180.0, 360.0);
            if (wrapped < 0.0) wrapped = wrapped + 360.0;
            if (wrapped == 0.0) wrapped = 0.0;
            const double delta = wrapped - 180.0;
            const double a2 = a * a, b2 = b * b;
            const double cos_2d = std::cos((2.0 * delta) * DEG_TO_RAD);
            const double squared = a2 * a2 + b2 * b2 + 2.0 * a2 * b2 * cos_2d;
            const double root = std::sqrt(0.0 > squared ? 0.0 : squared);
            const double major2 = 0.5 * (a2 + b2 + root);
            double minor2 = 0.5 * (a2 + b2 - root);
            if (0.0 > minor2) minor2 = 0.0;
            const double cos_d = std::cos(delta * DEG_TO_RAD);
            double tilt = 0.5 * (py_atan2(2.0 * a * b * cos_d, a2 - b2) * RAD_TO_DEG);
            const double ratio = std::sqrt(minor2 / major2);
            const bool linear = (major2 <= 0.0) || (ratio < 5.0e-6);
            const bool left = (delta * DEG_TO_RAD) >= 0.0;
            double axial = linear ? 0.0 : (left ? ratio : -ratio);
            std::string sense = linear ? "LINEAR" : (left ? "LEFT" : "RIGHT");
            if (blank) {
                axial = 0.0;
                tilt = 0.0;
                sense = "";
            }
            if (range_m != 0.0) {
                a = a / range_m;
                b = b / range_m;
                ad = ad + range_phase;
                bd = bd + range_phase;
            }
            PatternRowC row;
            row.theta = thetas[static_cast<std::size_t>(it)];
            row.phi = phis[static_cast<std::size_t>(ip)];
            row.vert = gain_db(g_v_rows[q]);
            row.hor = gain_db(g_h_rows[q]);
            row.total = gain_db(g_v_rows[q] + g_h_rows[q]);
            row.axial = axial;
            row.tilt = tilt;
            row.sense = sense;
            row.et_mag = a;
            row.et_deg = ad;
            row.ep_mag = b;
            row.ep_deg = bd;
            if (bl) {
                row.axial = row.tilt = row.et_mag = row.et_deg = row.ep_mag = row.ep_deg = 0.0;
                row.sense = "";
            }
            block.rows.push_back(row);
        }
    // XNDA's A digit asks for the average gain (Python's non-negative `% 10`)
    int64_t digit = rq.xnda.i64() % 10;
    if (digit < 0) digit += 10;
    if (digit != 0) {
        // A = 2 prints the trailer with no rows (momwire#1503).
        if (digit == 2) block.rows.clear();
        // `average_gain(g_v + g_h, thetas, d_theta, d_phi, n_phi)`
        const double d_theta = rq.d_theta_deg, d_phi = rq.d_phi_deg;
        const auto np_max = [](double x, double y) {
            return std::isnan(x) ? x : std::isnan(y) ? y : (x >= y ? x : y);
        };
        const auto np_min = [](double x, double y) {
            return std::isnan(x) ? x : std::isnan(y) ? y : (x <= y ? x : y);
        };
        std::vector<double> band(static_cast<std::size_t>(nt));
        for (int64_t it = 0; it < nt; it++) {
            const double lo = np_max(thetas[it] - 0.5 * d_theta, thetas[0]) * DEG_TO_RAD;
            const double hi = np_min(thetas[it] + 0.5 * d_theta, thetas[nt - 1]) * DEG_TO_RAD;
            band[it] = std::cos(lo) - std::cos(hi);
        }
        const int64_t columns = std::max<int64_t>(nph - 1, 1);
        const double step = d_phi != 0.0 ? d_phi * DEG_TO_RAD : 2.0 * 3.141592653589793;
        std::vector<double> weighted;
        for (int64_t it = 0; it < nt; it++)
            for (int64_t ip = 0; ip < columns; ip++) {
                const std::size_t q = static_cast<std::size_t>(it * nph + ip);
                weighted.push_back((g_v[q] + g_h[q]) * band[static_cast<std::size_t>(it)]);
            }
        const double total = (0.0 + np_pairwise(weighted.data(), static_cast<int64_t>(weighted.size()))) * step;
        const double solid = ((0.0 + np_pairwise(band.data(), nt)) * static_cast<double>(columns)) * step;
        const double average = solid != 0.0 ? total / solid : 0.0;
        block.has_average = true;
        block.average_power_gain = average;
        block.solid_angle_pi = std::fabs(solid) / 3.141592653589793;
        block.power_radiated_4pi = average * p_in;
    }
    return block;
}

// ============================================================================
// the near field (`_serve._near_field`, `_near_readout.near_field_at`)
// ============================================================================

using Elements = ElementCurrents;

struct NearRowC {  // `NearFieldRow`
    double point[3] = {0, 0, 0};
    double mag[3] = {0, 0, 0}, deg[3] = {0, 0, 0};
};

struct NearBlockC {  // `NearFieldBlock`
    std::vector<NearRowC> rows;
    bool magnetic = false;
};

// CPython's `divmod(x, 90.0)` (float_divmod), for `cos_sin_deg`.
void py_divmod(double vx, double wx, double &floordiv, double &mod) {
    mod = std::fmod(vx, wx);
    double div = (vx - mod) / wx;
    if (mod != 0.0) {
        if ((wx < 0) != (mod < 0)) {
            mod += wx;
            div -= 1.0;
        }
    } else {
        mod = std::copysign(0.0, wx);
    }
    if (div != 0.0) {
        floordiv = std::floor(div);
        if (div - floordiv > 0.5) floordiv += 1.0;
    } else {
        floordiv = std::copysign(0.0, vx / wx);
    }
}

// `cos_sin_deg`: exact on the four axes, libm elsewhere.
void cos_sin_deg(double a, double &c, double &s) {
    double quarter, rest;
    py_divmod(a, 90.0, quarter, rest);
    if (rest == 0.0) {
        long long q = static_cast<long long>(quarter) % 4;
        if (q < 0) q += 4;
        static const double table[4][2] = {{1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}};
        c = table[q][0];
        s = table[q][1];
        return;
    }
    const double rad = a * DEG_TO_RAD;
    c = std::cos(rad);
    s = std::sin(rad);
}

// `grid_points(coordinates, counts, origin, step)`: X fastest, then Y, then Z
// (R, theta, phi for a spherical card), each `start + idx * step`.
std::vector<double> near_points(const Request &rq) {
    for (int i = 0; i < 3; i++)
        if (!rq.counts[i].fits_i64()) throw Decline{"a near-field count past 64 bits"};
    if (!rq.coordinates.fits_i64() || (rq.coordinates.i64() != 0 && rq.coordinates.i64() != 1))
        throw Decline{"a near-field coordinate system other than 0 or 1 (the reference raises)"};
    const int64_t nx = rq.counts[0].i64(), ny = rq.counts[1].i64(), nz = rq.counts[2].i64();
    if (nx < 0 || ny < 0 || nz < 0 || nx * ny * nz > 10000000) throw Decline{"a near-field grid out of range"};
    std::vector<double> out;
    for (int64_t iz = 0; iz < nz; iz++)
        for (int64_t iy = 0; iy < ny; iy++)
            for (int64_t ix = 0; ix < nx; ix++) {
                const double idx[3] = {static_cast<double>(ix), static_cast<double>(iy), static_cast<double>(iz)};
                double v[3];
                for (int c = 0; c < 3; c++) v[c] = rq.origin[c] + idx[c] * rq.step[c];
                if (rq.coordinates.i64() == 0) {
                    out.insert(out.end(), v, v + 3);
                } else {
                    double ct, st, cp, sp;
                    cos_sin_deg(v[1], ct, st);
                    cos_sin_deg(v[2], cp, sp);
                    out.push_back((v[0] * st) * cp);
                    out.push_back((v[0] * st) * sp);
                    out.push_back(v[0] * ct);
                }
            }
    return out;
}

// numpy's `interp(x, xp, fp)` (no slopes cache difference: the same quotient).
double np_interp(double x, const double *xp, const double *fp, int64_t n) {
    if (std::isnan(x)) return x;
    if (x > xp[n - 1]) return fp[n - 1];
    if (x < xp[0]) return fp[0];
    if (x == xp[n - 1]) return fp[n - 1];
    const int64_t j = static_cast<int64_t>(std::upper_bound(xp, xp + n, x) - xp) - 1;
    if (x == xp[j]) return fp[j];
    const double slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    double r = slope * (x - xp[j]) + fp[j];
    if (std::isnan(r)) {
        r = slope * (x - xp[j + 1]) + fp[j + 1];
        if (std::isnan(r) && fp[j] == fp[j + 1]) r = fp[j];
    }
    return r;
}

// `element_currents(coeffs)` (subdiv 1, `_ElementCurrents`): per wire the knots
// along its edges (each edge an np.linspace, later edges without their first
// knot), the element midpoints `0.5 * (k1 + k0)` and moments
// `(0.5 * (I1 + I0))[:, None] * (k1 - k0)`.
void element_moments(const std::vector<WireWalk> &walk, const std::vector<std::vector<cd>> &knot_i,
                     std::vector<double> &mid, std::vector<cd> &moment) {
    for (std::size_t w = 0; w < walk.size(); w++) {
        std::vector<double> knots;
        const std::vector<double> &poly = walk[w].polyline;
        const std::vector<int64_t> &npe = walk[w].npe;
        for (std::size_t e = 0; e < npe.size(); e++) {
            std::vector<double> seg;
            linspace3(&poly[3 * e], &poly[3 * (e + 1)], npe[e] + 1, seg);
            knots.insert(knots.end(), seg.begin() + (e == 0 ? 0 : 3), seg.end());
        }
        const std::vector<cd> &cur = knot_i[static_cast<std::size_t>(w)];
        const std::size_t n_el = knots.size() / 3 - 1;
        if (cur.size() != n_el + 1) throw std::logic_error("knot currents and knots disagree");
        for (std::size_t e = 0; e < n_el; e++) {
            const double *k0 = &knots[3 * e], *k1 = &knots[3 * e + 3];
            const cd el = np_cmul(cd(0.5, 0.0), c_add(cur[e + 1], cur[e]));  // `0.5 * (...)`
            for (int c = 0; c < 3; c++) {
                mid.push_back(0.5 * (k1[c] + k0[c]));
                moment.push_back(c_mul_real(el, k1[c] - k0[c]));
            }
        }
    }
}

// `element_currents(coeffs, subdiv=8)`: each edge split 8 ways (linspace),
// the arc along the chords (`norm3` steps, `np.cumsum`), the current there by
// the basis' sampler (razor: `np.interp` of the knot currents), and the
// elements' midpoints, moments, nodes and steps.
ElementCurrents element_currents_sub(const std::vector<WireWalk> &walk, const CurrentSampler &sampler,
                                     int64_t subdiv) {
    ElementCurrents el;
    for (std::size_t w = 0; w < walk.size(); w++) {
        std::vector<double> knots;
        const std::vector<double> &poly = walk[w].polyline;
        const std::vector<int64_t> &npe = walk[w].npe;
        for (std::size_t e = 0; e < npe.size(); e++) {
            std::vector<double> seg;
            linspace3(&poly[3 * e], &poly[3 * (e + 1)], npe[e] * subdiv + 1, seg);
            knots.insert(knots.end(), seg.begin() + (e == 0 ? 0 : 3), seg.end());
        }
        const std::size_t nk = knots.size() / 3;
        std::vector<double> arcs(nk, 0.0);
        double run = 0.0;
        for (std::size_t q = 1; q < nk; q++) {
            const double dx = knots[3 * q] - knots[3 * q - 3], dy = knots[3 * q + 1] - knots[3 * q - 2],
                         dz = knots[3 * q + 2] - knots[3 * q - 1];
            const double step = std::sqrt((dx * dx + dy * dy) + dz * dz);
            run = q == 1 ? step : run + step;
            arcs[q] = run;
        }
        // a dropped copy: zeros (momwire#1042); else the basis' own samples
        const std::vector<cd> cur =
            walk[w].solved < 0 ? std::vector<cd>(nk, cd(0.0, 0.0)) : sampler.at(w, walk[w], arcs);
        std::vector<cd> element(nk - 1);
        for (std::size_t q = 0; q + 1 < nk; q++) {
            element[q] = np_cmul(cd(0.5, 0.0), c_add(cur[q + 1], cur[q]));
            for (int c = 0; c < 3; c++) {
                el.mid.push_back(0.5 * (knots[3 * q + 3 + c] + knots[3 * q + c]));
                el.moment.push_back(c_mul_real(element[q], knots[3 * q + 3 + c] - knots[3 * q + c]));
            }
        }
        el.nodes.insert(el.nodes.end(), knots.begin(), knots.end());
        for (std::size_t q = 0; q < nk; q++) {
            const cd in = q == 0 ? cd(0.0, 0.0) : element[q - 1];
            const cd out = q + 1 == nk ? cd(0.0, 0.0) : element[q];
            el.delta.push_back(c_sub(in, out));
        }
    }
    return el;
}

// `_element_fields(points, elements, k, radius, magnetic)`: the mixed-potential
// E (or H), every element sum over axis 1 in numpy's order (from the first
// term, in sequence).
std::vector<cd> element_fields(const std::vector<double> &points, const Elements &el, double k,
                               double radius, bool magnetic) {
    const std::size_t np_pts = points.size() / 3;
    std::vector<cd> out(np_pts * 3);
    const cd mjk = py_cmul(cd(-0.0, -1.0), cd(k, 0.0));  // `-1j * k`
    const auto geometry = [&](const double *p, const double *s, double rh[3], double &r, cd &green) {
        double rv[3];
        for (int c = 0; c < 3; c++) rv[c] = p[c] - s[c];
        r = std::sqrt(((rv[0] * rv[0] + rv[1] * rv[1]) + rv[2] * rv[2]) + radius * radius);
        for (int c = 0; c < 3; c++) rh[c] = rv[c] / r;
        const cd arg = c_mul_real(mjk, r);
        const double ex = std::exp(arg.real());
        green = np_cdiv(cd(ex * std::cos(arg.imag()), ex * std::sin(arg.imag())),
                        cd((4.0 * 3.141592653589793) * r, 0.0));
    };
    const std::size_t n_el = el.mid.size() / 3, n_nodes = el.nodes.size() / 3;
    if (magnetic) {
        const cd jk = py_cmul(cd(0.0, 1.0), cd(k, 0.0));
        for (std::size_t p = 0; p < np_pts; p++) {
            cd acc[3];
            for (std::size_t n = 0; n < n_el; n++) {
                double rh[3], r;
                cd green;
                geometry(&points[3 * p], &el.mid[3 * n], rh, r, green);
                const cd *m = &el.moment[3 * n];
                cd cross[3];
                for (int c = 0; c < 3; c++) {
                    const int i = (c + 1) % 3, j = (c + 2) % 3;
                    cross[c] = c_sub(c_mul_real(m[i], rh[j]), c_mul_real(m[j], rh[i]));
                }
                const cd weight = np_cmul(cd(jk.real() + 1.0 / r, jk.imag() + 0.0), green);
                for (int c = 0; c < 3; c++) {
                    const cd t = np_cmul(weight, cross[c]);
                    acc[c] = n == 0 ? t : c_add(acc[c], t);
                }
            }
            for (int c = 0; c < 3; c++) out[3 * p + c] = acc[c];
        }
        return out;
    }
    const cd pre = py_cmul(py_cmul(cd(-0.0, -1.0), cd(ETA0, 0.0)), cd(k, 0.0));        // `-1j * ETA0 * k`
    const cd pre_q = py_cdiv(py_cmul(cd(-0.0, -1.0), cd(ETA0, 0.0)), cd(k, 0.0));      // `-1j * ETA0 / k`
    const cd jk = py_cmul(cd(0.0, 1.0), cd(k, 0.0));                                     // `1j * k`
    for (std::size_t p = 0; p < np_pts; p++) {
        cd vec[3], sca[3];
        for (std::size_t n = 0; n < n_el; n++) {
            double rh[3], r;
            cd green;
            geometry(&points[3 * p], &el.mid[3 * n], rh, r, green);
            for (int c = 0; c < 3; c++) {
                const cd t = np_cmul(green, el.moment[3 * n + c]);
                vec[c] = n == 0 ? t : c_add(vec[c], t);
            }
        }
        for (std::size_t n = 0; n < n_nodes; n++) {
            double rh[3], r;
            cd green;
            geometry(&points[3 * p], &el.nodes[3 * n], rh, r, green);
            const cd jkr = c_mul_real(jk, r);
            const cd one_plus(1.0 + jkr.real(), 0.0 + jkr.imag());
            cd x = np_cmul(el.delta[n], one_plus);
            x = np_cdiv(x, cd(r, 0.0));
            x = np_cmul(x, green);
            const cd scalar = np_cmul(pre_q, x);
            for (int c = 0; c < 3; c++) {
                const cd t = c_mul_real(scalar, rh[c]);
                sca[c] = n == 0 ? t : c_add(sca[c], t);
            }
        }
        for (int c = 0; c < 3; c++) {
            const cd e = np_cmul(pre, n_el ? vec[c] : cd(0.0, 0.0));
            out[3 * p + c] = c_add(e, n_nodes ? sca[c] : cd(0.0, 0.0));
        }
    }
    return out;
}

// `_serve._near_field` over free space or GN 1, and `near_field_block`.
// The composing ground's near-field remainder (`_field_point.reflected_field_at`
// / `reflected_h_field_at` through `_somm_field_core`): the grid sized by
// `max_image_distance` over the observation points (or, for H, the six
// offset sets `points + shift`) and the element midpoints, at the solver's
// omega and `_sommerfeld._MU0`, then the kernel.
std::vector<cd> near_remainder(const std::vector<double> &points, const Elements &el, cd eps_t,
                               double k, double omega, bool magnetic) {
    const double mu0 = 4e-7 * 3.141592653589793;
    const double step = 1e-3 * (2.0 * 3.141592653589793 / k);  // CURL_STEP_WAVELENGTHS
    std::vector<double> ex;
    const std::size_t n = points.size() / 3;
    if (magnetic) {
        for (int axis = 0; axis < 3; axis++)
            for (double sign : {1.0, -1.0})
                for (std::size_t q = 0; q < n; q++)
                    for (int c = 0; c < 3; c++)
                        ex.push_back(points[3 * q + c] + (c == axis ? sign * step : 0.0));
    } else {
        ex = points;
    }
    ex.insert(ex.end(), el.mid.begin(), el.mid.end());
    const double r1_max = razor::max_image_distance(ex.data(), ex.data(), ex.size() / 3, 0.0);
    const somm_core::Grid &g = somm_core::get_grid_lazy(eps_t, k, r1_max, omega, mu0);
    const std::size_t n_reg = g.regions.size();
    if (n_reg > somm_proj::MAX_REGIONS) throw Decline{"a near-field grid layout"};
    double r0[somm_proj::MAX_REGIONS], dr[somm_proj::MAX_REGIONS], th0[somm_proj::MAX_REGIONS],
        dth[somm_proj::MAX_REGIONS];
    const cd *vals[somm_proj::MAX_REGIONS];
    std::ptrdiff_t shape0[somm_proj::MAX_REGIONS], nr[somm_proj::MAX_REGIONS],
        nth[somm_proj::MAX_REGIONS];
    for (std::size_t r = 0; r < n_reg; r++) {
        r0[r] = g.regions[r].r0;
        dr[r] = g.regions[r].dr;
        th0[r] = g.regions[r].th0;
        dth[r] = g.regions[r].dth;
        vals[r] = g.regions[r].vals.data();
        shape0[r] = 4;
        nr[r] = static_cast<std::ptrdiff_t>(g.regions[r].n_r);
        nth[r] = static_cast<std::ptrdiff_t>(g.regions[r].n_th);
    }
    somm_proj::GridView G = somm_proj::build_grid_view(g.r1_max, g.r_break, 30.0 * (M_PI / 180.0),
                                                       g.r_near, r0, dr, th0, dth, n_reg, vals,
                                                       shape0, nr, nth);
    const std::vector<cd> far = somm_core::far_pack(g);
    somm_proj::set_far(G, far.data(), static_cast<std::ptrdiff_t>(far.size()));
    std::vector<cd> out(3 * n);
    const std::ptrdiff_t n_src = static_cast<std::ptrdiff_t>(el.mid.size() / 3);
    {
        // Collect, then fill (momwire#1512): the nodes the kernel's reads
        // take at these points (H: its six offset points), then the read.
        std::vector<std::vector<uint8_t>> need(n_reg);
        somm_proj::NodeMarks marks;
        marks.n_reg = n_reg;
        for (std::size_t r = 0; r < n_reg; r++) {
            need[r].assign(static_cast<std::size_t>(nr[r] * nth[r]), 0);
            marks.m[r] = need[r].data();
            marks.nR[r] = nr[r];
            marks.nTh[r] = nth[r];
        }
        if (magnetic)
            somm_core::reflected_h_field_nodes(G, static_cast<std::ptrdiff_t>(n), points.data(),
                                               n_src, el.mid.data(), 0.0, step, marks);
        else
            somm_core::reflected_field_nodes(G, static_cast<std::ptrdiff_t>(n), points.data(), n_src,
                                             el.mid.data(), 0.0, marks);
        somm_core::ensure_nodes(g, need);
    }
    if (magnetic)
        somm_core::reflected_h_field(G, static_cast<std::ptrdiff_t>(n), points.data(), n_src,
                                     el.mid.data(), el.moment.data(), 0.0, k, omega, mu0, step,
                                     out.data());
    else
        somm_core::reflected_field(G, static_cast<std::ptrdiff_t>(n), points.data(), n_src,
                                   el.mid.data(), el.moment.data(), 0.0, k, out.data());
    if (somm_core::lazy_check()) somm_core::check_read(out.data(), out.size(), "the near-field remainder");
    return out;
}

// `near_ground` / `near_field_at`: free space, GN 1's image (`pec`), or the
// composing ground (`compose`: GN 0 / GN 2's eps~ from the solver's ground,
// GD's from its medium -- one expression, at the solver's own omega):
// `field + (C2 * image + remainder)`, C2's product numpy's fused one.
// The block over given elements (`near_field_at` + `near_field_block`):
// `compose` reads the medium (eps_r, sigma) at `omega` with `eps0`.
// `near_field_at` over the elements alone (no buried composition): free
// space, GN 1's image, or `field + (C2 * image + remainder)`.
std::vector<cd> near_field_values(bool magnetic, const std::vector<double> &points, const Elements &el, double k,
                                  double radius, bool pec, bool compose, double omega, double eps0, double eps_r,
                                  double sigma) {
    Request rq;
    rq.magnetic = magnetic;
    std::vector<cd> field = element_fields(points, el, k, radius, rq.magnetic);
    if (pec || compose) {
        Elements img = el;
        for (std::size_t n = 0; n < el.mid.size() / 3; n++) {
            img.mid[3 * n + 2] = 2.0 * 0.0 - el.mid[3 * n + 2];
            for (int c = 0; c < 3; c++) img.moment[3 * n + c] = c_mul_real(el.moment[3 * n + c], c == 2 ? 1.0 : -1.0);
        }
        for (std::size_t n = 0; n < el.nodes.size() / 3; n++) img.nodes[3 * n + 2] = 2.0 * 0.0 - el.nodes[3 * n + 2];
        for (std::size_t n = 0; n < el.delta.size(); n++) img.delta[n] = cd(-el.delta[n].real(), -el.delta[n].imag());
        std::vector<cd> image = element_fields(points, img, k, radius, rq.magnetic);
        if (compose) {
            // `eps_tilde((eps_r, sigma), omega, eps0)`; C2 = (eps_t - 1.0) /
            // (eps_t + 1.0).
            const cd eps_t(eps_r, (-sigma) / (omega * eps0));
            const cd c2 = somm_core::py_quot(cd(eps_t.real() - 1.0, eps_t.imag() - 0.0),
                                             cd(eps_t.real() + 1.0, eps_t.imag() + 0.0));
            const std::vector<cd> rem = near_remainder(points, el, eps_t, k, omega, rq.magnetic);
            for (std::size_t q = 0; q < field.size(); q++) {
                const cd im = image[q];
                const cd scaled(mw_fma::exact(c2.real(), im.real(), -(c2.imag() * im.imag())),
                                mw_fma::exact(c2.real(), im.imag(), c2.imag() * im.real()));
                field[q] = c_add(field[q], c_add(scaled, rem[q]));
            }
        } else {
            for (std::size_t q = 0; q < field.size(); q++) field[q] = c_add(field[q], image[q]);
        }
    }
    return field;
}

// The buried readout's thread budget: the process's OpenMP team (its values
// do not depend on it).
int mw_threads_for_near() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// `_near_readout._near_field_buried` (momwire#1653): over the composing
// ground with an element or a point below the plane, the above/above share of
// the above-ground elements first, then `near_buried::parts` -- the code the
// Python route calls (`near_buried_parts`).
std::vector<cd> near_field_buried(bool magnetic, const std::vector<double> &points, const Elements &el,
                                  const std::vector<int64_t> &counts, int subdiv, double k, double radius,
                                  double omega, double eps0, double eps_r, double sigma) {
    const cd eps_t(eps_r, (-sigma) / (omega * eps0));
    near_buried::Elements ne;
    ne.mid = el.mid.data();
    ne.moment = el.moment.data();
    ne.n_el = el.mid.size() / 3;
    ne.nodes = el.nodes.data();
    ne.delta = el.delta.data();
    ne.n_nodes = el.nodes.size() / 3;
    ne.counts = counts.data();
    ne.n_wires = counts.size();
    ne.subdiv = subdiv;
    near_buried::Sources above, below;
    near_buried::split(ne, 0.0, above, below);
    const std::size_t n = points.size() / 3;
    std::vector<cd> fresh(3 * n);
    near_buried::parts(points.data(), n, ne, eps_t, 0.0, k, omega, 4e-7 * 3.141592653589793, radius, magnetic,
                       mw_threads_for_near(), fresh.data());
    std::vector<double> pa;
    std::vector<std::size_t> ia;
    for (std::size_t q = 0; q < n; q++)
        if (points[3 * q + 2] >= 0.0) {
            pa.insert(pa.end(), &points[3 * q], &points[3 * q] + 3);
            ia.push_back(q);
        }
    std::vector<cd> out(3 * n, cd(0.0, 0.0));
    if (!ia.empty() && above.n_el() != 0) {
        Elements ea;
        ea.mid = above.mid;
        ea.moment = above.moment;
        ea.nodes = above.nodes;
        ea.delta = above.delta;
        const std::vector<cd> aa =
            near_field_values(magnetic, pa, ea, k, radius, false, true, omega, eps0, eps_r, sigma);
        for (std::size_t r = 0; r < ia.size(); r++)
            for (int c = 0; c < 3; c++) out[3 * ia[r] + c] = aa[3 * r + c];
    }
    for (std::size_t q = 0; q < n; q++)
        for (int c = 0; c < 3; c++) {
            const std::size_t i = 3 * q + static_cast<std::size_t>(c);
            out[i] = points[3 * q + 2] >= 0.0 ? c_add(out[i], fresh[i]) : fresh[i];
        }
    return out;
}

// `near_field_block`: the rows, each cell floored before it is read.
NearBlockC near_rows(const Request &rq, const std::vector<double> &points, const std::vector<cd> &field) {
    NearBlockC block;
    block.magnetic = rq.magnetic;
    for (std::size_t q = 0; q < points.size() / 3; q++) {
        NearRowC row;
        for (int c = 0; c < 3; c++) {
            row.point[c] = points[3 * q + c];
            cd cell = field[3 * q + c];
            if (cell.real() * cell.real() + cell.imag() * cell.imag() <= 1.0e-20) cell = cd(0.0, 0.0);
            row.mag[c] = std::hypot(cell.real(), cell.imag());
            row.deg[c] = py_degrees(py_atan2(cell.imag(), cell.real()));
        }
        block.rows.push_back(row);
    }
    return block;
}

NearBlockC near_block_of(const Request &rq, const std::vector<double> &points, const Elements &el, double k,
                         double radius, bool pec, bool compose, double omega, double eps0, double eps_r,
                         double sigma) {
    return near_rows(rq, points,
                     near_field_values(rq.magnetic, points, el, k, radius, pec, compose, omega, eps0, eps_r, sigma));
}

NearBlockC near_block(const Request &rq, const PrintoutEngine &engine,
                      const std::vector<WireWalk> &walk, const CurrentSampler &sampler,
                      const Prepared &p, bool pec, bool compose) {
    const std::vector<double> points = near_points(rq);
    const double k = 2.0 * 3.141592653589793 / p.wavelength;
    double radius = p.mesh.pieces.empty() ? 0.0 : p.mesh.pieces[0].radius;
    for (const SeamPiece &piece : p.mesh.pieces)
        if (piece.radius < radius) radius = piece.radius;  // Python's `min`
    if (p.mesh.pieces.empty()) throw Decline{"a near field with no wire"};
    const Elements el = element_currents_sub(walk, sampler, 8);
    // `solver.omega` = 2 * np.pi * (c / wavelength)
    const double omega = compose ? engine.solver_omega(p) : 0.0;
    const double eps0 = compose ? engine.eps0() : 0.0;
    if (compose) {
        // `_near_readout._crosses`: an element or a point below the plane
        bool crosses = false;
        for (std::size_t e = 0; e < el.mid.size() / 3 && !crosses; e++) crosses = el.mid[3 * e + 2] < 0.0;
        for (std::size_t q = 0; q < points.size() / 3 && !crosses; q++) crosses = points[3 * q + 2] < 0.0;
        if (crosses) {
            // `element_counts(solver)` at subdiv 8: each wire's segments x 8
            std::vector<int64_t> counts;
            for (const WireWalk &w : walk) {
                int64_t m = 0;
                for (int64_t n_e : w.npe) m += n_e;
                counts.push_back(m * 8);
            }
            return near_rows(rq, points,
                             near_field_buried(rq.magnetic, points, el, counts, 8, k, radius, omega, eps0,
                                               p.medium_eps_r, p.medium_sigma));
        }
    }
    return near_block_of(rq, points, el, k, radius, pec, compose, omega, eps0, p.medium_eps_r, p.medium_sigma);
}

RunDataC finish(const Deck &deck, const Prepared &p, const Solved &s, const NumpyLinalg &np,
                std::vector<RunArrays> *arrays, const PrintoutEngine &engine) {
    const PortState st = port_state(deck, p, s, np);
    // `coeffs = solution.coeffs @ (_transform(mesh) @ state.v_gap)`
    const int64_t nc = p.mesh.n_columns(), ns = static_cast<int64_t>(p.mesh.sites.size());
    const std::vector<cd> tc = transform_c(p);
    const std::vector<cd> w = mv(np, Op{tc.data(), nc, ns, ns, 1}, st.v_gap);
    const Op x = s.x_c_order ? Op{s.x.data(), s.n_basis, s.n_ports, s.n_ports, 1}
                             : Op{s.x.data(), s.n_basis, s.n_ports, 1, s.n_basis};
    const std::vector<cd> coeffs = mv(np, x, w);
    if (arrays != nullptr)
        arrays->push_back(RunArrays{s.y.v, coeffs, st.v_applied, st.v_gap, st.i_port, st.i_source});

    RunDataC data;
    // `loads=tuple(_loading_row(card) for card in deck.load_cards)`, deck
    // order. A zero field the card wrote prints BLANK (`spec.x or None`,
    // `spec.r or None`; an LD 2 / 3 field passed as written and blanked when
    // falsy).
    for (const LoadCardRef &ref : deck.load_cards) {
        const auto truthy = [](double v) { return v != 0.0 || std::isnan(v); };
        if (ref.kind == LoadCardRef::conductivity) {
            const Conductivity &cc = deck.conductivities[ref.index];
            LoadRowC row;
            row.tag = cc.tag;
            row.node_from = cc.segment_from;
            row.node_thru = cc.segment_thru;
            row.conductivity = cc.sigma;
            row.kind = "WIRE";
            data.loads.push_back(row);
            continue;
        }
        if (ref.kind == LoadCardRef::distributed_rlc) {
            const DistributedRLCCard &dc = deck.distributed_rlc[ref.index];
            LoadRowC row;
            row.tag = dc.tag;
            row.node_from = dc.segment_from;
            row.node_thru = dc.segment_thru;
            row.resistance = dc.spec.r;
            row.has_resistance = truthy(dc.spec.r);
            row.inductance = dc.spec.l;
            row.has_inductance = truthy(dc.spec.l);
            row.capacitance = dc.spec.c;
            row.has_capacitance = truthy(dc.spec.c);
            row.kind = dc.spec.kind == "series" ? "SERIES (PER METER)" : "PARALLEL (PER METER)";
            data.loads.push_back(row);
            continue;
        }
        const Load &load = deck.loads[ref.index];
        const LoadSpec &spec = load.spec;
        LoadRowC row;
        row.tag = load.at.tag;
        row.node_from = load.printed_location;
        row.node_thru = load.printed_location;
        if (spec.kind == "fixed") {
            row.kind = "FIXED IMPEDANCE";
            row.resistance = spec.r;
            row.has_resistance = true;
            row.reactance = spec.x;
            row.has_reactance = truthy(spec.x);
        } else {
            row.kind = spec.kind == "series" ? "SERIES" : "PARALLEL";
            row.resistance = spec.r;
            row.has_resistance = truthy(spec.r);
            row.inductance = spec.l;
            row.has_inductance = truthy(spec.l);
            row.capacitance = spec.c;
            row.has_capacitance = truthy(spec.c);
        }
        data.loads.push_back(row);
    }
    // the source rows, deck order
    // `_gyrator_drives`: an EX 0 on a phantom node behind an NT gyrator
    // (Y11 = Y22 = 0, Y12 = jB, |B| >= 1e-12) reports at the node the gyrator
    // drives, with the card's `-Y12 * V` as its current (momwire#1134).
    struct Gyrator {
        Node source_at, real_at;
        cd current;
    };
    std::vector<Gyrator> gyrators;
    {
        const auto is_phantom = [&](const Int &tag) {
            for (const Int &t : p.phantom_tags)
                if (t == tag) return true;
            return false;
        };
        std::vector<const Network *> candidates;
        for (const Network &net : deck.networks) {
            const bool y11 = net.y11[0] != 0.0 || net.y11[1] != 0.0;
            const bool y22 = net.y22[0] != 0.0 || net.y22[1] != 0.0;
            if (!(y11 || y22) && !(net.y12[0] != 0.0) && std::fabs(net.y12[1]) >= 1e-12)
                candidates.push_back(&net);
        }
        if (!candidates.empty() && !p.phantom_tags.empty()) {
            const auto card_count = [&](const Node &at) {
                long long n = 0;
                for (const TransmissionLine &t : deck.transmission_lines)
                    n += same_node(t.end_a, at) + same_node(t.end_b, at);
                for (const Network &x : deck.networks) n += same_node(x.end_a, at) + same_node(x.end_b, at);
                return n;
            };
            const auto drives_at = [&](const Node &at) {
                std::vector<const Source *> out;
                for (const Source &src : deck.sources)
                    if (same_node(src.at, at)) out.push_back(&src);
                return out;
            };
            std::vector<Node> claimed;
            for (const Network *net : candidates) {
                std::vector<Node> on_phantom;
                for (const Node &at : {net->end_a, net->end_b})
                    if (is_phantom(at.tag)) on_phantom.push_back(at);
                if (on_phantom.size() != 1) continue;
                const Node at = on_phantom[0];
                const Node real = same_node(at, net->end_a) ? net->end_b : net->end_a;
                bool taken = !drives_at(real).empty();
                for (const Node &c : claimed) taken = taken || same_node(c, real);
                if (taken) continue;
                const std::vector<const Source *> held = drives_at(at);
                if (held.size() != 1 || card_count(at) != 1) continue;
                const Source &src = *held[0];
                const cd drive(src.drive_re, src.drive_im);
                if (!src.kind.is_zero() || !py_ctruth(drive)) continue;
                claimed.push_back(real);
                bool seen = false;  // `found[at] = ...` (a dict: the last one wins)
                for (Gyrator &g : gyrators)
                    if (same_node(g.source_at, at)) {
                        g = Gyrator{at, real, py_cmul(cd(-net->y12[0], -net->y12[1]), drive)};
                        seen = true;
                    }
                if (!seen) gyrators.push_back(Gyrator{at, real, py_cmul(cd(-net->y12[0], -net->y12[1]), drive)});
            }
        }
    }
    const auto port_row_at = [&](const Node &at, cd voltage, cd current) {
        PortRowC row;
        row.tag = at.tag;
        const Int node = at.node < Int(1) ? Int(1) : at.node;
        row.segment = Int(first_element(deck, at.tag)) + node - Int(1);
        row.end_index = at.node.is_zero() ? 2 : 1;
        row.voltage = voltage;
        row.current = current;
        row.impedance = ratio(voltage, current);
        row.admittance = ratio(current, voltage);
        row.power = 0.5 * py_cmul(voltage, std::conj(current)).real();
        return row;
    };
    std::vector<double> powers;
    for (const Source &src : deck.sources) {
        const Gyrator *gy = nullptr;
        for (const Gyrator &g : gyrators)
            if (same_node(g.source_at, src.at)) gy = &g;
        if (gy != nullptr) {
            int64_t real_index = -1;
            for (const SeamSite &site : p.mesh.sites)
                if (same_node(site.at, gy->real_at)) real_index = site.index;
            const PortRowC row = port_row_at(gy->real_at, st.v_applied[static_cast<std::size_t>(real_index)],
                                             gy->current);
            powers.push_back(row.power);
            data.sources.push_back(row);
            continue;
        }
        int64_t index = -1;
        for (const SeamSite &site : p.mesh.sites)
            if (same_node(site.at, src.at)) index = site.index;
        cd voltage = st.v_applied[static_cast<std::size_t>(index)];
        cd current = st.i_source[static_cast<std::size_t>(index)];
        const cd drive(src.drive_re, src.drive_im);
        if (src.kind.is_zero())
            voltage = drive;
        else
            current = drive;
        PortRowC row;
        row.tag = src.at.tag;
        if (src.end_code == Int(1) || src.end_code == Int(2)) {
            row.segment = Int(first_element(deck, src.at.tag)) + src.printed_location - Int(1);
            row.end_index = src.end_code == Int(1) ? 2 : 1;
        } else {
            const Int node = src.at.node < Int(1) ? Int(1) : src.at.node;
            row.segment = Int(first_element(deck, src.at.tag)) + node - Int(1);
            row.end_index = src.at.node.is_zero() ? 2 : 1;
        }
        row.voltage = voltage;
        row.current = current;
        row.impedance = ratio(voltage, current);
        row.admittance = ratio(current, voltage);
        row.power = 0.5 * py_cmul(voltage, std::conj(current)).real();
        powers.push_back(row.power);
        data.sources.push_back(row);
    }
    const double p_in = py_sum(powers);
    // `0.5 * float(np.sum(np.real(z_load) * np.abs(i_port) ** 2))`: numpy's
    // hypot, its square, its product, and np.sum's identity plus pairwise sum
    std::vector<double> watts;
    for (std::size_t k = 0; k < st.i_port.size(); k++) {
        const double mag = std::hypot(st.i_port[k].real(), st.i_port[k].imag());
        watts.push_back(st.z_load[k].real() * (mag * mag));
    }
    double p_load = 0.5 * (0.0 + np_pairwise(watts.data(), static_cast<int64_t>(watts.size())));
    // `if deck.wire_conductivity: p_load += solver.wire_loss_power(coeffs, omega)[0]`
    if (!deck.wire_conductivity.empty()) p_load += engine.wire_loss_power(p, coeffs, p.omega);
    // `_connection_points(cards)`: each card's end A then end B, a site once
    std::vector<int64_t> points;
    for (const SeamCard &c : p.cards)
        for (int64_t site : {c.site_a, c.site_b})
            if (std::find(points.begin(), points.end(), site) == points.end()) points.push_back(site);
    std::vector<double> network_powers;
    for (int64_t index : points) {
        const SeamSite &site = p.mesh.sites[static_cast<std::size_t>(index)];
        const PortRowC row = port_row_at(site.at, st.v_applied[static_cast<std::size_t>(index)],
                                         st.i_port[static_cast<std::size_t>(index)]);
        network_powers.push_back(row.power);
        data.connections.push_back(row);
    }
    // `p_network = -sum(row.power for row in connections)` (0 with none)
    const double p_network = network_powers.empty() ? 0.0 : -py_sum(network_powers);
    const double p_radiated = (p_in - p_network) - p_load;
    if (p_network > 1e-9 * std::max(p_in, 1.0)) {
        data.has_network_loss = true;
        data.network_loss = p_network;
    }
    for (const SeamCard &c : p.cards) {
        NetworkRowC row;
        row.line = c.kind == SeamCard::tl;
        row.tag_from = c.tag_from;
        row.segment_from = c.segment_from;
        row.tag_to = c.tag_to;
        row.segment_to = c.segment_to;
        if (row.line) {
            row.cells[0] = std::fabs(c.payload[0]);
            row.cells[1] = c.payload[1];
            for (int k = 0; k < 4; k++) row.cells[2 + k] = c.payload[2 + k];
            row.crossed = c.payload[0] < 0.0;
        } else {
            std::copy(c.payload, c.payload + 6, row.cells);
        }
        data.networks.push_back(row);
    }

    // `_element_geometry`
    const double wavelength = p.wavelength;
    std::vector<double> centres, lengths;
    std::vector<Int> tags;
    for (const Wire &wire : deck.wires) {
        const long long n = wire.segment_count.i64();
        if (n <= 0) continue;
        std::vector<double> pts(static_cast<std::size_t>(3 * (n + 1)));
        for (long long k = 0; k <= n; k++) {
            double *q = &pts[static_cast<std::size_t>(3 * k)];
            if (k == 0) {
                std::copy(wire.end1, wire.end1 + 3, q);
            } else if (k == n) {
                std::copy(wire.end2, wire.end2 + 3, q);
            } else {
                const double t = static_cast<double>(k) / static_cast<double>(n);
                for (int c = 0; c < 3; c++) q[c] = wire.end1[c] + (wire.end2[c] - wire.end1[c]) * t;
            }
        }
        for (long long k = 0; k < n; k++) {
            const double *a = &pts[static_cast<std::size_t>(3 * k)], *b = a + 3;
            for (int c = 0; c < 3; c++) centres.push_back((0.5 * (a[c] + b[c])) / wavelength);
            const double dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
            lengths.push_back(std::sqrt((dx * dx + dy * dy) + dz * dz) / wavelength);
            tags.push_back(wire.tag);
        }
    }

    // `_element_currents_and_charges`
    const std::vector<std::vector<cd>> knots = engine.currents_at_knots(p, coeffs);
    // each piece's element centres along its arc; a deck element cut at
    // z = 0 (momwire#1281) reads its charge at its OLD centre, one extra
    // sample appended to the half that holds it
    std::vector<std::vector<double>> centres_per_piece;
    for (std::size_t w = 0; w < p.mesh.pieces.size(); w++) {
        const SeamPiece &piece = p.mesh.pieces[w];
        const int64_t ne = piece.n_elements();
        const double step = piece.length() / static_cast<double>(ne);
        std::vector<double> centres_s;
        for (int64_t k = 0; k < ne; k++) centres_s.push_back((static_cast<double>(k) + 0.5) * step);
        for (const auto &se : p.mesh.split_elements)
            if (se.second.centre_piece == static_cast<long long>(w)) centres_s.push_back(se.second.centre_s);
        centres_per_piece.push_back(std::move(centres_s));
    }
    const std::vector<std::vector<cd>> slopes = engine.current_slopes(p, coeffs, knots, centres_per_piece);
    const cd jw = py_cmul(cd(0.0, 1.0), cd(p.omega, 0.0));
    for (std::size_t index = 0; index < p.mesh.element_of.size(); index++) {
        const auto &pe = p.mesh.element_of[index];
        cd current(0.0, 0.0), charge(0.0, 0.0);  // a PHANTOM element's row is exactly 0
        const auto split = p.mesh.split_elements.find(static_cast<long long>(index));
        if (pe.first >= 0 && split != p.mesh.split_elements.end()) {
            const SplitElement &se = split->second;
            const cd first = knots[static_cast<std::size_t>(se.lower)].front();
            const cd last = knots[static_cast<std::size_t>(se.upper)].back();
            current = np_scalar_cmul(cd(0.5, 0.0), cd(first.real() + last.real(), first.imag() + last.imag()));
            const cd sl = slopes[static_cast<std::size_t>(se.centre_piece)].back();
            charge = np_cdiv(cd(-sl.real(), -sl.imag()), jw);
        } else if (pe.first >= 0) {
            const std::vector<cd> &k = knots[static_cast<std::size_t>(pe.first)];
            const int64_t el = pe.second;
            const cd sum(k[el].real() + k[el + 1].real(), k[el].imag() + k[el + 1].imag());
            current = np_scalar_cmul(cd(0.5, 0.0), sum);
            const cd sl = slopes[static_cast<std::size_t>(pe.first)][static_cast<std::size_t>(el)];
            charge = np_cdiv(cd(-sl.real(), -sl.imag()), jw);
        }
        ElementRowC row;
        row.element = static_cast<long long>(index) + 1;
        row.tag = tags[index];
        std::copy(&centres[3 * index], &centres[3 * index] + 3, row.center);
        row.length = lengths[index];
        row.value = current;
        row.magnitude = std::hypot(current.real(), current.imag());
        row.phase_deg = py_degrees(py_atan2(current.imag(), current.real()));
        data.currents.push_back(row);
        if (deck.has_pq) {
            ElementRowC q = row;
            q.value = charge;
            q.magnitude = std::hypot(charge.real(), charge.imag());
            q.phase_deg = py_degrees(py_atan2(charge.imag(), charge.real()));
            data.charges.push_back(q);
        }
    }

    data.node_count = p.structure.node_count;
    data.wire_element_count = p.structure.wire_element_count;
    data.unknown_count = p.structure.unknown_count;
    data.frequency_mhz = p.frequency_mhz;
    data.wavelength_m = p.wavelength;
    if (p.has_medium) {
        data.environment = RunDataC::finite_ground;
        data.eps_r = p.medium_eps_r;
        data.sigma = p.medium_sigma;
        data.eps_c[0] = p.medium_eps_c[0];
        data.eps_c[1] = p.medium_eps_c[1];
    } else if (deck.ground.kind == Ground::perfect) {
        data.environment = RunDataC::perfect_ground;
    }
    data.input_power = p_in;
    data.radiated_power = p_radiated;
    data.wire_loss = p_load;
    data.efficiency = p_in > 0 ? (100.0 * p_radiated) / p_in : 0.0;
    // `patterns=tuple(_pattern(...) for request in deck.requests if RP)`, over
    // `_far_ground(deck, medium)`: PEC for GN 1, the Fresnel-weighted image
    // for a medium -- "refl" for a bare GD's, "sommerfeld" for GN 0 / GN 2's,
    // which `_far_moments` serves as ONE branch above the plane (the seam
    // declines a buried wire) -- and free space otherwise.
    // `near_fields=tuple(_near_field(...) for request if NE / NH)`: free
    // space, GN 1's image, or the composing ground's `C2 image + remainder`
    // (GN 0 / GN 2, and GD's medium over its PEC-image currents).
    {
        bool any_near = false;
        for (const Request &rq : deck.requests) any_near = any_near || rq.kind == Request::near_field;
        if (any_near) {
            const mw_times::Scope timed(mw_times::near_field);
            const std::vector<WireWalk> walk = engine.wire_walk(p);
            const std::unique_ptr<CurrentSampler> sampler = engine.sampler(p, coeffs, knots);
            for (const Request &rq : deck.requests)
                if (rq.kind == Request::near_field)
                    data.near_fields.push_back(std::make_shared<NearBlockC>(
                        near_block(rq, engine, walk, *sampler, p, deck.ground.kind == Ground::perfect,
                                   p.has_medium)));
        }
    }
    bool any_rp = false;
    for (const Request &rq : deck.requests) any_rp = any_rp || rq.kind == Request::far_field;
    if (any_rp) {
        const mw_times::Scope timed(mw_times::far_field);
        FarGround fg = FarGround::free;
        if (deck.ground.kind == Ground::perfect)
            fg = FarGround::pec;
        else if (p.has_medium)
            fg = FarGround::refl;
        std::vector<double> mid;
        std::vector<cd> moment;
        element_moments(engine.wire_walk(p), knots, mid, moment);
        for (const Request &rq : deck.requests)
            if (rq.kind == Request::far_field)
                data.patterns.push_back(pattern_block(
                    rq, mid, moment, fg, p.medium_eps_r, p.medium_sigma, p.frequency_mhz,
                    p.wavelength, p_in, p.has_medium, cd(p.medium_eps_c[0], p.medium_eps_c[1]),
                    p_radiated));
    }
    return data;
}

// ============================================================================
// the writer (`_printout.py`)
// ============================================================================

typedef std::vector<std::string> Lines;  // UTF-8

void add(Lines &out, const Lines &more) { out.insert(out.end(), more.begin(), more.end()); }
void blanks(Lines &out, int n) { out.insert(out.end(), static_cast<std::size_t>(n), std::string()); }

std::string join(const Lines &lines) {
    std::string out;
    for (std::size_t k = 0; k < lines.size(); k++) {
        if (k) out += '\n';
        out += lines[k];
    }
    return out;
}

std::string sp(std::size_t n) { return std::string(n, ' '); }

// `_stamp_field`: "_".join(text.split())
std::string stamp_field(const std::string &text) {
    std::string out;
    for (const std::string &part : latin1_split(text)) {
        if (!out.empty()) out += '_';
        out += part;
    }
    return out;
}

Lines banner() {
    const std::string indent = sp(32), rule(47, '*');
    const std::string title = ljust("*  NUMERICAL ELECTROMAGNETICS CODE (NEC-5)", 46) + "*";
    return {indent + rule, indent + "*" + sp(45) + "*", indent + title,
            indent + "*" + sp(45) + "*", indent + rule, "", "", "", ""};
}

// `comment_cards` (latin-1 in, latin-1 out)
Lines comment_cards(const std::string &deck_text) {
    Lines texts;
    for (const std::string &line : latin1_splitlines(deck_text)) {
        const std::string mnemonic = latin1_upper(line.substr(0, std::min<std::size_t>(2, line.size())));
        const std::string text = line.size() > 2 ? line.substr(2, 78) : std::string();
        if (mnemonic == "CM") {
            texts.push_back(text);
        } else if (mnemonic == "CE") {
            texts.push_back(text);
            break;
        } else {
            break;
        }
    }
    return texts;
}

// `render_header`
std::string render_header(const std::string *deck_text, const std::string &basis,
                          const EngineStamp &stamp) {
    Lines lines = {"1", latin1_to_utf8(engine_stamp(basis, stamp)), "", "", ""};
    add(lines, banner());
    if (deck_text != nullptr) {
        const std::string rule = sp(22) + std::string(58, '*');
        lines.push_back(rule);
        lines.push_back("");
        for (const std::string &text : comment_cards(*deck_text))
            lines.push_back(sp(25) + latin1_to_utf8(ljust(text, 78)));
        lines.push_back("");
        lines.push_back(rule);
    }
    blanks(lines, 5);
    return join(lines) + "\n";
}

// `_post_ge_cards`
std::vector<Card> post_ge_cards(const std::string &deck_text) {
    std::vector<Card> cards;
    bool seen_ge = false;
    for (const std::string &line : latin1_splitlines(deck_text)) {
        Card card;
        if (!read_card(line, card)) continue;
        if (!seen_ge) {
            seen_ge = card.mnemonic == "GE";
            continue;
        }
        cards.push_back(card);
        if (card.mnemonic == "EN") break;
    }
    return cards;
}

// `_card_echo`
std::string card_echo(long long index, const Card &card) {
    std::string ints;
    for (std::size_t k = 0; k < 4; k++) ints += d(card.i(k), k == 0 ? 4 : 5);
    std::string reals;
    for (std::size_t k = 0; k < 6; k++) reals += e(card.f(4 + k), 13, 5);
    std::string line = " ***** INPUT LINE" + d(index, 3) + "  " + card.mnemonic + ints + reals;
    if (card.has_trailer)
        line += "\n" + sp(43) + e(card.f(10), 13, 5) + "  " +
                latin1_to_utf8(ljust(card.trailer, 40));
    return line;
}

bool is_terminator(const Card &c) { return c.mnemonic == "EN" || c.mnemonic == "NX"; }

Lines structure_specification(const Deck &deck, const RunDataC &data) {
    Lines lines = {sp(33) + "- - - STRUCTURE SPECIFICATION - - -",
                   "",
                   sp(37) + "COORDINATES MUST BE INPUT IN",
                   sp(37) + "METERS OR BE SCALED TO METERS",
                   sp(37) + "BEFORE STRUCTURE INPUT IS ENDED",
                   "",
                   " Model definition commands:",
                   ""};
    for (const Wire &w : deck.wires) {
        std::string coords;
        for (double v : {w.end1[0], w.end1[1], w.end1[2], w.end2[0], w.end2[1], w.end2[2], w.radius})
            coords += e(v, 13, 5);
        lines.push_back(" GW" + d(w.tag, 5) + d(w.segment_count, 6) + coords);
    }
    lines.push_back(" GE" + d(deck.ge_flag, 5) + d(deck.ge_second, 5));
    lines.push_back("");
    if (!deck.ge_flag.is_zero())
        add(lines, {"   GROUND PLANE SPECIFIED.", "",
                    "   WHERE WIRE ENDS TOUCH GROUND, CURRENT WILL BE INTERPOLATED TO IMAGE IN "
                    "GROUND PLANE.",
                    "", ""});
    const std::vector<std::string> notes = split_notes(deck);
    if (!notes.empty()) {
        add(lines, notes);
        lines.push_back("");
    }
    lines.push_back(" Number of nodes :" + d(data.node_count, 6));
    lines.push_back(" Number of wire elements :" + d(data.wire_element_count, 6));
    lines.push_back(" Number of patch elements:" + d(0LL, 6));
    lines.push_back(" Number unknowns:" + d(data.unknown_count, 6));
    return lines;
}

Lines frequency(const RunDataC &data) {
    return {sp(33) + "- - - - - - FREQUENCY - - - - - -", "",
            sp(36) + "FREQUENCY=" + e(data.frequency_mhz, 11, 4) + " MHZ",
            sp(36) + "WAVELENGTH=" + e(data.wavelength_m, 11, 4) + " METERS"};
}

Lines environment(const RunDataC &data) {
    const std::string heading = sp(34) + "- - - ANTENNA ENVIRONMENT - - -";
    if (data.environment != RunDataC::finite_ground)
        return {heading, "",
                rstrip_blanks(center(data.environment == RunDataC::perfect_ground ? "PERFECT GROUND"
                                                                           : "FREE SPACE",
                                     98))};
    const std::string m = sp(40);
    return {heading,
            "",
            m + "FINITE GROUND.  SOMMERFELD SOLUTION",
            m + "RELATIVE DIELECTRIC CONST.=" + f(data.eps_r, 7, 3),
            m + "CONDUCTIVITY=" + e(data.sigma, 10, 3) + " MHOS/METER",
            m + "COMPLEX DIELECTRIC CONSTANT=" + e(data.eps_c[0], 12, 5) + e(data.eps_c[1], 12, 5)};
}

// `canvas[end - len(text) : end] = list(text)` with Python's slice rules (a
// cell wider than its field shifts the rest of the row, as Python's does).
void place(std::string &canvas, const std::string &text, long long end) {
    const long long n = static_cast<long long>(canvas.size());
    long long a = end - static_cast<long long>(text.size()), b = end;
    if (a < 0) a = std::max(0LL, a + n);
    if (b < 0) b = std::max(0LL, b + n);
    a = std::min(a, n);
    b = std::min(b, n);
    if (b < a) b = a;
    canvas.replace(static_cast<std::size_t>(a), static_cast<std::size_t>(b - a), text);
}

// `_load_row` for the FIXED IMPEDANCE and SERIES / PARALLEL shapes.
std::string load_row(const LoadRowC &row) {
    if (row.kind == "SERIES (PER METER)" || row.kind == "PARALLEL (PER METER)") {
        std::string canvas(122, ' ');
        if (!row.tag.is_zero()) place(canvas, row.tag.str(), 8);
        place(canvas, row.node_from.str(), 13);
        place(canvas, row.node_thru.str(), 18);
        if (row.has_resistance) place(canvas, py_format_e(row.resistance, 11, 4), 34);
        if (row.has_inductance) place(canvas, py_format_e(row.inductance, 11, 4), 47);
        if (row.has_capacitance) place(canvas, py_format_e(row.capacitance, 11, 4), 60);
        place(canvas, center(row.kind, 20), 122);
        return canvas;
    }
    if (row.kind == "WIRE") {
        std::string canvas(110, ' ');
        if (!row.tag.is_zero()) place(canvas, row.tag.str(), 8);
        place(canvas, row.node_from.str(), 13);
        place(canvas, row.node_thru.str(), 18);
        place(canvas, py_format_e(row.conductivity, 11, 4), 99);
        place(canvas, row.kind, 108);
        return canvas;
    }
    if (row.kind == "SERIES" || row.kind == "PARALLEL") {
        std::string canvas(110, ' ');
        if (!row.tag.is_zero()) place(canvas, row.tag.str(), 8);
        place(canvas, row.node_from.str(), 13);
        place(canvas, row.node_thru.str(), 18);
        if (row.has_resistance) place(canvas, py_format_e(row.resistance, 11, 4), 34);
        if (row.has_inductance) place(canvas, py_format_e(row.inductance, 11, 4), 47);
        if (row.has_capacitance) place(canvas, py_format_e(row.capacitance, 11, 4), 60);
        place(canvas, center(row.kind, 8), 110);
        return canvas;
    }
    std::string canvas(118, ' ');
    place(canvas, row.tag.str(), 8);
    place(canvas, row.node_from.str(), 13);
    place(canvas, row.node_thru.str(), 18);
    place(canvas, py_format_e(row.resistance, 11, 4), 73);
    if (row.has_reactance) place(canvas, py_format_e(row.reactance, 11, 4), 86);
    std::string kind = row.kind;
    if (kind.size() < 16) kind += std::string(16 - kind.size(), ' ');  // ljust(16)
    place(canvas, kind, 118);
    return canvas;
}

Lines loading(const RunDataC &data, bool allocate = true) {
    Lines lines = {sp(31) + "- - - STRUCTURE IMPEDANCE LOADING - - -", ""};
    if (data.loads.empty()) {
        lines.push_back(sp(35) + "THIS STRUCTURE IS NOT LOADED");
    } else {
        lines.push_back("");
        lines.push_back("       LOCATION          RESISTANCE   INDUCTANCE  CAPACITANCE       "
                        "IMPEDANCE (OHMS)     CONDUCTIVITY    TYPE");
        lines.push_back("    ITAG FROM THRU          OHMS        HENRYS       FARADS        "
                        "REAL      IMAGINARY    MHOS/METER");
        lines.push_back("");
        for (const LoadRowC &row : data.loads) lines.push_back(load_row(row));
    }
    // a stepping FR allocates once: its first frequency prints the line
    if (allocate) lines.push_back("ALLOCATE CM:" + d(data.unknown_count * data.unknown_count, 15));
    return lines;
}

Lines timing(const RunDataC &data) {
    Lines lines = {sp(32) + "- - - MATRIX TIMING - - -", "",
                   sp(24) + "FILL=" + f(data.fill_seconds, 9, 3) + " SEC.,  FACTOR=" +
                       f(data.factor_seconds, 9, 3) + " SEC."};
    if (!data.phases) return lines;
    // MOMWIRE_TIMES: the run's phases, as the licensed engine's timed build
    // prints its own (`_printout._phase_lines`, the same text)
    auto ms = [](double seconds) { return std::to_string(std::llround(seconds * 1e3)); };
    lines.push_back(sp(24) + "GROUND TABLES=" + f(data.ground_seconds, 9, 3) + " SEC.  (BUILT " +
                    std::to_string(data.ground_built) + ", FROM DISK CACHE " +
                    std::to_string(data.ground_disk) + ", ANCHOR-BLENDED " +
                    std::to_string(data.ground_blended) + ")");
    lines.push_back(sp(24) + "FILL: " + ms(data.fill_seconds) + " msec (" +
                    std::to_string(data.fill_threads) + " threads) FACTOR: " +
                    ms(data.factor_seconds) + " msec (" + std::to_string(data.factor_threads) +
                    " threads)");
    lines.push_back(sp(24) + "SOLUTION (PORTS+CURRENTS+POWER)=" + f(data.solution_seconds, 9, 3) +
                    " SEC.");
    lines.push_back(sp(24) + "RADIATED FIELD (RP)=" + f(data.rp_seconds, 9, 3) + " SEC.");
    lines.push_back(sp(24) + "NEAR FIELD (NE/NH)=" + f(data.near_seconds, 9, 3) + " SEC.");
    return lines;
}

// MOMWIRE_TIMES only: the environment in effect, at the top of the printout
// (`_printout._times_environment`, the same text); nothing when it is off.
Lines times_environment(const std::string &variant) {
    if (!mw_times::active()) return {};
    auto var = [](const char *name) {
        const char *v = std::getenv(name);
        return std::string("  ") + name + " = " +
               (v == nullptr || v[0] == '\0' ? std::string("(NOT SET)") : std::string(v));
    };
    const mw_times::State &st = mw_times::state();
    std::filesystem::path dir;
    std::string tag;
    const std::string cache = somm_core::disk_location(dir, tag) ? dir.string() : "OFF";
    Lines lines = {sp(30) + "- - - MOMWIRE_TIMES ENVIRONMENT - - -",
                   "",
                   var("MOMWIRE_THREADS"),
                   var("OMP_NUM_THREADS"),
                   var("OPENBLAS_NUM_THREADS"),
                   "  THREADS = " + std::to_string(st.fill_threads) + " (FILL), " +
                       std::to_string(st.factor_threads) + " (FACTOR)",
                   "  VARIANT = " + variant,
                   "  MOMWIRE_BELOW_ANCHORS = " + somm_below::anchor_mode_from_env(),
                   "  GROUND TABLE CACHE = " + cache};
    blanks(lines, 3);
    return lines;
}

const Lines PORT_COLUMNS = {
    "   TAG   SEG.    VOLTAGE (VOLTS)         CURRENT (AMPS)         IMPEDANCE (OHMS)        "
    "ADMITTANCE (MHOS)      POWER",
    "   NO.   NO.    REAL        IMAG.       REAL        IMAG.       REAL        IMAG.       "
    "REAL        IMAG.     (WATTS)"};

std::string port_row(const PortRowC &r) {
    std::string out = d(r.tag, 4) + d(r.segment, 6) + d(static_cast<long long>(r.end_index), 2);
    for (double v : {r.voltage.real(), r.voltage.imag(), r.current.real(), r.current.imag(),
                     r.impedance.real(), r.impedance.imag(), r.admittance.real(),
                     r.admittance.imag(), r.power})
        out += e(v, 12, 4);
    return out;
}

std::string element_row(const ElementRowC &r, bool current) {
    std::string out = d(r.element, 7) + d(r.tag, 5);
    for (double v : r.center) out += e(v, 13, 5);
    out += e(r.length, 13, 5);
    if (current) {
        out += e(r.value.real(), 13, 5);
        out += e(r.value.imag(), 13, 5);
    }
    out += e(r.magnitude, 13, 5);
    out += f(r.phase_deg, 10, 4);
    return out;
}

// `_printout._pattern`: the heading (with the range form's lines), the rows
// through `_PATTERN_ROW`, and the 3-D trailer when it is carried.
Lines pattern_lines(const PatternBlockC &block) {
    const std::string heading = sp(48) + "- - - RADIATION PATTERNS - - -";
    const Lines columns = {
        std::string("  - - ANGLES - -") +
            (block.directive ? "       - DIRECTIVE GAINS -       "
                             : "          - POWER GAINS -        ") +
            "- - - POLARIZATION "
        "- - -    - - - E(THETA) - - -    - - - E(PHI) - - -",
        "  THETA     PHI        VERT.   HOR.    TOTAL      AXIAL     TILT   "
        "SENSE     MAGNITUDE    PHASE      MAGNITUDE    PHASE ",
        " DEGREES  DEGREES       DB      DB      DB        RATIO     DEG.       "
        "        VOLTS     DEGREES       VOLTS     DEGREES"};
    Lines lines;
    if (!block.has_range) {
        lines.push_back(heading);
        lines.push_back("");
        add(lines, columns);
    } else {
        const std::string ind = sp(54);
        lines.push_back(heading);
        lines.push_back(ind + "RANGE=" + e(block.range_m, 13, 6) + " METERS");
        lines.push_back(ind + "EXP(-JKR)/R=" + e(1.0 / block.range_m, 12, 5) + " AT PHASE" +
                        f(block.range_phase_deg, 7, 2) + " DEGREES");
        lines.push_back("");
        if (block.has_lower)
            lines.push_back(sp(39) + "LOWER MEDIUM - EXP(-JKR)/R=" +
                            e_fortran(block.lower_range_mag, 12, 5) + " AT PHASE" +
                            f(block.lower_range_phase_deg, 7, 2) + " DEGREES");
        lines.push_back("");
        lines.push_back(columns[0]);
        lines.push_back(columns[1]);
        lines.push_back(" DEGREES  DEGREES       DB      DB      DB        RATIO     DEG.       "
                        "       VOLTS/M    DEGREES      VOLTS/M    DEGREES");
    }
    for (const PatternRowC &r : block.rows)
        lines.push_back(f(r.theta, 8, 2) + f(r.phi, 9, 2) + f(r.vert, 11, 2) + f(r.hor, 8, 2) +
                        f(r.total, 8, 2) + f(r.axial, 11, 5) + f(r.tilt, 9, 2) + "  " +
                        ljust(r.sense, 6) + e(r.et_mag, 15, 5) + f(r.et_deg, 9, 2) +
                        e(r.ep_mag, 15, 5) + f(r.ep_deg, 9, 2));
    if (block.has_average) {
        add(lines, {"", "",
                    "   AVERAGE POWER GAIN=" + e(block.average_power_gain, 12, 5) +
                        "       SOLID ANGLE USED IN AVERAGING=(" + f(block.solid_angle_pi, 7, 4) +
                        ")*PI STERADIANS.",
                    "",
                    "   POWER RADIATED ASSUMING RADIATION INTO 4*PI STERADIANS =" +
                        e(block.power_radiated_4pi, 12, 5) + " WATTS"});
    }
    return lines;
}

// `_printout._near_field`: heading, three header lines, a row a point.
Lines near_lines(const NearBlockC &block) {
    const std::string kind = block.magnetic ? "MAGNETIC" : "ELECTRIC";
    const std::string c = block.magnetic ? "H" : "E";
    const std::string unit = block.magnetic ? "AMPS/M" : "VOLTS/M";
    const auto rjust = [](const std::string &t, std::size_t w) {
        return t.size() >= w ? t : std::string(w - t.size(), ' ') + t;
    };
    std::string units = "      METERS      METERS      METERS";
    for (std::size_t w : {16, 13, 13}) units += rjust(unit, w) + rjust("DEGREES", 10);
    Lines lines = {sp(38) + "- - - NEAR " + kind + " FIELDS - - -", "",
                   "              -  LOCATION  -                     -  " + c + "X  -"
                   "               -  " + c + "Y  -               -  " + c + "Z  -",
                   "        X           Y           Z           MAGNITUDE   PHASE"
                   "      MAGNITUDE   PHASE      MAGNITUDE   PHASE",
                   units};
    for (const NearRowC &r : block.rows) {
        std::string line = e(r.point[0], 14, 4) + e(r.point[1], 12, 4) + e(r.point[2], 12, 4);
        const int widths[3] = {15, 14, 14};
        for (int k = 0; k < 3; k++) line += e(r.mag[k], widths[k], 4) + f(r.deg[k], 9, 2);
        lines.push_back(line);
    }
    return lines;
}

// `_printout._network_data`: one heading, a sub-table per run of one row
// kind, a blank between runs; `_network_row`: I9 + 3 I6, a blank, six E14.4,
// and a TL's line type.
Lines network_data(const RunDataC &data) {
    const Lines nt_columns = {
        "      - FROM -    - TO -                          -  -  ADMITTANCE "
        "MATRIX ELEMENTS (MHOS)  -  -",
        "      TAG   SEG.   TAG  SEG.             (ONE,ONE)                   "
        "(ONE,TWO)                   (TWO,TWO)",
        "      NO.   NO.   NO.   NO.        REAL          IMAG.         REAL   "
        "       IMAG.         REAL          IMAG."};
    const Lines tl_columns = {
        "      - FROM -    - TO -           TRANSMISSION LINE               "
        "-  -  SHUNT ADMITTANCES (MHOS)  -  -              LINE",
        "      TAG  SEG.   TAG  SEG.      IMPEDANCE      LENGTH            "
        "- END ONE -                 - END TWO -            TYPE",
        "      NO.   NO.   NO.   NO.         OHMS        METERS         REAL   "
        "       IMAG.         REAL          IMAG."};
    const auto rjust = [](const std::string &t, std::size_t w) {
        return t.size() >= w ? t : std::string(w - t.size(), ' ') + t;
    };
    Lines lines = {sp(44) + "- - - NETWORK DATA - - -", ""};
    for (std::size_t i = 0; i < data.networks.size(); i++) {
        const NetworkRowC &r = data.networks[i];
        if (i == 0 || data.networks[i - 1].line != r.line) {
            if (i) lines.push_back("");
            add(lines, r.line ? tl_columns : nt_columns);
        }
        std::string line = rjust(r.tag_from.str(), 9) + rjust(std::to_string(r.segment_from), 6) +
                           rjust(r.tag_to.str(), 6) + rjust(std::to_string(r.segment_to), 6) + " ";
        for (double v : r.cells) line += e(v, 14, 4);
        if (r.line) line += std::string("   ") + (r.crossed ? "CROSSED" : "STRAIGHT");
        lines.push_back(line);
    }
    return lines;
}

Lines run_sections(const RunDataC &data, bool requests_only = false) {
    Lines body;
    if (requests_only) {
        // a run that changed no source: its near fields and patterns alone
        // (an RP after an XQ, momwire 0.79.2)
        for (const auto &near : data.near_fields) {
            add(body, near_lines(*near));
            blanks(body, 3);
        }
        for (const PatternBlockC &block : data.patterns) {
            add(body, pattern_lines(block));
            blanks(body, 5);
        }
        return body;
    }
    if (!data.connections.empty()) {
        body.push_back(sp(27) + "- - - STRUCTURE EXCITATION DATA AT NETWORK CONNECTION POINTS - - -");
        body.push_back("");
        add(body, PORT_COLUMNS);
        for (const PortRowC &r : data.connections) body.push_back(port_row(r));
        blanks(body, 3);
    }
    if (!data.sources.empty()) {
        body.push_back(sp(42) + "- - - ANTENNA INPUT PARAMETERS - - -");
        body.push_back("");
        add(body, PORT_COLUMNS);
        for (const PortRowC &r : data.sources) body.push_back(port_row(r));
        blanks(body, 3);
    }
    if (!data.currents.empty()) {
        add(body, {sp(39) + "- - - Wire Currents - - -", "",
                   sp(29) + "Lengths normalized by wavelength (or 2.*pi/CABS(k))", "",
                   "  Elem.  Tag            Coord. of seg. center          Seg.            "
                   "- - - Current (Amps) - - -",
                   "   No.   No.     X            Y            Z           Length       "
                   "Real         Imag.        Mag.         Phase"});
        for (const ElementRowC &r : data.currents) body.push_back(element_row(r, true));
        blanks(body, 3);
    }
    if (!data.charges.empty()) {
        add(body, {sp(29) + "- - - Wire Charge Densities - - -", "",
                   sp(21) + "Lengths normalized by wavelength (or 2.*pi/CABS(k))", "",
                   "  ELEM.  TAG          Coord. of Seg. Center           Seg.        "
                   "Charge Density (C/m)",
                   "   No.   No.     X            Y            Z          Length        "
                   "Mag.         Phase"});
        for (const ElementRowC &r : data.charges) body.push_back(element_row(r, false));
        blanks(body, 3);
    }
    const std::string ind = sp(43);
    auto watts = [&](const char *label, double v) {
        return ind + ljust(label, 14) + "=" + e(v, 11, 4) + " WATTS";
    };
    add(body, {sp(40) + "- - - POWER BUDGET - - -", "", watts("INPUT POWER", data.input_power),
               watts("RADIATED POWER", data.radiated_power), watts("WIRE LOSS", data.wire_loss)});
    if (data.has_network_loss) body.push_back(watts("NETWORK LOSS", data.network_loss));
    body.push_back(ind + ljust("EFFICIENCY", 14) + "=" + f(data.efficiency, 7, 2) + " PERCENT");
    blanks(body, 3);
    for (const auto &near : data.near_fields) {
        add(body, near_lines(*near));
        blanks(body, 3);
    }
    for (const PatternBlockC &block : data.patterns) {
        add(body, pattern_lines(block));
        blanks(body, 5);
    }
    return body;
}

Lines head_sections(const Deck &deck, const RunDataC &first) {
    Lines body = structure_specification(deck, first);
    blanks(body, 4);
    return body;
}

void tail_sections(Lines &body, const RunDataC &first) {
    blanks(body, 4);
    add(body, frequency(first));
    blanks(body, 3);
    add(body, environment(first));
    blanks(body, 3);
    add(body, loading(first));
    blanks(body, 3);
    add(body, timing(first));
    blanks(body, 3);
    if (!first.networks.empty()) {
        add(body, network_data(first));
        blanks(body, 3);
    }
}

// RUN TIME: the seconds since the process started when MOMWIRE_TIMES is on,
// else 0 (`mw_times::run_seconds`).
Lines closing_lines(bool closing) {
    if (!closing) return {};
    if (mw_times::active())
        return {"", " RUN TIME =" + f(mw_times::run_seconds(), 10, 3) + " SEC. (WALL CLOCK)"};
    return {"", " RUN TIME =" + f(0.0, 10, 3)};
}

// `render_printout`
std::string render_printout(const Deck &deck, const std::string &echo_text, const RunDataC &data,
                            const std::string &basis, const EngineStamp &stamp, bool closing) {
    const std::vector<Card> cards = post_ge_cards(echo_text);
    const bool has_term = !cards.empty() && is_terminator(cards.back());
    Lines body = times_environment(stamp.variant);
    add(body, head_sections(deck, data));
    for (std::size_t k = 0; k < cards.size(); k++) {
        if (has_term && k + 1 == cards.size()) continue;
        body.push_back(card_echo(static_cast<long long>(k) + 1, cards[k]));
    }
    tail_sections(body, data);
    add(body, run_sections(data));
    body.push_back("");
    if (has_term) body.push_back(card_echo(static_cast<long long>(cards.size()), cards.back()));
    add(body, closing_lines(closing));
    return render_header(&echo_text, basis, stamp) + join(body) + "\n";
}

// `render_stepped_printout` (momwire#1508): the structure, the echoes through
// the card that runs the sweep, a whole single-run body per frequency
// (ALLOCATE CM: in the first only; one blank more between blocks), the
// echoes after the sweep, EN and one RUN TIME.
std::string render_stepped_printout(const Deck &deck, const std::string &echo_text,
                                    const std::vector<RunDataC> &runs, const std::string &basis,
                                    const EngineStamp &stamp, bool closing) {
    const std::vector<Card> cards = post_ge_cards(echo_text);
    const bool has_term = !cards.empty() && is_terminator(cards.back());
    std::vector<std::pair<long long, const Card *>> numbered;
    for (std::size_t k = 0; k < cards.size(); k++) {
        if (has_term && k + 1 == cards.size()) continue;
        numbered.emplace_back(static_cast<long long>(k) + 1, &cards[k]);
    }
    std::size_t run_at = numbered.size();
    for (std::size_t k = 0; k < numbered.size(); k++)
        if (numbered[k].second->mnemonic == "RP" || numbered[k].second->mnemonic == "XQ") {
            run_at = k;
            break;
        }
    if (run_at == numbered.size() || runs.empty())
        throw PyRaise{"StopIteration", ""};  // `next(...)` with nothing to find
    Lines body = times_environment(stamp.variant);
    add(body, head_sections(deck, runs[0]));
    for (std::size_t k = 0; k <= run_at; k++)
        body.push_back(card_echo(numbered[k].first, *numbered[k].second));
    blanks(body, 4);
    for (std::size_t k = 0; k < runs.size(); k++) {
        const RunDataC &data = runs[k];
        if (k) body.push_back("");
        add(body, frequency(data));
        blanks(body, 3);
        add(body, environment(data));
        blanks(body, 3);
        add(body, loading(data, k == 0));
        blanks(body, 3);
        add(body, timing(data));
        blanks(body, 3);
        if (!data.networks.empty()) {
            add(body, network_data(data));
            blanks(body, 3);
        }
        add(body, run_sections(data));
    }
    body.push_back("");
    for (std::size_t k = run_at + 1; k < numbered.size(); k++)
        body.push_back(card_echo(numbered[k].first, *numbered[k].second));
    if (has_term) body.push_back(card_echo(static_cast<long long>(cards.size()), cards.back()));
    add(body, closing_lines(closing));
    return render_header(&echo_text, basis, stamp) + join(body) + "\n";
}

// `render_multi_printout`
std::string render_multi_printout(const Deck &deck, const std::string &echo_text,
                                  const std::vector<RunDataC> &runs, const std::string &basis,
                                  const EngineStamp &stamp, bool closing) {
    const std::vector<Card> cards = post_ge_cards(echo_text);
    const bool has_term = !cards.empty() && is_terminator(cards.back());
    std::vector<std::pair<long long, const Card *>> numbered;
    for (std::size_t k = 0; k < cards.size(); k++) {
        if (has_term && k + 1 == cards.size()) continue;
        numbered.emplace_back(static_cast<long long>(k) + 1, &cards[k]);
    }
    std::vector<std::vector<std::pair<long long, const Card *>>> blocks;
    std::vector<std::pair<long long, const Card *>> pending;
    bool changed = true;
    for (const auto &item : numbered) {
        pending.push_back(item);
        const std::string &mn = item.second->mnemonic;
        if (mn == "RP" || mn == "XQ") {
            if (changed) {
                blocks.push_back(pending);
                pending.clear();
                changed = false;
            }
        } else {
            changed = true;
        }
    }
    if (blocks.size() != runs.size())
        throw PyRaise{"ValueError", std::to_string(blocks.size()) + " executed runs and " +
                                        std::to_string(runs.size()) + " answers"};
    const RunDataC &first = runs[0];
    Lines body = times_environment(stamp.variant);
    add(body, head_sections(deck, first));
    for (const auto &item : blocks[0]) body.push_back(card_echo(item.first, *item.second));
    tail_sections(body, first);
    add(body, run_sections(first));
    for (std::size_t k = 1; k < blocks.size(); k++) {
        body.push_back("");
        for (const auto &item : blocks[k]) body.push_back(card_echo(item.first, *item.second));
        blanks(body, 3);
        add(body, run_sections(runs[k]));
    }
    body.push_back("");
    for (const auto &item : pending) body.push_back(card_echo(item.first, *item.second));
    if (has_term) body.push_back(card_echo(static_cast<long long>(cards.size()), cards.back()));
    add(body, closing_lines(closing));
    return render_header(&echo_text, basis, stamp) + join(body) + "\n";
}

// `render_sequence_printout` (momwire 0.79.2; momwire#1602): the deck read
// as a program. The structure, then per run its card echoes and either a
// whole block per frequency (the first run, or one carrying an FR) or its
// results only at the frequency in force (a run that changes no source:
// its near fields and patterns only), then the echoes after the last run,
// EN and one RUN TIME. `runs` are every run's answers in order, `counts`
// how many each run has.
std::string render_sequence_printout(const Deck &deck, const std::string &echo_text,
                                     const std::vector<SequenceGroup> &groups,
                                     const std::vector<std::size_t> &tail,
                                     const std::vector<RunDataC> &runs,
                                     const std::vector<std::size_t> &counts,
                                     const std::string &basis, const EngineStamp &stamp,
                                     bool closing) {
    const std::vector<Card> cards = post_ge_cards(echo_text);
    const bool has_term = !cards.empty() && is_terminator(cards.back());
    if (runs.empty()) throw PyRaise{"IndexError", "list index out of range"};
    auto echo = [&](std::size_t i) {
        if (i >= cards.size()) throw PyRaise{"IndexError", "list index out of range"};
        return card_echo(static_cast<long long>(i) + 1, cards[i]);
    };
    Lines body = times_environment(stamp.variant);
    add(body, head_sections(deck, runs[0]));
    bool allocate = true;
    std::size_t at = 0;
    for (std::size_t g = 0; g < groups.size(); g++) {
        const SequenceGroup &run = groups[g];
        if (g) body.push_back("");
        for (std::size_t i : run.cards) body.push_back(echo(i));
        const bool requests_only = !run.full && !run.sources;
        if (run.full) {
            blanks(body, 4);
            for (std::size_t k = 0; k < counts[g]; k++) {
                const RunDataC &data = runs[at + k];
                if (k) body.push_back("");
                add(body, frequency(data));
                blanks(body, 3);
                add(body, environment(data));
                blanks(body, 3);
                add(body, loading(data, allocate));
                allocate = false;
                blanks(body, 3);
                add(body, timing(data));
                blanks(body, 3);
                if (!data.networks.empty()) {
                    add(body, network_data(data));
                    blanks(body, 3);
                }
                add(body, run_sections(data));
            }
        } else {
            blanks(body, 3);
            for (std::size_t k = 0; k < counts[g]; k++)
                add(body, run_sections(runs[at + k], requests_only));
        }
        at += counts[g];
    }
    body.push_back("");
    for (std::size_t i : tail) body.push_back(echo(i));
    if (has_term) body.push_back(card_echo(static_cast<long long>(cards.size()), cards.back()));
    add(body, closing_lines(closing));
    return render_header(&echo_text, basis, stamp) + join(body) + "\n";
}

// ============================================================================
// `_shell.write_printout`: CRLF, and latin-1 with `transliterate`
// ============================================================================

std::vector<std::uint32_t> code_points(const std::string &utf8) {
    std::vector<std::uint32_t> out;
    for (std::size_t k = 0; k < utf8.size();) {
        const unsigned char c = static_cast<unsigned char>(utf8[k]);
        std::uint32_t cp;
        int extra;
        if (c < 0x80) {
            cp = c;
            extra = 0;
        } else if ((c >> 5) == 6) {
            cp = c & 0x1f;
            extra = 1;
        } else if ((c >> 4) == 14) {
            cp = c & 0x0f;
            extra = 2;
        } else {
            cp = c & 0x07;
            extra = 3;
        }
        k++;
        for (int j = 0; j < extra && k < utf8.size(); j++, k++)
            cp = (cp << 6) | (static_cast<unsigned char>(utf8[k]) & 0x3f);
        out.push_back(cp);
    }
    return out;
}

bool unicode_isspace(std::uint32_t c) {
    if (c < 0x100) return latin1_isspace(static_cast<unsigned char>(c));
    return c == 0x1680 || (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 ||
           c == 0x202f || c == 0x205f || c == 0x3000;
}

// `_shell._TRANSLITERATIONS`, then what `unicodedata.normalize("NFKD", c)`
// keeps inside latin-1 for every character outside it that momwire's own
// sentences carry (and the two `str.upper` takes a latin-1 letter to);
// `tests/test_razor_core_standalone.py` pins each against Python's
// `_spell`. Anything else is "?".
const char *spelling(std::uint32_t c) {
    static const std::map<std::uint32_t, const char *> table = {
        {0x2014, "-"},     {0x2013, "-"},     {0x2212, "-"},   {0x2010, "-"},
        {0x2011, "-"},     {0x2018, "'"},     {0x2019, "'"},   {0x201c, "\""},
        {0x201d, "\""},    {0x2026, "..."},   {0x2192, "->"},  {0x2190, "<-"},
        {0x2264, "<="},    {0x2265, ">="},    {0x2260, "!="},  {0x2248, "~"},
        {0x2261, "=="},    {0x221a, "sqrt"},  {0x221e, "inf"}, {0x03a9, "ohm"},
        {0x03bb, "lambda"}, {0x03bc, "u"},    {0x03b5, "eps"}, {0x03c3, "sigma"},
        {0x03c1, "rho"},   {0x03c0, "pi"},    {0x03c9, "omega"},
        // NFKD, kept inside latin-1
        {0x0109, "c"},     {0x0125, "h"},     {0x015d, "s"},   {0x0177, "y"},
        {0x0178, "Y"},     {0x1d38, "L"},     {0x1d40, "T"},   {0x1d62, "i"},
        {0x1d63, "r"},     {0x1e41, "m"},     {0x1e45, "n"},   {0x1e7c, "V"},
        {0x1e91, "z"},     {0x1ebc, "E"},     {0x2074, "4"},   {0x2075, "5"},
        {0x2078, "8"},     {0x207a, "+"},     {0x207d, "("},   {0x207e, ")"},
        {0x207f, "n"},     {0x2080, "0"},     {0x2081, "1"},   {0x2082, "2"},
        {0x2083, "3"},     {0x2084, "4"},     {0x2086, "6"},   {0x208a, "+"},
        {0x2099, "n"},     {0x2113, "l"},
    };
    auto it = table.find(c);
    return it == table.end() ? nullptr : it->second;
}

}  // namespace

std::string printout_bytes(const std::string &utf8_text) {
    std::string crlf;
    crlf.reserve(utf8_text.size() + utf8_text.size() / 40);
    for (char c : utf8_text) {
        if (c == '\n') crlf += '\r';
        crlf += c;
    }
    const std::vector<std::uint32_t> cps = code_points(crlf);
    std::string out;
    out.reserve(cps.size());
    for (std::size_t i = 0; i < cps.size(); i++) {
        const std::uint32_t c = cps[i];
        if (c < 0x100) {
            out += static_cast<char>(c);
            continue;
        }
        const char *s = spelling(c);
        if (s == nullptr) {
            out += '?';
            continue;
        }
        if (c == 0x2014) {
            const bool before = i > 0 ? unicode_isspace(cps[i - 1]) : true;
            const bool after = i + 1 < cps.size() ? unicode_isspace(cps[i + 1]) : true;
            out += std::string(before ? "" : " ") + s + (after ? "" : " ");
        } else {
            out += s;
        }
    }
    return out;
}

std::string engine_stamp(const std::string &basis, const EngineStamp &stamp) {
    std::string version = stamp_field(stamp.version);
    if (version.empty()) version = "unknown";
    std::string variant = stamp_field(stamp.variant);
    if (variant.empty()) variant = "none";
    const std::string tail = stamp.marker.empty() ? std::string() : " " + stamp.marker;
    const long long room = 79LL - 9 - static_cast<long long>(version.size()) -
                           static_cast<long long>(variant.size()) -
                           static_cast<long long>(tail.size()) - 2;
    std::string name = stamp_field(basis).substr(0, static_cast<std::size_t>(std::max(room, 0LL)));
    if (name.empty()) name = "-";
    const std::string line = " momwire " + version + " " + name + " " + variant + tail;
    return line.substr(0, 79);
}

std::string render_refusal(const std::string *latin1_deck_text, const std::string &utf8_reason,
                           const std::string &basis, const EngineStamp &stamp) {
    return render_header(latin1_deck_text, basis, stamp) + "\n" + " ***** NEC ERROR - " +
           utf8_reason + "\n";
}

namespace {

std::string echo_of(const StreamRead &s, std::size_t k, const std::string &text) {
    return s.chained ? s.structures[k].echo_text() : text;
}

// Every run the seam serves prints: LD 0 / 1 / 4, RP, NE / NH.
void check_scope(const Deck &run) { (void)run; }

}  // namespace

PrintoutResult printout(const std::string &text, const std::string &basis,
                        const EngineStamp &stamp, PrintoutEngine &engine, const NumpyLinalg &np,
                        std::vector<RunArrays> *arrays, bool open_end) {
    PrintoutResult out;
    SeamResult r = [&] {
        const mw_times::Scope timed(mw_times::seam);
        return seam(text, engine.seam(), open_end);
    }();
    out.seam = r.route;
    if (r.route == SeamResult::declined) {
        out.reason = r.message;
        return out;
    }
    const StreamRead &s = r.stream;
    // The scope check first, over every run Python finishes, in its order:
    // a decline must never hide an answer Python would print.
    try {
        for (std::size_t k = 0; k < r.structures.size(); k++) {
            const StructureOutcome &so = r.structures[k];
            const Answer &a = s.answers[k];
            for (std::size_t j = 0; j < so.runs.size(); j++) {
                const RunOutcome &o = so.runs[j];
                if (o.kind != RunOutcome::prepared && o.kind != RunOutcome::shared) break;
                check_scope(a.kind == Answer::multi ? a.runs[j] : a.deck);
            }
        }
    } catch (const Decline &dcl) {
        out.reason = dcl.reason;
        return out;
    }

    std::string utf8;
    try {
        std::vector<Solved> solved(r.prepared.size());
        std::vector<bool> have(r.prepared.size(), false);
        // MOMWIRE_TIMES: each solve's fill and factor seconds, printed in the
        // MATRIX TIMING of the run that paid for them (a run sharing an
        // earlier run's solve paid nothing)
        std::vector<double> fill_s(r.prepared.size(), 0.0), factor_s(r.prepared.size(), 0.0);
        auto solution = [&](long long index) -> const Solved & {
            const std::size_t i = static_cast<std::size_t>(index);
            if (!have[i]) {
                const double f0 = mw_times::ms_of(mw_times::fill);
                const double k0 = mw_times::ms_of(mw_times::factor);
                {
                    const mw_times::Scope timed(mw_times::fill);
                    solved[i] = engine.solve(r.prepared[i], np);
                }
                fill_s[i] = (mw_times::ms_of(mw_times::fill) - f0) / 1000.0;
                factor_s[i] = (mw_times::ms_of(mw_times::factor) - k0) / 1000.0;
                have[i] = true;
            }
            return solved[i];
        };
        auto timed_finish = [&](const Deck &run, Prepared &p, const Solved &sol, std::size_t i) {
            RunDataC data;
            const bool timing = mw_times::active();
            mw_times::Snap before;
            if (timing) before = mw_times::snap();
            {
                const mw_times::Scope timed(mw_times::run);
                data = finish(run, p, sol, np, arrays, engine);
            }
            if (timing) {
                // the ground tables counted since the last run (its seam's
                // and its solve's included), and this run's readouts
                const mw_times::Snap now = mw_times::snap();
                mw_times::State &st = mw_times::state();
                const mw_times::Snap &mark = st.mark;
                data.phases = true;
                data.ground_seconds = (now.ms[mw_times::ground] - mark.ms[mw_times::ground]) / 1e3;
                data.ground_built = now.built - mark.built;
                data.ground_disk = now.disk - mark.disk;
                data.ground_blended = now.blended - mark.blended;
                data.solution_seconds = (now.ms[mw_times::run] - before.ms[mw_times::run]) / 1e3;
                data.rp_seconds = (now.ms[mw_times::far_field] - before.ms[mw_times::far_field]) / 1e3;
                data.near_seconds =
                    (now.ms[mw_times::near_field] - before.ms[mw_times::near_field]) / 1e3;
                data.fill_threads = st.fill_threads;
                data.factor_threads = st.factor_threads;
                st.mark = now;
            }
            mw_times::count_run();
            data.fill_seconds = fill_s[i];
            data.factor_seconds = factor_s[i];
            fill_s[i] = factor_s[i] = 0.0;
            return data;
        };
        const std::size_t n_structures = s.chained ? s.structures.size() : 1;
        std::string printed;
        bool stopped = false;
        for (std::size_t k = 0; k < r.structures.size() && !stopped; k++) {
            const StructureOutcome &so = r.structures[k];
            const Answer &a = s.answers[k];
            const std::string echo = echo_of(s, k, text);
            // a structure NX closes never closes the run (`_render_chain`)
            const bool last = !s.chained ||
                              (k + 1 == n_structures && !s.structures[k].ends_on_nx());
            std::vector<RunDataC> runs;
            for (std::size_t j = 0; j < so.runs.size() && !stopped; j++) {
                const RunOutcome &o = so.runs[j];
                if (o.kind == RunOutcome::refused) {
                    printed += render_refusal(&echo, o.message, basis, stamp);
                    stopped = true;
                } else if (o.kind == RunOutcome::raised) {
                    throw PyRaise{o.error_kind, o.message};
                } else {
                    Prepared &p = r.prepared[static_cast<std::size_t>(o.prepared_index)];
                    const Solved &sol = solution(o.prepared_index);
                    if (a.kind == Answer::sequence) {
                        // `replace(deck_g, frequency_mhz=f)`, finished in
                        // turn; its front released as the stepped route's
                        const auto &step_at = so.steps[j];
                        Deck step = a.groups[step_at.first].parsed.deck;
                        step.frequency_mhz = step_at.second;
                        runs.push_back(
                            timed_finish(step, p, sol, static_cast<std::size_t>(o.prepared_index)));
                        engine.release(p);
                        solved[static_cast<std::size_t>(o.prepared_index)] = Solved();
                    } else if (a.kind == Answer::stepped) {
                        // `replace(deck, frequency_mhz=f)`, finished before
                        // the next frequency is solved; its front released
                        Deck step = a.deck;
                        step.frequency_mhz = a.deck.frequencies_mhz[j];
                        runs.push_back(
                            timed_finish(step, p, sol, static_cast<std::size_t>(o.prepared_index)));
                        engine.release(p);
                        solved[static_cast<std::size_t>(o.prepared_index)] = Solved();
                    } else {
                        runs.push_back(timed_finish(a.kind == Answer::multi ? a.runs[j] : a.deck, p,
                                                    sol, static_cast<std::size_t>(o.prepared_index)));
                    }
                }
            }
            if (stopped) break;
            if (so.runs.empty() || so.runs.back().kind == RunOutcome::declined)
                throw std::logic_error("a declined run reached the writer");
            const mw_times::Scope timed(mw_times::render);
            if (a.kind == Answer::multi)
                printed += render_multi_printout(a.deck, echo, runs, basis, stamp, last);
            else if (a.kind == Answer::stepped)
                printed += render_stepped_printout(a.deck, echo, runs, basis, stamp, last);
            else if (a.kind == Answer::sequence) {
                std::vector<std::size_t> counts(a.groups.size(), 0);
                for (const auto &step_at : so.steps) counts[step_at.first]++;
                printed += render_sequence_printout(a.deck, echo, a.groups, a.sequence_tail,
                                                    runs, counts, basis, stamp, last);
            }
            else
                printed += render_printout(a.deck, echo, runs[0], basis, stamp, last);
        }
        utf8 = printed;
    } catch (const PyRaise &raise) {
        out.raised_kind = raise.kind;
        utf8 = render_refusal(&text, internal_error_reason(raise.kind, raise.message), basis, stamp);
    } catch (const ParseError &pe) {
        out.raised_kind = error_kind_name(pe.kind);
        utf8 = render_refusal(&text, internal_error_reason(out.raised_kind, pe.what()), basis, stamp);
    } catch (const Decline &dcl) {
        out.raised_kind.clear();
        out.reason = dcl.reason;
        return out;
    } catch (const std::bad_alloc &ex) {
        out.raised_kind.clear();
        out.reason = std::string("C++ printout failed: ") + ex.what();
        out.cxx_failed = true;
        out.failure_kind = "MemoryError";
        out.failure_message = ex.what();
        return out;
    } catch (const std::exception &ex) {
        // Nothing Python prints (memory, a layout this file does not
        // replay): never a printout of our own; the caller falls back.
        out.raised_kind.clear();
        out.reason = std::string("C++ printout failed: ") + ex.what();
        out.cxx_failed = true;
        out.failure_kind = "RuntimeError";
        out.failure_message = ex.what();
        return out;
    }
    out.route = PrintoutResult::printed;
    const mw_times::Scope timed(mw_times::render);
    out.printout = printout_bytes(utf8);
    return out;
}

// The replayed arithmetic, for an engine's readouts: this file's own.
namespace replay {
cd np_cmul(cd a, cd b) { return nec5::np_cmul(a, b); }
cd np_cdiv(cd a, cd b) { return nec5::np_cdiv(a, b); }
double np_pairwise(const double *a, int64_t n) { return nec5::np_pairwise(a, n); }
cd np_csum(const std::vector<cd> &a) { return nec5::np_csum(a); }
void np_cpairwise(const cd *a, int64_t n, double &rr, double &ri) { nec5::np_cpairwise(a, n, rr, ri); }
double np_interp(double x, const double *xp, const double *fp, int64_t n) {
    return nec5::np_interp(x, xp, fp, n);
}
void linspace3(const double *a, const double *b, int64_t num, std::vector<double> &out) {
    nec5::linspace3(a, b, num, out);
}
void np_matmul(const NumpyLinalg &np, const Op &a, const Op &b, cd *out, int64_t os_m,
               int64_t os_p) {
    nec5::np_matmul(np, a, b, out, os_m, os_p);
}
PatternBlock far_pattern(const Request &rq, const std::vector<double> &mid,
                         const std::vector<cd> &moment, int ground, double eps_r, double sigma,
                         double frequency_mhz, double wavelength, double p_in, double p_radiated) {
    const FarGround fg = ground == 1 ? FarGround::pec : ground == 2 ? FarGround::refl : FarGround::free;
    const PatternBlockC c = nec5::pattern_block(rq, mid, moment, fg, eps_r, sigma,
                                                frequency_mhz, wavelength, p_in, false,
                                                cd(0.0, 0.0), p_radiated);
    PatternBlock out;
    for (const PatternRowC &r : c.rows) {
        PatternRow row;
        row.theta = r.theta;
        row.phi = r.phi;
        row.vert = r.vert;
        row.hor = r.hor;
        row.total = r.total;
        row.axial = r.axial;
        row.tilt = r.tilt;
        row.sense = r.sense;
        row.et_mag = r.et_mag;
        row.et_deg = r.et_deg;
        row.ep_mag = r.ep_mag;
        row.ep_deg = r.ep_deg;
        out.rows.push_back(row);
    }
    out.has_average = c.has_average;
    out.average_power_gain = c.average_power_gain;
    out.solid_angle_pi = c.solid_angle_pi;
    out.power_radiated_4pi = c.power_radiated_4pi;
    out.directive = c.directive;
    return out;
}
NetState composed(const std::vector<SeamCard> &cards, const CMat &y, const std::vector<cd> &z_load,
                  const std::vector<cd> &voltages, const std::vector<int64_t> &held,
                  const std::vector<int64_t> &driven, double wavelength, const NumpyLinalg &np) {
    const int64_t n = y.rows;
    bool any = false;  // `np.any(z_load)`
    for (const cd &z : z_load) any = any || py_ctruth(z) || std::isnan(z.real()) || std::isnan(z.imag());
    // `y_eff = solve((eye + z_load[:, None] * y).T, y.T).T`: the transposed
    // view of the C-ordered solution, read through its strides
    CMat xt;
    Op yeff_op = y.op();
    if (any) {
        CMat lt(n, n), yt(n, n);
        for (int64_t i = 0; i < n; i++)
            for (int64_t j = 0; j < n; j++) {
                const cd prod = nec5::np_cmul(z_load[static_cast<std::size_t>(i)], y.at(i, j));
                lt.at(j, i) = cd((i == j ? 1.0 : 0.0) + prod.real(), 0.0 + prod.imag());
                yt.at(i, j) = y.at(j, i);
            }
        xt = nec5::np_solve_matrix(np, lt, yt);
        yeff_op = Op{xt.v.data(), n, n, 1, n};
    }
    const std::function<cd(int64_t, int64_t)> y_eff_at = [&](int64_t i, int64_t j) {
        return any ? xt.at(j, i) : y.at(i, j);
    };
    const NetSolve r = nec5::reduced_state(cards, n, voltages, held, driven, yeff_op, y_eff_at, wavelength, np);
    NetState out;
    out.v_applied = r.v_applied;
    out.i_port = r.i_port;
    out.i_source = r.i_source;
    return out;
}
std::vector<double> near_grid(const Request &rq) { return nec5::near_points(rq); }
ElementCurrents element_currents_sub(const std::vector<WireWalk> &walk, const CurrentSampler &sampler,
                                     int64_t subdiv) {
    return nec5::element_currents_sub(walk, sampler, subdiv);
}
NearBlock near_field(const Request &rq, const std::vector<double> &points, const ElementCurrents &el,
                     double k, double radius, int ground, double eps_r, double sigma, double omega,
                     double eps0) {
    const NearBlockC c =
        nec5::near_block_of(rq, points, el, k, radius, ground == 1, ground == 2, omega, eps0, eps_r, sigma);
    NearBlock out;
    out.magnetic = c.magnetic;
    for (const NearRowC &r : c.rows) {
        NearRow row;
        for (int i = 0; i < 3; i++) {
            row.point[i] = r.point[i];
            row.mag[i] = r.mag[i];
            row.deg[i] = r.deg[i];
        }
        out.rows.push_back(row);
    }
    return out;
}
}  // namespace replay

}  // namespace nec5
}  // namespace mw
