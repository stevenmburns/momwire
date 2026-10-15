// The NEC-5 seam's run half and printout for razor-2p -- see
// `_nec5_razor_printout_core.h`. Each function names the Python it is a copy
// of (`eznec/_serve.py`, `eznec/_printout.py`, `eznec/_shell.py`, razor.py's
// readouts) or the numpy / CPython routine whose arithmetic it replays; the
// text it writes is that Python's, byte for byte, and so is the order in
// which it computes (an exception escapes where Python's does).
#include "_nec5_razor_printout_core.h"

// MSVC builds every TU /fp:fast (setup.py); this one replays numpy's and
// CPython's arithmetic one IEEE operation at a time, so it asks for precise
// semantics. GCC / clang build it with -ffp-contract=off.
#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>

namespace mw {
namespace nec5 {

namespace {

using razor::End;

// An exception Python lets escape: `_shell.main`'s last line of defence
// reports it as "INTERNAL ERROR IN MOMWIRE ENGINE - <kind>: <message>".
struct PyRaise {
    std::string kind, message;
};

// A run this unit does not print yet.
struct Decline {
    std::string reason;
};

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

// ============================================================================
// CPython 3.12's arithmetic
// ============================================================================

// `_Py_c_prod`
cd py_cmul(cd a, cd b) {
    return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}

// `_Py_c_quot` (3.12: Smith's method, no Annex G recovery). A zero divisor
// is Python's ZeroDivisionError, whose text is the interpreter's: it is
// DECLINED here rather than spelled in C++ (the one caller that can reach
// it is an EX 4 whose port carries no current), and Python answers it.
cd py_cdiv(cd a, cd b) {
    const double abs_breal = b.real() < 0 ? -b.real() : b.real();
    const double abs_bimag = b.imag() < 0 ? -b.imag() : b.imag();
    if (abs_breal >= abs_bimag) {
        if (abs_breal == 0.0) throw Decline{"an EX 4 port with no current (Python raises)"};
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
    return cd(std::fma(a.real(), b.real(), -(a.imag() * b.imag())),
              std::fma(a.real(), b.imag(), a.imag() * b.real()));
}

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

// A 2-D operand: its shape and its strides in ELEMENTS (numpy's are bytes;
// every operand here is complex128, so byte strides are 16x these). A 1-D
// vector is (n, 1) with a zero second stride, as the gufunc hands it.
struct Op {
    const cd *p;
    int64_t d0, d1, s0, s1;
};

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

// A C-ordered complex matrix (numpy's default layout for a result).
struct CMat {
    int64_t rows = 0, cols = 0;
    std::vector<cd> v;
    CMat() = default;
    CMat(int64_t r, int64_t c) : rows(r), cols(c), v(static_cast<std::size_t>(r * c)) {}
    cd &at(int64_t i, int64_t j) { return v[static_cast<std::size_t>(i * cols + j)]; }
    cd at(int64_t i, int64_t j) const { return v[static_cast<std::size_t>(i * cols + j)]; }
    Op op() const { return Op{v.data(), rows, cols, cols, 1}; }
};

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
    if (info > 0) throw Decline{"a singular drive block (numpy raises)"};
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

struct RunDataC {  // `RunData`, for the decks this unit prints
    long long node_count = 0, wire_element_count = 0, unknown_count = 0;
    double frequency_mhz = 0, wavelength_m = 0;
    enum Env { free_space, perfect_ground, finite_ground } environment = free_space;
    double eps_r = 0, sigma = 0, eps_c[2] = {0, 0};
    std::vector<PortRowC> sources;
    std::vector<ElementRowC> currents, charges;
    double input_power = 0, radiated_power = 0, wire_loss = 0, efficiency = 0;
};

// A preparation's solve: `PortSolution.coeffs` (column-major n_basis x
// n_ports) and `.y` (C-ordered n_ports x n_ports).
struct Solved {
    int64_t n_basis = 0, n_ports = 0;
    std::vector<cd> x;
    CMat y;
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
    std::vector<cd> v_applied, v_gap, i_port, i_source;
};

// `_port_state`, network-free and load-free (the decks this unit prints):
// `y = t.T @ y_solver @ t`, then one scaled unit probe, or the constrained
// multi-source drive (`_multi_drive_state`).
PortState port_state(const Deck &deck, const Prepared &p, const Solved &s, const NumpyLinalg &np) {
    const int64_t nc = p.mesh.n_columns(), n = static_cast<int64_t>(p.mesh.sites.size());
    const std::vector<cd> tc = transform_c(p);
    // `t.T @ y_solver`: t.T is the cast copy in its own (Fortran) order
    const CMat r1 = mm(np, Op{tc.data(), n, nc, 1, n}, s.y.op());
    const CMat y = mm(np, r1.op(), Op{tc.data(), nc, n, n, 1});
    const std::vector<cd> z_series(static_cast<std::size_t>(n), cd(0.0, 0.0));
    PortState st;
    if (deck.sources.size() > 1) {
        // `_multi_drive_state` with no cards and no loads: y_eff = y.
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
            for (int64_t i = 0; i < nd; i++)
                for (int64_t j = 0; j < nd; j++) m.at(i, j) = y.at(driven[i], driven[j]);
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
        st.v_applied = volts;
        st.i_port = mv(np, y.op(), st.v_applied);
        st.i_source = st.i_port;
        for (std::size_t k = 0; k < i_sites.size(); k++)
            st.i_source[static_cast<std::size_t>(i_sites[k])] = spec[static_cast<std::size_t>(i_rows[k])];
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
    std::vector<cd> v_gap = np_solve(np, loaded, probe);
    std::vector<cd> i_port = mv(np, y.op(), v_gap);
    std::vector<cd> v_applied(static_cast<std::size_t>(n));
    for (int64_t k = 0; k < n; k++) v_applied[k] = v_gap[k] + np_cmul(z_series[k], i_port[k]);
    const cd drive(source.drive_re, source.drive_im);
    const cd scale = source.kind.is_zero()
                         ? drive
                         : py_cdiv(drive, i_port[static_cast<std::size_t>(driven)]);
    for (int64_t k = 0; k < n; k++) {
        v_applied[k] = np_cmul(v_applied[k], scale);
        v_gap[k] = np_cmul(v_gap[k], scale);
        i_port[k] = np_cmul(i_port[k], scale);
    }
    st.v_applied = v_applied;
    st.v_gap = v_gap;
    st.i_source = i_port;
    st.i_port = i_port;
    return st;
}

// razor's `currents_at_knots` for one wire (`_end_wings` included).
std::vector<std::vector<cd>> knot_currents(const razor::Front &fr, const std::vector<cd> &coeffs) {
    std::vector<std::vector<cd>> out;
    for (int64_t w = 0; w < fr.n_wires; w++) {
        const int64_t n_knots = fr.seg_offsets[w + 1] - fr.seg_offsets[w] + 1;
        std::vector<cd> I(static_cast<std::size_t>(n_knots), cd(0.0, 0.0));
        for (int64_t b = fr.basis_offsets[w]; b < fr.basis_offsets[w + 1]; b++)
            I[static_cast<std::size_t>(b - fr.basis_offsets[w] + 1)] = coeffs[static_cast<std::size_t>(b)];
        out.push_back(std::move(I));
    }
    const int64_t n_total = fr.n_basis, n_interior = fr.n_interior;
    if (n_total > n_interior) {
        // `_end_wings`: the flattened junction wings by (segment, rise)
        std::map<std::pair<int64_t, bool>, std::vector<int64_t>> by_end;
        for (int64_t k = 0; k < 2 * (n_total - n_interior); k++) {
            const int64_t b = n_interior + k / 2, side = k % 2;
            by_end[{fr.wing_seg[static_cast<std::size_t>(2 * b + side)],
                    fr.wing_rise.data()[2 * b + side]}]
                .push_back(k);
        }
        auto knot = [&](const std::vector<int64_t> &wings) {
            std::vector<cd> terms;
            for (int64_t k : wings) {
                const int64_t b = n_interior + k / 2, side = k % 2;
                const double sigma = fr.wing_sigma[static_cast<std::size_t>(2 * b + side)];
                terms.push_back(np_cmul(cd(sigma, 0.0), coeffs[static_cast<std::size_t>(b)]));
            }
            return np_csum(terms);
        };
        for (int64_t w = 0; w < fr.n_wires; w++) {
            auto start = by_end.find({fr.seg_offsets[w], false});
            if (start != by_end.end() && !start->second.empty()) out[w].front() = knot(start->second);
            auto end = by_end.find({fr.seg_offsets[w + 1] - 1, true});
            if (end != by_end.end() && !end->second.empty()) out[w].back() = knot(end->second);
        }
    }
    return out;
}

const double *arc_of(const razor::Front &fr, int64_t w) {
    return fr.arc_at_knot.data() + fr.seg_offsets[w] + w;
}

RunDataC finish(const Deck &deck, const Prepared &p, const Solved &s, const NumpyLinalg &np,
                std::vector<RunArrays> *arrays) {
    const razor::Front &fr = *p.front;
    const PortState st = port_state(deck, p, s, np);
    // `coeffs = solution.coeffs @ (_transform(mesh) @ state.v_gap)`
    const int64_t nc = p.mesh.n_columns(), ns = static_cast<int64_t>(p.mesh.sites.size());
    const std::vector<cd> tc = transform_c(p);
    const std::vector<cd> w = mv(np, Op{tc.data(), nc, ns, ns, 1}, st.v_gap);
    const std::vector<cd> coeffs = mv(np, Op{s.x.data(), s.n_basis, s.n_ports, 1, s.n_basis}, w);
    if (arrays != nullptr)
        arrays->push_back(RunArrays{s.y.v, coeffs, st.v_applied, st.v_gap, st.i_port, st.i_source});

    RunDataC data;
    // the source rows, deck order
    std::vector<double> powers;
    for (const Source &src : deck.sources) {
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
    // `0.5 * float(np.sum(np.real(z_load) * np.abs(i_port) ** 2))`, z_load = 0
    bool nonfinite = false;
    for (const cd &i : st.i_port)
        if (!std::isfinite(i.real()) || !std::isfinite(i.imag())) nonfinite = true;
    const double p_load = 0.5 * (nonfinite ? std::numeric_limits<double>::quiet_NaN() : 0.0);
    const double p_radiated = (p_in - 0.0) - p_load;

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
    const std::vector<std::vector<cd>> knots = knot_currents(fr, coeffs);
    std::vector<std::vector<cd>> slopes;
    for (std::size_t w = 0; w < p.mesh.pieces.size(); w++) {
        const SeamPiece &piece = p.mesh.pieces[w];
        const int64_t ne = piece.n_elements();
        const double step = piece.length() / static_cast<double>(ne);
        const std::vector<cd> &I = knots[w];
        const double *arc = arc_of(fr, static_cast<int64_t>(w));
        const int64_t n_knots = static_cast<int64_t>(I.size()), nseg = n_knots - 1;
        std::vector<cd> slope(static_cast<std::size_t>(nseg));
        for (int64_t j = 0; j < nseg; j++) {
            const cd diff(I[j + 1].real() - I[j].real(), I[j + 1].imag() - I[j].imag());
            slope[j] = np_cdiv(diff, cd(arc[j + 1] - arc[j], 0.0));
        }
        std::vector<cd> out;
        for (int64_t k = 0; k < ne; k++) {
            double sv = (static_cast<double>(k) + 0.5) * step;
            sv = std::min(std::max(sv, arc[0]), arc[n_knots - 1]);  // np.clip
            // searchsorted(arc, s, side="right") - 1, clamped
            const int64_t right = std::upper_bound(arc, arc + n_knots, sv) - arc;
            const int64_t span = std::min(std::max<int64_t>(right - 1, 0), nseg - 1);
            out.push_back(slope[static_cast<std::size_t>(span)]);
        }
        slopes.push_back(std::move(out));
    }
    const cd jw = py_cmul(cd(0.0, 1.0), cd(p.omega, 0.0));
    for (std::size_t index = 0; index < p.mesh.element_of.size(); index++) {
        const auto &pe = p.mesh.element_of[index];
        const std::vector<cd> &k = knots[static_cast<std::size_t>(pe.first)];
        const int64_t el = pe.second;
        const cd sum(k[el].real() + k[el + 1].real(), k[el].imag() + k[el + 1].imag());
        const cd current = np_scalar_cmul(cd(0.5, 0.0), sum);
        const cd sl = slopes[static_cast<std::size_t>(pe.first)][static_cast<std::size_t>(el)];
        const cd charge = np_cdiv(cd(-sl.real(), -sl.imag()), jw);
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

Lines loading(const RunDataC &data) {
    return {sp(31) + "- - - STRUCTURE IMPEDANCE LOADING - - -", "",
            sp(35) + "THIS STRUCTURE IS NOT LOADED",
            "ALLOCATE CM:" + d(data.unknown_count * data.unknown_count, 15)};
}

Lines timing() {
    return {sp(32) + "- - - MATRIX TIMING - - -", "",
            sp(24) + "FILL=" + f(0.0, 9, 3) + " SEC.,  FACTOR=" + f(0.0, 9, 3) + " SEC."};
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

Lines run_sections(const RunDataC &data) {
    Lines body;
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
               watts("RADIATED POWER", data.radiated_power), watts("WIRE LOSS", data.wire_loss),
               ind + ljust("EFFICIENCY", 14) + "=" + f(data.efficiency, 7, 2) + " PERCENT"});
    blanks(body, 3);
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
    add(body, timing());
    blanks(body, 3);
}

Lines closing_lines(bool closing) {
    if (!closing) return {};
    return {"", " RUN TIME =" + f(0.0, 10, 3)};
}

// `render_printout`
std::string render_printout(const Deck &deck, const std::string &echo_text, const RunDataC &data,
                            const std::string &basis, const EngineStamp &stamp, bool closing) {
    const std::vector<Card> cards = post_ge_cards(echo_text);
    const bool has_term = !cards.empty() && is_terminator(cards.back());
    Lines body = head_sections(deck, data);
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
    Lines body = head_sections(deck, first);
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

// The runs Python FINISHES carry no LD / RP / NE / NH (this unit's scope).
void check_scope(const Deck &run) {
    if (!run.load_cards.empty()) throw Decline{"lumped loads not yet in C++"};
    for (const Request &r : run.requests) {
        if (r.kind == Request::far_field) throw Decline{"radiation pattern not yet in C++"};
        if (r.kind == Request::near_field) throw Decline{"near field not yet in C++"};
    }
}

Solved solve(const Prepared &p, razor::SolveContext &ctx, const razor::Lapack &lapack,
             const NumpyLinalg &np) {
    const razor::Front &front = *p.front;
    Solved s;
    s.n_basis = front.n_basis;
    s.n_ports = front.n_ports;
    s.x.assign(static_cast<std::size_t>(s.n_basis * s.n_ports), cd(0.0, 0.0));
    const int info = razor::solve_ports(front, ctx, lapack, s.x.data(), 0);
    if (info != 0) throw PyRaise{"LinAlgError", "singular matrix"};
    // `cols.T @ X`: cols.T is the cast copy in its own (Fortran) order
    std::vector<cd> cols(front.cols.size());
    for (std::size_t k = 0; k < cols.size(); k++) cols[k] = cd(front.cols[k], 0.0);
    s.y = CMat(s.n_ports, s.n_ports);
    np_matmul(np, Op{cols.data(), s.n_ports, s.n_basis, 1, s.n_ports},
              Op{s.x.data(), s.n_basis, s.n_ports, 1, s.n_basis}, s.y.v.data(), s.n_ports, 1);
    return s;
}

}  // namespace

PrintoutResult razor_printout(const std::string &text, const std::string &basis,
                              const EngineStamp &stamp, razor::SolveContext &ctx,
                              const razor::Lapack &lapack, const NumpyLinalg &np,
                              std::vector<RunArrays> *arrays) {
    PrintoutResult out;
    const SeamResult r = razor_seam(text);
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
        auto solution = [&](long long index) -> const Solved & {
            const std::size_t i = static_cast<std::size_t>(index);
            if (!have[i]) {
                solved[i] = solve(r.prepared[i], ctx, lapack, np);
                have[i] = true;
            }
            return solved[i];
        };
        const std::size_t n_structures = s.chained ? s.structures.size() : 1;
        std::string printed;
        bool stopped = false;
        for (std::size_t k = 0; k < r.structures.size() && !stopped; k++) {
            const StructureOutcome &so = r.structures[k];
            const Answer &a = s.answers[k];
            const std::string echo = echo_of(s, k, text);
            const bool last = !s.chained || k + 1 == n_structures;
            std::vector<RunDataC> runs;
            for (std::size_t j = 0; j < so.runs.size() && !stopped; j++) {
                const RunOutcome &o = so.runs[j];
                if (o.kind == RunOutcome::refused) {
                    printed += render_refusal(&echo, o.message, basis, stamp);
                    stopped = true;
                } else if (o.kind == RunOutcome::raised) {
                    throw PyRaise{o.error_kind, o.message};
                } else {
                    const Prepared &p = r.prepared[static_cast<std::size_t>(o.prepared_index)];
                    const Solved &sol = solution(o.prepared_index);
                    runs.push_back(
                        finish(a.kind == Answer::multi ? a.runs[j] : a.deck, p, sol, np, arrays));
                }
            }
            if (stopped) break;
            if (so.runs.empty() || so.runs.back().kind == RunOutcome::declined)
                throw std::logic_error("a declined run reached the writer");
            if (a.kind == Answer::multi)
                printed += render_multi_printout(a.deck, echo, runs, basis, stamp, last);
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
    } catch (const std::exception &ex) {
        // Nothing Python prints (memory, a layout this file does not
        // replay): never a printout of our own; the caller falls back.
        out.raised_kind.clear();
        out.reason = std::string("C++ printout failed: ") + ex.what();
        return out;
    }
    out.route = PrintoutResult::printed;
    out.printout = printout_bytes(utf8);
    return out;
}

}  // namespace nec5
}  // namespace mw
