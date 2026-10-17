// The NEC-5 seam's RUN HALF and PRINTOUT in C++, for any basis: deck text
// in, the printout's bytes out, with no Python anywhere. It is the hardened
// copy of what a `momwire-nec5` launcher does after the structure half
// (`_nec5_seam_core.h`) and the solve:
//
//   * `eznec/_serve.py`'s `_finish`: `_port_state` (one source, or the
//     constrained multi-source drive), `coeffs = X @ (T @ v_gap)`, the
//     source rows, the networks, the power budget, `_element_geometry` and
//     the current and charge tables, the radiation pattern and the near
//     fields;
//   * `eznec/_printout.py`'s writer (`render_printout`,
//     `render_multi_printout`, `render_refusal`, the header and its engine
//     stamp) and `eznec/_shell.py`'s frame around it (`render`,
//     `_render_chain`, the last line of defence's INTERNAL ERROR printout,
//     and `write_printout`'s CRLF and latin-1 transliteration).
//
// What the SOLVER answers -- `compute_port_solution` and the readouts
// (`currents_at_knots`, `current_slopes`, `element_currents`,
// `wire_loss_power`) -- is asked of a `PrintoutEngine` (below), with the
// structure half's `SeamEngine`. razor-2p's is `_nec5_razor_printout_core.h`;
// another basis plugs in the same way.
//
//     mw::nec5::PrintoutResult r = mw::nec5::printout(
//         latin1_text, basis, stamp, engine, numpy);
//     // r.route: printed (r.printout: the file's bytes) | declined (r.reason)
//
// SCOPE: every deck the structure half serves: the results, refusals and
// the INTERNAL ERROR frame (an exception Python lets escape: its type name
// and momwire's message), checked on every run Python would FINISH (all of
// them up to the run or structure that stops the stream), so a declined deck
// never hides an answer Python would have printed. Two run-half failures
// whose words are a LIBRARY's, not momwire's -- numpy's singular-matrix
// LinAlgError from the multi-source drive, CPython's ZeroDivisionError from
// an EX 4 port carrying no current -- decline instead of spelling that text
// in C++; so does any failure with no Python counterpart (an allocation,
// say).
//
// THE ARITHMETIC IS NUMPY'S AND CPYTHON'S, operation for operation, so the
// bits are theirs:
//   * every matrix product and `numpy.linalg.solve` goes through NUMPY'S OWN
//     BLAS / LAPACK, with the call numpy's matmul makes for those operands
//     (`NumpyLinalg`: cblas_zgemm / cblas_zgemv / cblas_zdotu_sub, zgesv,
//     the ILP64 OpenBLAS numpy's wheels bundle) -- a sum's order is the
//     kernel's, and only the kernel knows it;
//   * the run path's elementwise complex multiply is `_cmul.cmul`, FUSED on
//     every CPU (`fma(a.re, b.re, -(a.im * b.im))`); numpy's complex
//     division, scalar arithmetic and pairwise sums are replayed as written
//     (`replay`, which an engine's readouts share);
//   * Python's complex `*` / `/` are CPython 3.12's `_Py_c_prod` /
//     `_Py_c_quot`, `sum()` of floats its compensated sum, `math.atan2` its
//     special cases, `math.degrees` its `x * (180 / pi)`;
//   * every number is printed as Python's `%` / format spec prints it
//     (printf's `%E` / `%f` in the C locale, NaN spelled as Python spells it).
// The Python seam stays the reference: `scripts/core_split/printout_compare.py`
// and `tests/test_razor_core_standalone.py` gate the whole printout byte for
// byte.
#pragma once

#include "_mw_core.h"
#include "_nec5_seam_core.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mw {
namespace nec5 {

// --- numpy's BLAS / LAPACK ---------------------------------------------------
// The ILP64 interface numpy's own OpenBLAS exports (`numpy.libs/
// libscipy_openblas64_*`: `scipy_cblas_zgemm64_`, `scipy_cblas_zgemv64_`,
// `scipy_cblas_zdotu_sub64_`, `scipy_zgesv_64_`). CBLAS enums are ints.
using np_int = int64_t;
using cblas_zgemm_fn = void (*)(int order, int trans_a, int trans_b, np_int m, np_int n,
                                np_int k, const void *alpha, const void *a, np_int lda,
                                const void *b, np_int ldb, const void *beta, void *c,
                                np_int ldc);
using cblas_zgemv_fn = void (*)(int order, int trans, np_int m, np_int n, const void *alpha,
                                const void *a, np_int lda, const void *x, np_int incx,
                                const void *beta, void *y, np_int incy);
using cblas_zdotu_sub_fn = void (*)(np_int n, const void *x, np_int incx, const void *y,
                                    np_int incy, void *dotu);
using zgesv_fn = void (*)(const np_int *n, const np_int *nrhs, void *a, const np_int *lda,
                          np_int *ipiv, void *b, const np_int *ldb, np_int *info);

// --- the host route (`momwire._lapack`, ABI 2) --------------------------------
// The LP64 Fortran BLAS a host publishes as `sys._mw_host_lapack` (OpenBLAS's
// `zgemm_` / `zgemv_`: every argument by reference, no hidden string
// lengths), beside the LU pair (`razor::Lapack`'s types).
using f_zgemm_fn = void (*)(const char *transa, const char *transb, const int *m, const int *n,
                            const int *k, const void *alpha, const void *a, const int *lda,
                            const void *b, const int *ldb, const void *beta, void *c,
                            const int *ldc);
using f_zgemv_fn = void (*)(const char *trans, const int *m, const int *n, const void *alpha,
                            const void *a, const int *lda, const void *x, const int *incx,
                            const void *beta, void *y, const int *incy);

// The host's LU pair (LAPACK's `zgetrf_` / `zgetrs_`, LP64): the same
// function types as `razor::Lapack`'s.
using f_zgetrf_fn = void (*)(int *m, int *n, cd *a, int *lda, int *ipiv, int *info);
using f_zgetrs_fn = void (*)(char *trans, int *n, int *nrhs, cd *a, int *lda, int *ipiv, cd *b,
                             int *ldb, int *info);

struct HostLinalg {
    f_zgemm_fn zgemm = nullptr;
    f_zgemv_fn zgemv = nullptr;
    f_zgetrf_fn getrf = nullptr;
    f_zgetrs_fn getrs = nullptr;
};

// SciPy's LAPACK expert driver (`scipy.linalg.lapack.zgesvx`, LP64 Fortran,
// the three hidden CHARACTER lengths passed): what `networks.MNASystem.solve`
// calls, outside the `_lapack` seam, so it is SciPy's library on every route.
using zgesvx_fn = void (*)(const char *fact, const char *trans, const int *n, const int *nrhs,
                           void *a, const int *lda, void *af, const int *ldaf, int *ipiv,
                           char *equed, double *r, double *c, void *b, const int *ldb, void *x,
                           const int *ldx, double *rcond, double *ferr, double *berr, void *work,
                           double *rwork, int *info, std::size_t, std::size_t, std::size_t);

struct NumpyLinalg {
    cblas_zgemm_fn zgemm = nullptr;
    cblas_zgemv_fn zgemv = nullptr;
    cblas_zdotu_sub_fn zdotu_sub = nullptr;
    zgesv_fn zgesv = nullptr;
    // When `host.zgemm` is set, the Python run path runs with this table
    // published, and so does this: every product is `_lapack.matmul`'s ONE
    // fixed call per shape and every solve `_lapack.solve_stacked`'s getrf +
    // getrs, all on the host's library; numpy's four entries above go unused
    // (an empty operand still takes numpy's `@`, as `_lapack` hands it there).
    HostLinalg host;
    // the network solve's (TL / NT decks); null declines them
    zgesvx_fn zgesvx = nullptr;
};

// --- the printout ------------------------------------------------------------

// Line 2's three fields (`_printout.engine_stamp`): the release, the
// compiled variant, and the overlay marker ("" outside an overlaid bundle).
// The basis is the fourth, passed beside it.
struct EngineStamp {
    std::string version;
    std::string variant;
    std::string marker;
};

struct PrintoutResult {
    enum Route { printed, declined } route = declined;
    // What the structure half answered (served, refused, raised, declined).
    SeamResult::Route seam = SeamResult::declined;
    // printed: an exception escaped (the INTERNAL ERROR frame): its type.
    std::string raised_kind;
    std::string printout;  // printed: the file's bytes (latin-1, CRLF)
    std::string reason;    // declined: why
    // declined because a C++ exception escaped the run half (memory, ...):
    // its Python-style kind ("MemoryError" for std::bad_alloc, else the C++
    // type) and what(). The hybrid falls back; the no-Python build writes
    // the INTERNAL ERROR printout from it.
    bool cxx_failed = false;
    std::string failure_kind, failure_message;
};

// One finished run's intermediate arrays, for the gates that hold them to
// Python's bytes: the solve's `Y` (C order, n_ports x n_ports), `_finish`'s
// `coeffs`, and `_port_state`'s four vectors.
struct RunArrays {
    std::vector<cd> y, coeffs, v_applied, v_gap, i_port, i_source;
};

// --- what the run half asks of the solver ------------------------------------

// The run half's own answers, thrown by it and by an engine's readouts: an
// exception Python lets escape (`_shell.main`'s last line of defence reports
// it as "INTERNAL ERROR IN MOMWIRE ENGINE - <kind>: <message>"), and a run
// this unit does not print (why).
struct PrintoutRaise {
    std::string kind, message;
};
struct PrintoutDecline {
    std::string reason;
};

// A 2-D complex128 operand: its shape and its strides in ELEMENTS (numpy's
// are bytes; every operand here is complex128, so byte strides are 16x
// these). A 1-D vector is (n, 1) with a zero second stride, as the gufunc
// hands it.
struct Op {
    const cd *p;
    int64_t d0, d1, s0, s1;
};

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

// A preparation's solve: `PortSolution.coeffs` (column-major n_basis x
// n_ports) and `.y` (C-ordered n_ports x n_ports).
struct Solved {
    int64_t n_basis = 0, n_ports = 0;
    std::vector<cd> x;
    CMat y;
};

// `element_currents(coeffs, subdiv)`'s (mid, moment, nodes, delta): the
// elements' midpoints and current moments, (n, 3) row-major, and the nodes
// with the current step across each.
struct ElementCurrents {
    std::vector<double> mid, nodes;
    std::vector<cd> moment, delta;
};

// The run half's questions of one basis' constructed solver
// (`compute_port_solution` and the readouts `_finish`, `_pattern` and
// `_near_field` call), and the structure half it answers them for.
// razor-2p's is `_nec5_razor_printout_core.h`'s.
class PrintoutEngine {
  public:
    virtual ~PrintoutEngine() = default;
    virtual const SeamEngine &seam() const = 0;
    // `solver.compute_port_solution()`: X and Y. A singular system is
    // numpy's LinAlgError (`PrintoutRaise`).
    virtual Solved solve(const Prepared &p, const NumpyLinalg &np) = 0;
    // A stepping FR's run is finished before the next frequency is solved:
    // its solver is released (memory) once its run is finished.
    virtual void release(Prepared &p) = 0;
    // `solver.currents_at_knots(coeffs)`: per piece, the current at each knot.
    virtual std::vector<std::vector<cd>> currents_at_knots(const Prepared &p,
                                                           const std::vector<cd> &coeffs) const = 0;
    // `solver.current_slopes(coeffs, centres_per_piece)`: dI/ds per piece at
    // each arclength given (`knots`: `currents_at_knots(coeffs)`).
    virtual std::vector<std::vector<cd>> current_slopes(
        const Prepared &p, const std::vector<cd> &coeffs, const std::vector<std::vector<cd>> &knots,
        const std::vector<std::vector<double>> &centres_per_piece) const = 0;
    // `solver.element_currents(coeffs)` (subdiv 1): the far field's
    // midpoints and moments, appended to `mid` and `moment`.
    virtual void element_moments(const Prepared &p, const std::vector<std::vector<cd>> &knots,
                                 std::vector<double> &mid, std::vector<cd> &moment) const = 0;
    // `solver.element_currents(coeffs, subdiv)`: the near field's elements.
    virtual ElementCurrents element_currents(const Prepared &p,
                                             const std::vector<std::vector<cd>> &knots,
                                             int64_t subdiv) const = 0;
    // `solver.wire_loss_power(coeffs, omega)[0]`.
    virtual double wire_loss_power(const Prepared &p, const std::vector<cd> &coeffs,
                                   double omega) const = 0;
    // `solver.omega` and the eps0 `eps_tilde` folds the medium with, for the
    // near field's composing ground.
    virtual double solver_omega(const Prepared &p) const = 0;
    virtual double eps0() const = 0;
};

// The whole of a `momwire-nec5` launcher for one deck file's bytes, for
// `engine`'s basis (`basis`: the stamp's name for it). Never throws for a
// deck: whatever Python would print, this prints, or it declines. `arrays`,
// when given, collects every finished run's arrays in execution order.
// `open_end`: the text is a stream's piece (momwire#1512,
// `_shell._stream_piece`): a structure NX ends it prints without RUN TIME.
PrintoutResult printout(const std::string &latin1_text, const std::string &basis,
                        const EngineStamp &stamp, PrintoutEngine &engine, const NumpyLinalg &np,
                        std::vector<RunArrays> *arrays = nullptr, bool open_end = false);

// --- the replayed arithmetic an engine's readouts share ----------------------
// numpy's and CPython's operations as the run half replays them (see above),
// for an engine's readouts to compute with the same bits.
namespace replay {

// `_cmul.cmul`: each part one fused multiply-add.
cd np_cmul(cd a, cd b);
// numpy's complex add / subtract, and the fused product by a REAL operand
// promoted to complex (x, 0).
inline cd c_add(cd a, cd b) { return cd(a.real() + b.real(), a.imag() + b.imag()); }
inline cd c_sub(cd a, cd b) { return cd(a.real() - b.real(), a.imag() - b.imag()); }
inline cd c_mul_real(cd a, double x) { return np_cmul(a, cd(x, 0.0)); }
// `CDOUBLE_divide`.
cd np_cdiv(cd a, cd b);
// `DOUBLE_pairwise_sum` over `n` contiguous float64 values.
double np_pairwise(const double *a, int64_t n);
// `a.sum()` of a 1-D complex array.
cd np_csum(const std::vector<cd> &a);
// numpy's `interp(x, xp, fp)` over `n` points.
double np_interp(double x, const double *xp, const double *fp, int64_t n);
// `np.linspace(a, b, num)` for 3-vectors (`num >= 2`), appended to `out`.
void linspace3(const double *a, const double *b, int64_t num, std::vector<double> &out);
// `CDOUBLE_matmul` for one (m, n) @ (n, p) product into `out` (strides in
// elements; a 1-D result has `os_p` 0); with a host table, `_lapack.matmul`.
void np_matmul(const NumpyLinalg &np, const Op &a, const Op &b, cd *out, int64_t os_m,
               int64_t os_p);

}  // namespace replay

// `_printout.engine_stamp(basis)`: line 2, at most 79 columns.
std::string engine_stamp(const std::string &basis, const EngineStamp &stamp);

// `_printout.render_refusal`: the header (with the comment box when there
// is deck text) and one NEC ERROR line; `\n` line ends, UTF-8.
std::string render_refusal(const std::string *latin1_deck_text, const std::string &utf8_reason,
                           const std::string &basis, const EngineStamp &stamp);

// `_shell.write_printout`'s bytes for a rendered printout: `\n` -> CRLF, and
// every character outside latin-1 spelled inside it (`_shell.transliterate`).
std::string printout_bytes(const std::string &utf8_text);

// Python's `format(x, "W.DE")` / `format(x, "W.Df")` (and the `%` forms):
// printf in the C locale, NaN spelled NAN / nan without a sign.
std::string py_format_e(double x, int width, int digits);
std::string py_format_f(double x, int width, int digits);

}  // namespace nec5
}  // namespace mw
