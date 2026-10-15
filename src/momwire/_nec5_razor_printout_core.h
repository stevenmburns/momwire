// The NEC-5 seam's RUN HALF and PRINTOUT for razor-2p in C++: deck text in,
// the printout's bytes out, with no Python anywhere. It is the hardened copy
// of what `momwire-nec5-razor-2p` does after the structure half
// (`_nec5_razor_seam_core.h`) and the solve (`_razor_front_core.h` /
// `_razor_solve_core.h`):
//
//   * `RazorSolver.compute_port_solution`'s last line, `Y = cols.T @ X`;
//   * `eznec/_serve.py`'s `_finish`: `_port_state` (one source, or the
//     constrained multi-source drive), `coeffs = X @ (T @ v_gap)`, the
//     source rows, the power budget, `_element_geometry` and razor-2p's
//     current and charge readouts (`currents_at_knots`, `current_slopes`);
//   * `eznec/_printout.py`'s writer (`render_printout`,
//     `render_multi_printout`, `render_refusal`, the header and its engine
//     stamp) and `eznec/_shell.py`'s frame around it (`render`,
//     `_render_chain`, the last line of defence's INTERNAL ERROR printout,
//     and `write_printout`'s CRLF and latin-1 transliteration).
//
//     mw::nec5::PrintoutResult r = mw::nec5::razor_printout(
//         latin1_text, "razor-2p", stamp, ctx, lapack, numpy);
//     // r.route: printed (r.printout: the file's bytes) | declined (r.reason)
//
// SCOPE: every deck the structure half covers whose runs carry no LD card,
// no RP and no NE / NH: those DECLINE by name ("lumped loads / radiation
// pattern / near field not yet in C++"), as does everything the structure
// half declines. A decline is checked on every run Python would FINISH
// (all of them up to the run or structure that stops the stream), so a
// declined deck never hides an answer Python would have printed, and a
// printed one is Python's whole printout: refusals, the INTERNAL ERROR frame
// (an exception Python lets escape: its type name and momwire's message),
// and the results. Two run-half failures whose words are a LIBRARY's, not
// momwire's -- numpy's singular-matrix LinAlgError from the multi-source
// drive, CPython's ZeroDivisionError from an EX 4 port carrying no current
// -- decline instead of spelling that text in C++; so does any failure with
// no Python counterpart (an allocation, say).
//
// THE ARITHMETIC IS NUMPY'S AND CPYTHON'S, operation for operation, so the
// bits are theirs:
//   * every matrix product and `numpy.linalg.solve` goes through NUMPY'S OWN
//     BLAS / LAPACK, with the call numpy's matmul makes for those operands
//     (`NumpyLinalg`: cblas_zgemm / cblas_zgemv / cblas_zdotu_sub, zgesv,
//     the ILP64 OpenBLAS numpy's wheels bundle) -- a sum's order is the
//     kernel's, and only the kernel knows it;
//   * numpy's elementwise complex multiply is FUSED on the CPUs numpy
//     dispatches its x86-64-v3 / ASIMD loops to (`fma(a.re, b.re, -(a.im *
//     b.im))`), and plain elsewhere (`NumpyLinalg::fused_cmul`); its complex
//     division, scalar arithmetic and pairwise sums are replayed as written;
//   * Python's complex `*` / `/` are CPython 3.12's `_Py_c_prod` /
//     `_Py_c_quot`, `sum()` of floats its compensated sum, `math.atan2` its
//     special cases, `math.degrees` its `x * (180 / pi)`;
//   * every number is printed as Python's `%` / format spec prints it
//     (printf's `%E` / `%f` in the C locale, NaN spelled as Python spells it).
// The Python seam stays the reference: `scripts/core_split/printout_compare.py`
// and `tests/test_razor_core_standalone.py` gate the whole printout byte for
// byte.
#pragma once

#include "_nec5_razor_seam_core.h"
#include "_razor_solve_core.h"

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

struct NumpyLinalg {
    cblas_zgemm_fn zgemm = nullptr;
    cblas_zgemv_fn zgemv = nullptr;
    cblas_zdotu_sub_fn zdotu_sub = nullptr;
    zgesv_fn zgesv = nullptr;
    // numpy's elementwise complex multiply: fused multiply-add (its
    // x86-64-v3 and ASIMD loops) or two roundings (its x86-64-v2 loop).
    bool fused_cmul = true;
};

// What numpy's dispatcher picks on THIS CPU: x86-64 with AVX2 + FMA3 (the
// x86-64-v3 target) or aarch64 fuse; an older x86-64 does not.
bool numpy_fuses_complex_multiply();

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
};

// One finished run's intermediate arrays, for the gates that hold them to
// Python's bytes: the solve's `Y` (C order, n_ports x n_ports), `_finish`'s
// `coeffs`, and `_port_state`'s four vectors.
struct RunArrays {
    std::vector<cd> y, coeffs, v_applied, v_gap, i_port, i_source;
};

// The whole of `momwire-nec5-razor-2p` for one deck file's bytes. Never
// throws for a deck: whatever Python would print, this prints, or it
// declines. `arrays`, when given, collects every finished run's arrays in
// execution order.
PrintoutResult razor_printout(const std::string &latin1_text, const std::string &basis,
                              const EngineStamp &stamp, razor::SolveContext &ctx,
                              const razor::Lapack &lapack, const NumpyLinalg &np,
                              std::vector<RunArrays> *arrays = nullptr);

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
