// The C++ route of `momwire-nec5-bs2` (route.h, momwire#1517 Phase A):
// `momwire-nec5-bspline`'s printout for the decks bs2's C++ serves, on the
// executable's one OpenBLAS. Compiled once per namespace, as route.cpp.
//
// ONE LIBRARY FOR BOTH OF PYTHON'S. The launcher runs bs2 on two OpenBLAS
// builds: numpy's ILP64 one (the products: `tangents @ tangents.T`, the
// image weights, the loading Gram, the run half's matmuls) and SciPy's LP64
// one (the LU and the symmetric factor). The executable links one static
// OpenBLAS, the commit SciPy's wheel bundles (build_openblas.sh), and answers
// numpy's CBLAS calls from it too, through the LP64 adapters below. The gate
// (exe_compare --basis bspline) is what says their bits agree.
//
// MW_ROUTE_SG (momwire#1718 S10): the same route for
// `momwire-nec5-sinusoidal-galerkin`, over the Sinusoidal-Galerkin engine
// (`_nec5_sinusoidal_galerkin_core.h`); the adapters are this file's.
//
// MW_ROUTE_NEC4 (momwire#1607 Phase C2): the same route for `momwire-nec42`,
// the NEC-4.2 slot's default basis: bs2 on the slot's run half
// (`_bspline_nec4_run_core.h`, `_nec4.render(text, basis="bspline")`), on the
// same adapters.
#include "route.h"
#if MW_ROUTE_NEC4
#include "_bspline_nec4_run_core.h"
#endif

#include "_somm_below_fill_core.h"

#include "_nec5_bspline_printout_core.h"
#include "_nec5_bspline_seam_core.h"
#if MW_ROUTE_SG
#include "_nec5_sinusoidal_galerkin_core.h"
#endif
#include "_somm_grid_core.h"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include "blas.h"

extern "C" {
// The CBLAS face of the same static library (LP64: int dimensions).
void MWX(cblas_dsyrk)(int, int, int, int, int, double, const double *, int, double, double *, int);
void MWX(cblas_dgemm)(int, int, int, int, int, int, double, const double *, int, const double *, int, double,
                      double *, int);
void MWX(cblas_dgemv)(int, int, int, int, double, const double *, int, const double *, int, double, double *,
                      int);
double MWX(cblas_ddot)(int, const double *, int, const double *, int);
void MWX(cblas_zgemm)(int, int, int, int, int, int, const void *, const void *, int, const void *, int,
                      const void *, void *, int);
void MWX(cblas_zgemv)(int, int, int, int, const void *, const void *, int, const void *, int, const void *,
                      void *, int);
void MWX(cblas_zdotu_sub)(int, const void *, int, const void *, int, void *);
void MWX(cblas_zdotc_sub)(int, const void *, int, const void *, int, void *);
void MWX(zgesv_)(const int *, const int *, void *, const int *, int *, void *, const int *, int *);
void MWX(zsytrf_)(char *, int *, std::complex<double> *, int *, int *, std::complex<double> *, int *, int *);
void MWX(zsytrs_)(char *, int *, int *, std::complex<double> *, int *, int *, std::complex<double> *, int *,
                  int *);
}

namespace mw {
namespace exe {

namespace {

using nec5::np_int;

int i32(np_int v) { return static_cast<int>(v); }

void a_dsyrk(int order, int uplo, int trans, np_int n, np_int k, double alpha, const double *a, np_int lda,
             double beta, double *c, np_int ldc) {
    MWX(cblas_dsyrk)(order, uplo, trans, i32(n), i32(k), alpha, a, i32(lda), beta, c, i32(ldc));
}
void a_dgemm(int order, int ta, int tb, np_int m, np_int n, np_int k, double alpha, const double *a, np_int lda,
             const double *b, np_int ldb, double beta, double *c, np_int ldc) {
    MWX(cblas_dgemm)(order, ta, tb, i32(m), i32(n), i32(k), alpha, a, i32(lda), b, i32(ldb), beta, c, i32(ldc));
}
void a_dgemv(int order, int trans, np_int m, np_int n, double alpha, const double *a, np_int lda, const double *x,
             np_int incx, double beta, double *y, np_int incy) {
    MWX(cblas_dgemv)(order, trans, i32(m), i32(n), alpha, a, i32(lda), x, i32(incx), beta, y, i32(incy));
}
double a_ddot(np_int n, const double *x, np_int incx, const double *y, np_int incy) {
    return MWX(cblas_ddot)(i32(n), x, i32(incx), y, i32(incy));
}
void a_zgemm(int order, int ta, int tb, np_int m, np_int n, np_int k, const void *alpha, const void *a, np_int lda,
             const void *b, np_int ldb, const void *beta, void *c, np_int ldc) {
    MWX(cblas_zgemm)(order, ta, tb, i32(m), i32(n), i32(k), alpha, a, i32(lda), b, i32(ldb), beta, c, i32(ldc));
}
void a_zgemv(int order, int trans, np_int m, np_int n, const void *alpha, const void *a, np_int lda, const void *x,
             np_int incx, const void *beta, void *y, np_int incy) {
    MWX(cblas_zgemv)(order, trans, i32(m), i32(n), alpha, a, i32(lda), x, i32(incx), beta, y, i32(incy));
}
void a_zdotu(np_int n, const void *x, np_int incx, const void *y, np_int incy, void *out) {
    MWX(cblas_zdotu_sub)(i32(n), x, i32(incx), y, i32(incy), out);
}
void a_zdotc(np_int n, const void *x, np_int incx, const void *y, np_int incy, void *out) {
    MWX(cblas_zdotc_sub)(i32(n), x, i32(incx), y, i32(incy), out);
}
void a_zgesv(const np_int *n, const np_int *nrhs, void *a, const np_int *lda, np_int *ipiv, void *b,
             const np_int *ldb, np_int *info) {
    const int nn = i32(*n), nr = i32(*nrhs), la = i32(*lda), lb = i32(*ldb);
    std::vector<int> piv(static_cast<std::size_t>(std::max(nn, 1)));
    int inf = 0;
    MWX(zgesv_)(&nn, &nr, a, &la, piv.data(), b, &lb, &inf);
    for (int i = 0; i < nn; i++) ipiv[i] = piv[static_cast<std::size_t>(i)];
    *info = inf;
}

#if !MW_ROUTE_NEC4
void dump_arrays(const std::string &dir, const std::vector<nec5::RunArrays> &arrays) {
    for (std::size_t k = 0; k < arrays.size(); k++) {
        const nec5::RunArrays &a = arrays[k];
        const std::pair<const char *, const std::vector<cd> *> named[] = {
            {"y", &a.y},         {"coeffs", &a.coeffs}, {"v_applied", &a.v_applied},
            {"v_gap", &a.v_gap}, {"i_port", &a.i_port}, {"i_source", &a.i_source}};
        for (const auto &nv : named) {
            const std::string name = dir + "/" + std::to_string(k) + "_" + nv.first + ".bin";
            if (std::FILE *w = std::fopen(name.c_str(), "wb")) {
                std::fwrite(nv.second->data(), sizeof(cd), nv.second->size(), w);
                std::fclose(w);
            }
        }
    }
}
#endif

}  // namespace

void configure_grid_cache(const std::string &tag) {
    somm_core::configure_disk_cache(tag);
    // the below/below table (the buried fill's): filled here, kept in the
    // same store
    somm_below::configure_below_store(true);
    somm_below::set_source(somm_below::fill_source());
}

long long largest_solve(const std::string &latin1_text, bool open_end) {
#if MW_ROUTE_NEC4
    return nec4::largest_solve(latin1_text, "bspline", open_end);
#endif
#if MW_ROUTE_SG
    const nec5::SeamResult rs = nec5::sinusoidal_galerkin_seam(latin1_text, open_end);
    if (rs.route != nec5::SeamResult::served) return -1;
    long long ns = 0;
    for (const nec5::Prepared &p : rs.prepared) {
        const auto *e = dynamic_cast<const nec5::SinusoidalGalerkinPrepared *>(p.engine.get());
        if (e != nullptr && e->front) ns = std::max<long long>(ns, e->front->geometry.n_segs);
    }
    return ns;
#endif
    const nec5::SeamResult r = nec5::bspline_seam(latin1_text, open_end);
    if (r.route != nec5::SeamResult::served) return -1;
    long long n = 0;
    for (const nec5::Prepared &p : r.prepared)
        if (nec5::bspline_part(p).front) n = std::max<long long>(n, nec5::bspline_part(p).front->basis.n_basis);
    return n;
}

bool answer(const std::string &latin1_text, const std::string &version, const std::string &variant,
            const std::string &arrays_dir, std::string &printout, std::string &reason, bool *failed,
            std::string *failure_kind, std::string *failure_message, std::string *raised_kind,
            bool open_end, bool *served) {
    if (served) *served = false;
    razor::Lapack lp;
    lp.getrf = reinterpret_cast<razor::zgetrf_fn>(&MWX(zgetrf_));
    lp.getrs = reinterpret_cast<razor::zgetrs_fn>(&MWX(zgetrs_));
#if MW_ROUTE_SG
    // a crossing node's constrained solve (momwire#1718 S11b)
    lp.geqp3 = reinterpret_cast<razor::zgeqp3_fn>(&MWX(zgeqp3_));
#endif
    bspline::SymLapack sym;
    sym.sytrf = &MWX(zsytrf_);
    sym.sytrs = &MWX(zsytrs_);
    nec5::NumpyLinalg np;
    np.dsyrk = &a_dsyrk;
    np.dgemm = &a_dgemm;
    np.dgemv = &a_dgemv;
    np.ddot = &a_ddot;
    np.zgemm = &a_zgemm;
    np.zgemv = &a_zgemv;
    np.zdotu_sub = &a_zdotu;
    np.zdotc_sub = &a_zdotc;  // `np.vdot` (the crossing split's ACA)
    np.zgesv = &a_zgesv;
    np.zgesvx = &MWX(zgesvx_);  // the network solve's expert driver (SciPy's)
    nec5::EngineStamp stamp;
    stamp.version = version;
    stamp.variant = variant;
#if MW_ROUTE_NEC4
    // the NEC-4.2 slot's printout (route_sin.cpp's handling, over bs2)
    (void)arrays_dir;
    if (failed) *failed = false;
    if (raised_kind) raised_kind->clear();
    bspline_nec4::Linalg la;
    la.lu = lp;
    la.sym = sym;
    la.np = np;
    std::string out, why;
    bool ok, go_on = false;
    try {
        ok = open_end ? bspline_nec4::printout_piece(latin1_text, stamp, la, out, why, go_on)
                      : bspline_nec4::printout(latin1_text, stamp, la, out, why);
    } catch (const std::bad_alloc &) {
        if (failed) *failed = true;
        if (failure_kind) *failure_kind = "MemoryError";
        if (failure_message) *failure_message = "std::bad_alloc";
        reason = "an allocation failed";
        return false;
    } catch (const std::exception &e) {
        if (failed) *failed = true;
        if (failure_kind) *failure_kind = "RuntimeError";
        if (failure_message) *failure_message = e.what();
        reason = e.what();
        return false;
    }
    if (!ok) {
        reason = why;
        return false;
    }
    printout = nec5::printout_bytes(out);
    if (served) *served = go_on;
    return true;
#else
    std::vector<nec5::RunArrays> arrays;
#if MW_ROUTE_SG
    (void)sym;
    const nec5::PrintoutResult r = nec5::sinusoidal_galerkin_printout(
        latin1_text, "sinusoidal-galerkin", stamp, lp, np, arrays_dir.empty() ? nullptr : &arrays, open_end);
#else
    const nec5::PrintoutResult r = nec5::bspline_printout(latin1_text, "bspline", stamp, lp, sym, np,
                                                          arrays_dir.empty() ? nullptr : &arrays, open_end);
#endif
    if (r.route != nec5::PrintoutResult::printed) {
        reason = r.reason;
        if (failed) *failed = r.cxx_failed;
        if (failure_kind) *failure_kind = r.failure_kind;
        if (failure_message) *failure_message = r.failure_message;
        return false;
    }
    printout = r.printout;
    if (raised_kind) *raised_kind = r.raised_kind;
    if (served) *served = r.seam == nec5::SeamResult::served && r.raised_kind.empty();
    if (!arrays_dir.empty()) dump_arrays(arrays_dir, arrays);
    return true;
#endif
}

}  // namespace exe
}  // namespace mw
