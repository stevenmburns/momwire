// The C++ route of `momwire-nec5-bs2` (route.h, momwire#1517 Phase A):
// `momwire-nec5-bs2`'s printout for the decks bs2's C++ serves, on the
// executable's OpenBLAS. Compiled once per namespace, as route.cpp.
//
// PYTHON'S TWO LIBRARIES. The launcher runs bs2 on two OpenBLAS builds:
// numpy's ILP64 one (the products: `tangents @ tangents.T`, the image
// weights, the loading Gram, the run half's matmuls) and SciPy's LP64 one
// (the LU and the symmetric factor). On Linux the executable links both, at
// their wheels' commits (build_openblas.sh, momwire#1751): numpy's CBLAS
// calls go to numpy's (MWN, through the LP64 adapters below), the LAPACK to
// SciPy's (MWX). Elsewhere one library answers both. The gate
// (exe_compare --basis bs2) is what says their bits agree.
//
// MW_ROUTE_NEC4 (momwire#1607 Phase C2): the same route for `momwire-nec42`,
// the NEC-4.2 slot's default basis: bs2 on the slot's run half
// (`_bspline_nec4_run_core.h`, `_nec4.render(text, basis="bs2")`), on the
// same adapters.
#include "route.h"

#include "_near_interface_eval.h"  // the features this run asked for (momwire#1860)
#include "_razor_pair_reuse_core.h"  // the reuse plans' counters (momwire#1971)
#if MW_ROUTE_NEC4
#include "_bspline_nec4_run_core.h"
#endif

#include "_somm_below_fill_core.h"

#include "_nec5_bspline_printout_core.h"
#include "_nec5_bspline_seam_core.h"
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
void MWN(cblas_dsyrk)(int, int, int, int, int, double, const double *, int, double, double *, int);
void MWN(cblas_dgemm)(int, int, int, int, int, int, double, const double *, int, const double *, int, double,
                      double *, int);
void MWN(cblas_dgemv)(int, int, int, int, double, const double *, int, const double *, int, double, double *,
                      int);
double MWN(cblas_ddot)(int, const double *, int, const double *, int);
void MWN(cblas_zgemm)(int, int, int, int, int, int, const void *, const void *, int, const void *, int,
                      const void *, void *, int);
void MWN(cblas_zgemv)(int, int, int, int, const void *, const void *, int, const void *, int, const void *,
                      void *, int);
void MWN(cblas_zdotu_sub)(int, const void *, int, const void *, int, void *);
void MWN(cblas_zdotc_sub)(int, const void *, int, const void *, int, void *);
void MWN(zgesv_)(const int *, const int *, void *, const int *, int *, void *, const int *, int *);
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
    MWN(cblas_dsyrk)(order, uplo, trans, i32(n), i32(k), alpha, a, i32(lda), beta, c, i32(ldc));
}
void a_dgemm(int order, int ta, int tb, np_int m, np_int n, np_int k, double alpha, const double *a, np_int lda,
             const double *b, np_int ldb, double beta, double *c, np_int ldc) {
    MWN(cblas_dgemm)(order, ta, tb, i32(m), i32(n), i32(k), alpha, a, i32(lda), b, i32(ldb), beta, c, i32(ldc));
}
void a_dgemv(int order, int trans, np_int m, np_int n, double alpha, const double *a, np_int lda, const double *x,
             np_int incx, double beta, double *y, np_int incy) {
    MWN(cblas_dgemv)(order, trans, i32(m), i32(n), alpha, a, i32(lda), x, i32(incx), beta, y, i32(incy));
}
double a_ddot(np_int n, const double *x, np_int incx, const double *y, np_int incy) {
    return MWN(cblas_ddot)(i32(n), x, i32(incx), y, i32(incy));
}
void a_zgemm(int order, int ta, int tb, np_int m, np_int n, np_int k, const void *alpha, const void *a, np_int lda,
             const void *b, np_int ldb, const void *beta, void *c, np_int ldc) {
    MWN(cblas_zgemm)(order, ta, tb, i32(m), i32(n), i32(k), alpha, a, i32(lda), b, i32(ldb), beta, c, i32(ldc));
}
void a_zgemv(int order, int trans, np_int m, np_int n, const void *alpha, const void *a, np_int lda, const void *x,
             np_int incx, const void *beta, void *y, np_int incy) {
    MWN(cblas_zgemv)(order, trans, i32(m), i32(n), alpha, a, i32(lda), x, i32(incx), beta, y, i32(incy));
}
void a_zdotu(np_int n, const void *x, np_int incx, const void *y, np_int incy, void *out) {
    MWN(cblas_zdotu_sub)(i32(n), x, i32(incx), y, i32(incy), out);
}
void a_zdotc(np_int n, const void *x, np_int incx, const void *y, np_int incy, void *out) {
    MWN(cblas_zdotc_sub)(i32(n), x, i32(incx), y, i32(incy), out);
}
void a_zgesv(const np_int *n, const np_int *nrhs, void *a, const np_int *lda, np_int *ipiv, void *b,
             const np_int *ldb, np_int *info) {
    const int nn = i32(*n), nr = i32(*nrhs), la = i32(*lda), lb = i32(*ldb);
    std::vector<int> piv(static_cast<std::size_t>(std::max(nn, 1)));
    int inf = 0;
    MWN(zgesv_)(&nn, &nr, a, &la, piv.data(), b, &lb, &inf);
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

bool configure_features(const std::vector<std::string> &names, bool stamped) {
    bool ok = true;
    for (const std::string &n : names) ok = ni_eval::request_feature(n, stamped) && ok;
    return ok;
}

bool withdraw_features(const std::vector<std::string> &names) {
    bool ok = true;
    for (const std::string &n : names) ok = ni_eval::withdraw_feature(n) && ok;
    return ok;
}

void stamp_uncertified(bool on) { ni_eval::stamp_uncertified(on); }

// Line 2's last note (a deprecated executable name's, momwire#2090), after
// the features' tail. Function-local, so no avx2 copy gains a start-up entry.
namespace {
std::string &line2_note() {
    static std::string note;
    return note;
}
}  // namespace
void stamp_note(const std::string &note) { line2_note() = note; }

void reset_features() {
    ni_eval::reset_requested();
    line2_note().clear();
}

void reuse_plans(long long &built, long long &declined, long long &found) {
    const razor::reuse::CacheCounters c = razor::reuse::cache_counters();
    built = static_cast<long long>(c.built);
    declined = static_cast<long long>(c.declined);
    found = static_cast<long long>(c.found);
}

std::string reuse_outcome() { return std::string(); }

void configure_grid_cache(const std::string &tag) {
    somm_core::configure_disk_cache(tag);
    // the below/below table (the buried fill's): filled here, kept in the
    // same store
    somm_below::configure_below_store(true);
    somm_below::set_source(somm_below::fill_source());
}

long long largest_solve(const std::string &latin1_text, bool open_end, long long *crossing_pairs,
                        long long *below_pairs) {
    if (crossing_pairs) *crossing_pairs = 0;
    if (below_pairs) *below_pairs = 0;
#if MW_ROUTE_NEC4
    // the size policy reads n_basis alone here
    return nec4::largest_solve(latin1_text, "bs2", open_end);
#endif
    const nec5::SeamResult r = nec5::bspline_seam(latin1_text, open_end);
    if (r.route != nec5::SeamResult::served) return -1;
    long long n = 0;
    for (const nec5::Prepared &p : r.prepared) {
        const nec5::BSplineFront *f = nec5::bspline_part(p).front.get();
        if (!f) continue;
        n = std::max<long long>(n, f->basis.n_basis);
        // The size policy's other two measures, as the razor route reports
        // them (route.cpp): a buried fill's (`fill_buried`) above segments x
        // below segments when it has both sides (the crossing serve, or the
        // transmitted blocks of wires on both sides with no crossing node),
        // and a wholly-buried one's segments squared. Reported as 0 before
        // momwire#1956, so a crossing deck under 128 basis ran on the serial
        // copy: d3_hub without its near field (109 basis, 1,312 crossing
        // pairs) took 103 ms at the policy's one thread and takes 46 at
        // every core (Skylake, 4 cores, Sommerfeld store off, median of 3),
        // the printout byte for byte the same. 30 of the 168 decks with a
        // ground card in tests/fixtures move from 1 thread to every core this
        // way, each printing what it printed before.
        if (!f->buried) continue;
        const std::vector<uint8_t> &below_wire = f->basis.below_wire;
        const std::vector<int64_t> &offsets = f->geom.seg_offsets;
        long long above = 0, below = 0;
        for (size_t w = 0; w < below_wire.size() && w + 1 < offsets.size(); w++)
            (below_wire[w] != 0 ? below : above) += offsets[w + 1] - offsets[w];
        if (crossing_pairs && above > 0) *crossing_pairs = std::max(*crossing_pairs, above * below);
        if (below_pairs && above == 0) *below_pairs = std::max(*below_pairs, below * below);
    }
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
    stamp.marker = ni_eval::requested_stamp();  // `_features.stamp()`'s tail
    if (!line2_note().empty()) stamp.marker += (stamp.marker.empty() ? "" : " ") + line2_note();
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
    const nec5::PrintoutResult r = nec5::bspline_printout(latin1_text, "bs2", stamp, lp, sym, np,
                                                          arrays_dir.empty() ? nullptr : &arrays, open_end);
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
