// The C++ route of `momwire-nec42-sinusoidal` (route.h, momwire#1607 B4b):
// the NEC-4.2 slot's printout for the sinusoidal basis
// (`_nec4.render(text, basis="sinusoidal")`, `_sinusoidal_run_core.h`), on
// the executable's OpenBLAS. Compiled once per namespace, as route.cpp.
//
// PYTHON'S TWO LIBRARIES, as route_bs2.cpp: the launcher runs the basis on
// numpy's ILP64 OpenBLAS (the dense reduction's zgemm, the drive block's
// zgesv, the element match's dgemm, the run half's matmuls) and SciPy's LP64
// one (`scipy.linalg.solve`'s getrf / getrs). On Linux the executable links
// both at their wheels' commits (build_openblas.sh, momwire#1751) and sends
// numpy's CBLAS calls to numpy's (MWN, through LP64 adapters), the LAPACK to
// SciPy's (MWX); elsewhere one library answers both. The audit
// (scratch/standalone-audit-1021/run_audit_nec42.py in antennaknobs) is what
// says their bits agree.
#include "route.h"

#include "_near_interface_eval.h"  // the features this run asked for (momwire#1860)
#include "_sin_pair_reuse_core.h"  // the reuse plans' counters (momwire#1911)

#include "_sinusoidal_run_core.h"
#if MW_ROUTE_NEC2
#include "_nec2_printout_core.h"  // the NEC-2 portal's printout (momwire#2065)
#endif
#define MW_ROUTE_ENGINE sinusoidal
#define MW_ROUTE_BASIS "sinusoidal"
#include "_somm_below_fill_core.h"
#include "_somm_grid_core.h"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <vector>

#include "blas.h"

extern "C" {
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
void MWN(zgesv_)(const int *, const int *, void *, const int *, int *, void *, const int *, int *);
}

namespace mw {
namespace exe {

namespace {

using nec5::np_int;

int i32(np_int v) { return static_cast<int>(v); }

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
void a_zgesv(const np_int *n, const np_int *nrhs, void *a, const np_int *lda, np_int *ipiv, void *b,
             const np_int *ldb, np_int *info) {
    const int nn = i32(*n), nr = i32(*nrhs), la = i32(*lda), lb = i32(*ldb);
    std::vector<int> piv(static_cast<std::size_t>(std::max(nn, 1)));
    int inf = 0;
    MWN(zgesv_)(&nn, &nr, a, &la, piv.data(), b, &lb, &inf);
    for (int i = 0; i < nn; i++) ipiv[i] = piv[static_cast<std::size_t>(i)];
    *info = inf;
}

sinusoidal::FillLinalg linalg() {
    sinusoidal::FillLinalg la;
    la.lu.getrf = reinterpret_cast<razor::zgetrf_fn>(&MWX(zgetrf_));
    la.lu.getrs = reinterpret_cast<razor::zgetrs_fn>(&MWX(zgetrs_));
    la.np.dgemm = &a_dgemm;
    la.np.dgemv = &a_dgemv;
    la.np.ddot = &a_ddot;
    la.np.zgemm = &a_zgemm;
    la.np.zgemv = &a_zgemv;
    la.np.zdotu_sub = &a_zdotu;
    la.np.zgesv = &a_zgesv;
    la.np.zgesvx = &MWX(zgesvx_);  // the network solve's expert driver (SciPy's)
    return la;
}

}  // namespace

// The Sommerfeld store, as route.cpp and route_bs2.cpp keep it: the
// masters a GN 2 / GN 3 deck's fill reads (`somm_core::get_grid`) and the
// below/below table a buried deck's fill reads (momwire#1607 B6a), each
// kept on disk between runs. A stored value is the fill's, bit for bit, so
// a warm run prints what a cold one does; without it every run refilled
// both (exe warm == exe cold, the perf-1028 study's finding 3).
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

// This route's reuse is the sinusoidal fill's (`pair_reuse`, momwire#1911):
// its plans, not razor2p's.
void reuse_plans(long long &built, long long &declined, long long &found) {
    const mw::pair_reuse::Counters c = sinusoidal::pair_reuse::counters();
    built = static_cast<long long>(c.built);
    declined = static_cast<long long>(c.declined);
    found = static_cast<long long>(c.found);
}

std::string reuse_outcome() { return sinusoidal::pair_reuse::last_outcome(); }

void configure_grid_cache(const std::string &tag) {
    somm_core::configure_disk_cache(tag);
    somm_below::configure_below_store(true);
    somm_below::set_source(somm_below::fill_source());
}

long long largest_solve(const std::string &latin1_text, bool open_end, long long *crossing_pairs,
                        long long *below_pairs) {
    // the size policy reads n_basis alone here
    if (crossing_pairs) *crossing_pairs = 0;
    if (below_pairs) *below_pairs = 0;
    return nec4::largest_solve(latin1_text, MW_ROUTE_BASIS, open_end);
}

bool answer(const std::string &latin1_text, const std::string &version, const std::string &variant,
            const std::string &arrays_dir, std::string &printout, std::string &reason, bool *failed,
            std::string *failure_kind, std::string *failure_message, std::string *raised_kind,
            bool open_end, bool *served) {
    (void)arrays_dir;
    if (served) *served = false;
    if (failed) *failed = false;
    if (raised_kind) raised_kind->clear();
    const sinusoidal::FillLinalg la = linalg();
    nec5::EngineStamp stamp;
    stamp.version = version;
    stamp.variant = variant;
    stamp.marker = ni_eval::requested_stamp();  // `_features.stamp()`'s tail
    if (!line2_note().empty()) stamp.marker += (stamp.marker.empty() ? "" : " ") + line2_note();
    std::string out, why;
    bool ok, go_on = false;
    try {
        ok = open_end ? MW_ROUTE_ENGINE::printout_piece(latin1_text, stamp, la, out, why, go_on)
                      : MW_ROUTE_ENGINE::printout(latin1_text, stamp, la, out, why);
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
}

#if MW_ROUTE_NEC2
namespace {

// One engine per deck: the point-matched sinusoidal basis over its build.
class SinEngines : public nec4::EngineFactory {
  public:
    explicit SinEngines(const sinusoidal::FillLinalg &la) : la_(la) {}
    nec4::RunEngine &engine_for(const nec4::StructureBuild &structure) override {
        engine_.reset(new sinusoidal::Engine(structure, la_));
        return *engine_;
    }

  private:
    sinusoidal::FillLinalg la_;
    std::unique_ptr<sinusoidal::Engine> engine_;
};

}  // namespace

bool nec2_render(const std::string &latin1_body, int end_cap, std::vector<std::string> &out,
                 std::vector<std::string> &err, std::string &reason, bool *failed, std::string *failure_kind,
                 std::string *failure_message) {
    if (failed) *failed = false;
    const sinusoidal::FillLinalg la = linalg();
    nec2::DeckOptions options;
    options.basis = MW_ROUTE_BASIS;
    options.nec2_end_cap = end_cap;
    try {
        SinEngines engines(la);
        return nec2::render_deck(latin1_body, options, engines, la.np, out, err, reason);
    } catch (const std::bad_alloc &) {
        if (failed) *failed = true;
        if (failure_kind) *failure_kind = "MemoryError";
        if (failure_message) *failure_message = "std::bad_alloc";
        reason = "an allocation failed";
    } catch (const std::exception &e) {
        if (failed) *failed = true;
        if (failure_kind) *failure_kind = "RuntimeError";
        if (failure_message) *failure_message = e.what();
        reason = e.what();
    }
    out.clear();
    err.clear();
    return false;
}

long long nec2_largest_solve(const std::string &latin1_body) { return nec2::largest_solve(latin1_body); }
#endif

}  // namespace exe
}  // namespace mw
