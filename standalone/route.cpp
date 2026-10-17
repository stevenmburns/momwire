// The C++ route (route.h): compiled once per namespace, see there.
#include "route.h"

#include "_somm_below_fill_core.h"

#include "_nec5_razor_printout_core.h"
#include "_nec5_razor_seam_core.h"
#include "_razor_solve_core.h"
#include "_somm_grid_core.h"

#include <algorithm>
#include <cstdio>
#include <utility>
#include <vector>

#include "blas.h"

namespace mw {
namespace exe {

void configure_grid_cache(const std::string &tag) {
    somm_core::configure_disk_cache(tag);
    // The below/below table (momwire#1512 B5): filled here, kept in the same
    // store, its anchors' seen list beside it.
    somm_below::configure_below_store(true);
    somm_below::set_source(somm_below::fill_source());
}

long long largest_solve(const std::string &latin1_text, bool open_end) {
    const nec5::SeamResult r = nec5::razor_seam(latin1_text, open_end);
    if (r.route != nec5::SeamResult::served) return -1;
    long long n = 0;
    for (const nec5::Prepared &p : r.prepared)
        if (nec5::razor_part(p).front) n = std::max<long long>(n, nec5::razor_part(p).front->n_basis);
    return n;
}

namespace {

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

}  // namespace

bool answer(const std::string &latin1_text, const std::string &version, const std::string &variant,
            const std::string &arrays_dir, std::string &printout, std::string &reason, bool *failed,
            std::string *failure_kind, std::string *failure_message, std::string *raised_kind,
            bool open_end, bool *served) {
    if (served) *served = false;
    razor::Lapack lp;
    lp.getrf = reinterpret_cast<razor::zgetrf_fn>(&MWX(zgetrf_));
    lp.getrs = reinterpret_cast<razor::zgetrs_fn>(&MWX(zgetrs_));
    nec5::NumpyLinalg np;
    np.host.zgemm = &MWX(zgemm_);
    np.host.zgemv = &MWX(zgemv_);
    np.host.getrf = lp.getrf;
    np.host.getrs = lp.getrs;
    np.zgesvx = &MWX(zgesvx_);  // the network solve's expert driver
    nec5::EngineStamp stamp;
    stamp.version = version;
    stamp.variant = variant;
    razor::SolveContext ctx;
    std::vector<nec5::RunArrays> arrays;
    const nec5::PrintoutResult r =
        nec5::razor_printout(latin1_text, "razor-2p", stamp, ctx, lp, np,
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
}

}  // namespace exe
}  // namespace mw
