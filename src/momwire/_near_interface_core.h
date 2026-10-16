// The near-interface column twin and plane-sheet interpolator, Python-free
// (momwire#1512 B4a): `_near_interface_core.cpp`; `_near_interface_accel.cpp`
// binds them (`near_interface_six_columns`, `near_interface_grid_sheet`,
// `ColumnCache`), and the C++ crossing fill calls them directly.
#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>

namespace ni_core {

// The cooperative-cancellation abort (the binding registers it as the
// module's `AcceleratorAborted`).
struct Aborted : std::exception {
    const char *what() const noexcept override { return "accelerator solve aborted"; }
};

// The fill-held head + mid cache of the column twin (see the core TU): a
// value-neutral cache, bounded at `max_bytes`.
class ColumnCache {
  public:
    explicit ColumnCache(size_t max_bytes);
    ~ColumnCache();
    size_t hits() const;
    size_t misses() const;
    size_t evictions() const;
    size_t entries() const;
    size_t bytes() const;
    size_t max_bytes() const;
    struct Impl;
    std::unique_ptr<Impl> impl;
};

// `_near_interface.six_columns`' twin over `nc` exact-rho columns: column c's
// members are [offsets[c], offsets[c+1]) of (z, zp); out is (n, 6) row-major
// in KEYS order. Throws std::invalid_argument on a bad member (the walk's
// words), Aborted on cancellation. lanes: the rule's Bessel pairs and factors
// four nodes at a time where the build has them (momwire#1441 step 3); false
// is the scalar walk, bit for bit the same.
void six_columns(double k_p, std::complex<double> k_m, const double *rho,
                 const std::ptrdiff_t *offsets, std::ptrdiff_t nc, const double *z,
                 const double *zp, std::ptrdiff_t n, double lam_mult, int p, double detour,
                 int n_threads, const double *gx, const double *gw, int ng,
                 uintptr_t cancel_flag, ColumnCache *cache, std::complex<double> *out,
                 bool lanes = true);

// `PlaneSheet.interpolate`: out[idx[r]] (out (m, nk) row-major) = the table's
// kernels at rows sub[idx[r]] (sub (m, 3)).
void grid_sheet(const double *sub, std::ptrdiff_t m, const std::ptrdiff_t *idx, std::ptrdiff_t n,
                double fixed, bool height, const double *rho_edges, std::ptrdiff_t nr,
                const double *s_edges, std::ptrdiff_t ns, const std::int64_t *cell_off,
                const double *x, const double *bw, int p, const std::complex<double> *vals,
                std::complex<double> *out, int n_threads, const std::int64_t *rpow, int nk,
                bool generic);

// `near_interface_hankel_conj_mismatches`' count (TEST-ONLY) and
// `six_point`'s twin per triple (`near_interface_six_batch`, out (n, 6)).
int64_t hankel_conj_mismatches(const std::complex<double> *w, std::ptrdiff_t n);
void six_batch(double k_p, std::complex<double> k_m, const double *rho, const double *z,
               const double *zp, std::ptrdiff_t n, double rtol, double lam_mult, int depth,
               double detour, const double *gx, const double *gw, int ng,
               std::complex<double> *out);

// Whether this build interpolates six- and four-key tables on the fixed-width
// lanes (`grid_sheet_fixed_1290`).
bool grid_sheet_fixed();

}  // namespace ni_core
