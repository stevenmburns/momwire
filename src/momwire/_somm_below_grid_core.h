// The below/below Sommerfeld grid (`_sommerfeld_below.SommerfeldGridBelow`)
// as the C++ buried fill reads it (momwire#1512), with no Python.
//
// What is here now (B2): the grid's state as the projection reads it -- its
// fifteen regions (3 R1 zones x 5 theta bands, lazily filled) and the edges
// that route a query -- its cache key (`get_grid_below`: eps~, k2, the
// radius bucket `_r1_bucket_below_wl`, omega, mu, the spacing), the region
// set `_ensure_for` fills, and a SOURCE of grids. The fill itself (the
// layout, `iv_surfaces_direct_below`, the anchors of momwire#1471 and the
// executable's store) is B5's, and lands after the weighted-averages tail;
// until then the only source is the TEST BRIDGE, which reads the grids the
// Python route filled for the same deck (`scripts/core_split/
// buried_z_compare.py` writes them), so the stages after the table can be
// gated whole. The executable never installs the bridge.
#pragma once

#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace somm_below {

using cd = std::complex<double>;

constexpr int N_REGIONS = 15;
constexpr int N_BANDS = 5;
// `_SOMM_BELOW_R1_CAP_LAMBDA_M`, `_SOMM_BELOW_SPACING_FILL`.
constexpr double R1_CAP_LAMBDA_M = 4.0;
constexpr double SPACING_FILL[3] = {2.0, 8.0, 1.0};

struct Region {
    double r0 = 0, dr = 0, th0 = 0, dth = 0;  // th in radians
    int64_t n_r = 0, n_th = 0;
    bool filled = false;
    std::vector<cd> vals;  // (4, n_r, n_th) C order; NaN while unfilled
};

struct Key {
    cd eps_t;
    double k2 = 0, r1b_wl = 0, omega = 0, mu = 0;
    double spacing[3] = {0, 0, 0};
    bool operator==(const Key &o) const;
};

struct Grid {
    Key key;
    bool blended = false;
    cd k_m;
    double lam_m = 0, r1_max = 0, r1_cap = 0, r_break = 0, r_near = 0;
    double th_min = 0, th_band_floor_hi = 0, th_band_lo_hi = 0, th_band_hi = 0, th_split = 0;
    Region regions[N_REGIONS];
};

// `k_medium(eps_t, k2)`: k2 * numpy's scalar csqrt(eps~) (numpy's unfused
// scalar product by k2 + 0j), conjugated onto Im <= 0.
cd k_medium(cd eps_t, double k2);
// `_lambda_m`: 2 pi / CPython abs(k_m).
double lambda_m(cd eps_t, double k2);
// `_r1_bucket_below_wl(r1_max, lam_m)`: `_somm_r1_bucket_wl(min(r1_max /
// lam_m, R1_CAP_LAMBDA_M))`, CPython's math.log / float pow.
double r1_bucket_below_wl(double r1_max, double lam_m);
// `get_grid_below`'s key for a request (the exact eps~; the radius only
// through its bucket).
Key key_for(cd eps_t, double k2, double r1_max, double omega, double mu,
            const double spacing[3]);

// Which regions `_ensure_for(mx_r1, mn_th, mx_th)` fills, zone-major in its
// order (region index = zone * N_BANDS + band).
std::vector<int> regions_for(const Grid &g, double mx_r1, double mn_th, double mx_th);

class Source {
  public:
    virtual ~Source() = default;
    // `get_grid_below(eps_t, k2, r1_max, omega, mu, spacing=...)`.
    virtual Grid &get(cd eps_t, double k2, double r1_max, double omega, double mu,
                      const double spacing[3]) = 0;
    // `grid._ensure_for(mx_r1, mn_th, mx_th)`: every region it names filled.
    virtual void ensure_for(Grid &g, double mx_r1, double mn_th, double mx_th) = 0;
};

// The process's grid source: nullptr until one is installed (the buried fill
// then declines "below-table").
Source *source();
void set_source(std::shared_ptr<Source> s);

// The test bridge over DIR's grid files (one per Python grid; see
// buried_z_compare.py's `dump_below_grids`). `ensure_for` only checks that the
// regions it names were filled by the Python route (else std::runtime_error).
std::shared_ptr<Source> bridge_source(const std::string &dir);

}  // namespace somm_below
