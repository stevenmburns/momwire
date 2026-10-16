// The Sommerfeld interpolation grid with no Python (momwire#1479 Sommerfeld
// S2): `_sommerfeld.get_grid` and what it builds — the r1 bucket and the
// eps ladder's rung pair, the frequency-independent normalized masters
// (`SommerfeldGrid.__init__`: the region layout, `iv_surfaces_direct` over
// the six-integral kernel and the R1 = 0 limits), the rung pair's `blend`
// and the physical `scaled_to` view. Every value is the Python grid's, bit
// for bit: Python's complex scalars are CPython 3.12's arithmetic, numpy's
// arrays its fused complex product (`_cmul`'s formula) and its divide.
//
// The module constants are the shipped defaults (`_SOMM_R1_CAP_LAMBDA` 15,
// `_SOMM_R1_NEAR_LAMBDA` 4, `_SOMM_EPS_IM_BUCKET` 0.10,
// `_SOMM_R1_SWEEP_SLACK` 0.05); a caller that sees any of their
// `MOMWIRE_SOMM_*` environment overrides set must not use this.
#pragma once

#include <filesystem>

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

namespace somm_core {

using cd = std::complex<double>;

struct GridRegion {  // one of `SommerfeldGrid._regions`
    double r0 = 0, dr = 0;
    int64_t n_r = 0;
    double th0 = 0, dth = 0;  // radians
    int64_t n_th = 0;
    std::vector<cd> vals;  // (4, n_r, n_th), C order: IrhoV, IzV, IrhoH, IphiH
};

struct Grid {  // `SommerfeldGrid`
    cd eps_t, lattice_eps;
    double k2 = 0, omega = 0, mu = 0;
    double r1_max = 0, r_break = 0, r_near = 0, r0_fill = 0;
    std::vector<GridRegion> regions;
};

// `SommerfeldGrid(eps_t, k2, r1_max, rtol, omega, mu, lattice_eps=...)`.
Grid fill_grid(cd eps_t, double k2, double r1_max, double rtol, double omega, double mu,
               cd lattice_eps);

// `SommerfeldGrid.blend` / `.scaled_to`.
Grid blend(const Grid &a, const Grid &b, double t, cd eps_t);
Grid scaled_to(const Grid &g, double k2, double omega, double mu);

// `_sommerfeld.get_grid(eps_t, k2, r1_max, omega, mu)`, with this process's
// own memo of masters and views (the in-process caches' role; values are
// the fill's either way).
const Grid &get_grid(cd eps_t, double k2, double r1_max, double omega, double mu);

// `_sommerfeld.far_cpp_pack(grid)`: `_far_constants` in `_FAR_PACK_ORDER`
// (eps, k1, c2, c1k, ap, res_fv, g2k1, k2), empty for free space. numpy's
// complex scalar sqrt / arcsin / cos are glibc's csqrt / casin / ccos and its
// scalar products plain (measured, AVX2 and AVX-512).
std::vector<cd> far_pack(const Grid &g);

// The executable's own persistent store of the normalized masters (off
// until configured; the extension and the drivers never configure it).
// `tag` is the executable's identity (its version and build); the
// directory is MOMWIRE_SOMM_CACHE_DIR, else the platform cache dir's
// momwire/sommerfeld, then exe/<tag>. MOMWIRE_SOMM_CACHE=0 (off / false /
// no) leaves it off; MOMWIRE_SOMM_CACHE_MAX_MB (default 512) bounds it.
void configure_disk_cache(const std::string &tag);
struct DiskStats {
    uint64_t hits = 0, misses = 0, writes = 0, rejected = 0;
};
DiskStats disk_stats();
// The configured store's directory and tag (false while it is off), and its
// LRU eviction after a write -- for the below/below table's files, which
// share the directory and the size bound (`_somm_below_fill_core`).
bool disk_location(std::filesystem::path &dir, std::string &tag);
void disk_evict(const std::filesystem::path &keep);

// CPython 3.12's complex division (`_Py_c_quot`), for a caller spelling a
// Python complex expression (C2 = (eps~ - 1) / (eps~ + 1)).
cd py_quot(cd a, cd b);

// `_somm_r1_bucket_wl` and `_somm_eps_rungs` (exposed for the tests).
double r1_bucket_wl(double r1_wl);
void eps_rungs(cd eps_t, cd &lo, cd &hi, double &t);

}  // namespace somm_core
