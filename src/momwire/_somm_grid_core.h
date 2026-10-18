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
    // The layout's theta lattice in degrees (a node's theta is
    // np.radians(th0_deg + dth_deg * j)), and which nodes hold their value
    // (n_r * n_th, C order; a node not held reads NaN). momwire#1512.
    double th0_deg = 0, dth_deg = 0;
    std::vector<uint8_t> have;
};

struct Grid {  // `SommerfeldGrid`
    cd eps_t, lattice_eps;
    double k2 = 0, omega = 0, mu = 0;
    double r1_max = 0, r_break = 0, r_near = 0, r0_fill = 0;
    std::vector<GridRegion> regions;
    bool lazy = false;  // built by layout_grid: nodes filled as asked
};

// `SommerfeldGrid(eps_t, k2, r1_max, rtol, omega, mu, lattice_eps=...)`:
// `layout_grid` then every node.
Grid fill_grid(cd eps_t, double k2, double r1_max, double rtol, double omega, double mu,
               cd lattice_eps);

// The lazy grid (momwire#1512): the constructor's lattice with no node
// filled (values NaN, masks clear), and the nodes of `need` (a mask per
// region, n_r * n_th; an empty mask asks nothing there) filled in place by
// `iv_surfaces_direct` at `rtol`. A node's value is its own arithmetic, so it
// is the eager fill's bit for bit whichever nodes share the call. Returns
// how many nodes were filled.
Grid layout_grid(cd eps_t, double k2, double r1_max, double omega, double mu, cd lattice_eps);
size_t fill_nodes(Grid &g, const std::vector<std::vector<uint8_t>> &need, double rtol);

// `SommerfeldGrid.blend` / `.scaled_to`.
Grid blend(const Grid &a, const Grid &b, double t, cd eps_t);
Grid scaled_to(const Grid &g, double k2, double omega, double mu);

// `_sommerfeld.get_grid(eps_t, k2, r1_max, omega, mu)`, with this process's
// own memo of masters and views (the in-process caches' role; values are
// the fill's either way). Every node filled.
const Grid &get_grid(cd eps_t, double k2, double r1_max, double omega, double mu);

// The same view with only the nodes asked so far filled (momwire#1512): a
// reader takes its census of the nodes it will read (`field_proj_nodes`,
// `reflected_field_nodes`, ...) into a mask per region of the view, and
// `ensure_nodes` fills them -- the two rung masters' same nodes first (from
// the store when another run filled them, else the fill, then written back),
// then the view's blend and scale, node by node. Views and masters live in
// this process's memo, so the reference stays valid. MOMWIRE_SOMM_LAZY=0
// fills every node at once (the eager fill's work, the same values).
const Grid &get_grid_lazy(cd eps_t, double k2, double r1_max, double omega, double mu);
bool lazy_grids();  // MOMWIRE_SOMM_LAZY: on unless 0 / off / false / no
void ensure_nodes(const Grid &view, const std::vector<std::vector<uint8_t>> &need);

// Whether a reader's census is worth taking (the 0.79.0 pre-tag sweep): a
// census costs per PAIR it reads (~25 ns: the pair geometry and the stencil
// address), and saves at most the fill of the nodes the view's masters do not
// hold yet (`unfilled_nodes`: from 0.14 ms a node, the cheapest measured, to
// 2.7 ms). Where the census would cost over half that saving, the reader
// asks every node instead (`all_nodes_of`) -- the eager fill, the same
// values. `_sommerfeld.census_pays` is the same rule, the same constants.
std::size_t unfilled_nodes(const Grid &view);
bool census_pays(double n_pairs, std::size_t unfilled);
std::vector<std::vector<uint8_t>> all_nodes_of(const Grid &view);

// MOMWIRE_SOMM_LAZY_CHECK set (not 0): every lazy reader's output is checked
// finite after the read and the process aborts on a non-finite value -- with
// every node not filled reading NaN, a node the census missed cannot go
// unseen. The proof that a census is complete; off in normal runs.
bool lazy_check();
void check_read(const cd *v, size_t n, const char *who);

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
