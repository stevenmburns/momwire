// The transmitted Sommerfeld family with no Python: see
// `_somm_transmitted_core.h`.
//
// Python reads CPython's scalar arithmetic (`math.*`, `_Py_c_prod` /
// `_Py_c_quot`) and numpy's array arithmetic (the fused complex product,
// `CDOUBLE_divide`, libm's exp / cos / sin / cexp); each line here is
// spelled as the Python line it copies computes it.
#include "_somm_transmitted_core.h"

#include "_fma_inline.h"
#include "_leggauss_table.h"
#include "_libm_complex_inline.h"
#include "_mw_core.h"
#include "_somm_below_grid_core.h"
#include "_somm_grid_core.h"
#include "_somm_transmitted_six_inline.h"
#include "_mw_times.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <tuple>

namespace somm_trans {

namespace {

constexpr double PI = 3.141592653589793;  // `np.pi` / `math.pi`

// The served domain (`_sommerfeld_transmitted`'s module constants).
constexpr double ZPRIME_MAX_LAMBDA_M = 0.25;
constexpr double LADDER_PER_DECADE = 18.0, LADDER_PER_QUARTER = 8.0;
constexpr int64_t LADDER_MIN = 9;
constexpr double R_STEP_LOG = 0.10, DTH_DEG = 3.0;
constexpr int64_t PAD_ROWS = 2;
constexpr double R_MIN_LAMBDA_P = 0.001, R_MIN_FLOOR_LAMBDA_P = 1e-6, R_CAP_LAMBDA_P = 2.0;
constexpr int MAX_TAIL_PANELS_T = 6000;
constexpr double TAIL_PANELS_PER_COT = 16.0;

// The contour's machines (`_sommerfeld_below`'s, shared).
constexpr int GAUSS_N = 24, GAUSS_N_COARSE = 16;
constexpr int ADAPT_DEPTH = 16, ADAPT_DEPTH_COARSE = 12;
constexpr double DETOUR = 2.0, DETOUR_COARSE = 1.2;

// `_below_interface`'s cross-block order rule.
constexpr int64_t MAX_NEAR_Q_FACTOR = 16;

// ---- CPython's complex scalars and numpy's complex arrays -------------------

cd py_mul(cd a, cd b) {  // `_Py_c_prod` (a float operand is (x, 0.0))
    return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}
// numpy's array product: each part one fused multiply-add (x86-64's SIMD
// loop and arm64's contracted plain C alike); a real operand is (x, 0).
cd xmul(cd a, cd b) {
    return cd(mw_fma::exact(a.real(), b.real(), -(a.imag() * b.imag())),
              mw_fma::exact(a.real(), b.imag(), a.imag() * b.real()));
}
inline cd c_add(cd a, cd b) { return cd(a.real() + b.real(), a.imag() + b.imag()); }
// `CDOUBLE_divide`, numpy's plain C (contracted on arm64, `MW_NUMPY_CONTRACTS`)
cd xdiv(cd a, cd b) {
    const double in1r = a.real(), in1i = a.imag(), in2r = b.real(), in2i = b.imag();
    const double in2r_abs = std::fabs(in2r), in2i_abs = std::fabs(in2i);
    if (in2r_abs >= in2i_abs) {
        if (in2r_abs == 0 && in2i_abs == 0) return cd(in1r / in2r_abs, in1i / in2i_abs);
        const double rat = in2i / in2r;
#if MW_NUMPY_CONTRACTS
        const double scl = 1.0 / mw_fma::exact(in2i, rat, in2r);
        return cd(mw_fma::exact(in1i, rat, in1r) * scl, mw_fma::exact(-in1r, rat, in1i) * scl);
#else
        const double scl = 1.0 / (in2r + in2i * rat);
        return cd((in1r + in1i * rat) * scl, (in1i - in1r * rat) * scl);
#endif
    }
    const double rat = in2r / in2i;
#if MW_NUMPY_CONTRACTS
    const double scl = 1.0 / mw_fma::exact(in2r, rat, in2i);
    return cd(mw_fma::exact(in1r, rat, in1i) * scl, mw_fma::exact(in1i, rat, -in1r) * scl);
#else
    const double scl = 1.0 / (in2i + in2r * rat);
    return cd((in1r * rat + in1i) * scl, (in1i * rat - in1r) * scl);
#endif
}

// `_c1_moment(omega, mu)`: -1j * omega * mu / (4 pi), CPython left to right.
cd c1_moment(double omega, double mu) {
    return somm_core::py_quot(py_mul(py_mul(cd(-0.0, -1.0), cd(omega, 0.0)), cd(mu, 0.0)), cd(4.0 * PI, 0.0));
}

// CPython's `abs(complex)`.
double py_abs(cd z) { return std::hypot(z.real(), z.imag()); }

// `momwire._quadrature.leggauss(n)`: the frozen table's full rule.
void gauss(int n, std::vector<double> &x, std::vector<double> &w) {
    using namespace mw::leggauss_table;
    const int lo = OFFSET[n - 1], hi = OFFSET[n];
    x.clear();
    w.clear();
    for (int i = 0; i < n / 2; i++) {
        x.push_back(-NODES[hi - 1 - i]);
        w.push_back(WEIGHTS[hi - 1 - i]);
    }
    for (int k = lo; k < hi; k++) {
        x.push_back(NODES[k]);
        w.push_back(WEIGHTS[k]);
    }
}

struct Machines {
    std::vector<double> gx, gw, gxc, gwc;
    Machines() {
        gauss(GAUSS_N, gx, gw);
        gauss(GAUSS_N_COARSE, gxc, gwc);
    }
};
const Machines &machines() {
    static const Machines m;
    return m;
}

}  // namespace

// ---- the served domain ------------------------------------------------------

double r_min_bucket(bool has_r_min, double r_min, double lam_p) {
    const double dflt = R_MIN_LAMBDA_P * lam_p;
    if (!has_r_min) return dflt;
    if (r_min >= dflt) return dflt;
    const double floor = R_MIN_FLOOR_LAMBDA_P * lam_p;
    if (r_min < floor) throw Refused("transmitted-r-min-floor");
    const double steps = std::ceil(std::log(dflt / r_min) / R_STEP_LOG);  // math.ceil: an int
    return dflt * std::exp(-steps * R_STEP_LOG);
}

double grazing_floor(double r_max, double zp_min) {
    const double cot_max = static_cast<double>(MAX_TAIL_PANELS_T) / TAIL_PANELS_PER_COT;
    const double s = 1.0 / cot_max - std::fabs(zp_min) / r_max;
    return std::asin(std::min(std::max(s, 0.0), 1.0));
}

void grid_extent(double k2, double r_max, double zp_min, bool has_r_min, double r_min, double &rmin,
                 double &rmax, double &th_floor) {
    const double lam_p = (2.0 * PI) / k2;
    const double rb = somm_core::r1_bucket_wl(std::min(r_max / lam_p, R_CAP_LAMBDA_P)) * lam_p;
    rmin = r_min_bucket(has_r_min, r_min, lam_p);
    const double lo = 4.0 * rmin;
    rmax = std::min(lo > rb ? lo : rb, R_CAP_LAMBDA_P * lam_p);
    th_floor = grazing_floor(rmax, zp_min);
}

std::vector<double> ladder_nodes(double zp_min, double zp_max, double lam_m) {
    const double lo = std::fabs(zp_min), hi = std::fabs(zp_max);
    if (lo <= 0.0 || hi < lo) throw Refused("transmitted-ladder-range");
    if (hi <= lo * (1.0 + 1e-9)) return {lo};
    const double decades = std::log10(hi / lo);
    const double quarters = hi / (0.25 * lam_m);
    int64_t n = static_cast<int64_t>(std::ceil(LADDER_PER_DECADE * decades + LADDER_PER_QUARTER * quarters)) + 1;
    n = std::max(n, LADDER_MIN);
    // np.exp(np.linspace(log(lo), log(hi), n)): y = i * step + start, the
    // last node `stop` itself
    const double start = std::log(lo), stop = std::log(hi);
    const double step = (stop - start) / static_cast<double>(n - 1);
    std::vector<double> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; i++) {
        const double y = (i == n - 1) ? stop : static_cast<double>(i) * step + start;
        out[static_cast<size_t>(i)] = std::exp(y);
    }
    return out;
}

// ---- t_surfaces_direct ------------------------------------------------------

void t_surfaces(cd eps_t, double k2, double omega, double mu, const double *rho, const double *z, double zp,
                size_t n, cd *out, uintptr_t cancel_flag) {
    const double k_p = k2;
    const cd k_m = somm_below::k_medium(eps_t, k_p);
    // `_six_integrals_transmitted_many(..., rtol=1e-9)` on the accelerated
    // route: one contour per node, OpenMP across nodes (`dynamic`, the tail
    // cost runs as cot theta), as `transmitted_six_integrals_batch` runs them
    const Machines &mc = machines();
    const double rtol_fine = std::min(1e-9, 1e-11);
    std::vector<cd> six(6 * n);
    {
        MW_CANCEL_SETUP(cancel_flag);
        const std::ptrdiff_t nn = static_cast<std::ptrdiff_t>(n);
#pragma omp parallel for schedule(dynamic)
        for (std::ptrdiff_t i = 0; i < nn; ++i) {
            MW_CANCEL_POLL();
            mw568_trans::SixResultT r;
            mw568_trans::six_transmitted_one(rho[i], z[i], zp, k_p, k_m, false, rtol_fine, ADAPT_DEPTH, DETOUR,
                                             mc.gx.data(), mc.gw.data(), GAUSS_N, false, rtol_fine * 100.0,
                                             ADAPT_DEPTH_COARSE, DETOUR_COARSE, mc.gxc.data(), mc.gwc.data(),
                                             GAUSS_N_COARSE, MAX_TAIL_PANELS_T, r);
            for (int c = 0; c < 6; ++c) six[6 * static_cast<size_t>(i) + static_cast<size_t>(c)] = r.val[c];
        }
        MW_THROW_IF_ABORTED();
    }
    const cd cm = c1_moment(omega, mu);
    const cd neg_cm(-cm.real(), -cm.imag());
    const double azp = std::fabs(zp);
    for (size_t i = 0; i < n; i++) {
        const cd *v = &six[6 * i];
        const cd v_rz = v[0], v_zzk = v[1], v_rr = v[2], v_r1 = v[3], v_rzp = v[4], u = v[5];
        // `divide_out_transmitted`: exp(-1j (k_m |z'| + k_p R)) / R over
        // R = sqrt(rho^2 + (z + |z'|)^2)
        const double dz = z[i] + azp;
        const double r = std::sqrt(rho[i] * rho[i] + dz * dz);
        const cd arg = c_add(xmul(k_m, cd(azp, 0.0)), cd(k_p * r, 0.0));
        const cd g = xdiv(mw_libm_c::cexp(xmul(cd(-0.0, -1.0), arg)), cd(r, 0.0));
        const cd phase = xdiv(cd(1.0, 0.0), g);
        const cd cp = xmul(cm, phase), np_ = xmul(neg_cm, phase);
        out[0 * n + i] = xmul(cp, v_rz);
        out[1 * n + i] = xmul(cp, v_zzk);
        out[2 * n + i] = xmul(cp, c_add(v_rr, u));
        out[3 * n + i] = xmul(np_, c_add(v_r1, u));
        out[4 * n + i] = xmul(np_, v_rzp);
    }
}

// ---- TransmittedGrid --------------------------------------------------------

namespace {

// `TransmittedGrid`'s ln R axis over [r_min, r_max]: `PAD_ROWS` rows past
// either end. The anchored grid's served nodes (`exact_nodes`) are rows of it.
void exact_r_axis(double r_min, double r_max, double &lnr0, int64_t &n_r) {
    lnr0 = std::log(r_min) - static_cast<double>(PAD_ROWS) * R_STEP_LOG;
    n_r = static_cast<int64_t>(std::ceil(std::log(r_max / r_min) / R_STEP_LOG)) + 1 + 2 * PAD_ROWS;
}

// `TransmittedGrid`'s theta axis, the grazing floor `th_min` to pi/2 in about
// `DTH_DEG` steps, at least four.
void theta_axis(double th_min, int64_t &n_th, double &dth) {
    // `math.degrees` (180 / pi times x) and Python's half-even `round`
    const double deg = (180.0 / PI) * th_min;
    const int64_t n_int = std::max<int64_t>(static_cast<int64_t>(std::nearbyint((90.0 - deg) / DTH_DEG)), 4);
    n_th = n_int + 1;
    dth = (0.5 * PI - th_min) / static_cast<double>(n_int);
}

// `TransmittedGrid(...)`'s layout -- the ladder and the (ln R, theta)
// lattice -- with no node filled: `vals` NaN, `have` clear.
Grid layout_grid(cd eps_t, double k2, double r_max, double zp_min, double zp_max, double omega, double mu,
                 bool has_r_min, double r_min) {
    Grid g;
    g.eps_t = eps_t;
    g.k2 = k2;
    g.omega = omega;
    g.mu = mu;
    g.k_m = somm_below::k_medium(eps_t, k2);
    g.lam_p = (2.0 * PI) / k2;
    g.lam_m = (2.0 * PI) / py_abs(g.k_m);
    g.r_min = r_min_bucket(has_r_min, r_min, g.lam_p);
    g.r_cap = R_CAP_LAMBDA_P * g.lam_p;
    {
        const double lo = 4.0 * g.r_min;
        const double a = lo > r_max ? lo : r_max;  // Python's max(r_max, 4 r_min)
        g.r_max = g.r_cap < a ? g.r_cap : a;
    }
    const double zp_top = std::fabs(zp_max);
    if (zp_top > ZPRIME_MAX_LAMBDA_M * g.lam_m * (1.0 + 1e-9)) throw Refused("transmitted-deeper-than-ladder");
    g.zp_nodes = ladder_nodes(zp_min, zp_max, g.lam_m);
    g.n_zp = static_cast<int64_t>(g.zp_nodes.size());
    g.zp_min = g.zp_nodes.front();
    g.zp_max = g.zp_nodes.back();

    exact_r_axis(g.r_min, g.r_max, g.lnr0, g.n_r);
    g.dlnr = R_STEP_LOG;
    g.th_min = grazing_floor(g.r_max, g.zp_min);
    g.th0 = g.th_min;
    theta_axis(g.th_min, g.n_th, g.dth);
    g.lnz0 = std::log(g.zp_min);
    g.dlnz = g.n_zp > 1 ? (std::log(g.zp_max) - g.lnz0) / static_cast<double>(g.n_zp - 1) : 0.0;

    const size_t npl = static_cast<size_t>(g.n_r) * static_cast<size_t>(g.n_th);
    g.vals.assign(static_cast<size_t>(g.n_zp) * 5 * npl, cd(std::nan(""), std::nan("")));
    g.have.assign(static_cast<size_t>(g.n_zp) * npl, 0);
    return g;
}

// The nodes `need` names ((n_zp, n_r, n_th), C order) that `g` does not
// hold, filled: per rung, those nodes' (rho, z) as the whole lattice spells
// them, through one `t_surfaces` call. A node is its own six integrals and
// its own divide-out, so which nodes share the call moves no bit: a node
// filled here is the node the whole-grid fill computed. Returns the count.
size_t fill_nodes(Grid &g, const std::vector<uint8_t> &need, uintptr_t cancel_flag) {
    const size_t nr = static_cast<size_t>(g.n_r), nt = static_cast<size_t>(g.n_th), npl = nr * nt;
    std::vector<double> r_nodes(nr), cth(nt), sth(nt);
    for (size_t i = 0; i < nr; i++) r_nodes[i] = std::exp(g.lnr0 + g.dlnr * static_cast<double>(i));
    for (size_t j = 0; j < nt; j++) {
        const double th = g.th0 + g.dth * static_cast<double>(j);
        cth[j] = mw_libm_c::rcos(th);
        sth[j] = mw_libm_c::rsin(th);
    }
    size_t filled = 0;
    std::vector<size_t> at;
    std::vector<double> rho, zz;
    std::vector<cd> sub;
    for (int64_t k = 0; k < g.n_zp; k++) {
        const size_t base = static_cast<size_t>(k) * npl;
        at.clear();
        rho.clear();
        zz.clear();
        for (size_t q = 0; q < npl; q++) {
            if (!need[base + q] || g.have[base + q]) continue;
            const size_t i = q / nt, j = q % nt;
            const double x = r_nodes[i] * cth[j], y = r_nodes[i] * sth[j];
            at.push_back(q);
            rho.push_back(x >= 0.0 ? x : 0.0);  // np.maximum(x, 0.0)
            zz.push_back(y >= 0.0 ? y : 0.0);
        }
        const size_t n = at.size();
        if (n == 0) continue;
        sub.assign(5 * n, cd(0.0, 0.0));
        t_surfaces(g.eps_t, g.k2, g.omega, g.mu, rho.data(), zz.data(), -g.zp_nodes[static_cast<size_t>(k)], n,
                   sub.data(), cancel_flag);
        cd *plane = g.vals.data() + static_cast<size_t>(k) * 5 * npl;
        for (size_t u = 0; u < n; u++) {
            for (size_t c = 0; c < 5; c++) plane[c * npl + at[u]] = sub[c * n + u];
            g.have[base + at[u]] = 1;
        }
        filled += n;
    }
    return filled;
}

}  // namespace

Grid make_grid(cd eps_t, double k2, double r_max, double zp_min, double zp_max, double omega, double mu,
               bool has_r_min, double r_min, uintptr_t cancel_flag) {
    Grid g = layout_grid(eps_t, k2, r_max, zp_min, zp_max, omega, mu, has_r_min, r_min);
    fill_nodes(g, std::vector<uint8_t>(g.have.size(), 1), cancel_flag);
    return g;
}

std::vector<uint8_t> read_nodes(const Grid &g, const double *above, size_t A, const double *below, size_t B,
                                double ground_z) {
    // `tgrid_eval`'s stencil at every (above, below) pair, from
    // `proj_one_transmitted`'s geometry spelled as it spells it: the same
    // floats, so the same corner (i0, j0, k0) and the same 4 x 4 (x 4) nodes.
    const std::ptrdiff_t nth = static_cast<std::ptrdiff_t>(g.n_th), nr = static_cast<std::ptrdiff_t>(g.n_r);
    const size_t npl = static_cast<size_t>(nr * nth);
    const double half_pi = 0.5 * M_PI;
    std::vector<uint8_t> need(static_cast<size_t>(g.n_zp) * npl, 0);
    const std::ptrdiff_t An = static_cast<std::ptrdiff_t>(A);
#pragma omp parallel
    {
        std::vector<uint8_t> mine(need.size(), 0);
#pragma omp for schedule(static)
        for (std::ptrdiff_t a = 0; a < An; ++a) {
            const double ax = above[3 * a], ay = above[3 * a + 1], az = above[3 * a + 2];
            for (size_t b = 0; b < B; ++b) {
                const double z = az - ground_z;
                const double depth = ground_z - below[3 * b + 2];
                const double dx = ax - below[3 * b];
                const double dy = ay - below[3 * b + 1];
                const double rho = std::hypot(dx, dy);
                const double r_obs = std::hypot(rho, z);
                const double theta = std::atan2(z, rho);
                const double fr = (std::log(std::max(r_obs, 1e-300)) - g.lnr0) / g.dlnr;
                int i0 = (int)std::floor(fr) - 1;
                if (i0 < 0) i0 = 0;
                else if (i0 > (int)nr - 4) i0 = (int)nr - 4;
                double thc = theta;
                if (thc < 0.0) thc = 0.0;
                else if (thc > half_pi) thc = half_pi;
                const double ft = (thc - g.th0) / g.dth;
                int j0 = (int)std::floor(ft) - 1;
                if (j0 < 0) j0 = 0;
                else if (j0 > (int)nth - 4) j0 = (int)nth - 4;
                int k0 = 0, nk = 1;
                if (g.n_zp != 1) {
                    const double fz = (std::log(std::max(depth, 1e-300)) - g.lnz0) / g.dlnz;
                    k0 = (int)std::floor(fz) - 1;
                    if (k0 < 0) k0 = 0;
                    else if (k0 > (int)g.n_zp - 4) k0 = (int)g.n_zp - 4;
                    nk = 4;
                }
                for (int kk = 0; kk < nk; ++kk)
                    for (int i = 0; i < 4; ++i) {
                        uint8_t *row = mine.data() + static_cast<size_t>(k0 + kk) * npl +
                                       static_cast<size_t>(i0 + i) * static_cast<size_t>(nth) + static_cast<size_t>(j0);
                        row[0] = row[1] = row[2] = row[3] = 1;
                    }
            }
        }
#pragma omp critical
        for (size_t q = 0; q < need.size(); q++) need[q] |= mine[q];
    }
    return need;
}

namespace {

using Key = std::tuple<double, double, double, double, double, double, double, double, double>;
std::map<Key, std::unique_ptr<Grid>> g_grids;
std::mutex g_mutex;

// ---- the executable's persistent store (momwire#1752 T1) --------------------
//
// Python keeps this grid in `_somm_disk_cache` (momwire#1224); the executable
// kept it in process only, so every run of a mixed deck (wires on both sides,
// no crossing node) refilled it -- 80 % of `mixed_vert_over_buried8`'s exe
// run, against a Python warm call that reads it from memory. The store is the
// one the above masters and the below table use (`somm_core::disk_location`:
// on only in the executables, keyed by the build's tag, the same LRU bound),
// one file per Python key, exact bits; a file that does not read back whole
// (magic, tag, key, sizes, checksum) is a miss and is overwritten. A hit is
// the fill's own bytes, so the answer moves no bit.
//
// MWTRANSGRID2: the grid is filled node by node as its reads ask
// (`get_grid_reading`), so a file carries each node's held mark after the
// values (NaN where the mark is clear) and the fingerprints of the pair sets
// it has served; a later run that reads more of the lattice fills the rest
// and writes the union back.

namespace fs = std::filesystem;

const char MAGIC[] = "MWTRANSGRID2\n";

uint64_t fnv1a(const std::string &b) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : b) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

template <class T>
void put(std::string &b, const T &v) {
    b.append(reinterpret_cast<const char *>(&v), sizeof(T));
}

template <class T>
bool get(const std::string &b, size_t &at, T &v) {
    if (at > b.size() || b.size() - at < sizeof(T)) return false;
    std::memcpy(&v, b.data() + at, sizeof(T));
    at += sizeof(T);
    return true;
}

uint64_t bits(double x) {
    uint64_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}

double key_at(const Key &k, int i) {
    const double v[9] = {std::get<0>(k), std::get<1>(k), std::get<2>(k), std::get<3>(k), std::get<4>(k),
                         std::get<5>(k), std::get<6>(k), std::get<7>(k), std::get<8>(k)};
    return v[i];
}

// The key's hash names the file and the file carries the key itself, which a
// read checks bit for bit (a collision is a miss). A short name, because nine
// spelled-out keys would be ~160 characters, and under a deep temporary
// directory that passes Windows' 260-character MAX_PATH.
std::string file_name(const Key &k) {
    std::string kb;
    for (int i = 0; i < 9; i++) put(kb, key_at(k, i));
    char buf[40];
    std::snprintf(buf, sizeof buf, "trans-%016llx.grid", static_cast<unsigned long long>(fnv1a(kb)));
    return buf;
}

std::string serialize(const std::string &tag, const Key &k, const Grid &g) {
    std::string b(MAGIC);
    put(b, static_cast<uint64_t>(tag.size()));
    b += tag;
    for (int i = 0; i < 9; i++) put(b, key_at(k, i));
    const double sc[19] = {g.eps_t.real(), g.eps_t.imag(), g.k_m.real(), g.k_m.imag(), g.k2,     g.omega, g.mu,
                           g.lam_p,        g.lam_m,        g.r_min,      g.r_cap,      g.r_max,  g.zp_min,
                           g.zp_max,       g.lnr0,         g.dlnr,       g.th_min,     g.th0,    g.dth};
    for (double x : sc) put(b, x);
    put(b, g.lnz0);
    put(b, g.dlnz);
    put(b, g.n_zp);
    put(b, g.n_r);
    put(b, g.n_th);
    put(b, static_cast<uint64_t>(g.zp_nodes.size()));
    for (double x : g.zp_nodes) put(b, x);
    put(b, static_cast<uint64_t>(g.vals.size()));
    b.append(reinterpret_cast<const char *>(g.vals.data()), g.vals.size() * sizeof(cd));
    put(b, static_cast<uint64_t>(g.have.size()));
    b.append(reinterpret_cast<const char *>(g.have.data()), g.have.size());
    put(b, static_cast<uint64_t>(g.served.size()));
    for (uint64_t h : g.served) put(b, h);
    put(b, fnv1a(b));
    return b;
}

bool deserialize(const std::string &b, const std::string &tag, const Key &k, Grid &g) {
    const size_t m = sizeof(MAGIC) - 1;
    if (b.size() < m + 8 || b.compare(0, m, MAGIC) != 0) return false;
    uint64_t sum;
    std::memcpy(&sum, b.data() + b.size() - sizeof sum, sizeof sum);
    if (fnv1a(b.substr(0, b.size() - sizeof sum)) != sum) return false;
    size_t at = m;
    uint64_t tl;
    if (!get(b, at, tl) || tl != tag.size() || b.size() - at < tl || b.compare(at, tl, tag) != 0) return false;
    at += tl;
    for (int i = 0; i < 9; i++) {
        double y;
        if (!get(b, at, y) || bits(y) != bits(key_at(k, i))) return false;
    }
    double sc[19];
    for (double &x : sc)
        if (!get(b, at, x)) return false;
    g.eps_t = cd(sc[0], sc[1]);
    g.k_m = cd(sc[2], sc[3]);
    g.k2 = sc[4];
    g.omega = sc[5];
    g.mu = sc[6];
    g.lam_p = sc[7];
    g.lam_m = sc[8];
    g.r_min = sc[9];
    g.r_cap = sc[10];
    g.r_max = sc[11];
    g.zp_min = sc[12];
    g.zp_max = sc[13];
    g.lnr0 = sc[14];
    g.dlnr = sc[15];
    g.th_min = sc[16];
    g.th0 = sc[17];
    g.dth = sc[18];
    uint64_t nz, nv;
    if (!get(b, at, g.lnz0) || !get(b, at, g.dlnz) || !get(b, at, g.n_zp) || !get(b, at, g.n_r) ||
        !get(b, at, g.n_th) || !get(b, at, nz))
        return false;
    if (g.n_zp < 1 || g.n_r < 0 || g.n_th < 0 || g.n_r > (1 << 24) || g.n_th > (1 << 24) ||
        nz != static_cast<uint64_t>(g.n_zp) || nz > (b.size() - at) / sizeof(double))
        return false;
    g.zp_nodes.resize(static_cast<size_t>(nz));
    for (double &x : g.zp_nodes)
        if (!get(b, at, x)) return false;
    if (!get(b, at, nv)) return false;
    const uint64_t want = static_cast<uint64_t>(g.n_zp) * 5u * static_cast<uint64_t>(g.n_r) *
                          static_cast<uint64_t>(g.n_th);
    if (nv != want || nv > (b.size() - at) / sizeof(cd)) return false;
    g.vals.resize(static_cast<size_t>(nv));
    std::memcpy(g.vals.data(), b.data() + at, static_cast<size_t>(nv) * sizeof(cd));
    at += static_cast<size_t>(nv) * sizeof(cd);
    uint64_t nh;
    if (!get(b, at, nh) || nh != nv / 5u || nh > b.size() - at) return false;
    g.have.resize(static_cast<size_t>(nh));
    std::memcpy(g.have.data(), b.data() + at, static_cast<size_t>(nh));
    at += static_cast<size_t>(nh);
    uint64_t ns;
    if (!get(b, at, ns) || ns > (b.size() - at) / sizeof(uint64_t)) return false;
    g.served.resize(static_cast<size_t>(ns));
    for (uint64_t &h : g.served)
        if (!get(b, at, h)) return false;
    return at + sizeof(uint64_t) == b.size();
}

std::unique_ptr<Grid> store_fetch(const Key &k) {
    fs::path dir;
    std::string tag;
    if (!somm_core::disk_location(dir, tag)) return nullptr;
    const fs::path path = dir / file_name(k);
    std::ifstream in(path, std::ios::binary);
    if (!in) return nullptr;
    std::stringstream ss;
    ss << in.rdbuf();
    std::unique_ptr<Grid> g(new Grid());
    if (!deserialize(ss.str(), tag, k, *g)) return nullptr;
    std::error_code ec;
    fs::last_write_time(path, fs::file_time_type::clock::now(), ec);  // LRU
    return g;
}

void store_put(const Key &k, const Grid &g) {
    fs::path dir;
    std::string tag;
    if (!somm_core::disk_location(dir, tag)) return;
    const fs::path path = dir / file_name(k);
    const std::string bytes = serialize(tag, k, g);
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) return;
    std::random_device rd;
    char tmpname[64];
    std::snprintf(tmpname, sizeof tmpname, ".%08x%08x.tmp", rd(), rd());
    const fs::path tmp = dir / (path.filename().string() + tmpname);
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return;
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!out) {
            out.close();
            fs::remove(tmp, ec);
            return;
        }
    }
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::remove(tmp, ec);
        return;
    }
    somm_core::disk_evict(path);
}

}  // namespace

namespace {

// Whether `need` names a node `g` does not hold.
bool missing(const Grid &g, const std::vector<uint8_t> &need) {
    for (size_t q = 0; q < need.size(); q++)
        if (need[q] && !g.have[q]) return true;
    return false;
}

// The fingerprints a grid keeps (the newest; an older set is censused again).
constexpr size_t SERVED_KEEP = 32;

void add_served(Grid &g, uint64_t h) {
    g.served.push_back(h);
    if (g.served.size() > SERVED_KEEP) g.served.erase(g.served.begin());
}

// The stored copy's held nodes that `g` lacks, copied in (the file's grid is
// the same key's, so the same lattice). Whether a file was read.
bool store_merge(const Key &k, Grid &g) {
    std::unique_ptr<Grid> f = store_fetch(k);
    if (f == nullptr || f->have.size() != g.have.size() || f->vals.size() != g.vals.size()) return false;
    const size_t npl = static_cast<size_t>(g.n_r) * static_cast<size_t>(g.n_th);
    for (size_t q = 0; q < g.have.size(); q++) {
        if (!f->have[q] || g.have[q]) continue;
        const size_t k0 = q / npl, at = q % npl;
        for (size_t c = 0; c < 5; c++) g.vals[(k0 * 5 + c) * npl + at] = f->vals[(k0 * 5 + c) * npl + at];
        g.have[q] = 1;
    }
    for (uint64_t h : f->served)
        if (std::find(g.served.begin(), g.served.end(), h) == g.served.end()) add_served(g, h);
    return true;
}

// The pair set's fingerprint: its points' exact bytes, counts and the
// interface height. A grid that has served a set holds every node the set
// reads (nodes are never dropped), so a repeat of it skips the census.
uint64_t reads_hash(const double *above, size_t A, const double *below, size_t B, double ground_z) {
    std::string b;
    b.reserve(sizeof(double) * (3 * (A + B) + 1) + 2 * sizeof(uint64_t));
    put(b, static_cast<uint64_t>(A));
    put(b, static_cast<uint64_t>(B));
    put(b, ground_z);
    b.append(reinterpret_cast<const char *>(above), 3 * A * sizeof(double));
    b.append(reinterpret_cast<const char *>(below), 3 * B * sizeof(double));
    return fnv1a(b);
}

// The pairs a fill will project, (above, below) in role order.
struct Reads {
    const double *above;
    size_t A;
    const double *below;
    size_t B;
    double ground_z;
};

// `get_grid_below_above`'s grid with the nodes `reads` names held (every
// node when `reads` is null): the process's memo under the Python key,
// then the store, then the fill of what is still missing, written back.
const Grid &ensure_grid(cd eps_t, double k2, double r_max, double zp_min, double zp_max, double omega, double mu,
                        double r_min, uintptr_t cancel_flag, const Reads *reads) {
    const double lam_p = (2.0 * PI) / k2;
    const double rb_wl = somm_core::r1_bucket_wl(std::min(r_max / lam_p, R_CAP_LAMBDA_P));
    const double r_min_b = r_min_bucket(true, r_min, lam_p);
    const Key key{eps_t.real(), eps_t.imag(), k2, rb_wl, r_min_b, std::fabs(zp_min), std::fabs(zp_max), omega, mu};
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_grids.find(key);
    bool fresh = false;
    if (it == g_grids.end()) {
        std::unique_ptr<Grid> g(new Grid(layout_grid(eps_t, k2, rb_wl * lam_p, zp_min, zp_max, omega, mu, true, r_min_b)));
        it = g_grids.emplace(key, std::move(g)).first;
        fresh = true;
    }
    Grid &g = *it->second;
    const mw_times::Scope timed(mw_times::ground);
    // The store once per process, on the key's first use (as the whole grid
    // was read then); a later call adds only what it fills itself.
    const bool from_disk = fresh && store_merge(key, g);
    const uint64_t h = reads ? reads_hash(reads->above, reads->A, reads->below, reads->B, reads->ground_z) : 0;
    const bool seen = reads && std::find(g.served.begin(), g.served.end(), h) != g.served.end();
    if (!seen) {
        const std::vector<uint8_t> need =
            reads ? read_nodes(g, reads->above, reads->A, reads->below, reads->B, reads->ground_z)
                  : std::vector<uint8_t>(g.have.size(), 1);
        if (missing(g, need)) {
            // Another run may have filled these since this one read the store.
            if (!fresh) store_merge(key, g);
            if (missing(g, need)) {
                mw_times::count_built();
                fill_nodes(g, need, cancel_flag);
                if (reads) add_served(g, h);
                store_put(key, g);
                return g;
            }
        }
        if (reads) {
            // Held already (the store's or an earlier call's nodes): kept so
            // the next run of these pairs skips the census too.
            add_served(g, h);
            store_put(key, g);
        }
    }
    if (from_disk) mw_times::count_disk();
    return g;
}

}  // namespace

const Grid &get_grid(cd eps_t, double k2, double r_max, double zp_min, double zp_max, double omega, double mu,
                     double r_min, uintptr_t cancel_flag) {
    return ensure_grid(eps_t, k2, r_max, zp_min, zp_max, omega, mu, r_min, cancel_flag, nullptr);
}

const Grid &get_grid_reading(cd eps_t, double k2, double r_max, double zp_min, double zp_max, double omega,
                             double mu, double r_min, const double *above, size_t A, const double *below,
                             size_t B, double ground_z, uintptr_t cancel_flag) {
    const Reads r{above, A, below, B, ground_z};
    return ensure_grid(eps_t, k2, r_max, zp_min, zp_max, omega, mu, r_min, cancel_flag, &r);
}

// ---- the anchored grid (momwire#1847, `transmitted_anchors`) ----------------
//
// `_transmitted_anchors.get_grid`: a sweep point's below->above grid as the
// degree-7 Lagrange blend, in ln f, of exact grids (ANCHORS) tabulated at
// F_REF * 1.1**j on a lattice fixed in metres, each at the soil's own eps_t
// there, then resampled (ten Lagrange points per axis) onto the nodes the
// exact grid has at this request (momwire#2022). Each line spells the Python
// line it copies. CPython's `math.*` are libm's. The numpy calls on the
// lattice (`np.exp` of the z' ladder's `np.linspace`, `np.log` of the served
// ladder, and the anchor fill's `np.exp` of the R nodes and `np.sin` /
// `np.cos` of the theta nodes, which `fill_nodes` shares with the exact grid)
// agree with the C++ only where the box's libm makes them: numpy's AVX-512
// exp and log differ from libm's in the last bit on a few per cent of
// arguments, and gcc fuses `fill_nodes`' sine and cosine into `sincos`,
// whose sine glibc 2.35 does not keep equal to `sin` (2.39 does). Where
// either differs, the two grids agree to 2e-13 of scale (the declared drift
// is 3e-8) rather than bit for bit; the tests probe both.
//
// The anchors fill node by node as the reads ask (`fill_nodes`, as
// `get_grid_reading` fills the exact grid): the served nodes the pairs read,
// the lattice nodes their resampling stencils read, and the anchor nodes the
// blend of those reads. Every node is its own arithmetic, so each one held
// is the whole grid's bit for bit. The lattice is fixed in metres, so a
// deck's pairs read the same lattice nodes at every frequency of a sweep, and
// a later point in the same process fills no anchor node its stencil already
// holds.
//
// NOT ported: `_somm_disk_cache`'s copy of each anchor. The anchors live in
// this process only, so an executable run once per frequency fills its eight
// anchors on every run.

namespace {

constexpr double ANCHOR_F_REF_HZ = 1.0e6;
constexpr int64_t ANCHOR_POINTS = 8;
// `_RESAMPLE_POINTS`, `_INNER_ROWS`, `_OUTER_ROWS`: see the Python module.
constexpr int64_t RESAMPLE_POINTS = 10;
constexpr int64_t INNER_ROWS = PAD_ROWS + RESAMPLE_POINTS / 2;
constexpr int64_t OUTER_ROWS = 3;
constexpr int Q_DIGITS = 12;
constexpr int64_t LADDER_BUCKETS[] = {9, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512};
constexpr size_t ANCHORS_MAX = 96, BLENDS_MAX = 64;
constexpr double C_LIGHT = 299792458.0;  // `_sommerfeld._C_LIGHT`

// libm's exp and log at run time: a call the compiler folds is its own
// correctly rounded value, which libm's need not be.
double libm_exp(double x) {
    volatile double v = x;
    return std::exp(v);
}

// `math.log(1.1)`
double anchor_step() {
    static const double s = [] {
        volatile double x = 1.1;
        return std::log(x);
    }();
    return s;
}

// `_f_anchor(j)`
double f_anchor(int64_t j) { return ANCHOR_F_REF_HZ * std::exp(anchor_step() * static_cast<double>(j)); }

// `_eps_at(eps_r, q, omega)`
cd eps_at(double eps_r, double q, double omega) { return cd(eps_r, -q / omega); }

// `k_medium(_eps_at(eps_r, q, 2 pi f), 2 pi f / C)` at anchor j's frequency.
cd anchor_k_m(double eps_r, double q, double fj) {
    const double om = 2.0 * PI * fj;
    return somm_below::k_medium(eps_at(eps_r, q, om), om / C_LIGHT);
}

int64_t ladder_bucket(int64_t n) {
    for (int64_t b : LADDER_BUCKETS)
        if (n <= b) return b;
    return -1;
}

// `_weights(u, js)` over the n integer nodes j0 .. j0 + n - 1: exactly 1 and
// 0s when u is a node.
void lagrange(double u, int64_t j0, int64_t n, double *w) {
    for (int64_t a = 0; a < n; a++) {
        double p = 1.0;
        for (int64_t b = 0; b < n; b++)
            if (b != a) p *= (u - static_cast<double>(j0 + b)) / static_cast<double>(a - b);
        w[a] = p;
    }
}

// `_Lattice` (the metres-fixed lattice every anchor and blend of a family
// tabulates on) or `_Nodes` (the exact grid's nodes at one request): the
// axes `_set_lattice` copies.
struct Lattice {
    double r_min = 0, r_max = 0, lnr0 = 0, th_min = 0, dth = 0;
    int64_t n_r = 0, n_th = 0;
    std::vector<double> zp_nodes;
};

// `_Lattice(r_min, r_max, zp_nodes)`: `INNER_ROWS` rows inside r_min and
// none past r_max.
Lattice make_lattice(double r_min, double r_max, std::vector<double> zp_nodes) {
    Lattice L;
    L.r_min = r_min;
    L.r_max = r_max;
    L.zp_nodes = std::move(zp_nodes);
    L.lnr0 = std::log(L.r_min) - static_cast<double>(INNER_ROWS) * R_STEP_LOG;
    L.n_r = static_cast<int64_t>(std::floor(std::log(L.r_max / L.r_min) / R_STEP_LOG + 1e-9)) + 1 + INNER_ROWS;
    L.th_min = grazing_floor(L.r_max, L.zp_nodes[0]);
    theta_axis(L.th_min, L.n_th, L.dth);
    return L;
}

// What one request blends: its soil curve, stencil, weights and lattice.
struct Plan {
    double eps_r = 0, q = 0, u = 0;
    int64_t js[ANCHOR_POINTS] = {};
    double w[ANCHOR_POINTS] = {};
    Lattice lat;
};

// `_lattice(...)`: false (and `why`) where Python returns None.
bool lattice(double f_hz, const Plan &p, double r_max, double zp_min, double zp_max, bool has_r_min,
             double r_min, Lattice &out, std::string &why) {
    const double lam_p = C_LIGHT / f_hz;
    const double octave_top =
        ANCHOR_F_REF_HZ * std::pow(2.0, std::ceil(std::log2(f_hz / ANCHOR_F_REF_HZ) - 1e-12));
    const double lo_default = R_MIN_LAMBDA_P * C_LIGHT / octave_top;
    const double want = has_r_min ? r_min : lo_default;
    // on the log lattice exp(m * R_STEP_LOG) metres, snapped down
    const double r_lo = std::exp(R_STEP_LOG * std::floor(std::log(want) / R_STEP_LOG + 1e-12));
    // max(r_max * exp(_OUTER_ROWS * _R_STEP_LOG), 4 r_lo)
    const double r_out = r_max * libm_exp(static_cast<double>(OUTER_ROWS) * R_STEP_LOG);
    const double r4 = 4.0 * r_lo;
    const double r_hi = somm_core::r1_bucket_wl(r4 > r_out ? r4 : r_out);
    if (r_hi > R_CAP_LAMBDA_P * lam_p) {
        why = "r-cap";
        return false;
    }
    const double lo = std::fabs(zp_min), hi = std::fabs(zp_max);
    std::vector<double> zp_nodes;
    if (hi <= lo * (1.0 + 1e-9)) {
        zp_nodes.push_back(lo);
    } else {
        int64_t need = LADDER_MIN;
        for (int64_t j : p.js) {
            const cd k_m = anchor_k_m(p.eps_r, p.q, f_anchor(j));
            need = std::max(need, static_cast<int64_t>(ladder_nodes(lo, hi, 2.0 * PI / py_abs(k_m)).size()));
        }
        const int64_t n = ladder_bucket(need);
        if (n < 0) {
            why = "ladder-bucket";
            return false;
        }
        // np.exp(np.linspace(log(lo), log(hi), n)), spelled as `ladder_nodes`
        const double start = std::log(lo), stop = std::log(hi);
        const double step = (stop - start) / static_cast<double>(n - 1);
        zp_nodes.resize(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; i++) {
            const double y = (i == n - 1) ? stop : static_cast<double>(i) * step + start;
            zp_nodes[static_cast<size_t>(i)] = std::exp(y);
        }
    }
    // r_lo * exp(-_INNER_ROWS * _R_STEP_LOG): the lattice's innermost row
    const double r_in = r_lo * libm_exp(static_cast<double>(-INNER_ROWS) * R_STEP_LOG);
    for (int64_t j : p.js) {
        const double fj = f_anchor(j);
        const double lam_pj = C_LIGHT / fj;
        const double lam_mj = 2.0 * PI / py_abs(anchor_k_m(p.eps_r, p.q, fj));
        if (r_hi > R_CAP_LAMBDA_P * lam_pj * (1.0 + 1e-9)) {
            why = "anchor-r-cap";
            return false;
        }
        if (r_in < R_MIN_FLOOR_LAMBDA_P * lam_pj) {
            why = "anchor-r-min-floor";
            return false;
        }
        if (hi > ZPRIME_MAX_LAMBDA_M * lam_mj * (1.0 + 1e-9)) {
            why = "anchor-deeper-than-ladder";
            return false;
        }
    }
    out = make_lattice(r_lo, r_hi, std::move(zp_nodes));
    return true;
}

// `get_grid`'s plan up to the blend: false (and `why`) where it returns None.
bool plan(cd eps_t, double k2, double r_max, double zp_min, double zp_max, double omega, bool has_r_min,
          double r_min, Plan &p, std::string &why) {
    const double f_hz = omega / (2.0 * PI);
    if (!(f_hz > 0.0) || eps_t.imag() > 0.0) {
        why = "not-a-soil";
        return false;
    }
    // `_soil`: q = sigma / eps0 to `_Q_DIGITS` significant digits, through
    // the decimal string Python's `f"{q:.11e}"` spells (both correctly rounded)
    p.eps_r = eps_t.real();
    p.q = -eps_t.imag() * omega;
    if (p.q != 0.0) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.*e", Q_DIGITS - 1, p.q);
        p.q = std::strtod(buf, nullptr);
    }
    // `_stencil` and `_weights`: anchor j sits at u = j
    p.u = std::log(f_hz / ANCHOR_F_REF_HZ) / anchor_step();
    const int64_t j0 = static_cast<int64_t>(std::floor(p.u)) - ANCHOR_POINTS / 2 + 1;
    for (int64_t i = 0; i < ANCHOR_POINTS; i++) p.js[i] = j0 + i;
    lagrange(p.u, j0, ANCHOR_POINTS, p.w);
    if (!lattice(f_hz, p, r_max, zp_min, zp_max, has_r_min, r_min, p.lat, why)) return false;
    if (p.lat.r_max > R_CAP_LAMBDA_P * (2.0 * PI / k2)) {
        why = "r-cap";
        return false;
    }
    // the serve plan checked its grazing angle against the EXACT grid's
    // theta floor; a lattice whose floor is higher must not serve (the wide
    // outer radius of a long deck over a very shallow source lands here)
    double rmin_x, rmax_x, th_exact;
    try {
        grid_extent(k2, r_max, zp_min, has_r_min, r_min, rmin_x, rmax_x, th_exact);
    } catch (const Refused &) {
        why = "extent";
        return false;
    }
    if (p.lat.th_min > th_exact * (1.0 + 1e-9) + 1e-15) {
        why = "grazing-floor";
        return false;
    }
    return true;
}

// `_Nodes(blend, k2, r_max, zp_min, zp_max, r_min)`: the rows of the exact
// grid's own axes (`exact_r_axis`, `theta_axis`, `ladder_nodes`, as
// `layout_grid` lays them) that this request can read, inside the lattice.
Lattice exact_nodes(const Grid &blend, double k2, double r_max, double zp_min, double zp_max, bool has_r_min,
                    double r_min) {
    double rmin_e, rmax_e, th_e;
    grid_extent(k2, r_max, zp_min, has_r_min, r_min, rmin_e, rmax_e, th_e);
    double lnr0;
    int64_t n_exact;
    exact_r_axis(rmin_e, rmax_e, lnr0, n_exact);
    Lattice N;
    N.r_min = blend.r_min > rmin_e ? blend.r_min : rmin_e;  // max(rmin_e, blend.r_min)
    N.r_max = r_max < rmax_e ? r_max : rmax_e;              // min(rmax_e, r_max)
    // `tgrid_eval` opens its stencil one row under the query and closes it
    // two above; the 1e-9 holds a query on a row inside whichever way its
    // logarithm rounds
    const double lo = (std::log(N.r_min) - lnr0) / R_STEP_LOG;
    const double hi = (std::log(N.r_max) - lnr0) / R_STEP_LOG;
    const int64_t i_lo = std::max<int64_t>(static_cast<int64_t>(std::floor(lo - 1e-9)) - 1, 0);
    const int64_t i_hi = std::min<int64_t>(static_cast<int64_t>(std::floor(hi + 1e-9)) + 2, n_exact - 1);
    N.lnr0 = lnr0 + static_cast<double>(i_lo) * R_STEP_LOG;
    N.n_r = i_hi - i_lo + 1;
    N.th_min = th_e;
    theta_axis(th_e, N.n_th, N.dth);
    N.zp_nodes = ladder_nodes(zp_min, zp_max, blend.lam_m);
    return N;
}

// One axis of `_resample_matrix(x0, dx, n, targets)`: target t reads the p
// nodes i0[t] .. i0[t] + p - 1 with weights w[t * p ..]; the matrix's other
// entries are exact zeros.
struct Stencil {
    int64_t p = 0;
    std::vector<int64_t> i0;
    std::vector<double> w;
};

Stencil resample_stencil(double x0, double dx, int64_t n, const std::vector<double> &targets) {
    Stencil s;
    s.p = std::min(RESAMPLE_POINTS, n);
    const size_t T = targets.size(), p = static_cast<size_t>(s.p);
    s.i0.resize(T);
    s.w.resize(T * p);
    for (size_t t = 0; t < T; t++) {
        const double f = (targets[t] - x0) / dx;
        const int64_t i = static_cast<int64_t>(std::floor(f)) - s.p / 2 + 1;
        s.i0[t] = std::min(std::max<int64_t>(i, 0), n - s.p);
        lagrange(f, s.i0[t], s.p, &s.w[t * p]);
    }
    return s;
}

// `_set_lattice(g, lat, eps_t, k2, omega, mu)`, no node filled.
Grid lattice_grid(const Lattice &L, cd eps_t, double k2, double omega, double mu) {
    Grid g;
    g.eps_t = eps_t;
    g.k2 = k2;
    g.omega = omega;
    g.mu = mu;
    g.k_m = somm_below::k_medium(eps_t, k2);
    g.lam_p = 2.0 * PI / k2;
    g.lam_m = 2.0 * PI / py_abs(g.k_m);
    g.r_min = L.r_min;
    g.r_cap = R_CAP_LAMBDA_P * g.lam_p;
    g.r_max = L.r_max;
    g.zp_nodes = L.zp_nodes;
    g.n_zp = static_cast<int64_t>(L.zp_nodes.size());
    g.zp_min = L.zp_nodes.front();
    g.zp_max = L.zp_nodes.back();
    g.lnr0 = L.lnr0;
    g.dlnr = R_STEP_LOG;
    g.n_r = L.n_r;
    g.th_min = L.th_min;
    g.th0 = L.th_min;
    g.n_th = L.n_th;
    g.dth = L.dth;
    g.lnz0 = std::log(g.zp_min);
    g.dlnz = g.n_zp > 1 ? (std::log(g.zp_max) - g.lnz0) / static_cast<double>(g.n_zp - 1) : 0.0;
    const size_t npl = static_cast<size_t>(g.n_r) * static_cast<size_t>(g.n_th);
    g.vals.assign(static_cast<size_t>(g.n_zp) * 5 * npl, cd(std::nan(""), std::nan("")));
    g.have.assign(static_cast<size_t>(g.n_zp) * npl, 0);
    return g;
}

// One request's grids: the blend on the family's lattice (`_blend`) and the
// grid served, that blend on the exact grid's nodes (`_on_exact_nodes`),
// with the resampling's three axes. `sz` is empty on one rung, where Python
// takes no z' step.
struct Blend {
    Grid lat, served;
    Stencil sr, st, sz;
    // `_on_exact_nodes`' einsum contracts R first, else theta (`r_first`).
    bool r_first = true;
};

// The pair numpy's greedy `einsum_path` contracts first in
// "zsrt,ar,bt->zsab" (`optimize=True`): the largest size removed, then the
// fewest flops, then the first pair, among the pairs whose result fits the
// largest operand (inputs and output). Either pair's result fits whenever the
// other's does not, so one of the two is always taken.
bool contracts_r_first(int64_t Z, int64_t R, int64_t T, int64_t A, int64_t B) {
    const int64_t V = Z * 5 * R * T, out = Z * 5 * A * B;
    const int64_t limit = std::max({V, A * R, B * T, out});
    const int64_t size_r = Z * 5 * A * T, size_t_ = Z * 5 * R * B;
    const bool fit_r = size_r <= limit, fit_t = size_t_ <= limit;
    if (fit_r != fit_t) return fit_r;
    const int64_t removed_r = V + A * R - size_r, removed_t = V + B * T - size_t_;
    if (removed_r != removed_t) return removed_r > removed_t;
    return A <= B;  // flops 2 Z 5 R T A against 2 Z 5 R T B; a tie is the first pair
}

// One output of numpy's BLAS contraction (`zgemm` over the weight matrix
// cast to complex): the products of the stencil's nodes in index order, each
// fused into the running sum from +0. The matrix's zeros outside the stencil
// add exact zeros, so they move no bit.
inline cd fma_into(cd acc, cd v, double w) {
    return cd(mw_fma::exact(v.real(), w, acc.real()), mw_fma::exact(v.imag(), w, acc.imag()));
}

// The resampling's three axes, from `b.lat`'s onto `b.served`'s targets as
// numpy spells them: x0 + step * np.arange(n), and np.log of the served
// ladder.
void set_stencils(Blend &b) {
    const Grid &l = b.lat, &s = b.served;
    std::vector<double> x(static_cast<size_t>(s.n_r));
    for (size_t i = 0; i < x.size(); i++) x[i] = s.lnr0 + R_STEP_LOG * static_cast<double>(i);
    b.sr = resample_stencil(l.lnr0, l.dlnr, l.n_r, x);
    x.assign(static_cast<size_t>(s.n_th), 0.0);
    for (size_t i = 0; i < x.size(); i++) x[i] = s.th_min + s.dth * static_cast<double>(i);
    b.st = resample_stencil(l.th0, l.dth, l.n_th, x);
    if (s.n_zp > 1) {
        x.assign(static_cast<size_t>(s.n_zp), 0.0);
        for (size_t i = 0; i < x.size(); i++) x[i] = std::log(s.zp_nodes[i]);
        b.sz = resample_stencil(l.lnz0, l.dlnz, l.n_zp, x);
    }
    b.r_first = contracts_r_first(l.n_zp, l.n_r, l.n_th, s.n_r, s.n_th);
}

Blend make_blend(const Lattice &L, cd eps_t, double k2, double omega, double mu, double r_max, double zp_min,
                 double zp_max, bool has_r_min, double r_min) {
    Blend b;
    b.lat = lattice_grid(L, eps_t, k2, omega, mu);
    b.served = lattice_grid(exact_nodes(b.lat, k2, r_max, zp_min, zp_max, has_r_min, r_min), eps_t, k2, omega, mu);
    set_stencils(b);
    return b;
}

// The served nodes `todo` names ((served n_zp, n_r, n_th), C order),
// resampled from the lattice blend, which must hold every node their
// stencils read (`lattice_reads`). Python's order: the first einsum pair
// over its contracted axis, then the other, then (multi-rung) z'.
void resample(Blend &b, const std::vector<uint8_t> &todo) {
    const Grid &l = b.lat;
    Grid &s = b.served;
    const int64_t R = l.n_r, T = l.n_th, A = s.n_r, Bn = s.n_th;
    const int64_t pr = b.sr.p, pt = b.st.p, pz = b.sz.p;
    auto lv = [&](int64_t z, int64_t c, int64_t r, int64_t t) { return l.vals[((z * 5 + c) * R + r) * T + t]; };
    // The two in-plane steps into lattice rung z at served (a, bb): the
    // intermediate is (z, c, a, t) R first, (z, c, r, bb) theta first.
    auto plane = [&](int64_t z, int64_t c, int64_t a, int64_t bb) {
        cd out(0.0, 0.0);
        if (b.r_first) {
            for (int64_t j = 0; j < pt; j++) {
                const int64_t t = b.st.i0[bb] + j;
                cd mid(0.0, 0.0);
                for (int64_t i = 0; i < pr; i++) mid = fma_into(mid, lv(z, c, b.sr.i0[a] + i, t), b.sr.w[a * pr + i]);
                out = fma_into(out, mid, b.st.w[bb * pt + j]);
            }
        } else {
            for (int64_t i = 0; i < pr; i++) {
                const int64_t r = b.sr.i0[a] + i;
                cd mid(0.0, 0.0);
                for (int64_t j = 0; j < pt; j++) mid = fma_into(mid, lv(z, c, r, b.st.i0[bb] + j), b.st.w[bb * pt + j]);
                out = fma_into(out, mid, b.sr.w[a * pr + i]);
            }
        }
        return out;
    };
    const size_t npl = static_cast<size_t>(A) * static_cast<size_t>(Bn);
    if (pz == 0) {  // one rung: the plane is the grid
        for (size_t q = 0; q < todo.size(); q++) {
            if (!todo[q]) continue;
            const int64_t a = static_cast<int64_t>(q / static_cast<size_t>(Bn)), bb = static_cast<int64_t>(q % static_cast<size_t>(Bn));
            for (int64_t c = 0; c < 5; c++) s.vals[(c * A + a) * Bn + bb] = plane(0, c, a, bb);
            s.have[q] = 1;
        }
        return;
    }
    // The in-plane result on every lattice rung a served node's z' stencil
    // reads, once each: (lat n_zp, 5, A, Bn).
    std::vector<uint8_t> want(static_cast<size_t>(l.n_zp) * npl, 0);
    for (size_t q = 0; q < todo.size(); q++) {
        if (!todo[q]) continue;
        const size_t k = q / npl, at = q % npl, z0 = static_cast<size_t>(b.sz.i0[k]);
        for (size_t z = z0; z < z0 + static_cast<size_t>(pz); z++) want[z * npl + at] = 1;
    }
    std::vector<cd> mid(static_cast<size_t>(l.n_zp) * 5 * npl);
    for (size_t q = 0; q < want.size(); q++) {
        if (!want[q]) continue;
        const int64_t z = static_cast<int64_t>(q / npl), a = static_cast<int64_t>((q % npl) / static_cast<size_t>(Bn)),
                      bb = static_cast<int64_t>(q % static_cast<size_t>(Bn));
        for (int64_t c = 0; c < 5; c++) mid[(static_cast<size_t>(z) * 5 + static_cast<size_t>(c)) * npl + q % npl] = plane(z, c, a, bb);
    }
    for (size_t q = 0; q < todo.size(); q++) {
        if (!todo[q]) continue;
        const size_t k = q / npl, at = q % npl;
        for (size_t c = 0; c < 5; c++) {
            cd v(0.0, 0.0);
            for (int64_t m = 0; m < pz; m++) {
                const size_t z = static_cast<size_t>(b.sz.i0[k] + m);
                v = fma_into(v, mid[(z * 5 + c) * npl + at], b.sz.w[k * static_cast<size_t>(pz) + static_cast<size_t>(m)]);
            }
            s.vals[(k * 5 + c) * npl + at] = v;
        }
        s.have[q] = 1;
    }
}

// The lattice nodes ((lat n_zp, n_r, n_th), C order) the resampling of the
// served nodes `todo` reads: each one's stencil on every axis.
std::vector<uint8_t> lattice_reads(const Blend &b, const std::vector<uint8_t> &todo) {
    const Grid &l = b.lat, &s = b.served;
    const size_t T = static_cast<size_t>(l.n_th), R = static_cast<size_t>(l.n_r);
    const size_t Bn = static_cast<size_t>(s.n_th), npl = static_cast<size_t>(s.n_r) * Bn;
    std::vector<uint8_t> need(static_cast<size_t>(l.n_zp) * R * T, 0);
    for (size_t q = 0; q < todo.size(); q++) {
        if (!todo[q]) continue;
        const size_t k = q / npl, a = (q % npl) / Bn, bb = q % Bn;
        const size_t z0 = b.sz.p ? static_cast<size_t>(b.sz.i0[k]) : k, nz = b.sz.p ? static_cast<size_t>(b.sz.p) : 1;
        const size_t r0 = static_cast<size_t>(b.sr.i0[a]), t0 = static_cast<size_t>(b.st.i0[bb]);
        for (size_t z = z0; z < z0 + nz; z++)
            for (size_t r = r0; r < r0 + static_cast<size_t>(b.sr.p); r++) {
                uint8_t *row = need.data() + (z * R + r) * T + t0;
                std::fill(row, row + b.st.p, uint8_t(1));
            }
    }
    return need;
}

// The anchor key, as Python's ("below-above-anchor", eps_r, q, j, lat.key(),
// mu), and the blend's, as (eps_t, k2, omega, mu, lat.key(), r_min or None,
// r_max, zp_min, zp_max) -- rtol is 1e-9 on every C++ route, and None is
// (0, NaN) against (1, r_min).
using AnchorKey = Key;
using BlendKey = std::array<double, 15>;

template <class K, class V>
struct Fifo {  // `_evict_fifo`: the oldest entry out first
    // Shared, not owned: an eviction drops the store's hold only, so an
    // entry a caller (or the stencil being summed) still holds stays alive.
    std::map<K, std::shared_ptr<V>> map;
    std::deque<K> order;
    size_t max;
    explicit Fifo(size_t m) : max(m) {}
    std::shared_ptr<V> find(const K &k) {
        auto it = map.find(k);
        return it == map.end() ? nullptr : it->second;
    }
    std::shared_ptr<V> add(const K &k, V &&g) {
        while (map.size() >= max && !order.empty()) {
            map.erase(order.front());
            order.pop_front();
        }
        order.push_back(k);
        return map[k] = std::make_shared<V>(std::move(g));
    }
    void clear() {
        map.clear();
        order.clear();
    }
};

Fifo<AnchorKey, Grid> g_anchors(ANCHORS_MAX);
Fifo<BlendKey, Blend> g_blends(BLENDS_MAX);
AnchorStats g_anchor_stats;
std::mutex g_anchor_mutex;

std::shared_ptr<const Grid> anchored(cd eps_t, double k2, double r_max, double zp_min, double zp_max, double omega,
                                     double mu, bool has_r_min, double r_min, uintptr_t cancel_flag,
                                     const Reads *reads, AnchorParts *parts, std::string *why_out) {
    Plan p;
    std::string why;
    const bool served = plan(eps_t, k2, r_max, zp_min, zp_max, omega, has_r_min, r_min, p, why);
    std::lock_guard<std::mutex> lock(g_anchor_mutex);
    if (!served) {
        g_anchor_stats.declined++;
        if (why_out) *why_out = why;
        return nullptr;
    }
    const Lattice &L = p.lat;
    const double lk[5] = {L.r_min, L.r_max, L.zp_nodes.front(), L.zp_nodes.back(),
                          static_cast<double>(L.zp_nodes.size())};
    const BlendKey bkey{eps_t.real(), eps_t.imag(), k2,     omega,   mu,
                        lk[0],        lk[1],        lk[2],  lk[3],   lk[4],
                        has_r_min ? 1.0 : 0.0,      has_r_min ? r_min : std::nan(""),
                        r_max,        zp_min,       zp_max};
    std::shared_ptr<Blend> be = g_blends.find(bkey);
    if (be == nullptr) {
        be = g_blends.add(bkey, make_blend(L, eps_t, k2, omega, mu, r_max, zp_min, zp_max, has_r_min, r_min));
        g_anchor_stats.blends++;
    }
    Blend &E = *be;
    Grid &S = E.served, &B = E.lat;
    const std::shared_ptr<const Grid> out(be, &E.served);
    const mw_times::Scope timed(mw_times::ground);
    std::vector<uint8_t> todo_s =
        reads ? read_nodes(S, reads->above, reads->A, reads->below, reads->B, reads->ground_z)
              : std::vector<uint8_t>(S.have.size(), 1);
    bool any_s = false;
    for (size_t q = 0; q < todo_s.size(); q++) {
        todo_s[q] = todo_s[q] && !S.have[q];
        any_s = any_s || todo_s[q];
    }
    if (!any_s && parts == nullptr) return out;
    const std::vector<uint8_t> need = parts ? std::vector<uint8_t>(B.have.size(), 1) : lattice_reads(E, todo_s);
    std::vector<uint8_t> todo(need.size(), 0);
    for (size_t q = 0; q < need.size(); q++) todo[q] = need[q] && !B.have[q];
    // Held, not borrowed: adding anchor k can evict an earlier one of this
    // same stencil (a full store whose oldest entry is a[0]), which a raw
    // pointer would then read freed.
    std::shared_ptr<Grid> a[ANCHOR_POINTS];
    for (int64_t k = 0; k < ANCHOR_POINTS; k++) {
        const AnchorKey akey{p.eps_r, p.q, static_cast<double>(p.js[k]), lk[0], lk[1], lk[2], lk[3], lk[4], mu};
        std::shared_ptr<Grid> g = g_anchors.find(akey);
        if (g == nullptr) {
            const double omega_j = 2.0 * PI * f_anchor(p.js[k]);
            g = g_anchors.add(akey, lattice_grid(L, eps_at(p.eps_r, p.q, omega_j), omega_j / C_LIGHT, omega_j, mu));
            g_anchor_stats.anchor_fills++;
        }
        if (missing(*g, todo)) {
            mw_times::count_built();
            g_anchor_stats.nodes_filled += static_cast<int64_t>(fill_nodes(*g, todo, cancel_flag));
        }
        a[k] = std::move(g);
    }
    // `vals = a._vals * w if vals is None else vals + a._vals * w`, node by
    // node: numpy's complex product by the weight as (w, 0), then the sum
    const size_t npl = static_cast<size_t>(B.n_r) * static_cast<size_t>(B.n_th);
    for (size_t q = 0; q < todo.size(); q++) {
        if (!todo[q]) continue;
        const size_t k0 = q / npl, at = q % npl;
        for (size_t c = 0; c < 5; c++) {
            const size_t i = (k0 * 5 + c) * npl + at;
            cd v = xmul(a[0]->vals[i], cd(p.w[0], 0.0));
            for (int64_t k = 1; k < ANCHOR_POINTS; k++) v = c_add(v, xmul(a[k]->vals[i], cd(p.w[k], 0.0)));
            B.vals[i] = v;
        }
        B.have[q] = 1;
    }
    resample(E, todo_s);
    if (parts) {
        parts->anchors.clear();
        for (int64_t k = 0; k < ANCHOR_POINTS; k++) parts->anchors.push_back(a[k]);
        parts->blend = std::shared_ptr<const Grid>(be, &E.lat);
        parts->r_first = E.r_first;
    }
    return out;
}

}  // namespace

std::shared_ptr<const Grid> anchored_grid(cd eps_t, double k2, double r_max, double zp_min, double zp_max,
                                          double omega, double mu, bool has_r_min, double r_min,
                                          uintptr_t cancel_flag, AnchorParts *parts, std::string *why) {
    return anchored(eps_t, k2, r_max, zp_min, zp_max, omega, mu, has_r_min, r_min, cancel_flag, nullptr, parts,
                    why);
}

std::shared_ptr<const Grid> anchored_grid_reading(cd eps_t, double k2, double r_max, double zp_min, double zp_max,
                                                  double omega, double mu, double r_min, const double *above,
                                                  size_t A, const double *below, size_t B, double ground_z,
                                                  uintptr_t cancel_flag) {
    const Reads r{above, A, below, B, ground_z};
    return anchored(eps_t, k2, r_max, zp_min, zp_max, omega, mu, true, r_min, cancel_flag, &r, nullptr, nullptr);
}

Grid resample_onto(const Grid &lat, double lnr0, int64_t n_r, double th0, double dth, int64_t n_th,
                   const std::vector<double> &zp_nodes, bool *r_first) {
    Blend b;
    b.lat = lat;
    Grid &s = b.served;
    s.lnr0 = lnr0;
    s.dlnr = R_STEP_LOG;
    s.n_r = n_r;
    s.th_min = s.th0 = th0;
    s.dth = dth;
    s.n_th = n_th;
    s.zp_nodes = zp_nodes;
    s.n_zp = static_cast<int64_t>(zp_nodes.size());
    const size_t n = static_cast<size_t>(s.n_zp * s.n_r * s.n_th);
    s.vals.assign(5 * n, cd(std::nan(""), std::nan("")));
    s.have.assign(n, 0);
    set_stencils(b);
    resample(b, std::vector<uint8_t>(n, 1));
    if (r_first) *r_first = b.r_first;
    return std::move(b.served);
}

AnchorStats anchor_stats() {
    std::lock_guard<std::mutex> lock(g_anchor_mutex);
    return g_anchor_stats;
}

void anchor_reset() {
    std::lock_guard<std::mutex> lock(g_anchor_mutex);
    g_anchors.clear();
    g_anchors.max = ANCHORS_MAX;
    g_blends.clear();
    g_anchor_stats = AnchorStats();
}

void anchor_capacity(size_t max) {
    std::lock_guard<std::mutex> lock(g_anchor_mutex);
    g_anchors.max = max;
}

// ---- the projection ---------------------------------------------------------

void project(const Grid &g, const double *above, const double *t_above, size_t A, const double *below,
             const double *t_below, size_t B, double ground_z, double k_p, cd k_m, bool transposed, cd *out,
             Extremes &ext) {
    mw568_trans::TGridView G;
    G.vals = g.vals.data();
    G.n_zp = static_cast<std::ptrdiff_t>(g.n_zp);
    G.n_r = static_cast<std::ptrdiff_t>(g.n_r);
    G.n_th = static_cast<std::ptrdiff_t>(g.n_th);
    if (G.n_r < 4 || G.n_th < 4 || (G.n_zp != 1 && G.n_zp < 4)) throw Refused("transmitted-grid-too-small");
    G.lnr0 = g.lnr0;
    G.dlnr = g.dlnr;
    G.th0 = g.th0;
    G.dth = g.dth;
    G.lnz0 = g.lnz0;
    G.dlnz = g.dlnz;
    G.tiny = 1e-12 * g.r_max;
    G.half_pi = 0.5 * M_PI;
    const mw_contour::cd km(k_m);
    const std::ptrdiff_t AA = static_cast<std::ptrdiff_t>(A > 0 ? A : 1);
    std::vector<double> row_rlo(static_cast<size_t>(AA), std::numeric_limits<double>::infinity());
    std::vector<double> row_rhi(static_cast<size_t>(AA), 0.0);
    std::vector<double> row_thlo(static_cast<size_t>(AA), 0.5 * M_PI);
    std::vector<double> row_thhi(static_cast<size_t>(AA), 0.0);
    std::vector<double> row_zlo(static_cast<size_t>(AA), std::numeric_limits<double>::infinity());
    std::vector<double> row_zhi(static_cast<size_t>(AA), 0.0);
    const std::ptrdiff_t An = static_cast<std::ptrdiff_t>(A), Bn = static_cast<std::ptrdiff_t>(B);
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t a = 0; a < An; ++a) {
        const double ax = above[3 * a], ay = above[3 * a + 1], az = above[3 * a + 2];
        const double ta[3] = {t_above[3 * a], t_above[3 * a + 1], t_above[3 * a + 2]};
        double rlo = std::numeric_limits<double>::infinity(), rhi = 0.0;
        double tlo = 0.5 * M_PI, thi = 0.0;
        double zlo = std::numeric_limits<double>::infinity(), zhi = 0.0;
        for (std::ptrdiff_t b = 0; b < Bn; ++b) {
            const double tb[3] = {t_below[3 * b], t_below[3 * b + 1], t_below[3 * b + 2]};
            const double *to = transposed ? tb : ta;
            const double *ts = transposed ? ta : tb;
            double rq, thq, zq;
            out[a * Bn + b] = mw568_trans::proj_one_transmitted(G, ground_z, k_p, km, ax, ay, az, below[3 * b],
                                                                below[3 * b + 1], below[3 * b + 2], to, ts,
                                                                transposed, rq, thq, zq);
            if (rq < rlo) rlo = rq;
            if (rq > rhi) rhi = rq;
            if (thq < tlo) tlo = thq;
            if (thq > thi) thi = thq;
            if (zq < zlo) zlo = zq;
            if (zq > zhi) zhi = zq;
        }
        row_rlo[static_cast<size_t>(a)] = rlo;
        row_rhi[static_cast<size_t>(a)] = rhi;
        row_thlo[static_cast<size_t>(a)] = tlo;
        row_thhi[static_cast<size_t>(a)] = thi;
        row_zlo[static_cast<size_t>(a)] = zlo;
        row_zhi[static_cast<size_t>(a)] = zhi;
    }
    double mn_r = std::numeric_limits<double>::infinity(), mx_r = 0.0;
    double mn_th = 0.5 * M_PI, mx_th = 0.0;
    double mn_z = std::numeric_limits<double>::infinity(), mx_z = 0.0;
    for (std::ptrdiff_t a = 0; a < An; ++a) {
        const size_t s = static_cast<size_t>(a);
        if (row_rlo[s] < mn_r) mn_r = row_rlo[s];
        if (row_rhi[s] > mx_r) mx_r = row_rhi[s];
        if (row_thlo[s] < mn_th) mn_th = row_thlo[s];
        if (row_thhi[s] > mx_th) mx_th = row_thhi[s];
        if (row_zlo[s] < mn_z) mn_z = row_zlo[s];
        if (row_zhi[s] > mx_z) mx_z = row_zhi[s];
    }
    if (!std::isfinite(mn_r)) mn_r = 0.0;
    if (!std::isfinite(mn_z)) mn_z = 0.0;
    ext.r_lo = mn_r;
    ext.r_hi = mx_r;
    ext.th_lo = mn_th;
    ext.th_hi = mx_th;
    ext.zp_lo = mn_z;
    ext.zp_hi = mx_z;
}

std::string eval_refusal(const Grid &g, const Extremes &e) {
    // `TransmittedGrid.eval(np.array([r_lo, r_hi]), np.array([th_lo,
    // th_hi]), np.array([-zp_lo, -zp_hi]))`'s four refusals, in its order
    const double rmin = std::min(e.r_lo, e.r_hi), rmax = std::max(e.r_lo, e.r_hi);
    const double tmin = std::min(e.th_lo, e.th_hi), tmax = std::max(e.th_lo, e.th_hi);
    const double zmin = std::min(std::fabs(e.zp_lo), std::fabs(e.zp_hi));
    const double zmax = std::max(std::fabs(e.zp_lo), std::fabs(e.zp_hi));
    if (!(rmin == rmin) || !(rmax == rmax) || !(tmin == tmin) || !(tmax == tmax) || !(zmin == zmin) ||
        !(zmax == zmax))
        return "transmitted-eval-nan";
    if (rmin < g.r_min * (1.0 - 1e-9)) return "transmitted-eval-r-min";
    if (rmax > g.r_max * (1.0 + 1e-9)) return "transmitted-eval-r-max";
    if (tmax > 0.5 * PI + 1e-12 || tmin < -1e-12) return "transmitted-eval-theta-range";
    if (tmin < g.th_min * (1.0 - 1e-9)) return "transmitted-eval-theta-floor";
    if (zmin < g.zp_min * (1.0 - 1e-9) || zmax > g.zp_max * (1.0 + 1e-9)) return "transmitted-eval-ladder";
    return std::string();
}

// ---- the plan ---------------------------------------------------------------

CrossPlan cross_extents(const double *obs_a, size_t n_a, const double *obs_b, size_t n_b, double ground_z,
                        double k_p, cd k_m) {
    const double gz = ground_z;
    const double lam_p = (2.0 * PI) / k_p;
    const double lam_m = (2.0 * PI) / py_abs(k_m);
    if (n_a == 0 || n_b == 0) throw Refused("transmitted-empty-class");
    double r_lo = std::numeric_limits<double>::infinity(), r_hi = -std::numeric_limits<double>::infinity();
    double th_cross = std::numeric_limits<double>::infinity();
    bool nan = false;
    for (size_t i = 0; i < n_a; i++) {
        const double z_a = obs_a[3 * i + 2] - gz;
        for (size_t j = 0; j < n_b; j++) {
            const double cdx = obs_a[3 * i] - obs_b[3 * j], cdy = obs_a[3 * i + 1] - obs_b[3 * j + 1];
            const double crho = std::hypot(cdx, cdy);
            const double r_obs = std::hypot(crho, z_a);
            const double th = std::atan2(z_a, crho);
            nan = nan || r_obs != r_obs || th != th;
            r_lo = std::min(r_lo, r_obs);
            r_hi = std::max(r_hi, r_obs);
            th_cross = std::min(th_cross, th);
        }
    }
    double zp_lo = std::numeric_limits<double>::infinity(), zp_hi = -std::numeric_limits<double>::infinity();
    for (size_t j = 0; j < n_b; j++) {
        const double d = gz - obs_b[3 * j + 2];
        nan = nan || d != d;
        zp_lo = std::min(zp_lo, d);
        zp_hi = std::max(zp_hi, d);
    }
    if (nan) throw Refused("transmitted-plan-nan");
    if (r_hi > R_CAP_LAMBDA_P * lam_p) throw Refused("buried-cross-range");
    if (zp_hi > ZPRIME_MAX_LAMBDA_M * lam_m) throw Refused("buried-depth");
    double rmin_eff, r_hi_eff, th_floor;
    grid_extent(k_p, r_hi, zp_lo, true, r_lo, rmin_eff, r_hi_eff, th_floor);
    if (th_cross < th_floor) throw Refused("buried-cross-grazing");
    CrossPlan p;
    p.r_cross_max = r_hi;
    p.r_cross_min = r_lo;
    p.zp_min = zp_lo;
    p.zp_max = zp_hi;
    return p;
}

int64_t near_q_factor(const double *seg_l, const double *seg_r, const std::vector<int64_t> &a_idx,
                      const std::vector<int64_t> &b_idx) {
    if (a_idx.empty() || b_idx.empty()) return 1;
    // `cross_pair_separation`: centres 0.5 * (l + r), `norm3` sqrt((x x + y y) + z z)
    const auto centre = [&](int64_t s, double *c) {
        for (int k = 0; k < 3; k++) c[k] = 0.5 * (seg_l[3 * s + k] + seg_r[3 * s + k]);
    };
    const auto norm3 = [](double x, double y, double z) { return std::sqrt((x * x + y * y) + z * z); };
    double sep = std::numeric_limits<double>::infinity();
    bool nan = false;
    for (int64_t a : a_idx) {
        double ca[3];
        centre(a, ca);
        for (int64_t b : b_idx) {
            double cb[3];
            centre(b, cb);
            const double d = norm3(ca[0] - cb[0], ca[1] - cb[1], ca[2] - cb[2]);
            nan = nan || d != d;
            sep = std::min(sep, d);
        }
    }
    double h_a = -std::numeric_limits<double>::infinity(), h_b = h_a;
    for (int64_t s : a_idx) {
        const double h = norm3(seg_r[3 * s] - seg_l[3 * s], seg_r[3 * s + 1] - seg_l[3 * s + 1],
                               seg_r[3 * s + 2] - seg_l[3 * s + 2]);
        nan = nan || h != h;
        h_a = std::max(h_a, h);
    }
    for (int64_t s : b_idx) {
        const double h = norm3(seg_r[3 * s] - seg_l[3 * s], seg_r[3 * s + 1] - seg_l[3 * s + 1],
                               seg_r[3 * s + 2] - seg_l[3 * s + 2]);
        nan = nan || h != h;
        h_b = std::max(h_b, h);
    }
    if (nan) throw Refused("transmitted-separation-nan");
    const double seg_h = h_b > h_a ? h_b : h_a;  // Python's max(a, b)
    if (sep <= 0.0 || seg_h <= 0.0) return 1;
    const double ratio = seg_h / sep;
    if (ratio <= 1.0 + 1e-9) return 1;
    return std::min(static_cast<int64_t>(std::ceil(ratio - 1e-9)), MAX_NEAR_Q_FACTOR);
}

}  // namespace somm_trans
