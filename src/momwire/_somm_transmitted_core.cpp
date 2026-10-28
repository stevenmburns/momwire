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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
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

Grid make_grid(cd eps_t, double k2, double r_max, double zp_min, double zp_max, double omega, double mu,
               bool has_r_min, double r_min, uintptr_t cancel_flag) {
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

    g.lnr0 = std::log(g.r_min) - static_cast<double>(PAD_ROWS) * R_STEP_LOG;
    g.dlnr = R_STEP_LOG;
    g.n_r = static_cast<int64_t>(std::ceil(std::log(g.r_max / g.r_min) / R_STEP_LOG)) + 1 + 2 * PAD_ROWS;
    g.th_min = grazing_floor(g.r_max, g.zp_min);
    g.th0 = g.th_min;
    // `math.degrees` (180 / pi times x) and Python's half-even `round`
    const double deg = (180.0 / PI) * g.th_min;
    const int64_t n_int = std::max<int64_t>(static_cast<int64_t>(std::nearbyint((90.0 - deg) / DTH_DEG)), 4);
    g.n_th = n_int + 1;
    g.dth = (0.5 * PI - g.th_min) / static_cast<double>(n_int);

    const size_t nr = static_cast<size_t>(g.n_r), nt = static_cast<size_t>(g.n_th), npl = nr * nt;
    std::vector<double> r_nodes(nr), cth(nt), sth(nt);
    for (size_t i = 0; i < nr; i++) r_nodes[i] = std::exp(g.lnr0 + g.dlnr * static_cast<double>(i));
    for (size_t j = 0; j < nt; j++) {
        const double th = g.th0 + g.dth * static_cast<double>(j);
        cth[j] = mw_libm_c::rcos(th);
        sth[j] = mw_libm_c::rsin(th);
    }
    std::vector<double> rho(npl), zz(npl);
    for (size_t i = 0; i < nr; i++)
        for (size_t j = 0; j < nt; j++) {
            const double x = r_nodes[i] * cth[j], y = r_nodes[i] * sth[j];
            rho[i * nt + j] = x >= 0.0 ? x : 0.0;  // np.maximum(x, 0.0)
            zz[i * nt + j] = y >= 0.0 ? y : 0.0;
        }
    g.vals.assign(static_cast<size_t>(g.n_zp) * 5 * npl, cd(0.0, 0.0));
    for (int64_t k = 0; k < g.n_zp; k++) {
        t_surfaces(g.eps_t, g.k2, g.omega, g.mu, rho.data(), zz.data(), -g.zp_nodes[static_cast<size_t>(k)], npl,
                   g.vals.data() + static_cast<size_t>(k) * 5 * npl, cancel_flag);
    }
    g.lnz0 = std::log(g.zp_min);
    g.dlnz = g.n_zp > 1 ? (std::log(g.zp_max) - g.lnz0) / static_cast<double>(g.n_zp - 1) : 0.0;
    return g;
}

namespace {

using Key = std::tuple<double, double, double, double, double, double, double, double, double>;
std::map<Key, std::unique_ptr<Grid>> g_grids;
std::mutex g_mutex;

}  // namespace

const Grid &get_grid(cd eps_t, double k2, double r_max, double zp_min, double zp_max, double omega, double mu,
                     double r_min, uintptr_t cancel_flag) {
    const double lam_p = (2.0 * PI) / k2;
    const double rb_wl = somm_core::r1_bucket_wl(std::min(r_max / lam_p, R_CAP_LAMBDA_P));
    const double r_min_b = r_min_bucket(true, r_min, lam_p);
    const Key key{eps_t.real(), eps_t.imag(), k2, rb_wl, r_min_b, std::fabs(zp_min), std::fabs(zp_max), omega, mu};
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_grids.find(key);
    if (it != g_grids.end()) return *it->second;
    auto g = std::make_unique<Grid>(
        make_grid(eps_t, k2, rb_wl * lam_p, zp_min, zp_max, omega, mu, true, r_min_b, cancel_flag));
    const Grid &ref = *g;
    g_grids.emplace(key, std::move(g));
    return ref;
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
