// The below/below table filled in C++: see `_somm_below_fill_core.h`.
#include "_somm_below_fill_core.h"

#include "_leggauss_table.h"
#include "_mw_core.h"
#include "_somm_below_six_inline.h"
#include "_somm_grid_core.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <tuple>
#include <unistd.h>
#include <utility>
#include <vector>

namespace somm_below {

namespace {

constexpr double PI = 3.141592653589793;
constexpr double C_LIGHT = 299792458.0;     // `_constants.C_LIGHT`
const double MU0 = 4e-7 * PI;               // `_MU0 = 4e-7 * np.pi`
constexpr double DEG_TO_RAD = PI / 180.0;   // np.radians / math.radians

// The contour's machines (`_sommerfeld_below`'s module constants).
constexpr int GAUSS_N = 24, GAUSS_N_COARSE = 16;
constexpr int ADAPT_DEPTH = 16, ADAPT_DEPTH_COARSE = 12;
constexpr double DETOUR = 2.0, DETOUR_COARSE = 1.2;
constexpr int MAX_TAIL_PANELS = 24000;
// `_tail_rule()`: the weighted-averages stop (momwire#1441).
constexpr int WA_ORDER = 10, WA_MIN_SUMS = 4, WA_AGREE = 2;
const double WA_POW[6] = {-0.5, -0.5, -0.5, -1.5, -2.5, -0.5};
const char TAIL_RULE[] = "wa-1441:10:4:2:-0.5,-0.5,-0.5,-1.5,-2.5,-0.5";

// The lattice (`_SOMM_BELOW_*`).
constexpr double R1_NEAR_CAP_LAMBDA_M = 2.0;
constexpr int FAR_STEEP_CELLS = 72, FAR_GRAZE_CELLS = 72;
constexpr int PAD_ROWS = 2;
constexpr double DR_LAMBDA_M = 0.05, DR_NEAR_LAMBDA_M = 0.01, R_BREAK_LAMBDA_M = 0.2;
constexpr double DTH_GRAZE_DEG = 1.0, DTH_STEEP_DEG = 2.5;
constexpr double DTH_BAND_DEG = 0.225, TH_BAND_HI_DEG = 1.0;
constexpr double DTH_BAND_LO_DEG = 0.05 / 3.0, TH_BAND_LO_HI_DEG = 0.1;
constexpr double TH_BAND_FLOOR_HI_DEG = 0.05;
constexpr int BAND_FLOOR_NODES = 4;
constexpr int FLOOR_SHARED[2][2] = {{2, 0}, {3, 1}};
constexpr double TH_MIN_DEG = 0.05 - 2.0 * DTH_BAND_LO_DEG;
constexpr double TH_SPLIT_DEG = 30.0;  // `_sommerfeld._SOMM_TH_SPLIT_DEG`
constexpr int BAND_FLOOR = 0, BAND_LO = 1, BAND_MID = 2;
constexpr int ZONE_INNER = 0, ZONE_NEAR = 1, ZONE_FAR = 2;

// The anchors (momwire#1471).
const double ANCHOR_STEP = std::log(1.1);  // `math.log(1.1)`
constexpr int ANCHOR_POINTS = 8, ANCHOR_TRIGGER = 8;
constexpr double ANCHOR_WINDOW_STEPS = 2.0, ANCHOR_LAMBDA0_M = 1.0;
constexpr size_t SWEEP_SEEN_MAX = 256;  // `_SWEEP_SEEN_MAX`
constexpr size_t SEEN_MAX_DISK = 64;     // `_somm_disk_cache._SEEN_MAX`

int region_index(int zone, int band) { return zone * N_BANDS + band; }

bool deferred(int idx) {
    const int zone = idx / N_BANDS, band = idx % N_BANDS;
    return zone == ZONE_FAR || band == BAND_FLOOR || band == BAND_LO || band == BAND_MID;
}

// ---- CPython's complex scalars and numpy's complex arrays -------------------

cd py_mul(cd a, cd b) {  // `_Py_c_prod` (a float operand is (x, 0.0))
    return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}
cd py_div(cd a, cd b) { return somm_core::py_quot(a, b); }  // `_Py_c_quot`
cd py_neg(cd a) { return cd(-a.real(), -a.imag()); }

// numpy's array product: each part one fused multiply-add (the run path's
// `np_cmul`, measured on AVX2 and AVX-512 alike); a real operand is (x, 0).
cd np_mul(cd a, cd b) {
    return cd(std::fma(a.real(), b.real(), -(a.imag() * b.imag())),
              std::fma(a.real(), b.imag(), a.imag() * b.real()));
}
inline cd c_add(cd a, cd b) { return cd(a.real() + b.real(), a.imag() + b.imag()); }
inline cd c_sub(cd a, cd b) { return cd(a.real() - b.real(), a.imag() - b.imag()); }

cd np_div(cd a, cd b) {  // `CDOUBLE_divide`
    const double in1r = a.real(), in1i = a.imag(), in2r = b.real(), in2i = b.imag();
    const double in2r_abs = std::fabs(in2r), in2i_abs = std::fabs(in2i);
    if (in2r_abs >= in2i_abs) {
        if (in2r_abs == 0 && in2i_abs == 0) return cd(in1r / in2r_abs, in1i / in2i_abs);
        const double rat = in2i / in2r;
        const double scl = 1.0 / (in2r + in2i * rat);
        return cd((in1r + in1i * rat) * scl, (in1i - in1r * rat) * scl);
    }
    const double rat = in2r / in2i;
    const double scl = 1.0 / (in2i + in2r * rat);
    return cd((in1r * rat + in1i) * scl, (in1i * rat - in1r) * scl);
}

cd np_exp(cd z) {  // numpy's complex exp: libm's cexp
#if defined(__GNUC__)
    __complex__ double c;
    __real__ c = z.real();
    __imag__ c = z.imag();
    c = __builtin_cexp(c);
    return cd(__real__ c, __imag__ c);
#else
    return std::exp(z);
#endif
}

// CPython's `x ** 2` on a complex (`c_powu`: r = 1 * (x * x)).
cd py_square(cd x) { return py_mul(cd(1.0, 0.0), py_mul(x, x)); }

// `_c1_moment(omega, mu)`: -1j * omega * mu / (4 pi), CPython left to right.
cd c1_moment(double omega, double mu) {
    return py_div(py_mul(py_mul(cd(-0.0, -1.0), cd(omega, 0.0)), cd(mu, 0.0)), cd(4.0 * PI, 0.0));
}

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

// Python's `f"{x:.<p>g}"`.
std::string fmt_g(double x, int p) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*g", p, x);
    return buf;
}

// Python's `repr(float)`: the shortest string that reads back, "%r"'s form.
std::string py_repr(double x) {
    if (std::isnan(x)) return "nan";
    if (std::isinf(x)) return x > 0 ? "inf" : "-inf";
    char buf[64];
    for (int p = 1; p <= 17; p++) {
        std::snprintf(buf, sizeof buf, "%.*g", p, x);
        if (std::strtod(buf, nullptr) == x) break;
    }
    std::string s = buf;
    if (s.find_first_of(".en") == std::string::npos) s += ".0";
    return s;
}

// `_refuse_if_capped`'s sentence (`where` the grid node (R1, theta)).
[[noreturn]] void refuse_capped(int panels, double rho, double h, double w_r1, double w_th) {
    const double r1 = std::hypot(rho, h);
    const double th_deg = std::atan2(h, rho) * (180.0 / PI);
    throw TailCapped(
        "the below/below contour exhausted its tail budget at theta = " + fmt_g(th_deg, 4) +
        " deg, R1 = " + fmt_g(r1, 6) + " (rho = " + fmt_g(rho, 6) + ", h = " + fmt_g(h, 6) +
        "): " + std::to_string(panels) + " panels, the whole of _MAX_TAIL_PANELS = " +
        std::to_string(MAX_TAIL_PANELS) +
        ". The panel count grows as ~6.4/tan(theta), so this is a grazing limit: the tabulated "
        "floor is " + py_repr(TH_MIN_DEG) +
        " deg, where the worst SPEC soil uses ~93 % of the budget. Past the cap the contour falls "
        "back to a Wynn extrapolation whose error grows with the distance under the floor "
        "(momwire#935 measured ~3e-9 to ~2e-6 an octave under its own floor), so there is no "
        "honest value to return \u2014 raise the pair's depth sum relative to its horizontal "
        "separation, or refuse the geometry [at (" + py_repr(w_r1) + ", " + py_repr(w_th) + ")]");
}

// `_six_integrals_below_many(eps_t, k2, rho, h, rtol=1e-9)` on the
// accelerator path: one contour per node (`six_below_one`), OpenMP across
// nodes as `below_six_integrals_batch` runs them; `out` is (n, 6).
void six_many(cd eps_t, double k_p, const std::vector<double> &rho, const std::vector<double> &h,
              cd *out, const double *w_r1, const double *w_th, uintptr_t cancel_flag) {
    const int64_t n = static_cast<int64_t>(rho.size());
    if (eps_t.real() == 1.0 && eps_t.imag() == 0.0) {
        std::fill(out, out + 6 * n, cd(0.0, 0.0));
        return;
    }
    const cd k_m = k_medium(eps_t, k_p);
    const Machines &mc = machines();
    const double rtol_fine = std::min(1e-9, 1e-11);
    mw_contour::WaRule wa;
    wa.pow = WA_POW;
    wa.order = WA_ORDER;
    wa.min_sums = WA_MIN_SUMS;
    wa.agree = WA_AGREE;
    std::vector<int> tail(static_cast<size_t>(n));
    std::vector<char> conv(static_cast<size_t>(n));
    {
        MW_CANCEL_SETUP(cancel_flag);
#pragma omp parallel for schedule(dynamic)
        for (int64_t i = 0; i < n; ++i) {
            MW_CANCEL_POLL();
            mw568_below::SixResult r;
            mw568_below::six_below_one(rho[i], h[i], k_p, k_m, rtol_fine, ADAPT_DEPTH, DETOUR,
                                       mc.gx.data(), mc.gw.data(), GAUSS_N, false,
                                       rtol_fine * 100.0, ADAPT_DEPTH_COARSE, DETOUR_COARSE,
                                       mc.gxc.data(), mc.gwc.data(), GAUSS_N_COARSE,
                                       MAX_TAIL_PANELS, &wa, r);
            for (int c = 0; c < 6; ++c) out[6 * i + c] = r.val[c];
            tail[static_cast<size_t>(i)] = r.tail_panels;
            conv[static_cast<size_t>(i)] = r.converged;
        }
        MW_THROW_IF_ABORTED();
    }
    for (int64_t i = 0; i < n; i++)
        if (!conv[static_cast<size_t>(i)])
            refuse_capped(tail[static_cast<size_t>(i)], rho[i], h[i], w_r1[i], w_th[i]);
}

}  // namespace

// ---- iv_surfaces_direct_below ------------------------------------------------

void iv_surfaces(cd eps_t, double k2, double omega, double mu, const double *r1,
                 const double *theta, size_t n, cd *out, uintptr_t cancel_flag) {
    const double k_p = k2;
    const cd k_m = k_medium(eps_t, k_p);
    std::fill(out, out + 4 * n, cd(0.0, 0.0));
    const cd cm = c1_moment(omega, mu);
    // `_limits_r1_zero_below`: `_limits_r1_zero_shared(k_m, k_p, -1.0, ...)`
    {
        const cd ks2 = py_mul(k_m, k_m);
        const double ko2 = k_p * k_p;
        const cd num = cd(ko2 - ks2.real(), 0.0 - ks2.imag());  // ko2 - ks2
        const cd den = cd(ko2 + ks2.real(), 0.0 + ks2.imag());  // ko2 + ks2
        const cd c2 = py_div(num, den);
        const cd c3 = py_div(py_mul(ks2, num), py_square(den));
        const cd ratio = py_div(cd(ko2, 0.0), ks2);
        const cd v_lead = py_mul(py_mul(py_mul(cd(-1.0, 0.0), cm), ratio), c3);  // sign_rz*cm*ratio*c3
        const cd iz = py_mul(py_mul(cm, ratio), c3);
        const cd c2_c3 = c_sub(c2, c3);
        const cd neg_cm = py_neg(cm);
        for (size_t i = 0; i < n; i++) {
            if (r1[i] != 0.0) continue;
            const double s = std::sin(theta[i]), co = std::cos(theta[i]);
            const bool near = std::fabs(co) < 1e-8;
            const double co_safe = near ? 1.0 : co;
            const double q1 = near ? 0.0 : (1.0 - s) / co_safe;
            const double q2 = near ? 0.5 : (1.0 - s) / (co_safe * co_safe);
            out[0 * n + i] = np_mul(v_lead, cd(q1, 0.0));
            out[1 * n + i] = iz;
            out[2 * n + i] = np_mul(cm, c_add(c2_c3, np_mul(c3, cd(q2, 0.0))));
            out[3 * n + i] = np_mul(neg_cm, c_sub(c2, np_mul(c3, cd(q2, 0.0))));
        }
    }
    std::vector<size_t> nz;
    std::vector<double> rho, h, w_r1, w_th;
    for (size_t i = 0; i < n; i++) {
        if (r1[i] == 0.0) continue;
        nz.push_back(i);
        w_r1.push_back(r1[i]);
        w_th.push_back(theta[i]);
        rho.push_back(std::max(r1[i] * std::cos(theta[i]), 0.0));
        h.push_back(std::max(r1[i] * std::sin(theta[i]), 0.0));
    }
    if (nz.empty()) return;
    std::vector<cd> six(6 * nz.size());
    six_many(eps_t, k_p, rho, h, six.data(), w_r1.data(), w_th.data(), cancel_flag);
    const cd km2 = py_mul(k_m, k_m);
    const double kp2 = k_p * k_p;
    const cd ratio = py_div(cd(kp2, 0.0), km2);
    const cd neg_cm = py_neg(cm);
    for (size_t j = 0; j < nz.size(); j++) {
        const size_t i = nz[j];
        const cd *v6 = &six[6 * j];
        const cd v_rr = v6[0], v_zz = v6[1], v_rz = v6[2], v_r1 = v6[3], v = v6[4], u = v6[5];
        // `divide_out_below`: exp(-1j * (k_p rho + k_m hh)) / sqrt(rho^2 + hh^2)
        const double rr = std::sqrt(rho[j] * rho[j] + h[j] * h[j]);
        const cd kmh = np_mul(k_m, cd(h[j], 0.0));
        const cd arg = c_add(cd(k_p * rho[j], 0.0), kmh);
        const cd g = np_div(np_exp(np_mul(cd(-0.0, -1.0), arg)), cd(rr, 0.0));
        const cd phase = np_div(cd(1.0, 0.0), g);
        const cd cp = np_mul(cm, phase);
        out[0 * n + i] = np_mul(np_mul(cp, ratio), v_rz);
        out[1 * n + i] = np_mul(np_mul(cp, ratio), c_add(v_zz, np_mul(km2, v)));
        out[2 * n + i] = np_mul(cp, c_add(v_rr, u));
        out[3 * n + i] = np_mul(np_mul(neg_cm, phase), c_add(v_r1, u));
    }
}

// ---- the grid ---------------------------------------------------------------

namespace {

double scaled_dth(double base_deg, double scale, double width_deg) {
    if (scale == 1.0) return base_deg;
    return width_deg / std::max(1.0, std::nearbyint(width_deg / (base_deg * scale)));
}

double far_cells(int cells, double scale) {
    if (scale == 1.0) return cells;
    return std::max(2.0, std::nearbyint(cells / scale));
}

// Evaluate the (rows x cols) block of nodes r_nodes x th_nodes (meshgrid,
// "ij") into vals (4, rows, cols).
void eval_block(const Grid &g, const std::vector<double> &r_nodes, const std::vector<double> &th_nodes,
                std::vector<cd> &vals, uintptr_t cancel_flag) {
    const size_t nr = r_nodes.size(), nt = th_nodes.size(), n = nr * nt;
    std::vector<double> rr(n), tt(n);
    for (size_t i = 0; i < nr; i++)
        for (size_t j = 0; j < nt; j++) {
            rr[i * nt + j] = r_nodes[i];
            tt[i * nt + j] = th_nodes[j];
        }
    vals.assign(4 * n, cd(0.0, 0.0));
    if (n == 0) return;
    iv_surfaces(g.key.eps_t, g.key.k2, g.key.omega, g.key.mu, rr.data(), tt.data(), n, vals.data(),
                cancel_flag);
}

}  // namespace

namespace {

void th_nodes_of(const Region &reg, double th0_deg, double dth_deg, std::vector<double> &th) {
    th.resize(static_cast<size_t>(reg.n_th));
    for (int64_t j = 0; j < reg.n_th; j++)
        th[j] = (th0_deg + dth_deg * static_cast<double>(j)) * DEG_TO_RAD;
}

}  // namespace

std::unique_ptr<Grid> make_grid(cd eps_t, double k2, double r1_max, double omega, double mu,
                                const double spacing[3], bool eager, const Grid *seed,
                                uintptr_t cancel_flag, bool lazy) {
    if (lazy && seed != nullptr) throw std::logic_error("a lazy below grid takes no seed");
    std::unique_ptr<Grid> gp(new Grid());
    Grid &g = *gp;
    g.lazy = lazy;
    g.key.eps_t = eps_t;
    g.key.k2 = k2;
    g.key.omega = omega;
    g.key.mu = mu;
    for (int i = 0; i < 3; i++) g.key.spacing[i] = spacing[i];
    const double s_dr = spacing[0], s_th = spacing[1], s_band = spacing[2];
    g.k_m = k_medium(eps_t, k2);
    const double lam_m = 2.0 * PI / std::hypot(g.k_m.real(), g.k_m.imag());
    g.lam_m = lam_m;
    const double cap = R1_CAP_LAMBDA_M * lam_m;
    g.r1_max = std::min(std::max(r1_max, 0.35 * lam_m), cap);
    g.r1_cap = cap;
    const cd kmk = cd(g.k_m.real() - k2, g.k_m.imag() - 0.0);
    const double beat = 2.0 * PI / std::max(std::hypot(kmk.real(), kmk.imag()), 1e-30);
    g.r_break = R_BREAK_LAMBDA_M * lam_m;
    g.r_near = std::min(R1_NEAR_CAP_LAMBDA_M * lam_m, g.r1_max);
    g.th_min = TH_MIN_DEG * DEG_TO_RAD;
    const double split = TH_SPLIT_DEG;
    const double dr_in = DR_NEAR_LAMBDA_M * s_dr * lam_m;
    const double dr_out = std::min(DR_LAMBDA_M * s_dr * lam_m, beat / 6.0);
    const double th0 = TH_MIN_DEG, band_hi = TH_BAND_HI_DEG, band_lo_hi = TH_BAND_LO_HI_DEG;
    const double dthb = scaled_dth(DTH_BAND_DEG, s_band, band_hi - band_lo_hi);
    const double band_floor_hi = TH_BAND_FLOOR_HI_DEG, dthb_lo = DTH_BAND_LO_DEG;
    const double floor_top = th0 + dthb_lo * (BAND_FLOOR_NODES - 1);
    const double dthg = scaled_dth(DTH_GRAZE_DEG, s_th, split - band_hi);
    const double dths = scaled_dth(DTH_STEEP_DEG, s_th, 90.0 - split);
    g.th_band_hi = band_hi * DEG_TO_RAD;
    g.th_band_lo_hi = band_lo_hi * DEG_TO_RAD;
    g.th_band_floor_hi = band_floor_hi * DEG_TO_RAD;
    g.th_split = split * DEG_TO_RAD;
    const double dthg_far = (split - band_hi) / far_cells(FAR_GRAZE_CELLS, s_th);
    const double dths_far = (90.0 - split) / far_cells(FAR_STEEP_CELLS, s_th);
    struct L {
        double r0, r1, dr, th0, th1, dth;
    };
    const L layout[N_REGIONS] = {
        {0.0, g.r_break, dr_in, th0, floor_top, dthb_lo},
        {0.0, g.r_break, dr_in, band_floor_hi, band_lo_hi, dthb_lo},
        {0.0, g.r_break, dr_in, band_lo_hi, band_hi, dthb},
        {0.0, g.r_break, dr_in, band_hi, split, dthg},
        {0.0, g.r_break, dr_in, split, 90.0, dths},
        {g.r_break, g.r_near, dr_out, th0, floor_top, dthb_lo},
        {g.r_break, g.r_near, dr_out, band_floor_hi, band_lo_hi, dthb_lo},
        {g.r_break, g.r_near, dr_out, band_lo_hi, band_hi, dthb},
        {g.r_break, g.r_near, dr_out, band_hi, split, dthg},
        {g.r_break, g.r_near, dr_out, split, 90.0, dths},
        {g.r_near, g.r1_max, dr_out, th0, floor_top, dthb_lo},
        {g.r_near, g.r1_max, dr_out, band_floor_hi, band_lo_hi, dthb_lo},
        {g.r_near, g.r1_max, dr_out, band_lo_hi, band_hi, dthb},
        {g.r_near, g.r1_max, dr_out, band_hi, split, dthg_far},
        {g.r_near, g.r1_max, dr_out, split, 90.0, dths_far},
    };
    for (int ridx = 0; ridx < N_REGIONS; ridx++) {
        const L &l = layout[ridx];
        int pad_lo = l.r0 > 0.0 ? PAD_ROWS : 0;
        while (pad_lo > 0 && l.r0 - pad_lo * l.dr <= 1e-9 * l.dr) pad_lo -= 1;
        const double r_start = l.r0 - pad_lo * l.dr;
        const int64_t n_r = std::max<int64_t>(
            static_cast<int64_t>(std::ceil((l.r1 - l.r0) / l.dr)) + 1 + pad_lo + PAD_ROWS, 4);
        const int64_t n_th = static_cast<int64_t>(std::nearbyint((l.th1 - l.th0) / l.dth)) + 1;
        Region &reg = g.regions[ridx];
        reg.r0 = r_start;
        reg.dr = l.dr;
        reg.n_r = n_r;
        reg.th0 = l.th0 * DEG_TO_RAD;
        reg.dth = l.dth * DEG_TO_RAD;
        reg.n_th = n_th;
        g.th0_deg[ridx] = l.th0;
        g.dth_deg[ridx] = l.dth;
        std::vector<double> r_nodes(static_cast<size_t>(n_r)), th_nodes;
        for (int64_t i = 0; i < n_r; i++) r_nodes[i] = r_start + l.dr * static_cast<double>(i);
        th_nodes_of(reg, l.th0, l.dth, th_nodes);
        // `_seed_rows`
        const Region *kept = nullptr;
        if (seed != nullptr) {
            const Region &sr = seed->regions[ridx];
            if (sr.filled && sr.n_th == n_th && std::fabs(sr.r0 - r_start) <= 1e-9 * l.dr &&
                std::fabs(sr.dr - l.dr) <= 1e-12 * l.dr &&
                std::fabs(sr.th0 - l.th0 * DEG_TO_RAD) <= 1e-15 &&
                std::fabs(sr.dth - l.dth * DEG_TO_RAD) <= 1e-15)
                kept = &sr;
        }
        if (lazy) {
            // In the table where the eager fill would fill it now, no node in.
            reg.vals.assign(static_cast<size_t>(4 * n_r * n_th), cd(std::nan(""), std::nan("")));
            reg.have.assign(static_cast<size_t>(n_r * n_th), 0);
            reg.filled = eager && !deferred(ridx);
        } else if (kept != nullptr) {
            const int64_t n_kept = std::min(kept->n_r, n_r);
            reg.vals.assign(static_cast<size_t>(4 * n_r * n_th), cd(0.0, 0.0));
            for (int s = 0; s < 4; s++)
                for (int64_t i = 0; i < n_kept; i++)
                    for (int64_t j = 0; j < n_th; j++)
                        reg.vals[(s * n_r + i) * n_th + j] = kept->vals[(s * kept->n_r + i) * n_th + j];
            if (n_kept < n_r) {
                std::vector<double> more_r(r_nodes.begin() + n_kept, r_nodes.end());
                std::vector<cd> more;
                eval_block(g, more_r, th_nodes, more, cancel_flag);
                const int64_t m = n_r - n_kept;
                for (int s = 0; s < 4; s++)
                    for (int64_t i = 0; i < m; i++)
                        for (int64_t j = 0; j < n_th; j++)
                            reg.vals[(s * n_r + n_kept + i) * n_th + j] = more[(s * m + i) * n_th + j];
            }
            reg.filled = true;
        } else if (deferred(ridx) || !eager) {
            reg.vals.assign(static_cast<size_t>(4 * n_r * n_th),
                            cd(std::nan(""), std::nan("")));
            reg.filled = false;
        } else {
            eval_block(g, r_nodes, th_nodes, reg.vals, cancel_flag);
            reg.filled = true;
        }
    }
    return gp;
}

void fill_region(Grid &g, int idx, uintptr_t cancel_flag) {
    Region &reg = g.regions[idx];
    if (reg.filled) return;
    std::vector<double> r_nodes(static_cast<size_t>(reg.n_r)), th_nodes;
    for (int64_t i = 0; i < reg.n_r; i++) r_nodes[i] = reg.r0 + reg.dr * static_cast<double>(i);
    th_nodes_of(reg, g.th0_deg[idx], g.dth_deg[idx], th_nodes);
    if (idx % N_BANDS == BAND_FLOOR) {
        // The floor band's top nodes ARE the low band's first ones; the low
        // band owns them. A node's value does not depend on its batch, so
        // the low band first and then the floor's own columns are the bits
        // of Python's joint call.
        const int lo_idx = region_index(idx / N_BANDS, BAND_LO);
        Region &lo = g.regions[lo_idx];
        if (lo.n_r != reg.n_r || lo.r0 != reg.r0 || lo.dr != reg.dr)
            throw std::logic_error("the floor and low bands must share an R1 axis");
        fill_region(g, lo_idx, cancel_flag);
        std::vector<int64_t> own;
        for (int64_t j = 0; j < reg.n_th; j++) {
            bool taken = false;
            for (const auto &fs : FLOOR_SHARED) taken = taken || fs[0] == j;
            if (!taken) own.push_back(j);
        }
        std::vector<double> th_own;
        for (int64_t j : own) th_own.push_back(th_nodes[static_cast<size_t>(j)]);
        std::vector<cd> v;
        eval_block(g, r_nodes, th_own, v, cancel_flag);
        const int64_t n_r = reg.n_r, n_th = reg.n_th, n_own = static_cast<int64_t>(own.size());
        reg.vals.assign(static_cast<size_t>(4 * n_r * n_th), cd(0.0, 0.0));
        for (int s = 0; s < 4; s++)
            for (int64_t i = 0; i < n_r; i++) {
                for (int64_t k = 0; k < n_own; k++)
                    reg.vals[(s * n_r + i) * n_th + own[k]] = v[(s * n_r + i) * n_own + k];
                for (const auto &fs : FLOOR_SHARED)
                    reg.vals[(s * n_r + i) * n_th + fs[0]] = lo.vals[(s * lo.n_r + i) * lo.n_th + fs[1]];
            }
        reg.filled = true;
        return;
    }
    eval_block(g, r_nodes, th_nodes, reg.vals, cancel_flag);
    reg.filled = true;
}

// ---- the lazy grid's regions and nodes (momwire#1512) -------------------------

namespace {

// `fill_region`'s table membership without its fill: the floor band brings
// its low band in first, as `_fill_region` fills it first.
bool activate(Grid &g, int idx) {
    Region &reg = g.regions[idx];
    if (reg.filled) return false;
    if (idx % N_BANDS == BAND_FLOOR) activate(g, region_index(idx / N_BANDS, BAND_LO));
    reg.filled = true;
    return true;
}

// The asked nodes of the grid's filled regions not yet in, filled through
// one `iv_surfaces_direct_below` call (a node's value is its own contour, so
// the call's other nodes move no bit); the floor band's shared columns are
// the low band's nodes (`_fill_region`'s copy), filled there and copied.
size_t fill_lazy(Grid &g, const std::vector<std::vector<uint8_t>> &need, uintptr_t cancel_flag) {
    std::vector<std::pair<int, size_t>> at;
    std::vector<double> rr, tt;
    std::vector<std::vector<uint8_t>> queued(N_REGIONS);
    auto queue = [&](int r, size_t q) {
        Region &reg = g.regions[r];
        if (reg.have[q]) return;
        if (queued[r].empty()) queued[r].assign(reg.have.size(), 0);
        if (queued[r][q]) return;
        queued[r][q] = 1;
        const int64_t i = static_cast<int64_t>(q) / reg.n_th, j = static_cast<int64_t>(q) % reg.n_th;
        at.emplace_back(r, q);
        rr.push_back(reg.r0 + reg.dr * static_cast<double>(i));
        tt.push_back((g.th0_deg[r] + g.dth_deg[r] * static_cast<double>(j)) * DEG_TO_RAD);
    };
    struct Copy {
        int r;
        size_t q;
        int lo;
        size_t q_lo;
    };
    std::vector<Copy> copies;
    for (int r = 0; r < N_REGIONS; r++) {
        Region &reg = g.regions[r];
        if (!reg.filled || static_cast<size_t>(r) >= need.size() || need[r].empty()) continue;
        if (need[r].size() != reg.have.size())
            throw std::logic_error("below fill_nodes: a mask is not the region's");
        const bool floor = r % N_BANDS == BAND_FLOOR;
        const int lo_idx = region_index(r / N_BANDS, BAND_LO);
        for (size_t q = 0; q < reg.have.size(); q++) {
            if (!need[r][q] || reg.have[q]) continue;
            const int64_t i = static_cast<int64_t>(q) / reg.n_th, j = static_cast<int64_t>(q) % reg.n_th;
            int shared = -1;
            if (floor)
                for (const auto &fs : FLOOR_SHARED)
                    if (fs[0] == j) shared = fs[1];
            if (shared < 0) {
                queue(r, q);
            } else {
                const Region &lo = g.regions[lo_idx];
                if (!lo.filled || lo.n_r != reg.n_r || lo.r0 != reg.r0 || lo.dr != reg.dr)
                    throw std::logic_error("the floor and low bands must share an R1 axis");
                const size_t q_lo = static_cast<size_t>(i * lo.n_th + shared);
                queue(lo_idx, q_lo);
                copies.push_back(Copy{r, q, lo_idx, q_lo});
            }
        }
    }
    if (!at.empty()) {
        const size_t n = at.size();
        std::vector<cd> out(4 * n);
        iv_surfaces(g.key.eps_t, g.key.k2, g.key.omega, g.key.mu, rr.data(), tt.data(), n, out.data(),
                    cancel_flag);
        for (size_t p = 0; p < n; p++) {
            Region &reg = g.regions[at[p].first];
            const size_t m = reg.have.size(), q = at[p].second;
            for (int s = 0; s < 4; s++) reg.vals[s * m + q] = out[s * n + p];
            reg.have[q] = 1;
        }
    }
    for (const Copy &c : copies) {
        Region &reg = g.regions[c.r];
        const Region &lo = g.regions[c.lo];
        const size_t m = reg.have.size(), m_lo = lo.have.size();
        for (int s = 0; s < 4; s++) reg.vals[s * m + c.q] = lo.vals[s * m_lo + c.q_lo];
        reg.have[c.q] = 1;
    }
    return at.size() + copies.size();
}

// A grid read from the store into the lazy form: masks everywhere, a filled
// region with no mask (an eager file's) held whole.
void give_masks(Grid &g) {
    g.lazy = true;
    for (Region &reg : g.regions)
        if (reg.have.empty()) reg.have.assign(static_cast<size_t>(reg.n_r * reg.n_th), reg.filled ? 1 : 0);
}

}  // namespace

// ---- get_grid_below, the anchors, the store ----------------------------------

namespace {

namespace fs = std::filesystem;

struct Store {
    bool on = false;
    fs::path dir;
    std::string tag;
};
Store g_store;

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

// A store key: the grid's identity, exact bits (an anchor's size slot says
// "anchor", as Python's key does).
struct SKey {
    double v[9];  // eps re, eps im, k2, r1b (NaN: anchor), omega, mu, spacing x3
    std::string name(const char *kind) const {
        std::string s = kind;
        char buf[24];
        for (double x : v) {
            std::snprintf(buf, sizeof buf, "-%016llx", static_cast<unsigned long long>(bits(x)));
            s += buf;
        }
        return s;
    }
};

// MWBELOWGRID1: a filled region's every node (the eager table's files).
// MWBELOWGRID2 (momwire#1512): a region's node mask after its flag when the
// grid is lazy, then its values (NaN where the mask is clear). Both read;
// 2 is written.
const char MAGIC1[] = "MWBELOWGRID1\n";
const char MAGIC2[] = "MWBELOWGRID2\n";

std::string serialize(const SKey &k, const Grid &g) {
    std::string b(MAGIC2);
    const std::string id = g_store.tag + "|" + TAIL_RULE;
    put(b, static_cast<uint64_t>(id.size()));
    b += id;
    for (double x : k.v) put(b, x);
    const double sc[12] = {g.lam_m, g.r1_max, g.r1_cap, g.r_break, g.r_near, g.th_min,
                           g.th_band_floor_hi, g.th_band_lo_hi, g.th_band_hi, g.th_split,
                           g.k_m.real(), g.k_m.imag()};
    for (double x : sc) put(b, x);
    for (int r = 0; r < N_REGIONS; r++) {
        const Region &reg = g.regions[r];
        put(b, reg.r0);
        put(b, reg.dr);
        put(b, reg.th0);
        put(b, reg.dth);
        put(b, g.th0_deg[r]);
        put(b, g.dth_deg[r]);
        put(b, reg.n_r);
        put(b, reg.n_th);
        put(b, static_cast<int64_t>(reg.filled));
        put(b, static_cast<int64_t>(!reg.have.empty()));
        if (!reg.have.empty()) b.append(reinterpret_cast<const char *>(reg.have.data()), reg.have.size());
        if (reg.filled) b.append(reinterpret_cast<const char *>(reg.vals.data()), reg.vals.size() * sizeof(cd));
    }
    put(b, fnv1a(b));
    return b;
}

bool deserialize(const std::string &b, const SKey &k, Grid &g) {
    const size_t m = sizeof(MAGIC2) - 1;
    static_assert(sizeof(MAGIC1) == sizeof(MAGIC2), "one magic length");
    if (b.size() < m + 8) return false;
    const bool v2 = b.compare(0, m, MAGIC2) == 0;
    if (!v2 && b.compare(0, m, MAGIC1) != 0) return false;
    uint64_t sum;
    std::memcpy(&sum, b.data() + b.size() - sizeof sum, sizeof sum);
    if (fnv1a(b.substr(0, b.size() - sizeof sum)) != sum) return false;
    size_t at = m;
    const std::string id = g_store.tag + "|" + TAIL_RULE;
    uint64_t tl;
    if (!get(b, at, tl) || tl != id.size() || b.compare(at, tl, id) != 0) return false;
    at += tl;
    for (double x : k.v) {
        double y;
        if (!get(b, at, y) || bits(x) != bits(y)) return false;
    }
    double sc[12];
    for (double &x : sc)
        if (!get(b, at, x)) return false;
    g.lam_m = sc[0];
    g.r1_max = sc[1];
    g.r1_cap = sc[2];
    g.r_break = sc[3];
    g.r_near = sc[4];
    g.th_min = sc[5];
    g.th_band_floor_hi = sc[6];
    g.th_band_lo_hi = sc[7];
    g.th_band_hi = sc[8];
    g.th_split = sc[9];
    g.k_m = cd(sc[10], sc[11]);
    g.lazy = false;
    for (int r = 0; r < N_REGIONS; r++) {
        Region &reg = g.regions[r];
        int64_t filled, masked = 0;
        if (!get(b, at, reg.r0) || !get(b, at, reg.dr) || !get(b, at, reg.th0) || !get(b, at, reg.dth) ||
            !get(b, at, g.th0_deg[r]) || !get(b, at, g.dth_deg[r]) || !get(b, at, reg.n_r) ||
            !get(b, at, reg.n_th) || !get(b, at, filled))
            return false;
        if (v2 && !get(b, at, masked)) return false;
        if (reg.n_r < 0 || reg.n_th < 0 || reg.n_r > (1 << 24) || reg.n_th > (1 << 24)) return false;
        const size_t nn = static_cast<size_t>(reg.n_r * reg.n_th), nv = 4 * nn;
        reg.filled = filled != 0;
        reg.have.clear();
        if (masked) {
            if (nn > b.size() - at) return false;
            reg.have.assign(b.data() + at, b.data() + at + nn);
            at += nn;
            g.lazy = true;
        }
        if (reg.filled) {
            if (nv > (b.size() - at) / sizeof(cd)) return false;
            reg.vals.resize(nv);
            std::memcpy(reg.vals.data(), b.data() + at, nv * sizeof(cd));
            at += nv * sizeof(cd);
        } else {
            reg.vals.assign(nv, cd(std::nan(""), std::nan("")));
        }
    }
    return at + sizeof(uint64_t) == b.size();
}

std::unique_ptr<Grid> store_fetch(const char *kind, const SKey &k, const Key &key) {
    if (!g_store.on) return nullptr;
    const fs::path path = g_store.dir / (k.name(kind) + ".grid");
    std::ifstream in(path, std::ios::binary);
    if (!in) return nullptr;
    std::stringstream ss;
    ss << in.rdbuf();
    std::unique_ptr<Grid> g(new Grid());
    g->key = key;
    if (!deserialize(ss.str(), k, *g)) return nullptr;
    std::error_code ec;
    fs::last_write_time(path, fs::file_time_type::clock::now(), ec);  // LRU
    return g;
}

void write_atomic(const fs::path &path, const std::string &bytes) {
    std::error_code ec;
    fs::create_directories(g_store.dir, ec);
    if (ec) return;
    std::random_device rd;
    char tmpname[64];
    std::snprintf(tmpname, sizeof tmpname, ".%08x%08x.tmp", rd(), rd());
    const fs::path tmp = g_store.dir / (path.filename().string() + tmpname);
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
    if (ec) fs::remove(tmp, ec);
}

void store_put(const char *kind, const SKey &k, const Grid &g) {
    if (!g_store.on) return;
    const fs::path path = g_store.dir / (k.name(kind) + ".grid");
    write_atomic(path, serialize(k, g));
    somm_core::disk_evict(path);
}

// Whether two grids are one lattice to the bit (the store's copy of a lazy
// grid and the grid in hand).
bool one_lattice(const Grid &a, const Grid &b) {
    auto same = [](double x, double y) { return bits(x) == bits(y); };
    if (!same(a.r1_max, b.r1_max) || !same(a.r1_cap, b.r1_cap) || !same(a.r_break, b.r_break) ||
        !same(a.r_near, b.r_near))
        return false;
    for (int r = 0; r < N_REGIONS; r++) {
        const Region &x = a.regions[r], &y = b.regions[r];
        if (!same(x.r0, y.r0) || !same(x.dr, y.dr) || !same(x.th0, y.th0) || !same(x.dth, y.dth) ||
            x.n_r != y.n_r || x.n_th != y.n_th || !same(a.th0_deg[r], b.th0_deg[r]) ||
            !same(a.dth_deg[r], b.dth_deg[r]))
            return false;
    }
    return true;
}

// The stored copy's nodes into a lazy grid's filled regions where it does
// not hold them yet (another process filled them; a node's value is the
// fill's whoever filled it). `regions` too: the stored regions join the
// table (only for the copy about to be written back, so the file is the
// union of both runs). Returns how many nodes were taken, or -1.
long long store_merge(const char *kind, const SKey &k, Grid &g, bool regions) {
    if (!g_store.on || !g.lazy) return -1;
    std::unique_ptr<Grid> d = store_fetch(kind, k, g.key);
    if (d == nullptr || !one_lattice(*d, g)) return -1;
    give_masks(*d);
    long long taken = 0;
    for (int r = 0; r < N_REGIONS; r++) {
        Region &mine = g.regions[r];
        const Region &theirs = d->regions[r];
        if (!theirs.filled) continue;
        if (!mine.filled) {
            if (!regions) continue;
            mine.filled = true;
        }
        const size_t m = mine.have.size();
        for (size_t q = 0; q < m; q++) {
            if (mine.have[q] || !theirs.have[q]) continue;
            for (int s = 0; s < 4; s++) mine.vals[s * m + q] = theirs.vals[s * m + q];
            mine.have[q] = 1;
            taken++;
        }
    }
    return taken;
}

// A lazy grid written back: the stored copy merged in first (regions and
// nodes), into a copy, so the run's own table keeps the regions it named.
void store_put_merged(const char *kind, const SKey &k, const Grid &g) {
    if (!g_store.on) return;
    if (!g.lazy) {
        store_put(kind, k, g);
        return;
    }
    Grid w = g;
    store_merge(kind, k, w, true);
    store_put(kind, k, w);
}

// `_somm_disk_cache.note_seen(tag, value)`: the values recorded under `tag`
// so far, this one included (`_SEEN_MAX` kept).
std::vector<std::pair<double, double>> note_seen(const std::string &tag, std::pair<double, double> v) {
    std::vector<std::pair<double, double>> seen;
    if (!g_store.on) return {v};
    char hb[24];
    std::snprintf(hb, sizeof hb, "%016llx", static_cast<unsigned long long>(fnv1a(tag)));
    const fs::path path = g_store.dir / (std::string("seen-") + hb + ".bin");
    {
        std::ifstream in(path, std::ios::binary);
        if (in) {
            std::stringstream ss;
            ss << in.rdbuf();
            const std::string b = ss.str();
            size_t at = 0;
            uint64_t tl;
            if (get(b, at, tl) && tl <= b.size() - at && b.compare(at, tl, tag) == 0) {
                at += tl;
                double a, c;
                while (get(b, at, a) && get(b, at, c)) seen.emplace_back(a, c);
            }
        }
    }
    for (const auto &s : seen)
        if (s.first == v.first && s.second == v.second) return seen;
    seen.push_back(v);
    if (seen.size() > SEEN_MAX_DISK) seen.erase(seen.begin(), seen.end() - SEEN_MAX_DISK);
    std::string b;
    put(b, static_cast<uint64_t>(tag.size()));
    b += tag;
    for (const auto &s : seen) {
        put(b, s.first);
        put(b, s.second);
    }
    write_atomic(path, b);
    return seen;
}

// `_lagrange_weights(t, n)`
std::vector<double> lagrange_weights(double t, int n) {
    std::vector<double> w(static_cast<size_t>(n), 1.0);
    for (int a = 0; a < n; a++)
        for (int b = 0; b < n; b++)
            if (a != b) w[a] *= (t - b) / static_cast<double>(a - b);
    return w;
}

double anchor_k2() { return 2.0 * PI / ANCHOR_LAMBDA0_M; }

struct Plan {
    double s = 0, eps_r = 0;
    std::vector<cd> rungs;
    std::vector<double> weights;
    long long n0 = 0;
    double spacing[3] = {1, 1, 1};
    std::string family() const {
        char buf[160];
        std::snprintf(buf, sizeof buf, "below-sweep:%a:%a,%a,%a", eps_r, spacing[0], spacing[1], spacing[2]);
        return buf;
    }
    SKey key(size_t j) const {
        const double k2 = anchor_k2();
        return SKey{{rungs[j].real(), rungs[j].imag(), k2, std::nan(""), k2 * C_LIGHT, MU0, spacing[0],
                     spacing[1], spacing[2]}};
    }
};

// `_anchor_plan(eps_t, spacing)`
bool anchor_plan(cd eps_t, const double spacing[3], Plan &p) {
    const double s = -eps_t.imag();
    if (!(std::isfinite(s) && s > 0.0 && std::isfinite(eps_t.real()))) return false;
    const int n = ANCHOR_POINTS;
    const double x = std::log(s) / ANCHOR_STEP;
    const long long n0 = static_cast<long long>(std::floor(x)) - (n / 2 - 1);
    p.s = s;
    p.eps_r = eps_t.real();
    p.weights = lagrange_weights(x - static_cast<double>(n0), n);
    p.rungs.clear();
    for (int j = 0; j < n; j++)
        p.rungs.emplace_back(eps_t.real(), -std::exp(static_cast<double>(n0 + j) * ANCHOR_STEP));
    p.n0 = n0;
    for (int i = 0; i < 3; i++) p.spacing[i] = spacing[i];
    return true;
}

std::string anchor_mode_env() {
    const char *e = std::getenv("MOMWIRE_BELOW_ANCHORS");
    std::string v = e == nullptr ? "auto" : e;
    v.erase(0, v.find_first_not_of(" \t\n\r\f\v"));
    v.erase(v.find_last_not_of(" \t\n\r\f\v") + 1);
    for (char &c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (v == "0" || v == "off" || v == "false" || v == "no" || v == "exact") return "off";
    if (v == "1" || v == "on" || v == "true" || v == "yes" || v == "always") return "on";
    return "auto";
}

class FillSource : public Source {
  public:
    explicit FillSource(std::string mode) : mode_(std::move(mode)), lazy_(somm_core::lazy_grids()) {}

    Grid &get(cd eps_t, double k2, double r1_max, double omega, double mu,
              const double spacing[3]) override {
        std::lock_guard<std::recursive_mutex> lock(mu_);
        const double lam_m = lambda_m(eps_t, k2);
        const double r1b_wl = r1_bucket_below_wl(r1_max, lam_m);
        const Key key = key_for(eps_t, k2, r1_max, omega, mu, spacing);
        const SKey sk{{eps_t.real(), eps_t.imag(), k2, r1b_wl, omega, mu, spacing[0], spacing[1], spacing[2]}};
        if (Grid *g = find(exact_, key)) return *g;
        if (Grid *g = find(blended_, key)) return *g;
        const std::string mode = mode_.empty() ? anchor_mode_env() : mode_;
        Plan plan;
        bool have_plan = mode != "off" && anchor_plan(eps_t, spacing, plan);
        const double r1_wl = std::min(std::max(r1b_wl, 0.35), R1_CAP_LAMBDA_M);
        std::unique_ptr<Grid> grid;
        if (have_plan && mode == "auto") {
            const bool sweep = is_sweep(plan, r1_wl);
            grid = fetch_exact(sk, key);
            if (grid == nullptr && !(sweep || anchors_ready(plan, r1_wl))) have_plan = false;
        }
        if (grid == nullptr && have_plan) {
            grid = make_grid(eps_t, k2, r1b_wl * lam_m, omega, mu, spacing, false, nullptr, 0);
            grid->key = key;
            grid->blended = true;
            Grid *g = grid.get();
            plans_[g] = plan;
            blended_.emplace_back(std::move(grid));
            for (int r = 0; r < N_REGIONS; r++)
                if (!deferred(r)) blended_fill(*g, r);
            return *g;
        }
        if (grid == nullptr) grid = fetch_exact(sk, key);
        if (grid == nullptr) {
            grid = make_grid(eps_t, k2, r1b_wl * lam_m, omega, mu, spacing, true, nullptr, 0, lazy_);
            grid->key = key;
            store_put_merged("below", sk, *grid);
        }
        grid->key = key;
        stored_[grid.get()] = sk;
        exact_.emplace_back(std::move(grid));
        return *exact_.back();
    }

    void ensure_for(Grid &g, double mx_r1, double mn_th, double mx_th) override {
        std::lock_guard<std::recursive_mutex> lock(mu_);
        bool changed = false;
        for (int r : regions_for(g, mx_r1, mn_th, mx_th)) {
            if (g.regions[r].filled) continue;
            if (g.blended)
                blended_fill(g, r);
            else if (g.lazy)
                activate(g, r);  // its nodes come from fill_nodes
            else
                fill_region(g, r, 0);
            changed = true;
        }
        if (changed && g.lazy && !lazy_) fill_lazy(g, whole(g), 0);
        // `note_mutated`: a stored exact grid re-written as it fills.
        auto it = stored_.find(&g);
        if (changed && it != stored_.end()) store_put_merged("below", it->second, g);
    }

    void fill_nodes(Grid &g, const std::vector<std::vector<uint8_t>> &need) override {
        std::lock_guard<std::recursive_mutex> lock(mu_);
        if (!g.lazy) return;
        auto it = stored_.find(&g);
        auto missing = [&]() {
            for (int r = 0; r < N_REGIONS; r++) {
                const Region &reg = g.regions[r];
                if (!reg.filled || static_cast<size_t>(r) >= need.size() || need[r].empty()) continue;
                for (size_t q = 0; q < reg.have.size(); q++)
                    if (need[r][q] && !reg.have[q]) return true;
            }
            return false;
        };
        if (!missing()) return;
        // Another process may have filled them since this one read the file.
        if (it != stored_.end() && store_merge("below", it->second, g, false) > 0 && !missing()) return;
        if (fill_lazy(g, need, 0) > 0 && it != stored_.end()) store_put_merged("below", it->second, g);
    }

  private:
    std::string mode_;
    bool lazy_ = true;
    std::recursive_mutex mu_;

    // Every node of every region: the eager mode's ask on a lazy grid.
    static std::vector<std::vector<uint8_t>> whole(const Grid &g) {
        std::vector<std::vector<uint8_t>> w(N_REGIONS);
        for (int r = 0; r < N_REGIONS; r++) w[r].assign(g.regions[r].have.size(), 1);
        return w;
    }

    // `store_fetch("below")`, in the form this process runs: a lazy grid
    // with masks (an eager file's filled regions held whole), and, when the
    // lazy grids are off, every node of a lazy file's filled regions in.
    std::unique_ptr<Grid> fetch_exact(const SKey &sk, const Key &key) {
        std::unique_ptr<Grid> g = store_fetch("below", sk, key);
        if (g == nullptr) return g;
        if (lazy_) give_masks(*g);
        else if (g->lazy) fill_lazy(*g, whole(*g), 0);
        return g;
    }
    std::vector<std::unique_ptr<Grid>> exact_, blended_;
    std::map<const Grid *, Plan> plans_;
    std::map<const Grid *, SKey> stored_;
    std::map<std::string, std::vector<std::pair<double, double>>> seen_;
    std::map<std::string, std::unique_ptr<Grid>> anchors_;  // by store name

    static Grid *find(const std::vector<std::unique_ptr<Grid>> &v, const Key &k) {
        for (const auto &g : v)
            if (g->key == k) return g.get();
        return nullptr;
    }

    // `_is_sweep(plan, r1_wl)`
    bool is_sweep(const Plan &plan, double r1_wl) {
        const std::pair<double, double> mine(plan.s, r1_wl);
        std::vector<std::pair<double, double>> &seen = seen_[plan.family()];
        bool have = false;
        for (const auto &s : seen) have = have || (s.first == mine.first && s.second == mine.second);
        if (!have) {
            seen.push_back(mine);
            if (seen.size() > SWEEP_SEEN_MAX) seen.erase(seen.begin(), seen.end() - SWEEP_SEEN_MAX);
        }
        std::vector<std::pair<double, double>> both = seen;
        for (const auto &s : note_seen(plan.family(), mine)) both.push_back(s);
        const double window = ANCHOR_WINDOW_STEPS * ANCHOR_STEP;
        std::set<double> near;
        for (const auto &e : both) {
            const double s = e.first, r = e.second;
            if (s > 0.0 && std::fabs(std::log(s / plan.s)) <= window && 1.0 / 1.26 <= r / r1_wl &&
                r / r1_wl <= 1.26)
                near.insert(s);
        }
        return near.size() >= static_cast<size_t>(ANCHOR_TRIGGER);
    }

    static bool serves(const Grid *g, double r1_wl, int idx) {
        return g != nullptr && g->r1_max / g->lam_m >= r1_wl * (1.0 - 1e-9) &&
               (idx < 0 || g->regions[idx].filled);
    }

    // `_adopt_regions(grid, other)`
    static std::unique_ptr<Grid> adopt(std::unique_ptr<Grid> grid, std::unique_ptr<Grid> other) {
        if (grid == nullptr || other == nullptr) return grid == nullptr ? std::move(other) : std::move(grid);
        if (other->r1_max > grid->r1_max) std::swap(grid, other);
        for (int r = 0; r < N_REGIONS; r++) {
            Region &mine = grid->regions[r];
            const Region &theirs = other->regions[r];
            if (mine.filled || !theirs.filled) continue;
            if (theirs.n_r >= mine.n_r && theirs.n_th == mine.n_th &&
                std::fabs(theirs.r0 - mine.r0) <= 1e-9 * mine.dr &&
                std::fabs(theirs.dr - mine.dr) <= 1e-12 * mine.dr) {
                mine.vals.assign(static_cast<size_t>(4 * mine.n_r * mine.n_th), cd(0.0, 0.0));
                for (int s = 0; s < 4; s++)
                    for (int64_t i = 0; i < mine.n_r; i++)
                        for (int64_t j = 0; j < mine.n_th; j++)
                            mine.vals[(s * mine.n_r + i) * mine.n_th + j] =
                                theirs.vals[(s * theirs.n_r + i) * theirs.n_th + j];
                mine.filled = true;
            }
        }
        return grid;
    }

    Key anchor_key(const Plan &plan, size_t j) const {
        const SKey sk = plan.key(j);
        Key k;
        k.eps_t = plan.rungs[j];
        k.k2 = sk.v[2];
        k.r1b_wl = std::nan("");
        k.omega = sk.v[4];
        k.mu = sk.v[5];
        for (int i = 0; i < 3; i++) k.spacing[i] = plan.spacing[i];
        return k;
    }

    // `_ensure_anchors(plan, r1_wl, idx)`
    std::vector<Grid *> ensure_anchors(const Plan &plan, double r1_wl, int idx) {
        const size_t n = plan.rungs.size();
        std::vector<Grid *> out(n, nullptr);
        const double k2 = anchor_k2();
        const long long start_ll = (static_cast<long long>(getpid()) - plan.n0) % static_cast<long long>(n);
        const size_t start = static_cast<size_t>(start_ll < 0 ? start_ll + static_cast<long long>(n) : start_ll);
        for (size_t step = 0; step < n; step++) {
            const size_t j = (start + step) % n;
            const SKey sk = plan.key(j);
            const std::string name = sk.name("anchor");
            std::unique_ptr<Grid> &slot = anchors_[name];
            if (!serves(slot.get(), r1_wl, idx)) {
                std::unique_ptr<Grid> grid = adopt(std::move(slot), store_fetch("anchor", sk, anchor_key(plan, j)));
                bool changed = false;
                if (!serves(grid.get(), r1_wl, -1)) {
                    const cd eps_a = plan.rungs[j];
                    std::unique_ptr<Grid> grown =
                        make_grid(eps_a, k2, r1_wl * lambda_m(eps_a, k2), sk.v[4], sk.v[5], plan.spacing,
                                  true, grid.get(), 0);
                    grid = std::move(grown);
                    changed = true;
                }
                if (idx >= 0 && !grid->regions[idx].filled) {
                    fill_region(*grid, idx, 0);
                    changed = true;
                }
                grid->key = anchor_key(plan, j);
                if (changed) store_put("anchor", sk, *grid);
                slot = std::move(grid);
            }
            out[j] = slot.get();
        }
        return out;
    }

    // `_anchors_ready(plan, r1_wl)`
    bool anchors_ready(const Plan &plan, double r1_wl) {
        for (size_t j = 0; j < plan.rungs.size(); j++) {
            const SKey sk = plan.key(j);
            std::unique_ptr<Grid> &slot = anchors_[sk.name("anchor")];
            if (!serves(slot.get(), r1_wl, -1)) {
                std::unique_ptr<Grid> grid = adopt(std::move(slot), store_fetch("anchor", sk, anchor_key(plan, j)));
                const bool ok = serves(grid.get(), r1_wl, -1);
                slot = std::move(grid);
                if (!ok) return false;
            }
        }
        return true;
    }

    // `SommerfeldGridBelowBlended._rows_served(idx, anchors)`
    static int64_t rows_served(const Grid &g, int idx, const std::vector<Grid *> &anchors) {
        const Region &reg = g.regions[idx];
        const double r_hi = idx / N_BANDS == 0 ? g.r_break : idx / N_BANDS == 1 ? g.r_near : g.r1_max;
        if (r_hi <= reg.r0 + 2.0 * reg.dr) return 0;
        int64_t rows = reg.n_r;
        const double tol = 1e-9 * reg.dr / g.lam_m;
        for (const Grid *a : anchors) {
            const Region &ar = a->regions[idx];
            const bool same = ar.n_th == reg.n_th && std::fabs(ar.th0 - reg.th0) <= 1e-15 &&
                              std::fabs(ar.dth - reg.dth) <= 1e-15 &&
                              std::fabs(ar.dr / a->lam_m - reg.dr / g.lam_m) <= tol &&
                              std::fabs(ar.r0 / a->lam_m - reg.r0 / g.lam_m) <= tol;
            if (!same) return 0;
            rows = std::min(rows, ar.n_r);
        }
        const int64_t need =
            std::max<int64_t>(static_cast<int64_t>(std::floor((r_hi - reg.r0) / reg.dr + 1e-6)) + 3, 4);
        return rows >= need ? rows : 0;
    }

    // `SommerfeldGridBelowBlended._fill_region(idx)`
    void blended_fill(Grid &g, int idx) {
        Region &reg = g.regions[idx];
        if (reg.filled) return;
        const Plan &plan = plans_.at(&g);
        const double r1_wl = g.r1_max / g.lam_m;
        std::vector<Grid *> anchors = ensure_anchors(plan, r1_wl, -1);
        const int64_t rows = rows_served(g, idx, anchors);
        if (rows == 0) {
            fill_region(g, idx, 0);
            return;
        }
        anchors = ensure_anchors(plan, r1_wl, idx);
        const int64_t n_th = reg.n_th;
        std::vector<cd> acc;
        for (size_t j = 0; j < anchors.size(); j++) {
            const Grid &a = *anchors[j];
            const Region &ar = a.regions[idx];
            const double c = plan.weights[j] / (a.key.omega * a.key.mu);
            std::vector<cd> term(static_cast<size_t>(4 * rows * n_th));
            for (int s = 0; s < 4; s++)
                for (int64_t i = 0; i < rows; i++)
                    for (int64_t k = 0; k < n_th; k++)
                        term[(s * rows + i) * n_th + k] = np_mul(cd(c, 0.0), ar.vals[(s * ar.n_r + i) * n_th + k]);
            if (acc.empty()) {
                acc = std::move(term);
            } else {
                for (size_t q = 0; q < acc.size(); q++) acc[q] = c_add(acc[q], term[q]);
            }
        }
        const double om = g.key.omega * g.key.mu;
        for (cd &v : acc) v = np_mul(v, cd(om, 0.0));
        reg.vals = std::move(acc);
        reg.n_r = rows;
        reg.filled = true;
    }
};

}  // namespace

std::shared_ptr<Source> fill_source(const std::string &anchors) {
    return std::make_shared<FillSource>(anchors);
}

void configure_below_store(bool on) {
    g_store = Store();
    if (!on) return;
    std::string tag;
    fs::path dir;
    if (!somm_core::disk_location(dir, tag)) return;
    g_store.on = true;
    g_store.dir = dir;
    g_store.tag = tag;
}

}  // namespace somm_below
