// `_sommerfeld.get_grid` with no Python — see `_somm_grid_core.h`. Each
// function names the Python it copies; the arithmetic is that Python's, one
// IEEE operation at a time (this TU is built -ffp-contract=off; MSVC asks for
// precise semantics below), the fused complex product spelled with std::fma.
#include "_study_timers.h"
#include "_somm_grid_core.h"

#include "_somm_kernel_core.h"

#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <thread>
#include <cmath>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>

namespace somm_core {

namespace {

constexpr double PI = 3.141592653589793;
constexpr double C_LIGHT = 299792458.0;
const double MU0 = 4e-7 * PI;  // `_MU0 = 4e-7 * np.pi`
const double K2_REF = 2.0 * PI;
constexpr double R1_CAP_LAMBDA = 15.0;
constexpr double R1_NEAR_LAMBDA = 4.0;
constexpr double DR_FAR_LAMBDA = 0.2;
constexpr double DTH_FAR_DEG = 2.5;
constexpr double N0_CAP = 513;
constexpr double TH_SPLIT_DEG = 30.0;
constexpr double EPS_IM_BUCKET = 0.10;
constexpr double R1_SWEEP_SLACK = 0.05;
constexpr double DEG_TO_RAD = PI / 180.0;  // np.radians
constexpr double RAD_TO_DEG = 180.0 / PI;  // np.degrees

// CPython 3.12's complex arithmetic (a float operand is (x, 0) first).
cd py_mul(cd a, cd b) {
    return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}
cd py_div(cd a, cd b) {  // `_Py_c_quot`
    const double abs_breal = b.real() < 0 ? -b.real() : b.real();
    const double abs_bimag = b.imag() < 0 ? -b.imag() : b.imag();
    if (abs_breal >= abs_bimag) {
        if (abs_breal == 0.0) throw std::domain_error("complex division by zero");
        const double ratio = b.imag() / b.real();
        const double denom = b.real() + b.imag() * ratio;
        return cd((a.real() + a.imag() * ratio) / denom, (a.imag() - a.real() * ratio) / denom);
    }
    if (abs_bimag >= abs_breal) {
        const double ratio = b.real() / b.imag();
        const double denom = b.real() * ratio + b.imag();
        return cd((a.real() * ratio + a.imag()) / denom, (a.imag() * ratio - a.real()) / denom);
    }
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return cd(nan, nan);
}
cd py_sub(cd a, cd b) { return cd(a.real() - b.real(), a.imag() - b.imag()); }
cd py_add(cd a, cd b) { return cd(a.real() + b.real(), a.imag() + b.imag()); }
cd py_square(cd x) {  // `x ** 2`: c_powu(x, 2) = c_prod(c_1, c_prod(x, x))
    return py_mul(cd(1.0, 0.0), py_mul(x, x));
}


// numpy's elementwise complex multiply (fused, `_cmul`'s formula).
inline cd np_mul(cd a, cd b) {
    return cd(std::fma(a.real(), b.real(), -(a.imag() * b.imag())),
              std::fma(a.real(), b.imag(), a.imag() * b.real()));
}
inline cd np_add(cd a, cd b) { return cd(a.real() + b.real(), a.imag() + b.imag()); }
inline cd np_sub(cd a, cd b) { return cd(a.real() - b.real(), a.imag() - b.imag()); }

// Python's builtin min / max of two floats (the first wins a tie).
inline double py_max(double a, double b) { return b > a ? b : a; }
inline double py_min(double a, double b) { return b < a ? b : a; }

// `_c1(k2, omega, mu)`: -1j * omega * mu / (4.0 * np.pi * k2 * k2)
cd c1_of(double k2, double omega, double mu) {
    const cd a = py_mul(py_mul(cd(-0.0, -1.0), cd(omega, 0.0)), cd(mu, 0.0));
    return py_div(a, cd(((4.0 * PI) * k2) * k2, 0.0));
}

struct Surf {  // the four surfaces at one node
    cd v[4];
};

// `iv_surfaces_direct` over (R1, theta) node lists (C order of the region's
// meshgrid), rtol as the grid passes it.
std::vector<Surf> iv_surfaces(cd eps_t, double k2, const std::vector<double> &r1,
                              const std::vector<double> &th, double rtol, double omega,
                              double mu) {
    STUDY_T("above.iv_surfaces");
    const std::size_t n = r1.size();
    std::vector<Surf> out(n);
    const cd k1s = py_mul(cd(k2 * k2, 0.0), eps_t);  // k2 * k2 * eps_t
    const double k2s = k2 * k2;
    const cd c1 = c1_of(k2, omega, mu);
    // `_limits_r1_zero` (only its scalars here; per node below)
    const cd c2 = py_div(py_sub(k1s, cd(k2s, 0.0)), py_add(k1s, cd(k2s, 0.0)));
    const cd c3 = py_div(py_mul(cd(k2s, 0.0), py_sub(k1s, cd(k2s, 0.0))),
                         py_square(py_add(k1s, cd(k2s, 0.0))));
    const cd c1c3k1s = py_mul(py_mul(c1, c3), k1s);
    const cd c1k2s = py_mul(c1, cd(k2s, 0.0));
    const cd mc1k2s = py_mul(cd(-c1.real(), -c1.imag()), cd(k2s, 0.0));
    const cd c2mc3 = py_sub(c2, c3);
    // the six integrals at every nonzero node, in parallel (independent)
    std::vector<std::size_t> nz;
    for (std::size_t i = 0; i < n; i++)
        if (!(r1[i] == 0.0)) nz.push_back(i);
    std::vector<cd> six(nz.size() * 6);
    const bool free_space = eps_t == cd(1.0, 0.0);
    auto node = [&](std::ptrdiff_t q) {
        const std::size_t i = nz[static_cast<std::size_t>(q)];
        const double c = std::cos(th[i]), s = std::sin(th[i]);
        double rho = r1[i] * c, h = r1[i] * s;
        rho = (rho >= 0.0 || std::isnan(rho)) ? rho : 0.0;  // np.maximum(x, 0.0)
        h = (h >= 0.0 || std::isnan(h)) ? h : 0.0;
        cd res[6];
        if (free_space) {
            for (int k = 0; k < 6; k++) res[k] = cd(0.0, 0.0);
        } else {
            if (rho < 0.0 || h < 0.0 || (rho == 0.0 && h == 0.0))
                throw std::invalid_argument("need rho, h >= 0 and R1 > 0 at every node");
            somm_core::six_integrals(eps_t, k2, rho, h, rtol, 0, res);
        }
        for (int k = 0; k < 6; k++) six[static_cast<std::size_t>(q) * 6 + k] = res[k];
    };
    const std::ptrdiff_t n_nz = static_cast<std::ptrdiff_t>(nz.size());
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t q = 0; q < n_nz; q++) node(q);
#else
    // The executable's serial copy has no OpenMP, and a cold fill is seconds
    // of independent nodes: plain threads, each node its own arithmetic, so
    // which thread takes it moves no bit. MOMWIRE_THREADS (the executable's
    // own override) bounds them.
    unsigned n_thr = std::thread::hardware_concurrency();
    if (const char *mt = std::getenv("MOMWIRE_THREADS")) {
        const long v = std::strtol(mt, nullptr, 10);
        if (v > 0) n_thr = static_cast<unsigned>(v);
    }
    n_thr = std::max(1u, std::min<unsigned>(n_thr, 64u));
    if (n_thr == 1 || n_nz < 64) {
        for (std::ptrdiff_t q = 0; q < n_nz; q++) node(q);
    } else {
        std::atomic<std::ptrdiff_t> next{0};
        std::exception_ptr failed;
        std::mutex failed_mu;
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < n_thr; t++)
            pool.emplace_back([&] {
                try {
                    for (std::ptrdiff_t q = next++; q < n_nz; q = next++) node(q);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(failed_mu);
                    if (!failed) failed = std::current_exception();
                    next = n_nz;
                }
            });
        for (std::thread &th_ : pool) th_.join();
        if (failed) std::rethrow_exception(failed);
    }
#endif
    const cd jk2 = py_mul(cd(0.0, 1.0), cd(k2, 0.0));  // 1j * k2
    for (std::size_t i = 0; i < n; i++) {
        Surf &o = out[i];
        if (r1[i] == 0.0) {
            const double s = std::sin(th[i]), co = std::cos(th[i]);
            const bool near = std::fabs(co) < 1e-8;
            const double co_safe = near ? 1.0 : co;
            const double q1 = near ? 0.0 : (1.0 - s) / co_safe;
            const double q2 = near ? 0.5 : (1.0 - s) / (co_safe * co_safe);
            o.v[0] = np_mul(c1c3k1s, cd(q1, 0.0));
            o.v[1] = c1c3k1s;
            o.v[2] = np_mul(c1k2s, np_add(c2mc3, np_mul(c3, cd(q2, 0.0))));
            o.v[3] = np_mul(mc1k2s, np_sub(c2, np_mul(c3, cd(q2, 0.0))));
            continue;
        }
        const std::size_t q = static_cast<std::size_t>(
            std::lower_bound(nz.begin(), nz.end(), i) - nz.begin());
        const cd *sx = &six[q * 6];
        const cd v_rr = sx[0], v_zz = sx[1], v_rz = sx[2], v_r1 = sx[3], v = sx[4], u = sx[5];
        // phase = r1 * np.exp(1j * k2 * r1)
        const cd arg = np_mul(jk2, cd(r1[i], 0.0));
        const double ex = std::exp(arg.real());
        const cd e(ex * std::cos(arg.imag()), ex * std::sin(arg.imag()));
        const cd phase = np_mul(cd(r1[i], 0.0), e);
        const cd cp = np_mul(c1, phase);
        o.v[0] = np_mul(np_mul(cp, k1s), v_rz);
        o.v[1] = np_mul(np_mul(cp, k1s), np_add(v_zz, np_mul(cd(k2s, 0.0), v)));
        o.v[2] = np_mul(np_mul(cp, cd(k2s, 0.0)), np_add(v_rr, u));
        const cd mcp = np_mul(cd(-c1.real(), -c1.imag()), phase);
        o.v[3] = np_mul(np_mul(mcp, cd(k2s, 0.0)), np_add(v_r1, u));
    }
    return out;
}

}  // namespace

namespace {

// The (R1, theta) of node (i, j) of region `reg_idx`, as the constructor's
// meshgrid builds it: R1 = r0 + dr * i (region 1's first row at r0_fill when
// that is set), theta = np.radians(th0 + dth * j) in the layout's degrees.
inline void node_coords(const Grid &g, size_t reg_idx, int64_t i, int64_t j, double &r,
                        double &t) {
    const GridRegion &R = g.regions[reg_idx];
    r = R.r0 + R.dr * static_cast<double>(i);
    if (reg_idx == 1 && g.r0_fill > 0.0 && i == 0) r = g.r0_fill;
    t = (R.th0_deg + R.dth_deg * static_cast<double>(j)) * DEG_TO_RAD;
}

}  // namespace

Grid layout_grid(cd eps_t, double k2, double r1_max, double omega, double mu, cd lattice_eps) {
    Grid g;
    g.eps_t = eps_t;
    g.lattice_eps = lattice_eps;
    g.k2 = k2;
    g.omega = omega;
    g.mu = mu;
    const double lam = 2.0 * PI / g.k2;
    g.r1_max = py_min(py_max(r1_max, 0.35 * lam), R1_CAP_LAMBDA * lam);
    // k1 = self.k2 * np.sqrt(self.lattice_eps): numpy scalar arithmetic
    // numpy's complex sqrt is libm's csqrt (4cbda950: cmath.sqrt differs from
    // it on purely imaginary input; equal elsewhere on the samples measured).
    const cd root = std::sqrt(lattice_eps);
    cd k1(g.k2 * root.real() - 0.0 * root.imag(), g.k2 * root.imag() + 0.0 * root.real());
    if (k1.imag() > 0) k1 = std::conj(k1);
    const double abs_k1 = std::hypot(k1.real(), k1.imag());
    double beat;
    if (abs_k1 <= 12.0 * g.k2) {
        const cd d(k1.real() - g.k2, k1.imag() - 0.0);
        beat = 2.0 * PI / py_max(std::hypot(d.real(), d.imag()), 1e-30);
    } else {
        beat = std::numeric_limits<double>::infinity();
    }
    g.r_near = py_min(g.r1_max, R1_NEAR_LAMBDA * lam);
    const double dth2_target = py_min(5.0, ((0.07 * lam) / g.r_near) * RAD_TO_DEG);
    const long long n_th2 = static_cast<long long>(std::ceil(TH_SPLIT_DEG / dth2_target)) + 1;
    const double dth2 = TH_SPLIT_DEG / static_cast<double>(n_th2 - 1);
    g.r_break = 0.2 * lam;
    const double layer = 1.0 / py_max(abs_k1, 1e-300);
    const double dr0 = py_min(0.01 * lam, py_max(layer / 4.0, g.r_break / (N0_CAP - 1)));
    g.r0_fill = layer < 0.25 * dr0 ? 0.5 * dr0 : 0.0;
    const double split = TH_SPLIT_DEG;
    struct Lay {
        double r0, r1, dr, th0, th1, dth;
    };
    std::vector<Lay> layout = {
        {0.0, g.r_break, 0.01 * lam, 0.0, split, 10.0},
        {0.0, g.r_break, dr0, split, 90.0, 10.0},
        {g.r_break, g.r_near, py_min(0.05 * lam, beat / 6.0), 0.0, split, dth2},
        {g.r_break, g.r_near, py_min(0.1 * lam, beat / 6.0), split, 90.0, 10.0},
    };
    if (g.r1_max > g.r_near * (1.0 + 1e-9)) {
        const double dr_far = DR_FAR_LAMBDA * lam;
        layout.push_back({g.r_near, g.r1_max, dr_far, 0.0, split, DTH_FAR_DEG});
        layout.push_back({g.r_near, g.r1_max, dr_far, split, 90.0, 10.0});
    }
    for (std::size_t reg_idx = 0; reg_idx < layout.size(); reg_idx++) {
        const Lay &L = layout[reg_idx];
        const long long n_r = std::max<long long>(static_cast<long long>(std::ceil((L.r1 - L.r0) / L.dr)) + 1, 4);
        const long long n_th = static_cast<long long>(std::nearbyint((L.th1 - L.th0) / L.dth)) + 1;
        GridRegion R;
        R.r0 = L.r0;
        R.dr = L.dr;
        R.n_r = n_r;
        R.th0 = L.th0 * DEG_TO_RAD;
        R.dth = L.dth * DEG_TO_RAD;
        R.n_th = n_th;
        R.th0_deg = L.th0;
        R.dth_deg = L.dth;
        const std::size_t m = static_cast<std::size_t>(n_r * n_th);
        R.vals.assign(4 * m, cd(std::nan(""), std::nan("")));
        R.have.assign(m, 0);
        g.regions.push_back(std::move(R));
    }
    g.lazy = true;
    return g;
}

size_t fill_nodes(Grid &g, const std::vector<std::vector<uint8_t>> &need, double rtol) {
    STUDY_T("above.fill_nodes");
    // The nodes asked and not yet held, region-major in C order, through one
    // `iv_surfaces_direct` call: a node's surfaces are its own arithmetic
    // (the six integrals at its (rho, h), the R1 = 0 limits, the phase), so
    // which other nodes share the call moves no bit.
    std::vector<std::pair<std::size_t, std::size_t>> at;
    std::vector<double> rr, tt;
    for (std::size_t r = 0; r < g.regions.size(); r++) {
        GridRegion &R = g.regions[r];
        const std::size_t m = R.have.size();
        if (r >= need.size() || need[r].empty()) continue;
        if (need[r].size() != m) throw std::logic_error("fill_nodes: a mask is not the region's");
        for (std::size_t q = 0; q < m; q++) {
            if (!need[r][q] || R.have[q]) continue;
            double rv, tv;
            node_coords(g, r, static_cast<int64_t>(q) / R.n_th, static_cast<int64_t>(q) % R.n_th, rv, tv);
            at.emplace_back(r, q);
            rr.push_back(rv);
            tt.push_back(tv);
        }
    }
    if (at.empty()) return 0;
    const std::vector<Surf> surf = iv_surfaces(g.eps_t, g.k2, rr, tt, rtol, g.omega, g.mu);
    for (std::size_t p = 0; p < at.size(); p++) {
        GridRegion &R = g.regions[at[p].first];
        const std::size_t m = R.have.size(), q = at[p].second;
        for (int k = 0; k < 4; k++) R.vals[k * m + q] = surf[p].v[k];
        R.have[q] = 1;
    }
    return at.size();
}

Grid fill_grid(cd eps_t, double k2, double r1_max, double rtol, double omega, double mu,
               cd lattice_eps) {
    STUDY_T("above.fill_grid");
    Grid g = layout_grid(eps_t, k2, r1_max, omega, mu, lattice_eps);
    std::vector<std::vector<uint8_t>> all;
    for (const GridRegion &R : g.regions) all.emplace_back(R.have.size(), 1);
    fill_nodes(g, all, rtol);
    return g;
}

Grid blend(const Grid &a, const Grid &b, double t, cd eps_t) {
    if (a.regions.size() != b.regions.size())
        throw std::invalid_argument("blend: the two grids are not one lattice");
    for (std::size_t r = 0; r < a.regions.size(); r++) {
        const GridRegion &x = a.regions[r], &y = b.regions[r];
        if (x.n_r != y.n_r || x.n_th != y.n_th || x.r0 != y.r0 || x.dr != y.dr || x.th0 != y.th0 ||
            x.dth != y.dth)
            throw std::invalid_argument("blend: the two grids are not one lattice");
    }
    Grid g = a;
    g.eps_t = eps_t;
    const cd wa(1.0 - t, 0.0), wb(t, 0.0);
    for (std::size_t r = 0; r < g.regions.size(); r++) {
        std::vector<cd> &v = g.regions[r].vals;
        const std::vector<cd> &w = b.regions[r].vals;
        for (std::size_t i = 0; i < v.size(); i++) v[i] = np_add(np_mul(wa, a.regions[r].vals[i]), np_mul(wb, w[i]));
    }
    return g;
}

Grid scaled_to(const Grid &src, double k2, double omega, double mu) {
    const double scale = src.k2 / k2;
    const double factor = (omega * mu) / (src.omega * src.mu);
    Grid g = src;
    g.k2 = k2;
    g.omega = omega;
    g.mu = mu;
    g.r1_max = src.r1_max * scale;
    g.r_break = src.r_break * scale;
    g.r_near = src.r_near * scale;
    g.r0_fill = src.r0_fill * scale;
    for (GridRegion &R : g.regions) {
        R.r0 = R.r0 * scale;
        R.dr = R.dr * scale;
        for (cd &v : R.vals) v = np_mul(v, cd(factor, 0.0));
    }
    return g;
}

namespace {

// numpy's complex SCALAR arithmetic (scalarmath: plain products, numpy's
// divide); a float operand is (x, 0).
inline cd ns_mul(cd a, cd b) {
    return cd(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}
inline cd ns_div(cd a, cd b) {  // `CDOUBLE_divide`
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

}  // namespace

std::vector<cd> far_pack(const Grid &g) {
    const cd eps = g.eps_t;
    if (eps == cd(1.0, 0.0)) return {};
    const double k = g.k2, ks = k * k;
    const cd root = std::sqrt(eps);  // np.sqrt: glibc csqrt
    cd k1 = ns_mul(cd(k, 0.0), root);
    if (k1.imag() > 0) k1 = std::conj(k1);
    const cd k1s = ns_mul(k1, k1);
    const cd sap = ns_div(cd(-1.0, 0.0), std::sqrt(cd(eps.real() + 1.0, eps.imag() + 0.0)));
    const cd ap = std::asin(sap);   // np.arcsin: glibc casin
    const cd cap = std::cos(ap);    // np.cos: glibc ccos
    const cd g1p_arg = ns_mul(ns_mul(cd(ks, 0.0), cap), cap) - k1s;
    const cd g1p = std::sqrt(cd(g1p_arg.real() + 0.0, g1p_arg.imag() + 0.0));
    // dden = k1s * 1j * k * cap_ - ks * ks * cap_ * sap / g1p
    const cd left = ns_mul(ns_mul(ns_mul(k1s, cd(0.0, 1.0)), cd(k, 0.0)), cap);
    const cd right = ns_div(ns_mul(ns_mul(cd(ks * ks, 0.0), cap), sap), g1p);
    const cd dden(left.real() - right.real(), left.imag() - right.imag());
    // res_fv = 2.0 * (1j * k * sap) / dden; `1j * k` is CPython's, `* sap`
    // numpy's reflected scalar product
    const cd jk = py_mul(cd(0.0, 1.0), cd(k, 0.0));
    const cd res_fv = ns_div(ns_mul(cd(2.0, 0.0), ns_mul(jk, sap)), dden);
    const cd g2_arg(k1s.real() - ks, k1s.imag() - 0.0);
    cd g2k1 = std::sqrt(cd(g2_arg.real() + 0.0, g2_arg.imag() + 0.0));
    if (g2k1.real() < 0) g2k1 = cd(-g2k1.real(), -g2k1.imag());
    const cd c2 = py_div(cd(eps.real() - 1.0, eps.imag() - 0.0), cd(eps.real() + 1.0, eps.imag() + 0.0));
    const cd c1k = py_div(py_mul(py_mul(cd(-0.0, -1.0), cd(g.omega, 0.0)), cd(g.mu, 0.0)),
                          cd(4.0 * PI, 0.0));
    return {eps, k1, c2, c1k, ap, res_fv, g2k1, cd(k, 0.0)};
}

double r1_bucket_wl(double r1_wl) {
    const double x = py_max(r1_wl, 0.1);
    // math.log(x, 1.25) is log(x) / log(1.25)
    const double n = std::ceil(std::log(x) / std::log(1.25) - 1e-12);
    double bucket = std::pow(1.25, n);
    if (bucket < r1_wl) bucket *= 1.25;
    return bucket;
}

void eps_rungs(cd eps_t, cd &lo, cd &hi, double &t) {
    const double step = 1.0 + EPS_IM_BUCKET;
    const double re = eps_t.real(), im = eps_t.imag();
    if (step <= 1.0 || !(re > 0.0) || im >= 0.0) {
        lo = hi = eps_t;
        t = 0.0;
        return;
    }
    const double x = std::log(-im) / std::log(step);
    const double n = std::floor(x);
    const double lo_im = std::pow(step, n), hi_im = std::pow(step, n + 1.0);
    if (std::fabs(-im - lo_im) <= 1e-12 * lo_im) {
        lo = hi = cd(re, -lo_im);
        t = 0.0;
        return;
    }
    if (std::fabs(-im - hi_im) <= 1e-12 * hi_im) {
        lo = hi = cd(re, -hi_im);
        t = 0.0;
        return;
    }
    t = (-im - lo_im) / (hi_im - lo_im);
    lo = cd(re, -lo_im);
    hi = cd(re, -hi_im);
}

// ---------------------------------------------------------------------------
// The executable's own persistent store of the normalized masters (Steve
// 10-15: the exe keeps ITS OWN cache, beside Python's, keyed by the exe's
// version plus the grid's inputs; the first new frequency or ground pays the
// cold fill once). The masters are the expensive, frequency-independent
// level; the physical views over them stay in-process, as in Python.
//
// The contract is `_somm_disk_cache`'s: a hit is the fill's bytes (raw
// IEEE doubles, the key re-checked, a checksum over the whole file); any
// failure is a miss or a skipped write, never an error; writes go to a
// unique temporary file renamed into place, so concurrent processes and a
// killed one leave only whole files. Least-recently-used eviction (mtime;
// a hit refreshes it) keeps the directory under MOMWIRE_SOMM_CACHE_MAX_MB.
namespace {

namespace fs = std::filesystem;

struct DiskConfig {
    bool on = false;
    fs::path dir;
    std::string tag;
    uint64_t max_bytes = 512ull << 20;
};
DiskConfig g_disk;
DiskStats g_stats;

// MWSOMMGRID1: every node held (the eager fill's files). MWSOMMGRID2
// (momwire#1512's lazy masters): each region's node mask, then its values
// (NaN where the mask is clear). Both read; only 2 is written.
const char MAGIC1[] = "MWSOMMGRID1\n";
const char MAGIC2[] = "MWSOMMGRID2\n";

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
    if (b.size() - at < sizeof(T) || at > b.size()) return false;
    std::memcpy(&v, b.data() + at, sizeof(T));
    at += sizeof(T);
    return true;
}

uint64_t bits(double x) {
    uint64_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}

// The file for one master: its key's exact bits, in hex.
fs::path master_path(const double key[5]) {
    std::string name = "master";
    char buf[24];
    for (int i = 0; i < 5; i++) {
        std::snprintf(buf, sizeof buf, "-%016llx", static_cast<unsigned long long>(bits(key[i])));
        name += buf;
    }
    return g_disk.dir / (name + ".grid");
}

std::string serialize(const double key[5], const Grid &g) {
    std::string b(MAGIC2);
    const uint64_t tl = g_disk.tag.size();
    put(b, tl);
    b += g_disk.tag;
    for (int i = 0; i < 5; i++) put(b, key[i]);
    const double scalars[13] = {g.eps_t.real(), g.eps_t.imag(), g.lattice_eps.real(),
                                g.lattice_eps.imag(), g.k2, g.omega, g.mu, g.r1_max,
                                g.r_break, g.r_near, g.r0_fill, 0.0, 0.0};
    for (double v : scalars) put(b, v);
    put(b, static_cast<uint64_t>(g.regions.size()));
    for (const GridRegion &r : g.regions) {
        put(b, r.r0);
        put(b, r.dr);
        put(b, r.n_r);
        put(b, r.th0);
        put(b, r.dth);
        put(b, r.n_th);
        put(b, static_cast<uint64_t>(r.vals.size()));
        b.append(reinterpret_cast<const char *>(r.have.data()), r.have.size());
        b.append(reinterpret_cast<const char *>(r.vals.data()), r.vals.size() * sizeof(cd));
    }
    put(b, fnv1a(b));
    return b;
}

// A file's master into `g`, whose layout is the key's (`layout_grid`): the
// file's region geometry must be that layout to the bit, and its held nodes
// are taken where `g` does not hold them yet (a node's value is the fill's,
// whoever filled it). Returns how many nodes were taken, or -1 when the
// file is not this master's.
long long deserialize_merge(const std::string &b, const double key[5], Grid &g) {
    const size_t m = sizeof(MAGIC2) - 1;
    static_assert(sizeof(MAGIC1) == sizeof(MAGIC2), "one magic length");
    if (b.size() < m + sizeof(uint64_t)) return -1;
    const bool v2 = b.compare(0, m, MAGIC2) == 0;
    if (!v2 && b.compare(0, m, MAGIC1) != 0) return -1;
    uint64_t sum;
    std::memcpy(&sum, b.data() + b.size() - sizeof sum, sizeof sum);
    if (fnv1a(b.substr(0, b.size() - sizeof sum)) != sum) return -1;
    size_t at = m;
    uint64_t tl;
    if (!get(b, at, tl) || tl > b.size() - at || b.compare(at, tl, g_disk.tag) != 0 ||
        tl != g_disk.tag.size())
        return -1;
    at += tl;
    for (int i = 0; i < 5; i++) {
        double k;
        if (!get(b, at, k) || bits(k) != bits(key[i])) return -1;
    }
    double sc[13];
    for (double &v : sc)
        if (!get(b, at, v)) return -1;
    const double mine[11] = {g.eps_t.real(), g.eps_t.imag(), g.lattice_eps.real(),
                             g.lattice_eps.imag(), g.k2, g.omega, g.mu, g.r1_max,
                             g.r_break, g.r_near, g.r0_fill};
    for (int i = 0; i < 11; i++)
        if (bits(sc[i]) != bits(mine[i])) return -1;
    uint64_t n_reg;
    if (!get(b, at, n_reg) || n_reg != g.regions.size()) return -1;
    struct Take {
        size_t reg, mask_at, vals_at;
    };
    std::vector<Take> takes;
    for (size_t r = 0; r < n_reg; r++) {
        const GridRegion &R = g.regions[r];
        double r0, dr, th0, dth;
        int64_t n_r, n_th;
        uint64_t nv;
        if (!get(b, at, r0) || !get(b, at, dr) || !get(b, at, n_r) || !get(b, at, th0) ||
            !get(b, at, dth) || !get(b, at, n_th) || !get(b, at, nv))
            return -1;
        if (bits(r0) != bits(R.r0) || bits(dr) != bits(R.dr) || n_r != R.n_r ||
            bits(th0) != bits(R.th0) || bits(dth) != bits(R.dth) || n_th != R.n_th ||
            nv != R.vals.size())
            return -1;
        const size_t nm = v2 ? R.have.size() : 0;
        if (nm > b.size() - at) return -1;
        const size_t mask_at = v2 ? at : std::string::npos;
        at += nm;
        if (nv > (b.size() - at) / sizeof(cd)) return -1;
        takes.push_back(Take{r, mask_at, at});
        at += nv * sizeof(cd);
    }
    if (at + sizeof(uint64_t) != b.size()) return -1;
    long long taken = 0;
    for (const Take &t : takes) {
        GridRegion &R = g.regions[t.reg];
        const size_t mm = R.have.size();
        for (size_t q = 0; q < mm; q++) {
            if (R.have[q]) continue;
            if (t.mask_at != std::string::npos && b[t.mask_at + q] == 0) continue;
            for (int k = 0; k < 4; k++)
                std::memcpy(&R.vals[k * mm + q], b.data() + t.vals_at + (k * mm + q) * sizeof(cd),
                            sizeof(cd));
            R.have[q] = 1;
            taken++;
        }
    }
    return taken;
}

// The stored master's nodes into `g` (-1: no usable file).
long long disk_merge(const double key[5], Grid &g) {
    std::error_code ec;
    const fs::path path = master_path(key);
    std::ifstream in(path, std::ios::binary);
    if (!in) return -1;
    std::stringstream ss;
    ss << in.rdbuf();
    if (!in.good() && !in.eof()) return -1;
    const long long got = deserialize_merge(ss.str(), key, g);
    if (got < 0) {
        g_stats.rejected++;
        return -1;
    }
    fs::last_write_time(path, fs::file_time_type::clock::now(), ec);  // LRU
    return got;
}

void evict(const fs::path &keep) {
    std::error_code ec;
    std::vector<std::pair<fs::file_time_type, fs::path>> files;
    uint64_t total = 0;
    const auto now = fs::file_time_type::clock::now();
    for (const auto &e : fs::directory_iterator(g_disk.dir, ec)) {
        std::error_code e2;
        if (!e.is_regular_file(e2)) continue;
        const fs::path p = e.path();
        const auto t = fs::last_write_time(p, e2);
        if (e2) continue;
        if (p.extension() == ".tmp") {
            if (now - t > std::chrono::hours(1)) fs::remove(p, e2);
            continue;
        }
        if (p.extension() != ".grid") continue;
        const uint64_t size = fs::file_size(p, e2);
        if (e2) continue;
        total += size;
        files.emplace_back(t, p);
    }
    std::sort(files.begin(), files.end());
    for (const auto &f : files) {
        if (total <= g_disk.max_bytes) break;
        if (f.second == keep) continue;
        std::error_code e2;
        const uint64_t size = fs::file_size(f.second, e2);
        if (!e2 && fs::remove(f.second, e2)) total -= size;
    }
}

// Write `g` back, first taking any node another process stored since this
// one read the file (merge-on-write: a node's value is the fill's whoever
// filled it, so the union is what both would have written). The write is a
// unique temporary renamed into place; a race between two writers can lose
// the other's newest nodes from the file (they are filled again on demand),
// never mix bytes.
void disk_store(const double key[5], Grid &g) {
    std::error_code ec;
    fs::create_directories(g_disk.dir, ec);
    if (ec) return;
    disk_merge(key, g);
    const fs::path path = master_path(key);
    std::random_device rd;
    char tmpname[64];
    std::snprintf(tmpname, sizeof tmpname, ".%08x%08x.tmp", rd(), rd());
    const fs::path tmp = g_disk.dir / (path.filename().string() + tmpname);
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return;
        const std::string b = serialize(key, g);
        out.write(b.data(), static_cast<std::streamsize>(b.size()));
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
    g_stats.writes++;
    evict(path);
}

}  // namespace

void configure_disk_cache(const std::string &tag) {
    g_disk = DiskConfig();
    const char *en = std::getenv("MOMWIRE_SOMM_CACHE");
    if (en != nullptr) {
        std::string v(en);
        v.erase(0, v.find_first_not_of(" \t"));
        v.erase(v.find_last_not_of(" \t") + 1);
        for (char &c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (v == "0" || v == "off" || v == "false" || v == "no") return;
    }
    fs::path base;
    const char *dir = std::getenv("MOMWIRE_SOMM_CACHE_DIR");
    if (dir != nullptr && dir[0] != '\0') {
        base = dir;
    } else {
#if defined(_WIN32)
        const char *la = std::getenv("LOCALAPPDATA");
        const char *home = std::getenv("USERPROFILE");
        if (la != nullptr && la[0] != '\0')
            base = fs::path(la);
        else if (home != nullptr)
            base = fs::path(home) / "AppData" / "Local";
        else
            return;
        base = base / "momwire" / "Cache" / "sommerfeld";
#elif defined(__APPLE__)
        const char *home = std::getenv("HOME");
        if (home == nullptr) return;
        base = fs::path(home) / "Library" / "Caches" / "momwire" / "sommerfeld";
#else
        const char *xdg = std::getenv("XDG_CACHE_HOME");
        const char *home = std::getenv("HOME");
        if (xdg != nullptr && xdg[0] != '\0')
            base = fs::path(xdg);
        else if (home != nullptr)
            base = fs::path(home) / ".cache";
        else
            return;
        base = base / "momwire" / "sommerfeld";
#endif
    }
    // The tag names a directory: keep it to a portable alphabet.
    std::string safe;
    for (char c : tag) safe += (std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '-') ? c : '_';
    if (safe.empty()) return;
    const char *mb = std::getenv("MOMWIRE_SOMM_CACHE_MAX_MB");
    if (mb != nullptr && mb[0] != '\0') {
        char *end = nullptr;
        const double v = std::strtod(mb, &end);
        if (end != mb && std::isfinite(v)) g_disk.max_bytes = static_cast<uint64_t>(std::max(v, 0.0) * (1 << 20));
    }
    g_disk.on = true;
    g_disk.dir = base / "exe" / safe;
    g_disk.tag = tag;
}

DiskStats disk_stats() { return g_stats; }

bool disk_location(std::filesystem::path &dir, std::string &tag) {
    if (!g_disk.on) return false;
    dir = g_disk.dir;
    tag = g_disk.tag;
    return true;
}

void disk_evict(const std::filesystem::path &keep) {
    if (g_disk.on) evict(keep);
}

namespace {

using CKey = std::tuple<double, double, double, double, double>;  // eps re/im, r1b, lattice re/im
using VKey = std::tuple<double, double, double, double, double, double>;

// A normalized master: its layout always, its nodes as asked (momwire#1512).
struct Master {
    double key[5];
    Grid g;
};

// A physical view (`get_grid`'s answer): `scaled_to(blend(lo, hi, t))`, or
// `scaled_to(lo)` for one rung, node by node as asked.
struct View {
    Grid g;
    Master *lo = nullptr, *hi = nullptr;
    double t = 0.0, factor = 1.0;
};

std::mutex g_memo_mu;

Master &norm_master(cd rung, double r1b_wl, cd lattice_eps) {
    static std::map<CKey, std::unique_ptr<Master>> cache;
    const CKey key{rung.real(), rung.imag(), r1b_wl, lattice_eps.real(), lattice_eps.imag()};
    auto it = cache.find(key);
    if (it == cache.end()) {
        auto m = std::make_unique<Master>();
        const double dkey[5] = {rung.real(), rung.imag(), r1b_wl, lattice_eps.real(),
                                lattice_eps.imag()};
        std::copy(dkey, dkey + 5, m->key);
        m->g = layout_grid(rung, K2_REF, r1b_wl, K2_REF * C_LIGHT, MU0, lattice_eps);
        if (g_disk.on) {
            if (disk_merge(dkey, m->g) >= 0)
                g_stats.hits++;
            else
                g_stats.misses++;
        }
        it = cache.emplace(key, std::move(m)).first;
    }
    return *it->second;
}

// The nodes of `need` the master does not hold, filled: a stored copy first
// (another process may have filled them since), then the fill, then the
// file written back with the union.
void ensure_master(Master &m, const std::vector<std::vector<uint8_t>> &need) {
    auto missing = [&]() {
        for (size_t r = 0; r < m.g.regions.size(); r++) {
            const std::vector<uint8_t> &h = m.g.regions[r].have;
            if (r >= need.size() || need[r].empty()) continue;
            for (size_t q = 0; q < h.size(); q++)
                if (need[r][q] && !h[q]) return true;
        }
        return false;
    };
    if (!missing()) return;
    if (g_disk.on && disk_merge(m.key, m.g) > 0 && !missing()) return;
    fill_nodes(m.g, need, 1e-6);
    if (g_disk.on) disk_store(m.key, m.g);
}

std::map<VKey, std::unique_ptr<View>> &views() {
    static std::map<VKey, std::unique_ptr<View>> v;
    return v;
}

View *view_of(const Grid &g) {
    for (auto &kv : views())
        if (&kv.second->g == &g) return kv.second.get();
    return nullptr;
}

}  // namespace

// `MOMWIRE_SOMM_LAZY` = 0 / off / false / no: every node filled when a
// view is made (the eager fill's work; the values are the same either way).
bool lazy_grids() {
    static const bool on = [] {
        const char *e = std::getenv("MOMWIRE_SOMM_LAZY");
        if (e == nullptr) return true;
        std::string v(e);
        for (char &c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return !(v == "0" || v == "off" || v == "false" || v == "no");
    }();
    return on;
}

namespace {

void ensure_view(View &v, const std::vector<std::vector<uint8_t>> &need) {
    // The view's nodes asked and not held, then the masters' same nodes.
    std::vector<std::vector<uint8_t>> fresh(v.g.regions.size());
    bool any = false;
    for (size_t r = 0; r < v.g.regions.size(); r++) {
        const std::vector<uint8_t> &h = v.g.regions[r].have;
        if (r >= need.size() || need[r].empty()) continue;
        if (need[r].size() != h.size()) throw std::logic_error("ensure_nodes: a mask is not the region's");
        fresh[r].assign(h.size(), 0);
        for (size_t q = 0; q < h.size(); q++)
            if (need[r][q] && !h[q]) {
                fresh[r][q] = 1;
                any = true;
            }
    }
    if (!any) return;
    ensure_master(*v.lo, fresh);
    if (v.hi != nullptr) ensure_master(*v.hi, fresh);
    // `blend` then `scaled_to`, per node: np_add(np_mul(1 - t, a), np_mul(t,
    // b)), then np_mul(., factor) -- each node its own elementwise ops.
    const cd wa(1.0 - v.t, 0.0), wb(v.t, 0.0), f(v.factor, 0.0);
    for (size_t r = 0; r < v.g.regions.size(); r++) {
        if (fresh[r].empty()) continue;
        GridRegion &R = v.g.regions[r];
        const GridRegion &A = v.lo->g.regions[r];
        const size_t mm = R.have.size();
        for (size_t q = 0; q < mm; q++) {
            if (!fresh[r][q]) continue;
            for (int k = 0; k < 4; k++) {
                const size_t i = k * mm + q;
                cd x = A.vals[i];
                if (v.hi != nullptr) x = np_add(np_mul(wa, x), np_mul(wb, v.hi->g.regions[r].vals[i]));
                R.vals[i] = np_mul(x, f);
            }
            R.have[q] = 1;
        }
    }
}

View &make_view(cd eps_t, double k2, double r1_max, double omega, double mu) {
    const double lam = 2.0 * PI / k2;
    const double r1_wl = (r1_max / lam) * (1.0 + R1_SWEEP_SLACK);
    const double r1b_wl = r1_bucket_wl(py_min(r1_wl, R1_CAP_LAMBDA));
    cd lo, hi;
    double t;
    eps_rungs(eps_t, lo, hi, t);
    const cd keyed = lo == hi ? lo : eps_t;
    const VKey key{keyed.real(), keyed.imag(), k2, r1b_wl, omega, mu};
    auto it = views().find(key);
    if (it == views().end()) {
        auto v = std::make_unique<View>();
        v->lo = &norm_master(lo, r1b_wl, hi);
        if (lo != hi) v->hi = &norm_master(hi, r1b_wl, hi);
        v->t = t;
        // The view's layout: `blend` keeps lo's lattice and takes eps_t;
        // `scaled_to` scales the radii (the masks and values start empty).
        Grid master = v->lo->g;
        if (lo != hi) master.eps_t = eps_t;
        const double scale = master.k2 / k2;
        v->factor = (omega * mu) / (master.omega * master.mu);
        Grid &g = v->g;
        g = master;
        g.k2 = k2;
        g.omega = omega;
        g.mu = mu;
        g.r1_max = master.r1_max * scale;
        g.r_break = master.r_break * scale;
        g.r_near = master.r_near * scale;
        g.r0_fill = master.r0_fill * scale;
        for (GridRegion &R : g.regions) {
            R.r0 = R.r0 * scale;
            R.dr = R.dr * scale;
            std::fill(R.vals.begin(), R.vals.end(), cd(std::nan(""), std::nan("")));
            std::fill(R.have.begin(), R.have.end(), uint8_t(0));
        }
        it = views().emplace(key, std::move(v)).first;
    }
    return *it->second;
}

std::vector<std::vector<uint8_t>> all_nodes(const Grid &g) {
    std::vector<std::vector<uint8_t>> all;
    for (const GridRegion &R : g.regions) all.emplace_back(R.have.size(), 1);
    return all;
}

}  // namespace

cd py_quot(cd a, cd b) { return py_div(a, b); }

const Grid &get_grid_lazy(cd eps_t, double k2, double r1_max, double omega, double mu) {
    std::lock_guard<std::mutex> lock(g_memo_mu);
    View &v = make_view(eps_t, k2, r1_max, omega, mu);
    if (!lazy_grids()) ensure_view(v, all_nodes(v.g));
    return v.g;
}

void ensure_nodes(const Grid &view, const std::vector<std::vector<uint8_t>> &need) {
    std::lock_guard<std::mutex> lock(g_memo_mu);
    View *v = view_of(view);
    if (v == nullptr) throw std::logic_error("ensure_nodes: not a view get_grid_lazy made");
    ensure_view(*v, need);
}

const Grid &get_grid(cd eps_t, double k2, double r1_max, double omega, double mu) {
    std::lock_guard<std::mutex> lock(g_memo_mu);
    View &v = make_view(eps_t, k2, r1_max, omega, mu);
    ensure_view(v, all_nodes(v.g));
    return v.g;
}

bool lazy_check() {
    static const bool on = [] {
        const char *e = std::getenv("MOMWIRE_SOMM_LAZY_CHECK");
        return e != nullptr && e[0] != '\0' && std::string(e) != "0";
    }();
    return on;
}

namespace {
std::mutex g_check_mu;
std::map<std::string, std::pair<uint64_t, uint64_t>> *g_checked = nullptr;  // reads, values
void report_checked() {
    std::lock_guard<std::mutex> lock(g_check_mu);
    if (g_checked == nullptr) return;
    for (const auto &kv : *g_checked)
        std::fprintf(stderr, "MOMWIRE_SOMM_LAZY_CHECK ok: %s: %llu reads, %llu values finite\n",
                     kv.first.c_str(), static_cast<unsigned long long>(kv.second.first),
                     static_cast<unsigned long long>(kv.second.second));
}
}  // namespace

void check_read(const cd *v, size_t n, const char *who) {
    {
        // What was checked, reported at exit: a check that ran over nothing
        // proves nothing.
        std::lock_guard<std::mutex> lock(g_check_mu);
        if (g_checked == nullptr) {
            g_checked = new std::map<std::string, std::pair<uint64_t, uint64_t>>();
            std::atexit(report_checked);
        }
        auto &c = (*g_checked)[who];
        c.first++;
        c.second += n;
    }
    for (size_t i = 0; i < n; i++)
        if (!std::isfinite(v[i].real()) || !std::isfinite(v[i].imag())) {
            std::fprintf(stderr, "MOMWIRE_SOMM_LAZY_CHECK: %s read a non-finite value at %zu of %zu\n",
                         who, i, n);
            std::fflush(stderr);
            std::abort();
        }
}

}  // namespace somm_core
