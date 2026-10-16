// `_sommerfeld.get_grid` with no Python — see `_somm_grid_core.h`. Each
// function names the Python it copies; the arithmetic is that Python's, one
// IEEE operation at a time (this TU is built -ffp-contract=off; MSVC asks for
// precise semantics below), the fused complex product spelled with std::fma.
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

// CPython's `cmath.sqrt` for a finite value (numpy's complex sqrt equals it:
// measured on AVX2 and AVX-512); glibc's csqrt is another algorithm.
cd py_csqrt(cd z) {
    if (!std::isfinite(z.real()) || !std::isfinite(z.imag()))
        throw std::domain_error("a non-finite complex square root");
    if (z.real() == 0.0 && z.imag() == 0.0) return cd(0.0, z.imag());
    double ax = std::fabs(z.real());
    const double ay = std::fabs(z.imag());
    double sq;
    if (ax < std::numeric_limits<double>::min() && ay < std::numeric_limits<double>::min()) {
        ax = std::ldexp(ax, 53);
        sq = std::ldexp(std::sqrt(ax + std::hypot(ax, std::ldexp(ay, 53))), -27);
    } else {
        ax /= 8.0;
        sq = 2.0 * std::sqrt(ax + std::hypot(ax, ay / 8.0));
    }
    const double d = ay / (2.0 * sq);
    if (z.real() >= 0.0) return cd(sq, std::copysign(d, z.imag()));
    return cd(d, std::copysign(sq, z.imag()));
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

Grid fill_grid(cd eps_t, double k2, double r1_max, double rtol, double omega, double mu,
               cd lattice_eps) {
    Grid g;
    g.eps_t = eps_t;
    g.lattice_eps = lattice_eps;
    g.k2 = k2;
    g.omega = omega;
    g.mu = mu;
    const double lam = 2.0 * PI / g.k2;
    g.r1_max = py_min(py_max(r1_max, 0.35 * lam), R1_CAP_LAMBDA * lam);
    // k1 = self.k2 * np.sqrt(self.lattice_eps): numpy scalar arithmetic
    const cd root = py_csqrt(lattice_eps);  // np.sqrt == cmath.sqrt (measured)
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
        std::vector<double> rr, tt;
        for (long long i = 0; i < n_r; i++)
            for (long long j = 0; j < n_th; j++) {
                double r = L.r0 + L.dr * static_cast<double>(i);
                if (reg_idx == 1 && g.r0_fill > 0.0 && i == 0) r = g.r0_fill;
                rr.push_back(r);
                tt.push_back((L.th0 + L.dth * static_cast<double>(j)) * DEG_TO_RAD);
            }
        const std::vector<Surf> surf = iv_surfaces(g.eps_t, g.k2, rr, tt, rtol, g.omega, g.mu);
        GridRegion R;
        R.r0 = L.r0;
        R.dr = L.dr;
        R.n_r = n_r;
        R.th0 = L.th0 * DEG_TO_RAD;
        R.dth = L.dth * DEG_TO_RAD;
        R.n_th = n_th;
        const std::size_t m = surf.size();
        R.vals.resize(4 * m);
        for (int k = 0; k < 4; k++)
            for (std::size_t i = 0; i < m; i++) R.vals[k * m + i] = surf[i].v[k];
        g.regions.push_back(std::move(R));
    }
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

const char MAGIC[] = "MWSOMMGRID1\n";

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
    std::string b(MAGIC);
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
        b.append(reinterpret_cast<const char *>(r.vals.data()), r.vals.size() * sizeof(cd));
    }
    put(b, fnv1a(b));
    return b;
}

bool deserialize(const std::string &b, const double key[5], Grid &g) {
    const size_t m = sizeof(MAGIC) - 1;
    if (b.size() < m + sizeof(uint64_t) || b.compare(0, m, MAGIC) != 0) return false;
    uint64_t sum;
    std::memcpy(&sum, b.data() + b.size() - sizeof sum, sizeof sum);
    if (fnv1a(b.substr(0, b.size() - sizeof sum)) != sum) return false;
    size_t at = m;
    uint64_t tl;
    if (!get(b, at, tl) || tl > b.size() - at || b.compare(at, tl, g_disk.tag) != 0 ||
        tl != g_disk.tag.size())
        return false;
    at += tl;
    for (int i = 0; i < 5; i++) {
        double k;
        if (!get(b, at, k) || bits(k) != bits(key[i])) return false;
    }
    double sc[13];
    for (double &v : sc)
        if (!get(b, at, v)) return false;
    g.eps_t = cd(sc[0], sc[1]);
    g.lattice_eps = cd(sc[2], sc[3]);
    g.k2 = sc[4];
    g.omega = sc[5];
    g.mu = sc[6];
    g.r1_max = sc[7];
    g.r_break = sc[8];
    g.r_near = sc[9];
    g.r0_fill = sc[10];
    uint64_t n_reg;
    if (!get(b, at, n_reg) || n_reg > 64) return false;
    g.regions.assign(static_cast<size_t>(n_reg), GridRegion());
    for (GridRegion &r : g.regions) {
        uint64_t nv;
        if (!get(b, at, r.r0) || !get(b, at, r.dr) || !get(b, at, r.n_r) || !get(b, at, r.th0) ||
            !get(b, at, r.dth) || !get(b, at, r.n_th) || !get(b, at, nv))
            return false;
        if (r.n_r < 0 || r.n_th < 0 || nv != static_cast<uint64_t>(4 * r.n_r * r.n_th) ||
            nv > (b.size() - at) / sizeof(cd))
            return false;
        r.vals.resize(static_cast<size_t>(nv));
        std::memcpy(r.vals.data(), b.data() + at, nv * sizeof(cd));
        at += nv * sizeof(cd);
    }
    return at + sizeof(uint64_t) == b.size();
}

bool disk_load(const double key[5], Grid &g) {
    std::error_code ec;
    const fs::path path = master_path(key);
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::stringstream ss;
    ss << in.rdbuf();
    if (!in.good() && !in.eof()) return false;
    if (!deserialize(ss.str(), key, g)) {
        g_stats.rejected++;
        return false;
    }
    fs::last_write_time(path, fs::file_time_type::clock::now(), ec);  // LRU
    return true;
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

void disk_store(const double key[5], const Grid &g) {
    std::error_code ec;
    fs::create_directories(g_disk.dir, ec);
    if (ec) return;
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

namespace {

using CKey = std::tuple<double, double, double, double, double>;  // eps re/im, r1b, lattice re/im
using VKey = std::tuple<double, double, double, double, double, double>;

const Grid &norm_master(cd rung, double r1b_wl, cd lattice_eps) {
    static std::map<CKey, std::unique_ptr<Grid>> cache;
    const CKey key{rung.real(), rung.imag(), r1b_wl, lattice_eps.real(), lattice_eps.imag()};
    auto it = cache.find(key);
    if (it == cache.end()) {
        const double dkey[5] = {rung.real(), rung.imag(), r1b_wl, lattice_eps.real(),
                                lattice_eps.imag()};
        auto g = std::make_unique<Grid>();
        if (g_disk.on && disk_load(dkey, *g)) {
            g_stats.hits++;
        } else {
            *g = fill_grid(rung, K2_REF, r1b_wl, 1e-6, K2_REF * C_LIGHT, MU0, lattice_eps);
            if (g_disk.on) {
                g_stats.misses++;
                disk_store(dkey, *g);
            }
        }
        it = cache.emplace(key, std::move(g)).first;
    }
    return *it->second;
}

}  // namespace

cd py_quot(cd a, cd b) { return py_div(a, b); }

const Grid &get_grid(cd eps_t, double k2, double r1_max, double omega, double mu) {
    static std::map<VKey, std::unique_ptr<Grid>> views;
    const double lam = 2.0 * PI / k2;
    const double r1_wl = (r1_max / lam) * (1.0 + R1_SWEEP_SLACK);
    const double r1b_wl = r1_bucket_wl(py_min(r1_wl, R1_CAP_LAMBDA));
    cd lo, hi;
    double t;
    eps_rungs(eps_t, lo, hi, t);
    const cd keyed = lo == hi ? lo : eps_t;
    const VKey key{keyed.real(), keyed.imag(), k2, r1b_wl, omega, mu};
    auto it = views.find(key);
    if (it == views.end()) {
        Grid master = norm_master(lo, r1b_wl, hi);
        if (lo != hi) master = blend(master, norm_master(hi, r1b_wl, hi), t, eps_t);
        it = views.emplace(key, std::make_unique<Grid>(scaled_to(master, k2, omega, mu))).first;
    }
    return *it->second;
}

}  // namespace somm_core
