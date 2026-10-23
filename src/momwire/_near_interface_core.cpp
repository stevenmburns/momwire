// The near-interface column twin and the plane-sheet interpolator with no
// Python (momwire#1512 B4a): the kernels behind `near_interface_six_columns`
// and `near_interface_grid_sheet`, moved verbatim from
// `_near_interface_accel.cpp`, which binds them. The TU keeps that file's
// preamble exactly -- the `omp declare simd` exp / cos / sin (libmvec, simdlen
// 4) and `MW_NI_SIMD` before `_near_interface_columns_inline.h` -- so the
// columns vectorize as they did. The standalone executable links it.
#define _USE_MATH_DEFINES

#include "_near_interface_core.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// The sheet's fixed-width cell sums (MW_SHEET_FIXED below) mirror an
// mw_fma::fma chain, so the x86-64 baseline build may run them on the lane
// layer's unfused portable backend and keep its bits (_lanes.h). Only this
// TU opts in.
#define MW_LANES_BASELINE_UNFUSED 1
#include "_lanes.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "_contour_engine_inline.h"
#include "_contour_lanes_inline.h"
#include "_branch_cut_inline.h"
// The vendored xsf defines non-inline functions, so one link unit may hold one
// copy of each: this TU's are renamed (the extension's near module, and the
// standalone executable, which also links `_razor_wire_loading_core.cpp`'s;
// `mw` is the executable's serial copy's rename when it is one).
#undef xsf
#define MW_NI_CAT2(a, b) a##b
#define MW_NI_CAT(a, b) MW_NI_CAT2(a, b)
#define xsf MW_NI_CAT(mw, _ni_xsf)
#include "xsf/bessel.h"
#include <atomic>

#include "_lanes.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "_contour_engine_inline.h"
#include "_branch_cut_inline.h"
#include "xsf/bessel.h"
#include "xsf/faddeeva.h"

// The column entry's inner loop is exp + sincos per node per member, which
// is exactly what glibc's libmvec vectorizes (`-lmvec` is already on this
// extension's link line). Ubuntu's <cmath> carries no `omp declare simd`
// markers for them, so GCC cannot substitute `_ZGVdN4v_exp` / `_ZGVdN4v_cos`
// / `_ZGVdN4v_sin` inside an `omp simd` loop without these declarations; the
// std:: overloads still resolve to the same extern-C symbols, so the calls
// below pick the vectorized form up for free. Same block, same gating and
// same reason as `_accel_common.h`'s (the sibling extension's) — MSVC has no
// libmvec and would choke on redeclaring the CRT's exp/sin/cos, and macOS
// has neither libmvec nor the AVX2 simdlen(4) form.
#if defined(__GNUC__) && !defined(_MSC_VER) && !defined(__APPLE__)
#pragma omp declare simd notinbranch simdlen(4)
extern "C" double exp(double);

#pragma omp declare simd notinbranch simdlen(4)
extern "C" double cos(double);

#pragma omp declare simd notinbranch simdlen(4)
extern "C" double sin(double);
#endif

// `omp simd` neutralization for MSVC, whose /openmp:llvm rejects the
// directive outright (the `_accel_common.h` note has the long form). The
// pragmas below are a vectorization hint and nothing else: the arithmetic is
// a plain reduction either way, so dropping them costs speed, never bits.
#if defined(_MSC_VER)
#define MW_NI_SIMD(clauses)
#else
#define MW_NI_PRAGMA_(x) _Pragma(#x)
#define MW_NI_SIMD(clauses) MW_NI_PRAGMA_(omp simd clauses)
#endif

#include "_vmath_inline.h"
#include "_near_interface_columns_inline.h"

#define NI_CANCEL_SETUP(flag_addr)                                          \
    const volatile int32_t *ni_cancel =                                        \
        reinterpret_cast<const volatile int32_t *>(flag_addr);                 \
    std::atomic<bool> ni_aborted { false }
#define NI_CANCEL_POLL()                                                    \
    if (ni_aborted.load(std::memory_order_relaxed)) continue;                  \
    if (ni_cancel && *ni_cancel) {                                             \
        ni_aborted.store(true, std::memory_order_relaxed);                     \
        continue;                                                              \
    }
#define NI_THROW_IF_ABORTED()                                               \
    if (ni_aborted.load(std::memory_order_relaxed) || (ni_cancel && *ni_cancel)) \
        throw ni_core::Aborted {}

namespace ni_core {
namespace {
using mw899::HeadMid;
using mw899::cd;
struct ColumnCacheImpl {
  public:
    explicit ColumnCacheImpl(size_t max_bytes) : max_bytes_(max_bytes) {}

    struct Key {
        uint64_t rho, a_head, lam_top;
        bool operator==(const Key &o) const {
            return rho == o.rho && a_head == o.a_head && lam_top == o.lam_top;
        }
    };
    struct KeyHash {
        size_t operator()(const Key &k) const {
            uint64_t h = k.rho * 0x9E3779B97F4A7C15ULL;
            h ^= k.a_head + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
            h ^= k.lam_top + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
            return static_cast<size_t>(h);
        }
    };
    static uint64_t bits(double x) {
        uint64_t u;
        std::memcpy(&u, &x, sizeof u);
        return u;
    }
    static Key key(double rho, double a_head, double lam_top) {
        return Key{bits(rho), bits(a_head), bits(lam_top)};
    }

    // Empties the cache unless (k_p, k_m, p, detour, gx, gw) are the ones its
    // entries were built under, bit for bit.
    void set_context(double k_p, const cd &k_m, int p, double detour,
                     const double *gx, const double *gw, int ng) {
        std::vector<uint64_t> c;
        c.reserve(6 + 2 * ng);
        c.push_back(bits(k_p));
        c.push_back(bits(k_m.real()));
        c.push_back(bits(k_m.imag()));
        c.push_back(static_cast<uint64_t>(p));
        c.push_back(bits(detour));
        c.push_back(static_cast<uint64_t>(ng));
        for (int i = 0; i < ng; ++i) c.push_back(bits(gx[i]));
        for (int i = 0; i < ng; ++i) c.push_back(bits(gw[i]));
        std::lock_guard<std::mutex> lk(mu_);
        if (c != context_) {
            map_.clear();
            order_.clear();
            bytes_ = 0;
            context_.swap(c);
        }
    }

    std::shared_ptr<const HeadMid> find(const Key &k) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = map_.find(k);
        if (it == map_.end()) {
            ++misses_;
            return nullptr;
        }
        ++hits_;
        return it->second;
    }

    void insert(const Key &k, std::shared_ptr<const HeadMid> v) {
        std::lock_guard<std::mutex> lk(mu_);
        const size_t b = v->bytes();
        if (b > max_bytes_ || map_.count(k)) return;
        while (bytes_ + b > max_bytes_ && !order_.empty()) {
            auto it = map_.find(order_.front());
            bytes_ -= it->second->bytes();
            map_.erase(it);
            order_.pop_front();
            ++evictions_;
        }
        map_.emplace(k, std::move(v));
        order_.push_back(k);
        bytes_ += b;
    }

    size_t hits() const { return hits_; }
    size_t misses() const { return misses_; }
    size_t evictions() const { return evictions_; }
    size_t entries() const { return map_.size(); }
    size_t bytes() const { return bytes_; }
    size_t max_bytes() const { return max_bytes_; }

  private:
    size_t max_bytes_;
    size_t bytes_ = 0, hits_ = 0, misses_ = 0, evictions_ = 0;
    std::vector<uint64_t> context_;
    std::unordered_map<Key, std::shared_ptr<const HeadMid>, KeyHash> map_;
    std::deque<Key> order_;
    std::mutex mu_;
};

}  // namespace

struct ColumnCache::Impl : ColumnCacheImpl {
    using ColumnCacheImpl::ColumnCacheImpl;
};
ColumnCache::ColumnCache(size_t max_bytes) : impl(new Impl(max_bytes)) {}
ColumnCache::~ColumnCache() = default;
size_t ColumnCache::hits() const { return impl->hits(); }
size_t ColumnCache::misses() const { return impl->misses(); }
size_t ColumnCache::evictions() const { return impl->evictions(); }
size_t ColumnCache::entries() const { return impl->entries(); }
size_t ColumnCache::bytes() const { return impl->bytes(); }
size_t ColumnCache::max_bytes() const { return impl->max_bytes(); }

}  // namespace ni_core

namespace mw899 {

// One column's rule and factors into `s`, through `cache` when there is one:
// the head + mid factors from `hm` (a hit) or built here into `*fresh` (a
// miss), the tail built as always. Returns K, the rule's node count.
static size_t column_build(double rho, double k_p, const cd &kpc,
                           const cd &km, double s_min, double lam_mult, int p,
                           double detour, const double *gx, const double *gw,
                           int ng, bool cached, double a_head, double lam_top,
                           const HeadMid *hm,
                           std::shared_ptr<const HeadMid> *fresh, Scratch &s) {
    if (!cached) {
        column_rule(rho, k_p, km, s_min, lam_mult, p, detour, gx, gw, ng, s);
        column_factors(kpc, km, s);
        return s.lam.size();
    }
    if (hm == nullptr) {
        auto h = std::make_shared<HeadMid>();
        s.lam.clear();
        s.w.clear();
        column_head_mid(rho, k_p, km, a_head, lam_top, p, detour, gx, gw, ng, s);
        head_mid_factors(kpc, km, s, *h);
        hm = h.get();
        *fresh = std::move(h);
    }
    s.lam.clear();
    s.w.clear();
    column_tail(rho, s_min, lam_top, p, gx, gw, ng, s);
    return column_factors_joined(kpc, km, *hm, s);
}
}  // namespace mw899

// The steepest-descent route for FAR columns (momwire#1595 D1b):
// `_near_interface._six_columns_sdp`'s twin, the same nodes and the same
// decisions. With lam = k_p sin w and gamma_p = j k_p cos w, the H0^(2) half
// of the J0 integral is a Gaussian e^{-k_p R s^2} on the path cos(w - theta)
// = 1 - j s^2, so one fixed rule (96 Gauss nodes on s, plus a pseudo-node for
// the subtracted Zenneck pole and 2 x 32 keyhole nodes down k_m's cut when
// the path passes left of it) serves every far column that the head / mid /
// tail rule needs thousands of nodes for. The Python walk is the reference;
// the constants are its, transcribed (a twin that let them drift would not
// be a twin). The rule is built into the Scratch planes `column_factors`
// writes, so `column_member` sums it unchanged.
namespace mw1595 {
using cd = std::complex<double>;
using mw899::Scratch;

static const double SDP_KR_MIN = 30.0;
static const double SDP_KRHO_MIN = 20.0;
static const double SDP_SPREAD = 2.0;
static const int SDP_N = 96;
static const int SDP_NCUT = 32;
static const double SDP_TAIL = 60.0;
static const double SDP_CUT_T = 80.0;
static const double SDP_A_MAX = 6.0;

// numpy's `leggauss(96)` and `leggauss(32)`, transcribed (repr, so the same
// floats): a Newton rule here differs from numpy's in the last bits of the
// small end weights (1.5e-12 relative at n = 96), and the twin reads the
// walk's own nodes, as `MW_TAIL_GX` does.
static const double SDP_GX[] = {-0.9996895038832307, -0.9983643758631817, -0.9959818429872093, -0.9925439003237626, -0.9880541263296237, -0.9825172635630147, -0.9759391745851365, -0.9683268284632642, -0.9596882914487426, -0.9500327177844377, -0.9393703397527552, -0.9277124567223087, -0.9150714231208981, -0.9014606353158523, -0.8868945174024204, -0.8713885059092965, -0.8549590334346014, -0.8376235112281871, -0.8194003107379316, -0.8003087441391408, -0.7803690438674332, -0.7596023411766475, -0.7380306437444001, -0.7156768123489676, -0.6925645366421715, -0.6687183100439161, -0.6441634037849671, -0.6189258401254686, -0.593032364777572, -0.5665104185613972, -0.5393881083243575, -0.5116941771546677, -0.48345797392059636, -0.454709422167743, -0.42547898840730053, -0.3957976498289086, -0.3656968614723136, -0.3352085228926254, -0.30436494435449635, -0.27319881259104917, -0.24174315616384, -0.2100313104605672, -0.17809688236761861, -0.14597371465489695, -0.11369585011066592, -0.08129749546442555, -0.04881298513604974, -0.016276744849602967, 0.016276744849602967, 0.04881298513604974, 0.08129749546442555, 0.11369585011066592, 0.14597371465489695, 0.17809688236761861, 0.2100313104605672, 0.24174315616384, 0.27319881259104917, 0.30436494435449635, 0.3352085228926254, 0.3656968614723136, 0.3957976498289086, 0.42547898840730053, 0.454709422167743, 0.48345797392059636, 0.5116941771546677, 0.5393881083243575, 0.5665104185613972, 0.593032364777572, 0.6189258401254686, 0.6441634037849671, 0.6687183100439161, 0.6925645366421715, 0.7156768123489676, 0.7380306437444001, 0.7596023411766475, 0.7803690438674332, 0.8003087441391408, 0.8194003107379316, 0.8376235112281871, 0.8549590334346014, 0.8713885059092965, 0.8868945174024204, 0.9014606353158523, 0.9150714231208981, 0.9277124567223087, 0.9393703397527552, 0.9500327177844377, 0.9596882914487426, 0.9683268284632642, 0.9759391745851365, 0.9825172635630147, 0.9880541263296237, 0.9925439003237626, 0.9959818429872093, 0.9983643758631817, 0.9996895038832307};
static const double SDP_GW[] = {0.0007967920655518723, 0.0018539607889441585, 0.0029107318179352943, 0.003964554338444405, 0.005014202742928604, 0.006058545504235195, 0.007096470791153821, 0.0081268769256983, 0.009148671230783011, 0.010160770535008306, 0.01116210209983861, 0.012151604671088057, 0.013128229566961646, 0.014090941772314894, 0.015038721026994927, 0.015970562902562345, 0.016885479864245195, 0.017782502316045286, 0.01866067962741174, 0.019519081140145382, 0.020356797154333365, 0.021172939892191354, 0.02196664443874457, 0.022737069658329466, 0.023483399085926292, 0.02420484179236482, 0.024900633222483814, 0.025570036005349364, 0.026212340735672593, 0.02682686672559185, 0.027412962726029232, 0.02797000761684837, 0.028497411065085413, 0.02899461415055532, 0.029461089958168016, 0.029896344136328506, 0.03029991542082777, 0.030671376123669266, 0.03101033258631393, 0.03131642559686141, 0.03158933077072725, 0.03182875889441112, 0.032034456231992796, 0.03220620479403032, 0.03234382256857602, 0.0324471637140644, 0.03251611871386895, 0.03255061449236328, 0.03255061449236328, 0.03251611871386895, 0.0324471637140644, 0.03234382256857602, 0.03220620479403032, 0.032034456231992796, 0.03182875889441112, 0.03158933077072725, 0.03131642559686141, 0.03101033258631393, 0.030671376123669266, 0.03029991542082777, 0.029896344136328506, 0.029461089958168016, 0.02899461415055532, 0.028497411065085413, 0.02797000761684837, 0.027412962726029232, 0.02682686672559185, 0.026212340735672593, 0.025570036005349364, 0.024900633222483814, 0.02420484179236482, 0.023483399085926292, 0.022737069658329466, 0.02196664443874457, 0.021172939892191354, 0.020356797154333365, 0.019519081140145382, 0.01866067962741174, 0.017782502316045286, 0.016885479864245195, 0.015970562902562345, 0.015038721026994927, 0.014090941772314894, 0.013128229566961646, 0.012151604671088057, 0.01116210209983861, 0.010160770535008306, 0.009148671230783011, 0.0081268769256983, 0.007096470791153821, 0.006058545504235195, 0.005014202742928604, 0.003964554338444405, 0.0029107318179352943, 0.0018539607889441585, 0.0007967920655518723};
static const double SDP_CX[] = {-0.9972638618494816, -0.9856115115452684, -0.9647622555875064, -0.9349060759377397, -0.8963211557660521, -0.84936761373257, -0.7944837959679424, -0.7321821187402897, -0.6630442669302152, -0.5877157572407623, -0.5068999089322294, -0.42135127613063533, -0.33186860228212767, -0.23928736225213706, -0.1444719615827965, -0.048307665687738324, 0.048307665687738324, 0.1444719615827965, 0.23928736225213706, 0.33186860228212767, 0.42135127613063533, 0.5068999089322294, 0.5877157572407623, 0.6630442669302152, 0.7321821187402897, 0.7944837959679424, 0.84936761373257, 0.8963211557660521, 0.9349060759377397, 0.9647622555875064, 0.9856115115452684, 0.9972638618494816};
static const double SDP_CW[] = {0.007018610009470506, 0.016274394730905743, 0.025392065309262024, 0.034273862913021765, 0.042835898022226836, 0.05099805926237609, 0.058684093478535565, 0.06582222277636168, 0.07234579410884834, 0.07819389578707023, 0.08331192422694671, 0.08765209300440378, 0.09117387869576378, 0.09384439908080451, 0.09563872007927471, 0.09654008851472766, 0.09654008851472766, 0.09563872007927471, 0.09384439908080451, 0.09117387869576378, 0.08765209300440378, 0.08331192422694671, 0.07819389578707023, 0.07234579410884834, 0.06582222277636168, 0.058684093478535565, 0.05099805926237609, 0.042835898022226836, 0.034273862913021765, 0.025392065309262024, 0.016274394730905743, 0.007018610009470506};
static_assert(sizeof(SDP_GX) == SDP_N * sizeof(double) && sizeof(SDP_CX) == SDP_NCUT * sizeof(double),
              "the transcribed rules are the walk's sizes");
struct Rule {
    const double *x, *w;
};
static const Rule &rule_n() {
    static const Rule r{SDP_GX, SDP_GW};
    return r;
}
static const Rule &rule_cut() {
    static const Rule r{SDP_CX, SDP_CW};
    return r;
}

static inline cd hankel2_0(const cd &x) { return xsf::cyl_hankel_2(0.0, x); }

// `_sdp_path`: w and dw/ds on cos(w - theta) = 1 - j s^2. numpy's `<` on a
// complex operand orders by the real part, then the imaginary.
static inline void path(double theta, const cd &s, cd &w, cd &dpsi) {
    const cd j(0.0, 1.0);
    cd psi = std::acos(1.0 - j * s * s);
    const cd q = psi.real() * s;
    if (q.real() < 0.0 || (q.real() == 0.0 && q.imag() < 0.0)) psi = -psi;
    w = theta + psi;
    dpsi = 2.0 * j * s / std::sin(psi);
}

// `_column_factors` at one node with both gammas given and the weight wk.
static inline void factors(const cd &lam, const cd &wk, const cd &g_p, const cd &g_m,
                           const cd &k_p, const cd &k_m, cd *f) {
    const cd u = 2.0 * lam / (g_p + g_m);
    const cd v = 2.0 * lam / (k_m * k_m * g_p + k_p * k_p * g_m);
    const cd wv = (g_p - g_m) * v;
    f[0] = u * wk;
    f[1] = v * wk;
    f[2] = wv * wk;
    f[3] = -g_p * wv * wk;
    f[4] = -g_p * g_m * v * wk;
    f[5] = g_m * wv * wk;
}

// `_sdp_factors` at one path point s, times the quadrature weight `ws`.
static inline void path_node(const cd &s, double theta, double rho, double k_p, const cd &k_m,
                             const cd &ws, cd *f, cd &g_p, cd &g_m) {
    cd w, dpsi;
    path(theta, s, w, dpsi);
    const cd lam = k_p * std::sin(w);
    const cd wn = 0.5 * hankel2_0(lam * rho) * (k_p * std::cos(w) * dpsi);
    g_p = cd(0.0, 1.0) * k_p * std::cos(w);
    g_m = mw_branch::gamma_cut(lam, k_m);
    factors(lam, wn * ws, g_p, g_m, cd(k_p, 0.0), k_m, f);
}

// `_sdp_theta`: the column's path angle and k_p R, false when the column is
// not a far one.
static bool far_column(double rho, const double *z, std::ptrdiff_t lo, std::ptrdiff_t hi,
                       double k_p, double &theta, double &kr) {
    if (!(rho > 0.0) || k_p * rho < SDP_KRHO_MIN || hi <= lo) return false;
    double rmin = std::numeric_limits<double>::infinity();
    double tmin = rmin, tmax = -rmin;
    for (std::ptrdiff_t i = lo; i < hi; ++i) {
        rmin = std::min(rmin, std::hypot(rho, z[i]));
        const double t = std::atan2(rho, z[i]);
        tmin = std::min(tmin, t);
        tmax = std::max(tmax, t);
    }
    kr = k_p * rmin;
    if (kr < SDP_KR_MIN) return false;
    theta = 0.5 * (tmin + tmax);
    return !((tmax - tmin) * 0.5 * std::sqrt(kr) > SDP_SPREAD);
}

// `_sdp_pole`: the Zenneck pole's s_p and its residue in s, false when there
// is no pole on this sheet or the contour round it does not settle.
static bool pole(double k_p, const cd &k_m, double theta, double rho, cd &sp, cd *res) {
    const cd j(0.0, 1.0);
    const cd lam = k_p * k_m / std::sqrt(k_p * k_p + k_m * k_m);
    const cd cw = j * k_p * mw_branch::gamma_cut(lam, k_m) / (k_m * k_m);
    const cd wp = -j * std::log(cw + j * lam / k_p);
    sp = std::sqrt(-j * (1.0 - std::cos(wp - theta)));
    const cd guess = (wp - theta) / cd(1.0, 1.0);
    if (std::abs(-sp - guess) < std::abs(sp - guess)) sp = -sp;
    {
        cd w, dpsi;
        path(theta, sp, w, dpsi);
        const cd lam_p = k_p * std::sin(w);
        const cd d = k_m * k_m * (j * k_p * std::cos(w)) +
                     k_p * k_p * mw_branch::gamma_cut(lam_p, k_m);
        if (std::abs(d) > 1e-9 * std::norm(k_m) * k_p) return false;
    }
    // A trapezoid contour integral round s_p; two radii must agree.
    // A contour that is not finite is no pole (the walk's words: where the
    // formula's root is spurious the circle can cross gamma_p + gamma_m = 0).
    auto contour = [&](double radius, cd *out) {
        const int m = 64;
        for (int c = 0; c < 6; ++c) out[c] = 0.0;
        for (int i = 0; i < m; ++i) {
            const double ang = 2.0 * M_PI * (i + 0.5) / m;
            const cd off = radius * cd(std::cos(ang), std::sin(ang));
            cd f[6], gp, gm;
            path_node(sp + off, theta, rho, k_p, k_m, cd(1.0, 0.0), f, gp, gm);
            for (int c = 0; c < 6; ++c) out[c] += f[c] * off;
        }
        bool finite = true;
        for (int c = 0; c < 6; ++c) {
            out[c] /= double(m);
            finite = finite && std::isfinite(out[c].real()) && std::isfinite(out[c].imag());
        }
        return finite;
    };
    double radius = (sp != cd(0.0, 0.0)) ? std::min(1e-2, 0.5 * std::abs(sp)) : 1e-2;
    if (!contour(radius, res)) return false;
    for (int it = 0; it < 6; ++it) {
        cd half[6];
        if (!contour(0.5 * radius, half)) return false;
        double dmax = 0.0, rmax = 0.0;
        for (int c = 0; c < 6; ++c) {
            dmax = std::max(dmax, std::abs(half[c] - res[c]));
            rmax = std::max(rmax, std::abs(res[c]));
        }
        if (dmax <= 1e-12 * std::max(rmax, 1e-300)) return true;
        radius *= 0.5;
        for (int c = 0; c < 6; ++c) res[c] = half[c];
    }
    return false;
}

// `_sdp_encloses`: whether the path passes left of k_m, where its s > 0
// branch reaches Im lam = Im k_m, by the walk's own bisection.
static bool encloses(double k_p, const cd &k_m, double theta, double S) {
    auto im_gap = [&](double q) {
        cd w, dpsi;
        path(theta, cd(q, 0.0), w, dpsi);
        return (k_p * std::sin(w)).imag() - k_m.imag();
    };
    double hi = S;
    while (im_gap(hi) > 0.0 && hi < 1e3) hi *= 2.0;
    if (im_gap(hi) > 0.0) return false;
    double lo = 0.0;
    for (int it = 0; it < 200 && hi - lo > 1e-15 + 8.88e-16 * hi; ++it) {
        const double mid = 0.5 * (lo + hi);
        (im_gap(mid) > 0.0 ? lo : hi) = mid;
    }
    cd w, dpsi;
    path(theta, cd(0.5 * (lo + hi), 0.0), w, dpsi);
    return (k_p * std::sin(w)).real() < k_m.real();
}

// `_sdp_nodes` into the Scratch planes (`column_factors`' layout). Returns K.
static size_t build(double rho, double k_p, const cd &k_m, double theta, double kr,
                    Scratch &s) {
    const cd j(0.0, 1.0);
    const Rule &g = rule_n();
    const double S = std::sqrt(SDP_TAIL / kr);
    std::vector<cd> F;  // node-major, 6 per node
    std::vector<cd> GP, GM;
    F.reserve(6 * (SDP_N + 1 + 2 * SDP_NCUT));
    auto push = [&](const cd *f, const cd &gp, const cd &gm) {
        for (int c = 0; c < 6; ++c) F.push_back(f[c]);
        GP.push_back(gp);
        GM.push_back(gm);
    };
    for (int i = 0; i < SDP_N; ++i) {
        cd f[6], gp, gm;
        path_node(cd(S * g.x[i], 0.0), theta, rho, k_p, k_m, cd(S * g.w[i], 0.0), f, gp, gm);
        push(f, gp, gm);
    }
    cd sp, res[6];
    if (pole(k_p, k_m, theta, rho, sp, res) && std::abs(std::sqrt(kr) * sp) <= SDP_A_MAX) {
        const cd a = std::sqrt(kr) * sp;
        const cd whole = a.imag() > 0.0 ? j * M_PI * Faddeeva::w(a, 0.0)
                                        : -j * M_PI * Faddeeva::w(-a, 0.0);
        cd gsum = 0.0;
        for (int i = 0; i < SDP_N; ++i) {
            const double si = S * g.x[i];
            gsum += std::exp(-kr * (si * si - sp * sp)) / (si - sp) * (S * g.w[i]);
        }
        const cd corr = std::exp(kr * sp * sp) * whole - gsum;
        cd f1[6], gp1, gm1;
        path_node(sp, theta, rho, k_p, k_m, cd(1.0, 0.0), f1, gp1, gm1);
        cd f[6];
        for (int c = 0; c < 6; ++c) f[c] = res[c] * corr;
        push(f, gp1, gm1);
    }
    if (k_m.imag() < 0.0 && encloses(k_p, k_m, theta, S)) {
        const Rule &c = rule_cut();
        const double t_max = SDP_CUT_T / rho;
        const double rt = std::sqrt(t_max);
        std::vector<cd> fb, gpb, gmb;
        for (int i = 0; i < SDP_NCUT; ++i) {
            const double u = 0.5 * (c.x[i] + 1.0) * rt;
            const double dt = 2.0 * u * 0.5 * rt * c.w[i];
            const cd lam(k_m.real(), k_m.imag() - u * u);
            const cd gp = mw_branch::gamma_cut(lam, cd(k_p, 0.0));
            const cd gm = mw_branch::gamma_cut(lam, k_m);
            const cd h = 0.5 * hankel2_0(lam * rho) * j * dt;
            cd f[6];
            factors(lam, h, gp, gm, cd(k_p, 0.0), k_m, f);
            push(f, gp, gm);
            factors(lam, -h, gp, -gm, cd(k_p, 0.0), k_m, f);
            for (int q = 0; q < 6; ++q) fb.push_back(f[q]);
            gpb.push_back(gp);
            gmb.push_back(-gm);
        }
        // Python's order: the 32 nodes of one cut side, then the other's.
        const size_t n0 = GP.size() - SDP_NCUT;
        std::vector<cd> Fa(F.begin() + 6 * n0, F.end()), gpa(GP.begin() + n0, GP.end()),
            gma(GM.begin() + n0, GM.end());
        F.resize(6 * n0);
        GP.resize(n0);
        GM.resize(n0);
        F.insert(F.end(), Fa.begin(), Fa.end());
        F.insert(F.end(), fb.begin(), fb.end());
        GP.insert(GP.end(), gpa.begin(), gpa.end());
        GP.insert(GP.end(), gpb.begin(), gpb.end());
        GM.insert(GM.end(), gma.begin(), gma.end());
        GM.insert(GM.end(), gmb.begin(), gmb.end());
    }
    const size_t K = GP.size();
    s.gpr.resize(K);
    s.gpi.resize(K);
    s.gmr.resize(K);
    s.gmi.resize(K);
    s.fr.resize(6 * K);
    s.fi.resize(6 * K);
    for (size_t k = 0; k < K; ++k) {
        s.gpr[k] = GP[k].real();
        s.gpi[k] = GP[k].imag();
        s.gmr[k] = GM[k].real();
        s.gmi[k] = GM[k].imag();
        for (int c = 0; c < 6; ++c) {
            s.fr[c * K + k] = F[6 * k + c].real();
            s.fi[c * K + k] = F[6 * k + c].imag();
        }
    }
    return K;
}
}  // namespace mw1595

namespace ni_core {

namespace {
// A raw array read as `unchecked<1>` reads, for `s_min_of`.
struct Arr {
    const double *p;
    double operator()(std::ptrdiff_t i) const { return p[i]; }
};
}  // namespace

void six_columns(double k_p, std::complex<double> k_m, const double *rho,
                 const std::ptrdiff_t *offsets, std::ptrdiff_t nc, const double *z,
                 const double *zp, std::ptrdiff_t n, double lam_mult, int p, double detour,
                 int n_threads, const double *gx, const double *gw, int ng,
                 uintptr_t cancel_flag, ColumnCache *cache, std::complex<double> *out,
                 bool lanes, bool sdp) {
    if (offsets[0] != 0 || offsets[nc] != n)
        throw std::invalid_argument("offsets must span 0 .. len(z)");
    for (std::ptrdiff_t c = 0; c < nc; ++c)
        if (offsets[c + 1] < offsets[c])
            throw std::invalid_argument("offsets must be non-decreasing");
    if (ng < 1) throw std::invalid_argument("bad Gauss rule");
    // `p` is the resolution ladder, not a size knob: 2^p panels per seeded
    // interval. Bounded so a typo asks for a refusal rather than a terabyte.
    if (p < 0 || p > 20) throw std::invalid_argument("column p out of range");
    const double *gxp = gx;
    const double *gwp = gw;

    // The Python walk's domain raises, BEFORE the parallel region (a throw
    // cannot cross an omp boundary) and before any column is built — a bad
    // member anywhere refuses the whole call, as `six_columns` does.
    for (std::ptrdiff_t c = 0; c < nc; ++c) {
        for (std::ptrdiff_t i = offsets[c]; i < offsets[c + 1]; ++i) {
            if (!(z[i] >= 0.0 && zp[i] <= 0.0))
                throw std::invalid_argument("need z >= 0 >= zp");
            if (rho[c] < 0.0 || (z[i] - zp[i]) + rho[c] <= 0.0)
                throw std::invalid_argument("need R > 0");
        }
    }

    const mw_contour::cd kpc(k_p, 0.0);
    const mw_contour::cd km(k_m);

    // Which columns are FAR ones (momwire#1595 D1b): those take the
    // steepest-descent rule, built per column like any other, and never
    // touch the column cache (theirs is a different rule, and cheap).
    std::vector<char> c_far(nc, 0);
    std::vector<double> c_theta(nc, 0.0), c_kr(nc, 0.0);
    if (sdp)
        for (std::ptrdiff_t c = 0; c < nc; ++c)
            c_far[c] = mw1595::far_column(rho[c], z, offsets[c], offsets[c + 1], k_p,
                                          c_theta[c], c_kr[c]);

    // The thread count is the CALLER's policy, held to what OpenMP would
    // have used anyway so a pin below it survives: `_near_interface` passes
    // the PHYSICAL core count, for the reason #898 measured on the numpy
    // route's gemm and this kernel repeats — libmvec exp/sincos saturates a
    // core's FPU, so the hyperthread siblings contend rather than add. On
    // this 4c/8t box the BLE N = 4 kernel reads 95 ms at 1 thread, 40 ms at
    // 4 and 46 ms at 8.
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
    if (n_threads > 0) nt = std::min(nt, n_threads);
    nt = std::max(nt, 1);
#endif

    // TWO parallel units, because a fill's columns differ by three orders in
    // member count and one unit cannot serve both. Measured on BLE 45 ft
    // N = 4 (the #899 census): 84 columns over four `designed_tables` calls,
    // and the largest call is one column of 3,024 members beside forty of
    // 108. Parallelising over columns alone left that one column — 40 % of
    // the call's members — on a single thread, and the kernel measured 1.0x
    // from one thread to four.
    //
    // The split is a FAIR SHARE, not a fixed size: a column carrying more
    // members than one thread's share of the whole call cannot be a unit,
    // because no schedule can divide it. Everything else stays a column,
    // which matters in the other direction — those forty 108-member columns
    // are better parallelised as columns than as forty serial rule builds.
    // At one thread the share is the whole call and nothing splits, which is
    // the right answer there too.
    //
    //   * by_column: `dynamic`, since K grows with rho (one panel per J0
    //     oscillation) and a fill's columns arrive in geometry order;
    //   * by_member: the rule built once in front of the loop, serial, then
    //     every member of it on all threads.
    const std::ptrdiff_t share = (n + nt - 1) / nt;
    std::vector<std::ptrdiff_t> by_column, by_member;
    for (std::ptrdiff_t c = 0; c < nc; ++c) {
        if (offsets[c + 1] == offsets[c]) continue;
        (offsets[c + 1] - offsets[c] > share ? by_member : by_column).push_back(c);
    }
    const std::ptrdiff_t n_col = static_cast<std::ptrdiff_t>(by_column.size());
    const std::ptrdiff_t n_mem = static_cast<std::ptrdiff_t>(by_member.size());

    // The column cache (see `mw899::ColumnCache`): every column's extents and
    // its head + mid lookup, here with the GIL held and before any thread
    // starts; the misses' new entries go in after the parallel regions.
    ColumnCacheImpl *cc = cache ? cache->impl.get() : nullptr;
    std::vector<double> c_ah, c_lt;
    std::vector<std::shared_ptr<const mw899::HeadMid>> c_hit, c_new;
    if (cc) {
        cc->set_context(k_p, km, p, detour, gxp, gwp, ng);
        c_ah.assign(nc, 0.0);
        c_lt.assign(nc, 0.0);
        c_hit.resize(nc);
        c_new.resize(nc);
        for (std::ptrdiff_t c = 0; c < nc; ++c) {
            if (offsets[c + 1] == offsets[c] || c_far[c]) continue;
            mw899::column_extents(rho[c], k_p, km,
                                  mw899::s_min_of(Arr{z}, Arr{zp}, offsets[c], offsets[c + 1]),
                                  lam_mult, c_ah[c], c_lt[c]);
            c_hit[c] = cc->find(ColumnCacheImpl::key(rho[c], c_ah[c], c_lt[c]));
        }
    }
    const bool cached = cc != nullptr;

    {
        NI_CANCEL_SETUP(cancel_flag);
        // Polled per column, and per member of a column too big to be one
        // work item; checked between those columns, whose rules are built
        // serially.
        //
        // One Scratch per THREAD, not per column (momwire#1512): every array
        // in it is rewritten in full before it is read (`column_build`), so a
        // column sees nothing of the last one but the capacity, and the
        // per-column allocate-and-zero of a dozen K-sized vectors goes.
        #pragma omp parallel num_threads(nt)
        {
        mw899::Scratch s;
        #pragma omp for schedule(dynamic)
        for (std::ptrdiff_t j = 0; j < n_col; ++j) {
            NI_CANCEL_POLL();
            const std::ptrdiff_t c = by_column[j];
            const std::ptrdiff_t lo = offsets[c], hi = offsets[c + 1];
            s.lanes = lanes && mw_sl::kActive;
            const size_t K =
                c_far[c] ? mw1595::build(rho[c], k_p, km, c_theta[c], c_kr[c], s)
                         : mw899::column_build(
                               rho[c], k_p, kpc, km, mw899::s_min_of(Arr{z}, Arr{zp}, lo, hi),
                               lam_mult, p, detour, gxp, gwp, ng, cached,
                               cached ? c_ah[c] : 0.0, cached ? c_lt[c] : 0.0,
                               cached ? c_hit[c].get() : nullptr,
                               cached ? &c_new[c] : nullptr, s);
            for (std::ptrdiff_t i = lo; i < hi; ++i) {
                mw_contour::cd o6[6];
                mw899::column_member(s, K, z[i], zp[i], o6);
                for (int q = 0; q < 6; ++q) out[6 * i + q] = o6[q];
            }
        }
        }
        for (std::ptrdiff_t j = 0; j < n_mem; ++j) {
            NI_THROW_IF_ABORTED();
            const std::ptrdiff_t c = by_member[j];
            const std::ptrdiff_t lo = offsets[c], hi = offsets[c + 1];
            mw899::Scratch s;
            s.lanes = lanes && mw_sl::kActive;
            const size_t K =
                c_far[c] ? mw1595::build(rho[c], k_p, km, c_theta[c], c_kr[c], s)
                         : mw899::column_build(
                               rho[c], k_p, kpc, km, mw899::s_min_of(Arr{z}, Arr{zp}, lo, hi),
                               lam_mult, p, detour, gxp, gwp, ng, cached,
                               cached ? c_ah[c] : 0.0, cached ? c_lt[c] : 0.0,
                               cached ? c_hit[c].get() : nullptr,
                               cached ? &c_new[c] : nullptr, s);
            // `dynamic`: members of one column do NOT cost the same. The
            // rule is converged for the column's smallest s, so a member
            // with a larger s underflows part of its own tail and skips it
            // (see MW_EXP_ZERO) — the exact saving the numpy route cannot
            // take, and it makes the cheap members the deep ones.
            #pragma omp parallel for schedule(dynamic, 32) num_threads(nt)
            for (std::ptrdiff_t i = lo; i < hi; ++i) {
                NI_CANCEL_POLL();
                mw_contour::cd o6[6];
                mw899::column_member(s, K, z[i], zp[i], o6);
                for (int q = 0; q < 6; ++q) out[6 * i + q] = o6[q];
            }
        }
        NI_THROW_IF_ABORTED();
    }
    if (cc) {
        for (std::ptrdiff_t c = 0; c < nc; ++c)
            if (c_new[c])
                cc->insert(ColumnCacheImpl::key(rho[c], c_ah[c], c_lt[c]),
                           std::move(c_new[c]));
    }
}

}  // namespace ni_core

// ---------------------------------------------------------------------------
// Sheets (momwire#1173 Design E).
//
// A PLANE sheet is one source depth z' = -d: the six kernels are analytic in
// (rho, z) over the whole above half-plane z >= 0, their only nearby singular
// point (rho, z) = (0, z'), at least d away. A HEIGHT sheet (`height`) is its
// mirror, one observer height z = h over (rho, z' <= 0), at least h from its
// singular point (0, h). Either way the table is in (rho, s), s = z - z' the
// vertical separation (s >= d), and the singular point is (0, 0).
//
// `_near_interface.PlaneSheet` lays the table on a tensor grid of cells: rho
// edges and s edges from dyadic strips (each strip at least as far from the
// singular point as it is wide, split to a fraction of the wavelength alive
// there), p x p Chebyshev (first-kind) nodes per cell, each node evaluated by
// the column twin above -- a rho node shares its column setup with every s
// node of its dyadic s strip. Cell (a, b)'s block starts at node cell_off[a,
// b] and is laid out [rho node][s node][6 kernels].
//
// The stored values are f * R^pw with R = hypot(rho, s), pw = 1 for U, V, W
// and 2 for the three derivatives, so the table carries the smooth part and
// the 1/R^pw is put back here, exactly. Interpolation is barycentric (second
// form) in each direction: no Vandermonde, no conditioning question.
//
// Each row is independent and reduces over nothing shared, so the answer does
// not depend on the thread count. The multiply-adds are spelled through
// `mw_fma` (the build passes -ffp-contract=off), so they round as written.
// Only scalar hypot is called: no vector libm, so no symbol newer than the
// Linux wheels' glibc floor (test_glibc_floor.py).
// The widest table `near_interface_grid_sheet` takes (momwire#1221). Six
// kernels today; the headroom is for the point-observer keys #1220 stage 2
// adds.
#define MW_SHEET_MAX_KEYS 16

namespace mw_sheet {

// The node test, the divisions and the normalising products are each a loop
// of their own (momwire#1290) so that the divisions vectorise: a division is
// correctly rounded in any lane, so they are the floats the one fused loop
// gave, and the sum still runs a = 0, 1, ... in order. The fused loop
// returned on the FIRST node within 1e-15; so does this one.
static inline void bary(double t, double lo, double hi, const double *x,
                        const double *bw, int p, double *w) {
    const double xx = (2.0 * t - (lo + hi)) / (hi - lo);
    int on = 0;
    for (int a = 0; a < p; ++a) on |= std::fabs(xx - x[a]) < 1e-15;
    if (on) {
        int a = 0;
        while (!(std::fabs(xx - x[a]) < 1e-15)) ++a;
        for (int b = 0; b < p; ++b) w[b] = (a == b) ? 1.0 : 0.0;
        return;
    }
    for (int a = 0; a < p; ++a) w[a] = bw[a] / (xx - x[a]);
    double sum = 0.0;
    for (int a = 0; a < p; ++a) sum += w[a];
    const double inv = 1.0 / sum;
    for (int a = 0; a < p; ++a) w[a] *= inv;
}

// The cell of t among edges e[0] < ... < e[n], or -1 outside [e[0], e[n]]
// by more than a relative 1e-12 (the ends clamp into the table).
static inline std::ptrdiff_t cell_of(double t, const double *e, std::ptrdiff_t n) {
    const double tol = 1e-12 * std::max(1.0, std::fabs(e[n]));
    if (!(t >= e[0] - tol && t <= e[n] + tol)) return -1;
    std::ptrdiff_t j =
        static_cast<std::ptrdiff_t>(std::upper_bound(e, e + n + 1, t) - e) - 1;
    if (j < 0) j = 0;
    if (j > n - 1) j = n - 1;
    return j;
}

// One row's tensor-product sum over its cell's p x p block, into acc[0, nd):
//
//     acc[k] = sum_i wr[i] * (sum_j ws[j] * V[i][j][k])
//
// each sum in index order, every step one fused multiply-add, the inner sum
// started at +0.0 and restarted per i. `cell_sum` spells it for any width;
// `cell_sum_fixed<ND>` is the same arithmetic at a compile-time width of
// ND doubles, held in registers (momwire#1290): with `nd` a runtime value
// GCC keeps `inner` in memory and every j step is a store-to-load round
// trip. It also runs the inner sums of rows i and i + 1 together. They are
// independent chains, so that changes when each fused op issues, not which
// ops run or their order within any one sum, and every acc[k] still takes i
// in order. The two are therefore the same floats entry for entry; the
// width-generic reference stays reachable (`generic=True`) and is gated
// against this one as uint64 (tests/test_sheet_fixed_width_1290.py). Only
// a lanes build (the AVX2 one) has the fixed route: the baseline and arm64 builds
// take the generic loop, which is the same floats anyway. On MSVC (/fp:fast,
// an unfused mw_fma::fma) the two are not the same floats, and the gate there
// is the derived win32 tolerance (momwire#1371).
static inline void cell_sum(const double *blk, const double *wr,
                            const double *ws, int p, int nd, double *acc) {
    for (int k = 0; k < nd; ++k) acc[k] = 0.0;
    for (int i = 0; i < p; ++i) {
        double inner[2 * MW_SHEET_MAX_KEYS] = {0.0};
        const double *Vi = blk + static_cast<std::int64_t>(i) * p * nd;
        for (int j = 0; j < p; ++j) {
            const double w = ws[j];
            const double *Vj = Vi + j * nd;
            for (int k = 0; k < nd; ++k)
                inner[k] = mw_fma::fma(w, Vj[k], inner[k]);
        }
        const double wi = wr[i];
        for (int k = 0; k < nd; ++k)
            acc[k] = mw_fma::fma(wi, inner[k], acc[k]);
    }
}

#if MW_LANES_SIMD && MW_LANES_FUSED
#define MW_SHEET_FIXED 1
// ND doubles in ND / W lane-layer vectors (momwire#1372). `fmadd` is W
// independent multiply-adds, one per lane, each rounding exactly as the
// walk's `mw_fma::fma` does on that lane's operands (fused where the build
// has FMA, a*b + c on the x86-64 baseline's unfused portable backend), so a
// lane is the scalar chain.
// Spelled in vectors because GCC 11, left to vectorise the plain loops at a
// fixed width, spilled the inner sums to the stack and gathered across j,
// which was barely faster than the generic loop. A width the backend's W does
// not divide takes the generic loop, which is the same floats.
template <int ND>
static inline void cell_sum_fixed(const double *blk, const double *wr,
                                  const double *ws, int p, double *acc) {
    using namespace mw_lanes;
    if constexpr (ND % W != 0) {
        cell_sum(blk, wr, ws, p, ND, acc);
    } else {
    constexpr int NV = ND / W;
    const std::int64_t row = static_cast<std::int64_t>(p) * ND;
    vd a[NV];
    for (int v = 0; v < NV; ++v) a[v] = zero();
    int i = 0;
    for (; i + 2 <= p; i += 2) {
        vd x0[NV], x1[NV];
        for (int v = 0; v < NV; ++v) x0[v] = x1[v] = zero();
        const double *V0 = blk + i * row;
        const double *V1 = V0 + row;
        for (int j = 0; j < p; ++j) {
            const vd w = set1(ws[j]);
            for (int v = 0; v < NV; ++v) {
                x0[v] = fmadd(w, loadu(V0 + j * ND + W * v), x0[v]);
                x1[v] = fmadd(w, loadu(V1 + j * ND + W * v), x1[v]);
            }
        }
        const vd w0 = set1(wr[i]), w1 = set1(wr[i + 1]);
        for (int v = 0; v < NV; ++v) a[v] = fmadd(w0, x0[v], a[v]);
        for (int v = 0; v < NV; ++v) a[v] = fmadd(w1, x1[v], a[v]);
    }
    for (; i < p; ++i) {
        vd x0[NV];
        for (int v = 0; v < NV; ++v) x0[v] = zero();
        const double *V0 = blk + i * row;
        for (int j = 0; j < p; ++j) {
            const vd w = set1(ws[j]);
            for (int v = 0; v < NV; ++v)
                x0[v] = fmadd(w, loadu(V0 + j * ND + W * v), x0[v]);
        }
        const vd w0 = set1(wr[i]);
        for (int v = 0; v < NV; ++v) a[v] = fmadd(w0, x0[v], a[v]);
    }
    for (int v = 0; v < NV; ++v) storeu(acc + W * v, a[v]);
    }
}
#else
#define MW_SHEET_FIXED 0
#endif

}  // namespace mw_sheet


namespace ni_core {

void grid_sheet(const double *sub, std::ptrdiff_t m, const std::ptrdiff_t *idx, std::ptrdiff_t n,
                double fixed, bool height, const double *rho_edges, std::ptrdiff_t nr,
                const double *s_edges, std::ptrdiff_t ns, const std::int64_t *cell_off,
                const double *x, const double *bw, int p, const std::complex<double> *vals,
                std::complex<double> *out, int n_threads, const std::int64_t *rpow, int nk,
                bool generic) {
    (void)m;
    const bool fixed_ok = !generic;
    (void)fixed_ok;  // read only where the fixed route is built
    const int nd = 2 * nk;  // interleaved doubles per node
    const std::int64_t *rp = rpow;
    for (int k = 0; k < nk; ++k)
        if (rp[k] < 0 || rp[k] > 4)
            throw std::invalid_argument("rpow entries must be in 0..4");
    if (height ? !(fixed > 0.0) : !(fixed < 0.0))
        throw std::invalid_argument(height ? "a height sheet needs h > 0"
                                           : "a plane sheet needs zp < 0");
    if (nr < 1 || ns < 1) throw std::invalid_argument("edges must be 1-D with at least 2 entries");
    if (p < 1 || p > 32) throw std::invalid_argument("bad Chebyshev nodes (1..32)");
    const double *re = rho_edges, *se = s_edges;
    for (std::ptrdiff_t j = 0; j < nr; ++j)
        if (!(re[j + 1] > re[j])) throw std::invalid_argument("rho edges must rise");
    for (std::ptrdiff_t j = 0; j < ns; ++j)
        if (!(se[j + 1] > se[j])) throw std::invalid_argument("s edges must rise");
    const std::int64_t *co = cell_off;
    const std::int64_t per = static_cast<std::int64_t>(p) * p;
    const std::ptrdiff_t *ix = idx;
    const double *xp = x, *bwp = bw;
    // (nodes, nk) complex as interleaved doubles: node q's kernel k at
    // nd q + 2 k (re) and + 1 (im).
    const double *V = reinterpret_cast<const double *>(vals);
    const double *sb = sub;
    double *ob = reinterpret_cast<double *>(out);

    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
    if (n_threads > 0) nt = std::min(nt, n_threads);
    nt = std::max(nt, 1);
#endif
    // A row the table cannot answer (off the sheet, on the wrong side, beyond
    // its cells) is flagged in the loop -- a throw cannot cross the omp
    // boundary -- and refused after.
    int bad = 0;
    {
        #pragma omp parallel for schedule(static) num_threads(nt) reduction(|:bad)
        for (std::ptrdiff_t r = 0; r < n; ++r) {
            double wr[32], ws[32];
            const std::ptrdiff_t row = ix[r];
            const double rho = sb[row * 3], z = sb[row * 3 + 1];
            const double zq = sb[row * 3 + 2];
            double *o = ob + row * nd;
            const bool on =
                height ? (z == fixed && zq <= 0.0) : (zq == fixed && z >= 0.0);
            const double s = z - zq;
            const std::ptrdiff_t a = on && rho >= 0.0 ? mw_sheet::cell_of(rho, re, nr) : -1;
            const std::ptrdiff_t b = a >= 0 ? mw_sheet::cell_of(s, se, ns) : -1;
            if (b < 0) {
                bad = 1;
                for (int k = 0; k < nd; ++k) o[k] = 0.0;
                continue;
            }
            mw_sheet::bary(rho, re[a], re[a + 1], xp, bwp, p, wr);
            mw_sheet::bary(s, se[b], se[b + 1], xp, bwp, p, ws);
            const double *blk = V + co[a * ns + b] * nd;
            double acc[2 * MW_SHEET_MAX_KEYS];
            // The two widths the families have today (`_SHEET_FAMILIES`:
            // six kernels, four point kernels) at a fixed width; any other
            // width, and the reference, through the generic loop.
#if MW_SHEET_FIXED
            if (fixed_ok && nk == 6)
                mw_sheet::cell_sum_fixed<12>(blk, wr, ws, p, acc);
            else if (fixed_ok && nk == 4)
                mw_sheet::cell_sum_fixed<8>(blk, wr, ws, p, acc);
            else
#endif
                mw_sheet::cell_sum(blk, wr, ws, p, nd, acc);
            // 1/R^pw per column, as a product of 1/R's: pw = 2 is i1 * i1,
            // the spelling the fixed six-column loop used.
            const double i1 = 1.0 / std::hypot(rho, s);
            for (int k = 0; k < nk; ++k) {
                double f = 1.0;
                if (rp[k] > 0) {
                    f = i1;
                    for (std::int64_t e = 1; e < rp[k]; ++e) f *= i1;
                }
                o[2 * k] = acc[2 * k] * f;
                o[2 * k + 1] = acc[2 * k + 1] * f;
            }
        }
    }
    if (bad)
        throw std::invalid_argument(
            "a row lies off the sheet (need z' == the plane's and z >= 0, or on "
            "a height sheet z == its h and z' <= 0; rho >= 0 and (rho, z - z') "
            "inside the table)");
}

}  // namespace ni_core

namespace ni_core {
bool grid_sheet_fixed() { return bool(MW_SHEET_FIXED); }
}  // namespace ni_core

namespace ni_core {

// The cone line sheets' interpolator (momwire#1595 option A): one surface
// z' = zf + beta * u tabulated along u at a fixed set of heights `zs`, each
// u cell `p` Chebyshev nodes, each node a column of every height. Row r is
// (u[r], height zs[zi[r]]); its value is the barycentric sum over its u
// cell's nodes, in node order, one fused multiply-add a step from +0.0
// (`cell_sum`'s inner loop), then 1/R^pw with R = hypot(u, z + d), the
// spelling the table was stored times. vals is [node][height][key].
void cone_line(const double *u, const std::int64_t *zi, std::ptrdiff_t n, const double *zs,
               std::ptrdiff_t n_z, double d, const double *u_edges, std::ptrdiff_t nu,
               const std::int64_t *cell_off, const double *x, const double *bw, int p,
               const std::complex<double> *vals, std::complex<double> *out, int n_threads,
               const std::int64_t *rpow, int nk) {
    if (nk < 1 || nk > MW_SHEET_MAX_KEYS) throw std::invalid_argument("bad key count");
    if (nu < 1) throw std::invalid_argument("edges must be 1-D with at least 2 entries");
    if (p < 1 || p > 32) throw std::invalid_argument("bad Chebyshev nodes (1..32)");
    if (!(d > 0.0)) throw std::invalid_argument("a cone sheet needs d > 0");
    for (std::ptrdiff_t j = 0; j < nu; ++j)
        if (!(u_edges[j + 1] > u_edges[j])) throw std::invalid_argument("u edges must rise");
    const int nd = 2 * nk;
    const double *V = reinterpret_cast<const double *>(vals);
    double *ob = reinterpret_cast<double *>(out);
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
    if (n_threads > 0) nt = std::min(nt, n_threads);
    nt = std::max(nt, 1);
#else
    (void)n_threads;
#endif
    int bad = 0;
    #pragma omp parallel for schedule(static) num_threads(nt) reduction(|:bad)
    for (std::ptrdiff_t r = 0; r < n; ++r) {
        double w[32];
        double *o = ob + r * nd;
        const std::int64_t j = zi[r];
        const std::ptrdiff_t c = (j >= 0 && j < n_z) ? mw_sheet::cell_of(u[r], u_edges, nu) : -1;
        if (c < 0) {
            bad = 1;
            for (int k = 0; k < nd; ++k) o[k] = 0.0;
            continue;
        }
        mw_sheet::bary(u[r], u_edges[c], u_edges[c + 1], x, bw, p, w);
        double acc[2 * MW_SHEET_MAX_KEYS];
        for (int k = 0; k < nd; ++k) acc[k] = 0.0;
        for (int a = 0; a < p; ++a) {
            const double *Va = V + ((cell_off[c] + a) * n_z + j) * nd;
            const double wa = w[a];
            for (int k = 0; k < nd; ++k) acc[k] = mw_fma::fma(wa, Va[k], acc[k]);
        }
        const double i1 = 1.0 / std::hypot(u[r], zs[j] + d);
        for (int k = 0; k < nk; ++k) {
            double f = 1.0;
            if (rpow[k] > 0) {
                f = i1;
                for (std::int64_t e = 1; e < rpow[k]; ++e) f *= i1;
            }
            o[2 * k] = acc[2 * k] * f;
            o[2 * k + 1] = acc[2 * k + 1] * f;
        }
    }
    if (bad)
        throw std::invalid_argument(
            "a row lies off the cone sheet (its height is not one of the table's, or u is "
            "outside its cells)");
}

}  // namespace ni_core

namespace ni_core {

int64_t hankel_conj_mismatches(const std::complex<double> *wp, std::ptrdiff_t n) {
    int64_t bad = 0;
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const std::complex<double> a = std::conj(xsf::cyl_hankel_1(0.0, wp[i]));
        const std::complex<double> b = xsf::cyl_hankel_2(0.0, std::conj(wp[i]));
        if (std::memcmp(&a, &b, sizeof a) != 0) ++bad;
    }
    return bad;
}

void six_batch(double k_p, std::complex<double> k_m, const double *rb, const double *zb,
               const double *pb, std::ptrdiff_t n, double rtol, double lam_mult, int depth,
               double detour, const double *gxp, const double *gwp, int ng,
               std::complex<double> *out) {
    // The Python walk's domain raises, verbatim and BEFORE the parallel
    // region (a throw cannot cross an omp boundary).
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        if (!(zb[i] >= 0.0 && pb[i] <= 0.0))
            throw std::invalid_argument("need z >= 0 >= zp");
        if (rb[i] < 0.0 || (zb[i] - pb[i]) + rb[i] <= 0.0)
            throw std::invalid_argument("need R > 0");
    }
    const mw_contour::cd km(k_m);
    bool ray_failed = false;
    // `dynamic`: a corner point costs many more adaptive levels than a
    // mid-range pair, and a fill's unique triples arrive sorted by
    // geometry -- static scheduling would hand one thread the hard band.
    #pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        mw_contour::cd o[6];
        if (!mw680::six_point_one(k_p, km, rb[i], zb[i], pb[i], rtol, lam_mult, depth, detour,
                                  gxp, gwp, ng, o)) {
            #pragma omp atomic write
            ray_failed = true;
            continue;
        }
        for (int c = 0; c < 6; ++c) out[6 * i + c] = o[c];
    }
    if (ray_failed)
        throw std::runtime_error("rotated tail did not go quiet inside the panel budget");
}

}  // namespace ni_core
