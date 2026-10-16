// The below/below grid's state, key and sources: see `_somm_below_grid_core.h`.
#include "_somm_below_grid_core.h"
#include "_somm_grid_core.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <stdexcept>

namespace somm_below {

namespace {

// numpy's complex SCALAR sqrt: glibc csqrt (S2b's measurement).
cd np_csqrt(cd z) {
#if defined(__GNUC__)
    __complex__ double c;
    __real__ c = z.real();
    __imag__ c = z.imag();
    c = __builtin_csqrt(c);
    return cd(__real__ c, __imag__ c);
#else
    return std::sqrt(z);
#endif
}

std::shared_ptr<Source> g_source;
std::mutex g_mutex;

}  // namespace

bool Key::operator==(const Key &o) const {
    return eps_t.real() == o.eps_t.real() && eps_t.imag() == o.eps_t.imag() && k2 == o.k2 &&
           r1b_wl == o.r1b_wl && omega == o.omega && mu == o.mu && spacing[0] == o.spacing[0] &&
           spacing[1] == o.spacing[1] && spacing[2] == o.spacing[2];
}

cd k_medium(cd eps_t, double k2) {
    const cd s = np_csqrt(eps_t);
    // numpy's scalar complex product (k2 + 0j) * s, never fused.
    cd km(k2 * s.real() - 0.0 * s.imag(), k2 * s.imag() + 0.0 * s.real());
    if (km.imag() > 0) km = cd(km.real(), -km.imag());
    return km;
}

double lambda_m(cd eps_t, double k2) {
    const cd km = k_medium(eps_t, k2);
    return 2.0 * M_PI / std::hypot(km.real(), km.imag());
}

double r1_bucket_below_wl(double r1_max, double lam_m) {
    return somm_core::r1_bucket_wl(std::min(r1_max / lam_m, R1_CAP_LAMBDA_M));
}

Key key_for(cd eps_t, double k2, double r1_max, double omega, double mu,
            const double spacing[3]) {
    Key k;
    k.eps_t = eps_t;
    k.k2 = k2;
    k.r1b_wl = r1_bucket_below_wl(r1_max, lambda_m(eps_t, k2));
    k.omega = omega;
    k.mu = mu;
    for (int i = 0; i < 3; i++) k.spacing[i] = spacing[i];
    return k;
}

std::vector<int> regions_for(const Grid &g, double mx_r1, double mn_th, double mx_th) {
    std::vector<int> zones{0};
    if (mx_r1 > g.r_break) zones.push_back(1);
    if (mx_r1 > g.r_near) zones.push_back(2);
    std::vector<int> bands;
    if (mn_th < g.th_band_floor_hi) bands.push_back(0);
    if (mx_th >= g.th_band_floor_hi && mn_th < g.th_band_lo_hi) bands.push_back(1);
    if (mx_th >= g.th_band_lo_hi && mn_th < g.th_band_hi) bands.push_back(2);
    if (mx_th >= g.th_band_hi && mn_th <= g.th_split) bands.push_back(3);
    if (mx_th > g.th_split) bands.push_back(4);
    std::vector<int> out;
    for (int z : zones)
        for (int b : bands) out.push_back(z * N_BANDS + b);
    return out;
}

Source *source() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_source.get();
}

void set_source(std::shared_ptr<Source> s) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_source = std::move(s);
}

namespace {

// The bridge's file: little-endian, written by buried_z_compare.py:
//   "MWBELOW1", key (eps re, eps im, k2, r1b_wl, omega, mu, spacing x3: 9 f64),
//   blended (i64), lam_m r1_max r1_cap r_break r_near th_min th_band_floor_hi
//   th_band_lo_hi th_band_hi th_split k_m.re k_m.im (12 f64), then per region
//   r0 dr th0 dth (4 f64), n_r n_th filled (3 i64), vals (4 n_r n_th c128).
class Bridge : public Source {
  public:
    explicit Bridge(const std::string &dir) {
        std::error_code ec;
        std::filesystem::directory_iterator it(dir, ec), end;
        if (ec) throw std::runtime_error("below-table bridge: no directory " + dir);
        for (; it != end; ++it)
            if (it->path().extension() == ".bin") load(it->path().string());
    }
    Grid &get(cd eps_t, double k2, double r1_max, double omega, double mu,
              const double spacing[3]) override {
        const Key k = key_for(eps_t, k2, r1_max, omega, mu, spacing);
        for (auto &g : grids_)
            if (g->key == k && !g->blended) return *g;
        for (auto &g : grids_)
            if (g->key == k) return *g;
        char buf[256];
        std::snprintf(buf, sizeof buf, "below-table bridge: no grid for eps %.17g%+.17gj r1b %.17g",
                      eps_t.real(), eps_t.imag(), k.r1b_wl);
        throw std::runtime_error(buf);
    }
    void ensure_for(Grid &g, double mx_r1, double mn_th, double mx_th) override {
        for (int r : regions_for(g, mx_r1, mn_th, mx_th))
            if (!g.regions[r].filled)
                throw std::runtime_error("below-table bridge: region " + std::to_string(r) +
                                         " was not in the dumped tables");
    }

  private:
    std::vector<std::unique_ptr<Grid>> grids_;
    void load(const std::string &path) {
        std::FILE *f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) throw std::runtime_error("below-table bridge: cannot read " + path);
        auto rd = [&](void *p, size_t n) {
            if (std::fread(p, 1, n, f) != n) {
                std::fclose(f);
                throw std::runtime_error("below-table bridge: short file " + path);
            }
        };
        char magic[8];
        rd(magic, 8);
        if (std::memcmp(magic, "MWBELOW1", 8) != 0) {
            std::fclose(f);
            throw std::runtime_error("below-table bridge: bad magic in " + path);
        }
        std::unique_ptr<Grid> g(new Grid());
        double kd[9];
        rd(kd, sizeof kd);
        g->key.eps_t = cd(kd[0], kd[1]);
        g->key.k2 = kd[2];
        g->key.r1b_wl = kd[3];
        g->key.omega = kd[4];
        g->key.mu = kd[5];
        for (int i = 0; i < 3; i++) g->key.spacing[i] = kd[6 + i];
        int64_t blended;
        rd(&blended, 8);
        g->blended = blended != 0;
        double s[12];
        rd(s, sizeof s);
        g->lam_m = s[0];
        g->r1_max = s[1];
        g->r1_cap = s[2];
        g->r_break = s[3];
        g->r_near = s[4];
        g->th_min = s[5];
        g->th_band_floor_hi = s[6];
        g->th_band_lo_hi = s[7];
        g->th_band_hi = s[8];
        g->th_split = s[9];
        g->k_m = cd(s[10], s[11]);
        for (int r = 0; r < N_REGIONS; r++) {
            Region &reg = g->regions[r];
            double a[4];
            rd(a, sizeof a);
            reg.r0 = a[0];
            reg.dr = a[1];
            reg.th0 = a[2];
            reg.dth = a[3];
            int64_t b[3];
            rd(b, sizeof b);
            reg.n_r = b[0];
            reg.n_th = b[1];
            reg.filled = b[2] != 0;
            reg.vals.resize(static_cast<size_t>(4 * reg.n_r * reg.n_th));
            rd(reg.vals.data(), reg.vals.size() * sizeof(cd));
        }
        std::fclose(f);
        grids_.push_back(std::move(g));
    }
};

}  // namespace

std::shared_ptr<Source> bridge_source(const std::string &dir) {
    return std::make_shared<Bridge>(dir);
}

}  // namespace somm_below
