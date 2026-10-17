// The designed near-interface tables, Python-free: see `_near_interface_eval.h`.
// Every rule below names the `_near_interface.py` function it replays.
#include "_study_timers.h"
#include "_near_interface_eval.h"

#include "_razor_front_core.h"  // leggauss (the frozen Gauss table)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace ni_eval {

namespace {

constexpr int SHEET_P = 12;
constexpr double SHEET_WAVES = 0.7;
constexpr double SHEET_DEAD = 18.0;
constexpr double SHEET_ROW_WORTH = 1.0 / 40.0;
constexpr double SHEET_BUILD_WEIGHT = 0.15;
constexpr double SHEET_HEIGHT_WEIGHT = 0.15;
constexpr double SHEET_MIN_ROWS = 512;
constexpr double SHEET_MIN_DEPTH = 1e-3;
constexpr int64_t CENSUS_CHUNK = 1 << 18;
constexpr int COLUMN_P = 0;
constexpr double DETOUR = 2.0;
constexpr double LAM_MULT = 8.0;
constexpr int64_t KEY_RPOW[N_KEYS] = {1, 1, 1, 2, 2, 2};

uint64_t bits(double x) {
    uint64_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}

const std::vector<double> &gx24() {
    static const std::vector<double> x = [] {
        std::vector<double> a, b;
        mw::razor::leggauss(24, a, b);
        return a;
    }();
    return x;
}
const std::vector<double> &gw24() {
    static const std::vector<double> w = [] {
        std::vector<double> a, b;
        mw::razor::leggauss(24, a, b);
        return b;
    }();
    return w;
}

// CPython's abs(complex): hypot.
double cabs(cd z) { return std::hypot(z.real(), z.imag()); }

// `_sheet_wavelength(k_p, k_m, r_lo)`.
double sheet_wavelength(double k_p, cd k_m, double r_lo) {
    double k = k_p;
    if (std::fabs(k_m.imag()) * r_lo < SHEET_DEAD) k = std::max(k, cabs(k_m));
    return 2.0 * M_PI / k;
}

struct Strip {
    double lo, hi;
    int64_t n;
};

// `_sheet_strip(k_p, k_m, d, i, axis, depth)`: false past a height sheet's
// reach.
bool sheet_strip(double k_p, cd k_m, double d, int i, bool rho_axis, double depth, Strip &st) {
    double lo, hi, wl;
    if (rho_axis) {
        if (i == 0) {
            lo = 0.0;
            hi = 0.5 * d;
        } else if (i == 1) {
            lo = 0.5 * d;
            hi = d;
        } else {
            lo = d * std::pow(2.0, static_cast<double>(i - 2));
            hi = d * std::pow(2.0, static_cast<double>(i - 1));
        }
        wl = sheet_wavelength(k_p, k_m, std::max(lo, d));
    } else {
        lo = d * std::pow(2.0, static_cast<double>(i));
        hi = d * std::pow(2.0, static_cast<double>(i + 1));
        if (depth < 0) {
            wl = sheet_wavelength(k_p, k_m, lo);
        } else {
            if (lo >= d + depth) return false;
            hi = std::min(hi, d + depth);
            wl = 2.0 * M_PI / cabs(k_m);
        }
    }
    st.lo = lo;
    st.hi = hi;
    st.n = std::max<int64_t>(1, static_cast<int64_t>(std::ceil((hi - lo) / (SHEET_WAVES * wl))));
    return true;
}

// `_sheet_axis`: the strips held once the axis reaches `reach`.
std::vector<Strip> sheet_axis(double k_p, cd k_m, double d, bool rho_axis, double reach, double depth) {
    std::vector<Strip> out;
    for (int i = 0;; i++) {
        Strip st;
        if (!sheet_strip(k_p, k_m, d, i, rho_axis, depth, st)) return out;
        out.push_back(st);
        if (st.hi >= reach) return out;
    }
}

// `_sheet_nodes_to`.
double sheet_nodes_to(double k_p, cd k_m, double d, double rho_max, double s_max, double depth) {
    int64_t n_r = 0, n_s = 0;
    for (const Strip &s : sheet_axis(k_p, k_m, d, true, rho_max, -1)) n_r += s.n;
    for (const Strip &s : sheet_axis(k_p, k_m, d, false, s_max, depth)) n_s += s.n;
    return static_cast<double>(n_r * n_s * SHEET_P * SHEET_P);
}

// --------------------------------------------------------------------------
// PlaneSheet

struct Cheb {
    double x[SHEET_P], bw[SHEET_P];
    Cheb() {
        for (int k = 0; k < SHEET_P; k++) {
            const double t = (static_cast<double>(2 * k + 1) * M_PI) / static_cast<double>(2 * SHEET_P);
            x[k] = -std::cos(t);
            bw[k] = ((k % 2) ? -1.0 : 1.0) * std::sin(t);
        }
    }
};
const Cheb &cheb() {
    static const Cheb c;
    return c;
}

class PlaneSheet {
  public:
    PlaneSheet(double k_p, cd k_m, double zp, bool height, double depth)
        : k_p_(k_p), k_m_(k_m), zp_(zp), d_(std::fabs(zp)), height_(height), depth_(height ? depth : -1) {}

    // `cover(rho_max, s_max)`: grow by whole strips, evaluate the new cells.
    void cover(double rho_max, double s_max, int n_threads) {
        const std::vector<int> new_r = grow(true, rho_max);
        const std::vector<int> new_s = grow(false, s_max);
        const int n_r = static_cast<int>(strips_[0].size()), n_s = static_cast<int>(strips_[1].size());
        std::vector<std::pair<int, int>> pairs;
        for (int i = 0; i < n_r; i++)
            for (int j = 0; j < n_s; j++)
                if (std::count(new_r.begin(), new_r.end(), i) || std::count(new_s.begin(), new_s.end(), j))
                    pairs.emplace_back(i, j);
        if (pairs.empty()) return;
        const int p = SHEET_P;
        std::vector<double> rho_c, s_all;
        std::vector<std::ptrdiff_t> sizes;
        struct Place {
            int a, k;
            std::vector<int> s_ids;
        };
        std::vector<Place> place;
        for (const auto &ij : pairs) {
            std::vector<int> r_ids, s_ids;
            std::vector<double> rn, sn;
            nodes(true, ij.first, r_ids, rn);
            nodes(false, ij.second, s_ids, sn);
            for (size_t k = 0; k < rn.size(); k++) {
                rho_c.push_back(rn[k]);
                sizes.push_back(static_cast<std::ptrdiff_t>(sn.size()));
                s_all.insert(s_all.end(), sn.begin(), sn.end());
                place.push_back(Place{r_ids[k / p], static_cast<int>(k % p), s_ids});
            }
        }
        const std::ptrdiff_t nc = static_cast<std::ptrdiff_t>(rho_c.size());
        std::vector<std::ptrdiff_t> offsets(static_cast<size_t>(nc + 1), 0);
        for (std::ptrdiff_t c = 0; c < nc; c++) offsets[c + 1] = offsets[c] + sizes[c];
        const size_t n = s_all.size();
        std::vector<double> z(n), zq(n);
        for (size_t i = 0; i < n; i++) {
            if (height_) {
                z[i] = zp_;
                zq[i] = std::min(zp_ - s_all[i], 0.0);
            } else {
                z[i] = std::max(zp_ + s_all[i], 0.0);
                zq[i] = zp_;
            }
        }
        std::vector<cd> vals(n * N_KEYS);
        ni_core::six_columns(k_p_, k_m_, rho_c.data(), offsets.data(), nc, z.data(), zq.data(),
                             static_cast<std::ptrdiff_t>(n), LAM_MULT, COLUMN_P, DETOUR, n_threads,
                             gx24().data(), gw24().data(), 24, 0, nullptr, vals.data());
        // `vals[:, cols] *= f`, f = R ** pw (numpy's complex by real-as-complex
        // product: each part one rounding).
        size_t i = 0;
        for (std::ptrdiff_t c = 0; c < nc; c++)
            for (std::ptrdiff_t q = 0; q < sizes[c]; q++, i++) {
                const double Rn = std::hypot(rho_c[c], z[i] - zq[i]);
                const double f2 = Rn * Rn;
                for (int k = 0; k < N_KEYS; k++) {
                    const double f = KEY_RPOW[k] == 1 ? Rn : f2;
                    cd &v = vals[i * N_KEYS + k];
                    v = cd(std::fma(v.real(), f, -(v.imag() * 0.0)), std::fma(v.real(), 0.0, v.imag() * f));
                }
            }
        std::map<std::pair<int, int>, std::vector<cd>> blocks;
        for (std::ptrdiff_t c = 0; c < nc; c++) {
            const Place &pl = place[static_cast<size_t>(c)];
            for (size_t q = 0; q < pl.s_ids.size(); q++) {
                std::vector<cd> &blk = blocks[{pl.a, pl.s_ids[q]}];
                if (blk.empty()) blk.assign(static_cast<size_t>(p * p * N_KEYS), cd(0, 0));
                for (int ks = 0; ks < p; ks++)
                    for (int k = 0; k < N_KEYS; k++)
                        blk[(static_cast<size_t>(pl.k) * p + ks) * N_KEYS + k] =
                            vals[(static_cast<size_t>(offsets[c]) + q * p + ks) * N_KEYS + k];
            }
        }
        for (auto &kv : blocks) {
            off_[kv.first] = static_cast<int64_t>(vals_.size() / N_KEYS);
            vals_.insert(vals_.end(), kv.second.begin(), kv.second.end());
        }
        flat_ok_ = false;
    }

    void interpolate(const double *sub, std::ptrdiff_t m, const std::vector<std::ptrdiff_t> &idx,
                     Six *out, int n_threads) {
        if (!flat_ok_) flatten();
        // `out` is (m, N_KEYS) row-major already (Six is N_KEYS contiguous
        // complexes), and grid_sheet writes only rows idx: in place.
        static_assert(sizeof(Six) == N_KEYS * sizeof(cd), "Six is N_KEYS contiguous complexes");
        ni_core::grid_sheet(sub, m, idx.data(), static_cast<std::ptrdiff_t>(idx.size()), zp_, height_,
                            rho_edges_.data(), static_cast<std::ptrdiff_t>(rho_edges_.size() - 1),
                            s_edges_.data(), static_cast<std::ptrdiff_t>(s_edges_.size() - 1),
                            cell_off_.data(), cheb().x, cheb().bw, SHEET_P, vals_.data(),
                            reinterpret_cast<cd *>(out), n_threads, KEY_RPOW, N_KEYS, false);
    }

  private:
    struct Cell {
        double a, b;
        int strip;
    };
    double k_p_;
    cd k_m_;
    double zp_, d_;
    bool height_;
    double depth_;
    std::vector<Strip> strips_[2];  // rho, s
    std::vector<Cell> cells_[2];
    std::map<std::pair<int, int>, int64_t> off_;
    std::vector<cd> vals_;
    bool flat_ok_ = false;
    std::vector<double> rho_edges_, s_edges_;
    std::vector<int64_t> cell_off_;

    std::vector<int> grow(bool rho_axis, double reach) {
        std::vector<Strip> &have = strips_[rho_axis ? 0 : 1];
        const std::vector<Strip> want = sheet_axis(k_p_, k_m_, d_, rho_axis, reach, rho_axis ? -1 : depth_);
        std::vector<int> fresh;
        for (size_t i = have.size(); i < want.size(); i++) {
            fresh.push_back(static_cast<int>(i));
            const Strip &st = want[i];
            have.push_back(st);
            for (int64_t q = 0; q < st.n; q++) {
                const double a = st.lo + (st.hi - st.lo) * static_cast<double>(q) / static_cast<double>(st.n);
                const double b = q == st.n - 1 ? st.hi
                                               : st.lo + (st.hi - st.lo) * static_cast<double>(q + 1) /
                                                             static_cast<double>(st.n);
                cells_[rho_axis ? 0 : 1].push_back(Cell{a, b, static_cast<int>(i)});
            }
        }
        return fresh;
    }

    void nodes(bool rho_axis, int strip, std::vector<int> &ids, std::vector<double> &pts) const {
        const std::vector<Cell> &cells = cells_[rho_axis ? 0 : 1];
        for (size_t c = 0; c < cells.size(); c++) {
            if (cells[c].strip != strip) continue;
            ids.push_back(static_cast<int>(c));
            const double a = cells[c].a, b = cells[c].b;
            const double mid = 0.5 * (a + b), half = 0.5 * (b - a);
            for (int k = 0; k < SHEET_P; k++) pts.push_back(mid + half * cheb().x[k]);
        }
    }

    void flatten() {
        rho_edges_.clear();
        s_edges_.clear();
        for (const Cell &c : cells_[0]) rho_edges_.push_back(c.a);
        rho_edges_.push_back(cells_[0].back().b);
        for (const Cell &c : cells_[1]) s_edges_.push_back(c.a);
        s_edges_.push_back(cells_[1].back().b);
        const size_t n_r = cells_[0].size(), n_s = cells_[1].size();
        cell_off_.assign(n_r * n_s, 0);
        for (const auto &kv : off_)
            cell_off_[static_cast<size_t>(kv.first.first) * n_s + static_cast<size_t>(kv.first.second)] = kv.second;
        flat_ok_ = true;
    }
};

// `_plane_sheet`: the process's sheets, keyed by what a node depends on (a
// node's value never depends on how far its sheet was grown).
std::mutex g_sheet_mutex;
std::map<std::tuple<bool, double, double, double, double, double>, std::unique_ptr<PlaneSheet>> g_sheets;

PlaneSheet &plane_sheet(double k_p, cd k_m, double zp, bool height, double depth) {
    STUDY_T("ni.plane_sheet");
    const auto key = std::make_tuple(height, height ? depth : -1.0, k_p, k_m.real(), k_m.imag(), zp);
    auto it = g_sheets.find(key);
    if (it == g_sheets.end())
        it = g_sheets.emplace(key, std::unique_ptr<PlaneSheet>(new PlaneSheet(k_p, k_m, zp, height, depth))).first;
    return *it->second;
}

// --------------------------------------------------------------------------
// The column twin over a row set (`_column_twin`, `_column_blocks`,
// `_refuse_bad_members`).

void column_twin(const Evaluator &ev, const double *sub, size_t m, const int64_t *labels,
                 ni_core::ColumnCache *cache, Six *out) {
    STUDY_T("ni.column_twin");
    if (m == 0) return;
    // Groups: (label, rho) with -0.0 == 0.0, in first-seen order.
    struct GKey {
        int64_t lab;
        uint64_t r;
        bool operator<(const GKey &o) const { return lab != o.lab ? lab < o.lab : r < o.r; }
    };
    std::map<GKey, size_t> gid_of;
    std::vector<size_t> gid(m);
    std::vector<double> rho_c;
    std::vector<std::ptrdiff_t> sizes;
    for (size_t i = 0; i < m; i++) {
        const GKey k{labels ? labels[i] : 0, bits(sub[3 * i] + 0.0)};
        auto it = gid_of.find(k);
        if (it == gid_of.end()) {
            it = gid_of.emplace(k, rho_c.size()).first;
            rho_c.push_back(sub[3 * i]);
            sizes.push_back(0);
        }
        gid[i] = it->second;
        sizes[it->second]++;
    }
    const size_t nc = rho_c.size();
    std::vector<std::ptrdiff_t> offsets(nc + 1, 0);
    for (size_t c = 0; c < nc; c++) offsets[c + 1] = offsets[c] + sizes[c];
    std::vector<std::ptrdiff_t> fill(offsets.begin(), offsets.end() - 1);
    std::vector<size_t> member_order(m);
    for (size_t i = 0; i < m; i++) member_order[static_cast<size_t>(fill[gid[i]]++)] = i;
    std::vector<double> zs(m), zps(m);
    for (size_t j = 0; j < m; j++) {
        zs[j] = sub[3 * member_order[j] + 1];
        zps[j] = sub[3 * member_order[j] + 2];
    }
    // `_refuse_bad_members`, in the walk's words.
    for (size_t j = 0; j < m; j++)
        if (zs[j] < 0.0 || zps[j] > 0.0)
            throw std::invalid_argument("need z >= 0 >= zp, got (" + py_repr(zs[j]) + ", " +
                                        py_repr(zps[j]) + ")");
    {
        size_t j = 0;
        for (size_t c = 0; c < nc; c++)
            for (std::ptrdiff_t q = 0; q < sizes[c]; q++, j++) {
                const double s = zs[j] - zps[j];
                if (rho_c[c] < 0.0 || s + rho_c[c] <= 0.0)
                    throw std::invalid_argument("need R > 0, got rho=" + py_repr(rho_c[c]) +
                                                ", s=" + py_repr(s));
            }
    }
    std::vector<cd> vals(m * N_KEYS);
    ni_core::six_columns(ev.k_p, ev.k_m, rho_c.data(), offsets.data(), static_cast<std::ptrdiff_t>(nc),
                         zs.data(), zps.data(), static_cast<std::ptrdiff_t>(m), LAM_MULT, COLUMN_P,
                         DETOUR, ev.n_threads, gx24().data(), gw24().data(), 24, ev.cancel_flag, cache,
                         vals.data());
    for (size_t j = 0; j < m; j++)
        for (int k = 0; k < N_KEYS; k++) out[member_order[j]][k] = vals[j * N_KEYS + k];
}

}  // namespace

// --------------------------------------------------------------------------

std::string py_repr(double x) {
    if (std::isnan(x)) return "nan";
    if (std::isinf(x)) return x > 0 ? "inf" : "-inf";
    char buf[64];
    int prec = 1;
    for (; prec <= 17; prec++) {
        std::snprintf(buf, sizeof buf, "%.*e", prec - 1, x);
        if (std::strtod(buf, nullptr) == x) break;
    }
    // buf: d.ddde[+-]XX ; Python's repr: fixed for -4 <= exp < 16
    std::string s = buf;
    const size_t e = s.find('e');
    std::string mant = s.substr(0, e);
    const int exp = std::atoi(s.c_str() + e + 1);
    bool neg = false;
    if (!mant.empty() && mant[0] == '-') {
        neg = true;
        mant = mant.substr(1);
    }
    std::string digits;
    for (char c : mant)
        if (c != '.') digits += c;
    std::string out;
    if (exp >= -4 && exp < 16) {
        if (exp >= 0) {
            if (static_cast<int>(digits.size()) <= exp + 1) {
                out = digits + std::string(static_cast<size_t>(exp + 1 - static_cast<int>(digits.size())), '0') + ".0";
            } else {
                out = digits.substr(0, static_cast<size_t>(exp + 1)) + "." + digits.substr(static_cast<size_t>(exp + 1));
            }
        } else {
            out = "0." + std::string(static_cast<size_t>(-exp - 1), '0') + digits;
        }
    } else {
        out = digits.substr(0, 1);
        if (digits.size() > 1) out += "." + digits.substr(1);
        char eb[16];
        std::snprintf(eb, sizeof eb, "e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
        out += eb;
    }
    return neg ? "-" + out : out;
}

Memo::Memo() : cache(new ni_core::ColumnCache(16u << 20)) {}

uint64_t TripleIndex::mix(uint64_t a, uint64_t b, uint64_t c) {
    uint64_t h = a * 0x9E3779B97F4A7C15ULL;
    h ^= b + 0x7F4A7C159E3779B9ULL + (h << 6) + (h >> 2);
    h ^= c + 0x94D049BB133111EBULL + (h << 6) + (h >> 2);
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33;
    return h;
}

void TripleIndex::rehash(size_t want) {
    size_t cap = 1024;
    while (cap < want) cap <<= 1;
    std::vector<Slot> old;
    old.swap(slots_);
    slots_.assign(cap, Slot{0, NONE});
    mask_ = cap - 1;
    for (const Slot &s : old) {
        if (s.v == NONE) continue;
        const Key &k = keys_[s.v];
        size_t p = static_cast<size_t>(mix(k.a, k.b, k.c)) & mask_;
        while (slots_[p].v != NONE) p = (p + 1) & mask_;
        slots_[p] = s;
    }
}

uint32_t TripleIndex::find(const double *row) const {
    const uint64_t a = bits(row[0] + 0.0), b = bits(row[1] + 0.0), c = bits(row[2] + 0.0);
    const uint64_t h = mix(a, b, c);
    const uint32_t tag = static_cast<uint32_t>(h >> 32);
    for (size_t p = static_cast<size_t>(h) & mask_;; p = (p + 1) & mask_) {
        const Slot &s = slots_[p];
        if (s.v == NONE) return NONE;
        if (s.tag == tag) {
            const Key &k = keys_[s.v];
            if (k.a == a && k.b == b && k.c == c) return s.v;
        }
    }
}

uint32_t TripleIndex::find_or_insert(const double *row, uint32_t v, bool &inserted) {
    if (2 * (n_ + 1) > slots_.size()) rehash(2 * slots_.size());
    const uint64_t a = bits(row[0] + 0.0), b = bits(row[1] + 0.0), c = bits(row[2] + 0.0);
    const uint64_t h = mix(a, b, c);
    const uint32_t tag = static_cast<uint32_t>(h >> 32);
    for (size_t p = static_cast<size_t>(h) & mask_;; p = (p + 1) & mask_) {
        Slot &s = slots_[p];
        if (s.v == NONE) {
            s = Slot{tag, v};
            if (keys_.size() <= v) keys_.resize(static_cast<size_t>(v) + 1, Key{0, 0, 0});
            keys_[v] = Key{a, b, c};
            n_++;
            inserted = true;
            return v;
        }
        if (s.tag == tag) {
            const Key &k = keys_[s.v];
            if (k.a == a && k.b == b && k.c == c) {
                inserted = false;
                return s.v;
            }
        }
    }
}

uint32_t Memo::find(const double *row) const {
    if (std::isnan(row[0]) || std::isnan(row[1]) || std::isnan(row[2])) return TripleIndex::NONE;
    return idx_.find(row);
}

bool Memo::lookup(const double *row, Six &out) const {
    const uint32_t i = find(row);
    if (i == TripleIndex::NONE) return false;
    out = vals_[i];
    return true;
}

void Memo::insert(const double *row, const Six &v) {
    bool ins = false;
    idx_.find_or_insert(row, static_cast<uint32_t>(vals_.size()), ins);
    if (ins) vals_.push_back(v);
}

void evaluate_fresh(const Evaluator &ev, const double *rows, size_t m, const int64_t *labels,
                    const SheetPlan *plan, ni_core::ColumnCache *cache, Six *out) {
    STUDY_T("ni.evaluate_fresh");
    if (m == 0) return;
    // `_sheet_take`
    std::vector<char> take(m, 0), on_plane(m, 0);
    bool any = false;
    if (plan != nullptr && plan->any()) {
        for (size_t i = 0; i < m; i++) {
            const double rho = rows[3 * i], z = rows[3 * i + 1], zp = rows[3 * i + 2];
            const bool ok = rho >= 0.0 && std::isfinite(rho) && std::isfinite(z) && std::isfinite(zp);
            const bool onp = ok && z >= 0.0 &&
                             std::binary_search(plan->planes.begin(), plan->planes.end(), zp);
            on_plane[i] = onp;
            take[i] = onp;
        }
        for (size_t h = 0; h < plan->heights.size(); h++)
            for (size_t i = 0; i < m; i++) {
                const double rho = rows[3 * i], z = rows[3 * i + 1], zp = rows[3 * i + 2];
                const bool ok = rho >= 0.0 && std::isfinite(rho) && std::isfinite(z) && std::isfinite(zp);
                if (z == plan->heights[h] && ok && zp <= 0.0 && zp >= -plan->depths[h] && !on_plane[i])
                    take[i] = 1;
            }
        for (size_t i = 0; i < m; i++) any |= take[i] != 0;
    }
    if (!any) {
        column_twin(ev, rows, m, labels, cache, out);
        return;
    }
    // `_evaluate_with_sheets`: the rest through the twin, alone.
    std::vector<double> rest;
    std::vector<int64_t> rest_lab;
    std::vector<size_t> rest_i;
    for (size_t i = 0; i < m; i++)
        if (!take[i]) {
            rest_i.push_back(i);
            rest.insert(rest.end(), rows + 3 * i, rows + 3 * i + 3);
            if (labels) rest_lab.push_back(labels[i]);
        }
    if (!rest_i.empty()) {
        std::vector<Six> got(rest_i.size());
        column_twin(ev, rest.data(), rest_i.size(), labels ? rest_lab.data() : nullptr, cache, got.data());
        for (size_t j = 0; j < rest_i.size(); j++) out[rest_i[j]] = got[j];
    }
    // `_serve_from_sheets`: planes, then heights, each grown to this call's
    // farthest row on it.
    std::lock_guard<std::mutex> lock(g_sheet_mutex);
    for (int pass = 0; pass < 2; pass++) {
        const bool height = pass == 1;
        const std::vector<double> &values = height ? plan->heights : plan->planes;
        for (size_t k = 0; k < values.size(); k++) {
            const double v = values[k];
            std::vector<std::ptrdiff_t> idx;
            double rho_max = -std::numeric_limits<double>::infinity();
            double ext = height ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < m; i++) {
                const double col = rows[3 * i + (height ? 1 : 2)];
                bool sel = col == v;
                if (height)
                    sel = sel && take[i] && !on_plane[i];
                else
                    sel = sel && on_plane[i];
                if (!sel) continue;
                idx.push_back(static_cast<std::ptrdiff_t>(i));
                rho_max = std::max(rho_max, rows[3 * i]);
                if (height)
                    ext = std::min(ext, rows[3 * i + 2]);
                else
                    ext = std::max(ext, rows[3 * i + 1]);
            }
            if (idx.empty()) continue;
            const double depth = height ? plan->depths[k] : -1.0;
            const double s_max = height ? v - ext : ext - v;
            PlaneSheet &sh = plane_sheet(ev.k_p, ev.k_m, v, height, depth);
            sh.cover(rho_max, s_max, ev.n_threads);
            sh.interpolate(rows, static_cast<std::ptrdiff_t>(m), idx, out, ev.n_threads);
        }
    }
}

void designed_index(const Evaluator &ev, Memo &memo, const double *rho, const double *z,
                    const double *zp, size_t n, const int64_t *labels, uint32_t *idx,
                    int n_threads) {
    STUDY_T("ni.designed_index");
    // The memo's hits first (read-only, so any split). A row's hit or miss
    // depends on its triple alone, so the misses' first appearances are the
    // order `_unique_rows` then the memo filter would give the fresh rows.
    const std::ptrdiff_t nn = static_cast<std::ptrdiff_t>(n);
    const bool memo_empty = memo.size() == 0;
#if defined(_OPENMP)
    int nt = omp_get_max_threads();
    if (n_threads > 0) nt = std::min(nt, n_threads);
    if (n < 16384) nt = 1;
    #pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
#else
    (void)n_threads;
#endif
    for (std::ptrdiff_t i = 0; i < nn; i++) {
        const double row[3] = {rho[i], z[i], zp[i]};
        idx[i] = memo_empty ? TripleIndex::NONE : memo.find(row);
    }
    size_t n_miss = 0;
    for (size_t i = 0; i < n; i++) n_miss += idx[i] == TripleIndex::NONE;
    if (n_miss == 0) return;
    // The misses, each distinct triple claimed in the memo at its value's
    // future index (base + its place among the fresh rows); nothing reads
    // the memo before the values land. A NaN row never hits the memo, so
    // NaN triples are deduplicated apart (their key bits alike).
    const uint32_t base = static_cast<uint32_t>(memo.size());
    TripleIndex &keys = memo.keys();
    TripleIndex nan_rows;
    std::vector<double> rows;
    std::vector<int64_t> lab;
    auto claim = [&](size_t i) {
        const double row[3] = {rho[i], z[i], zp[i]};
        const uint32_t next = base + static_cast<uint32_t>(rows.size() / 3);
        bool ins = false;
        const bool nan = std::isnan(row[0]) || std::isnan(row[1]) || std::isnan(row[2]);
        const uint32_t v = nan ? nan_rows.find_or_insert(row, next, ins) : keys.find_or_insert(row, next, ins);
        if (ins) {
            rows.insert(rows.end(), row, row + 3);
            if (labels) lab.push_back(labels[i]);
        }
        return v;
    };
#if defined(_OPENMP)
    if (nt > 1) {
        // Each chunk's misses deduplicated on their own (any thread), then
        // the chunks' distinct triples claimed in chunk order, each at its
        // first appearance within the chunk: the claim order is the serial
        // walk's (first appearance over the whole call), with one claim per
        // chunk-distinct triple instead of one per point.
        std::vector<uint32_t> loc(n, TripleIndex::NONE);
        std::vector<std::vector<size_t>> first(static_cast<size_t>(nt));
        std::vector<std::vector<uint32_t>> to(static_cast<size_t>(nt));
        #pragma omp parallel for schedule(static, 1) num_threads(nt)
        for (int c = 0; c < nt; c++) {
            const size_t i0 = n * static_cast<size_t>(c) / static_cast<size_t>(nt);
            const size_t i1 = n * static_cast<size_t>(c + 1) / static_cast<size_t>(nt);
            TripleIndex li;
            li.reserve((i1 - i0) / 8);
            std::vector<size_t> &fc = first[static_cast<size_t>(c)];
            for (size_t i = i0; i < i1; i++) {
                if (idx[i] != TripleIndex::NONE) continue;
                const double row[3] = {rho[i], z[i], zp[i]};
                bool ins = false;
                loc[i] = li.find_or_insert(row, static_cast<uint32_t>(fc.size()), ins);
                if (ins) fc.push_back(i);
            }
        }
        size_t n_first = 0;
        for (const auto &fc : first) n_first += fc.size();
        keys.reserve(keys.size() + n_first);
        rows.reserve(3 * n_first);
        if (labels) lab.reserve(n_first);
        for (int c = 0; c < nt; c++) {
            const std::vector<size_t> &fc = first[static_cast<size_t>(c)];
            std::vector<uint32_t> &tc = to[static_cast<size_t>(c)];
            tc.resize(fc.size());
            for (size_t u = 0; u < fc.size(); u++) tc[u] = claim(fc[u]);
        }
        #pragma omp parallel for schedule(static, 1) num_threads(nt)
        for (int c = 0; c < nt; c++) {
            const size_t i0 = n * static_cast<size_t>(c) / static_cast<size_t>(nt);
            const size_t i1 = n * static_cast<size_t>(c + 1) / static_cast<size_t>(nt);
            const std::vector<uint32_t> &tc = to[static_cast<size_t>(c)];
            for (size_t i = i0; i < i1; i++)
                if (loc[i] != TripleIndex::NONE) idx[i] = tc[loc[i]];
        }
    } else
#endif
    {
        for (size_t i = 0; i < n; i++)
            if (idx[i] == TripleIndex::NONE) idx[i] = claim(i);
    }
    const size_t m = rows.size() / 3;
    if (m == 0) return;
    std::vector<Six> vals(m);
    evaluate_fresh(ev, rows.data(), m, labels ? lab.data() : nullptr,
                   memo.have_plan ? &memo.plan : nullptr, memo.cache.get(), vals.data());
    memo.append_values(vals.data(), m);
}

void designed_tables(const Evaluator &ev, Memo *memo, const double *rho, const double *z,
                     const double *zp, size_t n, const int64_t *labels, Six *out) {
    if (memo != nullptr) {
        std::vector<uint32_t> idx(n);
        designed_index(ev, *memo, rho, z, zp, n, labels, idx.data(), 1);
        for (size_t i = 0; i < n; i++) out[i] = memo->at(idx[i]);
        return;
    }
    // `_unique_rows`: distinct rows (-0.0 == 0.0) in first-appearance order.
    TripleIndex seen;
    seen.reserve(n);
    std::vector<double> rows;
    std::vector<int64_t> lab;
    std::vector<uint32_t> inverse(n);
    for (size_t i = 0; i < n; i++) {
        const double row[3] = {rho[i], z[i], zp[i]};
        bool ins = false;
        inverse[i] = seen.find_or_insert(row, static_cast<uint32_t>(rows.size() / 3), ins);
        if (ins) {
            rows.insert(rows.end(), row, row + 3);
            if (labels) lab.push_back(labels[i]);
        }
    }
    const size_t m = rows.size() / 3;
    std::vector<Six> block(m);
    evaluate_fresh(ev, rows.data(), m, labels ? lab.data() : nullptr, nullptr, nullptr, block.data());
    for (size_t i = 0; i < n; i++) out[i] = block[inverse[i]];
}

// --------------------------------------------------------------------------
// The sheet plan (`sheet_plan`, `_census`).

namespace {

// `_xy_groups`: the distinct exact (x, y) (sorted lexicographically, as
// np.unique(axis=0)) and each node's group.
void xy_groups(const std::vector<double> &x, const std::vector<double> &y, std::vector<double> &gx,
               std::vector<double> &gy, std::vector<int64_t> &inv) {
    std::map<std::pair<double, double>, int64_t> ids;
    for (size_t i = 0; i < x.size(); i++) ids.emplace(std::make_pair(x[i] + 0.0, y[i] + 0.0), 0);
    int64_t g = 0;
    gx.clear();
    gy.clear();
    for (auto &kv : ids) {
        kv.second = g++;
        gx.push_back(kv.first.first);
        gy.push_back(kv.first.second);
    }
    inv.resize(x.size());
    for (size_t i = 0; i < x.size(); i++) inv[i] = ids[std::make_pair(x[i] + 0.0, y[i] + 0.0)];
}

struct Grouped {
    std::vector<double> gx, gy;
    std::vector<std::vector<double>> vals;  // per group, distinct coord ascending
    std::vector<double> lo, hi;
};

// `_grouped(nodes, 2)`.
Grouped grouped(const std::vector<double> &nodes) {
    const size_t n = nodes.size() / 3;
    std::vector<double> x(n), y(n);
    for (size_t i = 0; i < n; i++) {
        x[i] = nodes[3 * i];
        y[i] = nodes[3 * i + 1];
    }
    Grouped g;
    std::vector<int64_t> inv;
    xy_groups(x, y, g.gx, g.gy, inv);
    std::vector<std::set<double>> sets(g.gx.size());
    for (size_t i = 0; i < n; i++) sets[static_cast<size_t>(inv[i])].insert(nodes[3 * i + 2] + 0.0);
    for (auto &s : sets) {
        g.vals.emplace_back(s.begin(), s.end());
        g.lo.push_back(*s.begin());
        g.hi.push_back(*s.rbegin());
    }
    return g;
}

// `_rho_matrix(gxy, hxy, a)`: hypot(hypot(gx - hx, gy - hy), a), (G, H).
std::vector<double> rho_matrix(const std::vector<double> &gx, const std::vector<double> &gy, size_t g_rows,
                               const std::vector<double> &hx, const std::vector<double> &hy, double a) {
    std::vector<double> out(g_rows * hx.size());
    for (size_t g = 0; g < g_rows; g++)
        for (size_t h = 0; h < hx.size(); h++)
            out[g * hx.size() + h] = std::hypot(std::hypot(gx[g] - hx[h], gy[g] - hy[h]), a);
    return out;
}

size_t n_unique(std::vector<double> v) {
    for (double &x : v) x += 0.0;
    std::sort(v.begin(), v.end());
    return static_cast<size_t>(std::unique(v.begin(), v.end()) - v.begin());
}

struct Part {
    std::vector<double> r;  // (G, H) row-major
    size_t G = 0, H = 0;
    std::vector<std::vector<double>> zg;
    std::vector<double> sep;  // per group
    std::vector<char> credit;  // per H
};

// `_distinct_pairs(parts)`.
size_t distinct_pairs(const std::vector<Part> &parts) {
    std::vector<std::vector<double>> rows_of;
    std::vector<size_t> count_of;
    std::map<double, std::vector<size_t>> by_z;
    for (const Part &p : parts) {
        for (size_t g = 0; g < p.G; g++) {
            std::vector<double> rs(p.r.begin() + static_cast<std::ptrdiff_t>(g * p.H),
                                   p.r.begin() + static_cast<std::ptrdiff_t>((g + 1) * p.H));
            std::sort(rs.begin(), rs.end());
            size_t cnt = 1;
            for (size_t j = 1; j < rs.size(); j++) cnt += rs[j] != rs[j - 1];
            const size_t gid = rows_of.size();
            rows_of.push_back(rs);
            count_of.push_back(cnt);
            for (double zv : p.zg[g]) by_z[zv + 0.0].push_back(gid);
        }
    }
    size_t n = 0;
    std::map<std::vector<size_t>, size_t> uni;
    for (auto &kv : by_z) {
        const std::vector<size_t> &gids = kv.second;
        if (gids.size() == 1) {
            n += count_of[gids[0]];
            continue;
        }
        auto it = uni.find(gids);
        if (it == uni.end()) {
            std::vector<double> all;
            for (size_t g : gids) all.insert(all.end(), rows_of[g].begin(), rows_of[g].end());
            it = uni.emplace(gids, n_unique(all)).first;
        }
        n += it->second;
    }
    return n;
}

// `_plane_pays(k_p, k_m, d, parts, depth)`.
bool plane_pays(double k_p, cd k_m, double d, const std::vector<Part> &parts, double depth) {
    double rho_max = -std::numeric_limits<double>::infinity(), s_max = rho_max;
    for (const Part &p : parts) {
        for (double v : p.r) rho_max = std::max(rho_max, v);
        for (double v : p.sep) s_max = std::max(s_max, v);
    }
    const double nodes = sheet_nodes_to(k_p, k_m, d, rho_max, s_max, depth);
    const double need = nodes * (depth < 0 ? SHEET_BUILD_WEIGHT : SHEET_HEIGHT_WEIGHT);
    const double w = SHEET_ROW_WORTH;
    double n_rows_ub = 0, n_rho_ub = 0;
    std::vector<std::vector<double>> credited;
    for (const Part &p : parts) {
        size_t zs = 0;
        for (const auto &z : p.zg) zs += z.size();
        n_rows_ub += static_cast<double>(p.H * zs);
        std::vector<double> c;
        for (size_t g = 0; g < p.G; g++)
            for (size_t h = 0; h < p.H; h++)
                if (p.credit[h]) c.push_back(p.r[g * p.H + h]);
        n_rho_ub += static_cast<double>(c.size());
        credited.push_back(std::move(c));
    }
    if (n_rho_ub + n_rows_ub * w < need) return false;
    const size_t cap = static_cast<size_t>(std::max(4096.0, 4 * need));
    std::vector<double> pre;
    size_t got = 0;
    for (const auto &c : credited) {
        const size_t take = std::min(c.size(), cap - got);
        pre.insert(pre.end(), c.begin(), c.begin() + static_cast<std::ptrdiff_t>(take));
        got += take;
        if (got >= cap) break;
    }
    if (static_cast<double>(n_unique(pre)) * (1.0 + w) >= need) return true;
    std::vector<double> all;
    for (const auto &c : credited) all.insert(all.end(), c.begin(), c.end());
    return static_cast<double>(n_unique(all)) + static_cast<double>(distinct_pairs(parts)) * w >= need;
}

struct Side {
    std::vector<double> above, below;  // (n, 3)
};

struct Fixed {
    const Grouped *g;
    std::vector<double> hx, hy;
    std::vector<char> credit;
};

// `_prefix_pays`.
bool prefix_pays(double k_p, cd k_m, double d, const std::vector<Fixed> &fixed, double rho_far,
                 double s_max, double a) {
    const double nodes = sheet_nodes_to(k_p, k_m, d, std::hypot(rho_far, a), s_max, -1);
    const double need = nodes * SHEET_BUILD_WEIGHT;
    const size_t cap = static_cast<size_t>(std::max(4096.0, 4 * need));
    std::vector<double> pre;
    size_t got = 0;
    for (const Fixed &f : fixed) {
        std::vector<double> hx, hy;
        for (size_t h = 0; h < f.hx.size(); h++)
            if (f.credit[h]) {
                hx.push_back(f.hx[h]);
                hy.push_back(f.hy[h]);
            }
        if (hx.empty()) continue;
        const size_t rows = std::min(f.g->gx.size(), (cap - got + hx.size() - 1) / hx.size());
        const std::vector<double> blk = rho_matrix(f.g->gx, f.g->gy, rows, hx, hy, a);
        pre.insert(pre.end(), blk.begin(), blk.end());
        got += blk.size();
        if (got >= cap) break;
    }
    return static_cast<double>(n_unique(pre)) * (1.0 + SHEET_ROW_WORTH) >= need;
}

// `_height_reach(deep)`.
double height_reach(double deep) {
    return std::pow(2.0, std::ceil(std::log2(std::max(deep, std::pow(2.0, -6.0)))));
}

struct Census {
    std::vector<double> planes;
    std::vector<std::pair<double, double>> heights;
};

// `_census(sides, height, k_p, k_m, a, skip)`.
void census(const std::vector<Side> &sides, bool height, double k_p, cd k_m, double a,
            const std::vector<double> &skip, Census &out) {
    struct Per {
        Grouped g;
        std::vector<double> fx;  // (n, 3)
        std::vector<double> vals;
        std::vector<int64_t> inv;
        std::vector<char> big, lone;
    };
    std::vector<Per> per;
    std::map<double, std::array<double, 4>> cand;  // n_rows, n_rho, rho_lb, s_lb
    for (const Side &sd : sides) {
        std::vector<double> vary, fixed;
        if (height) {
            for (size_t i = 0; i < sd.below.size() / 3; i++)
                if (!std::binary_search(skip.begin(), skip.end(), sd.below[3 * i + 2]) &&
                    std::find(skip.begin(), skip.end(), sd.below[3 * i + 2]) == skip.end())
                    vary.insert(vary.end(), sd.below.begin() + 3 * i, sd.below.begin() + 3 * i + 3);
            fixed = sd.above;
        } else {
            vary = sd.above;
            fixed = sd.below;
        }
        std::vector<double> fx;
        for (size_t i = 0; i < fixed.size() / 3; i++)
            if (height ? fixed[3 * i + 2] > 0.0 : fixed[3 * i + 2] < 0.0)
                fx.insert(fx.end(), fixed.begin() + 3 * i, fixed.begin() + 3 * i + 3);
        const size_t n_vary = vary.size() / 3, nfx = fx.size() / 3;
        if (n_vary == 0 || nfx == 0) continue;
        Per P;
        // np.unique(fx z, return_inverse, return_counts)
        std::vector<double> zz(nfx);
        for (size_t i = 0; i < nfx; i++) zz[i] = fx[3 * i + 2];
        std::vector<double> vals(zz);
        std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
        std::vector<int64_t> inv(nfx);
        std::vector<int64_t> cnt(vals.size(), 0);
        for (size_t i = 0; i < nfx; i++) {
            inv[i] = std::lower_bound(vals.begin(), vals.end(), zz[i]) - vals.begin();
            cnt[static_cast<size_t>(inv[i])]++;
        }
        std::vector<char> lone(nfx, 0);
        if (!height) {
            std::vector<double> x(nfx), y(nfx), fgx, fgy;
            for (size_t i = 0; i < nfx; i++) {
                x[i] = fx[3 * i];
                y[i] = fx[3 * i + 1];
            }
            std::vector<int64_t> finv;
            xy_groups(x, y, fgx, fgy, finv);
            std::set<std::pair<int64_t, int64_t>> pairs;
            for (size_t i = 0; i < nfx; i++) pairs.emplace(finv[i], inv[i]);
            std::vector<int64_t> per_group(fgx.size(), 0);
            for (const auto &pr : pairs) per_group[static_cast<size_t>(pr.first)]++;
            for (size_t i = 0; i < nfx; i++) lone[i] = per_group[static_cast<size_t>(finv[i])] == 1;
        }
        std::vector<double> n_lone(vals.size(), 0.0);
        for (size_t i = 0; i < nfx; i++) n_lone[static_cast<size_t>(inv[i])] += lone[i] ? 1.0 : 0.0;
        std::vector<char> big(vals.size(), 0);
        bool any_big = false;
        for (size_t v = 0; v < vals.size(); v++) {
            big[v] = static_cast<double>(cnt[v] * static_cast<int64_t>(n_vary)) >= SHEET_MIN_ROWS;
            any_big |= big[v] != 0;
        }
        if (!any_big) continue;
        P.g = grouped(vary);
        double zlo = std::numeric_limits<double>::infinity(), zhi = -zlo;
        for (double v : P.g.lo) zlo = std::min(zlo, v);
        for (double v : P.g.hi) zhi = std::max(zhi, v);
        std::vector<double> far(vals.size(), 0.0);
        for (size_t j = 0; j < nfx; j++) {
            if (!big[static_cast<size_t>(inv[j])]) continue;
            double dj = -std::numeric_limits<double>::infinity();
            for (size_t g = 0; g < P.g.gx.size(); g++)
                dj = std::max(dj, std::hypot(fx[3 * j] - P.g.gx[g], fx[3 * j + 1] - P.g.gy[g]));
            far[static_cast<size_t>(inv[j])] = std::max(far[static_cast<size_t>(inv[j])], dj);
        }
        for (size_t i = 0; i < vals.size(); i++) {
            if (!big[i]) continue;
            const double v = vals[i];
            const double sep = height ? v - zlo : zhi - v;
            auto it = cand.find(v);
            if (it == cand.end()) it = cand.emplace(v, std::array<double, 4>{0, 0, 0, 0}).first;
            it->second[0] += static_cast<double>(cnt[i] * static_cast<int64_t>(n_vary));
            it->second[1] += n_lone[i] * static_cast<double>(P.g.gx.size());
            it->second[2] = std::max(it->second[2], far[i]);
            it->second[3] = std::max(it->second[3], sep);
        }
        P.fx = fx;
        P.vals = vals;
        P.inv = inv;
        P.big = big;
        P.lone = lone;
        per.push_back(std::move(P));
    }
    const double w = SHEET_ROW_WORTH;
    const double weight = height ? SHEET_HEIGHT_WEIGHT : SHEET_BUILD_WEIGHT;
    const double per_cell = SHEET_P * SHEET_P;
    for (const auto &kv : cand) {
        const double v = kv.first;
        const double n_rows = kv.second[0], n_rho = kv.second[1], rho_lb = kv.second[2], s_lb = kv.second[3];
        if (std::fabs(v) < SHEET_MIN_DEPTH) continue;
        const double d = std::fabs(v);
        const double n_r = 2 + std::max(0.0, std::ceil(std::log2(std::max(rho_lb, d) / d)));
        const double n_s = std::max(1.0, std::ceil(std::log2(std::max(s_lb, d) / d)));
        const double best = n_rows * w + (height ? 0.0 : n_rho);
        if (best < n_r * (height ? 1.0 : n_s) * per_cell * weight ||
            (!height && best < sheet_nodes_to(k_p, k_m, d, rho_lb, s_lb, -1) * weight))
            continue;
        std::vector<Fixed> fixed;
        for (const Per &P : per) {
            const size_t i = static_cast<size_t>(std::lower_bound(P.vals.begin(), P.vals.end(), v) - P.vals.begin());
            if (i >= P.vals.size() || P.vals[i] != v || !P.big[i]) continue;
            std::vector<double> x, y;
            std::vector<char> lone_on;
            for (size_t j = 0; j < P.inv.size(); j++)
                if (static_cast<size_t>(P.inv[j]) == i) {
                    x.push_back(P.fx[3 * j]);
                    y.push_back(P.fx[3 * j + 1]);
                    lone_on.push_back(P.lone[j]);
                }
            Fixed f;
            f.g = &P.g;
            std::vector<int64_t> hinv;
            xy_groups(x, y, f.hx, f.hy, hinv);
            f.credit.assign(f.hx.size(), 1);
            for (size_t j = 0; j < hinv.size(); j++)
                f.credit[static_cast<size_t>(hinv[j])] = f.credit[static_cast<size_t>(hinv[j])] && lone_on[j];
            fixed.push_back(std::move(f));
        }
        if (!height && prefix_pays(k_p, k_m, d, fixed, rho_lb, s_lb, a)) {
            out.planes.push_back(v);
            continue;
        }
        std::vector<Part> parts;
        double deep = 0.0;
        for (const Fixed &f : fixed) {
            Part pt;
            pt.G = f.g->gx.size();
            pt.H = f.hx.size();
            pt.r = rho_matrix(f.g->gx, f.g->gy, pt.G, f.hx, f.hy, a);
            pt.zg = f.g->vals;
            pt.credit = f.credit;
            for (size_t g = 0; g < pt.G; g++) pt.sep.push_back(height ? v - f.g->lo[g] : f.g->hi[g] - v);
            if (height) {
                double mn = std::numeric_limits<double>::infinity();
                for (double l : f.g->lo) mn = std::min(mn, l);
                deep = std::max(deep, -mn);
            }
            parts.push_back(std::move(pt));
        }
        const double depth = height ? height_reach(deep) : -1.0;
        if (!plane_pays(k_p, k_m, d, parts, depth)) continue;
        if (height)
            out.heights.emplace_back(v, depth);
        else
            out.planes.push_back(v);
    }
}

}  // namespace

SheetPlan sheet_plan(double k_p, cd k_m, const std::vector<NodePair> &specs, double wire_radius) {
    STUDY_T("ni.sheet_plan");
    std::vector<Side> sides;
    for (const NodePair &np_ : specs) {
        Side sd;
        for (size_t i = 0; i < np_.n_above; i++) {
            const double *q = np_.above + 3 * i;
            if (std::isfinite(q[0]) && std::isfinite(q[1]) && std::isfinite(q[2]) && q[2] >= 0.0)
                sd.above.insert(sd.above.end(), q, q + 3);
        }
        for (size_t i = 0; i < np_.n_below; i++) {
            const double *q = np_.below + 3 * i;
            if (std::isfinite(q[0]) && std::isfinite(q[1]) && std::isfinite(q[2]) && q[2] <= 0.0)
                sd.below.insert(sd.below.end(), q, q + 3);
        }
        if (!sd.above.empty() && !sd.below.empty()) sides.push_back(std::move(sd));
    }
    Census planes, heights;
    census(sides, false, k_p, k_m, wire_radius, {}, planes);
    std::vector<double> skip = planes.planes;
    std::sort(skip.begin(), skip.end());
    census(sides, true, k_p, k_m, wire_radius, skip, heights);
    SheetPlan plan;
    plan.planes = planes.planes;
    std::sort(plan.planes.begin(), plan.planes.end());
    std::vector<std::pair<double, double>> hd = heights.heights;
    std::sort(hd.begin(), hd.end());
    for (auto &p : hd) {
        plan.heights.push_back(p.first);
        plan.depths.push_back(p.second);
    }
    return plan;
}

}  // namespace ni_eval
