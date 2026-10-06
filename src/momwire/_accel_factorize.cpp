#include "_accel_common.h"

#include <cstring>
#include <limits>
#include <stdexcept>

// momwire#1224: exact-equality grouping of rows in FIRST-APPEARANCE order, by
// hashing instead of sorting.
//
// The crossing fill dedups (rho, z, z') triples, (x, y) pairs and integer
// codes everywhere (`_near_interface._unique_tri`, `_crossing_fill.
// _first_groups` / `_first_ints`), and the numpy spelling of that is a
// lexsort plus a first-appearance pass: O(n log n), and at razor's
// inverted-L x8 ~0.6 s of a 3.6 s crossing fill. The answer is a pure
// function of the equality classes -- each row's group number, groups
// numbered by their first occurrence -- so any machine that finds the same
// classes returns the same integers, and a hash table finds them in one
// pass.
//
// The classes are numpy's `!=` ones, which is what the sorted spelling
// compares adjacent rows with:
//   * -0.0 and 0.0 are one value (they compare equal);
//   * a NaN is never equal to anything, so a row holding one is a group of
//     its own (it is never entered in the table, and never matched);
//   * everything else is equal iff the bits are.
// That is decided on the BIT PATTERNS, with -0.0 folded to +0.0 and NaN read
// off the exponent and mantissa -- never with floating-point compares -- so
// no compiler mode can change it (MSVC builds this TU with /fp:fast, under
// which `x != x` and `x + 0.0` are both licensed to fold away).
//
// No floating-point arithmetic is done at all, so the FMA and inlining
// hazards of the numeric TUs (momwire#1194) do not apply; its own TU anyway,
// so nothing here can move another kernel's codegen.

// A read prefetch: a hint only. It never faults and changes no value, so a
// stale one (a slot since filled, a key vector since moved) costs nothing.
#if defined(__GNUC__) || defined(__clang__)
#define MW_PREFETCH(p) __builtin_prefetch(static_cast<const void *>(p))
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <xmmintrin.h>
#define MW_PREFETCH(p) _mm_prefetch(reinterpret_cast<const char *>(p), _MM_HINT_T0)
#else
#define MW_PREFETCH(p) ((void)(p))
#endif

namespace factorize {

// Rows hashed, and their home slots prefetched, ahead of a walk.
static constexpr py::ssize_t kPrefetchBlock = 32;

// One column: element i of a 1-D array of 8-byte values, any stride.
struct Col {
    const char *base;
    py::ssize_t stride;
    inline uint64_t at(py::ssize_t i) const {
        uint64_t b;
        std::memcpy(&b, base + i * stride, 8);
        return b;
    }
};

// A float64's equality key: its bits, -0.0 folded onto +0.0.
static inline uint64_t float_key(uint64_t b) {
    return (b << 1) == 0 ? 0 : b;
}

static inline bool is_nan(uint64_t b) {
    return (b & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL &&
           (b & 0x000fffffffffffffULL) != 0;
}

static inline uint64_t mix(uint64_t h) {
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

// The grouping itself. `floats`: the columns are float64 (fold -0.0, NaN
// unequal); otherwise they are int64 and equal iff their bits are.
//
// The table is open addressing with linear probing, at least twice as many
// slots as rows. A slot packs the hash's top 32 bits (a cheap reject before
// the row compare) over the group number plus one (0 = empty).
static py::tuple group_rows(const std::vector<Col> &cols, py::ssize_t n,
                            bool floats, uintptr_t cancel_flag) {
    if (n >= static_cast<py::ssize_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("factorize: too many rows for 32-bit groups");
    py::array_t<py::ssize_t> inverse(n);
    py::ssize_t *inv = inverse.mutable_data();
    std::vector<int64_t> first;
    {
        py::gil_scoped_release nogil;
        MW_CANCEL_SERIAL_SETUP(cancel_flag);
        first.reserve(static_cast<size_t>(n / 4 + 16));
        size_t cap = 16;
        while (cap < 2 * static_cast<size_t>(n)) cap <<= 1;
        std::vector<uint64_t> table(cap, 0);
        const size_t mask = cap - 1;
        const size_t K = cols.size();
        uint64_t key[3], hs[kPrefetchBlock];
        // Hashed a block ahead with each home slot prefetched, then walked
        // row by row exactly as before: `RowGroups::add`'s argument (a
        // prefetch reads nothing it could change).
        for (py::ssize_t i0 = 0; i0 < n; i0 += kPrefetchBlock) {
            MW_CANCEL_SERIAL_POLL();  // per 32-row block
            const py::ssize_t i1 = std::min(n, i0 + kPrefetchBlock);
            for (py::ssize_t i = i0; i < i1; ++i) {
                uint64_t h = 0x9e3779b97f4a7c15ULL;
                for (size_t k = 0; k < K; ++k) {
                    const uint64_t b = cols[k].at(i);
                    h = mix(h ^ (floats ? float_key(b) : b));
                }
                hs[i - i0] = h;
                MW_PREFETCH(table.data() + (static_cast<size_t>(h) & mask));
            }
            for (py::ssize_t i = i0; i < i1; ++i) {
                bool nan = false;
                for (size_t k = 0; k < K; ++k) {
                    uint64_t b = cols[k].at(i);
                    if (floats) {
                        nan |= is_nan(b);
                        b = float_key(b);
                    }
                    key[k] = b;
                }
                const uint64_t h = hs[i - i0];
                if (nan) {
                    inv[i] = static_cast<py::ssize_t>(first.size());
                    first.push_back(i);
                    continue;
                }
                const uint64_t tag = h & 0xffffffff00000000ULL;
                size_t s = static_cast<size_t>(h) & mask;
                for (;;) {
                    const uint64_t slot = table[s];
                    if (slot == 0) {
                        const uint64_t g = first.size();
                        table[s] = tag | (g + 1);
                        inv[i] = static_cast<py::ssize_t>(g);
                        first.push_back(i);
                        break;
                    }
                    if ((slot & 0xffffffff00000000ULL) == tag) {
                        const uint64_t g = (slot & 0xffffffffULL) - 1;
                        const py::ssize_t f = first[g];
                        bool eq = true;
                        for (size_t k = 0; k < K && eq; ++k) {
                            uint64_t b = cols[k].at(f);
                            eq = (floats ? float_key(b) : b) == key[k];
                        }
                        if (eq) {
                            inv[i] = static_cast<py::ssize_t>(g);
                            break;
                        }
                    }
                    s = (s + 1) & mask;
                }
            }
        }
    }
    py::array_t<int64_t> out_first(static_cast<py::ssize_t>(first.size()));
    if (!first.empty())
        std::memcpy(out_first.mutable_data(), first.data(),
                    first.size() * sizeof(int64_t));
    return py::make_tuple(out_first, inverse);
}

template <class T>
static std::vector<Col> columns(const std::vector<py::array_t<T>> &arrs,
                                py::ssize_t *n) {
    if (arrs.empty() || arrs.size() > 3)
        throw std::runtime_error("factorize: 1 to 3 columns");
    std::vector<Col> cols;
    for (const auto &a : arrs) {
        if (a.ndim() != 1)
            throw std::runtime_error("factorize: columns must be 1-D");
        if (a.shape(0) != arrs[0].shape(0))
            throw std::runtime_error("factorize: columns of unequal length");
        cols.push_back(
            Col{reinterpret_cast<const char *>(a.data()), a.strides(0)});
    }
    *n = arrs[0].shape(0);
    return cols;
}

// The arrays are taken WITHOUT `c_style`, so a strided column view (a column
// of an (n, 3) array) is read in place rather than copied; `forcecast` only
// converts a column of another dtype.
static py::tuple factorize_rows(
    std::vector<py::array_t<double, py::array::forcecast>> arrs,
    uintptr_t cancel_flag = 0) {
    py::ssize_t n = 0;
    auto cols = columns(arrs, &n);
    return group_rows(cols, n, true, cancel_flag);
}

static py::tuple factorize_ints(
    std::vector<py::array_t<int64_t, py::array::forcecast>> arrs,
    uintptr_t cancel_flag = 0) {
    py::ssize_t n = 0;
    auto cols = columns(arrs, &n);
    return group_rows(cols, n, false, cancel_flag);
}

// `factorize_rows` of the crossing plan's (groups, line) key table
// (momwire#1335): row i is (line.flat[i], lz[i % nL]) -- the two columns
// `_crossing_fill._product_plan` handed `factorize_rows` as `line.ravel()` and
// `broadcast_to(lz, line.shape).ravel()` -- grouped by the same equality
// (`group_rows`: -0.0 folded, a NaN row its own group, bits otherwise) in
// first-appearance order, so the groups and their numbers are the same
// integers. Only the memory differs, which is the whole point: at razor's
// inverted L x32 that call was 31.5 M rows and a ~1.1 GB transient beside
// the line itself. Here lz is read in place (no broadcast copy), the ids come
// back int32 in the table's own (groups, line) shape (no int64 inverse and no
// narrowing copy after it), and the hash table grows with the GROUPS
// (rehash at half load, re-entering each group from its first row) instead
// of being sized at twice the rows up front. Which slot a group occupies
// never decides its number: numbers are handed out in walk order.
static py::tuple factorize_line_keys(py::array_t<double, py::array::c_style> line,
                                     py::array_t<double, py::array::c_style> lz,
                                     uintptr_t cancel_flag = 0) {
    if (line.ndim() != 2)
        throw std::runtime_error("factorize_line_keys: line must be 2-D");
    const py::ssize_t nG = line.shape(0), nL = line.shape(1);
    if (lz.ndim() != 1 || lz.shape(0) != nL)
        throw std::runtime_error("factorize_line_keys: one lz per line node");
    const py::ssize_t n = nG * nL;
    if (n >= static_cast<py::ssize_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("factorize_line_keys: too many rows for 32-bit ids");
    const double *Lp = line.data();
    const double *Zp = lz.data();
    py::array_t<int32_t> kid(std::vector<py::ssize_t>{nG, nL});
    int32_t *out = kid.mutable_data();
    std::vector<int64_t> first;
    if (n > 0) {
        py::gil_scoped_release nogil;
        MW_CANCEL_SERIAL_SETUP(cancel_flag);
        auto col_bits = [&](py::ssize_t i, uint64_t *b) {
            std::memcpy(&b[0], Lp + i, 8);
            std::memcpy(&b[1], Zp + (i % nL), 8);
        };
        auto hash_of = [](const uint64_t *k) {
            uint64_t h = 0x9e3779b97f4a7c15ULL;
            h = mix(h ^ k[0]);
            h = mix(h ^ k[1]);
            return h;
        };
        // Sized for half the rows (a quarter-full table if every row were
        // its own key) and grown at half load from there: the crossing
        // plan's tables hold about one key per three rows, so this skips the
        // early rehashes without reaching past the size growth ends at.
        size_t cap = 1024;
        while (cap < static_cast<size_t>(n / 2)) cap <<= 1;
        std::vector<uint64_t> table(cap, 0);
        size_t mask = cap - 1;
        size_t n_entered = 0;
        // Each entered group's hash low 32 bits (momwire#1335): a rehash
        // re-enters the old table's entries from their tag and these, never
        // re-reading a group's first row (a random read per group). A slot
        // only ever holds its group, so where it lands moves no number.
        std::vector<uint32_t> hlo;
        auto rehash = [&]() {
            const size_t cap2 = cap * 2;
            std::vector<uint64_t> fresh(cap2, 0);
            const size_t mask2 = cap2 - 1;
            for (size_t t0 = 0; t0 < cap; ++t0) {
                const uint64_t slot = table[t0];
                if (slot == 0) continue;
                const uint64_t g = (slot & 0xffffffffULL) - 1;
                size_t t = static_cast<size_t>(hlo[g]) & mask2;
                while (fresh[t] != 0) t = (t + 1) & mask2;
                fresh[t] = slot;
            }
            table.swap(fresh);
            cap = cap2;
            mask = mask2;
        };
        // Walked row by row of the table and a block of line nodes at a
        // time, so a row's z is lz[l] with no division (the flat index is
        // only ever g * nL + l); the order is the flat order.
        uint64_t hs[kPrefetchBlock];
        for (py::ssize_t g = 0; g < nG; ++g) {
            const double *Lg = Lp + g * nL;
            for (py::ssize_t l0 = 0; l0 < nL; l0 += kPrefetchBlock) {
                MW_CANCEL_SERIAL_POLL();  // per 32-row block
                const py::ssize_t l1 = std::min(nL, l0 + kPrefetchBlock);
                for (py::ssize_t l = l0; l < l1; ++l) {
                    uint64_t b[2];
                    std::memcpy(&b[0], Lg + l, 8);
                    std::memcpy(&b[1], Zp + l, 8);
                    b[0] = float_key(b[0]);
                    b[1] = float_key(b[1]);
                    hs[l - l0] = hash_of(b);
                    MW_PREFETCH(table.data() + (static_cast<size_t>(hs[l - l0]) & mask));
                }
                for (py::ssize_t l = l0; l < l1; ++l) {
                    const py::ssize_t i = g * nL + l;
                    uint64_t b[2];
                    std::memcpy(&b[0], Lg + l, 8);
                    std::memcpy(&b[1], Zp + l, 8);
                    if (is_nan(b[0]) || is_nan(b[1])) {
                        out[i] = static_cast<int32_t>(first.size());
                        first.push_back(i);
                        hlo.push_back(0);  // never entered
                        continue;
                    }
                    b[0] = float_key(b[0]);
                    b[1] = float_key(b[1]);
                    if (2 * (n_entered + 1) > cap) rehash();
                    const uint64_t h = hs[l - l0];
                    const uint64_t tag = h & 0xffffffff00000000ULL;
                    size_t s = static_cast<size_t>(h) & mask;
                    for (;;) {
                        const uint64_t slot = table[s];
                        if (slot == 0) {
                            const uint64_t gg = first.size();
                            table[s] = tag | (gg + 1);
                            ++n_entered;
                            out[i] = static_cast<int32_t>(gg);
                            first.push_back(i);
                            hlo.push_back(static_cast<uint32_t>(h));
                            break;
                        }
                        if ((slot & 0xffffffff00000000ULL) == tag) {
                            const uint64_t gg = (slot & 0xffffffffULL) - 1;
                            uint64_t f[2];
                            col_bits(static_cast<py::ssize_t>(first[gg]), f);
                            if (float_key(f[0]) == b[0] && float_key(f[1]) == b[1]) {
                                out[i] = static_cast<int32_t>(gg);
                                break;
                            }
                        }
                        s = (s + 1) & mask;
                    }
                }
            }
        }
    }
    py::array_t<int64_t> out_first(static_cast<py::ssize_t>(first.size()));
    if (!first.empty())
        std::memcpy(out_first.mutable_data(), first.data(), first.size() * sizeof(int64_t));
    return py::make_tuple(out_first, kid);
}

// A persistent exact-equality index over the rows of 1-3 float64 columns:
// `find(cols)` is, per query row, the index of the FIRST stored row equal to
// it, or -1 (a NaN row is never stored and never found) -- the lookup of
// `_near_interface.KeyIndex`, whose keys are distinct pairs (momwire#1224).
// The canonical keys are copied in, so the arrays it was built from may go.
class RowIndex {
  public:
    explicit RowIndex(
        std::vector<py::array_t<double, py::array::forcecast>> arrs) {
        py::ssize_t n = 0;
        auto cols = columns(arrs, &n);
        if (n >= static_cast<py::ssize_t>(std::numeric_limits<int32_t>::max()))
            throw std::runtime_error("RowIndex: too many rows");
        K_ = cols.size();
        n_ = n;
        py::gil_scoped_release nogil;
        keys_.resize(K_ * static_cast<size_t>(n));
        size_t cap = 16;
        while (cap < 2 * static_cast<size_t>(n)) cap <<= 1;
        table_.assign(cap, 0);
        mask_ = cap - 1;
        uint64_t key[3];
        for (py::ssize_t i = 0; i < n; ++i) {
            uint64_t h;
            const bool nan = row_key(cols, i, key, &h);
            uint64_t *dst = &keys_[K_ * static_cast<size_t>(i)];
            for (size_t k = 0; k < K_; ++k) dst[k] = key[k];
            if (nan) continue;
            const uint64_t tag = h & 0xffffffff00000000ULL;
            size_t s = static_cast<size_t>(h) & mask_;
            for (;;) {
                const uint64_t slot = table_[s];
                if (slot == 0) {
                    table_[s] = tag | (static_cast<uint64_t>(i) + 1);
                    break;
                }
                if ((slot & 0xffffffff00000000ULL) == tag &&
                    same(key, (slot & 0xffffffffULL) - 1))
                    break;  // a repeat: the first row keeps the key
                s = (s + 1) & mask_;
            }
        }
    }

    py::array_t<int64_t> find(
        std::vector<py::array_t<double, py::array::forcecast>> arrs,
        uintptr_t cancel_flag = 0) const {
        py::ssize_t q = 0;
        auto cols = columns(arrs, &q);
        if (cols.size() != K_)
            throw std::runtime_error("RowIndex.find: wrong number of columns");
        py::array_t<int64_t> out(q);
        int64_t *o = out.mutable_data();
        {
            py::gil_scoped_release nogil;
            MW_CANCEL_SERIAL_SETUP(cancel_flag);
            uint64_t key[3];
            for (py::ssize_t i = 0; i < q; ++i) {
                MW_CANCEL_SERIAL_POLL();
                uint64_t h;
                o[i] = -1;
                if (row_key(cols, i, key, &h)) continue;
                const uint64_t tag = h & 0xffffffff00000000ULL;
                size_t s = static_cast<size_t>(h) & mask_;
                for (;;) {
                    const uint64_t slot = table_[s];
                    if (slot == 0) break;
                    const uint64_t r = (slot & 0xffffffffULL) - 1;
                    if ((slot & 0xffffffff00000000ULL) == tag && same(key, r)) {
                        o[i] = static_cast<int64_t>(r);
                        break;
                    }
                    s = (s + 1) & mask_;
                }
            }
        }
        return out;
    }

    py::ssize_t size() const { return n_; }

  private:
    // Row i's canonical key and hash; true when it holds a NaN.
    bool row_key(const std::vector<Col> &cols, py::ssize_t i, uint64_t *key,
                 uint64_t *h) const {
        bool nan = false;
        uint64_t hh = 0x9e3779b97f4a7c15ULL;
        for (size_t k = 0; k < K_; ++k) {
            const uint64_t b = cols[k].at(i);
            nan |= is_nan(b);
            key[k] = float_key(b);
            hh = mix(hh ^ key[k]);
        }
        *h = hh;
        return nan;
    }

    bool same(const uint64_t *key, uint64_t r) const {
        const uint64_t *stored = &keys_[K_ * static_cast<size_t>(r)];
        for (size_t k = 0; k < K_; ++k)
            if (stored[k] != key[k]) return false;
        return true;
    }

    size_t K_ = 0;
    py::ssize_t n_ = 0;
    size_t mask_ = 0;
    std::vector<uint64_t> keys_;
    std::vector<uint64_t> table_;
};

// `_near_interface.TripleMemo`'s store as one hash table (momwire#1224): the
// (rho, z, z') keys of every inserted row, in insertion order, with their
// `width` complex values. `lookup` answers each asked row's stored values
// (the FIRST insertion of an equal key; a NaN row is stored but never
// found), `contains` the hit mask alone. The memo's contract is exact-`==`
// keys with -0.0 folded onto 0.0 (it stored `row + 0.0`), which is
// `float_key`; the values are copied in and out, never computed on.
class TripleTable {
  public:
    explicit TripleTable(py::ssize_t width) : width_(width) {
        if (width < 0) throw std::runtime_error("TripleTable: negative width");
        table_.assign(16, 0);
        mask_ = 15;
    }

    void insert(
        py::array_t<double, py::array::c_style | py::array::forcecast> rows,
        py::array_t<std::complex<double>,
                    py::array::c_style | py::array::forcecast>
            vals) {
        if (rows.ndim() != 2 || rows.shape(1) != 3)
            throw std::runtime_error("TripleTable.insert: rows must be (m, 3)");
        const py::ssize_t m = rows.shape(0);
        if (vals.ndim() != 2 || vals.shape(0) != m || vals.shape(1) != width_)
            throw std::runtime_error("TripleTable.insert: vals must be (m, width)");
        if (n_ + m >= static_cast<py::ssize_t>(std::numeric_limits<int32_t>::max()))
            throw std::runtime_error("TripleTable: too many rows");
        const double *r = rows.data();
        const std::complex<double> *v = vals.data();
        py::gil_scoped_release nogil;
        const size_t n_new = static_cast<size_t>(n_ + m);
        grow(vals_, n_new * static_cast<size_t>(width_));
        vals_.insert(vals_.end(), v, v + static_cast<size_t>(m * width_));
        grow(keys_, 3 * n_new);
        for (py::ssize_t i = 0; i < 3 * m; ++i) {
            uint64_t b;
            std::memcpy(&b, r + i, 8);
            keys_.push_back(float_key(b));
        }
        const py::ssize_t first_new = n_;
        n_ += m;
        if (2 * static_cast<size_t>(n_) > table_.size()) {
            rehash();  // places every row, the new ones included
            return;
        }
        for (py::ssize_t i = first_new; i < n_; ++i) place(i);
    }

    py::tuple lookup(
        py::array_t<double, py::array::c_style | py::array::forcecast> rows)
        const {
        const py::ssize_t q = check_rows(rows);
        py::array_t<bool> hit(q);
        py::array_t<std::complex<double>> block({q, width_});
        bool *h = hit.mutable_data();
        std::complex<double> *out = block.mutable_data();
        const double *r = rows.data();
        {
            py::gil_scoped_release nogil;
            for (py::ssize_t i = 0; i < q; ++i) {
                const int64_t id = find(r + 3 * i);
                h[i] = id >= 0;
                if (id >= 0)
                    std::memcpy(out + i * width_, &vals_[static_cast<size_t>(id * width_)],
                                static_cast<size_t>(width_) * sizeof(std::complex<double>));
            }
        }
        return py::make_tuple(hit, block);
    }

    py::array_t<bool> contains(
        py::array_t<double, py::array::c_style | py::array::forcecast> rows)
        const {
        const py::ssize_t q = check_rows(rows);
        py::array_t<bool> hit(q);
        bool *h = hit.mutable_data();
        const double *r = rows.data();
        {
            py::gil_scoped_release nogil;
            for (py::ssize_t i = 0; i < q; ++i) h[i] = find(r + 3 * i) >= 0;
        }
        return hit;
    }

    // The stored keys (folded) and values, in insertion order.
    py::array_t<double> keys() const {
        py::array_t<double> out({n_, static_cast<py::ssize_t>(3)});
        if (n_) std::memcpy(out.mutable_data(), keys_.data(), keys_.size() * 8);
        return out;
    }

    py::array_t<std::complex<double>> values() const {
        py::array_t<std::complex<double>> out({n_, width_});
        if (!vals_.empty())
            std::memcpy(out.mutable_data(), vals_.data(),
                        vals_.size() * sizeof(std::complex<double>));
        return out;
    }

    py::ssize_t size() const { return n_; }

  private:
    template <class T>
    static void grow(std::vector<T> &v, size_t need) {
        if (need > v.capacity()) v.reserve(std::max(need, v.capacity() + v.capacity() / 2));
    }

    static py::ssize_t check_rows(
        const py::array_t<double, py::array::c_style | py::array::forcecast> &rows) {
        if (rows.ndim() != 2 || rows.shape(1) != 3)
            throw std::runtime_error("TripleTable: rows must be (n, 3)");
        return rows.shape(0);
    }

    static uint64_t hash3(const uint64_t *k) {
        uint64_t h = 0x9e3779b97f4a7c15ULL;
        for (int c = 0; c < 3; ++c) h = mix(h ^ k[c]);
        return h;
    }

    // Stored row i into the table, unless a NaN row or an equal key is
    // already there (the first insertion keeps the key).
    void place(py::ssize_t i) {
        const uint64_t *k = &keys_[3 * static_cast<size_t>(i)];
        if (is_nan(k[0]) || is_nan(k[1]) || is_nan(k[2])) return;
        const uint64_t h = hash3(k);
        const uint64_t tag = h & 0xffffffff00000000ULL;
        size_t s = static_cast<size_t>(h) & mask_;
        for (;;) {
            const uint64_t slot = table_[s];
            if (slot == 0) {
                table_[s] = tag | (static_cast<uint64_t>(i) + 1);
                return;
            }
            if ((slot & 0xffffffff00000000ULL) == tag &&
                same(k, (slot & 0xffffffffULL) - 1))
                return;
            s = (s + 1) & mask_;
        }
    }

    void rehash() {
        size_t cap = table_.size();
        while (cap < 2 * static_cast<size_t>(n_)) cap <<= 1;
        table_.assign(cap, 0);
        mask_ = cap - 1;
        for (py::ssize_t i = 0; i < n_; ++i) place(i);
    }

    bool same(const uint64_t *k, uint64_t id) const {
        const uint64_t *s = &keys_[3 * static_cast<size_t>(id)];
        return s[0] == k[0] && s[1] == k[1] && s[2] == k[2];
    }

    int64_t find(const double *row) const {
        uint64_t k[3];
        for (int c = 0; c < 3; ++c) {
            uint64_t b;
            std::memcpy(&b, row + c, 8);
            if (is_nan(b)) return -1;
            k[c] = float_key(b);
        }
        const uint64_t h = hash3(k);
        const uint64_t tag = h & 0xffffffff00000000ULL;
        size_t s = static_cast<size_t>(h) & mask_;
        for (;;) {
            const uint64_t slot = table_[s];
            if (slot == 0) return -1;
            const uint64_t id = (slot & 0xffffffffULL) - 1;
            if ((slot & 0xffffffff00000000ULL) == tag && same(k, id))
                return static_cast<int64_t>(id);
            s = (s + 1) & mask_;
        }
    }

    py::ssize_t width_;
    py::ssize_t n_ = 0;
    size_t mask_ = 0;
    std::vector<uint64_t> keys_;
    std::vector<std::complex<double>> vals_;
    std::vector<uint64_t> table_;
};

// `factorize_rows` over a sequence of calls (momwire#1224): `add(cols)`
// numbers each row's group in first-appearance order ACROSS every add so
// far, as one `factorize_rows` over the concatenation would, and `rows()`
// hands back each group's first row with its original floats (sign of zero
// and NaN payload kept, as `_unique_tri` returns `tri[first]`). The
// crossing fill's chunked point tables feed it the grid a chunk at a time.
class RowGroups {
  public:
    RowGroups() {
        table_.assign(16, 0);
        mask_ = 15;
    }

    py::array_t<py::ssize_t> add(
        std::vector<py::array_t<double, py::array::forcecast>> arrs,
        uintptr_t cancel_flag = 0) {
        py::ssize_t n = 0;
        auto cols = columns(arrs, &n);
        if (cols.size() != 3)
            throw std::runtime_error("RowGroups.add: three columns");
        if (static_cast<py::ssize_t>(rows_.size() / 3) + n >=
            static_cast<py::ssize_t>(std::numeric_limits<int32_t>::max()))
            throw std::runtime_error("RowGroups: too many groups");
        py::array_t<py::ssize_t> ids(n);
        py::ssize_t *out = ids.mutable_data();
        {
            py::gil_scoped_release nogil;
            uint64_t key[3], raw[3], hs[kPrefetchBlock];
            MW_CANCEL_SERIAL_SETUP(cancel_flag);
            for (py::ssize_t i0 = 0; i0 < n; i0 += kPrefetchBlock) {
                // Between blocks the grouping is whole; an abort leaves the
                // rows added so far, and the caller discards the object.
                MW_CANCEL_SERIAL_POLL();
                const py::ssize_t i1 = std::min(n, i0 + kPrefetchBlock);
                // Two prefetch passes over the block, then the rows in order.
                // Past a few million groups the table and the keys are far
                // out of cache, and a row probed alone waits on two misses in
                // turn (its slot, then the group's keys it names): sin's
                // inverted-L x32 grid, 30 M rows into 10.9 M groups, took
                // 3.6 s. Issued a block at a time the misses overlap, and with
                // the rehash below it takes 1.75 s (Skylake, momwire#1224 perf
                // item 6). The passes only READ (a slot a
                // later row of the block fills, a mask a rehash replaces or a
                // key vector a push moves just make a hint useless), and the
                // third pass is the row-at-a-time walk itself, so the groups
                // and their numbering are untouched.
                for (py::ssize_t i = i0; i < i1; ++i) {
                    uint64_t h = 0x9e3779b97f4a7c15ULL;
                    for (int k = 0; k < 3; ++k) h = mix(h ^ float_key(cols[k].at(i)));
                    hs[i - i0] = h;
                    MW_PREFETCH(table_.data() + (static_cast<size_t>(h) & mask_));
                }
                for (py::ssize_t i = i0; i < i1; ++i) {
                    const uint64_t slot =
                        table_[static_cast<size_t>(hs[i - i0]) & mask_];
                    if (slot != 0)
                        MW_PREFETCH(keys_.data() + 3 * ((slot & 0xffffffffULL) - 1));
                }
                for (py::ssize_t i = i0; i < i1; ++i) {
                    bool nan = false;
                    for (int k = 0; k < 3; ++k) {
                        raw[k] = cols[k].at(i);
                        nan |= is_nan(raw[k]);
                        key[k] = float_key(raw[k]);
                    }
                    const uint64_t h = hs[i - i0];
                    if (nan) {
                        out[i] = new_group(raw, key, 0, false);
                        continue;
                    }
                    if (2 * (n_groups() + 1) > table_.size()) rehash();
                    const uint64_t tag = h & 0xffffffff00000000ULL;
                    size_t s = static_cast<size_t>(h) & mask_;
                    for (;;) {
                        const uint64_t slot = table_[s];
                        if (slot == 0) {
                            out[i] = new_group(raw, key, s, true, tag);
                            break;
                        }
                        const uint64_t g = (slot & 0xffffffffULL) - 1;
                        if ((slot & 0xffffffff00000000ULL) == tag && same(key, g)) {
                            out[i] = static_cast<py::ssize_t>(g);
                            break;
                        }
                        s = (s + 1) & mask_;
                    }
                }
            }
        }
        return ids;
    }

    py::array_t<double> rows() const {
        py::array_t<double> out({n_groups(), static_cast<py::ssize_t>(3)});
        if (!rows_.empty())
            std::memcpy(out.mutable_data(), rows_.data(), rows_.size() * 8);
        return out;
    }

    py::ssize_t n_groups() const { return static_cast<py::ssize_t>(keys_.size() / 3); }

  private:
    py::ssize_t new_group(const uint64_t *raw, const uint64_t *key, size_t s,
                          bool enter, uint64_t tag = 0) {
        const py::ssize_t g = n_groups();
        for (int k = 0; k < 3; ++k) {
            keys_.push_back(key[k]);
            rows_.push_back(raw[k]);
        }
        if (enter) table_[s] = tag | (static_cast<uint64_t>(g) + 1);
        return g;
    }

    bool same(const uint64_t *key, uint64_t g) const {
        const uint64_t *k = &keys_[3 * g];
        return k[0] == key[0] && k[1] == key[1] && k[2] == key[2];
    }

    // Re-enters every group in NUMBER order: the keys are read in order and
    // only the fresh slot is a miss (prefetched a block ahead, as in `add`),
    // where walking the old table read each group's keys at random as well. The slots land elsewhere (linear
    // probing follows entry order), which changes no group or number: a
    // group's slot only ever holds that group. A row holding a NaN is a group
    // of its own that was never entered, and is not entered now.
    void rehash() {
        size_t cap = table_.size() * 2;
        std::vector<uint64_t> fresh(cap, 0);
        const size_t mask = cap - 1;
        const size_t n_g = keys_.size() / 3;
        uint64_t hs[kPrefetchBlock];
        for (size_t g0 = 0; g0 < n_g; g0 += kPrefetchBlock) {
            const size_t g1 = std::min(n_g, g0 + static_cast<size_t>(kPrefetchBlock));
            for (size_t g = g0; g < g1; ++g) {
                const uint64_t *k = &keys_[3 * g];
                uint64_t h = 0x9e3779b97f4a7c15ULL;
                for (int c = 0; c < 3; ++c) h = mix(h ^ k[c]);
                hs[g - g0] = h;
                MW_PREFETCH(fresh.data() + (static_cast<size_t>(h) & mask));
            }
            for (size_t g = g0; g < g1; ++g) {
                const uint64_t *k = &keys_[3 * g];
                if (is_nan(k[0]) || is_nan(k[1]) || is_nan(k[2])) continue;
                const uint64_t h = hs[g - g0];
                size_t t = static_cast<size_t>(h) & mask;
                while (fresh[t] != 0) t = (t + 1) & mask;
                fresh[t] = (h & 0xffffffff00000000ULL) | (static_cast<uint64_t>(g) + 1);
            }
        }
        table_.swap(fresh);
        mask_ = cap - 1;
    }

    size_t mask_ = 0;
    std::vector<uint64_t> keys_;  // folded, for the compare
    std::vector<uint64_t> rows_;  // as given, for `rows()`
    std::vector<uint64_t> table_;
};

// Each row's `factorize_ints` of a (groups, line) table of non-negative ids
// below `n_id` (the crossing plan's global key ids, `_product_plan`), in one
// pass: per row g, `rank[g, l]` is the local id of kid[g, l] in first-
// appearance order along the row, `n[g]` the row's distinct count, and
// `first` / `ids` (concatenated over the rows) each local id's first
// position and its global id -- what `_first_ints(kid[g])` and
// `kid[g, first]` answer row by row. A stamp per global id (the row that
// last saw it) replaces the per-row hash table; the integers are the same.
static py::tuple group_first_ranks(
    py::array_t<int32_t, py::array::c_style> kid, int64_t n_id,
    uintptr_t cancel_flag = 0) {
    if (kid.ndim() != 2) throw std::runtime_error("group_first_ranks: kid must be 2-D");
    const py::ssize_t nG = kid.shape(0), nL = kid.shape(1);
    if (n_id < 0 || n_id >= static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("group_first_ranks: n_id out of range");
    const int32_t *K = kid.data();
    for (py::ssize_t q = 0; q < nG * nL; ++q)
        if (K[q] < 0 || K[q] >= n_id)
            throw std::runtime_error("group_first_ranks: id out of range");
    py::array_t<int32_t> rank(std::vector<py::ssize_t>{nG, nL});
    py::array_t<int64_t> n(nG);
    int32_t *R = rank.mutable_data();
    int64_t *N = n.mutable_data();
    int64_t total = 0;
    {
        py::gil_scoped_release nogil;
        MW_CANCEL_SERIAL_SETUP(cancel_flag);
        std::vector<int32_t> stamp(static_cast<size_t>(n_id), -1), local(static_cast<size_t>(n_id), 0);
        for (py::ssize_t g = 0; g < nG; ++g) {
            MW_CANCEL_SERIAL_POLL();  // per row of nL ids
            const int32_t *row = K + g * nL;
            int32_t *out = R + g * nL;
            int32_t next = 0;
            const int32_t tag = static_cast<int32_t>(g);
            for (py::ssize_t l = 0; l < nL; ++l) {
                const int32_t k = row[l];
                if (stamp[k] != tag) {
                    stamp[k] = tag;
                    local[k] = next++;
                }
                out[l] = local[k];
            }
            N[g] = next;
            total += next;
        }
    }
    // The first positions and global ids, written straight into arrays of
    // their exact size (momwire#1335: growing two vectors and copying them
    // out was ~3x their size at the inverted L x32). Local id j of row g first
    // occurs at the first l with rank[g, l] == j, which is where the walk's
    // count of ids seen so far in that row is j.
    py::array_t<int64_t> out_first(static_cast<py::ssize_t>(total));
    py::array_t<int32_t> out_ids(static_cast<py::ssize_t>(total));
    {
        int64_t *F = out_first.mutable_data();
        int32_t *I = out_ids.mutable_data();
        py::gil_scoped_release nogil;
        MW_CANCEL_SERIAL_SETUP(cancel_flag);
        int64_t q = 0;
        for (py::ssize_t g = 0; g < nG; ++g) {
            MW_CANCEL_SERIAL_POLL();
            const int32_t *row = K + g * nL;
            const int32_t *rk = R + g * nL;
            int32_t next = 0;
            for (py::ssize_t l = 0; l < nL; ++l) {
                if (rk[l] == next) {
                    F[q] = static_cast<int64_t>(l);
                    I[q] = row[l];
                    ++q;
                    ++next;
                }
            }
        }
    }
    return py::make_tuple(rank, n, out_first, out_ids);
}

// The multi-group merge's row numbering (`_crossing_fill._merge_groups_z`)
// without hashing the candidates. The candidates are walked in grid order as
// BLOCKS: block b is grouped node a_b with z id z[b] against its group's
// keys `ids[off[g_b] : off[g_b] + n[g_b]]` (ascending line position), and
// candidate j of block b sits at walk position start[b] + j. A row is a
// distinct (z id, key) pair numbered by its first walk position -- which is
// `factorize_ints` of the codes z * n_key + key along the walk.
//
// Two candidates can only share a code when they share the z id, so the
// blocks are taken z id by z id (each z id's blocks in walk order) with one
// stamp per key: a key's first block under that z id is where its code
// first appears. The rows are then numbered in walk order. Returns (the
// row of each candidate in walk order, each row's first block, each row's
// offset in that block).
// The walk itself, its positions and row numbers in `I` (int32 whenever
// they fit, momwire#1335: at the inverted L x32 the candidate and row arrays
// are ~19 M and ~17 M entries, and int64 doubled the merge's spike). The
// integers are the same in either width.
template <class I>
static py::tuple merge_rows_walk(const int64_t *Z, const int64_t *G, const int64_t *S,
                                 const int32_t *K, const int64_t *O, py::ssize_t nb,
                                 int64_t n_cand, int64_t n_key, int64_t n_z,
                                 uintptr_t cancel_flag) {
    py::array_t<I> row(n_cand);
    I *R = row.mutable_data();
    // first[e]: the walk position where candidate e's code first appears.
    // Kept across the two passes below: the rows' first block and offset
    // are written into arrays of their exact size in a second pass (growing
    // two vectors and copying them out was ~3x their size at x32).
    std::vector<I> first(static_cast<size_t>(n_cand));
    int64_t n_rows = 0;
    {
        py::gil_scoped_release nogil;
        MW_CANCEL_SERIAL_SETUP(cancel_flag);
        // Blocks by z id, each z id's in walk order (a counting sort).
        std::vector<int64_t> cnt(static_cast<size_t>(n_z) + 1, 0);
        for (py::ssize_t b = 0; b < nb; ++b) cnt[Z[b] + 1]++;
        for (int64_t i = 0; i < n_z; ++i) cnt[i + 1] += cnt[i];
        std::vector<int64_t> by_z(static_cast<size_t>(nb));
        {
            std::vector<int64_t> at(cnt.begin(), cnt.end() - 1);
            for (py::ssize_t b = 0; b < nb; ++b) by_z[at[Z[b]]++] = b;
        }
        {
            std::vector<I> tag(static_cast<size_t>(n_key), static_cast<I>(-1));
            std::vector<I> pos(static_cast<size_t>(n_key), 0);
            for (int64_t zi = 0; zi < n_z; ++zi) {
                MW_CANCEL_SERIAL_POLL();
                if (cnt[zi + 1] - cnt[zi] == 1) {
                    // One block under this z id: its keys are distinct (a
                    // group's own), so every code is new where it stands.
                    const int64_t b = by_z[cnt[zi]];
                    const int64_t len = O[G[b] + 1] - O[G[b]];
                    for (int64_t j = 0; j < len; ++j)
                        first[S[b] + j] = static_cast<I>(S[b] + j);
                    continue;
                }
                const I zt = static_cast<I>(zi);
                for (int64_t i = cnt[zi]; i < cnt[zi + 1]; ++i) {
                    MW_CANCEL_SERIAL_POLL();  // per block
                    const int64_t b = by_z[i];
                    const int64_t g = G[b];
                    for (int64_t j = 0; j < O[g + 1] - O[g]; ++j) {
                        const int32_t k = K[O[g] + j];
                        const int64_t e = S[b] + j;
                        if (tag[k] != zt) {
                            tag[k] = zt;
                            pos[k] = static_cast<I>(e);
                        }
                        first[e] = pos[k];
                    }
                }
            }
        }
        // Rows numbered by first walk position, in walk order.
        int64_t next = 0;
        for (py::ssize_t b = 0; b < nb; ++b) {
            MW_CANCEL_SERIAL_POLL();
            const int64_t len = O[G[b] + 1] - O[G[b]];
            for (int64_t j = 0; j < len; ++j) {
                const int64_t e = S[b] + j;
                R[e] = static_cast<int64_t>(first[e]) == e ? static_cast<I>(next++)
                                                           : R[first[e]];
            }
        }
        n_rows = next;
    }
    // Each row's first block and offset there, in row order: the walk meets
    // the rows' first positions in the order it numbered them.
    py::array_t<I> fb(static_cast<py::ssize_t>(n_rows));
    py::array_t<I> fj(static_cast<py::ssize_t>(n_rows));
    {
        I *FB = fb.mutable_data();
        I *FJ = fj.mutable_data();
        py::gil_scoped_release nogil;
        MW_CANCEL_SERIAL_SETUP(cancel_flag);
        int64_t q = 0;
        for (py::ssize_t b = 0; b < nb; ++b) {
            MW_CANCEL_SERIAL_POLL();
            const int64_t len = O[G[b] + 1] - O[G[b]];
            for (int64_t j = 0; j < len; ++j) {
                const int64_t e = S[b] + j;
                if (static_cast<int64_t>(first[e]) == e) {
                    FB[q] = static_cast<I>(b);
                    FJ[q] = static_cast<I>(j);
                    ++q;
                }
            }
        }
        if (q != n_rows) throw std::runtime_error("merge_rows_by_z: row count");
    }
    return py::make_tuple(row, fb, fj);
}

static py::tuple merge_rows_by_z(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> z,      // (blocks,)
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> grp,    // (blocks,)
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> start,  // (blocks,)
    py::array_t<int32_t, py::array::c_style> ids,                           // keys, concatenated by group
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> off,    // (groups + 1,)
    int64_t n_key, int64_t n_z, uintptr_t cancel_flag = 0) {
    const py::ssize_t nb = z.size();
    if (grp.size() != nb || start.size() != nb)
        throw std::runtime_error("merge_rows_by_z: one z, group and start per block");
    const int64_t *Z = z.data(), *G = grp.data(), *S = start.data(), *O = off.data();
    const py::ssize_t ng = off.size() - 1;
    const int32_t *K = ids.data();
    int64_t n_cand = 0;
    for (py::ssize_t b = 0; b < nb; ++b) {
        if (Z[b] < 0 || Z[b] >= n_z || G[b] < 0 || G[b] >= ng)
            throw std::runtime_error("merge_rows_by_z: block z or group out of range");
        if (S[b] != n_cand)
            throw std::runtime_error("merge_rows_by_z: blocks must tile the walk");
        n_cand += O[G[b] + 1] - O[G[b]];
    }
    for (py::ssize_t q = 0; q < ids.size(); ++q)
        if (K[q] < 0 || K[q] >= n_key)
            throw std::runtime_error("merge_rows_by_z: key out of range");
    const int64_t lim = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    if (n_cand < lim && n_z < lim && static_cast<int64_t>(nb) < lim)
        return merge_rows_walk<int32_t>(Z, G, S, K, O, nb, n_cand, n_key, n_z, cancel_flag);
    return merge_rows_walk<int64_t>(Z, G, S, K, O, nb, n_cand, n_key, n_z, cancel_flag);
}

}  // namespace factorize

void register_factorize(py::module_ &m) {
    m.def("factorize_rows", &factorize::factorize_rows,
          "Exact-equality groups of the rows of 1-3 equal-length float64 "
          "columns (-0.0 == 0.0, a NaN row its own group), in first-appearance "
          "order: (first, inverse), first[g] the row where group g first "
          "occurs (ascending), inverse[i] row i's group. momwire#1224.",
          py::arg("cols"), py::arg("cancel_flag") = 0);
    m.def("factorize_ints", &factorize::factorize_ints,
          "`factorize_rows` for 1-3 int64 columns (equal iff the values are).",
          py::arg("cols"), py::arg("cancel_flag") = 0);
    py::class_<factorize::RowIndex>(m, "RowIndex")
        .def(py::init<std::vector<py::array_t<double, py::array::forcecast>>>(),
             py::arg("cols"))
        .def("find", &factorize::RowIndex::find, py::arg("cols"),
             py::arg("cancel_flag") = 0)
        .def("__len__", &factorize::RowIndex::size);
    py::class_<factorize::TripleTable>(m, "TripleTable")
        .def(py::init<py::ssize_t>(), py::arg("width"))
        .def("insert", &factorize::TripleTable::insert, py::arg("rows"),
             py::arg("vals"))
        .def("lookup", &factorize::TripleTable::lookup, py::arg("rows"))
        .def("contains", &factorize::TripleTable::contains, py::arg("rows"))
        .def("keys", &factorize::TripleTable::keys)
        .def("values", &factorize::TripleTable::values)
        .def("__len__", &factorize::TripleTable::size);
    py::class_<factorize::RowGroups>(m, "RowGroups")
        .def(py::init<>())
        .def("add", &factorize::RowGroups::add, py::arg("cols"),
             py::arg("cancel_flag") = 0)
        .def("rows", &factorize::RowGroups::rows)
        .def("__len__", &factorize::RowGroups::n_groups);
    m.def("group_first_ranks", &factorize::group_first_ranks,
          "Per row of a (groups, line) int32 id table, `factorize_ints` of "
          "the row in one pass: (rank (groups, line) int32, n (groups,), "
          "first positions and global ids of each row's local ids, "
          "concatenated). momwire#1290.",
          py::arg("kid"), py::arg("n_id"), py::arg("cancel_flag") = 0);
    m.attr("group_first_ranks_1290") = true;
    m.def("merge_rows_by_z", &factorize::merge_rows_by_z,
          "The multi-group merge's rows by first walk position, z id by z id "
          "with a stamp per key: (row per candidate in walk order, each row's "
          "first block, its offset there), int32 where they fit. momwire#1290, "
          "#1335.",
          py::arg("z"), py::arg("grp"), py::arg("start"), py::arg("ids"),
          py::arg("off"), py::arg("n_key"), py::arg("n_z"), py::arg("cancel_flag") = 0);
    m.attr("merge_rows_by_z_1290") = true;
    m.def("factorize_line_keys", &factorize::factorize_line_keys,
          "`factorize_rows` of the rows (line.flat[i], lz[i % nL]) of a "
          "(groups, line) float64 table and its line's z: (first int64, ids "
          "int32 (groups, line)), the same integers, lz read in place and the "
          "table grown with the groups. momwire#1335.",
          py::arg("line"), py::arg("lz"), py::arg("cancel_flag") = 0);
    m.attr("factorize_line_keys_1335") = true;
    // The capability flag, beside the bindings it vouches for (#710).
    m.attr("exact_factorize_1224") = true;
    m.attr("row_groups_1224") = true;
}
