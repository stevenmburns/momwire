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
namespace factorize {

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
                            bool floats) {
    if (n >= static_cast<py::ssize_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("factorize: too many rows for 32-bit groups");
    py::array_t<py::ssize_t> inverse(n);
    py::ssize_t *inv = inverse.mutable_data();
    std::vector<int64_t> first;
    {
        py::gil_scoped_release nogil;
        first.reserve(static_cast<size_t>(n / 4 + 16));
        size_t cap = 16;
        while (cap < 2 * static_cast<size_t>(n)) cap <<= 1;
        std::vector<uint64_t> table(cap, 0);
        const size_t mask = cap - 1;
        const size_t K = cols.size();
        uint64_t key[3];
        for (py::ssize_t i = 0; i < n; ++i) {
            bool nan = false;
            uint64_t h = 0x9e3779b97f4a7c15ULL;
            for (size_t k = 0; k < K; ++k) {
                uint64_t b = cols[k].at(i);
                if (floats) {
                    nan |= is_nan(b);
                    b = float_key(b);
                }
                key[k] = b;
                h = mix(h ^ b);
            }
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
    std::vector<py::array_t<double, py::array::forcecast>> arrs) {
    py::ssize_t n = 0;
    auto cols = columns(arrs, &n);
    return group_rows(cols, n, true);
}

static py::tuple factorize_ints(
    std::vector<py::array_t<int64_t, py::array::forcecast>> arrs) {
    py::ssize_t n = 0;
    auto cols = columns(arrs, &n);
    return group_rows(cols, n, false);
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
        std::vector<py::array_t<double, py::array::forcecast>> arrs) const {
        py::ssize_t q = 0;
        auto cols = columns(arrs, &q);
        if (cols.size() != K_)
            throw std::runtime_error("RowIndex.find: wrong number of columns");
        py::array_t<int64_t> out(q);
        int64_t *o = out.mutable_data();
        {
            py::gil_scoped_release nogil;
            uint64_t key[3];
            for (py::ssize_t i = 0; i < q; ++i) {
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
        std::vector<py::array_t<double, py::array::forcecast>> arrs) {
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
            uint64_t key[3], raw[3];
            for (py::ssize_t i = 0; i < n; ++i) {
                bool nan = false;
                uint64_t h = 0x9e3779b97f4a7c15ULL;
                for (int k = 0; k < 3; ++k) {
                    raw[k] = cols[k].at(i);
                    nan |= is_nan(raw[k]);
                    key[k] = float_key(raw[k]);
                    h = mix(h ^ key[k]);
                }
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

    void rehash() {
        size_t cap = table_.size() * 2;
        std::vector<uint64_t> fresh(cap, 0);
        const size_t mask = cap - 1;
        for (size_t s = 0; s < table_.size(); ++s) {
            const uint64_t slot = table_[s];
            if (slot == 0) continue;
            const uint64_t g = (slot & 0xffffffffULL) - 1;
            uint64_t h = 0x9e3779b97f4a7c15ULL;
            for (int k = 0; k < 3; ++k) h = mix(h ^ keys_[3 * g + k]);
            size_t t = static_cast<size_t>(h) & mask;
            while (fresh[t] != 0) t = (t + 1) & mask;
            fresh[t] = slot;
        }
        table_.swap(fresh);
        mask_ = cap - 1;
    }

    size_t mask_ = 0;
    std::vector<uint64_t> keys_;  // folded, for the compare
    std::vector<uint64_t> rows_;  // as given, for `rows()`
    std::vector<uint64_t> table_;
};

}  // namespace factorize

void register_factorize(py::module_ &m) {
    m.def("factorize_rows", &factorize::factorize_rows,
          "Exact-equality groups of the rows of 1-3 equal-length float64 "
          "columns (-0.0 == 0.0, a NaN row its own group), in first-appearance "
          "order: (first, inverse), first[g] the row where group g first "
          "occurs (ascending), inverse[i] row i's group. momwire#1224.",
          py::arg("cols"));
    m.def("factorize_ints", &factorize::factorize_ints,
          "`factorize_rows` for 1-3 int64 columns (equal iff the values are).",
          py::arg("cols"));
    py::class_<factorize::RowIndex>(m, "RowIndex")
        .def(py::init<std::vector<py::array_t<double, py::array::forcecast>>>(),
             py::arg("cols"))
        .def("find", &factorize::RowIndex::find, py::arg("cols"))
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
        .def("add", &factorize::RowGroups::add, py::arg("cols"))
        .def("rows", &factorize::RowGroups::rows)
        .def("__len__", &factorize::RowGroups::n_groups);
    // The capability flag, beside the bindings it vouches for (#710).
    m.attr("exact_factorize_1224") = true;
    m.attr("row_groups_1224") = true;
}
