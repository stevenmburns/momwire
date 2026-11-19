// `factorize_ints` over a sequence of calls, on 1 to 8 int64 columns
// (momwire#2042): `add` numbers each row's group in first-appearance order
// ACROSS every add so far, as one `factorize_ints` over the concatenation
// would (and so as `_sin_pair_reuse._group`'s two-stage spelling of five or
// six columns does: its second stage numbers a row where its whole key first
// occurs). The sinusoidal pair-reuse census folds its bands into one of these
// in band order, in Python (`_accel_factorize.cpp`'s `IntGroups`, which wraps
// this) and in the C++ fill (`_sin_pair_reuse_core.cpp`), so both hold the
// same groups, numbered alike, in the same bytes.
//
// Memory: the keys, `width` int64 per group in blocks of `kBlock` groups (so
// growth never copies them, nor holds twice their size while it does), and
// the table, at least twice as many slots as groups (8 bytes each, grown at
// half load); `nbytes` is both, what the census counts against its budget.
//
// Integers only: no floating-point arithmetic, so no compiler mode moves a
// group or its number. Python-free.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

// A read prefetch: a hint only. It never faults and changes no value, so a
// stale one (a slot since filled, a key vector since moved) costs nothing.
#ifndef MW_PREFETCH
#if defined(__GNUC__) || defined(__clang__)
#define MW_PREFETCH(p) __builtin_prefetch(static_cast<const void *>(p))
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <xmmintrin.h>
#define MW_PREFETCH(p) _mm_prefetch(reinterpret_cast<const char *>(p), _MM_HINT_T0)
#else
#define MW_PREFETCH(p) ((void)(p))
#endif
#endif

namespace mw {
namespace int_groups {

inline uint64_t mix(uint64_t h) {
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

// Rows hashed, and their home slots prefetched, ahead of a walk.
constexpr std::ptrdiff_t kPrefetchBlock = 32;

// One column: element i of a 1-D array of 8-byte values, any stride.
struct Col {
    const char *base;
    std::ptrdiff_t stride;
    inline uint64_t at(std::ptrdiff_t i) const {
        uint64_t b;
        std::memcpy(&b, base + i * stride, 8);
        return b;
    }
};

class IntGroups {
  public:
    // 1 <= width <= 8; the caller checks.
    explicit IntGroups(std::size_t width) : width_(width) {
        table_.assign(16, 0);
        mask_ = 15;
    }

    // Each of rows [0, n)'s group into out[i]; `poll()` before every
    // prefetch block (a caller's cancellation check, which may throw: the
    // rows added so far stay grouped, and the caller discards the object).
    template <class Poll>
    void add(const Col *cols, std::ptrdiff_t n, int64_t *out, Poll &&poll) {
        uint64_t key[8], hs[kPrefetchBlock];
        for (std::ptrdiff_t i0 = 0; i0 < n; i0 += kPrefetchBlock) {
            poll();
            const std::ptrdiff_t i1 = std::min(n, i0 + kPrefetchBlock);
            // Two read-only prefetch passes, then the rows in order: the
            // groups and their numbers are the walk's.
            for (std::ptrdiff_t i = i0; i < i1; ++i) {
                uint64_t h = 0x9e3779b97f4a7c15ULL;
                for (std::size_t k = 0; k < width_; ++k) h = mix(h ^ cols[k].at(i));
                hs[i - i0] = h;
                MW_PREFETCH(table_.data() + (static_cast<std::size_t>(h) & mask_));
            }
            for (std::ptrdiff_t i = i0; i < i1; ++i) {
                const uint64_t slot = table_[static_cast<std::size_t>(hs[i - i0]) & mask_];
                if (slot != 0) MW_PREFETCH(group_key((slot & 0xffffffffULL) - 1));
            }
            for (std::ptrdiff_t i = i0; i < i1; ++i) {
                for (std::size_t k = 0; k < width_; ++k) key[k] = cols[k].at(i);
                out[i] = insert(key, hs[i - i0]);
            }
        }
    }

    void add(const Col *cols, std::ptrdiff_t n, int64_t *out) {
        add(cols, n, out, [] {});
    }

    // One row, its hash computed here: the walk `add` makes, unprefetched.
    int64_t add_row(const uint64_t *key) {
        uint64_t h = 0x9e3779b97f4a7c15ULL;
        for (std::size_t k = 0; k < width_; ++k) h = mix(h ^ key[k]);
        return insert(key, h);
    }

    std::ptrdiff_t n_groups() const { return static_cast<std::ptrdiff_t>(n_); }
    std::size_t width() const { return width_; }

    std::ptrdiff_t nbytes() const {
        return static_cast<std::ptrdiff_t>((blocks_.size() * kBlock * width_ + table_.size()) * 8);
    }

    // Group g's key, `width` values. Groups g0 .. g0 + kBlock - 1 with g0 a
    // multiple of kBlock lie contiguous, row after row.
    const uint64_t *group_key(uint64_t g) const {
        return blocks_[g / kBlock].get() + width_ * (g % kBlock);
    }
    static constexpr std::size_t kBlock = std::size_t(1) << 12;

  private:

    int64_t insert(const uint64_t *key, uint64_t h) {
        if (2 * (n_ + 1) > table_.size()) rehash();
        const uint64_t tag = h & 0xffffffff00000000ULL;
        std::size_t s = static_cast<std::size_t>(h) & mask_;
        for (;;) {
            const uint64_t slot = table_[s];
            if (slot == 0) {
                const uint64_t g = n_;
                push_key(key);
                table_[s] = tag | (g + 1);
                return static_cast<int64_t>(g);
            }
            const uint64_t g = (slot & 0xffffffffULL) - 1;
            if ((slot & 0xffffffff00000000ULL) == tag && same(key, g)) return static_cast<int64_t>(g);
            s = (s + 1) & mask_;
        }
    }

    void push_key(const uint64_t *key) {
        if (n_ % kBlock == 0) blocks_.emplace_back(new uint64_t[kBlock * width_]);
        std::memcpy(blocks_.back().get() + width_ * (n_ % kBlock), key, width_ * 8);
        ++n_;
    }

    bool same(const uint64_t *key, uint64_t g) const {
        const uint64_t *k = group_key(g);
        for (std::size_t c = 0; c < width_; ++c)
            if (k[c] != key[c]) return false;
        return true;
    }

    // Every group re-entered in number order, which moves slots and never a
    // group or its number.
    void rehash() {
        const std::size_t cap = table_.size() * 2;
        std::vector<uint64_t> fresh(cap, 0);
        const std::size_t mask = cap - 1;
        const std::size_t n_g = n_;
        uint64_t hs[kPrefetchBlock];
        for (std::size_t g0 = 0; g0 < n_g; g0 += kPrefetchBlock) {
            const std::size_t g1 = std::min(n_g, g0 + static_cast<std::size_t>(kPrefetchBlock));
            for (std::size_t g = g0; g < g1; ++g) {
                const uint64_t *k = group_key(g);
                uint64_t h = 0x9e3779b97f4a7c15ULL;
                for (std::size_t c = 0; c < width_; ++c) h = mix(h ^ k[c]);
                hs[g - g0] = h;
                MW_PREFETCH(fresh.data() + (static_cast<std::size_t>(h) & mask));
            }
            for (std::size_t g = g0; g < g1; ++g) {
                const uint64_t h = hs[g - g0];
                std::size_t t = static_cast<std::size_t>(h) & mask;
                while (fresh[t] != 0) t = (t + 1) & mask;
                fresh[t] = (h & 0xffffffff00000000ULL) | (static_cast<uint64_t>(g) + 1);
            }
        }
        table_.swap(fresh);
        mask_ = cap - 1;
    }

    std::size_t width_;
    std::size_t mask_ = 0;
    std::size_t n_ = 0;
    std::vector<std::unique_ptr<uint64_t[]>> blocks_;
    std::vector<uint64_t> table_;
};

}  // namespace int_groups
}  // namespace mw
