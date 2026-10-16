// NpVec (numpy-backed growable arrays, so tracemalloc sees them), the dedup
// Key and its open-addressing KeyTable: shared by the 1410 and 1433 TUs, as
// `_exact_kernel_ring_inline.h` is. Included inside each TU's namespace.
// tracemalloc sees numpy's buffers but not a std::vector's, and the memgate
// lane measures the correction with tracemalloc (tests/
// test_exact_kernel_rows_1411.py). So every buffer here that scales with the
// window or the cache is a numpy array (numpy reports its allocations to
// tracemalloc): without that the gate would pass on the C++ route by not
// seeing its memory at all. (Reporting a std::vector through
// PyTraceMalloc_Track is not portable: CPython 3.12.3's header declares it
// without C linkage, so a C++ caller links against a mangled name.)
//
// A growable array of a trivially copyable T. Every operation that
// allocates, and the destructor, need the GIL; element access does not.
#ifdef MW_EK_NO_PYTHON
// The Python-free core TU (momwire#1479): the same interface over a plain
// vector (nothing there for tracemalloc to see).
template <class T>
class NpVec {
  public:
    void reserve(size_t c) { v_.reserve(c); }
    void resize(size_t n) { v_.resize(n); }
    void assign(size_t n, const T &v) { v_.assign(n, v); }
    void push_back(const T &v) { v_.push_back(v); }
    T &operator[](size_t i) { return v_[i]; }
    const T &operator[](size_t i) const { return v_[i]; }
    size_t size() const { return v_.size(); }
    bool empty() const { return v_.empty(); }
    T *data() { return v_.data(); }
    const T *data() const { return v_.data(); }

  private:
    std::vector<T> v_;
};
#else
template <class T>
class NpVec {
  public:
    NpVec() : n_(0), cap_(0), p_(0) {}
    void reserve(size_t c) {
        if (c <= cap_) return;
        py::array_t<uint8_t> a(static_cast<py::ssize_t>(c * sizeof(T)));
        T *q = reinterpret_cast<T *>(a.mutable_data());
        if (n_) std::memcpy(q, p_, n_ * sizeof(T));
        buf_ = a;
        p_ = q;
        cap_ = c;
    }
    void resize(size_t n) {  // new elements uninitialised
        reserve(n);
        n_ = n;
    }
    void assign(size_t n, const T &v) {
        resize(n);
        for (size_t i = 0; i < n; ++i) p_[i] = v;
    }
    void push_back(const T &v) {
        if (n_ == cap_) reserve(cap_ ? 2 * cap_ : 16);
        p_[n_++] = v;
    }
    T &operator[](size_t i) { return p_[i]; }
    const T &operator[](size_t i) const { return p_[i]; }
    size_t size() const { return n_; }
    bool empty() const { return n_ == 0; }
    T *data() { return p_; }
    const T *data() const { return p_; }

  private:
    size_t n_, cap_;
    T *p_;
    py::object buf_;
};

#endif

// ---------------------------------------------------------------------------
// the dedup key and its table
// ---------------------------------------------------------------------------
struct Key {
    int64_t hi, hj, c;
    int64_t s;
};

static inline bool key_eq(const Key &x, const Key &y) {
    return x.hi == y.hi && x.hj == y.hj && x.c == y.c && x.s == y.s;
}

static inline bool key_lt(const Key &x, const Key &y) {
    if (x.hi != y.hi) return x.hi < y.hi;
    if (x.hj != y.hj) return x.hj < y.hj;
    if (x.c != y.c) return x.c < y.c;
    return x.s < y.s;
}

static inline uint64_t mix64(uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline uint64_t key_hash(const Key &k) {
    uint64_t h = mix64(static_cast<uint64_t>(k.hi));
    h = mix64(h ^ static_cast<uint64_t>(k.hj));
    h = mix64(h ^ static_cast<uint64_t>(k.c));
    return mix64(h ^ static_cast<uint64_t>(k.s));
}

// Open addressing, linear probing, power-of-two capacity, load <= 1/2.
class KeyTable {
  public:
    KeyTable() : mask_(0), count_(0) {}
    void reserve(size_t n) {
        size_t cap = 16;
        while (cap < 2 * n) cap <<= 1;
        if (cap > keys_.size()) rehash(cap);
    }
    int64_t find(const Key &k) const {
        if (keys_.empty()) return -1;
        size_t i = key_hash(k) & mask_;
        while (true) {
            const int64_t v = vals_[i];
            if (v < 0) return -1;
            if (key_eq(keys_[i], k)) return v;
            i = (i + 1) & mask_;
        }
    }
    void insert(const Key &k, int64_t v) {  // k must be absent
        if (2 * (count_ + 1) > keys_.size()) rehash(keys_.empty() ? 16 : 2 * keys_.size());
        size_t i = key_hash(k) & mask_;
        while (vals_[i] >= 0) i = (i + 1) & mask_;
        keys_[i] = k;
        vals_[i] = v;
        ++count_;
    }
    size_t size() const { return count_; }
  private:
    void rehash(size_t cap) {
        NpVec<Key> ok = keys_;
        NpVec<int64_t> ov = vals_;
        keys_ = NpVec<Key>();
        vals_ = NpVec<int64_t>();
        keys_.assign(cap, Key());
        vals_.assign(cap, -1);
        mask_ = cap - 1;
        count_ = 0;
        for (size_t i = 0; i < ov.size(); ++i)
            if (ov[i] >= 0) insert(ok[i], ov[i]);
    }
    NpVec<Key> keys_;
    NpVec<int64_t> vals_;
    size_t mask_, count_;
};
