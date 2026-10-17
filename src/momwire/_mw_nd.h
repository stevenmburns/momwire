// pybind11's array vocabulary over memory a HOST provides, with no Python: the
// types a kernel TU written against `py::array_t` compiles against when it is
// built as a Python-free core (`_bspline_core.cpp`, momwire#1517 step 0).
//
// The core TU says `namespace py = mw::nd;` and its kernel text -- `array_t`
// parameters, `unchecked<N>()` / `mutable_unchecked<N>()` proxies, `shape`,
// `data`, an output array constructed from its shape, `gil_scoped_release`,
// an `object` that may be None -- is the text the extension compiled, token
// for token. The proxies are `mw::view`s, which address memory as pybind11's
// proxies do (`_mw_core.h`); the checks and their wording are pybind11's
// (`unchecked<N>()` on the wrong rank, `shape(dim)` past the rank, a write
// into a read-only array).
//
// What Python did, a `Host` does: allocating an output array, releasing and
// re-acquiring the GIL, and casting an `object` argument to an array. The
// binding installs a host that does each through pybind11 (numpy allocates,
// so tracemalloc still sees every output; `_mw_nd_binding.h`); with none
// installed, the default host allocates with `new` and the GIL calls do
// nothing, which is the standalone executable's world.
//
// An array keeps its memory alive through `owner`: a host's allocation, or
// nothing for one that borrows a caller's buffer (a binding's arguments).
#pragma once

#include "_mw_core.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace mw {
namespace nd {

using ssize_t = std::ptrdiff_t;

// The element types the kernels' arrays hold.
enum class dtype : int { f64, c128, i64, i32, u8, boolean };

template <class T>
struct dtype_of;
template <>
struct dtype_of<double> {
    static constexpr dtype value = dtype::f64;
};
template <>
struct dtype_of<std::complex<double>> {
    static constexpr dtype value = dtype::c128;
};
template <>
struct dtype_of<int64_t> {
    static constexpr dtype value = dtype::i64;
};
template <>
struct dtype_of<int32_t> {
    static constexpr dtype value = dtype::i32;
};
template <>
struct dtype_of<uint8_t> {
    static constexpr dtype value = dtype::u8;
};
template <>
struct dtype_of<bool> {
    static constexpr dtype value = dtype::boolean;
};

// `py::array`'s flag names, for `array_t`'s second parameter.
struct array {
    static constexpr int c_style = 0x0001;
    static constexpr int f_style = 0x0002;
    static constexpr int forcecast = 0x0010;
};

constexpr int MAX_NDIM = 8;

// One array's memory as a host hands it over: the first element, the rank,
// shape and BYTE strides, whether it may be written, and what keeps it alive.
struct Buffer {
    void *ptr = nullptr;
    int ndim = 0;
    ssize_t shape[MAX_NDIM] = {};
    ssize_t strides[MAX_NDIM] = {};
    bool writeable = true;
    std::shared_ptr<void> owner;
};

class Host {
  public:
    virtual ~Host() = default;
    // A new C-contiguous array of `ndim` axes, its elements uninitialised.
    virtual Buffer allocate(dtype type, std::size_t itemsize, int ndim, const ssize_t *shape);
    // `py::gil_scoped_release` / `py::gil_scoped_acquire`: a token for the
    // matching restore.
    virtual void *release_gil() { return nullptr; }
    virtual void restore_gil(void *token) { (void)token; }
    virtual void *acquire_gil() { return nullptr; }
    virtual void drop_gil(void *token) { (void)token; }
    // `object.cast<py::array_t<T, flags>>()` of a non-None argument.
    virtual Buffer cast_array(const void *handle, dtype type, int flags) {
        (void)handle;
        (void)type;
        (void)flags;
        throw std::logic_error("mw::nd: this host has no objects to cast");
    }
};

// The host every array allocation and GIL call goes to: the one installed on
// this thread (`HostScope`), else the default.
inline Host *&installed_host() {
    static thread_local Host *h = nullptr;
    return h;
}
inline Host &default_host() {
    static Host h;
    return h;
}
inline Host &current_host() {
    Host *h = installed_host();
    return h != nullptr ? *h : default_host();
}

// Installs `h` on this thread for the scope's lifetime.
class HostScope {
  public:
    explicit HostScope(Host &h) : prev_(installed_host()) { installed_host() = &h; }
    ~HostScope() { installed_host() = prev_; }
    HostScope(const HostScope &) = delete;
    HostScope &operator=(const HostScope &) = delete;

  private:
    Host *prev_;
};

inline Buffer Host::allocate(dtype type, std::size_t itemsize, int ndim, const ssize_t *shape) {
    (void)type;
    Buffer b;
    b.ndim = ndim;
    std::size_t count = 1;
    for (int k = 0; k < ndim; k++) {
        b.shape[k] = shape[k];
        count *= static_cast<std::size_t>(shape[k]);
    }
    ssize_t acc = static_cast<ssize_t>(itemsize);
    for (int k = ndim - 1; k >= 0; k--) {
        b.strides[k] = acc;
        acc *= shape[k];
    }
    // 64-byte aligned, as numpy's allocations are at least 16
    const std::size_t bytes = count * itemsize;
    void *raw = ::operator new(bytes + 64);
    void *aligned = reinterpret_cast<void *>((reinterpret_cast<std::uintptr_t>(raw) + 64) &
                                             ~static_cast<std::uintptr_t>(63));
    b.owner = std::shared_ptr<void>(raw, [](void *p) { ::operator delete(p); });
    b.ptr = aligned;
    b.writeable = true;
    return b;
}

class gil_scoped_release {
  public:
    gil_scoped_release() : host_(current_host()), token_(host_.release_gil()) {}
    ~gil_scoped_release() { host_.restore_gil(token_); }
    gil_scoped_release(const gil_scoped_release &) = delete;
    gil_scoped_release &operator=(const gil_scoped_release &) = delete;

  private:
    Host &host_;
    void *token_;
};

class gil_scoped_acquire {
  public:
    gil_scoped_acquire() : host_(current_host()), token_(host_.acquire_gil()) {}
    ~gil_scoped_acquire() { host_.drop_gil(token_); }
    gil_scoped_acquire(const gil_scoped_acquire &) = delete;
    gil_scoped_acquire &operator=(const gil_scoped_acquire &) = delete;

  private:
    Host &host_;
    void *token_;
};

namespace detail {
// What `unchecked<N>()` returns, by its pybind11 name.
template <class T, int N>
using unchecked_reference = view<const T, N>;
}  // namespace detail

template <class T, int ExtraFlags = array::forcecast>
class array_t {
  public:
    // pybind11's default: a 1-D array of size 0.
    array_t() {
        buf_.ndim = 1;
        buf_.shape[0] = 0;
        buf_.strides[0] = static_cast<ssize_t>(sizeof(T));
    }
    // A new C-contiguous array of this shape, from the current host.
    template <class I, class = typename std::enable_if<std::is_integral<I>::value>::type>
    array_t(std::initializer_list<I> shape) {
        if (shape.size() > static_cast<std::size_t>(MAX_NDIM))
            throw std::length_error("mw::nd: too many dimensions");
        ssize_t s[MAX_NDIM];
        int n = 0;
        for (I v : shape) s[n++] = static_cast<ssize_t>(v);
        allocate(s, n);
    }
    explicit array_t(const std::vector<ssize_t> &shape) {
        if (shape.size() > static_cast<std::size_t>(MAX_NDIM))
            throw std::length_error("mw::nd: too many dimensions");
        allocate(shape.data(), static_cast<int>(shape.size()));
    }

    // An array over memory the caller provides (a binding's argument).
    static array_t borrow(const Buffer &b) {
        array_t a;
        a.buf_ = b;
        return a;
    }
    const Buffer &buffer() const { return buf_; }

    template <int N>
    view<const T, N> unchecked() const {
        check_rank(N);
        return view<const T, N>(static_cast<const T *>(buf_.ptr), buf_.shape, buf_.strides);
    }
    template <int N>
    view<T, N> mutable_unchecked() {
        check_rank(N);
        check_writeable();
        return view<T, N>(static_cast<T *>(buf_.ptr), buf_.shape, buf_.strides);
    }

    ssize_t ndim() const { return buf_.ndim; }
    const ssize_t *shape() const { return buf_.shape; }
    ssize_t shape(ssize_t dim) const {
        if (dim >= ndim()) fail_dim_check(dim, "invalid axis");
        return buf_.shape[dim];
    }
    const ssize_t *strides() const { return buf_.strides; }
    ssize_t size() const {
        ssize_t s = 1;
        for (int k = 0; k < buf_.ndim; k++) s *= buf_.shape[k];
        return s;
    }
    bool writeable() const { return buf_.writeable; }
    const T *data() const { return static_cast<const T *>(buf_.ptr); }
    T *mutable_data() {
        check_writeable();
        return static_cast<T *>(buf_.ptr);
    }

  private:
    void allocate(const ssize_t *shape, int ndim) {
        buf_ = current_host().allocate(dtype_of<T>::value, sizeof(T), ndim, shape);
    }
    void check_rank(int n) const {
        if (ndim() != n)
            throw std::domain_error("array has incorrect number of dimensions: " +
                                    std::to_string(ndim()) + "; expected " + std::to_string(n));
    }
    void check_writeable() const {
        if (!writeable()) throw std::domain_error("array is not writeable");
    }
    void fail_dim_check(ssize_t dim, const std::string &msg) const {
        // pybind11's `index_error`, which reaches Python as IndexError, as
        // std::out_of_range does
        throw std::out_of_range(msg + ": " + std::to_string(dim) + " (ndim = " +
                                std::to_string(ndim()) + ')');
    }

    Buffer buf_;
};

// A `py::object` argument that is None or an array: `is_none()`, and
// `cast<array_t<T, F>>()` through the host (the binding performs pybind11's
// own cast, conversion and errors included, at the point the kernel asks).
class object {
  public:
    object() = default;
    explicit object(const void *handle) : handle_(handle) {}
    bool is_none() const { return handle_ == nullptr; }
    const void *handle() const { return handle_; }
    template <class A>
    A cast() const {
        return cast_to(static_cast<A *>(nullptr));
    }

  private:
    template <class T, int F>
    array_t<T, F> cast_to(array_t<T, F> *) const {
        return array_t<T, F>::borrow(current_host().cast_array(handle_, dtype_of<T>::value, F));
    }
    const void *handle_ = nullptr;
};

inline object none() { return object(); }

}  // namespace nd
}  // namespace mw
