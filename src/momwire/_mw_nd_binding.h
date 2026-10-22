// numpy <-> mw::nd at the binding seam (`_mw_nd.h`): the host a binding
// installs around a core call, and the conversions of its arguments and
// results. Like `_mw_binding.h`, the ONLY place a numpy array meets a core
// array type.
//
//   * An argument is BORROWED: the core's array points into the numpy
//     array's own buffer with its own strides and writeability, and keeps
//     nothing alive (the binding's frame does). Whatever copy happens,
//     happens before, in pybind11's argument conversion, decided by the
//     binding's parameter type, as it always was.
//   * A result the core allocates is a numpy array (`NumpyHost::allocate`:
//     `py::array_t<T>(shape)`, exactly the allocation the kernels made), so
//     tracemalloc sees it; `out` hands that same array back.
//   * `gil_scoped_release` / `gil_scoped_acquire` in the core are pybind11's,
//     constructed where the kernel text declares them.
//   * An optional argument (`py::object`, None or an array) is cast by
//     pybind11 at the point the kernel casts it, conversion and errors
//     included; the cast array lives as long as the core's array does.
#pragma once

#include "_accel_common.h"
#include "_mw_nd.h"

#include <complex>
#include <memory>
#include <new>
#include <type_traits>
#include <stdexcept>
#include <tuple>

namespace ndb {

using mw::nd::Buffer;
using mw::nd::dtype;

namespace detail {

// A Buffer over a numpy array, `owner` keeping `keep` alive (or nothing).
inline Buffer buffer_of(const py::array &a, std::shared_ptr<void> owner) {
    Buffer b;
    if (a.ndim() > mw::nd::MAX_NDIM) throw std::length_error("mw::nd: too many dimensions");
    b.ptr = const_cast<void *>(a.data());
    b.ndim = static_cast<int>(a.ndim());
    for (int k = 0; k < b.ndim; k++) {
        b.shape[k] = a.shape()[k];
        b.strides[k] = a.strides()[k];
    }
    b.writeable = a.writeable();
    b.owner = std::move(owner);
    return b;
}

// Keeps a Python object alive; dropped with the GIL held, as every array the
// kernels made always was.
inline std::shared_ptr<void> keep(const py::object &o) { return std::make_shared<py::object>(o); }

template <class T>
Buffer allocate_as(int ndim, const mw::nd::ssize_t *shape) {
    std::vector<py::ssize_t> s(shape, shape + ndim);
    py::array_t<T> a(s);
    return buffer_of(a, keep(a));
}

template <class T, int F>
Buffer cast_as(const void *handle) {
    py::handle h(reinterpret_cast<PyObject *>(const_cast<void *>(handle)));
    py::array_t<T, F> a = h.cast<py::array_t<T, F>>();
    return buffer_of(a, keep(a));
}

}  // namespace detail

class NumpyHost final : public mw::nd::Host {
  public:
    Buffer allocate(dtype type, std::size_t itemsize, int ndim,
                    const mw::nd::ssize_t *shape) override {
        (void)itemsize;
        switch (type) {
            case dtype::f64: return detail::allocate_as<double>(ndim, shape);
            case dtype::c128: return detail::allocate_as<std::complex<double>>(ndim, shape);
            case dtype::i64: return detail::allocate_as<int64_t>(ndim, shape);
            case dtype::i32: return detail::allocate_as<int32_t>(ndim, shape);
            case dtype::u8: return detail::allocate_as<uint8_t>(ndim, shape);
            case dtype::boolean: return detail::allocate_as<bool>(ndim, shape);
            case dtype::i8: return detail::allocate_as<int8_t>(ndim, shape);
        }
        throw std::logic_error("mw::nd: an element type with no numpy array");
    }
    // pybind11's own scoped objects, built in place where the kernel text
    // declares its scope (they nest, innermost last).
    void *release_gil() override {
        if (n_release_ == SLOTS) throw std::logic_error("mw::nd: GIL scopes nested too deep");
        return new (&release_[n_release_++]) py::gil_scoped_release();
    }
    void restore_gil(void *token) override {
        static_cast<py::gil_scoped_release *>(token)->~gil_scoped_release();
        n_release_--;
    }
    void *acquire_gil() override {
        if (n_acquire_ == SLOTS) throw std::logic_error("mw::nd: GIL scopes nested too deep");
        return new (&acquire_[n_acquire_++]) py::gil_scoped_acquire();
    }
    void drop_gil(void *token) override {
        static_cast<py::gil_scoped_acquire *>(token)->~gil_scoped_acquire();
        n_acquire_--;
    }
    Buffer cast_array(const void *handle, dtype type, int flags) override {
        constexpr int CF = py::array::c_style | py::array::forcecast;
        static_assert(CF == (mw::nd::array::c_style | mw::nd::array::forcecast),
                      "mw::nd's flags are pybind11's");
        if (flags == CF) {
            switch (type) {
                case dtype::f64: return detail::cast_as<double, CF>(handle);
                case dtype::i64: return detail::cast_as<int64_t, CF>(handle);
                case dtype::c128: return detail::cast_as<std::complex<double>, CF>(handle);
                case dtype::u8: return detail::cast_as<uint8_t, CF>(handle);
                default: break;
            }
        }
        throw std::logic_error("mw::nd: an object cast this binding does not perform");
    }

  private:
    static constexpr int SLOTS = 4;
    using ReleaseSlot = typename std::aligned_storage<sizeof(py::gil_scoped_release),
                                                      alignof(py::gil_scoped_release)>::type;
    using AcquireSlot = typename std::aligned_storage<sizeof(py::gil_scoped_acquire),
                                                      alignof(py::gil_scoped_acquire)>::type;
    ReleaseSlot release_[SLOTS];
    AcquireSlot acquire_[SLOTS];
    int n_release_ = 0, n_acquire_ = 0;
};

// A numpy argument, borrowed.
template <class T, int F>
mw::nd::array_t<T, F> in(const py::array_t<T, F> &a) {
    return mw::nd::array_t<T, F>::borrow(detail::buffer_of(a, nullptr));
}

// A `py::object` argument: None, or the object the core may cast.
inline mw::nd::object object(const py::object &o) {
    return o.is_none() ? mw::nd::object() : mw::nd::object(o.ptr());
}

// A result the core allocated through the NumpyHost: that numpy array.
template <class T, int F>
py::array_t<T> out(const mw::nd::array_t<T, F> &a) {
    const std::shared_ptr<void> &owner = a.buffer().owner;
    if (!owner) throw std::logic_error("mw::nd: a result the binding's host did not allocate");
    return py::reinterpret_borrow<py::array_t<T>>(*static_cast<py::object *>(owner.get()));
}

// A tuple of three such results.
template <class T0, int F0, class T1, int F1, class T2, int F2>
std::tuple<py::array_t<T0>, py::array_t<T1>, py::array_t<T2>> out_tuple(
    const std::tuple<mw::nd::array_t<T0, F0>, mw::nd::array_t<T1, F1>, mw::nd::array_t<T2, F2>> &t) {
    return std::make_tuple(out(std::get<0>(t)), out(std::get<1>(t)), out(std::get<2>(t)));
}

}  // namespace ndb
