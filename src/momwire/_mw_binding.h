// numpy <-> mw::view at the binding seam (explore/core-binding-split).
//
// The ONLY place a numpy array becomes a core view. No copy is made here:
// a view points into the array's own buffer with the array's own strides.
// Whatever copy happens, happens BEFORE, in pybind11's argument conversion,
// and is decided by the binding's parameter type, exactly as before the split:
//
//   py::array_t<T, c_style | forcecast>  -> pybind11 hands over the caller's
//       array when it is already C-contiguous T, and otherwise a converted
//       C-contiguous COPY (dtype cast, Fortran order, a strided slice). A
//       write-through output must therefore NOT be taken this way.
//   py::array (no flags)                  -> the caller's array itself, any
//       strides; the binding checks dtype and writeability and passes the
//       strides through (`razor_t2_rows`' `out`, a window of column-major Z).
//
// The view does not own or keep alive anything. It is valid while the
// py::array it came from is alive, which is the binding function's frame;
// the core never stores a view past its return.
#pragma once

#include "_accel_common.h"

#include <sstream>

namespace mw {
namespace binding {

template <class T, int N, class A>
inline view<T, N> raw_view(const A &arr, T *ptr) {
    if (arr.ndim() != N) {
        // pybind11's own wording for unchecked<N>() on the wrong rank, so the
        // Python-visible error is unchanged.
        std::ostringstream m;
        m << "array has incorrect number of dimensions: " << arr.ndim()
          << "; expected " << N;
        throw std::domain_error(m.str());
    }
    ptrdiff_t shp[N > 0 ? N : 1], st[N > 0 ? N : 1];
    for (int k = 0; k < N; k++) {
        shp[k] = arr.shape(k);
        st[k] = arr.strides(k);
    }
    return view<T, N>(ptr, shp, st);
}

// A read-only input.
template <class T, int N, int F>
inline view<const T, N> in(const py::array_t<T, F> &arr) {
    return raw_view<const T, N>(arr, arr.data());
}

// A writable output the binding allocated (or the caller handed over).
template <class T, int N, int F>
inline view<T, N> out(py::array_t<T, F> &arr) {
    return raw_view<T, N>(arr, arr.mutable_data());
}

// An optional input: size 0 means absent (an empty view); otherwise its
// elements as one contiguous run, which is how the pre-split kernels read
// these arrays (through .data() and .size()).
template <class T, int F>
inline view<const T, 1> opt(const py::array_t<T, F> &arr) {
    if (!arr || arr.size() == 0) return view<const T, 1>();
    return view<const T, 1>::contiguous(arr.data(), {static_cast<ptrdiff_t>(arr.size())});
}

}  // namespace binding
}  // namespace mw
