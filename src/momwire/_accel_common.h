// Shared preamble for the split _accelerators translation units (momwire#687).
// Generated from the monolith's lines 1-120, then slimmed by the #710 review:
// the kernel headers (`_bspline_*_moments_inline.h`, `_contour_engine_inline.h`)
// moved into the TUs that use them, so editing one kernel no longer
// ccache-misses every TU — per-TU staleness is the split's whole point.
#pragma once

// M_PI is not in the C++ standard. GCC/glibc define it unconditionally, but
// MSVC only exposes it when _USE_MATH_DEFINES is set *before* the first math
// header is pulled in (directly or transitively via pybind11). Must stay at
// the very top of the file.
#define _USE_MATH_DEFINES

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <complex>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

// Explicit fused multiply-adds, since the build no longer contracts on its
// own (momwire#1194).
#include "_fma_inline.h"

namespace py = pybind11;

// The per-section registration seam. Declared here so every section TU
// compiles its definition against the same prototype the module TU calls —
// a drift would otherwise surface only as an undefined-symbol link error
// (or, for a forgotten call, not at all).
void register_bspline(py::module_ &m);
void register_sinusoidal(py::module_ &m);
void register_somm(py::module_ &m);
void register_mw568(py::module_ &m);
void register_razor(py::module_ &m);
void register_somm_pairs(py::module_ &m);
void register_factorize(py::module_ &m);
void register_left_gather(py::module_ &m);
void register_razor_t2(py::module_ &m);
void register_row_scatter(py::module_ &m);
void register_qp_pairs(py::module_ &m);
void register_exact_kernel(py::module_ &m);
void register_exact_kernel_point(py::module_ &m);
void register_exact_kernel_series(py::module_ &m);
void register_cmul(py::module_ &m);
void register_razor_solve(py::module_ &m);
void register_nec5_deck(py::module_ &m);
void register_nec4_printout(py::module_ &m);
void register_near_buried(py::module_ &m);

// The Python-free half: libmvec declarations, constants, OpenMP macros and
// the cancellation machinery now live in `_mw_core.h`, included here at the
// position they always occupied.
#include "_mw_core.h"
