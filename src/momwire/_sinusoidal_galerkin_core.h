// The Sinusoidal-Galerkin far fill, Python-free (momwire#1718 S1): the fused
// far fills `_accel_sinusoidal.cpp` binds -- `sinusoidal_galerkin_far_fill`,
// its extended-kernel twin `_ek` and its complex-wavenumber twin `_cplx` --
// by their bound names and with their bound arguments, over `mw::nd` arrays
// (`_mw_nd.h`) instead of numpy's. The arithmetic is
// `_sinusoidal_galerkin_core.cpp`'s, which is the text the extension TU
// compiled before the split (`galerkin_far_fill_impl` and its staged real
// rows, `galerkin_far_fill_cplx_impl` and its staged complex rows); the
// extension now calls these. The point-matched kernels' core is
// `_sinusoidal_core.h` (momwire#1607 B1), on the same pattern.
//
// Arrays: an argument is a `mw::nd::array_t` over the caller's memory
// (`array_t::borrow`), C-contiguous; a result is allocated by the current
// host (`mw::nd::current_host()`: the default host allocates with `new`; the
// extension installs numpy's). Each returns the three (nnz, N) contribution
// arrays of the source shapes (const, sin, cos).
//
// `fold` is the bound functions' `out` / `scale` (momwire#356), already
// checked: null, the fill allocates and returns its own three arrays; not
// null, nothing is allocated, each entry is folded on as
// `dst += scale * value` and the three destinations come back. The binding
// checks the Python `out` argument (a 3-sequence of complex128 (nnz, N)
// C-contiguous writeable arrays) before the call, as it always did; a C++
// caller hands the destinations it owns, which must meet the same contract.
// `reference` selects the per-pair scalar body over the staged vector one
// (bit-identical; the flag exists for the tests that say so).
//
// Errors are C++ exceptions with the extension's wording: std::runtime_error
// for a bad call, std::domain_error / std::out_of_range for pybind11's own
// array checks, and the cancellation machinery's `mw::AbortedError`.
#pragma once

#include "_mw_nd.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

namespace mw {
namespace sinusoidal_galerkin {

// The extended kernel's payload for the fused far fill (momwire#246 unit C).
// `src_a` is one radius per SOURCE segment (N) and `gx`/`gw` are the composite
// sinh-mapped rule `SinusoidalSolver._ek_delta_rule` built, passed in rather
// than rebuilt here so the two backends integrate against the same nodes by
// construction.
//
// Eligibility arrives as the pair rule's GROUP LABELS rather than as its mask
// (momwire#358): `g_obs` per TEST segment (M) and `g_src` per source segment
// (N), with the pair rule evaluated in the sweep as
//
//     eligible(observer row of test segment m, source n)
//         = g_obs[m] >= 0 && g_obs[m] == g_src[n]
//
// which is `SinusoidalGalerkinSolver._ek_pairs`' `(gi == gj) & (gi >= 0)`
// verbatim. The mask this replaces was an (M*nq, N) bool — 11.5 MB of
// resident input at N = 1200 — whose rows were constant over one test
// segment's nq observers by construction, because the caller built it by
// indexing the observer axis through `m_of_obs = repeat(arange(M), nq)` and
// the kernel already requires `obs_centers` to have exactly M*nq rows in that
// order. So the labels carry the same information at 8·(M + N) bytes, the
// blocked numpy fill's own spelling (its per-block mask has always been a
// slice of the same predicate), and the test segment's label is hoisted out
// of the source loop instead of being re-read from memory per pair.
struct GalerkinEkBlock {
    const double *src_a;
    const int64_t *g_obs;
    const int64_t *g_src;
    const double *gx;
    const double *gw;
    size_t n_gl;
};

// The FOLD destination for the fused far fill (momwire#356). Without one the
// fill allocates its own three (nnz, N) arrays and the caller differences them
// off its own triple afterwards, so any accelerated grounded block floors at
// two triples live. With one, the caller's arrays ARE the output buffers and
// each test segment's rows are folded on as
//
//     dst[e, n] += scale * value[e, n]
//
// with `scale` on the LEFT of the complex product — `sinusoidal.py`'s
// documented C2 convention, because complex multiply evaluates the imaginary
// part as x.re*y.im + x.im*y.re and swapping the operands reorders that sum
// and moves the last bit.
//
// The value folded on is the FULLY reduced entry, not a running partial: the
// test-quadrature accumulation runs into a per-test-segment scratch band and
// the fold happens once, after it. Accumulating `dst -= t_qt` node by node
// instead would be `((dst − t1) − t2) − …` where the caller's spelling is
// `dst − (t1 + t2 + …)`, which is a reassociation and not a fold.
//
// `scale` with a zero imaginary part takes a real path — `dst.re += s·v.re`,
// `dst.im += s·v.im`. At the fold's own scale, −1, that is bit-for-bit
// `np.subtract(dst, value, out=dst)`: IEEE addition of an exactly-negated
// operand is the subtraction, signed zeros included. The general complex path
// spells numpy's `np.multiply(scale, value)` term for term.
//
// A cancelled fill leaves the destination partly folded. That is the same
// contract the allocating path has — the caller drops what the kernel was
// building when `SolveAborted` comes out of it — one array further along.
struct GalerkinFoldBlock {
    nd::array_t<std::complex<double>> dst_const, dst_sin, dst_cos;
    double s_re, s_im;
};


std::tuple<nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>>
sinusoidal_galerkin_far_fill(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_centers,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_tangents,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_radius,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_centers,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_tangents,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_hh,
    double k,
    double eta,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_w,
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> w_entry,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> starts,
    uintptr_t cancel_flag = 0,
    const GalerkinFoldBlock *fold = nullptr,
    bool reference = false);

// The extended-kernel payload checked and gathered, exactly as
// `sinusoidal_galerkin_far_fill_ek` does before it fills: one radius and one
// group label per source segment (`n_src` of them), one label per test
// segment (`n_starts - 1`), and a delta rule of matching, non-zero length.
// The block points into the arrays, which must outlive it.
GalerkinEkBlock galerkin_ek_block(
    const nd::array_t<double, nd::array::c_style | nd::array::forcecast> &src_a,
    const nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> &group_obs,
    const nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> &group_src,
    const nd::array_t<double, nd::array::c_style | nd::array::forcecast> &ek_gx,
    const nd::array_t<double, nd::array::c_style | nd::array::forcecast> &ek_gw,
    nd::ssize_t n_src,
    nd::ssize_t n_starts);

std::tuple<nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>>
sinusoidal_galerkin_far_fill_ek(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_centers,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_tangents,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_radius,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_centers,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_tangents,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_hh,
    double k,
    double eta,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_w,
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> w_entry,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> starts,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_a,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> group_obs,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> group_src,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> ek_gx,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> ek_gw,
    uintptr_t cancel_flag = 0,
    const GalerkinFoldBlock *fold = nullptr);

std::tuple<nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>, nd::array_t<std::complex<double>>>
sinusoidal_galerkin_far_fill_cplx(
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_centers,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_tangents,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> obs_radius,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_centers,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_tangents,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> src_hh,
    std::complex<double> k,
    std::complex<double> eta,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_t,
    nd::array_t<double, nd::array::c_style | nd::array::forcecast> gl_w,
    nd::array_t<std::complex<double>, nd::array::c_style | nd::array::forcecast> w_entry,
    nd::array_t<int64_t, nd::array::c_style | nd::array::forcecast> starts,
    uintptr_t cancel_flag = 0,
    const GalerkinFoldBlock *fold = nullptr,
    bool reference = false);

// (staged, reference): how many reduced (EK-off) real fills, and how many
// complex fills, each body has served in this process (perf item 9,
// momwire#1224). The bodies' outputs are bit-identical, so this is the only
// way to see which one ran.
std::pair<unsigned long long, unsigned long long> sg_real_far_fill_calls();
std::pair<unsigned long long, unsigned long long> sg_cplx_far_fill_calls();

}  // namespace sinusoidal_galerkin
}  // namespace mw
