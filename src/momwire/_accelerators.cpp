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
#include <iostream>
#include <tuple>
#include <vector>

#include "_bspline_static_moments_inline.h"
// The extended-thin-wire static correction D_pq^EK (momwire#249's codegen,
// wired in by #270 unit 1). Pulls in the J header itself; the duplicate
// include above is harmless (`#pragma once`) and kept for legibility.
#include "_bspline_ek_moments_inline.h"

namespace py = pybind11;

// Ubuntu/glibc <cmath> headers don't carry `omp declare simd` markers for the
// libmvec routines, so GCC's auto-vectorizer can't substitute the vectorized
// `_ZGVdN4v_sin` / `_ZGVdN4v_cos` (AVX2, 4 doubles) inside an `omp simd` loop
// without these explicit declarations. The std::cos / std::sin overloads in
// <cmath> still resolve to these underlying extern-C symbols, so the rest of
// the file's calls pick up the simd-vectorized form for free once the linker
// has libmvec available (-lmvec in setup.py).
//
// Gated to GNU-compatible, non-MSVC compilers: this trick targets glibc's
// libmvec specifically. MSVC has no libmvec and would choke on redeclaring the
// CRT's cos/sin; there the sincos calls stay scalar/autovectorized. macOS is
// also excluded: Apple clang defines __GNUC__ but there is no libmvec on
// macOS (and arm64 has no AVX2 simdlen(4) form either), so the sincos calls
// stay scalar/NEON-autovectorized there too.
#if defined(__GNUC__) && !defined(_MSC_VER) && !defined(__APPLE__)
#pragma omp declare simd notinbranch simdlen(4)
extern "C" double cos(double);

#pragma omp declare simd notinbranch simdlen(4)
extern "C" double sin(double);
#endif

// Stringizing helpers so a token sequence can be turned into a _Pragma.
#define PYSIM_STR_(x) #x
#define PYSIM_PRAGMA_(x) _Pragma(PYSIM_STR_(x))

// The (i, j)-grid parallel loops below use `collapse(2)` for finer load
// balancing. We keep the Windows OpenMP usage conservative and drop it to plain
// outer-loop parallelism under MSVC — same results, only coarser scheduling
// across the grid. GCC (and MSVC's /openmp:llvm) keeps collapse(2).
#if defined(_MSC_VER)
#  define PYSIM_OMP_PARALLEL_FOR_COLLAPSE2 _Pragma("omp parallel for schedule(static)")
#else
#  define PYSIM_OMP_PARALLEL_FOR_COLLAPSE2 \
       _Pragma("omp parallel for collapse(2) schedule(static)")
#endif

// `omp simd` neutralization for MSVC. MSVC's /openmp:llvm (which we build with,
// for its collapse + unsigned-index support) rejects the `omp simd` directive
// outright, and /openmp:experimental silently drops simd `reduction` clauses —
// a correctness hazard. Turn the simd directives into no-ops under MSVC and let
// /arch:AVX2 autovectorize the inner loops as correct scalar-reduction code.
// GCC keeps real `omp simd` (bound to libmvec via the declare-simd block above).
// Reduction-clause commas are protected by the inner parens, so they pass as a
// single macro argument.
#if defined(_MSC_VER)
#  define PYSIM_OMP_SIMD(clauses)
#else
#  define PYSIM_OMP_SIMD(clauses) PYSIM_PRAGMA_(omp simd clauses)
#endif

// --------------------------------------------------------------------------
// Cooperative cancellation (Phase 2).
//
// Long kernels take a trailing `uintptr_t cancel_flag` (0 = no cancellation).
// It is the raw address of a CancelToken's int32 flag on the Python side. A
// monotonic 0 -> 1 write from any thread means "abort"; we read it with a
// `volatile` load per outer loop iteration (naturally-aligned int32, single
// writer -> safe in practice, and avoids the C++20 std::atomic_ref / Apple
// Clang libc++ support gap that would jeopardise the macOS wheel).
//
// An exception must never escape an OpenMP parallel region (UB), so we use the
// standard drain pattern: on cancel, set a shared atomic<bool> and `continue`
// so every remaining iteration becomes a no-op; the loop finishes normally and
// we throw AFTER it. AbortedError is registered as the Python exception
// `AcceleratorAborted`, which the `_accel.py` wrappers remap to SolveAborted.
struct AbortedError : std::exception {
    const char *what() const noexcept override { return "accelerator solve aborted"; }
};

#define PYSIM_CANCEL_SETUP(flag_addr)                                          \
    const volatile int32_t *pysim_cancel =                                     \
        reinterpret_cast<const volatile int32_t *>(flag_addr);                 \
    std::atomic<bool> pysim_aborted { false }

// Drain-poll. Place at the very top of a parallel loop body; `continue` targets
// the enclosing for-loop by design, so use only inside a braced loop body.
#define PYSIM_CANCEL_POLL()                                                    \
    if (pysim_aborted.load(std::memory_order_relaxed)) continue;              \
    if (pysim_cancel && *pysim_cancel) {                                       \
        pysim_aborted.store(true, std::memory_order_relaxed);                  \
        continue;                                                              \
    }

#define PYSIM_THROW_IF_ABORTED()                                               \
    if (pysim_aborted.load(std::memory_order_relaxed)) throw AbortedError {}

// Streaming swept reg-moment kernel for the B-spline Galerkin MoM.
//
// Computes, for every wavenumber k in k_array, the same-edge regularized
// smooth-kernel polynomial moment block
//
//   J[k, p, P, i, j] = sum_{q,r} wu_pow[p, i, q] * Greg(k; iq, jr)
//                                 * wu_pow[P, j, r]
//
// with Greg = (exp(-j k R) - 1) / (4 pi R) on the precomputed pair-distance
// table R (shape (N*n_qp, N*n_qp)) and the weight-folded local-coordinate
// powers wu_pow (shape (n_d, N, n_qp)) produced by the Python
// _seg_seg_reg_geometry. This is the streaming C++ replacement for numpy's
// _seg_seg_reg_moments_from_geometry_swept einsum "piq,kiqjr,Pjr->kpPij": it
// evaluates exp(-jkR) once per (iq, jr, k) and accumulates straight into the
// (n_d x n_d) moment block, never materializing the (n_k, N*n_qp, N*n_qp)
// phase intermediate the numpy path builds (and chunks at 256 MB). Output is
// the identical (n_k, n_d, n_d, N, N) tensor.
//
// Loop order (kk, i) collapse-parallel, inner j: the o(kk, p, P, i, j) writes
// run contiguously over the trailing (i, j) axes, and the (N*n_qp)^2 R table
// stays L2-resident as it is re-read across the k axis.
//
// THE EXTENDED-KERNEL TWIN (momwire#270 unit 1)
// ---------------------------------------------
// `EK == true` swaps Greg for the extended kernel's smooth remainder,
// `_bspline_kernels._ek_reg_kernel`:
//
//     Greg_ek = [ (e^{-jkR} − 1)·fac + extra ] / (4 π R)
//     fac     = 1 + T1·C2 − T2·C1        (Eq 89's coaxial factor of R)
//     extra   = T1·(C2 − 3) − T2·(C1 − 1) = fac − fac_static
//     C1 = 1 + jkR,  C2 = 3·C1 − (kR)²,  T1 = a⁴/(4R⁴),  T2 = a²/(2R²)
//
// with `a_ek` the EK radius (`_ek_radius(ek, geo["a"])` on the Python side).
// The static half of the same coaxial factor is carried in closed form by
// D_ek_pq / seg_seg_static_moments_bspline_uniform_ek, so what is left here
// really is a bounded remainder — the same class as the reduced kernel's
// (e^{-jkR}−1)/R → −jk, resolved by the caller's existing n_qp rule.
//
// The arithmetic is a LITERAL transcription of the numpy spelling, in the
// same multi-step order (momwire#205): every intermediate below has a named
// counterpart in `_ek_factor` / `_ek_reg_extra` / `_ek_reg_kernel` and the
// pointwise kernel comes out bit-identical to numpy's on this box, including
// the final `num / (4 π R)` — numpy's complex-by-real divide is Smith's
// algorithm with a zero divisor imaginary part, i.e. a multiply by
// `1.0 / (4 π R)`, which is what `scl` below is. (The MOMENT still differs in
// the last bits: the (q, r) reduction order is not the einsum's. Gates are
// relative-tolerance, not bit equality.)
//
// T1, T2 and `scl` are functions of R and a_ek alone, so they hoist out of
// the k loop next to `inv_R_4pi`; only C1/C2 and the phase are per-k.
template <bool EK>
static py::array_t<std::complex<double>>
seg_seg_reg_moments_bspline_swept_impl(
    py::array_t<double, py::array::c_style | py::array::forcecast> R,
    py::array_t<double, py::array::c_style | py::array::forcecast> wu_pow,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array,
    double a_ek
) {
    auto Rr = R.unchecked<2>();
    auto wu = wu_pow.unchecked<3>();
    auto ka = k_array.unchecked<1>();

    size_t n_d  = wu.shape(0);
    size_t N    = wu.shape(1);
    size_t n_qp = wu.shape(2);
    size_t n_k  = ka.shape(0);

    if (Rr.shape(0) != (py::ssize_t)(N * n_qp) ||
        Rr.shape(1) != (py::ssize_t)(N * n_qp)) {
        throw std::runtime_error(
            "R must be (N*n_qp, N*n_qp) consistent with wu_pow (n_d, N, n_qp)");
    }
    if (n_d > 8) {
        throw std::runtime_error("n_d too large (max_d must be <= 7)");
    }

    size_t n_pairs = n_qp * n_qp;
    if (n_pairs > 64) {
        throw std::runtime_error("n_qp too large (n_qp^2 must be <= 64)");
    }

    const double inv_4pi = 1.0 / (4.0 * M_PI);
    // numpy divides by `4 * np.pi * R`, associated as `(4*np.pi) * R`.
    const double four_pi = 4.0 * M_PI;
    const double a2_ek = a_ek * a_ek;
    const double a4_ek = a2_ek * a2_ek;

    py::array_t<std::complex<double>> out({n_k, n_d, n_d, N, N});
    auto o = out.mutable_unchecked<5>();

    // Phase 0: release the GIL for the heavy compute region below.
    py::gil_scoped_release release;

    // (i, j) parallel; the per-(i, j) R block is hoisted out of the k loop and
    // the inner sincos runs as `omp simd` so GCC substitutes the libmvec
    // vectorized cos/sin. The
    // n_d^2 moment accumulation reuses each (q, r) Greg value across all
    // polynomial orders — the streaming property that avoids the numpy
    // einsum's (n_k, N*n_qp, N*n_qp) phase intermediate.
    PYSIM_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            alignas(32) double R[64];
            alignas(32) double inv_R_4pi[64];
            alignas(32) double phases[64];
            alignas(32) double cos_phases[64];
            alignas(32) double sin_phases[64];
            alignas(32) double Gre[64];
            alignas(32) double Gim[64];
            alignas(32) double t1v[64];
            alignas(32) double t2v[64];
            alignas(32) double scl[64];

            for (size_t q = 0; q < n_qp; q++) {
                size_t iq = i * n_qp + q;
                for (size_t r = 0; r < n_qp; r++) {
                    R[q * n_qp + r] = Rr(iq, j * n_qp + r);
                }
            }
            PYSIM_OMP_SIMD()
            for (size_t qr = 0; qr < n_pairs; qr++) {
                inv_R_4pi[qr] = inv_4pi / R[qr];
            }
            if (EK) {
                // `_ek_factor`'s r2/r4 and T1/T2, plus the reciprocal of the
                // final 4πR divisor. All k-independent.
                PYSIM_OMP_SIMD()
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    double r2 = R[qr] * R[qr];
                    double r4 = r2 * r2;
                    t1v[qr] = 0.25 * a4_ek / r4;
                    t2v[qr] = 0.5 * a2_ek / r2;
                    scl[qr] = 1.0 / (four_pi * R[qr]);
                }
            }

            for (size_t kk = 0; kk < n_k; kk++) {
                double k = ka(kk);
                PYSIM_OMP_SIMD()
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    phases[qr] = -k * R[qr];
                }
                PYSIM_OMP_SIMD()
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    cos_phases[qr] = std::cos(phases[qr]);
                }
                PYSIM_OMP_SIMD()
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    sin_phases[qr] = std::sin(phases[qr]);
                }
                if (EK) {
                    PYSIM_OMP_SIMD()
                    for (size_t qr = 0; qr < n_pairs; qr++) {
                        double kr = k * R[qr];
                        double kr2 = kr * kr;
                        double t1 = t1v[qr];
                        double t2 = t2v[qr];
                        // C1 = 1 + jkR;  C2 = 3·C1 − (kR)².
                        double c1r = 1.0;
                        double c1i = kr;
                        double c2r = 3.0 * c1r - kr2;
                        double c2i = 3.0 * c1i;
                        // fac = T1·C2;  fac -= T2·C1;  fac += 1.
                        double facr = t1 * c2r;
                        double faci = t1 * c2i;
                        facr = facr - t2 * c1r;
                        faci = faci - t2 * c1i;
                        facr = facr + 1.0;
                        // extra = T1·(3jkR − (kR)²);  extra -= T2·(jkR).
                        double exr = t1 * (0.0 - kr2);
                        double exi = t1 * (3.0 * kr);
                        exi = exi - t2 * kr;
                        // phase = e^{-jkR} − 1 = (cos(-kR) − 1) + j sin(-kR).
                        double pr = cos_phases[qr] - 1.0;
                        double pim = sin_phases[qr];
                        // num = phase·fac;  num += extra;  num /= 4πR.
                        double numr = pr * facr - pim * faci;
                        double numi = pr * faci + pim * facr;
                        numr = numr + exr;
                        numi = numi + exi;
                        Gre[qr] = numr * scl[qr];
                        Gim[qr] = numi * scl[qr];
                    }
                } else {
                PYSIM_OMP_SIMD()
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    // exp(-j k R) - 1 = (cos(-kR) - 1) + j sin(-kR)
                    Gre[qr] = (cos_phases[qr] - 1.0) * inv_R_4pi[qr];
                    Gim[qr] = sin_phases[qr] * inv_R_4pi[qr];
                }
                }

                // J[p,P,i,j] = sum_{q,r} wu[p,i,q] Greg[q,r] wu[P,j,r].
                for (size_t p = 0; p < n_d; p++) {
                    for (size_t P = 0; P < n_d; P++) {
                        double mre = 0.0, mim = 0.0;
                        for (size_t q = 0; q < n_qp; q++) {
                            double wp = wu(p, i, q);
                            for (size_t r = 0; r < n_qp; r++) {
                                double w = wp * wu(P, j, r);
                                size_t qr = q * n_qp + r;
                                mre += w * Gre[qr];
                                mim += w * Gim[qr];
                            }
                        }
                        o(kk, p, P, i, j) = std::complex<double>(mre, mim);
                    }
                }
            }
        }
    }
    return out;
}

static py::array_t<std::complex<double>>
seg_seg_reg_moments_bspline_swept(
    py::array_t<double, py::array::c_style | py::array::forcecast> R,
    py::array_t<double, py::array::c_style | py::array::forcecast> wu_pow,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array
) {
    return seg_seg_reg_moments_bspline_swept_impl<false>(R, wu_pow, k_array, 0.0);
}

static py::array_t<std::complex<double>>
seg_seg_reg_moments_bspline_swept_ek(
    py::array_t<double, py::array::c_style | py::array::forcecast> R,
    py::array_t<double, py::array::c_style | py::array::forcecast> wu_pow,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array,
    double a_ek
) {
    return seg_seg_reg_moments_bspline_swept_impl<true>(R, wu_pow, k_array, a_ek);
}


// Templated B-spline moment-integral kernel.
//
// For each (i, j) segment pair, compute the (D+1)^2 polynomial moments
//   J[p, P, i, j] = sum_{q, r} wi[q] * ui[q]^p * wj[r] * uj[r]^P * G(R_qr)
// where
//   R_qr = sqrt(|pos_i(t_q) - pos_j(t_r)|^2 + a_squared)
//   G(R) = exp(-j*k*R) / (4*pi*R)
//   ui[q] = t_q * len_i,  uj[r] = t_r * len_j  (local arc lengths)
//
// Used by BSplineSolver._build_J_blocks for the all-pairs off-edge piece
// (same a^2 wire-radius regularization handles touching segments at kinks
// and at junctions). Single-k for now (BSplineSolver hasn't grown a swept
// path yet); add a batched k_array variant later if/when needed.
//
// Template parameter D = B-spline degree (1 or 2 currently — explicit
// instantiations below). Hardcoding D as a compile-time constant lets the
// compiler fully unroll the (D+1)^2 polynomial-moment inner loop, getting
// the same scalar-unrolled tight assembly the retired triangular solver's
// hand-rolled s00 / s10 / s01 / s11 accumulators achieved.
//
// n_qp <= 8 assumed (n_qp^2 <= 64 scratch buffer size).
template<int D>
static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_kernel(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    double k,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w
) {
    static constexpr int NM = D + 1;          // moments per axis
    static constexpr int NMM = NM * NM;       // total moments

    auto sli = seg_l_i.unchecked<2>();
    auto sri = seg_r_i.unchecked<2>();
    auto slj = seg_l_j.unchecked<2>();
    auto srj = seg_r_j.unchecked<2>();
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();

    if (sli.shape(1) != 3 || sri.shape(1) != 3 ||
        slj.shape(1) != 3 || srj.shape(1) != 3) {
        throw std::runtime_error("segment endpoint arrays must have shape (N, 3)");
    }
    if (sli.shape(0) != sri.shape(0) || slj.shape(0) != srj.shape(0)) {
        throw std::runtime_error("seg_l and seg_r must have matching N");
    }
    if (glt.shape(0) != glw.shape(0)) {
        throw std::runtime_error("gl_t and gl_w must have matching length");
    }
    size_t n_qp_in = glt.shape(0);
    if (n_qp_in > 8) {
        throw std::runtime_error("n_qp > 8 not supported (scratch buffer size)");
    }

    size_t N_i = sli.shape(0);
    size_t N_j = slj.shape(0);
    size_t n_qp = n_qp_in;

    py::array_t<std::complex<double>> J({(size_t)NM, (size_t)NM, N_i, N_j});
    auto j_view = J.mutable_unchecked<4>();

    // Phase 0: release the GIL for the heavy compute region below.
    py::gil_scoped_release release;

    const double inv_4pi = 1.0 / (4.0 * M_PI);

    // Per-segment quadrature-point positions and lengths -- k-independent,
    // computed once outside the parallel region.
    std::vector<double> pos_i(N_i * n_qp * 3);
    std::vector<double> pos_j(N_j * n_qp * 3);
    std::vector<double> len_i(N_i);
    std::vector<double> len_j(N_j);
    for (size_t i = 0; i < N_i; i++) {
        double dx = sri(i,0) - sli(i,0);
        double dy = sri(i,1) - sli(i,1);
        double dz = sri(i,2) - sli(i,2);
        len_i[i] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (size_t q = 0; q < n_qp; q++) {
            double t = glt(q);
            pos_i[(i*n_qp + q)*3 + 0] = (1.0 - t) * sli(i,0) + t * sri(i,0);
            pos_i[(i*n_qp + q)*3 + 1] = (1.0 - t) * sli(i,1) + t * sri(i,1);
            pos_i[(i*n_qp + q)*3 + 2] = (1.0 - t) * sli(i,2) + t * sri(i,2);
        }
    }
    for (size_t j = 0; j < N_j; j++) {
        double dx = srj(j,0) - slj(j,0);
        double dy = srj(j,1) - slj(j,1);
        double dz = srj(j,2) - slj(j,2);
        len_j[j] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (size_t r = 0; r < n_qp; r++) {
            double t = glt(r);
            pos_j[(j*n_qp + r)*3 + 0] = (1.0 - t) * slj(j,0) + t * srj(j,0);
            pos_j[(j*n_qp + r)*3 + 1] = (1.0 - t) * slj(j,1) + t * srj(j,1);
            pos_j[(j*n_qp + r)*3 + 2] = (1.0 - t) * slj(j,2) + t * srj(j,2);
        }
    }

    PYSIM_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t i = 0; i < N_i; i++) {
        for (size_t j = 0; j < N_j; j++) {
            alignas(32) double R[64];
            alignas(32) double inv_R_4pi[64];
            alignas(32) double phases[64];
            alignas(32) double cos_phases[64];
            alignas(32) double sin_phases[64];
            alignas(32) double G_re[64], G_im[64];
            // wuwu[pP, qr]: precomputed wi[q]*ui[q]^p * wj[r]*uj[r]^P,
            // flattened with pP = p*NM + P. For D=2: NMM*64 = 576 doubles = 4.5KB,
            // fits comfortably in L1.
            alignas(32) double wuwu[NMM * 64];

            const double *pi = &pos_i[i * n_qp * 3];
            const double *pj = &pos_j[j * n_qp * 3];
            for (size_t q = 0; q < n_qp; q++) {
                double pix = pi[q*3 + 0];
                double piy = pi[q*3 + 1];
                double piz = pi[q*3 + 2];
                for (size_t r = 0; r < n_qp; r++) {
                    double dx = pix - pj[r*3 + 0];
                    double dy = piy - pj[r*3 + 1];
                    double dz = piz - pj[r*3 + 2];
                    R[q*n_qp + r] = std::sqrt(dx*dx + dy*dy + dz*dz + a_squared);
                }
            }

            double Li = len_i[i];
            double Lj = len_j[j];
            size_t n_pairs = n_qp * n_qp;

            // Build wuwu[pP, qr]: pP indexes the moment (p*NM + P), qr the
            // quadrature pair (q*n_qp + r). The NM is a template constant so
            // ui_pow[p] / uj_pow[P] arrays unroll.
            for (size_t q = 0; q < n_qp; q++) {
                double wi = glw(q) * Li;
                double ui = glt(q) * Li;
                double ui_pow[NM];
                ui_pow[0] = 1.0;
                for (int p = 1; p < NM; p++) ui_pow[p] = ui_pow[p-1] * ui;
                for (size_t r = 0; r < n_qp; r++) {
                    double wj = glw(r) * Lj;
                    double uj = glt(r) * Lj;
                    double uj_pow[NM];
                    uj_pow[0] = 1.0;
                    for (int P = 1; P < NM; P++) uj_pow[P] = uj_pow[P-1] * uj;
                    double wij = wi * wj;
                    size_t qr = q * n_qp + r;
                    for (int p = 0; p < NM; p++) {
                        for (int P = 0; P < NM; P++) {
                            wuwu[(p * NM + P) * n_pairs + qr] = wij * ui_pow[p] * uj_pow[P];
                        }
                    }
                }
            }

            // Stage 1: phases = -k * R, then sincos via libmvec.
            PYSIM_OMP_SIMD()
            for (size_t qr = 0; qr < n_pairs; qr++) {
                phases[qr] = -k * R[qr];
            }
            PYSIM_OMP_SIMD()
            for (size_t qr = 0; qr < n_pairs; qr++) {
                cos_phases[qr] = std::cos(phases[qr]);
            }
            PYSIM_OMP_SIMD()
            for (size_t qr = 0; qr < n_pairs; qr++) {
                sin_phases[qr] = std::sin(phases[qr]);
            }
            PYSIM_OMP_SIMD()
            for (size_t qr = 0; qr < n_pairs; qr++) {
                inv_R_4pi[qr] = inv_4pi / R[qr];
                G_re[qr] = cos_phases[qr] * inv_R_4pi[qr];
                G_im[qr] = sin_phases[qr] * inv_R_4pi[qr];
            }

            // Stage 2: NMM moment reductions, each a vectorizable sum over qr.
            for (int pP = 0; pP < NMM; pP++) {
                double sr = 0.0, si = 0.0;
                const double *w_row = &wuwu[pP * n_pairs];
                PYSIM_OMP_SIMD(reduction(+:sr,si))
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    sr += w_row[qr] * G_re[qr];
                    si += w_row[qr] * G_im[qr];
                }
                j_view(pP / NM, pP % NM, i, j) = std::complex<double>(sr, si);
            }
        }
    }

    return J;
}

// Batched (swept-k) variant of seg_seg_full_moments_bspline_kernel.
//
// The per-(i, j) geometry — segment quadrature positions, the (q, r) distance
// table R, the 1/(4 pi R) factor, and the weight-folded moment weights wuwu —
// is all k-independent, so it is built once per (i, j) and reused across the
// whole k_array; only the exp(-jkR) phase varies per frequency. This is the
// off-edge analog of the same-edge streaming reg kernel above: it lets
// BSplineSolver.compute_impedance_swept build the off-edge moments in one call
// for the whole sweep instead of one single-k call per frequency.
//
// Output: (n_k, NM, NM, N_i, N_j) complex. Memory note: this materializes the
// full off-edge moment tensor for every frequency at once (n_k * NM^2 * N^2
// complex); the UI sweep chunker bounds the sweep width when N is large.
//
// n_qp <= 8 assumed (n_qp^2 <= 64 scratch buffer size).
template<int D>
static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_swept_kernel(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w
) {
    static constexpr int NM = D + 1;
    static constexpr int NMM = NM * NM;

    auto sli = seg_l_i.unchecked<2>();
    auto sri = seg_r_i.unchecked<2>();
    auto slj = seg_l_j.unchecked<2>();
    auto srj = seg_r_j.unchecked<2>();
    auto ka  = k_array.unchecked<1>();
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();

    if (sli.shape(1) != 3 || sri.shape(1) != 3 ||
        slj.shape(1) != 3 || srj.shape(1) != 3) {
        throw std::runtime_error("segment endpoint arrays must have shape (N, 3)");
    }
    if (sli.shape(0) != sri.shape(0) || slj.shape(0) != srj.shape(0)) {
        throw std::runtime_error("seg_l and seg_r must have matching N");
    }
    if (glt.shape(0) != glw.shape(0)) {
        throw std::runtime_error("gl_t and gl_w must have matching length");
    }
    size_t n_qp = glt.shape(0);
    if (n_qp > 8) {
        throw std::runtime_error("n_qp > 8 not supported (scratch buffer size)");
    }

    size_t N_i = sli.shape(0);
    size_t N_j = slj.shape(0);
    size_t n_k = ka.shape(0);

    py::array_t<std::complex<double>> J({n_k, (size_t)NM, (size_t)NM, N_i, N_j});
    auto j_view = J.mutable_unchecked<5>();

    // Release the GIL for the geometry precompute + heavy fill below.
    py::gil_scoped_release release;

    const double inv_4pi = 1.0 / (4.0 * M_PI);

    // k-independent per-segment quadrature positions and lengths.
    std::vector<double> pos_i(N_i * n_qp * 3);
    std::vector<double> pos_j(N_j * n_qp * 3);
    std::vector<double> len_i(N_i);
    std::vector<double> len_j(N_j);
    for (size_t i = 0; i < N_i; i++) {
        double dx = sri(i,0) - sli(i,0);
        double dy = sri(i,1) - sli(i,1);
        double dz = sri(i,2) - sli(i,2);
        len_i[i] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (size_t q = 0; q < n_qp; q++) {
            double t = glt(q);
            pos_i[(i*n_qp + q)*3 + 0] = (1.0 - t) * sli(i,0) + t * sri(i,0);
            pos_i[(i*n_qp + q)*3 + 1] = (1.0 - t) * sli(i,1) + t * sri(i,1);
            pos_i[(i*n_qp + q)*3 + 2] = (1.0 - t) * sli(i,2) + t * sri(i,2);
        }
    }
    for (size_t j = 0; j < N_j; j++) {
        double dx = srj(j,0) - slj(j,0);
        double dy = srj(j,1) - slj(j,1);
        double dz = srj(j,2) - slj(j,2);
        len_j[j] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (size_t r = 0; r < n_qp; r++) {
            double t = glt(r);
            pos_j[(j*n_qp + r)*3 + 0] = (1.0 - t) * slj(j,0) + t * srj(j,0);
            pos_j[(j*n_qp + r)*3 + 1] = (1.0 - t) * slj(j,1) + t * srj(j,1);
            pos_j[(j*n_qp + r)*3 + 2] = (1.0 - t) * slj(j,2) + t * srj(j,2);
        }
    }

    PYSIM_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t i = 0; i < N_i; i++) {
        for (size_t j = 0; j < N_j; j++) {
            alignas(32) double R[64];
            alignas(32) double inv_R_4pi[64];
            alignas(32) double phases[64];
            alignas(32) double cos_phases[64];
            alignas(32) double sin_phases[64];
            alignas(32) double G_re[64], G_im[64];
            alignas(32) double wuwu[NMM * 64];

            size_t n_pairs = n_qp * n_qp;
            const double *pi = &pos_i[i * n_qp * 3];
            const double *pj = &pos_j[j * n_qp * 3];

            // k-independent: R table, 1/(4 pi R), and the moment weights.
            for (size_t q = 0; q < n_qp; q++) {
                double pix = pi[q*3 + 0], piy = pi[q*3 + 1], piz = pi[q*3 + 2];
                for (size_t r = 0; r < n_qp; r++) {
                    double dx = pix - pj[r*3 + 0];
                    double dy = piy - pj[r*3 + 1];
                    double dz = piz - pj[r*3 + 2];
                    R[q*n_qp + r] = std::sqrt(dx*dx + dy*dy + dz*dz + a_squared);
                }
            }
            PYSIM_OMP_SIMD()
            for (size_t qr = 0; qr < n_pairs; qr++) {
                inv_R_4pi[qr] = inv_4pi / R[qr];
            }

            double Li = len_i[i];
            double Lj = len_j[j];
            for (size_t q = 0; q < n_qp; q++) {
                double wi = glw(q) * Li;
                double ui = glt(q) * Li;
                double ui_pow[NM];
                ui_pow[0] = 1.0;
                for (int p = 1; p < NM; p++) ui_pow[p] = ui_pow[p-1] * ui;
                for (size_t r = 0; r < n_qp; r++) {
                    double wj = glw(r) * Lj;
                    double uj = glt(r) * Lj;
                    double uj_pow[NM];
                    uj_pow[0] = 1.0;
                    for (int P = 1; P < NM; P++) uj_pow[P] = uj_pow[P-1] * uj;
                    double wij = wi * wj;
                    size_t qr = q * n_qp + r;
                    for (int p = 0; p < NM; p++) {
                        for (int P = 0; P < NM; P++) {
                            wuwu[(p * NM + P) * n_pairs + qr] = wij * ui_pow[p] * uj_pow[P];
                        }
                    }
                }
            }

            // Per-k: only the exp(-jkR) phase changes.
            for (size_t kk = 0; kk < n_k; kk++) {
                double k = ka(kk);
                PYSIM_OMP_SIMD()
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    phases[qr] = -k * R[qr];
                }
                PYSIM_OMP_SIMD()
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    cos_phases[qr] = std::cos(phases[qr]);
                }
                PYSIM_OMP_SIMD()
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    sin_phases[qr] = std::sin(phases[qr]);
                }
                PYSIM_OMP_SIMD()
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    G_re[qr] = cos_phases[qr] * inv_R_4pi[qr];
                    G_im[qr] = sin_phases[qr] * inv_R_4pi[qr];
                }
                for (int pP = 0; pP < NMM; pP++) {
                    double sr = 0.0, si = 0.0;
                    const double *w_row = &wuwu[pP * n_pairs];
                    PYSIM_OMP_SIMD(reduction(+:sr,si))
                    for (size_t qr = 0; qr < n_pairs; qr++) {
                        sr += w_row[qr] * G_re[qr];
                        si += w_row[qr] * G_im[qr];
                    }
                    j_view(kk, pP / NM, pP % NM, i, j) =
                        std::complex<double>(sr, si);
                }
            }
        }
    }

    return J;
}


// THE OFF-EDGE EXTENDED-KERNEL TWINS (momwire#270 unit 2)
// ---------------------------------------------------------
// Unit 1 added the C++ twins of the SAME-EDGE kernels, where a block is
// eligible in its entirety (one edge, one wire, one radius). The off-edge
// fill has no such luxury: eligibility is a property of the PAIR,
// `group_i[i] == group_j[j]` (momwire#249 §4.1), evaluated once per (i, j)
// segment pair and applied to every quadrature sub-pair inside it — the
// numpy reference (`_seg_seg_full_moments_offedge`'s `ek is not None`
// branch) builds the (N_i, N_j) mask once and broadcasts it over the
// (n_qp, n_qp) quadrature axes with `mask[:, None, :, None]`, so a pair's
// eligibility does not vary with (q, r) even though R does.
//
// Unlike the same-edge reg kernel (which starts from exp(-jkR) - 1), the
// off-edge kernel is the FULL G = exp(-jkR)/(4 pi R); numpy's spelling is
//
//     G = exp(-jkR) / (4 pi R)
//     if eligible: G = G * fac(R, a_ek, k)
//
// with `fac` the same NEC Eq 89 coaxial factor as `_ek_factor` / the
// same-edge twin's `fac` — literally the same multi-step spelling
// (T1, T2, C1, C2 in that order), transcribed again here rather than
// shared, because the eligibility gate sits between the two kernels' loop
// structures in a way a shared template parameter would only obscure.
//
// `a_ek` is the plain (unsquared) EK radius, exactly like the same-edge
// twins' own `a_ek` argument — `a_squared` stays the (already squared)
// observer-row regularization radius the reduced kernel already takes.
// Eligible pairs have equal radii by construction, so on this box the two
// agree on every pair the mask lets through; `a_ek` is kept separate so the
// C++ mirrors `_ek_radius(ek, a)` on the Python side rather than assuming
// it (unit 1's same rationale for `D_ek_dispatch`/`seg_seg_reg_..._ek`).
//
// Written as its own function rather than folded into
// `seg_seg_full_moments_bspline_kernel<D>` via a `template<bool EK>` (the
// same-edge kernels' choice): the reduced off-edge kernel's entry point and
// arithmetic stay completely untouched — zero lines of the frozen path
// change — at the cost of the geometry precompute being written twice.
template<int D>
static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_kernel_ek(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    double k,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_i,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_j,
    double a_ek
) {
    static constexpr int NM = D + 1;
    static constexpr int NMM = NM * NM;

    auto sli = seg_l_i.unchecked<2>();
    auto sri = seg_r_i.unchecked<2>();
    auto slj = seg_l_j.unchecked<2>();
    auto srj = seg_r_j.unchecked<2>();
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();
    auto gi_v = group_i.unchecked<1>();
    auto gj_v = group_j.unchecked<1>();

    if (sli.shape(1) != 3 || sri.shape(1) != 3 ||
        slj.shape(1) != 3 || srj.shape(1) != 3) {
        throw std::runtime_error("segment endpoint arrays must have shape (N, 3)");
    }
    if (sli.shape(0) != sri.shape(0) || slj.shape(0) != srj.shape(0)) {
        throw std::runtime_error("seg_l and seg_r must have matching N");
    }
    if (glt.shape(0) != glw.shape(0)) {
        throw std::runtime_error("gl_t and gl_w must have matching length");
    }
    size_t n_qp_in = glt.shape(0);
    if (n_qp_in > 8) {
        throw std::runtime_error("n_qp > 8 not supported (scratch buffer size)");
    }

    size_t N_i = sli.shape(0);
    size_t N_j = slj.shape(0);
    size_t n_qp = n_qp_in;
    if ((size_t)group_i.shape(0) != N_i || (size_t)group_j.shape(0) != N_j) {
        throw std::runtime_error("group_i/group_j must match N_i/N_j");
    }

    py::array_t<std::complex<double>> J({(size_t)NM, (size_t)NM, N_i, N_j});
    auto j_view = J.mutable_unchecked<4>();

    // Phase 0: release the GIL for the heavy compute region below.
    py::gil_scoped_release release;

    const double inv_4pi = 1.0 / (4.0 * M_PI);
    const double a2_ek = a_ek * a_ek;
    const double a4_ek = a2_ek * a2_ek;

    std::vector<double> pos_i(N_i * n_qp * 3);
    std::vector<double> pos_j(N_j * n_qp * 3);
    std::vector<double> len_i(N_i);
    std::vector<double> len_j(N_j);
    for (size_t i = 0; i < N_i; i++) {
        double dx = sri(i,0) - sli(i,0);
        double dy = sri(i,1) - sli(i,1);
        double dz = sri(i,2) - sli(i,2);
        len_i[i] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (size_t q = 0; q < n_qp; q++) {
            double t = glt(q);
            pos_i[(i*n_qp + q)*3 + 0] = (1.0 - t) * sli(i,0) + t * sri(i,0);
            pos_i[(i*n_qp + q)*3 + 1] = (1.0 - t) * sli(i,1) + t * sri(i,1);
            pos_i[(i*n_qp + q)*3 + 2] = (1.0 - t) * sli(i,2) + t * sri(i,2);
        }
    }
    for (size_t j = 0; j < N_j; j++) {
        double dx = srj(j,0) - slj(j,0);
        double dy = srj(j,1) - slj(j,1);
        double dz = srj(j,2) - slj(j,2);
        len_j[j] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (size_t r = 0; r < n_qp; r++) {
            double t = glt(r);
            pos_j[(j*n_qp + r)*3 + 0] = (1.0 - t) * slj(j,0) + t * srj(j,0);
            pos_j[(j*n_qp + r)*3 + 1] = (1.0 - t) * slj(j,1) + t * srj(j,1);
            pos_j[(j*n_qp + r)*3 + 2] = (1.0 - t) * slj(j,2) + t * srj(j,2);
        }
    }

    PYSIM_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t i = 0; i < N_i; i++) {
        for (size_t j = 0; j < N_j; j++) {
            alignas(32) double R[64];
            alignas(32) double inv_R_4pi[64];
            alignas(32) double phases[64];
            alignas(32) double cos_phases[64];
            alignas(32) double sin_phases[64];
            alignas(32) double G_re[64], G_im[64];
            alignas(32) double wuwu[NMM * 64];

            const double *pi = &pos_i[i * n_qp * 3];
            const double *pj = &pos_j[j * n_qp * 3];
            for (size_t q = 0; q < n_qp; q++) {
                double pix = pi[q*3 + 0];
                double piy = pi[q*3 + 1];
                double piz = pi[q*3 + 2];
                for (size_t r = 0; r < n_qp; r++) {
                    double dx = pix - pj[r*3 + 0];
                    double dy = piy - pj[r*3 + 1];
                    double dz = piz - pj[r*3 + 2];
                    R[q*n_qp + r] = std::sqrt(dx*dx + dy*dy + dz*dz + a_squared);
                }
            }

            double Li = len_i[i];
            double Lj = len_j[j];
            size_t n_pairs = n_qp * n_qp;

            for (size_t q = 0; q < n_qp; q++) {
                double wi = glw(q) * Li;
                double ui = glt(q) * Li;
                double ui_pow[NM];
                ui_pow[0] = 1.0;
                for (int p = 1; p < NM; p++) ui_pow[p] = ui_pow[p-1] * ui;
                for (size_t r = 0; r < n_qp; r++) {
                    double wj = glw(r) * Lj;
                    double uj = glt(r) * Lj;
                    double uj_pow[NM];
                    uj_pow[0] = 1.0;
                    for (int P = 1; P < NM; P++) uj_pow[P] = uj_pow[P-1] * uj;
                    double wij = wi * wj;
                    size_t qr = q * n_qp + r;
                    for (int p = 0; p < NM; p++) {
                        for (int P = 0; P < NM; P++) {
                            wuwu[(p * NM + P) * n_pairs + qr] = wij * ui_pow[p] * uj_pow[P];
                        }
                    }
                }
            }

            PYSIM_OMP_SIMD()
            for (size_t qr = 0; qr < n_pairs; qr++) {
                phases[qr] = -k * R[qr];
            }
            PYSIM_OMP_SIMD()
            for (size_t qr = 0; qr < n_pairs; qr++) {
                cos_phases[qr] = std::cos(phases[qr]);
            }
            PYSIM_OMP_SIMD()
            for (size_t qr = 0; qr < n_pairs; qr++) {
                sin_phases[qr] = std::sin(phases[qr]);
            }
            PYSIM_OMP_SIMD()
            for (size_t qr = 0; qr < n_pairs; qr++) {
                inv_R_4pi[qr] = inv_4pi / R[qr];
                G_re[qr] = cos_phases[qr] * inv_R_4pi[qr];
                G_im[qr] = sin_phases[qr] * inv_R_4pi[qr];
            }

            // Eligibility is a property of the (i, j) SEGMENT pair, not of
            // the quadrature sub-pair (numpy's `mask[:, None, :, None]`
            // broadcast) — one branch here serves every (q, r) below.
            bool eligible = (gi_v(i) == gj_v(j)) && (gi_v(i) >= 0);
            if (eligible) {
                // `_ek_factor`'s spelling, term by term: T1, T2, C1, C2,
                // fac = T1*C2 - T2*C1 + 1, then G *= fac (complex).
                PYSIM_OMP_SIMD()
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    double Rq = R[qr];
                    double r2 = Rq * Rq;
                    double r4 = r2 * r2;
                    double kr = k * Rq;
                    double kr2 = kr * kr;
                    double t1 = 0.25 * a4_ek / r4;
                    double t2 = 0.5 * a2_ek / r2;
                    double c1r = 1.0;
                    double c1i = kr;
                    double c2r = 3.0 * c1r - kr2;
                    double c2i = 3.0 * c1i;
                    double facr = t1 * c2r;
                    double faci = t1 * c2i;
                    facr = facr - t2 * c1r;
                    faci = faci - t2 * c1i;
                    facr = facr + 1.0;
                    double gre = G_re[qr];
                    double gim = G_im[qr];
                    G_re[qr] = gre * facr - gim * faci;
                    G_im[qr] = gre * faci + gim * facr;
                }
            }

            for (int pP = 0; pP < NMM; pP++) {
                double sr = 0.0, si = 0.0;
                const double *w_row = &wuwu[pP * n_pairs];
                PYSIM_OMP_SIMD(reduction(+:sr,si))
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    sr += w_row[qr] * G_re[qr];
                    si += w_row[qr] * G_im[qr];
                }
                j_view(pP / NM, pP % NM, i, j) = std::complex<double>(sr, si);
            }
        }
    }

    return J;
}

// Swept-k (batched) variant of seg_seg_full_moments_bspline_kernel_ek.
//
// Same relationship to seg_seg_full_moments_bspline_kernel_ek as the
// reduced swept kernel has to the reduced single-k one: the per-(i, j)
// geometry (positions, R, wuwu weights) and, under EK, the k-independent
// T1/T2 halves of `fac` are hoisted out of the k loop and reused across the
// whole k_array; only the phase and C1/C2 vary per frequency.
template<int D>
static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_swept_kernel_ek(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_i,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_j,
    double a_ek
) {
    static constexpr int NM = D + 1;
    static constexpr int NMM = NM * NM;

    auto sli = seg_l_i.unchecked<2>();
    auto sri = seg_r_i.unchecked<2>();
    auto slj = seg_l_j.unchecked<2>();
    auto srj = seg_r_j.unchecked<2>();
    auto ka  = k_array.unchecked<1>();
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();
    auto gi_v = group_i.unchecked<1>();
    auto gj_v = group_j.unchecked<1>();

    if (sli.shape(1) != 3 || sri.shape(1) != 3 ||
        slj.shape(1) != 3 || srj.shape(1) != 3) {
        throw std::runtime_error("segment endpoint arrays must have shape (N, 3)");
    }
    if (sli.shape(0) != sri.shape(0) || slj.shape(0) != srj.shape(0)) {
        throw std::runtime_error("seg_l and seg_r must have matching N");
    }
    if (glt.shape(0) != glw.shape(0)) {
        throw std::runtime_error("gl_t and gl_w must have matching length");
    }
    size_t n_qp = glt.shape(0);
    if (n_qp > 8) {
        throw std::runtime_error("n_qp > 8 not supported (scratch buffer size)");
    }

    size_t N_i = sli.shape(0);
    size_t N_j = slj.shape(0);
    size_t n_k = ka.shape(0);
    if ((size_t)group_i.shape(0) != N_i || (size_t)group_j.shape(0) != N_j) {
        throw std::runtime_error("group_i/group_j must match N_i/N_j");
    }

    py::array_t<std::complex<double>> J({n_k, (size_t)NM, (size_t)NM, N_i, N_j});
    auto j_view = J.mutable_unchecked<5>();

    py::gil_scoped_release release;

    const double inv_4pi = 1.0 / (4.0 * M_PI);
    const double a2_ek = a_ek * a_ek;
    const double a4_ek = a2_ek * a2_ek;

    std::vector<double> pos_i(N_i * n_qp * 3);
    std::vector<double> pos_j(N_j * n_qp * 3);
    std::vector<double> len_i(N_i);
    std::vector<double> len_j(N_j);
    for (size_t i = 0; i < N_i; i++) {
        double dx = sri(i,0) - sli(i,0);
        double dy = sri(i,1) - sli(i,1);
        double dz = sri(i,2) - sli(i,2);
        len_i[i] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (size_t q = 0; q < n_qp; q++) {
            double t = glt(q);
            pos_i[(i*n_qp + q)*3 + 0] = (1.0 - t) * sli(i,0) + t * sri(i,0);
            pos_i[(i*n_qp + q)*3 + 1] = (1.0 - t) * sli(i,1) + t * sri(i,1);
            pos_i[(i*n_qp + q)*3 + 2] = (1.0 - t) * sli(i,2) + t * sri(i,2);
        }
    }
    for (size_t j = 0; j < N_j; j++) {
        double dx = srj(j,0) - slj(j,0);
        double dy = srj(j,1) - slj(j,1);
        double dz = srj(j,2) - slj(j,2);
        len_j[j] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (size_t r = 0; r < n_qp; r++) {
            double t = glt(r);
            pos_j[(j*n_qp + r)*3 + 0] = (1.0 - t) * slj(j,0) + t * srj(j,0);
            pos_j[(j*n_qp + r)*3 + 1] = (1.0 - t) * slj(j,1) + t * srj(j,1);
            pos_j[(j*n_qp + r)*3 + 2] = (1.0 - t) * slj(j,2) + t * srj(j,2);
        }
    }

    PYSIM_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t i = 0; i < N_i; i++) {
        for (size_t j = 0; j < N_j; j++) {
            alignas(32) double R[64];
            alignas(32) double inv_R_4pi[64];
            alignas(32) double phases[64];
            alignas(32) double cos_phases[64];
            alignas(32) double sin_phases[64];
            alignas(32) double G_re[64], G_im[64];
            alignas(32) double t1v[64], t2v[64];
            alignas(32) double wuwu[NMM * 64];

            size_t n_pairs = n_qp * n_qp;
            const double *pi = &pos_i[i * n_qp * 3];
            const double *pj = &pos_j[j * n_qp * 3];

            for (size_t q = 0; q < n_qp; q++) {
                double pix = pi[q*3 + 0], piy = pi[q*3 + 1], piz = pi[q*3 + 2];
                for (size_t r = 0; r < n_qp; r++) {
                    double dx = pix - pj[r*3 + 0];
                    double dy = piy - pj[r*3 + 1];
                    double dz = piz - pj[r*3 + 2];
                    R[q*n_qp + r] = std::sqrt(dx*dx + dy*dy + dz*dz + a_squared);
                }
            }
            PYSIM_OMP_SIMD()
            for (size_t qr = 0; qr < n_pairs; qr++) {
                inv_R_4pi[qr] = inv_4pi / R[qr];
            }

            double Li = len_i[i];
            double Lj = len_j[j];
            for (size_t q = 0; q < n_qp; q++) {
                double wi = glw(q) * Li;
                double ui = glt(q) * Li;
                double ui_pow[NM];
                ui_pow[0] = 1.0;
                for (int p = 1; p < NM; p++) ui_pow[p] = ui_pow[p-1] * ui;
                for (size_t r = 0; r < n_qp; r++) {
                    double wj = glw(r) * Lj;
                    double uj = glt(r) * Lj;
                    double uj_pow[NM];
                    uj_pow[0] = 1.0;
                    for (int P = 1; P < NM; P++) uj_pow[P] = uj_pow[P-1] * uj;
                    double wij = wi * wj;
                    size_t qr = q * n_qp + r;
                    for (int p = 0; p < NM; p++) {
                        for (int P = 0; P < NM; P++) {
                            wuwu[(p * NM + P) * n_pairs + qr] = wij * ui_pow[p] * uj_pow[P];
                        }
                    }
                }
            }

            bool eligible = (gi_v(i) == gj_v(j)) && (gi_v(i) >= 0);
            if (eligible) {
                // T1, T2: functions of R and a_ek alone, so they hoist out
                // of the k loop exactly as the same-edge swept twin's do.
                PYSIM_OMP_SIMD()
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    double r2 = R[qr] * R[qr];
                    double r4 = r2 * r2;
                    t1v[qr] = 0.25 * a4_ek / r4;
                    t2v[qr] = 0.5 * a2_ek / r2;
                }
            }

            for (size_t kk = 0; kk < n_k; kk++) {
                double k = ka(kk);
                PYSIM_OMP_SIMD()
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    phases[qr] = -k * R[qr];
                }
                PYSIM_OMP_SIMD()
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    cos_phases[qr] = std::cos(phases[qr]);
                }
                PYSIM_OMP_SIMD()
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    sin_phases[qr] = std::sin(phases[qr]);
                }
                PYSIM_OMP_SIMD()
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    G_re[qr] = cos_phases[qr] * inv_R_4pi[qr];
                    G_im[qr] = sin_phases[qr] * inv_R_4pi[qr];
                }
                if (eligible) {
                    PYSIM_OMP_SIMD()
                    for (size_t qr = 0; qr < n_pairs; qr++) {
                        double kr = k * R[qr];
                        double kr2 = kr * kr;
                        double c1r = 1.0;
                        double c1i = kr;
                        double c2r = 3.0 * c1r - kr2;
                        double c2i = 3.0 * c1i;
                        double facr = t1v[qr] * c2r;
                        double faci = t1v[qr] * c2i;
                        facr = facr - t2v[qr] * c1r;
                        faci = faci - t2v[qr] * c1i;
                        facr = facr + 1.0;
                        double gre = G_re[qr];
                        double gim = G_im[qr];
                        G_re[qr] = gre * facr - gim * faci;
                        G_im[qr] = gre * faci + gim * facr;
                    }
                }
                for (int pP = 0; pP < NMM; pP++) {
                    double sr = 0.0, si = 0.0;
                    const double *w_row = &wuwu[pP * n_pairs];
                    PYSIM_OMP_SIMD(reduction(+:sr,si))
                    for (size_t qr = 0; qr < n_pairs; qr++) {
                        sr += w_row[qr] * G_re[qr];
                        si += w_row[qr] * G_im[qr];
                    }
                    j_view(kk, pP / NM, pP % NM, i, j) =
                        std::complex<double>(sr, si);
                }
            }
        }
    }

    return J;
}


// Templated B-spline Z assembly kernel.
//
// For each (m, n) basis pair, assembles the EFIE Galerkin entry from the
// polynomial-moment tensor J and the per-(basis, wing, poly-degree)
// coefficient table:
//   Z[m,n] = j*omega*mu * sum_{a,b} (t·t)[sm, sn]
//            * sum_{p, q} polys[m, a, p] * polys[n, b, q] * J[p, q, sm, sn]
//          + (1/jωε)    * sum_{a,b}
//            * sum_{p≥1, q≥1} p*q * polys[m, a, p] * polys[n, b, q]
//                           * J[p-1, q-1, sm, sn]
// where sm = support_seg[m, a], sn = support_seg[n, b].
//
// Inactive wings of boundary / junction-directional bases have polys = 0
// at every p, so they contribute nothing — no special handling needed.
//
// Template parameter D = B-spline degree (1 or 2). NM = D+1 wings per basis
// and D+1 polynomial moments per wing. Hardcoding NM as a compile-time
// constant unrolls the (D+1)^4 inner muladd loop.
//
// Single-k for now (BSplineSolver doesn't have a swept path yet); the inputs
// are scalar omega instead of an omega_array.
template<int D>
static py::array_t<std::complex<double>>
assemble_Z_bspline_kernel(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<double, py::array::c_style | py::array::forcecast> td_all,
    double omega,
    double eps_,
    double mu_,
    uintptr_t cancel_flag = 0
) {
    static constexpr int NM = D + 1;

    auto j_view = J.unchecked<4>();
    auto ss_view = support_seg.unchecked<2>();
    auto p_view = polys.unchecked<3>();
    auto td_view = td_all.unchecked<2>();

    size_t n_basis = (size_t)support_seg.shape(0);
    if (support_seg.shape(1) != NM) {
        throw std::runtime_error("support_seg.shape(1) must equal D+1");
    }
    if (polys.shape(0) != (long)n_basis || polys.shape(1) != NM ||
        polys.shape(2) != NM) {
        throw std::runtime_error("polys.shape must be (n_basis, D+1, D+1)");
    }
    if (J.shape(0) != NM || J.shape(1) != NM) {
        throw std::runtime_error("J.shape(0:2) must be (D+1, D+1)");
    }

    py::array_t<std::complex<double>> Z({n_basis, n_basis});
    auto z_view = Z.mutable_unchecked<2>();

    // Phase 0: release the GIL for the heavy compute region below.
    py::gil_scoped_release release;

    // Z = j*omega*mu * Z_A_accum + (1/(j*omega*eps)) * Z_Phi_accum
    // For Z_A_accum = re + j*im:    j*omega*mu * (re + j*im) = -omega*mu*im + j*omega*mu*re
    // For Z_Phi_accum = re + j*im:  (re + j*im)/(j*omega*eps) = im/(omega*eps) - j*re/(omega*eps)
    const double omega_mu = omega * mu_;
    const double inv_omega_eps = 1.0 / (omega * eps_);

    PYSIM_CANCEL_SETUP(cancel_flag);
    PYSIM_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t m = 0; m < n_basis; m++) {
        for (size_t n = 0; n < n_basis; n++) {
            PYSIM_CANCEL_POLL();
            double zA_re = 0.0, zA_im = 0.0;
            double zPhi_re = 0.0, zPhi_im = 0.0;

            for (int a = 0; a < NM; a++) {
                int64_t sm = ss_view(m, a);
                for (int b = 0; b < NM; b++) {
                    int64_t sn = ss_view(n, b);
                    double td = td_view(sm, sn);

                    double wA_re = 0.0, wA_im = 0.0;
                    double wPhi_re = 0.0, wPhi_im = 0.0;

                    for (int p = 0; p < NM; p++) {
                        double mp_ap = p_view(m, a, p);
                        for (int q = 0; q < NM; q++) {
                            double nq_bq = p_view(n, b, q);
                            std::complex<double> Jpq = j_view(p, q, sm, sn);
                            double prod = mp_ap * nq_bq;
                            wA_re += prod * Jpq.real();
                            wA_im += prod * Jpq.imag();
                            // p, q in {1..D}: Z_Phi contribution
                            if (p >= 1 && q >= 1) {
                                std::complex<double> Jpm1qm1 = j_view(p - 1, q - 1, sm, sn);
                                double pq = (double)(p * q) * prod;
                                wPhi_re += pq * Jpm1qm1.real();
                                wPhi_im += pq * Jpm1qm1.imag();
                            }
                        }
                    }

                    zA_re += td * wA_re;
                    zA_im += td * wA_im;
                    zPhi_re += wPhi_re;
                    zPhi_im += wPhi_im;
                }
            }

            double Zre = -omega_mu * zA_im + zPhi_im * inv_omega_eps;
            double Zim = omega_mu * zA_re - zPhi_re * inv_omega_eps;
            z_view(m, n) = std::complex<double>(Zre, Zim);
        }
    }

    PYSIM_THROW_IF_ABORTED();
    return Z;
}


// Windowed, accumulating variant of assemble_Z_bspline_kernel for the
// chunked dense build (issue #136): J_chunk holds the moment tensor for a
// rectangular segment window [i0, i1) x [j0, j1) only, and this kernel adds
// the window's contribution into a caller-provided Z. The (zA, zPhi) -> Z
// mixing is linear, so summing per-window contributions across calls
// reproduces the all-at-once assembly exactly; the caller never has to
// materialise the full (NM, NM, N, N) tensor. m_idx / n_idx list the basis
// rows/cols with at least one support wing inside the window (wings outside
// are skipped here), so per-chunk work stays proportional to the window.
// Each (m, n) pair is visited once per call, so the parallel += on
// z_view(m, n) is contention-free.
template<int D>
static void
assemble_Z_bspline_windowed_kernel(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J_chunk,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<double, py::array::c_style | py::array::forcecast> tangents,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> m_idx,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> n_idx,
    int64_t i0, int64_t i1, int64_t j0, int64_t j1,
    double omega,
    double eps_,
    double mu_,
    py::array_t<std::complex<double>> Z,  // any strides: F-order lets the caller's LAPACK solve factor in place
    uintptr_t cancel_flag = 0
) {
    static constexpr int NM = D + 1;

    auto j_view = J_chunk.unchecked<4>();
    auto ss_view = support_seg.unchecked<2>();
    auto p_view = polys.unchecked<3>();
    auto t_view = tangents.unchecked<2>();
    auto mi_view = m_idx.unchecked<1>();
    auto ni_view = n_idx.unchecked<1>();

    size_t n_basis = (size_t)support_seg.shape(0);
    if (support_seg.shape(1) != NM) {
        throw std::runtime_error("support_seg.shape(1) must equal D+1");
    }
    if (polys.shape(0) != (long)n_basis || polys.shape(1) != NM ||
        polys.shape(2) != NM) {
        throw std::runtime_error("polys.shape must be (n_basis, D+1, D+1)");
    }
    if (J_chunk.shape(0) != NM || J_chunk.shape(1) != NM ||
        J_chunk.shape(2) != i1 - i0 || J_chunk.shape(3) != j1 - j0) {
        throw std::runtime_error(
            "J_chunk.shape must be (D+1, D+1, i1-i0, j1-j0)");
    }
    // tangents is the per-segment unit tangent table, (n_segs_total, 3);
    // n_segs is not n_basis, so the only sound bound is the window's own
    // segment range — every support_seg id the loops below touch lies in
    // [i0, i1) or [j0, j1) by construction.
    if (tangents.shape(1) != 3) {
        throw std::runtime_error("tangents.shape must be (n_segs, 3)");
    }
    if (tangents.shape(0) < i1 || tangents.shape(0) < j1) {
        throw std::runtime_error(
            "tangents.shape(0) must cover the window's segment range");
    }
    if (Z.shape(0) != (long)n_basis || Z.shape(1) != (long)n_basis) {
        throw std::runtime_error("Z.shape must be (n_basis, n_basis)");
    }
    auto z_view = Z.mutable_unchecked<2>();

    py::gil_scoped_release release;

    const double omega_mu = omega * mu_;
    const double inv_omega_eps = 1.0 / (omega * eps_);
    size_t n_m = (size_t)m_idx.shape(0);
    size_t n_n = (size_t)n_idx.shape(0);

    PYSIM_CANCEL_SETUP(cancel_flag);
    PYSIM_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t mi = 0; mi < n_m; mi++) {
        for (size_t ni = 0; ni < n_n; ni++) {
            PYSIM_CANCEL_POLL();
            int64_t m = mi_view(mi);
            int64_t n = ni_view(ni);
            double zA_re = 0.0, zA_im = 0.0;
            double zPhi_re = 0.0, zPhi_im = 0.0;

            for (int a = 0; a < NM; a++) {
                int64_t sm = ss_view(m, a);
                if (sm < i0 || sm >= i1) continue;
                for (int b = 0; b < NM; b++) {
                    int64_t sn = ss_view(n, b);
                    if (sn < j0 || sn >= j1) continue;
                    // Tangent dot on the fly: the (N, N) table this used to
                    // read was N-squared doubles alive across the whole
                    // fill, right when Z is being accumulated (issue #318).
                    double td = t_view(sm, 0) * t_view(sn, 0) +
                                t_view(sm, 1) * t_view(sn, 1) +
                                t_view(sm, 2) * t_view(sn, 2);

                    double wA_re = 0.0, wA_im = 0.0;
                    double wPhi_re = 0.0, wPhi_im = 0.0;

                    for (int p = 0; p < NM; p++) {
                        double mp_ap = p_view(m, a, p);
                        for (int q = 0; q < NM; q++) {
                            double nq_bq = p_view(n, b, q);
                            std::complex<double> Jpq =
                                j_view(p, q, sm - i0, sn - j0);
                            double prod = mp_ap * nq_bq;
                            wA_re += prod * Jpq.real();
                            wA_im += prod * Jpq.imag();
                            if (p >= 1 && q >= 1) {
                                std::complex<double> Jpm1qm1 =
                                    j_view(p - 1, q - 1, sm - i0, sn - j0);
                                double pq = (double)(p * q) * prod;
                                wPhi_re += pq * Jpm1qm1.real();
                                wPhi_im += pq * Jpm1qm1.imag();
                            }
                        }
                    }

                    zA_re += td * wA_re;
                    zA_im += td * wA_im;
                    zPhi_re += wPhi_re;
                    zPhi_im += wPhi_im;
                }
            }

            double Zre = -omega_mu * zA_im + zPhi_im * inv_omega_eps;
            double Zim = omega_mu * zA_re - zPhi_re * inv_omega_eps;
            z_view(m, n) += std::complex<double>(Zre, Zim);
        }
    }

    PYSIM_THROW_IF_ABORTED();
}


static void
assemble_Z_bspline_windowed(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J_chunk,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<double, py::array::c_style | py::array::forcecast> tangents,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> m_idx,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> n_idx,
    int64_t i0, int64_t i1, int64_t j0, int64_t j1,
    double omega,
    double eps_,
    double mu_,
    py::array_t<std::complex<double>> Z,  // any strides: F-order lets the caller's LAPACK solve factor in place
    uintptr_t cancel_flag = 0
) {
    switch ((int)support_seg.shape(1) - 1) {
        case 1:
            assemble_Z_bspline_windowed_kernel<1>(
                J_chunk, support_seg, polys, tangents, m_idx, n_idx,
                i0, i1, j0, j1, omega, eps_, mu_, Z, cancel_flag);
            return;
        case 2:
            assemble_Z_bspline_windowed_kernel<2>(
                J_chunk, support_seg, polys, tangents, m_idx, n_idx,
                i0, i1, j0, j1, omega, eps_, mu_, Z, cancel_flag);
            return;
        default:
            throw std::runtime_error(
                "assemble_Z_bspline_windowed: max_d must be 1 or 2");
    }
}


// Weighted + scaled windowed accumulator — the ground-image counterpart of
// assemble_Z_bspline_windowed (issue #136 ground scope). Complex per-pair
// weights replace the real tangent-dot, exactly as
// assemble_Z_bspline_weighted generalises assemble_Z_bspline; `scale`
// multiplies the window's contribution before the += so the caller's
// Z -= image convention (PEC: scale = -1; Sommerfeld exact image:
// constant-C2 weights, scale = -1) needs no intermediate n_basis² matrix.
// Image pairs are never singular, so the caller runs no same-edge
// correction pass.
//
// wA_win / wPhi_win are WINDOWS, not global tables: shape (i1-i0, j1-j0),
// aligned with J_chunk's trailing axes and covering exactly the observer
// rows [i0, i1) and source cols [j0, j1) of the conceptual global (N, N)
// table (issue #323). Segment ids read out of support_seg are absolute, so
// every weight lookup is window-relative — (sm - i0, sn - j0) — the same
// shift J_chunk already needs. The caller therefore never has to keep two
// global complex (N, N) tables alive across the whole fill.
template<int D>
static void
assemble_Z_bspline_weighted_windowed_kernel(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J_chunk,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> wA_win,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> wPhi_win,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> m_idx,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> n_idx,
    int64_t i0, int64_t i1, int64_t j0, int64_t j1,
    double omega,
    double eps_,
    double mu_,
    std::complex<double> scale,
    py::array_t<std::complex<double>> Z,  // any strides: F-order lets the caller's LAPACK solve factor in place
    uintptr_t cancel_flag = 0
) {
    static constexpr int NM = D + 1;

    auto j_view = J_chunk.unchecked<4>();
    auto ss_view = support_seg.unchecked<2>();
    auto p_view = polys.unchecked<3>();
    auto wa_view = wA_win.unchecked<2>();
    auto wp_view = wPhi_win.unchecked<2>();
    auto mi_view = m_idx.unchecked<1>();
    auto ni_view = n_idx.unchecked<1>();

    size_t n_basis = (size_t)support_seg.shape(0);
    if (support_seg.shape(1) != NM) {
        throw std::runtime_error("support_seg.shape(1) must equal D+1");
    }
    if (polys.shape(0) != (long)n_basis || polys.shape(1) != NM ||
        polys.shape(2) != NM) {
        throw std::runtime_error("polys.shape must be (n_basis, D+1, D+1)");
    }
    if (J_chunk.shape(0) != NM || J_chunk.shape(1) != NM ||
        J_chunk.shape(2) != i1 - i0 || J_chunk.shape(3) != j1 - j0) {
        throw std::runtime_error(
            "J_chunk.shape must be (D+1, D+1, i1-i0, j1-j0)");
    }
    if (wA_win.shape(0) != i1 - i0 || wA_win.shape(1) != j1 - j0) {
        throw std::runtime_error("wA_win.shape must be (i1-i0, j1-j0)");
    }
    if (wPhi_win.shape(0) != i1 - i0 || wPhi_win.shape(1) != j1 - j0) {
        throw std::runtime_error("wPhi_win.shape must be (i1-i0, j1-j0)");
    }
    if (Z.shape(0) != (long)n_basis || Z.shape(1) != (long)n_basis) {
        throw std::runtime_error("Z.shape must be (n_basis, n_basis)");
    }
    auto z_view = Z.mutable_unchecked<2>();

    py::gil_scoped_release release;

    const double omega_mu = omega * mu_;
    const double inv_omega_eps = 1.0 / (omega * eps_);
    size_t n_m = (size_t)m_idx.shape(0);
    size_t n_n = (size_t)n_idx.shape(0);

    PYSIM_CANCEL_SETUP(cancel_flag);
    PYSIM_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t mi = 0; mi < n_m; mi++) {
        for (size_t ni = 0; ni < n_n; ni++) {
            PYSIM_CANCEL_POLL();
            int64_t m = mi_view(mi);
            int64_t n = ni_view(ni);
            std::complex<double> zA(0.0, 0.0);
            std::complex<double> zPhi(0.0, 0.0);

            for (int a = 0; a < NM; a++) {
                int64_t sm = ss_view(m, a);
                if (sm < i0 || sm >= i1) continue;
                for (int b = 0; b < NM; b++) {
                    int64_t sn = ss_view(n, b);
                    if (sn < j0 || sn >= j1) continue;
                    std::complex<double> wa = wa_view(sm - i0, sn - j0);
                    std::complex<double> wp = wp_view(sm - i0, sn - j0);

                    std::complex<double> iA(0.0, 0.0);
                    std::complex<double> iPhi(0.0, 0.0);

                    for (int p = 0; p < NM; p++) {
                        double mp_ap = p_view(m, a, p);
                        for (int q = 0; q < NM; q++) {
                            double nq_bq = p_view(n, b, q);
                            std::complex<double> Jpq =
                                j_view(p, q, sm - i0, sn - j0);
                            double prod = mp_ap * nq_bq;
                            iA += prod * Jpq;
                            if (p >= 1 && q >= 1) {
                                std::complex<double> Jpm1qm1 =
                                    j_view(p - 1, q - 1, sm - i0, sn - j0);
                                iPhi += ((double)(p * q) * prod) * Jpm1qm1;
                            }
                        }
                    }

                    zA += wa * iA;
                    zPhi += wp * iPhi;
                }
            }

            // Z_win = j*omega*mu*zA + zPhi/(j*omega*eps), then scaled.
            std::complex<double> Zc(
                -omega_mu * zA.imag() + zPhi.imag() * inv_omega_eps,
                omega_mu * zA.real() - zPhi.real() * inv_omega_eps);
            std::complex<double> add = scale * Zc;
            z_view(m, n) += add;
        }
    }

    PYSIM_THROW_IF_ABORTED();
}


static void
assemble_Z_bspline_weighted_windowed(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J_chunk,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> wA_win,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> wPhi_win,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> m_idx,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> n_idx,
    int64_t i0, int64_t i1, int64_t j0, int64_t j1,
    double omega,
    double eps_,
    double mu_,
    std::complex<double> scale,
    py::array_t<std::complex<double>> Z,  // any strides: F-order lets the caller's LAPACK solve factor in place
    uintptr_t cancel_flag = 0
) {
    switch ((int)support_seg.shape(1) - 1) {
        case 1:
            assemble_Z_bspline_weighted_windowed_kernel<1>(
                J_chunk, support_seg, polys, wA_win, wPhi_win, m_idx, n_idx,
                i0, i1, j0, j1, omega, eps_, mu_, scale, Z, cancel_flag);
            return;
        case 2:
            assemble_Z_bspline_weighted_windowed_kernel<2>(
                J_chunk, support_seg, polys, wA_win, wPhi_win, m_idx, n_idx,
                i0, i1, j0, j1, omega, eps_, mu_, scale, Z, cancel_flag);
            return;
        default:
            throw std::runtime_error(
                "assemble_Z_bspline_weighted_windowed: max_d must be 1 or 2");
    }
}


// Batched (swept-k) variant of assemble_Z_bspline_kernel. J carries a leading
// k axis (n_k, NM, NM, N, N) and omega is an array; the basis tables
// (support_seg, polys, tangents_row, tangents_col) are k-independent and
// reused across the sweep. Returns (n_k, n_basis, n_basis). Lets
// compute_impedance_swept assemble the whole sweep in one call instead of
// one per frequency, the bspline analog of triangular's batched assemble_Z.
//
// tangents_row / tangents_col are (n_segs, 3) per-segment unit-tangent
// tables — NOT the (N, N) dot-product table the kernel used to take
// (issue #333, the swept-batched twin of #318's windowed fix). The tangent
// dot for a given (sm, sn) pair is formed in-kernel from the two rows, same
// as assemble_Z_bspline_windowed_kernel does. The caller passes
// (tangents, tangents) for the free-space term and (tangents,
// mirrored_tangents) for the PEC image term — row side is always the real
// geometry, column side carries the mirror when one applies — so one kernel
// serves both without ever materialising an N² table.
template<int D>
static py::array_t<std::complex<double>>
assemble_Z_bspline_swept_kernel(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<double, py::array::c_style | py::array::forcecast> tangents_row,
    py::array_t<double, py::array::c_style | py::array::forcecast> tangents_col,
    py::array_t<double, py::array::c_style | py::array::forcecast> omega_array,
    double eps_,
    double mu_
) {
    static constexpr int NM = D + 1;

    auto j_view = J.unchecked<5>();
    auto ss_view = support_seg.unchecked<2>();
    auto p_view = polys.unchecked<3>();
    auto tr_view = tangents_row.unchecked<2>();
    auto tc_view = tangents_col.unchecked<2>();
    auto om = omega_array.unchecked<1>();

    size_t n_k = (size_t)J.shape(0);
    size_t n_basis = (size_t)support_seg.shape(0);
    if (support_seg.shape(1) != NM) {
        throw std::runtime_error("support_seg.shape(1) must equal D+1");
    }
    if (J.shape(1) != NM || J.shape(2) != NM) {
        throw std::runtime_error("J.shape(1:3) must be (D+1, D+1)");
    }
    if ((size_t)om.shape(0) != n_k) {
        throw std::runtime_error("omega_array length must match J.shape(0)");
    }
    if (tangents_row.shape(1) != 3 || tangents_col.shape(1) != 3) {
        throw std::runtime_error(
            "tangents_row / tangents_col shape must be (n_segs, 3)");
    }
    // support_seg ids are absolute segment indices into J's trailing (N, N)
    // axes, so the only sound bound for the tangent tables is that they
    // cover J's segment range — same convention as the windowed kernel.
    if ((size_t)tangents_row.shape(0) < (size_t)J.shape(3) ||
        (size_t)tangents_col.shape(0) < (size_t)J.shape(4)) {
        throw std::runtime_error(
            "tangents_row / tangents_col must cover J's segment range");
    }

    py::array_t<std::complex<double>> Z({n_k, n_basis, n_basis});
    auto z_view = Z.mutable_unchecked<3>();

    // Release the GIL for the heavy compute region below.
    py::gil_scoped_release release;

    PYSIM_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t m = 0; m < n_basis; m++) {
        for (size_t n = 0; n < n_basis; n++) {
            for (size_t kk = 0; kk < n_k; kk++) {
                const double omega_mu = om(kk) * mu_;
                const double inv_omega_eps = 1.0 / (om(kk) * eps_);
                double zA_re = 0.0, zA_im = 0.0;
                double zPhi_re = 0.0, zPhi_im = 0.0;

                for (int a = 0; a < NM; a++) {
                    int64_t sm = ss_view(m, a);
                    for (int b = 0; b < NM; b++) {
                        int64_t sn = ss_view(n, b);
                        // Tangent dot on the fly, not read from an (N, N)
                        // table hoisted for the whole sweep (issue #333).
                        double td = tr_view(sm, 0) * tc_view(sn, 0) +
                                    tr_view(sm, 1) * tc_view(sn, 1) +
                                    tr_view(sm, 2) * tc_view(sn, 2);
                        double wA_re = 0.0, wA_im = 0.0;
                        double wPhi_re = 0.0, wPhi_im = 0.0;
                        for (int p = 0; p < NM; p++) {
                            double mp_ap = p_view(m, a, p);
                            for (int q = 0; q < NM; q++) {
                                double nq_bq = p_view(n, b, q);
                                std::complex<double> Jpq = j_view(kk, p, q, sm, sn);
                                double prod = mp_ap * nq_bq;
                                wA_re += prod * Jpq.real();
                                wA_im += prod * Jpq.imag();
                                if (p >= 1 && q >= 1) {
                                    std::complex<double> Jm =
                                        j_view(kk, p - 1, q - 1, sm, sn);
                                    double pq = (double)(p * q) * prod;
                                    wPhi_re += pq * Jm.real();
                                    wPhi_im += pq * Jm.imag();
                                }
                            }
                        }
                        zA_re += td * wA_re;
                        zA_im += td * wA_im;
                        zPhi_re += wPhi_re;
                        zPhi_im += wPhi_im;
                    }
                }
                double Zre = -omega_mu * zA_im + zPhi_im * inv_omega_eps;
                double Zim = omega_mu * zA_re - zPhi_re * inv_omega_eps;
                z_view(kk, m, n) = std::complex<double>(Zre, Zim);
            }
        }
    }

    return Z;
}

static py::array_t<std::complex<double>>
assemble_Z_bspline(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<double, py::array::c_style | py::array::forcecast> td_all,
    double omega,
    double eps_,
    double mu_,
    int max_d,
    uintptr_t cancel_flag = 0
) {
    switch (max_d) {
        case 1:
            return assemble_Z_bspline_kernel<1>(J, support_seg, polys, td_all, omega, eps_, mu_, cancel_flag);
        case 2:
            return assemble_Z_bspline_kernel<2>(J, support_seg, polys, td_all, omega, eps_, mu_, cancel_flag);
        default:
            throw std::runtime_error(
                "assemble_Z_bspline: max_d must be 1 or 2");
    }
}


// Weighted variant of assemble_Z_bspline_kernel for the reflection-
// coefficient finite ground (BSplineSolver ground_eps): the A term takes a
// COMPLEX per-segment-pair weight table wA_all (the Fresnel dyad tangent
// table, replacing the real tangent-dot), and the Φ term — unweighted in
// the PEC kernel — takes its own complex per-pair image-charge table
// wPhi_all. Same loop structure, same J tensor, one pass; the PEC kernel is
// the wA = t·Mt (real), wPhi = 1 special case.
template<int D>
static py::array_t<std::complex<double>>
assemble_Z_bspline_weighted_kernel(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> wA_all,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> wPhi_all,
    double omega,
    double eps_,
    double mu_,
    uintptr_t cancel_flag = 0
) {
    static constexpr int NM = D + 1;

    auto j_view = J.unchecked<4>();
    auto ss_view = support_seg.unchecked<2>();
    auto p_view = polys.unchecked<3>();
    auto wa_view = wA_all.unchecked<2>();
    auto wp_view = wPhi_all.unchecked<2>();

    size_t n_basis = (size_t)support_seg.shape(0);
    if (support_seg.shape(1) != NM) {
        throw std::runtime_error("support_seg.shape(1) must equal D+1");
    }
    if (polys.shape(0) != (long)n_basis || polys.shape(1) != NM ||
        polys.shape(2) != NM) {
        throw std::runtime_error("polys.shape must be (n_basis, D+1, D+1)");
    }
    if (J.shape(0) != NM || J.shape(1) != NM) {
        throw std::runtime_error("J.shape(0:2) must be (D+1, D+1)");
    }
    if (wA_all.shape(0) != wPhi_all.shape(0) ||
        wA_all.shape(1) != wPhi_all.shape(1)) {
        throw std::runtime_error("wA_all / wPhi_all shape mismatch");
    }

    py::array_t<std::complex<double>> Z({n_basis, n_basis});
    auto z_view = Z.mutable_unchecked<2>();

    py::gil_scoped_release release;

    const double omega_mu = omega * mu_;
    const double inv_omega_eps = 1.0 / (omega * eps_);

    PYSIM_CANCEL_SETUP(cancel_flag);
    PYSIM_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t m = 0; m < n_basis; m++) {
        for (size_t n = 0; n < n_basis; n++) {
            PYSIM_CANCEL_POLL();
            double zA_re = 0.0, zA_im = 0.0;
            double zPhi_re = 0.0, zPhi_im = 0.0;

            for (int a = 0; a < NM; a++) {
                int64_t sm = ss_view(m, a);
                for (int b = 0; b < NM; b++) {
                    int64_t sn = ss_view(n, b);
                    std::complex<double> wa = wa_view(sm, sn);
                    std::complex<double> wp = wp_view(sm, sn);

                    double iA_re = 0.0, iA_im = 0.0;
                    double iPhi_re = 0.0, iPhi_im = 0.0;

                    for (int p = 0; p < NM; p++) {
                        double mp_ap = p_view(m, a, p);
                        for (int q = 0; q < NM; q++) {
                            double nq_bq = p_view(n, b, q);
                            std::complex<double> Jpq = j_view(p, q, sm, sn);
                            double prod = mp_ap * nq_bq;
                            iA_re += prod * Jpq.real();
                            iA_im += prod * Jpq.imag();
                            if (p >= 1 && q >= 1) {
                                std::complex<double> Jpm1qm1 = j_view(p - 1, q - 1, sm, sn);
                                double pq = (double)(p * q) * prod;
                                iPhi_re += pq * Jpm1qm1.real();
                                iPhi_im += pq * Jpm1qm1.imag();
                            }
                        }
                    }

                    // complex weight × complex inner sum, by parts
                    zA_re += wa.real() * iA_re - wa.imag() * iA_im;
                    zA_im += wa.real() * iA_im + wa.imag() * iA_re;
                    zPhi_re += wp.real() * iPhi_re - wp.imag() * iPhi_im;
                    zPhi_im += wp.real() * iPhi_im + wp.imag() * iPhi_re;
                }
            }

            double Zre = -omega_mu * zA_im + zPhi_im * inv_omega_eps;
            double Zim = omega_mu * zA_re - zPhi_re * inv_omega_eps;
            z_view(m, n) = std::complex<double>(Zre, Zim);
        }
    }

    PYSIM_THROW_IF_ABORTED();
    return Z;
}

static py::array_t<std::complex<double>>
assemble_Z_bspline_weighted(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> wA_all,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> wPhi_all,
    double omega,
    double eps_,
    double mu_,
    int max_d,
    uintptr_t cancel_flag = 0
) {
    switch (max_d) {
        case 1:
            return assemble_Z_bspline_weighted_kernel<1>(J, support_seg, polys, wA_all, wPhi_all, omega, eps_, mu_, cancel_flag);
        case 2:
            return assemble_Z_bspline_weighted_kernel<2>(J, support_seg, polys, wA_all, wPhi_all, omega, eps_, mu_, cancel_flag);
        default:
            throw std::runtime_error(
                "assemble_Z_bspline_weighted: max_d must be 1 or 2");
    }
}


// Runtime dispatch wrapper for the batched (swept-k) assemble.
static py::array_t<std::complex<double>>
assemble_Z_bspline_swept(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<double, py::array::c_style | py::array::forcecast> tangents_row,
    py::array_t<double, py::array::c_style | py::array::forcecast> tangents_col,
    py::array_t<double, py::array::c_style | py::array::forcecast> omega_array,
    double eps_,
    double mu_,
    int max_d
) {
    switch (max_d) {
        case 1:
            return assemble_Z_bspline_swept_kernel<1>(
                J, support_seg, polys, tangents_row, tangents_col,
                omega_array, eps_, mu_);
        case 2:
            return assemble_Z_bspline_swept_kernel<2>(
                J, support_seg, polys, tangents_row, tangents_col,
                omega_array, eps_, mu_);
        default:
            throw std::runtime_error(
                "assemble_Z_bspline_swept: max_d must be 1 or 2");
    }
}


// Fused off-edge block assembler for the hierarchical (H-matrix / ACA) solver.
//
// Computes a dense Z[I, J] block where every basis pair is OFF-EDGE (the
// caller guarantees admissibility / well-separation), fusing the moment
// quadrature and the Galerkin assembly into one pass — no intermediate
// (D+1, D+1, N, N) moment tensor and no numpy einsum. ACA's row/column
// sampling calls this with a single-row or single-column basis slice, so the
// whole per-row Python orchestration (np.unique / np.vectorize / dict maps /
// einsum) is replaced by one C++ call.
//
// Segment data is passed as the union of segments referenced by the I-side
// and J-side bases (resolved once per block in Python); support_*_local index
// into those union arrays. Same EFIE Galerkin formula as
// assemble_Z_bspline_kernel, but the per-pair moments are quadratured inline
// from the segment endpoints (a²-regularised full kernel, the off-edge path).
//
// WEIGHTED=true is the reflection-coefficient finite-ground image variant
// (BSplineSolver ground_eps): the caller passes the J side PRE-MIRRORED
// (positions reflected across the ground plane, tangents z-flipped), exactly
// as for the PEC image evaluators, plus ε̃ and the Φ-weight coefficients
// (w_Φ = phi_c0 + phi_c1·ρ_v — every ground_phi_mode reduces to this form,
// see _ground_refl.phi_mode_coeffs). Everything the Fresnel dyad needs is
// then already in the inputs: the obs→image midpoint delta gives cos θ and
// the incidence plane, and the mirrored tangent dot IS the PEC mirror table
// td_img — the kernel never needs ground_z itself. Weights are evaluated
// from segment midpoints (the NEC-style per-pair-constant approximation,
// same as the dense path). WEIGHTED=false compiles to the original kernel;
// the weight parameters are ignored.
template<int D, bool WEIGHTED>
static py::array_t<std::complex<double>>
bspline_assemble_offedge_block_kernel(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_I,   // (nI, NM)
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_I,   // (nI, NM, NM)
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_I,    // (nSegI, 3)
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_I,    // (nSegI, 3)
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_I,     // (nSegI, 3)
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_J,   // (nJ, NM)
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_J,   // (nJ, NM, NM)
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_J,    // (nSegJ, 3)
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_J,    // (nSegJ, 3)
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_J,     // (nSegJ, 3)
    double a_squared,
    double k,
    double omega,
    double eps_,
    double mu_,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    uintptr_t cancel_flag = 0,
    std::complex<double> eps_t = std::complex<double>(0.0, 0.0),
    std::complex<double> phi_c0 = std::complex<double>(0.0, 0.0),
    std::complex<double> phi_c1 = std::complex<double>(0.0, 0.0)
) {
    static constexpr int NM = D + 1;

    auto sI = supp_I.unchecked<2>();
    auto pI = polys_I.unchecked<3>();
    auto slI = segl_I.unchecked<2>();
    auto srI = segr_I.unchecked<2>();
    auto tI = tan_I.unchecked<2>();
    auto sJ = supp_J.unchecked<2>();
    auto pJ = polys_J.unchecked<3>();
    auto slJ = segl_J.unchecked<2>();
    auto srJ = segr_J.unchecked<2>();
    auto tJ = tan_J.unchecked<2>();
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();

    size_t nI = (size_t)supp_I.shape(0);
    size_t nJ = (size_t)supp_J.shape(0);
    size_t nSegI = (size_t)segl_I.shape(0);
    size_t nSegJ = (size_t)segl_J.shape(0);
    size_t n_qp = (size_t)gl_t.shape(0);
    if (supp_I.shape(1) != NM || supp_J.shape(1) != NM) {
        throw std::runtime_error("support arrays must have shape (n, D+1)");
    }
    if (n_qp > 8) {
        throw std::runtime_error("n_qp > 8 not supported (scratch buffer size)");
    }

    // Per-segment quadrature positions + lengths, precomputed once.
    std::vector<double> posI(nSegI * n_qp * 3), lenI(nSegI);
    std::vector<double> posJ(nSegJ * n_qp * 3), lenJ(nSegJ);
    // Segment midpoints for the WEIGHTED specular geometry (J side is
    // already mirrored, so midI − midJ is the obs→image ray).
    std::vector<double> midI, midJ;
    if (WEIGHTED) {
        midI.resize(nSegI * 3);
        midJ.resize(nSegJ * 3);
        for (size_t s = 0; s < nSegI; s++)
            for (int c = 0; c < 3; c++)
                midI[s*3+c] = 0.5 * (slI(s,c) + srI(s,c));
        for (size_t s = 0; s < nSegJ; s++)
            for (int c = 0; c < 3; c++)
                midJ[s*3+c] = 0.5 * (slJ(s,c) + srJ(s,c));
    }
    for (size_t s = 0; s < nSegI; s++) {
        double dx = srI(s,0)-slI(s,0), dy = srI(s,1)-slI(s,1), dz = srI(s,2)-slI(s,2);
        lenI[s] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (size_t q = 0; q < n_qp; q++) {
            double t = glt(q);
            posI[(s*n_qp+q)*3+0] = (1.0-t)*slI(s,0) + t*srI(s,0);
            posI[(s*n_qp+q)*3+1] = (1.0-t)*slI(s,1) + t*srI(s,1);
            posI[(s*n_qp+q)*3+2] = (1.0-t)*slI(s,2) + t*srI(s,2);
        }
    }
    for (size_t s = 0; s < nSegJ; s++) {
        double dx = srJ(s,0)-slJ(s,0), dy = srJ(s,1)-slJ(s,1), dz = srJ(s,2)-slJ(s,2);
        lenJ[s] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (size_t q = 0; q < n_qp; q++) {
            double t = glt(q);
            posJ[(s*n_qp+q)*3+0] = (1.0-t)*slJ(s,0) + t*srJ(s,0);
            posJ[(s*n_qp+q)*3+1] = (1.0-t)*slJ(s,1) + t*srJ(s,1);
            posJ[(s*n_qp+q)*3+2] = (1.0-t)*slJ(s,2) + t*srJ(s,2);
        }
    }

    py::array_t<std::complex<double>> Z({nI, nJ});
    auto z_view = Z.mutable_unchecked<2>();

    // Phase 0: release the GIL for the heavy compute region below.
    py::gil_scoped_release release;

    const double inv_4pi = 1.0 / (4.0 * M_PI);
    const double omega_mu = omega * mu_;
    const double inv_omega_eps = 1.0 / (omega * eps_);

    PYSIM_CANCEL_SETUP(cancel_flag);
    PYSIM_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t m = 0; m < nI; m++) {
        for (size_t n = 0; n < nJ; n++) {
            PYSIM_CANCEL_POLL();
            double zA_re = 0.0, zA_im = 0.0, zPhi_re = 0.0, zPhi_im = 0.0;

            for (int a = 0; a < NM; a++) {
                int64_t smi = sI(m, a);
                double tix = tI(smi,0), tiy = tI(smi,1), tiz = tI(smi,2);
                const double *pi = &posI[smi * n_qp * 3];
                double Li = lenI[smi];
                for (int b = 0; b < NM; b++) {
                    int64_t snj = sJ(n, b);
                    const double *pj = &posJ[snj * n_qp * 3];
                    double Lj = lenJ[snj];
                    double td = tix*tJ(snj,0) + tiy*tJ(snj,1) + tiz*tJ(snj,2);

                    // Moment tensor Jc[p][P] for this single segment pair.
                    std::complex<double> Jc[NM][NM];
                    {
                        alignas(32) double R[64], G_re[64], G_im[64];
                        alignas(32) double wuwu[(NM*NM) * 64];
                        size_t n_pairs = n_qp * n_qp;
                        for (size_t q = 0; q < n_qp; q++) {
                            double pix = pi[q*3+0], piy = pi[q*3+1], piz = pi[q*3+2];
                            for (size_t r = 0; r < n_qp; r++) {
                                double dx = pix - pj[r*3+0];
                                double dy = piy - pj[r*3+1];
                                double dz = piz - pj[r*3+2];
                                R[q*n_qp+r] = std::sqrt(dx*dx+dy*dy+dz*dz+a_squared);
                            }
                        }
                        for (size_t q = 0; q < n_qp; q++) {
                            double wi = glw(q) * Li, ui = glt(q) * Li;
                            double uip[NM]; uip[0] = 1.0;
                            for (int p = 1; p < NM; p++) uip[p] = uip[p-1]*ui;
                            for (size_t r = 0; r < n_qp; r++) {
                                double wj = glw(r) * Lj, uj = glt(r) * Lj;
                                double ujp[NM]; ujp[0] = 1.0;
                                for (int P = 1; P < NM; P++) ujp[P] = ujp[P-1]*uj;
                                double wij = wi*wj;
                                size_t qr = q*n_qp + r;
                                for (int p = 0; p < NM; p++)
                                    for (int P = 0; P < NM; P++)
                                        wuwu[(p*NM+P)*n_pairs + qr] = wij*uip[p]*ujp[P];
                            }
                        }
                        PYSIM_OMP_SIMD()
                        for (size_t qr = 0; qr < n_pairs; qr++) {
                            double inv = inv_4pi / R[qr];
                            double ph = -k * R[qr];
                            G_re[qr] = std::cos(ph) * inv;
                            G_im[qr] = std::sin(ph) * inv;
                        }
                        for (int pP = 0; pP < NM*NM; pP++) {
                            double sr_ = 0.0, si_ = 0.0;
                            const double *w_row = &wuwu[pP * n_pairs];
                            PYSIM_OMP_SIMD(reduction(+:sr_,si_))
                            for (size_t qr = 0; qr < n_pairs; qr++) {
                                sr_ += w_row[qr]*G_re[qr];
                                si_ += w_row[qr]*G_im[qr];
                            }
                            Jc[pP/NM][pP%NM] = std::complex<double>(sr_, si_);
                        }
                    }

                    // Galerkin combine for this wing pair.
                    double iA_re = 0.0, iA_im = 0.0, iPhi_re = 0.0, iPhi_im = 0.0;
                    for (int p = 0; p < NM; p++) {
                        double mp = pI(m, a, p);
                        for (int q = 0; q < NM; q++) {
                            double nq = pJ(n, b, q);
                            double prod = mp * nq;
                            iA_re += prod * Jc[p][q].real();
                            iA_im += prod * Jc[p][q].imag();
                            if (p >= 1 && q >= 1) {
                                double pq = (double)(p*q) * prod;
                                iPhi_re += pq * Jc[p-1][q-1].real();
                                iPhi_im += pq * Jc[p-1][q-1].imag();
                            }
                        }
                    }
                    if (WEIGHTED) {
                        // Fresnel dyad at the pair's specular angle. J side
                        // is mirrored, so Δ = obs mid − image mid; td is
                        // already the PEC mirror tangent dot td_img.
                        double ddx = midI[smi*3+0] - midJ[snj*3+0];
                        double ddy = midI[smi*3+1] - midJ[snj*3+1];
                        double ddz = midI[smi*3+2] - midJ[snj*3+2];
                        double rmag = std::sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
                        double cth = ddz / (rmag > 1e-30 ? rmag : 1e-30);
                        double hyp = std::sqrt(ddx*ddx + ddy*ddy);
                        double px, py;
                        if (hyp > 1e-30) { px = -ddy/hyp; py = ddx/hyp; }
                        else             { px = 1.0;      py = 0.0;     }
                        // (t·p̂) uses xy components only, so the mirrored
                        // trial tangent gives the same value as the real one.
                        double tip = tix*px + tiy*py;
                        double tjp = tJ(snj,0)*px + tJ(snj,1)*py;
                        double P_ = tip * tjp;
                        std::complex<double> root =
                            std::sqrt(eps_t - (1.0 - cth*cth));
                        std::complex<double> rv =
                            (eps_t*cth - root) / (eps_t*cth + root);
                        std::complex<double> rh =
                            (cth - root) / (cth + root);
                        std::complex<double> wa = rv*(td - P_) - rh*P_;
                        std::complex<double> wp = phi_c0 + phi_c1*rv;
                        zA_re += wa.real()*iA_re - wa.imag()*iA_im;
                        zA_im += wa.real()*iA_im + wa.imag()*iA_re;
                        zPhi_re += wp.real()*iPhi_re - wp.imag()*iPhi_im;
                        zPhi_im += wp.real()*iPhi_im + wp.imag()*iPhi_re;
                    } else {
                        zA_re += td * iA_re;
                        zA_im += td * iA_im;
                        zPhi_re += iPhi_re;
                        zPhi_im += iPhi_im;
                    }
                }
            }

            double Zre = -omega_mu * zA_im + zPhi_im * inv_omega_eps;
            double Zim = omega_mu * zA_re - zPhi_re * inv_omega_eps;
            z_view(m, n) = std::complex<double>(Zre, Zim);
        }
    }

    PYSIM_THROW_IF_ABORTED();
    return Z;
}

static py::array_t<std::complex<double>>
bspline_assemble_offedge_block(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_I,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_J,
    double a_squared, double k, double omega, double eps_, double mu_, int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    uintptr_t cancel_flag = 0
) {
    switch (max_d) {
        case 1:
            return bspline_assemble_offedge_block_kernel<1, false>(
                supp_I, polys_I, segl_I, segr_I, tan_I, supp_J, polys_J,
                segl_J, segr_J, tan_J, a_squared, k, omega, eps_, mu_, gl_t, gl_w,
                cancel_flag);
        case 2:
            return bspline_assemble_offedge_block_kernel<2, false>(
                supp_I, polys_I, segl_I, segr_I, tan_I, supp_J, polys_J,
                segl_J, segr_J, tan_J, a_squared, k, omega, eps_, mu_, gl_t, gl_w,
                cancel_flag);
        default:
            throw std::runtime_error(
                "bspline_assemble_offedge_block: max_d must be 1 or 2");
    }
}

// Reflection-coefficient finite-ground image variant of the fused off-edge
// block assembler (see the WEIGHTED=true notes on the kernel). The J side
// must be passed pre-mirrored, exactly as for the PEC image evaluators.
static py::array_t<std::complex<double>>
bspline_assemble_offedge_block_refl(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_I,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_J,
    double a_squared, double k, double omega, double eps_, double mu_, int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    std::complex<double> eps_t,
    std::complex<double> phi_c0,
    std::complex<double> phi_c1,
    uintptr_t cancel_flag = 0
) {
    switch (max_d) {
        case 1:
            return bspline_assemble_offedge_block_kernel<1, true>(
                supp_I, polys_I, segl_I, segr_I, tan_I, supp_J, polys_J,
                segl_J, segr_J, tan_J, a_squared, k, omega, eps_, mu_, gl_t, gl_w,
                cancel_flag, eps_t, phi_c0, phi_c1);
        case 2:
            return bspline_assemble_offedge_block_kernel<2, true>(
                supp_I, polys_I, segl_I, segr_I, tan_I, supp_J, polys_J,
                segl_J, segr_J, tan_J, a_squared, k, omega, eps_, mu_, gl_t, gl_w,
                cancel_flag, eps_t, phi_c0, phi_c1);
        default:
            throw std::runtime_error(
                "bspline_assemble_offedge_block_refl: max_d must be 1 or 2");
    }
}

// No EK twin here: EK + finite ground (the reflection-coefficient image) is
// refused upstream at the solver level (momwire#269), so this variant never
// needs to serve an extended-kernel fill and is left reduced-only.


// THE FUSED OFF-EDGE BLOCK ASSEMBLER'S EXTENDED-KERNEL TWIN (momwire#270
// unit 3)
// -------------------------------------------------------------------
// `bspline_assemble_offedge_block_kernel<D, false>` above fuses the
// off-edge moment quadrature with the EFIE Galerkin combine so ACA
// row/col/dense block sampling never materialises an intermediate
// (d+1, d+1, N_i, N_j) moment tensor. This is that assembler's EK twin:
// same fusion, with NEC Eq 89's coaxial factor applied to G on eligible
// SEGMENT pairs before the Galerkin contraction — the fused-assembler
// analog of unit 2's `seg_seg_full_moments_bspline_kernel_ek`, whose
// eligibility rule and `fac` spelling (T1, T2, C1, C2, in that order) this
// transcribes verbatim.
//
// `group_I` (nSegI,) / `group_J` (nSegJ,) are per-segment coaxial-and-
// equal-radius labels over the SAME per-block segment unions `tan_I` /
// `tan_J` are already keyed by (the caller's `segI`/`segJ` — hmatrix.py's
// `_offedge_block_evaluators_uniform`) — NOT per-basis, because eligibility
// is a property of the (segment, segment) pair sampled inside the basis-
// pair wing loop below, exactly as it is in unit 2. A pair is eligible iff
// `group_I[smi] == group_J[snj] >= 0`, evaluated once per (a, b) wing and
// applied to every quadrature sub-pair inside it (the mask does not vary
// with (q, r), same as unit 2). `a_ek` is the plain (unsquared) EK radius,
// kept separate from `a_squared` for the same reason unit 1/2 keep it
// separate: on every eligible pair the two agree by construction, but the
// C++ side mirrors `_ek_radius(ek, a)` rather than assuming it.
//
// WEIGHTED=true is the reflection-coefficient finite-ground image variant
// (momwire#269 lifted #249's EK + `ground_eps` refusal), and it is the exact
// composition of the two halves already above: the EK factor multiplies G
// pair by pair BEFORE the Galerkin contraction, the Fresnel dyad weights the
// contracted A / Φ terms AFTER it. The two never interact — the dyad is a
// per-segment-pair scalar built from the specular geometry and ε̃ alone, and
// the EK factor is a function of R and a_ek alone — which is why the weight
// block below is `bspline_assemble_offedge_block_kernel<D, true>`'s verbatim,
// applied to Jc values the eligibility branch has already extended. Callers
// pass the J side pre-mirrored exactly as for WEIGHTED=false.
template<int D, bool WEIGHTED>
static py::array_t<std::complex<double>>
bspline_assemble_offedge_block_kernel_ek(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_I,   // (nI, NM)
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_I,   // (nI, NM, NM)
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_I,    // (nSegI, 3)
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_I,    // (nSegI, 3)
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_I,     // (nSegI, 3)
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_J,   // (nJ, NM)
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_J,   // (nJ, NM, NM)
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_J,    // (nSegJ, 3)
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_J,    // (nSegJ, 3)
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_J,     // (nSegJ, 3)
    double a_squared,
    double k,
    double omega,
    double eps_,
    double mu_,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_I,  // (nSegI,)
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_J,  // (nSegJ,)
    double a_ek,
    uintptr_t cancel_flag = 0,
    std::complex<double> eps_t = std::complex<double>(0.0, 0.0),
    std::complex<double> phi_c0 = std::complex<double>(0.0, 0.0),
    std::complex<double> phi_c1 = std::complex<double>(0.0, 0.0)
) {
    static constexpr int NM = D + 1;

    auto sI = supp_I.unchecked<2>();
    auto pI = polys_I.unchecked<3>();
    auto slI = segl_I.unchecked<2>();
    auto srI = segr_I.unchecked<2>();
    auto tI = tan_I.unchecked<2>();
    auto sJ = supp_J.unchecked<2>();
    auto pJ = polys_J.unchecked<3>();
    auto slJ = segl_J.unchecked<2>();
    auto srJ = segr_J.unchecked<2>();
    auto tJ = tan_J.unchecked<2>();
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();
    auto gI_v = group_I.unchecked<1>();
    auto gJ_v = group_J.unchecked<1>();

    size_t nI = (size_t)supp_I.shape(0);
    size_t nJ = (size_t)supp_J.shape(0);
    size_t nSegI = (size_t)segl_I.shape(0);
    size_t nSegJ = (size_t)segl_J.shape(0);
    size_t n_qp = (size_t)gl_t.shape(0);
    if (supp_I.shape(1) != NM || supp_J.shape(1) != NM) {
        throw std::runtime_error("support arrays must have shape (n, D+1)");
    }
    if (n_qp > 8) {
        throw std::runtime_error("n_qp > 8 not supported (scratch buffer size)");
    }
    if ((size_t)group_I.shape(0) != nSegI || (size_t)group_J.shape(0) != nSegJ) {
        throw std::runtime_error(
            "bspline_assemble_offedge_block_kernel_ek: group_I/group_J must "
            "match the segment unions");
    }

    // Per-segment quadrature positions + lengths, precomputed once — the
    // reduced kernel's own precompute, including the WEIGHTED-only midpoint
    // tables the Fresnel dyad reads (J side already mirrored by the caller,
    // so midI − midJ is the obs→image ray).
    std::vector<double> posI(nSegI * n_qp * 3), lenI(nSegI);
    std::vector<double> posJ(nSegJ * n_qp * 3), lenJ(nSegJ);
    std::vector<double> midI, midJ;
    if (WEIGHTED) {
        midI.resize(nSegI * 3);
        midJ.resize(nSegJ * 3);
        for (size_t s = 0; s < nSegI; s++)
            for (int c = 0; c < 3; c++)
                midI[s*3+c] = 0.5 * (slI(s,c) + srI(s,c));
        for (size_t s = 0; s < nSegJ; s++)
            for (int c = 0; c < 3; c++)
                midJ[s*3+c] = 0.5 * (slJ(s,c) + srJ(s,c));
    }
    for (size_t s = 0; s < nSegI; s++) {
        double dx = srI(s,0)-slI(s,0), dy = srI(s,1)-slI(s,1), dz = srI(s,2)-slI(s,2);
        lenI[s] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (size_t q = 0; q < n_qp; q++) {
            double t = glt(q);
            posI[(s*n_qp+q)*3+0] = (1.0-t)*slI(s,0) + t*srI(s,0);
            posI[(s*n_qp+q)*3+1] = (1.0-t)*slI(s,1) + t*srI(s,1);
            posI[(s*n_qp+q)*3+2] = (1.0-t)*slI(s,2) + t*srI(s,2);
        }
    }
    for (size_t s = 0; s < nSegJ; s++) {
        double dx = srJ(s,0)-slJ(s,0), dy = srJ(s,1)-slJ(s,1), dz = srJ(s,2)-slJ(s,2);
        lenJ[s] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (size_t q = 0; q < n_qp; q++) {
            double t = glt(q);
            posJ[(s*n_qp+q)*3+0] = (1.0-t)*slJ(s,0) + t*srJ(s,0);
            posJ[(s*n_qp+q)*3+1] = (1.0-t)*slJ(s,1) + t*srJ(s,1);
            posJ[(s*n_qp+q)*3+2] = (1.0-t)*slJ(s,2) + t*srJ(s,2);
        }
    }

    py::array_t<std::complex<double>> Z({nI, nJ});
    auto z_view = Z.mutable_unchecked<2>();

    // Phase 0: release the GIL for the heavy compute region below.
    py::gil_scoped_release release;

    const double inv_4pi = 1.0 / (4.0 * M_PI);
    const double omega_mu = omega * mu_;
    const double inv_omega_eps = 1.0 / (omega * eps_);
    const double a2_ek = a_ek * a_ek;
    const double a4_ek = a2_ek * a2_ek;

    PYSIM_CANCEL_SETUP(cancel_flag);
    PYSIM_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t m = 0; m < nI; m++) {
        for (size_t n = 0; n < nJ; n++) {
            PYSIM_CANCEL_POLL();
            double zA_re = 0.0, zA_im = 0.0, zPhi_re = 0.0, zPhi_im = 0.0;

            for (int a = 0; a < NM; a++) {
                int64_t smi = sI(m, a);
                double tix = tI(smi,0), tiy = tI(smi,1), tiz = tI(smi,2);
                const double *pi = &posI[smi * n_qp * 3];
                double Li = lenI[smi];
                for (int b = 0; b < NM; b++) {
                    int64_t snj = sJ(n, b);
                    const double *pj = &posJ[snj * n_qp * 3];
                    double Lj = lenJ[snj];
                    double td = tix*tJ(snj,0) + tiy*tJ(snj,1) + tiz*tJ(snj,2);
                    // Eligibility is a property of THIS (smi, snj) SEGMENT
                    // pair, not of the quadrature sub-pair — one branch
                    // below serves every (q, r), exactly as unit 2's
                    // off-edge twin.
                    bool eligible = (gI_v(smi) == gJ_v(snj)) && (gI_v(smi) >= 0);

                    // Moment tensor Jc[p][P] for this single segment pair.
                    std::complex<double> Jc[NM][NM];
                    {
                        alignas(32) double R[64], G_re[64], G_im[64];
                        alignas(32) double wuwu[(NM*NM) * 64];
                        size_t n_pairs = n_qp * n_qp;
                        for (size_t q = 0; q < n_qp; q++) {
                            double pix = pi[q*3+0], piy = pi[q*3+1], piz = pi[q*3+2];
                            for (size_t r = 0; r < n_qp; r++) {
                                double dx = pix - pj[r*3+0];
                                double dy = piy - pj[r*3+1];
                                double dz = piz - pj[r*3+2];
                                R[q*n_qp+r] = std::sqrt(dx*dx+dy*dy+dz*dz+a_squared);
                            }
                        }
                        for (size_t q = 0; q < n_qp; q++) {
                            double wi = glw(q) * Li, ui = glt(q) * Li;
                            double uip[NM]; uip[0] = 1.0;
                            for (int p = 1; p < NM; p++) uip[p] = uip[p-1]*ui;
                            for (size_t r = 0; r < n_qp; r++) {
                                double wj = glw(r) * Lj, uj = glt(r) * Lj;
                                double ujp[NM]; ujp[0] = 1.0;
                                for (int P = 1; P < NM; P++) ujp[P] = ujp[P-1]*uj;
                                double wij = wi*wj;
                                size_t qr = q*n_qp + r;
                                for (int p = 0; p < NM; p++)
                                    for (int P = 0; P < NM; P++)
                                        wuwu[(p*NM+P)*n_pairs + qr] = wij*uip[p]*ujp[P];
                            }
                        }
                        PYSIM_OMP_SIMD()
                        for (size_t qr = 0; qr < n_pairs; qr++) {
                            double inv = inv_4pi / R[qr];
                            double ph = -k * R[qr];
                            G_re[qr] = std::cos(ph) * inv;
                            G_im[qr] = std::sin(ph) * inv;
                        }
                        if (eligible) {
                            // `_ek_factor`'s spelling, term by term: T1, T2,
                            // C1, C2, fac = T1*C2 - T2*C1 + 1, G *= fac —
                            // unit 2's off-edge twin, transcribed again.
                            PYSIM_OMP_SIMD()
                            for (size_t qr = 0; qr < n_pairs; qr++) {
                                double Rq = R[qr];
                                double r2 = Rq * Rq;
                                double r4 = r2 * r2;
                                double kr = k * Rq;
                                double kr2 = kr * kr;
                                double t1 = 0.25 * a4_ek / r4;
                                double t2 = 0.5 * a2_ek / r2;
                                double c1r = 1.0;
                                double c1i = kr;
                                double c2r = 3.0 * c1r - kr2;
                                double c2i = 3.0 * c1i;
                                double facr = t1 * c2r;
                                double faci = t1 * c2i;
                                facr = facr - t2 * c1r;
                                faci = faci - t2 * c1i;
                                facr = facr + 1.0;
                                double gre = G_re[qr];
                                double gim = G_im[qr];
                                G_re[qr] = gre * facr - gim * faci;
                                G_im[qr] = gre * faci + gim * facr;
                            }
                        }
                        for (int pP = 0; pP < NM*NM; pP++) {
                            double sr_ = 0.0, si_ = 0.0;
                            const double *w_row = &wuwu[pP * n_pairs];
                            PYSIM_OMP_SIMD(reduction(+:sr_,si_))
                            for (size_t qr = 0; qr < n_pairs; qr++) {
                                sr_ += w_row[qr]*G_re[qr];
                                si_ += w_row[qr]*G_im[qr];
                            }
                            Jc[pP/NM][pP%NM] = std::complex<double>(sr_, si_);
                        }
                    }

                    // Galerkin combine for this wing pair.
                    double iA_re = 0.0, iA_im = 0.0, iPhi_re = 0.0, iPhi_im = 0.0;
                    for (int p = 0; p < NM; p++) {
                        double mp = pI(m, a, p);
                        for (int q = 0; q < NM; q++) {
                            double nq = pJ(n, b, q);
                            double prod = mp * nq;
                            iA_re += prod * Jc[p][q].real();
                            iA_im += prod * Jc[p][q].imag();
                            if (p >= 1 && q >= 1) {
                                double pq = (double)(p*q) * prod;
                                iPhi_re += pq * Jc[p-1][q-1].real();
                                iPhi_im += pq * Jc[p-1][q-1].imag();
                            }
                        }
                    }
                    if (WEIGHTED) {
                        // The reduced kernel's WEIGHTED=true tail, verbatim:
                        // Fresnel dyad at the pair's specular angle, applied
                        // to the already-extended contracted moments.
                        double ddx = midI[smi*3+0] - midJ[snj*3+0];
                        double ddy = midI[smi*3+1] - midJ[snj*3+1];
                        double ddz = midI[smi*3+2] - midJ[snj*3+2];
                        double rmag = std::sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
                        double cth = ddz / (rmag > 1e-30 ? rmag : 1e-30);
                        double hyp = std::sqrt(ddx*ddx + ddy*ddy);
                        double px, py;
                        if (hyp > 1e-30) { px = -ddy/hyp; py = ddx/hyp; }
                        else             { px = 1.0;      py = 0.0;     }
                        double tip = tix*px + tiy*py;
                        double tjp = tJ(snj,0)*px + tJ(snj,1)*py;
                        double P_ = tip * tjp;
                        std::complex<double> root =
                            std::sqrt(eps_t - (1.0 - cth*cth));
                        std::complex<double> rv =
                            (eps_t*cth - root) / (eps_t*cth + root);
                        std::complex<double> rh =
                            (cth - root) / (cth + root);
                        std::complex<double> wa = rv*(td - P_) - rh*P_;
                        std::complex<double> wp = phi_c0 + phi_c1*rv;
                        zA_re += wa.real()*iA_re - wa.imag()*iA_im;
                        zA_im += wa.real()*iA_im + wa.imag()*iA_re;
                        zPhi_re += wp.real()*iPhi_re - wp.imag()*iPhi_im;
                        zPhi_im += wp.real()*iPhi_im + wp.imag()*iPhi_re;
                    } else {
                        zA_re += td * iA_re;
                        zA_im += td * iA_im;
                        zPhi_re += iPhi_re;
                        zPhi_im += iPhi_im;
                    }
                }
            }

            double Zre = -omega_mu * zA_im + zPhi_im * inv_omega_eps;
            double Zim = omega_mu * zA_re - zPhi_re * inv_omega_eps;
            z_view(m, n) = std::complex<double>(Zre, Zim);
        }
    }

    PYSIM_THROW_IF_ABORTED();
    return Z;
}

static py::array_t<std::complex<double>>
bspline_assemble_offedge_block_ek(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_I,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_J,
    double a_squared, double k, double omega, double eps_, double mu_, int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_I,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_J,
    double a_ek,
    uintptr_t cancel_flag = 0
) {
    switch (max_d) {
        case 1:
            return bspline_assemble_offedge_block_kernel_ek<1, false>(
                supp_I, polys_I, segl_I, segr_I, tan_I, supp_J, polys_J,
                segl_J, segr_J, tan_J, a_squared, k, omega, eps_, mu_, gl_t, gl_w,
                group_I, group_J, a_ek, cancel_flag);
        case 2:
            return bspline_assemble_offedge_block_kernel_ek<2, false>(
                supp_I, polys_I, segl_I, segr_I, tan_I, supp_J, polys_J,
                segl_J, segr_J, tan_J, a_squared, k, omega, eps_, mu_, gl_t, gl_w,
                group_I, group_J, a_ek, cancel_flag);
        default:
            throw std::runtime_error(
                "bspline_assemble_offedge_block_ek: max_d must be 1 or 2");
    }
}

// The extended-kernel twin of `bspline_assemble_offedge_block_refl`
// (momwire#269): the reflection-coefficient finite-ground image block with
// the coaxial factor applied on eligible segment pairs. J side pre-mirrored,
// exactly as for both parents.
static py::array_t<std::complex<double>>
bspline_assemble_offedge_block_refl_ek(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_I,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_J,
    double a_squared, double k, double omega, double eps_, double mu_, int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_I,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_J,
    double a_ek,
    std::complex<double> eps_t,
    std::complex<double> phi_c0,
    std::complex<double> phi_c1,
    uintptr_t cancel_flag = 0
) {
    switch (max_d) {
        case 1:
            return bspline_assemble_offedge_block_kernel_ek<1, true>(
                supp_I, polys_I, segl_I, segr_I, tan_I, supp_J, polys_J,
                segl_J, segr_J, tan_J, a_squared, k, omega, eps_, mu_, gl_t, gl_w,
                group_I, group_J, a_ek, cancel_flag, eps_t, phi_c0, phi_c1);
        case 2:
            return bspline_assemble_offedge_block_kernel_ek<2, true>(
                supp_I, polys_I, segl_I, segr_I, tan_I, supp_J, polys_J,
                segl_J, segr_J, tan_J, a_squared, k, omega, eps_, mu_, gl_t, gl_w,
                group_I, group_J, a_ek, cancel_flag, eps_t, phi_c0, phi_c1);
        default:
            throw std::runtime_error(
                "bspline_assemble_offedge_block_refl_ek: max_d must be 1 or 2");
    }
}


// Runtime dispatch wrapper. Picks the right template instantiation based on
// max_d (the maximum polynomial moment degree, == B-spline degree D).
static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    double k,
    int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w
) {
    switch (max_d) {
        case 1:
            return seg_seg_full_moments_bspline_kernel<1>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k, gl_t, gl_w);
        case 2:
            return seg_seg_full_moments_bspline_kernel<2>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k, gl_t, gl_w);
        default:
            throw std::runtime_error(
                "seg_seg_full_moments_bspline: max_d must be 1 or 2 "
                "(add an explicit template instantiation in _accelerators.cpp)");
    }
}


// Runtime dispatch wrapper for the batched (swept-k) off-edge kernel.
static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_swept(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array,
    int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w
) {
    switch (max_d) {
        case 1:
            return seg_seg_full_moments_bspline_swept_kernel<1>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k_array, gl_t, gl_w);
        case 2:
            return seg_seg_full_moments_bspline_swept_kernel<2>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k_array, gl_t, gl_w);
        default:
            throw std::runtime_error(
                "seg_seg_full_moments_bspline_swept: max_d must be 1 or 2 "
                "(add an explicit template instantiation in _accelerators.cpp)");
    }
}


// Runtime dispatch wrapper for the off-edge extended-kernel twin
// (momwire#270 unit 2).
static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_ek(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    double k,
    int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_i,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_j,
    double a_ek
) {
    switch (max_d) {
        case 1:
            return seg_seg_full_moments_bspline_kernel_ek<1>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k, gl_t, gl_w,
                group_i, group_j, a_ek);
        case 2:
            return seg_seg_full_moments_bspline_kernel_ek<2>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k, gl_t, gl_w,
                group_i, group_j, a_ek);
        default:
            throw std::runtime_error(
                "seg_seg_full_moments_bspline_ek: max_d must be 1 or 2 "
                "(add an explicit template instantiation in _accelerators.cpp)");
    }
}


// Runtime dispatch wrapper for the batched (swept-k) off-edge
// extended-kernel twin (momwire#270 unit 2).
static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_swept_ek(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array,
    int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_i,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_j,
    double a_ek
) {
    switch (max_d) {
        case 1:
            return seg_seg_full_moments_bspline_swept_kernel_ek<1>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k_array, gl_t, gl_w,
                group_i, group_j, a_ek);
        case 2:
            return seg_seg_full_moments_bspline_swept_kernel_ek<2>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k_array, gl_t, gl_w,
                group_i, group_j, a_ek);
        default:
            throw std::runtime_error(
                "seg_seg_full_moments_bspline_swept_ek: max_d must be 1 or 2 "
                "(add an explicit template instantiation in _accelerators.cpp)");
    }
}


// Toeplitz fast-path B-spline static-moment evaluation.
//
// For a single straight edge with uniform-h segments, the J_pq^static[i, j]
// integrals are translation-invariant in the arc direction — the matrix is
// Toeplitz with 2N-1 unique values per (p, q) moment. This function computes
// those 2N-1 values via the sympy-derived closed forms (inlined from
// _bspline_static_moments_inline.h) and gathers them to the (max_d+1,
// max_d+1, N, N) output.
//
// Replaces the per-edge numpy loop in `_seg_seg_static_moments` — that path
// took ~5 ms / call mainly from numpy dispatch overhead; the C++ inlined
// closed forms run in ~0.1 ms / call. Big win on multi-edge polylines like
// the hentenna where the static moments dominate after the all-pairs J kernel.
//
// max_d ∈ {0, 1, 2} currently — extends automatically when the header file
// is regenerated for larger MAX_D in scripts/derive_bspline_static_moments.py
// (and the case-list in J_static_dispatch below is extended).
static double J_static_dispatch(int p, int q,
                                double alpha, double beta,
                                double A, double B, double a) {
    int pq = p * 3 + q;
    switch (pq) {
        case 0: return J_static_pq_0_0(alpha, beta, A, B, a);
        case 1: return J_static_pq_0_1(alpha, beta, A, B, a);
        case 2: return J_static_pq_0_2(alpha, beta, A, B, a);
        case 3: return J_static_pq_1_0(alpha, beta, A, B, a);
        case 4: return J_static_pq_1_1(alpha, beta, A, B, a);
        case 5: return J_static_pq_1_2(alpha, beta, A, B, a);
        case 6: return J_static_pq_2_0(alpha, beta, A, B, a);
        case 7: return J_static_pq_2_1(alpha, beta, A, B, a);
        case 8: return J_static_pq_2_2(alpha, beta, A, B, a);
        default:
            throw std::runtime_error("J_static: (p, q) out of inline range");
    }
}

// The extended thin-wire kernel's static correction, same shape of dispatch
// (momwire#270 unit 1).
//
//   D_pq^EK = ∫∫ (s-α)^p (s'-A)^q [ -a²/(2R³) + 3a⁴/(4R⁵) ] ds' ds
//
// i.e. the k → 0 limit of Eq 89's coaxial factor minus 1, integrated against
// the same polynomial moments as J. It is a function of (α, β, A, B) through
// the same four corner differences J is, so it is translation-invariant along
// the edge exactly as J is and rides the Toeplitz gather below unchanged.
//
// `a_ek` is a SEPARATE argument from the regularization radius `a`: on every
// eligible pair they are equal by construction (eligibility requires equal
// radii), but `_EK.a` lets a caller override, and keeping them apart here
// means the C++ mirrors `_ek_radius(ek, a)` on the Python side rather than
// assuming it.
static double D_ek_dispatch(int p, int q,
                            double alpha, double beta,
                            double A, double B, double a) {
    int pq = p * 3 + q;
    switch (pq) {
        case 0: return D_ek_pq_0_0(alpha, beta, A, B, a);
        case 1: return D_ek_pq_0_1(alpha, beta, A, B, a);
        case 2: return D_ek_pq_0_2(alpha, beta, A, B, a);
        case 3: return D_ek_pq_1_0(alpha, beta, A, B, a);
        case 4: return D_ek_pq_1_1(alpha, beta, A, B, a);
        case 5: return D_ek_pq_1_2(alpha, beta, A, B, a);
        case 6: return D_ek_pq_2_0(alpha, beta, A, B, a);
        case 7: return D_ek_pq_2_1(alpha, beta, A, B, a);
        case 8: return D_ek_pq_2_2(alpha, beta, A, B, a);
        default:
            throw std::runtime_error("D_ek: (p, q) out of inline range");
    }
}

// Shared body of the reduced and extended Toeplitz static kernels. `EK` is a
// compile-time template parameter, so the EK-off instantiation is the pre-#270
// loop with the `if (EK)` block folded away — no runtime branch, no change to
// the reduced arithmetic.
//
// A NOTE ON LAST BITS (momwire#270 unit 1, measured). The reduced kernel's
// OUTPUT still moves by 1-3 ulp against a pre-#270 build (30 of 441 entries on
// the gate deck, max 1.8e-15 relative), and not because of anything above:
// D_ek_pq_0_2 / 1_2 / 2_2 call J_static_pq_0_0 / 1_0 / 2_0, which gives those
// three header inlines a second call site, which changes how GCC inlines them
// into THIS function, which at `-mfma -ffp-contract=fast` is a different set
// of fused multiply-adds. Confirmed by bisection: including the D header
// without using it is bit-identical; both a shared body and a fully duplicated
// one shift the same 30 entries; giving the EK twin its own copy of
// J_static_dispatch does not help either.
//
// So absolute cross-build bit stability is not a property this translation
// unit has, and it is not one to pin (the same argument as antennaknobs#253:
// never pin cross-machine bit equality — this is the cross-build case of it).
// What IS armored, and stays armored, is the within-build claim the tests
// actually make: EK-off is the same code path and the same bits as the
// default, and no EK code is entered to produce it.
template <bool EK>
static py::array_t<double>
seg_seg_static_moments_bspline_uniform_impl(double h, double a, size_t N,
                                            int max_d, double a_ek) {
    if (max_d < 0 || max_d > 2) {
        throw std::runtime_error("max_d out of range [0, 2]");
    }
    size_t NM = (size_t)(max_d + 1);
    py::array_t<double> out({NM, NM, N, N});
    auto v = out.mutable_unchecked<4>();

    // Phase 0: release the GIL for the heavy compute region below.
    py::gil_scoped_release release;

    // 2N-1 unique Toeplitz values per moment, indexed by Δ = j - i ∈ [-(N-1), N-1].
    // delta_idx = Δ + (N - 1) ∈ [0, 2N-2].
    size_t n_delta = 2 * N - 1;
    const double inv_4pi = 1.0 / (4.0 * M_PI);

    // Build (NM, NM, n_delta) Toeplitz table
    std::vector<double> table(NM * NM * n_delta);
    for (size_t p = 0; p < NM; p++) {
        for (size_t q = 0; q < NM; q++) {
            for (size_t di = 0; di < n_delta; di++) {
                long long delta = (long long)di - (long long)(N - 1);
                double alpha = 0.0;
                double beta = h;
                double A_ = (double)delta * h;
                double B_ = ((double)delta + 1.0) * h;
                double val = J_static_dispatch((int)p, (int)q, alpha, beta, A_, B_, a);
                if (EK) {
                    // numpy's `vals = vals + D_ek_moment(...)` then `* inv4pi`,
                    // in that order (_bspline_kernels._seg_seg_static_moments).
                    val = val + D_ek_dispatch((int)p, (int)q, alpha, beta, A_, B_,
                                              a_ek);
                }
                table[(p * NM + q) * n_delta + di] = val * inv_4pi;
            }
        }
    }

    // Gather: v(p, q, i, j) = table[p, q, j - i + (N - 1)]
    for (size_t p = 0; p < NM; p++) {
        for (size_t q = 0; q < NM; q++) {
            const double *row = &table[(p * NM + q) * n_delta];
            for (size_t i = 0; i < N; i++) {
                for (size_t j = 0; j < N; j++) {
                    size_t di = (size_t)((long long)j - (long long)i + (long long)(N - 1));
                    v(p, q, i, j) = row[di];
                }
            }
        }
    }
    return out;
}

static py::array_t<double>
seg_seg_static_moments_bspline_uniform(double h, double a, size_t N, int max_d) {
    return seg_seg_static_moments_bspline_uniform_impl<false>(h, a, N, max_d, 0.0);
}

static py::array_t<double>
seg_seg_static_moments_bspline_uniform_ek(double h, double a, size_t N, int max_d,
                                          double a_ek) {
    return seg_seg_static_moments_bspline_uniform_impl<true>(h, a, N, max_d, a_ek);
}


// Assemble the (Z_pe, Z_ep, Z_ee) blocks for the singular basis enrichment at
// K≥3 junctions (PR #47 productized path).
//
// Each enrichment basis e lives on a single segment adjacent to a junction.
// The shape on that segment is Φ_sing(u) = (u/h)·log(u/h) where u is measured
// from the junction node (u_origin=0 → u = t·h_e, u_origin=1 → u = (1-t)·h_e).
// dΦ_sing/du = (log(u/h) + 1) / h  — log-singular at u=0, matching the K≥3
// junction charge-density singularity.
//
// Integrals (all complex, single k):
//   Z_ee[e, f] = j*ω*μ * td * I_A  +  I_Phi / (j*ω*ε)
//     I_A   = ∫∫ Φ_e(u) Φ_f(u') G du du'
//     I_Phi = ∫∫ Φ_e'(u) Φ_f'(u') G du du'
//   Z_pe[m, e] = same, with polynomial basis m on one side and Φ_e on the other.
//   Z_ep[e, m] = same, but computed independently (no .T shortcut) — the two
//     match to floating-point precision when the same GL rule is used on both
//     axes, but computing them separately verifies that and keeps the path
//     robust if a future quadrature change breaks the symmetry.
//
// Parallelism: outer loop over m (polynomial basis index) for the (Z_pe, Z_ep)
// work, which dominates cost (n_poly ≫ n_enrich). Z_ee is small (n_enrich²);
// computed serially after.
//
// td (the tangent dot on a segment pair) is formed in-kernel from the
// (n_segs, 3) `tangents` table rather than read from a precomputed (N, N)
// td_all matrix (issue #334) — that table was N-squared doubles alive
// across the whole free-space enrichment fill, rebuilt per k in an
// enrichment sweep, though `assemble_Z_enrich` only ever reads it at the
// handful of (spec_seg[e], n) pairs this kernel actually visits.
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
assemble_Z_enrich(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> spec_seg,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> spec_origin,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r,
    py::array_t<double, py::array::c_style | py::array::forcecast> h_per_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> tangents,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_seg_poly,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_poly,
    double a_squared,
    double k,
    double omega,
    double eps_,
    double mu_,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t01,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w01,
    py::array_t<double, py::array::c_style | py::array::forcecast> proj_coeffs
) {
    auto specs_v   = spec_seg.unchecked<1>();
    auto origin_v  = spec_origin.unchecked<1>();
    auto sl_v      = seg_l.unchecked<2>();
    auto sr_v      = seg_r.unchecked<2>();
    auto h_v       = h_per_seg.unchecked<1>();
    if (tangents.shape(1) != 3) {
        throw std::runtime_error("tangents.shape must be (n_segs, 3)");
    }
    auto tan_v     = tangents.unchecked<2>();
    // Tangent dot on the fly rather than reading an (N, N) td_all table:
    // that table was N-squared doubles alive across the whole enrichment
    // fill, and rebuilt per k in an enrichment sweep (issue #334, same
    // shape as #318's assemble_Z_bspline_windowed_kernel fix). td_me and
    // td_em below are each a fixed-order 3-term dot, so swapping the
    // argument order still produces bit-identical values term-by-term
    // (float multiplication is commutative), matching the old table's
    // (mirror-index) symmetry.
    auto tdot = [&tan_v](int64_t i, int64_t j) -> double {
        return tan_v(i, 0) * tan_v(j, 0) + tan_v(i, 1) * tan_v(j, 1) +
               tan_v(i, 2) * tan_v(j, 2);
    };
    auto ss_v      = supp_seg_poly.unchecked<2>();
    auto polys_v   = polys_poly.unchecked<3>();
    auto t01_v     = gl_t01.unchecked<1>();
    auto w01_v     = gl_w01.unchecked<1>();
    auto pc_v      = proj_coeffs.unchecked<1>();

    size_t n_enrich = (size_t)spec_seg.shape(0);
    size_t n_poly   = (size_t)supp_seg_poly.shape(0);
    size_t n_wings  = (size_t)supp_seg_poly.shape(1);
    size_t n_qp     = (size_t)gl_t01.shape(0);

    if ((size_t)spec_origin.shape(0) != n_enrich) {
        throw std::runtime_error("spec_origin must match spec_seg length");
    }
    if ((size_t)polys_poly.shape(0) != n_poly ||
        (size_t)polys_poly.shape(1) != n_wings) {
        throw std::runtime_error("polys_poly first two dims must match supp_seg_poly");
    }
    size_t d_plus_1 = (size_t)polys_poly.shape(2);
    if ((size_t)gl_w01.shape(0) != n_qp) {
        throw std::runtime_error("gl_t01 and gl_w01 must have matching length");
    }
    if (sl_v.shape(1) != 3 || sr_v.shape(1) != 3) {
        throw std::runtime_error("seg_l/seg_r must have shape (N_seg, 3)");
    }
    if ((size_t)proj_coeffs.shape(0) != d_plus_1) {
        throw std::runtime_error("proj_coeffs length must equal d+1 (degree + 1)");
    }

    py::array_t<std::complex<double>> Z_pe({n_poly, n_enrich});
    py::array_t<std::complex<double>> Z_ep({n_enrich, n_poly});
    py::array_t<std::complex<double>> Z_ee({n_enrich, n_enrich});
    auto zpe_v = Z_pe.mutable_unchecked<2>();
    auto zep_v = Z_ep.mutable_unchecked<2>();
    auto zee_v = Z_ee.mutable_unchecked<2>();

    if (n_enrich == 0) {
        // Nothing to compute. Return empty arrays.
        return std::make_tuple(Z_pe, Z_ep, Z_ee);
    }

    // Phase 0: release the GIL for the heavy compute region below.
    py::gil_scoped_release release;

    const double inv_4pi = 1.0 / (4.0 * M_PI);
    const double omega_mu = omega * mu_;
    const double inv_omega_eps = 1.0 / (omega * eps_);

    // -----------------------------------------------------------------
    // Per-enrichment precompute: 3D quad-point positions, Φ_sing values,
    // dΦ_sing/du in arc length, and quadrature weights pre-scaled by h_e.
    // -----------------------------------------------------------------
    std::vector<double> pos_e_all(n_enrich * n_qp * 3);
    std::vector<double> sing_val_all(n_enrich * n_qp);
    std::vector<double> sing_dval_all(n_enrich * n_qp);
    std::vector<double> w_e_all(n_enrich * n_qp);
    std::vector<double> h_e_arr(n_enrich);
    std::vector<int64_t> seg_e_arr(n_enrich);

    const double eps_tiny = 1e-300;
    for (size_t e = 0; e < n_enrich; e++) {
        int64_t se = specs_v(e);
        int orig = (int)origin_v(e);
        double he = h_v(se);
        h_e_arr[e] = he;
        seg_e_arr[e] = se;
        // d(u_norm)/d(u_arc_along_wire): for orig=0 the junction is at the
        // segment's left endpoint, so u_norm = t = u_arc/h and the derivative
        // is +1/h. For orig=1 the junction is at the right endpoint, so
        // u_norm = 1 − t = 1 − u_arc/h and the derivative is −1/h. The
        // singular basis's slope dΦ/du_arc inherits that sign — without
        // it, every "end"-orientation enrichment basis enters the Φ-piece
        // of Z_pe/Z_ep (and the mixed-orig off-diagonals of Z_ee) with the
        // wrong sign, breaking L-R symmetry on geometries like hentenna
        // where mirror junctions have opposite orig.
        double dphi_sign = (orig == 0) ? 1.0 : -1.0;
        for (size_t q = 0; q < n_qp; q++) {
            double t = t01_v(q);
            double w = w01_v(q);
            double u_norm = (orig == 0) ? t : (1.0 - t);
            double u_safe = u_norm > eps_tiny ? u_norm : eps_tiny;
            double log_u = std::log(u_safe);
            // Stable XFEM: Φ_sing_stable(t) = t·log(t) − Σ c_p t^p, so the
            // enrichment basis is L²-orthogonal to the local polynomial
            // space {1, t, …, t^d} on the segment. The projection is in
            // u_norm — the segment's natural orientation-aware coordinate
            // — so it carries through both orientations unchanged.
            // dΦ/du_arc = (dΦ/du_norm) · (du_norm/du_arc) = (...) · sign/h.
            double poly_val = 0.0;
            double poly_dval = 0.0;
            // Horner on Σ c_p t^p and its derivative Σ p·c_p t^(p-1).
            poly_val  = pc_v(d_plus_1 - 1);
            poly_dval = (double)(d_plus_1 - 1) * pc_v(d_plus_1 - 1);
            for (size_t pp = d_plus_1 - 1; pp-- > 0; ) {
                poly_val = poly_val * u_norm + pc_v(pp);
                if (pp >= 1) {
                    poly_dval = poly_dval * u_norm + (double)pp * pc_v(pp);
                }
            }
            sing_val_all[e * n_qp + q] = u_norm * log_u - poly_val;
            sing_dval_all[e * n_qp + q] = dphi_sign * (log_u + 1.0 - poly_dval) / he;
            w_e_all[e * n_qp + q] = w * he;
            double *pe = &pos_e_all[(e * n_qp + q) * 3];
            pe[0] = (1.0 - t) * sl_v(se, 0) + t * sr_v(se, 0);
            pe[1] = (1.0 - t) * sl_v(se, 1) + t * sr_v(se, 1);
            pe[2] = (1.0 - t) * sl_v(se, 2) + t * sr_v(se, 2);
        }
    }

    // -----------------------------------------------------------------
    // Z_ee assembly: pairs (e, f). Symmetric; fill upper triangle then mirror.
    // -----------------------------------------------------------------
    for (size_t e = 0; e < n_enrich; e++) {
        for (size_t f = e; f < n_enrich; f++) {
            double td = tdot(seg_e_arr[e], seg_e_arr[f]);
            double IA_re = 0.0, IA_im = 0.0;
            double IPhi_re = 0.0, IPhi_im = 0.0;
            for (size_t q = 0; q < n_qp; q++) {
                double wq    = w_e_all[e * n_qp + q];
                double phiq  = sing_val_all[e * n_qp + q];
                double dphiq = sing_dval_all[e * n_qp + q];
                const double *pq = &pos_e_all[(e * n_qp + q) * 3];
                double wq_phi  = wq * phiq;
                double wq_dphi = wq * dphiq;
                for (size_t r = 0; r < n_qp; r++) {
                    double wr    = w_e_all[f * n_qp + r];
                    double phir  = sing_val_all[f * n_qp + r];
                    double dphir = sing_dval_all[f * n_qp + r];
                    const double *pr = &pos_e_all[(f * n_qp + r) * 3];
                    double dx = pq[0] - pr[0];
                    double dy = pq[1] - pr[1];
                    double dz = pq[2] - pr[2];
                    double R = std::sqrt(dx*dx + dy*dy + dz*dz + a_squared);
                    double phase = -k * R;
                    double iR_4pi = inv_4pi / R;
                    double Gre = std::cos(phase) * iR_4pi;
                    double Gim = std::sin(phase) * iR_4pi;
                    double wprod_A   = wq_phi  * (wr * phir);
                    double wprod_Phi = wq_dphi * (wr * dphir);
                    IA_re   += wprod_A * Gre;
                    IA_im   += wprod_A * Gim;
                    IPhi_re += wprod_Phi * Gre;
                    IPhi_im += wprod_Phi * Gim;
                }
            }
            // Z = j*ωμ*td*I_A + I_Phi/(jωε)
            // j*ωμ * (re + j im) = -ωμ im + j ωμ re
            // (re + j im) / (j ωε) = im/(ωε) - j re/(ωε)
            double Zre = -omega_mu * td * IA_im + IPhi_im * inv_omega_eps;
            double Zim =  omega_mu * td * IA_re - IPhi_re * inv_omega_eps;
            std::complex<double> Z_val(Zre, Zim);
            zee_v(e, f) = Z_val;
            if (e != f) zee_v(f, e) = Z_val;
        }
    }

    // -----------------------------------------------------------------
    // (Z_pe, Z_ep) assembly. For each polynomial basis m, for each wing of m,
    // for each enrichment e: integrate (poly_m vs Φ_e) and (Φ_e vs poly_m).
    //
    // Z_pe[m, e] integrates with quad-i = polynomial axis, quad-j = singular.
    // Z_ep[e, m] integrates with quad-i = singular axis, quad-j = polynomial.
    // The kernel G is symmetric so the two yield identical sums in exact
    // arithmetic; floating-point rounding is bit-for-bit identical given
    // matching summation order, which we deliberately mirror below.
    //
    // OpenMP parallelizes over m. Z_pe is row-disjoint by m; Z_ep is column-
    // disjoint by m. No reductions needed.
    // -----------------------------------------------------------------
    #pragma omp parallel
    {
        // Per-thread scratch for the polynomial-basis quad-point values.
        std::vector<double> pos_m(n_qp * 3);
        std::vector<double> poly_val(n_qp);
        std::vector<double> poly_dval(n_qp);
        std::vector<double> w_m(n_qp);

        #pragma omp for schedule(static)
        for (size_t m = 0; m < n_poly; m++) {
            for (size_t e = 0; e < n_enrich; e++) {
                zpe_v(m, e) = std::complex<double>(0.0, 0.0);
                zep_v(e, m) = std::complex<double>(0.0, 0.0);
            }
            for (size_t w = 0; w < n_wings; w++) {
                // Skip inactive wings (all-zero polynomial coefficients).
                bool any_nz = false;
                for (size_t p = 0; p < d_plus_1; p++) {
                    if (polys_v(m, w, p) != 0.0) { any_nz = true; break; }
                }
                if (!any_nz) continue;

                int64_t seg_m = ss_v(m, w);
                double hm = h_v(seg_m);

                for (size_t q = 0; q < n_qp; q++) {
                    double t = t01_v(q);
                    double u_arc = t * hm;
                    // Horner over polys_v(m, w, :) — evaluate poly value and derivative.
                    double pv = 0.0, dv = 0.0;
                    // value: P(u) = Σ c_p u^p ; deriv: Σ p·c_p u^(p-1)
                    // Evaluate as: pv = c_D ; for p = D-1..0: pv = pv*u + c_p
                    // and dv with Horner on (p+1)*c_{p+1}: dv = D·c_D ; ...
                    pv = polys_v(m, w, d_plus_1 - 1);
                    dv = (double)(d_plus_1 - 1) * polys_v(m, w, d_plus_1 - 1);
                    for (size_t pp = d_plus_1 - 1; pp-- > 0; ) {
                        pv = pv * u_arc + polys_v(m, w, pp);
                        if (pp >= 1) {
                            dv = dv * u_arc + (double)pp * polys_v(m, w, pp);
                        }
                    }
                    poly_val[q] = pv;
                    poly_dval[q] = dv;
                    w_m[q] = w01_v(q) * hm;
                    pos_m[q*3 + 0] = (1.0 - t) * sl_v(seg_m, 0) + t * sr_v(seg_m, 0);
                    pos_m[q*3 + 1] = (1.0 - t) * sl_v(seg_m, 1) + t * sr_v(seg_m, 1);
                    pos_m[q*3 + 2] = (1.0 - t) * sl_v(seg_m, 2) + t * sr_v(seg_m, 2);
                }

                for (size_t e = 0; e < n_enrich; e++) {
                    int64_t seg_e = seg_e_arr[e];
                    double td_me = tdot(seg_m, seg_e);
                    double td_em = tdot(seg_e, seg_m);

                    // Z_pe[m, e]: i = m-axis, j = e-axis.
                    double pe_IA_re = 0.0, pe_IA_im = 0.0;
                    double pe_IP_re = 0.0, pe_IP_im = 0.0;
                    // Z_ep[e, m]: i = e-axis, j = m-axis.
                    double ep_IA_re = 0.0, ep_IA_im = 0.0;
                    double ep_IP_re = 0.0, ep_IP_im = 0.0;

                    for (size_t q = 0; q < n_qp; q++) {
                        double wmq      = w_m[q];
                        double pvq      = poly_val[q];
                        double dvq      = poly_dval[q];
                        const double *pmq = &pos_m[q*3];

                        double weq_eax  = w_e_all[e * n_qp + q];
                        double phiq_eax = sing_val_all[e * n_qp + q];
                        double dphiq_eax= sing_dval_all[e * n_qp + q];
                        const double *peq_eax = &pos_e_all[(e * n_qp + q) * 3];

                        double wmq_pv  = wmq * pvq;
                        double wmq_dv  = wmq * dvq;
                        double weq_phi = weq_eax * phiq_eax;
                        double weq_dphi= weq_eax * dphiq_eax;

                        for (size_t r = 0; r < n_qp; r++) {
                            // -- Z_pe leg: i on m, j on e --
                            {
                                double wer      = w_e_all[e * n_qp + r];
                                double phir     = sing_val_all[e * n_qp + r];
                                double dphir    = sing_dval_all[e * n_qp + r];
                                const double *per = &pos_e_all[(e * n_qp + r) * 3];
                                double dx = pmq[0] - per[0];
                                double dy = pmq[1] - per[1];
                                double dz = pmq[2] - per[2];
                                double R = std::sqrt(dx*dx + dy*dy + dz*dz + a_squared);
                                double phase = -k * R;
                                double iR_4pi = inv_4pi / R;
                                double Gre = std::cos(phase) * iR_4pi;
                                double Gim = std::sin(phase) * iR_4pi;
                                double wprod_A   = wmq_pv * (wer * phir);
                                double wprod_Phi = wmq_dv * (wer * dphir);
                                pe_IA_re += wprod_A   * Gre;
                                pe_IA_im += wprod_A   * Gim;
                                pe_IP_re += wprod_Phi * Gre;
                                pe_IP_im += wprod_Phi * Gim;
                            }
                            // -- Z_ep leg: i on e, j on m --
                            {
                                double wmr      = w_m[r];
                                double pvr      = poly_val[r];
                                double dvr      = poly_dval[r];
                                const double *pmr = &pos_m[r*3];
                                double dx = peq_eax[0] - pmr[0];
                                double dy = peq_eax[1] - pmr[1];
                                double dz = peq_eax[2] - pmr[2];
                                double R = std::sqrt(dx*dx + dy*dy + dz*dz + a_squared);
                                double phase = -k * R;
                                double iR_4pi = inv_4pi / R;
                                double Gre = std::cos(phase) * iR_4pi;
                                double Gim = std::sin(phase) * iR_4pi;
                                double wprod_A   = weq_phi  * (wmr * pvr);
                                double wprod_Phi = weq_dphi * (wmr * dvr);
                                ep_IA_re += wprod_A   * Gre;
                                ep_IA_im += wprod_A   * Gim;
                                ep_IP_re += wprod_Phi * Gre;
                                ep_IP_im += wprod_Phi * Gim;
                            }
                        }
                    }
                    double Zpe_re = -omega_mu * td_me * pe_IA_im + pe_IP_im * inv_omega_eps;
                    double Zpe_im =  omega_mu * td_me * pe_IA_re - pe_IP_re * inv_omega_eps;
                    double Zep_re = -omega_mu * td_em * ep_IA_im + ep_IP_im * inv_omega_eps;
                    double Zep_im =  omega_mu * td_em * ep_IA_re - ep_IP_re * inv_omega_eps;
                    zpe_v(m, e) += std::complex<double>(Zpe_re, Zpe_im);
                    zep_v(e, m) += std::complex<double>(Zep_re, Zep_im);
                }
            }
        }
    }  // end omp parallel

    return std::make_tuple(Z_pe, Z_ep, Z_ee);
}


// ---------------------------------------------------------------------------
// Fresnel field-dyad projection tail — the reflection-coefficient finite
// ground (NEC IPERF=0). Called by the EXTENDED-kernel refl entry point
// (`sinusoidal_field_tensor_ek_refl`, momwire#259); the reduced one
// (`sinusoidal_field_tensor_refl`) carries a byte-frozen open-coded copy of
// exactly this algebra — see the note at its REFL branch for why it is not
// routed through here.
//
// At each (observer m, IMAGE source n) pair the specular ray's incidence
// cosine cos θ fixes the Fresnel coefficients (principal-branch sqrt, matching
// `_ground_refl.fresnel_rho`)
//   ρ_v = (ε̃ cosθ − √(ε̃ − sin²θ)) / (ε̃ cosθ + √(ε̃ − sin²θ))
//   ρ_h = (   cosθ − √(ε̃ − sin²θ)) / (   cosθ + √(ε̃ − sin²θ))
// and the dyad D = ρ_v(I − p̂p̂) − ρ_h p̂p̂ collapses under the observer
// projection to
//   t_m · D · E = ρ_v·(t_m·E) − (ρ_v + ρ_h)·(t_m·p̂)·(E·p̂)
// with both scalars read off the UNPROJECTED (E_z, E_ρ) tables:
//   t_m·E = td·E_z + rho_proj·E_ρ      (the plain projection)
//   E·p̂  = (t_n·p̂)·E_z + (ρ̂·p̂)·E_ρ    (t_img·p̂ = t_n·p̂; ρ̂ = rho_vec/rho_eval)
//
// That is exactly why one tail serves both kernels: the extended kernel
// changes WHAT E_z/E_ρ are, not how they are weighted, and the four geometric
// factors fed in (td, rho_proj, t_n·p̂, ρ̂·p̂) are EFLD's, computed before NEC
// picks between EKSC and EKSCX and therefore identical in both — in
// particular ρ̂ and rho_proj use the UNSWAPPED radial distance RHX, never
// EKSCX's ordered RH. The numpy side says the same thing by construction:
// `_field_tensor_image_refl`'s `_project_weighted` is ONE expression, applied
// to whichever tables `_field_components_bcast` returned, and it is the oracle
// for the term placement below.
struct ReflPair {
    std::complex<double> rho_v;
    std::complex<double> rvh;  // ρ_v + ρ_h — → 0 in the PEC limit
    double tm_p;               // t_m · p̂
    double tn_p;               // t_n · p̂
    double rho_p;              // ρ̂ · p̂
    double td;                 // t_m · t_n
    double rho_proj;           // (rho_vec · t_m) / rho_eval
};

static inline ReflPair refl_pair(std::complex<double> eps_t, double ct,
                                 double tm_p, double tn_p, double rho_p,
                                 double td, double rho_proj) {
    std::complex<double> root = std::sqrt(eps_t - (1.0 - ct * ct));
    std::complex<double> rho_v = (eps_t * ct - root) / (eps_t * ct + root);
    std::complex<double> rho_h = (ct - root) / (ct + root);
    ReflPair p;
    p.rho_v = rho_v;
    p.rvh = rho_v + rho_h;
    p.tm_p = tm_p;
    p.tn_p = tn_p;
    p.rho_p = rho_p;
    p.td = td;
    p.rho_proj = rho_proj;
    return p;
}

static inline std::complex<double> refl_project(const ReflPair &p,
                                                std::complex<double> Ez,
                                                std::complex<double> Erho) {
    std::complex<double> tm_E = p.td * Ez + p.rho_proj * Erho;
    std::complex<double> E_p = p.tn_p * Ez + p.rho_p * Erho;
    return p.rho_v * tm_E - p.rvh * (p.tm_p * E_p);
}


// Sinusoidal-basis (NEC2 three-term) tangential-field tensor.
//
// For each (m=obs, n=src) pair of segments, compute the three scalar tensors
//   Phi_const[m, n] = ŝ_m · E^const_n(c_m)
//   Phi_sin  [m, n] = ŝ_m · E^sin_n  (c_m)
//   Phi_cos  [m, n] = ŝ_m · E^cos_n  (c_m)
// where the source's local frame is centered on segment n with z-axis along
// src_tangents[n]; the const/sin/cos sources are I(z')=1 / sin(k z') /
// cos(k z') over z' ∈ [-H_n, +H_n], H_n = h_n/2. Result is in NEC's natural-
// arc convention (σ accounting is the caller's job).
//
// Closed forms for the const-source `int G_0 dz'` are 1/r_0 singularity
// extraction: ∫ 1/r_0 dz' = arcsinh((H-z)/ρ) - arcsinh((-H-z)/ρ); regular
// remainder via Gauss-Legendre on the (G_0 - 1/r_0) integrand. Sin/cos
// sources are fully closed-form per Eqs 76-79 of the LLNL theory manual
// (mirrored by the numpy reference in src/momwire/sinusoidal.py _field_tensor).
//
// Parallelism: each (m, n) pair is independent. OpenMP collapse(2) over the
// (m, n) grid; per-n constants (H_n, sin(kH_n), cos(kH_n)) are precomputed
// outside the parallel region.
//
// REFL=true is the reflection-coefficient finite-ground variant
// (`sinusoidal_field_tensor_refl`): the caller passes the k-independent
// per-pair specular tables (cos_th, px, py, tm_p, tn_p — see
// _ground_refl.specular_ray_tables and SinusoidalSolver._image_refl_prep)
// plus the complex ε̃, and the projection tail applies NEC's field dyad
//     t_m · D · E = ρ_v·(t_m·E) − (ρ_v + ρ_h)·(t_m·p̂)·(E·p̂)
// with the Fresnel ρ_v/ρ_h computed in-kernel at each pair's specular
// angle (principal-branch sqrt, matching _ground_refl.fresnel_rho). The
// unprojected E_z/E_ρ physics (Eqs 76-79) is shared with REFL=false —
// this template exists so the reflection path stops paying the pure-numpy
// `_field_components` fill (~35× the kernel cost, super-quadratic once
// its (M,N,n_qp) temporaries fall out of cache).
template<bool REFL>
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_field_tensor_impl(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,
    double a, double k, double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<double, py::array::c_style | py::array::forcecast> cos_th,
    py::array_t<double, py::array::c_style | py::array::forcecast> px_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> py_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> tm_p,
    py::array_t<double, py::array::c_style | py::array::forcecast> tn_p,
    std::complex<double> eps_t,
    uintptr_t cancel_flag = 0
) {
    auto oc = obs_centers.unchecked<2>();
    auto ot = obs_tangents.unchecked<2>();
    auto sc = src_centers.unchecked<2>();
    auto st = src_tangents.unchecked<2>();
    auto sh = seg_h.unchecked<1>();
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();

    if (oc.shape(1) != 3 || ot.shape(1) != 3 ||
        sc.shape(1) != 3 || st.shape(1) != 3) {
        throw std::runtime_error("center/tangent arrays must have shape (N, 3)");
    }
    if (oc.shape(0) != ot.shape(0)) {
        throw std::runtime_error("obs_centers and obs_tangents must have matching N");
    }
    if (sc.shape(0) != st.shape(0) || sc.shape(0) != sh.shape(0)) {
        throw std::runtime_error("src arrays must all have matching N");
    }
    if (glt.shape(0) != glw.shape(0)) {
        throw std::runtime_error("gl_t and gl_w must have matching length");
    }

    size_t M = oc.shape(0);
    size_t N = sc.shape(0);
    size_t n_qp = glt.shape(0);

    // REFL-only per-pair specular tables, flat (M, N) row-major.
    const double *cth_p = nullptr, *px_p = nullptr, *py_p = nullptr;
    const double *tmp_p = nullptr, *tnp_p = nullptr;
    if (REFL) {
        for (const auto *t : {&cos_th, &px_t, &py_t, &tm_p, &tn_p}) {
            if ((*t).ndim() != 2 || (size_t)(*t).shape(0) != M ||
                (size_t)(*t).shape(1) != N) {
                throw std::runtime_error(
                    "refl tables must all have shape (M_obs, N_src)");
            }
        }
        cth_p = cos_th.data();
        px_p = px_t.data();
        py_p = py_t.data();
        tmp_p = tm_p.data();
        tnp_p = tn_p.data();
    }

    py::array_t<std::complex<double>> Phi_const({M, N});
    py::array_t<std::complex<double>> Phi_sin({M, N});
    py::array_t<std::complex<double>> Phi_cos({M, N});
    auto pc = Phi_const.mutable_unchecked<2>();
    auto ps = Phi_sin.mutable_unchecked<2>();
    auto pco = Phi_cos.mutable_unchecked<2>();

    // Phase 0: release the GIL for the heavy compute region below.
    py::gil_scoped_release release;

    // Per-source-segment precompute: H_n = h_n/2, sin(kH_n), cos(kH_n).
    std::vector<double> H_n(N), sin_kH(N), cos_kH(N);
    for (size_t n = 0; n < N; n++) {
        H_n[n] = 0.5 * sh(n);
        sin_kH[n] = std::sin(k * H_n[n]);
        cos_kH[n] = std::cos(k * H_n[n]);
    }

    // Cache GL nodes/weights in std::vector so the per-pair loops avoid
    // bouncing through the pybind11 unchecked accessor in the inner loop.
    std::vector<double> glt_v(n_qp), glw_v(n_qp);
    for (size_t q = 0; q < n_qp; q++) {
        glt_v[q] = glt(q);
        glw_v[q] = glw(q);
    }

    // Scalar prefactors. pref_z = +j eta / (4 pi k); pref_rho_const has the
    // same form (the per-pair 1/rho_eval scaling lives inside pref_rho).
    const double four_pi_k = 4.0 * M_PI * k;
    const double pref_z_im = eta / four_pi_k;
    const double pref_rho_const_im = -eta / four_pi_k;
    const double a_sq = a * a;

    // The oscillatory sincos is batched across all N source segments (per
    // observer m) into one flat buffer so it vectorizes — the per-pair scalar
    // calls were unvectorizable. Three stages per m: (A) geometry + phases,
    // (B) one omp-simd sincos sweep over the buffer, (C) assembly.
    //
    // Phases per source segment: 2 boundary (r0_2, r0_1) + n_qp quadrature nodes.
    const size_t S = n_qp + 2;
    const size_t P = N * S;

    PYSIM_CANCEL_SETUP(cancel_flag);
    #pragma omp parallel for schedule(static)
    for (size_t m = 0; m < M; m++) {
        PYSIM_CANCEL_POLL();
        // Per-iteration scratch (per-thread under the parallel-for). Sizes are
        // tiny (P ~ N*(n_qp+2)); allocation cost is negligible next to the
        // sincos + assembly work it feeds.
        std::vector<double> ph(P), cphb(P), sphb(P);
        std::vector<double> rho_eval_a(N), dz1_a(N), dz2_a(N),
                            r0_1_a(N), r0_2_a(N), td_a(N), rpf_a(N);
        std::vector<double> r0q_inv_a(N * n_qp);
        // REFL: ρ̂·p̂ per source segment for this observer row (ρ̂ =
        // rho_vec/rho_eval; p̂ is horizontal so only x/y contribute).
        std::vector<double> rho_p_a(REFL ? N : 0);

        double cmx = oc(m, 0), cmy = oc(m, 1), cmz = oc(m, 2);
        double tmx = ot(m, 0), tmy = ot(m, 1), tmz = ot(m, 2);

        // ---- Stage A: geometry + phase generation -------------------------
        for (size_t n = 0; n < N; n++) {
            double cnx = sc(n, 0), cny = sc(n, 1), cnz = sc(n, 2);
            double tnx = st(n, 0), tny = st(n, 1), tnz = st(n, 2);
            double rvx = cmx - cnx, rvy = cmy - cny, rvz = cmz - cnz;
            double z_eval = rvx * tnx + rvy * tny + rvz * tnz;
            double rho_vx = rvx - z_eval * tnx;
            double rho_vy = rvy - z_eval * tny;
            double rho_vz = rvz - z_eval * tnz;
            double rho_axis = std::sqrt(rho_vx*rho_vx + rho_vy*rho_vy + rho_vz*rho_vz);
            double rho_eval = std::sqrt(rho_axis*rho_axis + a_sq);
            double td = tmx*tnx + tmy*tny + tmz*tnz;
            double rho_dot_tobs = rho_vx*tmx + rho_vy*tmy + rho_vz*tmz;

            double H = H_n[n];
            double dz2 = z_eval - H;
            double dz1 = z_eval + H;
            double r0_2 = std::sqrt(rho_eval*rho_eval + dz2*dz2);
            double r0_1 = std::sqrt(rho_eval*rho_eval + dz1*dz1);

            rho_eval_a[n] = rho_eval;
            dz1_a[n] = dz1; dz2_a[n] = dz2;
            r0_1_a[n] = r0_1; r0_2_a[n] = r0_2;
            td_a[n] = td; rpf_a[n] = rho_dot_tobs / rho_eval;
            if (REFL) {
                size_t mn = m * N + n;
                rho_p_a[n] = (rho_vx * px_p[mn] + rho_vy * py_p[mn]) / rho_eval;
            }

            size_t base = n * S;
            ph[base + 0] = -k * r0_2;
            ph[base + 1] = -k * r0_1;
            for (size_t q = 0; q < n_qp; q++) {
                double z_q = H * glt_v[q];
                double dz_q = z_eval - z_q;
                double r0_q = std::sqrt(rho_eval*rho_eval + dz_q*dz_q);
                ph[base + 2 + q] = -k * r0_q;
                r0q_inv_a[n * n_qp + q] = 1.0 / r0_q;
            }
        }

        // ---- Stage B: vectorized sincos over every phase ------------------
        // Split cos and sin into separate omp-simd loops (libmvec has no vector
        // sincos) so each body stays vectorizable to _ZGVdN4v_{cos,sin}
        // (AVX2, 4 doubles per call).
        PYSIM_OMP_SIMD()
        for (size_t i = 0; i < P; i++) cphb[i] = std::cos(ph[i]);
        PYSIM_OMP_SIMD()
        for (size_t i = 0; i < P; i++) sphb[i] = std::sin(ph[i]);

        // ---- Stage C: assembly --------------------------------------------
        for (size_t n = 0; n < N; n++) {
            size_t base = n * S;
            double cph_2 = cphb[base + 0], sph_2 = sphb[base + 0];
            double cph_1 = cphb[base + 1], sph_1 = sphb[base + 1];
            double rho_eval = rho_eval_a[n];
            double dz1 = dz1_a[n], dz2 = dz2_a[n];
            double r0_1 = r0_1_a[n], r0_2 = r0_2_a[n];
            double td = td_a[n], rho_proj_factor = rpf_a[n];
            double H = H_n[n];
            double inv_r0_2 = 1.0 / r0_2;
            double inv_r0_1 = 1.0 / r0_1;
            double G0_2_re = cph_2 * inv_r0_2, G0_2_im = sph_2 * inv_r0_2;
            double G0_1_re = cph_1 * inv_r0_1, G0_1_im = sph_1 * inv_r0_1;

            // (1 + j k r0) / r0² split into re/im
            double inv_r0_2_sq = inv_r0_2 * inv_r0_2;
            double inv_r0_1_sq = inv_r0_1 * inv_r0_1;
            double one_jkr_2_re = inv_r0_2_sq;
            double one_jkr_2_im = k * r0_2 * inv_r0_2_sq;
            double one_jkr_1_re = inv_r0_1_sq;
            double one_jkr_1_im = k * r0_1 * inv_r0_1_sq;

            // ---- Const source -------------------------------------------------
            // Erho_const = pref_rho_const * (
            //     (1 + j k r0_2) * rho_eval * G0_2 / r0_2²
            //   - (1 + j k r0_1) * rho_eval * G0_1 / r0_1²
            // )
            // Let A_2 = (1 + jk r0_2) / r0_2² = one_jkr_2 (complex).
            // term_2 = A_2 * G0_2 — complex product.
            auto cmul = [](double ar, double ai, double br, double bi,
                           double &cr, double &ci) {
                cr = ar*br - ai*bi;
                ci = ar*bi + ai*br;
            };

            double term_const2_re, term_const2_im;
            cmul(one_jkr_2_re, one_jkr_2_im, G0_2_re, G0_2_im,
                 term_const2_re, term_const2_im);
            double term_const1_re, term_const1_im;
            cmul(one_jkr_1_re, one_jkr_1_im, G0_1_re, G0_1_im,
                 term_const1_re, term_const1_im);
            double rho_diff_re = rho_eval * (term_const2_re - term_const1_re);
            double rho_diff_im = rho_eval * (term_const2_im - term_const1_im);
            // pref_rho_const = j * pref_rho_const_im  (pure imaginary scalar)
            double Erho_const_re = -pref_rho_const_im * rho_diff_im;
            double Erho_const_im =  pref_rho_const_im * rho_diff_re;

            // u2, u1, int_inv_r0.  H - z_eval = -dz2;  -H - z_eval = -dz1.
            double inv_rho_eval = 1.0 / rho_eval;
            double u2 = -dz2 * inv_rho_eval;
            double u1 = -dz1 * inv_rho_eval;
            double int_inv_r0 = std::asinh(u2) - std::asinh(u1);

            // Quadrature for the smooth remainder of int_G0:
            //   reg(q) = (exp(-jk r0_q) - 1) / r0_q,   r0_q = sqrt(ρ² + (z - H gx[q])²)
            //   int_reg = H * Σ_q reg(q) * gw[q]
            double int_reg_re = 0.0, int_reg_im = 0.0;
            for (size_t q = 0; q < n_qp; q++) {
                double cph_q = cphb[base + 2 + q];
                double sph_q = sphb[base + 2 + q];
                double inv_r0_q = r0q_inv_a[n * n_qp + q];
                // (exp(jphase) - 1) / r0
                double reg_re = (cph_q - 1.0) * inv_r0_q;
                double reg_im = sph_q * inv_r0_q;
                int_reg_re += reg_re * glw_v[q];
                int_reg_im += reg_im * glw_v[q];
            }
            int_reg_re *= H;
            int_reg_im *= H;
            double int_G0_re = int_inv_r0 + int_reg_re;
            double int_G0_im = int_reg_im;

            // Ez_const_boundary = (1+jk r0_2) dz2 G0_2 / r0_2² - (1+jk r0_1) dz1 G0_1 / r0_1²
            double Ez_boundary_re = dz2 * term_const2_re - dz1 * term_const1_re;
            double Ez_boundary_im = dz2 * term_const2_im - dz1 * term_const1_im;
            double k_sq = k * k;
            // Ez_const = -pref_z * (Ez_boundary + k² int_G0). pref_z = j * pref_z_im.
            double inside_re = Ez_boundary_re + k_sq * int_G0_re;
            double inside_im = Ez_boundary_im + k_sq * int_G0_im;
            // Multiply by -j * pref_z_im
            double Ez_const_re =  pref_z_im * inside_im;
            double Ez_const_im = -pref_z_im * inside_re;

            // ---- Sine source (Eq 76, 77) --------------------------------------
            double sin2 = sin_kH[n];
            double cos2 = cos_kH[n];
            double sin1 = -sin2;
            double cos1 =  cos2;

            // bracket_sin_2 = G0_2 * (k dz2 cos2 + (1 - dz2² (1+jk r0_2)/r0_2²) sin2)
            double inner_2_re = 1.0 - dz2*dz2 * one_jkr_2_re;
            double inner_2_im =     - dz2*dz2 * one_jkr_2_im;
            double bracket_sin_2_re = k*dz2*cos2 + inner_2_re*sin2;
            double bracket_sin_2_im =              inner_2_im*sin2;
            double bsin2_re, bsin2_im;
            cmul(G0_2_re, G0_2_im, bracket_sin_2_re, bracket_sin_2_im,
                 bsin2_re, bsin2_im);

            double inner_1_re = 1.0 - dz1*dz1 * one_jkr_1_re;
            double inner_1_im =     - dz1*dz1 * one_jkr_1_im;
            double bracket_sin_1_re = k*dz1*cos1 + inner_1_re*sin1;
            double bracket_sin_1_im =              inner_1_im*sin1;
            double bsin1_re, bsin1_im;
            cmul(G0_1_re, G0_1_im, bracket_sin_1_re, bracket_sin_1_im,
                 bsin1_re, bsin1_im);
            double Erho_sin_inner_re = bsin2_re - bsin1_re;
            double Erho_sin_inner_im = bsin2_im - bsin1_im;
            // pref_rho = -j eta / (4 pi k rho_eval)
            double pref_rho_im = pref_rho_const_im * inv_rho_eval;
            double Erho_sin_re = -pref_rho_im * Erho_sin_inner_im;
            double Erho_sin_im =  pref_rho_im * Erho_sin_inner_re;

            // bracket_sin_z = G0 * (k cos - (1+jk r0) dz / r0² sin)
            double bracket_sin_z_2_re = k*cos2 - dz2*one_jkr_2_re*sin2;
            double bracket_sin_z_2_im =        - dz2*one_jkr_2_im*sin2;
            double bszin2_re, bszin2_im;
            cmul(G0_2_re, G0_2_im, bracket_sin_z_2_re, bracket_sin_z_2_im,
                 bszin2_re, bszin2_im);
            double bracket_sin_z_1_re = k*cos1 - dz1*one_jkr_1_re*sin1;
            double bracket_sin_z_1_im =        - dz1*one_jkr_1_im*sin1;
            double bszin1_re, bszin1_im;
            cmul(G0_1_re, G0_1_im, bracket_sin_z_1_re, bracket_sin_z_1_im,
                 bszin1_re, bszin1_im);
            double Ez_sin_inner_re = bszin2_re - bszin1_re;
            double Ez_sin_inner_im = bszin2_im - bszin1_im;
            // pref_z = +j pref_z_im
            double Ez_sin_re = -pref_z_im * Ez_sin_inner_im;
            double Ez_sin_im =  pref_z_im * Ez_sin_inner_re;

            // ---- Cosine source ------------------------------------------------
            // bracket_cos_2 = G0_2 * (-k dz2 sin2 + (1 - dz2² (1+jk r0_2)/r0_2²) cos2)
            double bracket_cos_2_re = -k*dz2*sin2 + inner_2_re*cos2;
            double bracket_cos_2_im =              inner_2_im*cos2;
            double bcos2_re, bcos2_im;
            cmul(G0_2_re, G0_2_im, bracket_cos_2_re, bracket_cos_2_im,
                 bcos2_re, bcos2_im);
            double bracket_cos_1_re = -k*dz1*sin1 + inner_1_re*cos1;
            double bracket_cos_1_im =              inner_1_im*cos1;
            double bcos1_re, bcos1_im;
            cmul(G0_1_re, G0_1_im, bracket_cos_1_re, bracket_cos_1_im,
                 bcos1_re, bcos1_im);
            double Erho_cos_inner_re = bcos2_re - bcos1_re;
            double Erho_cos_inner_im = bcos2_im - bcos1_im;
            double Erho_cos_re = -pref_rho_im * Erho_cos_inner_im;
            double Erho_cos_im =  pref_rho_im * Erho_cos_inner_re;

            double bracket_cos_z_2_re = -k*sin2 - dz2*one_jkr_2_re*cos2;
            double bracket_cos_z_2_im =         - dz2*one_jkr_2_im*cos2;
            double bczin2_re, bczin2_im;
            cmul(G0_2_re, G0_2_im, bracket_cos_z_2_re, bracket_cos_z_2_im,
                 bczin2_re, bczin2_im);
            double bracket_cos_z_1_re = -k*sin1 - dz1*one_jkr_1_re*cos1;
            double bracket_cos_z_1_im =         - dz1*one_jkr_1_im*cos1;
            double bczin1_re, bczin1_im;
            cmul(G0_1_re, G0_1_im, bracket_cos_z_1_re, bracket_cos_z_1_im,
                 bczin1_re, bczin1_im);
            double Ez_cos_inner_re = bczin2_re - bczin1_re;
            double Ez_cos_inner_im = bczin2_im - bczin1_im;
            double Ez_cos_re = -pref_z_im * Ez_cos_inner_im;
            double Ez_cos_im =  pref_z_im * Ez_cos_inner_re;

            if (REFL) {
                // Fresnel ρ_v / ρ_h at this pair's specular angle
                // (principal-branch sqrt — matches _ground_refl.fresnel_rho),
                // then NEC's field dyad:
                //   t_m · D · E = ρ_v·(t_m·E) − (ρ_v + ρ_h)·(t_m·p̂)·(E·p̂)
                //   t_m·E = td·E_z + rho_proj·E_ρ
                //   E·p̂  = (t_n·p̂)·E_z + (ρ̂·p̂)·E_ρ
                //
                // `refl_pair` / `refl_project` above are this same algebra,
                // and the extended kernel (momwire#259) calls them — but THIS
                // copy stays open-coded on purpose. Routing the reduced path
                // through the helpers is a byte-level change: the factors then
                // reach the multiply-adds through a struct and GCC contracts
                // them differently, which measured out at ≤3.6e-15 on the
                // tensor and 7.1e-14 on a grounded Z. The reduced refl path is
                // frozen armor (#233 gate 4 / #259 gate 3), so the duplication
                // is the cheaper price. Both copies are independently anchored
                // to the same numpy oracle, `_field_tensor_image_refl`'s
                // `_project_weighted` — that, not code sharing, is what keeps
                // them from drifting.
                size_t mn = m * N + n;
                double ct = cth_p[mn];
                std::complex<double> root =
                    std::sqrt(eps_t - (1.0 - ct * ct));
                std::complex<double> rho_v =
                    (eps_t * ct - root) / (eps_t * ct + root);
                std::complex<double> rho_h = (ct - root) / (ct + root);
                std::complex<double> rvh = rho_v + rho_h;
                double tmp = tmp_p[mn], tnp = tnp_p[mn], rp = rho_p_a[n];

                auto weighted = [&](double Ez_re, double Ez_im,
                                    double Erho_re, double Erho_im) {
                    std::complex<double> Ez(Ez_re, Ez_im);
                    std::complex<double> Erho(Erho_re, Erho_im);
                    std::complex<double> tm_E =
                        td * Ez + rho_proj_factor * Erho;
                    std::complex<double> E_p = tnp * Ez + rp * Erho;
                    return rho_v * tm_E - rvh * (tmp * E_p);
                };
                pc(m, n) = weighted(Ez_const_re, Ez_const_im,
                                    Erho_const_re, Erho_const_im);
                ps(m, n) = weighted(Ez_sin_re, Ez_sin_im,
                                    Erho_sin_re, Erho_sin_im);
                pco(m, n) = weighted(Ez_cos_re, Ez_cos_im,
                                     Erho_cos_re, Erho_cos_im);
            } else {
                // Project to obs tangent: Phi = td * Ez + rho_proj * Erho.
                pc(m, n) = std::complex<double>(
                    td * Ez_const_re + rho_proj_factor * Erho_const_re,
                    td * Ez_const_im + rho_proj_factor * Erho_const_im);
                ps(m, n) = std::complex<double>(
                    td * Ez_sin_re + rho_proj_factor * Erho_sin_re,
                    td * Ez_sin_im + rho_proj_factor * Erho_sin_im);
                pco(m, n) = std::complex<double>(
                    td * Ez_cos_re + rho_proj_factor * Erho_cos_re,
                    td * Ez_cos_im + rho_proj_factor * Erho_cos_im);
            }
        }
    }

    PYSIM_THROW_IF_ABORTED();
    return std::make_tuple(Phi_const, Phi_sin, Phi_cos);
}

// Thin non-template wrappers for pybind registration. The plain tensor
// passes empty refl tables; the refl variant forwards them.
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_field_tensor(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,
    double a, double k, double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    uintptr_t cancel_flag = 0
) {
    py::array_t<double> empty(std::vector<py::ssize_t>{0, 0});
    return sinusoidal_field_tensor_impl<false>(
        obs_centers, obs_tangents, src_centers, src_tangents, seg_h,
        a, k, eta, gl_t, gl_w,
        empty, empty, empty, empty, empty,
        std::complex<double>(0.0, 0.0), cancel_flag);
}

static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_field_tensor_refl(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,
    double a, double k, double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<double, py::array::c_style | py::array::forcecast> cos_th,
    py::array_t<double, py::array::c_style | py::array::forcecast> px,
    py::array_t<double, py::array::c_style | py::array::forcecast> py_,
    py::array_t<double, py::array::c_style | py::array::forcecast> tm_p,
    py::array_t<double, py::array::c_style | py::array::forcecast> tn_p,
    std::complex<double> eps_t,
    uintptr_t cancel_flag = 0
) {
    return sinusoidal_field_tensor_impl<true>(
        obs_centers, obs_tangents, src_centers, src_tangents, seg_h,
        a, k, eta, gl_t, gl_w,
        cos_th, px, py_, tm_p, tn_p, eps_t, cancel_flag);
}


// NEC's EXTENDED thin-wire kernel for the sinusoidal field tensor (#245).
//
// `sinusoidal_field_tensor` above is EKSC (nec2-1.2.1.2.f:3124-3166) — the
// REDUCED kernel, rearranged into per-endpoint brackets. This entry point is
// EKSCX (f.3170-3234): the same brackets, with the per-end reduced routine GX
// (f.4842-4852) swapped for the second-order tube-average GXX (f.4857-4897)
// wherever the fill's per-end gating says the source current really continues
// straight through that end, plus EKSCX's ungated (kB/2)² correction on the
// constant-current term. Theory manual Eqs 84-98 (Burke & Poggio Part I,
// pp. 30-34).
//
// This is a TRANSCRIPTION of `SinusoidalSolver._ek_end_gxx` / `_ek_end_gx` /
// `_extended_kernel_fields`, which stay the reference implementation and the
// oracle: tests/test_extended_kernel.py holds the two paths to 1e-13 on every
// gating branch. Nothing here may be "improved" relative to the numpy spelling
// without moving that oracle first — see the IRA note below.
//
// Two tables the reduced entry point does not take:
//   * `src_a` (N,) — the SOURCE segment's radius, NEC's BX = BI(J). The
//     reduced kernel only ever needs the OBSERVER's (necpp EFLD's `ai`, the
//     scalar `a` argument), which is why THAT entry point can serve mixed
//     per-wire radii one constant-radius observer-row run at a time. The O(a²)
//     expansion is about the SOURCE conductor's girth, so it needs the other
//     radius, per source, inside every run.
//   * `ind1` / `ind2` (N,) int8 — the per-source-end gating codes of
//     `_ek_gating` (f.2019-2053). Codes 0 and 1 take GXX, code 2 takes GX,
//     exactly as EKSCX routes at f.3193 (`IF( INX1.EQ.2) GOTO 3`) and f.3199.
//     They are computed in Python because they are a connectivity walk, not
//     arithmetic.
//
// EKSCX's IRA (f.3186-3192) is decided PER PAIR, in stage A, from this pair's
// own `rhx < src_a` — the same comparison that already orders (rh, b) right
// beside it. #245 could not do that: `_extended_kernel_fields` reduced the
// test to one `np.any` before calling `_ek_end_gxx`, so every pair took the
// same arm, and this kernel took a build-wide `want_swapped` scalar to
// reproduce that faithfully rather than silently repair it. momwire#258
// repaired it on both sides, so the scalar is GONE from this signature — the
// arm now rides the same local `sw` the ordering does, which is the only
// spelling in which the two cannot disagree at a knife-edge pair. Python
// detects the new arity through the `ek_ira_per_pair` module attribute below;
// a stale extension exports the symbols with the old arity and is refused
// there rather than called wrongly.
//
// Parallelism / staging follow `sinusoidal_field_tensor_impl` exactly: OpenMP
// over observer rows, per-row scratch, and the same three stages (geometry +
// phase generation, one omp-simd sincos sweep over the row's phase buffer,
// assembly). Nothing (M, N, n_qp)-sized is ever materialized, which is the
// other half of #245 — the numpy path's source-quadrature temporaries are what
// make large EK meshes memory-bound.
//
// REFL=true (momwire#259) is the same marriage the reduced kernel's REFL arm
// makes: EKSCX's E_z/E_ρ tables with the Fresnel field dyad applied instead of
// the plain tangential projection, from the same k-independent specular tables
// (cos_th, px, py, tm_p, tn_p) and the same complex ε̃. It closes the last
// numpy-speed extended-kernel path on the sinusoidal family — before it,
// `extended_kernel=True` with `ground_eps` (and no `ground_model=
// "sommerfeld"`) took the pure-numpy image block. PEC and Sommerfeld grounds
// need nothing here: the PEC image is `sinusoidal_field_tensor_ek` with
// mirrored sources, and the Sommerfeld image block is that same PEC tensor
// scaled by the scalar C₂, so both already ride REFL=false.

// Minimal POD complex for the extended-kernel inner loop. libstdc++'s
// std::complex<double> multiplication lowers to a libgcc `__muldc3` CALL (the
// Inf/NaN fixup path) at our flag set, and each pair here does ~60 complex
// products; the naive four-multiply form below is what the reduced kernel's
// `cmul` lambda already open-codes inline.
struct EkC {
    double re, im;
};
static inline EkC operator+(EkC a, EkC b) { return EkC{a.re + b.re, a.im + b.im}; }
static inline EkC operator-(EkC a, EkC b) { return EkC{a.re - b.re, a.im - b.im}; }
static inline EkC operator-(EkC a) { return EkC{-a.re, -a.im}; }
static inline EkC operator+(double s, EkC a) { return EkC{s + a.re, a.im}; }
static inline EkC operator-(EkC a, double s) { return EkC{a.re - s, a.im}; }
static inline EkC operator*(EkC a, EkC b) {
    return EkC{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
static inline EkC operator*(EkC a, double s) { return EkC{a.re * s, a.im * s}; }
static inline EkC operator*(double s, EkC a) { return EkC{s * a.re, s * a.im}; }
static inline EkC operator/(EkC a, double s) { return EkC{a.re / s, a.im / s}; }
// (j·s)·x. CON (f.3181) is pure imaginary, and numpy's complex product with a
// zero real part is bit-identical to this: 0·x is an exact zero and ±0 + y == y.
static inline EkC ek_mul_j(double s, EkC x) { return EkC{-s * x.im, s * x.re}; }

// The six per-end quantities EKSCX consumes, in NEC's argument order:
// (G1, G1P, G2, G2P, G3, GZP) — the extended scalar kernel (Eq 89), its
// z-derivative, the ρ-kernel and its ρ- and z-derivatives (Eqs 90-96), and the
// reduced-kernel z-derivative that only the constant-current correction uses.
struct EkEnd {
    EkC g1, g1p, g2, g2p, g3, gzp;
};

// GXX (f.4857-4897) — `SinusoidalSolver._ek_end_gxx`. `r2`/`r` are the already
// formed R² = zz² + rh² and R; `cph`/`sph` are cos(−kR)/sin(−kR) out of the
// batched sincos sweep. `rh >= b` is the caller's ordering (f.3186-3192), and
// `want_swapped` is THIS pair's IRA out of that same ordering (momwire#258).
static inline EkEnd ek_end_gxx(double k, double zz, double rh, double b,
                               double r2, double r, double cph, double sph,
                               bool want_swapped) {
    double kr = k * r;
    double kr2 = kr * kr;
    // C1, C2, C3 of f.4869-4871 — the polynomial coefficients of the
    // second-order Taylor terms of Eqs 86-88.
    EkC c1 = EkC{1.0, kr};
    EkC c2 = 3.0 * c1 - kr2;
    EkC c3 = EkC{6.0, kr} * kr2 - 15.0 * c1;
    double a2 = b * b;
    // T1, T2 of f.4867-4868: the two O(a²) shape factors of Eq 89.
    double rh2 = rh * rh;
    double r4 = r2 * r2;
    double t1 = 0.25 * a2 * rh2 / r4;
    double t2 = 0.5 * a2 / r2;
    EkC gz = EkC{cph / r, sph / r};  // reduced kernel e^{-jkR}/R
    // Eq 89: the circumferentially averaged kernel, ρ-flavoured (G2) and
    // z-flavoured (G1, which carries the extra -T2·C1 term).
    EkC g2 = gz * (1.0 + t1 * c2);
    EkC g1 = g2 - t2 * c1 * gz;
    EkC gzr = gz / r2;
    EkC g2p = gzr * (t1 * c3 - c1);
    EkC gzp_t = t2 * c2 * gzr;
    EkC g3 = g2p + gzp_t;
    EkEnd out;
    out.g1 = g1;
    out.g1p = g3 * zz;
    // GZP: the plain reduced-kernel z-derivative. It is a-independent and only
    // enters EKSCX's constant-current term, scaled by (ka/2)².
    out.gzp = (-zz) * c1 * gzr;
    if (want_swapped) {
        // IRA == 1 (f.4886-4896): observation point inside the conductor.
        double t2b = 0.5 * b;
        out.g2 = (-t2b) * c1 * gzr;
        EkC g2p_s = (t2b * gzr) * c2 / r2;
        out.g3 = rh2 * g2p_s - (b * gzr) * c1;
        out.g2p = g2p_s * zz;
    } else {
        // IRA == 0 (f.4879-4885), the ordinary case. `rh` is never zero in
        // momwire — it is at least the observer radius — so NEC's RH < 1e-10
        // guard at f.4881 is unreachable here.
        out.g3 = (g3 + gzp_t) * rh;
        out.g2 = g2 / rh;
        out.g2p = g2p * zz / rh;
    }
    return out;
}

// GX (f.4842-4852) repackaged into GXX's six-value contract —
// `SinusoidalSolver._ek_end_gx`. EKSCX's `INX == 2` arm (f.3195-3207) passes
// RHX, the ORIGINAL radial distance, not the possibly-swapped RH, and zeroes
// GZP so the end contributes nothing to the constant-current correction.
static inline EkEnd ek_end_gx(double zz, double rhx, double r2, double r,
                              double k, double cph, double sph) {
    double kr = k * r;
    EkC c1 = EkC{1.0, kr};
    EkC gz = EkC{cph / r, sph / r};
    EkC gp = (-c1) * gz / r2;
    EkC g1p = gp * zz;
    EkEnd out;
    out.g1 = gz;
    out.g1p = g1p;
    out.g2 = gz / rhx;
    out.g2p = g1p / rhx;
    out.g3 = gp * rhx;
    out.gzp = EkC{0.0, 0.0};
    return out;
}

// Per-(observer row, source) geometry carried from stage A to stage C.
struct EkGeomRow {
    double z_eval, rhx, rh, b, z1, z2;
    double r2_1, r_1, r2_2, r_2;
    double td, rpf;
    bool ext1, ext2;
    // EKSCX's IRA for THIS pair (momwire#258): the `rhx < src_a` that ordered
    // (rh, b) just above, carried through to the arm it selects.
    bool swapped;
};

template<bool REFL>
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_field_tensor_ek_impl(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,
    double a, double k, double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_a,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> ind1,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> ind2,
    py::array_t<double, py::array::c_style | py::array::forcecast> cos_th,
    py::array_t<double, py::array::c_style | py::array::forcecast> px_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> py_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> tm_p,
    py::array_t<double, py::array::c_style | py::array::forcecast> tn_p,
    std::complex<double> eps_t,
    uintptr_t cancel_flag = 0
) {
    auto oc = obs_centers.unchecked<2>();
    auto ot = obs_tangents.unchecked<2>();
    auto sc = src_centers.unchecked<2>();
    auto st = src_tangents.unchecked<2>();
    auto sh = seg_h.unchecked<1>();
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();
    auto sa = src_a.unchecked<1>();
    auto i1 = ind1.unchecked<1>();
    auto i2 = ind2.unchecked<1>();

    if (oc.shape(1) != 3 || ot.shape(1) != 3 ||
        sc.shape(1) != 3 || st.shape(1) != 3) {
        throw std::runtime_error("center/tangent arrays must have shape (N, 3)");
    }
    if (oc.shape(0) != ot.shape(0)) {
        throw std::runtime_error("obs_centers and obs_tangents must have matching N");
    }
    if (sc.shape(0) != st.shape(0) || sc.shape(0) != sh.shape(0) ||
        sc.shape(0) != sa.shape(0) || sc.shape(0) != i1.shape(0) ||
        sc.shape(0) != i2.shape(0)) {
        throw std::runtime_error(
            "src arrays (centers, tangents, seg_h, src_a, ind1, ind2) must all "
            "have matching N");
    }
    if (glt.shape(0) != glw.shape(0)) {
        throw std::runtime_error("gl_t and gl_w must have matching length");
    }

    size_t M = oc.shape(0);
    size_t N = sc.shape(0);
    size_t n_qp = glt.shape(0);

    // REFL-only per-pair specular tables, flat (M, N) row-major — the same
    // contract `sinusoidal_field_tensor_impl<true>` takes.
    const double *cth_p = nullptr, *px_p = nullptr, *py_p = nullptr;
    const double *tmp_p = nullptr, *tnp_p = nullptr;
    if (REFL) {
        for (const auto *t : {&cos_th, &px_t, &py_t, &tm_p, &tn_p}) {
            if ((*t).ndim() != 2 || (size_t)(*t).shape(0) != M ||
                (size_t)(*t).shape(1) != N) {
                throw std::runtime_error(
                    "refl tables must all have shape (M_obs, N_src)");
            }
        }
        cth_p = cos_th.data();
        px_p = px_t.data();
        py_p = py_t.data();
        tmp_p = tm_p.data();
        tnp_p = tn_p.data();
    }

    py::array_t<std::complex<double>> Phi_const({M, N});
    py::array_t<std::complex<double>> Phi_sin({M, N});
    py::array_t<std::complex<double>> Phi_cos({M, N});
    auto pc = Phi_const.mutable_unchecked<2>();
    auto ps = Phi_sin.mutable_unchecked<2>();
    auto pco = Phi_cos.mutable_unchecked<2>();

    py::gil_scoped_release release;

    // Per-source-segment precompute: H_n = h_n/2, sin(kH_n), cos(kH_n) — NEC's
    // SS/CS at f.3183-3184.
    std::vector<double> H_n(N), sin_kH(N), cos_kH(N), sa_v(N);
    std::vector<unsigned char> ext1_v(N), ext2_v(N);
    for (size_t n = 0; n < N; n++) {
        H_n[n] = 0.5 * sh(n);
        sin_kH[n] = std::sin(k * H_n[n]);
        cos_kH[n] = std::cos(k * H_n[n]);
        sa_v[n] = sa(n);
        ext1_v[n] = (i1(n) != 2) ? 1 : 0;
        ext2_v[n] = (i2(n) != 2) ? 1 : 0;
    }

    std::vector<double> glt_v(n_qp), glw_v(n_qp);
    for (size_t q = 0; q < n_qp; q++) {
        glt_v[q] = glt(q);
        glw_v[q] = glw(q);
    }

    // CON of f.3181 — NEC's DATA CONX/0., 4.771341189/ is jη/(4πk) with NEC's
    // k = 2π. Pure imaginary, so only the magnitude is carried (see ek_mul_j).
    const double con_im = eta / (4.0 * M_PI * k);
    const double a_sq = a * a;
    const double k_sq = k * k;

    // Phases per source segment: 2 segment ends + n_qp const-term quadrature
    // nodes. Same layout as `sinusoidal_field_tensor_impl`, with slot 0 = end 1
    // (NEC's Z1 side) and slot 1 = end 2.
    const size_t S = n_qp + 2;
    const size_t P = N * S;

    PYSIM_CANCEL_SETUP(cancel_flag);
    #pragma omp parallel for schedule(static)
    for (size_t m = 0; m < M; m++) {
        PYSIM_CANCEL_POLL();
        std::vector<double> ph(P), cphb(P), sphb(P);
        std::vector<EkGeomRow> gmr(N);
        std::vector<double> r0q_inv_a(N * n_qp);
        // REFL: ρ̂·p̂ per source segment for this observer row, off the
        // UNSWAPPED rho_eval (= RHX) — p̂ is horizontal so only x/y contribute.
        std::vector<double> rho_p_a(REFL ? N : 0);

        double cmx = oc(m, 0), cmy = oc(m, 1), cmz = oc(m, 2);
        double tmx = ot(m, 0), tmy = ot(m, 1), tmz = ot(m, 2);

        // ---- Stage A: geometry + phase generation -------------------------
        for (size_t n = 0; n < N; n++) {
            double cnx = sc(n, 0), cny = sc(n, 1), cnz = sc(n, 2);
            double tnx = st(n, 0), tny = st(n, 1), tnz = st(n, 2);
            double rvx = cmx - cnx, rvy = cmy - cny, rvz = cmz - cnz;
            double z_eval = rvx * tnx + rvy * tny + rvz * tnz;
            double rho_vx = rvx - z_eval * tnx;
            double rho_vy = rvy - z_eval * tny;
            double rho_vz = rvz - z_eval * tnz;
            double rho_axis = std::sqrt(rho_vx*rho_vx + rho_vy*rho_vy + rho_vz*rho_vz);
            // EFLD's RHX: the a-regularized radial distance, on the OBSERVER's
            // radius. Shared with the reduced kernel — NEC computes it before
            // it chooses between EKSC and EKSCX (f.2919-2962), and the ρ̂
            // projection uses this UNSWAPPED distance either way.
            double rho_eval = std::sqrt(rho_axis*rho_axis + a_sq);
            double td = tmx*tnx + tmy*tny + tmz*tnz;
            double rho_dot_tobs = rho_vx*tmx + rho_vy*tmy + rho_vz*tmz;
            if (REFL) {
                size_t mn = m * N + n;
                rho_p_a[n] = (rho_vx * px_p[mn] + rho_vy * py_p[mn]) / rho_eval;
            }

            double H = H_n[n];
            double rhx = rho_eval;
            double src_an = sa_v[n];
            // f.3186-3192. One comparison decides both the ordering and the
            // IRA arm, per pair — see the IRA note at the top of this block.
            bool sw = rhx < src_an;
            double rh = sw ? src_an : rhx;
            double b = sw ? rhx : src_an;
            // NEC's Z2 = SH - Z and Z1 = -(SH + Z); momwire's dz_i is -Z_i.
            double z2 = H - z_eval;
            double z1 = -(H + z_eval);
            bool ext1 = ext1_v[n] != 0;
            bool ext2 = ext2_v[n] != 0;
            // GXX integrates at the ORDERED radius, GX at the original one.
            double rad1 = ext1 ? rh : rhx;
            double rad2 = ext2 ? rh : rhx;
            double r2_1 = z1*z1 + rad1*rad1;
            double r2_2 = z2*z2 + rad2*rad2;
            double r_1 = std::sqrt(r2_1);
            double r_2 = std::sqrt(r2_2);

            EkGeomRow &g = gmr[n];
            g.z_eval = z_eval; g.rhx = rhx; g.rh = rh; g.b = b;
            g.z1 = z1; g.z2 = z2;
            g.r2_1 = r2_1; g.r_1 = r_1; g.r2_2 = r2_2; g.r_2 = r_2;
            g.td = td; g.rpf = rho_dot_tobs / rho_eval;
            g.ext1 = ext1; g.ext2 = ext2;
            g.swapped = sw;

            size_t base = n * S;
            ph[base + 0] = -k * r_1;
            ph[base + 1] = -k * r_2;
            // INTX (f.3223-3227) integrates e^{-jkR}/R along the segment at the
            // ORDERED radius RH, not at RHX.
            for (size_t q = 0; q < n_qp; q++) {
                double z_q = H * glt_v[q];
                double dz_q = z_eval - z_q;
                double r0_q = std::sqrt(rh*rh + dz_q*dz_q);
                ph[base + 2 + q] = -k * r0_q;
                r0q_inv_a[n * n_qp + q] = 1.0 / r0_q;
            }
        }

        // ---- Stage B: vectorized sincos over every phase ------------------
        PYSIM_OMP_SIMD()
        for (size_t i = 0; i < P; i++) cphb[i] = std::cos(ph[i]);
        PYSIM_OMP_SIMD()
        for (size_t i = 0; i < P; i++) sphb[i] = std::sin(ph[i]);

        // ---- Stage C: assembly --------------------------------------------
        for (size_t n = 0; n < N; n++) {
            const EkGeomRow &g = gmr[n];
            size_t base = n * S;
            double H = H_n[n];
            double ss = sin_kH[n];
            double cs = cos_kH[n];

            EkEnd e1 = g.ext1
                ? ek_end_gxx(k, g.z1, g.rh, g.b, g.r2_1, g.r_1,
                             cphb[base + 0], sphb[base + 0], g.swapped)
                : ek_end_gx(g.z1, g.rhx, g.r2_1, g.r_1, k,
                            cphb[base + 0], sphb[base + 0]);
            EkEnd e2 = g.ext2
                ? ek_end_gxx(k, g.z2, g.rh, g.b, g.r2_2, g.r_2,
                             cphb[base + 1], sphb[base + 1], g.swapped)
                : ek_end_gx(g.z2, g.rhx, g.r2_2, g.r_2, k,
                            cphb[base + 1], sphb[base + 1]);

            // f.3213-3222, verbatim — algebraically the same brackets the
            // reduced path spells per endpoint; only the G-quantities changed.
            EkC ez_sin = ek_mul_j(con_im, (e2.g1 - e1.g1) * cs * k
                                          - (e2.g1p + e1.g1p) * ss);
            EkC ez_cos = ek_mul_j(-con_im, (e2.g1 + e1.g1) * ss * k
                                           + (e2.g1p - e1.g1p) * cs);
            EkC erho_sin = ek_mul_j(
                -con_im,
                (g.z2 * e2.g2p + g.z1 * e1.g2p + e2.g2 + e1.g2) * ss
                    - (g.z2 * e2.g2 - g.z1 * e1.g2) * cs * k);
            EkC erho_cos = ek_mul_j(
                -con_im,
                (g.z2 * e2.g2p - g.z1 * e1.g2p + e2.g2 - e1.g2) * cs
                    + (g.z2 * e2.g2 + g.z1 * e1.g2) * ss * k);
            EkC erho_const = ek_mul_j(con_im, e2.g3 - e1.g3);

            // f.3223-3227: the constant-current term. INTX integrates
            // e^{-jkR}/R along the segment at RH, and the extended kernel
            // scales that integral by (1 - (kB/2)²) while adding back a
            // (kB/2)²-weighted reduced-kernel end difference — the O(a²) piece
            // of Eq 98 that the ρ-expansion of the axial term leaves behind.
            // Unlike the per-end substitutions this factor is NOT gated: it
            // applies whenever EK is on, even when both ends fell back to GX.
            // Spelled as two divisions, matching the numpy `(H - z)/rh` and
            // `(-H - z)/rh`, rather than one reciprocal and two products.
            double int_inv_r0 = std::asinh(g.z2 / g.rh) - std::asinh(g.z1 / g.rh);
            double int_reg_re = 0.0, int_reg_im = 0.0;
            for (size_t q = 0; q < n_qp; q++) {
                double inv_r0_q = r0q_inv_a[n * n_qp + q];
                double reg_re = (cphb[base + 2 + q] - 1.0) * inv_r0_q;
                double reg_im = sphb[base + 2 + q] * inv_r0_q;
                int_reg_re += reg_re * glw_v[q];
                int_reg_im += reg_im * glw_v[q];
            }
            EkC int_G0 = EkC{int_inv_r0 + int_reg_re * H, int_reg_im * H};
            double bk = k * g.b;
            double bk2 = 0.25 * bk * bk;
            EkC ez_const = ek_mul_j(
                -con_im,
                e2.g1p - e1.g1p + (k_sq * (1.0 - bk2)) * int_G0
                    - bk2 * (e2.gzp - e1.gzp));

            double td = g.td, rpf = g.rpf;
            if (REFL) {
                // The Fresnel tail, byte-for-byte the reduced kernel's (see
                // `refl_pair` / `refl_project`). The EK tables replace E_z/E_ρ
                // and nothing else: `td`, `rpf` and `rho_p_a` are EFLD's
                // pre-kernel geometry, so the dyad sees the same four factors
                // it sees with EK off — which is exactly how the numpy path
                // spells it, `_field_tensor_image_refl`'s `_project_weighted`
                // consuming whichever tables `_field_components_bcast`
                // returned.
                size_t mn = m * N + n;
                ReflPair rp = refl_pair(eps_t, cth_p[mn], tmp_p[mn], tnp_p[mn],
                                        rho_p_a[n], td, rpf);
                auto weighted = [&](EkC ez, EkC erho) {
                    return refl_project(rp,
                                        std::complex<double>(ez.re, ez.im),
                                        std::complex<double>(erho.re, erho.im));
                };
                pc(m, n) = weighted(ez_const, erho_const);
                ps(m, n) = weighted(ez_sin, erho_sin);
                pco(m, n) = weighted(ez_cos, erho_cos);
            } else {
                // The consuming projection is the reduced kernel's:
                //   Phi = td·E_z + rho_proj·E_ρ   (`_field_tensor`'s tail).
                pc(m, n) = std::complex<double>(
                    td * ez_const.re + rpf * erho_const.re,
                    td * ez_const.im + rpf * erho_const.im);
                ps(m, n) = std::complex<double>(
                    td * ez_sin.re + rpf * erho_sin.re,
                    td * ez_sin.im + rpf * erho_sin.im);
                pco(m, n) = std::complex<double>(
                    td * ez_cos.re + rpf * erho_cos.re,
                    td * ez_cos.im + rpf * erho_cos.im);
            }
        }
    }

    PYSIM_THROW_IF_ABORTED();
    return std::make_tuple(Phi_const, Phi_sin, Phi_cos);
}


// Thin non-template wrappers for pybind registration, mirroring the reduced
// kernel's pair: the plain EK entry point passes empty refl tables, the refl
// one forwards them.
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_field_tensor_ek(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,
    double a, double k, double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_a,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> ind1,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> ind2,
    uintptr_t cancel_flag = 0
) {
    py::array_t<double> empty(std::vector<py::ssize_t>{0, 0});
    return sinusoidal_field_tensor_ek_impl<false>(
        obs_centers, obs_tangents, src_centers, src_tangents, seg_h,
        a, k, eta, gl_t, gl_w, src_a, ind1, ind2,
        empty, empty, empty, empty, empty,
        std::complex<double>(0.0, 0.0), cancel_flag);
}

static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_field_tensor_ek_refl(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,
    double a, double k, double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_a,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> ind1,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> ind2,
    py::array_t<double, py::array::c_style | py::array::forcecast> cos_th,
    py::array_t<double, py::array::c_style | py::array::forcecast> px,
    py::array_t<double, py::array::c_style | py::array::forcecast> py_,
    py::array_t<double, py::array::c_style | py::array::forcecast> tm_p,
    py::array_t<double, py::array::c_style | py::array::forcecast> tn_p,
    std::complex<double> eps_t,
    uintptr_t cancel_flag = 0
) {
    return sinusoidal_field_tensor_ek_impl<true>(
        obs_centers, obs_tangents, src_centers, src_tangents, seg_h,
        a, k, eta, gl_t, gl_w, src_a, ind1, ind2,
        cos_th, px, py_, tm_p, tn_p, eps_t, cancel_flag);
}


// Fused far fill for the sinusoidal GALERKIN solver (momwire#194).
//
// The Galerkin row for test segment m is a quadrature over m, so its field
// evaluation is the tensor above with the observers at m's nq Gauss points
// instead of its centre — and then reduced against the test weights:
//
//   contrib[e, n] = Σ_q w_entry[e, q] · Phi[m(e), q, n]
//
// where `e` runs over the support entries (segment, basis) of the CSR basis
// view and Phi is the plainly-projected tensor of the three source shapes.
// Evaluating Phi and reducing it in ONE kernel is the whole point: the numpy
// path has to materialize the (rows·nq, N) tables (and the (rows·nq, N,
// n_qp_const) source-quadrature scratch under them) before it can contract
// the q axis away, which is what forces the Python fill to block over test
// segments at all. Here nothing but the (nnz, N) result is ever stored.
//
// Serves the PLAIN projection only — the free-space block and the PEC-image
// block (`src_*` mirrored), which is where the fill time is. The
// reflection-coefficient projector's per-pair Fresnel tables and the
// Sommerfeld remainder stay on the numpy path.
//
// Parallelism: over test segments m. Each m owns rows starts[m]:starts[m+1]
// of the output exclusively, so the accumulation needs no reduction and no
// atomics; the rows are contiguous, so they stay cache-resident across the
// nq observers that write them. Per observer the kernel repeats
// `sinusoidal_field_tensor_impl`'s three stages (geometry+phases, one
// omp-simd sincos sweep, assembly).
//
// THIRD SHAPE (momwire#205): the const and sin blocks are the point-matched
// kernel's arithmetic verbatim; the third is the (cos kξ − 1) source shape,
// not cos kξ — the folded contract the Galerkin product pairs with σC. It is
// a transcription of `SinusoidalSolver._folded_cos_fields`, whose docstring
// carries the derivation and the reason the literal difference cannot be
// used. The two helpers below are that method's `_sin_minus_arg` /
// `_asinh_minus_arg`, and the phase table grows to carry the half-angles the
// spelling needs (see `S` at the top of the fill).

// sin(u) − u, without the u²/6 cancellation of the literal difference.
static inline double sin_minus_arg(double u) {
    double u2 = u * u;
    if (std::fabs(u) < 0.1) {
        return -(u * u2) / 6.0 *
               (1.0 - u2 / 20.0 *
                          (1.0 - u2 / 42.0 *
                                     (1.0 - u2 / 72.0 * (1.0 - u2 / 110.0))));
    }
    return std::sin(u) - u;
}

// asinh(x) − x, taking the caller's t = asinh(x) so the fill pays for one
// asinh either way. The series is in t because sinh t − t converges
// factorially where the asinh series does not.
static inline double asinh_minus_arg_from_t(double t) {
    double t2 = t * t;
    if (std::fabs(t) < 1.0) {
        return -(t * t2) / 6.0 *
               (1.0 + t2 / 20.0 *
                          (1.0 + t2 / 42.0 *
                                     (1.0 + t2 / 72.0 *
                                                (1.0 + t2 / 110.0 *
                                                           (1.0 + t2 / 156.0)))));
    }
    return -(std::sinh(t) - t);
}

// The extended kernel's payload for the fused far fill (momwire#246 unit C):
// `SinusoidalGalerkinSolver._ek_pairs`' `_EKPairs`, flattened. `src_a` is one
// radius per SOURCE segment (N), `eligible` the pair rule's mask over
// (observer row, source segment) — M*nq rows, exactly the rows `obs_centers`
// has — and `gx`/`gw` the composite sinh-mapped rule
// `SinusoidalSolver._ek_delta_rule` built, passed in rather than rebuilt here
// so the two backends integrate against the same nodes by construction.
struct GalerkinEkBlock {
    const double *src_a;
    const bool *eligible;
    const double *gx;
    const double *gw;
    size_t n_gl;
};

// One implementation, two instantiations. `WITH_EK` is a compile-time
// constant, so the reduced entry point below compiles with every line of the
// delta gone — which is what keeps `sinusoidal_galerkin_far_fill` byte-frozen
// against its pre-#246 build (gate G-C2) while the two paths stay one piece of
// code. The alternative — a copied 450-line twin — freezes the reduced fill
// just as well and rots twice as fast.
template <bool WITH_EK>
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
galerkin_far_fill_impl(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_radius,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_hh,
    double k, double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<std::complex<double>,
                py::array::c_style | py::array::forcecast> w_entry,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> starts,
    uintptr_t cancel_flag,
    const GalerkinEkBlock *ekb
) {
    auto oc = obs_centers.unchecked<2>();
    auto ot = obs_tangents.unchecked<2>();
    auto ar = obs_radius.unchecked<1>();
    auto sc = src_centers.unchecked<2>();
    auto st = src_tangents.unchecked<2>();
    auto sh = src_hh.unchecked<1>();
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();
    auto we = w_entry.unchecked<2>();
    auto st_ = starts.unchecked<1>();

    if (oc.shape(1) != 3 || ot.shape(1) != 3 ||
        sc.shape(1) != 3 || st.shape(1) != 3) {
        throw std::runtime_error("center/tangent arrays must have shape (N, 3)");
    }
    if (sc.shape(0) != st.shape(0) || sc.shape(0) != sh.shape(0)) {
        throw std::runtime_error("src arrays must all have matching N");
    }
    if (glt.shape(0) != glw.shape(0)) {
        throw std::runtime_error("gl_t and gl_w must have matching length");
    }
    if (st_.shape(0) < 1) {
        throw std::runtime_error("starts must have length n_test_segments + 1");
    }

    size_t M = (size_t)st_.shape(0) - 1;          // test segments
    size_t nq = (size_t)we.shape(1);              // test quadrature nodes
    size_t nnz = (size_t)we.shape(0);             // CSR support entries
    size_t N = (size_t)sc.shape(0);               // source segments
    size_t n_qp = (size_t)glt.shape(0);           // source quadrature nodes

    if ((size_t)oc.shape(0) != M * nq || (size_t)ot.shape(0) != M * nq ||
        (size_t)ar.shape(0) != M * nq) {
        throw std::runtime_error(
            "obs_centers/obs_tangents/obs_radius must have M*nq rows");
    }
    if ((size_t)st_(st_.shape(0) - 1) != nnz) {
        throw std::runtime_error("starts[-1] must equal w_entry's row count");
    }

    py::array_t<std::complex<double>> out_const({nnz, N});
    py::array_t<std::complex<double>> out_sin({nnz, N});
    py::array_t<std::complex<double>> out_cos({nnz, N});
    std::complex<double> *oc_p = out_const.mutable_data();
    std::complex<double> *os_p = out_sin.mutable_data();
    std::complex<double> *oco_p = out_cos.mutable_data();
    const std::complex<double> *w_p = w_entry.data();

    py::gil_scoped_release release;

    std::vector<double> H_n(N), sin_kH(N), cos_kH(N), cos_kH_m1(N), smarg_kH(N);
    for (size_t n = 0; n < N; n++) {
        H_n[n] = sh(n);
        sin_kH[n] = std::sin(k * H_n[n]);
        cos_kH[n] = std::cos(k * H_n[n]);
        // cos kH − 1 and sin kH − kH, both at their own size (#205).
        double hs = std::sin(0.5 * k * H_n[n]);
        cos_kH_m1[n] = -2.0 * hs * hs;
        smarg_kH[n] = sin_minus_arg(k * H_n[n]);
    }
    std::vector<double> glt_v(n_qp), glw_v(n_qp);
    // Which endpoint each source-quadrature node is referred to (#205), and
    // the two halves' weight sums — the rule's own departure from 1 each,
    // which is part of the value the const shape already carries.
    std::vector<double> gl_step(n_qp);
    std::vector<char> gl_near2(n_qp);
    double w_hi = 0.0, w_lo = 0.0;
    for (size_t q = 0; q < n_qp; q++) {
        glt_v[q] = glt(q);
        glw_v[q] = glw(q);
        gl_near2[q] = glt_v[q] >= 0.0;
        gl_step[q] = gl_near2[q] ? (1.0 - glt_v[q]) : -(1.0 + glt_v[q]);
        (gl_near2[q] ? w_hi : w_lo) += glw_v[q];
    }
    w_hi -= 1.0;
    w_lo -= 1.0;

    const double four_pi_k = 4.0 * M_PI * k;
    const double pref_z_im = eta / four_pi_k;
    const double pref_rho_const_im = -eta / four_pi_k;

    // Phase table per source segment. The folded third shape (#205) needs
    // HALF angles the const and sin shapes do not: cos kr − 1 has to be
    // −2sin²(kr/2) rather than a subtraction, and each node's e^{−jkδ_q} − 1
    // is taken at that node's offset δ_q from its own endpoint. So the table
    // carries the halves and derives the full angles from them by the
    // double-angle identities — e^{−jkr} = (e^{−jkr/2})², two multiplies —
    // rather than evaluating both. That keeps the sweep at n_qp + 3 phases
    // against the pre-#205 n_qp + 2: the whole cost of the folded shape is
    // one extra sincos per (observer, source) pair, for the E_ρ reference
    // angle φ = k(r₁−r₂)/2. Evaluating the full angles as well measured 2×
    // slower on the N=2401 dipole, and the sweep is most of the kernel.
    //
    // The reconstruction costs the const and sin shapes ~1 ulp on their
    // Green's values against the numpy path's own exp() — the two paths have
    // agreed at reassociation level since #194 and still do.
    const size_t IX_H2 = 0;         // half-phase of r₀_2
    const size_t IX_H1 = 1;         // half-phase of r₀_1
    const size_t IX_PHI = 2;        // φ
    const size_t IX_DQ = 3;         // half-phases of the node offsets
    const size_t S = n_qp + 3;
    const size_t P = N * S;

    PYSIM_CANCEL_SETUP(cancel_flag);
    #pragma omp parallel for schedule(static)
    for (size_t m = 0; m < M; m++) {
        PYSIM_CANCEL_POLL();
        size_t e0 = (size_t)st_(m), e1 = (size_t)st_(m + 1);
        if (e1 == e0) continue;  // segment carries no basis support

        // Rows e0:e1 belong to this test segment alone; zero them here rather
        // than allocating a zeroed output, so the accumulation below can add
        // straight into the result.
        std::fill(oc_p + e0 * N, oc_p + e1 * N, std::complex<double>(0.0, 0.0));
        std::fill(os_p + e0 * N, os_p + e1 * N, std::complex<double>(0.0, 0.0));
        std::fill(oco_p + e0 * N, oco_p + e1 * N, std::complex<double>(0.0, 0.0));

        std::vector<double> ph(P), cphb(P), sphb(P);
        std::vector<double> rho_eval_a(N), dz1_a(N), dz2_a(N),
                            r0_1_a(N), r0_2_a(N), td_a(N), rpf_a(N), z_a(N);
        std::vector<double> r0q_inv_a(N * n_qp), delta_a(N * n_qp);
        // Projected field of the three source shapes at ONE observer, split
        // re/im so the reduction below stays vectorizable.
        std::vector<double> phi_c_re(N), phi_c_im(N), phi_s_re(N), phi_s_im(N),
                            phi_co_re(N), phi_co_im(N);

        for (size_t qt = 0; qt < nq; qt++) {
            size_t o = m * nq + qt;
            double cmx = oc(o, 0), cmy = oc(o, 1), cmz = oc(o, 2);
            double tmx = ot(o, 0), tmy = ot(o, 1), tmz = ot(o, 2);
            double a_sq = ar(o) * ar(o);

            // ---- Stage A: geometry + phase generation ---------------------
            for (size_t n = 0; n < N; n++) {
                double cnx = sc(n, 0), cny = sc(n, 1), cnz = sc(n, 2);
                double tnx = st(n, 0), tny = st(n, 1), tnz = st(n, 2);
                double rvx = cmx - cnx, rvy = cmy - cny, rvz = cmz - cnz;
                double z_eval = rvx * tnx + rvy * tny + rvz * tnz;
                double rho_vx = rvx - z_eval * tnx;
                double rho_vy = rvy - z_eval * tny;
                double rho_vz = rvz - z_eval * tnz;
                double rho_axis =
                    std::sqrt(rho_vx*rho_vx + rho_vy*rho_vy + rho_vz*rho_vz);
                double rho_eval = std::sqrt(rho_axis*rho_axis + a_sq);
                double td = tmx*tnx + tmy*tny + tmz*tnz;
                double rho_dot_tobs = rho_vx*tmx + rho_vy*tmy + rho_vz*tmz;

                double H = H_n[n];
                double dz2 = z_eval - H;
                double dz1 = z_eval + H;
                double r0_2 = std::sqrt(rho_eval*rho_eval + dz2*dz2);
                double r0_1 = std::sqrt(rho_eval*rho_eval + dz1*dz1);

                rho_eval_a[n] = rho_eval;
                z_a[n] = z_eval;
                dz1_a[n] = dz1; dz2_a[n] = dz2;
                r0_1_a[n] = r0_1; r0_2_a[n] = r0_2;
                td_a[n] = td; rpf_a[n] = rho_dot_tobs / rho_eval;

                size_t base = n * S;
                ph[base + IX_H2] = -0.5 * k * r0_2;
                ph[base + IX_H1] = -0.5 * k * r0_1;
                // φ from r₁ − r₂ = 4Hz/(r₁+r₂), which is exact where the
                // subtraction would not be (#205).
                ph[base + IX_PHI] = 2.0 * k * H * z_eval / (r0_1 + r0_2);
                for (size_t q = 0; q < n_qp; q++) {
                    double z_q = H * glt_v[q];
                    double dz_q = z_eval - z_q;
                    double r0_q = std::sqrt(rho_eval*rho_eval + dz_q*dz_q);
                    r0q_inv_a[n * n_qp + q] = 1.0 / r0_q;
                    // δ_q = r_q − r_ref for the endpoint on the node's own
                    // side. Δz_q − Δz_ref is ±H(1 ∓ t_q) exactly — the
                    // observer cancels out of it — so δ_q is exact however
                    // far away the observer sits.
                    double dz_ref = gl_near2[q] ? dz2 : dz1;
                    double r_ref = gl_near2[q] ? r0_2 : r0_1;
                    double delta = H * gl_step[q] * (dz_q + dz_ref)
                                   / (r0_q + r_ref);
                    delta_a[n * n_qp + q] = delta;
                    ph[base + IX_DQ + q] = -0.5 * k * delta;
                }
            }

            // ---- Stage B: vectorized sincos over every phase --------------
            PYSIM_OMP_SIMD()
            for (size_t i = 0; i < P; i++) cphb[i] = std::cos(ph[i]);
            PYSIM_OMP_SIMD()
            for (size_t i = 0; i < P; i++) sphb[i] = std::sin(ph[i]);

            // ---- Stage C: assembly + plain tangential projection ----------
            for (size_t n = 0; n < N; n++) {
                size_t base = n * S;
                // Full boundary phases from their halves: e^{−jkr} =
                // (e^{−jkr/2})². The halves are what the folded shape needs
                // (below), and squaring is cheaper than a second sincos.
                double ch2 = cphb[base + IX_H2], sh2 = sphb[base + IX_H2];
                double ch1 = cphb[base + IX_H1], sh1 = sphb[base + IX_H1];
                double cph_2 = ch2 * ch2 - sh2 * sh2, sph_2 = 2.0 * ch2 * sh2;
                double cph_1 = ch1 * ch1 - sh1 * sh1, sph_1 = 2.0 * ch1 * sh1;
                double rho_eval = rho_eval_a[n];
                double dz1 = dz1_a[n], dz2 = dz2_a[n];
                double r0_1 = r0_1_a[n], r0_2 = r0_2_a[n];
                double td = td_a[n], rho_proj_factor = rpf_a[n];
                double H = H_n[n];
                double inv_r0_2 = 1.0 / r0_2;
                double inv_r0_1 = 1.0 / r0_1;
                double G0_2_re = cph_2 * inv_r0_2, G0_2_im = sph_2 * inv_r0_2;
                double G0_1_re = cph_1 * inv_r0_1, G0_1_im = sph_1 * inv_r0_1;

                double inv_r0_2_sq = inv_r0_2 * inv_r0_2;
                double inv_r0_1_sq = inv_r0_1 * inv_r0_1;
                double one_jkr_2_re = inv_r0_2_sq;
                double one_jkr_2_im = k * r0_2 * inv_r0_2_sq;
                double one_jkr_1_re = inv_r0_1_sq;
                double one_jkr_1_im = k * r0_1 * inv_r0_1_sq;

                auto cmul = [](double ar_, double ai, double br, double bi,
                               double &cr, double &ci) {
                    cr = ar_*br - ai*bi;
                    ci = ar_*bi + ai*br;
                };

                // ---- Const source (Eqs 78, 79) ---------------------------
                double term_const2_re, term_const2_im;
                cmul(one_jkr_2_re, one_jkr_2_im, G0_2_re, G0_2_im,
                     term_const2_re, term_const2_im);
                double term_const1_re, term_const1_im;
                cmul(one_jkr_1_re, one_jkr_1_im, G0_1_re, G0_1_im,
                     term_const1_re, term_const1_im);
                double rho_diff_re = rho_eval * (term_const2_re - term_const1_re);
                double rho_diff_im = rho_eval * (term_const2_im - term_const1_im);
                double Erho_const_re = -pref_rho_const_im * rho_diff_im;
                double Erho_const_im =  pref_rho_const_im * rho_diff_re;

                double inv_rho_eval = 1.0 / rho_eval;
                double u2 = -dz2 * inv_rho_eval;
                double u1 = -dz1 * inv_rho_eval;
                double int_inv_r0 = std::asinh(u2) - std::asinh(u1);

                double int_reg_re = 0.0, int_reg_im = 0.0;
                for (size_t q = 0; q < n_qp; q++) {
                    // e^{−jkr_q} = e^{−jkr_ref}·(e^{−jkδ_q/2})², the node's
                    // own endpoint carrying the reference phase (#205). The
                    // half-angle is the one the folded shape needs below.
                    double cd = cphb[base + IX_DQ + q];
                    double sd = sphb[base + IX_DQ + q];
                    double ed_re = cd * cd - sd * sd;
                    double ed_im = 2.0 * cd * sd;
                    bool hi = gl_near2[q];
                    double er = hi ? cph_2 : cph_1;
                    double ei = hi ? sph_2 : sph_1;
                    double cph_q = er * ed_re - ei * ed_im;
                    double sph_q = er * ed_im + ei * ed_re;
                    double inv_r0_q = r0q_inv_a[n * n_qp + q];
                    int_reg_re += (cph_q - 1.0) * inv_r0_q * glw_v[q];
                    int_reg_im += sph_q * inv_r0_q * glw_v[q];
                }
                int_reg_re *= H;
                int_reg_im *= H;
                double int_G0_re = int_inv_r0 + int_reg_re;
                double int_G0_im = int_reg_im;

                double Ez_boundary_re = dz2 * term_const2_re - dz1 * term_const1_re;
                double Ez_boundary_im = dz2 * term_const2_im - dz1 * term_const1_im;
                double k_sq = k * k;
                double inside_re = Ez_boundary_re + k_sq * int_G0_re;
                double inside_im = Ez_boundary_im + k_sq * int_G0_im;
                double Ez_const_re =  pref_z_im * inside_im;
                double Ez_const_im = -pref_z_im * inside_re;

                // ---- Sine source (Eqs 76, 77) ----------------------------
                double sin2 = sin_kH[n];
                double cos2 = cos_kH[n];
                double sin1 = -sin2;
                double cos1 =  cos2;

                double inner_2_re = 1.0 - dz2*dz2 * one_jkr_2_re;
                double inner_2_im =     - dz2*dz2 * one_jkr_2_im;
                double bracket_sin_2_re = k*dz2*cos2 + inner_2_re*sin2;
                double bracket_sin_2_im =              inner_2_im*sin2;
                double bsin2_re, bsin2_im;
                cmul(G0_2_re, G0_2_im, bracket_sin_2_re, bracket_sin_2_im,
                     bsin2_re, bsin2_im);

                double inner_1_re = 1.0 - dz1*dz1 * one_jkr_1_re;
                double inner_1_im =     - dz1*dz1 * one_jkr_1_im;
                double bracket_sin_1_re = k*dz1*cos1 + inner_1_re*sin1;
                double bracket_sin_1_im =              inner_1_im*sin1;
                double bsin1_re, bsin1_im;
                cmul(G0_1_re, G0_1_im, bracket_sin_1_re, bracket_sin_1_im,
                     bsin1_re, bsin1_im);
                double pref_rho_im = pref_rho_const_im * inv_rho_eval;
                double Erho_sin_re = -pref_rho_im * (bsin2_im - bsin1_im);
                double Erho_sin_im =  pref_rho_im * (bsin2_re - bsin1_re);

                double bracket_sin_z_2_re = k*cos2 - dz2*one_jkr_2_re*sin2;
                double bracket_sin_z_2_im =        - dz2*one_jkr_2_im*sin2;
                double bszin2_re, bszin2_im;
                cmul(G0_2_re, G0_2_im, bracket_sin_z_2_re, bracket_sin_z_2_im,
                     bszin2_re, bszin2_im);
                double bracket_sin_z_1_re = k*cos1 - dz1*one_jkr_1_re*sin1;
                double bracket_sin_z_1_im =        - dz1*one_jkr_1_im*sin1;
                double bszin1_re, bszin1_im;
                cmul(G0_1_re, G0_1_im, bracket_sin_z_1_re, bracket_sin_z_1_im,
                     bszin1_re, bszin1_im);
                double Ez_sin_re = -pref_z_im * (bszin2_im - bszin1_im);
                double Ez_sin_im =  pref_z_im * (bszin2_re - bszin1_re);

                // ---- Folded source (I = cos kξ − 1), #205 ----------------
                // Transcription of `SinusoidalSolver._folded_cos_fields`;
                // the derivation, and why the two closed forms may not
                // simply be subtracted, are in that method's docstring.
                double rho2 = rho_eval * rho_eval;
                double cm1 = cos_kH_m1[n];

                // X: sinh of the arcsinh difference ∫(1/r)dξ, rationalized
                // on whichever endpoint pair does not cancel.
                double X = (dz1 * dz2 >= 0.0)
                    ? 2.0 * H * (dz1 + dz2) / (dz1 * r0_2 + dz2 * r0_1)
                    : (dz1 * r0_2 - dz2 * r0_1) / rho2;
                double t_asx = std::asinh(X);
                double t_sing =
                    (std::fabs(X) < 1.0)
                        ? asinh_minus_arg_from_t(t_asx)
                              + H * rho2 * X * X / ((r0_1 + r0_2) * r0_1 * r0_2)
                        : t_asx - H * (inv_r0_1 + inv_r0_2);

                // g_e = (e^{−jkr_e} − 1)/r_e at the two endpoints, real part
                // from the half-angle (cos kr − 1 = −2sin²(kr/2)) so it is
                // relative rather than absolute.
                double g2_re = -2.0 * sh2 * sh2 * inv_r0_2;
                double g2_im = sph_2 * inv_r0_2;
                double g1_re = -2.0 * sh1 * sh1 * inv_r0_1;
                double g1_im = sph_1 * inv_r0_1;

                // Σ_q gw_q·g(ξ_q) − (g₁+g₂), each node against the endpoint
                // on its own side: g(r_ref+δ) − g(r_ref), both terms O(kδ).
                double m_reg_re = w_hi * g2_re + w_lo * g1_re;
                double m_reg_im = w_hi * g2_im + w_lo * g1_im;
                for (size_t q = 0; q < n_qp; q++) {
                    bool hi = gl_near2[q];
                    double e_ref_re = hi ? cph_2 : cph_1;
                    double e_ref_im = hi ? sph_2 : sph_1;
                    double gr_re = hi ? g2_re : g1_re;
                    double gr_im = hi ? g2_im : g1_im;
                    double sd = sphb[base + IX_DQ + q];
                    double cd = cphb[base + IX_DQ + q];
                    // e^{−jkδ} − 1 from the half-angle: −2s² + 2jsc.
                    double em1_re = -2.0 * sd * sd;
                    double em1_im = 2.0 * sd * cd;
                    double t_re, t_im;
                    cmul(e_ref_re, e_ref_im, em1_re, em1_im, t_re, t_im);
                    double dl = delta_a[n * n_qp + q];
                    double inv_r0_q = r0q_inv_a[n * n_qp + q];
                    double w = glw_v[q] * inv_r0_q;
                    m_reg_re += w * (t_re - gr_re * dl);
                    m_reg_im += w * (t_im - gr_im * dl);
                }

                double smarg = smarg_kH[n];
                double d_int_re = t_sing + H * m_reg_re
                                  - smarg / k * (G0_1_re + G0_2_re);
                double d_int_im = H * m_reg_im - smarg / k * (G0_1_im + G0_2_im);
                double inner_cos_re =
                    k_sq * d_int_re - cm1 * Ez_boundary_re;
                double inner_cos_im =
                    k_sq * d_int_im - cm1 * Ez_boundary_im;
                double Ez_cos_re = -pref_z_im * inner_cos_im;
                double Ez_cos_im =  pref_z_im * inner_cos_re;

                // E_ρ: both endpoint terms referred to the pair's mean
                // radius, which splits them into an even part that
                // telescopes onto sin_minus_arg and an odd part in c₂ − c₁.
                double cph_p = cphb[base + IX_PHI], sph_p = sphb[base + IX_PHI];
                double phi_ang = ph[base + IX_PHI];
                double kH = k * H;
                double A_ang = kH + phi_ang, B_ang = kH - phi_ang;
                // H(c₁+c₂) − (r₁−r₂), cancellation-free.
                double d_lin = -8.0 * H * H * H * z_a[n] * rho2
                               / ((rho2 + dz1 * dz2 + r0_1 * r0_2)
                                  * (r0_1 + r0_2) * r0_1 * r0_2);
                double w_even =
                    (A_ang * sin_minus_arg(B_ang) - B_ang * sin_minus_arg(A_ang))
                        / kH
                    + (d_lin / H) * sin2 * cph_p;
                // c₂ − c₁ = −ρ²X/(r₁r₂), from the same numerator as X.
                double w_odd = sin2 * (-(rho2 * X) / (r0_1 * r0_2)) * sph_p;
                // W = e^{−jk(r₁+r₂)/2}·(w_even + j·w_odd), the reference
                // phase built from e^{−jkr₂} and φ rather than a new exp.
                double ref_re = cph_2 * cph_p + sph_2 * sph_p;
                double ref_im = sph_2 * cph_p - cph_2 * sph_p;
                double W_re, W_im;
                cmul(ref_re, ref_im, w_even, w_odd, W_re, W_im);
                double b_rho_re = -k * W_re + rho2 * cm1
                                  * (term_const2_re - term_const1_re);
                double b_rho_im = -k * W_im + rho2 * cm1
                                  * (term_const2_im - term_const1_im);
                double Erho_cos_re = -pref_rho_im * b_rho_im;
                double Erho_cos_im =  pref_rho_im * b_rho_re;

                // ---- Extended kernel: the folded delta (momwire#246) ------
                // Transcription of `SinusoidalSolver._folded_ek_delta_fields`;
                // the derivation of the delta kernel W = a²g′ + a²ρ²g″, the
                // two field operators, and the floor underneath the whole
                // decomposition are in that method's docstring. Nothing above
                // this point moves: the reduced tables are #205's own, and the
                // delta is a separate sum at its own size added to them before
                // the projection — exactly where the numpy seam adds it.
                if (WITH_EK && ekb->eligible[o * N + n]) {
                    // ζ = ρ·sinh t, R = ρ·cosh t. The variable is NOT ξ: in ξ
                    // the delta is a spike of width ρ inside a segment of half
                    // length H, so a fixed rule's accuracy is governed by ρ/H
                    // and a 16-node rule at the self pair is wrong by 1e6 at
                    // Δ/a = 122. In t the spike is O(1) wide however thin the
                    // wire, and the kernel's poles sit at t = ±jπ/2 whatever ρ
                    // is. `gx`/`gw` are the caller's composite rule over that
                    // mapped interval (`_ek_delta_rule`).
                    double t_lo = std::asinh((z_a[n] - H) / rho_eval);
                    double t_hi = std::asinh((z_a[n] + H) / rho_eval);
                    double t_mid = 0.5 * (t_hi + t_lo);
                    double t_half = 0.5 * (t_hi - t_lo);
                    double src_a = ekb->src_a[n];
                    double dz_c_re = 0.0, dz_c_im = 0.0;  // E_z,   const shape
                    double dr_c_re = 0.0, dr_c_im = 0.0;  // E_ρ,   const shape
                    double dz_s_re = 0.0, dz_s_im = 0.0;  // sin shape
                    double dr_s_re = 0.0, dr_s_im = 0.0;
                    double dz_o_re = 0.0, dz_o_im = 0.0;  // folded cos shape
                    double dr_o_re = 0.0, dr_o_im = 0.0;
                    for (size_t qd = 0; qd < ekb->n_gl; qd++) {
                        double t = t_mid + t_half * ekb->gx[qd];
                        double cosh_t = std::cosh(t);
                        // R = ρ·cosh t exactly, so the near-singular
                        // denominator never goes through a difference of
                        // squares; ξ = z − ζ recovers the source arc.
                        double R = rho_eval * cosh_t;
                        double zeta = rho_eval * std::sinh(t);
                        double xi = z_a[n] - zeta;
                        double wq = (t_half * ekb->gw[qd]) * (rho_eval * cosh_t);
                        double r2 = R * R;

                        // g′…g⁗ of e^{−jkR}/R in u = R², through the reverse
                        // Bessel polynomials Aₙ(x) at x = jkR. x is purely
                        // imaginary, so x², x³, x⁴ are exactly real or exactly
                        // imaginary and the four polynomials split by hand
                        // into the same values numpy's complex products give.
                        double kR = k * R;
                        double kr2 = kR * kR;
                        double kr3 = kr2 * kR;
                        double kr4 = kr2 * kr2;
                        double a1_re = 1.0,               a1_im = kR;
                        double a2_re = 3.0 - kr2,         a2_im = 3.0 * kR;
                        double a3_re = 15.0 - 6.0 * kr2,  a3_im = 15.0 * kR - kr3;
                        double a4_re = 105.0 - 45.0 * kr2 + kr4;
                        double a4_im = 105.0 * kR - 10.0 * kr3;

                        double inv2 = 1.0 / r2;
                        double inv4 = inv2 * inv2;
                        double inv6 = inv4 * inv2;
                        double inv8 = inv6 * inv2;
                        double base_re =  std::cos(kR) / R;
                        double base_im = -std::sin(kR) / R;
                        double gd1_re, gd1_im, gd2_re, gd2_im;
                        double gd3_re, gd3_im, gd4_re, gd4_im;
                        // Named steps in the numpy path's own order — the
                        // scale factor last — so the two backends' roundings
                        // stay one reassociation apart and no further.
                        cmul(base_re, base_im, a1_re, a1_im, gd1_re, gd1_im);
                        gd1_re *= inv2;   gd1_im *= inv2;
                        gd1_re *= -0.5;   gd1_im *= -0.5;
                        cmul(base_re, base_im, a2_re, a2_im, gd2_re, gd2_im);
                        gd2_re *= inv4;   gd2_im *= inv4;
                        gd2_re *= 0.25;   gd2_im *= 0.25;
                        cmul(base_re, base_im, a3_re, a3_im, gd3_re, gd3_im);
                        gd3_re *= inv6;   gd3_im *= inv6;
                        gd3_re *= -0.125; gd3_im *= -0.125;
                        cmul(base_re, base_im, a4_re, a4_im, gd4_re, gd4_im);
                        gd4_re *= inv8;   gd4_im *= inv8;
                        gd4_re *= 0.0625; gd4_im *= 0.0625;

                        // The two operators, with a² held back to the end:
                        //   (k² + ∂²_z)W = a²[k²(g′+ρ²g″) + 2(g″+ρ²g‴)
                        //                     + 4ζ²(g‴+ρ²g⁗)]
                        //   ∂²_{ρz} W    = a²·ρζ·[8g‴ + 4ρ²g⁗]
                        double z4 = 4.0 * zeta * zeta;
                        double lz_re = k_sq * (gd1_re + rho2 * gd2_re)
                                       + 2.0 * (gd2_re + rho2 * gd3_re)
                                       + z4 * (gd3_re + rho2 * gd4_re);
                        double lz_im = k_sq * (gd1_im + rho2 * gd2_im)
                                       + 2.0 * (gd2_im + rho2 * gd3_im)
                                       + z4 * (gd3_im + rho2 * gd4_im);
                        double lr_re = ((8.0 * gd3_re + 4.0 * rho2 * gd4_re)
                                        * rho_eval) * zeta;
                        double lr_im = ((8.0 * gd3_im + 4.0 * rho2 * gd4_im)
                                        * rho_eval) * zeta;

                        // The folded shape POINTWISE as −2sin²(kξ/2) — never
                        // cos − 1 by subtraction. There is no cancellation
                        // discipline to keep here because there is no
                        // difference of closed forms anywhere in the delta.
                        // The sin shape is evaluated at the full angle rather
                        // than reconstructed from the half one: measured, the
                        // reconstruction saves nothing here (the loop is
                        // bound by cosh/sinh, not by sincos) and costs a ulp.
                        double kxi = k * xi;
                        double hsx = std::sin(0.5 * kxi);
                        double sw_s = std::sin(kxi) * wq;
                        double sw_o = (-2.0 * hsx * hsx) * wq;
                        dz_c_re += wq * lz_re;    dz_c_im += wq * lz_im;
                        dr_c_re += wq * lr_re;    dr_c_im += wq * lr_im;
                        dz_s_re += sw_s * lz_re;  dz_s_im += sw_s * lz_im;
                        dr_s_re += sw_s * lr_re;  dr_s_im += sw_s * lr_im;
                        dz_o_re += sw_o * lz_re;  dz_o_im += sw_o * lz_im;
                        dr_o_re += sw_o * lr_re;  dr_o_im += sw_o * lr_im;
                    }
                    // −pref_z·a²: one multiplication at the end, which is what
                    // makes the whole correction exactly 0.0 at a = 0 in IEEE
                    // and not merely in the limit. `src_a` is the SOURCE
                    // segment's radius, the a of NEC Eq 89.
                    double gain = pref_z_im * (src_a * src_a);
                    Ez_const_re  += gain * dz_c_im;
                    Ez_const_im  -= gain * dz_c_re;
                    Erho_const_re += gain * dr_c_im;
                    Erho_const_im -= gain * dr_c_re;
                    Ez_sin_re  += gain * dz_s_im;
                    Ez_sin_im  -= gain * dz_s_re;
                    Erho_sin_re += gain * dr_s_im;
                    Erho_sin_im -= gain * dr_s_re;
                    Ez_cos_re  += gain * dz_o_im;
                    Ez_cos_im  -= gain * dz_o_re;
                    Erho_cos_re += gain * dr_o_im;
                    Erho_cos_im -= gain * dr_o_re;
                }

                phi_c_re[n]  = td * Ez_const_re + rho_proj_factor * Erho_const_re;
                phi_c_im[n]  = td * Ez_const_im + rho_proj_factor * Erho_const_im;
                phi_s_re[n]  = td * Ez_sin_re   + rho_proj_factor * Erho_sin_re;
                phi_s_im[n]  = td * Ez_sin_im   + rho_proj_factor * Erho_sin_im;
                phi_co_re[n] = td * Ez_cos_re   + rho_proj_factor * Erho_cos_re;
                phi_co_im[n] = td * Ez_cos_im   + rho_proj_factor * Erho_cos_im;
            }

            // ---- Test reduction: contrib[e, :] += w_entry[e, qt] * Phi ----
            // One axpy per support entry of this test segment, over the whole
            // source row. Accumulating quadrature node by quadrature node
            // matches `_tested_contrib_rows`' summation order.
            for (size_t e = e0; e < e1; e++) {
                double wr = w_p[e * nq + qt].real();
                double wi = w_p[e * nq + qt].imag();
                double *rc  = reinterpret_cast<double *>(oc_p + e * N);
                double *rs  = reinterpret_cast<double *>(os_p + e * N);
                double *rco = reinterpret_cast<double *>(oco_p + e * N);
                PYSIM_OMP_SIMD()
                for (size_t n = 0; n < N; n++) {
                    rc[2*n]     += wr * phi_c_re[n]  - wi * phi_c_im[n];
                    rc[2*n + 1] += wr * phi_c_im[n]  + wi * phi_c_re[n];
                    rs[2*n]     += wr * phi_s_re[n]  - wi * phi_s_im[n];
                    rs[2*n + 1] += wr * phi_s_im[n]  + wi * phi_s_re[n];
                    rco[2*n]    += wr * phi_co_re[n] - wi * phi_co_im[n];
                    rco[2*n + 1]+= wr * phi_co_im[n] + wi * phi_co_re[n];
                }
            }
        }
    }

    PYSIM_THROW_IF_ABORTED();
    return std::make_tuple(out_const, out_sin, out_cos);
}

// The reduced entry point. Byte-frozen against its pre-#246 build: the
// instantiation below has WITH_EK false, so not one line of the delta is
// compiled into it (gate G-C2).
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_galerkin_far_fill(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_radius,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_hh,
    double k, double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<std::complex<double>,
                py::array::c_style | py::array::forcecast> w_entry,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> starts,
    uintptr_t cancel_flag = 0
) {
    return galerkin_far_fill_impl<false>(
        obs_centers, obs_tangents, obs_radius, src_centers, src_tangents,
        src_hh, k, eta, gl_t, gl_w, w_entry, starts, cancel_flag, nullptr);
}

// The extended-kernel twin (momwire#246 unit C). Same arguments plus the
// `_EKPairs` payload — one source radius per segment, the pair rule's mask
// over (observer row, source segment), and the delta quadrature's composite
// rule — and the same three arrays out.
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_galerkin_far_fill_ek(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_radius,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_hh,
    double k, double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<std::complex<double>,
                py::array::c_style | py::array::forcecast> w_entry,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> starts,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_a,
    py::array_t<bool, py::array::c_style | py::array::forcecast> eligible,
    py::array_t<double, py::array::c_style | py::array::forcecast> ek_gx,
    py::array_t<double, py::array::c_style | py::array::forcecast> ek_gw,
    uintptr_t cancel_flag = 0
) {
    auto sa = src_a.unchecked<1>();
    auto el = eligible.unchecked<2>();
    auto ex = ek_gx.unchecked<1>();
    auto ew = ek_gw.unchecked<1>();
    if (sa.shape(0) != src_centers.shape(0)) {
        throw std::runtime_error("src_a must have one radius per source segment");
    }
    if (el.shape(0) != obs_centers.shape(0) ||
        el.shape(1) != src_centers.shape(0)) {
        throw std::runtime_error(
            "eligible must have shape (obs rows, source segments)");
    }
    if (ex.shape(0) != ew.shape(0) || ex.shape(0) < 1) {
        throw std::runtime_error("ek_gx and ek_gw must have matching length");
    }
    GalerkinEkBlock ekb;
    ekb.src_a = src_a.data();
    ekb.eligible = eligible.data();
    ekb.gx = ek_gx.data();
    ekb.gw = ek_gw.data();
    ekb.n_gl = (size_t)ex.shape(0);
    return galerkin_far_fill_impl<true>(
        obs_centers, obs_tangents, obs_radius, src_centers, src_tangents,
        src_hh, k, eta, gl_t, gl_w, w_entry, starts, cancel_flag, &ekb);
}


// ==========================================================================
// Sommerfeld ground: batched six-integral evaluation (sommerfeld-perf-plan
// Phase 3). A faithful port of _sommerfeld.py's `_six_integrals` machinery
// (same 24-point Gauss rule, same adaptive bisection, same fig 13/14
// contours and tail handling), evaluated per (rho, h) node with OpenMP
// across nodes — the nodes are independent, and the pure-Python fill spends
// ~75% of its time on interpreter/ufunc dispatch this port removes.
//
// Clean-room note: ported from momwire's own _sommerfeld.py (itself
// implemented from the public-domain NEC-2 theory manual equations); the
// complex-argument Bessel/Hankel functions below are implemented from the
// Abramowitz & Stegun 9.1/9.2 ascending series and asymptotic expansions.
// No GPL Sommerfeld source (nec2c, nec2++/PyNEC, somnec) was consulted.
// ==========================================================================

namespace somm {

using cd = std::complex<double>;
static const cd CI(0.0, 1.0);
static const double SPI = 3.14159265358979323846;
static const double EULER_GAMMA = 0.57721566490153286061;
static const int ADAPT_DEPTH = 14;  // = _sommerfeld._ADAPT_DEPTH

// 24-point Gauss-Legendre rule — identical to _sommerfeld._GX/_GW.
static const double GX[24] = {
    -9.95187219997021311e-01, -9.74728555971309474e-01, -9.38274552002732798e-01,
    -8.86415527004401071e-01, -8.20001985973902947e-01, -7.40124191578554358e-01,
    -6.48093651936975546e-01, -5.45421471388839563e-01, -4.33793507626045127e-01,
    -3.15042679696163397e-01, -1.91118867473616311e-01, -6.40568928626056300e-02,
    6.40568928626056300e-02, 1.91118867473616311e-01, 3.15042679696163397e-01,
    4.33793507626045127e-01, 5.45421471388839563e-01, 6.48093651936975546e-01,
    7.40124191578554358e-01, 8.20001985973902947e-01, 8.86415527004401071e-01,
    9.38274552002732798e-01, 9.74728555971309474e-01, 9.95187219997021311e-01,
};
static const double GW[24] = {
    1.23412297999886903e-02, 2.85313886289335593e-02, 4.42774388174194122e-02,
    5.92985849154363601e-02, 7.33464814110801611e-02, 8.61901615319532050e-02,
    9.76186521041139260e-02, 1.07444270115965565e-01, 1.15505668053725516e-01,
    1.21670472927803294e-01, 1.25837456346828247e-01, 1.27938195346752021e-01,
    1.27938195346752021e-01, 1.25837456346828247e-01, 1.21670472927803294e-01,
    1.15505668053725516e-01, 1.07444270115965565e-01, 9.76186521041139260e-02,
    8.61901615319532050e-02, 7.33464814110801611e-02, 5.92985849154363601e-02,
    4.42774388174194122e-02, 2.85313886289335593e-02, 1.23412297999886903e-02,
};

// ---- complex-argument Bessel/Hankel, orders 0 and 1 ----------------------
//
// Domain (measured from real contour fills): |x| up to ~110, arg(x) in
// [-100 deg, +45 deg] — never near the negative-real-axis branch cut, so
// principal-branch log/sqrt are safe throughout. Ascending series for
// |x| <= 12 (~3 digits of cancellation, ~1e-12 abs), A&S 9.2 asymptotic
// expansions with optimal truncation beyond (~1e-11 at the switch, far
// better further out). Both are validated pointwise against scipy over the
// sampled domain in tests/test_sommerfeld_accel.py.

// These routines are the measured hot spot of the whole grid fill
// (sommerfeld-perf-plan Phase 8): ~85% of the per-contour-point cost, and
// the fill is ~95% of a cold Sommerfeld solve. Three shares are exploited
// below, none of which changes the mathematics:
//
//   * every truncation test is "is this term small next to the running
//     sum", a predicate identical under squaring — so |.| (a libm hypot,
//     ~75 of them per small-|x| Hankel call) becomes three flops;
//   * the Y series ride the same two ladders as the J series, so one pass
//     yields all four sums instead of two passes rebuilding the ladders;
//   * in the asymptotic regime the two orders share the (i/z) ladder and
//     the sqrt/exp prefactor, and the kind-1 terms are the kind-2 terms
//     with alternating signs.

// |z|^2 — the convergence-test currency (see above).
static inline double cnorm(cd z) {
    return z.real() * z.real() + z.imag() * z.imag();
}
static const double SER_EPS2 = 1e-34;  // (1e-17)^2

// J0 and J1/z ascending series (A&S 9.1.10/9.1.12): safe as z -> 0.
static void j01_series(cd z, cd &j0, cd &j1x) {
    const cd q = -0.25 * z * z;
    cd t0(1.0, 0.0), t1(0.5, 0.0);
    j0 = t0;
    j1x = t1;
    for (int k = 1; k <= 60; ++k) {
        t0 *= q / double(k * k);
        t1 *= q / double(k * (k + 1));
        j0 += t0;
        j1x += t1;
        if (cnorm(t0) <= SER_EPS2 * cnorm(j0) &&
            cnorm(t1) <= SER_EPS2 * cnorm(j1x))
            break;
    }
}

// J0, J1/z, Y0 and Y1 from ONE ascending-series pass (A&S 9.1.10-9.1.13,
// principal log).
//
// With q = -(z/2)^2 the J ladders are t0_k = q^k/(k!)^2 and t1_k =
// (1/2) q^k/(k!(k+1)!), i.e. t0_k = (-1)^k (z/2)^{2k}/(k!)^2 and 2 t1_k =
// (-1)^k (z/2)^{2k}/(k!(k+1)!). The manual's alternating Y sums are then
// exactly those ladders reweighted by harmonic numbers:
//   sum_{k>=1} (-1)^{k+1} H_k (z/2)^{2k}/(k!)^2        = -sum_k H_k t0_k
//   sum_{k>=0} (-1)^k (H_k+H_{k+1}) (z/2)^{2k}/(k!(k+1)!)
//                                            = 2 sum_k (H_k+H_{k+1}) t1_k
// so the four sums cost two complex multiplies per term, not four.
static void jy01_series(cd z, cd &j0, cd &j1x, cd &y0, cd &y1) {
    const cd q = -0.25 * z * z;
    cd t0(1.0, 0.0), t1(0.5, 0.0);
    j0 = t0;
    j1x = t1;
    cd s0(0.0, 0.0), s1 = 2.0 * t1;  // k = 0: H_0 = 0, H_0 + H_1 = 1
    double hk = 0.0, hk1 = 1.0;
    for (int k = 1; k <= 60; ++k) {
        t0 *= q / double(k * k);
        t1 *= q / double(k * (k + 1));
        hk += 1.0 / double(k);
        hk1 += 1.0 / double(k + 1);
        j0 += t0;
        j1x += t1;
        const cd term0 = hk * t0;
        const cd term1 = (2.0 * (hk + hk1)) * t1;
        s0 -= term0;
        s1 += term1;
        if (cnorm(t0) <= SER_EPS2 * cnorm(j0) &&
            cnorm(t1) <= SER_EPS2 * cnorm(j1x) &&
            cnorm(term0) <= SER_EPS2 * (cnorm(s0) + 1.0) &&
            cnorm(term1) <= SER_EPS2 * (cnorm(s1) + 1.0))
            break;
    }
    const cd lg = std::log(0.5 * z) + EULER_GAMMA;
    y0 = (2.0 / SPI) * (lg * j0 + s0);
    y1 = (2.0 / SPI) * (lg * (j1x * z) - 1.0 / z) - (z / (2.0 * SPI)) * s1;
}

// The A&S 9.2 asymptotic sums for orders 0 and 1 at once, each with its own
// optimal truncation:
//   H^(kind)_nu(z) ~ sqrt(2/(pi z)) e^{s i (z - nu pi/2 - pi/4)} sum_k t^nu_k
//   t^nu_0 = 1,  t^nu_{k+1} = t^nu_k (4 nu^2 - (2k+1)^2)/(8(k+1)) (s i/z)
// with s = +1 for kind 1, -1 for kind 2. Valid away from the negative real
// axis, which the contours never approach.
//
// The orders differ only in the numerator 4 nu^2 - (2k+1)^2, so one loop
// over the shared (i/z) ladder yields both; and t^nu_k carries (s i/z)^k,
// so the kind-1 terms are the kind-2 terms with alternating signs — no
// second pass. `want_plus` skips the kind-1 accumulation for the Hankel
// form, which never needs it. Each order keeps its own divergence-onset
// break, so both truncate exactly where a per-order loop would.
static void hankel_asym_sums(cd z, bool want_plus, cd &s0m, cd &s1m, cd &s0p,
                             cd &s1p) {
    const cd iz = -CI / z;  // s = -1
    cd t0(1.0, 0.0), t1(1.0, 0.0);
    s0m = s1m = s0p = s1p = cd(1.0, 0.0);
    double prev0 = 1e300, prev1 = 1e300;
    bool live0 = true, live1 = true;
    for (int k = 0; k < 40 && (live0 || live1); ++k) {
        const double odd = double(2 * k + 1);
        const double odd2 = odd * odd;
        const double den = 8.0 * double(k + 1);
        const double sgn = (k & 1) ? 1.0 : -1.0;  // (-1)^{k+1}: term k+1
        if (live0) {
            t0 *= (-odd2 / den) * iz;
            const double a = cnorm(t0);
            if (a >= prev0) {
                live0 = false;  // divergence onset: stop at the optimum
            } else {
                s0m += t0;
                if (want_plus) s0p += sgn * t0;
                prev0 = a;
                if (a <= SER_EPS2 * cnorm(s0m)) live0 = false;
            }
        }
        if (live1) {
            t1 *= ((4.0 - odd2) / den) * iz;
            const double a = cnorm(t1);
            if (a >= prev1) {
                live1 = false;
            } else {
                s1m += t1;
                if (want_plus) s1p += sgn * t1;
                prev1 = a;
                if (a <= SER_EPS2 * cnorm(s1m)) live1 = false;
            }
        }
    }
}

static const double BESSEL_SWITCH2 = 144.0;  // (12.0)^2

// (J0(x), J1(x)/x) — the Bessel-form pair of _bessel_j0_j1x.
static void bessel_j0_j1x(cd x, cd &b0, cd &b1x) {
    if (cnorm(x) <= BESSEL_SWITCH2) {
        j01_series(x, b0, b1x);
        return;
    }
    // J_nu = (H1_nu + H2_nu)/2. With W = sqrt(2/(pi x)) and E = e^{-i(x -
    // pi/4)}: H2_0 = W E S0m, H2_1 = i W E S1m, H1_0 = (W/E) S0p and
    // H1_1 = -i (W/E) S1p — one sqrt and one exp for all four, where the
    // per-order calls did four of each.
    cd s0m, s1m, s0p, s1p;
    hankel_asym_sums(x, true, s0m, s1m, s0p, s1p);
    const cd w = std::sqrt(2.0 / (SPI * x));
    const cd e = std::exp(-CI * (x - 0.25 * SPI));
    const cd p = w * e, pinv = w / e;
    b0 = 0.5 * (pinv * s0p + p * s0m);
    b1x = (0.5 * CI) * (p * s1m - pinv * s1p) / x;
}

// (H2_0(x)/2, H2_1(x)/(2x)) — the Hankel-form pair of _integrand_six.
static void hankel2_half(cd x, cd &b0, cd &b1x) {
    if (cnorm(x) <= BESSEL_SWITCH2) {
        cd j0, j1x, y0, y1;
        jy01_series(x, j0, j1x, y0, y1);
        b0 = 0.5 * (j0 - CI * y0);
        b1x = 0.5 * (j1x * x - CI * y1) / x;
        return;
    }
    cd s0m, s1m, s0p, s1p;
    hankel_asym_sums(x, false, s0m, s1m, s0p, s1p);
    const cd p = std::sqrt(2.0 / (SPI * x)) * std::exp(-CI * (x - 0.25 * SPI));
    b0 = 0.5 * p * s0m;
    b1x = (0.5 * CI) * p * s1m / x;
}

// ---- the six integrands and quadrature (ports of the Python names) -------

// gamma(lam, k) = sqrt(-j(lam-k)) sqrt(j(lam+k)): NEC's vertical cuts.
static inline cd gam(cd lam, cd k) {
    return std::sqrt(-CI * (lam - k)) * std::sqrt(CI * (lam + k));
}

struct Six {
    cd v[6];
    Six() : v{} {}
    Six &operator+=(const Six &o) {
        for (int i = 0; i < 6; ++i) v[i] += o.v[i];
        return *this;
    }
    Six &operator-=(const Six &o) {
        for (int i = 0; i < 6; ++i) v[i] -= o.v[i];
        return *this;
    }
};
static inline Six operator+(Six a, const Six &b) { return a += b; }

static inline double absmax(const Six &s) {
    double m = 0.0;
    for (int i = 0; i < 6; ++i) m = std::max(m, std::abs(s.v[i]));
    return m;
}
static inline double absmax_diff(const Six &a, const Six &b) {
    double m = 0.0;
    for (int i = 0; i < 6; ++i) m = std::max(m, std::abs(a.v[i] - b.v[i]));
    return m;
}

struct SommCtx {
    double rho, h;
    cd k1, k2;
    bool bessel;
};

// The six lambda-integrands of NEC eqs 148-153 (= _integrand_six).
static inline void integrand_six(const SommCtx &c, cd lam, cd out[6]) {
    const cd g1 = gam(lam, c.k1);
    const cd g2 = gam(lam, c.k2);
    const cd k1s = c.k1 * c.k1;
    const cd k2s = c.k2 * c.k2;
    const cd d1 = 2.0 / (g1 + g2) - 2.0 * k2s / (g2 * (k1s + k2s));
    const cd d2 = 2.0 / (k1s * g2 + k2s * g1) - 2.0 / (g2 * (k1s + k2s));
    const cd e = std::exp(-g2 * c.h);
    const cd x = lam * c.rho;
    cd b0, b1x;
    if (c.bessel)
        bessel_j0_j1x(x, b0, b1x);
    else
        hankel2_half(x, b0, b1x);
    const cd l2 = lam * lam;
    const cd l3 = l2 * lam;
    const cd common = d2 * e;
    out[0] = common * (b1x - b0) * l3;
    out[1] = common * g2 * g2 * b0 * lam;
    out[2] = common * g2 * (b1x * x) * l2;
    out[3] = -common * b1x * l3;
    out[4] = common * b0 * lam;
    out[5] = d1 * e * b0 * lam;
}

static Six gauss_segment(const SommCtx &c, cd z0, cd z1) {
    const cd mid = 0.5 * (z0 + z1);
    const cd half = 0.5 * (z1 - z0);
    Six acc;
    cd f[6];
    for (int q = 0; q < 24; ++q) {
        integrand_six(c, mid + half * GX[q], f);
        const cd w = GW[q] * half;
        for (int i = 0; i < 6; ++i) acc.v[i] += f[i] * w;
    }
    return acc;
}

// Recursive bisection Gauss quadrature, relative tolerance
// (= _adaptive_segment; see its docstring for why relative).
static Six adaptive_segment(const SommCtx &c, cd z0, cd z1, double rtol,
                            int depth, const Six *whole_in) {
    Six whole = whole_in ? *whole_in : gauss_segment(c, z0, z1);
    const cd mid = 0.5 * (z0 + z1);
    Six left = gauss_segment(c, z0, mid);
    Six right = gauss_segment(c, mid, z1);
    Six better = left + right;
    const double err = absmax_diff(better, whole);
    const double scale = absmax(better);
    if (depth <= 0 || err <= rtol * std::max(scale, 1e-300)) return better;
    return adaptive_segment(c, z0, mid, rtol, depth - 1, &left) +
           adaptive_segment(c, mid, z1, rtol, depth - 1, &right);
}

// Panel tail with geometric ramp from panel0 (<= 0 means "none") — see
// _tail's docstring for the ramp rationale and the `== 0.0` quiet trigger.
static Six tail(const SommCtx &c, cd z0, cd direction, double panel,
                double rtol, double ref_scale, double panel0) {
    const int max_panels = 800;
    Six total;
    int quiet = 0;
    cd z = z0;
    double step = (panel0 > 0.0) ? std::min(panel0, panel) : panel;
    for (int i = 0; i < max_panels; ++i) {
        const cd z_next = z + step * direction;
        Six contrib = gauss_segment(c, z, z_next);
        total += contrib;
        z = z_next;
        step = std::min(2.0 * step, panel);
        const double scale = std::max(absmax(total), ref_scale);
        const double cmax = absmax(contrib);
        if (cmax == 0.0 || cmax < rtol * scale) {
            if (++quiet >= 2) break;
        } else {
            quiet = 0;
        }
    }
    return total;
}

// = _six_integrals for one (rho, h); form: 0 auto (rho < 2h -> Bessel),
// 1 force Bessel, 2 force Hankel. eps_t == 1 is short-circuited by the
// caller (and again here for safety).
static void six_integrals(cd eps_t, double k2d, double rho, double h,
                          double rtol, int form, cd out[6]) {
    for (int i = 0; i < 6; ++i) out[i] = cd(0.0, 0.0);
    if (eps_t == cd(1.0, 0.0)) return;
    const cd k2(k2d, 0.0);
    cd k1 = k2d * std::sqrt(eps_t);
    if (k1.imag() > 0) k1 = std::conj(k1);
    const double scale = std::max(rho, h);
    const double panel = 0.2 * SPI / scale;
    const double kmax = std::max(std::abs(k1), k2d);
    const double kcap = 1.2 * k2d + 50.0 / scale;
    const double kmax_eff = std::min(kmax, kcap);
    const double qtol = std::min(rtol, 1e-11);
    const bool use_bessel = (form == 0) ? (rho < 2.0 * h) : (form == 1);
    SommCtx c{rho, h, k1, k2, use_bessel};
    Six total;
    if (use_bessel) {
        const double p = std::min(rho > 0.0 ? 1.0 / rho : 1e300, 1.0 / h);
        const cd brk(p, p);
        const cd end_adapt(1.3 * kmax_eff + 3.0 * p, p);
        total = adaptive_segment(c, cd(0.0, 0.0), brk, qtol, ADAPT_DEPTH, nullptr);
        cd tail_start = brk;
        if (end_adapt.real() > brk.real()) {
            total += adaptive_segment(c, brk, end_adapt, qtol, ADAPT_DEPTH, nullptr);
            tail_start = end_adapt;
        }
        total += tail(c, tail_start, cd(1.0, 0.0), panel, rtol, absmax(total), -1.0);
    } else {
        const double r1 = std::hypot(rho, h);
        const cd dir_right = cd(c.h, -c.rho) / r1;
        const cd dir_left = cd(-c.h, -c.rho) / r1;
        const cd a(0.0, -0.4 * k2d);
        const cd b = cd(0.6, 0.2) * k2d;
        const cd cc = cd(1.02, 0.2) * k2d;
        // Waypoint d must clear the k1 branch point: gamma_1's cut runs
        // straight DOWN from +k1, so a d left of k1.real starts the
        // descending tail on the far side of that cut and flips gamma_1's
        // sign over the live part of the contour (issue #161). `kcap` is
        // keyed to max(rho, h) and at grazing falls below k1, so cap only
        // once the branch point is numerically dead -- the a->d run
        // carries e^{-gamma_2 h} * H0(2)(lam*rho) ~ e^{-(k1r*h - k1i*rho)}
        // there. See _sommerfeld._six_integrals for the full rationale and
        // the |k1| > 200 k2 (PEC-limit) escape.
        const bool k1_dead = k1.real() * h - k1.imag() * rho >= 50.0;
        const double cap_d = (k1_dead || std::abs(k1) > 200.0 * k2d)
                                 ? kcap
                                 : std::max(kcap, 1.01 * k1.real());
        cd d;
        if (1.01 * k1.real() <= cap_d)
            d = cd(1.01 * k1.real(), 0.99 * std::max(k1.imag(), -cap_d));
        else
            d = cd(cap_d, 0.0);
        if (d.real() < 1.1 * k2d) d = cd(1.1 * k2d, d.imag());
        total = adaptive_segment(c, a, b, qtol, ADAPT_DEPTH, nullptr);
        total += adaptive_segment(c, b, cc, qtol, ADAPT_DEPTH, nullptr);
        total += adaptive_segment(c, cc, d, qtol, ADAPT_DEPTH, nullptr);
        const double ref = absmax(total);
        const double p0 = 0.5 * kmax;
        total += tail(c, d, dir_right, panel, rtol, ref, p0);
        total -= tail(c, a, dir_left, panel, rtol, ref, p0);
    }
    for (int i = 0; i < 6; ++i) out[i] = total.v[i];
}

}  // namespace somm

// Batched entry point: the six NEC lambda-integrals at each (rho[i], h[i]),
// OpenMP across nodes (each node's adaptive quadrature is independent).
// Returns (n, 6) complex in _six_integrals order (Vrr, Vzz, Vrz, Vr1, V, U).
static py::array_t<std::complex<double>> somm_six_integrals_batch(
    std::complex<double> eps_t, double k2,
    py::array_t<double, py::array::c_style | py::array::forcecast> rho,
    py::array_t<double, py::array::c_style | py::array::forcecast> h,
    double rtol, int form, uintptr_t cancel_flag = 0) {
    auto rb = rho.unchecked<1>();
    auto hb = h.unchecked<1>();
    const py::ssize_t n = rb.shape(0);
    if (hb.shape(0) != n)
        throw std::invalid_argument("rho and h must have the same length");
    if (form < 0 || form > 2)
        throw std::invalid_argument("form must be 0 (auto), 1 (J) or 2 (H)");
    for (py::ssize_t i = 0; i < n; ++i) {
        if (rb(i) < 0.0 || hb(i) < 0.0 || (rb(i) == 0.0 && hb(i) == 0.0))
            throw std::invalid_argument(
                "need rho, h >= 0 and R1 > 0 at every node");
    }
    py::array_t<std::complex<double>> out({n, py::ssize_t(6)});
    auto ob = out.mutable_unchecked<2>();
    const somm::cd et(eps_t);
    PYSIM_CANCEL_SETUP(cancel_flag);

    #pragma omp parallel for schedule(dynamic)
    for (py::ssize_t i = 0; i < n; ++i) {
        PYSIM_CANCEL_POLL();
        somm::cd res[6];
        somm::six_integrals(et, k2, rb(i), hb(i), rtol, form, res);
        for (int j = 0; j < 6; ++j) ob(i, j) = res[j];
    }
    PYSIM_THROW_IF_ABORTED();
    return out;
}

// ---------------------------------------------------------------------------
// Sommerfeld remainder assembly (sommerfeld-perf-plan Phase 4b).
//
// Fused C++ port of _sommerfeld.remainder_field_proj (which internally calls
// SommerfeldGrid.eval). Per (observer m, source n) pair: interpolate the four
// smooth-remainder surfaces from the tabulated grid with the same 4x4 Lagrange
// (bivariate cubic) stencil, combine per the theory-manual eqs 143-147 azimuth
// factors, and project onto the observer tangent -- with NO materialized
// (4,n,4,4) intermediate (the numpy bottleneck). One OpenMP loop over observer
// rows. Bit-for-bit the same arithmetic as the Python fallback; cross-checked
// in tests/test_sommerfeld_accel.py.
//
// Clean-room: ported from momwire's own _sommerfeld.py. No GPL Sommerfeld
// source (nec2c, nec2++/PyNEC, somnec) was consulted.
namespace somm_proj {
using cd = std::complex<double>;

// Cubic Lagrange weights for nodes at 0,1,2,3 evaluated at u (== _lagrange4).
static inline void lagrange4(double u, double *w) {
    const double u0 = u, u1 = u - 1.0, u2 = u - 2.0, u3 = u - 3.0;
    w[0] = -u1 * u2 * u3 / 6.0;
    w[1] = u0 * u2 * u3 / 2.0;
    w[2] = -u0 * u1 * u3 / 2.0;
    w[3] = u0 * u1 * u2 / 6.0;
}

// The tabulated SommerfeldGrid, flattened for the inner loop: raw pointers to
// the three near (plus optionally two far, issue #159) regions'
// (4, n_r, n_th) C-contiguous value tables plus their axis origins/spacings,
// and the region-select breakpoints. Populated from the pybind arrays by both
// callers (build_grid_view).
struct GridView {
    const cd *vptr[5];
    py::ssize_t nR[5], nTh[5];
    double rr0[5], rdr[5], rth0[5], rdth[5];
    double r1_max, r_break, th_split, r_near, tiny, half_pi;
};

static GridView build_grid_view(
    double r1_max, double r_break, double th_split, double r_near,
    const py::detail::unchecked_reference<double, 1> &r0b,
    const py::detail::unchecked_reference<double, 1> &drb,
    const py::detail::unchecked_reference<double, 1> &th0b,
    const py::detail::unchecked_reference<double, 1> &dthb,
    const std::vector<py::array_t<cd, py::array::c_style | py::array::forcecast>>
        &reg_vals) {
    const size_t n_reg = reg_vals.size();
    if (n_reg != 3 && n_reg != 5)
        throw std::runtime_error("expected 3 or 5 region value tables");
    GridView G;
    for (size_t g = 0; g < n_reg; ++g) {
        auto v = reg_vals[g].template unchecked<3>();
        if (v.shape(0) != 4)
            throw std::runtime_error("region values must have shape (4, n_r, n_th)");
        G.vptr[g] = reg_vals[g].data();
        G.nR[g] = v.shape(1);
        G.nTh[g] = v.shape(2);
        G.rr0[g] = r0b(g);
        G.rdr[g] = drb(g);
        G.rth0[g] = th0b(g);
        G.rdth[g] = dthb(g);
    }
    G.r1_max = r1_max;
    G.r_break = r_break;
    G.th_split = th_split;
    // 3-region grids have r_near == r1_max, so clamped queries never route
    // far; guard anyway so a stale r_near can't index missing tables.
    G.r_near = n_reg == 5 ? r_near : r1_max;
    G.tiny = 1e-12 * r1_max;
    G.half_pi = 0.5 * M_PI;
    return G;
}

// Interpolated + projected smooth-remainder field for ONE (observer, source)
// pair: t_obs . F(r_obs, r_src) . t_src. `sux/suy/sthsrc/stzsrc` are the
// source tangent's horizontal-unit / horizontal-magnitude / vertical parts
// (precomputed once per source point by the caller). Inlines
// SommerfeldGrid.eval + remainder_field_proj's eqs 143-147 with no
// intermediates. Bit-for-bit the numpy body.
static inline cd proj_one(
    const GridView &G, double ground_z, double k,
    double ox, double oy, double oz, double tox, double toy, double toz,
    double sx, double sy, double sz, double sux, double suy, double sthsrc,
    double stzsrc) {
    const double dx = ox - sx;
    const double dy = oy - sy;
    const double rho = std::hypot(dx, dy);
    const double hh = (oz - ground_z) + (sz - ground_z);
    const double r1 = std::sqrt(rho * rho + hh * hh);

    // --- inline SommerfeldGrid.eval(r1, theta) ---
    double theta = std::atan2(hh, rho);
    if (theta < 0.0) theta = 0.0; else if (theta > G.half_pi) theta = G.half_pi;
    const double r1c = r1 > G.r1_max ? G.r1_max : r1;  // interp clamps; g uses r1
    const int reg = (r1c <= G.r_break)
                        ? 0
                        : (r1c <= G.r_near ? (theta <= G.th_split ? 1 : 2)
                                           : (theta <= G.th_split ? 3 : 4));
    const double fr = (r1c - G.rr0[reg]) / G.rdr[reg];
    const double ft = (theta - G.rth0[reg]) / G.rdth[reg];
    int i0 = (int)std::floor(fr) - 1;
    int j0 = (int)std::floor(ft) - 1;
    if (i0 < 0) i0 = 0; else if (i0 > G.nR[reg] - 4) i0 = (int)G.nR[reg] - 4;
    if (j0 < 0) j0 = 0; else if (j0 > G.nTh[reg] - 4) j0 = (int)G.nTh[reg] - 4;
    double wr[4], wt[4];
    lagrange4(fr - i0, wr);
    lagrange4(ft - j0, wt);
    const cd *V = G.vptr[reg];
    const py::ssize_t nth = G.nTh[reg], nr = G.nR[reg];
    cd surf[4];
    for (int s = 0; s < 4; ++s) {
        const cd *plane = V + (py::ssize_t)s * nr * nth;
        cd acc(0.0, 0.0);
        for (int i = 0; i < 4; ++i) {
            const cd *row = plane + (py::ssize_t)(i0 + i) * nth + j0;
            cd rs = row[0] * wt[0] + row[1] * wt[1] + row[2] * wt[2] +
                    row[3] * wt[3];
            acc += rs * wr[i];
        }
        surf[s] = acc;
    }
    const cd IrhoV = surf[0], IzV = surf[1], IrhoH = surf[2], IphiH = surf[3];

    // --- projection (eqs 143-147) ---
    const cd g = std::polar(1.0 / r1, -k * r1);
    const bool safe_r = rho > G.tiny;
    const double inv_rho = safe_r ? 1.0 / rho : 0.0;
    const double dhx = safe_r ? dx * inv_rho : sux;
    const double dhy = safe_r ? dy * inv_rho : suy;
    const double cphi = sux * dhx + suy * dhy;
    const double sphi = sux * dhy - suy * dhx;
    const cd e_rho = g * (stzsrc * IrhoV + sthsrc * cphi * IrhoH);
    const cd e_phi = g * (sthsrc * sphi * IphiH);
    const cd e_z = g * (stzsrc * IzV - sthsrc * cphi * IrhoV);
    return tox * (dhx * e_rho - dhy * e_phi) +
           toy * (dhy * e_rho + dhx * e_phi) + toz * e_z;
}

// Decompose a tangent (tx,ty,tz) into (horizontal unit x,y; horizontal
// magnitude; vertical), matching remainder_field_proj's th_src/ux/uy/tz_src.
static inline void tangent_decomp(double tx, double ty, double tz, double &ux,
                                  double &uy, double &th, double &tzc) {
    th = std::hypot(tx, ty);
    const bool safe = th > 1e-12;
    ux = safe ? tx / th : 1.0;
    uy = safe ? ty / th : 0.0;
    tzc = tz;
}
}  // namespace somm_proj

// obs/t_obs (M,3), src/t_src (S,3); returns (M,S) complex. The grid is passed
// flattened: the three regions' (r0, dr, th0, dth) as length-3 arrays, plus a
// list of three (4, n_r, n_th) complex value tables. r_break / th_split select
// the region exactly as SommerfeldGrid.eval.
static py::array_t<std::complex<double>> remainder_field_proj_batch(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_obs,
    py::array_t<double, py::array::c_style | py::array::forcecast> src,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_src,
    double ground_z, double k, double r1_max, double r_break, double th_split,
    double r_near,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_r0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dr,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_th0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dth,
    std::vector<py::array_t<std::complex<double>,
                            py::array::c_style | py::array::forcecast>> reg_vals,
    uintptr_t cancel_flag = 0) {
    using somm_proj::cd;
    auto ob = obs.unchecked<2>();
    auto tob = t_obs.unchecked<2>();
    auto sb = src.unchecked<2>();
    auto tsb = t_src.unchecked<2>();
    if (ob.shape(1) != 3 || tob.shape(1) != 3 || sb.shape(1) != 3 ||
        tsb.shape(1) != 3)
        throw std::runtime_error("obs/src/tangent arrays must have shape (*, 3)");
    if (ob.shape(0) != tob.shape(0) || sb.shape(0) != tsb.shape(0))
        throw std::runtime_error("points and tangents must have matching length");

    const py::ssize_t M = ob.shape(0);
    const py::ssize_t S = sb.shape(0);
    somm_proj::GridView G = somm_proj::build_grid_view(
        r1_max, r_break, th_split, r_near, reg_r0.unchecked<1>(),
        reg_dr.unchecked<1>(), reg_th0.unchecked<1>(), reg_dth.unchecked<1>(),
        reg_vals);

    py::array_t<std::complex<double>> out({M, S});
    auto out_m = out.mutable_unchecked<2>();

    py::gil_scoped_release release;

    // Per-source constants (n-only): position + tangent decomposition.
    std::vector<double> sx(S), sy(S), sz(S), ux(S), uy(S), thsrc(S), tzsrc(S);
    for (py::ssize_t n = 0; n < S; ++n) {
        sx[n] = sb(n, 0);
        sy[n] = sb(n, 1);
        sz[n] = sb(n, 2);
        somm_proj::tangent_decomp(tsb(n, 0), tsb(n, 1), tsb(n, 2), ux[n], uy[n],
                                  thsrc[n], tzsrc[n]);
    }

    PYSIM_CANCEL_SETUP(cancel_flag);
    #pragma omp parallel for schedule(static)
    for (py::ssize_t m = 0; m < M; ++m) {
        PYSIM_CANCEL_POLL();
        const double ox = ob(m, 0), oy = ob(m, 1), oz = ob(m, 2);
        const double tox = tob(m, 0), toy = tob(m, 1), toz = tob(m, 2);
        for (py::ssize_t n = 0; n < S; ++n) {
            out_m(m, n) = somm_proj::proj_one(
                G, ground_z, k, ox, oy, oz, tox, toy, toz, sx[n], sy[n], sz[n],
                ux[n], uy[n], thsrc[n], tzsrc[n]);
        }
    }
    PYSIM_THROW_IF_ABORTED();
    return out;
}

// Fully-fused b-spline Galerkin Sommerfeld remainder (Phase 4b, stage 2):
// returns the (n_basis, n_basis) block Q directly, absorbing the moment
// quadrature and the basis assembly that the Python code did with two einsums
// and a large fancy-index gather. Segments are the shared obs=src set.
//
//   Jf[p,P,i,j] = sum_{qi,rj} W[p,i,qi] * proj(node[i,qi], node[j,rj]) * W[P,j,rj]
//   Q[m,n]      = sum_{a,b,p,P} polys[m,a,p] * Jf[p,P, supp[m,a], supp[n,b]]
//                                            * polys[n,b,P]
//
// Rectangular obs/src form: the dense b-spline block is the symmetric case
// (obs == src, loc == supp_seg), the ACA sampler passes thin obs/src segment
// subsets with local support maps. obs_nodes (nsI, q, 3), obs_tang (nsI, 3),
// W_obs (d+1, nsI, q); likewise src_*; loc_I (nI, d+1) int64 indexes obs
// segments, pI (nI, d+1, d+1); loc_J/pJ index src. Grid passed as in
// remainder_field_proj_batch. Returns Q (nI, nJ).
static py::array_t<std::complex<double>> sommerfeld_remainder_bspline_Q(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_nodes,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tang,
    py::array_t<double, py::array::c_style | py::array::forcecast> W_obs,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_nodes,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tang,
    py::array_t<double, py::array::c_style | py::array::forcecast> W_src,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> loc_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> pI,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> loc_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> pJ,
    double ground_z, double k, double r1_max, double r_break, double th_split,
    double r_near,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_r0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dr,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_th0,
    py::array_t<double, py::array::c_style | py::array::forcecast> reg_dth,
    std::vector<py::array_t<std::complex<double>,
                            py::array::c_style | py::array::forcecast>> reg_vals,
    uintptr_t cancel_flag = 0) {
    using somm_proj::cd;
    auto ndI = obs_nodes.unchecked<3>();
    auto tgI = obs_tang.unchecked<2>();
    auto WvI = W_obs.unchecked<3>();
    auto ndJ = src_nodes.unchecked<3>();
    auto tgJ = src_tang.unchecked<2>();
    auto WvJ = W_src.unchecked<3>();
    auto lI = loc_I.unchecked<2>();
    auto lJ = loc_J.unchecked<2>();
    auto plI = pI.unchecked<3>();
    auto plJ = pJ.unchecked<3>();

    const py::ssize_t nsI = ndI.shape(0);
    const py::ssize_t nsJ = ndJ.shape(0);
    const py::ssize_t q = ndI.shape(1);
    const int d1 = (int)WvI.shape(0);  // degree + 1
    const py::ssize_t nI = lI.shape(0);
    const py::ssize_t nJ = lJ.shape(0);
    if (ndI.shape(2) != 3 || ndJ.shape(2) != 3 || ndJ.shape(1) != q)
        throw std::runtime_error("obs/src nodes must be (n_seg, q, 3), same q");
    if (WvI.shape(1) != nsI || WvI.shape(2) != q || WvJ.shape(1) != nsJ ||
        WvJ.shape(2) != q || (int)WvJ.shape(0) != d1)
        throw std::runtime_error("W_obs/W_src inconsistent with nodes/degree");
    if (lI.shape(1) != d1 || lJ.shape(1) != d1 || plI.shape(1) != d1 ||
        plI.shape(2) != d1 || plJ.shape(1) != d1 || plJ.shape(2) != d1)
        throw std::runtime_error("loc/polys inconsistent with degree");

    somm_proj::GridView G = somm_proj::build_grid_view(
        r1_max, r_break, th_split, r_near, reg_r0.unchecked<1>(),
        reg_dr.unchecked<1>(), reg_th0.unchecked<1>(), reg_dth.unchecked<1>(),
        reg_vals);

    py::array_t<std::complex<double>> Q({nI, nJ});
    auto Qm = Q.mutable_unchecked<2>();

    py::gil_scoped_release release;

    // Per-src-segment tangent decomposition.
    std::vector<double> sux(nsJ), suy(nsJ), sth(nsJ), stz(nsJ);
    for (py::ssize_t j = 0; j < nsJ; ++j)
        somm_proj::tangent_decomp(tgJ(j, 0), tgJ(j, 1), tgJ(j, 2), sux[j],
                                  suy[j], sth[j], stz[j]);

    // Stage 1: Jf[p,P,i,j] over the (nsI, nsJ) segment rectangle. Flat index
    // (((p*d1+P)*nsI)+i)*nsJ+j.
    std::vector<cd> Jf((size_t)d1 * d1 * nsI * nsJ);
    const size_t seg2 = (size_t)nsI * nsJ;

    PYSIM_CANCEL_SETUP(cancel_flag);
    #pragma omp parallel for schedule(dynamic)
    for (py::ssize_t i = 0; i < nsI; ++i) {
        PYSIM_CANCEL_POLL();
        const double tox = tgI(i, 0), toy = tgI(i, 1), toz = tgI(i, 2);
        std::vector<cd> fblk((size_t)q * q);
        for (py::ssize_t j = 0; j < nsJ; ++j) {
            for (py::ssize_t qi = 0; qi < q; ++qi) {
                const double ox = ndI(i, qi, 0), oy = ndI(i, qi, 1),
                             oz = ndI(i, qi, 2);
                for (py::ssize_t rj = 0; rj < q; ++rj) {
                    fblk[qi * q + rj] = somm_proj::proj_one(
                        G, ground_z, k, ox, oy, oz, tox, toy, toz,
                        ndJ(j, rj, 0), ndJ(j, rj, 1), ndJ(j, rj, 2),
                        sux[j], suy[j], sth[j], stz[j]);
                }
            }
            for (int p = 0; p < d1; ++p) {
                for (int P = 0; P < d1; ++P) {
                    cd acc(0.0, 0.0);
                    for (py::ssize_t qi = 0; qi < q; ++qi) {
                        const double wp = WvI(p, i, qi);
                        cd row(0.0, 0.0);
                        for (py::ssize_t rj = 0; rj < q; ++rj)
                            row += fblk[qi * q + rj] * WvJ(P, j, rj);
                        acc += wp * row;
                    }
                    Jf[((size_t)(p * d1 + P) * nsI + i) * nsJ + j] = acc;
                }
            }
        }
    }
    PYSIM_THROW_IF_ABORTED();

    // Stage 2: basis assembly Q[m,n] from the segment moment tensor.
    #pragma omp parallel for schedule(static)
    for (py::ssize_t m = 0; m < nI; ++m) {
        PYSIM_CANCEL_POLL();
        for (py::ssize_t n = 0; n < nJ; ++n) {
            cd qmn(0.0, 0.0);
            for (int a = 0; a < d1; ++a) {
                const py::ssize_t si = lI(m, a);
                for (int b = 0; b < d1; ++b) {
                    const py::ssize_t sj = lJ(n, b);
                    cd inner(0.0, 0.0);
                    for (int p = 0; p < d1; ++p) {
                        const double pma = plI(m, a, p);
                        const cd *jfp =
                            &Jf[((size_t)(p * d1) * nsI + si) * nsJ + sj];
                        cd s(0.0, 0.0);
                        for (int P = 0; P < d1; ++P)
                            s += jfp[(size_t)P * seg2] * plJ(n, b, P);
                        inner += pma * s;
                    }
                    qmn += inner;
                }
            }
            Qm(m, n) = qmn;
        }
    }
    PYSIM_THROW_IF_ABORTED();
    return Q;
}


PYBIND11_MODULE(_accelerators, m) {
    // Phase 2: raised by the long kernels when their cancel_flag is tripped;
    // the _accel.py wrappers remap it to momwire.SolveAborted.
    py::register_exception<AbortedError>(m, "AcceleratorAborted");

    // Capability flag, not a value (momwire#258). The two EKSCX entry points
    // dropped their build-wide `want_swapped` argument when the IRA arm went
    // per pair, and a STALE extension still exports both symbols under the
    // old arity — so `hasattr` alone would hand the new caller a TypeError
    // instead of the graceful numpy fallback the guards exist to give.
    // `sinusoidal.py` requires this attribute before it claims either
    // accelerator; an older build simply lacks it and takes the numpy
    // reference, which carries the same per-pair fix.
    m.attr("ek_ira_per_pair") = true;

    m.def("seg_seg_reg_moments_bspline_swept",
          &seg_seg_reg_moments_bspline_swept,
          "Streaming swept reg-moment kernel for the B-spline Galerkin MoM. "
          "From the precomputed pair-distance table R (N*n_qp, N*n_qp) and "
          "weight-folded local-coordinate powers wu_pow (n_d, N, n_qp), "
          "compute J[k,p,P,i,j] = sum_{q,r} wu_pow[p,i,q] (exp(-jkR)-1)/(4 pi "
          "R) wu_pow[P,j,r] for every k. Returns (n_k, n_d, n_d, N, N) "
          "complex — the streaming C++ replacement for the numpy einsum in "
          "_seg_seg_reg_moments_from_geometry_swept.",
          py::arg("R"), py::arg("wu_pow"), py::arg("k_array"));
    m.def("seg_seg_reg_moments_bspline_swept_ek",
          &seg_seg_reg_moments_bspline_swept_ek,
          "Extended-thin-wire-kernel twin of seg_seg_reg_moments_bspline_swept "
          "(momwire#270 unit 1). Same (R, wu_pow, k_array) contract and the "
          "same (n_k, n_d, n_d, N, N) output, but the smooth remainder is "
          "[(exp(-jkR) - 1)*fac + extra] / (4 pi R) with NEC Eq 89's coaxial "
          "equal-radius factor fac = 1 + T1*C2 - T2*C1 about the EK radius "
          "`a_ek` and extra = fac - fac_static — a literal transcription of "
          "_bspline_kernels._ek_reg_kernel, whose static half rides in "
          "seg_seg_static_moments_bspline_uniform_ek. Same-edge blocks are "
          "eligible in their entirety, so there are no per-pair group labels.",
          py::arg("R"), py::arg("wu_pow"), py::arg("k_array"), py::arg("a_ek"));
    m.def("seg_seg_full_moments_bspline", &seg_seg_full_moments_bspline,
          "Single-k full-kernel polynomial moment integrals for the B-spline "
          "Galerkin MoM. Returns J of shape (max_d+1, max_d+1, N_i, N_j) "
          "complex. Templated on max_d at compile time; currently "
          "instantiated for max_d in {1, 2}.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k"),
          py::arg("max_d"),
          py::arg("gl_t"), py::arg("gl_w"));
    m.def("seg_seg_full_moments_bspline_swept",
          &seg_seg_full_moments_bspline_swept,
          "Batched (swept-k) off-edge full-kernel polynomial moments for the "
          "B-spline Galerkin MoM. The per-(i,j) geometry (R table, moment "
          "weights) is built once and reused across k_array; only exp(-jkR) "
          "varies per frequency. Returns (n_k, max_d+1, max_d+1, N_i, N_j) "
          "complex; per-pair R tables are reused across the k axis.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k_array"),
          py::arg("max_d"),
          py::arg("gl_t"), py::arg("gl_w"));
    m.def("seg_seg_full_moments_bspline_ek", &seg_seg_full_moments_bspline_ek,
          "Extended-thin-wire-kernel twin of seg_seg_full_moments_bspline "
          "(momwire#270 unit 2). Same (seg_l_i, seg_r_i, seg_l_j, seg_r_j, "
          "a_squared, k, max_d, gl_t, gl_w) contract, plus per-segment "
          "coaxial-and-equal-radius group labels group_i (N_i,) / group_j "
          "(N_j,) int64 and the plain (unsquared) EK radius a_ek. A pair "
          "(i, j) is eligible iff group_i[i] == group_j[j] >= 0, evaluated "
          "once per segment pair and applied to every quadrature sub-pair "
          "inside it; G = exp(-jkR)/(4 pi R) is multiplied by NEC Eq 89's "
          "coaxial factor fac = 1 + T1*C2 - T2*C1 on eligible pairs and left "
          "alone otherwise — a literal transcription of "
          "_bspline_kernels._seg_seg_full_moments_offedge's `ek is not "
          "None` branch, evaluated eagerly instead of via np.where.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k"),
          py::arg("max_d"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("group_i"), py::arg("group_j"), py::arg("a_ek"));
    m.def("seg_seg_full_moments_bspline_swept_ek",
          &seg_seg_full_moments_bspline_swept_ek,
          "Batched (swept-k) twin of seg_seg_full_moments_bspline_ek "
          "(momwire#270 unit 2). Same (max_d+1, max_d+1, N_i, N_j) "
          "per-(i,j) geometry hoist as seg_seg_full_moments_bspline_swept, "
          "plus the EK eligibility mask and the k-independent T1/T2 halves "
          "of the coaxial factor hoisted out of the k loop the same way the "
          "same-edge swept twin hoists them. Returns (n_k, max_d+1, max_d+1, "
          "N_i, N_j) complex.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k_array"),
          py::arg("max_d"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("group_i"), py::arg("group_j"), py::arg("a_ek"));
    m.def("seg_seg_static_moments_bspline_uniform",
          &seg_seg_static_moments_bspline_uniform,
          "Closed-form same-edge static-kernel polynomial moments J_pq for a "
          "uniform-h edge with N segments. Uses Toeplitz structure (2N-1 "
          "unique values per moment) and inlined sympy-derived closed forms. "
          "Returns J_static of shape (max_d+1, max_d+1, N, N), with the "
          "1/(4π) prefactor folded in.",
          py::arg("h"), py::arg("a"), py::arg("N"), py::arg("max_d"));
    m.def("seg_seg_static_moments_bspline_uniform_ek",
          &seg_seg_static_moments_bspline_uniform_ek,
          "Extended-thin-wire-kernel twin of "
          "seg_seg_static_moments_bspline_uniform (momwire#270 unit 1). Adds "
          "the generated closed-form correction D_pq^EK — the moments of "
          "-a_ek^2/(2R^3) + 3 a_ek^4/(4R^5), the k -> 0 limit of Eq 89's "
          "coaxial factor minus 1 — to each J_pq before the 1/(4 pi) "
          "prefactor. D is translation-invariant along the edge exactly as J "
          "is, so it rides the same 2N-1 Toeplitz table. `a_ek` is separate "
          "from the regularization radius `a` because _EK.a may override it; "
          "on every eligible pair they are equal.",
          py::arg("h"), py::arg("a"), py::arg("N"), py::arg("max_d"),
          py::arg("a_ek"));
    m.def("assemble_Z_bspline_weighted_windowed", &assemble_Z_bspline_weighted_windowed,
          "Weighted + scaled windowed accumulator: like "
          "assemble_Z_bspline_windowed but with complex per-pair weights "
          "wA_win / wPhi_win on the A and charge terms and a complex scale "
          "on the window's contribution before the +=. Serves the chunked "
          "ground-image builds (PEC mirror-dot, Fresnel refl-coef, "
          "Sommerfeld constant-C2) with scale = -1 for the Z -= image "
          "convention. The weights are WINDOWS of shape (i1-i0, j1-j0) "
          "aligned with J_chunk's trailing axes, not global (N, N) tables: "
          "lookups are window-relative, so the caller never keeps two "
          "global complex (N, N) tables alive across the fill (issue #323). "
          "max_d inferred from support_seg.",
          py::arg("J_chunk"), py::arg("support_seg"),
          py::arg("polys"), py::arg("wA_win"), py::arg("wPhi_win"),
          py::arg("m_idx"), py::arg("n_idx"),
          py::arg("i0"), py::arg("i1"), py::arg("j0"), py::arg("j1"),
          py::arg("omega"), py::arg("eps_"), py::arg("mu_"),
          py::arg("scale"), py::arg("Z"), py::arg("cancel_flag") = 0);
    m.def("assemble_Z_bspline_windowed", &assemble_Z_bspline_windowed,
          "Accumulate one rectangular segment window's contribution into a "
          "caller-provided Z from a chunked moment tensor J_chunk of shape "
          "(D+1, D+1, i1-i0, j1-j0). m_idx / n_idx select the basis rows / "
          "cols with support in the window; the (zA, zPhi) -> Z mixing is "
          "linear so per-window accumulation equals the all-at-once "
          "assembly. Lets the dense build skip the full (D+1, D+1, N, N) "
          "tensor (issue #136). `tangents` is the per-segment unit tangent "
          "table, (n_segs, 3): the pair tangent dot is formed here from two "
          "rows rather than read out of an (N, N) table the caller would "
          "have to keep alive across the whole fill (issue #318). max_d is "
          "inferred from support_seg.",
          py::arg("J_chunk"), py::arg("support_seg"),
          py::arg("polys"), py::arg("tangents"),
          py::arg("m_idx"), py::arg("n_idx"),
          py::arg("i0"), py::arg("i1"), py::arg("j0"), py::arg("j1"),
          py::arg("omega"), py::arg("eps_"), py::arg("mu_"),
          py::arg("Z"), py::arg("cancel_flag") = 0);
    m.def("assemble_Z_bspline", &assemble_Z_bspline,
          "Assemble the (n_basis, n_basis) Z matrix from the polynomial-"
          "moment tensor J, per-basis polynomial coefficients, support-segment "
          "map, and tangent-dot table. Templated on max_d at compile time; "
          "currently instantiated for max_d in {1, 2}. Single-k.",
          py::arg("J"), py::arg("support_seg"),
          py::arg("polys"), py::arg("td_all"),
          py::arg("omega"), py::arg("eps"), py::arg("mu"),
          py::arg("max_d"),
          py::arg("cancel_flag") = 0);
    m.def("assemble_Z_bspline_weighted", &assemble_Z_bspline_weighted,
          "Weighted assemble_Z_bspline for the reflection-coefficient "
          "finite ground: complex per-segment-pair weight tables on both "
          "terms — wA_all (Fresnel dyad tangent table) on the A term, "
          "wPhi_all (image-charge weight) on the Φ term. Templated on "
          "max_d in {1, 2}; single-k.",
          py::arg("J"), py::arg("support_seg"),
          py::arg("polys"), py::arg("wA_all"), py::arg("wPhi_all"),
          py::arg("omega"), py::arg("eps"), py::arg("mu"),
          py::arg("max_d"),
          py::arg("cancel_flag") = 0);
    m.def("assemble_Z_bspline_swept", &assemble_Z_bspline_swept,
          "Batched (swept-k) assemble: J is (n_k, max_d+1, max_d+1, N, N) and "
          "omega is an array; the basis tables are k-independent. "
          "tangents_row / tangents_col are (n_segs, 3) per-segment tangent "
          "tables — the kernel forms each pair's dot in-kernel rather than "
          "reading an (N, N) table (issue #333); pass (tangents, tangents) "
          "for free space and (tangents, mirrored_tangents) for the PEC "
          "image term. Returns (n_k, n_basis, n_basis) — the bspline analog "
          "of triangular's batched assemble_Z.",
          py::arg("J"), py::arg("support_seg"),
          py::arg("polys"), py::arg("tangents_row"), py::arg("tangents_col"),
          py::arg("omega_array"), py::arg("eps"), py::arg("mu"),
          py::arg("max_d"));
    m.def("bspline_assemble_offedge_block", &bspline_assemble_offedge_block,
          "Fused off-edge Z[I, J] block assembly for the H-matrix / ACA "
          "solver: quadratures the a²-regularised full-kernel moments and "
          "performs the EFIE Galerkin combine in one pass, with no "
          "intermediate moment tensor. Segments are the per-block union "
          "referenced by the I/J bases; support_*_local index into them. "
          "Templated on max_d in {1, 2}; single-k.",
          py::arg("supp_I"), py::arg("polys_I"), py::arg("segl_I"),
          py::arg("segr_I"), py::arg("tan_I"),
          py::arg("supp_J"), py::arg("polys_J"), py::arg("segl_J"),
          py::arg("segr_J"), py::arg("tan_J"),
          py::arg("a_squared"), py::arg("k"), py::arg("omega"),
          py::arg("eps"), py::arg("mu"), py::arg("max_d"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("cancel_flag") = 0);
    m.def("bspline_assemble_offedge_block_refl", &bspline_assemble_offedge_block_refl,
          "Reflection-coefficient finite-ground image variant of "
          "bspline_assemble_offedge_block: J side passed pre-mirrored, the "
          "Fresnel dyad (from eps_t) weights the A term per segment pair and "
          "w_Phi = phi_c0 + phi_c1*rho_v weights the charge term. Templated "
          "on max_d in {1, 2}; single-k.",
          py::arg("supp_I"), py::arg("polys_I"), py::arg("segl_I"),
          py::arg("segr_I"), py::arg("tan_I"),
          py::arg("supp_J"), py::arg("polys_J"), py::arg("segl_J"),
          py::arg("segr_J"), py::arg("tan_J"),
          py::arg("a_squared"), py::arg("k"), py::arg("omega"),
          py::arg("eps"), py::arg("mu"), py::arg("max_d"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("eps_t"), py::arg("phi_c0"), py::arg("phi_c1"),
          py::arg("cancel_flag") = 0);
    m.def("bspline_assemble_offedge_block_ek", &bspline_assemble_offedge_block_ek,
          "Extended-thin-wire-kernel twin of bspline_assemble_offedge_block "
          "(momwire#270 unit 3). Same fused off-edge Z[I, J] block assembly "
          "contract, plus per-segment coaxial-and-equal-radius group labels "
          "group_I (nSegI,) / group_J (nSegJ,) int64 over the SAME segment "
          "unions segl_I/segl_J index, and the plain (unsquared) EK radius "
          "a_ek. A segment pair (smi, snj) is eligible iff "
          "group_I[smi] == group_J[snj] >= 0, evaluated once per basis-pair "
          "wing and applied to every quadrature sub-pair inside it before "
          "the Galerkin combine — the fused-assembler analog of "
          "seg_seg_full_moments_bspline_ek. Serves free space and the "
          "mirror_J PEC-ground image (J-side positions/tangents pre-mirrored "
          "by the caller); the reflection-coefficient image is the separate "
          "bspline_assemble_offedge_block_refl_ek. Templated on "
          "max_d in {1, 2}; single-k.",
          py::arg("supp_I"), py::arg("polys_I"), py::arg("segl_I"),
          py::arg("segr_I"), py::arg("tan_I"),
          py::arg("supp_J"), py::arg("polys_J"), py::arg("segl_J"),
          py::arg("segr_J"), py::arg("tan_J"),
          py::arg("a_squared"), py::arg("k"), py::arg("omega"),
          py::arg("eps"), py::arg("mu"), py::arg("max_d"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("group_I"), py::arg("group_J"), py::arg("a_ek"),
          py::arg("cancel_flag") = 0);
    m.def("bspline_assemble_offedge_block_refl_ek",
          &bspline_assemble_offedge_block_refl_ek,
          "bspline_assemble_offedge_block_ek and "
          "bspline_assemble_offedge_block_refl composed (momwire#269): the "
          "reflection-coefficient finite-ground image block under the "
          "extended thin-wire kernel. The coaxial factor multiplies G on "
          "eligible segment pairs before the Galerkin contraction; the "
          "Fresnel dyad (from eps_t) and w_Phi = phi_c0 + phi_c1*rho_v "
          "weight the contracted A / charge terms after it. J side passed "
          "pre-mirrored. Templated on max_d in {1, 2}; single-k.",
          py::arg("supp_I"), py::arg("polys_I"), py::arg("segl_I"),
          py::arg("segr_I"), py::arg("tan_I"),
          py::arg("supp_J"), py::arg("polys_J"), py::arg("segl_J"),
          py::arg("segr_J"), py::arg("tan_J"),
          py::arg("a_squared"), py::arg("k"), py::arg("omega"),
          py::arg("eps"), py::arg("mu"), py::arg("max_d"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("group_I"), py::arg("group_J"), py::arg("a_ek"),
          py::arg("eps_t"), py::arg("phi_c0"), py::arg("phi_c1"),
          py::arg("cancel_flag") = 0);
    m.def("assemble_Z_enrich", &assemble_Z_enrich,
          "Assemble (Z_pe, Z_ep, Z_ee) for the stable XFEM singular basis "
          "enrichment at K≥3 junctions. Each enrichment basis is "
          "Φ_sing_stable(t) = t·log(t) − Σ_p proj_coeffs[p]·t^p with "
          "t = u_norm = u/h (origin=0) or 1 − u/h (origin=1), so the "
          "enrichment is L²-orthogonal to the local polynomial space on "
          "each segment. proj_coeffs must have length degree+1 and match "
          "the polys_poly third dim. Z_ep is computed independently from "
          "Z_pe (no .T shortcut). tangents is the (n_segs, 3) per-segment "
          "unit tangent table (#334) — the tangent dot is formed in-kernel "
          "per (m, e)/(e, f) pair rather than reading an (N, N) td_all "
          "table. Single-k.",
          py::arg("spec_seg"), py::arg("spec_origin"),
          py::arg("seg_l"), py::arg("seg_r"),
          py::arg("h_per_seg"), py::arg("tangents"),
          py::arg("supp_seg_poly"), py::arg("polys_poly"),
          py::arg("a_squared"), py::arg("k"),
          py::arg("omega"), py::arg("eps"), py::arg("mu"),
          py::arg("gl_t01"), py::arg("gl_w01"),
          py::arg("proj_coeffs"));
    m.def("somm_six_integrals_batch", &somm_six_integrals_batch,
          "Batched Sommerfeld six-integral evaluation at (rho[i], h[i]) "
          "nodes; OpenMP across nodes. form: 0 auto, 1 Bessel, 2 Hankel. "
          "Returns (n, 6) complex in _six_integrals order.",
          py::arg("eps_t"), py::arg("k2"), py::arg("rho"), py::arg("h"),
          py::arg("rtol") = 1e-9, py::arg("form") = 0,
          py::arg("cancel_flag") = 0);
    m.def("sinusoidal_field_tensor", &sinusoidal_field_tensor,
          "Tangential field tensor for the NEC2 three-term basis. Returns "
          "(Phi_const, Phi_sin, Phi_cos), each (M, N) complex. obs_*/src_* "
          "can be the same arrays (free-space build) or src_* mirrored "
          "(PEC image build).",
          py::arg("obs_centers"), py::arg("obs_tangents"),
          py::arg("src_centers"), py::arg("src_tangents"),
          py::arg("seg_h"),
          py::arg("a"), py::arg("k"), py::arg("eta"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("cancel_flag") = 0);
    m.def("sinusoidal_field_tensor_refl", &sinusoidal_field_tensor_refl,
          "Reflection-coefficient finite-ground variant of "
          "sinusoidal_field_tensor: src_* are the MIRRORED image sources, "
          "and the tangential projection applies NEC's Fresnel field dyad "
          "t_m·D·E = rho_v·(t_m·E) − (rho_v+rho_h)·(t_m·p̂)(E·p̂) with "
          "rho_v/rho_h computed in-kernel at each pair's specular angle "
          "from eps_t and the k-independent tables (cos_th, px, py, tm_p, "
          "tn_p — see SinusoidalSolver._image_refl_prep). Returns "
          "(Phi_const, Phi_sin, Phi_cos), each (M, N) complex.",
          py::arg("obs_centers"), py::arg("obs_tangents"),
          py::arg("src_centers"), py::arg("src_tangents"),
          py::arg("seg_h"),
          py::arg("a"), py::arg("k"), py::arg("eta"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("cos_th"), py::arg("px"), py::arg("py"),
          py::arg("tm_p"), py::arg("tn_p"),
          py::arg("eps_t"),
          py::arg("cancel_flag") = 0);
    m.def("sinusoidal_field_tensor_ek", &sinusoidal_field_tensor_ek,
          "NEC extended-thin-wire-kernel variant of sinusoidal_field_tensor "
          "(EKSCX, nec2-1.2.1.2.f:3170-3234). Same (Phi_const, Phi_sin, "
          "Phi_cos) contract and the same observer-side scalar radius `a`, "
          "plus three per-SOURCE tables: `src_a` (NEC's BX, the source "
          "conductor radius the O(a²) expansion is about) and the int8 "
          "per-end gating codes `ind1`/`ind2` from "
          "SinusoidalSolver._ek_gating (0/1 -> GXX, 2 -> GX). "
          "EKSCX's IRA is NOT an argument: it is resolved per pair inside "
          "the kernel from that pair's own `rhx < src_a`, the same "
          "comparison that orders (rh, b) — see momwire#258, which removed "
          "the build-wide `want_swapped` scalar #245 took. "
          "src_* may be the MIRRORED image sources (the gating tables are "
          "unchanged there, as in NEC's KSYMP loop).",
          py::arg("obs_centers"), py::arg("obs_tangents"),
          py::arg("src_centers"), py::arg("src_tangents"),
          py::arg("seg_h"),
          py::arg("a"), py::arg("k"), py::arg("eta"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("src_a"), py::arg("ind1"), py::arg("ind2"),
          py::arg("cancel_flag") = 0);
    m.def("sinusoidal_field_tensor_ek_refl", &sinusoidal_field_tensor_ek_refl,
          "Extended-thin-wire-kernel (EKSCX) variant of "
          "sinusoidal_field_tensor_refl: the EK tables of "
          "sinusoidal_field_tensor_ek (src_a / ind1 / ind2, with EKSCX's IRA "
          "resolved per pair in-kernel) "
          "with the Fresnel field dyad projection tail of "
          "sinusoidal_field_tensor_refl (cos_th, px, py, tm_p, tn_p, eps_t) "
          "in place of the plain tangential one. src_* are the MIRRORED "
          "image sources. Returns (Phi_const, Phi_sin, Phi_cos), each "
          "(M, N) complex.",
          py::arg("obs_centers"), py::arg("obs_tangents"),
          py::arg("src_centers"), py::arg("src_tangents"),
          py::arg("seg_h"),
          py::arg("a"), py::arg("k"), py::arg("eta"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("src_a"), py::arg("ind1"), py::arg("ind2"),
          py::arg("cos_th"), py::arg("px"), py::arg("py"),
          py::arg("tm_p"), py::arg("tn_p"),
          py::arg("eps_t"),
          py::arg("cancel_flag") = 0);
    m.def("sinusoidal_galerkin_far_fill", &sinusoidal_galerkin_far_fill,
          "Fused far fill for SinusoidalGalerkinSolver: the plainly-projected "
          "Eqs 76-79 tensor at each test segment's nq Gauss points, reduced "
          "on the fly against the test weights. Returns (contrib_const, "
          "contrib_sin, contrib_cos), each (nnz, N) complex, where nnz is the "
          "CSR support-entry count (`starts` is that CSR's per-test-segment "
          "row index). src_* are the geometry's segments (free-space block) "
          "or the MIRRORED ones (PEC image block); the reflection-coefficient "
          "and Sommerfeld blocks stay on the numpy path.",
          py::arg("obs_centers"), py::arg("obs_tangents"),
          py::arg("obs_radius"),
          py::arg("src_centers"), py::arg("src_tangents"), py::arg("src_hh"),
          py::arg("k"), py::arg("eta"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("w_entry"), py::arg("starts"),
          py::arg("cancel_flag") = 0);
    m.def("sinusoidal_galerkin_far_fill_ek", &sinusoidal_galerkin_far_fill_ek,
          "Extended-kernel twin of sinusoidal_galerkin_far_fill "
          "(momwire#246): the same fused far fill, plus the folded EK delta "
          "on the pairs `eligible` selects. `src_a` is one radius per SOURCE "
          "segment, `eligible` the pair rule's (obs rows, N) mask, and "
          "`ek_gx`/`ek_gw` the composite sinh-mapped rule "
          "`SinusoidalSolver._ek_delta_rule` built — passed in, not rebuilt "
          "here, so both backends integrate the delta on the same nodes. "
          "Returns the same three (nnz, N) arrays.",
          py::arg("obs_centers"), py::arg("obs_tangents"),
          py::arg("obs_radius"),
          py::arg("src_centers"), py::arg("src_tangents"), py::arg("src_hh"),
          py::arg("k"), py::arg("eta"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("w_entry"), py::arg("starts"),
          py::arg("src_a"), py::arg("eligible"),
          py::arg("ek_gx"), py::arg("ek_gw"),
          py::arg("cancel_flag") = 0);
    m.def("remainder_field_proj_batch", &remainder_field_proj_batch,
          "Fused Sommerfeld smooth-remainder assembly: interpolate the four "
          "grid surfaces (4x4 Lagrange) and project t_m.F(r_m,r_n).t_n per "
          "(observer, source) pair; OpenMP over observer rows. Returns (M, S) "
          "complex. The grid is passed flattened (per-region r0/dr/th0/dth "
          "arrays + a list of 3 near — or 5 with the #159 far zone — "
          "(4,n_r,n_th) value tables).",
          py::arg("obs"), py::arg("t_obs"), py::arg("src"), py::arg("t_src"),
          py::arg("ground_z"), py::arg("k"), py::arg("r1_max"),
          py::arg("r_break"), py::arg("th_split"), py::arg("r_near"),
          py::arg("reg_r0"), py::arg("reg_dr"), py::arg("reg_th0"),
          py::arg("reg_dth"), py::arg("reg_vals"),
          py::arg("cancel_flag") = 0);
    m.def("sommerfeld_remainder_bspline_Q", &sommerfeld_remainder_bspline_Q,
          "Fully-fused b-spline Galerkin Sommerfeld remainder over an obs/src "
          "rectangle: interpolate + project + moment-quadrature + basis-assemble "
          "into the (nI, nJ) Q block directly (no Jf / einsum intermediates). "
          "Dense block = symmetric case (obs==src, loc==supp_seg); the ACA "
          "sampler passes thin segment subsets with local support maps. Grid as "
          "in remainder_field_proj_batch.",
          py::arg("obs_nodes"), py::arg("obs_tang"), py::arg("W_obs"),
          py::arg("src_nodes"), py::arg("src_tang"), py::arg("W_src"),
          py::arg("loc_I"), py::arg("pI"), py::arg("loc_J"), py::arg("pJ"),
          py::arg("ground_z"), py::arg("k"),
          py::arg("r1_max"), py::arg("r_break"), py::arg("th_split"),
          py::arg("r_near"), py::arg("reg_r0"), py::arg("reg_dr"), py::arg("reg_th0"),
          py::arg("reg_dth"), py::arg("reg_vals"),
          py::arg("cancel_flag") = 0);
}
