#include "_accel_common.h"
#include <algorithm>
#include <string>
#include <vector>
#include "_stable_inline.h"

#include "_bspline_static_moments_inline.h"
#include "_bspline_static_far_inline.h"
// The extended-thin-wire static correction D_pq^EK (momwire#249's codegen,
// wired in by #270 unit 1). Pulls in the J header itself; the duplicate
// include above is harmless (`#pragma once`) and kept for legibility.
// Included HERE and not in _accel_common.h: this TU is the only consumer,
// and a codegen regeneration should rebuild one TU, not five.
#include "_bspline_ek_moments_inline.h"

// bspline section of the former _accelerators.cpp monolith (momwire#687).
// Code below is byte-identical to the monolith's lines 121-3348.


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
// RECTANGULAR SINCE momwire#968. The observer axis (`i`, rows) and the source
// axis (`j`, columns) were always independent in the loop nest below — `wu` was
// indexed `wu(p, i, q)` on one side and `wu(P, j, r)` on the other, and R as
// `Rr(i*n_qp+q, j*n_qp+r)`. Nothing here ever used `i == j`, a transpose, or a
// triangle. So taking a SEPARATE row-side and column-side `wu_pow` is the whole
// generalisation: the square call passes the same array twice and executes the
// identical instruction stream in the identical order, which is why the
// existing entry point stays bit-identical rather than merely close.
//
// It exists so a caller can build one observer WINDOW of a same-edge block
// instead of the whole (d+1, d+1, N, N) — the other half of momwire#966, whose
// first half chunked everything except this and `A_st`.
template <bool EK>
static py::array_t<std::complex<double>>
seg_seg_reg_moments_bspline_swept_impl(
    py::array_t<double, py::array::c_style | py::array::forcecast> R,
    py::array_t<double, py::array::c_style | py::array::forcecast> wu_row,
    py::array_t<double, py::array::c_style | py::array::forcecast> wu_col,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array,
    double a_ek
) {
    auto Rr = R.unchecked<2>();
    auto wu = wu_row.unchecked<3>();
    auto wuc = wu_col.unchecked<3>();
    auto ka = k_array.unchecked<1>();

    size_t n_d  = wu.shape(0);
    size_t N    = wu.shape(1);
    size_t n_qp = wu.shape(2);
    size_t n_col = wuc.shape(1);
    size_t n_k  = ka.shape(0);

    if ((size_t)wuc.shape(0) != n_d || (size_t)wuc.shape(2) != n_qp) {
        throw std::runtime_error(
            "wu_col must agree with wu_row in (n_d, n_qp)");
    }
    if (Rr.shape(0) != (py::ssize_t)(N * n_qp) ||
        Rr.shape(1) != (py::ssize_t)(n_col * n_qp)) {
        throw std::runtime_error(
            "R must be (n_row*n_qp, n_col*n_qp) consistent with wu_row "
            "(n_d, n_row, n_qp) and wu_col (n_d, n_col, n_qp)");
    }
    if (n_d > 8) {
        throw std::runtime_error("n_d too large (max_d must be <= 7)");
    }

    size_t n_pairs = n_qp * n_qp;
    if (n_qp > BSPLINE_SAME_EDGE_MAX_N_QP) {
        throw std::runtime_error("n_qp > "
                                 + std::to_string(BSPLINE_SAME_EDGE_MAX_N_QP)
                                 + " not supported on the same-edge reg kernel"
                                 " (L1-sized stack scratch, not yet tiled);"
                                 " the caller should have taken the numpy path");
    }

    const double inv_4pi = 1.0 / (4.0 * M_PI);
    // numpy divides by `4 * np.pi * R`, associated as `(4*np.pi) * R`.
    const double four_pi = 4.0 * M_PI;
    const double a2_ek = a_ek * a_ek;
    const double a4_ek = a2_ek * a2_ek;

    py::array_t<std::complex<double>> out({n_k, n_d, n_d, N, n_col});
    auto o = out.mutable_unchecked<5>();

    // Phase 0: release the GIL for the heavy compute region below.
    py::gil_scoped_release release;

    // (i, j) parallel; the per-(i, j) R block is hoisted out of the k loop and
    // the inner sincos runs as `omp simd` so GCC substitutes the libmvec
    // vectorized cos/sin. The
    // n_d^2 moment accumulation reuses each (q, r) Greg value across all
    // polynomial orders — the streaming property that avoids the numpy
    // einsum's (n_k, N*n_qp, N*n_qp) phase intermediate.
    MW_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < n_col; j++) {
            alignas(32) double R[64];
            alignas(32) double inv_R_4pi[64];
            alignas(32) double phases[64];
            // sin(phase/2), not cos(phase): this kernel forms the REMAINDER,
            // so the only thing it ever wanted `cos` for was `cos − 1`, and
            // that is `-2 sin²(phase/2)` (momwire#799). Same two transcendental
            // stages as before, one of them at half the argument.
            alignas(32) double half_phases[64];
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
            MW_OMP_SIMD()
            for (size_t qr = 0; qr < n_pairs; qr++) {
                inv_R_4pi[qr] = inv_4pi / R[qr];
            }
            if (EK) {
                // `_ek_factor`'s r2/r4 and T1/T2, plus the reciprocal of the
                // final 4πR divisor. All k-independent.
                MW_OMP_SIMD()
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
                MW_OMP_SIMD()
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    phases[qr] = -k * R[qr];
                }
                MW_OMP_SIMD()
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    half_phases[qr] = std::sin(0.5 * phases[qr]);
                }
                MW_OMP_SIMD()
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    sin_phases[qr] = std::sin(phases[qr]);
                }
                if (EK) {
                    MW_OMP_SIMD()
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
                        // phase = e^{-jkR} − 1 = −2 sin²(kR/2) + j sin(-kR),
                        // the reduced branch's bracket (momwire#799).
                        double hp = half_phases[qr];
                        double pr = -2.0 * hp * hp;
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
                MW_OMP_SIMD()
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    // exp(-j k R) - 1 = -2 sin²(kR/2) + j sin(-kR)
                    double hp = half_phases[qr];
                    Gre[qr] = (-2.0 * hp * hp) * inv_R_4pi[qr];
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
                                double w = wp * wuc(P, j, r);
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

// The square entry points: the same array on both axes. Deliberately a
// DELEGATION and not a copy of the loop — a second body is how the two spellings
// drift, and bit-identity here is the whole contract (momwire#762, #968).
static py::array_t<std::complex<double>>
seg_seg_reg_moments_bspline_swept(
    py::array_t<double, py::array::c_style | py::array::forcecast> R,
    py::array_t<double, py::array::c_style | py::array::forcecast> wu_pow,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array
) {
    return seg_seg_reg_moments_bspline_swept_impl<false>(R, wu_pow, wu_pow,
                                                        k_array, 0.0);
}

static py::array_t<std::complex<double>>
seg_seg_reg_moments_bspline_swept_ek(
    py::array_t<double, py::array::c_style | py::array::forcecast> R,
    py::array_t<double, py::array::c_style | py::array::forcecast> wu_pow,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array,
    double a_ek
) {
    return seg_seg_reg_moments_bspline_swept_impl<true>(R, wu_pow, wu_pow,
                                                       k_array, a_ek);
}

// The rectangular twins (momwire#968): one observer WINDOW of the same-edge
// block. `wu_row` carries the window's segments, `wu_col` the whole edge's, and
// R is (n_row*n_qp, n_col*n_qp). Output is (n_k, n_d, n_d, n_row, n_col).
static py::array_t<std::complex<double>>
seg_seg_reg_moments_bspline_swept_window(
    py::array_t<double, py::array::c_style | py::array::forcecast> R,
    py::array_t<double, py::array::c_style | py::array::forcecast> wu_row,
    py::array_t<double, py::array::c_style | py::array::forcecast> wu_col,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array
) {
    return seg_seg_reg_moments_bspline_swept_impl<false>(R, wu_row, wu_col,
                                                        k_array, 0.0);
}

static py::array_t<std::complex<double>>
seg_seg_reg_moments_bspline_swept_ek_window(
    py::array_t<double, py::array::c_style | py::array::forcecast> R,
    py::array_t<double, py::array::c_style | py::array::forcecast> wu_row,
    py::array_t<double, py::array::c_style | py::array::forcecast> wu_col,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array,
    double a_ek
) {
    return seg_seg_reg_moments_bspline_swept_impl<true>(R, wu_row, wu_col,
                                                       k_array, a_ek);
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
// COMPLEX_K continues this kernel to an in-medium (lossy) wavenumber
// k = k_re + j*k_im with Im k <= 0 (momwire#778). The continuation is a real
// SCALE FACTOR, not a substitution:
//
//     exp(-jkR) = exp(k_im * R) * (cos(k_re*R) - j*sin(k_re*R))
//                 ^^^^^^^^^^^^^ real, monotone in (0, 1] since k_im <= 0
//
// so the complex-k case is the real-k case with both trig arrays scaled by one
// extra real exp() per point, and every loop below stays in real SIMD lanes.
// That is the point: this kernel is fast BECAUSE its complex arithmetic is
// hand-split into real arrays libmvec can vectorize, and a
// std::complex<double> inner loop would give most of that back.
//
// THE TRAP (King & Smith 1981 eqs. 3.20b,c, recorded at momwire#553 U1): a
// lossy-medium kernel split into e^{-aR}cos(bR)/R and e^{-aR}sin(bR)/R does
// NOT generalize by substituting |k| for a real k. That audit found no such
// spelling in the PYTHON modules; THIS kernel is exactly that spelling, which
// was harmless only while it never served complex k. The factorization above
// is the correct continuation; |k| is the wrong one.
//
// `if (COMPLEX_K)` rather than `if constexpr`: the build is -std=gnu++11, and
// this is the idiom `if (EK)` already uses here. The branch folds at -O3.
// The distance-adaptive pair-order ladder (momwire#906). Tier 0 is the base
// Gauss-Legendre rule and always eligible; tier t > 0 serves a pair whose
// centre distance over the longer segment is >= ratio[t]. Thresholds ascend,
// so the highest eligible tier is the pair's order. Rules are concatenated:
// tier t's nodes are t[off[t] .. off[t+1]) and its weights likewise.
//
// Why this shape and not a per-pair n_qp array: the selector is O(1) per pair
// from data the kernel already holds, so the ladder is three small vectors
// and the (N_i, N_j) decision never leaves the parallel loop.
struct PairOrderLadder {
    std::vector<double> t;
    std::vector<double> w;
    std::vector<size_t> off;    // n_tiers + 1 prefix offsets into t / w
    std::vector<double> ratio;  // n_tiers thresholds; ratio[0] is unused
    size_t n_tiers() const { return ratio.size(); }
    size_t n_qp(size_t tier) const { return off[tier + 1] - off[tier]; }
    const double *t_at(size_t tier) const { return &t[off[tier]]; }
    const double *w_at(size_t tier) const { return &w[off[tier]]; }
};

// One tier: the pre-#906 (gl_t, gl_w) contract, unchanged for every caller
// that does not pass a ladder.
static PairOrderLadder ladder_from_rule(
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w
) {
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();
    if (glt.shape(0) != glw.shape(0)) {
        throw std::runtime_error("gl_t and gl_w must have matching length");
    }
    PairOrderLadder L;
    const size_t n = (size_t)glt.shape(0);
    L.t.resize(n);
    L.w.resize(n);
    for (size_t q = 0; q < n; q++) { L.t[q] = glt(q); L.w[q] = glw(q); }
    L.off = {0, n};
    L.ratio = {0.0};
    return L;
}

// A full ladder from the Python side's concatenated arrays. Validated here
// rather than trusted: a non-ascending threshold would make the selector's
// early `break` pick the wrong tier silently.
static PairOrderLadder ladder_from_arrays(
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tier_n_qp,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_ratio
) {
    auto tt = tier_t.unchecked<1>();
    auto tw = tier_w.unchecked<1>();
    auto tn = tier_n_qp.unchecked<1>();
    auto tr = tier_ratio.unchecked<1>();
    const size_t n_tiers = (size_t)tn.shape(0);
    if (n_tiers < 1 || (size_t)tr.shape(0) != n_tiers) {
        throw std::runtime_error(
            "pair-order ladder: tier_n_qp and tier_ratio must have the same "
            "length, at least 1");
    }
    if (tt.shape(0) != tw.shape(0)) {
        throw std::runtime_error("pair-order ladder: tier_t and tier_w must match");
    }
    PairOrderLadder L;
    L.off.resize(n_tiers + 1);
    L.off[0] = 0;
    for (size_t tier = 0; tier < n_tiers; tier++) {
        if (tn(tier) < 1) {
            throw std::runtime_error("pair-order ladder: every tier needs n_qp >= 1");
        }
        L.off[tier + 1] = L.off[tier] + (size_t)tn(tier);
        if (tier >= 1 && !(tr(tier) > tr(tier - 1))) {
            throw std::runtime_error(
                "pair-order ladder: ratio thresholds must strictly ascend");
        }
    }
    if (L.off[n_tiers] != (size_t)tt.shape(0)) {
        throw std::runtime_error(
            "pair-order ladder: tier_n_qp must sum to the length of tier_t");
    }
    L.t.resize(L.off[n_tiers]);
    L.w.resize(L.off[n_tiers]);
    for (size_t q = 0; q < L.off[n_tiers]; q++) { L.t[q] = tt(q); L.w[q] = tw(q); }
    L.ratio.resize(n_tiers);
    for (size_t tier = 0; tier < n_tiers; tier++) L.ratio[tier] = tr(tier);
    return L;
}

// momwire#1290: the off-edge kernels' lane paths, written against the lane
// layer (momwire#1372, `_lanes.h`) at its width W. Only where the layer has a
// vector backend (the x86 AVX2 build, `_avx2`); the baseline (`_sse2`), arm64
// and MSVC builds keep the per-pair walk alone, so their bits cannot move.
#include "_lanes.h"
#define MW_OFFEDGE_LANES_1290 MW_LANES_SIMD

#if MW_OFFEDGE_LANES_1290
// The pieces the three off-edge lane kernels (single k, swept, H-matrix
// table) share. Lane l of every [t][lane] buffer is one pair's point t, and
// each helper is the walk's arithmetic for that pair, operation for operation.
namespace offedge_lanes {
using namespace mw_lanes;

// R at the W pairs (i, j0 .. j0+W-1): the walk's `dx*dx + dy*dy + dz*dz +
// a_squared`, left to right, then a correctly rounded sqrt. `pi` is the
// observer's n_qp points (xyz-interleaved, broadcast to every lane); `pT` the
// sources' lane-major (c, r, j) table.
MW_LANES_INLINE void r_bcast(double *R, const double *pi, const double *pT,
                             size_t n_qp, size_t N_j, size_t j0, vd v_a2) {
    for (size_t q = 0; q < n_qp; q++) {
        const vd px = set1(pi[q*3 + 0]);
        const vd py = set1(pi[q*3 + 1]);
        const vd pz = set1(pi[q*3 + 2]);
        for (size_t r = 0; r < n_qp; r++) {
            const vd dx = sub(px, loadu(pT + r * N_j + j0));
            const vd dy = sub(py, loadu(pT + (n_qp + r) * N_j + j0));
            const vd dz = sub(pz, loadu(pT + (2 * n_qp + r) * N_j + j0));
            vd s = add(mul(dx, dx), mul(dy, dy));
            s = add(s, mul(dz, dz));
            s = add(s, v_a2);
            store(R + (q * n_qp + r) * W, sqrt(s));
        }
    }
}

// The moment weights wv[t][pP][lane] for one observer length Li and W source
// lengths Lj, per lane in the walk's order: wij = wi*wj, then
// (wij*ui^p)*uj^P. The source-side factors, the only ones that differ across
// lanes, are vectors; the observer's are scalars broadcast.
template<int NM>
MW_LANES_INLINE void wv_bcast(double *wv, const double *gt, const double *gw,
                              size_t n_qp, double Li, const double *Lj) {
    constexpr int NMM = NM * NM;
    vd wjv[8], ujp[8][NM];
    const vd vLj = loadu(Lj);
    for (size_t r = 0; r < n_qp; r++) {
        wjv[r] = mul(set1(gw[r]), vLj);
        const vd uj = mul(set1(gt[r]), vLj);
        ujp[r][0] = set1(1.0);
        for (int e = 1; e < NM; e++) ujp[r][e] = mul(ujp[r][e-1], uj);
    }
    for (size_t q = 0; q < n_qp; q++) {
        const double wi = gw[q] * Li;
        const double ui = gt[q] * Li;
        double ui_pow[NM];
        ui_pow[0] = 1.0;
        for (int e = 1; e < NM; e++) ui_pow[e] = ui_pow[e-1] * ui;
        for (size_t r = 0; r < n_qp; r++) {
            const size_t t = q * n_qp + r;
            const vd wij = mul(set1(wi), wjv[r]);
            for (int pp = 0; pp < NM; pp++) {
                const vd x = mul(wij, set1(ui_pow[pp]));
                for (int PP = 0; PP < NM; PP++) {
                    store(wv + (t * NMM + pp * NM + PP) * W, mul(x, ujp[r][PP]));
                }
            }
        }
    }
}

// EK (momwire#1362): the walk's coaxial factor on G at one point t of every
// lane, from that point's T1, T2 and kR. Eligibility is per PAIR, so the
// lanes of a group can disagree (a group straddling a wire's end): every lane
// computes G*fac and an ineligible lane keeps its G by a blend, which moves no
// bits. The walk's `t2 * c1r` and `3.0 * c1r` multiply by c1r = 1.0, which is
// exact, so they are spelled t2 and 3.0.
MW_LANES_INLINE void ek_factor(double *gre_p, double *gim_p, vd t1, vd t2,
                               vd kr, vm emask) {
    const vd three = set1(3.0);
    const vd kr2 = mul(kr, kr);
    const vd c2r = sub(three, kr2);
    const vd c2i = mul(three, kr);
    vd facr = mul(t1, c2r);
    vd faci = mul(t1, c2i);
    facr = sub(facr, t2);
    faci = sub(faci, mul(t2, kr));
    facr = add(facr, set1(1.0));
    const vd gre = load(gre_p);
    const vd gim = load(gim_p);
    const vd nre = sub(mul(gre, facr), mul(gim, faci));
    const vd nim = add(mul(gre, faci), mul(gim, facr));
    store(gre_p, blend(gre, nre, emask));
    store(gim_p, blend(gim, nim, emask));
}

// Stage 2, one part (re or im) of G: per moment and lane, the walk's chain --
// start at 0.0, add w*G in ascending t, one rounded multiply and one rounded
// add per term. NMM vector accumulators; the parts never mixed in the walk.
template<int NMM>
MW_LANES_INLINE void stage2(const double *G, const double *wv, size_t m,
                            vd *acc) {
    for (int pP = 0; pP < NMM; pP++) acc[pP] = zero();
    for (size_t t = 0; t < m; t++) {
        const vd gv = load(G + t * W);
        const double *w_t = wv + t * NMM * W;
        for (int pP = 0; pP < NMM; pP++) {
            acc[pP] = add(acc[pP], mul(load(w_t + pP * W), gv));
        }
    }
}
}  // namespace offedge_lanes
#endif

// EK (momwire#1362): the extended-kernel twin is this kernel with NEC Eq 89's
// coaxial factor applied to G on the pairs whose group labels match
// (`grp_i[i] == grp_j[j] >= 0`), between stage 1 and stage 2; see
// `seg_seg_full_moments_bspline_kernel_ek` below for the contract. With EK
// false every EK statement is a constant-false branch, so the reduced
// instantiations compile to what they did.
template<int D, bool COMPLEX_K, bool EK = false>
static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_kernel_impl(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    double k,
    double k_im,
    const PairOrderLadder& ladder,
    // momwire#1290: true walks every pair alone even where the lane path is
    // compiled -- the reference that path is gated against to the bit.
    bool reference = false,
    // EK only: the per-segment group labels (N_i,) / (N_j,) and the plain
    // (unsquared) EK radius.
    const int64_t *grp_i = nullptr,
    const int64_t *grp_j = nullptr,
    double a_ek = 0.0,
    // momwire#1368: the SOURCE tube's radius b of NEC Eq 89's two-radius
    // factor (T1 = b^2 a^2 / 4R^4, T2 = b^2 / 2R^2, `a_ek` the observer's);
    // negative = b = a_ek, the equal-radius factor to the bit.
    double a_ek_src = -1.0
) {
    static_assert(!(EK && COMPLEX_K),
                  "the extended kernel is served for a real k only");
    static constexpr int NM = D + 1;          // moments per axis
    static constexpr int NMM = NM * NM;       // total moments
    const double b_ek = a_ek_src < 0.0 ? a_ek : a_ek_src;
    const double a2_ek = b_ek * b_ek;                  // b^2
    const double a4_ek = a2_ek * (a_ek * a_ek);        // b^2 a^2
    // momwire#1368: an observer inside the source tube's radius (only at a
    // radius-step junction, where the thin wire's quadrature reaches within
    // b of the fat wire) is outside the expansion Eq 89 is: its b^2/R^2
    // exceeds 1 and the factor grows without bound (bs2 diverged under
    // refinement at 5:1 and 20:1). The factor is evaluated no closer than
    // the tube's surface, R_f = max(R, b). It bites only when b > a_ek
    // (R >= a_ek always), so an equal-radius or thinner-source call never
    // enters it and is the same arithmetic as before.
    const bool ek_floor = EK && b_ek > a_ek;

    auto sli = seg_l_i.unchecked<2>();
    auto sri = seg_r_i.unchecked<2>();
    auto slj = seg_l_j.unchecked<2>();
    auto srj = seg_r_j.unchecked<2>();

    if (sli.shape(1) != 3 || sri.shape(1) != 3 ||
        slj.shape(1) != 3 || srj.shape(1) != 3) {
        throw std::runtime_error("segment endpoint arrays must have shape (N, 3)");
    }
    if (sli.shape(0) != sri.shape(0) || slj.shape(0) != srj.shape(0)) {
        throw std::runtime_error("seg_l and seg_r must have matching N");
    }

    size_t N_i = sli.shape(0);
    size_t N_j = slj.shape(0);
    const size_t n_tiers = ladder.n_tiers();

    py::array_t<std::complex<double>> J({(size_t)NM, (size_t)NM, N_i, N_j});
    auto j_view = J.mutable_unchecked<4>();

    // Phase 0: release the GIL for the heavy compute region below.
    py::gil_scoped_release release;

    const double inv_4pi = 1.0 / (4.0 * M_PI);

    // Per-segment lengths and, PER TIER, quadrature-point positions --
    // k-independent, computed once outside the parallel region. With one
    // tier this is the pre-#906 precompute split into two loops: the same
    // expressions produce the same doubles.
    std::vector<double> len_i(N_i);
    std::vector<double> len_j(N_j);
    for (size_t i = 0; i < N_i; i++) {
        double dx = sri(i,0) - sli(i,0);
        double dy = sri(i,1) - sli(i,1);
        double dz = sri(i,2) - sli(i,2);
        len_i[i] = std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    for (size_t j = 0; j < N_j; j++) {
        double dx = srj(j,0) - slj(j,0);
        double dy = srj(j,1) - slj(j,1);
        double dz = srj(j,2) - slj(j,2);
        len_j[j] = std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    std::vector<std::vector<double>> pos_i(n_tiers);
    std::vector<std::vector<double>> pos_j(n_tiers);
    for (size_t tier = 0; tier < n_tiers; tier++) {
        const size_t n_qp = ladder.n_qp(tier);
        const double *gt = ladder.t_at(tier);
        pos_i[tier].resize(N_i * n_qp * 3);
        pos_j[tier].resize(N_j * n_qp * 3);
        std::vector<double>& pi_t = pos_i[tier];
        std::vector<double>& pj_t = pos_j[tier];
        for (size_t i = 0; i < N_i; i++) {
            for (size_t q = 0; q < n_qp; q++) {
                double t = gt[q];
                pi_t[(i*n_qp + q)*3 + 0] = (1.0 - t) * sli(i,0) + t * sri(i,0);
                pi_t[(i*n_qp + q)*3 + 1] = (1.0 - t) * sli(i,1) + t * sri(i,1);
                pi_t[(i*n_qp + q)*3 + 2] = (1.0 - t) * sli(i,2) + t * sri(i,2);
            }
        }
        for (size_t j = 0; j < N_j; j++) {
            for (size_t r = 0; r < n_qp; r++) {
                double t = gt[r];
                // Source points coordinate-major per segment, (j, c, r), so
                // the R loop below reads each coordinate as a unit-stride run.
                pj_t[(j*3 + 0)*n_qp + r] = (1.0 - t) * slj(j,0) + t * srj(j,0);
                pj_t[(j*3 + 1)*n_qp + r] = (1.0 - t) * slj(j,1) + t * srj(j,1);
                pj_t[(j*3 + 2)*n_qp + r] = (1.0 - t) * slj(j,2) + t * srj(j,2);
            }
        }
    }
    // Segment centres, only read when there is a tier to choose (#906): the
    // selector is centre distance over the longer segment, the O(1)-per-pair
    // quantity the study binned by.
    std::vector<double> c_i, c_j;
    if (n_tiers > 1) {
        c_i.resize(N_i * 3);
        c_j.resize(N_j * 3);
        for (size_t i = 0; i < N_i; i++) {
            for (int d = 0; d < 3; d++) c_i[i*3 + d] = 0.5 * (sli(i,d) + sri(i,d));
        }
        for (size_t j = 0; j < N_j; j++) {
            for (int d = 0; d < 3; d++) c_j[j*3 + d] = 0.5 * (slj(j,d) + srj(j,d));
        }
    }

    // The pair's tier: the highest whose ratio threshold the pair meets
    // (thresholds ascend, validated at construction). One function for the
    // per-pair walk and the lane groups below, so the two cannot disagree
    // about a pair's rule.
    auto pair_tier = [&](size_t i, size_t j) -> size_t {
        size_t tier = 0;
        if (n_tiers > 1) {
            const double dx = c_i[i*3 + 0] - c_j[j*3 + 0];
            const double dy = c_i[i*3 + 1] - c_j[j*3 + 1];
            const double dz = c_i[i*3 + 2] - c_j[j*3 + 2];
            const double ratio = std::sqrt(dx*dx + dy*dy + dz*dz)
                                 / std::max(len_i[i], len_j[j]);
            for (size_t t = 1; t < n_tiers; t++) {
                if (ratio >= ladder.ratio[t]) tier = t; else break;
            }
        }
        return tier;
    };

    // TILED OVER qr (momwire#762). The scratch stays [64] — an L1 blocking
    // width, not a limit — and the qr range is walked in chunks of at most
    // that, accumulating the moment sums across chunks before a single
    // j_view write. Nothing carries state across quadrature pairs, so this is
    // exact; and at n_qp <= 8 there is exactly ONE chunk spanning the whole
    // range, so both the arithmetic and its order are unchanged and the
    // output is bit-identical to the untiled kernel.
    //
    // wuwu (below) is a function of the tier and the two segment LENGTHS
    // only, and a meshed wire repeats its segment length pair after pair --
    // on paper; to the bit a uniform mesh's lengths often differ (the lane
    // path below refills its table on 93 % of its groups on the free-space
    // array), so the hold pays only where they truly repeat. Each thread
    // keeps the last single-chunk table with its key and refills it only
    // when (tier, Li, Lj) changes. A hit reuses doubles produced by
    // the very expressions a refill would evaluate on bit-equal inputs, so
    // the moments do not move. The caller owns the table and its key: a
    // thread's own across its pairs, or a fresh one per pair (MSVC).
    auto pair_walk = [&](size_t i, size_t j, double *wuwu, bool &w_held,
                         size_t &w_tier, double &w_Li, double &w_Lj) {
        alignas(32) double R[BSPLINE_QR_TILE];
        alignas(32) double inv_R_4pi[BSPLINE_QR_TILE];
        alignas(32) double phases[BSPLINE_QR_TILE];
        alignas(32) double cos_phases[BSPLINE_QR_TILE];
        alignas(32) double sin_phases[BSPLINE_QR_TILE];
        alignas(32) double G_re[BSPLINE_QR_TILE], G_im[BSPLINE_QR_TILE];
        alignas(32) double decay[BSPLINE_QR_TILE];

        double acc_re[NMM], acc_im[NMM];
        for (int pP = 0; pP < NMM; pP++) { acc_re[pP] = 0.0; acc_im[pP] = 0.0; }

        const size_t tier = pair_tier(i, j);
        const size_t n_qp = ladder.n_qp(tier);
        const double *gt = ladder.t_at(tier);
        const double *gw = ladder.w_at(tier);
        const double *pi = &pos_i[tier][i * n_qp * 3];
        const double *pj = &pos_j[tier][j * n_qp * 3];
        const double Li = len_i[i];
        const double Lj = len_j[j];
        const size_t n_pairs = n_qp * n_qp;

        for (size_t base = 0; base < n_pairs; base += BSPLINE_QR_TILE) {
            const size_t m = (n_pairs - base < BSPLINE_QR_TILE)
                                 ? (n_pairs - base) : BSPLINE_QR_TILE;

            // One division per chunk, then the chunk's contiguous qr
            // range (qr = q*n_qp + r) as runs of r under one q: each run
            // is a unit-stride loop the compiler vectorizes, and sqrt is
            // correctly rounded in a vector lane as in a scalar one, so
            // R is the per-point walk's to the bit.
            size_t q = base / n_qp;
            size_t r = base % n_qp;
            for (size_t t = 0; t < m; r = 0, ++q) {
                const size_t run = std::min(n_qp - r, m - t);
                const double px = pi[q*3 + 0], py = pi[q*3 + 1], pz = pi[q*3 + 2];
                const double *xj = pj + r;
                const double *yj = pj + n_qp + r;
                const double *zj = pj + 2 * n_qp + r;
                double *Rt = R + t;
                for (size_t u = 0; u < run; u++) {
                    const double dx = px - xj[u];
                    const double dy = py - yj[u];
                    const double dz = pz - zj[u];
                    Rt[u] = std::sqrt(dx*dx + dy*dy + dz*dz + a_squared);
                }
                t += run;
            }

            // A multi-chunk pair rewrites wuwu per chunk, so only a
            // single-chunk table is ever held.
            const bool one_chunk = n_pairs <= BSPLINE_QR_TILE;
            if (!(one_chunk && w_held && tier == w_tier && Li == w_Li &&
                  Lj == w_Lj)) {
                q = base / n_qp;
                r = base % n_qp;
                for (size_t t = 0; t < m; t++) {
                    const double wi = gw[q] * Li;
                    const double ui = gt[q] * Li;
                    const double wj = gw[r] * Lj;
                    const double uj = gt[r] * Lj;
                    double ui_pow[NM], uj_pow[NM];
                    ui_pow[0] = 1.0;
                    uj_pow[0] = 1.0;
                    for (int e = 1; e < NM; e++) {
                        ui_pow[e] = ui_pow[e-1] * ui;
                        uj_pow[e] = uj_pow[e-1] * uj;
                    }
                    const double wij = wi * wj;
                    for (int pp = 0; pp < NM; pp++) {
                        for (int PP = 0; PP < NM; PP++) {
                            wuwu[t * NMM + pp * NM + PP] = wij * ui_pow[pp] * uj_pow[PP];
                        }
                    }

                    if (++r == n_qp) { r = 0; ++q; }
                }
                w_held = one_chunk;
                w_tier = tier;
                w_Li = Li;
                w_Lj = Lj;
            }

            // Stage 1: phases = -k_re * R, then sincos via libmvec.
            MW_OMP_SIMD()
            for (size_t t = 0; t < m; t++) {
                phases[t] = -k * R[t];
            }
            MW_OMP_SIMD()
            for (size_t t = 0; t < m; t++) {
                cos_phases[t] = std::cos(phases[t]);
            }
            MW_OMP_SIMD()
            for (size_t t = 0; t < m; t++) {
                sin_phases[t] = std::sin(phases[t]);
            }
            if (COMPLEX_K) {
                // exp(k_im * R) with k_im <= 0 — decaying, so no overflow,
                // and underflow to +0 at large R is the physical answer.
                MW_OMP_SIMD()
                for (size_t t = 0; t < m; t++) {
                    decay[t] = std::exp(k_im * R[t]);
                }
                MW_OMP_SIMD()
                for (size_t t = 0; t < m; t++) {
                    inv_R_4pi[t] = inv_4pi / R[t];
                    const double sc = decay[t] * inv_R_4pi[t];
                    G_re[t] = cos_phases[t] * sc;
                    G_im[t] = sin_phases[t] * sc;
                }
            } else {
                // Textually unchanged from the pre-#778 kernel so the
                // real-k path stays bit-identical — proven by rebuilding
                // and array_equal, not by reading (the #762 protocol).
                MW_OMP_SIMD()
                for (size_t t = 0; t < m; t++) {
                    inv_R_4pi[t] = inv_4pi / R[t];
                    G_re[t] = cos_phases[t] * inv_R_4pi[t];
                    G_im[t] = sin_phases[t] * inv_R_4pi[t];
                }
            }

            // EK: eligibility is a property of the (i, j) SEGMENT pair, not
            // of the quadrature sub-pair (numpy's `mask[:, None, :, None]`
            // broadcast), so one branch serves every point of the chunk. The
            // loop is the pre-#1362 EK kernel's, verbatim: `_ek_factor`'s
            // spelling term by term, T1, T2, C1, C2, fac = T1*C2 - T2*C1 + 1,
            // then G *= fac (complex).
            if (EK && grp_i[i] == grp_j[j] && grp_i[i] >= 0) {
                MW_OMP_SIMD()
                for (size_t t = 0; t < m; t++) {
                    double Rq = R[t];
                    if (ek_floor) Rq = std::max(Rq, b_ek);
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
                    double gre = G_re[t];
                    double gim = G_im[t];
                    G_re[t] = gre * facr - gim * faci;
                    G_im[t] = gre * faci + gim * facr;
                }
            }

            // Stage 2: the NMM moment sums, carried in acc_* across
            // chunks. Each accumulator adds its terms in ascending t, one
            // rounded multiply and one rounded add per term -- the order
            // the per-pP loop this replaced used -- so the moments are
            // bit-identical to it on every target. Only the nesting moved:
            // with t outermost the NMM (re, im) chains are independent
            // within a step, where per pP each was one serial chain of m
            // dependent adds (latency-bound, ~4 cycles a term). Nothing
            // here licenses reassociation: no `omp simd reduction`
            // (momwire#781), and any vectorizing the compiler does is
            // ACROSS accumulators, which is exact.
            for (size_t t = 0; t < m; t++) {
                const double gr = G_re[t], gi = G_im[t];
                const double *w_t = &wuwu[t * NMM];
                for (int pP = 0; pP < NMM; pP++) {
                    acc_re[pP] += w_t[pP] * gr;
                    acc_im[pP] += w_t[pP] * gi;
                }
            }
        }

        for (int pP = 0; pP < NMM; pP++) {
            j_view(pP / NM, pP % NM, i, j) =
                std::complex<double>(acc_re[pP], acc_im[pP]);
        }
    };

#if MW_OFFEDGE_LANES_1290
    if (!reference) {
        // The lane kernel (momwire#1290): W columns (i, j0..j0+W-1) at a
        // time, lane l holding the pair (i, j0+l), wherever the W share a
        // tier whose rule is one qr chunk of an EVEN order. Anything else --
        // a group with mixed tiers, an odd order, a rule over
        // BSPLINE_QR_TILE points, the N_j % W tail -- is walked pair by pair
        // by `pair_walk` above, unchanged.
        //
        // Each lane's arithmetic is the walk's for its pair, operation for
        // operation: the same R expression, the same G, the same wuwu
        // products, and the same ascending-t mul-then-add chain per moment
        // (this TU builds with -ffp-contract=off, and nothing here writes an
        // fma). The one thing that is not textual is cos/sin/exp: the walk's
        // simd loops call libmvec's 4-wide entry for whole vectors and the
        // SCALAR libm function for a remainder, and the two differ in the
        // last bit. So the order must be even: m = n_qp^2 is then a multiple
        // of 4, the walk has no remainder, and both routes (the lanes' loops
        // are m W long) put every point through the vector entry. An odd order leaves m % 4 == 1, whose
        // remainder point the walk sends to scalar libm, so it stays on the
        // walk. The moments are the walk's to the bit, and `reference=True`
        // keeps the walk reachable as the gate
        // (`tests/test_offedge_lanes_1290.py`).
        //
        // Measured (Haswell, the free-space 8-dipole array at N = 2816, its
        // 13 calls; 98 % of the pairs sit 16+ segment lengths apart and take
        // the order-4 tier, 16 points): the walk spent ~330 ns a pair on one
        // thread around those 16 points -- short simd loops and their
        // remainders, a 4-deep run loop for R, nine accumulators in three
        // part-empty vectors, and the wuwu refill, which a uniform mesh pays
        // on most pairs because its segment lengths differ in the last bits.
        // In lanes every loop is 4 m long, every accumulator a full vector,
        // and the refill vectorised across the four lengths: the 13 calls
        // went 2.87 -> 1.72 s on one thread and 0.75 -> 0.47 s on four, with
        // 99.6 % of the pairs in lanes. Left as they were, because they are
        // what bits are made of: cos and sin (a third of the lane time;
        // libmvec's fused sincos is slower here AND differs from them in the
        // last bit), the divide and sqrt, and stage 2's separate mul and add.
        using namespace offedge_lanes;
        constexpr size_t LN = W;
        const size_t n_grp = (N_j + LN - 1) / LN;
        // Source points lane-major per tier, (c, r, j): a group reads one
        // coordinate of one node for its W columns as a single load.
        std::vector<std::vector<double>> pjT(n_tiers);
        for (size_t tier = 0; tier < n_tiers; tier++) {
            const size_t n_qp = ladder.n_qp(tier);
            const std::vector<double>& pj_t = pos_j[tier];
            std::vector<double>& out_t = pjT[tier];
            out_t.resize(3 * n_qp * N_j);
            for (size_t j = 0; j < N_j; j++) {
                for (size_t c = 0; c < 3; c++) {
                    for (size_t r = 0; r < n_qp; r++) {
                        out_t[(c * n_qp + r) * N_j + j] = pj_t[(j*3 + c)*n_qp + r];
                    }
                }
            }
        }
        const vd v_a2 = set1(a_squared);

        #pragma omp parallel
        {
        alignas(32) double wuwu[NMM * BSPLINE_QR_TILE];
        bool w_held = false;
        size_t w_tier = 0;
        double w_Li = 0.0, w_Lj = 0.0;
        // The lane twin of `wuwu`, [t][pP][lane], held by the same rule with
        // one Lj per lane.
        alignas(ALIGN) double wv[BSPLINE_QR_TILE * NMM * LN];
        bool v_held = false;
        size_t v_tier = 0;
        double v_Li = 0.0;
        double v_Lj[LN];
        for (size_t l = 0; l < LN; l++) v_Lj[l] = 0.0;
        // [t][lane]: point t of the walk's chunk for column j0 + lane.
        alignas(ALIGN) double R[BSPLINE_QR_TILE * LN];
        alignas(ALIGN) double inv_R_4pi[BSPLINE_QR_TILE * LN];
        alignas(ALIGN) double phases[BSPLINE_QR_TILE * LN];
        alignas(ALIGN) double cos_phases[BSPLINE_QR_TILE * LN];
        alignas(ALIGN) double sin_phases[BSPLINE_QR_TILE * LN];
        alignas(ALIGN) double G_re[BSPLINE_QR_TILE * LN];
        alignas(ALIGN) double G_im[BSPLINE_QR_TILE * LN];
        alignas(ALIGN) double decay[BSPLINE_QR_TILE * LN];
        MW_OMP_FOR_COLLAPSE2
        for (size_t i = 0; i < N_i; i++) {
            for (size_t g = 0; g < n_grp; g++) {
                const size_t j0 = g * LN;
                bool lanes = j0 + LN <= N_j;
                size_t tier = 0;
                if (lanes) {
                    tier = pair_tier(i, j0);
                    for (size_t l = 1; l < LN; l++) {
                        lanes = lanes && pair_tier(i, j0 + l) == tier;
                    }
                    const size_t nq = ladder.n_qp(tier);
                    lanes = lanes && nq % 2 == 0 && nq * nq <= BSPLINE_QR_TILE;
                }
                if (!lanes) {
                    const size_t j1 = std::min(j0 + LN, N_j);
                    for (size_t j = j0; j < j1; j++) {
                        pair_walk(i, j, wuwu, w_held, w_tier, w_Li, w_Lj);
                    }
                    continue;
                }

                const size_t n_qp = ladder.n_qp(tier);
                const size_t m = n_qp * n_qp;
                const size_t m4 = m * LN;
                const double *gt = ladder.t_at(tier);
                const double *gw = ladder.w_at(tier);
                const double *pi = &pos_i[tier][i * n_qp * 3];
                const double Li = len_i[i];
                double Lj[LN];
                for (size_t l = 0; l < LN; l++) Lj[l] = len_j[j0 + l];

                r_bcast(R, pi, pjT[tier].data(), n_qp, N_j, j0, v_a2);

                // A mesh's segment lengths agree to the bit far less often
                // than they agree on paper, so this refill runs on most
                // groups and is worth the vectors.
                bool same = v_held && tier == v_tier && Li == v_Li;
                for (size_t l = 0; l < LN; l++) same = same && Lj[l] == v_Lj[l];
                if (!same) {
                    wv_bcast<NM>(wv, gt, gw, n_qp, Li, Lj);
                    v_held = true;
                    v_tier = tier;
                    v_Li = Li;
                    for (size_t l = 0; l < LN; l++) v_Lj[l] = Lj[l];
                }

                // Stage 1, the walk's loops over all the lanes' points.
                MW_OMP_SIMD()
                for (size_t t = 0; t < m4; t++) {
                    phases[t] = -k * R[t];
                }
                MW_OMP_SIMD()
                for (size_t t = 0; t < m4; t++) {
                    cos_phases[t] = std::cos(phases[t]);
                }
                MW_OMP_SIMD()
                for (size_t t = 0; t < m4; t++) {
                    sin_phases[t] = std::sin(phases[t]);
                }
                if (COMPLEX_K) {
                    MW_OMP_SIMD()
                    for (size_t t = 0; t < m4; t++) {
                        decay[t] = std::exp(k_im * R[t]);
                    }
                    MW_OMP_SIMD()
                    for (size_t t = 0; t < m4; t++) {
                        inv_R_4pi[t] = inv_4pi / R[t];
                        const double sc = decay[t] * inv_R_4pi[t];
                        G_re[t] = cos_phases[t] * sc;
                        G_im[t] = sin_phases[t] * sc;
                    }
                } else {
                    MW_OMP_SIMD()
                    for (size_t t = 0; t < m4; t++) {
                        inv_R_4pi[t] = inv_4pi / R[t];
                        G_re[t] = cos_phases[t] * inv_R_4pi[t];
                        G_im[t] = sin_phases[t] * inv_R_4pi[t];
                    }
                }

                // EK (momwire#1362): the walk's coaxial factor, per lane
                // (`ek_factor`), from the floored R where the walk floors it.
                if (EK && grp_i[i] >= 0) {
                    const int64_t gi = grp_i[i];
                    bool e[LN];
                    bool any_e = false;
                    for (size_t l = 0; l < LN; l++) {
                        e[l] = grp_j[j0 + l] == gi;
                        any_e = any_e || e[l];
                    }
                    if (any_e) {
                        const vm emask = mask_from(e);
                        const vd v_k = set1(k);
                        const vd v_t1n = set1(0.25 * a4_ek);
                        const vd v_t2n = set1(0.5 * a2_ek);
                        const vd v_bek = set1(b_ek);
                        for (size_t t = 0; t < m; t++) {
                            vd Rq = load(R + t * LN);
                            if (ek_floor) Rq = max(Rq, v_bek);
                            const vd r2 = mul(Rq, Rq);
                            const vd r4 = mul(r2, r2);
                            ek_factor(G_re + t * LN, G_im + t * LN, div(v_t1n, r4),
                                      div(v_t2n, r2), mul(v_k, Rq), emask);
                        }
                    }
                }

                // Stage 2, the real and imaginary sums in separate passes so
                // each holds NMM accumulators, not 2 NMM, in registers.
                // `phases` holds the real sums once its points are spent; the
                // imaginary pass writes each moment's W complex outputs.
                vd acc[NMM];
                stage2<NMM>(G_re, wv, m, acc);
                for (int pP = 0; pP < NMM; pP++) store(phases + pP * LN, acc[pP]);
                stage2<NMM>(G_im, wv, m, acc);
                for (int pP = 0; pP < NMM; pP++) {
                    store_interleaved(
                        reinterpret_cast<double *>(&j_view(pP / NM, pP % NM, i, j0)),
                        load(phases + pP * LN), acc[pP]);
                }
            }
        }
        }
    } else
#else
    (void)reference;
#endif
    {
    // MSVC keeps main's parallel structure: one `omp parallel for` spelled as
    // main spells it, the table on each iteration's stack and refilled per
    // pair. Holding the table needs a region with a work-sharing loop inside
    // it, and on the Windows wheel (momwire#1302) that shape coincided with
    // the bspline tests running slower than main; the hold is a few percent,
    // not worth a structure that build has not been measured on.
#if !defined(_MSC_VER)
    #pragma omp parallel
    {
    // wuwu[t, pP]: precomputed wi[q]*ui[q]^p * wj[r]*uj[r]^P for the
    // chunk's pairs, flattened with pP = p*NM + P innermost (stage 2 walks t
    // outermost). For D=2: NMM*64 = 576 doubles = 4.5KB, fits comfortably
    // in L1.
    alignas(32) double wuwu[NMM * BSPLINE_QR_TILE];
    bool w_held = false;
    size_t w_tier = 0;
    double w_Li = 0.0, w_Lj = 0.0;
    MW_OMP_FOR_COLLAPSE2
#else
    MW_OMP_PARALLEL_FOR_COLLAPSE2
#endif
    for (size_t i = 0; i < N_i; i++) {
        for (size_t j = 0; j < N_j; j++) {
#if defined(_MSC_VER)
            alignas(32) double wuwu[NMM * BSPLINE_QR_TILE];
            bool w_held = false;  // refilled every pair
            size_t w_tier = 0;
            double w_Li = 0.0, w_Lj = 0.0;
#endif
            pair_walk(i, j, wuwu, w_held, w_tier, w_Li, w_Lj);
        }
    }
#if !defined(_MSC_VER)
    }
#endif
    }

    return J;
}

// The single-rule entry the pre-#906 callers use: one tier, the same loop.
template<int D, bool COMPLEX_K>
static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_kernel(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    double k,
    double k_im,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    bool reference = false
) {
    return seg_seg_full_moments_bspline_kernel_impl<D, COMPLEX_K>(
        seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k, k_im,
        ladder_from_rule(gl_t, gl_w), reference);
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

    // TILED OVER qr (momwire#762), with the chunk OUTSIDE the k loop so the
    // k-independent tables (R, 1/(4 pi R), wuwu) are still built once per
    // chunk and reused across the whole sweep — tiling must not cost this
    // kernel the very reuse it exists for. The first chunk assigns and later
    // chunks accumulate, so at n_qp <= 8 (one chunk) the write is the same
    // plain assignment it always was and the output is bit-identical.
    MW_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t i = 0; i < N_i; i++) {
        for (size_t j = 0; j < N_j; j++) {
            alignas(32) double R[BSPLINE_QR_TILE];
            alignas(32) double inv_R_4pi[BSPLINE_QR_TILE];
            alignas(32) double phases[BSPLINE_QR_TILE];
            alignas(32) double cos_phases[BSPLINE_QR_TILE];
            alignas(32) double sin_phases[BSPLINE_QR_TILE];
            alignas(32) double G_re[BSPLINE_QR_TILE], G_im[BSPLINE_QR_TILE];
            alignas(32) double wuwu[NMM * BSPLINE_QR_TILE];

            const size_t n_pairs = n_qp * n_qp;
            const double *pi = &pos_i[i * n_qp * 3];
            const double *pj = &pos_j[j * n_qp * 3];
            const double Li = len_i[i];
            const double Lj = len_j[j];

            bool first_chunk = true;
            for (size_t base = 0; base < n_pairs; base += BSPLINE_QR_TILE) {
                const size_t m = (n_pairs - base < BSPLINE_QR_TILE)
                                     ? (n_pairs - base) : BSPLINE_QR_TILE;

                // k-independent: R table, 1/(4 pi R), and the moment weights.
                size_t q = base / n_qp;
                size_t r = base % n_qp;
                for (size_t t = 0; t < m; t++) {
                    const double dx = pi[q*3 + 0] - pj[r*3 + 0];
                    const double dy = pi[q*3 + 1] - pj[r*3 + 1];
                    const double dz = pi[q*3 + 2] - pj[r*3 + 2];
                    R[t] = std::sqrt(dx*dx + dy*dy + dz*dz + a_squared);

                    const double wi = glw(q) * Li;
                    const double ui = glt(q) * Li;
                    const double wj = glw(r) * Lj;
                    const double uj = glt(r) * Lj;
                    double ui_pow[NM], uj_pow[NM];
                    ui_pow[0] = 1.0;
                    uj_pow[0] = 1.0;
                    for (int e = 1; e < NM; e++) {
                        ui_pow[e] = ui_pow[e-1] * ui;
                        uj_pow[e] = uj_pow[e-1] * uj;
                    }
                    const double wij = wi * wj;
                    for (int pp = 0; pp < NM; pp++) {
                        for (int PP = 0; PP < NM; PP++) {
                            wuwu[(pp * NM + PP) * m + t] = wij * ui_pow[pp] * uj_pow[PP];
                        }
                    }

                    if (++r == n_qp) { r = 0; ++q; }
                }
                MW_OMP_SIMD()
                for (size_t t = 0; t < m; t++) {
                    inv_R_4pi[t] = inv_4pi / R[t];
                }

                // Per-k: only the exp(-jkR) phase changes.
                for (size_t kk = 0; kk < n_k; kk++) {
                    double k = ka(kk);
                    MW_OMP_SIMD()
                    for (size_t t = 0; t < m; t++) {
                        phases[t] = -k * R[t];
                    }
                    MW_OMP_SIMD()
                    for (size_t t = 0; t < m; t++) {
                        cos_phases[t] = std::cos(phases[t]);
                    }
                    MW_OMP_SIMD()
                    for (size_t t = 0; t < m; t++) {
                        sin_phases[t] = std::sin(phases[t]);
                    }
                    MW_OMP_SIMD()
                    for (size_t t = 0; t < m; t++) {
                        G_re[t] = cos_phases[t] * inv_R_4pi[t];
                        G_im[t] = sin_phases[t] * inv_R_4pi[t];
                    }
                    for (int pP = 0; pP < NMM; pP++) {
                        double sr = 0.0, si = 0.0;
                        const double *w_row = &wuwu[pP * m];
                        // No `omp simd reduction` here (momwire#781): the clause LICENSES
                        // reassociation, so the reduction tree follows whatever
                        // vectorization factor the compiler picks per FUNCTION -- and the
                        // reduced and EK kernels differ in register pressure. That made
                        // their all-ineligible outputs disagree by 1 ulp on arm64 while
                        // matching on x86-64, breaking the exact-reduction gates that
                        // momwire#270 U2 relies on. Measured single-threaded (pinned,
                        // passive wait, min of 5, 3 alternating rounds), the clause is
                        // worth -0.2%/+0.4% on the bspline fills -- i.e. nothing. It IS
                        // worth ~4.6% in _accel_razor.cpp, which keeps its clause and has
                        // no cross-kernel equality gate to protect.
                        for (size_t t = 0; t < m; t++) {
                            sr += w_row[t] * G_re[t];
                            si += w_row[t] * G_im[t];
                        }
                        std::complex<double> part(sr, si);
                        if (first_chunk) {
                            j_view(kk, pP / NM, pP % NM, i, j) = part;
                        } else {
                            j_view(kk, pP / NM, pP % NM, i, j) += part;
                        }
                    }
                }
                first_chunk = false;
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
// Unit 2 wrote this as its own function so the reduced kernel's lines stayed
// untouched. momwire#1362 folded it into `seg_seg_full_moments_bspline_
// kernel_impl` as `EK = true` once the reduced kernel had grown the t-outer
// stage 2 and the AVX2 lanes the twin lacked (EK on cost bs2 2-3x on free
// and above decks); the reduced instantiations' Z is gated unchanged.
template<int D>
static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_kernel_ek(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    double k,
    const PairOrderLadder& ladder,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_i,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_j,
    double a_ek,
    bool reference = false,
    double a_ek_src = -1.0
) {
    if (group_i.ndim() != 1 || group_j.ndim() != 1 ||
        (seg_l_i.ndim() == 2 && group_i.shape(0) != seg_l_i.shape(0)) ||
        (seg_l_j.ndim() == 2 && group_j.shape(0) != seg_l_j.shape(0))) {
        throw std::runtime_error("group_i/group_j must match N_i/N_j");
    }
    // momwire#1362: the reduced kernel's own walk and AVX2 lanes, with the
    // coaxial factor applied between stage 1 and stage 2 on eligible pairs.
    // Each pair's arithmetic at a given order is the pre-#1362 EK kernel's,
    // which this replaced: the same R, the same G, the same factor loop, and
    // each moment the same ascending-t chain of one rounded multiply and one
    // rounded add -- with t outermost (8c311afb) and four column pairs per
    // vector lane (momwire#1290) only reordering work ACROSS chains. Gated to
    // the bit against the walk (`reference=True`).
    //
    // The ladder is the reduced kernel's (momwire#906): a pair's order is
    // chosen by the same selector, and the factor is applied at whatever
    // order the pair runs. A one-tier ladder is the single-rule entry bit for
    // bit; the tiered entry's movement against it is gated in
    // `tests/test_ek_pair_order_ladder_1362.py`.
    return seg_seg_full_moments_bspline_kernel_impl<D, false, true>(
        seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k, 0.0,
        ladder, reference, group_i.data(), group_j.data(), a_ek, a_ek_src);
}

// Swept-k (batched) variant of seg_seg_full_moments_bspline_kernel_ek.
//
// Same relationship to seg_seg_full_moments_bspline_kernel_ek as the
// reduced swept kernel has to the reduced single-k one: the per-(i, j)
// geometry (positions, R, wuwu weights) and, under EK, the k-independent
// T1/T2 halves of `fac` are hoisted out of the k loop and reused across the
// whole k_array; only the phase and C1/C2 vary per frequency.
//
// momwire#1362 (follow-up to #1363/#1365): this twin was the last EK
// off-edge fill still on the pre-#1290 per-pair loop. It now has
//
// * a t-outer stage 2 in the walk (8c311afb's reordering: each moment is
//   still one ascending-t chain of one rounded multiply and one rounded add
//   starting at 0.0, so the walk's moments are the pre-#1362 kernel's to the
//   bit; only work ACROSS chains moved);
// * the AVX2 four-column lanes of momwire#1290 (`!reference`, AVX2 build
//   only): R, 1/(4 pi R), the moment weights and T1/T2 built once per group
//   of four columns, then per k the walk's own loops over all four lanes'
//   points, the coaxial factor applied per lane by a blend (eligibility is
//   per PAIR, so the four lanes of a group can disagree), and stage 2 in
//   vector accumulators. Gated to the bit against the walk
//   (`reference=True`), which is the pre-#1362 arithmetic;
// * the pair-order ladder of momwire#906 (the `_tiered` entry): a pair's
//   order is chosen by the same selector as the single-k kernel and the
//   factor is applied at whatever order the pair runs. A one-tier ladder is
//   the single-rule entry bit for bit.
//
// A multi-chunk rule (n_qp > 8) keeps the original chunk semantics: each
// chunk's sums start at 0.0 and the first chunk assigns while later chunks
// add, exactly as before; such a pair is always walked.
template<int D>
static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_swept_kernel_ek(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array,
    const PairOrderLadder& ladder,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_i,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_j,
    double a_ek,
    bool reference = false,
    double a_ek_src = -1.0
) {
    static constexpr int NM = D + 1;
    static constexpr int NMM = NM * NM;

    auto sli = seg_l_i.unchecked<2>();
    auto sri = seg_r_i.unchecked<2>();
    auto slj = seg_l_j.unchecked<2>();
    auto srj = seg_r_j.unchecked<2>();
    auto ka  = k_array.unchecked<1>();

    if (sli.shape(1) != 3 || sri.shape(1) != 3 ||
        slj.shape(1) != 3 || srj.shape(1) != 3) {
        throw std::runtime_error("segment endpoint arrays must have shape (N, 3)");
    }
    if (sli.shape(0) != sri.shape(0) || slj.shape(0) != srj.shape(0)) {
        throw std::runtime_error("seg_l and seg_r must have matching N");
    }

    size_t N_i = sli.shape(0);
    size_t N_j = slj.shape(0);
    size_t n_k = ka.shape(0);
    const size_t n_tiers = ladder.n_tiers();
    if (group_i.ndim() != 1 || group_j.ndim() != 1 ||
        (size_t)group_i.shape(0) != N_i || (size_t)group_j.shape(0) != N_j) {
        throw std::runtime_error("group_i/group_j must match N_i/N_j");
    }
    const int64_t *grp_i = group_i.data();
    const int64_t *grp_j = group_j.data();
    std::vector<double> ks(n_k);
    for (size_t kk = 0; kk < n_k; kk++) ks[kk] = ka(kk);

    py::array_t<std::complex<double>> J({n_k, (size_t)NM, (size_t)NM, N_i, N_j});
    auto j_view = J.mutable_unchecked<5>();

    py::gil_scoped_release release;

    const double inv_4pi = 1.0 / (4.0 * M_PI);
    // momwire#1368: b = the source tube's radius (`a_ek_src`, negative = a_ek).
    const double b_ek = a_ek_src < 0.0 ? a_ek : a_ek_src;
    const double a2_ek = b_ek * b_ek;                  // b^2
    const double a4_ek = a2_ek * (a_ek * a_ek);        // b^2 a^2
    // The factor's R floored at the tube, as `..._kernel_impl` (momwire#1368).
    const bool ek_floor = b_ek > a_ek;

    // Lengths, then PER TIER the quadrature positions -- with one tier the
    // pre-#1362 precompute split into two loops: the same expressions
    // produce the same doubles.
    std::vector<double> len_i(N_i);
    std::vector<double> len_j(N_j);
    for (size_t i = 0; i < N_i; i++) {
        double dx = sri(i,0) - sli(i,0);
        double dy = sri(i,1) - sli(i,1);
        double dz = sri(i,2) - sli(i,2);
        len_i[i] = std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    for (size_t j = 0; j < N_j; j++) {
        double dx = srj(j,0) - slj(j,0);
        double dy = srj(j,1) - slj(j,1);
        double dz = srj(j,2) - slj(j,2);
        len_j[j] = std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    std::vector<std::vector<double>> pos_i(n_tiers);
    std::vector<std::vector<double>> pos_j(n_tiers);
    for (size_t tier = 0; tier < n_tiers; tier++) {
        const size_t n_qp = ladder.n_qp(tier);
        const double *gt = ladder.t_at(tier);
        pos_i[tier].resize(N_i * n_qp * 3);
        pos_j[tier].resize(N_j * n_qp * 3);
        std::vector<double>& pi_t = pos_i[tier];
        std::vector<double>& pj_t = pos_j[tier];
        for (size_t i = 0; i < N_i; i++) {
            for (size_t q = 0; q < n_qp; q++) {
                double t = gt[q];
                pi_t[(i*n_qp + q)*3 + 0] = (1.0 - t) * sli(i,0) + t * sri(i,0);
                pi_t[(i*n_qp + q)*3 + 1] = (1.0 - t) * sli(i,1) + t * sri(i,1);
                pi_t[(i*n_qp + q)*3 + 2] = (1.0 - t) * sli(i,2) + t * sri(i,2);
            }
        }
        for (size_t j = 0; j < N_j; j++) {
            for (size_t r = 0; r < n_qp; r++) {
                double t = gt[r];
                pj_t[(j*n_qp + r)*3 + 0] = (1.0 - t) * slj(j,0) + t * srj(j,0);
                pj_t[(j*n_qp + r)*3 + 1] = (1.0 - t) * slj(j,1) + t * srj(j,1);
                pj_t[(j*n_qp + r)*3 + 2] = (1.0 - t) * slj(j,2) + t * srj(j,2);
            }
        }
    }
    std::vector<double> c_i, c_j;
    if (n_tiers > 1) {
        c_i.resize(N_i * 3);
        c_j.resize(N_j * 3);
        for (size_t i = 0; i < N_i; i++) {
            for (int d = 0; d < 3; d++) c_i[i*3 + d] = 0.5 * (sli(i,d) + sri(i,d));
        }
        for (size_t j = 0; j < N_j; j++) {
            for (int d = 0; d < 3; d++) c_j[j*3 + d] = 0.5 * (slj(j,d) + srj(j,d));
        }
    }
    // The single-k kernel's selector, verbatim, so the two fills put a pair
    // on the same rule.
    auto pair_tier = [&](size_t i, size_t j) -> size_t {
        size_t tier = 0;
        if (n_tiers > 1) {
            const double dx = c_i[i*3 + 0] - c_j[j*3 + 0];
            const double dy = c_i[i*3 + 1] - c_j[j*3 + 1];
            const double dz = c_i[i*3 + 2] - c_j[j*3 + 2];
            const double ratio = std::sqrt(dx*dx + dy*dy + dz*dz)
                                 / std::max(len_i[i], len_j[j]);
            for (size_t t = 1; t < n_tiers; t++) {
                if (ratio >= ladder.ratio[t]) tier = t; else break;
            }
        }
        return tier;
    };

    // The per-pair walk: the pre-#1362 loop body, tier-aware, with stage 2
    // t-outer. TILED OVER qr (momwire#762) with the chunk OUTSIDE the k loop
    // so R / 1/(4 pi R) / wuwu / T1 / T2 are built once per chunk and reused
    // across the sweep. First chunk assigns, later chunks accumulate; one
    // chunk at n_qp <= 8.
    auto pair_walk = [&](size_t i, size_t j) {
        alignas(32) double R[BSPLINE_QR_TILE];
        alignas(32) double inv_R_4pi[BSPLINE_QR_TILE];
        alignas(32) double phases[BSPLINE_QR_TILE];
        alignas(32) double cos_phases[BSPLINE_QR_TILE];
        alignas(32) double sin_phases[BSPLINE_QR_TILE];
        alignas(32) double G_re[BSPLINE_QR_TILE], G_im[BSPLINE_QR_TILE];
        alignas(32) double t1v[BSPLINE_QR_TILE], t2v[BSPLINE_QR_TILE];
        alignas(32) double wuwu[NMM * BSPLINE_QR_TILE];

        const size_t tier = pair_tier(i, j);
        const size_t n_qp = ladder.n_qp(tier);
        const double *gt = ladder.t_at(tier);
        const double *gw = ladder.w_at(tier);
        const size_t n_pairs = n_qp * n_qp;
        const double *pi = &pos_i[tier][i * n_qp * 3];
        const double *pj = &pos_j[tier][j * n_qp * 3];
        const double Li = len_i[i];
        const double Lj = len_j[j];
        const bool eligible = (grp_i[i] == grp_j[j]) && (grp_i[i] >= 0);

        bool first_chunk = true;
        for (size_t base = 0; base < n_pairs; base += BSPLINE_QR_TILE) {
            const size_t m = (n_pairs - base < BSPLINE_QR_TILE)
                                 ? (n_pairs - base) : BSPLINE_QR_TILE;

            size_t q = base / n_qp;
            size_t r = base % n_qp;
            for (size_t t = 0; t < m; t++) {
                const double dx = pi[q*3 + 0] - pj[r*3 + 0];
                const double dy = pi[q*3 + 1] - pj[r*3 + 1];
                const double dz = pi[q*3 + 2] - pj[r*3 + 2];
                R[t] = std::sqrt(dx*dx + dy*dy + dz*dz + a_squared);

                const double wi = gw[q] * Li;
                const double ui = gt[q] * Li;
                const double wj = gw[r] * Lj;
                const double uj = gt[r] * Lj;
                double ui_pow[NM], uj_pow[NM];
                ui_pow[0] = 1.0;
                uj_pow[0] = 1.0;
                for (int e = 1; e < NM; e++) {
                    ui_pow[e] = ui_pow[e-1] * ui;
                    uj_pow[e] = uj_pow[e-1] * uj;
                }
                const double wij = wi * wj;
                for (int pp = 0; pp < NM; pp++) {
                    for (int PP = 0; PP < NM; PP++) {
                        wuwu[t * NMM + pp * NM + PP] = wij * ui_pow[pp] * uj_pow[PP];
                    }
                }

                if (++r == n_qp) { r = 0; ++q; }
            }
            MW_OMP_SIMD()
            for (size_t t = 0; t < m; t++) {
                inv_R_4pi[t] = inv_4pi / R[t];
            }
            if (eligible) {
                // T1, T2: functions of R and a_ek alone, so they hoist out
                // of the k loop exactly as the same-edge swept twin's do.
                MW_OMP_SIMD()
                for (size_t t = 0; t < m; t++) {
                    const double Rf = ek_floor ? std::max(R[t], b_ek) : R[t];
                    double r2 = Rf * Rf;
                    double r4 = r2 * r2;
                    t1v[t] = 0.25 * a4_ek / r4;
                    t2v[t] = 0.5 * a2_ek / r2;
                }
            }

            for (size_t kk = 0; kk < n_k; kk++) {
                const double k = ks[kk];
                MW_OMP_SIMD()
                for (size_t t = 0; t < m; t++) {
                    phases[t] = -k * R[t];
                }
                MW_OMP_SIMD()
                for (size_t t = 0; t < m; t++) {
                    cos_phases[t] = std::cos(phases[t]);
                }
                MW_OMP_SIMD()
                for (size_t t = 0; t < m; t++) {
                    sin_phases[t] = std::sin(phases[t]);
                }
                MW_OMP_SIMD()
                for (size_t t = 0; t < m; t++) {
                    G_re[t] = cos_phases[t] * inv_R_4pi[t];
                    G_im[t] = sin_phases[t] * inv_R_4pi[t];
                }
                if (eligible) {
                    MW_OMP_SIMD()
                    for (size_t t = 0; t < m; t++) {
                        double kr = k * (ek_floor ? std::max(R[t], b_ek) : R[t]);
                        double kr2 = kr * kr;
                        double c1r = 1.0;
                        double c1i = kr;
                        double c2r = 3.0 * c1r - kr2;
                        double c2i = 3.0 * c1i;
                        double facr = t1v[t] * c2r;
                        double faci = t1v[t] * c2i;
                        facr = facr - t2v[t] * c1r;
                        faci = faci - t2v[t] * c1i;
                        facr = facr + 1.0;
                        double gre = G_re[t];
                        double gim = G_im[t];
                        G_re[t] = gre * facr - gim * faci;
                        G_im[t] = gre * faci + gim * facr;
                    }
                }
                // Stage 2, t outermost: each moment's chain starts at 0.0
                // and adds w*G in ascending t, one rounded multiply and one
                // rounded add per term -- the per-pP loop's order, so the
                // moments do not move. No `omp simd reduction` (momwire#781):
                // any vectorizing is ACROSS accumulators, which is exact.
                double acc_re[NMM], acc_im[NMM];
                for (int pP = 0; pP < NMM; pP++) { acc_re[pP] = 0.0; acc_im[pP] = 0.0; }
                for (size_t t = 0; t < m; t++) {
                    const double gr = G_re[t], gi = G_im[t];
                    const double *w_t = &wuwu[t * NMM];
                    for (int pP = 0; pP < NMM; pP++) {
                        acc_re[pP] += w_t[pP] * gr;
                        acc_im[pP] += w_t[pP] * gi;
                    }
                }
                for (int pP = 0; pP < NMM; pP++) {
                    std::complex<double> part(acc_re[pP], acc_im[pP]);
                    if (first_chunk) {
                        j_view(kk, pP / NM, pP % NM, i, j) = part;
                    } else {
                        j_view(kk, pP / NM, pP % NM, i, j) += part;
                    }
                }
            }
            first_chunk = false;
        }
    };

#if MW_OFFEDGE_LANES_1290
    if (!reference) {
        // The lane kernel (momwire#1290's, for a sweep): W columns
        // (i, j0..j0+W-1) at a time wherever they share a tier whose rule is
        // one qr chunk of an EVEN order; anything else is walked. The
        // k-independent tables are built once per group, then each k runs
        // the walk's stage 1, factor and stage 2 over all the lanes. Every
        // lane's arithmetic is the walk's for its pair, operation for
        // operation (-ffp-contract=off, no fma written); the even order is
        // what puts every cos/sin point through libmvec's vector entry on
        // both routes (see the single-k kernel's lane notes).
        using namespace offedge_lanes;
        constexpr size_t LN = W;
        const size_t n_grp = (N_j + LN - 1) / LN;
        std::vector<std::vector<double>> pjT(n_tiers);
        for (size_t tier = 0; tier < n_tiers; tier++) {
            const size_t n_qp = ladder.n_qp(tier);
            const std::vector<double>& pj_t = pos_j[tier];
            std::vector<double>& out_t = pjT[tier];
            out_t.resize(3 * n_qp * N_j);
            for (size_t j = 0; j < N_j; j++) {
                for (size_t c = 0; c < 3; c++) {
                    for (size_t r = 0; r < n_qp; r++) {
                        out_t[(c * n_qp + r) * N_j + j] = pj_t[(j*n_qp + r)*3 + c];
                    }
                }
            }
        }
        const vd v_a2 = set1(a_squared);
        const vd v_t1n = set1(0.25 * a4_ek);
        const vd v_t2n = set1(0.5 * a2_ek);
        const vd v_bek = set1(b_ek);

        #pragma omp parallel
        {
        alignas(ALIGN) double wv[BSPLINE_QR_TILE * NMM * LN];
        alignas(ALIGN) double R[BSPLINE_QR_TILE * LN];
        alignas(ALIGN) double inv_R_4pi[BSPLINE_QR_TILE * LN];
        alignas(ALIGN) double t1v[BSPLINE_QR_TILE * LN];
        alignas(ALIGN) double t2v[BSPLINE_QR_TILE * LN];
        alignas(ALIGN) double phases[BSPLINE_QR_TILE * LN];
        alignas(ALIGN) double cos_phases[BSPLINE_QR_TILE * LN];
        alignas(ALIGN) double sin_phases[BSPLINE_QR_TILE * LN];
        alignas(ALIGN) double G_re[BSPLINE_QR_TILE * LN];
        alignas(ALIGN) double G_im[BSPLINE_QR_TILE * LN];
        MW_OMP_FOR_COLLAPSE2
        for (size_t i = 0; i < N_i; i++) {
            for (size_t g = 0; g < n_grp; g++) {
                const size_t j0 = g * LN;
                bool lanes = j0 + LN <= N_j;
                size_t tier = 0;
                if (lanes) {
                    tier = pair_tier(i, j0);
                    for (size_t l = 1; l < LN; l++) {
                        lanes = lanes && pair_tier(i, j0 + l) == tier;
                    }
                    const size_t nq = ladder.n_qp(tier);
                    lanes = lanes && nq % 2 == 0 && nq * nq <= BSPLINE_QR_TILE;
                }
                if (!lanes) {
                    const size_t j1 = std::min(j0 + LN, N_j);
                    for (size_t j = j0; j < j1; j++) pair_walk(i, j);
                    continue;
                }

                const size_t n_qp = ladder.n_qp(tier);
                const size_t m = n_qp * n_qp;
                const size_t m4 = m * LN;
                const double *gt = ladder.t_at(tier);
                const double *gw = ladder.w_at(tier);
                const double *pi = &pos_i[tier][i * n_qp * 3];
                const double Li = len_i[i];
                double Lj[LN];
                for (size_t l = 0; l < LN; l++) Lj[l] = len_j[j0 + l];

                r_bcast(R, pi, pjT[tier].data(), n_qp, N_j, j0, v_a2);
                MW_OMP_SIMD()
                for (size_t t = 0; t < m4; t++) {
                    inv_R_4pi[t] = inv_4pi / R[t];
                }
                wv_bcast<NM>(wv, gt, gw, n_qp, Li, Lj);

                // EK eligibility per lane, and the k-independent T1/T2 when
                // any lane is eligible (computed for every lane; an
                // ineligible lane's are never used).
                const int64_t gi_i = grp_i[i];
                bool e[LN];
                bool any_e = false;
                for (size_t l = 0; l < LN; l++) {
                    e[l] = gi_i >= 0 && grp_j[j0 + l] == gi_i;
                    any_e = any_e || e[l];
                }
                const vm emask = mask_from(e);
                if (any_e) {
                    for (size_t t = 0; t < m; t++) {
                        vd Rq = load(R + t * LN);
                        if (ek_floor) Rq = max(Rq, v_bek);
                        const vd r2 = mul(Rq, Rq);
                        const vd r4 = mul(r2, r2);
                        store(t1v + t * LN, div(v_t1n, r4));
                        store(t2v + t * LN, div(v_t2n, r2));
                    }
                }

                for (size_t kk = 0; kk < n_k; kk++) {
                    const double k = ks[kk];
                    MW_OMP_SIMD()
                    for (size_t t = 0; t < m4; t++) {
                        phases[t] = -k * R[t];
                    }
                    MW_OMP_SIMD()
                    for (size_t t = 0; t < m4; t++) {
                        cos_phases[t] = std::cos(phases[t]);
                    }
                    MW_OMP_SIMD()
                    for (size_t t = 0; t < m4; t++) {
                        sin_phases[t] = std::sin(phases[t]);
                    }
                    MW_OMP_SIMD()
                    for (size_t t = 0; t < m4; t++) {
                        G_re[t] = cos_phases[t] * inv_R_4pi[t];
                        G_im[t] = sin_phases[t] * inv_R_4pi[t];
                    }
                    if (any_e) {
                        const vd v_k = set1(k);
                        for (size_t t = 0; t < m; t++) {
                            vd Rq = load(R + t * LN);
                            if (ek_floor) Rq = max(Rq, v_bek);
                            ek_factor(G_re + t * LN, G_im + t * LN,
                                      load(t1v + t * LN), load(t2v + t * LN),
                                      mul(v_k, Rq), emask);
                        }
                    }

                    // Stage 2: real and imaginary sums in separate passes;
                    // `phases` holds the real sums once its points are spent.
                    vd acc[NMM];
                    stage2<NMM>(G_re, wv, m, acc);
                    for (int pP = 0; pP < NMM; pP++) store(phases + pP * LN, acc[pP]);
                    stage2<NMM>(G_im, wv, m, acc);
                    for (int pP = 0; pP < NMM; pP++) {
                        store_interleaved(
                            reinterpret_cast<double *>(&j_view(kk, pP / NM, pP % NM, i, j0)),
                            load(phases + pP * LN), acc[pP]);
                    }
                }
            }
        }
        }
    } else
#else
    (void)reference;
#endif
    {
    MW_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t i = 0; i < N_i; i++) {
        for (size_t j = 0; j < N_j; j++) {
            pair_walk(i, j);
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
template<int D, bool COMPLEX_EPS>
static py::array_t<std::complex<double>>
assemble_Z_bspline_kernel(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<double, py::array::c_style | py::array::forcecast> td_all,
    double omega,
    double eps_,
    double mu_,
    uintptr_t cancel_flag = 0,
    double c_re = 0.0,
    double c_im = 0.0
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
    // With COMPLEX_EPS the real eps_ is unused (the wrapper passes 1.0 so the
    // divide below is finite) and c = 1/(j*omega*eps~) carries the medium.
    const double inv_omega_eps = 1.0 / (omega * eps_);

    MW_CANCEL_SETUP(cancel_flag);
    MW_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t m = 0; m < n_basis; m++) {
        for (size_t n = 0; n < n_basis; n++) {
            MW_CANCEL_POLL();
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

            double Zre, Zim;
            if (COMPLEX_EPS) {
                // Z = j*omega*mu*zA + c*zPhi with c = 1/(j*omega*eps~) COMPLEX
                // (momwire#910): the medium's eps~ enters this kernel only
                // here, so the buried assembly is the real-eps loop above
                // with a complex scalar in the combine.
                Zre = -omega_mu * zA_im + (c_re * zPhi_re - c_im * zPhi_im);
                Zim = omega_mu * zA_re + (c_re * zPhi_im + c_im * zPhi_re);
            } else {
                Zre = -omega_mu * zA_im + zPhi_im * inv_omega_eps;
                Zim = omega_mu * zA_re - zPhi_re * inv_omega_eps;
            }
            z_view(m, n) = std::complex<double>(Zre, Zim);
        }
    }

    MW_THROW_IF_ABORTED();
    return Z;
}


// momwire#1290: the windowed assembler's AVX2 lane kernel. Only where the
// vector fma IS the scalar fma -- an x86 build with -mfma, whose mw_fma::fma
// is std::fma, one vfmadd. The baseline (_sse2) variant, arm64 and MSVC keep
// the reference loop alone, so their bits cannot move.
#if defined(__AVX2__) && defined(__FMA__) && !defined(_MSC_VER)
#define MW_WINDOWED_LANES_1290 1
#include <immintrin.h>
#else
#define MW_WINDOWED_LANES_1290 0
#endif


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
template<int D, bool COMPLEX_EPS, bool ROW_MAP = false>
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
    uintptr_t cancel_flag = 0,
    double c_re = 0.0,
    double c_im = 0.0,
    // momwire#1132: with ROW_MAP, Z is ROW-COMPACT -- (n_rows, n_basis) --
    // and basis row m lands in Z row row_of[m]. The wrapper validates
    // the map against m_idx before the GIL is released. Without it this
    // pointer is never read and the instantiation is the shipped one.
    const int64_t *row_of = nullptr,
    // momwire#1290: true runs the per-entry loop below even where the lane
    // kernel is compiled -- the reference it is gated against to the bit.
    bool reference = false
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
    if (ROW_MAP) {
        if (Z.shape(1) != (long)n_basis) {
            throw std::runtime_error("a row-compact Z must be (n_rows, n_basis)");
        }
    } else if (Z.shape(0) != (long)n_basis || Z.shape(1) != (long)n_basis) {
        throw std::runtime_error("Z.shape must be (n_basis, n_basis)");
    }
    auto z_view = Z.mutable_unchecked<2>();

    py::gil_scoped_release release;

    const double omega_mu = omega * mu_;
    // With COMPLEX_EPS the real eps_ is unused (the wrapper passes 1.0 so the
    // divide below is finite) and c = 1/(j*omega*eps~) carries the medium.
    const double inv_omega_eps = 1.0 / (omega * eps_);
    size_t n_m = (size_t)m_idx.shape(0);
    size_t n_n = (size_t)n_idx.shape(0);

    MW_CANCEL_SETUP(cancel_flag);
#if MW_WINDOWED_LANES_1290
    if (!reference) {
        // The lane kernel (momwire#1290): L entries (m, n..n+L-1) at a time,
        // over TM x TN tiles of (mi, ni).
        //
        // Every entry's arithmetic is the reference loop's, operation for
        // operation: the same products, the same fma chain over (p, q) for
        // wA / wPhi, the same fma(td, wA, zA) and `zPhi + wPhi` over (a, b) in
        // the same order, the same tangent dot, the same combine. An (a, b)
        // the reference skips (sn outside [j0, j1)) is computed here for that
        // lane and then dropped by a blend, never added, so the accumulator is
        // left as `continue` left it. Nothing is reassociated: each Z entry is
        // the reference's to the bit.
        //
        // Measured (Haswell, free-space array at N = 2816, 13 calls, 9.0 M
        // entries): the reference ran 2.09 s on one thread and 0.66 s on four;
        // this kernel 0.75 s and 0.29 s. What the reference paid for, in
        // order: one entry at a time is scalar and serial (each wA is a
        // 9-deep fma chain), and the row-at-a-time sweep wrote a column-major
        // Z one cache line per entry, 45 KB apart (a C-ordered Z ran the
        // same call 1.5x faster). Lanes vectorise across entries; tiles keep
        // a tile's Z lines and its J rows cache-resident while it fills. The
        // entries are packed (re, im) per lane so a contiguous run of J --
        // the common case, consecutive bases on consecutive segments -- is
        // read as it lies.
        constexpr int V = 4;         // ymm registers per quantity, two entries' (re, im) each
        constexpr size_t L = 2 * V;  // entries per lane group
        constexpr size_t K = 2 * L;
        constexpr size_t TM = 64;  // tile rows (mi)
        constexpr size_t TN = 16;  // tile columns (ni), a multiple of L
        constexpr int NP = NM * NM;
        const size_t n_grp = (n_n + L - 1) / L;
        const size_t n_pad = n_grp * L;
        const size_t lane_stride = 2 * n_pad;
        // Column side, once per call. Per (b, lane): the window-relative J
        // offset in doubles, the polynomial row and the source tangent, each
        // (re, im)-duplicated; a lane the reference would skip carries a zero
        // mask. Per (b, group): whether its lanes read one contiguous run of J
        // (then `goff` is that run's start, else -1), and `gcov` 1 when every
        // lane is live, 2 when none is (the reference skips that (a, b) for
        // all of them, and an empty window has no J column to stage).
        std::vector<int64_t> live2((size_t)NM * lane_stride, 0);
        std::vector<int64_t> loff((size_t)NM * n_pad, 0);
        std::vector<double> pn2((size_t)NM * NM * lane_stride, 0.0);
        std::vector<double> tn2((size_t)NM * 3 * lane_stride, 0.0);
        std::vector<int64_t> goff((size_t)NM * n_grp, -1);
        std::vector<char> gcov((size_t)NM * n_grp, 0);
        for (size_t ni = 0; ni < n_n; ni++) {
            const int64_t n = ni_view(ni);
            for (int b = 0; b < NM; b++) {
                const int64_t sn = ss_view(n, b);
                if (sn >= j0 && sn < j1) {
                    live2[(size_t)b * lane_stride + 2 * ni] = -1;
                    live2[(size_t)b * lane_stride + 2 * ni + 1] = -1;
                    loff[(size_t)b * n_pad + ni] = 2 * (sn - j0);
                    for (int c = 0; c < 3; c++) {
                        const double t = t_view(sn, c);
                        tn2[((size_t)b * 3 + c) * lane_stride + 2 * ni] = t;
                        tn2[((size_t)b * 3 + c) * lane_stride + 2 * ni + 1] = t;
                    }
                }
                for (int q = 0; q < NM; q++) {
                    const double v = p_view(n, b, q);
                    pn2[((size_t)b * NM + q) * lane_stride + 2 * ni] = v;
                    pn2[((size_t)b * NM + q) * lane_stride + 2 * ni + 1] = v;
                }
            }
        }
        for (int b = 0; b < NM; b++) {
            for (size_t gi = 0; gi < n_grp; gi++) {
                const size_t g = gi * L;
                bool all = true, none = true, contig = true;
                for (size_t l = 0; l < L; l++) {
                    const bool on = live2[(size_t)b * lane_stride + 2 * (g + l)] != 0;
                    all = all && on;
                    none = none && !on;
                    contig = contig && on &&
                             loff[(size_t)b * n_pad + g + l] == loff[(size_t)b * n_pad + g] + 2 * (int64_t)l;
                }
                gcov[(size_t)b * n_grp + gi] = all ? 1 : (none ? 2 : 0);
                if (contig) goff[(size_t)b * n_grp + gi] = loff[(size_t)b * n_pad + g];
            }
        }
        const double *Jd = reinterpret_cast<const double *>(J_chunk.data());
        const size_t J_row = 2 * (size_t)(j1 - j0);
        const size_t J_plane = (size_t)(i1 - i0) * J_row;
        size_t poff_j[NP], poff_s[NP];
        for (int pq = 0; pq < NP; pq++) {
            poff_j[pq] = (size_t)pq * J_plane;
            poff_s[pq] = (size_t)pq * K;
        }
        const size_t n_tm = (n_m + TM - 1) / TM;
        const size_t n_tn = (n_pad + TN - 1) / TN;

        MW_OMP_PARALLEL_FOR_COLLAPSE2
        for (size_t tm = 0; tm < n_tm; tm++) {
            for (size_t tn = 0; tn < n_tn; tn++) {
                MW_CANCEL_POLL();
                const size_t mi_end = std::min(n_m, (tm + 1) * TM);
                const size_t g_end = std::min(n_pad, (tn + 1) * TN);
                double jst[NP * K];
                for (size_t mi = tm * TM; mi < mi_end; mi++) {
                    const int64_t m = mi_view(mi);
                    for (size_t g = tn * TN; g < g_end; g += L) {
                        const size_t gi = g / L;
                        __m256d zA[V], zP[V];
                        for (int v = 0; v < V; v++) {
                            zA[v] = _mm256_setzero_pd();
                            zP[v] = _mm256_setzero_pd();
                        }
                        for (int a = 0; a < NM; a++) {
                            const int64_t sm = ss_view(m, a);
                            if (sm < i0 || sm >= i1) continue;
                            const __m256d tm0 = _mm256_set1_pd(t_view(sm, 0));
                            const __m256d tm1 = _mm256_set1_pd(t_view(sm, 1));
                            const __m256d tm2 = _mm256_set1_pd(t_view(sm, 2));
                            double pm[NM];
                            for (int p = 0; p < NM; p++) pm[p] = p_view(m, a, p);
                            const double *Jrow = Jd + (size_t)(sm - i0) * J_row;
                            for (int b = 0; b < NM; b++) {
                                const char cov = gcov[(size_t)b * n_grp + gi];
                                if (cov == 2) continue;
                                const int64_t go = goff[(size_t)b * n_grp + gi];
                                const double *base;
                                const size_t *po;
                                if (go >= 0) {
                                    base = Jrow + go;
                                    po = poff_j;
                                } else {
                                    // Stage the lanes' J into a contiguous block;
                                    // a dead lane reads column 0 and is masked.
                                    const int64_t *lo = &loff[(size_t)b * n_pad + g];
                                    for (int pq = 0; pq < NP; pq++)
                                        for (size_t l = 0; l < L; l++) {
                                            jst[pq * K + 2 * l] = Jrow[poff_j[pq] + lo[l]];
                                            jst[pq * K + 2 * l + 1] = Jrow[poff_j[pq] + lo[l] + 1];
                                        }
                                    base = jst;
                                    po = poff_s;
                                }
                                __m256d wA[V], wP[V];
                                for (int v = 0; v < V; v++) {
                                    wA[v] = _mm256_setzero_pd();
                                    wP[v] = _mm256_setzero_pd();
                                }
                                for (int p = 0; p < NM; p++) {
                                    const __m256d mp = _mm256_set1_pd(pm[p]);
                                    for (int q = 0; q < NM; q++) {
                                        const double *pnq =
                                            &pn2[((size_t)b * NM + q) * lane_stride + 2 * g];
                                        const double *jp = base + po[p * NM + q];
                                        __m256d pr[V];
                                        for (int v = 0; v < V; v++) {
                                            pr[v] = _mm256_mul_pd(mp, _mm256_loadu_pd(pnq + 4 * v));
                                            wA[v] = _mm256_fmadd_pd(pr[v], _mm256_loadu_pd(jp + 4 * v), wA[v]);
                                        }
                                        if (p >= 1 && q >= 1) {
                                            const __m256d s = _mm256_set1_pd((double)(p * q));
                                            const double *jl = base + po[(p - 1) * NM + (q - 1)];
                                            for (int v = 0; v < V; v++)
                                                wP[v] = _mm256_fmadd_pd(_mm256_mul_pd(s, pr[v]),
                                                                        _mm256_loadu_pd(jl + 4 * v), wP[v]);
                                        }
                                    }
                                }
                                const double *tb = &tn2[(size_t)b * 3 * lane_stride + 2 * g];
                                for (int v = 0; v < V; v++) {
                                    const __m256d td = _mm256_add_pd(
                                        _mm256_add_pd(_mm256_mul_pd(tm0, _mm256_loadu_pd(tb + 4 * v)),
                                                      _mm256_mul_pd(tm1, _mm256_loadu_pd(tb + lane_stride + 4 * v))),
                                        _mm256_mul_pd(tm2, _mm256_loadu_pd(tb + 2 * lane_stride + 4 * v)));
                                    const __m256d nA = _mm256_fmadd_pd(td, wA[v], zA[v]);
                                    const __m256d nP = _mm256_add_pd(zP[v], wP[v]);
                                    if (cov == 1) {
                                        zA[v] = nA;
                                        zP[v] = nP;
                                    } else {
                                        const __m256d on = _mm256_loadu_pd(reinterpret_cast<const double *>(
                                            &live2[(size_t)b * lane_stride + 2 * g + 4 * v]));
                                        zA[v] = _mm256_blendv_pd(zA[v], nA, on);
                                        zP[v] = _mm256_blendv_pd(zP[v], nP, on);
                                    }
                                }
                            }
                        }
                        double zAs[K], zPs[K];
                        for (int v = 0; v < V; v++) {
                            _mm256_storeu_pd(zAs + 4 * v, zA[v]);
                            _mm256_storeu_pd(zPs + 4 * v, zP[v]);
                        }
                        for (size_t l = 0; l < L && g + l < n_n; l++) {
                            const int64_t n = ni_view(g + l);
                            const double zA_re = zAs[2 * l], zA_im = zAs[2 * l + 1];
                            const double zPhi_re = zPs[2 * l], zPhi_im = zPs[2 * l + 1];
                            double Zre, Zim;
                            if (COMPLEX_EPS) {
                                Zre = -omega_mu * zA_im + (c_re * zPhi_re - c_im * zPhi_im);
                                Zim = omega_mu * zA_re + (c_re * zPhi_im + c_im * zPhi_re);
                            } else {
                                Zre = -omega_mu * zA_im + zPhi_im * inv_omega_eps;
                                Zim = omega_mu * zA_re - zPhi_re * inv_omega_eps;
                            }
                            z_view(ROW_MAP ? row_of[m] : m, n) += std::complex<double>(Zre, Zim);
                        }
                    }
                }
            }
        }
        MW_THROW_IF_ABORTED();
        return;
    }
#else
    (void)reference;
#endif
    MW_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t mi = 0; mi < n_m; mi++) {
        for (size_t ni = 0; ni < n_n; ni++) {
            MW_CANCEL_POLL();
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
                            // Fused accumulation (momwire#1194, _fma_inline.h):
                            // without it this window loop measured +12 %.
                            wA_re = mw_fma::fma(prod, Jpq.real(), wA_re);
                            wA_im = mw_fma::fma(prod, Jpq.imag(), wA_im);
                            if (p >= 1 && q >= 1) {
                                std::complex<double> Jpm1qm1 =
                                    j_view(p - 1, q - 1, sm - i0, sn - j0);
                                double pq = (double)(p * q) * prod;
                                wPhi_re = mw_fma::fma(pq, Jpm1qm1.real(), wPhi_re);
                                wPhi_im = mw_fma::fma(pq, Jpm1qm1.imag(), wPhi_im);
                            }
                        }
                    }

                    zA_re = mw_fma::fma(td, wA_re, zA_re);
                    zA_im = mw_fma::fma(td, wA_im, zA_im);
                    zPhi_re += wPhi_re;
                    zPhi_im += wPhi_im;
                }
            }

            double Zre, Zim;
            if (COMPLEX_EPS) {
                // Z = j*omega*mu*zA + c*zPhi, c = 1/(j*omega*eps~) complex
                // (momwire#915, the #910 pattern on the windowed twin).
                Zre = -omega_mu * zA_im + (c_re * zPhi_re - c_im * zPhi_im);
                Zim = omega_mu * zA_re + (c_re * zPhi_im + c_im * zPhi_re);
            } else {
                Zre = -omega_mu * zA_im + zPhi_im * inv_omega_eps;
                Zim = omega_mu * zA_re - zPhi_re * inv_omega_eps;
            }
            z_view(ROW_MAP ? row_of[m] : m, n) += std::complex<double>(Zre, Zim);
        }
    }

    MW_THROW_IF_ABORTED();
}


// momwire#1132: the ROW-COMPACT target of the four windowed assemblers. The
// sector route (momwire#1029) reads a few rows of Z, so it hands over an
// (n_rows, n_basis) Z plus `row_of`, one entry per basis: the compact row a
// basis row lands in, or -1 for a row the fill must never write. Checked
// HERE, with the GIL held and before any kernel runs, rather than trusted
// under a released GIL: every m the call will visit must map inside Z, and no
// two of them to the same row -- the kernels' parallel `+=` is contention-free
// only because each (m, n) is its own entry. The arithmetic is untouched; only
// the address of the `+=` moves, so the compact rows are the dense rows bit
// for bit.
typedef py::array_t<int64_t, py::array::c_style | py::array::forcecast> RowOf1132;

static RowOf1132
checked_row_of_1132(
    const py::object &row_of,
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast> &m_idx,
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast> &support_seg,
    const py::array_t<std::complex<double>> &Z
) {
    RowOf1132 ro = row_of.cast<RowOf1132>();
    const py::ssize_t n_basis = support_seg.shape(0);
    if (ro.ndim() != 1 || ro.shape(0) != n_basis) {
        throw std::runtime_error("row_of must be 1-D with one entry per basis");
    }
    if (Z.ndim() != 2) {
        throw std::runtime_error("Z must be 2-D");
    }
    const py::ssize_t n_rows = Z.shape(0);
    auto rv = ro.unchecked<1>();
    auto mv = m_idx.unchecked<1>();
    std::vector<char> seen((size_t)n_rows, 0);
    for (py::ssize_t i = 0; i < m_idx.shape(0); i++) {
        const int64_t m = mv(i);
        if (m < 0 || m >= n_basis) {
            throw std::runtime_error("m_idx holds a basis outside support_seg");
        }
        const int64_t r = rv(m);
        if (r < 0 || r >= n_rows) {
            throw std::runtime_error(
                "row_of maps a basis this window writes outside the compact Z "
                "(a row the caller did not ask for, or a map sized for another Z)");
        }
        if (seen[(size_t)r]) {
            throw std::runtime_error("row_of maps two written bases to one Z row");
        }
        seen[(size_t)r] = 1;
    }
    return ro;
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
    uintptr_t cancel_flag = 0,
    py::object row_of = py::none(),
    bool reference = false
) {
    if (!row_of.is_none()) {
        RowOf1132 ro = checked_row_of_1132(row_of, m_idx, support_seg, Z);
        const int64_t *rp = ro.data();
        switch ((int)support_seg.shape(1) - 1) {
            case 1:
                assemble_Z_bspline_windowed_kernel<1, false, true>(
                    J_chunk, support_seg, polys, tangents, m_idx, n_idx,
                    i0, i1, j0, j1, omega, eps_, mu_, Z, cancel_flag, 0.0, 0.0, rp, reference);
                return;
            case 2:
                assemble_Z_bspline_windowed_kernel<2, false, true>(
                    J_chunk, support_seg, polys, tangents, m_idx, n_idx,
                    i0, i1, j0, j1, omega, eps_, mu_, Z, cancel_flag, 0.0, 0.0, rp, reference);
                return;
            case 3:
                assemble_Z_bspline_windowed_kernel<3, false, true>(
                    J_chunk, support_seg, polys, tangents, m_idx, n_idx,
                    i0, i1, j0, j1, omega, eps_, mu_, Z, cancel_flag, 0.0, 0.0, rp, reference);
                return;
            default:
                throw std::runtime_error("assemble_Z_bspline_windowed: max_d must be 1, 2 or 3");
        }
    }
    switch ((int)support_seg.shape(1) - 1) {
        case 1:
            assemble_Z_bspline_windowed_kernel<1, false>(
                J_chunk, support_seg, polys, tangents, m_idx, n_idx,
                i0, i1, j0, j1, omega, eps_, mu_, Z, cancel_flag, 0.0, 0.0, nullptr, reference);
            return;
        case 2:
            assemble_Z_bspline_windowed_kernel<2, false>(
                J_chunk, support_seg, polys, tangents, m_idx, n_idx,
                i0, i1, j0, j1, omega, eps_, mu_, Z, cancel_flag, 0.0, 0.0, nullptr, reference);
            return;
        case 3:
            assemble_Z_bspline_windowed_kernel<3, false>(
                J_chunk, support_seg, polys, tangents, m_idx, n_idx,
                i0, i1, j0, j1, omega, eps_, mu_, Z, cancel_flag, 0.0, 0.0, nullptr, reference);
            return;
        default:
            throw std::runtime_error(
                "assemble_Z_bspline_windowed: max_d must be 1, 2 or 3");
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
template<int D, bool COMPLEX_EPS, bool ROW_MAP = false>
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
    uintptr_t cancel_flag = 0,
    double c_re = 0.0,
    double c_im = 0.0,
    // momwire#1132: with ROW_MAP, Z is ROW-COMPACT -- (n_rows, n_basis) --
    // and basis row m lands in Z row row_of[m]. The wrapper validates
    // the map against m_idx before the GIL is released. Without it this
    // pointer is never read and the instantiation is the shipped one.
    const int64_t *row_of = nullptr,
    // true runs the per-entry loop below even where the lane kernel is
    // compiled -- the reference it is gated against to the bit.
    bool reference = false
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
    if (ROW_MAP) {
        if (Z.shape(1) != (long)n_basis) {
            throw std::runtime_error("a row-compact Z must be (n_rows, n_basis)");
        }
    } else if (Z.shape(0) != (long)n_basis || Z.shape(1) != (long)n_basis) {
        throw std::runtime_error("Z.shape must be (n_basis, n_basis)");
    }
    auto z_view = Z.mutable_unchecked<2>();

    py::gil_scoped_release release;

    const double omega_mu = omega * mu_;
    // With COMPLEX_EPS the real eps_ is unused (the wrapper passes 1.0 so the
    // divide below is finite) and c = 1/(j*omega*eps~) carries the medium.
    const double inv_omega_eps = 1.0 / (omega * eps_);
    size_t n_m = (size_t)m_idx.shape(0);
    size_t n_n = (size_t)n_idx.shape(0);

    MW_CANCEL_SETUP(cancel_flag);
#if MW_WINDOWED_LANES_1290
    if (!reference) {
        // The lane kernel: the unweighted twin's (momwire#1290) shape --
        // L entries (m, n..n+L-1) a step, (re, im)-packed, over TM x TN
        // tiles of (mi, ni) -- carrying the per-pair complex weights.
        //
        // Every entry's arithmetic is the reference loop's, operation for
        // operation. That loop has no explicit fma and the build does not
        // contract (momwire#1194), so here too every product and sum is its
        // own rounded op, never a vfmadd: iA += prod * J and
        // iPhi += (p q prod) * J' as multiply-then-add in the same (p, q)
        // order; zA += wa * iA with the complex product as the compiler
        // expands std::complex's -- re = ac - bd, im = ad + bc, four rounded
        // products -- which is exactly mul, mul, addsub on (re, im)-packed
        // lanes; the same (a, b) order; the same scalar combine and scale
        // per entry. A wing the reference skips (sn outside [j0, j1)) is
        // computed for that lane and blended away, never added. So each Z
        // entry is the reference's to the bit on finite inputs. (A product
        // whose parts are BOTH NaN is where the compiler's complex multiply
        // calls __muldc3 and this kernel does not; no finite fill reaches it.)
        constexpr int V = 4;         // ymm registers per quantity, two entries' (re, im) each
        constexpr size_t L = 2 * V;  // entries per lane group
        constexpr size_t K = 2 * L;
        constexpr size_t TM = 64;  // tile rows (mi)
        constexpr size_t TN = 16;  // tile columns (ni), a multiple of L
        constexpr int NP = NM * NM;
        const size_t n_grp = (n_n + L - 1) / L;
        const size_t n_pad = n_grp * L;
        const size_t lane_stride = 2 * n_pad;
        // Column side, once per call, as the unweighted twin builds it (the
        // window-relative J offset in doubles is also the weight windows'
        // offset within a row: they share J's trailing axes).
        std::vector<int64_t> live2((size_t)NM * lane_stride, 0);
        std::vector<int64_t> loff((size_t)NM * n_pad, 0);
        std::vector<double> pn2((size_t)NM * NM * lane_stride, 0.0);
        std::vector<int64_t> goff((size_t)NM * n_grp, -1);
        std::vector<char> gcov((size_t)NM * n_grp, 0);
        for (size_t ni = 0; ni < n_n; ni++) {
            const int64_t n = ni_view(ni);
            for (int b = 0; b < NM; b++) {
                const int64_t sn = ss_view(n, b);
                if (sn >= j0 && sn < j1) {
                    live2[(size_t)b * lane_stride + 2 * ni] = -1;
                    live2[(size_t)b * lane_stride + 2 * ni + 1] = -1;
                    loff[(size_t)b * n_pad + ni] = 2 * (sn - j0);
                }
                for (int q = 0; q < NM; q++) {
                    const double v = p_view(n, b, q);
                    pn2[((size_t)b * NM + q) * lane_stride + 2 * ni] = v;
                    pn2[((size_t)b * NM + q) * lane_stride + 2 * ni + 1] = v;
                }
            }
        }
        for (int b = 0; b < NM; b++) {
            for (size_t gi = 0; gi < n_grp; gi++) {
                const size_t g = gi * L;
                bool all = true, none = true, contig = true;
                for (size_t l = 0; l < L; l++) {
                    const bool on = live2[(size_t)b * lane_stride + 2 * (g + l)] != 0;
                    all = all && on;
                    none = none && !on;
                    contig = contig && on &&
                             loff[(size_t)b * n_pad + g + l] == loff[(size_t)b * n_pad + g] + 2 * (int64_t)l;
                }
                gcov[(size_t)b * n_grp + gi] = all ? 1 : (none ? 2 : 0);
                if (contig) goff[(size_t)b * n_grp + gi] = loff[(size_t)b * n_pad + g];
            }
        }
        const double *Jd = reinterpret_cast<const double *>(J_chunk.data());
        const double *WAd = reinterpret_cast<const double *>(wA_win.data());
        const double *WPd = reinterpret_cast<const double *>(wPhi_win.data());
        const size_t J_row = 2 * (size_t)(j1 - j0);
        const size_t J_plane = (size_t)(i1 - i0) * J_row;
        size_t poff_j[NP], poff_s[NP];
        for (int pq = 0; pq < NP; pq++) {
            poff_j[pq] = (size_t)pq * J_plane;
            poff_s[pq] = (size_t)pq * K;
        }
        const size_t n_tm = (n_m + TM - 1) / TM;
        const size_t n_tn = (n_pad + TN - 1) / TN;

        MW_OMP_PARALLEL_FOR_COLLAPSE2
        for (size_t tm = 0; tm < n_tm; tm++) {
            for (size_t tn = 0; tn < n_tn; tn++) {
                MW_CANCEL_POLL();
                const size_t mi_end = std::min(n_m, (tm + 1) * TM);
                const size_t g_end = std::min(n_pad, (tn + 1) * TN);
                double jst[NP * K], wast[K], wpst[K];
                for (size_t mi = tm * TM; mi < mi_end; mi++) {
                    const int64_t m = mi_view(mi);
                    for (size_t g = tn * TN; g < g_end; g += L) {
                        const size_t gi = g / L;
                        __m256d zA[V], zP[V];
                        for (int v = 0; v < V; v++) {
                            zA[v] = _mm256_setzero_pd();
                            zP[v] = _mm256_setzero_pd();
                        }
                        for (int a = 0; a < NM; a++) {
                            const int64_t sm = ss_view(m, a);
                            if (sm < i0 || sm >= i1) continue;
                            double pm[NM];
                            for (int p = 0; p < NM; p++) pm[p] = p_view(m, a, p);
                            const size_t row = (size_t)(sm - i0) * J_row;
                            const double *Jrow = Jd + row;
                            for (int b = 0; b < NM; b++) {
                                const char cov = gcov[(size_t)b * n_grp + gi];
                                if (cov == 2) continue;
                                const int64_t go = goff[(size_t)b * n_grp + gi];
                                const double *base, *wab, *wpb;
                                const size_t *po;
                                if (go >= 0) {
                                    base = Jrow + go;
                                    wab = WAd + row + go;
                                    wpb = WPd + row + go;
                                    po = poff_j;
                                } else {
                                    // Stage the lanes' J and weights contiguously;
                                    // a dead lane reads column 0 and is masked.
                                    const int64_t *lo = &loff[(size_t)b * n_pad + g];
                                    for (int pq = 0; pq < NP; pq++)
                                        for (size_t l = 0; l < L; l++) {
                                            jst[pq * K + 2 * l] = Jrow[poff_j[pq] + lo[l]];
                                            jst[pq * K + 2 * l + 1] = Jrow[poff_j[pq] + lo[l] + 1];
                                        }
                                    for (size_t l = 0; l < L; l++) {
                                        wast[2 * l] = WAd[row + lo[l]];
                                        wast[2 * l + 1] = WAd[row + lo[l] + 1];
                                        wpst[2 * l] = WPd[row + lo[l]];
                                        wpst[2 * l + 1] = WPd[row + lo[l] + 1];
                                    }
                                    base = jst;
                                    wab = wast;
                                    wpb = wpst;
                                    po = poff_s;
                                }
                                __m256d iA[V], iP[V];
                                for (int v = 0; v < V; v++) {
                                    iA[v] = _mm256_setzero_pd();
                                    iP[v] = _mm256_setzero_pd();
                                }
                                for (int p = 0; p < NM; p++) {
                                    const __m256d mp = _mm256_set1_pd(pm[p]);
                                    for (int q = 0; q < NM; q++) {
                                        const double *pnq =
                                            &pn2[((size_t)b * NM + q) * lane_stride + 2 * g];
                                        const double *jp = base + po[p * NM + q];
                                        __m256d pr[V];
                                        for (int v = 0; v < V; v++) {
                                            pr[v] = _mm256_mul_pd(mp, _mm256_loadu_pd(pnq + 4 * v));
                                            iA[v] = _mm256_add_pd(
                                                iA[v], _mm256_mul_pd(pr[v], _mm256_loadu_pd(jp + 4 * v)));
                                        }
                                        if (p >= 1 && q >= 1) {
                                            const __m256d s = _mm256_set1_pd((double)(p * q));
                                            const double *jl = base + po[(p - 1) * NM + (q - 1)];
                                            for (int v = 0; v < V; v++)
                                                iP[v] = _mm256_add_pd(
                                                    iP[v], _mm256_mul_pd(_mm256_mul_pd(s, pr[v]),
                                                                         _mm256_loadu_pd(jl + 4 * v)));
                                        }
                                    }
                                }
                                for (int v = 0; v < V; v++) {
                                    // (wr, wi) x (xr, xi): [wr xr - wi xi, wr xi + wi xr].
                                    const __m256d wa = _mm256_loadu_pd(wab + 4 * v);
                                    const __m256d wp = _mm256_loadu_pd(wpb + 4 * v);
                                    const __m256d cA = _mm256_addsub_pd(
                                        _mm256_mul_pd(_mm256_movedup_pd(wa), iA[v]),
                                        _mm256_mul_pd(_mm256_permute_pd(wa, 0xF),
                                                      _mm256_permute_pd(iA[v], 0x5)));
                                    const __m256d cP = _mm256_addsub_pd(
                                        _mm256_mul_pd(_mm256_movedup_pd(wp), iP[v]),
                                        _mm256_mul_pd(_mm256_permute_pd(wp, 0xF),
                                                      _mm256_permute_pd(iP[v], 0x5)));
                                    const __m256d nA = _mm256_add_pd(zA[v], cA);
                                    const __m256d nP = _mm256_add_pd(zP[v], cP);
                                    if (cov == 1) {
                                        zA[v] = nA;
                                        zP[v] = nP;
                                    } else {
                                        const __m256d on = _mm256_loadu_pd(reinterpret_cast<const double *>(
                                            &live2[(size_t)b * lane_stride + 2 * g + 4 * v]));
                                        zA[v] = _mm256_blendv_pd(zA[v], nA, on);
                                        zP[v] = _mm256_blendv_pd(zP[v], nP, on);
                                    }
                                }
                            }
                        }
                        double zAs[K], zPs[K];
                        for (int v = 0; v < V; v++) {
                            _mm256_storeu_pd(zAs + 4 * v, zA[v]);
                            _mm256_storeu_pd(zPs + 4 * v, zP[v]);
                        }
                        for (size_t l = 0; l < L && g + l < n_n; l++) {
                            const int64_t n = ni_view(g + l);
                            const std::complex<double> zAc(zAs[2 * l], zAs[2 * l + 1]);
                            const std::complex<double> zPhi(zPs[2 * l], zPs[2 * l + 1]);
                            std::complex<double> Zc;
                            if (COMPLEX_EPS) {
                                Zc = std::complex<double>(
                                    -omega_mu * zAc.imag()
                                        + (c_re * zPhi.real() - c_im * zPhi.imag()),
                                    omega_mu * zAc.real()
                                        + (c_re * zPhi.imag() + c_im * zPhi.real()));
                            } else {
                                Zc = std::complex<double>(
                                    -omega_mu * zAc.imag() + zPhi.imag() * inv_omega_eps,
                                    omega_mu * zAc.real() - zPhi.real() * inv_omega_eps);
                            }
                            std::complex<double> add = scale * Zc;
                            z_view(ROW_MAP ? row_of[m] : m, n) += add;
                        }
                    }
                }
            }
        }
        MW_THROW_IF_ABORTED();
        return;
    }
#else
    (void)reference;
#endif
    MW_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t mi = 0; mi < n_m; mi++) {
        for (size_t ni = 0; ni < n_n; ni++) {
            MW_CANCEL_POLL();
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
            std::complex<double> Zc;
            if (COMPLEX_EPS) {
                // c = 1/(j*omega*eps~) complex (momwire#915).
                Zc = std::complex<double>(
                    -omega_mu * zA.imag()
                        + (c_re * zPhi.real() - c_im * zPhi.imag()),
                    omega_mu * zA.real()
                        + (c_re * zPhi.imag() + c_im * zPhi.real()));
            } else {
                Zc = std::complex<double>(
                    -omega_mu * zA.imag() + zPhi.imag() * inv_omega_eps,
                    omega_mu * zA.real() - zPhi.real() * inv_omega_eps);
            }
            std::complex<double> add = scale * Zc;
            z_view(ROW_MAP ? row_of[m] : m, n) += add;
        }
    }

    MW_THROW_IF_ABORTED();
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
    uintptr_t cancel_flag = 0,
    py::object row_of = py::none(),
    bool reference = false
) {
    if (!row_of.is_none()) {
        RowOf1132 ro = checked_row_of_1132(row_of, m_idx, support_seg, Z);
        const int64_t *rp = ro.data();
        switch ((int)support_seg.shape(1) - 1) {
            case 1:
                assemble_Z_bspline_weighted_windowed_kernel<1, false, true>(
                    J_chunk, support_seg, polys, wA_win, wPhi_win, m_idx, n_idx,
                    i0, i1, j0, j1, omega, eps_, mu_, scale, Z, cancel_flag, 0.0, 0.0, rp, reference);
                return;
            case 2:
                assemble_Z_bspline_weighted_windowed_kernel<2, false, true>(
                    J_chunk, support_seg, polys, wA_win, wPhi_win, m_idx, n_idx,
                    i0, i1, j0, j1, omega, eps_, mu_, scale, Z, cancel_flag, 0.0, 0.0, rp, reference);
                return;
            case 3:
                assemble_Z_bspline_weighted_windowed_kernel<3, false, true>(
                    J_chunk, support_seg, polys, wA_win, wPhi_win, m_idx, n_idx,
                    i0, i1, j0, j1, omega, eps_, mu_, scale, Z, cancel_flag, 0.0, 0.0, rp, reference);
                return;
            default:
                throw std::runtime_error("assemble_Z_bspline_weighted_windowed: max_d must be 1, 2 or 3");
        }
    }
    switch ((int)support_seg.shape(1) - 1) {
        case 1:
            assemble_Z_bspline_weighted_windowed_kernel<1, false>(
                J_chunk, support_seg, polys, wA_win, wPhi_win, m_idx, n_idx,
                i0, i1, j0, j1, omega, eps_, mu_, scale, Z, cancel_flag, 0.0, 0.0,
                nullptr, reference);
            return;
        case 2:
            assemble_Z_bspline_weighted_windowed_kernel<2, false>(
                J_chunk, support_seg, polys, wA_win, wPhi_win, m_idx, n_idx,
                i0, i1, j0, j1, omega, eps_, mu_, scale, Z, cancel_flag, 0.0, 0.0,
                nullptr, reference);
            return;
        case 3:
            assemble_Z_bspline_weighted_windowed_kernel<3, false>(
                J_chunk, support_seg, polys, wA_win, wPhi_win, m_idx, n_idx,
                i0, i1, j0, j1, omega, eps_, mu_, scale, Z, cancel_flag, 0.0, 0.0,
                nullptr, reference);
            return;
        default:
            throw std::runtime_error(
                "assemble_Z_bspline_weighted_windowed: max_d must be 1, 2 or 3");
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

    MW_OMP_PARALLEL_FOR_COLLAPSE2
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
            return assemble_Z_bspline_kernel<1, false>(J, support_seg, polys, td_all, omega, eps_, mu_, cancel_flag);
        case 2:
            return assemble_Z_bspline_kernel<2, false>(J, support_seg, polys, td_all, omega, eps_, mu_, cancel_flag);
        case 3:
            return assemble_Z_bspline_kernel<3, false>(J, support_seg, polys, td_all, omega, eps_, mu_, cancel_flag);
        default:
            throw std::runtime_error(
                "assemble_Z_bspline: max_d must be 1, 2 or 3");
    }
}


// Weighted variant of assemble_Z_bspline_kernel for the reflection-
// coefficient finite ground (BSplineSolver ground_eps): the A term takes a
// COMPLEX per-segment-pair weight table wA_all (the Fresnel dyad tangent
// table, replacing the real tangent-dot), and the Φ term — unweighted in
// the PEC kernel — takes its own complex per-pair image-charge table
// wPhi_all. Same loop structure, same J tensor, one pass; the PEC kernel is
// the wA = t·Mt (real), wPhi = 1 special case.
template<int D, bool COMPLEX_EPS>
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
    uintptr_t cancel_flag = 0,
    double c_re = 0.0,
    double c_im = 0.0
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
    // With COMPLEX_EPS the real eps_ is unused (the wrapper passes 1.0 so the
    // divide below is finite) and c = 1/(j*omega*eps~) carries the medium.
    const double inv_omega_eps = 1.0 / (omega * eps_);

    MW_CANCEL_SETUP(cancel_flag);
    MW_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t m = 0; m < n_basis; m++) {
        for (size_t n = 0; n < n_basis; n++) {
            MW_CANCEL_POLL();
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

            double Zre, Zim;
            if (COMPLEX_EPS) {
                // Z = j*omega*mu*zA + c*zPhi with c = 1/(j*omega*eps~) COMPLEX
                // (momwire#910): the medium's eps~ enters this kernel only
                // here, so the buried assembly is the real-eps loop above
                // with a complex scalar in the combine.
                Zre = -omega_mu * zA_im + (c_re * zPhi_re - c_im * zPhi_im);
                Zim = omega_mu * zA_re + (c_re * zPhi_im + c_im * zPhi_re);
            } else {
                Zre = -omega_mu * zA_im + zPhi_im * inv_omega_eps;
                Zim = omega_mu * zA_re - zPhi_re * inv_omega_eps;
            }
            z_view(m, n) = std::complex<double>(Zre, Zim);
        }
    }

    MW_THROW_IF_ABORTED();
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
            return assemble_Z_bspline_weighted_kernel<1, false>(J, support_seg, polys, wA_all, wPhi_all, omega, eps_, mu_, cancel_flag);
        case 2:
            return assemble_Z_bspline_weighted_kernel<2, false>(J, support_seg, polys, wA_all, wPhi_all, omega, eps_, mu_, cancel_flag);
        case 3:
            return assemble_Z_bspline_weighted_kernel<3, false>(J, support_seg, polys, wA_all, wPhi_all, omega, eps_, mu_, cancel_flag);
        default:
            throw std::runtime_error(
                "assemble_Z_bspline_weighted: max_d must be 1, 2 or 3");
    }
}


// In-medium twins of the two assemblers above (momwire#910). The buried fill
// divides its charge term by eps~_m = eps0 * eps~ with eps~ COMPLEX, and the
// real-eps entries' `double eps` would truncate it — the same hazard class as
// `float(k)` on the pair kernels (momwire#553 U1) — so, as there, the complex
// case is a separate ENTRY POINT taking std::complex<double>, not a widened
// argument. The kernel body is the real-eps loop; only the final combine
// differs, and the real instantiation is textually unchanged.
static py::array_t<std::complex<double>>
assemble_Z_bspline_cplx_eps(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<double, py::array::c_style | py::array::forcecast> td_all,
    double omega,
    std::complex<double> eps_,
    double mu_,
    int max_d,
    uintptr_t cancel_flag = 0
) {
    const std::complex<double> c = 1.0 / (std::complex<double>(0.0, omega) * eps_);
    switch (max_d) {
        case 1:
            return assemble_Z_bspline_kernel<1, true>(
                J, support_seg, polys, td_all, omega, 1.0, mu_, cancel_flag,
                c.real(), c.imag());
        case 2:
            return assemble_Z_bspline_kernel<2, true>(
                J, support_seg, polys, td_all, omega, 1.0, mu_, cancel_flag,
                c.real(), c.imag());
        case 3:
            return assemble_Z_bspline_kernel<3, true>(
                J, support_seg, polys, td_all, omega, 1.0, mu_, cancel_flag,
                c.real(), c.imag());
        default:
            throw std::runtime_error(
                "assemble_Z_bspline_cplx_eps: max_d must be 1, 2 or 3");
    }
}

static py::array_t<std::complex<double>>
assemble_Z_bspline_weighted_cplx_eps(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> wA_all,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> wPhi_all,
    double omega,
    std::complex<double> eps_,
    double mu_,
    int max_d,
    uintptr_t cancel_flag = 0
) {
    const std::complex<double> c = 1.0 / (std::complex<double>(0.0, omega) * eps_);
    switch (max_d) {
        case 1:
            return assemble_Z_bspline_weighted_kernel<1, true>(
                J, support_seg, polys, wA_all, wPhi_all, omega, 1.0, mu_,
                cancel_flag, c.real(), c.imag());
        case 2:
            return assemble_Z_bspline_weighted_kernel<2, true>(
                J, support_seg, polys, wA_all, wPhi_all, omega, 1.0, mu_,
                cancel_flag, c.real(), c.imag());
        case 3:
            return assemble_Z_bspline_weighted_kernel<3, true>(
                J, support_seg, polys, wA_all, wPhi_all, omega, 1.0, mu_,
                cancel_flag, c.real(), c.imag());
        default:
            throw std::runtime_error(
                "assemble_Z_bspline_weighted_cplx_eps: max_d must be 1, 2 or 3");
    }
}


// In-medium twins of the two WINDOWED assemblers (momwire#915): the chunked
// fill+assemble route for a buried deck whose dense tensor does not fit the
// budget. Same discipline as #910's twins — a separate entry point taking
// std::complex<double> eps, the real instantiations textually unchanged.
static void
assemble_Z_bspline_windowed_cplx_eps(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J_chunk,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<double, py::array::c_style | py::array::forcecast> tangents,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> m_idx,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> n_idx,
    int64_t i0, int64_t i1, int64_t j0, int64_t j1,
    double omega,
    std::complex<double> eps_,
    double mu_,
    py::array_t<std::complex<double>> Z,
    uintptr_t cancel_flag = 0,
    py::object row_of = py::none(),
    bool reference = false
) {
    const std::complex<double> c = 1.0 / (std::complex<double>(0.0, omega) * eps_);
    if (!row_of.is_none()) {
        RowOf1132 ro = checked_row_of_1132(row_of, m_idx, support_seg, Z);
        const int64_t *rp = ro.data();
        switch ((int)support_seg.shape(1) - 1) {
            case 1:
                assemble_Z_bspline_windowed_kernel<1, true, true>(
                    J_chunk, support_seg, polys, tangents, m_idx, n_idx,
                    i0, i1, j0, j1, omega, 1.0, mu_, Z, cancel_flag, c.real(), c.imag(), rp, reference);
                return;
            case 2:
                assemble_Z_bspline_windowed_kernel<2, true, true>(
                    J_chunk, support_seg, polys, tangents, m_idx, n_idx,
                    i0, i1, j0, j1, omega, 1.0, mu_, Z, cancel_flag, c.real(), c.imag(), rp, reference);
                return;
            case 3:
                assemble_Z_bspline_windowed_kernel<3, true, true>(
                    J_chunk, support_seg, polys, tangents, m_idx, n_idx,
                    i0, i1, j0, j1, omega, 1.0, mu_, Z, cancel_flag, c.real(), c.imag(), rp, reference);
                return;
            default:
                throw std::runtime_error("assemble_Z_bspline_windowed_cplx_eps: max_d must be 1, 2 or 3");
        }
    }
    switch ((int)support_seg.shape(1) - 1) {
        case 1:
            assemble_Z_bspline_windowed_kernel<1, true>(
                J_chunk, support_seg, polys, tangents, m_idx, n_idx,
                i0, i1, j0, j1, omega, 1.0, mu_, Z, cancel_flag, c.real(), c.imag(), nullptr, reference);
            return;
        case 2:
            assemble_Z_bspline_windowed_kernel<2, true>(
                J_chunk, support_seg, polys, tangents, m_idx, n_idx,
                i0, i1, j0, j1, omega, 1.0, mu_, Z, cancel_flag, c.real(), c.imag(), nullptr, reference);
            return;
        case 3:
            assemble_Z_bspline_windowed_kernel<3, true>(
                J_chunk, support_seg, polys, tangents, m_idx, n_idx,
                i0, i1, j0, j1, omega, 1.0, mu_, Z, cancel_flag, c.real(), c.imag(), nullptr, reference);
            return;
        default:
            throw std::runtime_error(
                "assemble_Z_bspline_windowed_cplx_eps: max_d must be 1, 2 or 3");
    }
}

static void
assemble_Z_bspline_weighted_windowed_cplx_eps(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J_chunk,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> wA_win,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> wPhi_win,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> m_idx,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> n_idx,
    int64_t i0, int64_t i1, int64_t j0, int64_t j1,
    double omega,
    std::complex<double> eps_,
    double mu_,
    std::complex<double> scale,
    py::array_t<std::complex<double>> Z,
    uintptr_t cancel_flag = 0,
    py::object row_of = py::none(),
    bool reference = false
) {
    const std::complex<double> c = 1.0 / (std::complex<double>(0.0, omega) * eps_);
    if (!row_of.is_none()) {
        RowOf1132 ro = checked_row_of_1132(row_of, m_idx, support_seg, Z);
        const int64_t *rp = ro.data();
        switch ((int)support_seg.shape(1) - 1) {
            case 1:
                assemble_Z_bspline_weighted_windowed_kernel<1, true, true>(
                    J_chunk, support_seg, polys, wA_win, wPhi_win, m_idx, n_idx,
                    i0, i1, j0, j1, omega, 1.0, mu_, scale, Z, cancel_flag, c.real(), c.imag(), rp, reference);
                return;
            case 2:
                assemble_Z_bspline_weighted_windowed_kernel<2, true, true>(
                    J_chunk, support_seg, polys, wA_win, wPhi_win, m_idx, n_idx,
                    i0, i1, j0, j1, omega, 1.0, mu_, scale, Z, cancel_flag, c.real(), c.imag(), rp, reference);
                return;
            case 3:
                assemble_Z_bspline_weighted_windowed_kernel<3, true, true>(
                    J_chunk, support_seg, polys, wA_win, wPhi_win, m_idx, n_idx,
                    i0, i1, j0, j1, omega, 1.0, mu_, scale, Z, cancel_flag, c.real(), c.imag(), rp, reference);
                return;
            default:
                throw std::runtime_error("assemble_Z_bspline_weighted_windowed_cplx_eps: max_d must be 1, 2 or 3");
        }
    }
    switch ((int)support_seg.shape(1) - 1) {
        case 1:
            assemble_Z_bspline_weighted_windowed_kernel<1, true>(
                J_chunk, support_seg, polys, wA_win, wPhi_win, m_idx, n_idx,
                i0, i1, j0, j1, omega, 1.0, mu_, scale, Z, cancel_flag,
                c.real(), c.imag(), nullptr, reference);
            return;
        case 2:
            assemble_Z_bspline_weighted_windowed_kernel<2, true>(
                J_chunk, support_seg, polys, wA_win, wPhi_win, m_idx, n_idx,
                i0, i1, j0, j1, omega, 1.0, mu_, scale, Z, cancel_flag,
                c.real(), c.imag(), nullptr, reference);
            return;
        case 3:
            assemble_Z_bspline_weighted_windowed_kernel<3, true>(
                J_chunk, support_seg, polys, wA_win, wPhi_win, m_idx, n_idx,
                i0, i1, j0, j1, omega, 1.0, mu_, scale, Z, cancel_flag,
                c.real(), c.imag(), nullptr, reference);
            return;
        default:
            throw std::runtime_error(
                "assemble_Z_bspline_weighted_windowed_cplx_eps: max_d must be 1, 2 or 3");
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
        case 3:
            return assemble_Z_bspline_swept_kernel<3>(
                J, support_seg, polys, tangents_row, tangents_col,
                omega_array, eps_, mu_);
        default:
            throw std::runtime_error(
                "assemble_Z_bspline_swept: max_d must be 1, 2 or 3");
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

    MW_CANCEL_SETUP(cancel_flag);
    MW_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t m = 0; m < nI; m++) {
        for (size_t n = 0; n < nJ; n++) {
            MW_CANCEL_POLL();
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
                        // TILED OVER qr (momwire#762). `mm` and `tt` rather
                        // than the usual m/t: this scope already has m and n
                        // as the block's segment-pair indices.
                        alignas(32) double R[BSPLINE_QR_TILE];
                        alignas(32) double G_re[BSPLINE_QR_TILE], G_im[BSPLINE_QR_TILE];
                        alignas(32) double wuwu[(NM*NM) * BSPLINE_QR_TILE];
                        const size_t n_pairs = n_qp * n_qp;
                        double acc_re[NM*NM], acc_im[NM*NM];
                        for (int pP = 0; pP < NM*NM; pP++) { acc_re[pP] = 0.0; acc_im[pP] = 0.0; }

                        for (size_t base = 0; base < n_pairs; base += BSPLINE_QR_TILE) {
                            const size_t mm = (n_pairs - base < BSPLINE_QR_TILE)
                                                  ? (n_pairs - base) : BSPLINE_QR_TILE;
                            size_t q = base / n_qp;
                            size_t r = base % n_qp;
                            for (size_t tt = 0; tt < mm; tt++) {
                                double dx = pi[q*3+0] - pj[r*3+0];
                                double dy = pi[q*3+1] - pj[r*3+1];
                                double dz = pi[q*3+2] - pj[r*3+2];
                                R[tt] = std::sqrt(dx*dx+dy*dy+dz*dz+a_squared);

                                double wi = glw(q) * Li, ui = glt(q) * Li;
                                double wj = glw(r) * Lj, uj = glt(r) * Lj;
                                double uip[NM], ujp[NM];
                                uip[0] = 1.0; ujp[0] = 1.0;
                                for (int e = 1; e < NM; e++) {
                                    uip[e] = uip[e-1]*ui;
                                    ujp[e] = ujp[e-1]*uj;
                                }
                                double wij = wi*wj;
                                for (int p = 0; p < NM; p++)
                                    for (int P = 0; P < NM; P++)
                                        wuwu[(p*NM+P)*mm + tt] = wij*uip[p]*ujp[P];

                                if (++r == n_qp) { r = 0; ++q; }
                            }
                            MW_OMP_SIMD()
                            for (size_t tt = 0; tt < mm; tt++) {
                                double inv = inv_4pi / R[tt];
                                double ph = -k * R[tt];
                                G_re[tt] = std::cos(ph) * inv;
                                G_im[tt] = std::sin(ph) * inv;
                            }
                            for (int pP = 0; pP < NM*NM; pP++) {
                                double sr_ = acc_re[pP], si_ = acc_im[pP];
                                const double *w_row = &wuwu[pP * mm];
                                // No `omp simd reduction` here (momwire#781): the clause LICENSES
                                // reassociation, so the reduction tree follows whatever
                                // vectorization factor the compiler picks per FUNCTION -- and the
                                // reduced and EK kernels differ in register pressure. That made
                                // their all-ineligible outputs disagree by 1 ulp on arm64 while
                                // matching on x86-64, breaking the exact-reduction gates that
                                // momwire#270 U2 relies on. Measured single-threaded (pinned,
                                // passive wait, min of 5, 3 alternating rounds), the clause is
                                // worth -0.2%/+0.4% on the bspline fills -- i.e. nothing. It IS
                                // worth ~4.6% in _accel_razor.cpp, which keeps its clause and has
                                // no cross-kernel equality gate to protect.
                                for (size_t tt = 0; tt < mm; tt++) {
                                    sr_ += w_row[tt]*G_re[tt];
                                    si_ += w_row[tt]*G_im[tt];
                                }
                                acc_re[pP] = sr_;
                                acc_im[pP] = si_;
                            }
                        }
                        for (int pP = 0; pP < NM*NM; pP++) {
                            Jc[pP/NM][pP%NM] =
                                std::complex<double>(acc_re[pP], acc_im[pP]);
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

    MW_THROW_IF_ABORTED();
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
        case 3:
            return bspline_assemble_offedge_block_kernel<3, false>(
                supp_I, polys_I, segl_I, segr_I, tan_I, supp_J, polys_J,
                segl_J, segr_J, tan_J, a_squared, k, omega, eps_, mu_, gl_t, gl_w,
                cancel_flag);
        default:
            throw std::runtime_error(
                "bspline_assemble_offedge_block: max_d must be 1, 2 or 3");
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
        case 3:
            return bspline_assemble_offedge_block_kernel<3, true>(
                supp_I, polys_I, segl_I, segr_I, tan_I, supp_J, polys_J,
                segl_J, segr_J, tan_J, a_squared, k, omega, eps_, mu_, gl_t, gl_w,
                cancel_flag, eps_t, phi_c0, phi_c1);
        default:
            throw std::runtime_error(
                "bspline_assemble_offedge_block_refl: max_d must be 1, 2 or 3");
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
//
// momwire#1362 gave this twin what the dense EK kernel got in #1363/#1365:
// the moments are computed once per SEGMENT pair into a table (phase A) and
// combined per basis pair from it (phase B), phase A runs the pre-#1362
// arithmetic per pair with stage 2 t-outer and, in the AVX2 build, four
// pairs to a vector lane (`reference=True` walks every pair alone, the gate
// the lanes are held to), and the `_tiered` entries take the pair-order
// ladder. The reduced (EK-off) assembler above is untouched.
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
    const PairOrderLadder& ladder,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_I,  // (nSegI,)
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_J,  // (nSegJ,)
    double a_ek,
    // The ladder's per-pair phase guard (momwire#920): a tier below
    // `limited_below` points serves a pair only when both its segments pass
    // (`phase_ok_*[s] != 0`). Null means every segment passes.
    const uint8_t *phase_ok_I,
    const uint8_t *phase_ok_J,
    int64_t limited_below,
    uintptr_t cancel_flag,
    bool reference,
    std::complex<double> eps_t = std::complex<double>(0.0, 0.0),
    std::complex<double> phi_c0 = std::complex<double>(0.0, 0.0),
    std::complex<double> phi_c1 = std::complex<double>(0.0, 0.0)
) {
    static constexpr int NM = D + 1;
    static constexpr int NMM = NM * NM;

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

    size_t nI = (size_t)supp_I.shape(0);
    size_t nJ = (size_t)supp_J.shape(0);
    size_t nSegI = (size_t)segl_I.shape(0);
    size_t nSegJ = (size_t)segl_J.shape(0);
    const size_t n_tiers = ladder.n_tiers();
    if (supp_I.shape(1) != NM || supp_J.shape(1) != NM) {
        throw std::runtime_error("support arrays must have shape (n, D+1)");
    }
    if ((size_t)group_I.shape(0) != nSegI || (size_t)group_J.shape(0) != nSegJ) {
        throw std::runtime_error(
            "bspline_assemble_offedge_block_kernel_ek: group_I/group_J must "
            "match the segment unions");
    }
    const int64_t *gI = group_I.data();
    const int64_t *gJ = group_J.data();

    // Per-segment lengths, then PER TIER the quadrature positions -- with
    // one tier the pre-#1362 precompute split in two: the same expressions
    // give the same doubles. Midpoints for the WEIGHTED specular geometry
    // (J side already mirrored, so midI - midJ is the obs->image ray).
    std::vector<double> lenI(nSegI), lenJ(nSegJ);
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
    }
    for (size_t s = 0; s < nSegJ; s++) {
        double dx = srJ(s,0)-slJ(s,0), dy = srJ(s,1)-slJ(s,1), dz = srJ(s,2)-slJ(s,2);
        lenJ[s] = std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    std::vector<std::vector<double>> posI(n_tiers), posJ(n_tiers);
    for (size_t tier = 0; tier < n_tiers; tier++) {
        const size_t n_qp = ladder.n_qp(tier);
        const double *gt = ladder.t_at(tier);
        posI[tier].resize(nSegI * n_qp * 3);
        posJ[tier].resize(nSegJ * n_qp * 3);
        double *pIt = posI[tier].data();
        double *pJt = posJ[tier].data();
        for (size_t s = 0; s < nSegI; s++) {
            for (size_t q = 0; q < n_qp; q++) {
                double t = gt[q];
                pIt[(s*n_qp+q)*3+0] = (1.0-t)*slI(s,0) + t*srI(s,0);
                pIt[(s*n_qp+q)*3+1] = (1.0-t)*slI(s,1) + t*srI(s,1);
                pIt[(s*n_qp+q)*3+2] = (1.0-t)*slI(s,2) + t*srI(s,2);
            }
        }
        for (size_t s = 0; s < nSegJ; s++) {
            for (size_t q = 0; q < n_qp; q++) {
                double t = gt[q];
                pJt[(s*n_qp+q)*3+0] = (1.0-t)*slJ(s,0) + t*srJ(s,0);
                pJt[(s*n_qp+q)*3+1] = (1.0-t)*slJ(s,1) + t*srJ(s,1);
                pJt[(s*n_qp+q)*3+2] = (1.0-t)*slJ(s,2) + t*srJ(s,2);
            }
        }
    }
    std::vector<double> cI, cJ;
    if (n_tiers > 1) {
        cI.resize(nSegI * 3);
        cJ.resize(nSegJ * 3);
        for (size_t s = 0; s < nSegI; s++)
            for (int c = 0; c < 3; c++) cI[s*3 + c] = 0.5 * (slI(s,c) + srI(s,c));
        for (size_t s = 0; s < nSegJ; s++)
            for (int c = 0; c < 3; c++) cJ[s*3 + c] = 0.5 * (slJ(s,c) + srJ(s,c));
    }
    // The single-k kernel's selector (centre distance over the longer
    // segment, thresholds ascending), with the phase guard answered per pair:
    // a phase-limited tier is skipped, not a stop, for a pair that fails it --
    // which is the tier list `_ladder_for_block` trims for such a pair.
    auto pair_tier = [&](size_t si, size_t sj) -> size_t {
        size_t tier = 0;
        if (n_tiers > 1) {
            const double dx = cI[si*3 + 0] - cJ[sj*3 + 0];
            const double dy = cI[si*3 + 1] - cJ[sj*3 + 1];
            const double dz = cI[si*3 + 2] - cJ[sj*3 + 2];
            const double ratio = std::sqrt(dx*dx + dy*dy + dz*dz)
                                 / std::max(lenI[si], lenJ[sj]);
            const bool ok = (phase_ok_I == nullptr || phase_ok_I[si]) &&
                            (phase_ok_J == nullptr || phase_ok_J[sj]);
            for (size_t t = 1; t < n_tiers; t++) {
                if (!(ratio >= ladder.ratio[t])) break;
                if (ok || (int64_t)ladder.n_qp(t) >= limited_below) tier = t;
            }
        }
        return tier;
    };

    py::array_t<std::complex<double>> Z({nI, nJ});
    auto z_view = Z.mutable_unchecked<2>();

    // Phase 0: release the GIL for the heavy compute region below.
    py::gil_scoped_release release;

    const double inv_4pi = 1.0 / (4.0 * M_PI);
    const double omega_mu = omega * mu_;
    const double inv_omega_eps = 1.0 / (omega * eps_);
    const double a2_ek = a_ek * a_ek;
    const double a4_ek = a2_ek * a2_ek;

    // momwire#1362: ONE moment tensor per SEGMENT pair. The fused walk this
    // replaced quadratured Jc for every (basis pair, wing pair), and a
    // segment pair sits under up to NM^2 of those (adjacent bases share
    // segments), so every moment was computed ~3x on an ACA row or column
    // and up to NM^2 times on a dense block. Jc is a pure function of the
    // segment pair, so phase A fills a table of them once and phase B is the
    // walk's Galerkin combine reading it: the same doubles in the same
    // order, so Z does not move.
    //
    // Phase A walks a pair with the pre-#1362 arithmetic verbatim (R, the
    // fused G loop, the EK factor loop, each moment one ascending-t chain
    // carried across qr chunks; stage 2 t-outer, which reorders work only
    // ACROSS chains), and in the AVX2 build groups four pairs into lanes the
    // way momwire#1290 does, each lane the walk's arithmetic for its pair.
    //
    // The table is tiled over observer bases so it stays bounded on a big
    // dense block: a tile's observer segments times every source segment,
    // at most ~16 MB of moments.
    const size_t tile_pairs_cap = ((size_t)16 << 20) / (2 * NMM * sizeof(double));
    const size_t TM = std::max<size_t>(
        1, tile_pairs_cap / (std::max<size_t>(nSegJ, 1) * NM));

    std::vector<int64_t> loc(nSegI, -1);
    std::vector<size_t> tseg;
    std::vector<double> tab;  // [pair][re NMM, im NMM], pair = ti * nSegJ + sj

    // One segment pair, the fused walk's moment block. Writes 2*NMM doubles.
    auto pair_walk = [&](size_t si, size_t sj, double *out) {
        alignas(32) double R[BSPLINE_QR_TILE];
        alignas(32) double G_re[BSPLINE_QR_TILE], G_im[BSPLINE_QR_TILE];
        alignas(32) double wuwu[NMM * BSPLINE_QR_TILE];
        const size_t tier = pair_tier(si, sj);
        const size_t n_qp = ladder.n_qp(tier);
        const double *gt = ladder.t_at(tier);
        const double *gw = ladder.w_at(tier);
        const double *pi = &posI[tier][si * n_qp * 3];
        const double *pj = &posJ[tier][sj * n_qp * 3];
        const double Li = lenI[si];
        const double Lj = lenJ[sj];
        const bool eligible = (gI[si] == gJ[sj]) && (gI[si] >= 0);
        const size_t n_pairs = n_qp * n_qp;
        double acc_re[NMM], acc_im[NMM];
        for (int pP = 0; pP < NMM; pP++) { acc_re[pP] = 0.0; acc_im[pP] = 0.0; }

        for (size_t base = 0; base < n_pairs; base += BSPLINE_QR_TILE) {
            const size_t mm = (n_pairs - base < BSPLINE_QR_TILE)
                                  ? (n_pairs - base) : BSPLINE_QR_TILE;
            size_t q = base / n_qp;
            size_t r = base % n_qp;
            for (size_t tt = 0; tt < mm; tt++) {
                double dx = pi[q*3+0] - pj[r*3+0];
                double dy = pi[q*3+1] - pj[r*3+1];
                double dz = pi[q*3+2] - pj[r*3+2];
                R[tt] = std::sqrt(dx*dx+dy*dy+dz*dz+a_squared);

                double wi = gw[q] * Li, ui = gt[q] * Li;
                double wj = gw[r] * Lj, uj = gt[r] * Lj;
                double uip[NM], ujp[NM];
                uip[0] = 1.0; ujp[0] = 1.0;
                for (int e = 1; e < NM; e++) {
                    uip[e] = uip[e-1]*ui;
                    ujp[e] = ujp[e-1]*uj;
                }
                double wij = wi*wj;
                for (int p = 0; p < NM; p++)
                    for (int P = 0; P < NM; P++)
                        wuwu[tt*NMM + p*NM+P] = wij*uip[p]*ujp[P];

                if (++r == n_qp) { r = 0; ++q; }
            }
            MW_OMP_SIMD()
            for (size_t tt = 0; tt < mm; tt++) {
                double inv = inv_4pi / R[tt];
                double ph = -k * R[tt];
                G_re[tt] = std::cos(ph) * inv;
                G_im[tt] = std::sin(ph) * inv;
            }
            if (eligible) {
                MW_OMP_SIMD()
                for (size_t tt = 0; tt < mm; tt++) {
                    double Rq = R[tt];
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
                    double gre = G_re[tt];
                    double gim = G_im[tt];
                    G_re[tt] = gre * facr - gim * faci;
                    G_im[tt] = gre * faci + gim * facr;
                }
            }
            // Stage 2, t outermost: each moment's chain continues from the
            // previous chunk and adds w*G in ascending t -- the order of the
            // per-pP loop it replaced. No `omp simd reduction` (momwire#781).
            for (size_t tt = 0; tt < mm; tt++) {
                const double gr = G_re[tt], gi = G_im[tt];
                const double *w_t = &wuwu[tt * NMM];
                for (int pP = 0; pP < NMM; pP++) {
                    acc_re[pP] += w_t[pP] * gr;
                    acc_im[pP] += w_t[pP] * gi;
                }
            }
        }
        for (int pP = 0; pP < NMM; pP++) {
            out[pP] = acc_re[pP];
            out[NMM + pP] = acc_im[pP];
        }
    };

    MW_CANCEL_SETUP(cancel_flag);
    for (size_t m0 = 0; m0 < nI; m0 += TM) {
        const size_t m1 = std::min(nI, m0 + TM);
        // The tile's observer segments, ascending, and their local rows.
        tseg.clear();
        for (size_t m = m0; m < m1; m++) {
            for (int a = 0; a < NM; a++) {
                const int64_t s = sI(m, a);
                if (loc[s] < 0) { loc[s] = 0; tseg.push_back((size_t)s); }
            }
        }
        std::sort(tseg.begin(), tseg.end());
        for (size_t ti = 0; ti < tseg.size(); ti++) loc[tseg[ti]] = (int64_t)ti;
        const size_t n_pair = tseg.size() * nSegJ;
        tab.resize(n_pair * 2 * NMM);
        double *tab_p = tab.data();

        // Phase A: the table.
#if MW_OFFEDGE_LANES_1290
        if (!reference) {
            using namespace offedge_lanes;
            constexpr size_t LN = W;
            const size_t n_grp = (n_pair + LN - 1) / LN;
            const vd v_a2 = set1(a_squared);
            const vd v_t1n = set1(0.25 * a4_ek);
            const vd v_t2n = set1(0.5 * a2_ek);
            const vd v_k = set1(k);
            #pragma omp parallel
            {
            // [t][lane] point t of lane l's pair; wv [t][pP][lane].
            alignas(ALIGN) double R[BSPLINE_QR_TILE * LN];
            alignas(ALIGN) double G_re[BSPLINE_QR_TILE * LN];
            alignas(ALIGN) double G_im[BSPLINE_QR_TILE * LN];
            alignas(ALIGN) double wv[BSPLINE_QR_TILE * NMM * LN];
            alignas(ALIGN) double PI[3 * 8 * LN], PJ[3 * 8 * LN];
            alignas(ALIGN) double res[2 * NMM * LN];
            #pragma omp for schedule(static)
            for (size_t g = 0; g < n_grp; g++) {
                MW_CANCEL_POLL();
                const size_t p0 = g * LN;
                size_t si[LN], sj[LN];
                bool lanes = p0 + LN <= n_pair;
                size_t tier = 0;
                if (lanes) {
                    for (size_t l = 0; l < LN; l++) {
                        si[l] = tseg[(p0 + l) / nSegJ];
                        sj[l] = (p0 + l) % nSegJ;
                    }
                    tier = pair_tier(si[0], sj[0]);
                    for (size_t l = 1; l < LN; l++) {
                        lanes = lanes && pair_tier(si[l], sj[l]) == tier;
                    }
                    const size_t nq = ladder.n_qp(tier);
                    lanes = lanes && nq % 2 == 0 && nq * nq <= BSPLINE_QR_TILE;
                }
                if (!lanes) {
                    const size_t p1 = std::min(p0 + LN, n_pair);
                    for (size_t p = p0; p < p1; p++) {
                        pair_walk(tseg[p / nSegJ], p % nSegJ, tab_p + p * 2 * NMM);
                    }
                    continue;
                }
                const size_t n_qp = ladder.n_qp(tier);
                const size_t m = n_qp * n_qp;
                const size_t m4 = m * LN;
                const double *gt = ladder.t_at(tier);
                const double *gw = ladder.w_at(tier);
                const double *pIt = posI[tier].data();
                const double *pJt = posJ[tier].data();
                // Gather both sides lane-major, [c][q][lane]: W arbitrary
                // pairs, so a column call (three source segments) fills its
                // lanes as readily as a row call.
                for (size_t c = 0; c < 3; c++) {
                    for (size_t q = 0; q < n_qp; q++) {
                        for (size_t l = 0; l < LN; l++) {
                            PI[(c * n_qp + q) * LN + l] = pIt[(si[l] * n_qp + q) * 3 + c];
                            PJ[(c * n_qp + q) * LN + l] = pJt[(sj[l] * n_qp + q) * 3 + c];
                        }
                    }
                }
                double Li4[LN], Lj4[LN];
                for (size_t l = 0; l < LN; l++) { Li4[l] = lenI[si[l]]; Lj4[l] = lenJ[sj[l]]; }
                const vd vLi = loadu(Li4);
                const vd vLj = loadu(Lj4);

                // R and the moment weights, per lane in the walk's order:
                // `dx*dx+dy*dy+dz*dz+a_squared` left to right; wi = gw*Li,
                // wj = gw*Lj, wij = wi*wj, then (wij*ui^p)*uj^P. Both sides
                // vary across lanes here, so neither is broadcast.
                vd ujp[8][NM], wjv[8];
                for (size_t r = 0; r < n_qp; r++) {
                    wjv[r] = mul(set1(gw[r]), vLj);
                    const vd uj = mul(set1(gt[r]), vLj);
                    ujp[r][0] = set1(1.0);
                    for (int e = 1; e < NM; e++) ujp[r][e] = mul(ujp[r][e-1], uj);
                }
                for (size_t q = 0; q < n_qp; q++) {
                    const vd px = load(PI + (0 * n_qp + q) * LN);
                    const vd py = load(PI + (1 * n_qp + q) * LN);
                    const vd pz = load(PI + (2 * n_qp + q) * LN);
                    const vd wi = mul(set1(gw[q]), vLi);
                    const vd ui = mul(set1(gt[q]), vLi);
                    vd uip[NM];
                    uip[0] = set1(1.0);
                    for (int e = 1; e < NM; e++) uip[e] = mul(uip[e-1], ui);
                    for (size_t r = 0; r < n_qp; r++) {
                        const size_t t = q * n_qp + r;
                        const vd dx = sub(px, load(PJ + (0 * n_qp + r) * LN));
                        const vd dy = sub(py, load(PJ + (1 * n_qp + r) * LN));
                        const vd dz = sub(pz, load(PJ + (2 * n_qp + r) * LN));
                        vd s = add(mul(dx, dx), mul(dy, dy));
                        s = add(s, mul(dz, dz));
                        s = add(s, v_a2);
                        store(R + t * LN, sqrt(s));
                        const vd wij = mul(wi, wjv[r]);
                        for (int p = 0; p < NM; p++) {
                            const vd x = mul(wij, uip[p]);
                            for (int P = 0; P < NM; P++) {
                                store(wv + (t * NMM + p * NM + P) * LN, mul(x, ujp[r][P]));
                            }
                        }
                    }
                }

                // Stage 1, the walk's fused loop body over all the lanes'
                // points (same body, so the same libmvec entries; the even
                // order leaves no scalar remainder on either route).
                MW_OMP_SIMD()
                for (size_t tt = 0; tt < m4; tt++) {
                    double inv = inv_4pi / R[tt];
                    double ph = -k * R[tt];
                    G_re[tt] = std::cos(ph) * inv;
                    G_im[tt] = std::sin(ph) * inv;
                }

                // EK: per lane, `ek_factor`; this kernel's walk has no floor.
                bool e[LN];
                bool any_e = false;
                for (size_t l = 0; l < LN; l++) {
                    e[l] = gI[si[l]] == gJ[sj[l]] && gI[si[l]] >= 0;
                    any_e = any_e || e[l];
                }
                if (any_e) {
                    const vm emask = mask_from(e);
                    for (size_t t = 0; t < m; t++) {
                        const vd Rq = load(R + t * LN);
                        const vd r2 = mul(Rq, Rq);
                        const vd r4 = mul(r2, r2);
                        ek_factor(G_re + t * LN, G_im + t * LN, div(v_t1n, r4),
                                  div(v_t2n, r2), mul(v_k, Rq), emask);
                    }
                }

                // Stage 2: per moment and lane, the walk's chain from 0.0 in
                // ascending t (one chunk), real and imaginary in two passes.
                vd acc[NMM];
                stage2<NMM>(G_re, wv, m, acc);
                for (int pP = 0; pP < NMM; pP++) store(res + pP * LN, acc[pP]);
                stage2<NMM>(G_im, wv, m, acc);
                for (int pP = 0; pP < NMM; pP++) store(res + (NMM + pP) * LN, acc[pP]);
                for (size_t l = 0; l < LN; l++) {
                    double *o = tab_p + (p0 + l) * 2 * NMM;
                    for (int x = 0; x < 2 * NMM; x++) o[x] = res[x * LN + l];
                }
            }
            }
        } else
#else
        (void)reference;
#endif
        {
        MW_OMP_PARALLEL_FOR_COLLAPSE2
        for (size_t ti = 0; ti < tseg.size(); ti++) {
            for (size_t sj = 0; sj < nSegJ; sj++) {
                MW_CANCEL_POLL();
                pair_walk(tseg[ti], sj, tab_p + (ti * nSegJ + sj) * 2 * NMM);
            }
        }
        }
        MW_THROW_IF_ABORTED();

        // Phase B: the walk's Galerkin combine, per basis pair and wing pair
        // in the walk's order, reading the pair's moments from the table.
        MW_OMP_PARALLEL_FOR_COLLAPSE2
        for (size_t m = m0; m < m1; m++) {
            for (size_t n = 0; n < nJ; n++) {
                MW_CANCEL_POLL();
                double zA_re = 0.0, zA_im = 0.0, zPhi_re = 0.0, zPhi_im = 0.0;

                for (int a = 0; a < NM; a++) {
                    int64_t smi = sI(m, a);
                    double tix = tI(smi,0), tiy = tI(smi,1), tiz = tI(smi,2);
                    const double *row = tab_p + (size_t)loc[smi] * nSegJ * 2 * NMM;
                    for (int b = 0; b < NM; b++) {
                        int64_t snj = sJ(n, b);
                        double td = tix*tJ(snj,0) + tiy*tJ(snj,1) + tiz*tJ(snj,2);
                        const double *Jre = row + (size_t)snj * 2 * NMM;
                        const double *Jim = Jre + NMM;

                        double iA_re = 0.0, iA_im = 0.0, iPhi_re = 0.0, iPhi_im = 0.0;
                        for (int p = 0; p < NM; p++) {
                            double mp = pI(m, a, p);
                            for (int q = 0; q < NM; q++) {
                                double nq = pJ(n, b, q);
                                double prod = mp * nq;
                                iA_re += prod * Jre[p*NM + q];
                                iA_im += prod * Jim[p*NM + q];
                                if (p >= 1 && q >= 1) {
                                    double pq = (double)(p*q) * prod;
                                    iPhi_re += pq * Jre[(p-1)*NM + (q-1)];
                                    iPhi_im += pq * Jim[(p-1)*NM + (q-1)];
                                }
                            }
                        }
                        if (WEIGHTED) {
                            // The reduced kernel's WEIGHTED=true tail,
                            // verbatim: Fresnel dyad at the pair's specular
                            // angle, applied to the extended moments.
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
        MW_THROW_IF_ABORTED();
        for (size_t s : tseg) loc[s] = -1;
    }

    return Z;
}

// The flat-rule and laddered dispatchers share one argument list.
#define MW_BLOCK_EK_ARGS                                                        \
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_I,     \
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_I,     \
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_I,      \
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_I,      \
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_I,       \
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_J,     \
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_J,     \
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_J,      \
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_J,      \
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_J,       \
    double a_squared, double k, double omega, double eps_, double mu_, int max_d
#define MW_BLOCK_EK_PASS                                                        \
    supp_I, polys_I, segl_I, segr_I, tan_I, supp_J, polys_J, segl_J, segr_J,    \
    tan_J, a_squared, k, omega, eps_, mu_

// The per-pair phase guard's masks: absent (None) means every segment passes.
static const uint8_t *phase_mask_or_null(
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& mask,
    size_t n, const char *who
) {
    if (mask.ndim() == 1 && mask.shape(0) == 0) return nullptr;
    if (mask.ndim() != 1 || (size_t)mask.shape(0) != n) {
        throw std::runtime_error(std::string(who) +
                                 ": phase_ok arrays must match the segment unions");
    }
    return mask.data();
}

template<bool WEIGHTED>
static py::array_t<std::complex<double>>
bspline_assemble_offedge_block_ek_dispatch(
    MW_BLOCK_EK_ARGS,
    const PairOrderLadder& ladder,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_I,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_J,
    double a_ek,
    const uint8_t *ok_I, const uint8_t *ok_J, int64_t limited_below,
    uintptr_t cancel_flag, bool reference,
    std::complex<double> eps_t, std::complex<double> phi_c0,
    std::complex<double> phi_c1, const char *who
) {
    switch (max_d) {
        case 1:
            return bspline_assemble_offedge_block_kernel_ek<1, WEIGHTED>(
                MW_BLOCK_EK_PASS, ladder, group_I, group_J, a_ek, ok_I, ok_J,
                limited_below, cancel_flag, reference, eps_t, phi_c0, phi_c1);
        case 2:
            return bspline_assemble_offedge_block_kernel_ek<2, WEIGHTED>(
                MW_BLOCK_EK_PASS, ladder, group_I, group_J, a_ek, ok_I, ok_J,
                limited_below, cancel_flag, reference, eps_t, phi_c0, phi_c1);
        case 3:
            return bspline_assemble_offedge_block_kernel_ek<3, WEIGHTED>(
                MW_BLOCK_EK_PASS, ladder, group_I, group_J, a_ek, ok_I, ok_J,
                limited_below, cancel_flag, reference, eps_t, phi_c0, phi_c1);
        default:
            throw std::runtime_error(std::string(who) +
                                     ": max_d must be 1, 2 or 3");
    }
}

static py::array_t<std::complex<double>>
bspline_assemble_offedge_block_ek(
    MW_BLOCK_EK_ARGS,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_I,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_J,
    double a_ek,
    uintptr_t cancel_flag = 0,
    bool reference = false
) {
    return bspline_assemble_offedge_block_ek_dispatch<false>(
        MW_BLOCK_EK_PASS, max_d, ladder_from_rule(gl_t, gl_w), group_I, group_J,
        a_ek, nullptr, nullptr, 0, cancel_flag, reference,
        std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0),
        std::complex<double>(0.0, 0.0), "bspline_assemble_offedge_block_ek");
}

// The extended-kernel twin of `bspline_assemble_offedge_block_refl`
// (momwire#269): the reflection-coefficient finite-ground image block with
// the coaxial factor applied on eligible segment pairs. J side pre-mirrored,
// exactly as for both parents.
static py::array_t<std::complex<double>>
bspline_assemble_offedge_block_refl_ek(
    MW_BLOCK_EK_ARGS,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_I,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_J,
    double a_ek,
    std::complex<double> eps_t,
    std::complex<double> phi_c0,
    std::complex<double> phi_c1,
    uintptr_t cancel_flag = 0,
    bool reference = false
) {
    return bspline_assemble_offedge_block_ek_dispatch<true>(
        MW_BLOCK_EK_PASS, max_d, ladder_from_rule(gl_t, gl_w), group_I, group_J,
        a_ek, nullptr, nullptr, 0, cancel_flag, reference, eps_t, phi_c0, phi_c1,
        "bspline_assemble_offedge_block_refl_ek");
}

// The distance-adaptive (ladder) twins of the two entries above
// (momwire#1362): `seg_seg_full_moments_bspline_tiered`'s ladder contract,
// with the per-pair phase guard answered in the kernel from the caller's
// per-segment masks (`phase_ok_I` / `phase_ok_J`, uint8, empty = all pass)
// so a pair's order depends on the pair alone -- an ACA row and column
// sampling the same pair cannot disagree about it (momwire#921's invariant).
// One tier reproduces the flat entries bit for bit.
static py::array_t<std::complex<double>>
bspline_assemble_offedge_block_ek_tiered(
    MW_BLOCK_EK_ARGS,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tier_n_qp,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_ratio,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_I,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_J,
    double a_ek,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> phase_ok_I,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> phase_ok_J,
    int64_t limited_below,
    uintptr_t cancel_flag = 0,
    bool reference = false
) {
    const char *who = "bspline_assemble_offedge_block_ek_tiered";
    return bspline_assemble_offedge_block_ek_dispatch<false>(
        MW_BLOCK_EK_PASS, max_d,
        ladder_from_arrays(tier_t, tier_w, tier_n_qp, tier_ratio),
        group_I, group_J, a_ek,
        phase_mask_or_null(phase_ok_I, (size_t)segl_I.shape(0), who),
        phase_mask_or_null(phase_ok_J, (size_t)segl_J.shape(0), who),
        limited_below, cancel_flag, reference,
        std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0),
        std::complex<double>(0.0, 0.0), who);
}

static py::array_t<std::complex<double>>
bspline_assemble_offedge_block_refl_ek_tiered(
    MW_BLOCK_EK_ARGS,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tier_n_qp,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_ratio,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_I,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_J,
    double a_ek,
    std::complex<double> eps_t,
    std::complex<double> phi_c0,
    std::complex<double> phi_c1,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> phase_ok_I,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> phase_ok_J,
    int64_t limited_below,
    uintptr_t cancel_flag = 0,
    bool reference = false
) {
    const char *who = "bspline_assemble_offedge_block_refl_ek_tiered";
    return bspline_assemble_offedge_block_ek_dispatch<true>(
        MW_BLOCK_EK_PASS, max_d,
        ladder_from_arrays(tier_t, tier_w, tier_n_qp, tier_ratio),
        group_I, group_J, a_ek,
        phase_mask_or_null(phase_ok_I, (size_t)segl_I.shape(0), who),
        phase_mask_or_null(phase_ok_J, (size_t)segl_J.shape(0), who),
        limited_below, cancel_flag, reference, eps_t, phi_c0, phi_c1, who);
}

#undef MW_BLOCK_EK_ARGS
#undef MW_BLOCK_EK_PASS


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
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    bool reference
) {
    switch (max_d) {
        case 1:
            return seg_seg_full_moments_bspline_kernel<1, false>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k, 0.0, gl_t, gl_w, reference);
        case 2:
            return seg_seg_full_moments_bspline_kernel<2, false>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k, 0.0, gl_t, gl_w, reference);
        case 3:
            return seg_seg_full_moments_bspline_kernel<3, false>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k, 0.0, gl_t, gl_w, reference);
        default:
            throw std::runtime_error(
                "seg_seg_full_moments_bspline: max_d must be 1, 2 or 3 "
                "(add an explicit template instantiation in _accelerators.cpp)");
    }
}


// In-medium (complex-k) twin of the wrapper above (momwire#778). Takes the
// wavenumber as a std::complex<double> and splits it at the boundary, so the
// kernel itself never sees a complex type and its inner loops stay real.
//
// The Im k <= 0 convention is enforced on the PYTHON side (`_complex_k` raises
// on the growing-exponential branch) before any call reaches here; a positive
// Im k would silently produce exp(+|k_im|R) growth, so the guard is not
// optional. Re-asserted here because this entry point is reachable directly
// from a test.
static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_cplx(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    std::complex<double> k,
    int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w,
    bool reference
) {
    if (k.imag() > 0.0) {
        throw std::runtime_error(
            "seg_seg_full_moments_bspline_cplx: Im k > 0 is the growing "
            "exponential branch; e^{+jwt} requires Im k <= 0 so e^{-jkR} decays");
    }
    switch (max_d) {
        case 1:
            return seg_seg_full_moments_bspline_kernel<1, true>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared,
                k.real(), k.imag(), gl_t, gl_w, reference);
        case 2:
            return seg_seg_full_moments_bspline_kernel<2, true>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared,
                k.real(), k.imag(), gl_t, gl_w, reference);
        case 3:
            return seg_seg_full_moments_bspline_kernel<3, true>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared,
                k.real(), k.imag(), gl_t, gl_w, reference);
        default:
            throw std::runtime_error(
                "seg_seg_full_moments_bspline_cplx: max_d must be 1, 2 or 3 "
                "(add an explicit template instantiation in _accelerators.cpp)");
    }
}


// Distance-adaptive twins of the two entries above (momwire#906): the same
// kernel with a pair-order LADDER instead of one rule. tier 0 is the base
// order; tier t >= 1 serves pairs at centre distance >= tier_ratio[t]
// segment lengths. Real k and complex k are separate entry points for the
// same reason the plain pair is (`float(k)` must never truncate a medium's
// wavenumber, momwire#553 U1).
static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_tiered(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    double k,
    int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tier_n_qp,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_ratio,
    bool reference
) {
    PairOrderLadder ladder = ladder_from_arrays(tier_t, tier_w, tier_n_qp, tier_ratio);
    switch (max_d) {
        case 1:
            return seg_seg_full_moments_bspline_kernel_impl<1, false>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k, 0.0, ladder, reference);
        case 2:
            return seg_seg_full_moments_bspline_kernel_impl<2, false>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k, 0.0, ladder, reference);
        case 3:
            return seg_seg_full_moments_bspline_kernel_impl<3, false>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k, 0.0, ladder, reference);
        default:
            throw std::runtime_error(
                "seg_seg_full_moments_bspline_tiered: max_d must be 1, 2 or 3 "
                "(add an explicit template instantiation in _accelerators.cpp)");
    }
}

static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_cplx_tiered(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    std::complex<double> k,
    int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tier_n_qp,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_ratio,
    bool reference
) {
    if (k.imag() > 0.0) {
        throw std::runtime_error(
            "seg_seg_full_moments_bspline_cplx_tiered: Im k > 0 is the growing "
            "exponential branch; e^{+jwt} requires Im k <= 0 so e^{-jkR} decays");
    }
    PairOrderLadder ladder = ladder_from_arrays(tier_t, tier_w, tier_n_qp, tier_ratio);
    switch (max_d) {
        case 1:
            return seg_seg_full_moments_bspline_kernel_impl<1, true>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared,
                k.real(), k.imag(), ladder, reference);
        case 2:
            return seg_seg_full_moments_bspline_kernel_impl<2, true>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared,
                k.real(), k.imag(), ladder, reference);
        case 3:
            return seg_seg_full_moments_bspline_kernel_impl<3, true>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared,
                k.real(), k.imag(), ladder, reference);
        default:
            throw std::runtime_error(
                "seg_seg_full_moments_bspline_cplx_tiered: max_d must be 1, 2 or 3 "
                "(add an explicit template instantiation in _accelerators.cpp)");
    }
}


// ===========================================================================
// The B-spline pair machinery with the SINUSOIDAL shape hook (momwire#1354).
//
// `seg_seg_full_moments_bspline_kernel_impl` above evaluates the monomial
// shapes u^p at each Gauss node and contracts them against G over the pair's
// (q, r) nodes; its ladder picks the order per pair by distance. The
// sinusoidal Galerkin fill is the same reaction integral on NEC's three-term
// basis, whose shapes on a segment of length L are the folded set
//
//     S_0 = 1,   S_1 = sin(k xi),   S_2 = cos(k xi) - 1 = -2 sin^2(k xi / 2),
//
// xi = u - L/2 the arc from the segment centre (`_sinusoidal_mp.py`). The
// kernel below is that contraction with those shapes: ladder, tier selection,
// positions and the exp(-jkR)/(4 pi R) factorisation are the B-spline kernel's
// (the monomial lane kernel is left untouched, bit for bit, since the shapes
// here are complex in the medium and a shared template would have to widen
// its real wuwu tables). The shape values are evaluated once per (tier,
// segment, node) and reused across every pair the segment takes part in.
//
// In the medium k is complex and so are the shapes; at a real k the shape
// tables are real and the contraction is a real x complex multiply, which is
// why COMPLEX_K is a template parameter rather than a widened argument.
// ===========================================================================

template<bool COMPLEX_K>
struct SinShapeValue {
    using type = std::complex<double>;
};
template<>
struct SinShapeValue<false> {
    using type = double;
};

template<bool COMPLEX_K, bool EK = false>
static py::array_t<std::complex<double>>
seg_seg_full_moments_sinusoidal_kernel(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    double k_re,
    double k_im,
    const PairOrderLadder& ladder,
    // EK only (momwire#1362): per-segment coaxial group labels (N_i,) /
    // (N_j,) and the plain EK radius; a pair is extended iff its labels match.
    const int64_t *grp_i = nullptr,
    const int64_t *grp_j = nullptr,
    double a_ek = 0.0,
    // momwire#1368: Eq 89's source tube b per SOURCE segment (N_j,), or
    // nullptr for b = a_ek (the equal-radius factor).
    const double *b_j = nullptr
) {
    static_assert(!(EK && COMPLEX_K),
                  "the extended kernel is served at a real k only");
    using shape_t = typename SinShapeValue<COMPLEX_K>::type;
    const double a2_ek = a_ek * a_ek;
    static constexpr int NM = 3;
    static constexpr int NMM = NM * NM;

    auto sli = seg_l_i.unchecked<2>();
    auto sri = seg_r_i.unchecked<2>();
    auto slj = seg_l_j.unchecked<2>();
    auto srj = seg_r_j.unchecked<2>();
    if (sli.shape(1) != 3 || sri.shape(1) != 3 ||
        slj.shape(1) != 3 || srj.shape(1) != 3) {
        throw std::runtime_error("segment endpoint arrays must have shape (N, 3)");
    }
    if (sli.shape(0) != sri.shape(0) || slj.shape(0) != srj.shape(0)) {
        throw std::runtime_error("seg_l and seg_r must have matching N");
    }
    const size_t N_i = sli.shape(0);
    const size_t N_j = slj.shape(0);
    const size_t n_tiers = ladder.n_tiers();
    const std::complex<double> k(k_re, k_im);

    py::array_t<std::complex<double>> J({(size_t)NM, (size_t)NM, N_i, N_j});
    auto j_view = J.mutable_unchecked<4>();

    py::gil_scoped_release release;

    const double inv_4pi = 1.0 / (4.0 * M_PI);

    std::vector<double> len_i(N_i), len_j(N_j);
    std::vector<double> c_i(N_i * 3), c_j(N_j * 3);
    for (size_t i = 0; i < N_i; i++) {
        const double dx = sri(i,0) - sli(i,0);
        const double dy = sri(i,1) - sli(i,1);
        const double dz = sri(i,2) - sli(i,2);
        len_i[i] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (int d = 0; d < 3; d++) c_i[i*3 + d] = 0.5 * (sli(i,d) + sri(i,d));
    }
    for (size_t j = 0; j < N_j; j++) {
        const double dx = srj(j,0) - slj(j,0);
        const double dy = srj(j,1) - slj(j,1);
        const double dz = srj(j,2) - slj(j,2);
        len_j[j] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (int d = 0; d < 3; d++) c_j[j*3 + d] = 0.5 * (slj(j,d) + srj(j,d));
    }
    // Per tier: node positions, and the WEIGHTED shape values w_q L S_p(xi_q)
    // per (segment, node, shape). The weight and length ride in the shape
    // table so the pair loop multiplies nothing but G.
    std::vector<std::vector<double>> pos_i(n_tiers), pos_j(n_tiers);
    std::vector<std::vector<shape_t>> ws_i(n_tiers), ws_j(n_tiers);
    auto fill_tables = [&](size_t tier, size_t N, const double *len,
                           const py::detail::unchecked_reference<double, 2> &sl,
                           const py::detail::unchecked_reference<double, 2> &sr,
                           std::vector<double> &pos, std::vector<shape_t> &ws) {
        const size_t n_qp = ladder.n_qp(tier);
        const double *gt = ladder.t_at(tier);
        const double *gw = ladder.w_at(tier);
        pos.resize(N * n_qp * 3);
        ws.resize(N * n_qp * NM);
        for (size_t s = 0; s < N; s++) {
            for (size_t q = 0; q < n_qp; q++) {
                const double t = gt[q];
                pos[(s*n_qp + q)*3 + 0] = (1.0 - t) * sl(s,0) + t * sr(s,0);
                pos[(s*n_qp + q)*3 + 1] = (1.0 - t) * sl(s,1) + t * sr(s,1);
                pos[(s*n_qp + q)*3 + 2] = (1.0 - t) * sl(s,2) + t * sr(s,2);
                const double w = gw[q] * len[s];
                const double xi = (t - 0.5) * len[s];
                shape_t *o = &ws[(s*n_qp + q) * NM];
                if constexpr (COMPLEX_K) {
                    const std::complex<double> arg = k * xi;
                    const std::complex<double> half = std::sin(0.5 * arg);
                    o[0] = shape_t(w);
                    o[1] = shape_t(w * std::sin(arg));
                    o[2] = shape_t(w * (-2.0 * half * half));
                } else {
                    const double arg = k_re * xi;
                    const double half = std::sin(0.5 * arg);
                    o[0] = shape_t(w);
                    o[1] = shape_t(w * std::sin(arg));
                    o[2] = shape_t(w * (-2.0 * half * half));
                }
            }
        }
    };
    for (size_t tier = 0; tier < n_tiers; tier++) {
        fill_tables(tier, N_i, len_i.data(), sli, sri, pos_i[tier], ws_i[tier]);
        fill_tables(tier, N_j, len_j.data(), slj, srj, pos_j[tier], ws_j[tier]);
    }

    // The ladder's selector, as the B-spline kernel computes it: centre
    // distance over the longer segment, highest tier whose threshold the
    // ratio reaches.
    auto pair_tier = [&](size_t i, size_t j) -> size_t {
        size_t tier = 0;
        if (n_tiers > 1) {
            const double dx = c_i[i*3 + 0] - c_j[j*3 + 0];
            const double dy = c_i[i*3 + 1] - c_j[j*3 + 1];
            const double dz = c_i[i*3 + 2] - c_j[j*3 + 2];
            const double ratio = std::sqrt(dx*dx + dy*dy + dz*dz)
                                 / std::max(len_i[i], len_j[j]);
            for (size_t t = 1; t < n_tiers; t++) {
                if (ratio >= ladder.ratio[t]) tier = t; else break;
            }
        }
        return tier;
    };

    // Shape tables split into (re, im) planes so the contraction below is
    // explicit real arithmetic: std::complex's operator* carries the C99
    // NaN/Inf fixup (`__muldc3`) under this build's strict FP flags, which
    // is most of a scalar complex multiply's cost.
    std::vector<std::vector<double>> wsi_re(n_tiers), wsi_im(n_tiers), wsj_re(n_tiers), wsj_im(n_tiers);
    for (size_t tier = 0; tier < n_tiers; tier++) {
        auto split = [](const std::vector<shape_t> &src, std::vector<double> &re, std::vector<double> &im) {
            re.resize(src.size());
            im.resize(src.size());
            for (size_t x = 0; x < src.size(); x++) {
                if constexpr (COMPLEX_K) { re[x] = src[x].real(); im[x] = src[x].imag(); }
                else { re[x] = src[x]; im[x] = 0.0; }
            }
        };
        split(ws_i[tier], wsi_re[tier], wsi_im[tier]);
        split(ws_j[tier], wsj_re[tier], wsj_im[tier]);
    }

    MW_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t i = 0; i < N_i; i++) {
        for (size_t j = 0; j < N_j; j++) {
            const size_t tier = pair_tier(i, j);
            const size_t n_qp = ladder.n_qp(tier);
            const double *pi = &pos_i[tier][i * n_qp * 3];
            const double *pj = &pos_j[tier][j * n_qp * 3];
            const double *si_re = &wsi_re[tier][i * n_qp * NM];
            const double *si_im = &wsi_im[tier][i * n_qp * NM];
            const double *sj_re = &wsj_re[tier][j * n_qp * NM];
            const double *sj_im = &wsj_im[tier][j * n_qp * NM];
            double acc_re[NMM], acc_im[NMM];
            for (int pP = 0; pP < NMM; pP++) { acc_re[pP] = 0.0; acc_im[pP] = 0.0; }
            const size_t n_pairs = n_qp * n_qp;
            alignas(32) double R[BSPLINE_QR_TILE];
            alignas(32) double ph[BSPLINE_QR_TILE];
            alignas(32) double G_re[BSPLINE_QR_TILE], G_im[BSPLINE_QR_TILE];
            for (size_t base = 0; base < n_pairs; base += BSPLINE_QR_TILE) {
                const size_t m = std::min(n_pairs - base, (size_t)BSPLINE_QR_TILE);
                for (size_t t = 0; t < m; t++) {
                    const size_t q = (base + t) / n_qp, r = (base + t) % n_qp;
                    const double dx = pi[q*3 + 0] - pj[r*3 + 0];
                    const double dy = pi[q*3 + 1] - pj[r*3 + 1];
                    const double dz = pi[q*3 + 2] - pj[r*3 + 2];
                    R[t] = std::sqrt(dx*dx + dy*dy + dz*dz + a_squared);
                }
                MW_OMP_SIMD()
                for (size_t t = 0; t < m; t++) ph[t] = -k_re * R[t];
                MW_OMP_SIMD()
                for (size_t t = 0; t < m; t++) G_re[t] = std::cos(ph[t]);
                MW_OMP_SIMD()
                for (size_t t = 0; t < m; t++) G_im[t] = std::sin(ph[t]);
                if constexpr (COMPLEX_K) {
                    MW_OMP_SIMD()
                    for (size_t t = 0; t < m; t++) ph[t] = std::exp(k_im * R[t]);
                    MW_OMP_SIMD()
                    for (size_t t = 0; t < m; t++) {
                        const double sc = ph[t] * inv_4pi / R[t];
                        G_re[t] *= sc;
                        G_im[t] *= sc;
                    }
                } else {
                    MW_OMP_SIMD()
                    for (size_t t = 0; t < m; t++) {
                        const double sc = inv_4pi / R[t];
                        G_re[t] *= sc;
                        G_im[t] *= sc;
                    }
                }
                if (EK && grp_i[i] == grp_j[j] && grp_i[i] >= 0) {
                    const double b2 = b_j ? b_j[j] * b_j[j] : a2_ek;
                    // momwire#1368: the factor no closer than the source
                    // tube (bspline's `ek_floor`); only b > a can bite.
                    const double bj = b_j ? b_j[j] : a_ek;
                    const bool ek_floor = bj > a_ek;
                    // NEC Eq 89's coaxial factor on G (`_bspline_kernels.
                    // _ek_factor`): fac = 1 + T1 C2 - T2 C1, C1 = 1 + jkR,
                    // C2 = 3 C1 - (kR)^2, T1 = a^4 / 4R^4, T2 = a^2 / 2R^2,
                    // in that order of operations.
                    for (size_t t = 0; t < m; t++) {
                        const double Rf = ek_floor ? std::max(R[t], bj) : R[t];
                        const double r2 = Rf * Rf;
                        const double r4 = r2 * r2;
                        const double kr = k_re * Rf;
                        const double kr2 = kr * kr;
                        const double t1 = 0.25 * (b2 * a2_ek) / r4;
                        const double t2 = 0.5 * b2 / r2;
                        double fr = t1 * (3.0 - kr2);
                        double fi = t1 * (3.0 * kr);
                        fr = fr - t2;
                        fi = fi - t2 * kr;
                        fr = fr + 1.0;
                        const double gr = G_re[t], gi = G_im[t];
                        G_re[t] = gr * fr - gi * fi;
                        G_im[t] = gr * fi + gi * fr;
                    }
                }
                // Source-side contraction per test node q: gq[P] = sum_r G sj[r][P];
                // then the 3x3 outer product with the test shapes once per q.
                size_t t = 0;
                while (t < m) {
                    const size_t q = (base + t) / n_qp;
                    size_t r = (base + t) % n_qp;
                    double gq_re[NM] = {0.0, 0.0, 0.0}, gq_im[NM] = {0.0, 0.0, 0.0};
                    for (; r < n_qp && t < m; r++, t++) {
                        const double gr = G_re[t], gi = G_im[t];
                        for (int P = 0; P < NM; P++) {
                            const double sr = sj_re[r*NM + P], sim = sj_im[r*NM + P];
                            gq_re[P] += gr * sr - gi * sim;
                            gq_im[P] += gr * sim + gi * sr;
                        }
                    }
                    for (int p = 0; p < NM; p++) {
                        const double ar = si_re[q*NM + p], ai = si_im[q*NM + p];
                        for (int P = 0; P < NM; P++) {
                            acc_re[p*NM + P] += ar * gq_re[P] - ai * gq_im[P];
                            acc_im[p*NM + P] += ar * gq_im[P] + ai * gq_re[P];
                        }
                    }
                }
            }
            for (int pP = 0; pP < NMM; pP++)
                j_view(pP / NM, pP % NM, i, j) = std::complex<double>(acc_re[pP], acc_im[pP]);
        }
    }
    return J;
}

static py::array_t<std::complex<double>>
seg_seg_full_moments_sinusoidal_tiered(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    std::complex<double> k,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tier_n_qp,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_ratio
) {
    if (k.imag() > 0.0) {
        throw std::runtime_error(
            "seg_seg_full_moments_sinusoidal_tiered: Im k > 0 is the growing "
            "exponential branch; e^{+jwt} requires Im k <= 0 so e^{-jkR} decays");
    }
    PairOrderLadder ladder = ladder_from_arrays(tier_t, tier_w, tier_n_qp, tier_ratio);
    if (k.imag() == 0.0) {
        return seg_seg_full_moments_sinusoidal_kernel<false>(
            seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k.real(), 0.0, ladder);
    }
    return seg_seg_full_moments_sinusoidal_kernel<true>(
        seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k.real(), k.imag(), ladder);
}

// The extended-kernel twin of `seg_seg_full_moments_sinusoidal_tiered`
// (momwire#1362): the same contraction with NEC Eq 89's coaxial factor on G
// for the pairs whose group labels match (`_bspline_kernels._ek_axis_groups`,
// the eligibility rule bspline's EK fill uses). A real k only: the extended
// kernel is refused in the medium.
static py::array_t<std::complex<double>>
seg_seg_full_moments_sinusoidal_tiered_ek(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    double k,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tier_n_qp,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_ratio,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_i,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_j,
    double a_ek,
    py::object b_j_obj
) {
    py::array_t<double, py::array::c_style | py::array::forcecast> b_j_arr;
    if (!b_j_obj.is_none()) {
        b_j_arr = b_j_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        if ((size_t)b_j_arr.size() != (size_t)seg_l_j.shape(0))
            throw std::runtime_error(
                "seg_seg_full_moments_sinusoidal_tiered_ek: one b per source segment");
    }
    if ((size_t)group_i.size() != (size_t)seg_l_i.shape(0) ||
        (size_t)group_j.size() != (size_t)seg_l_j.shape(0)) {
        throw std::runtime_error(
            "seg_seg_full_moments_sinusoidal_tiered_ek: one group label per segment");
    }
    PairOrderLadder ladder = ladder_from_arrays(tier_t, tier_w, tier_n_qp, tier_ratio);
    return seg_seg_full_moments_sinusoidal_kernel<false, true>(
        seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k, 0.0, ladder,
        group_i.data(), group_j.data(), a_ek,
        b_j_obj.is_none() ? nullptr : b_j_arr.data());
}

// The sinusoidal fill's windowed assembler (momwire#1354): the B-spline
// `assemble_Z_bspline_windowed` with the basis handed in as CSR-by-segment
// coefficient tables on the folded shape set and its derivative coefficients
// ALREADY ROTATED (`_sinusoidal_mp.dshape_coefs`), rather than as fixed-width
// support wings of monomial polynomials. One window of pair moments
// J (3, 3, n_rows, n_cols) over the segments `rows_seg` x `cols_seg`
// accumulates
//
//   Z[i, j] += c_a . w_a[m, n] . (t_m . t_n) . coef_e^T J[:, :, m, n] coef_e'
//            + c_phi . w_phi[m, n] . dcoef_e^T J[:, :, m, n] dcoef_e'
//
// over the CSR entries e of segment rows_seg[m] (basis i = jbasis[e]) and e'
// of cols_seg[n]; `c_a` = jk eta and `c_phi` = eta / (jk) carry the block's
// (k, eta) and any global scale, `tangents_*` are the window's own tangents
// (an image window hands the mirrored source tangents) and `w_a` / `w_phi`
// are optional per-pair complex tables (Fresnel or C2), 1 when absent.
//
// Threads own distinct BASIS rows of Z: the window's entries are grouped by
// basis first, so two segments carrying the same basis never race on its
// row, and no atomic is needed on a complex accumulate.
static void
assemble_Z_sinusoidal_windowed(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> rows_seg,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> cols_seg,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> starts,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> jbasis,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> coef,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> dcoef,
    py::array_t<double, py::array::c_style | py::array::forcecast> tangents_rows,
    py::array_t<double, py::array::c_style | py::array::forcecast> tangents_cols,
    std::complex<double> c_a,
    std::complex<double> c_phi,
    py::object w_a_obj,
    py::object w_phi_obj,
    py::array_t<std::complex<double>> Z,
    uintptr_t cancel_flag = 0
) {
    static constexpr int NM = 3;
    auto jv = J.unchecked<4>();
    auto rs = rows_seg.unchecked<1>();
    auto cs = cols_seg.unchecked<1>();
    auto st = starts.unchecked<1>();
    auto jb = jbasis.unchecked<1>();
    auto cf = coef.unchecked<2>();
    auto dcf = dcoef.unchecked<2>();
    auto tr = tangents_rows.unchecked<2>();
    auto tc = tangents_cols.unchecked<2>();
    const size_t n_rows = (size_t)rs.shape(0);
    const size_t n_cols = (size_t)cs.shape(0);
    if (jv.shape(0) != NM || jv.shape(1) != NM ||
        (size_t)jv.shape(2) != n_rows || (size_t)jv.shape(3) != n_cols) {
        throw std::runtime_error("J must be (3, 3, n_rows, n_cols)");
    }
    if ((size_t)tr.shape(0) != n_rows || tr.shape(1) != 3 ||
        (size_t)tc.shape(0) != n_cols || tc.shape(1) != 3) {
        throw std::runtime_error("tangents must be (n_rows, 3) and (n_cols, 3)");
    }
    const int64_t nnz = jb.shape(0);
    if (cf.shape(0) != nnz || cf.shape(1) != NM || dcf.shape(0) != nnz || dcf.shape(1) != NM) {
        throw std::runtime_error("coef / dcoef must be (nnz, 3)");
    }
    const bool has_wa = !w_a_obj.is_none();
    const bool has_wp = !w_phi_obj.is_none();
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> w_a_arr, w_phi_arr;
    if (has_wa) w_a_arr = w_a_obj.cast<decltype(w_a_arr)>();
    if (has_wp) w_phi_arr = w_phi_obj.cast<decltype(w_phi_arr)>();
    if (has_wa && ((size_t)w_a_arr.shape(0) != n_rows || (size_t)w_a_arr.shape(1) != n_cols)) {
        throw std::runtime_error("w_a must be (n_rows, n_cols)");
    }
    if (has_wp && ((size_t)w_phi_arr.shape(0) != n_rows || (size_t)w_phi_arr.shape(1) != n_cols)) {
        throw std::runtime_error("w_phi must be (n_rows, n_cols)");
    }
    const std::complex<double> *wa = has_wa ? w_a_arr.data() : nullptr;
    const std::complex<double> *wp = has_wp ? w_phi_arr.data() : nullptr;
    auto zv = Z.mutable_unchecked<2>();
    const int64_t n_basis = zv.shape(0);
    if (zv.shape(1) != n_basis) throw std::runtime_error("Z must be square");
    for (size_t m = 0; m < n_rows; m++) {
        const int64_t s = rs(m);
        if (s < 0 || s + 1 >= st.shape(0)) throw std::runtime_error("rows_seg out of range");
    }
    for (size_t n = 0; n < n_cols; n++) {
        const int64_t s = cs(n);
        if (s < 0 || s + 1 >= st.shape(0)) throw std::runtime_error("cols_seg out of range");
    }
    for (int64_t e = 0; e < nnz; e++) {
        if (jb(e) < 0 || jb(e) >= n_basis) throw std::runtime_error("jbasis out of range");
    }

    py::gil_scoped_release release;

    // Window entries grouped by basis: (basis, m, e) sorted by basis, with
    // group offsets. Built once per window, O(entries).
    struct Entry { int64_t basis; size_t m; int64_t e; };
    std::vector<Entry> ents;
    for (size_t m = 0; m < n_rows; m++) {
        const int64_t s = rs(m);
        for (int64_t e = st(s); e < st(s + 1); e++) ents.push_back(Entry{jb(e), m, e});
    }
    std::sort(ents.begin(), ents.end(),
              [](const Entry &a, const Entry &b) { return a.basis < b.basis; });
    std::vector<size_t> grp;
    for (size_t x = 0; x < ents.size(); x++) {
        if (x == 0 || ents[x].basis != ents[x - 1].basis) grp.push_back(x);
    }
    grp.push_back(ents.size());
    const size_t n_grp = grp.size() - 1;
    // Column entries, flattened: per column n the run of (e', basis').
    std::vector<size_t> col_off(n_cols + 1, 0);
    for (size_t n = 0; n < n_cols; n++) {
        const int64_t s = cs(n);
        col_off[n + 1] = col_off[n] + (size_t)(st(s + 1) - st(s));
    }
    std::vector<int64_t> col_e(col_off[n_cols]);
    for (size_t n = 0; n < n_cols; n++) {
        const int64_t s = cs(n);
        size_t o = col_off[n];
        for (int64_t e = st(s); e < st(s + 1); e++) col_e[o++] = e;
    }
    // The distinct basis columns this window touches: each group's row is
    // accumulated in a contiguous buffer and added to Z's (strided, column-
    // major) row once per touched column, not once per entry.
    std::vector<int64_t> col_bases;
    {
        std::vector<char> seen((size_t)n_basis, 0);
        for (size_t o = 0; o < col_e.size(); o++) {
            const int64_t j = jb(col_e[o]);
            if (!seen[(size_t)j]) { seen[(size_t)j] = 1; col_bases.push_back(j); }
        }
        std::sort(col_bases.begin(), col_bases.end());
    }
    const size_t plane = n_rows * n_cols;
    const double *Jd_d = reinterpret_cast<const double *>(J.data());

    MW_CANCEL_SETUP(cancel_flag);
    #pragma omp parallel
    {
        std::vector<std::complex<double>> rowbuf((size_t)n_basis, std::complex<double>(0.0, 0.0));
        #pragma omp for schedule(dynamic, 8)
        for (size_t g = 0; g < n_grp; g++) {
            MW_CANCEL_POLL();
            const int64_t i = ents[grp[g]].basis;
            for (size_t x = grp[g]; x < grp[g + 1]; x++) {
                const size_t m = ents[x].m;
                const int64_t e = ents[x].e;
                // Explicit (re, im) arithmetic throughout: see the moment
                // kernel above for why std::complex's operator* is avoided.
                double ce_re[NM], ce_im[NM], de_re[NM], de_im[NM];
                for (int p = 0; p < NM; p++) {
                    const std::complex<double> a = c_a * cf(e, p), b = c_phi * dcf(e, p);
                    ce_re[p] = a.real(); ce_im[p] = a.imag();
                    de_re[p] = b.real(); de_im[p] = b.imag();
                }
                const double tmx = tr(m, 0), tmy = tr(m, 1), tmz = tr(m, 2);
                const double *Jm = Jd_d + 2 * (m * n_cols);
                for (size_t n = 0; n < n_cols; n++) {
                    const double td = tmx * tc(n, 0) + tmy * tc(n, 1) + tmz * tc(n, 2);
                    double fa_re = td, fa_im = 0.0, fp_re = 1.0, fp_im = 0.0;
                    if (has_wa) { fa_re = wa[m * n_cols + n].real() * td; fa_im = wa[m * n_cols + n].imag() * td; }
                    if (has_wp) { fp_re = wp[m * n_cols + n].real(); fp_im = wp[m * n_cols + n].imag(); }
                    // ta[q] = fa sum_p ce[p] J[p][q]; tp[q] = fp sum_p de[p] J[p][q].
                    double ta_re[NM], ta_im[NM], tp_re[NM], tp_im[NM];
                    for (int q = 0; q < NM; q++) {
                        double sa_re = 0.0, sa_im = 0.0, sp_re = 0.0, sp_im = 0.0;
                        for (int p = 0; p < NM; p++) {
                            const size_t off = 2 * ((size_t)(p * NM + q) * plane + n);
                            const double jr = Jm[off], ji = Jm[off + 1];
                            sa_re += ce_re[p] * jr - ce_im[p] * ji;
                            sa_im += ce_re[p] * ji + ce_im[p] * jr;
                            sp_re += de_re[p] * jr - de_im[p] * ji;
                            sp_im += de_re[p] * ji + de_im[p] * jr;
                        }
                        ta_re[q] = fa_re * sa_re - fa_im * sa_im;
                        ta_im[q] = fa_re * sa_im + fa_im * sa_re;
                        tp_re[q] = fp_re * sp_re - fp_im * sp_im;
                        tp_im[q] = fp_re * sp_im + fp_im * sp_re;
                    }
                    for (size_t o = col_off[n]; o < col_off[n + 1]; o++) {
                        const int64_t ep = col_e[o];
                        double v_re = 0.0, v_im = 0.0;
                        for (int q = 0; q < NM; q++) {
                            const double cr = cf(ep, q).real(), ci = cf(ep, q).imag();
                            const double dr = dcf(ep, q).real(), di = dcf(ep, q).imag();
                            v_re += ta_re[q] * cr - ta_im[q] * ci + tp_re[q] * dr - tp_im[q] * di;
                            v_im += ta_re[q] * ci + ta_im[q] * cr + tp_re[q] * di + tp_im[q] * dr;
                        }
                        rowbuf[(size_t)jb(ep)] += std::complex<double>(v_re, v_im);
                    }
                }
            }
            for (size_t c = 0; c < col_bases.size(); c++) {
                const int64_t j = col_bases[c];
                zv(i, j) += rowbuf[(size_t)j];
                rowbuf[(size_t)j] = std::complex<double>(0.0, 0.0);
            }
        }
    }
    MW_THROW_IF_ABORTED();
}

// Field-form pair moments from a projected field table (momwire#1354): the
// sinusoidal fill's remainder blocks need `Jf[p, P, i, j] = sum_{q, r}
// W_obs[p, i, q] . F[i q, j r] . W_src[P, j, r]` with COMPLEX shape weights
// (the shapes are complex in the medium), which `assemble_field_galerkin`'s
// real tables cannot carry. This is that double sum alone, (3, 3, n_obs,
// n_src); the basis assembly is `assemble_Z_sinusoidal_windowed`'s with the
// tangent dot already inside F.
static py::array_t<std::complex<double>>
field_pair_moments_sinusoidal(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> F,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> W_obs,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> W_src
) {
    static constexpr int NM = 3;
    auto Fv = F.unchecked<2>();
    auto Wo = W_obs.unchecked<3>();
    auto Ws = W_src.unchecked<3>();
    if (Wo.shape(0) != NM || Ws.shape(0) != NM) throw std::runtime_error("W must be (3, n, q)");
    const size_t n_obs = (size_t)Wo.shape(1), n_src = (size_t)Ws.shape(1), q = (size_t)Wo.shape(2);
    if ((size_t)Ws.shape(2) != q) throw std::runtime_error("W_obs and W_src must share q");
    if ((size_t)Fv.shape(0) != n_obs * q || (size_t)Fv.shape(1) != n_src * q)
        throw std::runtime_error("F must be (n_obs q, n_src q)");
    py::array_t<std::complex<double>> J({(size_t)NM, (size_t)NM, n_obs, n_src});
    auto jv = J.mutable_unchecked<4>();
    // Split tables: explicit (re, im) arithmetic, see the moment kernel.
    std::vector<double> wo_re(n_obs * q * NM), wo_im(n_obs * q * NM), ws_re(n_src * q * NM), ws_im(n_src * q * NM);
    for (size_t i = 0; i < n_obs; i++)
        for (size_t a = 0; a < q; a++)
            for (int p = 0; p < NM; p++) {
                wo_re[(i * q + a) * NM + p] = Wo(p, i, a).real();
                wo_im[(i * q + a) * NM + p] = Wo(p, i, a).imag();
            }
    for (size_t j = 0; j < n_src; j++)
        for (size_t b = 0; b < q; b++)
            for (int P = 0; P < NM; P++) {
                ws_re[(j * q + b) * NM + P] = Ws(P, j, b).real();
                ws_im[(j * q + b) * NM + P] = Ws(P, j, b).imag();
            }
    const double *Fd = reinterpret_cast<const double *>(F.data());
    const size_t ld = 2 * n_src * q;
    py::gil_scoped_release release;
    MW_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t i = 0; i < n_obs; i++) {
        for (size_t j = 0; j < n_src; j++) {
            double acc_re[NM * NM], acc_im[NM * NM];
            for (int x = 0; x < NM * NM; x++) { acc_re[x] = 0.0; acc_im[x] = 0.0; }
            for (size_t a = 0; a < q; a++) {
                double g_re[NM] = {0.0, 0.0, 0.0}, g_im[NM] = {0.0, 0.0, 0.0};
                const double *row = Fd + (i * q + a) * ld + 2 * (j * q);
                for (size_t b = 0; b < q; b++) {
                    const double fr = row[2 * b], fi = row[2 * b + 1];
                    const double *sr = &ws_re[(j * q + b) * NM], *si = &ws_im[(j * q + b) * NM];
                    for (int P = 0; P < NM; P++) {
                        g_re[P] += fr * sr[P] - fi * si[P];
                        g_im[P] += fr * si[P] + fi * sr[P];
                    }
                }
                const double *orr = &wo_re[(i * q + a) * NM], *oi = &wo_im[(i * q + a) * NM];
                for (int p = 0; p < NM; p++)
                    for (int P = 0; P < NM; P++) {
                        acc_re[p * NM + P] += orr[p] * g_re[P] - oi[p] * g_im[P];
                        acc_im[p * NM + P] += orr[p] * g_im[P] + oi[p] * g_re[P];
                    }
            }
            for (int x = 0; x < NM * NM; x++) jv(x / NM, x % NM, i, j) = std::complex<double>(acc_re[x], acc_im[x]);
        }
    }
    return J;
}

// The parallel-pair reduction of the sinusoidal fill (momwire#1354,
// `_sinusoidal_mp.parallel_pair_moments`): for a pair of PARALLEL segments the
// kernel depends on the separation delta along the common line alone,
// J[p, q] = int G(delta) Lambda_pq(delta) d delta, Lambda the shapes'
// correlation over the overlap. Each of the three pieces of Lambda (its
// breakpoints are the four corner separations) is integrated in t with
// delta = rho sinh t, which removes the 1/R exactly, Gauss-Legendre in t on
// the outside and in xi on the inside. Same arithmetic as the numpy reference,
// one pair per thread. Complex k serves the medium.
static py::array_t<std::complex<double>>
parallel_pair_moments_sinusoidal(
    py::array_t<double, py::array::c_style | py::array::forcecast> c_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> h_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> c_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> h_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> a2,
    std::complex<double> k,
    py::array_t<double, py::array::c_style | py::array::forcecast> gt,
    py::array_t<double, py::array::c_style | py::array::forcecast> gw,
    py::array_t<double, py::array::c_style | py::array::forcecast> gx,
    py::array_t<double, py::array::c_style | py::array::forcecast> gwx,
    py::object a_ek_obj,
    // momwire#1368: Eq 89's source tube b per pair (n,), or None for b = a_ek.
    py::object b_ek_obj
) {
    static constexpr int NM = 3;
    // momwire#1362: `a_ek` (n,) is each pair's extended-kernel radius, 0 for a
    // pair the extended kernel does not reach (None: no pair). An extended
    // pair takes NEC Eq 89's coaxial factor on G inside the t integral, where
    // it is smooth: R = rho cosh t with rho >= a.
    const bool has_ek = !a_ek_obj.is_none();
    py::array_t<double, py::array::c_style | py::array::forcecast> a_ek_arr;
    if (has_ek) a_ek_arr = a_ek_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    const double *a_ekp = has_ek ? a_ek_arr.data() : nullptr;
    py::array_t<double, py::array::c_style | py::array::forcecast> b_ek_arr;
    const bool has_b = has_ek && !b_ek_obj.is_none();
    if (has_b) b_ek_arr = b_ek_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    const double *b_ekp = has_b ? b_ek_arr.data() : nullptr;
    auto ci = c_i.unchecked<2>(); auto ti = t_i.unchecked<2>(); auto hi_ = h_i.unchecked<1>();
    auto cj = c_j.unchecked<2>(); auto tj = t_j.unchecked<2>(); auto hj_ = h_j.unchecked<1>();
    auto a2v = a2.unchecked<1>();
    auto gtv = gt.unchecked<1>(); auto gwv = gw.unchecked<1>();
    auto gxv = gx.unchecked<1>(); auto gwxv = gwx.unchecked<1>();
    const size_t n = (size_t)ci.shape(0);
    if ((size_t)cj.shape(0) != n || (size_t)a2v.shape(0) != n || (size_t)hi_.shape(0) != n ||
        (size_t)hj_.shape(0) != n || ci.shape(1) != 3 || cj.shape(1) != 3 ||
        ti.shape(1) != 3 || tj.shape(1) != 3)
        throw std::runtime_error("parallel_pair_moments_sinusoidal: n pairs of (3,) rows expected");
    const size_t n_t = (size_t)gtv.shape(0), n_xi = (size_t)gxv.shape(0);
    if ((size_t)gwv.shape(0) != n_t || (size_t)gwxv.shape(0) != n_xi)
        throw std::runtime_error("rules must be (nodes, weights) pairs of one length");
    if (has_ek && (size_t)a_ek_arr.size() != n)
        throw std::runtime_error("parallel_pair_moments_sinusoidal: a_ek must be (n,)");
    if (has_b && (size_t)b_ek_arr.size() != n)
        throw std::runtime_error("parallel_pair_moments_sinusoidal: b_ek must be (n,)");
    py::array_t<std::complex<double>> J({(size_t)NM, (size_t)NM, n});
    auto jv = J.mutable_unchecked<3>();
    const double kr = k.real(), kim = k.imag();
    const bool cplx = kim != 0.0;
    if (has_ek && cplx)
        throw std::runtime_error("parallel_pair_moments_sinusoidal: the extended kernel needs a real k");
    py::gil_scoped_release release;
    const double inv_4pi = 1.0 / (4.0 * M_PI);
    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t p = 0; p < n; p++) {
        double dot = 0.0, dpar = 0.0, dd = 0.0;
        for (int d = 0; d < 3; d++) {
            dot += ti(p, d) * tj(p, d);
            const double dc = ci(p, d) - cj(p, d);
            dpar += dc * tj(p, d);
            dd += dc * dc;
        }
        const double s = dot > 0.0 ? 1.0 : (dot < 0.0 ? -1.0 : 0.0);
        const double dperp2 = std::max(dd - dpar * dpar, 0.0);
        const double rho = std::sqrt(dperp2 + a2v(p));
        const double hI = hi_(p), hJ = hj_(p);
        double br[4] = {dpar - 0.5 * (hI + hJ), dpar - 0.5 * (hI - hJ),
                        dpar + 0.5 * (hI - hJ), dpar + 0.5 * (hI + hJ)};
        std::sort(br, br + 4);
        double acc_re[NM * NM], acc_im[NM * NM];
        for (int x = 0; x < NM * NM; x++) { acc_re[x] = 0.0; acc_im[x] = 0.0; }
        for (int piece = 0; piece < 3; piece++) {
            const double lo_t = std::asinh(br[piece] / rho), hi_t = std::asinh(br[piece + 1] / rho);
            const double mid = 0.5 * (hi_t + lo_t), half = 0.5 * (hi_t - lo_t);
            for (size_t a = 0; a < n_t; a++) {
                const double t = mid + half * gtv(a);
                const double w_t = half * gwv(a);
                const double delta = rho * std::sinh(t), R = rho * std::cosh(t);
                // G d delta = exp(-jkR) / (4 pi) dt
                double sc = inv_4pi * w_t;
                if (cplx) sc *= std::exp(kim * R);
                const double ph = -kr * R;
                double g_re = std::cos(ph) * sc, g_im = std::sin(ph) * sc;
                if (has_ek && a_ekp[p] > 0.0) {
                    const double a2e = a_ekp[p] * a_ekp[p];
                    const double b2e = has_b ? b_ekp[p] * b_ekp[p] : a2e;
                    // The factor no closer than the source tube (momwire#1368).
                    const double Rf = (has_b && b_ekp[p] > a_ekp[p]) ? std::max(R, b_ekp[p]) : R;
                    const double r2 = Rf * Rf, r4 = r2 * r2;
                    const double kR = kr * Rf, kR2 = kR * kR;
                    const double t1 = 0.25 * (b2e * a2e) / r4;
                    const double t2 = 0.5 * b2e / r2;
                    double fr = t1 * (3.0 - kR2);
                    double fi = t1 * (3.0 * kR);
                    fr = fr - t2;
                    fi = fi - t2 * kR;
                    fr = fr + 1.0;
                    const double gr = g_re, gi = g_im;
                    g_re = gr * fr - gi * fi;
                    g_im = gr * fi + gi * fr;
                }
                const double beta = dpar - delta;
                double lo, hi;
                if (s > 0.0) { lo = std::max(-0.5 * hI, -beta - 0.5 * hJ); hi = std::min(0.5 * hI, -beta + 0.5 * hJ); }
                else { lo = std::max(-0.5 * hI, beta - 0.5 * hJ); hi = std::min(0.5 * hI, beta + 0.5 * hJ); }
                const double width = std::max(hi - lo, 0.0);
                const double xm = 0.5 * (hi + lo), xh = 0.5 * width;
                double lam_re[NM * NM], lam_im[NM * NM];
                for (int x = 0; x < NM * NM; x++) { lam_re[x] = 0.0; lam_im[x] = 0.0; }
                for (size_t b = 0; b < n_xi; b++) {
                    const double xi = xm + xh * gxv(b);
                    const double w_xi = xh * gwxv(b);
                    const double xi_src = s * xi + beta;
                    // shapes {1, sin k xi, -2 sin^2(k xi / 2)} at xi and xi_src
                    double si_re[NM], si_im[NM], sj_re[NM], sj_im[NM];
                    auto shapes = [&](double x, double *re, double *im) {
                        re[0] = 1.0; im[0] = 0.0;
                        if (cplx) {
                            const std::complex<double> arg = k * x;
                            const std::complex<double> sn = std::sin(arg);
                            const std::complex<double> hf = std::sin(0.5 * arg);
                            const std::complex<double> c2 = -2.0 * hf * hf;
                            re[1] = sn.real(); im[1] = sn.imag(); re[2] = c2.real(); im[2] = c2.imag();
                        } else {
                            const double arg = kr * x, hf = std::sin(0.5 * arg);
                            re[1] = std::sin(arg); im[1] = 0.0; re[2] = -2.0 * hf * hf; im[2] = 0.0;
                        }
                    };
                    shapes(xi, si_re, si_im);
                    shapes(xi_src, sj_re, sj_im);
                    for (int pp = 0; pp < NM; pp++)
                        for (int q = 0; q < NM; q++) {
                            const double pr = si_re[pp] * sj_re[q] - si_im[pp] * sj_im[q];
                            const double pi_ = si_re[pp] * sj_im[q] + si_im[pp] * sj_re[q];
                            lam_re[pp * NM + q] += w_xi * pr;
                            lam_im[pp * NM + q] += w_xi * pi_;
                        }
                }
                for (int x = 0; x < NM * NM; x++) {
                    acc_re[x] += lam_re[x] * g_re - lam_im[x] * g_im;
                    acc_im[x] += lam_re[x] * g_im + lam_im[x] * g_re;
                }
            }
        }
        for (int x = 0; x < NM * NM; x++) jv(x / NM, x % NM, p) = std::complex<double>(acc_re[x], acc_im[x]);
    }
    return J;
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
        case 3:
            return seg_seg_full_moments_bspline_swept_kernel<3>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k_array, gl_t, gl_w);
        default:
            throw std::runtime_error(
                "seg_seg_full_moments_bspline_swept: max_d must be 1, 2 or 3 "
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
    double a_ek,
    bool reference,
    double a_ek_src
) {
    const PairOrderLadder ladder = ladder_from_rule(gl_t, gl_w);
    switch (max_d) {
        case 1:
            return seg_seg_full_moments_bspline_kernel_ek<1>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k, ladder,
                group_i, group_j, a_ek, reference, a_ek_src);
        case 2:
            return seg_seg_full_moments_bspline_kernel_ek<2>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k, ladder,
                group_i, group_j, a_ek, reference, a_ek_src);
        case 3:
            return seg_seg_full_moments_bspline_kernel_ek<3>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k, ladder,
                group_i, group_j, a_ek, reference, a_ek_src);
        default:
            throw std::runtime_error(
                "seg_seg_full_moments_bspline_ek: max_d must be 1, 2 or 3 "
                "(add an explicit template instantiation in _accelerators.cpp)");
    }
}


// The distance-adaptive (ladder) twin of the entry above (momwire#1362): the
// extended kernel on the pair-order ladder of momwire#906, with the
// `seg_seg_full_moments_bspline_tiered` ladder contract and this entry's EK
// arguments. Real k only, like every EK off-edge entry.
static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_ek_tiered(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    double k,
    int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tier_n_qp,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_ratio,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_i,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_j,
    double a_ek,
    bool reference,
    double a_ek_src
) {
    const PairOrderLadder ladder =
        ladder_from_arrays(tier_t, tier_w, tier_n_qp, tier_ratio);
    switch (max_d) {
        case 1:
            return seg_seg_full_moments_bspline_kernel_ek<1>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k, ladder,
                group_i, group_j, a_ek, reference, a_ek_src);
        case 2:
            return seg_seg_full_moments_bspline_kernel_ek<2>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k, ladder,
                group_i, group_j, a_ek, reference, a_ek_src);
        case 3:
            return seg_seg_full_moments_bspline_kernel_ek<3>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k, ladder,
                group_i, group_j, a_ek, reference, a_ek_src);
        default:
            throw std::runtime_error(
                "seg_seg_full_moments_bspline_ek_tiered: max_d must be 1, 2 or 3 "
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
    double a_ek,
    bool reference,
    double a_ek_src
) {
    const PairOrderLadder ladder = ladder_from_rule(gl_t, gl_w);
    switch (max_d) {
        case 1:
            return seg_seg_full_moments_bspline_swept_kernel_ek<1>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k_array, ladder,
                group_i, group_j, a_ek, reference, a_ek_src);
        case 2:
            return seg_seg_full_moments_bspline_swept_kernel_ek<2>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k_array, ladder,
                group_i, group_j, a_ek, reference, a_ek_src);
        case 3:
            return seg_seg_full_moments_bspline_swept_kernel_ek<3>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k_array, ladder,
                group_i, group_j, a_ek, reference, a_ek_src);
        default:
            throw std::runtime_error(
                "seg_seg_full_moments_bspline_swept_ek: max_d must be 1, 2 or 3 "
                "(add an explicit template instantiation in _accelerators.cpp)");
    }
}


// The distance-adaptive (ladder) twin of the swept EK entry above
// (momwire#1362): `seg_seg_full_moments_bspline_ek_tiered`'s ladder contract
// over a k_array. The caller trims the ladder for the sweep's electrical
// length (`_ladder_for_block`) per group of k it serves.
static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_swept_ek_tiered(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array,
    int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_w,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tier_n_qp,
    py::array_t<double, py::array::c_style | py::array::forcecast> tier_ratio,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_i,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> group_j,
    double a_ek,
    bool reference,
    double a_ek_src
) {
    const PairOrderLadder ladder =
        ladder_from_arrays(tier_t, tier_w, tier_n_qp, tier_ratio);
    switch (max_d) {
        case 1:
            return seg_seg_full_moments_bspline_swept_kernel_ek<1>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k_array, ladder,
                group_i, group_j, a_ek, reference, a_ek_src);
        case 2:
            return seg_seg_full_moments_bspline_swept_kernel_ek<2>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k_array, ladder,
                group_i, group_j, a_ek, reference, a_ek_src);
        case 3:
            return seg_seg_full_moments_bspline_swept_kernel_ek<3>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k_array, ladder,
                group_i, group_j, a_ek, reference, a_ek_src);
        default:
            throw std::runtime_error(
                "seg_seg_full_moments_bspline_swept_ek_tiered: "
                "max_d must be 1, 2 or 3 "
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
// The dispatch below is a [p][q] TABLE, not a flattened switch, and that is
// load-bearing rather than stylistic (momwire#999).
//
// It was `switch (p * 3 + q)` with nine cases. That is a correct bijection for
// p, q ∈ {0,1,2} and it COLLIDES at degree 3: (0,3) and (1,0) both flatten to
// 3, (1,3) and (2,0) both to 6. So extending the case list alone -- which the
// comment that stood here used to recommend, saying the dispatch "extends
// automatically" when the header is regenerated -- compiles, runs, and returns
// the wrong moment. Measured at a near pair: J(0,3) is 1.302e-06 and the
// collision hands back J(1,0) = 2.191e-03, wrong by a factor of 1683. Half the
// degree-3 pairs collided and half fell through to the throw, which is the
// worst available mix, because a smoke test asserting "degree 3 no longer
// raises" goes green on the wrong numbers.
//
// That comment was also false on its own terms: the header WAS regenerated for
// MAX_D = 3 by momwire#883, a release before #999, and nothing extended
// automatically. Two indices cannot collide, and degree 4 is a row.
// The degree the generated families cover. Mirrors
// `_bspline_static_moments.MAX_D`, and `static_assert`ed against the far
// series' copy so the two C++ constants cannot drift from each other; the
// python side is pinned to `BSPLINE_FAR_MAX_P` by a test.
static constexpr int BSPLINE_MOMENT_MAX_D = 3;
static_assert(BSPLINE_MOMENT_MAX_D == BSPLINE_FAR_MAX_P,
              "the near and far spellings must cover the same degrees");

// Out of range is a throw, not a fall-through: an index with no bound check is
// momwire#999 step 1 one dimension up. It may throw where it stands.
// `_accel_common.h` is emphatic that an exception must never escape an OpenMP
// parallel region, and this file has exactly one (the `#pragma omp parallel`
// at the far end, in a function that does not reach here) -- both dispatches
// run in the serial Toeplitz loops below, inside a released GIL, which unwinds
// correctly.
[[noreturn]] static void bspline_unreachable_pq(int p, int q, const char *who) {
    throw std::runtime_error(
        std::string(who) + ": (p, q) = (" + std::to_string(p) + ", " +
        std::to_string(q) + ") not in [0, " +
        std::to_string(BSPLINE_MOMENT_MAX_D) + "]^2");
}

// NESTED SWITCHES, not a function-pointer table, and the choice is measured.
// A `BsplineMomentFn tab[4][4]` is the obvious spelling and reads better, but
// every call through it is INDIRECT. Measured on this box against a d <= 2
// baseline taken before the change: the pointer table moved all 27 cells,
// nested switches move 2. Both are last-ulp effects of gcc's codegen, not of
// arithmetic -- a wrong closed form differs by a factor of 1683, not 1e-21 --
// but 25 cells bit-identical beats none, and the direct calls cost nothing:
// the two spellings timed the same to under 0.4% (51.14 vs 51.16 ms at
// max_d=1, 128.7 vs 128.9 at max_d=2, N=401).
//
// The residual 2 cells are `D_ek` at max_d = 2, and the cause is the NESTING
// rather than the seven arms it added -- a diagnostic build that kept the
// nesting and dropped the degree-3 arms moved exactly the same 2 cells. The
// moved entries sit at 7.6e-07 of their table's maximum, i.e. deep in the
// four-corner cancellation regime #808 documents, where per-element relative
// error is the wrong bar; against the house bar (max|diff| <= 1e-14 * max
// |table|) the worst is 2.0e-21.
//
// Either spelling is immune to the collision that motivated #999: p and q
// index independently, so there is no stride to get wrong at degree 4 either.
static inline double bspline_moment_J(int p, int q, double alpha, double beta,
                                     double A, double B, double a) {
    switch (p) {
        case 0:
            switch (q) {
                case 0: return J_static_pq_0_0(alpha, beta, A, B, a);
                case 1: return J_static_pq_0_1(alpha, beta, A, B, a);
                case 2: return J_static_pq_0_2(alpha, beta, A, B, a);
                case 3: return J_static_pq_0_3(alpha, beta, A, B, a);
            }
            break;
        case 1:
            switch (q) {
                case 0: return J_static_pq_1_0(alpha, beta, A, B, a);
                case 1: return J_static_pq_1_1(alpha, beta, A, B, a);
                case 2: return J_static_pq_1_2(alpha, beta, A, B, a);
                case 3: return J_static_pq_1_3(alpha, beta, A, B, a);
            }
            break;
        case 2:
            switch (q) {
                case 0: return J_static_pq_2_0(alpha, beta, A, B, a);
                case 1: return J_static_pq_2_1(alpha, beta, A, B, a);
                case 2: return J_static_pq_2_2(alpha, beta, A, B, a);
                case 3: return J_static_pq_2_3(alpha, beta, A, B, a);
            }
            break;
        case 3:
            switch (q) {
                case 0: return J_static_pq_3_0(alpha, beta, A, B, a);
                case 1: return J_static_pq_3_1(alpha, beta, A, B, a);
                case 2: return J_static_pq_3_2(alpha, beta, A, B, a);
                case 3: return J_static_pq_3_3(alpha, beta, A, B, a);
            }
            break;
    }
    bspline_unreachable_pq(p, q, "J_static");
}

static inline double bspline_moment_D_ek(int p, int q, double alpha, double beta,
                                        double A, double B, double a) {
    switch (p) {
        case 0:
            switch (q) {
                case 0: return D_ek_pq_0_0(alpha, beta, A, B, a);
                case 1: return D_ek_pq_0_1(alpha, beta, A, B, a);
                case 2: return D_ek_pq_0_2(alpha, beta, A, B, a);
                case 3: return D_ek_pq_0_3(alpha, beta, A, B, a);
            }
            break;
        case 1:
            switch (q) {
                case 0: return D_ek_pq_1_0(alpha, beta, A, B, a);
                case 1: return D_ek_pq_1_1(alpha, beta, A, B, a);
                case 2: return D_ek_pq_1_2(alpha, beta, A, B, a);
                case 3: return D_ek_pq_1_3(alpha, beta, A, B, a);
            }
            break;
        case 2:
            switch (q) {
                case 0: return D_ek_pq_2_0(alpha, beta, A, B, a);
                case 1: return D_ek_pq_2_1(alpha, beta, A, B, a);
                case 2: return D_ek_pq_2_2(alpha, beta, A, B, a);
                case 3: return D_ek_pq_2_3(alpha, beta, A, B, a);
            }
            break;
        case 3:
            switch (q) {
                case 0: return D_ek_pq_3_0(alpha, beta, A, B, a);
                case 1: return D_ek_pq_3_1(alpha, beta, A, B, a);
                case 2: return D_ek_pq_3_2(alpha, beta, A, B, a);
                case 3: return D_ek_pq_3_3(alpha, beta, A, B, a);
            }
            break;
    }
    bspline_unreachable_pq(p, q, "D_ek");
}

static double J_static_dispatch(int p, int q,
                                double alpha, double beta,
                                double A, double B, double a) {
    // The far branch needs its own check: `bspline_J_static_far` is
    // deliberately total -- its guard lives in the py wrapper, because the leaf
    // runs inside a released GIL -- so an out-of-range (p, q) reaching it would
    // be answered rather than refused. The near branch checks itself, in the
    // nested switch's fall-through. Both regimes share one domain.
    if (p < 0 || p > BSPLINE_MOMENT_MAX_D || q < 0 || q > BSPLINE_MOMENT_MAX_D) {
        bspline_unreachable_pq(p, q, "J_static");
    }
    // The closed forms below are sympy's, and they are the NEAR-field half
    // (momwire#808): the value decays like h^5/D while every term in a
    // four-corner closed form grows like D^5, so at 401 segments the (2, 2)
    // moment between the two ends of an edge comes out 2.94e+01 relative --
    // no correct digits. Past `BSPLINE_FAR_RATIO` the centred multipole
    // series is the spelling instead, and it measures 3e-14 there. Same
    // predicate, same truncation, same stepping as the numpy twin.
    if (bspline_far_ratio(alpha, beta, A, B, a) <= BSPLINE_FAR_RATIO) {
        return bspline_J_static_far(p, q, alpha, beta, A, B, a);
    }
    return bspline_moment_J(p, q, alpha, beta, A, B, a);
}

// `J_static_dispatch` with the far series' delta-independent half supplied by
// the caller (momwire#1006). Same predicate, same two regimes, same values --
// the only change is that the coefficient vector is computed once per (p, q)
// instead of once per offset. See `bspline_far_coeffs` for why that is legal
// on a uniform edge and why it is worth 47x on the branch that serves almost
// every offset.
// NOTHROW: the (p, q) range check is the caller's, made ONCE before the offset
// loop, because p and q are invariant across it. That is what lets the loop be
// an OpenMP region: `_accel_common.h` is emphatic that an exception must never
// escape one, and the house answer is the drain pattern (set a flag, continue,
// throw after). Draining is the right tool when the condition can BECOME true
// mid-loop, which is what cancellation does. Here it cannot: the only throw was
// a bound on loop-invariant indices, so hoisting removes the throw site from
// the region entirely rather than arranging to survive it. There is nothing
// left inside to drain, and no "unreachable" claim that could rot -- the call
// is not there.
static double J_static_dispatch_coeffs(int p, int q,
                                       double alpha, double beta,
                                       double A, double B, double a,
                                       const double *coef) {
    if (bspline_far_ratio(alpha, beta, A, B, a) <= BSPLINE_FAR_RATIO) {
        return bspline_J_static_far_with_coeffs(alpha, beta, A, B, a, coef);
    }
    return bspline_moment_J(p, q, alpha, beta, A, B, a);
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
    return bspline_moment_D_ek(p, q, alpha, beta, A, B, a);
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
// That was the mechanism momwire#1194 removed: the GCC/clang builds now pass
// -ffp-contract=off (setup.py), so an a*b+c rounds twice as written whatever
// the inliner decides, and a changed inlining decision no longer changes the
// arithmetic. Cross-COMPILER and cross-machine bit equality is still not
// promised (libm and libmvec differ between glibc versions, and MSVC builds
// with /fp:fast), so it is still not one to pin (antennaknobs#253). What IS
// armored, and stays armored, is the within-build claim the tests actually
// make: EK-off is the same code path and the same bits as the default, and
// no EK code is entered to produce it.
template <bool EK>
static py::array_t<double>
seg_seg_static_moments_bspline_table_impl(double h, double a, size_t N,
                                          int max_d, double a_ek) {
    if (max_d < 0 || max_d > BSPLINE_MOMENT_MAX_D) {
        throw std::runtime_error("max_d out of range [0, " +
                                 std::to_string(BSPLINE_MOMENT_MAX_D) + "]");
    }
    size_t NM = (size_t)(max_d + 1);
    size_t n_delta = 2 * N - 1;
    py::array_t<double> out({NM, NM, n_delta});
    auto v = out.mutable_unchecked<3>();
    py::gil_scoped_release release;
    const double inv_4pi = 1.0 / (4.0 * M_PI);
    for (size_t p = 0; p < NM; p++) {
        for (size_t q = 0; q < NM; q++) {
            // Hoisted out of the offset loop (momwire#1006): on this uniform
            // edge every offset has h1 = h2 = h, so the far series' whole
            // coefficient vector is the same for all 2N-1 of them.
            //
            // The range check happens HERE, once, outside the parallel region
            // below -- see `J_static_dispatch_coeffs`. p and q are invariant
            // across offsets, so this is the only place it can fire.
            if (p > (size_t)BSPLINE_MOMENT_MAX_D || q > (size_t)BSPLINE_MOMENT_MAX_D) {
                bspline_unreachable_pq((int)p, (int)q, "J_static");
            }
            double coef[BSPLINE_FAR_TERMS + 1];
            bspline_far_coeffs((int)p, (int)q, h, h, coef);
            // The offsets are independent and, since the hoist, this loop IS
            // the cost of a table.
            //
            // NO DRAIN HERE, AND THE REASON IS NOT THAT CANCELLATION DOES NOT
            // MATTER -- it is that this entry point has no cancel_flag to poll.
            //
            // What the flag is FOR: a user changes a knob in the app while a
            // solve is running, so the in-flight run must stop and a new one be
            // issued. The twelve entry points that take a `cancel_flag` are the
            // O(N^2) fills where that latency is felt, and they use
            // `_accel_common.h`'s drain -- set a flag, `continue` so the
            // remaining iterations are no-ops, throw AFTER the loop, because an
            // exception must never escape an OpenMP region and cancellation can
            // become true part-way through one.
            //
            // The static-moment table is NOT one of those: `bspline.py` calls it
            // separately from the cancellable fills, so it is an uncancellable
            // window inside an otherwise-cancellable solve. Adding a drain here
            // would mean threading a cancel_flag through this entry point and
            // its Python caller, which is a real change and not this one.
            //
            // What #1006 did do is make that window small. Table build at
            // max_d=2, this box:
            //
            //     N=81    25.6 ms -> 0.44 ms
            //     N=401  130.2 ms -> 1.98 ms
            //     N=801  264.9 ms -> 8.84 ms
            //
            // A knob-change used to wait out a quarter of a second on a long
            // edge; now it waits ~9 ms. That is why the missing drain is
            // tolerable rather than why it is unnecessary.
            //
            // Separately: nothing in this body can throw, so the region is safe
            // as written. The only throw it ever had was `bspline_unreachable_pq`,
            // a bound on (p, q), which do not vary across offsets -- so the check
            // sits above the region and the throw site is not in here at all.
            // That is stronger than a drain for THAT hazard, and it is not a
            // substitute for cancellation, which is a different thing. If you add
            // anything to this body that can throw, or a cancel_flag to poll,
            // then the drain becomes necessary and this comment is wrong.
            #pragma omp parallel for schedule(static)
            for (long long di_s = 0; di_s < (long long)n_delta; di_s++) {
                const size_t di = (size_t)di_s;
                long long delta = (long long)di - (long long)(N - 1);
                double alpha = 0.0;
                double beta = h;
                double A_ = (double)delta * h;
                double B_ = ((double)delta + 1.0) * h;
                double val = J_static_dispatch_coeffs((int)p, (int)q, alpha, beta,
                                                      A_, B_, a, coef);
                if (EK) {
                    val = val + D_ek_dispatch((int)p, (int)q, alpha, beta, A_, B_,
                                              a_ek);
                }
                v(p, q, di) = val * inv_4pi;
            }
        }
    }
    return out;
}

template <bool EK>
static py::array_t<double>
seg_seg_static_moments_bspline_uniform_impl(double h, double a, size_t N,
                                            int max_d, double a_ek,
                                            uintptr_t cancel_flag,
                                            uintptr_t poll_probe = 0) {
    if (max_d < 0 || max_d > BSPLINE_MOMENT_MAX_D) {
        throw std::runtime_error("max_d out of range [0, " +
                                 std::to_string(BSPLINE_MOMENT_MAX_D) + "]");
    }
    size_t NM = (size_t)(max_d + 1);
    py::array_t<double> out({NM, NM, N, N});
    auto v = out.mutable_unchecked<4>();

    // The gather below is O(N^2) and it is the WINDOW A KNOB CHANGE WAITS OUT.
    // Measured after momwire#1006: table build 4.4 ms at N=3201 against 107 ms
    // for the whole call -- the gather is 85% of it at N=801, 92% at 1601, 96%
    // at 3201, and the total grows quadratically (420 ms at N=6401). So this is
    // where the poll belongs; polling only the table build would drain 4% of
    // the wait.
    MW_CANCEL_SETUP(cancel_flag);

    // THE POLL PROBE (momwire#1158) -- a test seam, 0 in production. It is the
    // address of an int64[2]: probe[0] = k, probe[1] = a counter of the flag
    // reads this gather makes. On the k-th read the probe writes the cancel
    // flag itself, just before the poll reads it, so "a flag raised while the
    // gather is running is seen at the next row" is tested at a known row
    // instead of by racing a wall-clock timer against the gather (which the
    // macOS runner lost). The flag is only ever read here as raw memory, never
    // through Python, so a counting Python token cannot stand in for this.
    int64_t *probe = reinterpret_cast<int64_t *>(poll_probe);
    volatile int32_t *probe_flag =
        reinterpret_cast<volatile int32_t *>(cancel_flag);

    // Phase 0: release the GIL for the heavy compute region below.
    py::gil_scoped_release release;

    // 2N-1 unique Toeplitz values per moment, indexed by Δ = j - i ∈ [-(N-1), N-1].
    // delta_idx = Δ + (N - 1) ∈ [0, 2N-2].
    size_t n_delta = 2 * N - 1;

    // ONE HOME FOR THE CLOSED FORMS (momwire#968). The table loop used to be
    // written out here as well as in the table producer below, and the
    // duplicate cost bit-identity: with two copies in the translation unit,
    // GCC made different inlining choices for `J_static_dispatch` in each and
    // the square answer moved by 5.3e-15 relative on a handful of entries
    // against a rebuild of the pre-change source. Nothing about the algebra
    // changed — but "bit-identical" is the contract this entry point carries
    // (momwire#762), and a second copy of a 9 x (2N-1) sympy-derived
    // evaluation is exactly the kind of thing a compiler is free to schedule
    // differently. Calling the producer keeps one copy, so there is nothing to
    // schedule two ways.
    std::vector<double> table(NM * NM * n_delta);
    {
        py::gil_scoped_acquire acquire;
        auto tab = seg_seg_static_moments_bspline_table_impl<EK>(h, a, N, max_d,
                                                                 a_ek);
        auto t = tab.template unchecked<3>();
        for (size_t p = 0; p < NM; p++) {
            for (size_t q = 0; q < NM; q++) {
                for (size_t di = 0; di < n_delta; di++) {
                    table[(p * NM + q) * n_delta + di] = t(p, q, di);
                }
            }
        }
    }

    // Gather: v(p, q, i, j) = table[p, q, j - i + (N - 1)]
    //
    // Polled per ROW, not per element: a row is N copies, so the check costs
    // O(1) per O(N) work, and the worst-case latency after a knob change is one
    // row -- 6401 doubles, microseconds. Per element would be a branch in the
    // innermost loop of a memcpy.
    for (size_t p = 0; p < NM; p++) {
        for (size_t q = 0; q < NM; q++) {
            const double *row = &table[(p * NM + q) * n_delta];
            for (size_t i = 0; i < N; i++) {
                // Counted only while the poll below still reads the flag: once
                // aborted, MW_CANCEL_POLL skips on its first test.
                if (probe && !pysim_aborted.load(std::memory_order_relaxed) &&
                    ++probe[1] == probe[0] && probe_flag) {
                    *probe_flag = 1;
                }
                MW_CANCEL_POLL();
                for (size_t j = 0; j < N; j++) {
                    size_t di = (size_t)((long long)j - (long long)i + (long long)(N - 1));
                    v(p, q, i, j) = row[di];
                }
            }
        }
    }
    MW_THROW_IF_ABORTED();
    return out;
}

static py::array_t<double>
seg_seg_static_moments_bspline_uniform(double h, double a, size_t N, int max_d,
                                       uintptr_t cancel_flag = 0,
                                       uintptr_t poll_probe = 0) {
    return seg_seg_static_moments_bspline_uniform_impl<false>(h, a, N, max_d, 0.0,
                                                              cancel_flag,
                                                              poll_probe);
}

static py::array_t<double>
seg_seg_static_moments_bspline_uniform_ek(double h, double a, size_t N, int max_d,
                                          double a_ek,
                                          uintptr_t cancel_flag = 0) {
    return seg_seg_static_moments_bspline_uniform_impl<true>(h, a, N, max_d, a_ek,
                                                             cancel_flag);
}

// THE TABLE ALONE (momwire#968). A windowed caller wants rows [r0, r1) of the
// gather, and the first spelling of this exposed exactly that — an
// `..._uniform_window(h, a, N, max_d, r0, r1)`. Measured, it was 3.7x SLOWER
// across a whole edge than the single square call, because each window rebuilt
// the 2N-1 Toeplitz table: 9 x 8,001 sympy-derived closed forms per window at
// d=2, nine times over on the ladder's finest rung. The gather is trivial; the
// table is the work.
//
// So the table is what crosses the boundary. It is (NM, NM, 2N-1) — 576 kB at
// N = 4001 against the 2.3 GB gather it feeds — and the caller gathers whatever
// rows it wants from it in numpy. Bit-identical to the square call by
// construction: same table, and a gather copies values rather than computing
// them.
static py::array_t<double>
seg_seg_static_moments_bspline_uniform_table(double h, double a, size_t N,
                                             int max_d) {
    return seg_seg_static_moments_bspline_table_impl<false>(h, a, N, max_d, 0.0);
}

static py::array_t<double>
seg_seg_static_moments_bspline_uniform_ek_table(double h, double a, size_t N,
                                                int max_d, double a_ek) {
    return seg_seg_static_moments_bspline_table_impl<true>(h, a, N, max_d, a_ek);
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



// The far series on its own, exposed so that its DOMAIN and its VALUES can be
// gated directly against `_bspline_static_far.J_static_far` (momwire#999).
//
// Reaching it through a table entry point cannot do that job: those stop at
// max_d = 2, so the p = 3 arm of this family -- the arm that read a 3x3
// binomial table out of bounds from momwire#883 until #999 -- had no path from
// a test at all. A leaf with no caller a test can reach is a leaf that goes
// wrong quietly, which is what happened here for a release.
//
// Validation sits in this wrapper rather than in `bspline_J_static_far`
// itself: the leaf runs inside the Toeplitz builders' `gil_scoped_release`
// region, and every other entry point in this file validates BEFORE releasing
// the GIL. Keeping the leaf total preserves that.
static double bspline_j_static_far_py(int p, int q, double alpha, double beta,
                                      double A, double B, double a) {
    if (p < 0 || p > BSPLINE_FAR_MAX_P || q < 0 || q > BSPLINE_FAR_MAX_P) {
        throw std::runtime_error(
            "bspline_j_static_far: (p, q) = (" + std::to_string(p) + ", " +
            std::to_string(q) + ") not in [0, " +
            std::to_string(BSPLINE_FAR_MAX_P) + "]^2");
    }
    return bspline_J_static_far(p, q, alpha, beta, A, B, a);
}


// The two NEAR dispatches, exposed on the same reasoning as
// `bspline_j_static_far` above (momwire#999). Without these the only path to
// `J_static_dispatch` / `D_ek_dispatch` is a table entry point, and those stop
// at max_d = 2 -- so the degree-3 half of the generated families, which has
// been compiled into this .so since momwire#883, had no caller a test could
// reach. That is how the `p * 3 + q` collision could have shipped: not a weak
// gate, but no gate, because nothing could call the code.
//
// These take the scalar path and never release the GIL, so the dispatch's own
// range check is the only one needed.
static double bspline_j_static_moment_py(int p, int q, double alpha, double beta,
                                         double A, double B, double a) {
    return J_static_dispatch(p, q, alpha, beta, A, B, a);
}

static double bspline_d_ek_moment_py(int p, int q, double alpha, double beta,
                                     double A, double B, double a) {
    return D_ek_dispatch(p, q, alpha, beta, A, B, a);
}


void register_bspline(py::module_ &m) {

    // Read by momwire._accel so the Python routing guard cannot drift from the
    // kernels' real ceiling (momwire#769).
    m.attr("BSPLINE_MAX_N_QP") = py::int_(BSPLINE_MAX_N_QP);
    m.attr("BSPLINE_SAME_EDGE_MAX_N_QP") = py::int_(BSPLINE_SAME_EDGE_MAX_N_QP);

    // Same contract as the two above, for the far series' shared domain:
    // exported so a test can pin it to `_bspline_static_moments.MAX_D` rather
    // than a comment claiming they match (momwire#999).
    m.attr("BSPLINE_FAR_MAX_P") = py::int_(BSPLINE_FAR_MAX_P);
    m.attr("BSPLINE_MOMENT_MAX_D") = py::int_(BSPLINE_MOMENT_MAX_D);

    m.def("bspline_j_static_moment", &bspline_j_static_moment_py,
          "One same-edge static moment, dispatching near/far exactly as the "
          "Toeplitz builders do -- the C++ twin of "
          "_bspline_static_far.J_static_stable. Exposed for the cross-lane "
          "gate, which otherwise cannot reach p or q = 3.",
          py::arg("p"), py::arg("q"), py::arg("alpha"), py::arg("beta"),
          py::arg("A"), py::arg("B"), py::arg("a"));

    m.def("bspline_d_ek_moment", &bspline_d_ek_moment_py,
          "One same-edge EXTENDED-kernel static correction -- the C++ twin of "
          "_bspline_ek_moments.D_ek_moment. No far branch: D_ek has only the "
          "closed form, unlike J. Same reason for existing as the twin above.",
          py::arg("p"), py::arg("q"), py::arg("alpha"), py::arg("beta"),
          py::arg("A"), py::arg("B"), py::arg("a"));

    m.def("bspline_j_static_far", &bspline_j_static_far_py,
          "One same-edge static moment by the centred multipole series -- the "
          "C++ twin of _bspline_static_far.J_static_far, same truncation "
          "(BSPLINE_FAR_TERMS) and same domain guard. Correct only where "
          "bspline_far_ratio <= BSPLINE_FAR_RATIO; the caller checks, exactly "
          "as the numpy twin's docstring says. Exposed for the cross-lane "
          "gate, which otherwise has no path to p = 3.",
          py::arg("p"), py::arg("q"), py::arg("alpha"), py::arg("beta"),
          py::arg("A"), py::arg("B"), py::arg("a"));

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
    m.def("seg_seg_reg_moments_bspline_swept_window",
          &seg_seg_reg_moments_bspline_swept_window,
          "Rectangular twin of seg_seg_reg_moments_bspline_swept "
          "(momwire#968): ONE OBSERVER WINDOW of a same-edge block. R is "
          "(n_row*n_qp, n_col*n_qp), wu_row (n_d, n_row, n_qp) carries the "
          "window's segments and wu_col (n_d, n_col, n_qp) the whole edge's; "
          "the result is (n_k, n_d, n_d, n_row, n_col). Same kernel body as "
          "the square call, which delegates to it with one array on both "
          "axes — the observer and source axes were always independent.",
          py::arg("R"), py::arg("wu_row"), py::arg("wu_col"),
          py::arg("k_array"));
    m.def("seg_seg_reg_moments_bspline_swept_ek_window",
          &seg_seg_reg_moments_bspline_swept_ek_window,
          "Rectangular twin of seg_seg_reg_moments_bspline_swept_ek "
          "(momwire#968). Same windowing contract as "
          "seg_seg_reg_moments_bspline_swept_window, same EK remainder as the "
          "square call.",
          py::arg("R"), py::arg("wu_row"), py::arg("wu_col"),
          py::arg("k_array"), py::arg("a_ek"));
    m.def("seg_seg_full_moments_bspline", &seg_seg_full_moments_bspline,
          "Single-k full-kernel polynomial moment integrals for the B-spline "
          "Galerkin MoM. Returns J of shape (max_d+1, max_d+1, N_i, N_j) "
          "complex. Templated on max_d at compile time; currently "
          "instantiated for max_d in {1, 2}.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k"),
          py::arg("max_d"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("reference") = false);
    m.def("seg_seg_full_moments_bspline_cplx",
          &seg_seg_full_moments_bspline_cplx,
          "In-medium (complex-k) twin of seg_seg_full_moments_bspline "
          "(momwire#778). Same contract and same (max_d+1, max_d+1, N_i, N_j) "
          "output, with k a complex wavenumber k_re + j*k_im, Im k <= 0. The "
          "continuation is exp(-jkR) = exp(k_im*R) * exp(-j*k_re*R): one extra "
          "real exp() per quadrature point, all loops still real-lane SIMD.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k"),
          py::arg("max_d"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("reference") = false);
    m.def("seg_seg_full_moments_bspline_tiered",
          &seg_seg_full_moments_bspline_tiered,
          "Distance-adaptive twin of seg_seg_full_moments_bspline "
          "(momwire#906). Same contract and output, but the Gauss-Legendre "
          "rule is a LADDER: tier 0 (the first tier_n_qp[0] entries of "
          "tier_t / tier_w) is the base order, and tier t >= 1 serves every "
          "pair whose centre distance over the longer segment is >= "
          "tier_ratio[t] (thresholds strictly ascending). One tier reproduces "
          "the plain entry bit for bit.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k"),
          py::arg("max_d"),
          py::arg("tier_t"), py::arg("tier_w"),
          py::arg("tier_n_qp"), py::arg("tier_ratio"),
          py::arg("reference") = false);
    m.def("seg_seg_full_moments_bspline_cplx_tiered",
          &seg_seg_full_moments_bspline_cplx_tiered,
          "In-medium (complex-k) twin of seg_seg_full_moments_bspline_tiered "
          "(momwire#906): the ladder contract above with the #778 complex-k "
          "continuation, Im k <= 0.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k"),
          py::arg("max_d"),
          py::arg("tier_t"), py::arg("tier_w"),
          py::arg("tier_n_qp"), py::arg("tier_ratio"),
          py::arg("reference") = false);
    m.def("seg_seg_full_moments_sinusoidal_tiered",
          &seg_seg_full_moments_sinusoidal_tiered,
          "The tiered pair moments on the sinusoidal Galerkin basis's folded "
          "shape set {1, sin k xi, cos k xi - 1} (momwire#1354): the B-spline "
          "ladder contract with the shapes evaluated at the nodes in place of "
          "the monomials, (3, 3, N_i, N_j). k complex, Im k <= 0; a real k "
          "takes the real-shape instantiation.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k"),
          py::arg("tier_t"), py::arg("tier_w"),
          py::arg("tier_n_qp"), py::arg("tier_ratio"));
    m.def("seg_seg_full_moments_sinusoidal_tiered_ek",
          &seg_seg_full_moments_sinusoidal_tiered_ek,
          "The extended-kernel twin of seg_seg_full_moments_sinusoidal_tiered "
          "(momwire#1362): NEC Eq 89's coaxial factor on G for the pairs whose "
          "group labels match. Real k only.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k"),
          py::arg("tier_t"), py::arg("tier_w"),
          py::arg("tier_n_qp"), py::arg("tier_ratio"),
          py::arg("group_i"), py::arg("group_j"), py::arg("a_ek"),
          py::arg("b_j") = py::none());
    m.def("assemble_Z_sinusoidal_windowed", &assemble_Z_sinusoidal_windowed,
          "Accumulate one window of sinusoidal pair moments into Z "
          "(momwire#1354): the windowed B-spline assembler with the basis as "
          "CSR-by-segment coefficient tables (coef, dcoef) on the folded "
          "shape set, per-pair complex weights optional. Threads own basis "
          "rows.",
          py::arg("J"), py::arg("rows_seg"), py::arg("cols_seg"),
          py::arg("starts"), py::arg("jbasis"), py::arg("coef"), py::arg("dcoef"),
          py::arg("tangents_rows"), py::arg("tangents_cols"),
          py::arg("c_a"), py::arg("c_phi"),
          py::arg("w_a") = py::none(), py::arg("w_phi") = py::none(),
          py::arg("Z"), py::arg("cancel_flag") = 0);
    m.def("field_pair_moments_sinusoidal", &field_pair_moments_sinusoidal,
          "Field-form pair moments sum_{q,r} W_obs[p,i,q] F[iq,jr] W_src[P,j,r] "
          "with complex shape weights, (3, 3, n_obs, n_src) (momwire#1354).",
          py::arg("F"), py::arg("W_obs"), py::arg("W_src"));
    m.def("parallel_pair_moments_sinusoidal", &parallel_pair_moments_sinusoidal,
          "The sinusoidal fill's parallel-pair reduction (momwire#1354): "
          "(3, 3, n) moments of n parallel segment pairs by the sinh-substituted "
          "separation integral, a pair per thread.",
          py::arg("c_i"), py::arg("t_i"), py::arg("h_i"), py::arg("c_j"), py::arg("t_j"),
          py::arg("h_j"), py::arg("a2"), py::arg("k"), py::arg("gt"), py::arg("gw"),
          py::arg("gx"), py::arg("gwx"), py::arg("a_ek") = py::none(),
          py::arg("b_ek") = py::none());
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
          "None` branch, evaluated eagerly instead of via np.where. "
          "reference=True walks every pair alone where the AVX2 lane path "
          "is compiled (momwire#1362), the gate that path is held to.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k"),
          py::arg("max_d"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("group_i"), py::arg("group_j"), py::arg("a_ek"),
          py::arg("reference") = false, py::arg("a_ek_src") = -1.0);
    m.def("seg_seg_full_moments_bspline_ek_tiered",
          &seg_seg_full_moments_bspline_ek_tiered,
          "Distance-adaptive twin of seg_seg_full_moments_bspline_ek "
          "(momwire#1362): the pair-order ladder contract of "
          "seg_seg_full_moments_bspline_tiered (tier 0 the base order, tier "
          "t >= 1 for pairs at centre distance over the longer segment >= "
          "tier_ratio[t]) with the EK group labels and radius. The coaxial "
          "factor is applied at whatever order a pair runs; one tier "
          "reproduces seg_seg_full_moments_bspline_ek bit for bit.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k"),
          py::arg("max_d"),
          py::arg("tier_t"), py::arg("tier_w"),
          py::arg("tier_n_qp"), py::arg("tier_ratio"),
          py::arg("group_i"), py::arg("group_j"), py::arg("a_ek"),
          py::arg("reference") = false, py::arg("a_ek_src") = -1.0);
    m.def("seg_seg_full_moments_bspline_swept_ek",
          &seg_seg_full_moments_bspline_swept_ek,
          "Batched (swept-k) twin of seg_seg_full_moments_bspline_ek "
          "(momwire#270 unit 2). Same (max_d+1, max_d+1, N_i, N_j) "
          "per-(i,j) geometry hoist as seg_seg_full_moments_bspline_swept, "
          "plus the EK eligibility mask and the k-independent T1/T2 halves "
          "of the coaxial factor hoisted out of the k loop the same way the "
          "same-edge swept twin hoists them. Returns (n_k, max_d+1, max_d+1, "
          "N_i, N_j) complex. reference=True walks every pair alone where the "
          "AVX2 lane path is compiled (momwire#1362), the gate that path is "
          "held to.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k_array"),
          py::arg("max_d"),
          py::arg("gl_t"), py::arg("gl_w"),
          py::arg("group_i"), py::arg("group_j"), py::arg("a_ek"),
          py::arg("reference") = false, py::arg("a_ek_src") = -1.0);
    m.def("seg_seg_full_moments_bspline_swept_ek_tiered",
          &seg_seg_full_moments_bspline_swept_ek_tiered,
          "Distance-adaptive twin of seg_seg_full_moments_bspline_swept_ek "
          "(momwire#1362): the pair-order ladder contract of "
          "seg_seg_full_moments_bspline_ek_tiered over a k_array. One tier "
          "reproduces seg_seg_full_moments_bspline_swept_ek bit for bit; "
          "reference=True walks every pair alone where the AVX2 lane path is "
          "compiled.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k_array"),
          py::arg("max_d"),
          py::arg("tier_t"), py::arg("tier_w"),
          py::arg("tier_n_qp"), py::arg("tier_ratio"),
          py::arg("group_i"), py::arg("group_j"), py::arg("a_ek"),
          py::arg("reference") = false, py::arg("a_ek_src") = -1.0);
    m.def("seg_seg_static_moments_bspline_uniform",
          &seg_seg_static_moments_bspline_uniform,
          "Closed-form same-edge static-kernel polynomial moments J_pq for a "
          "uniform-h edge with N segments. Uses Toeplitz structure (2N-1 "
          "unique values per moment) and inlined sympy-derived closed forms. "
          "Returns J_static of shape (max_d+1, max_d+1, N, N), with the "
          "1/(4π) prefactor folded in.",
          py::arg("h"), py::arg("a"), py::arg("N"), py::arg("max_d"),
          py::arg("cancel_flag") = 0,
          // Test seam (momwire#1158): see the probe comment in the impl.
          py::arg("poll_probe") = 0);
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
          py::arg("a_ek"), py::arg("cancel_flag") = 0);
    m.def("seg_seg_static_moments_bspline_uniform_table",
          &seg_seg_static_moments_bspline_uniform_table,
          "The 2N-1 Toeplitz TABLE behind "
          "seg_seg_static_moments_bspline_uniform (momwire#968), shape "
          "(max_d+1, max_d+1, 2N-1) with the 1/(4 pi) prefactor folded in. The "
          "square call gathers it into (max_d+1, max_d+1, N, N); a caller "
          "streaming one edge in observer windows gathers whatever rows it "
          "wants instead, and gets the square answer's sub-block bit for bit. "
          "Exposed rather than a row-window gather because the TABLE is the "
          "work: rebuilding it per window measured 3.7x slower across an edge.",
          py::arg("h"), py::arg("a"), py::arg("N"), py::arg("max_d"));
    m.def("seg_seg_static_moments_bspline_uniform_ek_table",
          &seg_seg_static_moments_bspline_uniform_ek_table,
          "EK twin of seg_seg_static_moments_bspline_uniform_table "
          "(momwire#968): each entry carries the D_pq^EK correction the square "
          "EK call adds.",
          py::arg("h"), py::arg("a"), py::arg("N"), py::arg("max_d"),
          py::arg("a_ek"));
    // momwire#1132: the four windowed assemblers take `row_of`, a ROW-COMPACT
    // Z target. Its own flag: a .so built before this answers `row_of=` with a
    // TypeError, so the sector route's compact fill gates on THIS.
    m.attr("windowed_row_of_1132") = true;
    // momwire#1290: whether the windowed assemblers carry the AVX2 lane
    // kernel; either way they take `reference`, which runs the per-entry
    // loop that kernel is gated against.
    m.attr("windowed_lanes_1290") = (bool)MW_WINDOWED_LANES_1290;
    // momwire#1290: whether the off-edge moment kernel (plain, cplx and both
    // tiered entries) carries the AVX2 lane path; either way they take
    // `reference`, which walks every pair alone.
    m.attr("offedge_lanes_1290") = (bool)MW_OFFEDGE_LANES_1290;
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
          "max_d inferred from support_seg. On an AVX2+FMA x86 build a lane "
          "kernel fills eight entries a step; reference=True runs the "
          "per-entry loop instead, the same bits.",
          py::arg("J_chunk"), py::arg("support_seg"),
          py::arg("polys"), py::arg("wA_win"), py::arg("wPhi_win"),
          py::arg("m_idx"), py::arg("n_idx"),
          py::arg("i0"), py::arg("i1"), py::arg("j0"), py::arg("j1"),
          py::arg("omega"), py::arg("eps_"), py::arg("mu_"),
          py::arg("scale"), py::arg("Z"), py::arg("cancel_flag") = 0,
          py::arg("row_of") = py::none(),
          py::arg("reference") = false);
    m.def("assemble_Z_bspline_windowed_cplx_eps", &assemble_Z_bspline_windowed_cplx_eps,
          "In-medium twin of assemble_Z_bspline_windowed (momwire#915): the "
          "same window contract with eps a COMPLEX permittivity, and the same "
          "`reference` switch (momwire#1290).",
          py::arg("J_chunk"), py::arg("support_seg"), py::arg("polys"),
          py::arg("tangents"), py::arg("m_idx"), py::arg("n_idx"),
          py::arg("i0"), py::arg("i1"), py::arg("j0"), py::arg("j1"),
          py::arg("omega"), py::arg("eps"), py::arg("mu"),
          py::arg("Z"), py::arg("cancel_flag") = 0,
          py::arg("row_of") = py::none(),
          py::arg("reference") = false);
    m.def("assemble_Z_bspline_weighted_windowed_cplx_eps",
          &assemble_Z_bspline_weighted_windowed_cplx_eps,
          "In-medium twin of assemble_Z_bspline_weighted_windowed "
          "(momwire#915): the same window contract with eps a COMPLEX "
          "permittivity.",
          py::arg("J_chunk"), py::arg("support_seg"), py::arg("polys"),
          py::arg("wA_win"), py::arg("wPhi_win"), py::arg("m_idx"), py::arg("n_idx"),
          py::arg("i0"), py::arg("i1"), py::arg("j0"), py::arg("j1"),
          py::arg("omega"), py::arg("eps"), py::arg("mu"), py::arg("scale"),
          py::arg("Z"), py::arg("cancel_flag") = 0,
          py::arg("row_of") = py::none(),
          py::arg("reference") = false);
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
          "inferred from support_seg. On the AVX2 build the fill runs eight "
          "entries at a time over cache tiles (momwire#1290); "
          "reference=True runs the per-entry loop instead, the same bits.",
          py::arg("J_chunk"), py::arg("support_seg"),
          py::arg("polys"), py::arg("tangents"),
          py::arg("m_idx"), py::arg("n_idx"),
          py::arg("i0"), py::arg("i1"), py::arg("j0"), py::arg("j1"),
          py::arg("omega"), py::arg("eps_"), py::arg("mu_"),
          py::arg("Z"), py::arg("cancel_flag") = 0,
          py::arg("row_of") = py::none(),
          py::arg("reference") = false);
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
    m.def("assemble_Z_bspline_cplx_eps", &assemble_Z_bspline_cplx_eps,
          "In-medium twin of assemble_Z_bspline (momwire#910): the same "
          "contract with eps a COMPLEX permittivity, for the buried fill's "
          "charge term divided by eps0 * eps~.",
          py::arg("J"), py::arg("support_seg"),
          py::arg("polys"), py::arg("td_all"),
          py::arg("omega"), py::arg("eps"), py::arg("mu"),
          py::arg("max_d"),
          py::arg("cancel_flag") = 0);
    m.def("assemble_Z_bspline_weighted_cplx_eps",
          &assemble_Z_bspline_weighted_cplx_eps,
          "In-medium twin of assemble_Z_bspline_weighted (momwire#910): the "
          "same contract with eps a COMPLEX permittivity, for the buried "
          "image block.",
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
          py::arg("cancel_flag") = 0, py::arg("reference") = false);
    m.def("bspline_assemble_offedge_block_ek_tiered",
          &bspline_assemble_offedge_block_ek_tiered,
          "Distance-adaptive twin of bspline_assemble_offedge_block_ek "
          "(momwire#1362): the pair-order ladder contract of "
          "seg_seg_full_moments_bspline_tiered, with the per-pair phase guard "
          "answered in the kernel -- a tier of fewer than limited_below points "
          "serves a segment pair only when phase_ok_I[si] and phase_ok_J[sj] "
          "are both set (uint8 per segment of each union; empty = all pass). "
          "One tier reproduces bspline_assemble_offedge_block_ek bit for bit.",
          py::arg("supp_I"), py::arg("polys_I"), py::arg("segl_I"),
          py::arg("segr_I"), py::arg("tan_I"),
          py::arg("supp_J"), py::arg("polys_J"), py::arg("segl_J"),
          py::arg("segr_J"), py::arg("tan_J"),
          py::arg("a_squared"), py::arg("k"), py::arg("omega"),
          py::arg("eps"), py::arg("mu"), py::arg("max_d"),
          py::arg("tier_t"), py::arg("tier_w"),
          py::arg("tier_n_qp"), py::arg("tier_ratio"),
          py::arg("group_I"), py::arg("group_J"), py::arg("a_ek"),
          py::arg("phase_ok_I"), py::arg("phase_ok_J"), py::arg("limited_below"),
          py::arg("cancel_flag") = 0, py::arg("reference") = false);
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
          py::arg("cancel_flag") = 0, py::arg("reference") = false);
    m.def("bspline_assemble_offedge_block_refl_ek_tiered",
          &bspline_assemble_offedge_block_refl_ek_tiered,
          "bspline_assemble_offedge_block_ek_tiered's ladder and phase-guard "
          "contract on the reflection-coefficient image block of "
          "bspline_assemble_offedge_block_refl_ek (momwire#1362). One tier "
          "reproduces that entry bit for bit.",
          py::arg("supp_I"), py::arg("polys_I"), py::arg("segl_I"),
          py::arg("segr_I"), py::arg("tan_I"),
          py::arg("supp_J"), py::arg("polys_J"), py::arg("segl_J"),
          py::arg("segr_J"), py::arg("tan_J"),
          py::arg("a_squared"), py::arg("k"), py::arg("omega"),
          py::arg("eps"), py::arg("mu"), py::arg("max_d"),
          py::arg("tier_t"), py::arg("tier_w"),
          py::arg("tier_n_qp"), py::arg("tier_ratio"),
          py::arg("group_I"), py::arg("group_J"), py::arg("a_ek"),
          py::arg("eps_t"), py::arg("phi_c0"), py::arg("phi_c1"),
          py::arg("phase_ok_I"), py::arg("phase_ok_J"), py::arg("limited_below"),
          py::arg("cancel_flag") = 0, py::arg("reference") = false);
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
}

