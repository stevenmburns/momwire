#include "_accel_common.h"

#include "_bspline_static_moments_inline.h"
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
    if (n_qp_in > BSPLINE_MAX_N_QP) {
        throw std::runtime_error("n_qp > " + std::to_string(BSPLINE_MAX_N_QP)
                                 + " not supported (L1-sized stack scratch);"
                                 " the caller should have taken the numpy path");
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
    if (n_qp > BSPLINE_MAX_N_QP) {
        throw std::runtime_error("n_qp > " + std::to_string(BSPLINE_MAX_N_QP)
                                 + " not supported (L1-sized stack scratch);"
                                 " the caller should have taken the numpy path");
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
    if (n_qp_in > BSPLINE_MAX_N_QP) {
        throw std::runtime_error("n_qp > " + std::to_string(BSPLINE_MAX_N_QP)
                                 + " not supported (L1-sized stack scratch);"
                                 " the caller should have taken the numpy path");
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
    if (n_qp > BSPLINE_MAX_N_QP) {
        throw std::runtime_error("n_qp > " + std::to_string(BSPLINE_MAX_N_QP)
                                 + " not supported (L1-sized stack scratch);"
                                 " the caller should have taken the numpy path");
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
    if (n_qp > BSPLINE_MAX_N_QP) {
        throw std::runtime_error("n_qp > " + std::to_string(BSPLINE_MAX_N_QP)
                                 + " not supported (L1-sized stack scratch);"
                                 " the caller should have taken the numpy path");
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
    if (n_qp > BSPLINE_MAX_N_QP) {
        throw std::runtime_error("n_qp > " + std::to_string(BSPLINE_MAX_N_QP)
                                 + " not supported (L1-sized stack scratch);"
                                 " the caller should have taken the numpy path");
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



void register_bspline(py::module_ &m) {

    // Read by momwire._accel so the Python routing guard cannot drift from the
    // kernels' real ceiling (momwire#769).
    m.attr("BSPLINE_MAX_N_QP") = py::int_(BSPLINE_MAX_N_QP);

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
}

