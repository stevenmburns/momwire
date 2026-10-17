// The Sommerfeld near-field remainder: see `_somm_field_core.h`.
//
// It replaced `_field_point`'s numpy body (kept as the fallback without the
// accelerator) because that body's last bits depended on the CPU: numpy's
// `arctan2` takes an AVX-512 SVML loop where the CPU has one and libm's
// `atan2` elsewhere (measured: 15654 of 200000 random pairs differ on an
// AVX-512 machine, none on an AVX2 one), and its stencil was an einsum whose
// summation order is numpy's choice. Here every operation is spelled, libm
// answers every transcendental, and the sum over sources runs in source
// order, so one build gives one answer on every x86-64 CPU -- and the
// standalone executable, which links this file, gives the same.
//
// `_mw_core.h` first, for the TU-wide `omp declare simd` set (momwire#1194).
#include "_mw_core.h"
#include "_somm_field_core.h"
#include "_accel_somm_proj_inline.h"

namespace somm_core {

namespace {

using cd = std::complex<double>;

// E's remainder at one point (x, y, z), into e[3].
void field_at(const somm_proj::GridView &G, double x, double y, double z, std::ptrdiff_t n_src,
              const double *src_mid, const cd *src_moment, double ground_z, double k, cd e[3]) {
    cd ax(0.0, 0.0), ay(0.0, 0.0), az(0.0, 0.0);
    for (std::ptrdiff_t s = 0; s < n_src; ++s) {
        const double *m = src_mid + 3 * s;
        const cd *p = src_moment + 3 * s;
        double dx, dy, rho, hh;
        somm_proj::pair_geometry(ground_z, x, y, z, m[0], m[1], m[2], dx, dy, rho, hh);
        cd surf[4], g;
        somm_proj::proj_core(G, k, rho, hh, surf, g);
        const cd IrhoV = surf[0], IzV = surf[1], IrhoH = surf[2], IphiH = surf[3];
        // rho -> 0: any horizontal unit vector serves (I_rho^H(90 deg) =
        // -I_phi^H there), and a complex moment has no direction, so (1, 0).
        const bool safe_r = rho > G.tiny;
        const double inv_rho = safe_r ? 1.0 / rho : 0.0;
        const double dhx = safe_r ? dx * inv_rho : 1.0;
        const double dhy = safe_r ? dy * inv_rho : 0.0;
        const cd cph = p[0] * dhx + p[1] * dhy;
        const cd sph = p[0] * dhy - p[1] * dhx;
        const cd e_rho = g * (p[2] * IrhoV + cph * IrhoH);
        const cd e_phi = g * (sph * IphiH);
        const cd e_z = g * (p[2] * IzV - cph * IrhoV);
        ax += dhx * e_rho - dhy * e_phi;
        ay += dhy * e_rho + dhx * e_phi;
        az += e_z;
    }
    e[0] = ax;
    e[1] = ay;
    e[2] = az;
}

// `field_at`'s reads at one point, marked (`proj_core`'s address).
void field_at_nodes(const somm_proj::GridView &G, double x, double y, double z,
                    std::ptrdiff_t n_src, const double *src_mid, double ground_z,
                    somm_proj::NodeMarks &marks) {
    for (std::ptrdiff_t s = 0; s < n_src; ++s) {
        const double *m = src_mid + 3 * s;
        double dx, dy, rho, hh, r1, theta, fr, ft;
        int i0, j0;
        somm_proj::pair_geometry(ground_z, x, y, z, m[0], m[1], m[2], dx, dy, rho, hh);
        somm_proj::proj_r1_theta(G, rho, hh, r1, theta);
        const int reg = somm_proj::proj_address(G, r1, theta, fr, ft, i0, j0);
        marks.mark(reg, i0, j0);
    }
}

// `reflected_h_field`'s six offset points about q, as it forms them.
inline void h_offsets(const double *q, double step, double pts[6][3]) {
    for (int axis = 0; axis < 3; ++axis)
        for (int side = 0; side < 2; ++side) {
            double shift[3] = {0.0, 0.0, 0.0};
            shift[axis] = (side == 0 ? 1.0 : -1.0) * step;
            for (int c = 0; c < 3; ++c) pts[2 * axis + side][c] = q[c] + shift[c];
        }
}

}  // namespace

void reflected_field_nodes(const somm_proj::GridView &G, std::ptrdiff_t n_pts, const double *points,
                           std::ptrdiff_t n_src, const double *src_mid, double ground_z,
                           somm_proj::NodeMarks &marks) {
    for (std::ptrdiff_t i = 0; i < n_pts; ++i) {
        const double *q = points + 3 * i;
        field_at_nodes(G, q[0], q[1], q[2], n_src, src_mid, ground_z, marks);
    }
}

void reflected_h_field_nodes(const somm_proj::GridView &G, std::ptrdiff_t n_pts,
                             const double *points, std::ptrdiff_t n_src, const double *src_mid,
                             double ground_z, double step, somm_proj::NodeMarks &marks) {
    for (std::ptrdiff_t i = 0; i < n_pts; ++i) {
        double pts[6][3];
        h_offsets(points + 3 * i, step, pts);
        for (int f = 0; f < 6; ++f)
            field_at_nodes(G, pts[f][0], pts[f][1], pts[f][2], n_src, src_mid, ground_z, marks);
    }
}

void reflected_field(const somm_proj::GridView &G, std::ptrdiff_t n_pts, const double *points,
                     std::ptrdiff_t n_src, const double *src_mid, const cd *src_moment,
                     double ground_z, double k, cd *out) {
    // Points are independent and each sums its sources in order, so the
    // schedule moves no bit.
    #pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t i = 0; i < n_pts; ++i) {
        const double *q = points + 3 * i;
        field_at(G, q[0], q[1], q[2], n_src, src_mid, src_moment, ground_z, k, out + 3 * i);
    }
}

void reflected_h_field(const somm_proj::GridView &G, std::ptrdiff_t n_pts, const double *points,
                       std::ptrdiff_t n_src, const double *src_mid, const cd *src_moment,
                       double ground_z, double k, double omega, double mu, double step,
                       cd *out) {
    const double two_step = 2.0 * step;
    const double a = omega * mu;
    #pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t i = 0; i < n_pts; ++i) {
        const double *q = points + 3 * i;
        cd f[6][3];
        double pts[6][3];
        h_offsets(q, step, pts);
        for (int p = 0; p < 6; ++p)
            field_at(G, pts[p][0], pts[p][1], pts[p][2], n_src, src_mid, src_moment, ground_z, k,
                     f[p]);
        cd d_e[3][3];
        for (int axis = 0; axis < 3; ++axis)
            for (int c = 0; c < 3; ++c) {
                const cd diff = f[2 * axis][c] - f[2 * axis + 1][c];
                d_e[axis][c] = cd(diff.real() / two_step, diff.imag() / two_step);
            }
        const cd curl[3] = {d_e[1][2] - d_e[2][1], d_e[2][0] - d_e[0][2],
                            d_e[0][1] - d_e[1][0]};
        for (int c = 0; c < 3; ++c) out[3 * i + c] = cd(-curl[c].imag() / a, curl[c].real() / a);
    }
}

}  // namespace somm_core
