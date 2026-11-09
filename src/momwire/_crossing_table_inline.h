// The crossing table's per-row arithmetic (momwire#1860 L5, the opt-in
// feature `crossing_table`): the closed-form part it subtracts and the
// tricubic stencil it reads. Header-only and Python-free so that the binding
// (`_crossing_table_core.cpp`) and, at promotion, the executables read the
// same lines. No BLAS and no reductions whose order the build could change.
//
// The approach follows the NEC-4 theory manual sec. 5.1.3: tabulate the
// transmitted kernels on one fixed grid per frequency, after taking out the
// part that is singular as source and observer meet at the interface, and
// store what is left times R so that it goes to zero there. The subtracted
// forms here are NOT the manual's. They are the first two orders of the
// large-lambda expansion of this module's own six spectral factors
// (`_near_interface._core`), with gamma ~ lambda - k^2 / (2 lambda):
//
//   leading:  U 1/R, V A/R, W C L, dzW -C/R, dzpV -A (2 s^2 - rho^2)/R^5,
//             dzpW C/R;
//   next:     U, V (times A), dzW (times -C) and dzpW (times C) gain tau L;
//             dzpV gains -A (tau s/R^3 + c3/R),
//
// where s = z - z', R = hypot(rho, s), L = -ln(s + R) (the 1/lambda moment,
// up to a constant that cancels between build and read), A = 2/(km^2+kp^2),
// C = (km^2-kp^2)/(km^2+kp^2), tau = (kp^2 z - km^2 z')/2 (the exponent's
// 1/lambda term) and c3 = tau^2/2 + kp^2 km^2/(kp^2+km^2) - (kp^2+km^2)/2
// (dzpV's lambda^0 bracket). With both orders out, R (f - S) vanishes
// linearly at R = 0 for all six kernels, which is what lets one coarse grid
// reach a source on the interface.
#pragma once

#include <cmath>
#include <complex>

namespace mw1860 {

using cplx = std::complex<double>;

inline void crossing_singular(double rho, double z, double zp, double kp, cplx km,
                              cplx out[6]) {
    const double s = z - zp;
    const double R = std::hypot(rho, s);
    const double al = kp * kp;
    const cplx be = km * km;
    const cplx sum = al + be;
    const cplx A = 2.0 / sum;
    const cplx C = (be - al) / sum;
    const cplx tau = 0.5 * (al * z - be * zp);
    const double L = -std::log(s + R);
    const double R3 = R * R * R;
    const double R5 = R3 * R * R;
    const cplx c3 = 0.5 * tau * tau + al * be / sum - 0.5 * sum;
    const cplx tL = tau * L;
    out[0] = 1.0 / R + tL;
    out[1] = A / R + A * tL;
    out[2] = C * L;
    out[3] = -C / R - C * tL;
    out[4] = -A * (2.0 * s * s - rho * rho) / R5 - A * (tau * s / R3 + c3 / R);
    out[5] = C / R + C * tL;
}

// The four-point Lagrange stencil on a uniform axis of n >= 4 nodes, spacing
// h: its first node and its weights at t.
inline void crossing_stencil(double t, double h, long n, long &i0, double w[4]) {
    const double u = t / h;
    long i = static_cast<long>(std::floor(u)) - 1;
    if (i < 0) i = 0;
    if (i > n - 4) i = n - 4;
    const double x = u - static_cast<double>(i);
    w[0] = -(x - 1.0) * (x - 2.0) * (x - 3.0) / 6.0;
    w[1] = x * (x - 2.0) * (x - 3.0) / 2.0;
    w[2] = -x * (x - 1.0) * (x - 3.0) / 2.0;
    w[3] = x * (x - 1.0) * (x - 2.0) / 6.0;
    i0 = i;
}

}  // namespace mw1860
