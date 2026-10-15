// The Sommerfeld six-integral kernel with no Python (`_somm_kernel_core.cpp`).
#pragma once

#include <complex>

namespace somm_core {

// = `_sommerfeld._six_integrals` for one (rho, h): out = (Vrr, Vzz, Vrz,
// Vr1, V, U). form: 0 auto (rho < 2h -> Bessel), 1 Bessel, 2 Hankel.
// eps_t == 1 gives zeros. lanes: the Gauss nodes four at a time where the
// build has them (momwire#1441 step 3), bit-identical to the walk.
void six_integrals(std::complex<double> eps_t, double k2, double rho, double h, double rtol,
                   int form, std::complex<double> out[6], bool lanes = true);

}  // namespace somm_core
