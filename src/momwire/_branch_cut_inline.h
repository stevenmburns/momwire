// The Sommerfeld branch cut, in one place (momwire#714).
//
// `_sommerfeld._gamma`: (lam^2 - k^2)^{1/2} with vertical cuts running DOWN
// from +k and UP from -k. Transcribed as the two-sqrt product, NOT collapsed
// to sqrt(lam^2 - k^2): the collapsed form has its cut on the SEGMENT between
// the branch points, and the legitimacy of both the head's first-quadrant
// detour and the rotated rays is that they cross neither of the vertical ones.
// The spelling IS the branch choice.
//
// That last sentence is why this file exists. Until #714 the spelling stood as
// three formula-equivalent internal-linkage copies -- `somm::gam`,
// `mw568_below::gamma_cut`, `mw680::gamma_cut` -- across two extensions. Two
// copies of a branch choice can drift; one cannot. A drifted copy would not
// fail loudly: it would move the cut in ONE translation unit and return
// wrong-sheet integrals that only that section's gates could catch.
//
// Deliberately leaner than `_accel_somm_proj_inline.h` (which #714 first
// proposed as the home): one of the three call sites is in the SEPARATE
// `_near_interface_accel` extension, whose include set is kept minimal, so
// this header takes `<complex>` and nothing else -- no pybind11, no OpenMP
// declarations, no cancellation machinery. Same discipline as
// `_contour_engine_inline.h`: it must not depend on the includer having
// defined `_USE_MATH_DEFINES` first (MSVC).
//
// `static inline` keeps the monolith's semantics exactly: each including TU
// gets its own internal-linkage copy, so section-local inlining is preserved
// and there is no ODR surface across the two extensions.
#pragma once

#include <complex>

namespace mw_branch {

// The imaginary unit. Named as in the call sites this replaces (`CI` in the
// somm family, `MW_BJ` in the mw568/mw680 twins -- both were (0.0, 1.0)).
static const std::complex<double> MW_BRANCH_J(0.0, 1.0);

static inline std::complex<double> gamma_cut(const std::complex<double> &lam,
                                             const std::complex<double> &k) {
    return std::sqrt(-MW_BRANCH_J * (lam - k)) *
           std::sqrt(MW_BRANCH_J * (lam + k));
}

}  // namespace mw_branch
