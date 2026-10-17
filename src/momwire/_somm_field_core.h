// The Sommerfeld ground's near-field REMAINDER at observation points, in one
// C++ kernel (`_somm_field_core.cpp`): what `_field_point.reflected_field_at`
// and `reflected_h_field_at` return for the ('above', 'above') regime.
#pragma once

#include <complex>
#include <cstddef>

namespace somm_proj {
struct GridView;
struct NodeMarks;
}

namespace somm_core {

// E's remainder at n_pts points (n_pts, 3) from n_src element midpoints
// (n_src, 3) carrying complex moments (n_src, 3), all row-major: out (n_pts,
// 3). Per pair the fill's own interpolation (`proj_core`: libm's hypot,
// atan2 and polar, the momwire#1258 continuation), then the eqs 143-147
// combination of a complex moment, summed over the sources in order.
void reflected_field(const somm_proj::GridView &G, std::ptrdiff_t n_pts, const double *points,
                     std::ptrdiff_t n_src, const double *src_mid,
                     const std::complex<double> *src_moment, double ground_z, double k,
                     std::complex<double> *out);

// H's remainder, curl(E_rem) / (-j omega mu) by central differences at
// `step`: the six offset points p +- step along each axis (`p + shift`, as
// numpy adds the shift vector), d_e[a] = (E(p + step e_a) - E(p - step e_a))
// / (2 step) per part, and the curl's three differences, divided by
// -j omega mu as (-Im, Re) / (omega mu). One grid serves all six.
void reflected_h_field(const somm_proj::GridView &G, std::ptrdiff_t n_pts, const double *points,
                       std::ptrdiff_t n_src, const double *src_mid,
                       const std::complex<double> *src_moment, double ground_z, double k,
                       double omega, double mu, double step, std::complex<double> *out);

// The lazy grids' census of the two (momwire#1512): every node their reads
// take, marked (`proj_core`'s address over the same points, H's six offset
// points formed as `reflected_h_field` forms them).
void reflected_field_nodes(const somm_proj::GridView &G, std::ptrdiff_t n_pts, const double *points,
                           std::ptrdiff_t n_src, const double *src_mid, double ground_z,
                           somm_proj::NodeMarks &marks);
void reflected_h_field_nodes(const somm_proj::GridView &G, std::ptrdiff_t n_pts,
                             const double *points, std::ptrdiff_t n_src, const double *src_mid,
                             double ground_z, double step, somm_proj::NodeMarks &marks);

}  // namespace somm_core
