// The near field across and below a finite ground, Python-free
// (momwire#1653): `_near_buried.py`'s two new regimes in C++, so the
// standalone executables serve what the Python launchers serve, and the
// Python route calls this very code through `_accelerators`
// (`near_buried_parts`). The numpy spelling in `_near_buried.py` stays the
// reference, gated against this at a derived tolerance.
//
//   transmitted   a WHOLE field across the interface (either way), from the
//                 crossing fill's designed contour kernels: the six through
//                 `ni_eval::designed_tables` and the point family through
//                 `ni_eval::point_tables`, both over one `SheetPlan`;
//   below         direct(k_m, eps) + A_m * image(k_m, eps) + the below/below
//                 remainder (`somm_below::Remainder`'s projection over the
//                 point-spacing grid; past its cap, the surfaces by direct
//                 contour evaluation, `somm_below::iv_surfaces`).
//
// H is the curl of the non-gradient part, central differences, one-sided in
// z at the plane (`_near_buried._curl`).
#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace near_buried {

using cd = std::complex<double>;

// `_near_buried.TRANSMITTED_CURL_STEP_WAVELENGTHS`, `BELOW_CURL_STEP_WAVELENGTHS`,
// `BELOW_FAR_SEGMENT_PHASE`.
constexpr double TRANSMITTED_CURL_STEP_WAVELENGTHS = 1e-4;
constexpr double BELOW_CURL_STEP_WAVELENGTHS = 1e-3;
constexpr double BELOW_FAR_SEGMENT_PHASE = 0.1;

// `element_currents(coeffs, subdiv)`'s source terms, the whole structure:
// mid (n_el, 3), moment (n_el, 3), nodes (n_el + n_wires, 3), delta
// (n_el + n_wires), and each wire's element count (`counts`, summing to
// n_el) at resampling `subdiv`.
struct Elements {
    const double *mid = nullptr;
    const cd *moment = nullptr;
    size_t n_el = 0;
    const double *nodes = nullptr;
    const cd *delta = nullptr;
    size_t n_nodes = 0;
    const int64_t *counts = nullptr;
    size_t n_wires = 0;
    int subdiv = 1;
};

// One side's source terms (`_near_buried.Sources`).
struct Sources {
    std::vector<double> mid, nodes, start, end;  // start/end: each element's knots
    std::vector<cd> moment, delta;
    // mesh segments (`subdiv` consecutive elements): summed moments at their
    // midpoints, horizontal and vertical extents
    std::vector<double> seg_mid, seg_h, seg_v;
    std::vector<cd> seg_moment;
    bool whole = true;
    size_t n_el() const { return mid.size() / 3; }
    size_t n_nodes() const { return nodes.size() / 3; }
};

// `source_sides` + `Sources`: the wires below the plane (by their elements'
// midpoints; a wire straddling it is std::invalid_argument) and each side's
// sources.
void split(const Elements &el, double ground_z, Sources &above, Sources &below);

// Whether a near field over the composing ground takes this composition
// (`_near_readout._crosses`): an element midpoint or a point below the plane.
bool crosses(const double *points, size_t n, const Elements &el, double ground_z);

// `transmitted_field_at(points, sources, ...)`: (n, 3) row-major into out.
void transmitted(const double *points, size_t n, const Sources &src, cd eps_t, double ground_z, double k,
                 double omega, double mu, double radius, bool observers_above, bool magnetic, int n_threads,
                 cd *out);

// `below_field_at(points, elements, sources, ...)`: (n, 3) row-major.
void below(const double *points, size_t n, const Sources &src, cd eps_t, double ground_z, double k,
           double omega, double mu, double radius, bool magnetic, int n_threads, cd *out);

// The two regimes' share of `_near_field_buried` at every point: for a point
// at or above the plane, the transmitted field of the below sources (the
// caller adds the above/above composition of the above sources FIRST); for a
// point below it, below(...) + transmitted(above sources). (n, 3) row-major.
void parts(const double *points, size_t n, const Elements &el, cd eps_t, double ground_z, double k,
           double omega, double mu, double radius, bool magnetic, int n_threads, cd *out);

}  // namespace near_buried
