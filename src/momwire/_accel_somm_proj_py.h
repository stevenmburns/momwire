// The pybind11 half of `_accel_somm_proj_inline.h` (momwire#1479 S3): the
// GridView and far-pack builders from the Python-side arrays. They gather
// pointers and shapes and call the Python-free builders, which keep every
// check, so a binding and the standalone executable build the same view.
#pragma once

#include "_accel_common.h"
#include "_accel_somm_proj_inline.h"

namespace somm_proj {

static GridView build_grid_view(
    double r1_max, double r_break, double th_split, double r_near,
    const py::detail::unchecked_reference<double, 1> &r0b,
    const py::detail::unchecked_reference<double, 1> &drb,
    const py::detail::unchecked_reference<double, 1> &th0b,
    const py::detail::unchecked_reference<double, 1> &dthb,
    const std::vector<py::array_t<cd, py::array::c_style | py::array::forcecast>>
        &reg_vals) {
    const size_t n_reg = reg_vals.size();
    if (n_reg > MAX_REGIONS)
        throw std::runtime_error("expected 4, 6, 9, 12 or 15 region value tables");
    const cd *vals[MAX_REGIONS];
    std::ptrdiff_t shape0[MAX_REGIONS], nR[MAX_REGIONS], nTh[MAX_REGIONS];
    double r0[MAX_REGIONS], dr[MAX_REGIONS], th0[MAX_REGIONS], dth[MAX_REGIONS];
    for (size_t g = 0; g < n_reg; ++g) {
        auto v = reg_vals[g].template unchecked<3>();
        vals[g] = reg_vals[g].data();
        shape0[g] = v.shape(0);
        nR[g] = v.shape(1);
        nTh[g] = v.shape(2);
        r0[g] = r0b(g);
        dr[g] = drb(g);
        th0[g] = th0b(g);
        dth[g] = dthb(g);
    }
    return build_grid_view(r1_max, r_break, th_split, r_near, r0, dr, th0, dth,
                           n_reg, vals, shape0, nR, nTh);
}

static inline void set_far(
    GridView &G,
    const py::array_t<cd, py::array::c_style | py::array::forcecast> &far) {
    set_far(G, far.data(), far.size());
}

}  // namespace somm_proj
