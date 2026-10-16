// razor-2p's exact ring-kernel correction (momwire#1433) for the standalone
// path (momwire#1479), Python-free: `_razor_exact_plan.build_plan` (the
// geometry-only tables of every one-radius coaxial group) and
// `RazorSolver._add_exact_kernel_correction` (per group and observer window:
// the exact point moments, the base moments through the fill's own kernel,
// their difference assembled into Z). The point moments and the assembly are
// `_exact_kernel_point_core.h`, the extension's own text.
//
// SCOPE: one radius per coaxial line. A line that carries a radius step
// needs the two-radius cross jobs (`_exact_kernel.PointCrossRows`), numpy
// only today; `build_exact_plan` returns "exact-kernel-cross" for it and the
// front half declines by that name.
#pragma once

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

namespace mw {
namespace razor {

class Front;
using cd = std::complex<double>;

struct ExactWindow {
    int64_t o0 = 0, o1 = 0, e0 = 0, e1 = 0;
};

struct ExactGroup {  // `_razor_exact_plan._Group`, one-radius
    int64_t n = 0;
    std::vector<int64_t> S;                 // the group's segments
    std::vector<double> x0, sgn, h;         // `group_frame`
    double a = 0;                           // the group's radius
    std::vector<int64_t> col_ptr, col_n;    // CSR over the columns (B)
    std::vector<int64_t> ent_j;             // wing entries, by column
    std::vector<double> ent_t, ent_q;
    std::vector<uint8_t> ent_rise;
    std::vector<double> part_w, part_p;     // a_parts in pre-sort order
    std::vector<double> part_sign;          // p_parts' sign per pre-sort row
    std::vector<uint8_t> part_zero;         // grounded (cP = 0) per pre-sort row
    std::vector<int64_t> by_o, r_o, r_m;    // sorted by observer
    std::vector<double> obs_pt, obs_x;      // (n, 3) centroids, their axial x
    std::vector<double> sub_p0, sub_t, sub_h, sub_a;  // the source sub-geometry
    std::vector<int64_t> lab_obs, lab_src;  // EK labels (empty: reduced kernel)
    std::vector<double> b_obs;              // EK tube per observer (empty: one radius)
    std::vector<ExactWindow> windows;
};

struct ExactPlan {
    std::vector<ExactGroup> groups;
};

// "" and the plan; or the reason this is not served ("exact-kernel-cross").
std::string build_exact_plan(const Front &front, ExactPlan &plan);

// Z (column-major, n_basis x n_basis) += every group's correction, in plan
// order and window order.
void add_exact_correction(const Front &front, const ExactPlan &plan, cd *Z,
                          uintptr_t cancel_flag);

}  // namespace razor
}  // namespace mw
