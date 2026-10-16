// razor-2p's buried fill in C++: see `_razor_buried_core.h`.
#include "_razor_buried_core.h"

#include <algorithm>
#include <set>
#include <stdexcept>

namespace mw {
namespace razor {

namespace {

// The knot each basis is centred on, read off its side-A wing
// (`_knot_points`): seg_p0[s_a] + (rise ? h : 0) * seg_t[s_a].
void knot_point(const Front &g, int64_t m, double *out) {
    const int64_t s = g.wing_seg[static_cast<size_t>(2 * m)];
    const double along = g.wing_rise.p[2 * m] ? g.seg_h[static_cast<size_t>(s)] : 0.0;
    for (int c = 0; c < 3; c++)
        out[c] = g.seg_p0[static_cast<size_t>(3 * s + c)] + along * g.seg_t[static_cast<size_t>(3 * s + c)];
}

}  // namespace

MediumPart medium_part(const Front &f, bool below) {
    MediumPart mp;
    mp.front.reset(new Front());
    Front &s = *mp.front;
    const auto side_of = [&](int64_t w) { return f.media_below[static_cast<size_t>(w)] != 0; };

    // The medium's wires, segments and owned bases, in wire order.
    std::vector<int64_t> seg_i, wires;
    std::vector<int64_t> remap(static_cast<size_t>(f.n_seg), -1);
    for (int64_t w = 0; w < f.n_wires; w++) {
        if (side_of(w) != below) continue;
        wires.push_back(w);
        for (int64_t g = f.seg_offsets[static_cast<size_t>(w)]; g < f.seg_offsets[static_cast<size_t>(w) + 1]; g++) {
            remap[static_cast<size_t>(g)] = static_cast<int64_t>(seg_i.size());
            seg_i.push_back(g);
        }
    }
    std::vector<int64_t> &rows = mp.rows;
    for (int64_t w : wires)
        for (int64_t m = f.basis_offsets[static_cast<size_t>(w)]; m < f.basis_offsets[static_cast<size_t>(w) + 1]; m++)
            rows.push_back(m);
    for (int64_t m = f.basis_offsets.back(); m < f.n_basis; m++)
        for (int j = 0; j < 2; j++)
            if (remap[static_cast<size_t>(f.wing_seg[static_cast<size_t>(2 * m + j)])] >= 0) {
                rows.push_back(m);
                break;
            }

    // The scalar state the fill reads: the solver's, unchanged.
    s.has_ground = true;
    s.ground_z = f.ground_z;
    s.c = f.c;
    s.k = f.k;
    s.omega = f.omega;
    s.c_a_imag = f.c_a_imag;
    s.c_phi_imag = f.c_phi_imag;
    s.extended_kernel = false;
    s.exact = false;
    s.compose = f.compose;
    s.ground_eps_r_ = f.ground_eps_r_;
    s.ground_sigma_ = f.ground_sigma_;
    s.n_qp_sommerfeld = f.n_qp_sommerfeld;
    s.xg = f.xg;
    s.wg = f.wg;
    // The kernel's radius: the scalar on a uniform deck, else each segment
    // its wire's (`seg_a`), which the medium's wires spell by themselves.
    s.n_wires = static_cast<int64_t>(wires.size());
    s.uniform_radius = f.uniform_radius;
    s.radius_value = f.radius_value;
    s.seg_offsets.assign(1, 0);
    for (int64_t w : wires) {
        s.radius_per_wire.push_back(f.radius_per_wire[static_cast<size_t>(w)]);
        s.seg_offsets.push_back(s.seg_offsets.back() + f.seg_offsets[static_cast<size_t>(w) + 1] -
                                f.seg_offsets[static_cast<size_t>(w)]);
    }
    s.n_seg = static_cast<int64_t>(seg_i.size());
    for (int64_t g : seg_i) {
        for (int c = 0; c < 3; c++) {
            s.seg_p0.push_back(f.seg_p0[static_cast<size_t>(3 * g + c)]);
            s.seg_t.push_back(f.seg_t[static_cast<size_t>(3 * g + c)]);
        }
        s.seg_h.push_back(f.seg_h[static_cast<size_t>(g)]);
    }
    // The wings: a wing on another medium's segment becomes a ghost of the
    // live one (its segment and rise) at sigma = 0, and the row is chopped
    // ("A": wing A survives).
    const int64_t n = static_cast<int64_t>(rows.size());
    s.n_basis = n;
    s.wing_seg.assign(static_cast<size_t>(2 * n), 0);
    s.wing_rise.assign(static_cast<size_t>(2 * n), false);
    s.wing_sigma.assign(static_cast<size_t>(2 * n), 0.0);
    std::vector<std::pair<int64_t, bool>> chop;
    for (int64_t i = 0; i < n; i++) {
        const int64_t m = rows[static_cast<size_t>(i)];
        bool side[2];
        for (int j = 0; j < 2; j++) side[j] = remap[static_cast<size_t>(f.wing_seg[static_cast<size_t>(2 * m + j)])] >= 0;
        for (int j = 0; j < 2; j++) {
            const int src = side[j] ? j : 1 - j;
            s.wing_seg[static_cast<size_t>(2 * i + j)] =
                remap[static_cast<size_t>(f.wing_seg[static_cast<size_t>(2 * m + src)])];
            s.wing_rise.p[2 * i + j] = f.wing_rise.p[2 * m + src];
            s.wing_sigma[static_cast<size_t>(2 * i + j)] = side[j] ? f.wing_sigma[static_cast<size_t>(2 * m + j)] : 0.0;
        }
        if (!(side[0] && side[1])) chop.emplace_back(i, side[0]);
    }
    s.grounded_bases.clear();
    s.prepare();
    // `t2_chop`: the chopped rows sorted, which half survives, and each
    // one's knot on the SUB-geometry (`_knot_points(geom_side)`).
    std::sort(chop.begin(), chop.end());
    s.chop_rows.clear();
    s.chop_pts.clear();
    s.chop_keep_a.assign(chop.size(), false);
    for (size_t q = 0; q < chop.size(); q++) {
        s.chop_rows.push_back(chop[q].first);
        s.chop_keep_a.p[q] = chop[q].second;
        double kp[3];
        knot_point(s, chop[q].first, kp);
        s.chop_pts.insert(s.chop_pts.end(), kp, kp + 3);
    }
    // `_schedule.elem_rows(_WEIGHTED_CHUNK_ELEMS, n_path * n_basis)`: the
    // window size moves no bit.
    const int64_t budget = f.chunk_elems / 12;
    s.window_rows = std::max<int64_t>(1, budget / std::max<int64_t>(1, 2 * n));
    s.chunk_elems = f.chunk_elems;
    if (!below && s.compose) {
        const std::string why = s.prepare_compose(s.n_qp_sommerfeld);
        if (!why.empty()) throw std::runtime_error("buried above medium: " + why);
    }
    return mp;
}

std::string fill_buried(const Front &f, std::vector<cd> &out, BuriedStages *stages,
                        uintptr_t cancel_flag) {
    if (f.buried == Buried::none) throw std::invalid_argument("fill_buried: not a buried deck");
    const int64_t n = f.n_basis;
    const auto record = [&](const char *name) {
        if (stages == nullptr) return;
        stages->reached.push_back(name);
        stages->z.push_back(out);
    };
    if (f.buried == Buried::below_plane) return "below";

    // 1-2. The above medium's block, scattered into the zero target.
    out.assign(static_cast<size_t>(n * n), cd(0.0, 0.0));
    {
        MediumPart a = medium_part(f, false);
        SolveContext ctx;
        SolveInputs in = a.front->solve_inputs(cancel_flag);
        const ptrdiff_t nc = static_cast<ptrdiff_t>(a.front->chop_rows.size());
        if (nc) {
            in.chop_rows = view<const int64_t, 1>::contiguous(a.front->chop_rows.data(), {nc});
            in.chop_keep_a = view<const bool, 1>::contiguous(a.front->chop_keep_a.data(), {nc});
            in.chop_pts = view<const double, 2>::contiguous(a.front->chop_pts.data(), {nc, 3});
        }
        ctx.fill(in);
        const int64_t na = a.front->n_basis;
        const cd *za = ctx.z();
        // `_ix_accumulate(Z, rows_a, rows_a, Z_a)`: one IEEE add per entry.
        for (int64_t j = 0; j < na; j++) {
            const int64_t cj = a.rows[static_cast<size_t>(j)];
            for (int64_t i = 0; i < na; i++) {
                const int64_t ri = a.rows[static_cast<size_t>(i)];
                cd &e = out[static_cast<size_t>(cj * n + ri)];
                const cd v = za[static_cast<size_t>(j * na + i)];
                e = cd(e.real() + v.real(), e.imag() + v.imag());
            }
        }
    }
    record("above");
    return "below";
}

}  // namespace razor
}  // namespace mw
