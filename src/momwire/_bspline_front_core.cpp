// PROTOTYPE (momwire#1517 step 1): see `_bspline_front_core.h`. Each step is
// numpy's operation as `bspline.py` spells it, one IEEE operation at a time.
#include "_bspline_front_core.h"

#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include <cmath>
#include <stdexcept>
#include <string>

namespace mw {
namespace bspline {

namespace {

// `np.linspace(start, stop, num)` for scalars, num >= 2 (numpy 2.5's form):
// `arange(num) * step + start`, the step-zero branch `(i / div) * delta`,
// and the last entry `stop` exactly.
std::vector<double> linspace(double start, double stop, int64_t num) {
    std::vector<double> y(static_cast<size_t>(num));
    const double div = static_cast<double>(num - 1);
    const double delta = stop - start;
    const double step = delta / div;
    for (int64_t i = 0; i < num; i++) {
        const double fi = static_cast<double>(i);
        y[static_cast<size_t>(i)] = (step == 0.0 ? (fi / div) * delta : fi * step) + start;
    }
    if (num > 1) y[static_cast<size_t>(num - 1)] = stop;
    return y;
}

}  // namespace

Geometry build_geometry(const std::vector<std::vector<double>> &polylines,
                        const std::vector<std::vector<int64_t>> &n_per_edge) {
    Geometry g;
    g.seg_offsets.push_back(0);
    for (size_t w = 0; w < polylines.size(); w++) {
        const std::vector<double> &pl = polylines[w];
        const size_t n_pts = pl.size() / 3;
        WireGeometry wg;
        wg.edge_offsets.push_back(0);
        for (size_t e = 0; e + 1 < n_pts; e++) {
            const double *p0 = &pl[3 * e], *p1 = &pl[3 * e + 3];
            const double v[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
            // `float(norm3(edge_vec))`: math.sqrt((x*x + y*y) + z*z)
            const double edge_len = std::sqrt((v[0] * v[0] + v[1] * v[1]) + v[2] * v[2]);
            if (edge_len < 1e-15)
                throw std::invalid_argument("wire " + std::to_string(w) + " edge " +
                                            std::to_string(e) + " has zero length");
            const double tan[3] = {v[0] / edge_len, v[1] / edge_len, v[2] / edge_len};
            const int64_t n_e = n_per_edge[w][e];
            const double h_e = edge_len / static_cast<double>(n_e);
            const std::vector<double> t = linspace(0.0, 1.0, n_e + 1);
            // `(1 - t)[:, None] * p0 + t[:, None] * p1`
            std::vector<double> pts(static_cast<size_t>(3 * (n_e + 1)));
            for (int64_t i = 0; i <= n_e; i++)
                for (int c = 0; c < 3; c++) {
                    const double ti = t[static_cast<size_t>(i)];
                    pts[static_cast<size_t>(3 * i + c)] = (1.0 - ti) * p0[c] + ti * p1[c];
                }
            for (int64_t i = 0; i < n_e; i++)
                for (int c = 0; c < 3; c++) {
                    wg.seg_l.push_back(pts[static_cast<size_t>(3 * i + c)]);
                    wg.seg_r.push_back(pts[static_cast<size_t>(3 * (i + 1) + c)]);
                    wg.tangents.push_back(tan[c]);
                }
            for (int64_t i = 0; i < n_e; i++) wg.h_per_seg.push_back(h_e);
            wg.edge_arc_edges.push_back(linspace(0.0, edge_len, n_e + 1));
            wg.edge_offsets.push_back(wg.edge_offsets.back() + n_e);
        }
        wg.n_total = static_cast<int64_t>(wg.h_per_seg.size());
        // `np.concatenate([[0.0], np.cumsum(h_per_seg_w)])`
        wg.arc_at_knot.push_back(0.0);
        double run = 0.0;
        for (size_t i = 0; i < wg.h_per_seg.size(); i++) {
            run = i == 0 ? wg.h_per_seg[0] : run + wg.h_per_seg[i];
            wg.arc_at_knot.push_back(run);
        }
        g.seg_offsets.push_back(g.seg_offsets.back() + wg.n_total);
        g.h_per_seg.insert(g.h_per_seg.end(), wg.h_per_seg.begin(), wg.h_per_seg.end());
        g.tangents.insert(g.tangents.end(), wg.tangents.begin(), wg.tangents.end());
        g.seg_l.insert(g.seg_l.end(), wg.seg_l.begin(), wg.seg_l.end());
        g.seg_r.insert(g.seg_r.end(), wg.seg_r.begin(), wg.seg_r.end());
        g.per_wire.push_back(std::move(wg));
    }
    g.n_segs_total = g.seg_offsets.back();
    return g;
}

}  // namespace bspline
}  // namespace mw
