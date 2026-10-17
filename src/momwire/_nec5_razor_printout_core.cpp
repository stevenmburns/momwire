// The NEC-5 seam's run half for razor-2p: razor's answers to what the solver
// is asked -- see `_nec5_razor_printout_core.h`. The run half and the writer
// are the engine-neutral printout's (`_nec5_printout_core.cpp`); each reader
// here names razor.py's method it is a copy of, and computes with that
// file's replayed numpy arithmetic (`replay`), so the bits are numpy's.
#include "_nec5_razor_printout_core.h"

// MSVC builds every TU /fp:fast (setup.py); this one replays numpy's
// arithmetic one IEEE operation at a time, so it asks for precise semantics.
// GCC / clang build it with -ffp-contract=off.
#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>

namespace mw {
namespace nec5 {

namespace {

using namespace replay;

// razor's `currents_at_knots` for one wire (`_end_wings` included).
std::vector<std::vector<cd>> knot_currents(const razor::Front &fr, const std::vector<cd> &coeffs) {
    std::vector<std::vector<cd>> out;
    for (int64_t w = 0; w < fr.n_wires; w++) {
        const int64_t n_knots = fr.seg_offsets[w + 1] - fr.seg_offsets[w] + 1;
        std::vector<cd> I(static_cast<std::size_t>(n_knots), cd(0.0, 0.0));
        for (int64_t b = fr.basis_offsets[w]; b < fr.basis_offsets[w + 1]; b++)
            I[static_cast<std::size_t>(b - fr.basis_offsets[w] + 1)] = coeffs[static_cast<std::size_t>(b)];
        out.push_back(std::move(I));
    }
    const int64_t n_total = fr.n_basis, n_interior = fr.n_interior;
    if (n_total > n_interior) {
        // `_end_wings`: the flattened junction wings by (segment, rise)
        std::map<std::pair<int64_t, bool>, std::vector<int64_t>> by_end;
        for (int64_t k = 0; k < 2 * (n_total - n_interior); k++) {
            const int64_t b = n_interior + k / 2, side = k % 2;
            by_end[{fr.wing_seg[static_cast<std::size_t>(2 * b + side)],
                    fr.wing_rise.data()[2 * b + side]}]
                .push_back(k);
        }
        auto knot = [&](const std::vector<int64_t> &wings) {
            std::vector<cd> terms;
            for (int64_t k : wings) {
                const int64_t b = n_interior + k / 2, side = k % 2;
                const double sigma = fr.wing_sigma[static_cast<std::size_t>(2 * b + side)];
                terms.push_back(np_cmul(cd(sigma, 0.0), coeffs[static_cast<std::size_t>(b)]));
            }
            return np_csum(terms);
        };
        for (int64_t w = 0; w < fr.n_wires; w++) {
            auto start = by_end.find({fr.seg_offsets[w], false});
            if (start != by_end.end() && !start->second.empty()) out[w].front() = knot(start->second);
            auto end = by_end.find({fr.seg_offsets[w + 1] - 1, true});
            if (end != by_end.end() && !end->second.empty()) out[w].back() = knot(end->second);
        }
    }
    if (fr.has_dedup) {
        // `reads_input_wires("per_wire")` (momwire#1042): the caller's
        // numbering, each dropped copy at zero current, sized like its twin
        std::vector<std::vector<cd>> input(static_cast<std::size_t>(fr.n_input_wires));
        for (std::size_t i = 0; i < fr.dedup_kept.size(); i++)
            input[static_cast<std::size_t>(fr.dedup_kept[i])] = std::move(out[i]);
        for (const razor::Front::Dropped &d : fr.dedup_removed)
            input[static_cast<std::size_t>(d.copy)].assign(input[static_cast<std::size_t>(d.twin)].size(),
                                                           cd(0.0, 0.0));
        return input;
    }
    return out;
}

// `element_currents`' walk (momwire#1042): the caller's wires, a dropped copy
// over its twin's polyline (reversed when it runs the other way), with its
// per-edge counts alike. `solved` is -1 for a dropped copy.
struct WireWalk {
    std::vector<double> polyline;  // (n_points, 3)
    std::vector<int64_t> npe;
    int64_t solved = -1;
};
std::vector<WireWalk> wire_walk(const razor::Front &fr) {
    std::vector<WireWalk> out;
    const int64_t n = fr.has_dedup ? fr.n_input_wires : fr.n_wires;
    const auto of_solved = [&](int64_t s, WireWalk &ww) {
        const int64_t p0 = fr.pt_off[static_cast<std::size_t>(s)], p1 = fr.pt_off[static_cast<std::size_t>(s) + 1];
        ww.polyline.assign(fr.points.begin() + 3 * p0, fr.points.begin() + 3 * p1);
        ww.npe = fr.npe[static_cast<std::size_t>(s)];
    };
    for (int64_t w = 0; w < n; w++) {
        WireWalk ww;
        ww.solved = fr.solved_index(w);
        if (ww.solved >= 0) {
            of_solved(ww.solved, ww);
        } else {
            for (const razor::Front::Dropped &d : fr.dedup_removed)
                if (d.copy == w) {
                    of_solved(fr.solved_index(d.twin), ww);
                    if (d.reversed) {
                        const std::size_t m = ww.polyline.size() / 3;
                        std::vector<double> r(ww.polyline.size());
                        for (std::size_t i = 0; i < m; i++)
                            for (int c = 0; c < 3; c++) r[3 * i + c] = ww.polyline[3 * (m - 1 - i) + c];
                        ww.polyline = r;
                        std::reverse(ww.npe.begin(), ww.npe.end());
                    }
                }
        }
        out.push_back(std::move(ww));
    }
    return out;
}

const double *arc_of(const razor::Front &fr, int64_t w) {
    return fr.arc_at_knot.data() + fr.seg_offsets[w] + w;
}

// `RazorSolver.element_currents(coeffs)` (subdiv 1): per wire the knots
// along its edges (each edge an np.linspace, later edges without their first
// knot), the element midpoints `0.5 * (k1 + k0)` and moments
// `(0.5 * (I1 + I0))[:, None] * (k1 - k0)`.
void element_moments(const razor::Front &fr, const std::vector<std::vector<cd>> &knot_i,
                     std::vector<double> &mid, std::vector<cd> &moment) {
    const std::vector<WireWalk> walk = wire_walk(fr);
    for (std::size_t w = 0; w < walk.size(); w++) {
        std::vector<double> knots;
        const std::vector<double> &poly = walk[w].polyline;
        const std::vector<int64_t> &npe = walk[w].npe;
        for (std::size_t e = 0; e < npe.size(); e++) {
            std::vector<double> seg;
            linspace3(&poly[3 * e], &poly[3 * (e + 1)], npe[e] + 1, seg);
            knots.insert(knots.end(), seg.begin() + (e == 0 ? 0 : 3), seg.end());
        }
        const std::vector<cd> &cur = knot_i[static_cast<std::size_t>(w)];
        const std::size_t n_el = knots.size() / 3 - 1;
        if (cur.size() != n_el + 1) throw std::logic_error("knot currents and knots disagree");
        for (std::size_t e = 0; e < n_el; e++) {
            const double *k0 = &knots[3 * e], *k1 = &knots[3 * e + 3];
            const cd el = np_cmul(cd(0.5, 0.0), c_add(cur[e + 1], cur[e]));  // `0.5 * (...)`
            for (int c = 0; c < 3; c++) {
                mid.push_back(0.5 * (k1[c] + k0[c]));
                moment.push_back(c_mul_real(el, k1[c] - k0[c]));
            }
        }
    }
}

// `element_currents(coeffs, subdiv=8)`: each edge split 8 ways (linspace),
// the arc along the chords (`norm3` steps, `np.cumsum`), the current there by
// `np.interp` of the knot currents (real and imaginary apart, then
// `Ire + 1j * Iim`), and the elements' midpoints, moments, nodes and steps.
ElementCurrents element_currents_sub(const razor::Front &fr, const std::vector<std::vector<cd>> &knot_i,
                              int64_t subdiv) {
    ElementCurrents el;
    const std::vector<WireWalk> walk = wire_walk(fr);
    for (std::size_t w = 0; w < walk.size(); w++) {
        std::vector<double> knots;
        const std::vector<double> &poly = walk[w].polyline;
        const std::vector<int64_t> &npe = walk[w].npe;
        for (std::size_t e = 0; e < npe.size(); e++) {
            std::vector<double> seg;
            linspace3(&poly[3 * e], &poly[3 * (e + 1)], npe[e] * subdiv + 1, seg);
            knots.insert(knots.end(), seg.begin() + (e == 0 ? 0 : 3), seg.end());
        }
        const std::size_t nk = knots.size() / 3;
        std::vector<double> arcs(nk, 0.0);
        double run = 0.0;
        for (std::size_t q = 1; q < nk; q++) {
            const double dx = knots[3 * q] - knots[3 * q - 3], dy = knots[3 * q + 1] - knots[3 * q - 2],
                         dz = knots[3 * q + 2] - knots[3 * q - 1];
            const double step = std::sqrt((dx * dx + dy * dy) + dz * dz);
            run = q == 1 ? step : run + step;
            arcs[q] = run;
        }
        std::vector<cd> cur(nk, cd(0.0, 0.0));  // a dropped copy: zeros (momwire#1042)
        const int64_t s = walk[w].solved;
        const int64_t a0 = s < 0 ? 0 : fr.seg_offsets[static_cast<std::size_t>(s)] + s;
        const int64_t na = s < 0 ? 0 : fr.seg_offsets[static_cast<std::size_t>(s) + 1] - fr.seg_offsets[static_cast<std::size_t>(s)] + 1;
        const double *arc = &fr.arc_at_knot[static_cast<std::size_t>(a0)];
        const std::vector<cd> &kc = knot_i[w];
        std::vector<double> re(kc.size()), im(kc.size());
        for (std::size_t q = 0; q < kc.size(); q++) {
            re[q] = kc[q].real();
            im[q] = kc[q].imag();
        }
        for (std::size_t q = 0; q < nk && s >= 0; q++) {
            const double ir = np_interp(arcs[q], arc, re.data(), na);
            const double ii = np_interp(arcs[q], arc, im.data(), na);
            const cd j_ii = np_cmul(cd(0.0, 1.0), cd(ii, 0.0));  // `1j * Iim`
            cur[q] = cd(ir + j_ii.real(), 0.0 + j_ii.imag());
        }
        std::vector<cd> element(nk - 1);
        for (std::size_t q = 0; q + 1 < nk; q++) {
            element[q] = np_cmul(cd(0.5, 0.0), c_add(cur[q + 1], cur[q]));
            for (int c = 0; c < 3; c++) {
                el.mid.push_back(0.5 * (knots[3 * q + 3 + c] + knots[3 * q + c]));
                el.moment.push_back(c_mul_real(element[q], knots[3 * q + 3 + c] - knots[3 * q + c]));
            }
        }
        el.nodes.insert(el.nodes.end(), knots.begin(), knots.end());
        for (std::size_t q = 0; q < nk; q++) {
            const cd in = q == 0 ? cd(0.0, 0.0) : element[q - 1];
            const cd out = q + 1 == nk ? cd(0.0, 0.0) : element[q];
            el.delta.push_back(c_sub(in, out));
        }
    }
    return el;
}

// `RazorSolver.wire_loss_power(coeffs, omega)[0]` (momwire#1501's hypot):
// each segment's end currents from the wings (`np.add.at` in entry order),
// `h * (|a|^2 + Re(a conj b) + |b|^2) / 3`, weighted by ½ Re Z'_w at
// `omega`, summed per wire in segment order, then over the wires.
double wire_loss_power(const razor::Front &fr, const std::vector<cd> &coeffs, double omega) {
    if (!fr.loading.active()) return 0.0;
    const int64_t n_seg = fr.n_seg;
    std::vector<cd> i_lo(static_cast<std::size_t>(n_seg), cd(0.0, 0.0)), i_hi = i_lo;
    for (int64_t b = 0; b < fr.n_basis; b++)
        for (int w = 0; w < 2; w++) {
            const std::size_t q = static_cast<std::size_t>(2 * b + w);
            const cd v = np_cmul(coeffs[static_cast<std::size_t>(b)], cd(fr.wing_sigma[q], 0.0));
            cd &t = fr.wing_rise.p[q] ? i_hi[static_cast<std::size_t>(fr.wing_seg[q])]
                                      : i_lo[static_cast<std::size_t>(fr.wing_seg[q])];
            t = c_add(t, v);
        }
    const std::vector<cd> z_wire = razor::series_impedance_per_wire(omega, fr.radius_per_wire, fr.loading);
    std::vector<double> per_wire(static_cast<std::size_t>(fr.n_wires), 0.0);
    for (int64_t w = 0; w < fr.n_wires; w++)
        for (int64_t s = fr.seg_offsets[w]; s < fr.seg_offsets[w + 1]; s++) {
            const cd lo = i_lo[static_cast<std::size_t>(s)], hi = i_hi[static_cast<std::size_t>(s)];
            const double ml = std::hypot(lo.real(), lo.imag()), mh = std::hypot(hi.real(), hi.imag());
            const double cross = np_cmul(lo, std::conj(hi)).real();
            const double gram = (fr.seg_h[s] * ((ml * ml + cross) + mh * mh)) / 3.0;
            per_wire[static_cast<std::size_t>(w)] += (0.5 * z_wire[static_cast<std::size_t>(w)].real()) * gram;
        }
    return 0.0 + np_pairwise(per_wire.data(), fr.n_wires);
}

}  // namespace

Solved RazorPrintoutEngine::solve(const Prepared &p, const NumpyLinalg &np) {
    razor::SolveContext &ctx = ctx_;
    const razor::Lapack &lapack = lapack_;
    const razor::Front &front = *razor_part(p).front;
    Solved s;
    s.n_basis = front.n_basis;
    s.n_ports = front.n_ports;
    s.x.assign(static_cast<std::size_t>(s.n_basis * s.n_ports), cd(0.0, 0.0));
    const int info = razor::solve_ports(front, ctx, lapack, s.x.data(), 0);
    if (info != 0) throw PrintoutRaise{"LinAlgError", "singular matrix"};
    // `cols.T @ X`: cols.T is the cast copy in its own (Fortran) order
    std::vector<cd> cols(front.cols.size());
    for (std::size_t k = 0; k < cols.size(); k++) cols[k] = cd(front.cols[k], 0.0);
    s.y = CMat(s.n_ports, s.n_ports);
    np_matmul(np, Op{cols.data(), s.n_ports, s.n_basis, 1, s.n_ports},
              Op{s.x.data(), s.n_basis, s.n_ports, 1, s.n_basis}, s.y.v.data(), s.n_ports, 1);
    return s;
}


// The run is finished: its front (and the fill's tables) released.
void RazorPrintoutEngine::release(Prepared &p) { razor_part(p).front.reset(); }

std::vector<std::vector<cd>> RazorPrintoutEngine::currents_at_knots(
    const Prepared &p, const std::vector<cd> &coeffs) const {
    return knot_currents(*razor_part(p).front, coeffs);
}

// `RazorSolver.current_slopes(coeffs, centres_per_piece)`: the knot currents'
// differences over the arc between knots, read at each centre's span
// (`searchsorted(arc, s, side="right") - 1`, clamped); a dropped copy
// (momwire#1042) has zero slope at every centre.
std::vector<std::vector<cd>> RazorPrintoutEngine::current_slopes(
    const Prepared &p, const std::vector<cd> &coeffs, const std::vector<std::vector<cd>> &knots,
    const std::vector<std::vector<double>> &centres_per_piece) const {
    (void)coeffs;
    const razor::Front &fr = *razor_part(p).front;
    std::vector<std::vector<cd>> slopes;
    for (std::size_t w = 0; w < p.mesh.pieces.size(); w++) {
        const std::vector<cd> &I = knots[w];
        const std::vector<double> &centres_s = centres_per_piece[w];
        const int64_t solved = fr.solved_index(static_cast<int64_t>(w));
        if (solved < 0) {
            slopes.emplace_back(centres_s.size(), cd(0.0, 0.0));
            continue;
        }
        const double *arc = arc_of(fr, solved);
        const int64_t n_knots = static_cast<int64_t>(I.size()), nseg = n_knots - 1;
        std::vector<cd> slope(static_cast<std::size_t>(nseg));
        for (int64_t j = 0; j < nseg; j++) {
            const cd diff(I[j + 1].real() - I[j].real(), I[j + 1].imag() - I[j].imag());
            slope[j] = np_cdiv(diff, cd(arc[j + 1] - arc[j], 0.0));
        }
        std::vector<cd> out;
        for (double sv : centres_s) {
            sv = std::min(std::max(sv, arc[0]), arc[n_knots - 1]);  // np.clip
            // searchsorted(arc, s, side="right") - 1, clamped
            const int64_t right = std::upper_bound(arc, arc + n_knots, sv) - arc;
            const int64_t span = std::min(std::max<int64_t>(right - 1, 0), nseg - 1);
            out.push_back(slope[static_cast<std::size_t>(span)]);
        }
        slopes.push_back(std::move(out));
    }
    return slopes;
}

void RazorPrintoutEngine::element_moments(const Prepared &p,
                                          const std::vector<std::vector<cd>> &knots,
                                          std::vector<double> &mid, std::vector<cd> &moment) const {
    nec5::element_moments(*razor_part(p).front, knots, mid, moment);
}

ElementCurrents RazorPrintoutEngine::element_currents(const Prepared &p,
                                                      const std::vector<std::vector<cd>> &knots,
                                                      int64_t subdiv) const {
    return element_currents_sub(*razor_part(p).front, knots, subdiv);
}

double RazorPrintoutEngine::wire_loss_power(const Prepared &p, const std::vector<cd> &coeffs,
                                            double omega) const {
    return nec5::wire_loss_power(*razor_part(p).front, coeffs, omega);
}

// `solver.omega`: 2 * np.pi * (c / wavelength), at the solver's own c.
double RazorPrintoutEngine::solver_omega(const Prepared &p) const {
    return (2.0 * 3.141592653589793) * (razor_part(p).front->c / p.wavelength);
}

double RazorPrintoutEngine::eps0() const { return razor::EPS; }

PrintoutResult razor_printout(const std::string &latin1_text, const std::string &basis,
                              const EngineStamp &stamp, razor::SolveContext &ctx,
                              const razor::Lapack &lapack, const NumpyLinalg &np,
                              std::vector<RunArrays> *arrays, bool open_end) {
    RazorPrintoutEngine engine(ctx, lapack);
    return printout(latin1_text, basis, stamp, engine, np, arrays, open_end);
}

}  // namespace nec5
}  // namespace mw
