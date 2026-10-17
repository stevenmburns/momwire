// momwire#1517 step 1: bs2's answers to the run half -- see
// `_nec5_bspline_printout_core.h`.
#include "_nec5_bspline_printout_core.h"

#if defined(_MSC_VER)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace mw {
namespace nec5 {

namespace {

const BSplineFront &front_of(const Prepared &p) {
    const BSplinePrepared &part = bspline_part(p);
    if (!part.front) throw PrintoutDecline{"bspline: no front (released or never built)"};
    return *part.front;
}

// The spline expansion at the walk's subdivided arcs
// (`currents_at_knots(coeffs, arcs)` for one solved wire).
class SplineSampler final : public CurrentSampler {
  public:
    SplineSampler(const BSplineFront &fr, const std::vector<cd> &coeffs) : fr_(fr), coeffs_(coeffs) {}
    std::vector<cd> at(std::size_t w, const WireWalk &ww, const std::vector<double> &arcs) const override {
        (void)w;
        return bspline::spline_eval_wire(fr_.basis, fr_.geom, coeffs_, static_cast<std::size_t>(ww.solved),
                                         arcs, false);
    }

  private:
    const BSplineFront &fr_;
    const std::vector<cd> coeffs_;
};

}  // namespace

Solved BSplinePrintoutEngine::solve(const Prepared &p, const NumpyLinalg &np) {
    const BSplineFront &fr = front_of(p);
    if (np.dsyrk == nullptr) throw PrintoutDecline{"bspline: no dsyrk for the assembler"};
    if (np.host.zgemm != nullptr) throw PrintoutDecline{"bspline: the host BLAS route is not ported"};
    bspline::FillParams fp;
    fp.k = fr.k;
    fp.degree = fr.basis.degree;
    fp.n_qp_pair = 8;
    fp.ladder = bspline::default_ladder(8);
    fp.n_qp_same_edge = 4;
    fp.extended_kernel = fr.extended_kernel;
    // `_compute_Z_operator`: the dense tensor within `swept_mem_mb` (256),
    // else the chunked fill
    std::vector<cd> Z;
    if (bspline::dense_tensor_fits(fp.degree, fr.geom.n_segs_total, 256)) {
        const std::vector<cd> J = bspline::build_J_blocks(fr.geom, fr.radii, fp);
        Z = bspline::assemble_Z(J, fr.basis, fr.geom, fr.omega, BSPLINE_EPS, BSPLINE_MU, np.dsyrk);
    } else {
        Z = bspline::compute_Z_chunked(fr.geom, fr.basis, fr.radii, fp, fr.omega, BSPLINE_EPS, BSPLINE_MU, 256.0);
    }
    // `_add_exact_kernel_correction` (after the fill, before the loading)
    if (fr.exact_kernel)
        bspline::add_exact_correction(Z, fr.geom, fr.basis, fr.radii, fp, fr.omega, BSPLINE_EPS, BSPLINE_MU, 256.0);
    // `_apply_loading(Z)`: the distributed series loading, last
    if (fr.loading.active()) {
        BSplineFront &mf = const_cast<BSplineFront &>(fr);
        if (!mf.has_gram) {
            mf.gram = bspline::loading_gram(fr.geom, fr.basis, np);
            mf.has_gram = true;
        }
        bspline::add_wire_scaled(Z, fr.basis.n_basis, fr.gram,
                                 razor::series_impedance_per_wire(fr.omega, fr.radii, fr.loading));
    }
    const bspline::PortSolve ps =
        bspline::solve_ports(Z, fr.basis.n_basis, fr.B, fr.n_ports, fr.basis.kcl_A, fr.basis.n_kcl, lapack_, np,
                             &sym_);
    // A singular system: SciPy warns or raises in its own words, per solver
    if (ps.info != 0) throw PrintoutDecline{"bspline: a singular system"};
    Solved s;
    s.n_basis = fr.basis.n_basis;
    s.n_ports = fr.n_ports;
    s.x_c_order = ps.x_c_order;
    s.x = ps.X;
    s.y = CMat(s.n_ports, s.n_ports);
    s.y.v = ps.Y;
    return s;
}

void BSplinePrintoutEngine::release(Prepared &p) { bspline_part(p).front.reset(); }

std::vector<std::vector<cd>> BSplinePrintoutEngine::currents_at_knots(const Prepared &p,
                                                                      const std::vector<cd> &coeffs) const {
    const BSplineFront &fr = front_of(p);
    return bspline::spline_readout(fr.basis, fr.geom, fr.dedup, coeffs, nullptr, false);
}

std::vector<std::vector<cd>> BSplinePrintoutEngine::current_slopes(
    const Prepared &p, const std::vector<cd> &coeffs, const std::vector<std::vector<cd>> &knots,
    const std::vector<std::vector<double>> &centres_per_piece) const {
    (void)knots;
    const BSplineFront &fr = front_of(p);
    return bspline::spline_readout(fr.basis, fr.geom, fr.dedup, coeffs, &centres_per_piece, true);
}

// `element_currents`' walk: the caller's wires, a dropped copy over its
// twin's polyline and per-edge counts (reversed when it runs the other way).
std::vector<WireWalk> BSplinePrintoutEngine::wire_walk(const Prepared &p) const {
    const BSplineFront &fr = front_of(p);
    const bspline::WireDedup &dd = fr.dedup;
    const auto of_solved = [&](int64_t s, WireWalk &ww) {
        const std::vector<double> &poly = fr.args.polylines[static_cast<std::size_t>(s)];
        ww.polyline = poly;
        ww.npe = fr.args.n_per_edge[static_cast<std::size_t>(s)];
    };
    std::vector<WireWalk> out;
    if (!dd.active) {
        for (std::size_t w = 0; w < fr.args.polylines.size(); w++) {
            WireWalk ww;
            ww.solved = static_cast<int64_t>(w);
            of_solved(ww.solved, ww);
            out.push_back(std::move(ww));
        }
        return out;
    }
    const auto solved_of = [&](int64_t input) -> int64_t {
        for (std::size_t i = 0; i < dd.kept.size(); i++)
            if (dd.kept[i] == input) return static_cast<int64_t>(i);
        return -1;
    };
    for (int64_t w = 0; w < dd.n_input; w++) {
        WireWalk ww;
        ww.solved = solved_of(w);
        if (ww.solved >= 0) {
            of_solved(ww.solved, ww);
        } else {
            for (const auto &d : dd.removed)
                if (d.copy == w) {
                    of_solved(solved_of(d.twin), ww);
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

std::unique_ptr<CurrentSampler> BSplinePrintoutEngine::sampler(const Prepared &p, const std::vector<cd> &coeffs,
                                                               const std::vector<std::vector<cd>> &knots) const {
    (void)knots;
    return std::unique_ptr<CurrentSampler>(new SplineSampler(front_of(p), coeffs));
}

// `wire_loss_power(coeffs, omega)[0]`: 0.0 when unloaded; no jacket term
// (the NEC-5 seam never jackets a wire).
double BSplinePrintoutEngine::wire_loss_power(const Prepared &p, const std::vector<cd> &coeffs,
                                              double omega) const {
    const BSplineFront &fr = front_of(p);
    if (!fr.loading.active()) return 0.0;
    if (!fr.has_gram) throw PrintoutDecline{"bspline: the loss read before the solve"};
    return bspline::wire_loss_power(fr.gram, coeffs, razor::series_impedance_per_wire(omega, fr.radii, fr.loading),
                                    fr.args.polylines.size());
}

double BSplinePrintoutEngine::solver_omega(const Prepared &p) const { return front_of(p).omega; }

double BSplinePrintoutEngine::eps0() const { return BSPLINE_EPS; }

PrintoutResult bspline_printout(const std::string &latin1_text, const std::string &basis,
                                const EngineStamp &stamp, const razor::Lapack &lapack,
                                const bspline::SymLapack &sym, const NumpyLinalg &np,
                                std::vector<RunArrays> *arrays, bool open_end) {
    BSplinePrintoutEngine engine(lapack, sym);
    return printout(latin1_text, basis, stamp, engine, np, arrays, open_end);
}

}  // namespace nec5
}  // namespace mw
