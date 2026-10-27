// The Sinusoidal-Galerkin basis' MIXED-POTENTIAL FILL AND SOLVE in C++
// (momwire#1718 S2), with no Python in it. Over the sinusoidal front half
// (`sinusoidal::front`, which SinusoidalGalerkinSolver inherits: its geometry,
// feed snap and `_basis_coefs`) it gives what
// `SinusoidalGalerkinSolver.compute_port_solution` computes on the default
// `fill="mixed-potential"` route in free space:
// - `_assemble_Z_mp`: `_sinusoidal_mp.basis_csr`, then `WindowFill`'s free
//   block and, over a perfect ground (GN 1, momwire#1718 S5a), its image
//   block (the sources mirrored in the plane, scale `-DIRECT_SIGN`); each
//   block with the extended kernel's every-pair labels and the
//   source tube on a stepped radius, an observer window at a time
//   (`WINDOW_BYTES`): the tiered pair moments from the B-spline core's
//   sinusoidal kernels, one call per run of equal observer radius
//   (`seg_seg_full_moments_sinusoidal_tiered[_ek]`, the solver's
//   `DEFAULT_N_QP_PAIR` and `DEFAULT_PAIR_ORDER_LADDER`); the near pairs
//   (`classify_pairs`, a plain scan that gives the cKDTree ball query's set,
//   then the strict test and the lexsort order) overwritten by
//   `near_pair_moments` -- the parallel reduction's kernel
//   (`parallel_pair_moments_sinusoidal`) and numpy's own product rules for
//   the rest (`_pair_moments_two_rules_batch`: the graded endpoint rule on a
//   touching pair, `NEAR_N_QP` otherwise, numpy's elementwise order and its
//   three-operand einsum); and the window's
//   assembly (`assemble_Z_sinusoidal_windowed`) into G, F order;
// - `_apply_loading` (momwire#1718 S6): the series wire impedance's Galerkin
//   overlap, closed form on the shapes (`_loading_integrals`), subtracted
//   pair by pair in `_shared_segment_pairs`' order;
// - the port algebra: `_drive_columns` (the point gap, `-f_i(s0)` at
//   `feed_xi`, `np.add.at`), `_solve_in_place` (`lu_factor` in place and
//   `lu_solve`: getrf / getrs on whichever LAPACK the caller hands in) and
//   `_port_currents` (`_feed_segment_current` at `feed_xi`) into Y.
//
// Python stays the reference: G, the coefficients and Y are Python's to the
// bit, gated by `tests/test_sinusoidal_galerkin_fill_core.py` through
// `scripts/core_split/sinusoidal_galerkin_fill_compare.py`.
//
// REFUSED as Python refuses (`SolverRefusal`, the exception's type and
// words): a non-finite matrix (`lu_factor`'s check), a singular one
// (`_solve_in_place`), the extended kernel at a stepped-radius junction;
// the front's own refusals carry SinusoidalGalerkinSolver's name
// (`sinusoidal::front(args, galerkin=true)`).
//
// A wire end on a ground plane takes the direct-field fill
// (`_sinusoidal_galerkin_direct_core.h`, momwire#1718 S7). Node ports (S9,
// `node_gaps`) are served on the mixed-potential fill.
//
// A WHOLLY-BURIED deck (momwire#1718 S11a, `_operating_medium`) is solved in
// the lower medium: the same fill at k_m / eta_m (`WindowFill` at a complex
// k, its near pairs on the complex product rule), the image weighted A_m,
// and the below family's remainder (`_mp_remainder_below`: bspline's
// field-form route over the shared below projection,
// `somm_below::Remainder`, the grid from `somm_below::source()`), the
// loading's overlap at k_m, the drive and the gap readout at k_m (off the
// segment centre CPython's `cmath.sin`).
//
// DECLINED (`nec4::Declined`), never misread: an insulated wire, wires on
// both sides of the interface (the transmitted blocks; a crossing junction
// is S11b), junction ports (the direct route; no
// corpus deck carries one, S0), node gaps on a deck the duplicate drop has
// reduced, a below/below pair under the
// grazing floor (the refusal's words are not copied).
#pragma once

#include "_sinusoidal_fill_core.h"

#include <complex>
#include <string>
#include <vector>

namespace mw {
namespace sinusoidal_galerkin {

// A refusal the solver's constructor raises with another exception type than
// the front's ValueError (`NotImplementedError` for the extended kernel at a
// stepped-radius junction): `kind` is its Python type name.
struct SolverRefusal : std::runtime_error {
    std::string kind, message;  // UTF-8
    SolverRefusal(const std::string &k, const std::string &m) : std::runtime_error(m), kind(k), message(m) {}
};

struct PortSolution {
    long long n = 0, n_ports = 0;
    std::vector<std::complex<double>> G;       // (n, n), F order: `_assemble_Z`'s
    std::vector<std::complex<double>> coeffs;  // (n, n_ports), F order (`lu_solve`'s)
    std::vector<std::complex<double>> y;       // (n_ports, n_ports), C order
};

// The constructor's own refusals beyond the front's (`SolverRefusal`).
void check_constructor(const sinusoidal::Front &fr, const nec4::SolverArgs &args);

// `_assemble_Z(geom, k)` on the mixed-potential route in free space: G in F
// order, (n, n).
// `np`: numpy's products (the remainder's grazing-pair scan), needed over a
// Sommerfeld ground.
std::vector<std::complex<double>> assemble_Z(const sinusoidal::Front &fr, const nec4::SolverArgs &args,
                                             const nec5::NumpyLinalg *np = nullptr);

// A series EMF AT a junction node, named by one member (`node_gaps=`, the
// NEC-5 seam's node spelling, momwire#1718 S9): wire `wire`'s start or end.
struct NodeGap {
    long long wire = 0;
    bool start = true;
};

// The constructor's checks of `node_gaps=` (`SolverRefusal` ValueError).
void check_node_gaps(const sinusoidal::Front &fr, const std::vector<NodeGap> &node_gaps);

// `compute_port_solution()`: ports [gap feeds..., node gaps...].
PortSolution port_solution(const sinusoidal::Front &fr, const nec4::SolverArgs &args,
                           const sinusoidal::FillLinalg &la, const std::vector<NodeGap> &node_gaps = {});

// The comparison format (`sinusoidal_galerkin_fill_compare.py`), per operating
// point of each built structure of a deck: a matrix of up to `whole` columns
// element by element, a wider one by row digest. `g_only`: the fill alone
// (G, no BLAS and no solve), so the comparison runs where numpy bundles no
// OpenBLAS to hand the solve (macOS's Accelerate build); a Sommerfeld
// ground's remainder, which takes numpy's products, declines there.
std::string dump_fill(const std::string &latin1_text, const sinusoidal::FillLinalg &la, std::size_t whole = 64,
                      bool g_only = false);

}  // namespace sinusoidal_galerkin
}  // namespace mw
