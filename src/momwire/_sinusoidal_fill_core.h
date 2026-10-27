// The sinusoidal basis' free-space FILL AND SOLVE in C++ (momwire#1607 B3),
// with no Python in it. Over B2's front half (`front`) it gives what
// `SinusoidalSolver.compute_port_solution` computes for a structure in free
// space:
// - `_assemble_Z`: the observer-row chunks (`_schedule.mb_rows` at the
//   default `swept_mem_mb`), each chunk's field tensors from B1's kernels
//   exactly as `_field_tensor` calls them (`sinusoidal_field_tensor_ek` with
//   `_ek_gating`'s labels, or `sinusoidal_field_tensor`; one call per radius
//   run on a stepped radius) -- or, for the folded `cos-1` shape (min k·Δ
//   below `_WELL_SCALED_KD`), numpy's reference route (`_field_components`
//   with `_extended_kernel_fields`' cos-1 branch) element by element -- and
//   the reduction into Z: numpy's zgemm below
//   `_DENSE_ASSEMBLY_THRESHOLD`, scipy's `csr_matvecs` over `_REDUCE_TILE`-
//   row tiles above it (each element its own zero plus its column's stored
//   entries in row order, one unfused complex multiply-add each, the three
//   products summed `(A + B) + C`);
// - the port solve: `_lapack.solve(G, B)`, which is `scipy.linalg.solve`'s
//   general route -- getrf on the Fortran copy, getrs 'N' -- on whichever
//   LAPACK the caller hands in (the binding passes SciPy's own; an
//   executable its bundled OpenBLAS, as `_lapack`'s host route does);
// - `_feed_segment_current` at each feed segment's centre, into Y;
// - a wholly-buried deck (momwire#1607 B6a), the whole fill at the lower
//   medium's k_m / eta_m (`_operating_medium`): the complex twin
//   (`sinusoidal_field_tensor_cplx`), the image weighted A_m, and the
//   below-family remainder (`_somm_remainder_below_prepare` / `_replay`: the
//   below grid from `somm_below::source()`, `remainder_field_proj_below`
//   through the shared `somm_below::Remainder`), sized by `serve_plan`'s
//   below/below extents;
// - a crossing deck (momwire#1607 B6b, `_assemble_Z_mixed`): each medium's
//   class block over its own sources at its own (k, eta) and ground (above:
//   the reduced kernel and the Sommerfeld remainder over the class; below:
//   the complex twin and the below remainder), reduced into Z, then the
//   point rows (`_sinusoidal_crossing_core.h`) and the loading.
//
// Python stays the reference: Z, the coefficients and Y are Python's to the
// bit, gated by `tests/test_sinusoidal_fill_core.py` through
// `scripts/core_split/sinusoidal_fill_compare.py`.
//
// DECLINED (`nec4::Declined`), never misread:
// - a solver without the extended kernel above ground (the NEC-4.2 slot
//   never builds one), and one with it below (the reference refuses);
// - a below/below pair under the grazing floor (not worded here);
// - on a crossing deck, the folded cos-1 shape (the reduced kernel's numpy
//   route) and a repeated (segment, basis) entry;
// - an insulated wire;
// - a matrix scipy's solve would NOT take as general: exactly symmetric or
//   Hermitian, or of lower or upper bandwidth at most 1 (its diagonal,
//   triangular and tridiagonal detection), or not finite; a singular one
//   (scipy's LinAlgError);
// - a basis view with a repeated (segment, basis) pair (scipy would sum it,
//   numpy's assignment would keep the last).
#pragma once

#include "_nec5_printout_core.h"
#include "_razor_solve_core.h"
#include "_sinusoidal_front_core.h"

#include <complex>
#include <string>
#include <vector>

namespace mw {
namespace sinusoidal {

struct FillLinalg {
    // numpy's own CBLAS (the dense reduction's `Φ @ M`, N < 60)
    nec5::NumpyLinalg np;
    // the LU pair `scipy.linalg.solve` reaches
    razor::Lapack lu;
};

struct PortSolution {
    long long n = 0, n_ports = 0;
    std::vector<std::complex<double>> G;       // (n, n), C order: `_assemble_Z`'s
    std::vector<std::complex<double>> coeffs;  // (n, n_ports), C order
    std::vector<std::complex<double>> y;       // (n_ports, n_ports), C order
};

// `_assemble_Z(geom, k)` in free space: Z in C order.
std::vector<std::complex<double>> assemble_Z(const Front &fr, const nec4::SolverArgs &args,
                                             const nec5::NumpyLinalg &np);

// `loading_for(solver, omega).z_wire`: each wire's per-metre series
// impedance (LD 5, LD 2 / LD 3), false when the solver carries no wire
// loading (`_loading_active`). Declines an insulated wire.
bool wire_impedance_per_wire(const Front &fr, const nec4::SolverArgs &args, double omega,
                             std::vector<std::complex<double>> &z_wire);

// `compute_port_solution()` in free space.
PortSolution port_solution(const Front &fr, const nec4::SolverArgs &args, const FillLinalg &la);

// The comparison format (`sinusoidal_fill_compare.py`), per operating point
// of each built structure of a deck: a matrix of up to `whole` columns
// element by element, a wider one by row digest.
std::string dump_fill(const std::string &latin1_text, const FillLinalg &la, std::size_t whole = 64);

}  // namespace sinusoidal
}  // namespace mw
