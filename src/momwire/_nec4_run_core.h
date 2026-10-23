// The NEC-4.2 slot's RUN HALF in C++ (momwire#1607 B4a), with no Python in
// it: deck text to the `Nec4Printout` A3's writer prints, over one basis'
// solver (`RunEngine`). It is the copy of `eznec/_nec4.py`'s `solve` and
// `_run` and of the parts of `portal._portal.DeckSolver` they call:
// - the deck tables: the wire table (`_wire_rows`), the GE flag, the
//   junctions (`_segment_end_nodes`) and the segmentation rows with NEC-4.2's
//   junction rule for the connection columns (`_connections`);
// - each run's port algebra (`DeckSolver.solve_group` with no network):
//   the LD 0-4 port impedances (`_load_impedances`), `(1 + Z·Y)⁻¹·V` by
//   numpy's solve, a lone current source scaled to its set current, the
//   coefficients `X·(signs·V_gap)`, the segment currents
//   (`_segment_currents`: the element match and the knot means) and charges
//   (`_segment_charges`, and `_charges`' free-end charges), and the power
//   budget;
// - the rows (`_run`, `_run_records`' source rows, `_charge_rows`, `value_row`)
//   and RP (`_pattern` over the NEC-5 seam's row builder,
//   `nec5::replay::free_space_pattern`, then `_at_range`).
//
// The basis answers three questions (`RunEngine`): its port solution, its
// current at every knot, and dI/ds at given arclengths. The sinusoidal
// basis' answers are `_sinusoidal_run_core.h`'s.
//
// Python stays the reference: the printout this produces is
// `_nec4.render(text, basis=...)`'s byte for byte, except the two lines that
// carry measured time (FILL= ... SEC. and RUN TIME), gated by
// `tests/test_nec4_run_core.py` through `scripts/core_split/nec4_run_compare.py`.
//
// DECLINED (`Declined`), never misprinted: GD's far-field cliff (B5d), a
// wire below the ground plane (B6), an insulated wire, a TL / NT network and
// NE / NH (B4c), two sources on one segment of a phased drive, and
// any run Python would answer with a refusal this copy does not word (a
// singular drive block, a current source on a port that takes no current).
#pragma once

#include "_nec4_printout_core.h"

#include <complex>
#include <string>
#include <vector>

namespace mw {
namespace nec4 {

// A refusal the reference raises as a ValueError inside `solve` (the solver
// constructor's, say), which `_nec4.solve` turns into the structure's
// refusal printout with the exception's words: an engine throws it with
// Python's message.
struct Refusal : std::runtime_error {
    std::string message;  // UTF-8
    explicit Refusal(const std::string &m) : std::runtime_error(m), message(m) {}
};

// One operating point's `compute_port_solution()`: Y (n_ports x n_ports,
// C order) and the coefficients X (n_basis x n_ports) with the element
// strides numpy holds them at.
struct EngineSolve {
    long long n_basis = 0, n_ports = 0;
    std::vector<std::complex<double>> y;
    std::vector<std::complex<double>> x;
    long long x_s0 = 0, x_s1 = 0;
    double fill_seconds = 0;  // measured, for the printout's MATRIX TIMING
};

class RunEngine {
  public:
    virtual ~RunEngine() = default;
    // `solver.compute_port_solution()` at this operating point (`index`:
    // its position in the structure's `points`).
    virtual const EngineSolve &solve(std::size_t index) = 0;
    // `solver.currents_at_knots(coeffs)`: per solver polyline, the current at
    // each knot.
    virtual std::vector<std::vector<std::complex<double>>> currents_at_knots(
        std::size_t index, const std::vector<std::complex<double>> &coeffs) = 0;
    // `solver.current_slopes(coeffs, s)`: per polyline, dI/ds at each arclength.
    virtual std::vector<std::vector<std::complex<double>>> current_slopes(
        std::size_t index, const std::vector<std::complex<double>> &coeffs,
        const std::vector<std::vector<double>> &s) = 0;
    // `solver.wire_loss_power(coeffs)[0]`: the wires' own loss (0 with no
    // wire loading).
    virtual double wire_loss_power(std::size_t index, const std::vector<std::complex<double>> &coeffs) = 0;
};

// Builds the engine for one built structure (its operating points).
class EngineFactory {
  public:
    virtual ~EngineFactory() = default;
    virtual RunEngine &engine_for(const StructureBuild &structure) = 0;
};

// `_nec4.render(text, basis=...)` over the engine: true with the printout
// (UTF-8, LF), or false with why this copy declines the deck. An exception
// that is not a decline (an allocation, say) escapes, as Python's would.
bool render(const std::string &latin1_text, const std::string &basis,
            const nec5::EngineStamp &stamp, EngineFactory &engines,
            const nec5::NumpyLinalg &np, std::string &utf8_printout, std::string &why);

// `_shell._stream_piece(text, dialect="nec4")` (momwire#1512): a piece cut
// at an NX is that structure (`_nec4.render_structure` with its NX echo and
// no RUN TIME), and `go_on` says it solved; any other piece is a whole deck
// (`render`), after which the stream is over.
bool render_piece(const std::string &latin1_text, const std::string &basis,
                  const nec5::EngineStamp &stamp, EngineFactory &engines,
                  const nec5::NumpyLinalg &np, std::string &utf8_printout, std::string &why,
                  bool &go_on);

// The largest solve (segments) the deck's built structures prepare, or -1
// (the executable's thread policy).
long long largest_solve(const std::string &latin1_text, const std::string &basis, bool open_end);

}  // namespace nec4
}  // namespace mw
