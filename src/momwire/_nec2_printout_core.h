// The NEC-2 portal's printout in C++ (momwire#2065), with no Python in it:
// one deck body's frame, as `momwire.dropin.nec2c._portal`'s `render_deck`
// and `deck_frame` write it, over one basis' solver (`nec4::RunEngine`).
//
//   * the deck is read by the nec2 / nec4 reader in its nec2 dialect
//     (`_nec4_deck_core.h`) and walked by the portal's own card walk
//     (`nec4::parse_portal_deck`, `parse_deck(body)`);
//   * the solver is `DeckSolver`'s construction (`nec4::build_portal_structure`)
//     and its run half (`nec4::PortalSolve`: `solve_group`, the readouts,
//     the near field) -- the same code the NEC-4.2 slot's executables run;
//   * what is this file's own is the NEC-2 printout: the banner, the comment
//     echo, STRUCTURE SPECIFICATION (with GA / GH, the transforms and the
//     ground-plane lines), MULTIPLE WIRE JUNCTIONS, SEGMENTATION DATA with
//     NEC-2's connection columns, the DATA CARD echo, each execute group's
//     block (FREQUENCY, LOADING, ANTENNA ENVIRONMENT, the MP line, the end
//     cap's withdrawal line, MATRIX TIMING, NETWORK DATA and the excitation
//     at its connection points, ANTENNA INPUT PARAMETERS, CURRENTS AND
//     LOCATION, CHARGE DENSITIES, POWER BUDGET), RADIATION PATTERNS (RP 0,
//     2 and 3, through `_far_readout.pattern_fields`) and the NEAR FIELDS,
//     and the two-line ERROR frame every refusal takes.
//
// Python stays the reference: every frame is `deck_frame`'s, byte for byte,
// with the MATRIX TIMING line's two measured numbers the only difference
// (`tests/test_standalone_nec2c.py`). A deck this copy does not answer --
// one Python would raise on rather than refuse, or one the C++ route does
// not solve (a reflection-coefficient ground, say) -- is DECLINED: `why`
// says what, and the executable prints it as a refusal frame naming it.
//
// Lines are UTF-8 (a deck's text is latin-1 and is carried over as its code
// points), with no line ends: `\n` joins them, as `"\n".join(out)` does.
#pragma once

#include "_nec4_run_core.h"

#include <string>
#include <vector>

namespace mw {
namespace nec2 {

// The printout banner (`_banner_lines`), the basis' suffix on its VERSION
// line ("+sin" for the point-matched sinusoidal basis).
std::vector<std::string> banner_lines(const std::string &suffix);

// `fmt_data_card(number, card)`: the DATA CARD echo, and the NX / EN
// sentinel (`Card(terminator, (), terminator)`, every field zero).
std::string fmt_data_card(long long number, const nec5::Card &card);
std::string sentinel_card(long long number, const std::string &terminator);

// `_append_error(out, PortalError(reason))`'s two lines.
void append_error(std::vector<std::string> &out, const std::string &utf8_reason);

struct DeckOptions {
    std::string basis = "sinusoidal";
    // `nec2_end_cap` on every operating point (`nec4::SolverArgs`): 0 off,
    // 1 asked for, 2 the NEC-2 launcher's own default
    int nec2_end_cap = 0;
};

// `render_deck(body)`: (stdout lines, stderr lines) for one deck body, no
// banner and no NX echo. False (and nothing in `out`) with `why` when this
// copy declines the deck. An exception that is not a decline (memory, say)
// escapes.
bool render_deck(const std::string &latin1_body, const DeckOptions &options, nec4::EngineFactory &engines,
                 const nec5::NumpyLinalg &np, std::vector<std::string> &out, std::vector<std::string> &err,
                 std::string &why);

// The largest solve (segments) a deck body's structure asks for, or -1
// (the executable's thread policy).
long long largest_solve(const std::string &latin1_body);

}  // namespace nec2
}  // namespace mw
