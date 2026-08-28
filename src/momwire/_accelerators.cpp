#include "_accel_common.h"

// Thin module TU (momwire#687): the sections register themselves.

void register_bspline(py::module_ &m);
void register_sinusoidal(py::module_ &m);
void register_somm(py::module_ &m);
void register_mw568(py::module_ &m);

PYBIND11_MODULE(_accelerators, m) {
    // Phase 2: raised by the long kernels when their cancel_flag is tripped;
    // the _accel.py wrappers remap it to momwire.SolveAborted.
    py::register_exception<AbortedError>(m, "AcceleratorAborted");
    // Capability flag, not a value (momwire#258). The two EKSCX entry points
    // dropped their build-wide `want_swapped` argument when the IRA arm went
    // per pair, and a STALE extension still exports both symbols under the
    // old arity — so `hasattr` alone would hand the new caller a TypeError
    // instead of the graceful numpy fallback the guards exist to give.
    // `sinusoidal.py` requires this attribute before it claims either
    // accelerator; an older build simply lacks it and takes the numpy
    // reference, which carries the same per-pair fix.
    m.attr("ek_ira_per_pair") = true;
    // momwire#568 unit 1: the shared contour engine's TEST entry points. The
    // engine itself is header-only (`_contour_engine_inline.h`); these three
    // exist so the Python suite can gate it before U2/U3 ride on it.
    m.attr("contour_engine_568") = true;
    // momwire#568 unit 2: the below/below fills on that engine. Its OWN
    // capability flag, deliberately not `contour_engine_568` — a .so built at
    // U1 exports the engine's test entry points and would otherwise claim to
    // carry U2's contract too, handing `_sommerfeld_below` a missing symbol
    // instead of the graceful numpy fallback the guard exists to give.
    m.attr("below_fills_568") = true;
    // momwire#568 unit 3: the transmitted fills on that engine. Its OWN
    // capability flag, deliberately neither `contour_engine_568` nor
    // `below_fills_568` — a .so built at U1 or U2 exports those symbols and
    // would otherwise claim to carry U3's contract too, handing
    // `_sommerfeld_transmitted` a missing symbol instead of the graceful numpy
    // fallback the guard exists to give.
    m.attr("transmitted_fills_568") = true;

    register_bspline(m);
    register_sinusoidal(m);
    register_somm(m);
    register_mw568(m);
}

