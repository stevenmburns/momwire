#include "_accel_common.h"

// Thin module TU (momwire#687): the sections register themselves; their
// prototypes live in _accel_common.h so every TU compiles against the seam.
//
// The capability flags live INSIDE the register_* body that defines the
// symbols they vouch for (ek_ira_per_pair in register_sinusoidal, the three
// #568 flags in register_mw568, razor_fill_742 in register_razor) — flag and
// binding in one TU, so an edit
// cannot advertise a contract whose symbols moved out from under it
// (#710 review; `_sommerfeld_below`/`_sommerfeld_transmitted` trust the
// flag alone, with no hasattr backstop).

PYBIND11_MODULE(_accelerators, m) {
    // Phase 2: raised by the long kernels when their cancel_flag is tripped;
    // the _accel.py wrappers remap it to momwire.SolveAborted.
    py::register_exception<AbortedError>(m, "AcceleratorAborted");

    register_bspline(m);
    register_sinusoidal(m);
    register_somm(m);
    register_mw568(m);
    register_razor(m);
}

