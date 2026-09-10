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

// momwire#1032: the module NAME is a build parameter, so the same sources can
// be compiled twice — once with AVX2/FMA and once at the x86-64 baseline — and
// loaded by name at import time on a CPU that can run one but not the other.
// `PYBIND11_MODULE` pastes this token into the init symbol, so a `-D` here is
// what makes `PyInit__accelerators_sse2` exist. The default keeps an
// out-of-tree or single-variant build (macOS, arm64) building under the name
// it has always had.
#ifndef MOMWIRE_MODULE_NAME
#define MOMWIRE_MODULE_NAME _accelerators
#endif

PYBIND11_MODULE(MOMWIRE_MODULE_NAME, m) {
    // Phase 2: raised by the long kernels when their cancel_flag is tripped;
    // the _accel.py wrappers remap it to momwire.SolveAborted.
    py::register_exception<AbortedError>(m, "AcceleratorAborted");

    register_bspline(m);
    register_sinusoidal(m);
    register_somm(m);
    register_mw568(m);
    register_razor(m);
}

