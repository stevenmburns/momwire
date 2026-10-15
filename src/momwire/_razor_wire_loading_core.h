// WIRE loading for razor-2p's C++ path (momwire#1479): the per-metre series
// impedance `_wire_loading.series_impedance_per_wire` puts on each wire -- an
// LD 5 conductivity's internal impedance (skin effect, through the vendored
// xsf's `cyl_bessel_ie`, which IS scipy.special.ive's code: measured equal on
// 25,000 arguments on AVX-512 and AVX2) and an LD 2 / LD 3 per-metre RLC --
// with numpy's arithmetic, operation for operation (its 0-d array ops, then
// its scalar math). Python-free.
//
// The ONE translation unit that includes xsf in a link unit: its headers
// define functions non-inline (setup.py's note on the near-interface twins).
#pragma once

#include "_mw_core.h"

#include <cstdint>
#include <vector>

namespace mw {
namespace razor {

// One wire's `DistributedRLC` (LD 2 series / LD 3 parallel), or none.
struct WireRLC {
    enum Kind : int8_t { none = 0, series = 1, parallel = 2 } kind = none;
    double r = 0, l = 0, c = 0;
};

// The loading `configure_loading` keeps: `wire_conductivity` (absent, or one
// entry per wire, NaN / inf switching a wire off) and `distributed_rlc`
// (absent, or one entry per wire).
struct WireLoading {
    bool has_conductivity = false;
    std::vector<double> conductivity;
    bool has_rlc = false;
    std::vector<WireRLC> rlc;
    bool active() const { return has_conductivity || has_rlc; }
};

// `series_impedance_per_wire(omega, radius, conductivity, None, None, rlc)` at
// a real scalar omega: (n_wires,) complex, Ohm/m.
std::vector<cd> series_impedance_per_wire(double omega, const std::vector<double> &radius,
                                          const WireLoading &loading);

}  // namespace razor
}  // namespace mw
