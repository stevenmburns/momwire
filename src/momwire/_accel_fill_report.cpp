#include "_accel_common.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

// The fill report's wire map on the sinusoidal fill (momwire#2016): one pass
// over a `_sin_pair_reuse` plan's class map, run only when the map is read.
// The NumPy spelling of the same count took about 0.6 s at 3,000 segments.
// Its own TU, so that no kernel's codegen moves with it (momwire#1187).

namespace fill_report {

using I32 = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;
using I64 = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

// (W, W, 3) int64: per observer wire and source wire, the pairs computed (the
// first pair of a far class, row-major), reused and near. `near_flat` is the
// plan's sorted flat index of its near pairs, which share one sentinel class.
py::array_t<int64_t> wire_map(I32 cls, int code_bits, I64 near_flat, I32 wire_of_row, I32 wire_of_col,
                              int64_t n_classes, int64_t n_wires) {
    if (cls.ndim() != 2) throw std::invalid_argument("fill_report: the class map is (rows, sources)");
    const int64_t M = cls.shape(0), N = cls.shape(1), W = n_wires;
    if (wire_of_row.size() != M || wire_of_col.size() != N)
        throw std::invalid_argument("fill_report: one wire per row and per source");
    if (W < 1 || n_classes < 0 || code_bits < 0 || code_bits > 8)
        throw std::invalid_argument("fill_report: bad wire or class count");
    const int32_t *wr = wire_of_row.data(), *wc = wire_of_col.data();
    for (int64_t m = 0; m < M; m++)
        if (wr[m] < 0 || wr[m] >= W) throw std::invalid_argument("fill_report: wire out of range");
    for (int64_t n = 0; n < N; n++)
        if (wc[n] < 0 || wc[n] >= W) throw std::invalid_argument("fill_report: wire out of range");
    py::array_t<int64_t> out({static_cast<py::ssize_t>(W), static_cast<py::ssize_t>(W), static_cast<py::ssize_t>(3)});
    int64_t *o = out.mutable_data();
    const int32_t *map = cls.data();
    const int64_t *near = near_flat.data();
    const int64_t n_near = near_flat.size();
    {
        py::gil_scoped_release release;
        std::fill(o, o + W * W * 3, int64_t(0));
        std::vector<uint64_t> seen(static_cast<size_t>((n_classes + 63) / 64), 0);
        int64_t at = 0;
        for (int64_t m = 0; m < M; m++) {
            int64_t *row = o + static_cast<int64_t>(wr[m]) * W * 3;
            for (int64_t n = 0; n < N; n++) {
                int64_t *cell = row + static_cast<int64_t>(wc[n]) * 3;
                if (at < n_near && near[at] == m * N + n) {
                    at++;
                    cell[2]++;
                    continue;
                }
                const uint64_t c = static_cast<uint64_t>(map[m * N + n]) >> code_bits;
                if (static_cast<int64_t>(c) >= n_classes) throw std::invalid_argument("fill_report: class out of range");
                const uint64_t bit = uint64_t(1) << (c % 64);
                if (seen[c / 64] & bit) {
                    cell[1]++;
                } else {
                    seen[c / 64] |= bit;
                    cell[0]++;
                }
            }
        }
    }
    return out;
}

}  // namespace fill_report

void register_fill_report(py::module_ &m) {
    m.def("fill_report_wire_map", &fill_report::wire_map, py::arg("cls"), py::arg("code_bits"),
          py::arg("near_flat"), py::arg("wire_of_row"), py::arg("wire_of_col"), py::arg("n_classes"),
          py::arg("n_wires"),
          "The fill report's wire map of a sinusoidal reuse plan: (W, W, 3) int64, per observer "
          "wire and source wire the pairs computed, reused and near (momwire#2016).");
}
