// The fill report's wire map on razor-2p (momwire#2016): one pass over the
// reuse plan's class maps. Its own TU, so that no kernel's codegen moves with
// it (momwire#1187): integers only, and nothing here is called by a fill.
#include "_razor_pair_reuse_core.h"

#include <vector>

namespace mw {
namespace razor {
namespace reuse {

void wire_map(const Plan &p, const int32_t *wire_of_seg, int64_t n_wires, int64_t *out) {
    const int64_t n = p.n_seg, W = n_wires;
    std::fill(out, out + 3 * W * W * 3, int64_t(0));
    if (n == 0) return;
    // A class is computed at its first pair in the census's order (block by
    // block, row-major), which is the pair the census made its representative.
    std::vector<uint64_t> seen(static_cast<size_t>((p.n_dcls + 63) / 64), 0);
    for (int b = 0; b < (p.has_image ? 2 : 1); b++) {
        if (p.dmap[b].empty()) continue;
        const int32_t *map = p.dmap[b].data();
        int64_t *fam = out + b * W * W * 3;
        for (int64_t o = 0; o < n; o++) {
            int64_t *row = fam + static_cast<int64_t>(wire_of_seg[o]) * W * 3;
            for (int64_t s = 0; s < n; s++) {
                const uint64_t c = static_cast<uint64_t>(map[o * n + s] >> 1);
                const uint64_t bit = uint64_t(1) << (c % 64);
                int64_t *cell = row + static_cast<int64_t>(wire_of_seg[s]) * 3;
                if (p.dnear_bits[c / 64] & bit) {
                    cell[2]++;
                } else if (seen[c / 64] & bit) {
                    cell[1]++;
                } else {
                    seen[c / 64] |= bit;
                    cell[0]++;
                }
            }
        }
    }
    if (!p.remainder || p.qmap.empty()) return;
    std::vector<uint64_t> qseen(static_cast<size_t>((p.n_qcls + 63) / 64), 0);
    const int32_t *map = p.qmap.data();
    int64_t *fam = out + 2 * W * W * 3;
    for (int64_t o = 0; o < n; o++) {
        int64_t *row = fam + static_cast<int64_t>(wire_of_seg[o]) * W * 3;
        for (int64_t s = 0; s < n; s++) {
            const uint64_t c = static_cast<uint64_t>(map[o * n + s] >> 2);
            const uint64_t bit = uint64_t(1) << (c % 64);
            int64_t *cell = row + static_cast<int64_t>(wire_of_seg[s]) * 3;
            if (qseen[c / 64] & bit) {
                cell[1]++;
            } else {
                qseen[c / 64] |= bit;
                cell[0]++;
            }
        }
    }
}

}  // namespace reuse
}  // namespace razor
}  // namespace mw
