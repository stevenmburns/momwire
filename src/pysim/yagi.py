import numpy as np

from .abstract import AbstractPySim
from . import Integral_Standalone


class YagiPySim(AbstractPySim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_impedance(self, *, ntrap=0):
        N = self.nsegs
        h = self.halfdriver

        # Two parallel y-directed wires: driver at x=0, reflector at x=-h (5% longer).
        wires = [
            (np.array((0.0, -h, 0.0)), np.array((0.0, +h, 0.0))),
            (np.array((-h, -1.05 * h, 0.0)), np.array((-h, +1.05 * h, 0.0))),
        ]

        seg_l_list, seg_r_list = [], []
        sub_l_list, sub_r_list = [], []
        # Per-segment indices into the flattened sub-segment array.
        # Wire boundaries are NOT adjacent — handled by separate offsets.
        node_l_idx_list, node_r_idx_list = [], []
        sub_offset = 0

        for w0, w1 in wires:
            delta = (w1 - w0) / (2 * N)
            exnm = np.linspace(w0 - delta, w1 + delta, 2 * N + 3)

            wire_endpoints = exnm[1:-1:2, :]
            seg_l_list.append(wire_endpoints[:-1])
            seg_r_list.append(wire_endpoints[1:])

            midpoints_ext = exnm[::2, :]
            sub_l_list.append(midpoints_ext[:-1])
            sub_r_list.append(midpoints_ext[1:])

            node_l_idx_list.append(sub_offset + np.arange(N))
            node_r_idx_list.append(sub_offset + np.arange(1, N + 1))
            sub_offset += N + 1

        seg_l = np.vstack(seg_l_list)
        seg_r = np.vstack(seg_r_list)
        sub_l = np.vstack(sub_l_list)
        sub_r = np.vstack(sub_r_list)
        node_l_idx = np.concatenate(node_l_idx_list)
        node_r_idx = np.concatenate(node_r_idx_list)

        vec_delta_l = seg_l - seg_r

        z = (
            self.jomega
            * self.mu
            * (vec_delta_l[np.newaxis, :, :] * vec_delta_l[:, np.newaxis, :]).sum(
                axis=2
            )
        )
        z *= Integral_Standalone(
            seg_l, seg_r, ntrap=ntrap, wire_radius=self.wire_radius, k=self.k
        )

        s = (
            1
            / (self.jomega * self.eps)
            * Integral_Standalone(
                sub_l, sub_r, ntrap=ntrap, wire_radius=self.wire_radius, k=self.k
            )
        )

        z += (
            s[np.ix_(node_l_idx, node_l_idx)]
            + s[np.ix_(node_r_idx, node_r_idx)]
            - s[np.ix_(node_l_idx, node_r_idx)]
            - s[np.ix_(node_r_idx, node_l_idx)]
        )

        self.z = z

        return self.factor_and_solve()
