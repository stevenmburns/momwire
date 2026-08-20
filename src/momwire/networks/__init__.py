"""The network-solve core: transmission lines and a Modified Nodal Analysis
system, on top of a raw multiport antenna admittance matrix.

Arriving from antennaknobs per ``docs/design/networks-move-into-the-engine.md``
(momwire#456 workstream 2), which records why the network solve belongs to the
engine rather than the app: two of the three drop-in seams momwire stands at
emit ``TL``/``NT`` cards, and NEC-2 and NEC-5 both serve them natively.

numpy and scipy only — no first-party imports, and scipy stays lazy inside
:meth:`MNASystem.solve`. This unit lands the TYPE-FREE half: the closed-form
TL/chain math and the MNA primitives, which never see a port name, a branch
dataclass or a `Network`. The spec layer (ports, branches, sources, the flat
`Network` container) and the `NetworkReducer` that stamps them land in the
next unit, in the same ``_reduce`` module so the moved file keeps the ordering
it had in antennaknobs.

`tl_abcd` is what the reducer stamps; `tl_admittance_2x2` and
`balanced_admittance_4x4` are the closed forms the composition oracles are
written against (issue #746 moved the stamp off the admittance, which does not
exist at a lossless half-wave, onto the chain matrix, which is entire).
"""

from __future__ import annotations

from .._constants import C_LIGHT
from ._reduce import (
    RCOND_SINGULAR,
    RCOND_SUSPECT,
    MNASystem,
    SingularNetworkError,
    balanced_admittance_4x4,
    magnetizing_impedance,
    poison_singular_sample,
    tl_abcd,
    tl_admittance_2x2,
)

__all__ = [
    "C_LIGHT",
    "RCOND_SINGULAR",
    "RCOND_SUSPECT",
    "MNASystem",
    "SingularNetworkError",
    "balanced_admittance_4x4",
    "magnetizing_impedance",
    "poison_singular_sample",
    "tl_abcd",
    "tl_admittance_2x2",
]
