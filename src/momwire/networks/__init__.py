"""The network-solve core: transmission lines and a Modified Nodal Analysis
system, on top of a raw multiport antenna admittance matrix.

Arriving from antennaknobs per ``docs/design/networks-move-into-the-engine.md``
(momwire#456 workstream 2), which records why the network solve belongs to the
engine rather than the app: two of the three drop-in seams momwire stands at
emit ``TL``/``NT`` cards, and NEC-2 and NEC-5 both serve them natively.

numpy and scipy only — no first-party imports beyond `momwire._constants`, and
scipy stays lazy inside :meth:`MNASystem.solve`. Three modules:

* `._reduce` — the TYPE-FREE core: the closed-form TL/chain math and the MNA
  primitives, which never see a port name, a branch dataclass or a `Network`.
* `._spec` — ports, branches, sources and the flat `Network` container. It
  reuses `._reduce`'s series-RLC helper rather than carrying a second copy,
  which is why the dependency runs spec -> reduce and not the reverse.
* `._reducer` — `NetworkReducer`, which needs both and so is its own module
  (the antennaknobs original was one file and had no such cycle to dodge).

`tl_abcd` is what the reducer stamps; `tl_admittance_2x2` and
`balanced_admittance_4x4` are the closed forms the composition oracles are
written against (issue #746 moved the stamp off the admittance, which does not
exist at a lossless half-wave, onto the chain matrix, which is entire).

The `Network` here is FLAT: authoring hierarchy (`Composite`/`Instance`) stays
in antennaknobs, which flattens before calling down and passes the resulting
``branch_paths``. `Cable` is the formal line spec `TL.from_cable` takes; the
cable CATALOG stays in antennaknobs, as does the Touchstone parser behind the
`AdmittanceData` protocol.
"""

from __future__ import annotations

from .._constants import C_LIGHT
from ._reduce import (
    FEET_PER_M,
    NEPER_PER_DB,
    RCOND_SINGULAR,
    RCOND_SUSPECT,
    Z_REF_DEFAULT,
    MNASystem,
    SingularNetworkError,
    balanced_admittance_4x4,
    magnetizing_impedance,
    poison_singular_sample,
    tl_abcd,
    tl_admittance_2x2,
)
from ._reducer import NetworkReducer
from ._spec import (
    TL,
    Admittance,
    AdmittanceData,
    Autotransformer,
    BalancedLine,
    Branch,
    Cable,
    Driven,
    DrivenCurrent,
    FloatingBalun,
    Load,
    Network,
    Port,
    PortAtEdge,
    PortAtEnd,
    PortAtVertex,
    PortOnWire,
    PortOnWireFloating,
    PortVirtual,
    Shunt,
    Source,
    TouchstoneLoad,
    TouchstoneTwoPort,
    Transformer,
    TwoPort,
    load_impedance,
    load_series_admittance,
)

__all__ = [
    "C_LIGHT",
    "FEET_PER_M",
    "NEPER_PER_DB",
    "RCOND_SINGULAR",
    "RCOND_SUSPECT",
    "Z_REF_DEFAULT",
    "MNASystem",
    "SingularNetworkError",
    "balanced_admittance_4x4",
    "magnetizing_impedance",
    "poison_singular_sample",
    "tl_abcd",
    "tl_admittance_2x2",
    "NetworkReducer",
    "TL",
    "Admittance",
    "AdmittanceData",
    "Autotransformer",
    "BalancedLine",
    "Branch",
    "Cable",
    "Driven",
    "DrivenCurrent",
    "FloatingBalun",
    "Load",
    "Network",
    "Port",
    "PortAtEdge",
    "PortAtEnd",
    "PortAtVertex",
    "PortOnWire",
    "PortOnWireFloating",
    "PortVirtual",
    "Shunt",
    "Source",
    "TouchstoneLoad",
    "TouchstoneTwoPort",
    "Transformer",
    "TwoPort",
    "load_impedance",
    "load_series_admittance",
]
