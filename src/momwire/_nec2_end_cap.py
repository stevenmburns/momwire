"""NEC-2's free-end condition on the point-matched sinusoidal basis
(momwire#2033): a labelled compatibility option, off by default.

The default basis takes the current to zero at a free wire end. NEC-2 lets it
flow onto the end cap instead. The public program description (Burke and
Poggio, NEC Part I, 1981, Eq. 23 and the definition under Eq. 25) relates the
current at a free end to its derivative there,

    I(s_i +- D_i/2) = +-(X_i / k) dI/ds,    X_i = J1(k a_i) / J0(k a_i),

and writes the end-segment basis for that X in Eqs. 54-61 and 64. Setting
X = 0 gives the zero-current end.

`_sinusoidal_nec4.basis_coefs_per_end` already carries those end branches for
any X, so this module only supplies NEC-2's X. Both charge factors there are
the log rule's, Eq. 25, because NEC-2 has one constant per segment.

This is not a better model of a real wire end. A flat cap is worth about 0.10a
of extra length per end, and this condition measures 3-4 times that on coarse
meshes (momwire#1327). The option exists so that momwire can be compared with
NEC-2 engines on equal terms.

Verified against nec2c on the NEC-2 route (`momwire-nec2c-sinusoidal`) only.
`momwire-nec42-sinusoidal` honours the option on a deck with no junction the
NEC-4 rule solves, but no reference has been compared on that route, whose
kernel also differs from the NEC-2 route's. A deck with such a junction and a
free end refuses: the NEC-4 route carries its own end cap, so this one could
not act there.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j0, j1

from . import _sinusoidal_nec4

# NEC-2 has no wire below the interface, so nothing defines this condition at
# a complex wavenumber or across a crossing.
BELOW_INTERFACE_REFUSAL = (
    "nec2_end_cap is NEC-2's free-end condition and NEC-2 has no buried or "
    "ground-crossing wires; turn it off for this deck"
)

NEC4_JUNCTION_REFUSAL = (
    "nec2_end_cap cannot act on a deck the NEC-4 junction rule solves, whose "
    "route carries its own end cap; turn it off for this deck"
)


def has_free_end(geom) -> bool:
    """Whether any segment of `geom` ends in free space."""
    free_m, free_p = _sinusoidal_nec4.free_ends(geom)
    return bool(free_m.any() or free_p.any())


def end_x(geom, k, seg_radius):
    """NEC-2's X = J1(ka)/J0(ka) on every segment with a free end, else 0."""
    free_m, free_p = _sinusoidal_nec4.free_ends(geom)
    ka = k * np.asarray(seg_radius, dtype=np.float64)
    return np.where(free_m | free_p, j1(ka) / j0(ka), 0.0)


def basis_view(geom, k, a_const, seg_radius):
    """`_basis_coefs`' CSR-by-segment view with NEC-2's free ends (real `k`).

    `a_const` is the log-rule charge factor, a scalar or one per segment."""
    a = np.broadcast_to(a_const, (geom["n_segs"],)).astype(np.float64)
    return _sinusoidal_nec4.basis_coefs_per_end(
        geom, k, a, a, end_x(geom, k, seg_radius)
    )
