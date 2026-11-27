"""bs2's census apart from its same-edge Toeplitz form, for tests.

`pair_reuse` turns both on and off together (momwire#2063 step 4). A test
that measures one alone holds the other through the solver's private
attribute, which the fill reads when it runs (`bspline._offset_reuse_fill`).
"""

from __future__ import annotations

from momwire.bspline import BSplineSolver


def bs2(*, toeplitz=None, cls=BSplineSolver, **kw):
    """``cls(**kw)``, its same-edge Toeplitz form held at `toeplitz` when that
    is not None (what `pair_reuse` alone decides otherwise)."""
    s = cls(**kw)
    if toeplitz is not None:
        s._offset_reuse = bool(toeplitz)
    return s
