from ._accel import LOADED as accelerated
from ._cancel import CancelToken, SolveAborted
from .bspline import BSplineSolver
from .hmatrix import HMatrixSolver
from .array_block import ArrayBlockSolver
from .sinusoidal import SinusoidalSolver

# Wire-material physics helpers (#133): the per-metre quantities behind the
# distributed wire loading, exported for consumers that mirror the loading
# into other tools (e.g. antennaknobs' NEC LD-5/LD-2 card emission).
from ._wire_loading import insulation_inductance, wire_internal_impedance

# `accelerated` is True iff the optional C++ accelerator loaded; consumers can
# assert it to guard against a silent fall-back to the slow pure-Python path.
__all__ = [
    "SinusoidalSolver",
    "BSplineSolver",
    "HMatrixSolver",
    "ArrayBlockSolver",
    "CancelToken",
    "SolveAborted",
    "accelerated",
    "wire_internal_impedance",
    "insulation_inductance",
]
