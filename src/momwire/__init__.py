from ._accel import LOADED as accelerated
from ._cancel import CancelToken, SolveAborted
from .bspline import BSplineSolver
from .hmatrix import HMatrixSolver
from .array_block import ArrayBlockSolver
from .sinusoidal import SinusoidalSolver

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
]
