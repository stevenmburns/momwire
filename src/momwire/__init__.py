from ._accel import LOADED as accelerated
from ._cancel import CancelToken, SolveAborted

# The multi-port solve result (#232): every solver family's
# `compute_port_solution()` returns one of these — Y plus the per-port
# solution columns that one fill + one factorisation already produced.
from ._port_solution import PortSolution
from .bspline import BSplineSolver
from .hmatrix import HMatrixSolver
from .array_block import ArrayBlockSolver, LatticeFFTUnavailable
from .razor import RazorSolver
from .sinusoidal import SinusoidalSolver
from .sinusoidal_galerkin import SinusoidalGalerkinSolver

# Wire-material physics helpers (#133): the per-metre quantities behind the
# distributed wire loading, exported for consumers that mirror the loading
# into other tools (e.g. antennaknobs' NEC LD-5/LD-2 card emission).
from ._wire_loading import insulation_inductance, wire_internal_impedance

# `accelerated` is True iff the optional C++ accelerator loaded; consumers can
# assert it to guard against a silent fall-back to the slow pure-Python path.
__all__ = [
    "SinusoidalSolver",
    "SinusoidalGalerkinSolver",
    "BSplineSolver",
    "HMatrixSolver",
    "ArrayBlockSolver",
    "RazorSolver",
    "LatticeFFTUnavailable",
    "PortSolution",
    "CancelToken",
    "SolveAborted",
    "accelerated",
    "wire_internal_impedance",
    "insulation_inductance",
]
