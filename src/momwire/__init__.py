from ._accel import LOADED as accelerated
from ._cancel import CancelToken, SolveAborted
from ._capabilities import Capabilities

# The multi-port solve result (#232): every solver family's
# `compute_port_solution()` returns one of these — Y plus the per-port
# solution columns that one fill + one factorisation already produced.
from ._port_solution import PortSolution
from .bspline import BSplineSolver
from .harrington import HarringtonSolver
from .hmatrix import HMatrixSolver
from .array_block import ArrayBlockSolver, LatticeFFTUnavailable

# The element-grouping rule the array solver partitions by (#932), exported for
# consumers that must ask "is this an array?" before choosing a solver — one
# spelling of the geometry rule rather than a drifting copy.
from .array_block import wire_to_element
from .pulse import PulseSolver
from .razor import RazorSolver
from .sinusoidal import SinusoidalSolver
from .sinusoidal_galerkin import SinusoidalGalerkinSolver

# Wire-material physics helpers (#133): the per-metre quantities behind the
# distributed wire loading, exported for consumers that mirror the loading
# into other tools (e.g. antennaknobs' NEC LD-5/LD-2 card emission).
from ._wire_loading import insulation_inductance, wire_internal_impedance

# The two interface-side geometry answers (#855), on `wire_to_element`'s
# precedent and for the same reason: a consumer that must refuse a deck the
# way momwire refuses it has to answer these IDENTICALLY, and the only way to
# do that before this export was to reach through a private name.
#
# The irony is the argument. #848 exists because two copies of the exemption
# test — bspline's and razor's — drifted apart and answered differently; it
# put the geometry in one place so they could not. A consumer with no public
# path was then obliged to either import privately or write the third copy,
# which is the failure #848 had just finished repairing one layer down.
#
# Re-exported, never reimplemented: these ARE the objects the solvers call,
# and `test_the_public_names_are_the_private_objects` pins that they stay the
# same objects rather than becoming a fourth spelling. The private names keep
# working, so nothing inside this tree or in a consumer moves.
from ._ground_spec import ground_touch_tol
from ._medium_spec import grounded_crossing_exemption

# `accelerated` is True iff the optional C++ accelerator loaded; consumers can
# assert it to guard against a silent fall-back to the slow pure-Python path.
__all__ = [
    "SinusoidalSolver",
    "SinusoidalGalerkinSolver",
    "BSplineSolver",
    "HMatrixSolver",
    "ArrayBlockSolver",
    "RazorSolver",
    "PulseSolver",
    "HarringtonSolver",
    "LatticeFFTUnavailable",
    "PortSolution",
    "Capabilities",
    "CancelToken",
    "SolveAborted",
    "accelerated",
    "wire_internal_impedance",
    "insulation_inductance",
    "wire_to_element",
    "ground_touch_tol",
    "grounded_crossing_exemption",
]
