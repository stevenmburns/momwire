# Thread-pool wait policy, set before anything below loads NumPy/SciPy or the
# accelerator's OpenMP runtime. Each native pool reads these ONCE, when its
# library loads: OpenBLAS (every bundled copy) OPENBLAS_THREAD_TIMEOUT,
# libgomp OMP_WAIT_POLICY and GOMP_SPINCOUNT. At their defaults the idle
# workers busy-spin after every factorization and steal cores from the next
# fill: +29 % Sommerfeld, +60 % refl-coef, +77 % free space per repeated
# solve (0.63.0, Skylake, 4 threads, paired runs, 2026-09-24; antennaknobs'
# scratch/openblas-spin harness, its #1050). Here, first, so every process
# that imports momwire before NumPy gets them -- the SimNEC portal daemon,
# the EZNEC drop-in and plain scripts. setdefault keeps a caller's own
# values; a process that loaded NumPy first is unaffected (antennaknobs,
# which does, sets the same three at the top of its own package).
import os as _os

for _k, _v in (
    ("OMP_WAIT_POLICY", "PASSIVE"),
    ("GOMP_SPINCOUNT", "0"),
    ("OPENBLAS_THREAD_TIMEOUT", "1"),
):
    _os.environ.setdefault(_k, _v)
del _os, _k, _v

from ._accel import LOADED as accelerated
from ._accel import VARIANT as accelerator_variant
from ._cancel import CancelToken, SolveAborted
from ._capabilities import Capabilities

# The multi-port solve result (#232): every solver family's
# `compute_port_solution()` returns one of these — Y plus the per-port
# solution columns that one fill + one factorisation already produced.
from ._port_solution import PortSolution
from ._feed_snap import FeedPlacement
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
from ._wire_loading import (
    DistributedRLC,
    equivalent_radius,
    insulation_inductance,
    wire_internal_impedance,
)

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
from .bspline import below_reach_refusal
from ._medium_spec import grounded_crossing_exemption

# The capability axes (#884) and the coated-wire pair (#876), on the same
# precedent again. Both were reached through privately by antennaknobs, and in
# both cases A VERSION CHECK CANNOT REPLACE THE REACH: momwire's submodule
# pointer runs ahead of its PyPI release by convention, so a build WITH these
# names and a build WITHOUT them declare the same version. A consumer asking
# "does this build know X" has to ask the build, and promoting the names is
# what lets it ask without importing a private module.
#
# `axes_for` is the SINGLE derivation point for the derived axes
# (`ground_model` from `grounds`, `wire_position` from `buried`/`contact`);
# `AXIS_VALUES` is the vocabulary a consumer rendering those axes needs, and
# `DERIVED_AXES` says which are computed rather than declared. Promoting
# `axes_for` alone would just delay the next census entry.
#
# `SURFACE_HEIGHT_CLASS` is exported as the OBJECT rather than behind a
# `models_coated_wire()` predicate: identity is the point here (a predicate
# would be a new thing to keep true), the consumer probes for
# `equivalent_radius` and this separately so it can refuse by naming which
# half is absent, and the tuple carries the numbers (`floor_h_over_a`,
# `advisory_h_over_a`) that a consumer's own advisory quotes.
from ._capabilities import AXIS_VALUES, DERIVED_AXES, axes_for
from ._surface_height import SURFACE_HEIGHT_CLASS

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
    "FeedPlacement",
    "Capabilities",
    "CancelToken",
    "SolveAborted",
    "accelerated",
    "accelerator_variant",
    "wire_internal_impedance",
    "DistributedRLC",
    "insulation_inductance",
    "wire_to_element",
    "below_reach_refusal",
    "ground_touch_tol",
    "grounded_crossing_exemption",
    "axes_for",
    "AXIS_VALUES",
    "DERIVED_AXES",
    "equivalent_radius",
    "SURFACE_HEIGHT_CLASS",
]
