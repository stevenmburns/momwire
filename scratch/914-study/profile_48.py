"""momwire#914: per-term WARM profile at 48 radials, as it stands today.

Replaces the profile in #914's issue body, which predates units 1-2 and #915.
Reports self time per term and labels each as C++ or numpy, so "what is left"
is also "what could still move".
"""

import cProfile
import io
import pstats
import sys

sys.path.insert(0, "/home/smburns/stevenmburns/antennaknobs/scripts")

import bench_converge as bnc  # noqa: E402

from antennaknobs.engines.momwire import MomwireEngine  # noqa: E402
from momwire import BSplineSolver  # noqa: E402

# Terms whose work happens inside a pybind11 extension. cProfile attributes a
# C++ call to its Python caller, so these are labelled rather than inferred.
CPP = (
    "pair_extents_below",
    "assemble_field_galerkin",
    "remainder_field_proj_batch_below",
    "seg_seg_full_moments_bspline",
    "seg_seg_static_moments_bspline",
    "assemble_Z_bspline",
    "sommerfeld_remainder_bspline_Q",
    "_acc.",
)


def main():
    cls = bnc.load_design("verticals.buried_radial_vertical")
    b = cls()
    b.n_radials = 48
    eng = MomwireEngine(
        b,
        solver=BSplineSolver,
        solver_kwargs={"degree": 2, "swept_mem_mb": 8192},
        ground=("finite", 13.0, 0.005),
    )
    eng.impedance()  # cold, discarded
    eng._solved_cache = None

    pr = cProfile.Profile()
    pr.enable()
    z = eng.impedance()[0]
    pr.disable()

    buf = io.StringIO()
    st = pstats.Stats(pr, stream=buf).sort_stats("tottime")
    st.print_stats(28)
    print(f"Z = {z.real:.6f}{z.imag:+.6f}j\n")
    print(buf.getvalue())


main()
