"""The lane layer runs on Windows (momwire#1371).

Since #1371 the MSVC `_avx2` variant selects the avx2 backend of `_lanes.h`,
and its lanes are gated against their walks by the derived win32 tolerance
(tests/_lane_gate.py). A green Windows run of those gates means nothing unless
the lanes actually ran, so on win32 this module checks, against the installed
build:

* the AVX2 variant is the one loaded (the wheel smoke runs on CPUs with AVX2;
  a run that fell back to `_sse2` exercised no lane at all), and every lane
  path reports itself compiled in, fused kernels included;
* the remainder projection's lane counter moves on a solve, so the lanes were
  entered, not merely built;
* a fused kernel's lanes and its walk give DIFFERENT bits there. On MSVC the
  lanes' `fmadd` is fused and the walk's `mw_fma::fma` is not, so a difference
  is the signature of the lane branch having executed; equal bits on this
  synthetic case would mean the call never left the walk.

The wheel also carries `_sse2`, the variant a CPU without AVX2 loads (Dan's).
No `/arch:AVX2` means no `__AVX2__`, so `_lanes.h` selects no backend there and
every kernel keeps its walk. The last test forces that variant in a subprocess
and checks it loads, compiled no lane and entered none, and solves the
projection deck to the AVX2 variant's answer.

Off Windows the module skips: Linux and macOS have their own compiled-in
assertions in each lane test module, and bit equality there.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import momwire._accel as _accel
from _lane_gate import bits

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="the MSVC build's lane switch (momwire#1371)"
)


def test_the_avx2_variant_is_loaded():
    assert _accel.LOADED
    assert _accel.VARIANT == "avx2", (
        f"loaded {_accel.VARIANT!r}: the lane gates on this runner would "
        "compare the walk with itself"
    )


def test_every_lane_path_is_compiled_in():
    from momwire import _near_interface as ni

    acc = _accel.acc
    built = {
        "offedge_lanes_1290": acc.offedge_lanes_1290,
        "windowed_lanes_1290": acc.windowed_lanes_1290,
        "below_lanes_1290": acc.below_lanes_1290,
        "somm_proj_lanes_built": acc.somm_proj_lanes_built(),
        "grid_sheet_fixed_1290": ni._nia.grid_sheet_fixed_1290,
    }
    assert all(built.values()), built


def test_the_projection_lanes_are_entered_on_a_solve():
    from test_sg_cplx_far_fill_simd_1224 import _hub4

    from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

    before = _accel.acc.somm_proj_lane_pairs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        SinusoidalGalerkinSolver(**_hub4(), fill="direct").compute_impedance()
    assert _accel.acc.somm_proj_lane_pairs() > before


def test_a_fused_kernel_runs_its_lanes():
    from test_windowed_lanes_1290 import _case, _run

    moved = 0
    for seed in (0, 1):
        for d in (1, 2, 3):
            c = _case(seed, d)
            walk = _run(c, True)
            lanes = _run(c, False)
            moved += int(np.any(bits(lanes) != bits(walk)))
    assert moved > 0, "the windowed lanes gave the walk's bits on every case"


# Run under MOMWIRE_FORCE_VARIANT=sse2; prints what loaded and the deck's Z.
_SSE2_PROBE = """
import json, warnings
import momwire._accel as _accel
from momwire import _near_interface as ni
from test_sg_cplx_far_fill_simd_1224 import _hub4
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver
acc = _accel.acc
before = acc.somm_proj_lane_pairs()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Z, _ = SinusoidalGalerkinSolver(**_hub4(), fill="direct").compute_impedance()
print(json.dumps({
    "variant": _accel.VARIANT,
    "built": [acc.offedge_lanes_1290, acc.windowed_lanes_1290,
              acc.below_lanes_1290, acc.somm_proj_lanes_built(),
              ni._nia.grid_sheet_fixed_1290],
    "lane_pairs": acc.somm_proj_lane_pairs() - before,
    "z": [complex(Z).real, complex(Z).imag],
}))
"""


def test_the_sse2_variant_runs_its_walks():
    from test_sg_cplx_far_fill_simd_1224 import _hub4

    from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

    env = dict(os.environ, MOMWIRE_FORCE_VARIANT="sse2")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).parent), env.get("PYTHONPATH", "")]
    )
    out = subprocess.run(
        [sys.executable, "-c", _SSE2_PROBE],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["variant"] == "sse2", got
    assert not any(got["built"]), got
    assert got["lane_pairs"] == 0, got

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want, _ = SinusoidalGalerkinSolver(**_hub4(), fill="direct").compute_impedance()
        want = complex(want)
    # Two builds of one solver: a check the variant works, not a bit gate.
    # 1e-9 is the derived solver tolerances' floor (see tests/_lane_gate.py).
    assert abs(complex(*got["z"]) - want) <= 1e-9 * abs(want), (got["z"], want)
