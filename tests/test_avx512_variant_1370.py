"""The `_avx512` variant on a CPU that runs it (momwire#1370).

Everything here SKIPS, with the reason named, on a CPU without AVX-512
F/CD/BW/DQ/VL (or under an OS that does not save their state): the fleet's
Skylake, every AMD before Zen 4, most GitHub runners. The dispatch half that
runs everywhere is test_accel_avx512_dispatch_1370.py.

Where the CPU has them, this checks that:

  * the AVX-512 build is what `import momwire` chose, and every extension
    (`_accelerators` and both near-interface companions) is its `_avx512`
    build -- a stray `_avx2` companion is the mixed-variant state
    `import_companion` exists to prevent;
  * the lane layer is compiled into it (the 4-wide AVX2 backend, so the lane
    gates below compare lanes against the walk, not the walk against itself);
  * a small solve set agrees with the `_avx2` build, each variant run in its
    own interpreter (two builds of one extension never share a process);
    to `Z_RTOL`, below.

The lane gates themselves are the existing modules, run in the same session
so they execute under this variant. On the AVX-512 machine, from a checkout
of this repository with momwire installed (a wheel or `pip install .`):

    pytest -p no:cacheprovider -n0 -s -v \\
        tests/test_avx512_variant_1370.py \\
        tests/test_accel_avx512_dispatch_1370.py \\
        tests/test_windowed_lanes_1290.py tests/test_weighted_windowed_lanes.py \\
        tests/test_below_lanes_1290.py tests/test_field_galerkin_lanes_1290.py \\
        tests/test_pair_extents_lanes_1290.py tests/test_sheet_fixed_width_1290.py \\
        tests/test_offedge_lanes_1290.py tests/test_offedge_ek_lanes_1362.py \\
        tests/test_swept_ek_lanes_ladder_1362.py \\
        tests/test_ek_pair_order_ladder_1362.py tests/test_hmatrix_ek_block_1362.py \\
        tests/test_somm_proj_lanes_1290.py tests/test_sg_cplx_far_fill_simd_1224.py \\
        tests/test_sg_mixed_class_fill_1224.py

(the module list is pyproject's cp312-win_amd64 lane-gate set plus the two SG
modules whose vector-stage predicates name `_avx512` since #1370). On Linux
those gates are BIT equality, walk against lanes, inside one build -- so they
hold under `_avx512` exactly when the wider autovectorised loops and libmvec's
8-lane routines leave each lane's arithmetic alone. Under Intel SDE's
Skylake-X emulation (2026-10-07, 3d110fe0) they did NOT all hold: the
off-edge and EK lane gates and the SG class fill's column independence
differ in the last bits (<= 1.9e-16 relative), because which entries of a
libmvec loop fall to the scalar tail depends on the loop's length modulo the
vector width, and at width 8 the walk's and the lanes' tails no longer
coincide. Expect those failures here until that is resolved (momwire#1370).

`-s` prints the measured |dZ|/|Z| per deck; MOMWIRE_AVX512_LOG=<path> also
writes them as JSON, which is how Z_RTOL is to be re-derived.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from momwire import _accel

_MISSING = _accel._avx512_missing()
_FORCED = os.environ.get(_accel._FORCE_VARIANT_ENV)


def _skip_reason() -> str | None:
    if _MISSING is None:
        return (
            "AVX-512 cannot be checked on this platform/CPU (non-x86, macOS, or "
            "no /proc/cpuinfo flags line / cpuid probe), so _avx512 is never "
            "chosen here"
        )
    if _MISSING:
        return f"this CPU lacks {', '.join(_MISSING)} (momwire#1370 needs all of them)"
    if not _accel._extension_built("_avx2"):
        return "a single-variant install (no _avx2/_avx512 builds on this platform)"
    if _FORCED and _FORCED != "avx512":
        return f"{_accel._FORCE_VARIANT_ENV}={_FORCED!r} pins another variant"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")

# |Z_avx512 - Z_avx2| / |Z_avx2| per deck, from the variants' precedent until
# an AVX-512 measurement replaces it. The precedent is this very solve set run
# under `_sse2` against `_avx2` (Skylake, GCC 13.3, 2026-10-07):
#
#   free_dipole 3.5e-15   pec_dipole 5.4e-14   sommerfeld_dipole 1.1e-14
#   buried_hub  7.4e-14   sg_hub     4.1e-13   razor_invl        3.7e-15
#
# `_avx512` differs from `_avx2` in the same two ways `_sse2` does: libm
# routines that agree to a few ulp (libmvec's 8-lane forms against its 4-lane
# ones, as against scalar libm), and loops vectorised at another width that
# perform the same IEEE operations. So the same spread is expected, and the
# gate is the worst of it (4.1e-13) with ~250x headroom: 1e-10, a decade
# under the 1e-9 floor test_windows_lanes_1371 uses for sse2-vs-avx2.
# Re-derive on the AVX-512 machine: MOMWIRE_AVX512_LOG=<path> records each
# deck's ratio.
Z_RTOL = 1e-10

DECKS = (
    "free_dipole",
    "pec_dipole",
    "sommerfeld_dipole",
    "buried_hub",
    "sg_hub",
    "razor_invl",
)

# Run under MOMWIRE_FORCE_VARIANT; prints the variant, the module names and
# each deck's Z as JSON. Deck builders are the suite's own (tests/ on
# PYTHONPATH), so this needs no source tree beyond tests/.
_PROBE = r"""
import json, re, sys, warnings
import numpy as np
import momwire._accel as _accel
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver
from test_crossing_serve_524 import hub_deck, invl_deck

C0 = 299792458.0
WL = C0 / 7.1e6
SOIL = dict(ground_eps=(13.0, 0.005), ground_model="sommerfeld")

def dipole(z0, n=61, **kw):
    half = 0.48 * WL / 2
    return dict(
        wires=[np.array([(-half, 0.0, z0), (half, 0.0, z0)])],
        n_per_edge_per_wire=[[n]],
        feeds=[(0, half, 1 + 0j)],
        wavelength=WL,
        wire_radius=1e-3,
        **kw,
    )

DECKS = {
    "free_dipole": lambda: BSplineSolver(**dipole(0.0)),
    "pec_dipole": lambda: BSplineSolver(**dipole(10.0, ground_z=0.0)),
    "sommerfeld_dipole": lambda: BSplineSolver(**dipole(10.0, n=31, ground_z=0.0, **SOIL)),
    "buried_hub": lambda: BSplineSolver(**hub_deck(n_radials=4)),
    "sg_hub": lambda: SinusoidalGalerkinSolver(**hub_deck(n_radials=4)),
    "razor_invl": lambda: RazorSolver(**invl_deck(n_radials=4), nec5_quadrature=True),
}

z = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for name in sys.argv[1:]:
        out = DECKS[name]().compute_impedance()
        zz = complex(np.asarray(out[0] if isinstance(out, tuple) else out).ravel()[0])
        z[name] = [zz.real, zz.imag]
print(json.dumps({
    "variant": _accel.VARIANT,
    "modules": sorted(m for m in sys.modules if re.fullmatch(
        r"momwire\._(accelerators|near_interface_accel|near_interface_point_accel)_\w+", m)),
    "z": z,
}))
"""


def _run(variant: str) -> dict:
    env = dict(os.environ, **{_accel._FORCE_VARIANT_ENV: variant})
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(Path(__file__).parent), env.get("PYTHONPATH", "")) if p
    )
    out = subprocess.run(
        [sys.executable, "-c", _PROBE, *DECKS],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def both():
    return _run("avx512"), _run("avx2")


def test_the_avx512_variant_is_the_one_loaded():
    assert _accel.LOADED
    assert _accel.VARIANT == "avx512", (_accel.VARIANT, _MISSING)


def test_every_extension_is_the_avx512_build():
    from momwire import _near_interface as ni

    assert _accel.acc.__name__ == "momwire._accelerators_avx512"
    assert ni._nia is not None and ni._nia.__name__.endswith("_accel_avx512"), ni._nia
    assert ni._nipa is not None and ni._nipa.__name__.endswith("_accel_avx512"), (
        ni._nipa
    )
    others = sorted(
        m
        for m in sys.modules
        if m.startswith("momwire._") and ("_avx2" in m or "_sse2" in m)
    )
    assert others == [], f"another variant is loaded beside _avx512: {others}"


def test_every_lane_path_is_compiled_in():
    """The 4-wide AVX2 lane backend, in the AVX-512 build."""
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


def test_the_subprocesses_ran_the_variants_they_were_told(both):
    avx512, avx2 = both
    assert avx512["variant"] == "avx512", avx512
    assert avx2["variant"] == "avx2", avx2
    assert all(m.endswith("_avx512") for m in avx512["modules"]), avx512["modules"]
    assert all(m.endswith("_avx2") for m in avx2["modules"]), avx2["modules"]


@pytest.mark.parametrize("deck", DECKS)
def test_the_solve_set_matches_the_avx2_build(both, deck):
    avx512, avx2 = both
    got = complex(*avx512["z"][deck])
    want = complex(*avx2["z"][deck])
    rel = abs(got - want) / abs(want)
    print(f"\n[1370] {deck:18s} avx512 {got:.12g}  avx2 {want:.12g}  rel {rel:.3e}")
    log = os.environ.get("MOMWIRE_AVX512_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"deck": deck, "rel": rel, "bitwise": got == want}) + "\n"
            )
    assert rel <= Z_RTOL, (deck, got, want, rel)
