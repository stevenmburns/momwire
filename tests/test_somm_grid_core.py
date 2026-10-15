"""The Sommerfeld grid in C++ alone (`_somm_grid_core.cpp`, momwire#1479
Sommerfeld S2) is `_sommerfeld.get_grid`'s, bit for bit.

The driver is built from the two core TUs with no Python include path (the
standalone executable's link), asked for a handful of grids that reach the
fill's branches -- a ladder rung pair (the blend), a value on a rung, a grid
past the near/far split, a lossy and a near-lossless soil -- and every scalar,
lattice and value byte is compared with Python's fill (disk cache off).
Its `far_cpp_pack` (the momwire#1258 continuation's constants) too.
`scripts/core_split/somm_grid_compare.py` runs the same comparison over the
corpus's own calls and random ones.
"""

from __future__ import annotations

import importlib.util
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "scripts" / "core_split"


def _compare_module():
    spec = importlib.util.spec_from_file_location(
        "core_split_somm_grid_compare", SPLIT / "somm_grid_compare.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _call(eps_r, sigma, f_hz, r1_lambda):
    omega = 2.0 * math.pi * f_hz
    eps = complex(eps_r, -sigma / (omega * 8.854187817e-12))
    k2 = omega / 299792458.0
    lam = 2.0 * math.pi / k2
    return (eps.real, eps.imag, k2, r1_lambda * lam, omega, 4e-7 * math.pi)


@pytest.mark.skipif(
    shutil.which(os.environ.get("CXX", "g++")) is None, reason="no C++ compiler"
)
@pytest.mark.skipif(sys.platform != "linux", reason="the driver build is the Linux one")
def test_the_cxx_grid_is_get_grids(tmp_path, monkeypatch):
    monkeypatch.setenv("MOMWIRE_SOMM_CACHE", "0")
    cmp = _compare_module()
    calls = [
        _call(13.0, 0.005, 14.2e6, 1.0),  # average soil: a rung pair, blended
        _call(5.0, 0.001, 7.1e6, 6.0),  # poor soil past the near/far split
        _call(81.0, 4.0, 3.6e6, 0.5),  # sea water: the steep band's keyed rows
        _call(15.0, 0.0, 21.0e6, 0.3),  # lossless: no ladder, keyed exactly
    ]
    exe = cmp.build_driver(tmp_path)
    lines = "".join(" ".join(float(x).hex() for x in c) + "\n" for c in calls)
    run = subprocess.run(
        [str(exe), str(tmp_path)], input=lines, text=True, capture_output=True
    )
    assert run.returncode == 0, run.stderr
    from momwire import _sommerfeld as s

    for k, c in enumerate(calls):
        s._GRID_CACHE.clear()
        s._NORM_CACHE.clear()
        g = s.get_grid(complex(c[0], c[1]), c[2], c[3], c[4], c[5])
        head = [
            float.fromhex(x)
            for x in (tmp_path / f"{k}.txt").read_text().split("\n")[0].split()
        ]
        want = [g.eps_t.real, g.eps_t.imag, g.lattice_eps.real, g.lattice_eps.imag, g.k2,
                g.omega, g.mu, g.r1_max, g.r_break, g.r_near, g._r0_fill]  # fmt: skip
        assert [h.hex() for h in head] == [float(w).hex() for w in want], c
        blob = (tmp_path / f"{k}.bin").read_bytes()
        raw = b"".join(reg["vals"].tobytes() for reg in g._regions)
        assert blob == raw, c
        far = (tmp_path / f"{k}.far").read_bytes()
        assert far == s.far_cpp_pack(g).astype("complex128").tobytes(), c
