"""momwire#1168 U5 — the bit-identical tidy of the buried crossing fill.

What the tidy renamed or moved, pinned where a rename could silently disarm a
switch or a shared helper could silently stop being shared.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from momwire import _crossing_fill as CF
from momwire.bspline import BSplineSolver
from test_crossing_serve_524 import crossing_deck


def _flag_in_fresh_process(env_value):
    env = {k: v for k, v in os.environ.items() if k != "MOMWIRE_CROSSING_FORCE_DENSE"}
    if env_value is not None:
        env["MOMWIRE_CROSSING_FORCE_DENSE"] = env_value
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "from momwire import _crossing_fill as CF;"
            "print(CF._WHOLE_AXIS_NO_ACA, hasattr(CF, '_FORCE_DENSE'))",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.split()


@pytest.mark.parametrize("value, expect", [(None, "False"), ("1", "True")])
def test_the_legacy_env_var_still_sets_the_whole_axis_switch(value, expect):
    """The switch was renamed (`_FORCE_DENSE` -> `_WHOLE_AXIS_NO_ACA`); its
    environment variable kept the legacy name, so a bisect note that sets
    `MOMWIRE_CROSSING_FORCE_DENSE=1` still flips it, and the old attribute
    is gone rather than left as a dead twin."""
    assert _flag_in_fresh_process(value) == [expect, "False"]


def _crossing_axes():
    s = BSplineSolver(**crossing_deck(1))
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    ctx = s._crossing_context(geom, supp_seg, polys)
    so = np.asarray(geom["seg_offsets"])
    b_idx, a_idx = np.arange(so[0], so[1]), np.arange(so[1], so[2])
    return ctx, a_idx, b_idx, CF.axis_data(ctx, a_idx), CF.axis_data(ctx, b_idx)


def test_the_whole_axis_switch_routes_every_split_entry(monkeypatch):
    """With the switch on, the three split entries answer through the
    whole-axis fill and never partition: `_main_split` is not entered."""
    ctx, a_idx, b_idx, A, B = _crossing_axes()
    split_calls = []
    real_split = CF._main_split

    def spy(*a, **k):
        split_calls.append(1)
        return real_split(*a, **k)

    monkeypatch.setattr(CF, "_main_split", spy)
    monkeypatch.setattr(CF, "_WHOLE_AXIS_NO_ACA", True)
    whole = CF.cross_complete_block(ctx, A, B)
    assert np.array_equal(CF.cross_complete_block_split(ctx, a_idx, b_idx, A, B), whole)
    assert np.array_equal(
        CF.cross_complete_block_reversed_split(ctx, b_idx, a_idx, B, A),
        CF.cross_complete_block_reversed(ctx, B, A),
    )
    assert split_calls == []
    monkeypatch.setattr(CF, "_WHOLE_AXIS_NO_ACA", False)
    CF.cross_complete_block_split(ctx, a_idx, b_idx, A, B)
    assert split_calls == [1]
