"""momwire#1675: under the exact kernel the gap advisory fires at Δ/a < 1.

A zero-width gap has no limiting reactance under the exact ring kernel:
once the segments at it are shorter than the radius, X slides by tenths of
an ohm per halving instead of settling. bs2's point gap starts sliding at
about Δ/a = 1 (a 38 mm monopole over PEC at 14 MHz: X -0.00 / +0.01 / -0.05
/ -0.25 ohm at Δ/a 2.8 / 1.4 / 0.70 / 0.35), so `GAP_FLOOR_EXACT` is 1, above
the extended kernel's collapse floor of 0.6.
"""

from __future__ import annotations

import warnings

import pytest

from momwire import _wire_spec
from momwire.eznec._shell import render

LAM = 299.8 / 14.0
H, A = 5.05283, 0.0381


def _deck(segs_per_wl: float) -> str:
    n = lambda length: max(1, round(length / LAM * segs_per_wl))  # noqa: E731
    return (
        "CM 1675\nCE\n"
        f"GW 1,{n(0.5)},0.,0.,0.,0.,0.,0.5,{A}\n"
        f"GW 2,{n(H - 0.5)},0.,0.,0.5,0.,0.,{H},{A}\n"
        "GE 1\nFR 0,1,0,0,14.\nGN 1\nEX 4,1,-1,0,1.,0.\nXQ\nEN\n"
    )


def _gap_advisories(text: str, basis: str) -> list[str]:
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        render(text, basis=basis)
    return [str(w.message) for w in seen if w.category is _wire_spec.GapMeshFloor]


def test_the_exact_floor_sits_above_the_extended_one():
    assert _wire_spec.GAP_FLOOR_EXACT == 1.0
    assert _wire_spec.GAP_FLOOR_EXACT > _wire_spec.GAP_FLOOR_EXTENDED


@pytest.mark.parametrize(("segs", "fires"), [(400, False), (800, True)])
def test_bs2_names_the_feed_once_its_segments_pass_the_radius(segs, fires):
    """Δ/a 1.41 is silent; Δ/a 0.70 names the source and the end-port feed."""
    found = _gap_advisories(_deck(segs), "bspline")
    assert bool(found) is fires, found
    if fires:
        assert "exact ring kernel" in found[0]
        assert "end-port" in found[0]


def _fires(exact_kernel):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        _wire_spec.advise_gap_mesh_floor(
            "probe",
            [[(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]],
            [[round(1.0 / (0.7 * A))]],
            [A],
            True,
            [("source 0", [(0, "start")])],
            gap_model="knot",
            exact_kernel=exact_kernel,
        )
    return any(w.category is _wire_spec.GapMeshFloor for w in seen)


def test_the_extended_kernel_keeps_its_collapse_floor():
    """At Δ/a 0.70 the exact kernel's floor fires and the extended
    kernel's (0.6) does not: the floor follows the kernel that ran."""
    assert _fires(True)
    assert not _fires(False)
