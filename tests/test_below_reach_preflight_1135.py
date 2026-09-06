"""`below_reach_refusal` answers what the fill would, before the fill.

antennaknobs#1135 wants a knob combination to fail at construction rather than
20 s into a solve. The pre-flight is only worth anything if its verdict is the
FILL's verdict, so that is what these gates measure -- both refuse or both
serve, on decks swept across the two bounds, with no false refusals.

The two bounds it reproduces are `_buried_serve_plan`'s own:

  * R1_max <= `_SOMM_BELOW_R1_CAP_LAMBDA_M` in-medium wavelengths;
  * theta_min >= `_SOMM_BELOW_TH_MIN_DEG` (0.05 deg since momwire#935).

One copy of each constant and each sentence: the helper formats the same
templates the fill raises, which is what these gates hold together.
"""

import math
import sys

import numpy as np
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import momwire  # noqa: E402
from momwire import _sommerfeld_below  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from test_crossing_serve_524 import fan_rise_deck  # noqa: E402

FREQ = 7.1e6
SOILS = {"A": (13.0, 0.005), "B": (20.0, 0.03), "C": (5.0, 0.001)}


def _verdict(pts, soil):
    """The helper takes the SAME `(eps_r, sigma)` a solver takes, plus the
    frequency — not a derived medium, so no caller carries a third copy of
    that conversion."""
    return momwire.below_reach_refusal(np.asarray(pts, float), 0.0, SOILS[soil], FREQ)


def _screen(radial, depth, n_radials=4):
    """The connected screen's VERTICES: N radial tips, the hub, the node."""
    pts = [(0.0, 0.0, -depth), (0.0, 0.0, 0.0)]
    for i in range(n_radials):
        th = 2 * math.pi * i / n_radials
        pts.append((radial * math.cos(th), radial * math.sin(th), -depth))
    return pts


def test_g1135_1_nothing_below_is_not_a_refusal():
    assert _verdict([(0.0, 0.0, 1.0), (5.0, 0.0, 2.0)], "A") is None


def test_g1135_2_the_two_bounds_fire_by_name():
    far = _verdict([(30.0, 0.0, -0.15), (-30.0, 0.0, -0.15)], "A")
    assert far is not None and "pair separation" in far
    graze = _verdict([(5.0, 0.0, -5e-4), (-5.0, 0.0, -5e-4)], "A")
    assert graze is not None and "pair elevation" in graze
    # the sentences are the FILL's, so the floor they quote is the live one
    assert f"{_sommerfeld_below._SOMM_BELOW_TH_MIN_DEG:g} deg grazing floor" in graze


def test_g1135_3_the_floor_it_quotes_follows_the_constant(monkeypatch):
    """A second copy of the floor would not move with momwire#935's; this
    one must, so the pre-flight can never advertise a stale domain."""
    monkeypatch.setattr(_sommerfeld_below, "_SOMM_BELOW_TH_MIN_DEG", 0.5)
    msg = _verdict([(5.0, 0.0, -0.02), (-5.0, 0.0, -0.02)], "A")
    assert msg is not None and "0.5 deg grazing floor" in msg


@pytest.mark.slow
def test_g1135_4_the_preflight_agrees_with_the_fill():
    """The gate that matters: sweep the R1 bound and compare verdicts.

    Each deck is built for real and SOLVED far enough to reach
    `_buried_serve_plan`, which is where the fill raises. One test rather
    than a parametrisation because the sweep has to be scored as a WHOLE: a
    pre-flight that refused everything, or served everything, would pass
    every individual row of a one-sided sweep. The mix assertion at the end
    is what makes this a comparison.

    `slow` because it solves: five rows of the parametrised version breached
    the 20 s ceiling, and solving is the point.
    """
    served = refused = 0
    for soil in sorted(SOILS):
        for radial in (3.0, 6.0, 9.0, 12.0, 18.0, 24.0):
            for depth in (0.15, 0.5):
                build = fan_rise_deck(depth=depth)
                build["wires"] = [
                    np.array(
                        [
                            (radial * dx, radial * dy, -depth),
                            (0.0, 0.0, -depth),
                            (0.0, 0.0, 0.0),
                        ]
                    )
                    for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1))
                ] + [build["wires"][-1]]
                build["ground_eps"] = SOILS[soil]
                pre = _verdict(_screen(radial, depth), soil)
                try:
                    BSplineSolver(**build).compute_impedance()
                    fill = None
                except ValueError as e:
                    fill = str(e)
                if fill is not None and "below/below pair" not in fill:
                    continue  # refused for an unrelated reason; not this gate
                assert (pre is None) == (fill is None), (
                    f"{soil}/r={radial}/d={depth}: pre-flight "
                    f"{'REFUSED' if pre else 'served'} but the fill "
                    f"{'REFUSED' if fill else 'served'}\n  pre : "
                    f"{str(pre)[:160]}\n  fill: {str(fill)[:160]}"
                )
                if pre is None:
                    served += 1
                else:
                    refused += 1
                    # and for the SAME reason, not merely at the same time
                    assert pre.split(",")[0][:40] == fill.split(",")[0][:40]
    assert served and refused, (
        f"the sweep is one-sided: {served} served, {refused} refused. "
        "A pre-flight that always says the same thing agrees with the fill "
        "on whichever half it happens to cover, which is not a comparison."
    )


def test_g1135_5_the_shipped_screen_is_not_falsely_refused():
    """Vertices reach further and lie shallower than quadrature nodes, so the
    pre-flight is conservative -- the risk is over-refusal, and the deck the
    catalog actually ships is the one that must not trip it."""
    assert _verdict(_screen(5.0, 0.15), "A") is None


def test_g1135_6_a_deck_the_935_floor_just_made_servable_passes():
    """momwire#935 moved the grazing floor 0.1 deg -> 0.05. A 3 mm-class
    depth on a short radial sits between the two: theta = atan(d/L) with
    d = 3 mm and L = 3.17 m is 0.054 deg -- refused at the old floor, served
    at the new one. The pre-flight must agree with the floor as it IS.
    """
    depth, radial = 0.003, 3.17
    th = math.degrees(math.atan2(depth, radial))
    assert 0.05 < th < 0.1, f"the deck no longer straddles the two floors: {th}"
    assert _verdict(_screen(radial, depth), "A") is None
