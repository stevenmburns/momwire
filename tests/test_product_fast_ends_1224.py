"""The crossing fill's end loops on a product of MANY groups — momwire#1224.

`_fast_end_desc` names the product rows an end asks; a "line" end needs its
key in every group. That used to be one Python dict per group, so the plan
built the structures only up to 64 groups (`_PRODUCT_FAST_MAX_GROUPS`), and a
block past that — razor's inverted-L, whose reversed block groups the top
wire's path axis one node per group (97 groups at x1, 769 at x8) — sent every
end to the generic lookup: the asked points re-deduplicated and searched
against the whole product, 2.7 s of a 4.9 s block at x8 for ~7 k fresh rows.
The key is now found by array search (`KeyIndex`, then a sorted code
g * n_key + k), the same exact-`==` classes the dicts keyed on, and the cap
is lifted.

The gate is the bits: Z of a razor fill through its real constructor with
the cap lifted, against the same fill capped at the old 64 (the generic
lookup path for the many-group block: the code on main) and against every
product switch off, at `np.array_equal`. The counters prove the capped fill
took the lookup path for the many-group block and the uncapped one served
those ends fast. The red control serves every fast end one product row off
and must move Z. A unit row pins the array search to the dict it replaced.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _crossing_fill as cf
from momwire import _near_interface as ni

from test_crossing_serve_524 import invl_deck
from test_product_grid_1173 import _fill
from test_triple_memo_1168 import _razor

pytestmark = pytest.mark.usefixtures("sheet_modes")


def _make(lean=False):
    # Leaning, four radials make the below side the cheaper one to group
    # (41 groups); sixteen keep the above side's 161.
    return lambda: _razor(invl_deck(16 if lean else 4, x=1, lean=lean))


@pytest.mark.slow
@pytest.mark.parametrize("lean", [False, pytest.param(True, marks=pytest.mark.slow)])
def test_many_groups_take_the_fast_ends_to_the_bit(lean):
    """inverted-L x1: the reversed block has 97 groups (161 leaning over
    sixteen radials), past
    the old cap. Capped it is main's lookup path; uncapped, its ends are
    served by the gather. Z does not move. Upright, the many-group block's
    line ends exercise the array key search; leaning, its grouped ends."""
    make = _make(lean)
    ref, r_ref = _fill(make, off=True)
    old, r_old = _fill(make, **{"cf._PRODUCT_FAST_MAX_GROUPS": 64})
    got, r = _fill(make)
    assert np.array_equal(old, ref)
    assert np.array_equal(got, ref)
    assert r_ref["cf.ends_fast"] == 0
    # The many-group block exists and is the one the cap used to stop.
    assert r["cf.main_product"] == 2 and r["cf.main_product_groups"] > 64, r
    # Capped, only the forward block's ends were fast; lifted, the reversed
    # block's are too.
    assert r["cf.ends_fast"] > r_old["cf.ends_fast"], (r, r_old)
    # The product route counts every end it classifies, so capped it counted
    # only the forward block's, and lifted both blocks'.
    n_old = r_old["cf.ends_fast"] + r_old["cf.ends_slow"]
    assert r["cf.ends_fast"] + r["cf.ends_slow"] > n_old, (r, r_old)
    if not lean:
        assert r["cf.ends_fast_line"] > r_old["cf.ends_fast_line"], (r, r_old)


def test_many_groups_red_control_moves_z():
    """Every fast end served one product row off: the gate above can see a
    wrong row on the many-group block."""
    make = _make()
    ref, _r = _fill(make, off=True)
    got, r = _fill(make, **{"cf._PRODUCT_NEG_CONTROL": "row"})
    assert r["cf.main_product_groups"] > 64 and r["cf.ends_fast"] > 0, r
    assert int(np.count_nonzero(got != ref)) > 0, "the gate is blind to a wrong row"


def _dict_line_keys(plan, r, l0):
    """The dict route `_fast_end_desc` took for a line end (momwire#1173):
    group g's local key of (r[g], l0), None when any group lacks it."""
    kmap = [
        {
            (float(plan.line[g, n]), float(plan.fast.line_z[n])): j
            for j, n in enumerate(plan.kfirst[g].tolist())
        }
        for g in range(len(plan.rowtab))
    ]
    kg = []
    for g, rg in enumerate(r.tolist()):
        k = kmap[g].get((rg, l0))
        if k is None:
            return None
        kg.append(k)
    return np.array(kg)


def test_the_array_key_search_is_the_dict_it_replaced():
    """Three above groups over a below line whose z holds both signed zeros:
    every line end the dicts found, `_fast_end_desc` finds at the same local
    keys -- by the end's own line node or by the array search -- and an end
    whose ρ misses one group is refused by both."""

    class _Ctx:
        a_wire = 0.001

    xy_a = [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 2.0)]
    z_a = [0.5, 1.0, 0.5, 2.0, 1.0]
    xy_b = [(0.3, 0.0), (0.0, 0.4), (0.3, 0.0), (2.0, 2.0), (0.0, 0.4), (0.5, 0.5)]
    z_b = [0.0, -0.0, -0.5, -0.0, -1.0, -0.25]
    A = {"nodes": np.column_stack([np.asarray(xy_a, float), np.asarray(z_a, float)])}
    B = {"nodes": np.column_stack([np.asarray(xy_b, float), np.asarray(z_b, float)])}
    plan = cf._product_plan(_Ctx(), 1.0, 1.0, A, B, 0.0)
    assert plan.slot == "z" and len(plan.rowtab) == 3
    fast = plan.fast
    checked = 0
    on_node0 = cf._ROUTES["ends_line_on_node"]
    for n in range(B["nodes"].shape[0]):
        # A line end standing exactly on line node n: rho to every grouped
        # node, its own z in the line slot.
        x, y = B["nodes"][n, :2]
        rho = np.hypot(A["nodes"][:, 0] - x, A["nodes"][:, 1] - y)
        for l0 in (B["nodes"][n, 2], 0.0, -0.0, 7.0):
            zp = np.full(rho.shape, l0)
            r = ni.radius_fold(rho[fast.gfirst], 0.001)
            want = _dict_line_keys(plan, r, float(l0))
            # At the end's own (x, y), and at one no line node holds (the
            # node shortcut then has no candidate and the search answers).
            for pt in ((x, y, l0), (99.0, 99.0, l0)):
                desc = cf._fast_end_desc(fast, 0.001, pt, rho, A["nodes"][:, 2], zp)
                if want is None:
                    assert desc is None
                    continue
                assert desc[0] == "line" and np.array_equal(desc[1], want)
                checked += 1
    assert checked >= B["nodes"].shape[0]
    # Both ways in: ends standing on a line node (the node's own keys) and
    # ends whose line-slot float only matches a key (the array search).
    on_node = cf._ROUTES["ends_line_on_node"] - on_node0
    assert 0 < on_node < checked, (on_node, checked)
    # One group's rho nudged off its key: the dicts refuse, and so must this.
    x, y = B["nodes"][0, :2]
    rho = np.hypot(A["nodes"][:, 0] - x, A["nodes"][:, 1] - y)
    rho[fast.grank == 2] = np.nextafter(rho[fast.grank == 2], 10.0)
    zp = np.full(rho.shape, B["nodes"][0, 2])
    r = ni.radius_fold(rho[fast.gfirst], 0.001)
    assert _dict_line_keys(plan, r, 0.0) is None
    assert (
        cf._fast_end_desc(fast, 0.001, (x, y, 0.0), rho, A["nodes"][:, 2], zp) is None
    )
