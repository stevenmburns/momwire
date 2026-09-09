"""momwire#980 step C, part 1: the below-interface plumbing is a module of
functions over data, and both trunks call it.

`_below_interface` holds what `BSplineSolver` grew for the #553 buried serve
and `RazorSolver` copied for #813: the crossing-junction scope and its #698
exemption audit, the node-mesh advisory's arms, the field-form quadrature
nodes, the grid extents and every named refusal. The solvers keep thin
wrappers that hand in their own state. The bit gate is the existing buried,
crossing, transmitted and razor suites; what this file adds is the SHAPE:

    G-980C-1  the module names no solver attribute (the #801 gate, again).
    G-980C-2  the wrappers are wrappers: bspline's and razor's
              `_crossing_junctions` both route through the shared function,
              and the shared function refuses on the same decks by the same
              sentences the wrappers used to.
    G-980C-3  `bspline` still exports every historical name (`_BURIED_*`,
              `_N_QP_BURIED_FIELD`) as the module's own object — `_couplings`
              and the eznec seam import them there.
    G-980C-4  `_pair_extents_below` stays bspline's and is looked up at call
              time — its own tests monkeypatch it through `bspline`, and the
              plan must see the patch.
"""

import ast
import inspect
import sys
from pathlib import Path

import pytest

from momwire import BSplineSolver, RazorSolver
from momwire import _below_interface as BI
from momwire import bspline as _bs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_crossing_serve_524 import crossing_deck  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:crossing node")


def test_g980c_1_the_module_names_no_solver_attribute():
    tree = ast.parse(inspect.getsource(BI))
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for name in (
        "wires_polylines",
        "junctions",
        "ground_z",
        "ground_eps",
        "_radius_per_wire",
        "_wire_media",
        "_find_junctions",
        "_stand_off_floor",
        "_build_geometry",
        "n_qp_sommerfeld",
        "degree",
    ):
        assert name not in attrs, name
    imported = {
        alias.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module in (None, "bspline", "razor")
        for alias in n.names
    }
    assert not ({"bspline", "razor", "BSplineSolver", "RazorSolver"} & imported)


def test_g980c_2_both_trunks_route_through_the_shared_scope_check(monkeypatch):
    seen = []
    real = BI.crossing_junctions

    def spy(media, groups, grounded, polylines, ground_z, radii):
        seen.append((tuple(media), len(list(groups)), set(grounded)))
        return real(media, groups, grounded, polylines, ground_z, radii)

    monkeypatch.setattr(BI, "crossing_junctions", spy)
    build = crossing_deck(1)
    bs = BSplineSolver(**build)
    got_bs = bs._crossing_junctions()
    assert seen and got_bs == (0,), (seen, got_bs)
    # razor detects its groups rather than reading `junctions=`, and asks the
    # question at CONSTRUCTION (`_refuse_buried_geometry`), where a crossing
    # deck is refused by name — so the shared function is reached on the way
    # to that refusal, with the same media labels and the same grounded set.
    with pytest.raises(ValueError, match="razor does not serve buried decks"):
        RazorSolver(**{k: v for k, v in build.items() if k != "degree"})
    assert len(seen) == 2, seen
    assert seen[0][0] == seen[1][0], "media labels differ between trunks"
    assert seen[0][2] == seen[1][2], "grounded sets differ between trunks"


def test_g980c_2b_the_scope_refusals_are_the_shared_functions(monkeypatch):
    """Two above members at the node: refused by name through the module."""
    s = BSplineSolver(**crossing_deck(1))
    media = s._wire_media()
    groups = [list(g) for g in s.junctions]
    grounded = s._grounded_junctions()
    above = [w for w, lab in enumerate(media) if lab == "above"]
    assert len(above) == 1, media
    # a second above member at the crossing node: n_above == 2, refused
    two_above = [groups[0] + [(above[0], "start")]] + groups[1:]
    with pytest.raises(NotImplementedError, match="more than one above member"):
        BI.crossing_junctions(
            media,
            two_above,
            grounded,
            s.wires_polylines,
            s.ground_z,
            s._radius_per_wire,
        )
    with pytest.raises(NotImplementedError, match="per-wire radii"):
        BI.crossing_junctions(
            media, groups, grounded, s.wires_polylines, s.ground_z, [0.001, 0.002]
        )


def test_g980c_3_bspline_re_exports_every_historical_name():
    for name in (
        "ENRICHMENT",
        "EXTENDED_KERNEL",
        "DENSE_BUDGET",
        "PAST_CAP",
        "GRAZING",
        "CROSS_RANGE",
        "DEPTH",
        "CROSS_GRAZING",
    ):
        assert getattr(_bs, f"_BURIED_{name}_REFUSAL") is getattr(
            BI, f"BURIED_{name}_REFUSAL"
        ), name
    assert _bs._N_QP_BURIED_FIELD is BI.N_QP_BURIED_FIELD == 6
    from momwire import _couplings

    assert (
        _couplings._BURIED_EXTENDED_KERNEL_REFUSAL is BI.BURIED_EXTENDED_KERNEL_REFUSAL
    )


def test_g980c_4_the_plan_reads_pair_extents_through_bspline(monkeypatch):
    seen = {}
    real = _bs._pair_extents_below

    def capture(x, y, d_b, **kw):
        seen["n"] = x.shape[0]
        return real(x, y, d_b, **kw)

    monkeypatch.setattr(_bs, "_pair_extents_below", capture)
    s = BSplineSolver(**crossing_deck(1))
    s.compute_impedance()
    assert seen.get("n", 0) > 0, "the buried plan did not call bspline's extents"
    assert not hasattr(BI, "_pair_extents_below")
