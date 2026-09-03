"""Razor's crossing fill — momwire#813 unit 2, the assembly.

Four masked terms indexed by (row HALF) x (column WING), every crossing tent
in all four:

    Z[R_a, C_a] += the above fill at k_p, on the ABOVE geometry
    Z[R_b, C_b] += the below fill at k_m, on the BELOW geometry
    Z[R_a, C_b] -= the trunk's cross block           (corner=False)
    Z[R_b, C_a] -= the trunk's reversed cross block  (momwire#832)

**What the eps~ = 1 collapse below can and cannot see.** At eps~ = 1 the
interface vanishes and the crossing deck IS one straight wire, so razor's own
free-space fill is the truth and the whole matrix is checkable against it.
Three things it is blind to, named here so the bar is not read as more than
it is:

  * **the SW end term.** W vanishes at eps~ = 1, so derivation (b)'s choice
    to keep SW on razor rows cannot be read here at all. Soil (step 4) is
    what reads it.
  * **t_z against F'** on `crossing_deck(1)`: both its wires are vertical, so
    those two are collinear and no disagreement between them could show.
    `fan_rise_deck()` has horizontal below members and separates them, which
    is why both decks are gated and not just the cheap one.
  * **the below/below grazing floor**, also on `crossing_deck(1)`: it is a
    single vertical line, so every below/below pair is coaxial (rho = 0) and
    reads 90 deg. The fan carries that one too — its shallowest graded node
    sits 1.9855e-05 m below the plane and its worst real pair is 1.7199 deg
    against the 1 deg floor, which is 1.72x of margin and thin enough that
    grading the node harder for soil should re-read it.

`_SERVE_CROSSING` is off by default, so every test here turns it on
explicitly; the roster flip is momwire#814's.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

from momwire import _medium_spec as MS
from momwire import razor as _razor
from momwire.razor import RazorSolver

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from test_crossing_serve_524 import crossing_deck, fan_rise_deck  # noqa: E402

# measured 6.83e-13 / 7.26e-11 at (growth 2.0, panel 8, q 12), both lanes
BAR_COLLAPSE_CROSSING = 1e-11
BAR_COLLAPSE_FAN = 1e-9
LANES = {"two-point": {"nec5_quadrature": True}, "gauss-legendre": {}}


@pytest.fixture
def serve_crossing(monkeypatch):
    monkeypatch.setattr(_razor, "_SERVE_CROSSING", True)


def _decks(name):
    deck = crossing_deck(1) if name == "crossing" else fan_rise_deck()
    return {k: v for k, v in deck.items() if k != "junctions"}


def _at_eps1(deck):
    d = dict(deck)
    d["ground_eps"] = (1.0, 0.0)
    return d


def _free_space(deck):
    return {
        k: v
        for k, v in deck.items()
        if k not in ("ground_z", "ground_eps", "ground_model")
    }


def _geom(deck, **kw):
    """Geometry only. The basis gates do not need a fill, and filling for
    them put this module three tests over the time ceiling."""
    rs = RazorSolver(**deck, n_qp_path=8, **kw)
    return rs, rs._build_geometry()


def _fill(deck, **kw):
    rs = RazorSolver(**deck, n_qp_path=8, **kw)
    geom = rs._build_geometry()
    return (
        rs,
        geom,
        rs._assemble_Z_from_prepared(
            geom, rs._assemble_Z_prepare(geom), rs.k, rs.omega
        ),
    )


# ------------------------------------------------------- the basis (gates 1-3)


def test_a_crossing_node_carries_a_through_tent_not_two_contact_tents(serve_crossing):
    """Gate 2. With `ground_z` set razor gives K ends in the plane K contact
    tents; at a DECLARED crossing node it must give the free-space topology
    instead, because the plane there is a boundary between MEDIA and not a
    conductor (momwire#838 measured what the contact spelling does)."""
    rs, geom = _geom(_at_eps1(_decks("crossing")))
    _free_rs, free_geom = _geom(_free_space(_decks("crossing")))
    assert geom["n_basis_total"] == free_geom["n_basis_total"] == 29
    assert np.array_equal(geom["wing_seg"], free_geom["wing_seg"])
    assert np.array_equal(geom["wing_sigma"], free_geom["wing_sigma"])
    # and no plane reference is taken at that node
    assert np.asarray(geom["grounded_bases"]).size == 0


def test_the_fan_classifies_four_through_tents_by_their_wings(serve_crossing):
    """Gate 3. K = 5 gives 4 through tents; which of them CROSSES is decided
    by the deck's wire order, because `_junction_wings` pairs every member
    against `ends[0]`."""
    rs, geom = _geom(_at_eps1(_decks("fan")))
    bo = int(np.asarray(geom["basis_offsets"])[-1])
    assert geom["n_basis_total"] - bo == 4
    tents = rs._crossing_tents(geom)
    assert len(tents) == 1, tents  # the riser is the LAST wire as written
    media = rs._wire_media()
    assert media.count(MS.BELOW) == 4 and media.count(MS.ABOVE) == 1


def test_a_grounded_end_that_is_not_a_crossing_member_is_untouched(monkeypatch):
    """Gate 1. The demotion must reach a declared crossing node and nothing
    else: an ordinary contact deck keeps its contact tents and its grounded
    bases exactly as before."""
    lam = 299792458.0 / 7.0e6
    contact = dict(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, lam / 4]])],
        n_per_edge_per_wire=[[12]],
        wire_radius=0.005,
        wavelength=lam,
        ground_z=0.0,
        feeds=[(0, 0.0, 1 + 0j)],
    )
    _rs, geom, Z = _fill(contact)
    assert np.asarray(geom["grounded_bases"]).size == 1
    monkeypatch.setattr(_razor, "_SERVE_CROSSING", True)
    _rs2, geom2, Z2 = _fill(contact)
    assert np.array_equal(np.asarray(geom["grounded_bases"]), geom2["grounded_bases"])
    assert np.array_equal(Z, Z2)


# ------------------------------------------------- the collapse (gates 4, 5, 8)


@pytest.mark.slow
@pytest.mark.parametrize("lane", sorted(LANES))
@pytest.mark.parametrize(
    "name,bar", [("crossing", BAR_COLLAPSE_CROSSING), ("fan", BAR_COLLAPSE_FAN)]
)
def test_the_whole_matrix_collapses_to_free_space_at_eps_tilde_one(
    serve_crossing, name, bar, lane
):
    """Gates 4 and 5. At eps~ = 1 the assembled crossing matrix IS razor's own
    free-space fill on the same geometry — every block, every entry, not one
    port number. See the module docstring for what this cannot see."""
    deck = _decks(name)
    _rs, _g, Z = _fill(_at_eps1(deck), **LANES[lane])
    _rf, _gf, Zf = _fill(_free_space(deck), **LANES[lane])
    assert Z.shape == Zf.shape
    rel = np.abs(Z - Zf).max() / np.abs(Zf).max()
    assert rel < bar, rel


@pytest.mark.slow
def test_the_collapse_does_not_depend_on_the_wire_order(serve_crossing):
    """Gate 8. Listing the riser FIRST makes every tent a crossing tent — a
    different path through the classification at the same physics. Both
    spellings are legal and the assembly must serve both."""
    # Two radials, not four: the claim is about the CLASSIFICATION flipping,
    # which K = 3 shows as well as K = 5 does, and it reads the same numbers
    # (7.2554e-11 / 7.2634e-11) for a fifth of the time.
    base = fan_rise_deck(n_radials=2)
    n = len(base["wires"])
    seen = {}
    for label, order in (
        ("as written", list(range(n))),
        ("riser first", [n - 1] + list(range(n - 1))),
    ):
        inv = {o: i for i, o in enumerate(order)}
        d = {k: v for k, v in base.items() if k != "junctions"}
        d["wires"] = [base["wires"][i] for i in order]
        d["n_per_edge_per_wire"] = [base["n_per_edge_per_wire"][i] for i in order]
        d["feeds"] = [(inv[w], a, v) for w, a, v in base["feeds"]]
        rs, geom, Z = _fill(_at_eps1(d))
        _rf, _gf, Zf = _fill(_free_space(d))
        seen[label] = (
            len(rs._crossing_tents(geom)),
            np.abs(Z - Zf).max() / np.abs(Zf).max(),
        )
    # the gate is reading two code paths, not one
    assert seen["as written"][0] == 1, seen
    assert seen["riser first"][0] == 2, seen
    for label, (_n, rel) in seen.items():
        assert rel < BAR_COLLAPSE_FAN, (label, rel)


# ------------------------------------------------------- the near-miss (gate 9)


@pytest.mark.slow
def test_the_below_block_comes_from_the_below_fill(serve_crossing, monkeypatch):
    """Gate 9, and it exists because this branch nearly shipped a false pass.

    A probe called `_assemble_Z_from_prepared`, which dispatches on
    `_below_plane` — never set by that probe's constructor stub — so it ran
    razor's ORDINARY fill on the below geometry and reported OK. The rule the
    arc took from it: anything reaching past `_refuse_buried_geometry` must
    call the medium fill BY NAME. This asserts the assembly does.
    """
    calls = {"below": 0, "ordinary": 0}
    real_below = RazorSolver._assemble_Z_below_plane
    real_block = RazorSolver._assemble_Z_source_block

    def spy_below(self, geom, prepared, k, omega, **kw):
        calls["below"] += 1
        return real_below(self, geom, prepared, k, omega, **kw)

    def spy_block(self, geom, prepared, sources, k, omega, **kw):
        # the below fill reaches the source block too; count only the calls
        # made OUTSIDE it, i.e. the above family's
        if calls["in_below"] == 0:
            calls["ordinary"] += 1
        return real_block(self, geom, prepared, sources, k, omega, **kw)

    calls["in_below"] = 0

    def wrapped_below(self, geom, prepared, k, omega, **kw):
        calls["in_below"] += 1
        try:
            return spy_below(self, geom, prepared, k, omega, **kw)
        finally:
            calls["in_below"] -= 1

    monkeypatch.setattr(RazorSolver, "_assemble_Z_below_plane", wrapped_below)
    monkeypatch.setattr(RazorSolver, "_assemble_Z_source_block", spy_block)
    _fill(_at_eps1(_decks("crossing")))
    assert calls["below"] == 1, calls
    # the above family's two source-block calls (direct and image), and no
    # third one standing in for the below fill
    assert calls["ordinary"] == 2, calls


# ----------------------------------------------------------- the refusals


def test_a_crossing_deck_is_refused_by_name_when_the_switch_is_off():
    with pytest.raises(ValueError, match="momwire#813"):
        RazorSolver(**_decks("crossing"), n_qp_path=8)


def test_a_plane_touching_end_that_is_not_a_crossing_member_still_refuses(
    monkeypatch,
):
    """The `plan_skip` key must never widen to "any point at depth 0": a
    wholly-below wire whose end merely touches the plane has no crossing
    block to carry its current."""
    monkeypatch.setattr(_razor, "_SERVE_BELOW_PLANE", True)
    monkeypatch.setattr(_razor, "_SERVE_CROSSING", True)
    lam = 299792458.0 / 7.0e6
    deck = dict(
        wires=[np.array([[0.0, 0.0, -2.0], [0.0, 0.0, 0.0]])],
        n_per_edge_per_wire=[[8]],
        wire_radius=0.001,
        wavelength=lam,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
        feeds=[(0, 1.0, 1 + 0j)],
    )
    with pytest.raises(ValueError):
        RazorSolver(**deck, n_qp_path=8).compute_impedance()


def test_loading_on_a_crossing_deck_is_refused(serve_crossing):
    deck = _at_eps1(_decks("crossing"))
    deck["lumped_loads"] = [(1, 1.0, 50 + 0j)]
    with pytest.raises(NotImplementedError, match="loading on a crossing deck"):
        RazorSolver(**deck, n_qp_path=8).compute_impedance()
