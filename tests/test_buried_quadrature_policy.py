"""The buried pair quadrature policy record (`momwire._buried_quadrature`).

bspline and SG run the shared tiered pair moments differently on a buried
deck. This file holds what keeps that record honest:

- the record is what each solver reads (bspline's module names, SG's near
  rules in `_sinusoidal_mp`, SG's two routes through the production NEC-5
  render, each seen to reach the fill);
- bspline's C++ copies of its entry are its entry, read out of the source;
- the claim SG's entry rests on: its base order and far ladder serve NONE of
  its near pairs. Every pair under the near ratio goes through
  `near_pair_moments`, and raising the crossing route's base order leaves Z
  bit for bit where it was, while the graded rule (the touching pair at the
  crossing node) moves it. So a near-contact error on SG lives in the near
  rules, and the record says so.

And SG's graded rule per pair: a touching pair whose wedge is acute takes
the raised order, the rest the base order. On slant_base_source (the deck
that found the error: 0.425 ohm at 8 nodes a panel against a raised
reference) the acute pairs are the node segments against their own images,
and the gate here holds the shipped answer within 1e-2 ohm of the reference
(3.5e-3 measured, 11-17); a shallow bend stays on the base order bit for bit.
"""

import dataclasses
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from momwire import _buried_quadrature as BQ
from momwire import _sinusoidal_mp as SMP
from momwire import bspline
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "momwire"
FIX = ROOT / "tests" / "fixtures"

# A crossing deck (the below class of `_assemble_Z_mp_mixed`): the deck where
# bs2 at base 32 was 1.18 ohm off (momwire#1752).
SLANT = FIX / "razor_printout_crossing" / "slant_base_source.nec"
# A wholly-buried deck (`_assemble_Z_mp` at k_m).
BENT_PAIR = FIX / "bspline_printout_buried" / "ab_bent_pair.nec"


# --- the record -------------------------------------------------------------


def test_bspline_reads_its_entry():
    b = BQ.BSPLINE
    assert bspline.BURIED_N_QP_PAIR == b.base_order == 128
    assert bspline.BURIED_PAIR_ORDER_LADDER == b.ladder
    # bspline has one rule for every buried deck: its wholly-buried fields
    # are not a second knob, so they may not say something else
    assert (b.wholly_buried_order, b.wholly_buried_ladder) == (b.base_order, b.ladder)
    assert b.near is None
    assert BQ.POLICIES["bspline"] is b


def test_sg_reads_its_entry_and_the_divergence_is_declared():
    sg = BQ.SINUSOIDAL_GALERKIN
    assert BQ.POLICIES["sinusoidal-galerkin"] is sg
    assert (sg.base_order, sg.ladder) == (32, ((2.0, 8), (16.0, 4)))
    # the wholly-buried route is bspline's FREE-SPACE rule, by value
    assert (sg.wholly_buried_order, sg.wholly_buried_ladder) == (
        bspline.DEFAULT_N_QP_PAIR,
        bspline.DEFAULT_PAIR_ORDER_LADDER,
    )
    near = sg.near
    assert (
        (
            SMP.NEAR_RATIO,
            SMP.GRADED_N_PER_PANEL,
            SMP.NEAR_N_QP,
            SMP.PARALLEL_N_T,
            SMP.PARALLEL_N_XI,
        )
        == (
            near.near_ratio,
            near.graded_n_per_panel,
            near.near_n_qp,
            near.parallel_n_t,
            near.parallel_n_xi,
        )
        == (2.0, 8, 32, 24, 8)
    )
    assert (SMP.GRADED_ACUTE_WEDGE_DEG, SMP.GRADED_ACUTE_N_PER_PANEL) == (
        near.graded_acute_wedge_deg,
        near.graded_acute_n_per_panel,
    )
    assert (near.graded_acute_wedge_deg, near.graded_acute_n_per_panel) == (15.0, 24)
    assert (sg.base_order, sg.ladder) != (BQ.BSPLINE.base_order, BQ.BSPLINE.ladder)
    assert "momwire#1752" in BQ.BSPLINE.reason
    assert "fx_slant_base_source" in BQ.BSPLINE.reason
    assert "0.425 ohm" in sg.reason and "fx_slant_base_source" in sg.reason
    assert "15 degrees" in sg.reason


def test_the_crossing_base_order_is_under_the_near_ratio():
    # The record's own arithmetic for the claim tested on a deck below: the
    # base order serves the pairs under the ladder's first rung, and the
    # near rules overwrite every pair under the near ratio. On the crossing
    # route the first is inside the second, so the base order serves nothing;
    # on the wholly-buried route it serves [near_ratio, 16).
    sg = BQ.SINUSOIDAL_GALERKIN
    assert sg.ladder[0][0] <= sg.near.near_ratio
    assert sg.wholly_buried_ladder[0][0] > sg.near.near_ratio


def test_a_malformed_entry_is_refused():
    with pytest.raises(ValueError, match="strictly descend"):
        dataclasses.replace(BQ.SINUSOIDAL_GALERKIN, ladder=((2.0, 32),))
    with pytest.raises(ValueError, match="strictly ascend"):
        dataclasses.replace(
            BQ.SINUSOIDAL_GALERKIN, wholly_buried_ladder=((16.0, 4), (2.0, 2))
        )


# --- the C++ copies ---------------------------------------------------------

_BSPLINE_CXX = re.compile(
    r"// `BURIED_N_QP_PAIR` / `BURIED_PAIR_ORDER_LADDER` \(momwire#1752 S2\)\n"
    r"\s*fp\.n_qp_pair = (\d+);\n"
    r"\s*fp\.ladder = \{((?:\{[^{}]*\}(?:, )?)*)\};"
)


def _pairs(text):
    return tuple(
        (float(r), int(n)) for r, n in re.findall(r"\{([0-9.]+), (\d+)\}", text)
    )


def _cxx_sources():
    return sorted([*SRC.glob("*.cpp"), *SRC.glob("*.h")])


def test_bsplines_cxx_copies_are_its_entry():
    found = {}
    for path in _cxx_sources():
        for m in _BSPLINE_CXX.finditer(path.read_text(encoding="utf-8")):
            found.setdefault(path.name, []).append((int(m[1]), _pairs(m[2])))
    # the two known copies at least; a new copy that keeps the marker
    # comment is pinned here without an edit
    assert {"_nec5_bspline_printout_core.cpp", "_accel_razor_solve.cpp"} <= set(found)
    want = (BQ.BSPLINE.base_order, BQ.BSPLINE.ladder)
    assert all(v == want for vs in found.values() for v in vs), found
    # and no copy has dropped the marker: every buried `fp.n_qp_pair` literal
    # in the sources is one the regex above read
    bare = sum(
        len(re.findall(r"fp\.n_qp_pair = 128;", p.read_text(encoding="utf-8")))
        for p in _cxx_sources()
    )
    assert bare == sum(len(vs) for vs in found.values())


def test_sgs_cxx_copy_if_any_is_its_entry():
    # SG's C++ fill core (`_sinusoidal_galerkin_fill_core.cpp`) carried its
    # own `BURIED_N_QP_PAIR` and `buried_ladder()`; momwire's SG C++ removal
    # takes it out. While a copy exists, it is pinned.
    hits = []
    for path in _cxx_sources():
        src = path.read_text(encoding="utf-8")
        n = re.search(r"constexpr int64_t BURIED_N_QP_PAIR = (\d+);", src)
        lad = re.search(r"buried_ladder\(\) \{[^\n]*\n[^\n]*\bl\{(.*?)\};", src)
        if n or lad:
            hits.append((path.name, n, lad))
    if not hits:
        pytest.skip("no SG C++ copy of the buried constants in this tree")
    sg = BQ.SINUSOIDAL_GALERKIN
    for name, n, lad in hits:
        assert n and lad, name
        assert (int(n[1]), _pairs(lad[1])) == (sg.base_order, sg.ladder), name


# --- SG's near pairs, on the production route -------------------------------


def _sg_render(
    monkeypatch, deck, policy=None, *, graded=None, watch_near=False, text=None
):
    """Y of `deck` (or of the deck `text`) through the NEC-5 render SG serves
    (`momwire-nec5-sinusoidal-galerkin`), the fill routes that ran, the
    WindowFill (n_qp, ladder) pairs built, and with `watch_near` every
    near_pair_moments call checked against the near ratio. `graded` sets ONE
    graded order for every touching pair, acute or not."""
    from momwire.dropin.nec5 import _shell

    if policy is not None:
        monkeypatch.setitem(BQ.POLICIES, "sinusoidal-galerkin", policy)
    if graded is not None:
        monkeypatch.setattr(SMP, "GRADED_N_PER_PANEL", graded)
        monkeypatch.setattr(SMP, "GRADED_ACUTE_N_PER_PANEL", graded)
    seen = {"y": [], "routes": set(), "fills": [], "near": 0, "touch": 0, "min": 9.0}
    cls = SinusoidalGalerkinSolver
    orig_cps = cls.compute_port_solution

    def cps(self, *a, **k):
        sol = orig_cps(self, *a, **k)
        seen["y"].append(np.atleast_2d(np.asarray(sol.y)).copy())
        return sol

    monkeypatch.setattr(cls, "compute_port_solution", cps)
    for name in ("_assemble_Z_mp_mixed", "_assemble_Z_mp"):
        orig = getattr(cls, name)

        def wrap(self, *a, _o=orig, _n=name, **k):
            at_km = _n == "_assemble_Z_mp" and self._active_medium is not None
            seen["routes"].add(_n + ("@k_m" if at_km else ""))
            return _o(self, *a, **k)

        monkeypatch.setattr(cls, name, wrap)
    orig_init = SMP.WindowFill.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        seen["fills"].append((self.n_qp, self.ladder))

    monkeypatch.setattr(SMP.WindowFill, "__init__", init)
    if watch_near:
        orig_near = SMP.near_pair_moments

        def near(c_i, t_i, h_i, c_j, t_j, h_j, *a, **k):
            ii, jj, J = orig_near(c_i, t_i, h_i, c_j, t_j, h_j, *a, **k)
            ci, cj = np.asarray(c_i, float), np.asarray(c_j, float)
            hi, hj = np.asarray(h_i, float), np.asarray(h_j, float)
            ratio = np.linalg.norm(ci[:, None] - cj[None, :], axis=-1) / np.maximum(
                hi[:, None], hj[None, :]
            )
            want = set(zip(*np.nonzero(ratio < SMP.NEAR_RATIO), strict=True))
            got = set(zip(ii.tolist(), jj.tolist(), strict=True))
            # every pair under the near ratio is overwritten here, and
            # nothing else is
            assert got == want
            seen["near"] += len(got)
            pairs = SMP.classify_pairs(ci, t_i, hi, cj, t_j, hj)
            touch = pairs[3]
            seen["touch"] += int(np.count_nonzero(touch))
            if touch.any():
                r_touch = ratio[pairs[0][touch], pairs[1][touch]]
                seen["min"] = min(seen["min"], float(r_touch.min()))
            return ii, jj, J

        monkeypatch.setattr(SMP, "near_pair_moments", near)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if text is None:
            text = deck.read_text(encoding="latin-1")
        out = _shell.render(text, basis="sinusoidal-galerkin")
    monkeypatch.undo()
    assert seen["y"], out[-2000:]
    return seen


def test_every_near_pair_on_the_crossing_route_is_near_pair_moments(monkeypatch):
    seen = _sg_render(monkeypatch, SLANT, watch_near=True)
    assert "_assemble_Z_mp_mixed" in seen["routes"], seen["routes"]
    sg = BQ.SINUSOIDAL_GALERKIN
    assert (sg.base_order, sg.ladder) in seen["fills"], seen["fills"]
    assert seen["near"] > 0 and seen["touch"] > 0, seen
    # the near-contact pair #1752 was about, a node segment against its own
    # image touching at the crossing node (ratio 0.02), is a touching pair
    # among them: the graded rule's, not the base order's
    assert seen["min"] < 0.1, seen["min"]


def test_the_crossing_routes_base_order_moves_nothing(monkeypatch):
    base = _sg_render(monkeypatch, SLANT)
    raised = dataclasses.replace(BQ.SINUSOIDAL_GALERKIN, base_order=128)
    up = _sg_render(monkeypatch, SLANT, raised)
    # the raised order reached the fill ...
    assert (128, raised.ladder) in up["fills"], up["fills"]
    # ... and served no pair: bit for bit the same answer
    assert np.array_equal(base["y"][0].view(np.uint64), up["y"][0].view(np.uint64))
    # while the graded rule on the touching pairs is live, and is where a
    # near-contact error on SG sits (0.425 ohm at 8 nodes a panel, 11-17)
    graded = _sg_render(monkeypatch, SLANT, graded=8)
    z = complex(np.linalg.inv(base["y"][0])[0, 0])
    z8 = complex(np.linalg.inv(graded["y"][0])[0, 0])
    assert abs(z - z8) > 1e-1, (z, z8)


def _z_in(seen):
    return complex(np.linalg.inv(seen["y"][0])[0, 0])


def test_slants_near_contact_error_is_inside_1e_2_ohm(monkeypatch):
    # The reference raises every near rule the module reads at call time:
    # graded 48 a panel on every touching pair and the near product rule at
    # 96 (the far ladder and the parallel reduction moved this deck by under
    # 1e-12 ohm, 11-17). Measured 11-17: |Z - ref| = 0.425 ohm at graded 8,
    # 3.5e-3 at 24 on every pair, and the same 3.5e-3 with 24 on the acute
    # pairs only (the shipped rule), so the 1e-2 gate fails at the old value.
    # (`_sg_render` undoes every patch when it returns, this one included)
    monkeypatch.setattr(SMP, "NEAR_N_QP", 96)
    ref = _z_in(_sg_render(monkeypatch, SLANT, graded=48))
    shipped = _z_in(_sg_render(monkeypatch, SLANT))
    assert abs(shipped - ref) < 1e-2, (shipped, ref)
    # the gate can fail: the old graded order is well outside it
    old = _z_in(_sg_render(monkeypatch, SLANT, graded=8))
    assert abs(old - ref) > 1e-1, (old, ref)


def test_the_touching_wedge():
    o = np.zeros((1, 3))
    x, y = np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]])
    fold = np.array([[np.cos(np.radians(5)), np.sin(np.radians(5)), 0]])
    end0, end1 = np.array([0]), np.array([1])
    # i = o->x shares its start with j = o->y, o->fold, and with x->2x
    assert SMP.touching_wedge_deg(o, x, o, y, end0, end0)[0] == pytest.approx(90)
    assert SMP.touching_wedge_deg(o, x, o, fold, end0, end0)[0] == pytest.approx(5)
    # straight on: j = -x->o ends where i starts
    assert SMP.touching_wedge_deg(o, x, -x, o, end0, end1)[0] == pytest.approx(180)


def _bent_deck(wedge_deg):
    """A centre-fed wire with a second wire leaving its far end at
    `wedge_deg` to it (180 = straight on, small = folded back). Synthetic."""
    a = np.radians(180.0 - wedge_deg)
    L = 5.28
    p = (L * (1 + 0.7 * np.cos(a)), 0.7 * L * np.sin(a), 0.0)
    return (
        "CM synthetic bend\nCE\n"
        f"GW 1 21 0 0 0 {L:.6f} 0 0 0.001\n"
        f"GW 2 15 {L:.6f} 0 0 {p[0]:.6f} {p[1]:.6f} 0 0.001\n"
        "GE 0 0\nEX 0 1 11 0 1 0\nFR 0 1 0 0 14.2 0\nXQ 0\nEN\n"
    )


@pytest.mark.parametrize("wedge", [5.0, 165.0])
def test_the_graded_order_follows_the_wedge(monkeypatch, wedge):
    # an acute wedge takes the raised order, a shallow bend the base order,
    # each bit for bit what that order everywhere gives
    text = _bent_deck(wedge)
    near = BQ.SINUSOIDAL_GALERKIN.near
    want = (
        near.graded_acute_n_per_panel
        if wedge < near.graded_acute_wedge_deg
        else near.graded_n_per_panel
    )
    other = near.graded_n_per_panel + near.graded_acute_n_per_panel - want
    shipped = _sg_render(monkeypatch, None, text=text)["y"][0]
    same = _sg_render(monkeypatch, None, graded=want, text=text)["y"][0]
    diff = _sg_render(monkeypatch, None, graded=other, text=text)["y"][0]
    assert np.array_equal(shipped.view(np.uint64), same.view(np.uint64))
    assert not np.array_equal(shipped, diff)


def test_the_wholly_buried_route_reads_its_entry(monkeypatch):
    base = _sg_render(monkeypatch, BENT_PAIR)
    assert "_assemble_Z_mp@k_m" in base["routes"], base["routes"]
    sg = BQ.SINUSOIDAL_GALERKIN
    assert (sg.wholly_buried_order, sg.wholly_buried_ladder) in base["fills"]
    # here the base order DOES serve pairs, those from the near ratio to the
    # ladder's 16, so starving it moves Z (PD2: 7e-6 ohm)
    starved = dataclasses.replace(sg, wholly_buried_order=4, wholly_buried_ladder=())
    low = _sg_render(monkeypatch, BENT_PAIR, starved)
    assert (4, ()) in low["fills"], low["fills"]
    z = complex(np.linalg.inv(base["y"][0])[0, 0])
    z4 = complex(np.linalg.inv(low["y"][0])[0, 0])
    assert z != z4
