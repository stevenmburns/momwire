"""The extended kernel above ground on a deck with buried wires (momwire#1170
stage 0).

Before this, any wire below the ground plane refused `extended_kernel=True`,
and a NEC-4 or NEC-5 deck, whose dialect solves extended by default, fell
back to the reduced kernel on EVERY wire because of one ground stake. Stage 0
serves the combination in part on the B-spline family:

* a pair whose two segments are both above ground takes the extended kernel,
  direct and image, exactly as an all-above deck's pairs do;
* every pair that touches a buried segment keeps the reduced kernel (the
  later stages of #1170 are the extended kernel in and across the soil).

The mask is on the grouping the kernels already take: a buried segment's
eligibility label is the reserved never-extend marker, and a block with no
eligible pair is handed no spec at all, so it is the call it always was.

The gates count what RAN. A spy on the moment kernels records, per call,
the wavenumber, the pairs handed in and the pairs the spec extends; nothing
here reads a flag and believes it.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from momwire import BSplineSolver, HMatrixSolver, RazorSolver, bspline
from momwire import _bspline_kernels as _k
from momwire._below_interface import (
    BURIED_EXTENDED_KERNEL_ABOVE_ONLY,
    ExtendedKernelAboveGround,
)
from momwire.deck import ExtendedKernelDefault, build_solver, parse
from momwire.deck import _solver as _deck_solver
from test_buried_serve_553 import SOIL_A, WL7, _mono, _radial
from test_crossing_serve_524 import crossing_deck

GROUND = dict(ground_z=0.0, ground_eps=SOIL_A, ground_model="sommerfeld")
N_A, N_B = 15, 10
# Fat enough that the extended kernel moves the feed by a quarter of an ohm
# (0.26 at this radius; 0.009 at the buried suite's 1 mm), so a gate that
# compares the two kernels compares two different numbers.
A_FAT = 0.02


def deck(ek, *, radial=True, shift=0.0, a=A_FAT, a_radial=None, **kw) -> BSplineSolver:
    """An elevated 10 m monopole, and (optionally) a detached 5 m radial
    15 cm down, moved `shift` metres along x."""
    wires, npe = [_mono(11.0, 1.0)], [[N_A]]
    if radial:
        wires.append(_radial(depth=0.15) + np.array([shift, 0.0, 0.0]))
        npe.append([N_B])
    radius = [a, a_radial] if (radial and a_radial is not None) else a
    return BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        feeds=[(0, 4.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=radius,
        extended_kernel=ek,
        **GROUND,
        **kw,
    )


def _z(solver) -> complex:
    # Every advisory silenced, not only this unit's: an uncaught warning of a
    # momwire category makes the xdist controller import momwire to rebuild
    # it, and two workers doing that at once is a race this file stays out of.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return complex(solver.compute_impedance()[0])


def _operator(solver, **kw):
    geom = solver._build_geometry()
    supp, polys = solver._build_basis_polynomials(geom)[:2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExtendedKernelAboveGround)
        return solver._compute_Z_operator(geom, supp, polys, **kw), geom


class Census:
    """What the moment kernels were handed, per call."""

    def __init__(self, monkeypatch):
        self.offedge = []  # (medium, n_pairs, n_extended, had_ek_kwarg)
        self.same_edge = []  # (kernel, n_segs, had_ek_kwarg, ek_is_set)
        real_off = bspline._seg_seg_full_moments_offedge
        real_st = bspline._seg_seg_static_moments
        real_reg = bspline._seg_seg_reg_moments

        def off(sl_i, sr_i, sl_j, sr_j, a, k, *args, **kwargs):
            n_i, n_j = np.asarray(sl_i).shape[0], np.asarray(sl_j).shape[0]
            ek = kwargs.get("ek")
            ext = 0 if ek is None else int(_k._ek_pair_mask(ek, n_i, n_j).sum())
            medium = "soil" if complex(k).imag != 0.0 else "air"
            self.offedge.append((medium, n_i * n_j, ext, "ek" in kwargs))
            return real_off(sl_i, sr_i, sl_j, sr_j, a, k, *args, **kwargs)

        def same(name, real):
            def spy(arc, *args, **kwargs):
                self.same_edge.append(
                    (name, len(arc) - 1, "ek" in kwargs, kwargs.get("ek") is not None)
                )
                return real(arc, *args, **kwargs)

            return spy

        monkeypatch.setattr(bspline, "_seg_seg_full_moments_offedge", off)
        monkeypatch.setattr(bspline, "_seg_seg_static_moments", same("static", real_st))
        monkeypatch.setattr(bspline, "_seg_seg_reg_moments", same("reg", real_reg))

    def pairs(self, medium):
        """(pairs handed to the kernel, pairs it extended) in one medium."""
        rows = [r for r in self.offedge if r[0] == medium]
        return sum(r[1] for r in rows), sum(r[2] for r in rows)


def _dense_route(monkeypatch):
    monkeypatch.setattr(
        BSplineSolver, "_buried_chunked_serves", property(lambda self: False)
    )


# ---------------------------------------------------------------------------
# gate 1: a deck with no buried wire never reaches the mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ek", [False, True])
def test_a_deck_with_no_buried_wire_never_asks_the_mask(monkeypatch, ek):
    """The whole of the new code is behind the buried fill, which an
    all-above deck does not enter: nothing it runs changed, so its bits did
    not (the base-versus-change comparison is in the PR)."""

    def boom(*a, **k):  # pragma: no cover - the point is that it never runs
        raise AssertionError("an all-above deck reached the buried mask")

    for name in ("_ek_spec_above", "_ek_subset", "_compute_Z_operator_buried"):
        monkeypatch.setattr(BSplineSolver, name, boom)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExtendedKernelAboveGround)
        s = deck(ek, radial=False)
        s.compute_impedance()
    assert s.extended_kernel_wires() == (ek,)


# ---------------------------------------------------------------------------
# gate 2: extended_kernel=False hands no kernel a spec, buried or not
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ["chunked", "dense"])
def test_the_reduced_kernel_solve_makes_the_calls_it_always_made(monkeypatch, route):
    """Not `ek=None` — NO `ek` keyword, on every moment call of the buried
    fill, so an `extended_kernel=False` solve is the same call sequence it
    was before the mask existed."""
    if route == "dense":
        _dense_route(monkeypatch)
    census = Census(monkeypatch)
    s = deck(False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExtendedKernelAboveGround)
        s.compute_impedance()
    assert census.offedge and census.same_edge
    assert not any(row[3] for row in census.offedge)
    assert not any(row[2] for row in census.same_edge)
    assert s.extended_kernel_wires() == (False, False)


# ---------------------------------------------------------------------------
# gate 3: which pairs took which kernel, counted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ["chunked", "dense"])
def test_the_pairs_are_counted_per_kernel(monkeypatch, route):
    """15 above segments and 10 buried ones. The moment kernels see the two
    same-medium classes only (the 300 cross-medium pairs are a field-form
    block with no thin-wire kernel in it):

    * air: every pair extended, and none left reduced;
    * soil: every pair reduced, with no spec handed over at all.
    """
    if route == "dense":
        _dense_route(monkeypatch)
    census = Census(monkeypatch)
    s = deck(True)
    _z(s)

    air_pairs, air_ext = census.pairs("air")
    soil_pairs, soil_ext = census.pairs("soil")
    # Direct and image, each over the class's own segments. The chunked
    # route also refills each wire's same-edge block to subtract it.
    passes = 3 if route == "chunked" else 2
    assert (air_pairs, air_ext) == (passes * N_A * N_A, passes * N_A * N_A)
    assert (soil_pairs, soil_ext) == (passes * N_B * N_B, 0)
    assert all(row[3] for row in census.offedge if row[0] == "air")
    assert not any(row[3] for row in census.offedge if row[0] == "soil")
    # The same-edge blocks: the above wire's extended, the buried wire's not.
    assert sorted(census.same_edge) == sorted(
        [
            ("static", N_A, True, True),
            ("reg", N_A, True, True),
            ("static", N_B, False, False),
            ("reg", N_B, False, False),
        ]
    )
    assert s.extended_kernel_wires() == (True, False)


def test_the_extended_solve_is_a_different_number_and_both_routes_agree(monkeypatch):
    """The new branch moves the feed (it would be a dead branch if not), and
    the dense route, the reference the chunked one is gated on, agrees."""
    z_red, z_ext = _z(deck(False)), _z(deck(True))
    assert abs(z_ext - z_red) > 0.2, (z_red, z_ext)
    _dense_route(monkeypatch)
    assert abs(_z(deck(True)) - z_ext) <= 1e-11 * abs(z_ext)


def test_the_operator_stays_symmetric_and_the_buried_block_is_untouched():
    """The extension is symmetric in the pair, so Galerkin symmetry survives
    as an error detector; and every entry with a buried basis on either side
    is the reduced solve's, bit for bit, because no pair that touches a
    buried segment changed kernel."""
    s = deck(True)
    (z_red, geom), (z_ext, _) = _operator(deck(False)), _operator(s)
    assert np.abs(z_ext - z_ext.T).max() <= 1e-13 * np.abs(z_ext).max()
    supp, polys = s._build_basis_polynomials(geom)[:2]
    below = np.asarray(s._below_segments(geom), dtype=bool)
    live = np.any(polys != 0.0, axis=2)
    buried = np.any(below[supp] & live, axis=1)
    assert buried.any() and not buried.all()
    assert np.array_equal(z_ext[buried, :], z_red[buried, :])
    assert np.array_equal(z_ext[:, buried], z_red[:, buried])
    above = ~buried
    assert (z_ext[np.ix_(above, above)] != z_red[np.ix_(above, above)]).all()


def test_a_two_radius_deck_and_a_row_restricted_fill_take_the_same_mask():
    """A thinner buried wire makes the spec carry a per-source tube radius,
    cut to each block with its labels; and a fill restricted to the mast's
    rows (momwire#1029) is the full fill's rows."""
    s = deck(True, a_radial=0.005)
    full, geom = _operator(s)
    below = s._below_segments(geom)
    rows = np.nonzero(~below)[0]
    part, _ = _operator(deck(True, a_radial=0.005), rows=rows)
    keep = np.abs(part).max(axis=1) > 0
    assert keep.any() and not keep.all()
    assert np.array_equal(part[keep], full[keep])
    assert abs(_z(s) - _z(deck(False, a_radial=0.005))) > 0.2


# ---------------------------------------------------------------------------
# gate 4: a buried wire moved far away gives back the above-only answer
# ---------------------------------------------------------------------------


def test_a_far_buried_wire_reproduces_the_above_ground_extended_answer():
    """Detach the radial and move it 60 m off (1.4 wavelengths; the
    transmitted family is tabulated to 2). What is left of it at the feed is
    its coupling, which the REDUCED pair of solves measures: c = |Z_mixed -
    Z_above|, r = c / |Z|.

    * the extended answers differ by that same coupling, no more;
    * the kernel's own effect, Z_ext - Z_red, is the above-only deck's to
      first order in r: a relative r of it is the coupling acting on the
      shift, and the bound is three times that. Measured 5.7e-7 ohm against
      a derived 1.7e-6 on a 0.26 ohm shift.
    """
    above_red, above_ext = _z(deck(False, radial=False)), _z(deck(True, radial=False))
    far_red, far_ext = _z(deck(False, shift=60.0)), _z(deck(True, shift=60.0))
    coupling = abs(far_red - above_red)
    r = coupling / abs(above_red)
    assert 0.0 < r < 1e-5, r
    assert abs(far_ext - above_ext) <= 1.01 * coupling
    shift_above, shift_far = above_ext - above_red, far_ext - far_red
    assert abs(shift_above) > 0.2
    assert abs(shift_far - shift_above) <= 3.0 * r * abs(shift_above)


# ---------------------------------------------------------------------------
# the user-facing shape
# ---------------------------------------------------------------------------


def test_an_explicit_request_is_served_and_says_which_wires_once():
    s = deck(True)
    with pytest.warns(ExtendedKernelAboveGround) as seen:
        s.compute_impedance()
    (message,) = [str(w.message) for w in seen]
    assert BURIED_EXTENDED_KERNEL_ABOVE_ONLY in message
    assert "1 of 2 wires are below the plane" in message
    assert "(wire index 1)" in message
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExtendedKernelAboveGround)
        s.compute_impedance()  # once per solver


def test_a_crossing_deck_changes_kernel_at_the_node():
    """A wire continued below the plane through a junction in it: the above
    member extends, the buried one does not, and the solve is served."""
    s = BSplineSolver(**crossing_deck(wire_radius=A_FAT), extended_kernel=True)
    assert s.extended_kernel_wires() == (False, True)
    z_ext = _z(s)
    z_red = _z(BSplineSolver(**crossing_deck(wire_radius=A_FAT)))
    assert 0.02 < abs(z_ext - z_red) < 0.5, (z_red, z_ext)


def test_the_declarations_say_what_each_family_does():
    key = "buried+extended_kernel"
    assert BSplineSolver.capabilities.refusal("buried", "extended_kernel") is None
    assert BSplineSolver.capabilities.partials[key] is BURIED_EXTENDED_KERNEL_ABOVE_ONLY
    # The accelerated subclasses refuse a buried deck whole, so the partial
    # is not theirs; razor still refuses the combination.
    assert not HMatrixSolver.capabilities.partials
    assert not RazorSolver.capabilities.partials
    assert RazorSolver.capabilities.refusal("buried", "extended_kernel")


# ---------------------------------------------------------------------------
# the deck front end: the default no longer costs the whole deck its kernel
# ---------------------------------------------------------------------------

STAKE = (
    "CM a vertical on a ground stake\nCE\n"
    "GW 1 12 0 0 0 0 0 10 0.02\n"
    "GW 2 4 0 0 0 0 0 -1 0.02\n"
    "GE -1\nGN 3 0 0 0 13 0.005\n"
    "{ek}EX 0 1 3 0 1 0\nFR 0 1 0 0 7.0\nXQ\nEN\n"
)


def test_a_nec4_deck_with_a_stake_keeps_the_extended_kernel_above_ground():
    """The user case (a vertical on a ground stake, read as NEC-4): the
    solve is extended above ground, and the advisory says the buried wire
    kept the reduced kernel, not that the deck lost the extended one."""
    model = parse(STAKE.format(ek=""), dialect="nec4")
    with pytest.warns(ExtendedKernelDefault, match="kept the reduced kernel") as seen:
        built = build_solver(model)
    assert built.extended_kernel and built.solver.extended_kernel
    assert built.solver.extended_kernel_wires() == (True, False)
    (advisory,) = built.advisories
    assert "1 of 2 wires below the plane kept the reduced kernel" in advisory
    assert "above-ground wires solved with the extended kernel" in advisory
    assert "so it solved with the reduced kernel" not in advisory
    assert len(seen) == 1
    # Said once, by the front end: the solver does not repeat it at the fill.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the crossing node's mesh advisory
        warnings.simplefilter("error", ExtendedKernelAboveGround)
        z_ext = complex(built.solver.compute_port_solution().y[0, 0]) ** -1
        reduced = build_solver(model, extended_kernel=False)
        assert reduced.advisories == ()
        z_red = complex(reduced.solver.compute_port_solution().y[0, 0]) ** -1
    assert abs(z_ext - z_red) > 0.02, (z_red, z_ext)


def test_an_ek_card_on_a_nec2_deck_with_a_stake_is_served_in_part():
    """Asked for, not defaulted: served the same way and said under the
    solver's own category, where it used to be a refusal."""
    text = STAKE.format(ek="EK\n").replace("GN 3 ", "GN 2 ")
    model = parse(text, dialect="nec2")
    with pytest.warns(ExtendedKernelAboveGround, match="asks for the extended"):
        built = build_solver(model)
    assert built.solver.extended_kernel and len(built.advisories) == 1


def test_a_family_without_the_partial_still_falls_back_whole(monkeypatch):
    """The mask is the B-spline family's. A family that refuses the
    combination keeps the reduced kernel on every wire, as before."""
    monkeypatch.setattr(_deck_solver, "extended_kernel_partial", lambda *a: None)
    monkeypatch.setattr(
        _deck_solver,
        "extended_kernel_default_refusal",
        lambda cls, *a: RazorSolver.capabilities.refusal("buried", "extended_kernel"),
    )
    model = parse(STAKE.format(ek=""), dialect="nec4")
    with pytest.warns(ExtendedKernelDefault, match="below the ground plane"):
        built = build_solver(model)
    assert not built.solver.extended_kernel
    assert built.solver.extended_kernel_wires() == (False, False)
