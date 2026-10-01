"""Bit-identity gate for SG's class-restricted mixed fill — momwire#1224.

`SinusoidalGalerkinSolver._assemble_mixed_contribs` used to fill each
medium's pair class over the WHOLE plane (every test entry against every
source, free block, Sommerfeld image and remainder) and keep only the class's
quadrant. On hub16 x4 60 of 702 segments are above, so the above class threw
away ~99 % of what it filled, and the whole-plane block was the mixed peak's
largest owner. Each class is now filled over its own test entries x its own
source columns (`_class_view`) and scattered into the triple.

The claim is bit-identity, cell for cell, and the gates here are:

  * the class fill equals the whole-plane fill BYTE FOR BYTE — the contribs
    triple and G — on the production seam (`_assemble_Z`), with the
    whole-plane reference taken from the same process by refusing
    `_class_fill_serves` (the branch the numpy fill still takes). Decks: the
    crossing hub and inverted-L (the survey's buried/invl), the NON-crossing
    detached hub (the transmitted path), and detached two-wire decks whose
    segment counts walk the source padding through 0-3 with the real-k class
    both holding and not holding the deck's last segment;
  * the class fill ran: a spy on `_class_view` counts one view per class,
    and none on the reference;
  * the padding rule is load-bearing, at the kernel: the fused REAL far fill
    reproduces the whole-plane columns under `_class_view`'s padding for
    random column subsets, and on an AVX2 libmvec build it does NOT under a
    wrong padding (the red control — which is why the padding exists);
  * the fill's traced peak falls: hub16 x2's measured 3.32 -> 2.50 triples.

The bit-identity was also measured against 96f1ec7 (the commit before this
change) from saved arrays, on hub16 x1/x2/x4, invl x1/x4, the above-ground
array x1/x4 (a non-mixed control), the detached decks and two decks with a
one-segment class: all `np.array_equal`, G, the triple, Z and the currents.
"""

from __future__ import annotations

import sys
import tracemalloc
import warnings

import numpy as np
import pytest
from test_crossing_serve_524 import hub_deck, invl_deck
from test_mixed_fill_fused_1224 import detached_hub_deck

import momwire._accel as _accel
from momwire import sinusoidal_galerkin as sg
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

C0 = 299792458.0
WL7 = C0 / 7e6
SOIL_A = (13.0, 0.005)

pytestmark = pytest.mark.skipif(
    not sg._HAVE_GALERKIN_FAR_FILL or not sg._HAVE_GALERKIN_FAR_FILL_CPLX,
    reason="the class fill is the fused C++ path's; without it the mixed "
    "fill IS the whole-plane fill",
)


def detached(n_above, n_below, above_first):
    """A buried centre-fed wire and a centre-fed wire 1 m up
    (`test_sg_mixed_sign_1159.mixed`'s geometry), with free segment counts
    and wire order, so the class sizes and which class holds the deck's last
    segment can be chosen."""
    above = (np.array([(-2.5, 0.0, 1.0), (2.5, 0.0, 1.0)]), n_above)
    below = (np.array([(-2.5, 0.0, -0.5), (2.5, 0.0, -0.5)]), n_below)
    pair = (above, below) if above_first else (below, above)
    return dict(
        wires=[w for w, _ in pair],
        n_per_edge_per_wire=[[n] for _, n in pair],
        feeds=[(0, 2.5, 1 + 0j), (1, 2.5, 1 + 0j)],
        wavelength=WL7,
        wire_radius=1e-3,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )


def _assemble(d, monkeypatch, whole):
    """G and the mixed triple through the production seam, plus the
    `_class_view` calls as (n_class_segs, pad) pairs."""
    views, triples = [], []
    view = SinusoidalGalerkinSolver._class_view
    fill = SinusoidalGalerkinSolver._assemble_mixed_contribs

    def spy_view(self, ctx, keep, n_segs):
        cv = view(self, ctx, keep, n_segs)
        views.append((int(cv.segs.size), cv.pad))
        return cv

    def spy_fill(self, *a, **k):
        out = fill(self, *a, **k)
        triples.append(tuple(np.array(x, copy=True) for x in out))
        return out

    with monkeypatch.context() as m:
        m.setattr(SinusoidalGalerkinSolver, "_class_view", spy_view)
        m.setattr(SinusoidalGalerkinSolver, "_assemble_mixed_contribs", spy_fill)
        # The whole-triple fill, whose triple is what this file compares: the
        # fused banded fill (momwire#1224) never forms it, and is gated
        # against this one in `test_sg_fused_banded_fill_1224`.
        m.setattr(SinusoidalGalerkinSolver, "_band_fill_serves", lambda self, n: False)
        if whole:
            m.setattr(
                SinusoidalGalerkinSolver,
                "_class_fill_serves",
                lambda self, *a: False,
            )
        s = SinusoidalGalerkinSolver(**d)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            geom = s._build_geometry()
            assert s._is_mixed(geom)
            G = s._assemble_Z(geom, s.k, s.eta)[0]
    assert len(triples) == 1
    return G, triples[0], views


DECKS = {
    "hub16-crossing": lambda: hub_deck(n_radials=16),
    "invl-crossing": lambda: invl_deck(n_radials=16),
    "hub16-detached": lambda: detached_hub_deck(x=1),
    # (n_above, n_below, above_first): the real-k (above) class's pad is
    # (-n_above) % 4 when it does not hold the last segment; the below one's
    # is (N % 4 - n_below) % 4, since it does.
    "detached-9-10-above-first": lambda: detached(9, 10, True),
    "detached-11-12-above-first": lambda: detached(11, 12, True),
    "detached-12-9-above-first": lambda: detached(12, 9, True),
    "detached-10-10-below-first": lambda: detached(10, 10, False),
}


# The hub-sized decks run ~5-15 s a pair under xdist, over the PR lane's
# guardrail, so they carry `slow` (push lane); the detached decks cover every
# padding in the PR lane.
_HUB = ("hub16-crossing", "invl-crossing", "hub16-detached")


@pytest.mark.parametrize(
    "name",
    [pytest.param(n, marks=pytest.mark.slow) if n in _HUB else n for n in DECKS],
)
def test_class_fill_is_bit_equal_to_the_whole_plane_fill(name, monkeypatch):
    d = DECKS[name]()
    G, triple, views = _assemble(d, monkeypatch, whole=False)
    G_ref, triple_ref, views_ref = _assemble(d, monkeypatch, whole=True)

    # Both classes took the class fill, and the reference took neither.
    assert len(views) == 2, views
    assert views_ref == []

    for got, ref in zip(triple, triple_ref):
        assert got.shape == ref.shape
        assert np.array_equal(got, ref), (
            f"{name}: {int((got != ref).sum())} of {got.size} cells differ"
        )
    assert np.array_equal(G, G_ref)


def test_the_decks_walk_every_padding():
    """The detached decks exist to put each pad value 0-3 through the real
    kernel, both with and without the deck's last segment in the class.
    Pinned from `_class_view` itself so a deck edit cannot quietly drop one."""
    seen = set()
    for name in DECKS:
        if not name.startswith("detached-"):
            continue
        s = SinusoidalGalerkinSolver(**DECKS[name]())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            geom = s._build_geometry()
            below = s._below_segments(geom)
            medium = s._fill_medium(geom)
            view = s._stitch_basis_coefs(geom, below, medium.k_p, medium.k_m)
            ctx = s._stitch_test_context(geom, view, below, medium.k_p, medium.k_m)
        n = int(geom["n_segs"])
        keep = ~below  # the real-k class
        seen.add((s._class_view(ctx, keep, n).pad, bool(keep[n - 1])))
    assert {p for p, _ in seen} == {0, 1, 2, 3}, seen
    assert {last for _, last in seen} == {True, False}, seen


def _kernel_columns(s, ctx, geom, k, eta, cols, pad):
    idx = np.concatenate((np.full(pad, cols[0]), cols)).astype(np.int64)
    sub = dict(ctx, hh=np.asarray(ctx["hh"])[idx])
    out = s._far_fill_accel(
        k, sub, geom["seg_centers"][idx], geom["seg_tangents"][idx], eta=eta
    )
    return [o[:, pad:] for o in out]


def test_the_real_kernel_needs_the_padding():
    """`_class_view`'s padding rule at the kernel, on hub16's own geometry
    and the above class's real k: random column subsets, with and without
    the last segment. Under the rule the restricted columns are the
    whole-plane ones to the bit. The red control — some other padding moves
    bits — is asserted only on the build that has libmvec's 4-lane sincos
    (linux, AVX2), since elsewhere the sweep has no vector/tail split to
    expose."""
    s = SinusoidalGalerkinSolver(**hub_deck(n_radials=16))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geom = s._build_geometry()
        below = s._below_segments(geom)
        medium = s._fill_medium(geom)
        view = s._stitch_basis_coefs(geom, below, medium.k_p, medium.k_m)
        view = s._crossing_wing_view(geom, view, below, medium)
        ctx = s._test_context(geom, view, medium.k_p)
    k, eta = medium.k_p, s._fill_eta(medium.k_p, None)
    n = int(geom["n_segs"])
    whole = s._far_fill_accel(
        k, ctx, geom["seg_centers"], geom["seg_tangents"], eta=eta
    )
    rng = np.random.default_rng(1224)
    red = 0
    for trial in range(8):
        cols = np.sort(rng.choice(n, int(rng.integers(3, n)), replace=False))
        if trial % 2:
            cols = np.unique(np.append(cols, n - 1))
        keep = np.zeros(n, dtype=bool)
        keep[cols] = True
        cv = s._class_view(ctx, keep, n)
        assert np.array_equal(cv.segs, cols)
        got = _kernel_columns(s, ctx, geom, k, eta, cols, cv.pad)
        for w, g in zip(whole, got):
            assert np.array_equal(w[:, cols], g)
        wrong = _kernel_columns(s, ctx, geom, k, eta, cols, (cv.pad + 1) % 4)
        red += any(not np.array_equal(w[:, cols], g) for w, g in zip(whole, wrong))
    if sys.platform == "linux" and _accel.VARIANT == "avx2":
        assert red == 8, f"a wrong padding moved bits in only {red} of 8 subsets"


@pytest.mark.slow
def test_the_mixed_fill_peak_falls(monkeypatch):
    """hub16 x2 (N = 352): the fill's traced peak, in units of one (nnz, N)
    triple T. Measured 3.32 T whole-plane (the whole-plane class block and
    its image beside the triple) and 2.50 T class-restricted; x4 goes 3.12 ->
    1.92. The N-independent Sommerfeld grids dominate at x1, so x2 is the
    smallest rung where the bound means something."""
    d = hub_deck(n_radials=16)
    npe = d["n_per_edge_per_wire"]
    d["n_per_edge_per_wire"] = [[n * 2 for n in e] for e in npe[:16]] + [
        npe[16],
        [npe[17][0] * 2],
    ]
    fill = SinusoidalGalerkinSolver._assemble_mixed_contribs
    seen = {}

    def traced(self, geom, ctx, *a, **k):
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        try:
            out = fill(self, geom, ctx, *a, **k)
            seen["peak"] = tracemalloc.get_traced_memory()[1] - base
        finally:
            tracemalloc.stop()
        nnz = np.asarray(ctx["w_entry"]).shape[0]
        seen["T"] = 3 * 16 * nnz * int(geom["n_segs"])
        return out

    monkeypatch.setattr(SinusoidalGalerkinSolver, "_assemble_mixed_contribs", traced)
    # The whole-triple fill this bound is about (the fused banded fill holds
    # no triple at all; `test_sg_fused_banded_fill_1224` bounds that one).
    monkeypatch.setattr(
        SinusoidalGalerkinSolver, "_band_fill_serves", lambda self, n: False
    )
    s = SinusoidalGalerkinSolver(**d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geom = s._build_geometry()
        s._assemble_Z(geom, s.k, s.eta)
    ratio = seen["peak"] / seen["T"]
    assert ratio < 2.8, f"mixed fill traced peak {ratio:.2f} T (whole-plane: 3.32 T)"
