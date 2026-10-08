"""Bit-identity gate for SG's fused banded fill — momwire#1224.

`SinusoidalGalerkinSolver._assemble_Z` used to build the whole fill in
element layout — a "triple" of three (nnz, N) complex arrays, nnz ≈ 3N, so
about 9-10 Z (Z = 16·n_basis² bytes) — plus, under a Sommerfeld ground, a
second triple for the image, and only then contract it to G through the
sparse coefficient product (`_scatter_coef_product`). It now assembles G one
OBSERVER BAND at a time (`_assemble_G_banded`, `_assemble_mixed_G_banded`):
a band is a run of test segments and their support entries, filled (free
block, image, remainder, near cells, fold; on a mixed deck each pair class's
quadrant and the transmitted directions) and scattered straight into a
preallocated, F-ordered G (`_scatter_band`), which accumulates each basis's
T row across bands in the whole product's order and writes the basis's row
of G at the band holding its last entry. The whole triple never exists. Two more
(n_basis, n_basis) arrays went with it: the crossing block `t_ab` is asked
for its live rows only (and before G exists), and the extended kernel's
end-bracket `C` is taken by row bands.

The claim is bit-identity, and the gates here are:

  * the banded G equals the whole-triple G BYTE FOR BYTE through the
    production seam (`_assemble_Z_ported`), the reference recomputed
    in-process by refusing `_band_fill_serves` (the branch the dense regime,
    N < 60, still takes). Decks: free space, PEC (the fold), refl-coef (the
    numpy image fill under a band view), Sommerfeld above ground and fully
    buried (compose: image, remainder), a ground contact, the extended
    kernel with a bend (the bracket fires, in row bands), a junction-port
    pair, a non-crossing mixed deck (transmitted directions), and the
    crossing hub and inverted-L (class views, the crossing rows). Each in
    several bands, and the cheap decks in bands of one, two and three
    segments — every boundary, and boundaries that cut a basis's support
    after one entry and after two;
  * the fallbacks band too: a mixed class filled whole-plane, and the numpy
    far fill (no accelerator), on bands aligned to the numpy loop's blocks;
  * the banded fill ran: spies count the bands scattered (several) and the
    whole product (never), and the reference the reverse;
  * the red controls: summing the three shape products in another order, or
    adding a straddling basis's carried row after the band's entries instead
    of before, fails the byte gate (a faithful copy of the scatter passes);
    so does a numpy band NOT aligned to the numpy loop's blocks;
  * the fill's traced peak is G plus band scratch, not the triple.

Measured as well against 648f3c9 (the commit before this change) from saved
arrays on Haswell: G, Z, the currents and `compute_port_solution`'s Y and
coeffs `np.array_equal` on hub16 x1/x2/x4, invl x1/x4, the above-ground
array x1/x4 and the decks below, at the production band budget and at
budgets forcing tens of bands.
"""

from __future__ import annotations

import sys
import tracemalloc
import warnings

import numpy as np
import pytest
from test_crossing_serve_524 import hub_deck, invl_deck
from test_junction_ports import _port_pair_solver
from test_mixed_fill_fused_1224 import detached_hub_deck
from test_sg_mixed_sign_1159 import mixed

from momwire import _schedule
from momwire import sinusoidal_galerkin as sg
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver


def _direct_sg(*args, **kw):
    """This module's subject is the DIRECT-field fill (its near correction,
    test quadrature, folded shapes and fused far fill), so every solver here
    names it; `SinusoidalGalerkinSolver`'s default fill is the mixed-potential
    one since momwire#1354 and is gated in `test_sinusoidal_mp_1354.py`."""
    kw.setdefault("fill", "direct")
    return SinusoidalGalerkinSolver(*args, **kw)


C0 = 299792458.0
WL7 = C0 / 7e6
SOMM = dict(ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld")
REFL = dict(ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="refl-coef")


def dipole(n=81, z=0.0, **kw):
    return dict(
        wires=[np.array([(-10.0, 0.0, z), (10.0, 0.0, z)])],
        n_per_edge_per_wire=[[n]],
        feeds=[(0, 10.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=1e-3,
        **kw,
    )


def ell(n=60, a=0.05, **kw):
    """A bent wire thick enough for the extended kernel's end bracket to
    fire at the bend (`_ek_reduced_ends`)."""
    return dict(
        wires=[
            np.array([(0.0, 0.0, 3.0), (10.0, 0.0, 3.0)]),
            np.array([(10.0, 0.0, 3.0), (10.0, 0.0, 9.0)]),
        ],
        n_per_edge_per_wire=[[n], [n // 2 + 1]],
        junctions=[[(0, "end"), (1, "start")]],
        feeds=[(0, 5.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=a,
        **kw,
    )


def monopole(n=61, **kw):
    return dict(
        wires=[np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 10.0)])],
        n_per_edge_per_wire=[[n]],
        feeds=[(0, 0.4, 1 + 0j)],
        wavelength=WL7,
        wire_radius=1e-3,
        **kw,
    )


DECKS = {
    "free-dipole": lambda: _direct_sg(**dipole()),
    "pec-dipole": lambda: _direct_sg(**dipole(z=3.0, ground_z=0.0)),
    "refl-dipole": lambda: _direct_sg(**dipole(z=3.0, **REFL)),
    "somm-dipole": lambda: _direct_sg(**dipole(z=3.0, **SOMM)),
    "buried-dipole": lambda: _direct_sg(**dipole(z=-0.5, **SOMM)),
    "contact-monopole": lambda: _direct_sg(**monopole(**SOMM)),
    "ek-ell": lambda: _direct_sg(**ell(extended_kernel=True)),
    "junction-ports": lambda: _port_pair_solver(
        0.02, 0.01, 40, volts=(1 + 0j, 0.3 + 0j), cls=SinusoidalGalerkinSolver
    ),
    "mixed-detached": lambda: _direct_sg(**mixed(4)),
    "hub16-crossing": lambda: _direct_sg(**hub_deck(n_radials=16)),
    "invl-crossing": lambda: _direct_sg(**invl_deck(n_radials=16)),
    "hub16-detached": lambda: _direct_sg(**detached_hub_deck(x=1)),
}
# The hub-sized decks build Sommerfeld grids and run ~5-15 s a pair, over the
# PR lane's guardrail, so they carry `slow` (push lane).
_HUB = ("hub16-crossing", "invl-crossing", "hub16-detached")
# Bands of one, two and three test segments: every band boundary there is,
# and boundaries that cut a basis's support after one of its entries and
# after two, on the cheap decks.
_THIN = ("free-dipole", "somm-dipole", "mixed-detached", "ek-ell", "junction-ports")

_COUNTED = (
    "_assemble_G_banded",
    "_assemble_mixed_G_banded",
    "_scatter_band",
    "_scatter_coef_product",
    "_bracket_coef_mats",
)


def force_bands(m, step, *, unaligned=False):
    """`step` test segments a band, asked for through the PRODUCTION step
    rule (`_segment_bands`, which walks `_schedule.chunks`): the band budget
    is zeroed, so the rule's floor `_BAND_MIN_SEGMENTS` is the step. A band
    on the numpy far fill is still rounded up to that loop's blocks, as in
    production, unless `unaligned` drops the rounding (the red control).

    Replacing `_segment_bands` with a test-built generator, as this helper
    once did, would bypass the schedule layer the method now walks, and the
    gates riding on it would stay green whatever the layer did
    (momwire#1337 phase 2). The zero budget also bands the extended
    kernel's end bracket one basis row at a time."""
    m.setattr(sg, "_BAND_MIN_SEGMENTS", step)
    m.setattr(SinusoidalGalerkinSolver, "_band_budget_bytes", lambda self, n: 0)
    if unaligned:
        real = SinusoidalGalerkinSolver._segment_bands

        def bands(self, ctx, n_basis, n_cols, n_triples, align=None):
            return real(self, ctx, n_basis, n_cols, n_triples, align=None)

        m.setattr(SinusoidalGalerkinSolver, "_segment_bands", bands)


class Walked:
    """One `_schedule.chunks` walk as `record_walks` saw it: the function
    that iterated it (`by`), its arguments, the chunks it yielded, and how
    many checkpoints the walk itself called."""

    __slots__ = ("by", "spans", "chunk", "has_checkpoint", "got", "checkpoints")

    def __init__(self, by, spans, chunk, has_checkpoint):
        self.by, self.spans, self.chunk = by, list(spans), chunk
        self.has_checkpoint = has_checkpoint
        self.got, self.checkpoints = [], 0


def record_walks(m):
    """Log every `_schedule.chunks` walk (`Walked`) and reset
    `_schedule.STATS`. The real walk runs underneath, and counts."""
    log = []
    real = _schedule.chunks

    def chunks(spans, chunk, checkpoint=None):
        # The generator's body first runs inside the loop that iterates it,
        # so the caller's frame names the walker.
        w = Walked(
            sys._getframe(1).f_code.co_name, spans, chunk, checkpoint is not None
        )
        log.append(w)

        def tally():
            w.checkpoints += 1
            checkpoint()

        for c in real(spans, chunk, None if checkpoint is None else tally):
            w.got.append(c)
            yield c

    m.setattr(_schedule, "chunks", chunks)
    _schedule.reset_stats()
    return log


def logged_chunks(log):
    return sum(len(w.got) for w in log)


def segment_walks(log):
    """The bands `_segment_bands` walked on the layer, one list per fill."""
    return [w.got for w in log if w.by == "_segment_bands"]


def _assemble(
    name,
    monkeypatch,
    whole,
    budget=256 * 1024,
    step=None,
    scatter=None,
    fallback=None,
    unaligned=False,
):
    """G through the production seam (`_assemble_Z_ported`, which every
    solve calls), plus how often each fill branch ran and the bands the
    fill walked on the schedule layer (`counts["walks"]`, `force_bands`).
    `budget` is the band budget (it also bands the end bracket's C); `step`
    forces a number of test segments a band through the production step
    rule instead (`force_bands`; `unaligned` drops its numpy-block rounding,
    for the red control). `fallback` takes a path the
    shipped build does not: "whole-plane-class" refuses the class-restricted
    fill (a mixed class is then filled against the whole source list and its
    quadrant cut out, per band), "numpy" hides the fused C++ far fill."""
    counts = dict.fromkeys(_COUNTED, 0)
    with monkeypatch.context() as m:
        if fallback == "whole-plane-class":
            m.setattr(
                SinusoidalGalerkinSolver, "_class_fill_serves", lambda self, *a: False
            )
        elif fallback == "numpy":
            m.setattr(sg, "_HAVE_GALERKIN_FAR_FILL", False)
        # The budget decides the bands here, on decks a few hundred segments
        # long, so the production floor of 64 segments a band is lifted.
        m.setattr(sg, "_BAND_MIN_SEGMENTS", 1)
        for meth in _COUNTED:
            real = getattr(SinusoidalGalerkinSolver, meth)

            def spy(self, *a, _real=real, _meth=meth, **kw):
                counts[_meth] += 1
                return _real(self, *a, **kw)

            m.setattr(SinusoidalGalerkinSolver, meth, spy)
        if scatter is not None:
            m.setattr(SinusoidalGalerkinSolver, "_scatter_band", scatter)
        m.setattr(
            SinusoidalGalerkinSolver, "_band_budget_bytes", lambda self, n_basis: budget
        )
        if step is not None:
            force_bands(m, step, unaligned=unaligned)
        log = record_walks(m)
        if whole:
            m.setattr(
                SinusoidalGalerkinSolver, "_band_fill_serves", lambda self, n: False
            )
        s = DECKS[name]()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            geom = s._build_geometry()
            with s._operating_medium(geom) as medium:
                G = s._assemble_Z_ported(geom, s.k, s._medium_eta(medium))[0]
        # Every chunk the layer counted was a walk logged here, so the bands
        # below are what the fill walked through `_schedule`.
        assert _schedule.STATS["chunks"] == logged_chunks(log)
    counts["walks"] = segment_walks(log)
    return np.array(G, copy=True), counts


def _cases():
    for name in DECKS:
        marks = [pytest.mark.slow] if name in _HUB else []
        yield pytest.param(name, None, marks=marks, id=f"{name}-256k")
        if name in _THIN:
            for step in (1, 2, 3):
                yield pytest.param(name, step, id=f"{name}-{step}seg")


@pytest.mark.parametrize(("name", "step"), list(_cases()))
def test_banded_fill_is_bit_equal_to_the_whole_triple_fill(name, step, monkeypatch):
    # The refl-coef image takes the numpy far fill, whose bands align to its
    # own blocks (`_segment_bands`); shrink those so they are several here.
    _numpy_blocks_of(8, name, monkeypatch)
    G, counts = _assemble(name, monkeypatch, whole=False, step=step)
    G_ref, counts_ref = _assemble(name, monkeypatch, whole=True)

    # The banded fill ran, in several bands, and never formed the whole
    # product; the reference is the reverse.
    banded = counts["_assemble_G_banded"] + counts["_assemble_mixed_G_banded"]
    assert banded == 1, counts
    assert counts["_scatter_band"] >= 3, counts
    assert counts["_scatter_coef_product"] == 0, counts
    assert counts_ref["_scatter_band"] == 0 and counts_ref["_scatter_coef_product"] == 1
    # ... and its bands were the schedule layer's walk (momwire#1337): one
    # walk over every test segment, a band per scatter (no band is empty
    # on these decks), `step` segments a band where forced.
    _assert_walked_the_layer(counts, step)
    assert counts_ref["walks"] == []
    if name == "ek-ell":
        # The end bracket fired, and took its row bands.
        assert counts["_bracket_coef_mats"] >= 1, counts

    assert G.shape == G_ref.shape
    assert G.flags.f_contiguous, "the banded G must reach the solve F-ordered"
    assert np.array_equal(G, G_ref), (
        f"{name}: {int((G != G_ref).sum())} of {G.size} cells differ, max rel "
        f"{np.max(np.abs(G - G_ref)) / np.max(np.abs(G_ref)):.2e}"
    )


@pytest.mark.parametrize("fallback", ["whole-plane-class", "numpy"])
@pytest.mark.parametrize("name", ["mixed-detached", "somm-dipole", "ek-ell"])
def test_the_fallback_fills_band_too(name, fallback, monkeypatch):
    """The paths the shipped build does not take on these decks, banded and
    whole alike: a mixed class filled against the whole source list (a band
    view holding every entry, its class's quadrant cut out), and the numpy
    far fill under a band view (no accelerator; also what a weighted image
    projector takes). The numpy loop's block is shrunk to 8 test segments for
    both fills, so the production bands (aligned to it) are several."""
    if fallback == "whole-plane-class" and name != "mixed-detached":
        pytest.skip("only a mixed deck has pair classes")
    _numpy_blocks_of(8, name, monkeypatch)
    G, counts = _assemble(name, monkeypatch, whole=False, fallback=fallback)
    G_ref, _ = _assemble(name, monkeypatch, whole=True, fallback=fallback)
    _assert_walked_the_layer(counts)
    assert np.array_equal(G, G_ref), (
        f"{name}/{fallback}: {int((G != G_ref).sum())} of {G.size} cells differ"
    )


def _assert_walked_the_layer(counts, step=None):
    (walk,) = counts["walks"]
    assert walk[0][0] == 0 and all(a[1] == b[0] for a, b in zip(walk, walk[1:]))
    assert len(walk) == counts["_scatter_band"] >= 3, (walk, counts)
    if step is not None:
        heights = {i1 - i0 for i0, i1 in walk[:-1]}
        assert len(heights) == 1 and heights.pop() % step == 0, walk


def test_numpy_bands_must_align_to_the_numpy_blocks(monkeypatch):
    """The red control for `_segment_bands`' `align`: on the complex-k class
    the numpy loop's values move across numpy's 256 KB temporary-elision
    boundary, so bands of 3 segments — the whole fill takes this deck's 80
    segments in ONE block, above the boundary, and 3-segment bands sit below
    it — are NOT the whole fill's bits. (The aligned bands are, above.)"""
    G, counts = _assemble(
        "mixed-detached",
        monkeypatch,
        whole=False,
        step=3,
        fallback="numpy",
        unaligned=True,
    )
    # The production walk, with only the alignment taken out.
    (walk,) = counts["walks"]
    assert len(walk) > 1 and walk[0] == (0, 3), walk
    G_ref, _ = _assemble("mixed-detached", monkeypatch, whole=True, fallback="numpy")
    assert np.allclose(G, G_ref, rtol=0.0, atol=1e-12 * np.abs(G_ref).max())
    if np.array_equal(G, G_ref):
        # The elision is CPython-dependent: measured firing on 3.12 and not
        # on 3.14 (same numpy 2.5.3), where every band size gives the same
        # bits, so `align` is moot there and this control cannot go red.
        pytest.skip(
            "numpy's temporary elision does not fire on this Python "
            f"({sys.version.split()[0]}); band alignment cannot move a bit here"
        )


def _numpy_blocks_of(n_seg, name, monkeypatch):
    """Set `_FILL_WORKSPACE_BYTES` so `_fill_block` is `n_seg` test segments
    on this deck (both fills read it)."""
    s = DECKS[name]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n = int(s._build_geometry()["n_segs"])
    per_seg = s.n_qp_test * n * 16 * (5 * s.n_qp_const + 16)
    monkeypatch.setattr(sg, "_FILL_WORKSPACE_BYTES", n_seg * per_seg)
    assert sg._fill_block(n, s.n_qp_test, s.n_qp_const) == n_seg


def _scatter(shapes_reversed=False, carry_last=False):
    """`SinusoidalGalerkinSolver._scatter_band` with one reassociation put
    in, for the red controls: the three shape products summed `P2 + P1 + P0`,
    or a straddling basis's carried partial row added AFTER this band's
    entries instead of before them."""

    def scatter(self, rows, e0, e1, band):
        touched, local = np.unique(rows.i_of_entry[e0:e1], return_inverse=True)
        T = [
            np.zeros((touched.size, band[0].shape[1]), dtype=np.complex128)
            for _ in range(3)
        ]
        carried = {}
        for r, b in enumerate(touched):
            got = rows.carry.pop(int(b), None)
            if got is not None:
                carried[r] = got
                if not carry_last:
                    for t, g in zip(T, got):
                        t[r] = g
        for t, c in zip(T, band):
            np.add.at(t, local, c)
        if carry_last:
            for r, got in carried.items():
                for t, g in zip(T, got):
                    t[r] = t[r] + g
        fin = np.flatnonzero(rows.last[touched] < e1)
        if fin.size:
            P = [t[fin] @ M for t, M in zip(T, rows.Ms)]
            rows.G[touched[fin]] = (
                (P[2] + P[1]) + P[0] if shapes_reversed else (0 + P[0]) + P[1] + P[2]
            )
        for r in np.flatnonzero(rows.last[touched] >= e1):
            rows.carry[int(touched[r])] = tuple(t[r].copy() for t in T)

    return scatter


@pytest.mark.parametrize("name", ["free-dipole", "mixed-detached"])
@pytest.mark.parametrize("mutation", ["shapes-reversed", "carry-last"])
def test_a_reassociated_scatter_is_seen(name, mutation, monkeypatch):
    """The red controls. The scatter's two orderings are the whole product's:
    the shape products as `(0 + P0) + P1 + P2`, and a basis's entries in
    ascending order even across a band boundary. Either reassociation is the
    same algebra, and the byte gate above must notice it — on a fair share
    of the matrix. The faithful copy (no mutation) is bit-equal, so the copy
    is not where the difference comes from."""
    kw = dict(shapes_reversed=mutation == "shapes-reversed")
    kw["carry_last"] = not kw["shapes_reversed"]
    # Two segments a band: a straddling basis has entries on both sides of
    # a boundary, two of them after it at some boundaries — which is where
    # adding the carry last reassociates (one entry after it commutes).
    G_bad, _ = _assemble(name, monkeypatch, whole=False, step=2, scatter=_scatter(**kw))
    G_copy, _ = _assemble(name, monkeypatch, whole=False, step=2, scatter=_scatter())

    G_ref, _ = _assemble(name, monkeypatch, whole=True)
    assert np.array_equal(G_copy, G_ref)
    assert np.allclose(G_bad, G_ref, rtol=0.0, atol=1e-12 * np.abs(G_ref).max())
    differ = int((G_bad != G_ref).sum())
    assert differ > 0, f"{mutation}: a reassociated scatter went unseen"


@pytest.mark.slow
def test_the_fill_holds_G_plus_band_scratch(monkeypatch):
    """The 8-dipole array over Sommerfeld soil at 22·4 segments a dipole
    (N = 704, the survey's above x4): the traced peak of `_assemble_Z` past
    its start, in units of Z = 16·n_basis², with the Sommerfeld grid already
    cached (it is N-independent and would otherwise be the peak). The
    whole-triple fill holds the triple and its image, ~3 triples ≈ 9 Z plus
    G; the banded fill holds G and one band (a 4 MB budget, half a Z here)
    plus the near correction's fixed 8 MB block (`_NEAR_WORKSPACE_BYTES`),
    which at this size is another Z. Measured on Haswell: 2.8 Z banded,
    20.1 Z whole."""
    wires = [
        np.array([(-10.0, j * 8.5, 10.0), (10.0, j * 8.5, 10.0)]) for j in range(8)
    ]
    d = dict(
        wires=wires,
        n_per_edge_per_wire=[[88] for _ in wires],
        feeds=[(0, 10.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=1e-3,
        **SOMM,
    )

    def peak(whole):
        with monkeypatch.context() as m:
            m.setattr(
                SinusoidalGalerkinSolver,
                "_band_budget_bytes",
                lambda self, n_basis: 4 << 20,
            )
            m.setattr(sg, "_BAND_MIN_SEGMENTS", 1)
            if whole:
                m.setattr(
                    SinusoidalGalerkinSolver, "_band_fill_serves", lambda self, n: False
                )
            s = _direct_sg(**d)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                geom = s._build_geometry()
                G = s._assemble_Z(geom, s.k, s.eta)[0]  # warm the grid cache
                z_bytes = G.nbytes
                del G
                tracemalloc.start()
                base = tracemalloc.get_traced_memory()[0]
                try:
                    s._assemble_Z(geom, s.k, s.eta)
                    top = tracemalloc.get_traced_memory()[1] - base
                finally:
                    tracemalloc.stop()
        return top / z_bytes

    banded, whole = peak(False), peak(True)
    assert banded < 3.2, (
        f"banded fill traced peak {banded:.2f} Z (whole: {whole:.2f} Z)"
    )
    assert whole > 9.0, f"the whole-triple reference peaked at only {whole:.2f} Z"
