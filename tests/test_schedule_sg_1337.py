"""SG's fill walks the schedule layer — momwire#1337 phase 2, PR 2.

Three loops of `SinusoidalGalerkinSolver`'s fill moved onto `_schedule`,
loop for loop:

  * G1 the direct fill's test-segment bands, `_segment_bands`, walked by
    both banded fills (`_assemble_G_banded`, `_assemble_mixed_G_banded`):
    `chunks([(0, N)], step)` with no checkpoint. The step rule — the byte
    budget, the `_BAND_MIN_SEGMENTS` floor and the round-UP to whole
    numpy-fill blocks (`align`) — stays in the trunk, and the consumers keep
    their own checkpoint, which an EMPTY band skips;
  * G2 the mixed-potential fill's observer windows,
    `_sinusoidal_mp.WindowFill.accumulate`: `chunks([(0, n_obs)], rows,
    checkpoint)`;
  * G3 the extended kernel's end-bracket row bands
    (`_ek_bracket_correction_tested`): `chunks([(0, n_basis)], step)`, no
    checkpoint, as before.

The claim is that no float moved, and the gates are four:

  * BOUNDARIES — the bands each loop walked, recorded on the layer from the
    production fill (`record_walks`, which names the function iterating each
    walk), equal the pre-migration loops spelled out here verbatim, at
    forcings that give several bands, a ragged last one, and an `align`
    round-up;
  * BITS — Z (and the solved currents) at those forcings: the direct fill's
    byte-equal to the one-band fill (it is band-independent by design,
    #1224) free, over Sommerfeld soil, on the numpy far fill (aligned), with
    the extended kernel, and on a mixed deck; the mixed-potential fill's
    byte-equal to the PRE-MIGRATION window loop (`old_accumulate`, verbatim)
    at the same window size, free and over Sommerfeld soil — its Z depends
    on where the windows fall, so a one-window reference is not one;
  * STATS — `_schedule.STATS["chunks"]` equals the chunks the recorder saw,
    each loop's share of it is the bands it walked, and the checkpoints are
    one per band where the old loop took one (G1: per non-empty band, in the
    consumer; G2: per window, in the walk; G3: none);
  * NEGATIVE CONTROLS — a walk with every boundary one row early fails the
    boundary gate, and one that drops a row fails the bit gate.

Run against the tree before this change, every gate here fails: none of the
three loops walks `_schedule`, so no `_segment_bands` / `accumulate` /
bracket walk is recorded and STATS reads zero for them.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest
from test_sg_fused_banded_fill_1224 import (
    _numpy_blocks_of,
    dipole,
    ell,
    logged_chunks,
    record_walks,
)
from test_sg_mixed_sign_1159 import mixed

from momwire import _schedule
from momwire import _sinusoidal_mp as mp
from momwire import sinusoidal_galerkin as sg
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

SOMM = dict(ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld")

# The direct fill's decks (fill="direct"), and the mixed-potential ones (the
# default fill since momwire#1354).
DIRECT = {
    "free": lambda: dict(**dipole(), fill="direct"),
    "somm": lambda: dict(**dipole(z=3.0, **SOMM), fill="direct"),
    "ek-ell": lambda: dict(**ell(extended_kernel=True), fill="direct"),
    "mixed": lambda: dict(**mixed(4), fill="direct"),
}
MP = {
    "mp-free": lambda: dict(**dipole(), fill="mixed-potential"),
    "mp-somm": lambda: dict(**dipole(z=3.0, **SOMM), fill="mixed-potential"),
}

ONE_BAND = 1 << 40
BRACKET = "_ek_bracket_correction_tested"


def _bytes_equal(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


# ----------------------------------------------------------------------
# The pre-migration loops, verbatim (momwire fd987abd).
# ----------------------------------------------------------------------


def old_segment_bands(self, ctx, n_basis, n_cols, n_triples, align=None):
    N = int(ctx["N"])
    starts = np.asarray(ctx["starts"], dtype=np.int64)
    nnz = int(np.asarray(ctx["w_entry"]).shape[0])
    entries_per_seg = nnz / max(1, N)
    per_seg = 16 * (3 * n_triples * entries_per_seg * n_cols + 3 * n_cols + 4 * n_basis)
    step = max(
        min(sg._BAND_MIN_SEGMENTS, N),
        int(self._band_budget_bytes(n_basis) // per_seg),
    )
    if align:
        step = -(-step // int(align)) * int(align)
    for m0 in range(0, N, step):
        m1 = min(m0 + step, N)
        yield m0, m1, int(starts[m0]), nnz if m1 == N else int(starts[m1])


def old_window_rows(n_obs, n_src):
    rows = max(1, mp.WINDOW_BYTES // (mp.N_SHAPES * mp.N_SHAPES * 16 * max(n_src, 1)))
    return [(i0, min(i0 + rows, n_obs)) for i0 in range(0, n_obs, rows)]


def old_accumulate(
    self,
    obs_c,
    obs_t,
    obs_h,
    a_row,
    src_c,
    src_t,
    src_h,
    *,
    scale=1.0,
    obs_idx=None,
    src_idx=None,
    ek=None,
):
    """`WindowFill.accumulate` as it was, its window loop included."""
    obs_c, obs_t = np.asarray(obs_c, float), np.asarray(obs_t, float)
    src_c, src_t = np.asarray(src_c, float), np.asarray(src_t, float)
    obs_h, src_h = np.asarray(obs_h, float), np.asarray(src_h, float)
    a_row = np.broadcast_to(np.asarray(a_row, float), (obs_c.shape[0],))
    n_obs, n_src = obs_c.shape[0], src_c.shape[0]
    sl_o, sr_o = mp.segment_ends(obs_c, obs_t, obs_h)
    sl_s, sr_s = mp.segment_ends(src_c, src_t, src_h)
    rows = max(1, mp.WINDOW_BYTES // (mp.N_SHAPES * mp.N_SHAPES * 16 * max(n_src, 1)))
    obs_idx = (
        np.arange(n_obs, dtype=np.int64)
        if obs_idx is None
        else np.asarray(obs_idx, dtype=np.int64)
    )
    all_cols = (
        np.arange(n_src, dtype=np.int64)
        if src_idx is None
        else np.asarray(src_idx, dtype=np.int64)
    )
    for i0 in range(0, n_obs, rows):
        if self.checkpoint is not None:
            self.checkpoint()
        i1 = min(i0 + rows, n_obs)
        ek_win = None if ek is None else (ek[0][i0:i1],) + tuple(ek[1:])
        J = self._window_moments(sl_o, sr_o, sl_s, sr_s, a_row, i0, i1, ek_win)
        ii, jj, Jn = mp.near_pair_moments(
            obs_c[i0:i1],
            obs_t[i0:i1],
            obs_h[i0:i1],
            src_c,
            src_t,
            src_h,
            a_row[i0:i1] ** 2,
            self.k,
            ek=ek_win,
        )
        if ii.size:
            J[:, :, ii, jj] = Jn
        mp.assemble_window(
            self.G,
            J,
            obs_idx[i0:i1],
            all_cols,
            self.starts,
            self.jbasis,
            self.coef,
            self.dcoef,
            obs_t[i0:i1],
            src_t,
            self.c_a,
            self.c_phi,
            scale=scale,
        )
        del J


def old_bracket_bands(self, n_basis):
    step = max(1, self._band_budget_bytes(n_basis) // (16 * 6 * n_basis))
    return [(i0, min(i0 + step, n_basis)) for i0 in range(0, n_basis, step)]


# ----------------------------------------------------------------------
# Recorders: what the production fill walked.
# ----------------------------------------------------------------------


class Fill:
    """One fill through the production seam, with spies on its three loops."""

    def __init__(self, deck, m, *, step=None, budget=None, window_rows=None):
        self.seg_calls = []  # (args, kwargs, yielded 4-tuples, old 4-tuples)
        self.windows = []  # (n_obs, n_src, [(i0, i1), ...]) per accumulate
        self.scattered = []  # (e0, e1) per scattered band
        self.consumer_checkpoints = 0
        self.checkpoints = 0
        if step is not None:
            # `step` segments a band through the step rule's floor.
            m.setattr(sg, "_BAND_MIN_SEGMENTS", step)
            m.setattr(SinusoidalGalerkinSolver, "_band_budget_bytes", lambda s, n: 0)
        elif budget is not None:
            # Through the byte budget itself, the floor lifted.
            m.setattr(sg, "_BAND_MIN_SEGMENTS", 1)
            m.setattr(
                SinusoidalGalerkinSolver, "_band_budget_bytes", lambda s, n: budget
            )
        self._spy(m)
        self.sim = SinusoidalGalerkinSolver(**deck())
        if window_rows is not None:
            n_src = int(self.sim._build_geometry()["n_segs"])
            if window_rows == "one":
                m.setattr(mp, "WINDOW_BYTES", ONE_BAND)
            else:
                cell = mp.N_SHAPES * mp.N_SHAPES * 16 * n_src
                m.setattr(mp, "WINDOW_BYTES", window_rows * cell)
        fill = self

        def checkpoint():
            fill.checkpoints += 1
            if sys._getframe(1).f_code.co_name in (
                "_assemble_G_banded",
                "_assemble_mixed_G_banded",
            ):
                fill.consumer_checkpoints += 1

        self.sim._checkpoint = checkpoint
        log = record_walks(m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            geom = self.sim._build_geometry()
            with self.sim._operating_medium(geom) as medium:
                G = self.sim._assemble_Z_ported(
                    geom, self.sim.k, self.sim._medium_eta(medium)
                )[0]
        self.Z = np.array(G, copy=True)
        self.log = list(log)  # this fill's walks, not a later solve's
        self.n_basis = self.Z.shape[0]
        self.stats = dict(_schedule.STATS)

    def _spy(self, m):
        fill = self
        real_sb = SinusoidalGalerkinSolver._segment_bands
        real_sc = SinusoidalGalerkinSolver._scatter_band
        real_acc = mp.WindowFill.accumulate
        real_wm = mp.WindowFill._window_moments

        def segment_bands(self_, *a, **kw):
            got = []
            old = list(old_segment_bands(self_, *a, **kw))
            fill.seg_calls.append((a, kw, got, old))
            for item in real_sb(self_, *a, **kw):
                got.append(item)
                yield item

        def scatter_band(self_, rows, e0, e1, band):
            fill.scattered.append((e0, e1))
            return real_sc(self_, rows, e0, e1, band)

        def accumulate(self_, obs_c, obs_t, obs_h, a_row, src_c, *a, **kw):
            wins = []
            fill.windows.append((len(obs_c), len(src_c), wins))
            self_._wins = wins
            try:
                return real_acc(self_, obs_c, obs_t, obs_h, a_row, src_c, *a, **kw)
            finally:
                del self_._wins

        def window_moments(self_, sl_o, sr_o, sl_s, sr_s, a_row, i0, i1, ek=None):
            self_._wins.append((i0, i1))
            return real_wm(self_, sl_o, sr_o, sl_s, sr_s, a_row, i0, i1, ek)

        m.setattr(SinusoidalGalerkinSolver, "_segment_bands", segment_bands)
        m.setattr(SinusoidalGalerkinSolver, "_scatter_band", scatter_band)
        m.setattr(mp.WindowFill, "accumulate", accumulate)
        m.setattr(mp.WindowFill, "_window_moments", window_moments)

    def walks(self, by):
        return [w for w in self.log if w.by == by]

    def currents(self, deck):
        """The solved currents of a fresh solver under the same forcing (the
        spies record that solve's fill elsewhere, so this fill's records
        stand)."""
        kept = self.seg_calls, self.windows, self.scattered
        self.seg_calls, self.windows, self.scattered = [], [], []
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _z, cur = SinusoidalGalerkinSolver(**deck()).compute_impedance()
        finally:
            self.seg_calls, self.windows, self.scattered = kept
        return np.concatenate([np.ravel(c) for c in cur])


def _one_band(deck, monkeypatch):
    """The direct fill in ONE band over every test segment."""
    with monkeypatch.context() as m:
        m.setattr(sg, "_BAND_MIN_SEGMENTS", ONE_BAND)
        f = Fill(deck, m)
        assert len(f.seg_calls[0][2]) == 1
        return f.Z, f.currents(deck), f


def _old_windows(deck, monkeypatch, rows):
    """The mixed-potential fill through the pre-migration window loop
    (`old_accumulate`) at `rows` observers a window."""
    with monkeypatch.context() as m:
        m.setattr(mp.WindowFill, "accumulate", old_accumulate)
        f = Fill(deck, m, window_rows=rows)
        assert f.walks("accumulate") == [] and f.windows
        return f.Z, f.currents(deck), f


# ----------------------------------------------------------------------
# G1 (+ G3): the direct fill's bands.
# ----------------------------------------------------------------------

# (deck, forcing): `step` through the floor, `budget` through the bytes,
# `numpy` the far fill without its accelerator (bands aligned to its
# 8-segment blocks; a step of 3 rounds UP to 8).
DIRECT_CASES = [
    ("free", dict(step=7, ragged=True)),  # 81 = 11·7 + 4
    ("free", dict(step=1)),
    ("free", dict(budget=256 * 1024)),
    ("somm", dict(step=7, ragged=True)),
    ("somm", dict(budget=256 * 1024)),
    ("ek-ell", dict(step=6, ragged=True)),  # 91 = 15·6 + 1
    ("mixed", dict(step=7)),
    ("mixed", dict(budget=1 << 20)),
    ("free", dict(step=3, numpy=True, ragged=True)),  # 81 = 10·8 + 1
    ("mixed", dict(step=3, numpy=True)),
]


def _direct_fill(name, forcing, m):
    forcing = dict(forcing)
    forcing.pop("ragged", None)
    if forcing.pop("numpy", False):
        m.setattr(sg, "_HAVE_GALERKIN_FAR_FILL", False)
        _numpy_blocks_of(8, _numpy_name(name), m)
    return Fill(DIRECT[name], m, **forcing)


def _numpy_name(name):
    return {"free": "free-dipole", "mixed": "mixed-detached"}[name]


@pytest.mark.parametrize(
    ("name", "forcing"),
    [
        pytest.param(n, f, id=f"{n}-{'-'.join(map(str, f.items()))}")
        for n, f in DIRECT_CASES
    ],
)
def test_direct_fill_walks_the_old_bands_bit_for_bit(name, forcing, monkeypatch):
    with monkeypatch.context() as m:
        if forcing.get("numpy"):
            m.setattr(sg, "_HAVE_GALERKIN_FAR_FILL", False)
            _numpy_blocks_of(8, _numpy_name(name), m)
        Z_ref, I_ref, _ = _one_band(DIRECT[name], m)
    with monkeypatch.context() as m:
        f = _direct_fill(name, forcing, m)
        I = f.currents(DIRECT[name])
        old_bracket = old_bracket_bands(f.sim, f.n_basis)

    # BOUNDARIES: one call of `_segment_bands`, walked on the layer, and its
    # bands the pre-migration loop's for the same arguments — several of
    # them, the last one ragged.
    ((args, kw, got, old),) = f.seg_calls
    assert got == old
    (walk,) = f.walks("_segment_bands")
    assert walk.got == [(m0, m1) for m0, m1, _e0, _e1 in got]
    assert not walk.has_checkpoint
    N = got[-1][1]
    heights = [m1 - m0 for m0, m1, _e0, _e1 in got]
    assert len(got) > 1 and set(heights[:-1]) == {heights[0]}, heights
    assert heights[-1] == (N % heights[0] or heights[0]), heights
    if forcing.get("ragged"):
        assert heights[-1] < heights[0], heights
    if "step" in forcing:
        align = kw.get("align")
        if forcing.get("numpy"):
            # The round-up stayed in the trunk: 3 asked, 8 walked.
            assert align == 8 and heights[0] == 8, (align, heights)
        else:
            assert align is None and heights[0] == forcing["step"], heights
    # The consumer scattered every non-empty band, in order.
    assert f.scattered == [(e0, e1) for _m0, _m1, e0, e1 in got if e1 > e0]
    # G3 (the extended kernel's deck only): the old bracket bands.
    brackets = f.walks(BRACKET)
    if name == "ek-ell":
        (bw,) = brackets
        assert bw.got == old_bracket
        assert len(bw.got) > 1 and not bw.has_checkpoint
    else:
        assert brackets == []

    # STATS: every chunk the layer counted was logged, the bands above are
    # their share, and a checkpoint was taken per non-empty band by the
    # consumer, as before (the walk itself took none).
    assert f.stats["chunks"] == logged_chunks(f.log) > 0
    assert sum(len(w.got) for w in f.log if w.by in ("_segment_bands", BRACKET)) == (
        len(got) + sum(len(w.got) for w in brackets)
    )
    assert f.consumer_checkpoints == len(f.scattered)
    assert walk.checkpoints == 0
    assert N == int(args[0]["N"])

    # BITS: the one-band fill's Z and currents.
    assert _bytes_equal(f.Z, Z_ref)
    assert _bytes_equal(I, I_ref)


# ----------------------------------------------------------------------
# G2: the mixed-potential fill's windows.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", list(MP))
@pytest.mark.parametrize("rows", [1, 7])
def test_mixed_potential_fill_walks_the_old_windows_bit_for_bit(
    name, rows, monkeypatch
):
    """The mixed-potential fill's Z is NOT window-independent: each window
    is one `assemble_window` call, whose sums into G follow the window (its
    one-window and seven-row Z differ in the last bits, before this change
    as after). So the bit reference here is the PRE-MIGRATION loop at the
    same window size (`old_accumulate`), and the cross-commit byte check of
    the PR compares it against the tree before the change too."""
    Z_ref, I_ref, ref = _old_windows(MP[name], monkeypatch, rows)
    with monkeypatch.context() as m:
        f = Fill(MP[name], m, window_rows=rows)
        I = f.currents(MP[name])
        old = [
            (n_obs, n_src, old_window_rows(n_obs, n_src))
            for n_obs, n_src, _ in f.windows
        ]

    # BOUNDARIES: free, then (over soil) the image, each the old windows,
    # and the same windows the old loop walked in the reference.
    assert f.seg_calls == []  # the mixed-potential fill, not the direct one
    assert len(f.windows) == (2 if name == "mp-somm" else 1)
    assert f.windows == old == ref.windows
    for _n_obs, _n_src, wins in f.windows:
        assert len(wins) > 1
        if rows == 7:
            assert wins[-1][1] - wins[-1][0] == 4  # 81 = 11·7 + 4
    walks = f.walks("accumulate")
    assert [w.got for w in walks] == [wins for _o, _s, wins in f.windows]

    # STATS and checkpoints: one per window, taken by the walk, as the old
    # loop took one per window (the reference counts its own).
    n_win = sum(len(wins) for _o, _s, wins in f.windows)
    assert f.stats["chunks"] == logged_chunks(f.log)
    assert sum(len(w.got) for w in walks) == n_win
    assert all(w.has_checkpoint and w.checkpoints == len(w.got) for w in walks)
    assert f.checkpoints == ref.checkpoints

    # BITS: the old loop's Z and currents.
    assert _bytes_equal(f.Z, Z_ref)
    assert _bytes_equal(I, I_ref)


# ----------------------------------------------------------------------
# Negative controls.
# ----------------------------------------------------------------------


def _shifted_chunks(spans, chunk, checkpoint=None):
    """Every boundary one row early: the first chunk is chunk − 1 rows. All
    rows are still covered once."""
    for r0, r1 in spans:
        starts = [r0, *range(r0 + chunk - 1, r1, chunk)] if chunk > 1 else range(r0, r1)
        starts = [s for s in starts if s < r1]
        for a, b in zip(starts, [*starts[1:], r1]):
            if checkpoint is not None:
                checkpoint()
            _schedule.STATS["chunks"] += 1
            yield a, b


_real_chunks = _schedule.chunks


def _dropping_chunks(spans, chunk, checkpoint=None):
    """The last chunk of each span one row short: its last row is never
    filled."""
    for i0, i1 in _real_chunks(spans, chunk, checkpoint):
        last = any(i1 == r1 for _r0, r1 in spans)
        yield i0, (i1 - 1 if last and i1 - 1 > i0 else i1)


def test_an_off_by_one_boundary_is_caught_by_the_boundary_gate(monkeypatch):
    Z_ref, _I, _ = _one_band(DIRECT["free"], monkeypatch)
    with monkeypatch.context() as m:
        m.setattr(_schedule, "chunks", _shifted_chunks)
        f = Fill(DIRECT["free"], m, step=7)
    ((_a, _kw, got, old),) = f.seg_calls
    # Every test segment still filled once, so the direct fill's Z — which
    # is bit-independent of its band boundaries by design (#1224) — does not
    # see it ...
    assert got[0][:2] == (0, 6) and got[-1][1] == old[-1][1]
    assert _bytes_equal(f.Z, Z_ref)
    # ... and the boundary gate does.
    assert got != old
    with monkeypatch.context() as m:
        m.setattr(_schedule, "chunks", _shifted_chunks)
        g = Fill(MP["mp-free"], m, window_rows=7)
        ((n_obs, n_src, wins),) = g.windows
        assert wins != old_window_rows(n_obs, n_src)


def test_a_dropped_row_is_caught_by_the_bit_gate(monkeypatch):
    Z_ref, _I, _ = _old_windows(MP["mp-free"], monkeypatch, 7)
    with monkeypatch.context() as m:
        m.setattr(_schedule, "chunks", _dropping_chunks)
        g = Fill(MP["mp-free"], m, window_rows=7)
    assert not _bytes_equal(g.Z, Z_ref)
    # The direct fill: the dropped test segment leaves a basis row
    # unfinished, which the fill refuses (or, were that check gone, Z moves).
    Z_ref, _I, _ = _one_band(DIRECT["free"], monkeypatch)
    with monkeypatch.context() as m:
        m.setattr(_schedule, "chunks", _dropping_chunks)
        try:
            f = Fill(DIRECT["free"], m, step=7)
        except AssertionError as e:
            assert "unfinished" in str(e)
        else:
            assert not _bytes_equal(f.Z, Z_ref)
