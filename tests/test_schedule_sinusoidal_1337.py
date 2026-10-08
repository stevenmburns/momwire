"""The sinusoidal fill walks the schedule layer — momwire#1337 phase 2, PR 1.

Four loops of `SinusoidalSolver` moved onto `_schedule`, loop for loop:

  * S1 `_assemble_Z`'s observer bands: `mb_rows` + `chunks([(0, N)])`, the
    dense-threshold one-band override kept in the trunk;
  * S2 `_class_bands`' bands of a mixed deck: `chunks(runs(idx))` (the
    trunk's own `_index_runs` was a copy of `_schedule.runs` and is gone);
  * S3 the fused mixed fill's Z bands: `chunks([(s, e)], zband)`, no
    checkpoint, as before;
  * S4 `_replay_remainder`'s observer chunks: `elem_rows` + `chunks`, the
    `row_group` round-DOWN (and its raise on a partial group) kept in the
    trunk.

The claim is that no float moved. A Z gate alone cannot carry it: above the
dense threshold every one of these fills is bit-INDEPENDENT of where its
band boundaries fall (each Z row reduces one Φ row on the source axis), so a
walk that cut the bands one row off would still produce the same Z. The
gates here are therefore three:

  * BOUNDARIES — the bands each loop actually walks, recorded from the
    production fill, equal the pre-migration loop's bands, spelled out here
    verbatim (the deleted `_index_runs` included);
  * BITS — Z (and the solved currents) at a budget that forces several
    bands is byte-equal to the one-band fill, on a free-space deck, a
    Sommerfeld deck and a mixed (buried + elevated) deck, with the band
    count asserted > 1 so the forcing cannot silently fail;
  * STATS — `_schedule.STATS["chunks"]` moves by exactly the bands walked,
    so the gates cannot pass through a loop that no longer exists.

The negative controls at the bottom show the boundary gate catches an
off-by-one walk that the Z gate cannot see, and the Z gate catches one that
drops a row.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _schedule
from momwire import sinusoidal as sn
from momwire.sinusoidal import SinusoidalSolver

LAM = 22.0

SOMMERFELD = {
    "ground_z": 0.0,
    "ground_eps": (13.0, 0.005),
    "ground_model": "sommerfeld",
}


def _wire_deck(**ground):
    """One elevated half-wave wire, N = 81 (above the dense threshold)."""
    ys = np.linspace(-0.962 * LAM / 4, 0.962 * LAM / 4, 2)
    wire = np.column_stack([np.zeros_like(ys), ys, np.full_like(ys, 4.0)])
    return dict(wires=[wire], nsegs=81, wavelength=LAM, wire_radius=0.0005, **ground)


def _mixed_deck():
    """A detached mixed deck (`test_sin_buried_below_1222.detached(6)`): a
    buried wire under a bent elevated one, each class 61 segments, so both
    classes sit above the dense threshold and their bands can split."""
    from test_sin_buried_below_1222 import detached

    return detached(6)


DECKS = {
    "free": _wire_deck,
    "sommerfeld": lambda: _wire_deck(**SOMMERFELD),
    "mixed": _mixed_deck,
}


def _bytes_equal(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


# ----------------------------------------------------------------------
# The pre-migration loops, verbatim (momwire da841987, 0.75.0).
# ----------------------------------------------------------------------


def _old_mb_chunk(sim, n):
    if n < sn._DENSE_ASSEMBLY_THRESHOLD:
        return n
    return max(1, int(sim.swept_mem_mb * 1024 * 1024 // sim._fill_row_bytes(n)))


def _old_index_runs(idx):
    if idx.size == 0:
        return []
    cut = np.flatnonzero(np.diff(idx) != 1) + 1
    firsts = np.concatenate(([0], cut))
    lasts = np.concatenate((cut, [idx.size]))
    return [(int(idx[a]), int(idx[b - 1]) + 1) for a, b in zip(firsts, lasts)]


def old_s1_bands(sim, N):
    chunk = _old_mb_chunk(sim, N)
    return [(i0, min(i0 + chunk, N)) for i0 in range(0, N, chunk)]


def old_s2_bands(sim, idx):
    chunk = _old_mb_chunk(sim, idx.size)
    return [
        (b0, min(b0 + chunk, e0))
        for s0, e0 in _old_index_runs(idx)
        for b0 in range(s0, e0, chunk)
    ]


def old_s3_bands(s, e, zband):
    return [(z0, min(z0 + zband, e)) for z0 in range(s, e, zband)]


def old_s4_bands(n_src, M, row_group):
    chunk = max(1, sn._REMAINDER_CHUNK_ELEMS // max(n_src, 1))
    if row_group > 1:
        if M % row_group:
            raise ValueError
        chunk = max(row_group, (chunk // row_group) * row_group)
    return [(i0, min(i0 + chunk, M)) for i0 in range(0, M, chunk)]


# ----------------------------------------------------------------------
# Recorders: what the production fill walked.
# ----------------------------------------------------------------------


class Walk:
    """Spies on the four loops of one solver's fill."""

    def __init__(self, sim, monkeypatch):
        self.s1 = []  # obs_rows of each _assemble_Z band
        self.s2 = []  # (s, e) of each class band, per class call
        self.s3 = []  # (z0, z1) of each reduced Z band
        self.s4 = []  # (n_src, M, row_group, [(i0, i1), ...]) per replay
        self.checkpoints = 0
        self.top_rows = None

        orig_ft = SinusoidalSolver._field_tensor
        orig_cb = SinusoidalSolver._class_bands
        orig_rr = SinusoidalSolver._replay_remainder
        orig_reduce = sn._reduce_phi_band
        walk = self

        def field_tensor(self_, geom, k, *a, **kw):
            # S1 calls it with the deck's own sources (no src_centers).
            if kw.get("src_centers") is None and "obs_rows" in kw:
                walk.s1.append(tuple(kw["obs_rows"]))
            return orig_ft(self_, geom, k, *a, **kw)

        def class_bands(self_, geom, keep, idx, *a, **kw):
            bands = []
            walk.s2.append((np.array(idx), bands))
            for item in orig_cb(self_, geom, keep, idx, *a, **kw):
                bands.append((item[0], item[1]))
                yield item
                item = None

        def replay_remainder(self_, prepared, proj, obs_c, obs_t, consume, row_group):
            rows = []

            def spy_proj(c, t):
                rows.append(c.shape[0])
                return proj(c, t)

            out = orig_rr(self_, prepared, spy_proj, obs_c, obs_t, consume, row_group)
            M = sum(rows)
            bounds = list(
                zip(np.cumsum([0, *rows[:-1]]).tolist(), np.cumsum(rows).tolist())
            )
            walk.s4.append((prepared["n_src"], M, row_group, bounds))
            return out

        def reduce_phi_band(Phi, Ms, out):
            base = out.base if out.base is not None else out
            z0 = (out.ctypes.data - base.ctypes.data) // out.strides[0]
            walk.s3.append((int(z0), int(z0) + out.shape[0]))
            return orig_reduce(Phi, Ms, out)

        def checkpoint():
            walk.checkpoints += 1

        monkeypatch.setattr(SinusoidalSolver, "_field_tensor", field_tensor)
        monkeypatch.setattr(SinusoidalSolver, "_class_bands", class_bands)
        monkeypatch.setattr(SinusoidalSolver, "_replay_remainder", replay_remainder)
        monkeypatch.setattr(sn, "_reduce_phi_band", reduce_phi_band)
        sim._checkpoint = checkpoint


def _solver(name, rows_per_band=None):
    """A solver of deck `name` whose single-medium band (or, on the mixed
    deck, each class band) is `rows_per_band` rows, asked for through the
    solver's own budget arithmetic; None is one band over every row."""
    sim = SinusoidalSolver(**DECKS[name]())
    geom = sim._build_geometry()
    if rows_per_band is None:
        sim.swept_mem_mb = 1 << 20
    else:
        n = _class_n(sim, geom) if name == "mixed" else int(geom["n_segs"])
        sim.swept_mem_mb = rows_per_band * sim._fill_row_bytes(n) / (1 << 20)
    return sim, geom


def _class_n(sim, geom):
    below = sim._below_segments(geom)
    n = {int(np.count_nonzero(below)), int(np.count_nonzero(~below))}
    assert len(n) == 1  # both classes size the same, so one budget fits both
    return n.pop()


def _fill(sim, geom):
    with sim._operating_medium(geom) as medium:
        G, _view = sim._assemble_Z(geom, sim.k, sim._medium_eta(medium))
    return G


@pytest.fixture(scope="module")
def whole():
    """Each deck's one-band Z and solved currents, computed once."""
    out = {}
    for name in DECKS:
        sim, geom = _solver(name)
        Z = _fill(sim, geom)
        out[name] = (Z, _solver(name)[0].compute_impedance()[1])
    return out


# ----------------------------------------------------------------------
# The gates.
# ----------------------------------------------------------------------


def _expected_s4_chunks(walk):
    return sum(len(b) for *_, b in walk.s4)


@pytest.mark.parametrize("name", ["free", "sommerfeld"])
@pytest.mark.parametrize("rows_per_band", [1, 7])
def test_single_medium_fill_walks_the_old_bands_bit_for_bit(
    name, rows_per_band, whole, monkeypatch
):
    sim, geom = _solver(name, rows_per_band)
    N = int(geom["n_segs"])
    assert N >= sn._DENSE_ASSEMBLY_THRESHOLD  # else the band is clamped to N
    walk = Walk(sim, monkeypatch)
    _schedule.reset_stats()
    Z = _fill(sim, geom)

    # BOUNDARIES: the pre-migration loop's bands, ragged last band included
    # (81 = 11·7 + 4), and several of them.
    expect = old_s1_bands(sim, N)
    assert len(expect) > 1
    assert walk.s1 == expect
    # S4 (Sommerfeld only): each band's remainder replay walked the old
    # chunks of that band's observers.
    if name == "sommerfeld":
        assert [M for _n, M, _g, _b in walk.s4] == [e - s for s, e in expect]
        for n_src, M, row_group, bounds in walk.s4:
            assert bounds == old_s4_bands(n_src, M, row_group)
    else:
        assert walk.s4 == []

    # STATS: the layer walked exactly these bands, one checkpoint each.
    assert _schedule.STATS["chunks"] == len(expect) + _expected_s4_chunks(walk)
    assert walk.checkpoints == _schedule.STATS["chunks"]

    # BITS: the same Z as one band over every row, and the same currents.
    assert _bytes_equal(Z, whole[name][0])
    sim2, _ = _solver(name, rows_per_band)
    assert _bytes_equal(sim2.compute_impedance()[1], whole[name][1])


@pytest.mark.parametrize("rows_per_band", [5, 13])
@pytest.mark.parametrize("zband", [3, None])
def test_mixed_fill_walks_the_old_bands_bit_for_bit(
    rows_per_band, zband, whole, monkeypatch
):
    sim, geom = _solver("mixed", rows_per_band)
    N = int(geom["n_segs"])
    if zband is not None:
        monkeypatch.setattr(SinusoidalSolver, "_mixed_band_rows", lambda self, N: zband)
    zb = sim._mixed_band_rows(N)
    walk = Walk(sim, monkeypatch)
    _schedule.reset_stats()
    Z = _fill(sim, geom)
    assert walk.s1 == []  # the mixed fill, not the single-medium one

    # BOUNDARIES. S2: two classes, each cut into the old bands of its runs.
    assert len(walk.s2) == 2
    for idx, bands in walk.s2:
        assert idx.size >= sn._DENSE_ASSEMBLY_THRESHOLD
        assert bands == old_s2_bands(sim, idx)
        assert len(bands) > 1
    class_bands = [b for _idx, bands in walk.s2 for b in bands]
    # S3: each class band cut into the old Z bands, in order.
    want_s3 = [z for s, e in class_bands for z in old_s3_bands(s, e, zb)]
    assert walk.s3 == want_s3
    if zband is not None:
        assert len(walk.s3) > len(class_bands)
    # S4: every remainder replay walked the old chunks.
    assert walk.s4
    for n_src, M, row_group, bounds in walk.s4:
        assert bounds == old_s4_bands(n_src, M, row_group)

    # STATS: S2 + S3 + S4, nothing more and nothing less; a checkpoint per
    # S2 band and S4 chunk, none per S3 band.
    s4 = _expected_s4_chunks(walk)
    assert _schedule.STATS["chunks"] == len(class_bands) + len(walk.s3) + s4
    assert walk.checkpoints == len(class_bands) + s4

    # BITS.
    assert _bytes_equal(Z, whole["mixed"][0])
    sim2, _ = _solver("mixed", rows_per_band)
    if zband is not None:
        assert sim2._mixed_band_rows(N) == zband
    assert _bytes_equal(sim2.compute_impedance()[1], whole["mixed"][1])


def _somm_prepared():
    from momwire import _ground_refl, _sommerfeld

    sim = SinusoidalSolver(**_wire_deck(**SOMMERFELD))
    geom = sim._build_geometry()
    eps_t = _ground_refl.eps_tilde(sim.ground_eps, sim.omega, sim.eps)
    r1_max = _sommerfeld.max_image_distance(geom["seg_l"], geom["seg_r"], sim.ground_z)
    return (
        sim,
        geom,
        sim._sommerfeld_remainder_prepare(geom, sim.k, eps_t, r1_max=r1_max),
    )


@pytest.mark.parametrize("row_group", [1, 3])
@pytest.mark.parametrize("elems_rows", [1, 2, 10])
def test_remainder_replay_walks_the_old_chunks_bit_for_bit(
    row_group, elems_rows, monkeypatch
):
    sim, geom, prepared = _somm_prepared()
    M = int(geom["n_segs"])
    assert M % row_group == 0  # 81 = 27·3
    # The reference: one chunk over every observer.
    with monkeypatch.context() as m:
        m.setattr(sn, "_REMAINDER_CHUNK_ELEMS", M * prepared["n_src"])
        assert len(old_s4_bands(prepared["n_src"], M, row_group)) == 1
        whole = sim._replay_sommerfeld_remainder(prepared)

    # Force a chunk of `elems_rows` rows (before the row_group rounding).
    monkeypatch.setattr(sn, "_REMAINDER_CHUNK_ELEMS", elems_rows * prepared["n_src"])
    expect = old_s4_bands(prepared["n_src"], M, row_group)
    assert len(expect) > 1
    got = []
    blocks = np.empty_like(whole)

    def consume(i0, i1, block):
        got.append((i0, i1))
        blocks[:, i0:i1] = block

    calls = [0]
    sim._checkpoint = lambda: calls.__setitem__(0, calls[0] + 1)
    _schedule.reset_stats()
    sim._replay_sommerfeld_remainder(prepared, consume=consume, row_group=row_group)
    assert got == expect
    assert _schedule.STATS["chunks"] == len(expect) == calls[0]
    assert _bytes_equal(blocks, whole)
    # The returned spelling is the streamed one.
    S = sim._replay_sommerfeld_remainder(prepared, row_group=row_group)
    assert _bytes_equal(S, whole)


def test_remainder_replay_still_refuses_a_partial_group(monkeypatch):
    sim, geom, prepared = _somm_prepared()
    M = int(geom["n_segs"])
    assert M % 2
    _schedule.reset_stats()
    with pytest.raises(ValueError, match="not a multiple of row_group"):
        sim._replay_sommerfeld_remainder(prepared, row_group=2)
    assert _schedule.STATS["chunks"] == 0  # raised before the walk


# ----------------------------------------------------------------------
# Negative controls: the gates above see an off-by-one walk.
# ----------------------------------------------------------------------


def _shifted_chunks(spans, chunk, checkpoint=None):
    """Every boundary one row early: the first band is chunk − 1 rows. All
    rows are still covered once, so in the sparse regime Z does not move."""
    for r0, r1 in spans:
        starts = [r0, *range(r0 + chunk - 1, r1, chunk)] if chunk > 1 else range(r0, r1)
        starts = [s for s in starts if s < r1]
        for a, b in zip(starts, [*starts[1:], r1]):
            if checkpoint is not None:
                checkpoint()
            _schedule.STATS["chunks"] += 1
            yield a, b


def _dropping_chunks(spans, chunk, checkpoint=None):
    """The last band of each span one row short: its last row is never
    filled."""
    for i0, i1 in _schedule_chunks(spans, chunk, checkpoint):
        last = any(i1 == r1 for _r0, r1 in spans)
        yield i0, (i1 - 1 if last and i1 - 1 > i0 else i1)


_schedule_chunks = _schedule.chunks


def test_an_off_by_one_boundary_is_caught_by_the_boundary_gate(whole, monkeypatch):
    sim, geom = _solver("free", 7)
    N = int(geom["n_segs"])
    monkeypatch.setattr(_schedule, "chunks", _shifted_chunks)
    walk = Walk(sim, monkeypatch)
    Z = _fill(sim, geom)
    assert sorted(i for s, e in walk.s1 for i in range(s, e)) == list(range(N))
    # The Z gate is blind to it — the reason the boundary gate exists ...
    assert _bytes_equal(Z, whole["free"][0])
    # ... and the boundary gate is not.
    assert walk.s1 != old_s1_bands(sim, N)


def test_a_dropped_row_is_caught_by_the_bit_gate(whole, monkeypatch):
    sim, geom = _solver("free", 7)
    monkeypatch.setattr(_schedule, "chunks", _dropping_chunks)
    Z = _fill(sim, geom)
    assert not _bytes_equal(Z, whole["free"][0])
