"""B-spline's field block and hmatrix's block loops walk the schedule layer —
momwire#1337 phase 2, PR 3.

Two loops of the B-spline family moved onto `_schedule`, loop for loop:

  * B1 the field-form Galerkin block's observer chunks — the rectangle
    (`_field_galerkin_block`) and the halved square
    (`_field_galerkin_block_symmetric`), which serve the buried fill's
    below/below remainder and transmitted blocks: `chunks([(0, n_obs)],
    chunk, self._checkpoint)`. The chunk height stays the trunk's rule
    (`_FIELD_GALERKIN_CHUNK_ELEMS` projected-table entries a chunk, a module
    constant now only so a test can force several chunks on a small deck
    through the production rule);
  * H1 `HMatrixSolver.build_hmatrix`'s two block loops: `walk(part["near"],
    ..., checkpoint=)` then `walk(part["far"], ..., checkpoint=)`. The
    partition and ACA stay hmatrix's own schedule.

The claim is that no float moved, and the gates are four:

  * BOUNDARIES — the chunks each B1 loop walked, recorded on the layer from
    the production fill (`record_walks`), equal the pre-migration loop for
    the same call spelled out here verbatim, at the shipped chunk and at a
    forced one (several chunks, a ragged last one); H1's walks visit the
    partition's near blocks and then its far blocks, in its order;
  * BITS — B1: the buried Z (and, at the shipped chunk, the solved
    currents) byte-equal to the PRE-MIGRATION functions (`old_*`, verbatim
    from cc1550f5 save for qualifying the module's names and reading the
    chunk constant this change introduced). The field block is NOT
    chunk-independent — a basis row straddling a boundary reassociates — so
    a one-chunk fill is no reference. H1: every near, far and preconditioner
    block, and the solved impedance and currents, byte-equal to the
    pre-migration `build_hmatrix`, free, over PEC, refl-coef and Sommerfeld
    ground, and on a junction;
  * STATS — `_schedule.STATS["chunks"]` / `["walked"]` equal what the
    recorders saw, B1's share is the old loop's chunk count, H1's is
    ``len(near) + len(far)``, and the checkpoints are one per chunk / block,
    taken by the walk, as the old loops took one;
  * NEGATIVE CONTROLS — a walk with every boundary one row early fails the
    boundary gate, one that drops a row fails the bit gate, and an H1 walk
    that skips a block or reverses the order fails both.

Run against the tree before this change, every gate here fails: neither
loop walks `_schedule`, so no field-block / `build_hmatrix` walk is recorded
and STATS reads zero for them (and `_FIELD_GALERKIN_CHUNK_ELEMS` does not
exist).
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest
from test_field_galerkin_914 import hub_deck, screen_deck
from test_schedule_sg_1337 import _dropping_chunks, _shifted_chunks
from test_sg_fused_banded_fill_1224 import logged_chunks, record_walks

import momwire.bspline as _bs
import momwire.hmatrix as _hm
from momwire import _schedule
from momwire.bspline import BSplineSolver
from momwire.hmatrix import HMatrixSolver

pytestmark = [
    pytest.mark.filterwarnings("ignore:crossing node"),
    pytest.mark.skipif(
        not _bs._HAVE_FIELD_GALERKIN_STRIDED,
        reason="the field block's shipped routes need the strided accelerator",
    ),
]

FG = "_field_galerkin_block"
FG_SYM = "_field_galerkin_block_symmetric"
SHIPPED = 1 << 19


def _bytes_equal(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def _flat(cur):
    return np.concatenate([np.ravel(np.asarray(c)) for c in cur])


# ----------------------------------------------------------------------
# The pre-migration functions, verbatim (momwire cc1550f5; docstrings cut,
# module names qualified, `1 << 19` read from the constant it became).
# ----------------------------------------------------------------------


def old_field_galerkin_block(
    self,
    supp_seg,
    polys,
    proj_fn,
    obs_idx,
    src_idx,
    obs,
    t_obs,
    W_obs,
    src,
    t_src,
    W_src,
    *,
    out=None,
    scale=1.0,
    row_of=None,
    symmetric=False,
):
    d = self.degree
    n_obs = len(obs_idx)
    n_src = len(src_idx)
    n_basis = polys.shape[0]
    # `out=` accumulates into the caller's matrix instead of returning a
    # fresh one, which is the whole point of the lever: at 150 radials the
    # (n, n) transient this would otherwise allocate is 1.05 GB. `scale`
    # rides with it, so `out=Z, scale=-1.0` IS `Z -= block` with no
    # temporary. NOT bit-identical to the two-step form and not meant to
    # be -- the observer loop below chunks, and a basis row whose support
    # straddles a chunk boundary reassociates: `Z - (c1 + c2)` becomes
    # `(Z - c1) - c2`.
    # `row_of` (momwire#1132) makes `out` ROW-COMPACT: (n_rows, n_basis),
    # basis row m accumulated at row `row_of[m]`. Only the address of each
    # add moves, so a held row is the square target's row bit for bit.
    if row_of is not None and out is None:
        raise ValueError("row_of names the rows of a caller's `out`")
    Q = np.zeros((n_basis, n_basis), dtype=np.complex128) if out is None else out
    if n_obs == 0 or n_src == 0:
        return Q
    # The order is INFERRED from the arrays handed in and never asked of
    # the solver (momwire#1004). `q` RESHAPES the projected table below,
    # and the nodes were built at whatever order the caller chose for this
    # pair class — a cross block near the plane gets a raised one — so an
    # order taken from a second source that disagreed would reshape a
    # correct table into a wrong answer with no exception. One derivation,
    # at the consumer, cross-checked against the moment weights it is about
    # to contract: `W[p, i, q]` carries the same q, and it is what the C++
    # twin reads, so the two routes cannot part on it either.
    q, rem_obs = divmod(len(obs), n_obs)
    q_src, rem_src = divmod(len(src), n_src)
    assert rem_obs == 0 and rem_src == 0 and q == q_src, (
        len(obs),
        n_obs,
        len(src),
        n_src,
    )
    assert W_obs.shape[-1] == q and W_src.shape[-1] == q, (
        W_obs.shape,
        W_src.shape,
        q,
    )
    # Global segment index -> position on each axis; -1 means "not on
    # this axis", which is how a wing that belongs to the other medium
    # drops out of the block rather than being clamped into it.
    n_seg_total = int(np.max(supp_seg)) + 1
    pos_o = np.full(n_seg_total, -1, dtype=np.int64)
    pos_o[obs_idx] = np.arange(n_obs)
    pos_s = np.full(n_seg_total, -1, dtype=np.int64)
    pos_s[src_idx] = np.arange(n_src)

    chunk = max(1, _bs._FIELD_GALERKIN_CHUNK_ELEMS // max(n_src * q * q, 1))
    if symmetric:
        # The caller's claim, checked rather than trusted: a mirror
        # assembled over two DIFFERENT axes is a wrong block, not a slow one.
        same = (
            np.array_equal(obs_idx, src_idx)
            and (obs is src or np.array_equal(obs, src))
            and (t_obs is t_src or np.array_equal(t_obs, t_src))
            and (W_obs is W_src or np.array_equal(W_obs, W_src))
        )
        if not same:
            raise ValueError(
                "symmetric=True needs the observer and source axes to be "
                "the same nodes, tangents and moment weights"
            )
    if (
        symmetric
        and _bs._FIELD_GALERKIN_SYMMETRIC
        and row_of is None
        and _bs._HAVE_FIELD_GALERKIN_STRIDED
    ):
        old_field_galerkin_block_symmetric(
            self,
            supp_seg,
            polys,
            proj_fn,
            obs,
            t_obs,
            W_obs,
            pos_o,
            n_obs,
            q,
            chunk,
            Q,
            scale,
        )
        return Q
    for i0 in range(0, n_obs, chunk):
        self._checkpoint()
        i1 = min(i0 + chunk, n_obs)
        proj = proj_fn(
            obs[i0 * q : i1 * q],
            t_obs[i0 * q : i1 * q],
            src,
            t_src,
        )
        if _bs._HAVE_FIELD_GALERKIN_ACCEL and (
            row_of is None or _bs._HAVE_FIELD_GALERKIN_ROW_OF
        ):
            # momwire#914 unit 2. The C++ twin fuses both moment sums into
            # a q-vector per wing, so it never materialises `Jc` nor the
            # per-wing-pair gather below; it accumulates into `Q` in place.
            # Everything after this line is the reference it is gated
            # against, and the fallback when the .so predates the contract.
            _bs._acc.assemble_field_galerkin(
                proj,
                np.ascontiguousarray(W_obs[:, i0:i1]),
                W_src,
                supp_seg,
                polys,
                pos_o,
                pos_s,
                i0,
                Q,
                _bs._FIELD_GALERKIN_FUSED,
                scale,
                **({} if row_of is None else {"row_of": row_of}),
                **_bs._fg_lanes_kw(),
            )
            continue
        fq = proj.reshape(i1 - i0, q, n_src, q)
        # optimize=True: pairwise contraction, momwire#910 (see the
        # remainder's twin above for the measurement).
        Jc = np.einsum("piq,iqjr,Pjr->pPij", W_obs[:, i0:i1], fq, W_src, optimize=True)
        for a in range(d + 1):
            pm = pos_o[supp_seg[:, a]]
            rows = np.nonzero((pm >= i0) & (pm < i1))[0]
            if row_of is not None:
                # A row the compact target does not hold reaches here only
                # through a padded slot (an exact zero): drop it, as the
                # C++ twin does, rather than let -1 index the last row.
                rows = rows[row_of[rows] >= 0]
            if rows.size == 0:
                continue
            pml = pm[rows] - i0
            for b in range(d + 1):
                pn = pos_s[supp_seg[:, b]]
                cols = np.nonzero(pn >= 0)[0]
                if cols.size == 0:
                    continue
                J_blk = Jc[:, :, pml[:, None], pn[cols][None, :]]
                q_rows = rows if row_of is None else row_of[rows]
                Q[np.ix_(q_rows, cols)] += scale * np.einsum(
                    "mp,pPmn,nP->mn",
                    polys[rows, a, :],
                    J_blk,
                    polys[cols, b, :],
                )
    return Q


def old_field_galerkin_block_symmetric(
    self,
    supp_seg,
    polys,
    proj_fn,
    nodes,
    tangents,
    W,
    pos,
    n_axis,
    q,
    chunk,
    Q,
    scale,
):
    QT = Q.T
    for i0 in range(0, n_axis, chunk):
        self._checkpoint()
        i1 = min(i0 + chunk, n_axis)
        proj = proj_fn(
            nodes[i0 * q : i1 * q],
            tangents[i0 * q : i1 * q],
            nodes[i0 * q :],
            tangents[i0 * q :],
        )
        W_rows = np.ascontiguousarray(W[:, i0:i1])
        W_cols = np.ascontiguousarray(W[:, i0:])
        # Source positions relative to the table's first column; a
        # segment before i0 (or off this axis) drops out as -1.
        pos_from_i0 = np.where(pos >= i0, pos - i0, -1)
        mirror = {}
        if (
            i1 < n_axis
            and _bs._FIELD_GALERKIN_MIRROR
            and _bs._FIELD_GALERKIN_FUSED
            and _bs._HAVE_FIELD_GALERKIN_MIRROR
        ):
            # momwire#1290: the mirror below, in the same call.
            mirror = {
                "mirror_pos_s": np.where(pos >= i1, pos - i0, -1),
                "mirror_Q": QT,
            }
        _bs._acc.assemble_field_galerkin(
            proj,
            W_rows,
            W_cols,
            supp_seg,
            polys,
            pos,
            pos_from_i0,
            i0,
            Q,
            _bs._FIELD_GALERKIN_FUSED,
            scale,
            **mirror,
            **_bs._fg_lanes_kw(),
        )
        if i1 < n_axis and not mirror:
            # The mirror: the same table with the chunk's own columns
            # masked (the diagonal block is already in, in both orders),
            # written through the transposed target.
            pos_past_i1 = np.where(pos >= i1, pos - i0, -1)
            _bs._acc.assemble_field_galerkin(
                proj,
                W_rows,
                W_cols,
                supp_seg,
                polys,
                pos,
                pos_past_i1,
                i0,
                QT,
                _bs._FIELD_GALERKIN_FUSED,
                scale,
                **_bs._fg_lanes_kw(),
            )
        del proj


def old_build_hmatrix(self, eta=None, leaf_size=None, tol=None, k=None):
    if tol is None:
        tol = self.aca_tol
    if k is None:
        k = self.k
    self._refuse_buried_fast_operator("HMatrixSolver.build_hmatrix")
    part = self.build_partition(eta=eta, leaf_size=leaf_size)
    ctx = self._context()
    n = ctx["n_basis"]
    grounded = self.ground_z is not None
    somm = (
        grounded and self.ground_eps is not None and (self.ground_model == "sommerfeld")
    )
    if somm:
        eps_t, c2 = self._somm_eps_c2()

    near_blocks = []
    for s, t in part["near"]:
        self._checkpoint()  # per near-block dense fill
        I, J = s.indices, t.indices
        D = self.zblock(I, J, k=k)
        if grounded:
            if somm:
                # C2-weighted exact image = C2 x the PEC-image block;
                # the smooth remainder rides the single global
                # low-rank term appended below.
                D = D - c2 * self._zblock_image(I, J, k=k)
            elif self.ground_eps is not None:
                D = D - self._zblock_image_refl(I, J, k=k)
            else:
                D = D - self._zblock_image(I, J, k=k)
        near_blocks.append((I, J, D))

    use_accel = (
        _hm._HAVE_OFFEDGE_BLOCK_ACCEL
        and self.degree <= _hm._OFFEDGE_BLOCK_ACCEL_MAX_D
        and self.hmatrix_use_accel
        # momwire#769: the off-edge block kernels carry the same capped
        # scratch as the dense pair kernels.
        and self._accel_serves_n_qp_pair
    )

    far_blocks = []
    precond_extra = []  # first-ring far blocks, dense, for the preconditioner
    p_eta = self.precond_eta
    for s, t in part["far"]:
        self._checkpoint()  # per far-block ACA build
        I, J = s.indices, t.indices
        mI, nJ = I.size, J.size

        get_row, get_col, dense = self._offedge_aca_evaluators(ctx, I, J, k, use_accel)

        U, V = _hm.aca_partial(get_row, get_col, mI, nJ, tol=tol)
        r = U.shape[1]
        if r * (mI + nJ) >= mI * nJ:
            # No compression — store dense (off-edge kernel; the block is
            # admissible so the same-edge analytic path is unnecessary).
            near_blocks.append((I, J, dense()))
        else:
            far_blocks.append((I, J, U, V))
            # "First ring": leaf-scale blocks adjacent to the near band —
            # admissible for the operator but not at the tighter
            # precond_eta. Fold a dense reconstruction into the
            # preconditioner (free: reuses the low-rank factors). The
            # leaf-size cap keeps it a *thin* ring: large far blocks (big
            # well-separated clusters) are genuinely far and excluded, so
            # the preconditioner stays sparse.
            if (
                p_eta < self.aca_eta
                and max(mI, nJ) <= self.aca_leaf_size
                and not _hm.admissible(s, t, p_eta)
            ):
                precond_extra.append((I, J, U @ V))

    if somm:
        # The smooth remainder as ONE extra global low-rank far block
        # (Z subtracts Q, so the factors carry the minus). Not in the
        # near list, so the GMRES preconditioner ignores it — a smooth
        # perturbation it converges through.
        U, V = self._sommerfeld_global_lowrank(k, eps_t)
        idx = np.arange(n, dtype=np.int64)
        far_blocks.append((idx, idx, U, -V))

    return _hm.HMatrix(
        n, near_blocks, far_blocks, precond_extra=precond_extra, cancel=self._cancel
    )


def old_chunks(n, chunk):
    """The old B1 loops' chunks."""
    return [(i0, min(i0 + chunk, n)) for i0 in range(0, n, chunk)]


# ----------------------------------------------------------------------
# B1: the field block's observer chunks.
# ----------------------------------------------------------------------


class FieldFill:
    """The buried fill of a deck through the production seam
    (`_compute_Z_operator_buried`), with a spy on the field block that
    records each call's chunk arithmetic and route, and the layer's walks.
    `old` runs the pre-migration functions instead."""

    def __init__(
        self,
        deck,
        m,
        *,
        elems=None,
        symmetric=True,
        accel=True,
        rows=None,
        old=False,
        currents=False,
    ):
        if elems is not None:
            m.setattr(_bs, "_FIELD_GALERKIN_CHUNK_ELEMS", elems)
        m.setattr(_bs, "_FIELD_GALERKIN_SYMMETRIC", symmetric)
        if not accel:
            m.setattr(_bs, "_HAVE_FIELD_GALERKIN_ACCEL", False)
        impl = old_field_galerkin_block if old else BSplineSolver._field_galerkin_block
        calls = []  # (n_obs, chunk, symmetric route?) per non-empty call
        fill = self

        def spy(self_, *a, **kw):
            n_obs, n_src = len(a[3]), len(a[4])
            if n_obs and n_src:
                q = len(a[5]) // n_obs
                chunk = max(1, _bs._FIELD_GALERKIN_CHUNK_ELEMS // max(n_src * q * q, 1))
                sym = bool(
                    kw.get("symmetric")
                    and _bs._FIELD_GALERKIN_SYMMETRIC
                    and kw.get("row_of") is None
                    and _bs._HAVE_FIELD_GALERKIN_STRIDED
                )
                calls.append((n_obs, chunk, sym))
            return impl(self_, *a, **kw)

        m.setattr(BSplineSolver, FG, spy)
        self.checkpoints = 0

        def checkpoint():
            fill.checkpoints += 1

        log = record_walks(m)
        s = BSplineSolver(**deck())
        s._checkpoint = checkpoint
        geom = s._build_geometry()
        supp_seg, polys, *_ = s._build_basis_polynomials(geom)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.Z = np.array(
                s._compute_Z_operator_buried(geom, supp_seg, polys, rows=rows),
                copy=True,
            )
        # This fill's records, not the currents solve's below.
        self.log = list(log)
        self.calls = list(calls)
        self.stats = dict(_schedule.STATS)
        self.I = None
        if currents:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _z, cur = BSplineSolver(**deck()).compute_impedance()
            self.I = _flat(cur)

    def walks(self):
        return [w for w in self.log if w.by in (FG, FG_SYM)]


# (deck, forcing). Production is the symmetric route; `symmetric=False` is the
# rectangle, `accel=False` its numpy fallback, `rows=` the restricted fill
# (rectangle with a row map). `elems` forces the chunk through the step rule.
FIELD_CASES = {
    "screen12-shipped": (lambda: screen_deck(12), dict(currents=True)),
    "screen12-forced": (lambda: screen_deck(12), dict(elems=SHIPPED // 16)),
    "screen12-rect-forced": (
        lambda: screen_deck(12),
        dict(elems=SHIPPED // 16, symmetric=False),
    ),
    "screen12-rows-forced": (
        lambda: screen_deck(12),
        dict(elems=SHIPPED // 16, rows=np.arange(0, 40, dtype=np.int64)),
    ),
    "hub-numpy-forced": (
        lambda: hub_deck(n_radials=4),
        dict(elems=SHIPPED // 64, symmetric=False, accel=False),
    ),
}


@pytest.mark.parametrize("name", list(FIELD_CASES))
def test_field_block_walks_the_old_chunks_bit_for_bit(name, monkeypatch):
    deck, forcing = FIELD_CASES[name]
    with monkeypatch.context() as m:
        ref = FieldFill(deck, m, old=True, **forcing)
    with monkeypatch.context() as m:
        f = FieldFill(deck, m, **forcing)

    # BOUNDARIES: one walk per non-empty field block, on the route the call
    # took, its chunks the old loop's for the same n_obs and chunk.
    assert f.calls == ref.calls and f.calls
    assert ref.walks() == []
    walks = f.walks()
    assert [w.by for w in walks] == [FG_SYM if sym else FG for _n, _c, sym in f.calls]
    for w, (n_obs, chunk, _sym) in zip(walks, f.calls, strict=True):
        assert w.spans == [(0, n_obs)] and w.chunk == chunk
        assert w.got == old_chunks(n_obs, chunk)
    if forcing.get("symmetric", True) and "rows" not in forcing:
        assert any(sym for _n, _c, sym in f.calls)  # the halved route ran
    else:
        assert not any(sym for _n, _c, sym in f.calls)
    # Several chunks somewhere, and (when forced) a ragged last one.
    assert max(len(w.got) for w in walks) > 1
    if "elems" in forcing:
        assert any(
            len(w.got) > 1 and w.got[-1][1] - w.got[-1][0] < w.chunk for w in walks
        )

    # STATS and checkpoints: every chunk the layer counted was logged, the
    # field block's share is the old loops' count, one checkpoint per chunk
    # taken by the walk, and the fill's checkpoint total is the old one's.
    n_old = sum(len(old_chunks(n, c)) for n, c, _s in f.calls)
    assert f.stats["chunks"] == logged_chunks(f.log)
    assert sum(len(w.got) for w in walks) == n_old
    assert all(w.has_checkpoint and w.checkpoints == len(w.got) for w in walks)
    assert f.checkpoints == ref.checkpoints

    # BITS: the pre-migration functions' Z (and currents).
    assert _bytes_equal(f.Z, ref.Z)
    if forcing.get("currents"):
        assert _bytes_equal(f.I, ref.I)


# ----------------------------------------------------------------------
# H1: build_hmatrix's block loops.
# ----------------------------------------------------------------------

_GEPS = (10.0, 0.002)
_HALF = 0.962 * 22 / 4


def _vertical(z0, n, **kw):
    wire = np.array([[0.0, 0.0, z0], [0.0, 0.0, z0 + 2 * _HALF]])
    return dict(
        wires=[wire],
        degree=2,
        n_per_edge_per_wire=[[n]],
        wavelength=22.0,
        feeds=[(0, None, 1.0 + 0.0j)],
        **kw,
    )


def _junction():
    h = _HALF
    return dict(
        wires=[
            np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0 + h]]),
            np.array([[0.0, 0.0, 1.0 + h], [0.0, h, 1.0 + h]]),
        ],
        degree=2,
        n_per_edge_per_wire=[[60], [60]],
        wavelength=22.0,
        junctions=[[(0, "end"), (1, "start")]],
        feed_wire_index=0,
        ground_z=0.0,
    )


HM_DECKS = {
    "free": lambda: _vertical(0.0, 200),
    "pec": lambda: _vertical(2.0, 200, ground_z=0.0),
    "refl": lambda: _vertical(2.0, 200, ground_z=0.0, ground_eps=_GEPS),
    "somm": lambda: _vertical(
        2.0, 120, ground_z=0.0, ground_eps=_GEPS, ground_model="sommerfeld"
    ),
    "junction-pec": _junction,
}


class HFill:
    """`build_hmatrix` through the production seam (`_build_operator`, which
    every H-matrix solve calls), the layer's `walk`s recorded."""

    def __init__(self, deck, m, *, old=False):
        if old:
            m.setattr(HMatrixSolver, "build_hmatrix", old_build_hmatrix)
        self.walks = []  # (caller, [blocks walked], checkpoints)
        real = _schedule.walk
        fill = self

        def walk(blocks, fn, *, checkpoint=None, keep=None):
            got = []
            rec = [sys._getframe(1).f_code.co_name, got, 0]
            fill.walks.append(rec)

            def tally():
                rec[2] += 1
                checkpoint()

            def spy_fn(*block):
                got.append(block)
                return fn(*block)

            return real(
                blocks,
                spy_fn,
                checkpoint=None if checkpoint is None else tally,
                keep=keep,
            )

        m.setattr(_schedule, "walk", walk)
        _schedule.reset_stats()
        self.checkpoints = 0

        def checkpoint():
            fill.checkpoints += 1

        s = HMatrixSolver(**deck())
        s._checkpoint = checkpoint
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.H = s._build_operator()
        self.stats = dict(_schedule.STATS)
        self.part = s.build_partition()
        self.n_checkpoints_fill = self.checkpoints
        self.fill_walks = list(self.walks)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = HMatrixSolver(**deck())
            n_before = len(self.walks)
            self.z, cur = t.compute_impedance()
            self.solve_walked = any(
                w[0] == "build_hmatrix" for w in self.walks[n_before:]
            )
        self.I = np.ravel(np.asarray(cur))


def _same_blocks(a, b):
    return len(a) == len(b) and all(
        len(x) == len(y) and all(_bytes_equal(u, v) for u, v in zip(x, y, strict=True))
        for x, y in zip(a, b, strict=True)
    )


@pytest.mark.parametrize("name", list(HM_DECKS))
def test_build_hmatrix_walks_the_old_blocks_bit_for_bit(name, monkeypatch):
    with monkeypatch.context() as m:
        ref = HFill(HM_DECKS[name], m, old=True)
    with monkeypatch.context() as m:
        f = HFill(HM_DECKS[name], m)

    # BOUNDARIES: two walks by `build_hmatrix`, the near blocks then the far
    # ones, each the partition's list in its order; the old loop walked none.
    near, far = f.part["near"], f.part["far"]
    assert len(near) > 1 and len(far) > 1, (len(near), len(far))
    assert [w for w in ref.fill_walks if w[0] == "build_hmatrix"] == []
    hm = [w for w in f.fill_walks if w[0] == "build_hmatrix"]
    assert [w[1] for w in hm] == [list(near), list(far)]
    # ... and the solve itself went through `build_hmatrix`'s walks.
    assert f.solve_walked and not ref.solve_walked

    # STATS: every walked block was recorded, and the H-matrix's share is
    # len(near) + len(far), one checkpoint each, taken by the walk; the
    # fill's checkpoint total is the old loop's.
    assert f.stats["walked"] == sum(len(w[1]) for w in f.fill_walks)
    assert sum(len(w[1]) for w in hm) == len(near) + len(far)
    assert all(w[2] == len(w[1]) for w in hm)
    assert f.n_checkpoints_fill == ref.n_checkpoints_fill

    # BITS: every block of the operator, then the solve.
    assert _same_blocks(f.H.near, ref.H.near)
    assert _same_blocks(f.H.far, ref.H.far)
    assert _same_blocks(f.H.precond_extra, ref.H.precond_extra)
    assert _bytes_equal(np.asarray(f.z), np.asarray(ref.z))
    assert _bytes_equal(f.I, ref.I)


# ----------------------------------------------------------------------
# Negative controls.
# ----------------------------------------------------------------------


def test_an_off_by_one_field_chunk_is_caught(monkeypatch):
    deck, forcing = FIELD_CASES["screen12-forced"]
    with monkeypatch.context() as m:
        ref = FieldFill(deck, m, old=True, **forcing)
    with monkeypatch.context() as m:
        m.setattr(_schedule, "chunks", _shifted_chunks)
        f = FieldFill(deck, m, **forcing)
    # Every observer still filled once, at the same chunk height ...
    assert f.calls == ref.calls
    walks = f.walks()
    assert all(
        w.got[0][0] == 0 and w.got[-1][1] == n
        for w, (n, _c, _s) in zip(walks, f.calls, strict=True)
    )
    # ... and the boundary gate sees it.
    assert any(
        w.got != old_chunks(n, c) for w, (n, c, _s) in zip(walks, f.calls, strict=True)
    )


def test_a_dropped_field_row_is_caught(monkeypatch):
    deck, forcing = FIELD_CASES["screen12-forced"]
    with monkeypatch.context() as m:
        ref = FieldFill(deck, m, old=True, **forcing)
    with monkeypatch.context() as m:
        m.setattr(_schedule, "chunks", _dropping_chunks)
        f = FieldFill(deck, m, **forcing)
    assert not _bytes_equal(f.Z, ref.Z)


_real_walk = _schedule.walk


def _skipping_walk(blocks, fill, *, checkpoint=None, keep=None):
    """The last block of each walk never filled."""
    _real_walk(list(blocks)[:-1], fill, checkpoint=checkpoint, keep=keep)


def _reversed_walk(blocks, fill, *, checkpoint=None, keep=None):
    """Every block filled once, in reverse order."""
    _real_walk(list(blocks)[::-1], fill, checkpoint=checkpoint, keep=keep)


@pytest.mark.parametrize("bad", [_skipping_walk, _reversed_walk])
def test_a_wrong_hmatrix_walk_is_caught(bad, monkeypatch):
    with monkeypatch.context() as m:
        ref = HFill(HM_DECKS["free"], m, old=True)
    with monkeypatch.context() as m:
        m.setattr(_hm._schedule, "walk", bad)
        s = HMatrixSolver(**HM_DECKS["free"]())
        H = s.build_hmatrix()
    assert not (_same_blocks(H.near, ref.H.near) and _same_blocks(H.far, ref.H.far))
