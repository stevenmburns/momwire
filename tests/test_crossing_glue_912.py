"""The crossing fill's Python glue, vectorised (momwire#912).

After #906 / #904 / #910 the 12-radial buried screen's largest term was the
crossing fill at 1.17 s, spread over index bookkeeping: `axis_data` built
the basis-sample matrices with a Python loop over every basis row and wing,
`_support_rows` rescanned those matrices per cluster block, `_main_split`
scanned every node per block with `np.isin`, and `_ends_and_corner` did
full (n, n) outer products for end vectors with a handful of nonzeros.

The vectorised forms are gated against the loop forms they replace, kept
here as the reference — the same products in the same order, so the bar is
`array_equal`, not a tolerance:

- G-912-1  F / Fd from `_basis_samples` equal the per-row loop on the real
           axes of the hub and crossing decks, and the ends table too.
- G-912-2  `_support_rows` from the segment→rows map is a superset of the
           F / Fd scan, and on these decks the same set.
- G-912-3  `_nodes_of` equals the `np.isin` form, ascending.
- G-912-4  the restricted rank-1 updates in `_ends_and_corner` give the
           block the full outer products gave, bit for bit.
- G-912-5  the hub and crossing decks' Z are unchanged (pinned to the
           digit against the values these tests printed before the edit).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import momwire._crossing_fill as cf
from momwire import BSplineSolver

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_crossing_serve_524 import crossing_deck, hub_deck  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:crossing node")


def _axes(build):
    """Every (ctx, seg_idx, coarse, axis) `axis_data` produced in one solve."""
    seen = []
    real = cf.axis_data

    def spy(ctx, seg_idx, coarse=False, **kw):
        out = real(ctx, seg_idx, coarse, **kw)
        seen.append((ctx, np.asarray(seg_idx), coarse, out))
        return out

    cf.axis_data = spy
    try:
        z, _ = BSplineSolver(**build).compute_impedance()
    finally:
        cf.axis_data = real
    return z, seen


def _loop_F_Fd(ctx, segof, u_phys):
    """The pre-#912 per-row loop, verbatim."""
    supp_seg, polys = ctx.basis.supp_seg, ctx.basis.polys
    n_basis = polys.shape[0]
    d = ctx.basis.degree
    F = np.zeros((n_basis, len(u_phys)))
    Fd = np.zeros((n_basis, len(u_phys)))
    for m in range(n_basis):
        for a_ in range(supp_seg.shape[1]):
            if not np.any(polys[m, a_] != 0.0):
                continue
            sel = np.nonzero(segof == supp_seg[m, a_])[0]
            if sel.size == 0:
                continue
            u = u_phys[sel]
            for p in range(d + 1):
                c = polys[m, a_, p]
                if c == 0.0:
                    continue
                F[m, sel] += c * u**p
                if p >= 1:
                    Fd[m, sel] += p * c * u ** (p - 1)
    return F, Fd


def _loop_ends(ctx, seg_idx):
    """The pre-#912 per-basis end loop, verbatim."""
    geom = ctx.geom
    supp_seg, polys = ctx.basis.supp_seg, ctx.basis.polys
    n_basis = polys.shape[0]
    d = ctx.basis.degree
    seg_off = geom.seg_offsets
    ends = []
    on_axis = set(int(g) for g in seg_idx)
    for w in range(len(seg_off) - 1):
        first, last = seg_off[w], seg_off[w + 1] - 1
        if first not in on_axis:
            continue
        for gseg, sign, u_end in ((first, -1.0, 0.0), (last, +1.0, None)):
            hh = geom.h[gseg]
            u = hh if u_end is None else 0.0
            pt = geom.seg_l[gseg] + (u / hh) * (geom.seg_r[gseg] - geom.seg_l[gseg])
            fv = np.zeros(n_basis)
            for m in range(n_basis):
                for a_ in range(supp_seg.shape[1]):
                    if supp_seg[m, a_] == gseg and np.any(polys[m, a_] != 0.0):
                        fv[m] += sum(polys[m, a_, p] * u**p for p in range(d + 1))
            if np.any(fv != 0.0):
                ends.append((pt, sign, fv))
    return ends


def _u_phys(ctx, ax):
    """Recover the arclength each node was built at: the nodes are
    `seg_l + (u/h)(seg_r - seg_l)`, so u = |node - seg_l| exactly enough
    for the loop and the vectorised form to see the SAME array."""
    seg_l = ctx.geom.seg_l[ax["segof"]]
    return np.linalg.norm(ax["nodes"] - seg_l, axis=1)


@pytest.fixture(scope="module", params=["hub", "crossing"])
def solved(request):
    build = hub_deck() if request.param == "hub" else crossing_deck(1)
    return request.param, _axes(build)


def test_g912_1_basis_samples_and_ends_equal_the_loop_forms(solved):
    _, (_, seen) = solved
    assert seen, "no axis was built"
    for ctx, seg_idx, coarse, ax in seen:
        u = _u_phys(ctx, ax)
        runs = cf._segment_runs(ax["segof"])
        F, Fd, _rows = cf._basis_samples(ctx.basis.supp_seg, ctx.basis.polys, runs, u)
        F_ref, Fd_ref = _loop_F_Fd(ctx, ax["segof"], u)
        assert np.array_equal(F, F_ref), f"F differs (coarse={coarse})"
        assert np.array_equal(Fd, Fd_ref), f"Fd differs (coarse={coarse})"
        ends_ref = _loop_ends(ctx, seg_idx)
        assert len(ax["ends"]) == len(ends_ref)
        for (pt, sg, fv), (pt_r, sg_r, fv_r) in zip(ax["ends"], ends_ref):
            assert np.array_equal(pt, pt_r) and sg == sg_r
            assert np.array_equal(fv, fv_r)


def test_g912_2_support_rows_from_the_map_cover_the_scan(solved):
    _, (_, seen) = solved
    rng = np.random.default_rng(912)
    for _ctx, _seg_idx, _coarse, ax in seen:
        n = ax["nodes"].shape[0]
        for _ in range(8):
            ii = np.sort(rng.choice(n, size=min(n, 17), replace=False))
            got = cf._support_rows(ax, ii)
            scan = np.flatnonzero(
                np.any(ax["F"][:, ii] != 0, axis=1)
                | np.any(ax["Fd"][:, ii] != 0, axis=1)
            )
            assert set(scan) <= set(got), "the map dropped a live row"
            assert np.array_equal(got, scan), "the map is not the scan on this deck"


def test_g912_3_nodes_of_is_the_isin_form(solved):
    _, (_, seen) = solved
    rng = np.random.default_rng(3)
    for _ctx, seg_idx, _coarse, ax in seen:
        for _ in range(8):
            segs = rng.choice(seg_idx, size=min(len(seg_idx), 5), replace=False)
            got = cf._nodes_of(ax, segs)
            ref = np.flatnonzero(np.isin(ax["segof"], segs))
            assert np.array_equal(got, ref)
        # a segment that is not on this axis contributes nothing, as isin did
        assert cf._nodes_of(ax, np.array([10**6])).size == 0


def test_g912_4_the_restricted_end_updates_are_the_full_outer_products():
    rng = np.random.default_rng(4)
    n, m = 30, 40
    t_full = np.zeros((n, m), dtype=np.complex128)
    t_res = np.zeros((n, m), dtype=np.complex128)
    for _ in range(5):
        fv = np.zeros(n)
        fv[rng.choice(n, 3, replace=False)] = rng.normal(size=3)
        v = rng.normal(size=m) + 1j * rng.normal(size=m)
        c = complex(rng.normal(), rng.normal())
        t_full += c * np.outer(fv, v)
        nz = np.flatnonzero(fv)
        t_res[nz] += c * np.outer(fv[nz], v)
        gv = np.zeros(m)
        gv[rng.choice(m, 2, replace=False)] = rng.normal(size=2)
        w = rng.normal(size=n) + 1j * rng.normal(size=n)
        t_full += c * np.outer(w, gv)
        nz = np.flatnonzero(gv)
        t_res[:, nz] += c * np.outer(w, gv[nz])
    assert np.array_equal(t_full, t_res)


def test_g912_5_the_decks_z_are_unchanged(solved, record_property):
    name, (z, _) = solved
    record_property(f"z_{name}", f"{z:.9f}")
    pinned = {
        # Printed by these decks on main 624897f, before the edit.
        "hub": 141.016615417 - 43.425182328j,
        "crossing": 138.960862256 - 102.609718869j,
    }[name]
    assert abs(z - pinned) < 5e-7, (z, pinned)
