"""The field-form Galerkin assembly: the mirror in one call, the AVX2 stage 1
and the dead-column skip (momwire#1290).

The symmetric route (`_field_galerkin_block_symmetric`, momwire#1224)
assembled each projected chunk twice, into Q and through other source
positions into Q.T, and stage 1 -- the chunk's table against the row-wings'
g vectors -- does not depend on the target, so it ran twice for one F. One
call now forms F once and runs stage 2 into Q and then into the mirror, the
order of the two calls. Stage 1 runs two (j, r) entries to a vector in the
AVX2 build, each part the scalar `acc += g * v`; and a column whose every
wing drops out is skipped when `scale` is negative, where its add of
`scale * (+0, +0)` = (-0, -0) is the identity. None of it may move a bit:

- G-1290-F1  on the live calls of a buried solve's symmetric route, one
             mirror call into (Q, Q.T) is the two scalar calls
             (`lanes=False`), as uint64, into targets seeded with signed
             zeros and ordinary values.
- G-1290-F2  a positive scale still visits every column: a -0 entry in a
             dead column becomes +0 exactly as the scalar route makes it.
- G-1290-F3  stage 1's lanes on a synthetic table holding +0, -0 and an odd
             (j, r) count, every batch size, against `lanes=False`.
- G-1290-F4  end to end, the buried Z with the mirror and lanes and with
             both flags cleared, bit for bit, the mirror counted as run.

On Windows (MSVC, /fp:fast) the lanes run too since momwire#1371, and every
lanes-vs-walk comparison here takes the derived win32 tolerance instead of bit
equality (`assert_lanes_match`, tests/_lane_gate.py). Linux and macOS keep the
bit gates.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import momwire.bspline as _bs
from momwire import BSplineSolver
from _lane_gate import assert_lanes_match

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_field_galerkin_914 import screen_deck  # noqa: E402

pytestmark = [
    pytest.mark.filterwarnings("ignore:crossing node"),
    pytest.mark.skipif(
        not _bs._HAVE_FIELD_GALERKIN_MIRROR,
        reason="the .so predates the one-call mirror: rebuild",
    ),
]


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


@pytest.fixture(scope="module")
def live_calls():
    """The (args, kwargs) of every mirror call one buried solve made."""
    seen = []
    real = _bs._acc.assemble_field_galerkin

    def spy(*a, **kw):
        if "mirror_Q" in kw:
            seen.append((a, dict(kw)))
        return real(*a, **kw)

    mp = pytest.MonkeyPatch()
    mp.setattr(_bs._acc, "assemble_field_galerkin", spy)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            BSplineSolver(**screen_deck(12)).compute_impedance()
    finally:
        mp.undo()
    assert len(seen) > 2, "the solve never took the one-call mirror"
    return real, seen


def _target(n, rng):
    """A column-major (n, n) target like the buried Z, seeded with ordinary
    values and with signed zeros, the entries a skipped add could flip."""
    Q = np.asfortranarray(
        rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    )
    zeros = rng.random((n, n)) < 0.2
    Q[zeros] = complex(-0.0, -0.0)
    Q[rng.random((n, n)) < 0.1] = complex(0.0, -0.0)
    return Q


def _mirror_vs_two(real, a, kw, scale, seed):
    a = list(a)
    rng = np.random.default_rng(seed)
    n = a[8].shape[0]
    Q1 = _target(n, rng)
    Q2 = Q1.copy(order="F")
    a[10] = scale
    # one call, lanes
    a[8] = Q1
    real(*a, mirror_pos_s=kw["mirror_pos_s"], mirror_Q=Q1.T)
    # the two scalar calls it replaces
    a[8] = Q2
    real(*a, lanes=False)
    a[6] = kw["mirror_pos_s"]
    a[8] = Q2.T
    real(*a, lanes=False)
    return Q1, Q2


def test_g1290_f1_one_mirror_call_is_the_two_scalar_calls(live_calls):
    real, seen = live_calls
    for i, (a, kw) in enumerate(seen[:: max(1, len(seen) // 6)]):
        Q1, Q2 = _mirror_vs_two(real, a, kw, -1.0, i)
        assert_lanes_match(Q1, Q2, i)


def test_g1290_f2_a_positive_scale_visits_every_column(live_calls):
    real, seen = live_calls
    a, kw = seen[len(seen) // 2]
    Q1, Q2 = _mirror_vs_two(real, a, kw, 1.0, 7)
    assert_lanes_match(Q1, Q2)
    # and the scalar route did flip a -0 the skip would have kept
    rng = np.random.default_rng(7)
    seed = _target(Q1.shape[0], rng)
    neg0 = (np.signbit(seed.real)) & (seed.real == 0) & (Q1.real == 0)
    assert np.any(neg0 & ~np.signbit(Q1.real)), "no dead -0 entry was exercised"


@pytest.mark.parametrize("n_rows", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("n_src", [1, 2, 7])
def test_g1290_f3_stage_one_lanes_on_signed_zeros(n_rows, n_src):
    rng = np.random.default_rng(100 * n_rows + n_src)
    d, P, q = 2, 3, 3  # q odd, so n_src * q is odd for odd n_src
    n_seg = 6
    nc = 1  # one observer segment in the chunk
    nb = 8
    supp_seg = rng.integers(0, n_seg, (nb, d + 1)).astype(np.int64)
    supp_seg[:n_rows, 0] = 0  # n_rows basis rows with a wing on segment 0
    supp_seg[n_rows:, :] = rng.integers(1, n_seg, (nb - n_rows, d + 1))
    polys = rng.standard_normal((nb, d + 1, P))
    polys[0, 0, 0] = -0.0
    W_obs = rng.standard_normal((P, nc, q))
    W_src = rng.standard_normal((P, n_src, q))
    proj = rng.standard_normal((nc * q, n_src * q)) + 1j * rng.standard_normal(
        (nc * q, n_src * q)
    )
    proj[rng.random(proj.shape) < 0.3] = complex(-0.0, 0.0)
    proj[rng.random(proj.shape) < 0.2] = complex(0.0, -0.0)
    pos_o = np.full(n_seg, -1, dtype=np.int64)
    pos_o[0] = 0
    pos_s = np.full(n_seg, -1, dtype=np.int64)
    pos_s[: min(n_src, n_seg)] = np.arange(min(n_src, n_seg))
    out = []
    for lanes in (True, False):
        Q = np.zeros((nb, nb), dtype=complex)
        _bs._acc.assemble_field_galerkin(
            proj,
            W_obs,
            W_src,
            supp_seg,
            polys,
            pos_o,
            pos_s,
            0,
            Q,
            True,
            1.0,
            lanes=lanes,
        )
        out.append(Q)
    assert_lanes_match(out[0], out[1])


def test_g1290_f4_the_buried_z_with_and_without(monkeypatch):
    def fill(on):
        monkeypatch.setattr(_bs, "_FIELD_GALERKIN_MIRROR", on)
        monkeypatch.setattr(_bs, "_FIELD_GALERKIN_LANES", on)
        count = {"mirror": 0, "scalar": 0, "other": 0}
        real = _bs._acc.assemble_field_galerkin

        def spy(*a, **kw):
            if "mirror_Q" in kw:
                count["mirror"] += 1
            elif kw.get("lanes", True) is False:
                count["scalar"] += 1
            else:
                count["other"] += 1
            return real(*a, **kw)

        monkeypatch.setattr(_bs._acc, "assemble_field_galerkin", spy)
        s = BSplineSolver(**screen_deck(12))
        geom = s._build_geometry()
        supp_seg, polys, *_ = s._build_basis_polynomials(geom)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z = s._compute_Z_operator_buried(geom, supp_seg, polys)
        monkeypatch.undo()
        return Z, count

    Zm, cm = fill(True)
    Zs, cs = fill(False)
    assert cm["mirror"] > 2, cm
    assert cs["mirror"] == 0 and cs["other"] == 0 and cs["scalar"] > cm["mirror"], cs
    assert_lanes_match(Zm, Zs)
