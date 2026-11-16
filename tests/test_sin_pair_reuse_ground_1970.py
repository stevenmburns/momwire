"""`pair_reuse` over a finite ground (momwire#1970).

The reflected term (Sommerfeld's `C2 Φ_img - S`, or the Fresnel-weighted
image) is invariant only under the isometries that keep the ground plane,
so it takes its own classes, keyed on the height sum and the horizontal
configuration (`_sin_pair_reuse.build_ground_plan`). So:

- off is the fill, bit for bit, over both grounds;
- on moves Z by no more than the declared drift, and `pair_reuse="verify"`
  checks a sample of every band against the direct evaluators;
- the fused gather is the band path, bit for bit, with the reflected block;
- the key carries what the reflected term reads: a height-blind key, or the
  free-space (3-D) key, merges pairs whose reflected terms differ, and the
  verify sample catches it, where the real key does not merge them.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _features
from momwire import _sin_pair_reuse as pr
from momwire._accel import acc
from momwire.sinusoidal import SinusoidalSolver

# An experimental feature (`momwire._features`, reach 1).
pytestmark = pytest.mark.reach(1)

LAM = 21.1
DRIFT = _features.FEATURES["pair_reuse"].drift
SOMM = dict(ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld")
REFL = dict(ground_z=0.0, ground_eps=(13.0, 0.005))
GROUNDS = {"sommerfeld": SOMM, "refl-coef": REFL}


def _yagi(h=8.0, D=6, segs=21):
    """Reflector, driver (fed) and D identical directors at height h, plus a
    bent, tilted wire so not every pair is horizontal."""
    Ld = 0.47 * LAM
    wires = [
        np.array([[0.2 * LAM * i, -L / 2, h], [0.2 * LAM * i, L / 2, h]])
        for i, L in enumerate([1.05 * Ld, Ld] + [0.95 * Ld] * D)
    ]
    wires.append(np.array([[-2.0, 0.0, h - 3.0], [-2.0, 3.0, h - 1.0], [-2.0, 5.0, h]]))
    n = [[segs] for _ in wires[:-1]] + [[9, 7]]
    return dict(
        wires=wires,
        n_per_edge_per_wire=n,
        feed_wire_index=1,
        wavelength=LAM,
        wire_radius=1e-3,
    )


def _two_heights(z1=5.0, z2=7.0, segs=21):
    """Two congruent horizontal dipoles, the second 3 m along x and at
    another height, both image distances in one octave so a height-blind key
    puts their pairs in one class."""
    L = 0.47 * LAM
    wires = [
        np.array([[0.0, -L / 2, z1], [0.0, L / 2, z1]]),
        np.array([[3.0, -L / 2, z2], [3.0, L / 2, z2]]),
    ]
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[segs], [segs]],
        feed_wire_index=0,
        wavelength=LAM,
        wire_radius=1e-3,
    )


def _tilted(z=6.0, tilt_deg=40.0, segs=21):
    """A horizontal dipole and a congruent one tilted about its centre's
    horizontal axis: the same free-space pairs, other reflected ones."""
    L = 0.47 * LAM
    t = np.radians(tilt_deg)
    half = 0.5 * L * np.array([0.0, np.cos(t), np.sin(t)])
    c2 = np.array([6.0, 0.0, z])
    wires = [
        np.array([[0.0, -L / 2, z], [0.0, L / 2, z]]),
        np.array([c2 - half, c2 + half]),
    ]
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[segs], [segs]],
        feed_wire_index=0,
        wavelength=LAM,
        wire_radius=1e-3,
    )


def _z(deck, pair_reuse=None, **kw):
    s = SinusoidalSolver(**deck, **kw, pair_reuse=pair_reuse)
    geom = s._build_geometry()
    G, _ = s._assemble_Z(geom, s.k)
    return G, s


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("MOMWIRE_X_PAIR_REUSE", raising=False)
    pr.clear_shared_plans()
    yield
    pr.clear_shared_plans()


@pytest.mark.parametrize("ground", sorted(GROUNDS))
def test_off_is_the_fill_bit_for_bit(ground, monkeypatch):
    base, _ = _z(_yagi(), **GROUNDS[ground])
    monkeypatch.setenv("MOMWIRE_X_PAIR_REUSE", "1")
    off, s = _z(_yagi(), pair_reuse=False, **GROUNDS[ground])
    assert off.tobytes() == base.tobytes()
    assert s.pair_reuse_census == {}


@pytest.mark.parametrize("ek", [False, True])
@pytest.mark.parametrize("ground", sorted(GROUNDS))
def test_on_matches_off(ground, ek):
    kw = dict(GROUNDS[ground], extended_kernel=ek)
    base, _ = _z(_yagi(), **kw)
    on, s = _z(_yagi(), pair_reuse="verify", **kw)
    assert s.pair_reuse_declined is None
    assert set(s.pair_reuse_census) == {"direct", "reflected"}
    n_pairs, n_cls, _near = s.pair_reuse_census["reflected"]
    # The six directors repeat; the reflected classes are about as coarse
    # as the direct ones on a deck that is mostly at one height.
    assert n_cls < n_pairs / 2
    assert _rel(on, base) <= DRIFT
    assert 0.0 <= s.pair_reuse_verify_max <= pr.VERIFY_TOL


@pytest.mark.skipif(
    acc is None or not getattr(acc, "pair_reuse_reduce_1916", False),
    reason="accelerator None or without pair_reuse_reduce_1916",
)
@pytest.mark.parametrize("ground", sorted(GROUNDS))
@pytest.mark.parametrize("budget", [None, 1])
def test_the_fused_gather_is_the_band_path_bit_for_bit(ground, budget, monkeypatch):
    kw = dict(GROUNDS[ground])
    if budget is not None:
        kw["swept_mem_mb"] = budget
    calls = []
    real = pr.reduce_band

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(pr, "reduce_band", counting)
    fused, s = _z(_yagi(), pair_reuse=True, **kw)
    assert calls, "the fused path did not run"
    assert set(s.pair_reuse_census) == {"direct", "reflected"}
    monkeypatch.setattr(pr, "_HAVE_REDUCE", False)
    calls.clear()
    banded, _ = _z(_yagi(), pair_reuse=True, **kw)
    assert calls == []
    assert fused.tobytes() == banded.tobytes()


def test_a_sweep_keeps_one_census():
    deck = _yagi()
    ks = 2 * np.pi / LAM * np.array([0.97, 1.03])
    z_off = SinusoidalSolver(**deck, **SOMM).compute_impedance_swept(ks)
    s_on = SinusoidalSolver(**deck, **SOMM, pair_reuse=True)
    z_on = s_on.compute_impedance_swept(ks)
    # Neither block's classes hold k or the ground's constants.
    assert sorted(s_on._sin_reuse_plans) == ["direct", "reflected"]
    assert np.max(np.abs(z_on - z_off) / np.abs(z_off)) <= DRIFT


def _keys(oc, ot, sc, st, gz=0.0, H=0.05, a=1e-3):
    sc_ = pr.ground_pair_scalars(oc, ot, sc, st, gz)
    cols, _ = pr._ground_band_keys(
        sc_, np.array([H]), np.array([a]), np.zeros((1, 1), np.int64)
    )
    return [int(c[0]) for c in cols]


def test_the_key_is_the_ground_group_s():
    rng = np.random.default_rng(1970)
    src_c = np.array([[0.4, -0.2, 3.0]])
    src_t = np.array([[0.6, 0.0, 0.8]])
    obs_c = np.array([[1.1, 0.5, 2.2]])
    obs_t = np.array([[0.0, 0.8, 0.6]])
    base = _keys(obs_c, obs_t, src_c, src_t)

    def horizontal(theta, mirror):
        c, s = np.cos(theta), np.sin(theta)
        q = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        if mirror:
            q = q @ np.diag([1.0, -1.0, 1.0])
        return q

    for mirror in (False, True):
        q = horizontal(rng.uniform(0, 2 * np.pi), mirror)
        shift = np.array([*rng.normal(size=2), 0.0])
        moved = _keys(
            obs_c @ q.T + shift, obs_t @ q.T, src_c @ q.T + shift, src_t @ q.T
        )
        # A ground-preserving isometry moves each column by its rounding.
        assert all(abs(a - b) <= 1 for a, b in zip(base, moved))
    # Raised together: the height sum moves.
    up = np.array([[0.0, 0.0, 0.5]])
    assert _keys(obs_c + up, obs_t, src_c + up, src_t) != base
    # Observer up and source down by as much: the image geometry is the same.
    same = _keys(obs_c + up, obs_t, src_c - up, src_t)
    assert all(abs(a - b) <= 1 for a, b in zip(base, same))
    # A rotation about a HORIZONTAL axis keeps the free-space key, not this.
    q = np.array([[1.0, 0.0, 0.0], [0.0, 0.6, -0.8], [0.0, 0.8, 0.6]])
    tilted = _keys(obs_c @ q.T, obs_t @ q.T, src_c @ q.T, src_t @ q.T)
    assert tilted != base


def _height_blind(monkeypatch):
    real = pr._ground_band_keys

    def blind(scalars, H, a_obs, type_id):
        cols, near = real(scalars, H, a_obs, type_id)
        cols[1][:] = 0  # the height sum
        return cols, near

    monkeypatch.setattr(pr, "_ground_band_keys", blind)


def _free_space_key(monkeypatch):
    """The direct block's 3-D key standing in for the ground one: classes
    from `build_plan` over the real sources, the segment types without the
    vertical tangent."""
    monkeypatch.setattr(pr, "ground_segment_types", lambda s, o, t: (s, o))

    def as_ground(obs_c, obs_t, src_c, src_t, gz, seg_h, a_obs, st, ot, rows):
        plan = pr.build_plan(obs_c, obs_t, src_c, src_t, seg_h, a_obs, st, ot, rows)
        N = src_c.shape[0]
        _, first, inv = np.unique(
            plan.cls.reshape(-1), return_index=True, return_inverse=True
        )
        order = np.argsort(first)
        renum = np.empty_like(order)
        renum[order] = np.arange(order.size)
        rep_flat = first[order].astype(np.int64)
        return pr.GroundPlan(
            cls=renum[inv].reshape(plan.cls.shape).astype(np.int32),
            rep_flat=rep_flat,
            rep_blocks=pr.near_blocks(rep_flat, N, a_obs),
            near_flat=plan.near_flat,
            near_blocks=plan.near_blocks,
        )

    monkeypatch.setattr(pr, "build_ground_plan", as_ground)


@pytest.mark.parametrize("ground", sorted(GROUNDS))
@pytest.mark.parametrize(
    "deck, wrong",
    [(_two_heights, _height_blind), (_tilted, _free_space_key)],
    ids=["two-heights/height-blind", "tilted/free-space-key"],
)
def test_a_key_blind_to_the_ground_is_caught(ground, deck, wrong, monkeypatch):
    """Each deck holds pairs congruent under a motion that is not the
    ground's: a vertical translation, a rotation about a horizontal axis.
    The real key keeps them apart; the wrong key merges them, moves Z by
    far more than the drift, and fails the verify sample."""
    base, _ = _z(deck(), **GROUNDS[ground])
    on, s = _z(deck(), pair_reuse="verify", **GROUNDS[ground])
    n_pairs, good_cls, _ = s.pair_reuse_census["reflected"]
    assert _rel(on, base) <= DRIFT
    pr.clear_shared_plans()
    wrong(monkeypatch)
    bad, s = _z(deck(), pair_reuse=True, **GROUNDS[ground])
    _, bad_cls, _ = s.pair_reuse_census["reflected"]
    assert bad_cls < good_cls
    # Measured (Skylake): 2e-5 of max |Z| on all four, the feed impedance
    # moved 17 % (heights) and 1.2 % (tilt); the real key 1e-14 and 2e-13.
    assert _rel(bad, base) > 1e-6
    pr.clear_shared_plans()
    with pytest.raises(pr.PairReuseMismatch):
        _z(deck(), pair_reuse="verify", **GROUNDS[ground])


@pytest.mark.parametrize("ground", sorted(GROUNDS))
def test_the_two_height_classes_are_the_heights(ground):
    """On the two-height deck the direct block shares each dipole's pairs
    between the dipoles and the reflected block does not."""
    _, s = _z(_two_heights(), pair_reuse=True, **GROUNDS[ground])
    direct = s._sin_reuse_plans["direct"][1]
    refl = s._sin_reuse_plans["reflected"][1]
    n = 21
    # Pair (2, 5) of dipole 1 and its twin on dipole 2.
    assert direct.cls[2, 5] == direct.cls[n + 2, n + 5]
    assert refl.cls[2, 5] != refl.cls[n + 2, n + 5]
    # Along one dipole the reflected term repeats by offset.
    assert refl.cls[2, 5] == refl.cls[7, 10]
