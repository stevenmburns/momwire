"""`pair_reuse` on the point-matched sinusoidal fill (momwire#1911).

The fill evaluates each class of congruent (observer, source) pairs' field
tensors once with the shipped kernel and gathers them into the band; the
reduction is unchanged. So:

- off (the default, or False over the variable) is the fill, bit for bit;
- on moves Z by no more than the key's relative step can, far below
  the declared drift, in free space and over the PEC image, thin-wire and
  extended kernel, across the band cut and across a frequency sweep;
- `pair_reuse="verify"` checks a sample of every band against the direct
  kernel in the original frame;
- a finite ground declines with its reason and fills as before.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _features
from momwire import _sin_pair_reuse as pr
from momwire.sinusoidal import SinusoidalSolver

# An experimental feature (`momwire._features`, reach 1).
pytestmark = pytest.mark.reach(1)

LAM = 21.1
# The feature's declared bound; each gate below also states what it measured.
DRIFT = _features.FEATURES["pair_reuse"].drift


def _yagi(D=8, h=None, segs=21):
    """Reflector, driver (fed) and D identical directors, 0.2 lambda apart,
    plus a bent wire so not every pair is parallel."""
    Ld = 0.47 * LAM
    z = 8.0 if h is None else h
    wires = []
    for i, L in enumerate([1.05 * Ld, Ld] + [0.95 * Ld] * D):
        x = 0.2 * LAM * i
        wires.append(np.array([[x, -L / 2, z], [x, L / 2, z]]))
    wires.append(np.array([[-2.0, 0.0, z - 3.0], [-2.0, 3.0, z - 1.0], [-2.0, 5.0, z]]))
    n = [[segs] for _ in wires[:-1]] + [[9, 7]]
    return dict(
        wires=wires,
        n_per_edge_per_wire=n,
        feed_wire_index=1,
        wavelength=LAM,
        wire_radius=1e-3,
    )


def _z(pair_reuse=None, **kw):
    deck = _yagi(**{k: kw.pop(k) for k in ("D", "h", "segs") if k in kw})
    s = SinusoidalSolver(**deck, **kw, pair_reuse=pair_reuse)
    geom = s._build_geometry()
    G, _ = s._assemble_Z(geom, s.k)
    return G, s


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("MOMWIRE_X_PAIR_REUSE", raising=False)


@pytest.mark.parametrize("ek", [False, True])
def test_off_is_the_fill_bit_for_bit(ek, monkeypatch):
    base, _ = _z(extended_kernel=ek)
    off, s = _z(pair_reuse=False, extended_kernel=ek)
    assert off.tobytes() == base.tobytes()
    assert s.pair_reuse_census == {}
    # An explicit False wins over the variable.
    monkeypatch.setenv("MOMWIRE_X_PAIR_REUSE", "1")
    off_env, s = _z(pair_reuse=False, extended_kernel=ek)
    assert off_env.tobytes() == base.tobytes()
    assert s.pair_reuse_census == {}


@pytest.mark.parametrize("ek", [False, True])
@pytest.mark.parametrize("h", [None, 6.0])
def test_on_matches_off(ek, h):
    kw = {"extended_kernel": ek}
    if h is not None:
        kw.update(h=h, ground_z=0.0)
    base, _ = _z(**kw)
    on, s = _z(pair_reuse="verify", **kw)
    assert s.pair_reuse_declined is None
    blocks = {"direct"} if h is None else {"direct", "image"}
    assert set(s.pair_reuse_census) == blocks
    # Only the eight directors repeat (the other wires differ in length):
    # measured about 2.3 pairs per class.
    for n_pairs, n_cls, _near in s.pair_reuse_census.values():
        assert n_cls < n_pairs / 2
    # Measured: ~1e-15 on this deck, every case.
    assert _rel(on, base) <= DRIFT
    assert 0.0 <= s.pair_reuse_verify_max <= pr.VERIFY_TOL


def test_band_cut_and_a_hint_and_the_variable(monkeypatch):
    base, _ = _z()
    # A small budget cuts the fill (and the census) into many bands.
    on, s = _z(pair_reuse=[[((2, 0), (3, 0))]], swept_mem_mb=1)
    assert s.pair_reuse_declined is None and s.pair_reuse_census
    assert _rel(on, base) <= DRIFT
    monkeypatch.setenv("MOMWIRE_X_PAIR_REUSE", "1")
    env, s = _z()
    assert s.pair_reuse_census
    assert _rel(env, base) <= DRIFT


def test_a_sweep_keeps_one_census():
    deck = _yagi()
    ks = 2 * np.pi / LAM * np.array([0.95, 1.0, 1.05])
    z_off = SinusoidalSolver(**deck).compute_impedance_swept(ks)
    s_on = SinusoidalSolver(**deck, pair_reuse=True)
    z_on = s_on.compute_impedance_swept(ks)
    # The classes hold no k: one census serves every frequency.
    assert list(s_on._sin_reuse_plans) == ["direct"]
    assert np.max(np.abs(z_on - z_off) / np.abs(z_off)) <= DRIFT


def test_a_finite_ground_declines():
    kw = dict(h=6.0, ground_z=0.0, ground_eps=complex(13.0, -5.0))
    base, _ = _z(**kw)
    on, s = _z(pair_reuse=True, **kw)
    assert s.pair_reuse_declined == "a finite ground"
    assert on.tobytes() == base.tobytes()


def test_the_key_separates_what_differs():
    # Two pairs a rigid motion apart share a key; a pair moved by more than
    # the key's step does not.
    rng = np.random.default_rng(1911)
    src_c = np.zeros((1, 3))
    src_t = np.array([[0.0, 0.0, 1.0]])
    obs_c = np.array([[0.3, 0.0, 0.7]])
    obs_t = np.array([[0.6, 0.0, 0.8]])
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    shift = rng.normal(size=3)
    moved = [obs_c @ q.T + shift, obs_t @ q.T, src_c @ q.T + shift, src_t @ q.T]
    far = [obs_c + np.array([[1e-9, 0.0, 0.0]]), obs_t, src_c, src_t]
    H = np.array([0.05])
    keys = []
    for oc, ot, sc, st in ([obs_c, obs_t, src_c, src_t], moved, far):
        z, rho, td, rdt = pr.pair_scalars(oc, ot, sc, st)
        keys.append(
            [
                int(c[0])
                for c in pr._band_keys(
                    z, rho, td, rdt, H, np.array([1e-3]), np.zeros((1, 1), np.int64)
                )[0]
            ]
        )
    # A rigid motion moves each quantised scalar by at most its rounding.
    assert all(abs(a - b) <= 1 for a, b in zip(keys[0], keys[1]))
    assert keys[0] != keys[2]
