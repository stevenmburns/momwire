"""`pair_reuse`'s orientation-canonical key (momwire#1969).

A segment's tangent is a convention, so a pair keyed along its stored
tangents keys apart from its mirror image (a reflection reverses a loop's
direction, and so both tangents of a pair) and, along a straight run, offset
+k from offset -k. The plans key each pair in its canonical orientation and
carry its flips in the class map's low bits; the gather applies
`CODE_SIGNS`. So:

- off is the fill, bit for bit;
- on moves Z by no more than the declared drift, in free space, over the
  PEC image and over both finite grounds, thin wire and extended kernel,
  with `pair_reuse="verify"` checking a sample of every band against the
  shipped kernels;
- the loops' classes fold their mirror images (about 2x fewer), and a deck
  whose wires run either way round has the same classes;
- ADVERSARIAL: an asymmetrically driven pentagon, with wires listed either
  way round, under a sign table wrong in any one entry, or a reflected key
  whose flips are recorded wrongly, moves Z by far more than the drift and
  fails the verify sample;
- the fused gather decodes the coded map to the bit of `scatter` then
  `reduce_phi_band`.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse

from momwire import _features
from momwire import _sin_pair_reuse as pr
from momwire import sinusoidal as sin_mod
from momwire._accel import acc
from momwire.sinusoidal import SinusoidalSolver

# An experimental feature (`momwire._features`, reach 1).
pytestmark = pytest.mark.reach(1)

#: 28 MHz.
LAM = 10.7
DRIFT = _features.FEATURES["pair_reuse"].drift
#: The fill's own bound, max |dZ| / max |Z|, as the 1970 gates'.
MATRIX = 3e-12
GROUNDS = {
    "free": {},
    "pec": dict(ground_z=0.0),
    "refl-coef": dict(ground_z=0.0, ground_eps=(13.0, 0.005)),
    "sommerfeld": dict(
        ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld"
    ),
}


def _polygon(n=5, segs=15, h=8.0, reverse=(), feeds=None):
    """A regular n-gon loop of perimeter 1.05 lambda at height h, one wire
    per side, the sides in `reverse` listed end to start. Fed off every
    symmetry axis: 0.3 of side 0 at 1 V and 0.55 of side 2 at 0.5j V, so
    no mirror of the loop maps the drive to itself."""
    side = 1.05 * LAM / n
    r = side / (2.0 * np.sin(np.pi / n))
    ang = 2.0 * np.pi * np.arange(n + 1) / n + 0.1
    v = np.stack([r * np.cos(ang), r * np.sin(ang), np.full(n + 1, h)], axis=1)
    wires = []
    for i in range(n):
        w = np.array([v[i], v[i + 1]])
        wires.append(w[::-1].copy() if i in reverse else w)

    def arc(i, frac):
        return side * (1.0 - frac if i in reverse else frac)

    if feeds is None:
        feeds = [(0, arc(0, 0.3), 1.0 + 0j), (2, arc(2, 0.55), 0.5j)]
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[segs]] * n,
        feeds=feeds,
        wavelength=LAM,
        wire_radius=1e-3,
    )


#: The adversarial deck: sides 1 and 3 reversed, so the loop holds pairs
#: whose tangents agree with the loop's, pairs with one reversed and pairs
#: with both, every code.
REVERSED = (1, 3)


def _fill(deck, pair_reuse=None, **kw):
    s = SinusoidalSolver(**deck, **kw, pair_reuse=pair_reuse)
    geom = s._build_geometry()
    with s._operating_medium(geom) as medium:
        G, _ = s._assemble_Z(geom, s.k, s._medium_eta(medium))
    return G, s


def _zin(deck, pair_reuse=None, **kw):
    s = SinusoidalSolver(**deck, **kw, pair_reuse=pair_reuse)
    return np.asarray(s.compute_impedance()[0]).reshape(-1), s


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("MOMWIRE_X_PAIR_REUSE", raising=False)
    pr.clear_shared_plans()
    yield
    pr.clear_shared_plans()


# ---------------------------------------------------------------- the key


def test_the_four_orientations_key_alike():
    """Every orientation of a pair (random key columns, zeros included, so
    each tie rule runs) keys to one canonical row, and its code is the
    flips that take it there."""
    rng = np.random.default_rng(1969)
    R, N = 7, 40
    n_obs = 3
    src = rng.integers(0, 4, N)
    src_rev = rng.integers(0, 4, N)
    obs = rng.integers(0, n_obs, R)
    obs_rev = rng.integers(0, n_obs, R)

    def cols_of(a, b, t, fs, fo):
        so = src_rev if fs else src
        oo = obs_rev if fo else obs
        e = (so[None, :] * n_obs + oo[:, None]).reshape(-1) + (5 << 40)
        sa = -1 if fs else 1
        so_ = -1 if fo else 1
        return [e, a * sa, rng_hs.copy(), b * so_, t * sa * so_]

    base = [rng.integers(-2, 3, R * N) for _ in range(3)]
    rng_hs = rng.integers(0, 5, R * N)
    canon = None
    for fs in (False, True):
        for fo in (False, True):
            cols = cols_of(*base, fs, fo)
            # The deltas a reversal moves the packed types by, from this
            # orientation's own types (a reversal of a reversed one is back).
            so = src_rev if fs else src
            sx = src if fs else src_rev
            oo = obs_rev if fo else obs
            ox = obs if fo else obs_rev
            ds = (sx - so) * n_obs
            do = ox - oo
            code = pr._canonical_codes(cols, 1, 3, 4, None, N, ds=ds, do=do)
            got = np.stack(cols, axis=1)
            if canon is None:
                canon, code0, distinct = got, code, None
            assert np.array_equal(got, canon), (fs, fo)
            # Where the four orientations' keys differ, the code is the
            # base's composed with this orientation's flips.
            if distinct is None:
                imgs = [
                    np.stack(cols_of(*base, a, b), axis=1)
                    for a in (False, True)
                    for b in (False, True)
                ]
                distinct = np.array(
                    [len({x[i].tobytes() for x in imgs}) == 4 for i in range(R * N)]
                )
                assert distinct.any() and not distinct.all()
            flips = int(fs) | int(fo) << 1
            assert np.array_equal((code ^ flips)[distinct], code0[distinct])


def test_a_near_pair_keeps_code_zero():
    cols = [np.array([pr._NEAR_SENTINEL, 7], np.int64)] + [
        np.array([0, -3], np.int64) for _ in range(4)
    ]
    near = np.array([True, False])
    code = pr._canonical_codes(cols, 1, 3, 4, near, 2, ds=np.array([5, 5]))
    assert code[0] == 0 and cols[0][0] == pr._NEAR_SENTINEL
    assert code[1] == 3


# ------------------------------------------------------ the fill, on and off


@pytest.mark.parametrize("ground", sorted(GROUNDS))
def test_off_is_the_fill_bit_for_bit(ground):
    deck = _polygon(reverse=REVERSED)
    base, _ = _fill(deck, **GROUNDS[ground])
    off, s = _fill(deck, pair_reuse=False, **GROUNDS[ground])
    assert off.tobytes() == base.tobytes()
    assert s.pair_reuse_census == {}


@pytest.mark.parametrize("ek", [False, True])
@pytest.mark.parametrize("reverse", [(), REVERSED], ids=["loop", "reversed"])
@pytest.mark.parametrize("ground", sorted(GROUNDS))
def test_on_matches_off(ground, reverse, ek):
    deck = _polygon(reverse=reverse)
    kw = dict(GROUNDS[ground], extended_kernel=ek)
    base, _ = _fill(deck, **kw)
    on, s = _fill(deck, pair_reuse="verify", **kw)
    assert s.pair_reuse_declined is None
    plans = [p for _, p in s._sin_reuse_plans.values()]
    assert all(p.code_bits == pr.CODE_BITS for p in plans)
    assert _rel(on, base) <= MATRIX
    assert 0.0 < s.pair_reuse_verify_max <= pr.VERIFY_TOL
    z_off, _ = _zin(deck, **kw)
    z_on, _ = _zin(deck, pair_reuse=True, **kw)
    assert np.max(np.abs(z_on - z_off) / np.abs(z_off)) <= DRIFT


@pytest.mark.parametrize("ground", ["free", "sommerfeld"])
def test_mirror_images_share_a_class(ground, monkeypatch):
    """The pentagon's C5 alone allows ~5x; its mirrors double it. Measured
    (this deck, 75 segments): see the asserts' comments."""
    deck = _polygon()
    _, s = _fill(deck, pair_reuse=True, **GROUNDS[ground])
    canon = dict(s.pair_reuse_census)
    pr.clear_shared_plans()
    monkeypatch.setattr(pr, "CANONICAL", False)
    _, s = _fill(deck, pair_reuse=True, **GROUNDS[ground])
    stored = dict(s.pair_reuse_census)
    for block, (n_pairs, n_cls, n_near) in canon.items():
        p0, c0, nn0 = stored[block]
        assert (n_pairs, n_near) == (p0, nn0)
        assert n_cls * 1.8 < c0, (block, n_cls, c0)


def test_the_classes_do_not_depend_on_which_way_a_wire_runs():
    _, s = _fill(_polygon(), pair_reuse=True)
    loop = s.pair_reuse_census["direct"]
    _, s = _fill(_polygon(reverse=REVERSED), pair_reuse=True)
    rev = s.pair_reuse_census["direct"]
    # The reversed wires' centres are computed from the other end, so a
    # few pairs straddle a quantum.
    assert loop[0] == rev[0] and abs(loop[1] - rev[1]) <= 0.02 * loop[1]


def test_a_straight_run_folds_plus_and_minus_offsets(monkeypatch):
    """A dipole's pairs at offset +k and -k are the both-reversed relation:
    one class, the sin shape negated."""
    deck = dict(
        wires=[np.array([[0.0, -2.5, 8.0], [0.0, 2.5, 8.0]])],
        n_per_edge_per_wire=[[81]],
        feed_wire_index=0,
        wavelength=LAM,
        wire_radius=1e-3,
    )
    base, _ = _fill(deck)
    on, s = _fill(deck, pair_reuse="verify")
    n_pairs, canon, _ = s.pair_reuse_census["direct"]
    assert _rel(on, base) <= MATRIX
    plan = s._sin_reuse_plans["direct"][1]
    m, k = 40, 10
    assert plan.cls[m, m + k] >> 2 == plan.cls[m, m - k] >> 2
    assert {plan.cls[m, m + k] & 3, plan.cls[m, m - k] & 3} == {0, 3}
    pr.clear_shared_plans()
    monkeypatch.setattr(pr, "CANONICAL", False)
    _, s = _fill(deck, pair_reuse=True)
    stored = s.pair_reuse_census["direct"][1]
    assert canon * 1.8 < stored


# ------------------------------------------------------------ adversarial

#: Each of the nine entries the flips set, negated alone.
WRONG = [(c, sh) for c in (1, 2, 3) for sh in range(3)]


@pytest.mark.parametrize("ground", ["free", "pec"])
@pytest.mark.parametrize("entry", WRONG, ids=[f"code{c}-shape{sh}" for c, sh in WRONG])
def test_a_wrong_sign_is_caught(entry, ground, monkeypatch):
    """The asymmetrically driven, mixed-orientation pentagon under a sign
    table wrong in one entry: Z moves by far more than the drift, the fused
    gather and the band path alike, and the verify sample fails. The right
    table (`test_on_matches_off`) holds both to the rounding."""
    deck = _polygon(reverse=REVERSED)
    kw = GROUNDS[ground]
    base, _ = _fill(deck, **kw)
    z_off, _ = _zin(deck, **kw)
    bad = pr.CODE_SIGNS.copy()
    bad[entry] *= -1.0
    monkeypatch.setattr(pr, "CODE_SIGNS", bad)
    fused, s = _fill(deck, pair_reuse=True, **kw)
    # Measured (Skylake), both grounds: max |dZ| / max |Z| 3.8e-5 (a sin
    # entry) to 0.98, and the feed impedance 2.4e-3 to 49x; the right
    # table 2.3e-14 and 9e-13.
    assert _rel(fused, base) > 1e-6
    z_bad, _ = _zin(deck, pair_reuse=True, **kw)
    assert np.max(np.abs(z_bad - z_off) / np.abs(z_off)) > 1e3 * DRIFT
    pr.clear_shared_plans()
    with pytest.raises(pr.PairReuseMismatch):
        _fill(deck, pair_reuse="verify", **kw)


#: Code maps the reflected plan might record instead of the true flips.
WRONG_GROUND_CODES = {
    "swapped": np.array([0, 2, 1, 3], np.int8),
    "source-lost": np.array([0, 0, 2, 2], np.int8),
    "observer-lost": np.array([0, 1, 0, 1], np.int8),
}


@pytest.mark.parametrize("ground", ["refl-coef", "sommerfeld"])
@pytest.mark.parametrize("wrong", sorted(WRONG_GROUND_CODES))
def test_a_wrong_reflected_code_is_caught(wrong, ground, monkeypatch):
    """The reflected block alone mis-recording its flips (the direct block's
    are right): its classes are the same, its signs wrong, and Z and the
    verify sample both see it."""
    deck = _polygon(reverse=REVERSED)
    kw = GROUNDS[ground]
    base, _ = _fill(deck, **kw)
    real = pr._canonical_codes
    perm = WRONG_GROUND_CODES[wrong]

    def lying(cols, *a, **k):
        code = real(cols, *a, **k)
        return perm[code] if len(cols) == 6 else code

    monkeypatch.setattr(pr, "_canonical_codes", lying)
    bad, s = _fill(deck, pair_reuse=True, **kw)
    assert set(s.pair_reuse_census) == {"direct", "reflected"}
    # Measured (Skylake): "swapped" moves only the sin shape of the
    # one-reversed pairs, whose image is 16 m off, 3.0e-9 of max |Z|; the
    # two "lost" maps 4.6e-6. The real codes hold 4e-15 to 2e-14.
    assert _rel(bad, base) > 100 * MATRIX
    pr.clear_shared_plans()
    with pytest.raises(pr.PairReuseMismatch):
        _fill(deck, pair_reuse="verify", **kw)


# ------------------------------------------------------- the fused gather

needs_coded = pytest.mark.skipif(
    acc is None or not getattr(acc, "pair_reuse_reduce_1969", False),
    reason="accelerator None or without pair_reuse_reduce_1969",
)


@needs_coded
@pytest.mark.parametrize("ground", sorted(GROUNDS))
def test_the_fused_gather_decodes_to_the_bit(ground, monkeypatch):
    deck = _polygon(reverse=REVERSED)
    calls = []
    real = pr.reduce_band

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(pr, "reduce_band", counting)
    fused, s = _fill(deck, pair_reuse=True, **GROUNDS[ground])
    assert calls, "the fused path did not run"
    monkeypatch.setattr(pr, "_HAVE_REDUCE", False)
    calls.clear()
    banded, _ = _fill(deck, pair_reuse=True, **GROUNDS[ground])
    assert calls == []
    assert fused.tobytes() == banded.tobytes()


@needs_coded
def test_the_kernel_decodes_against_numpy_on_a_synthetic_band():
    rng = np.random.default_rng(1969)
    rows, n_src, n_cls = 37, 90, 400

    def cplx(*shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    def block():
        c = rng.integers(-3, n_cls + 3, size=(rows, n_src))
        code = rng.integers(0, 4, size=(rows, n_src))
        cls = ((c << 2) | code).astype(np.int32)
        rep = cplx(n_cls, 3)
        near = np.unique(rng.integers(0, rows * n_src, size=300)).astype(np.int64)
        return cls, rep, near, cplx(near.size, 3)

    class P:
        code_bits = pr.CODE_BITS

    def numpy_phi(cls, rep, near, vals, signs):
        P.cls = cls
        P.n_near = 0
        old = pr.CODE_SIGNS
        pr.CODE_SIGNS = signs
        try:
            phi = list(pr.scatter(rep, P, (0, rows)))
        finally:
            pr.CODE_SIGNS = old
        for s in range(3):
            phi[s].reshape(-1)[near] = vals[:, s]
        return phi

    Ms = []
    for _ in range(3):
        n_idx = rng.integers(0, n_src, size=3 * n_src)
        j_idx = rng.integers(0, n_src, size=3 * n_src)
        Ms.append(
            scipy.sparse.csc_matrix(
                (cplx(3 * n_src), (n_idx, j_idx)), shape=(n_src, n_src)
            )
        )
    m_args = []
    for M in Ms:
        m_args += [M.indptr, M.indices, M.data]
    d, im = block(), block()
    signs_d = pr.CODE_SIGNS
    signs_i = -pr.CODE_SIGNS[::-1].copy()  # any table: the kernel reads it
    for image in (None, im):
        phi = numpy_phi(*d, signs_d)
        if image is not None:
            phi_i = numpy_phi(*image, signs_i)
            for s in range(3):
                phi[s] -= phi_i[s]
        want = np.empty((rows, n_src), dtype=np.complex128)
        sin_mod._reduce_phi_band(phi, Ms, want)
        img = None if image is None else (*image, signs_i)
        for nt in (1, 3):
            got = np.full((rows, n_src), np.nan, dtype=np.complex128)
            acc.pair_reuse_reduce_band(
                got, *d, img, *m_args, n_threads=nt, signs=signs_d
            )
            assert got.tobytes() == want.tobytes(), (image is None, nt)
