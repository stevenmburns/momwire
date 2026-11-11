"""`pair_reuse` on the point-matched sinusoidal fill (momwire#1911).

The fill's field tensors Φ (one value per shape: constant, sin, cos) at an
(observer point, source segment) pair depend on the pair's geometry only
through four scalars, measured in the SOURCE's frame (origin at its centre,
z along its natural tangent):

* ``z``   -- the observer centre's coordinate along the source axis;
* ``rho`` -- its distance from that axis;
* ``td``  -- the cosine between the observer and source tangents;
* ``rdt`` -- the radial vector dotted with the observer tangent;

and, beyond those, on the source's half-length, the observer's radius and,
under the extended kernel, the source's radius and its two end codes. A
straight source's current is axisymmetric, so its field has no azimuthal
part and the projection onto the observer's tangent needs nothing more.
Both C++ kernels (`sinusoidal_field_tensor`, `_ek`) read exactly these
quantities (`_sinusoidal_core.cpp`, stage A), so two pairs agreeing on all
of them have the same Φ: every rigid motion, mirrored pairs included.

The KEY quantises the four scalars with a step of 2^-B times the pair's own
nearest singular distance (the observer centre to the source's centre or
either end, floored at the observer radius), rounded down to a power of two
so the step is a function of the value alone. A member therefore differs
from its class representative by at most 2^-(B+1) relative to the scale at
which Φ varies, and takes the representative's Φ. Pairs that straddle a
quantum land in neighbouring classes, which costs compression, never
accuracy. No pair is excluded: the self term sits at z = rho = 0 exactly
(the kernel's own difference of equal centres), and every other pair is
held to the same relative step.

The representative is evaluated by the shipped kernel in canonical form:
one source at the origin along z, observer centre (rho, 0, z) with tangent
(rdt/rho, ., td). Classes are found once per geometry (the key holds no k)
and Φ per class once per frequency, then gathered into each observer band,
so the reduction (`_reduce_phi_band`) is unchanged. Memory: an (N_obs,
N_src) int32 class map, a quarter of Z, plus the class table; no Φ-sized
copy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Relative resolution of the key, in bits.
KEY_BITS = 40

#: `pair_reuse="verify"`: per-pair agreement bound, |Φ_reused - Φ_direct|
#: over the pair's largest |Φ_direct| among the three shapes.
VERIFY_TOL = 1e-10

#: `pair_reuse="verify"`: pairs checked per observer band.
VERIFY_SAMPLES = 512

_M1 = np.uint64(0x9E3779B97F4A7C15)
_M2 = np.uint64(0xBF58476D1CE4E5B9)


class PairReuseMismatch(AssertionError):
    """`pair_reuse="verify"` found a reused Φ that the direct kernel does not
    reproduce."""


class PairReuseCollision(RuntimeError):
    """Two different keys hashed alike (exact check). The fill falls back to
    the direct kernel."""


@dataclass
class Plan:
    """Which class each (observer, source) pair belongs to, and how to
    evaluate each class once."""

    cls: np.ndarray  # (N_obs, N_src) int32 class of each pair
    canon: np.ndarray  # (n_cls, 4) float64 rho, z, rdt, td of the representative
    group: np.ndarray  # (n_cls,) int64 type group of the class
    group_src: np.ndarray  # (n_groups,) int64 source segment typifying the group
    group_obs: np.ndarray  # (n_groups,) int64 observer segment typifying it

    @property
    def n_pairs(self) -> int:
        return int(self.cls.size)

    @property
    def n_classes(self) -> int:
        return int(self.canon.shape[0])


def _types(cols):
    """Dense ids for the distinct rows of a few per-segment columns."""
    stacked = np.stack([np.asarray(c) for c in cols], axis=1)
    _, ids = np.unique(stacked, axis=0, return_inverse=True)
    return np.asarray(ids, dtype=np.int64).reshape(-1)


def _quantise_length(h):
    """(exponent, mantissa) of a positive length at KEY_BITS relative bits."""
    m, e = np.frexp(h)
    return e.astype(np.int64), np.rint(np.ldexp(m, KEY_BITS)).astype(np.int64)


def segment_types(seg_h, a_obs, src_a=None, ind1=None, ind2=None):
    """Per-segment source-type and observer-type ids.

    The source type holds what Φ reads of the source besides its pose: its
    half-length (quantised like the key) and, under the extended kernel, its
    radius and end codes (exact). The observer type is its radius (exact),
    the scalar each kernel call takes."""
    e, q = _quantise_length(np.asarray(seg_h, dtype=np.float64))
    cols = [e, q]
    if src_a is not None:
        cols += [
            np.asarray(src_a, dtype=np.float64).view(np.int64),
            np.asarray(ind1, dtype=np.int64),
            np.asarray(ind2, dtype=np.int64),
        ]
    src_type = _types(cols)
    obs_type = _types([np.asarray(a_obs, dtype=np.float64).view(np.int64)])
    return src_type, obs_type


def pair_scalars(obs_c, obs_t, src_c, src_t):
    """z, rho, td, rdt for every (observer, source) pair, (R, N) each, in
    the kernels' own order of operations."""
    rvx = obs_c[:, None, 0] - src_c[None, :, 0]
    rvy = obs_c[:, None, 1] - src_c[None, :, 1]
    rvz = obs_c[:, None, 2] - src_c[None, :, 2]
    tnx, tny, tnz = src_t[None, :, 0], src_t[None, :, 1], src_t[None, :, 2]
    z = rvx * tnx + rvy * tny + rvz * tnz
    rvx -= z * tnx
    rvy -= z * tny
    rvz -= z * tnz
    rho = np.sqrt(rvx * rvx + rvy * rvy + rvz * rvz)
    tmx, tmy, tmz = obs_t[:, None, 0], obs_t[:, None, 1], obs_t[:, None, 2]
    td = tmx * tnx + tmy * tny + tmz * tnz
    rdt = rvx * tmx + rvy * tmy + rvz * tmz
    return z, rho, td, rdt


def _band_keys(z, rho, td, rdt, H, a_obs, type_id):
    """The quantised key columns of one band (int64, (R*N,) each)."""
    # Nearest singular distance: to the source's centre or to its nearer
    # end, never below the observer radius.
    d_c = np.hypot(z, rho)
    d_end = np.hypot(np.abs(z) - H[None, :], rho)
    scale = np.minimum(d_c, d_end)
    del d_c, d_end
    np.maximum(scale, a_obs[:, None], out=scale)
    _, e = np.frexp(scale)
    del scale
    e = e.astype(np.int64)
    step_inv = np.ldexp(1.0, KEY_BITS - e)
    zq = np.rint(z * step_inv).astype(np.int64)
    rq = np.rint(rho * step_inv).astype(np.int64)
    dq = np.rint(rdt * step_inv).astype(np.int64)
    del step_inv
    tq = np.rint(np.ldexp(td, KEY_BITS)).astype(np.int64)
    return [c.reshape(-1) for c in (e, zq, rq, dq, tq, type_id)]


def _hash(cols):
    h = np.full(cols[0].shape, 0x243F6A8885A308D3, dtype=np.uint64)
    for c in cols:
        h ^= c.view(np.uint64)
        h *= _M1
        h ^= h >> np.uint64(31)
        h *= _M2
        h ^= h >> np.uint64(29)
    return h


def build_plan(obs_c, obs_t, src_c, src_t, seg_h, a_obs, src_type, obs_type, rows):
    """The class map over every (observer, source) pair, found in observer
    bands of `rows` rows (the transient is O(rows * N_src))."""
    obs_c = np.asarray(obs_c, dtype=np.float64)
    obs_t = np.asarray(obs_t, dtype=np.float64)
    src_c = np.asarray(src_c, dtype=np.float64)
    src_t = np.asarray(src_t, dtype=np.float64)
    a_obs = np.asarray(a_obs, dtype=np.float64)
    H = 0.5 * np.asarray(seg_h, dtype=np.float64)
    M, N = obs_c.shape[0], src_c.shape[0]
    n_obs_types = int(obs_type.max()) + 1
    cls = np.empty((M, N), dtype=np.int32)
    band_hash, band_keys, band_canon, band_pair = [], [], [], []
    rows = max(1, int(rows))
    for i0 in range(0, M, rows):
        i1 = min(M, i0 + rows)
        z, rho, td, rdt = pair_scalars(obs_c[i0:i1], obs_t[i0:i1], src_c, src_t)
        type_id = src_type[None, :] * n_obs_types + obs_type[i0:i1, None]
        cols = _band_keys(z, rho, td, rdt, H, a_obs[i0:i1], type_id)
        h = _hash(cols)
        u, first, inv = np.unique(h, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        for c in cols:  # exact: every member's key is its first's
            if not np.array_equal(c, c[first][inv]):
                raise PairReuseCollision("pair key hash collision within a band")
        cls[i0:i1] = inv.reshape(i1 - i0, N)
        band_hash.append(u)
        band_keys.append(np.stack([c[first] for c in cols], axis=1))
        band_canon.append(
            np.stack(
                [
                    rho.reshape(-1)[first],
                    z.reshape(-1)[first],
                    rdt.reshape(-1)[first],
                    td.reshape(-1)[first],
                ],
                axis=1,
            )
        )
        r_first, n_first = np.divmod(first, N)
        band_pair.append(np.stack([r_first + i0, n_first], axis=1))
        del z, rho, td, rdt, cols, h, inv, type_id
    all_hash = np.concatenate(band_hash)
    all_keys = np.concatenate(band_keys)
    gu, gfirst, ginv = np.unique(all_hash, return_index=True, return_inverse=True)
    ginv = ginv.reshape(-1)
    if not np.array_equal(all_keys, all_keys[gfirst][ginv]):
        raise PairReuseCollision("pair key hash collision across bands")
    off = 0
    for b, i0 in enumerate(range(0, M, rows)):
        i1 = min(M, i0 + rows)
        n_b = band_hash[b].shape[0]
        remap = ginv[off : off + n_b].astype(np.int32)
        cls[i0:i1] = remap[cls[i0:i1]]
        off += n_b
    canon = np.concatenate(band_canon)[gfirst]
    pair = np.concatenate(band_pair)[gfirst]
    type_of = src_type[pair[:, 1]] * n_obs_types + obs_type[pair[:, 0]]
    tu, gidx, group = np.unique(type_of, return_index=True, return_inverse=True)
    return Plan(
        cls=cls,
        canon=canon,
        group=np.asarray(group, dtype=np.int64).reshape(-1),
        group_src=pair[gidx, 1].astype(np.int64),
        group_obs=pair[gidx, 0].astype(np.int64),
    )


def canonical_observers(canon):
    """Observer centres and tangents placing each class in the canonical
    source frame (source at the origin along +z)."""
    rho, z, rdt, td = canon[:, 0], canon[:, 1], canon[:, 2], canon[:, 3]
    n = canon.shape[0]
    c = np.zeros((n, 3))
    c[:, 0] = rho
    c[:, 2] = z
    t = np.zeros((n, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        tr = np.where(rho > 0.0, rdt / rho, 0.0)
    t[:, 0] = tr
    t[:, 1] = np.sqrt(np.maximum(0.0, 1.0 - td * td - tr * tr))
    t[:, 2] = td
    return c, t


def evaluate(plan, call):
    """Φ per class: (3, n_cls) complex. `call(obs_c, obs_t, g_src, g_obs)`
    evaluates canonical observers against the canonical source standing for
    source segment `g_src`, at the radius of observer segment `g_obs`, and
    returns the kernel's (3, R, 1) tuple."""
    out = np.empty((3, plan.n_classes), dtype=np.complex128)
    obs_c, obs_t = canonical_observers(plan.canon)
    order = np.argsort(plan.group, kind="stable")
    bounds = np.searchsorted(plan.group[order], np.arange(len(plan.group_src) + 1))
    for g in range(len(plan.group_src)):
        idx = order[bounds[g] : bounds[g + 1]]
        if idx.size == 0:
            continue
        phis = call(
            np.ascontiguousarray(obs_c[idx]),
            np.ascontiguousarray(obs_t[idx]),
            int(plan.group_src[g]),
            int(plan.group_obs[g]),
        )
        for s in range(3):
            out[s, idx] = phis[s][:, 0]
    return out


def scatter(rep, plan, rows):
    """The band's three Φ, gathered from the class values."""
    i0, i1 = rows
    c = plan.cls[i0:i1]
    return tuple(np.take(rep[s], c) for s in range(3))


def verify_band(phis, rows, n_src, direct, samples=VERIFY_SAMPLES, tol=VERIFY_TOL):
    """Check `samples` pairs of the band against the direct kernel.

    `direct(m, ns)` returns the kernel's tuple for observer row `m` against
    source segments `ns`, in the original frame. Returns the largest
    relative difference seen; raises `PairReuseMismatch` above `tol`."""
    i0, i1 = rows
    n_pairs = (i1 - i0) * n_src
    flat = np.unique(
        np.linspace(0, n_pairs - 1, min(samples, n_pairs)).astype(np.int64)
    )
    r, n = np.divmod(flat, n_src)
    worst = 0.0
    for m in np.unique(r):
        ns = n[r == m]
        d = direct(int(m) + i0, ns)
        got = [phis[s][m, ns] for s in range(3)]
        want = [np.asarray(d[s])[0] for s in range(3)]
        scale = np.maximum.reduce([np.abs(w) for w in want])
        for g, w in zip(got, want):
            rel = np.abs(g - w) / np.where(scale > 0, scale, 1.0)
            worst = max(worst, float(rel.max()))
    if worst > tol:
        raise PairReuseMismatch(
            f"pair_reuse: a reused field tensor differs from the direct kernel "
            f"by {worst:.3g} relative (bound {tol:g})"
        )
    return worst
