"""The transmitted (below->above) grid blended from frequency anchors
(momwire#1847, the opt-in feature ``transmitted_anchors``).

`_sommerfeld_transmitted.get_grid_below_above` keys its grid on the EXACT
eps_t and frequency, so a sweep over a deck with wires on both sides of the
interface fills a fresh grid at every point: 0.82 s of a 1.4 s nec5-bspline
call on `mixed_vert_over_buried24`, the whole of the per-point ground cost.
Nothing in the exact grid is shared between frequencies, and `scaled_to`
records why no normalisation can share it: the observer axes are paced by
lambda_p and the z' ladder by lambda_m, and their ratio moves with frequency.

This module shares it a different way. It fixes the lattice IN METRES -- the
deck's geometry sets the R and theta axes and the z' ladder, whatever the
frequency -- and tabulates exact grids (ANCHORS) on that lattice at the
frequencies F_REF * 1.1**j, each at the soil's own eps_t there. A sweep
point's grid is then the degree-7 Lagrange blend, in ln f, of the eight
anchors nearest it: a node-by-node sum over tables that already exist.

Measured on `mixed_vert_over_buried24`'s geometry (R 0.4-22 m, theta 0-90
deg, z' = -0.15 m, soil 13 / 0.005 S/m, 10 points over 7.1 MHz +- 25 %),
worst per-surface error of scale against direct evaluation:

    ladder   4 pts    6 pts      8 pts
    10 %     6e-6     1-2e-7     7e-9 .. 5e-8
     5 %     4e-7     3e-9       4e-11 .. 3e-10

-- the below/below anchors' order of magnitude (2.3e-8 at 10 % / degree 7,
`_sommerfeld_below._SOMM_BELOW_ANCHOR_STEP`), and four decades under the
grid's own spatial interpolation (3e-4, `_sommerfeld_transmitted._R_STEP_LOG`).
Dividing out exp(-j k_p R) or exp(-j k_m |z'|) first made it WORSE (3e-7 at
10 % / 8): the surfaces are already in their smoothest form.

The soil's curve is eps_t(f) = eps_r - j q / omega with q = sigma / eps0,
read back off the request's own eps_t (`_soil`). A ground given as a complex
eps_t (no sigma) has a q that moves with frequency, so each point is its own
family: correct, and no reuse. q is rounded to 12 significant digits for the
key so the points of one sweep agree on it; the blend then serves the curve
through the rounded q, 1e-12 relative from the request in eps_t.

Whatever an anchor could not serve -- an R cap, z' ladder depth or R floor
that holds at the request but not at an anchor 40 % higher -- returns None,
and the caller takes the exact grid, which raises or serves exactly as it
does with the feature off.

Off by default; flag off, nothing here is imported on the solve path.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from . import _cancel, _somm_disk_cache
from ._sommerfeld import _C_LIGHT, _evict_fifo, _somm_r1_bucket_wl
from ._sommerfeld_below import k_medium
from ._sommerfeld_transmitted import (
    _DTH_DEG,
    _LADDER_MIN,
    _PAD_ROWS,
    _R_CAP_LAMBDA_P,
    _R_MIN_FLOOR_LAMBDA_P,
    _R_MIN_LAMBDA_P,
    _R_STEP_LOG,
    _SURF_KEYS_T,
    _ZPRIME_MAX_LAMBDA_M,
    TransmittedGrid,
    grazing_floor,
    grid_extent,
    ladder_nodes,
    t_surfaces_direct,
)

# The anchor ladder: frequencies F_REF * exp(j * STEP), blended over POINTS
# of them. One F_REF for every deck, so any two sweeps on the same soil and
# geometry share anchors whatever their band.
ANCHOR_F_REF_HZ = 1.0e6
ANCHOR_STEP = math.log(1.1)
ANCHOR_POINTS = 8

# Significant digits of q = sigma/eps0 in the family key (see the module note).
_Q_DIGITS = 12

# A multi-depth z' ladder's count rounds UP to one of these, so that the
# neighbouring points of a sweep (whose lambda_m differ a little) share one
# ladder and so one family.
_LADDER_BUCKETS = (9, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)

# The R axis's inner end is the request's `r_min` (the plan's nearest
# observer, which is geometry and so the same at every point of a sweep),
# snapped down onto exp(m * _R_STEP_LOG) metres. It is NOT clipped to the
# exact grid's default `_R_MIN_LAMBDA_P * lambda_p`: that default moves with
# frequency, and clipping to it split a 7.1 MHz +- 25 % sweep into two
# families at the octave edge (21 anchor fills where 13 serve). The lattice
# needs no row under the nearest observer; the padding rows give it the
# centred stencil there. A request that names no `r_min` gets the default at
# the top of its OCTAVE of F_REF instead, the same within an octave.

_ANCHORS: dict = {}
_ANCHORS_MAX = 96
_BLENDS: dict = {}
_BLENDS_MAX = 64

# What was built, for tests and benchmarks (never for control).
STATS = {"anchor_fills": 0, "anchor_disk": 0, "blends": 0, "declined": 0}


def reset():
    """Forget every anchor and blend held in this process (the disk keeps
    its copies). The tests call it so each starts from nothing."""
    _ANCHORS.clear()
    _BLENDS.clear()
    for k in STATS:
        STATS[k] = 0


def _soil(eps_t, omega):
    """(eps_r, q) of the curve eps_t(f) = eps_r - j q / omega through the
    request, q = sigma / eps0 rounded to `_Q_DIGITS` significant digits."""
    eps_t = complex(eps_t)
    q = -eps_t.imag * float(omega)
    if q != 0.0:
        q = float(f"{q:.{_Q_DIGITS - 1}e}")
    return float(eps_t.real), q


def _eps_at(eps_r, q, omega):
    return complex(eps_r, -q / omega)


def _f_anchor(j):
    return ANCHOR_F_REF_HZ * math.exp(ANCHOR_STEP * j)


def _stencil(f_hz):
    """The anchor indices a request at `f_hz` blends, and its ladder
    coordinate u (anchor j sits at u = j)."""
    u = math.log(f_hz / ANCHOR_F_REF_HZ) / ANCHOR_STEP
    j0 = math.floor(u) - ANCHOR_POINTS // 2 + 1
    return list(range(j0, j0 + ANCHOR_POINTS)), u


def _weights(u, js):
    """Lagrange weights at u through the integer nodes `js` -- exactly 1 and
    0s when u is a node, so a request ON an anchor serves that anchor."""
    w = []
    for i in js:
        p = 1.0
        for m in js:
            if m != i:
                p *= (u - m) / (i - m)
        w.append(p)
    return w


def _ladder_bucket(n):
    for b in _LADDER_BUCKETS:
        if n <= b:
            return b
    return None


class _Lattice:
    """The metres-fixed lattice of one family: what every anchor and every
    blend in it tabulates on."""

    __slots__ = ("r_min", "r_max", "zp_nodes", "n_r", "lnr0", "th_min", "n_th", "dth")

    def __init__(self, r_min, r_max, zp_nodes):
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.zp_nodes = np.asarray(zp_nodes, dtype=float)
        self.lnr0 = math.log(self.r_min) - _PAD_ROWS * _R_STEP_LOG
        self.n_r = (
            int(math.ceil(math.log(self.r_max / self.r_min) / _R_STEP_LOG))
            + 1
            + 2 * _PAD_ROWS
        )
        self.th_min = grazing_floor(self.r_max, float(self.zp_nodes[0]))
        n_int = max(int(round((90.0 - math.degrees(self.th_min)) / _DTH_DEG)), 4)
        self.n_th = n_int + 1
        self.dth = (0.5 * np.pi - self.th_min) / n_int

    def key(self):
        z = self.zp_nodes
        return (self.r_min, self.r_max, float(z[0]), float(z[-1]), int(z.size))


def _lattice(f_hz, js, r_max, zp_min, zp_max, r_min, eps_r, q):
    """The family's lattice, or None when an anchor in `js` could not
    tabulate it inside the exact grid's own served domain."""
    lam_p = _C_LIGHT / f_hz
    octave_top = ANCHOR_F_REF_HZ * 2.0 ** math.ceil(
        math.log2(f_hz / ANCHOR_F_REF_HZ) - 1e-12
    )
    lo_default = _R_MIN_LAMBDA_P * _C_LIGHT / octave_top
    want = lo_default if r_min is None else float(r_min)
    # On the log lattice exp(m * _R_STEP_LOG) metres, snapped down.
    r_lo = math.exp(_R_STEP_LOG * math.floor(math.log(want) / _R_STEP_LOG + 1e-12))
    r_hi = _somm_r1_bucket_wl(max(float(r_max), 4.0 * r_lo))
    if r_hi > _R_CAP_LAMBDA_P * lam_p:
        return None
    lo = abs(float(zp_min))
    hi = abs(float(zp_max))
    if hi <= lo * (1.0 + 1e-9):
        zp_nodes = np.array([lo])
    else:
        need = _LADDER_MIN
        for j in js:
            k_m = k_medium(
                _eps_at(eps_r, q, 2.0 * np.pi * _f_anchor(j)),
                2.0 * np.pi * _f_anchor(j) / _C_LIGHT,
            )
            need = max(need, ladder_nodes(lo, hi, 2.0 * np.pi / abs(k_m))[1])
        n = _ladder_bucket(need)
        if n is None:
            return None
        zp_nodes = np.exp(np.linspace(math.log(lo), math.log(hi), n))
    for j in js:
        fj = _f_anchor(j)
        lam_pj = _C_LIGHT / fj
        k_m = k_medium(_eps_at(eps_r, q, 2.0 * np.pi * fj), 2.0 * np.pi * fj / _C_LIGHT)
        lam_mj = 2.0 * np.pi / abs(k_m)
        if r_hi > _R_CAP_LAMBDA_P * lam_pj * (1.0 + 1e-9):
            return None
        if r_lo < _R_MIN_FLOOR_LAMBDA_P * lam_pj:
            return None
        if hi > _ZPRIME_MAX_LAMBDA_M * lam_mj * (1.0 + 1e-9):
            return None
    return _Lattice(r_lo, r_hi, zp_nodes)


def _fill_anchor(lat, eps_t, k2, omega, mu, rtol):
    """An exact TransmittedGrid on `lat`, filled by the shipped evaluator
    exactly as `TransmittedGrid.__init__` fills its own lattice."""
    g = object.__new__(TransmittedGrid)
    _set_lattice(g, lat, eps_t, k2, omega, mu)
    r_nodes = np.exp(g.lnr0 + g.dlnr * np.arange(g.n_r))
    th_nodes = g.th0 + g.dth * np.arange(g.n_th)
    rr, tt = np.meshgrid(r_nodes, th_nodes, indexing="ij")
    rho = np.maximum(rr * np.cos(tt), 0.0)
    zz = np.maximum(rr * np.sin(tt), 0.0)
    vals = np.empty((g.n_zp, len(_SURF_KEYS_T), g.n_r, g.n_th), dtype=np.complex128)
    for i, zpn in enumerate(g.zp_nodes):
        surf = t_surfaces_direct(
            g.eps_t, g.k2, rho, zz, -float(zpn), rtol=rtol, omega=g.omega, mu=g.mu
        )
        vals[i] = np.stack([surf[k] for k in _SURF_KEYS_T])
    g._vals = vals
    return g


def _set_lattice(g, lat, eps_t, k2, omega, mu):
    """Every attribute `TransmittedGrid.__init__` sets, on `lat`."""
    g.eps_t = complex(eps_t)
    g.k2 = float(k2)
    g.omega = float(omega)
    g.mu = float(mu)
    g.k_m = k_medium(g.eps_t, g.k2)
    g.lam_p = 2.0 * np.pi / g.k2
    g.lam_m = 2.0 * np.pi / abs(g.k_m)
    g.r_min = lat.r_min
    g.r_cap = _R_CAP_LAMBDA_P * g.lam_p
    g.r_max = lat.r_max
    g.zp_nodes = lat.zp_nodes.copy()
    g.n_zp = int(lat.zp_nodes.size)
    g.zp_min = float(lat.zp_nodes[0])
    g.zp_max = float(lat.zp_nodes[-1])
    g.lnr0 = lat.lnr0
    g.dlnr = _R_STEP_LOG
    g.n_r = lat.n_r
    g.th_min = lat.th_min
    g.th0 = lat.th_min
    g.n_th = lat.n_th
    g.dth = lat.dth
    g.lnz0 = math.log(g.zp_min)
    g.dlnz = (math.log(g.zp_max) - g.lnz0) / (g.n_zp - 1) if g.n_zp > 1 else 0.0


def _source_digest():
    """sha256 of this file: the anchor fill and lattice live here, outside the
    modules `_somm_disk_cache`'s code fingerprint hashes."""
    if not _SOURCE:
        with open(__file__, "rb") as fh:
            _SOURCE.append(hashlib.sha256(fh.read()).hexdigest())
    return _SOURCE[0]


_SOURCE: list = []


def _anchor(j, lat, eps_r, q, mu, rtol, cancel_flag):
    key = ("below-above-anchor", eps_r, q, int(j), lat.key(), float(mu))
    g = _ANCHORS.get(key)
    if g is not None:
        return g
    fj = _f_anchor(j)
    omega_j = 2.0 * np.pi * fj

    def fill():
        STATS["anchor_fills"] += 1
        return _fill_anchor(
            lat, _eps_at(eps_r, q, omega_j), omega_j / _C_LIGHT, omega_j, mu, rtol
        )

    hits = _somm_disk_cache.STATS["hits"]
    with _cancel.scope_flag(cancel_flag):
        g = _somm_disk_cache.fetch_or_fill(
            key,
            fill,
            extra=(float(rtol), ANCHOR_F_REF_HZ, ANCHOR_STEP, _source_digest()),
        )
    if _somm_disk_cache.STATS["hits"] != hits:
        STATS["anchor_disk"] += 1
    _evict_fifo(_ANCHORS, _ANCHORS_MAX)
    _ANCHORS[key] = g
    return g


def get_grid(
    eps_t, k2, r_max, zp_min, zp_max, omega, mu, r_min=None, rtol=1e-9, cancel_flag=0
):
    """The blended transmitted grid for this request, or None when the
    anchors cannot serve it (the caller then fills the exact grid).

    A `TransmittedGrid` in every attribute `eval` and the C++ projection
    read, carrying the REQUEST's eps_t, k2 and omega, on the family's
    metres-fixed lattice -- which covers the request's (r_min, r_max) and
    z' range, as the exact grid's bucketed lattice does."""
    omega = float(omega)
    f_hz = omega / (2.0 * np.pi)
    if not (f_hz > 0.0) or complex(eps_t).imag > 0.0:
        STATS["declined"] += 1
        return None
    eps_r, q = _soil(eps_t, omega)
    js, u = _stencil(f_hz)
    lat = _lattice(f_hz, js, r_max, zp_min, zp_max, r_min, eps_r, q)
    if lat is None or lat.r_max > _R_CAP_LAMBDA_P * (2.0 * np.pi / float(k2)):
        STATS["declined"] += 1
        return None
    # The serve plan checked its grazing angle against the EXACT grid's
    # theta floor (`grid_extent`). The floor rises with the lattice's outer
    # R, and the metres bucket can sit above the exact grid's wavelength
    # bucket, so a lattice whose floor is higher could be asked under it.
    try:
        th_exact = grid_extent(k2, r_max, zp_min, r_min=r_min)[2]
    except ValueError:
        STATS["declined"] += 1
        return None
    if lat.th_min > th_exact * (1.0 + 1e-9) + 1e-15:
        STATS["declined"] += 1
        return None
    bkey = (complex(eps_t), float(k2), omega, float(mu), lat.key(), float(rtol))
    g = _BLENDS.get(bkey)
    if g is not None:
        return g
    vals = None
    for j, w in zip(js, _weights(u, js), strict=True):
        a = _anchor(j, lat, eps_r, q, mu, rtol, cancel_flag)
        vals = a._vals * w if vals is None else vals + a._vals * w
    g = object.__new__(TransmittedGrid)
    _set_lattice(g, lat, eps_t, k2, omega, mu)
    g._vals = vals
    STATS["blends"] += 1
    _evict_fifo(_BLENDS, _BLENDS_MAX)
    _BLENDS[bkey] = g
    return g
