"""The exact ring kernel for coaxial same-radius segment pairs (momwire#1408).

For a current that does not vary around a straight wire of radius ``a``, the
field on the wire's SURFACE due to a ring of that current at axial separation
``z`` is carried by the exact ring kernel

    K(z) = (1/2π) ∮ e^{-jkR} / (4πR) dφ,   R² = z² + 4a² sin²(φ/2),

normalised like the solvers' ``e^{-jkR}/(4πR)`` (the 1/(4π) is folded in).
Unlike the reduced kernel (R² = z² + a², finite at z = 0) and the extended
kernel (a series in a²/R² about it), K is log-singular at z = 0 and is not an
approximation at any separation. Public context: King (1956), Werner (IEEE
TAP 1998), Fikioris & Valagiannopoulos (PIER 2005). The evaluation below is
our own derivation from public mathematics (the AGM and the Legendre-type
recurrence); no NEC-4 / NEC-5 material was used.

Evaluation (prototyped in antennaknobs ``scratch/1396-exact-kernel``):

* **Near, t = |z|/a < 6.** Expand e^{-jkR}/R = Σ (-jk)^n R^{n-1}/n! and use
  ∫₀^π R^p dθ = 2ρ^p J_{p/2}(m), ρ² = z² + 4a², m = 4a²/ρ², with
  J_ν(m) = ∫₀^{π/2} (1 - m sin²t)^ν dt. J_{-1/2} = K(m) and J_{1/2} = E(m)
  come from one AGM; J₀ = π/2, J₁ = (2 - m)π/4; every other order follows from
  the three-term recurrence
  J_{ν+1} = [2ν(m-1)J_{ν-1} + (2ν+1)(2-m)J_ν] / (2ν+2).
  The series is truncated where (kρ)^n/n! drops below 1e-17.
* **Far, t ≥ 6.** Expand g(s) = e^{-jk√s}/√s about the ring MEAN of R²,
  s₀ = z² + 2a², where the odd moments of R² - s₀ = -2a² cos φ vanish:
  K = (1/4π) Σ_{n even} ⟨ε^n⟩/n! · g^{(n)}(s₀), five terms (to a^16).

**The log singularity, split exactly.** With m₁ = 1 - m = z²/ρ²,

    K(m) = K_reg - (K(m₁)/π) ln m₁,   E(m) = E_reg - ((K(m₁) - E(m₁))/π) ln m₁,

and the recurrence is linear with coefficients polynomial in m, so every
half-integer J_ν splits the same way and the integer ones carry no log.
`ring_kernel_split` returns the two smooth functions A, B with
K(z) = A(z) ln|z| + B(z). `pair_moments` integrates A ln|z| against the basis
polynomials by product integration (exact log moments of the Legendre
polynomials), never by quadrature that has to resolve the singularity.

**Pair moments.** For two coaxial segments the 2-D moment integral

    J_pq = ∫₀^{h_i} ∫₀^{h_j} u^p v^q K(c + σ_j v - σ_i u) du dv

reduces exactly to one dimension, ∫ W_pq(y) K(c + y) dy, where the weight
W_pq is a piecewise polynomial (closed form, breakpoints at the four corner
values of y). Pieces away from the singular point take Gauss-Legendre, near
pieces take geometric grading, and a piece that reaches within 2a of the
singular point takes the exact log split above.

Complex k (a lossy medium) is not served: the AGM / series forms are written
for real k, and the callers refuse it before reaching here.
"""

from __future__ import annotations

from functools import lru_cache
from math import comb, factorial

import numpy as np

from ._accel import acc as _acc

_FOURPI = 4.0 * np.pi
_T_SWITCH = 6.0  # near (closed form) below, far series at and above, in |z|/a
_N_GL = 16  # Gauss-Legendre order per smooth piece
_N_SPLIT = 20  # order of the log-split rule on the piece touching z = 0
_GRADE = 3.0  # geometric chunk ratio away from the singular point


# --------------------------------------------------------------------------
# elliptic integrals by the AGM
# --------------------------------------------------------------------------
def _agm_KE(m1):
    """K(m), E(m) at parameter m = 1 - m1, with m1 passed exactly (so a tiny
    m1 loses nothing to cancellation). Iterates to convergence."""
    m1 = np.asarray(m1, dtype=np.float64)
    a = np.ones_like(m1)
    b = np.sqrt(m1)
    s = 0.5 * (1.0 - m1)  # 2^{-1} c0², c0² = m
    p = 0.5
    for _ in range(40):
        c = 0.5 * (a - b)
        a, b = 0.5 * (a + b), np.sqrt(a * b)
        p *= 2.0
        s = s + p * c * c
        if np.all(np.abs(c) <= 1e-17 * a):
            break
    K = np.pi / (2.0 * a)
    return K, K * (1.0 - s)


# --------------------------------------------------------------------------
# the kernel
# --------------------------------------------------------------------------
def _near_terms(kr_max):
    """Series length so that (kρ)^n/n! < 1e-17 for the largest kρ served."""
    n, term = 0, 1.0
    while term > 1e-17 and n < 80:
        n += 1
        term *= kr_max / n
    return max(n, 4)


def _near(z, a, k, split):
    """The near closed form at |z| = z (z ≥ 0 array). With ``split`` returns
    (A, B) such that K = A ln z + B; otherwise K."""
    rho2 = z * z + 4.0 * a * a
    rho = np.sqrt(rho2)
    m = 4.0 * a * a / rho2
    m1 = z * z / rho2
    Km, Em = _agm_KE(m1)
    N = _near_terms(float(k * np.max(rho)) if rho.size else 0.0)
    if split:
        with np.errstate(divide="ignore"):
            Kc, Ec = _agm_KE(m)  # K(m1), E(m1): the complementary pair
            lnm1 = np.log(m1)
        Lh = [-Kc / np.pi, -(Kc - Ec) / np.pi]  # log coefficients of K(m), E(m)
        with np.errstate(invalid="ignore"):
            Rh = [Km - Lh[0] * lnm1, Em - Lh[1] * lnm1]
        # z = 0 exactly: the regular parts' limits (K → ln 4 - ½ ln m1 ...).
        zero = m1 == 0.0
        if np.any(zero):
            Rh[0] = np.where(zero, np.log(4.0), Rh[0])
            Rh[1] = np.where(zero, 1.0, Rh[1])
    else:
        Rh = [Km, Em]
    Ji = [np.full_like(m, np.pi / 2.0), (2.0 - m) * np.pi / 4.0]
    nmax = (N - 1) // 2 + 2
    for i in range(1, nmax):
        nu = i - 0.5
        c0, c1, c2 = 2.0 * nu * (m - 1.0), (2.0 * nu + 1.0) * (2.0 - m), 2.0 * nu + 2.0
        Rh.append((c0 * Rh[i - 1] + c1 * Rh[i]) / c2)
        if split:
            Lh.append((c0 * Lh[i - 1] + c1 * Lh[i]) / c2)
        nu = float(i)
        Ji.append(
            (2.0 * nu * (m - 1.0) * Ji[i - 1] + (2.0 * nu + 1.0) * (2.0 - m) * Ji[i])
            / (2.0 * nu + 2.0)
        )
    reg = np.zeros(z.shape, dtype=np.complex128)
    log = np.zeros(z.shape, dtype=np.complex128) if split else None
    coef = 1.0 + 0.0j
    rp = 1.0 / rho  # ρ^{n-1}
    for n in range(N + 1):
        if n % 2 == 0:
            reg = reg + coef * rp * Rh[n // 2]
            if split:
                log = log + coef * rp * Lh[n // 2]
        else:
            reg = reg + coef * rp * Ji[(n - 1) // 2]
        coef = coef * (-1j * k) / (n + 1)
        rp = rp * rho
    scale = 2.0 / (4.0 * np.pi**2)
    if not split:
        return reg * scale
    # K = reg + log·ln m1 = reg + log·(2 ln z − ln ρ²)
    A = 2.0 * log * scale
    B = (reg - log * np.log(rho2)) * scale
    return A, B


@lru_cache(maxsize=1)
def _far_polys():
    """P_n with d^n/ds^n [e^{-jk√s}/√s] = e^{-x} P_n(x)/r^{2n+1}, x = jkr."""
    P = [np.poly1d([1.0])]
    x = np.poly1d([1.0, 0.0])
    for n in range(8):
        P.append((x * P[n].deriv() - (x + (2 * n + 1)) * P[n]) / 2.0)
    return P


# ⟨ε^n⟩/n! in units of a^{2n}, ε = R² - s₀ = -2a² cos φ
_FAR_C = {
    n: (2.0**n) * comb(n, n // 2) / 2.0**n / factorial(n) for n in (0, 2, 4, 6, 8)
}


def _far(z, a, k):
    s0 = z * z + 2.0 * a * a
    r = np.sqrt(s0)
    x = 1j * k * r
    P = _far_polys()
    acc = np.zeros(z.shape, dtype=np.complex128)
    for n in (0, 2, 4, 6, 8):
        acc = acc + _FAR_C[n] * a ** (2 * n) * P[n](x) / r ** (2 * n + 1)
    return np.exp(-x) * acc / _FOURPI


def ring_kernel(z, a, k):
    """The exact ring kernel K(z) (complex, per unit current-length squared).

    ``z`` is the axial separation (any sign), ``a`` the radius, ``k`` the real
    free-space wavenumber. Log-singular at z = 0 (returns inf there).
    """
    z = np.abs(np.asarray(z, dtype=np.float64))
    out = np.empty(z.shape, dtype=np.complex128)
    near = z < _T_SWITCH * a
    if np.any(near):
        with np.errstate(divide="ignore"):
            out[near] = _near(z[near], a, k, split=False)
    if not np.all(near):
        out[~near] = _far(z[~near], a, k)
    return out


def ring_kernel_split(z, a, k):
    """(A, B) with K(z) = A(z) ln|z| + B(z), A and B smooth; |z| < 6a only."""
    z = np.abs(np.asarray(z, dtype=np.float64))
    if np.any(z >= _T_SWITCH * a):
        raise ValueError("ring_kernel_split serves |z| < 6a only")
    return _near(z, a, k, split=True)


# --------------------------------------------------------------------------
# quadrature rules on [0, 1]
# --------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _gl01(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


@lru_cache(maxsize=8)
def _gl01_log(n):
    """Weights w_i with Σ w_i f(x_i) = ∫₀¹ f(x) ln x dx exactly for every
    polynomial f of degree < n, on the Gauss-Legendre nodes x_i.

    Product integration: the Lagrange basis on the GL nodes expands exactly
    in shifted Legendre polynomials (discrete orthogonality of the rule),
    and ∫₀¹ P̃_0 ln x = -1, ∫₀¹ P̃_m ln x = (-1)^{m+1}/(m(m+1)) for m ≥ 1.
    """
    x, w = _gl01(n)
    V = np.polynomial.legendre.legvander(2.0 * x - 1.0, n - 1)  # (n, n): P_m(x_i)
    m = np.arange(n)
    lm = np.empty(n)
    lm[0] = -1.0
    lm[1:] = (-1.0) ** (m[1:] + 1) / (m[1:] * (m[1:] + 1.0))
    return w * (V @ ((2.0 * m + 1.0) * lm))


# --------------------------------------------------------------------------
# the 1-D weight W_pq(y)
# --------------------------------------------------------------------------
def _u_limits(v0, s, hi, hj):
    """The two candidate expressions for each end of the u interval:
    (lower: 0 or the v-bound, upper: h_i or the v-bound)."""
    lo_v = np.where(s > 0, -v0, v0 - hj)
    hi_v = np.where(s > 0, hj - v0, v0)
    return lo_v, hi_v


def _weight(y, hi, hj, si, sj, nd, y_branch=None):
    """W_pq(y) = ∫ u^p v^q δ(y - (σ_j v - σ_i u)) du dv over the two segments'
    local coordinates, at every y (broadcast). Returns (..., nd, nd) real.

    W is a polynomial between the corner breakpoints. With ``y_branch`` (a
    point inside one piece) the u-interval's active ends are chosen THERE and
    held, so the result is that piece's polynomial continued analytically
    over every ``y`` — what the log-split rule needs when it integrates from
    the singular point across a breakpoint it must not see."""
    s = si * sj
    v0 = sj * y  # v = v0 + s·u
    s = np.broadcast_to(np.asarray(s, dtype=np.float64), np.shape(y))
    lo_v, hi_v = _u_limits(v0, s, hi, hj)
    if y_branch is None:
        u0 = np.maximum(0.0, lo_v)
        u1 = np.maximum(np.minimum(hi, hi_v), u0)
    else:
        bl, bh = _u_limits(sj * y_branch, si * sj, hi, hj)
        u0 = lo_v if bl > 0.0 else np.zeros_like(lo_v)
        u1 = hi_v if bh < hi else np.full_like(hi_v, hi)
    out = np.zeros(np.shape(y) + (nd, nd), dtype=np.float64)
    # powers of u0, u1 up to 2nd-1
    U0 = [np.ones_like(u0)]
    U1 = [np.ones_like(u1)]
    for _ in range(2 * nd):
        U0.append(U0[-1] * u0)
        U1.append(U1[-1] * u1)
    V0 = [np.ones_like(v0)]
    for _ in range(nd):
        V0.append(V0[-1] * v0)
    for p in range(nd):
        for q in range(nd):
            acc = np.zeros_like(u0)
            for r in range(q + 1):
                e = p + r + 1
                acc = acc + comb(q, r) * V0[q - r] * s**r * (U1[e] - U0[e]) / e
            out[..., p, q] = acc
    return out


# --------------------------------------------------------------------------
# pair moments
# --------------------------------------------------------------------------
def _chunks(d, e, ratio=_GRADE):
    """[d, e] cut geometrically outward from d > 0 (each chunk's distance to
    the singular point at least half its length)."""
    out = []
    lo = d
    while lo < e:
        hi = min(lo * ratio, e)
        if e - hi < 1e-3 * (hi - lo):
            hi = e
        out.append((lo, hi))
        lo = hi
    return out


def _gl_on(lo, hi, n=_N_GL):
    x, w = _gl01(n)
    return lo + (hi - lo) * x, (hi - lo) * w


class _Piece:
    """∫ over ζ of W(y(ζ)) K(ζ) dζ where y(ζ) = y_star + sgn·ζ."""

    def __init__(self, y_star, sgn, hi, hj, si, sj, nd, a, k, y_mid):
        self.y_star, self.sgn, self.y_mid = y_star, sgn, y_mid
        self.geom = (hi, hj, si, sj, nd)
        self.a, self.k = a, k

    def W(self, zeta):
        hi, hj, si, sj, nd = self.geom
        y = self.y_star + self.sgn * zeta
        return _weight(y, hi, hj, si, sj, nd, y_branch=self.y_mid)

    def smooth(self, lo, hi_):
        z, w = _gl_on(lo, hi_)
        Kz = ring_kernel(z, self.a, self.k)
        return np.einsum("i,i,ipq->pq", w, Kz, self.W(z))

    def graded(self, lo, hi_):
        tot = 0.0
        for c0, c1 in _chunks(lo, hi_):
            tot = tot + self.smooth(c0, c1)
        return tot

    def from_zero(self, b):
        """F(b) = ∫₀^b, the log split on [0, min(b, 2a)] and graded beyond."""
        b0 = min(b, 2.0 * self.a)
        x, w = _gl01(_N_SPLIT)
        wl = _gl01_log(_N_SPLIT)
        z = b0 * x
        A, B = ring_kernel_split(z, self.a, self.k)
        Wz = self.W(z)
        # ∫₀^{b0} W (A ln ζ + B) dζ, ln ζ = ln b0 + ln x
        f = b0 * (w * (B + A * np.log(b0)) + wl * A)
        tot = np.einsum("i,ipq->pq", f, Wz)
        if b > b0:
            tot = tot + self.graded(b0, b)
        return tot

    def integrate(self, d, ell):
        if ell <= 0.0:
            return 0.0
        if d >= ell:
            return self.smooth(d, d + ell)
        if d >= 2.0 * self.a:
            return self.graded(d, d + ell)
        out = self.from_zero(d + ell)
        if d > 0.0:
            out = out - self.from_zero(d)
        return out


def pair_moments(hi, hj, c, si, sj, a, k, nd):
    """(nd, nd) complex J_pq for ONE coaxial pair, any separation.

    ``c`` = axial start of segment j minus axial start of segment i; ``si``,
    ``sj`` = ±1 the segments' directions along the common axis; ``hi``, ``hj``
    their lengths. Local coordinates u, v run from each segment's start.
    """
    y_star = -c
    corners = np.sort([0.0, sj * hj, -si * hi, sj * hj - si * hi])
    out = np.zeros((nd, nd), dtype=np.complex128)
    for y0, y1 in zip(corners[:-1], corners[1:]):
        if y1 - y0 <= 0.0:
            continue
        subs = [(y0, y1)]
        if y0 < y_star < y1:
            subs = [(y0, y_star), (y_star, y1)]
        for ya, yb in subs:
            if y_star <= ya:
                piece = _Piece(y_star, 1.0, hi, hj, si, sj, nd, a, k, 0.5 * (ya + yb))
                d = ya - y_star
            else:
                piece = _Piece(y_star, -1.0, hi, hj, si, sj, nd, a, k, 0.5 * (ya + yb))
                d = y_star - yb
            out += piece.integrate(max(d, 0.0), yb - ya)
    return out


def pair_moments_far(hi, hj, c, si, sj, a, k, nd):
    """Vectorised `pair_moments` for pairs whose every piece sits at least its
    own length from the singular point (min |z| ≥ max(h_i, h_j)): one
    Gauss-Legendre rule per piece, no grading. Arrays of shape (P,);
    returns (nd, nd, P)."""
    hi, hj, c, si, sj = (np.asarray(v, dtype=np.float64) for v in (hi, hj, c, si, sj))
    corners = np.sort(
        np.stack([np.zeros_like(hi), sj * hj, -si * hi, sj * hj - si * hi], -1), -1
    )
    x, w = _gl01(_N_GL)
    lo = corners[:, :-1, None]
    ln = (corners[:, 1:] - corners[:, :-1])[:, :, None]
    y = lo + ln * x  # (P, 3, n)
    wy = ln * w
    Kz = ring_kernel(c[:, None, None] + y, a, k)
    Wy = _weight(
        y,
        hi[:, None, None],
        hj[:, None, None],
        si[:, None, None],
        sj[:, None, None],
        nd,
    )
    return np.einsum("Psi,Psi,Psipq->pqP", wy, Kz, Wy)


def _unique_pair_moments(h, sgn, ui, uj, cu, a, k, nd):
    """(nd, nd, U) moments of the representative pairs (ui[u], uj[u]) at
    axial offsets cu[u]: far pairs vectorised, the rest one at a time."""
    hi, hj, si, sj = h[ui], h[uj], sgn[ui], sgn[uj]
    # minimum |z| over the pair: |c + y| over the corner hull
    ylo = np.minimum.reduce([np.zeros_like(hi), sj * hj, -si * hi, sj * hj - si * hi])
    yhi = np.maximum.reduce([np.zeros_like(hi), sj * hj, -si * hi, sj * hj - si * hi])
    zlo, zhi = cu + ylo, cu + yhi
    dmin = np.where((zlo <= 0.0) & (zhi >= 0.0), 0.0, np.minimum(abs(zlo), abs(zhi)))
    far = dmin >= np.maximum(hi, hj)
    vals = np.empty((nd, nd, ui.shape[0]), dtype=np.complex128)
    fi = np.flatnonzero(far)
    step = 4096
    for s0 in range(0, fi.size, step):
        sl = fi[s0 : s0 + step]
        vals[:, :, sl] = pair_moments_far(
            hi[sl], hj[sl], cu[sl], si[sl], sj[sl], a, k, nd
        )
    for u in np.flatnonzero(~far):
        vals[:, :, u] = pair_moments(hi[u], hj[u], cu[u], si[u], sj[u], a, k, nd)
    return vals


class CoaxialRows:
    """Observer-row windows of one coaxial group's exact-kernel moment block
    (momwire#1411): ``rows(r0, r1)`` is ``coaxial_block(...)[:, :, r0:r1]``
    without the (nd, nd, n, n) block ever existing.

    Pairs are deduplicated on (h_i, h_j, c, σ_i, σ_j) as in `coaxial_block`,
    and the deduplication is carried ACROSS windows by a cache keyed the same
    way, so a uniform run still costs O(n) kernel work however many windows
    it is cut into. Walking the windows in row order, a key is first met in
    the earliest window holding it and, inside that window, at its first pair
    in row-major order -- the whole block's own first occurrence. So each
    unique moment is evaluated from the same representative pair as the
    whole-block route, and a window is that block's rows to the bit.

    ``max_cached`` bounds the cache (entries). A key met after the cache is
    full is still deduplicated inside its window, just not remembered: a
    graded mesh with O(n²) distinct pairs then costs what it costs today in
    kernel work, without the O(n²) residency. ``evaluated`` counts the unique
    moments computed, for gates that hold the O(n) claim.
    """

    def __init__(self, x0, sgn, h, a, k, nd, max_cached=None):
        self.x0 = np.asarray(x0, dtype=np.float64)
        self.sgn = np.asarray(sgn, dtype=np.float64)
        self.h = np.asarray(h, dtype=np.float64)
        self.n = self.x0.shape[0]
        self.a, self.k, self.nd = a, k, nd
        self.q = 1e-11 * float(np.max(self.h))
        self.max_cached = max_cached
        self.evaluated = 0
        self._index = {}
        self._vals = np.empty((nd, nd, 0), dtype=np.complex128)

    def _remember(self, keys, miss, vals):
        if self.max_cached is not None:
            room = self.max_cached - len(self._index)
            if room <= 0:
                return
            miss = miss[:room]
        n0 = len(self._index)
        need = n0 + miss.size
        if need > self._vals.shape[2]:
            grown = np.empty(
                (self.nd, self.nd, max(need, 2 * self._vals.shape[2])),
                dtype=np.complex128,
            )
            grown[:, :, :n0] = self._vals[:, :, :n0]
            self._vals = grown
        self._vals[:, :, n0:need] = vals[:, :, miss]
        for j, u in enumerate(miss.tolist()):
            self._index[keys[u]] = n0 + j

    def rows(self, r0, r1):
        """(nd, nd, r1 - r0, n): observer rows [r0, r1) of the block."""
        n, nd, q = self.n, self.nd, self.q
        m = r1 - r0
        I = np.repeat(np.arange(r0, r1), n)
        J = np.tile(np.arange(n), m)
        c = self.x0[J] - self.x0[I]
        key = np.stack(
            [
                np.round(self.h[I] / q),
                np.round(self.h[J] / q),
                np.round(c / q),
                self.sgn[I],
                self.sgn[J],
            ],
            -1,
        )
        uniq, first, inv = np.unique(
            key, axis=0, return_index=True, return_inverse=True
        )
        del key
        inv = inv.ravel()
        keys = list(map(tuple, uniq.tolist()))
        del uniq
        get = self._index.get
        pos = np.fromiter((get(t, -1) for t in keys), dtype=np.int64, count=len(keys))
        vals = np.empty((nd, nd, len(keys)), dtype=np.complex128)
        hit = np.flatnonzero(pos >= 0)
        vals[:, :, hit] = self._vals[:, :, pos[hit]]
        miss = np.flatnonzero(pos < 0)
        if miss.size:
            f = first[miss]
            vals[:, :, miss] = _unique_pair_moments(
                self.h, self.sgn, I[f], J[f], c[f], self.a, self.k, nd
            )
            self.evaluated += int(miss.size)
            self._remember(keys, miss, vals)
        del I, J, c, first
        return vals[:, :, inv].reshape(nd, nd, m, n)


def coaxial_block(x0, sgn, h, a, k, nd):
    """(nd, nd, n, n) exact-kernel moment block of one coaxial group.

    ``x0`` (n,) axial coordinate of each segment's START (its ``seg_l``),
    ``sgn`` (n,) ±1 its direction along the axis, ``h`` (n,) its length.
    Pairs are deduplicated on (h_i, h_j, c, σ_i, σ_j), so a uniform run costs
    O(n) kernel work rather than O(n²). The whole block at once: the solver
    walks it in row windows instead (`CoaxialRows`, momwire#1411), and this
    is the reference those windows are gated against.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    return CoaxialRows(x0, sgn, h, a, k, nd, max_cached=0).rows(0, x0.shape[0])


# --------------------------------------------------------------------------
# the C++ route (momwire#1410)
# --------------------------------------------------------------------------
# `_accel_exact_kernel.cpp` ports the kernel, the pair moments and the
# row windows with their cross-window cache; the dedup there hashes instead of
# sorting, and keys a pair on its least image under the coaxial pair's exact
# symmetries (forward segments, transpose, mirror), which halves a uniform
# run's unique pairs and extends the dedup to graded meshes. Not bit-exact
# with the numpy route above by decision: the two agree to a derived
# tolerance (tests/test_exact_kernel_accel_1410.py), and the numpy route stays
# the reference and the fallback where the accelerator is absent.
#
# `_USE_ACCEL` is the switch the tests flip to hold the routes against each
# other; `_SYMMETRIC = False` keeps numpy's own key in the C++ route.
_HAVE_ACCEL = _acc is not None and hasattr(_acc, "exact_kernel_1410")
_USE_ACCEL = True
_SYMMETRIC = True
_ACCEL_MAX_ND = 6


def _accel_rules():
    xg, wg = _gl01(_N_GL)
    xs, ws = _gl01(_N_SPLIT)
    return _acc.ExactKernelRules(xg, wg, xs, ws, _gl01_log(_N_SPLIT))


class CoaxialRowsAccel:
    """`CoaxialRows` through the C++ kernel: the same windows, the same
    `rows(r0, r1)` and `evaluated`, the cache bounded the same way."""

    def __init__(
        self, x0, sgn, h, a, k, nd, max_cached=None, *, symmetric=None, n_threads=0
    ):
        h = np.asarray(h, dtype=np.float64)
        self.n = int(h.shape[0])
        self.nd = nd
        self.q = 1e-11 * float(np.max(h)) if self.n else 1.0
        self._impl = _acc.ExactKernelCoaxialRows(
            np.asarray(x0, dtype=np.float64),
            np.asarray(sgn, dtype=np.float64),
            h,
            float(a),
            float(k),
            int(nd),
            -1 if max_cached is None else int(max_cached),
            self.q,
            _SYMMETRIC if symmetric is None else bool(symmetric),
            _accel_rules(),
            int(n_threads),
        )

    @property
    def evaluated(self):
        return int(self._impl.evaluated)

    def rows(self, r0, r1):
        """(nd, nd, r1 - r0, n): observer rows [r0, r1) of the block."""
        out = np.empty((self.nd, self.nd, r1 - r0, self.n), dtype=np.complex128)
        self._impl.rows(int(r0), int(r1), out)
        return out


def accel_serves(k, nd):
    """Whether the C++ route takes this group (and is switched on)."""
    return _HAVE_ACCEL and _USE_ACCEL and nd <= _ACCEL_MAX_ND and not np.iscomplexobj(k)


def coaxial_rows(x0, sgn, h, a, k, nd, max_cached=None):
    """The row-window source the solver walks: C++ where it serves, else
    the numpy `CoaxialRows`."""
    if accel_serves(k, nd):
        return CoaxialRowsAccel(x0, sgn, h, a, k, nd, max_cached=max_cached)
    return CoaxialRows(x0, sgn, h, a, k, nd, max_cached=max_cached)
