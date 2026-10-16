"""The exact ring kernel for coaxial segment pairs (momwire#1408; across a
radius step, momwire#1438: `ring_kernel2` and the section at the end).

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
from ._quadrature import leggauss

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
    m = 4.0 * a * a / rho2
    m1 = z * z / rho2
    return _near_series(z, rho2, m, m1, k, split)


def _near_series(z, rho2, m, m1, k, split):
    """The near series given ρ², the elliptic parameter m and m1 = 1 - m
    (passed exactly). One radius: ρ² = z² + 4a², m = 4a²/ρ²; two radii
    (`ring_kernel2`): ρ² = z² + (a₁ + a₂)², m = 4a₁a₂/ρ². ``split`` is the
    one-radius log split and is served for that form only."""
    rho = np.sqrt(rho2)
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
    x, w = leggauss(n)
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
    c = ((2.0 * m + 1.0) * lm).tolist()
    # `V @ c` as a plain left-to-right sum per node, not a BLAS matvec: the
    # matvec's summation order is the BLAS kernel's, which differs by CPU
    # (Rocketlake's AVX-512 kernel and Skylake's AVX2 one gave different
    # bits for n = 20), and these weights reach every exact-kernel fill.
    # A fixed order is one a C++ replay of the fill (momwire#1479) can match.
    rows = []
    for row in V.tolist():
        acc = 0.0
        for v, cm in zip(row, c, strict=True):
            acc += v * cm
        rows.append(acc)
    return w * np.array(rows)


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


# --------------------------------------------------------------------------
# point moments: the 1-D primitive (momwire#1433)
# --------------------------------------------------------------------------
# A point- or path-tested row (razor-2p's centroids, sin's match points)
# observes the field at a POINT on the wire's surface, so what it needs of a
# coaxial source segment is the 1-D moments of K against the source's local
# polynomial,
#
#     N_q(c, h) = ∫₀^h v^q K(c + v) dv,      q < nq,
#
# with the observer at the origin and the segment running forward from c.
# A segment that runs backward along the axis is the same integral with
# c = σ(x₀ - x_obs) (K is even), so (c, h) is the whole geometry. The
# singular point is v* = -c. The pieces from it outward take the same three
# rules as `pair_moments`' pieces (the log split within 2a of it, geometric
# grading beyond, one Gauss-Legendre rule on a piece at least its own length
# away), with the polynomial v^q in place of the 2-D weight W_pq.


def _point_piece(v0, sgn, d, ell, a, k, nq):
    """∫_d^{d+ell} f(v0 + sgn·ζ) K(ζ) dζ for f = v^q, q < nq: (nq,) complex."""
    if ell <= 0.0:
        return np.zeros(nq, dtype=np.complex128)

    def powers(zeta):
        v = v0 + sgn * zeta
        return v[:, None] ** np.arange(nq)[None, :]

    def smooth(lo, hi):
        z, w = _gl_on(lo, hi)
        return (w * ring_kernel(z, a, k)) @ powers(z)

    def graded(lo, hi):
        tot = np.zeros(nq, dtype=np.complex128)
        for c0, c1 in _chunks(lo, hi):
            tot = tot + smooth(c0, c1)
        return tot

    def from_zero(b):
        b0 = min(b, 2.0 * a)
        x, w = _gl01(_N_SPLIT)
        wl = _gl01_log(_N_SPLIT)
        z = b0 * x
        A, B = ring_kernel_split(z, a, k)
        f = b0 * (w * (B + A * np.log(b0)) + wl * A)
        tot = f @ powers(z)
        if b > b0:
            tot = tot + graded(b0, b)
        return tot

    if d >= ell:
        return smooth(d, d + ell)
    if d >= 2.0 * a:
        return graded(d, d + ell)
    out = from_zero(d + ell)
    if d > 0.0:
        out = out - from_zero(d)
    return out


def point_moments(c, h, a, k, nq):
    """(nq,) complex N_q(c, h) = ∫₀^h v^q K(c + v) dv for ONE observer and
    one coaxial source segment: `c` is the segment's start relative to the
    observer along the direction the segment runs, `h` its length."""
    vs = -float(c)
    h = float(h)
    if 0.0 < vs < h:
        return _point_piece(vs, -1.0, 0.0, vs, a, k, nq) + _point_piece(
            vs, 1.0, 0.0, h - vs, a, k, nq
        )
    if vs <= 0.0:
        return _point_piece(vs, 1.0, -vs, h, a, k, nq)
    return _point_piece(vs, -1.0, vs - h, h, a, k, nq)


def point_moments_far(c, h, a, k, nq):
    """Vectorised `point_moments` for segments at least their own length from
    the observer (min |c + v| ≥ h): one Gauss-Legendre rule each. Arrays of
    shape (P,); returns (nq, P)."""
    c = np.asarray(c, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    x, w = _gl01(_N_GL)
    v = h[:, None] * x[None, :]
    Kz = ring_kernel(c[:, None] + v, a, k)
    wv = h[:, None] * w[None, :] * Kz
    return np.stack([np.sum(wv * v**q, axis=1) for q in range(nq)])


class PointRows:
    """The exact-kernel point moments of one coaxial group's source segments
    at a set of observer points: the numpy reference of the C++
    `ExactKernelPointRows`.

    ``x0`` (n,) each segment's START along the axis, ``sgn`` (n,) ±1 its
    direction, ``h`` (n,) its length; `rows(x_obs)` returns (nq, m, n) with
    entry [q, o, j] = ∫₀^{h_j} τ^q K(x0_j + σ_j τ - x_o) dτ, τ the segment's
    own arc from its start -- the source polynomial's coordinate. Pairs are
    deduplicated on the quantised (c, h), and remembered across calls
    (bounded by `max_cached`), so a uniform run costs O(n) kernel work.
    """

    def __init__(self, x0, sgn, h, a, k, nq, max_cached=None):
        self.x0 = np.asarray(x0, dtype=np.float64)
        self.sgn = np.asarray(sgn, dtype=np.float64)
        self.h = np.asarray(h, dtype=np.float64)
        self.n = self.x0.shape[0]
        self.a, self.k, self.nq = a, k, nq
        self.q = 1e-11 * float(np.max(self.h)) if self.n else 1.0
        self.max_cached = max_cached
        self.evaluated = 0
        self._cache = {}

    def rows(self, x_obs):
        x_obs = np.asarray(x_obs, dtype=np.float64)
        m, n, nq = x_obs.shape[0], self.n, self.nq
        c = self.sgn[None, :] * (self.x0[None, :] - x_obs[:, None])
        hh = np.broadcast_to(self.h[None, :], (m, n))
        key = np.stack([np.round(c / self.q), np.round(hh / self.q)], -1).reshape(-1, 2)
        uniq, first, inv = np.unique(
            key, axis=0, return_index=True, return_inverse=True
        )
        inv = inv.ravel()
        cf, hf = c.ravel()[first], hh.ravel()[first]
        vals = np.empty((nq, uniq.shape[0]), dtype=np.complex128)
        keys = list(map(tuple, uniq.tolist()))
        miss = []
        for u, t in enumerate(keys):
            hit = self._cache.get(t)
            if hit is None:
                miss.append(u)
            else:
                vals[:, u] = hit
        miss = np.asarray(miss, dtype=np.int64)
        if miss.size:
            dmin = np.where(
                (cf[miss] <= 0.0) & (cf[miss] + hf[miss] >= 0.0),
                0.0,
                np.minimum(np.abs(cf[miss]), np.abs(cf[miss] + hf[miss])),
            )
            far = dmin >= hf[miss]
            fm = miss[far]
            if fm.size:
                vals[:, fm] = point_moments_far(cf[fm], hf[fm], self.a, self.k, nq)
            for u in miss[~far]:
                vals[:, u] = point_moments(cf[u], hf[u], self.a, self.k, nq)
            self.evaluated += int(miss.size)
            for u in miss.tolist():
                if self.max_cached is not None and len(self._cache) >= self.max_cached:
                    break
                self._cache[keys[u]] = vals[:, u].copy()
        # into the segment's own coordinate: τ = v forward, τ = v too on a
        # backward segment, whose c already carries σ (K is even).
        return vals[:, inv].reshape(nq, m, n)


_HAVE_POINT_ACCEL = _acc is not None and hasattr(_acc, "exact_kernel_point_1433")


def _point_rules():
    xg, wg = _gl01(_N_GL)
    xs, ws = _gl01(_N_SPLIT)
    return _acc.ExactKernelPointRules(xg, wg, xs, ws, _gl01_log(_N_SPLIT))


class PointRowsAccel:
    """`PointRows` through the C++ kernel (`_accel_exact_kernel_point.cpp`):
    the same `rows(x_obs)` and `evaluated`, the cache bounded the same way,
    and the pair keyed on the lesser of its mirror images."""

    def __init__(self, x0, sgn, h, a, k, nq, max_cached=None, *, n_threads=0):
        h = np.asarray(h, dtype=np.float64)
        self.n = int(h.shape[0])
        self.nq = nq
        q = 1e-11 * float(np.max(h)) if self.n else 1.0
        self._impl = _acc.ExactKernelPointRows(
            np.asarray(x0, dtype=np.float64),
            np.asarray(sgn, dtype=np.float64),
            h,
            float(a),
            float(k),
            int(nq),
            -1 if max_cached is None else int(max_cached),
            q,
            _point_rules(),
            int(n_threads),
        )

    @property
    def evaluated(self):
        return int(self._impl.evaluated)

    def rows(self, x_obs):
        x_obs = np.ascontiguousarray(x_obs, dtype=np.float64)
        out = np.empty((self.nq, x_obs.shape[0], self.n), dtype=np.complex128)
        self._impl.rows(x_obs, out)
        return out


def point_accel_serves(k, nq):
    """Whether the C++ point route takes this group (and is switched on)."""
    return (
        _HAVE_POINT_ACCEL
        and _USE_ACCEL
        and nq <= _ACCEL_MAX_ND
        and not np.iscomplexobj(k)
    )


def point_rows(x0, sgn, h, a, k, nq, max_cached=None):
    """The point-moment source a solver walks: C++ where it serves
    (`PointRowsAccel`), else the numpy `PointRows`."""
    if point_accel_serves(k, nq):
        return PointRowsAccel(x0, sgn, h, a, k, nq, max_cached=max_cached)
    return PointRows(x0, sgn, h, a, k, nq, max_cached=max_cached)


# --------------------------------------------------------------------------
# two radii: coaxial rings across a radius step (momwire#1438)
# --------------------------------------------------------------------------
# Each ring sits on its own wire's surface: the source current on a ring of
# the SOURCE segment's radius a₂, the field tested on the OBSERVER's own
# surface, radius a₁. For coaxial rings at axial separation z,
#
#     R² = z² + a₁² + a₂² - 2a₁a₂ cos φ = ρ² (1 - m cos²(φ/2)),
#     ρ² = z² + (a₁ + a₂)²,   m = 4a₁a₂/ρ²,   m₁ = 1 - m = (z² + δ²)/ρ²,
#
# δ = a₁ - a₂. That is the one-radius form with a² → a₁a₂ and 2a → a₁ + a₂,
# so every ring average ∫₀^π R^p dφ is the same 2ρ^p J_{p/2}(m), from the
# same AGM and the same recurrence (`_near_series`), and K₂ is symmetric in
# (a₁, a₂). At a₁ = a₂ it IS the one-radius kernel; the callers never ask
# for that case here (one radius keeps its own route, untouched).
#
# The singular split. The Legendre split K(m) = K_reg - (K(m₁)/π) ln m₁
# holds unchanged, and ln m₁ = ln(z² + δ²) - ln ρ², so
#
#     K₂(z) = A(z) ln(z² + δ²) + B(z),   A, B smooth and even in z,
#
# with A the one-radius log coefficient at the generalised (ρ, m). At δ = 0
# that is A·2 ln|z| + B, the one-radius split, and the step from one form to
# the other is continuous in δ (gate: tests/test_exact_kernel_two_radius_1438
# .py). At δ ≠ 0 there is NO real singularity: K₂ is analytic on the real z
# axis, bounded at z = 0, and its nearest singularities are the log branch
# points z = ±i|δ| (where R vanishes at φ = 0). So the moments need no
# product rule: Gauss-Legendre converges geometrically on any chunk whose
# distance to ±i|δ| is at least half its length, which is the bound the
# one-radius graded chunks already rest on. A piece is cut as one rule on
# [d, |δ|] (length ≤ |δ| ≤ its distance to the branch points) and geometric
# chunks from max(d, |δ|) outward. The chunk count grows as ln(ℓ/|δ|); the
# radius groups differ by more than 1e-6 relative (`_ek_axis_groups_coaxial`),
# which bounds it near 13.
#
# Far, |z| ≥ 6√(a₁a₂): the series about s₀ = z² + a₁² + a₂² with
# ε = -2a₁a₂ cos φ, ⟨εⁿ⟩ = (a₁a₂)ⁿ C(n, n/2). There 2a₁a₂/s₀ ≤ 2/38 because
# a₁² + a₂² ≥ 2a₁a₂: the one-radius bound at t = 6, so the same five terms
# hold the same accuracy.


def _near2(z, a1, a2, k):
    s = a1 + a2
    dl = a1 - a2
    rho2 = z * z + s * s
    m = 4.0 * a1 * a2 / rho2
    m1 = (z * z + dl * dl) / rho2
    return _near_series(z, rho2, m, m1, k, split=False)


def _far2(z, a1, a2, k):
    s0 = z * z + a1 * a1 + a2 * a2
    r = np.sqrt(s0)
    x = 1j * k * r
    p = a1 * a2
    P = _far_polys()
    acc = np.zeros(z.shape, dtype=np.complex128)
    for n in (0, 2, 4, 6, 8):
        acc = acc + _FAR_C[n] * p**n * P[n](x) / r ** (2 * n + 1)
    return np.exp(-x) * acc / _FOURPI


def ring_kernel2(z, a1, a2, k):
    """The exact ring kernel between coaxial rings of radii ``a1`` (the
    observer's) and ``a2`` (the source's) at axial separation ``z``:
    (1/2π)∮ e^{-jkR}/(4πR) dφ, R² = z² + a1² + a2² - 2 a1 a2 cos φ.

    Broadcasts over ``z``, ``a1``, ``a2``; ``k`` real. Finite at z = 0 when
    a1 ≠ a2. Symmetric in (a1, a2); the solvers still pass the observer's
    radius first, so a swapped-in asymmetric kernel (the mechanism gates)
    reaches the fill's own convention."""
    z, a1, a2 = np.broadcast_arrays(
        np.abs(np.asarray(z, dtype=np.float64)),
        np.asarray(a1, dtype=np.float64),
        np.asarray(a2, dtype=np.float64),
    )
    out = np.empty(z.shape, dtype=np.complex128)
    near = z < _T_SWITCH * np.sqrt(a1 * a2)
    if np.any(near):
        out[near] = _near2(z[near], a1[near], a2[near], k)
    if not np.all(near):
        far = ~near
        out[far] = _far2(z[far], a1[far], a2[far], k)
    return out


class _Piece2(_Piece):
    """`_Piece` between rings of two radii: analytic on the real axis, the
    branch points at ±i|δ| (`ring_kernel2`'s derivation), so the piece takes
    graded Gauss-Legendre from the scale |δ| and no log split."""

    def __init__(self, y_star, sgn, hi, hj, si, sj, nd, a1, a2, k, y_mid):
        super().__init__(y_star, sgn, hi, hj, si, sj, nd, a1, k, y_mid)
        self.a1, self.a2 = a1, a2
        self.dl = abs(a1 - a2)

    def smooth(self, lo, hi_):
        z, w = _gl_on(lo, hi_)
        Kz = ring_kernel2(z, self.a1, self.a2, self.k)
        return np.einsum("i,i,ipq->pq", w, Kz, self.W(z))

    def integrate(self, d, ell):
        return _integrate_two_radius(self, d, ell)


def _integrate_two_radius(piece, d, ell):
    """∫_d^{d+ell} of a two-radius piece: one rule on [d, |δ|], geometric
    chunks from max(d, |δ|) outward."""
    if ell <= 0.0:
        return 0.0
    if d >= ell:
        return piece.smooth(d, d + ell)
    dl = piece.dl
    if d >= dl:
        return piece.graded(d, d + ell)
    e = min(dl, d + ell)
    out = piece.smooth(d, e)
    if d + ell > e:
        out = out + piece.graded(e, d + ell)
    return out


def pair_moments2(hi, hj, c, si, sj, a1, a2, k, nd):
    """`pair_moments` between an observer segment i of radius ``a1`` and a
    coaxial source segment j of radius ``a2`` (a1 ≠ a2): the same 1-D
    reduction through W_pq, the two-radius kernel and its piece rule."""
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
            sgn = 1.0 if y_star <= ya else -1.0
            d = ya - y_star if sgn > 0 else y_star - yb
            piece = _Piece2(y_star, sgn, hi, hj, si, sj, nd, a1, a2, k, 0.5 * (ya + yb))
            out += piece.integrate(max(d, 0.0), yb - ya)
    return out


def pair_moments2_far(hi, hj, c, si, sj, a1, a2, k, nd):
    """Vectorised `pair_moments2` for pairs at least their own length from
    z = 0 (min |z| ≥ max(h_i, h_j)): `pair_moments_far` with the two-radius
    kernel. Arrays of shape (P,); returns (nd, nd, P)."""
    hi, hj, c, si, sj, a1, a2 = (
        np.asarray(v, dtype=np.float64) for v in (hi, hj, c, si, sj, a1, a2)
    )
    corners = np.sort(
        np.stack([np.zeros_like(hi), sj * hj, -si * hi, sj * hj - si * hi], -1), -1
    )
    x, w = _gl01(_N_GL)
    lo = corners[:, :-1, None]
    ln = (corners[:, 1:] - corners[:, :-1])[:, :, None]
    y = lo + ln * x
    wy = ln * w
    Kz = ring_kernel2(c[:, None, None] + y, a1[:, None, None], a2[:, None, None], k)
    Wy = _weight(
        y,
        hi[:, None, None],
        hj[:, None, None],
        si[:, None, None],
        sj[:, None, None],
        nd,
    )
    return np.einsum("Psi,Psi,Psipq->pqP", wy, Kz, Wy)


def _min_abs_z(c, ylo, yhi):
    zlo, zhi = c + ylo, c + yhi
    return np.where((zlo <= 0.0) & (zhi >= 0.0), 0.0, np.minimum(abs(zlo), abs(zhi)))


class CoaxialCrossRows:
    """Observer-row windows of the two-radius moments of one coaxial LINE
    (momwire#1438): `rows(r0, r1)` is (nd, nd, r1 - r0, n), holding J_pq of
    every pair whose two segments carry different radius-group labels
    (``lab``) and zero on every same-radius pair, which the one-radius
    groups own. Pairs are deduplicated on (h_i, h_j, c, σ_i, σ_j, a_i, a_j),
    and remembered across windows (bounded by ``max_cached``), so two
    uniform runs cost O(n) kernel work."""

    def __init__(self, x0, sgn, h, a, lab, k, nd, max_cached=None):
        self.x0 = np.asarray(x0, dtype=np.float64)
        self.sgn = np.asarray(sgn, dtype=np.float64)
        self.h = np.asarray(h, dtype=np.float64)
        self.a = np.asarray(a, dtype=np.float64)
        self.lab = np.asarray(lab)
        self.n = self.x0.shape[0]
        self.k, self.nd = k, nd
        self.q = 1e-11 * float(np.max(self.h))
        self.qa = 1e-11 * float(np.max(self.a))
        self.max_cached = max_cached
        self.evaluated = 0
        self._cache = {}

    def mask(self, r0, r1):
        """(r1 - r0, n) True on the window's cross-radius pairs."""
        return self.lab[r0:r1, None] != self.lab[None, :]

    def rows(self, r0, r1):
        n, nd = self.n, self.nd
        out = np.zeros((nd, nd, r1 - r0, n), dtype=np.complex128)
        I, J = np.nonzero(self.mask(r0, r1))
        if I.size == 0:
            return out
        I0 = I + r0
        c = self.x0[J] - self.x0[I0]
        key = np.stack(
            [
                np.round(self.h[I0] / self.q),
                np.round(self.h[J] / self.q),
                np.round(c / self.q),
                self.sgn[I0],
                self.sgn[J],
                np.round(self.a[I0] / self.qa),
                np.round(self.a[J] / self.qa),
            ],
            -1,
        )
        uniq, first, inv = np.unique(
            key, axis=0, return_index=True, return_inverse=True
        )
        inv = inv.ravel()
        keys = list(map(tuple, uniq.tolist()))
        vals = np.empty((nd, nd, len(keys)), dtype=np.complex128)
        miss = []
        for u, t in enumerate(keys):
            hit = self._cache.get(t)
            if hit is None:
                miss.append(u)
            else:
                vals[:, :, u] = hit
        miss = np.asarray(miss, dtype=np.int64)
        if miss.size:
            f = first[miss]
            ui, uj, cu = I0[f], J[f], c[f]
            hi, hj = self.h[ui], self.h[uj]
            si, sj = self.sgn[ui], self.sgn[uj]
            ai, aj = self.a[ui], self.a[uj]
            ys = [np.zeros_like(hi), sj * hj, -si * hi, sj * hj - si * hi]
            dmin = _min_abs_z(cu, np.minimum.reduce(ys), np.maximum.reduce(ys))
            far = dmin >= np.maximum(hi, hj)
            fm = np.flatnonzero(far)
            if fm.size:
                vals[:, :, miss[fm]] = pair_moments2_far(
                    hi[fm], hj[fm], cu[fm], si[fm], sj[fm], ai[fm], aj[fm], self.k, nd
                )
            for t in np.flatnonzero(~far):
                vals[:, :, miss[t]] = pair_moments2(
                    hi[t], hj[t], cu[t], si[t], sj[t], ai[t], aj[t], self.k, nd
                )
            self.evaluated += int(miss.size)
            for u in miss.tolist():
                if self.max_cached is not None and len(self._cache) >= self.max_cached:
                    break
                self._cache[keys[u]] = vals[:, :, u].copy()
        out[:, :, I, J] = vals[:, :, inv]
        return out


def coaxial_cross_block(x0, sgn, h, a, lab, k, nd):
    """(nd, nd, n, n) two-radius block of one coaxial line, whole (the
    reference the row windows are gated against)."""
    n = np.asarray(x0).shape[0]
    return CoaxialCrossRows(x0, sgn, h, a, lab, k, nd, max_cached=0).rows(0, n)


def _point_piece2(v0, sgn, d, ell, a1, a2, k, nq):
    """`_point_piece` between rings of two radii (`_integrate_two_radius`)."""

    class _P:
        dl = abs(a1 - a2)

        @staticmethod
        def smooth(lo, hi):
            z, w = _gl_on(lo, hi)
            v = v0 + sgn * z
            return (w * ring_kernel2(z, a1, a2, k)) @ (
                v[:, None] ** np.arange(nq)[None, :]
            )

        @staticmethod
        def graded(lo, hi):
            tot = np.zeros(nq, dtype=np.complex128)
            for c0, c1 in _chunks(lo, hi):
                tot = tot + _P.smooth(c0, c1)
            return tot

    out = _integrate_two_radius(_P, d, ell)
    return np.zeros(nq, dtype=np.complex128) if np.isscalar(out) else out


def point_moments2(c, h, a1, a2, k, nq):
    """`point_moments` with the observer on a ring of radius ``a1`` and the
    source segment's current on rings of radius ``a2`` (a1 ≠ a2)."""
    vs = -float(c)
    h = float(h)
    if 0.0 < vs < h:
        return _point_piece2(vs, -1.0, 0.0, vs, a1, a2, k, nq) + _point_piece2(
            vs, 1.0, 0.0, h - vs, a1, a2, k, nq
        )
    if vs <= 0.0:
        return _point_piece2(vs, 1.0, -vs, h, a1, a2, k, nq)
    return _point_piece2(vs, -1.0, vs - h, h, a1, a2, k, nq)


def point_moments2_far(c, h, a1, a2, k, nq):
    """Vectorised `point_moments2` for segments at least their own length
    from the observer. Arrays of shape (P,); returns (nq, P)."""
    c = np.asarray(c, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    x, w = _gl01(_N_GL)
    v = h[:, None] * x[None, :]
    Kz = ring_kernel2(
        c[:, None] + v,
        np.asarray(a1, dtype=np.float64)[:, None],
        np.asarray(a2, dtype=np.float64)[:, None],
        k,
    )
    wv = h[:, None] * w[None, :] * Kz
    return np.stack([np.sum(wv * v**q, axis=1) for q in range(nq)])


class PointCrossRows:
    """The two-radius point moments of one coaxial LINE (momwire#1438):
    `rows(x_obs, a_obs, lab_obs)` is (nq, m, n) with entry [q, o, j] =
    ∫₀^{h_j} τ^q K₂(x0_j + σ_j τ - x_o; a_obs[o], a_j) dτ on every pair whose
    observer and source carry different radius-group labels, and zero on the
    same-radius pairs the one-radius groups own. Deduplicated on the
    quantised (c, h, a_obs, a_src) and remembered across calls."""

    def __init__(self, x0, sgn, h, a, lab, k, nq, max_cached=None):
        self.x0 = np.asarray(x0, dtype=np.float64)
        self.sgn = np.asarray(sgn, dtype=np.float64)
        self.h = np.asarray(h, dtype=np.float64)
        self.a = np.asarray(a, dtype=np.float64)
        self.lab = np.asarray(lab)
        self.n = self.x0.shape[0]
        self.k, self.nq = k, nq
        self.q = 1e-11 * float(np.max(self.h)) if self.n else 1.0
        self.qa = 1e-11 * float(np.max(self.a)) if self.n else 1.0
        self.max_cached = max_cached
        self.evaluated = 0
        self._cache = {}

    def rows(self, x_obs, a_obs, lab_obs):
        x_obs = np.asarray(x_obs, dtype=np.float64)
        a_obs = np.asarray(a_obs, dtype=np.float64)
        m, n, nq = x_obs.shape[0], self.n, self.nq
        out = np.zeros((nq, m, n), dtype=np.complex128)
        O, J = np.nonzero(np.asarray(lab_obs)[:, None] != self.lab[None, :])
        if O.size == 0:
            return out
        c = self.sgn[J] * (self.x0[J] - x_obs[O])
        hh = self.h[J]
        ao, aj = a_obs[O], self.a[J]
        key = np.stack(
            [
                np.round(c / self.q),
                np.round(hh / self.q),
                np.round(ao / self.qa),
                np.round(aj / self.qa),
            ],
            -1,
        )
        uniq, first, inv = np.unique(
            key, axis=0, return_index=True, return_inverse=True
        )
        inv = inv.ravel()
        keys = list(map(tuple, uniq.tolist()))
        vals = np.empty((nq, len(keys)), dtype=np.complex128)
        miss = []
        for u, t in enumerate(keys):
            hit = self._cache.get(t)
            if hit is None:
                miss.append(u)
            else:
                vals[:, u] = hit
        miss = np.asarray(miss, dtype=np.int64)
        if miss.size:
            f = first[miss]
            cf, hf, af, bf = c[f], hh[f], ao[f], aj[f]
            dmin = _min_abs_z(cf, 0.0, hf)
            far = dmin >= hf
            fm = np.flatnonzero(far)
            if fm.size:
                vals[:, miss[fm]] = point_moments2_far(
                    cf[fm], hf[fm], af[fm], bf[fm], self.k, nq
                )
            for t in np.flatnonzero(~far):
                vals[:, miss[t]] = point_moments2(
                    cf[t], hf[t], af[t], bf[t], self.k, nq
                )
            self.evaluated += int(miss.size)
            for u in miss.tolist():
                if self.max_cached is not None and len(self._cache) >= self.max_cached:
                    break
                self._cache[keys[u]] = vals[:, u].copy()
        out[:, O, J] = vals[:, inv]
        return out


_HAVE_POINT_CROSS_ACCEL = _acc is not None and hasattr(
    _acc, "exact_kernel_point_cross_1438"
)


class PointCrossRowsAccel:
    """`PointCrossRows` through the C++ kernel (`_exact_kernel_cross_inline.h`,
    compiled into `_accel_exact_kernel_point.cpp`): the same `rows(x_obs,
    a_obs, lab_obs)` and `evaluated`, the same key (the quantised c, h,
    a_obs, a_src), the cache bounded the same way. Not bit-exact with the
    numpy class (that one sizes its near series per call, from the largest
    ρ in the batch): every value here is a function of its own pair alone,
    so the bits do not depend on the batch or the thread count; the
    agreement is gated at a tolerance (tests/test_exact_kernel_two_radius_
    1438.py). It is the one implementation the solvers and the standalone
    razor-2p path (momwire#1479) share."""

    def __init__(self, x0, sgn, h, a, lab, k, nq, max_cached=None, *, n_threads=0):
        h = np.asarray(h, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        self.n = int(h.shape[0])
        self.nq = nq
        q = 1e-11 * float(np.max(h)) if self.n else 1.0
        qa = 1e-11 * float(np.max(a)) if self.n else 1.0
        self._impl = _acc.ExactKernelPointCrossRows(
            np.asarray(x0, dtype=np.float64),
            np.asarray(sgn, dtype=np.float64),
            h,
            a,
            np.asarray(lab, dtype=np.int64),
            float(k),
            int(nq),
            -1 if max_cached is None else int(max_cached),
            q,
            qa,
            _point_rules(),
            int(n_threads),
        )

    @property
    def evaluated(self):
        return int(self._impl.evaluated)

    def rows(self, x_obs, a_obs, lab_obs):
        x_obs = np.ascontiguousarray(x_obs, dtype=np.float64)
        out = np.empty((self.nq, x_obs.shape[0], self.n), dtype=np.complex128)
        self._impl.rows(
            x_obs,
            np.ascontiguousarray(a_obs, dtype=np.float64),
            np.ascontiguousarray(lab_obs, dtype=np.int64),
            out,
        )
        return out


def point_cross_rows(x0, sgn, h, a, lab, k, nq, max_cached=None):
    """The two-radius point-moment source a solver walks: C++ where it
    serves (`PointCrossRowsAccel`), else the numpy `PointCrossRows`."""
    if _HAVE_POINT_CROSS_ACCEL and point_accel_serves(k, nq):
        return PointCrossRowsAccel(x0, sgn, h, a, lab, k, nq, max_cached=max_cached)
    return PointCrossRows(x0, sgn, h, a, lab, k, nq, max_cached=max_cached)
