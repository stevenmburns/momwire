"""Point ports that move along one wire against one factorisation (momwire#1915).

On a Galerkin basis a point gap at arclength x leaves Z alone. It enters the
solve only through its port vector u(x) = Σ_k Φ_k(x) e_k, the basis functions
evaluated at x, of which d+1 are nonzero on a degree-d B-spline. So put a
shorted port on every basis k of the chosen wire (a "site"), solve those
columns once beside the configured ports, and keep the small readout

    M = [B, E]ᵀ X,   X = the constrained solve of [B, E],

with B the configured port columns and E = [e_k] the site columns. The
constrained solve is linear in its right-hand side (the Schur form returns
W − X_c·Λ with Λ linear in W; the null-space form is Q(QᵀZQ)⁻¹Qᵀ), so the
solution for the drive u(x) is Σ_k Φ_k(x)·X[:, k], and the readout of any
port vector is the same contraction. With T(x) = blockdiag(I, F(x)) and
F[:, j] = Φ(x_j),

    Y(x) = T(x)ᵀ M T(x),
    ∂Y/∂x_j = T′_jᵀ M T + Tᵀ M T′_j,   T′_j = F′(x_j) in column j only.

Every term reads at most d+1 rows or columns of M per position, so a new x
costs O((d+1)·n_ports + (d+1)²·n_x²) and no back-substitution. An unloaded
site is a shorted port, i.e. the plain continuous wire, which is why the
sites need no termination of their own and drop out of `y`.

Smoothness in x is the basis's: Φ_k is C^{d-1} at a knot, so for d = 1 Y is
continuous with a derivative jump there, and for d = 2 it is C1. At a sample
exactly on an interior knot the span rule of `_bspline_basis_rows` takes the
span to the knot's RIGHT, so the d = 1 derivative returned there is the
right-hand (forward, increasing-arclength) one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def basis_derivative_rows(x, t, k, first):
    """dΦ/ds of the k+1 bases nonzero at each sample, on the spans `first`.

    `first` is `_bspline_basis_rows`'s second return for the same `x`, so the
    derivative is taken on exactly the span the values were (the knot's
    right-hand span at an interior knot). Uses
    Φ′_{i,k} = k·N_{i,k−1}/(t_{i+k} − t_i) − k·N_{i+1,k−1}/(t_{i+k+1} − t_{i+1}),
    with the degree-(k−1) values from the same Cox-de Boor recurrence run one
    level short. A zero-width knot pair carries no basis and contributes 0.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    t = np.asarray(t, dtype=np.float64)
    m = x.shape[0]
    ell = np.asarray(first) + k
    # Degree k-1 on span ell: h[:, a] is basis ell-(k-1)+a.
    h = np.ones((m, 1), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        for j in range(1, k):
            hh = h
            h = np.zeros((m, j + 1), dtype=np.float64)
            for nn in range(1, j + 1):
                xb = t[ell + nn]
                xa = t[ell + nn - j]
                w = np.where(xb == xa, 0.0, hh[:, nn - 1] / (xb - xa))
                h[:, nn - 1] += w * (xb - x)
                h[:, nn] = w * (x - xa)
        out = np.zeros((m, k + 1), dtype=np.float64)
        for b in range(k + 1):
            i = ell - k + b
            if b >= 1:
                den = t[i + k] - t[i]
                out[:, b] += np.where(den == 0.0, 0.0, k * h[:, b - 1] / den)
            if b <= k - 1:
                den = t[i + k + 1] - t[i + 1]
                out[:, b] -= np.where(den == 0.0, 0.0, k * h[:, b] / den)
    return out


def terminate_ports(y, z_loads: Mapping[int, complex]) -> np.ndarray:
    """Y over the ports NOT in `z_loads`, the others terminated in series loads.

    A load Z_L in a port's gap with no source holds V_L = −Z_L·I_L, so with
    the loaded set L, the kept set K and Z = diag(Z_L):

        I_L = (I + Y_LL Z)⁻¹ Y_LK V_K,
        Y_term = Y_KK − Y_KL Z (I + Y_LL Z)⁻¹ Y_LK.

    Written in Z rather than 1/Z so a shorted load (Z_L = 0) returns Y_KK
    exactly; the open limit Z_L → ∞ is the Schur complement Y_KK − Y_KL
    Y_LL⁻¹ Y_LK. The kept ports stay in their original relative order, so
    the feed's input impedance with the configured port 0 kept is
    ``1 / terminate_ports(y, loads)[0, 0]``.
    """
    y = np.asarray(y, dtype=np.complex128)
    n = y.shape[0]
    loaded = sorted(int(i) for i in z_loads)
    if any(not 0 <= i < n for i in loaded):
        raise ValueError(f"terminate_ports: load index outside 0..{n - 1}: {loaded}")
    kept = [i for i in range(n) if i not in set(loaded)]
    if not kept:
        raise ValueError("terminate_ports: every port is loaded; keep at least one")
    if not loaded:
        return y.copy()
    zl = np.array([complex(z_loads[i]) for i in loaded], dtype=np.complex128)
    y_kk = y[np.ix_(kept, kept)]
    y_kl = y[np.ix_(kept, loaded)]
    y_lk = y[np.ix_(loaded, kept)]
    y_ll = y[np.ix_(loaded, loaded)]
    a = np.eye(len(loaded), dtype=np.complex128) + y_ll * zl[None, :]
    return y_kk - (y_kl * zl[None, :]) @ np.linalg.solve(a, y_lk)


class PortSites:
    """The configured ports plus movable point ports on one wire (momwire#1915).

    Built by `BSplineSolver.port_sites`; holds one solve's site columns and
    their readout, and answers `y` / `dy_dx` / `coeffs` at any positions on
    the window without touching Z again. Positions are arclengths in metres
    from the wire's start, the convention of a feed's arclength.

    Attributes
    ----------
    wire, span:
        The wire index and the closed arclength window positions may take.
    n_ports:
        The solver's own port count; `y`'s first `n_ports` rows/columns are
        those ports in the solver's order, then one per requested position.
    n_sites:
        The number of basis sites cached (bases whose support meets `span`).
    nbytes:
        Bytes held: the solved columns (n_dof × (n_ports + n_sites)) and the
        readout matrix M.
    """

    def __init__(
        self,
        *,
        wire: int,
        span: tuple[float, float],
        degree: int,
        knots: np.ndarray,
        site_of_local: np.ndarray,
        n_ports: int,
        m: np.ndarray,
        columns: np.ndarray,
        n_poly: int,
        clip,
        basis_rows,
    ) -> None:
        self.wire = int(wire)
        self.span = (float(span[0]), float(span[1]))
        self.degree = int(degree)
        self.n_ports = int(n_ports)
        self._knots = knots
        self._site_of_local = site_of_local
        self._m = m
        self._columns = columns
        self._n_poly = int(n_poly)
        self._clip = clip
        self._basis_rows = basis_rows

    @property
    def n_sites(self) -> int:
        return int(self._m.shape[0] - self.n_ports)

    @property
    def nbytes(self) -> int:
        return int(self._columns.nbytes + self._m.nbytes)

    @property
    def site_y(self) -> np.ndarray:
        """Y over [ports…, every basis site…] — the cached readout M itself
        (a copy). Site k is the shorted port whose vector is e_k."""
        return self._m.copy()

    def _rows(self, xs, derivative):
        """`(idx, f, g)`: per position, the d+1 indices into M's site block
        (offset by n_ports), the basis values and (if asked) derivatives.
        A boundary basis the wire dropped carries index 0 and value 0."""
        xs = np.atleast_1d(np.asarray(xs, dtype=np.float64))
        if xs.ndim != 1 or xs.size == 0:
            raise ValueError("positions must be a non-empty 1-D sequence")
        lo, hi = self.span
        xs = np.array([self._clip(float(x)) for x in xs])
        bad = (xs < lo) | (xs > hi)
        if bad.any():
            raise ValueError(
                f"position(s) {xs[bad].tolist()} outside the cached window "
                f"[{lo!r}, {hi!r}] on wire {self.wire}"
            )
        vals, first = self._basis_rows(xs, self._knots, self.degree)
        local = first[:, None] + np.arange(self.degree + 1)[None, :]
        site = self._site_of_local[local]
        present = site >= 0
        idx = np.where(present, self.n_ports + site, 0)
        f = np.where(present, vals, 0.0)
        g = None
        if derivative:
            dv = basis_derivative_rows(xs, self._knots, self.degree, first)
            g = np.where(present, dv, 0.0)
        return idx, f, g

    def y(self, xs: float | Sequence[float]) -> np.ndarray:
        """Short-circuit Y over [ports…, a point port at each x in `xs`…]."""
        idx, f, _g = self._rows(xs, derivative=False)
        return self._assemble(idx, f)

    def dy_dx(self, xs: float | Sequence[float]) -> np.ndarray:
        """`(len(xs), n, n)`: entry j is ∂Y/∂x_j of `y(xs)`, per metre."""
        idx, f, g = self._rows(xs, derivative=True)
        m = self._m
        p = self.n_ports
        n_x = idx.shape[0]
        n = p + n_x
        # Q_fg[i, j] = f_iᵀ M[idx_i, idx_j] g_j; Q_gf likewise.
        block = m[idx[:, :, None, None], idx[None, None, :, :]]
        q_fg = np.einsum("ia,iajb,jb->ij", f, block, g)
        q_gf = np.einsum("ia,iajb,jb->ij", g, block, f)
        col = np.einsum("pja,ja->pj", m[:p][:, idx], g)  # M_ps g_j
        row = np.einsum("ja,jap->jp", g, m[idx][:, :, :p])  # g_jᵀ M_sp
        out = np.zeros((n_x, n, n), dtype=np.complex128)
        for j in range(n_x):
            out[j, :p, p + j] = col[:, j]
            out[j, p + j, :p] = row[j]
            out[j, p:, p + j] += q_fg[:, j]
            out[j, p + j, p:] += q_gf[j, :]
        return out

    def coeffs(self, xs: float | Sequence[float]) -> np.ndarray:
        """`(n_dof, n_ports + len(xs))`: solution columns for a 1 V drive at
        each port of `y(xs)`, others shorted — `PortSolution.coeffs`'s
        convention, enrichment block included where active."""
        idx, f, _g = self._rows(xs, derivative=False)
        cols = self._columns
        site_cols = np.einsum("nja,ja->nj", cols[:, idx], f)
        return np.hstack([cols[:, : self.n_ports], site_cols])

    def _assemble(self, idx, f):
        m = self._m
        p = self.n_ports
        n_x = idx.shape[0]
        out = np.empty((p + n_x, p + n_x), dtype=np.complex128)
        out[:p, :p] = m[:p, :p]
        out[:p, p:] = np.einsum("pja,ja->pj", m[:p][:, idx], f)
        out[p:, :p] = np.einsum("ja,jap->jp", f, m[idx][:, :, :p])
        block = m[idx[:, :, None, None], idx[None, None, :, :]]
        out[p:, p:] = np.einsum("ia,iajb,jb->ij", f, block, f)
        return out
