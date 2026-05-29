"""Probe scipy.interpolate.BSpline for the operations we need to build a
triangular-basis MoM solver.

Demonstrates:
  1. Construct triangular (k=1) basis on a clamped uniform knot vector
  2. Distinguish "all" basis vs "interior" basis (drop boundary tents for
     zero-current Dirichlet BC at wire endpoints)
  3. Evaluate basis at arbitrary points via design_matrix
  4. Take derivatives -> piecewise-constant per-segment (charge basis)
  5. Galerkin projection g_m = integral Phi_m(z) f(z) dz via per-segment
     Gauss-Legendre quadrature, cross-checked against scipy.integrate.quad
  6. Mass matrix M_mn = integral Phi_m(z) Phi_n(z) dz vs the analytic
     tridiagonal form (diag = 2h/3, off-diag = h/6) for uniform tents

Nothing here imports pysim; it's a pure scipy demonstration.
"""

import numpy as np
from scipy.interpolate import BSpline
from scipy.integrate import quad


def make_clamped_basis(L, N, k=1):
    """Clamped triangular basis on [0, L].

    Clamped knots = boundary knots repeated `k` times. With k=1 and N
    segments, knots are [0, 0, h, 2h, ..., (N-1)h, L, L] -- N+3 entries.
    scipy's design_matrix valid range is then the full [0, L].

    Returns (knots, n_basis, interior_slice) where interior_slice is the
    set of basis functions to use under zero-Dirichlet BC (drop the first
    and last, which are nonzero at z=0 and z=L respectively).
    """
    interior_knots = np.linspace(0.0, L, N + 1)
    knots = np.concatenate([np.full(k, 0.0), interior_knots, np.full(k, L)])
    n_basis = len(knots) - k - 1  # = N + k
    interior = slice(1, n_basis - 1)  # = N - 1 interior basis functions
    return knots, n_basis, interior


def design(z, knots, k=1):
    """(n_eval, n_basis) sparse matrix of Phi_j(z_i)."""
    return BSpline.design_matrix(z, knots, k)


def make_unit_basis_fn(knots, j, k=1):
    """The BSpline that is identically Phi_j."""
    n_basis = len(knots) - k - 1
    c = np.zeros(n_basis)
    c[j] = 1.0
    return BSpline(knots, c, k, extrapolate=False)


def gauss_legendre_per_segment(segment_endpoints, n_per_seg):
    """Map Gauss-Legendre nodes/weights onto every segment.

    `segment_endpoints` is a 1-D array of the N+1 unique knot positions
    that bound the N integration segments. Returns flat (z, w) such that
    integral f(z) dz from segment_endpoints[0] to segment_endpoints[-1]
    ~= sum_i w_i f(z_i).
    """
    gl_xi, gl_w = np.polynomial.legendre.leggauss(n_per_seg)
    seg_l = segment_endpoints[:-1]
    seg_r = segment_endpoints[1:]
    half = 0.5 * (seg_r - seg_l)
    mid = 0.5 * (seg_r + seg_l)
    z = (mid[:, None] + half[:, None] * gl_xi[None, :]).ravel()
    w = (half[:, None] * gl_w[None, :]).ravel()
    return z, w


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    L = 1.0
    N = 5
    knots, n_basis, interior = make_clamped_basis(L, N)
    n_interior = interior.stop - interior.start
    print(f"L={L}, N={N} segments, k=1 (triangular)")
    print(f"clamped knots: {knots}")
    print(f"n_basis (all): {n_basis}")
    print(f"n_interior (drop boundary tents for zero-current BC): {n_interior}")

    seg_endpoints = np.linspace(0.0, L, N + 1)  # the N physical segments
    h = L / N
    print(f"physical segment endpoints: {seg_endpoints}, h={h}")

    # ----------------------------------------------------------------
    section("(1) Design matrix at evenly spaced eval points")
    z_eval = np.linspace(0, L, 11)
    M_all = design(z_eval, knots).toarray()
    print(f"z_eval: {z_eval}")
    print(f"All {n_basis} basis functions (boundary tents in cols 0, {n_basis - 1}):")
    for i, zi in enumerate(z_eval):
        row = " ".join(f"{M_all[i, j]:5.3f}" for j in range(n_basis))
        print(f"  z={zi:5.2f}: {row}")
    print("row sums (partition of unity, all basis):")
    print(f"  {M_all.sum(axis=1)}")
    print()
    M_int = M_all[:, interior]
    print(f"Interior-only basis ({n_interior} cols): boundary tents dropped")
    print("  values at z=0 and z=L are all 0 -- zero-current BC for free")
    print(f"  z=0:   {M_int[0]}")
    print(f"  z=L:   {M_int[-1]}")
    print(f"  z=L/2: {M_int[len(z_eval) // 2]}")

    # ----------------------------------------------------------------
    section("(2) Derivative of an interior basis function: piecewise constant")
    j = 2  # an interior basis index (in the "all" indexing)
    phi = make_unit_basis_fn(knots, j=j)
    dphi = phi.derivative()
    print(f"Phi_{j}: support [knots[{j}]={knots[j]}, knots[{j + 2}]={knots[j + 2]}]")
    print(f"  apex at knots[{j + 1}]={knots[j + 1]}")
    z_fine = np.linspace(knots[j], knots[j + 2], 9)
    for zi in z_fine:
        v = float(phi(zi))
        d = float(dphi(zi))
        print(f"  z={zi:5.3f}  phi={v:6.3f}  dphi={d:+6.2f}")
    print(f"\nLeft half: dphi = +1/h = +{1 / h}")
    print(f"Right half: dphi = -1/h = -{1 / h}")
    print("(For MoM: dPhi/dz gives the per-segment charge-density basis,")
    print(" which is piecewise constant -- exactly the dual relationship")
    print(" between linear-current and constant-charge that thin-wire MoM wants.)")

    # ----------------------------------------------------------------
    section("(3) Galerkin projection g_m = integral Phi_m(z) f(z) dz")

    def f(z):
        return np.sin(np.pi * z / L)

    n_qp = 4
    z_q, w_q = gauss_legendre_per_segment(seg_endpoints, n_per_seg=n_qp)
    Phi_q = design(z_q, knots).toarray()[:, interior]  # (n_q, n_interior)
    g_quad = (Phi_q * f(z_q)[:, None]).T @ w_q
    print(f"Per-segment Gauss-Legendre, {n_qp} pts/seg (basis 0 at z=0, z=L):")
    for m, gm in enumerate(g_quad):
        print(f"  g_{m} = {gm:.10f}")

    print("\nReference via scipy.integrate.quad on per-basis BSpline objects:")
    max_rel = 0.0
    for m_local, m_global in enumerate(range(interior.start, interior.stop)):
        phi_m = make_unit_basis_fn(knots, m_global)
        # Phi_m support is [knots[m_global], knots[m_global + 2]].
        a, b = knots[m_global], knots[m_global + 2]
        g_ref, _ = quad(lambda z: phi_m(z) * f(z), a, b)
        rel = abs(g_quad[m_local] - g_ref) / abs(g_ref)
        max_rel = max(max_rel, rel)
        print(f"  g_{m_local} = {g_ref:.10f}   (rel err: {rel:.2e})")
    print(f"\nmax rel error vs scipy.quad: {max_rel:.2e}")

    # ----------------------------------------------------------------
    section("(4) Mass matrix M_mn = integral Phi_m(z) Phi_n(z) dz")
    Mass = Phi_q.T @ np.diag(w_q) @ Phi_q
    print("Computed Mass (n_interior x n_interior):")
    for row in Mass:
        print("  " + "  ".join(f"{v:+.6f}" for v in row))

    print(f"\nAnalytic tent mass matrix for uniform h={h}:")
    print(f"  diagonal:    2h/3 = {2 * h / 3:.6f}")
    print(f"  off-diag:    h/6  = {h / 6:.6f}")
    print("  elsewhere:   0")
    diag_err = max(abs(Mass[i, i] - 2 * h / 3) for i in range(n_interior))
    off_err = max(abs(Mass[i, i + 1] - h / 6) for i in range(n_interior - 1))
    other_err = max(
        abs(Mass[i, j])
        for i in range(n_interior)
        for j in range(n_interior)
        if abs(i - j) > 1
    )
    print("\nmax error vs analytic:")
    print(f"  diag      = {diag_err:.2e}")
    print(f"  off-diag  = {off_err:.2e}")
    print(f"  elsewhere = {other_err:.2e}")


if __name__ == "__main__":
    main()
