"""momwire#883 — validate the regenerated degree-3 moment tables.

Raising MAX_D to 3 adds 7 entries to each of the two families (every (p, q)
with p = 3 or q = 3). Nothing in the repo pins them, and the closed forms are
machine-generated sympy that no one will read, so they are checked against
direct 2-D numerical quadrature of the defining integrals.

Two families, one integrand each — never split Dg into its /R^3 and /R^5
halves, which is the generator's own stated invariant (they are individually
O(1) while their sum is O(a^2), so differencing them ships the answer as the
residue of a catastrophic cancellation).

    J_pq = int int (s-alpha)^p (t-A)^q / sqrt((s-t)^2 + a^2) dt ds
    D_pq = int int (s-alpha)^p (t-A)^q Dg(s-t) dt ds
           Dg(xi) = -a^2/(2 R^3) + 3 a^4/(4 R^5),  R = sqrt(xi^2 + a^2)

Three geometries, chosen so no single coincidence can carry a row: an offset
pair with distinct corners, the unit-square pair with alpha = A = 0 (where
several boundary terms collapse), and a well-separated pair. The quadrature's
own reported error bound is asserted before any comparison — an unconverged
reference is not a reference.
"""

import sys
from pathlib import Path

import numpy as np
from scipy import integrate

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from momwire._bspline_ek_moments import D_ek_moment  # noqa: E402
from momwire._bspline_static_moments import J_static_moment  # noqa: E402

MAX_D = 3

GEOMETRIES = [
    ("offset pair", 0.30, 0.95, 0.10, 0.70, 0.023),
    ("alpha=A=0", 0.00, 1.00, 0.00, 1.00, 0.010),
    ("separated", 0.50, 2.50, 4.00, 5.50, 0.050),
]


def _quad(kernel, p, q, alpha, beta, A, B, a):
    def inner(t, s):
        return (s - alpha) ** p * (t - A) ** q * kernel(s - t, a)

    return integrate.dblquad(
        inner, alpha, beta, lambda _s: A, lambda _s: B, epsabs=1e-15, epsrel=1e-13
    )


def reduced(xi, a):
    return 1.0 / np.sqrt(xi * xi + a * a)


def correction(xi, a):
    r2 = xi * xi + a * a
    r = np.sqrt(r2)
    return -(a**2) / (2.0 * r**3) + 3.0 * a**4 / (4.0 * r**5)


def main():
    print(f"momwire#883 — regenerated moments at MAX_D = {MAX_D}, against quadrature\n")
    worst_old = worst_new = 0.0
    n_new = 0
    for name, alpha, beta, A, B, a in GEOMETRIES:
        print(f"  {name}: alpha={alpha} beta={beta} A={A} B={B} a={a}")
        for fam, closed, kernel in (
            ("J", J_static_moment, reduced),
            ("D", D_ek_moment, correction),
        ):
            for p in range(MAX_D + 1):
                for q in range(MAX_D + 1):
                    ref, err = _quad(kernel, p, q, alpha, beta, A, B, a)
                    tol = 1e-9 * abs(ref) + 1e-15
                    assert abs(err) < tol, (
                        f"{fam}_{p}{q} on {name}: quadrature err {err:.3e} vs "
                        f"tol {tol:.3e} — the reference is not a reference"
                    )
                    got = closed(p, q, alpha, beta, A, B, a)
                    rel = abs(got - ref) / max(abs(ref), 1e-30)
                    is_new = p == MAX_D or q == MAX_D
                    if is_new:
                        n_new += 1
                        worst_new = max(worst_new, rel)
                    else:
                        worst_old = max(worst_old, rel)
                    flag = "NEW" if is_new else "   "
                    if rel > 1e-9:
                        print(
                            f"    {flag} {fam}_{p}{q}  FAIL  closed={got:.12e} "
                            f"quad={ref:.12e}  rel={rel:.2e}"
                        )
        print()
    print(f"  worst relative error, pre-existing (p,q <= 2): {worst_old:.3e}")
    print(
        f"  worst relative error, NEW degree-3 entries:    {worst_new:.3e}"
        f"   ({n_new} checked)"
    )
    assert n_new == 2 * len(GEOMETRIES) * (2 * (MAX_D + 1) - 1), (
        f"expected {2 * len(GEOMETRIES) * (2 * (MAX_D + 1) - 1)} new-entry "
        f"checks, ran {n_new} — the p==3-or-q==3 selector is wrong"
    )
    print("\n  OK" if max(worst_old, worst_new) < 1e-9 else "\n  MISMATCH")


if __name__ == "__main__":
    main()
