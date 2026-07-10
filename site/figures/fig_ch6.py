"""Figures for Act II chapter 6 — "Integrals done honestly".

Run from the repo root:  .venv/bin/python site/figures/fig_ch6.py
(needs momwire importable — the repo venv.)
"""

import numpy as np
import matplotlib.pyplot as plt

from style import AMBER, CYAN, GREEN, MUTED, RED, save

WAVELENGTH = 22.0
L = 2 * 0.962 * WAVELENGTH / 4
A_THIN = 0.0005
N = 41
DZ = L / N
K = 2 * np.pi / WAVELENGTH


def _kernel(s_obs, s_src):
    """Thin-wire kernel g = e^{-jkR}/(4πR), R = sqrt((Δs)² + a²)."""
    R = np.sqrt((s_obs - s_src) ** 2 + A_THIN**2)
    return np.exp(-1j * K * R) / (4 * np.pi * R)


def fig_integrand():
    """The integrand a matrix entry must integrate, along one source segment:
    for a far observation point it's flat and tame; for the self segment it
    spikes as the source passes under the observation point."""
    s = np.linspace(-DZ / 2, DZ / 2, 2000)  # arclength across one source segment
    self_obs = _kernel(0.0, s)  # observation at this segment's own center
    far_obs = _kernel(2.0, s)   # observation 2 m away

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(s * 1e3, np.abs(self_obs), color=RED, lw=2,
            label="same segment (self):  spikes to 1/4πa")
    ax.plot(s * 1e3, np.abs(far_obs), color=CYAN, lw=2,
            label="far segment (2 m away):  smooth and small")
    ax.axhline(1 / (4 * np.pi * A_THIN), color=MUTED, lw=0.8, ls=":")
    ax.annotate("only the a² floor keeps\nthis finite", xy=(0, 1 / (4 * np.pi * A_THIN)),
                xytext=(3, 120), color=MUTED, fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    ax.set_yscale("log")
    ax.set_xlabel("position of the source point across the segment  (mm)")
    ax.set_ylabel("|g(s, s′)|   (1/m)")
    ax.set_title("Same integrand, two neighbours: the self term is a near-singular spike")
    ax.legend(fontsize=9, loc="center right")
    save(fig, "ch6-integrand")


def _gauss_int(s_obs, n):
    """∫ over one source segment of g, by n-point Gauss-Legendre (naive)."""
    xi, w = np.polynomial.legendre.leggauss(n)
    s = 0.0 + (DZ / 2) * xi  # segment centered at 0
    return ((DZ / 2) * w * _kernel(s_obs, s)).sum()


def _exact_self():
    """Self integral with the 1/R part done analytically (asinh) + smooth
    Gauss — the singularity-extracted 'truth' momwire precomputes."""
    xi, w = np.polynomial.legendre.leggauss(64)
    s = (DZ / 2) * xi
    R = np.sqrt(s**2 + A_THIN**2)
    smooth = ((DZ / 2) * w * (np.exp(-1j * K * R) - 1.0) / R).sum()
    sing = 2 * np.arcsinh((DZ / 2) / A_THIN)  # ∫ 1/R over the segment
    return (smooth + sing) / (4 * np.pi)


def fig_quadrature():
    """Error vs Gauss order: the far pair is nailed by 4 points; the self pair
    crawls (hundreds), which is why momwire never brute-forces it — it uses 4
    points for far pairs and precomputed static moments for the singular ones."""
    ns = np.arange(2, 130, 2)
    exact_self = _exact_self()
    exact_far = _gauss_int(2.0, 256)
    err_self = [abs(_gauss_int(0.0, n) - exact_self) / abs(exact_self) for n in ns]
    err_far = [abs(_gauss_int(2.0, n) - exact_far) / abs(exact_far) + 1e-18 for n in ns]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.semilogy(ns, err_self, "-", color=RED, lw=2, label="self pair (naive Gauss)")
    ax.semilogy(ns, err_far, "-", color=CYAN, lw=2, label="far pair (naive Gauss)")
    ax.axvline(4, color=GREEN, lw=1.2, ls=":")
    ax.annotate("momwire: n_qp_pair = 4\nfor smooth far pairs", xy=(4, 1e-6),
                xytext=(14, 3e-5), color=GREEN, fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1))
    ax.axhline(1e-12, color=AMBER, lw=1.2, ls="--")
    ax.annotate("self pair: precomputed static moments (exact, O(1))",
                xy=(70, 1e-12), xytext=(24, 4e-11), color=AMBER, fontsize=8.5)
    ax.set_ylim(1e-16, 2)
    ax.set_xlabel("Gauss-Legendre points per segment")
    ax.set_ylabel("relative error of the matrix entry")
    ax.set_title("You can't out-integrate a singularity: far pairs are cheap, self pairs aren't")
    ax.legend(fontsize=9, loc="upper right")
    save(fig, "ch6-quadrature")


if __name__ == "__main__":
    fig_integrand()
    fig_quadrature()
