"""Figures for Act IV chapter 12 — "Matrices that are secretly small".

Run from the repo root:  .venv/bin/python site/figures/fig_ch12.py
(needs numpy only; the H-matrix scaling numbers are quoted from docs/hmatrix.md.)
"""

import numpy as np
import matplotlib.pyplot as plt

from style import AMBER, CYAN, GREEN, MUTED, RED, save

WAVELENGTH = 22.0
K = 2 * np.pi / WAVELENGTH
A_THIN = 0.0005


def fig_svd():
    """Singular values of a far block collapse (secretly rank ~2); a diagonal
    block stays full. That gap is exactly what ACA peels off."""
    N = 400
    z = np.linspace(0, 4 * WAVELENGTH, N)

    def block(rows, cols):
        R = np.sqrt((z[rows][:, None] - z[cols][None, :]) ** 2 + A_THIN**2)
        return np.exp(-1j * K * R) / (4 * np.pi * R)

    far = np.linalg.svd(block(np.arange(0, 60), np.arange(320, 380)), compute_uv=False)
    near = np.linalg.svd(block(np.arange(0, 60), np.arange(0, 60)), compute_uv=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.semilogy(np.arange(1, 61), near / near[0], "s-", color=RED, lw=1.6, ms=3,
                label="diagonal block (near) — full rank")
    ax.semilogy(np.arange(1, 61), far / far[0], "o-", color=CYAN, lw=1.6, ms=4,
                label="far block — collapses to rank ~2")
    ax.axhline(1e-5, color=AMBER, lw=1.0, ls="--")
    ax.text(30, 1.6e-5, "ACA tolerance 1e-5", color=AMBER, fontsize=8.5)
    ax.set_ylim(1e-17, 3)
    ax.set_xlabel("singular value index")
    ax.set_ylabel("singular value  (normalized)")
    ax.set_title("A far block is secretly small: two numbers carry a 60×60 chunk")
    ax.legend(fontsize=9, loc="lower left")
    save(fig, "ch12-svd")


def fig_hmatrix():
    """Dense fill vs H-matrix build and storage, quoted from the momwire
    benchmark (docs/hmatrix.md): O(N log N) storage and build beat dense."""
    N = np.array([250, 500, 1000, 2000, 4000])
    dfill = np.array([0.11, 0.43, 1.54, 6.16, 25.01])   # dense fill (C), s
    hbuild = np.array([0.14, 0.28, 0.53, 1.05, 2.31])    # H-matrix build, s
    store = np.array([49.4, 30.9, 18.9, 11.1, 6.4])      # H store, % of dense

    fig, (axt, axs) = plt.subplots(1, 2, figsize=(11, 4.2))
    axt.loglog(N, dfill, "o-", color=RED, lw=2, ms=5, label="dense fill")
    axt.loglog(N, hbuild, "s-", color=GREEN, lw=2, ms=5, label="H-matrix build")
    axt.set_xlabel("number of unknowns N")
    axt.set_ylabel("build time  (s)")
    axt.set_title("Build: H-matrix pulls away (11× at N=4000)")
    axt.legend(fontsize=9, loc="upper left")
    axt.grid(True, which="both", alpha=0.2)

    axs.semilogx(N, store, "o-", color=CYAN, lw=2, ms=5)
    axs.set_xlabel("number of unknowns N")
    axs.set_ylabel("H-matrix storage  (% of dense)")
    axs.set_title("Storage halves on every mesh doubling — O(N log N)")
    axs.set_ylim(0, 55)
    axs.grid(True, which="both", alpha=0.2)
    for x, y in zip(N, store):
        axs.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=8, color=MUTED)
    save(fig, "ch12-hmatrix")


if __name__ == "__main__":
    fig_svd()
    fig_hmatrix()
