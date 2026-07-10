"""Figures for Act I chapter 1 — "The question".

Run from the repo root:  .venv/bin/python site/figures/fig_ch1.py
"""

import numpy as np
import matplotlib.pyplot as plt

import style
from style import AMBER, CYAN, GRID, INK, MUTED, save

# The house specimen everywhere in Act I: the momwire default dipole.
WAVELENGTH = 22.0
L = 2 * 0.962 * WAVELENGTH / 4  # 10.582 m
A = 0.0005  # 0.5 mm wire radius


def fig_the_question():
    """The whole problem in one drawing: a wire, a gap, a voltage."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_axis_off()
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-2.2, 2.6)

    gap = 0.25
    # The two arms.
    for x0, x1 in [(-L / 2, -gap), (gap, L / 2)]:
        ax.plot([x0, x1], [0, 0], color=INK, lw=5, solid_capstyle="butt")
    # The generator in the gap.
    ax.annotate(
        "1 V",
        xy=(0, 0),
        xytext=(0, -1.55),
        ha="center",
        color=AMBER,
        fontsize=13,
        arrowprops=dict(arrowstyle="-", color=AMBER, lw=1.2),
    )
    ax.text(-0.45, 0.32, "−", color=AMBER, fontsize=15, ha="center")
    ax.text(0.45, 0.32, "+", color=AMBER, fontsize=15, ha="center")

    # The unknown: the current distribution I(z), drawn as the envelope the
    # solve will eventually produce (roughly a half sine).
    z = np.linspace(-L / 2, L / 2, 300)
    envelope = 1.6 * np.cos(np.pi * z / L)
    ax.plot(z, envelope, color=CYAN, lw=2)
    ax.fill_between(z, 0, envelope, color=CYAN, alpha=0.12, lw=0)
    ax.text(2.6, 1.55, "I(z)  — the unknown", color=CYAN, fontsize=12)

    ax.annotate(
        f"L = {L:.3f} m   (0.481 λ at λ = {WAVELENGTH:.0f} m)",
        xy=(-3.4, -0.75),
        ha="center",
        color=MUTED,
        fontsize=10,
    )
    ax.annotate(
        "radius a = 0.5 mm\n(21 000× thinner than it is long)",
        xy=(-L / 2 + 0.4, 0.15),
        xytext=(-5.6, 1.7),
        color=MUTED,
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color=GRID),
    )
    save(fig, "ch1-the-question")


def fig_kernel_floor():
    """The thin-wire trick: R = sqrt(d^2 + a^2) never falls below a, so the
    1/R of the free-space Green's function is capped instead of divergent."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    d = np.logspace(-5, 1, 400)  # source-to-match distance along the wire, m
    ax.loglog(
        d,
        1 / (4 * np.pi * d),
        color=style.RED,
        lw=1.6,
        ls="--",
        label=r"$1/4\pi|z-z'|$  (same filament: diverges)",
    )
    ax.loglog(
        d,
        1 / (4 * np.pi * np.sqrt(d**2 + A**2)),
        color=CYAN,
        lw=2,
        label=r"$1/4\pi R$,  $R=\sqrt{(z-z')^2+a^2}$  (thin-wire)",
    )
    ax.axvline(A, color=MUTED, lw=1, ls=":")
    ax.text(A * 1.25, 2.2e-1, "d = a", color=MUTED, fontsize=10)
    ax.axhline(1 / (4 * np.pi * A), color=MUTED, lw=1, ls=":")
    ax.text(2e-5, 1 / (4 * np.pi * A) * 0.55, r"cap: $1/4\pi a$", color=MUTED, fontsize=10)
    ax.set_xlabel("distance along the wire  |z − z′|   (m)")
    ax.set_ylabel("static part of the kernel   (1/m)")
    ax.set_title("Why the integral survives: the a² floor")
    ax.legend(loc="lower left", fontsize=10)
    save(fig, "ch1-kernel-floor")


if __name__ == "__main__":
    fig_the_question()
    fig_kernel_floor()
