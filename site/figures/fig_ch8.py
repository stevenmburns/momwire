"""Figures for Act III chapter 8 — "Mirror worlds".

Run from the repo root:  .venv/bin/python site/figures/fig_ch8.py
(needs momwire importable — the repo venv.)
"""

import numpy as np
import matplotlib.pyplot as plt

from style import AMBER, CYAN, INK, MUTED, RED, save

from momwire import BSplineSolver

WAVELENGTH = 22.0
HD = 0.962 * WAVELENGTH / 4  # halfdriver 5.291 m
A_THIN = 0.0005


def fig_image():
    """The method of images: a horizontal dipole over a PEC plane radiates as
    if a mirror copy — with reversed current — sat the same distance below."""
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    # ground plane at z=0
    ax.axhline(0, color=INK, lw=2)
    ax.fill_between([-6, 6], -0.55, 0, color=MUTED, alpha=0.15, hatch="////", lw=0)
    ax.text(5.6, -0.35, "PEC ground", color=MUTED, ha="right", fontsize=10)

    h = 2.4
    # real dipole (above) with current arrow +x
    ax.plot([-4, 4], [h, h], color=CYAN, lw=3)
    ax.annotate("", xy=(2.6, h), xytext=(-2.6, h),
                arrowprops=dict(arrowstyle="-|>", color=CYAN, lw=2))
    ax.text(0, h + 0.35, "real dipole  (current →)", color=CYAN, ha="center", fontsize=10)
    ax.plot([0, 0], [0, h], color=MUTED, lw=0.8, ls=":")
    ax.text(0.2, h / 2, "h", color=MUTED, fontsize=11)
    # image dipole (below) with reversed current arrow -x
    ax.plot([-4, 4], [-h, -h], color=AMBER, lw=3, alpha=0.8)
    ax.annotate("", xy=(-2.6, -h), xytext=(2.6, -h),
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=2))
    ax.text(0, -h - 0.5, "image  (current ←, reversed)", color=AMBER, ha="center", fontsize=10)
    ax.plot([0, 0], [-h, 0], color=MUTED, lw=0.8, ls=":")
    ax.text(0.2, -h / 2, "h", color=MUTED, fontsize=11)

    ax.set_xlim(-6, 6)
    ax.set_ylim(-4.2, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("One extra kernel term: the PEC image, a mirror dipole with reversed current")
    save(fig, "ch8-image")


def fig_height():
    """Drive-point impedance of a horizontal half-wave dipole vs height over
    PEC ground: the classic oscillation, damping toward the free-space value."""
    hs = np.linspace(0.05, 1.25, 49)
    Z = []
    for hl in hs:
        wire = np.array([[-HD, 0, hl * WAVELENGTH], [HD, 0, hl * WAVELENGTH]])
        z, _ = BSplineSolver(wires=[wire], nsegs=21, wavelength=WAVELENGTH,
                             wire_radius=A_THIN, degree=2, ground_z=0.0,
                             feed_wire_index=0, feed_arclength=HD).compute_impedance()
        Z.append(z)
    Z = np.array(Z)

    wf = np.array([[-HD, 0, 0], [HD, 0, 0]])
    zf, _ = BSplineSolver(wires=[wf], nsegs=21, wavelength=WAVELENGTH,
                          wire_radius=A_THIN, degree=2,
                          feed_wire_index=0, feed_arclength=HD).compute_impedance()

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(hs, Z.real, color=CYAN, lw=2, label="R")
    ax.plot(hs, Z.imag, color=AMBER, lw=2, label="X")
    ax.axhline(zf.real, color=CYAN, lw=1.0, ls="--", alpha=0.6,
               label=f"free-space R = {zf.real:.0f}")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.annotate("h → 0: the image cancels the\ncurrent, R collapses",
                xy=(0.06, Z.real[0]), xytext=(0.22, 8), color=RED, fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1))
    ax.set_xlabel("height above ground  h / λ")
    ax.set_ylabel("impedance  (Ω)")
    ax.set_title("Horizontal dipole over PEC: impedance oscillates with height")
    ax.legend(fontsize=9, loc="upper right")
    save(fig, "ch8-height")


if __name__ == "__main__":
    fig_image()
    fig_height()
