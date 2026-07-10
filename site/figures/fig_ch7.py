"""Figures for Act II chapter 7 — "How do you know it's right?".

Run from the repo root:  .venv/bin/python site/figures/fig_ch7.py
(needs momwire importable — the repo venv.)
"""

import numpy as np
import matplotlib.pyplot as plt

from style import AMBER, CYAN, GREEN, MUTED, RED, save
from toy_solver import toy_dipole

from momwire import SinusoidalSolver, BSplineSolver

WAVELENGTH = 22.0
HD = 0.962 * WAVELENGTH / 4  # halfdriver = 5.291 m
L = 2 * HD
A_THIN = 0.0005

# NEC-2 reference values (docs/convergence_analysis.md, PyNEC, N=101).
NEC_DIPOLE = 69.64 - 18.21j
NEC_YAGI = 77.28 + 6.74j


def _dipole(cls, N, **kw):
    wire = np.array([[0.0, -HD, 0.0], [0.0, HD, 0.0]])
    z, _ = cls(wires=[wire], nsegs=N, wavelength=WAVELENGTH, wire_radius=A_THIN,
               feed_wire_index=0, feed_arclength=HD, **kw).compute_impedance()
    return z


def _yagi(cls, N, **kw):
    driver = np.array([[0.0, -HD, 0.0], [0.0, HD, 0.0]])
    refl = np.array([[-HD, -1.05 * HD, 0.0], [-HD, 1.05 * HD, 0.0]])
    z, _ = cls(wires=[driver, refl], nsegs=N, wavelength=WAVELENGTH, wire_radius=A_THIN,
               feed_wire_index=0, feed_arclength=HD, **kw).compute_impedance()
    return z


def fig_knee():
    """Internal check: the impedance stops moving. momwire's good bases hit a
    knee by ~7 segments and sit on NEC; the Act I pulse toy never does."""
    Ns = np.array([3, 5, 7, 11, 21, 41])
    z_sin = np.array([_dipole(SinusoidalSolver, n) for n in Ns])
    z_bsp = np.array([_dipole(BSplineSolver, n, degree=2) for n in Ns])
    Nt = np.array([11, 21, 41, 81, 161, 321])
    z_toy = np.array([toy_dipole(L, A_THIN, WAVELENGTH, int(n))[0] for n in Nt])

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
    for ax, part, ref, lbl in [(axes[0], np.real, NEC_DIPOLE.real, "R  (Ω)"),
                               (axes[1], np.imag, NEC_DIPOLE.imag, "X  (Ω)")]:
        ax.axhline(ref, color=MUTED, lw=1.4, ls="--", label=f"NEC-2:  {ref:.1f}")
        ax.plot(Ns, part(z_sin), "o-", color=CYAN, lw=1.6, ms=5, label="sinusoidal")
        ax.plot(Ns, part(z_bsp), "s-", color=GREEN, lw=1.6, ms=5, label="B-spline d=2")
        ax.plot(Nt, part(z_toy), "^:", color=RED, lw=1.2, ms=4, label="Act I pulses")
        ax.set_xscale("log")
        ax.set_xticks([3, 5, 10, 20, 40, 80, 160, 320])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_ylabel(lbl)
    axes[0].set_ylim(66, 92)
    axes[1].set_ylim(-30, 60)
    axes[0].annotate("the knee", xy=(7, 69.9), xytext=(11, 82), color=CYAN, fontsize=9,
                     arrowprops=dict(arrowstyle="->", color=CYAN, lw=1))
    axes[0].set_title("The knee: momwire's bases stop moving; the pulse toy keeps crawling")
    axes[1].set_xlabel("number of segments N")
    axes[0].legend(fontsize=8.5, loc="upper right")
    save(fig, "ch7-knee")


def fig_validation():
    """External check: momwire − NEC residuals, two antennas × two bases.
    Everything lands inside a few tenths of an ohm."""
    rows = [
        ("dipole R", NEC_DIPOLE.real, _dipole(SinusoidalSolver, 41).real, _dipole(BSplineSolver, 41, degree=2).real),
        ("dipole X", NEC_DIPOLE.imag, _dipole(SinusoidalSolver, 41).imag, _dipole(BSplineSolver, 41, degree=2).imag),
        ("Yagi R", NEC_YAGI.real, _yagi(SinusoidalSolver, 41).real, _yagi(BSplineSolver, 41, degree=2).real),
        ("Yagi X", NEC_YAGI.imag, _yagi(SinusoidalSolver, 41).imag, _yagi(BSplineSolver, 41, degree=2).imag),
    ]
    labels = [r[0] for r in rows]
    d_sin = [r[2] - r[1] for r in rows]
    d_bsp = [r[3] - r[1] for r in rows]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.barh(y - 0.2, d_sin, height=0.36, color=CYAN, label="sinusoidal − NEC")
    ax.barh(y + 0.2, d_bsp, height=0.36, color=GREEN, label="B-spline d=2 − NEC")
    ax.axvline(0, color=MUTED, lw=1.0)
    for lim in (-0.2, 0.2):
        ax.axvline(lim, color=AMBER, lw=0.8, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("residual vs NEC-2  (Ω)   ·   dotted bands = ±0.2 Ω")
    ax.set_xlim(-0.35, 0.35)
    ax.set_title("Two engines, two antennas: agreement to a fraction of an ohm")
    ax.legend(fontsize=9, loc="lower right")
    save(fig, "ch7-validation")


if __name__ == "__main__":
    fig_knee()
    fig_validation()
