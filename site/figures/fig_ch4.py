"""Figures for Act II chapter 4 — "Sinusoids, NEC's bet".

Run from the repo root:  .venv/bin/python site/figures/fig_ch4.py
(needs momwire importable — the repo venv.)
"""

import numpy as np
import matplotlib.pyplot as plt

from style import AMBER, CYAN, GREEN, MUTED, RED, save
from toy_solver import toy_dipole

from momwire import SinusoidalSolver, BSplineSolver

WAVELENGTH = 22.0
L = 2 * 0.962 * WAVELENGTH / 4
A_THIN = 0.0005


def fig_basis():
    """The three-term sinusoidal basis functions, extracted from momwire:
    each peaks on its home segment and tapers *continuously* to zero across its
    neighbours — the opposite of Act I's discontinuous boxes."""
    N = 9
    wire = np.array([[0.0, -L / 2, 0.0], [0.0, L / 2, 0.0]])
    solver = SinusoidalSolver(wires=[wire], nsegs=N, wavelength=WAVELENGTH,
                              wire_radius=A_THIN)
    _, alpha = solver.compute_impedance()
    sd = np.linspace(0, L, 800)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = [CYAN, AMBER, GREEN]
    for c, j in zip(colors, (3, 4, 5)):
        e = np.zeros(N, dtype=complex)
        e[j] = 1.0
        shp = np.real(solver.currents_at_knots(e, s_array=[sd])[0])
        shp = shp / np.abs(shp).max()
        ax.plot(sd - L / 2, shp, color=c, lw=2,
                label=f"basis fn on segment {j}")
    # segment boundaries, to show each fn spans its segment + two neighbours
    for e0 in np.linspace(-L / 2, L / 2, N + 1):
        ax.axvline(e0, color=MUTED, lw=0.5, alpha=0.35)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("position along the dipole z  (m)")
    ax.set_ylabel("basis function  (peak-normalized)")
    ax.set_title("Three-term basis: A + B·sin ks + C·cos ks per segment, continuous across joints")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(-0.35, 1.15)
    save(fig, "ch4-basis")


def fig_currents():
    """Five segments. The sinusoidal current already sits on the converged
    answer; the Act I pulse staircase, at the same N, is nowhere — continuity
    is the difference, made visible."""
    N = 5
    wire = np.array([[0.0, -L / 2, 0.0], [0.0, L / 2, 0.0]])

    sin_solver = SinusoidalSolver(wires=[wire], nsegs=N, wavelength=WAVELENGTH,
                                  wire_radius=A_THIN)
    z_sin, a_sin = sin_solver.compute_impedance()
    sd = np.linspace(0, L, 500)
    I_sin = np.abs(sin_solver.currents_at_knots(a_sin, s_array=[sd])[0])

    ref = BSplineSolver(wires=[wire], nsegs=81, wavelength=WAVELENGTH,
                        wire_radius=A_THIN, degree=2)
    z_ref, c_ref = ref.compute_impedance()
    I_ref = np.abs(ref.currents_at_knots(c_ref, s_array=[sd])[0])

    z_toy, I_toy, z_mid = toy_dipole(L, A_THIN, WAVELENGTH, N)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(sd - L / 2, I_ref * 1e3, color=MUTED, lw=3, alpha=0.6,
            label=f"converged reference   ({z_ref.real:.1f} {z_ref.imag:+.1f}j Ω)")
    ax.plot(sd - L / 2, I_sin * 1e3, color=CYAN, lw=2,
            label=f"sinusoidal, N=5   ({z_sin.real:.1f} {z_sin.imag:+.1f}j Ω)")
    ax.plot(z_mid, np.abs(I_toy) * 1e3, drawstyle="steps-mid", color=RED, lw=1.6,
            label=f"Act I pulses, N=5   ({z_toy.real:.0f} {z_toy.imag:+.0f}j Ω)")
    ax.set_xlabel("position along the dipole z  (m)")
    ax.set_ylabel("|I(z)|   (mA, 1 V drive)")
    ax.set_title("Five segments: the sinusoidal basis is already there")
    ax.legend(fontsize=9, loc="upper right")
    save(fig, "ch4-currents")


if __name__ == "__main__":
    fig_basis()
    fig_currents()
