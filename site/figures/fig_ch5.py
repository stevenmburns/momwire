"""Figures for Act II chapter 5 — "Splines and junctions".

Run from the repo root:  .venv/bin/python site/figures/fig_ch5.py
(needs momwire importable — the repo venv.)
"""

import numpy as np
import matplotlib.pyplot as plt

from style import AMBER, CYAN, GREEN, MUTED, RED, save

from momwire import BSplineSolver

WAVELENGTH = 22.0
L = 2 * 0.962 * WAVELENGTH / 4
A_THIN = 0.0005


def _basis_fn(degree, N, j, sd):
    """Peak-normalized shape of B-spline basis function j on a straight wire."""
    wire = np.array([[0.0, -L / 2, 0.0], [0.0, L / 2, 0.0]])
    solver = BSplineSolver(wires=[wire], nsegs=N, wavelength=WAVELENGTH,
                           wire_radius=A_THIN, degree=degree)
    _, c = solver.compute_impedance()
    e = np.zeros(len(np.asarray(c)), dtype=complex)
    e[j] = 1.0
    shp = np.real(solver.currents_at_knots(e, s_array=[sd])[0])
    return shp / np.abs(shp).max()


def fig_basis():
    """The smoothness ladder: pulse (d0, cliffs) → tent (d1, kink) → quadratic
    (d2, smooth). Each rung is more continuous and converges faster."""
    N = 9
    dz = L / N
    sd = np.linspace(0, L, 900)
    x = sd - L / 2

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    # d0 pulse: a single box on the center segment (drawn analytically)
    c0 = 0.0
    box = np.where(np.abs(x - c0) <= dz / 2, 1.0, 0.0)
    ax.plot(x, box, color=RED, lw=1.8, label="d=0  pulse  (discontinuous — Act I)")
    # d1 tent and d2 quadratic, extracted from momwire (center basis fn)
    t1 = _basis_fn(1, N, N // 2, sd)
    ax.plot(x, t1, color=AMBER, lw=2, label="d=1  tent  (continuous value)")
    t2 = _basis_fn(2, N, N // 2, sd)
    ax.plot(x, t2, color=CYAN, lw=2, label="d=2  quadratic  (continuous slope)")

    for e0 in np.linspace(-L / 2, L / 2, N + 1):
        ax.axvline(e0, color=MUTED, lw=0.5, alpha=0.3)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.25, 1.2)
    ax.set_xlabel("position along the wire z  (m)")
    ax.set_ylabel("basis function  (peak-normalized)")
    ax.set_title("The smoothness ladder: each rung is more continuous, and converges faster")
    ax.legend(fontsize=9, loc="upper right")
    save(fig, "ch5-basis")


def fig_junction():
    """KCL at a wire junction: a T antenna's downlead current splits equally
    into its two top arms. Same B-spline basis, now on a branched geometry."""
    down = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, 0.0]])   # feed near the bottom
    left = np.array([[0.0, 0.0, 0.0], [-2.6, 0.0, 0.0]])
    right = np.array([[0.0, 0.0, 0.0], [2.6, 0.0, 0.0]])
    junc = [[(0, "end"), (1, "start"), (2, "start")]]
    s = BSplineSolver(wires=[down, left, right], nsegs=21, wavelength=WAVELENGTH,
                      wire_radius=A_THIN, degree=2, junctions=junc,
                      feed_wire_index=0, feed_arclength=0.25)
    _, c = s.compute_impedance()

    sd_down = np.linspace(0.0, 5.0, 300)
    sd_arm = np.linspace(0.0, 2.6, 160)
    Id = np.abs(s.currents_at_knots(c, s_array=[sd_down, np.array([0.03]), np.array([0.03])])[0])
    arr = s.currents_at_knots(c, s_array=[np.array([4.97]), sd_arm, sd_arm])
    Il, Ir = np.abs(arr[1]), np.abs(arr[2])
    # values at the node for the KCL callout
    node = s.currents_at_knots(c, s_array=[np.array([4.97]), np.array([0.03]), np.array([0.03])])
    I_d, I_l, I_r = abs(node[0][0]) * 1e3, abs(node[1][0]) * 1e3, abs(node[2][0]) * 1e3

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    # x = signed distance from the node: downlead on the left, arms on the right
    ax.plot(-(5.0 - sd_down), Id * 1e3, color=CYAN, lw=2, label="downlead")
    ax.plot(sd_arm, Il * 1e3, color=AMBER, lw=2, label="top arm (each)")
    ax.plot(sd_arm, Ir * 1e3, color=AMBER, lw=2, ls=(0, (1, 1)))
    ax.axvline(0, color=GREEN, lw=1.0, ls=":")
    ax.annotate(
        f"junction (KCL):\n{I_d:.2f} = {I_l:.2f} + {I_r:.2f} mA",
        xy=(0, I_d), xytext=(-3.4, I_d * 0.55), color=GREEN, fontsize=9,
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1))
    ax.set_xlabel("distance from the junction node  (m)   ·   downlead ← | → top arms")
    ax.set_ylabel("|I(s)|   (mA, 1 V drive)")
    ax.set_title("A T-antenna: the downlead current splits at the junction")
    ax.legend(fontsize=9, loc="upper right")
    save(fig, "ch5-junction")


if __name__ == "__main__":
    fig_basis()
    fig_junction()
