"""Figures for Act III chapter 9 — "Real dirt, cheap".

Run from the repo root:  .venv/bin/python site/figures/fig_ch9.py
(needs momwire importable — the repo venv.)
"""

import numpy as np
import matplotlib.pyplot as plt

from style import AMBER, CYAN, GREEN, MUTED, RED, save

from momwire import BSplineSolver

WAVELENGTH = 22.0
FREQ = 299792458.0 / WAVELENGTH  # 13.627 MHz
HD = 0.962 * WAVELENGTH / 4
A_THIN = 0.0005
EPS_R, SIGMA = 13.0, 0.005  # "average ground"
EPS0 = 8.8541878188e-12


def _eps_tilde():
    """Complex relative permittivity of the ground at the operating frequency."""
    return EPS_R - 1j * SIGMA / (2 * np.pi * FREQ * EPS0)


def fig_gamma():
    """Fresnel reflection coefficient of average ground vs grazing angle —
    a partial, complex mirror, unlike PEC's perfect |Γ| = 1."""
    et = _eps_tilde()
    psi = np.radians(np.linspace(0.5, 90, 400))  # grazing angle from the surface
    root = np.sqrt(et - np.cos(psi) ** 2)
    gh = (np.sin(psi) - root) / (np.sin(psi) + root)
    gv = (et * np.sin(psi) - root) / (et * np.sin(psi) + root)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(1.0, color=MUTED, lw=1.2, ls="--", label="PEC: |Γ| = 1 (perfect)")
    ax.plot(np.degrees(psi), np.abs(gh), color=CYAN, lw=2, label="|Γ| horizontal pol.")
    ax.plot(np.degrees(psi), np.abs(gv), color=AMBER, lw=2, label="|Γ| vertical pol.")
    imin = np.argmin(np.abs(gv))
    ax.annotate("pseudo-Brewster dip:\nvertical pol. barely reflects",
                xy=(np.degrees(psi[imin]), np.abs(gv[imin])),
                xytext=(38, 0.55), color=AMBER, fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1))
    ax.set_xlabel("grazing angle above the ground  ψ  (degrees)")
    ax.set_ylabel("|Γ|  reflection coefficient magnitude")
    ax.set_title(f"Average ground (εr={EPS_R:.0f}, σ={SIGMA} S/m): a dim, complex mirror")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="lower right")
    save(fig, "ch9-gamma")


def _Z(h, **kw):
    wire = np.array([[-HD, 0, h], [HD, 0, h]])
    z, _ = BSplineSolver(wires=[wire], nsegs=21, wavelength=WAVELENGTH,
                         wire_radius=A_THIN, degree=2, feed_wire_index=0,
                         feed_arclength=HD, **kw).compute_impedance()
    return z


def fig_height():
    """Reflection-coefficient ground vs the exact Sommerfeld solve vs PEC:
    the cheap Fresnel image tracks Sommerfeld beautifully above ~0.2 λ, then
    goes tens of ohms wrong as the antenna nears the dirt."""
    hs = np.linspace(0.03, 0.6, 24)
    eps = (EPS_R, SIGMA)
    zp = np.array([_Z(h * WAVELENGTH, ground_z=0.0) for h in hs])
    zf = np.array([_Z(h * WAVELENGTH, ground_z=0.0, ground_eps=eps) for h in hs])
    zs = np.array([_Z(h * WAVELENGTH, ground_z=0.0, ground_eps=eps,
                      ground_model="sommerfeld") for h in hs])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(hs, zp.real, color=MUTED, lw=1.4, ls=":", label="PEC (perfect mirror)")
    ax.plot(hs, zs.real, color=GREEN, lw=2.6, alpha=0.8, label="Sommerfeld (exact)")
    ax.plot(hs, zf.real, color=CYAN, lw=1.8, label="Fresnel reflection coef. (cheap)")
    ax.axvspan(0.03, 0.1, color=RED, alpha=0.08, lw=0)
    ax.annotate("below ~0.1 λ the cheap\nmirror is tens of Ω wrong",
                xy=(0.05, zf[np.argmin(np.abs(hs - 0.05))].real),
                xytext=(0.16, 30), color=RED, fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1))
    ax.set_xlabel("height above ground  h / λ")
    ax.set_ylabel("R  (Ω)")
    ax.set_title("Fresnel tracks Sommerfeld — until the antenna gets close to the dirt")
    ax.legend(fontsize=9, loc="lower right")
    save(fig, "ch9-height")


if __name__ == "__main__":
    fig_gamma()
    fig_height()
