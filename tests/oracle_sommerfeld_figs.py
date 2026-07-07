"""Pointwise oracles for the Sommerfeld-integral engine, hand-transcribed
from the NEC-2 theory manual (docs/nec2_theory_manual.pdf, figs 7-11,
pp. 46-49). See docs/sommerfeld-ground-plan.md Phase 0/1.

Each figure plots one interpolation surface I(R1, theta) (eqs 156-159:
the ground-remainder field component with the exp(-j k2 R1)/R1 factor and
the sin/cos(phi) azimuth factor removed) over R1 in [0, 1] wavelength and
theta in [0, 90] degrees, theta = atan((z + z') / rho), and prints the
Max/Min of the real and imaginary parts over the plotted mesh. Those
extrema are the literals below.

Normalization caveats for the Phase 1 test that consumes these:
  - The manual does not print the plot mesh density; extrema of a smooth
    surface sampled on a ~20x10 mesh can differ a few percent from the
    true extrema. Compare on a similar mesh and gate loosely (~5-10%).
  - Values are scanned-figure annotations for a unit current element at
    10 MHz with C1 = -j*omega*I*dl*mu0 / (4*pi*k2**2) (eq 123). If the
    computed surfaces disagree by a single global constant (unit-moment /
    length-unit convention), pin the constant against eq 123 once and
    document it in _sommerfeld.py -- but the constant must then be the
    SAME for all five figures, which is itself a strong check.
  - Real/imag split assumes NEC's e^{+j omega t} convention (momwire's
    own; see _ground_refl.py docstring).

Keys: (component, eps_r, sigma) -> {"re": (min, max), "im": (min, max)}.
Components: "IrhoV", "IzV" (vertical element), "IrhoH", "IphiH"
(horizontal element); the fifth field function needs no figure because
I_z^H = -cos(phi) * F_rho^V (eq 147). Ground constants: figs 7-10 are
eps_r=4, sigma=0.001 S/m; fig 11 is eps_r=16, sigma=0 (the low-loss
evanescent-wave stress case). All at 10 MHz.
"""

FIG_ORACLES = {
    # Fig 7: I_rho^V, eps_r=4, sigma=0.001, 10 MHz
    ("IrhoV", 4.0, 0.001): {"re": (-80.65, 0.0), "im": (-137.9, 0.0)},
    # Fig 8: I_z^V
    ("IzV", 4.0, 0.001): {"re": (-163.8, -16.31), "im": (-98.16, 219.9)},
    # Fig 9: I_rho^H
    ("IrhoH", 4.0, 0.001): {"re": (-74.07, -1.045), "im": (-121.3, 29.25)},
    # Fig 10: I_phi^H
    ("IphiH", 4.0, 0.001): {"re": (14.77, 102.1), "im": (-75.86, 109.8)},
    # Fig 11: I_rho^H, eps_r=16, sigma=0 (low-loss stress case)
    ("IrhoH", 16.0, 0.0): {"re": (-142.2, 32.85), "im": (-166.2, 84.85)},
}

FIG_FREQ_MHZ = 10.0
