# U8 — the buried far field (momwire#570): results

Registration and amendments: `PLAN.md`. Every number below is from a file in
this directory (`proto_results.json`, `p_d_ladder.json`, `p_d_richardson.json`,
`p_e_calib.json`, `p_e_ladder.json`, `p_h_i_nec5.json`, and the `.log` beside
each). Base: momwire main a925d37 (0.56.0); antennaknobs main 5de6cdde9
(v0.79.0); soil A = (13, 0.005) unless stated.

## The prototype gates

| id | gate | registered bar | measured | verdict |
|---|---|---|---|---|
| P-A | transversality of the 7a–7e components at the saddle | < 1e-12 | 2.5e-14 | HIT |
| P-B | ε̃ = 1 collapse to the free-space moment | < 1e-12 | 8.7e-16 (moment), 2.2e-16 (factors) | HIT |
| P-C | saddle spelling == Fresnel spelling | < 1e-12 at every θ | 1.1e-14 below 85°; **4.5e-9 at 89.99°** | **MISS** (A1: the saddle spelling's 1 − sinθ cancellation; 7.6e-16 with γ_p = j k_p cosθ) |
| P-D | numerical transmitted integral → closed form, R = 5…80 λ₀ | rel < 2e-3 at 80 λ₀ for θ ≤ 70°; halving ratio in [1.6, 2.4] | halving ratios **2.000** on every rung and angle; **7.3e-3 at 80 λ₀, VED 70°** | **MISS on the bar, HIT on the law** (A2) |
| P-D′ | Richardson in 1/R over (40, 80) λ₀ | rel < 1e-4 for θ ≤ 70° | 8.2e-7 … 8.6e-6 for θ ≤ 50°; **1.4e-4 at 70°**; second order 2.0e-5 at 70°, 6.4e-9 at 10° | **MISS by 1.4× at 70°**; the sequence converges to the closed form |
| P-E | empymod at 50–200 λ₀ | 5e-3 / 2e-3 | withdrawn (A3): the oracle is not converged there | — |
| P-E′ | empymod qwe/6000 ppd on 20/40/80 λ₀ with an above-soil CONTROL | control and subject < 3e-3 after Richardson | **control 2.2–3.1 (O(1)) on every soil**; subject 6e-2–3e-1 | **MISS on the control** = the oracle is disqualified, not the formula |
| P-F | grazing ∝ cosθ | ratio 89.9°/89.5° in [0.18, 0.22] | 0.200, 0.206, 0.206 (cos ratio 0.200) | HIT |

What the misses mean. P-C is conditioning: the build uses the Fresnel
spelling (k_pz = k_p cosθ, k_mz = √(k_m² − k_p² sin²θ)), which agrees with the
saddle form to 1e-14 wherever the saddle form is well conditioned. P-D and
P-D′ are bars set too tight for the angle dependence of the next-order
term: the numerical integrals approach the closed form as an exact 1/R law
at every angle, and the extrapolated sequence lands on it to 1e-5–1e-8. P-E′
says empymod (2.6.0) cannot be used as a far-zone oracle over real soil at
these ranges with any of its Hankel transforms: on the air-over-air control
every transform reads the exact free-space 1/(k_pR) residual, but with the
soil present the above-source control (reference = the direct + Fresnel
image far field, which `_far_readout._far_moments` reproduces to 5.2e-11)
is O(1) wrong, and the three transforms disagree with each other. Its
buried subject is closer (1.3e-2 at 20 λ₀ against an expected ~1e-2) but
cannot be quoted as a gate when its control fails.

## The cross-engine gate (NEC-5 x13, `nec5_pattern_gate.py`)

momwire's own currents at each catalog design's defaults, θ every 5°, φ
every 15°, gain normalised by input power as the app does. "Lit" = within
20 dB of NEC-5's peak.

| design | NEC-5 peak | closed form peak | Δ at NEC-5's peak | max \|Δ\| lit | shape (Δ − mean Δ) lit | Z momwire / NEC-5 |
|---|---|---|---|---|---|---|
| `buried_dipole` (wholly buried, 0.15 m) | −21.460 dBi | −21.427 dBi | +0.033 dB | 0.036 dB (450 directions) | 0.005 dB | 146.79+45.82j / 146.39+44.38j |
| `buried_radial_vertical` (connected, 89.5 % of Σ\|I·dl\| below) | −2.900 dBi | −2.904 dBi | −0.005 dB | 0.009 dB (400 directions) | 0.007 dB | 78.13+46.34j / 77.81+44.47j |

- **P-H HIT** (bar: shape 0.5 dB, absolute 1.0 dB): 0.005 dB shape, 0.03 dB
  absolute. The 0.03 dB is the size of the impedance difference's effect
  through P_in (momwire#1027's class), as registered.
- **P-I HIT** (bar: the honest pattern moves off the imaged one by less than
  the #1341 note's 0.46 dB, and toward NEC-5): the honest peak is
  **−0.186 dB** from the imaged readout's (−2.718 → −2.904 dBi; 0.229 dB is
  the largest move in the lit hemisphere), and the imaged readout sat
  +0.18 dB ABOVE NEC-5 while the honest one sits 0.005 dB from it.

NEC-5 is an independent implementation of the same physics; it is the
oracle P-E was meant to be, and it agrees to hundredths of a dB.

## What this settles

The far-field pattern of a buried current is the transmitted plane wave,
closed form, with the Fresnel spelling as the well-conditioned one; the
lateral wave and the critical-angle structure are 1/R² and not in a
pattern. The build (momwire `u8-far-field`, antennaknobs
`u8-buried-pattern`) carries these factors into `_far_readout` and both
antennaknobs readouts; P-G (above-ground bit-identity) is the build's own
gate. The NE/NH item on momwire#570 (phase 3) is untouched.
