# U8 — the buried far field (momwire#570): registration

- **Plan of record:** antennaknobs `docs/plan-buried-scope-closure.md`, section
  "U8 — the buried far field".
- **Base:** momwire main a925d37 (0.56.0). Branch `u8-far-field`.
- **Registered** 2026-09-16, before any source change and before any run.
  Amendments go in before the runs they amend. A miss is reported as a miss.

## What is claimed, before measuring

The pattern of a buried current is the coefficient of e^{−jk_pR}/R in the
transmitted field. That coefficient is the stationary-phase value of the
below→above Sommerfeld integrals (`_sommerfeld_transmitted`'s V_T, U_T and
their five derivatives), taken at the saddle λ_s = k_p sinθ. At the saddle:

    γ_p → j k_p cosθ,   γ_m → j k_mz,   k_mz = √(k_m² − k_p² sin²θ)  (Im ≤ 0)
    ∂/∂ρ → −j k_p sinθ,  ∂/∂z → −j k_p cosθ,  ∂/∂z′ → +γ_m,  (1/ρ)∂/∂ρ → 0

and the six surfaces reduce to three angular factors on a unit moment,
E = C₁·(e^{−jk_pR}/R)·M with C₁ = −jωμ₀/4π and

    M_θ = T_h(θ)·(m_x cosφ + m_y sinφ) + T_v(θ)·m_z
    M_φ = T_e(θ)·(−m_x sinφ + m_y cosφ)

    T_e = 2γ_p/(γ_m + γ_p)                                      (= Fresnel t_TE)
    T_v = −2 k_p² sinθ γ_p/(k_m²γ_p + k_p²γ_m)
    T_h = cosθ·T_e − j k_p sin²θ (γ_m − γ_p)·2γ_p/(k_m²γ_p + k_p²γ_m)

each segment carrying e^{+jk_p sinθ (x cosφ + y sinφ)}·e^{−γ_m d}, d = depth
below the plane. The same three factors follow from reciprocity (a plane
wave from direction θ transmitted into the ground with the Fresnel t_TE and
t_TM coefficients, read at the source), which is the second derivation in
`proto_far.py`. Above-ground segments keep their direct + Fresnel-image
moments; the total is the sum. Nothing here touches the fill.

**What is NOT in the pattern.** The lateral wave and the critical-angle
structure are O(1/R²) at an observer in air, so they are not in the 1/R
coefficient. At θ = 90° every factor is zero, as the finite-ground image
pattern is (NEC prints −999.99 there). The NE/NH near-field item on
momwire#570 is phase 3 and stays out.

## Predictions (registered)

| id | gate | prediction |
|---|---|---|
| P-A | transversality: r̂·E from the 7a–7e forms at the saddle | \|r̂·E\|/\|E\| < 1e-12 at every θ ∈ [0°, 89.9°], both source kinds |
| P-B | ε̃ = 1 collapse: T_e → 1, T_h → cosθ, T_v → −sinθ, depth factor → e^{−jk_p cosθ d}; the transmitted moment equals the free-space moment | rel < 1e-12 on a random 20-segment moment set |
| P-C | saddle form == reciprocity/Fresnel form | rel < 1e-12 at every θ, soils A/B/C, 7 and 21 MHz |
| P-D | numerical transmitted integral (momwire's `_six_integrals_transmitted`, C++ contour) at R = 5, 10, 20, 40, 80 λ₀ along rays θ ∈ {10, 30, 50, 70, 80}°, HED at d = 0.15 m and VED at d = 0.6 m, soil A 7 MHz: R·e^{+jk_pR}·E_num vs C₁M | rel err at 80 λ₀ < 2e-3 for θ ≤ 70°; err halves per doubling of R (ratio in [1.6, 2.4]) for θ ≤ 50°; at 80° the residual is larger and named |
| P-E | empymod (independent oracle, `ht` chosen by a free-space calibration first) at R = 50 and 200 λ₀, θ ∈ {5, 20, 40, 60, 75, 85}°, φ ∈ {0, 30, 90}°, HED d = 0.15 m and VED d = 0.6 m, soils A/B/C at 7 and 21 MHz | E_θ and E_φ (complex, phase included) within 5e-3 at 50 λ₀ and 2e-3 at 200 λ₀ for θ ≤ 75°; the 85° rows within 2e-2 |
| P-F | grazing: each of \|T_e\|, \|T_h\|, \|T_v\| vanishes as θ → 90° | proportional to cosθ over the last degree (ratio at 89.9°/89.5° in [0.18, 0.22]) |

Later gates, registered here and run after the build:

| id | gate | prediction |
|---|---|---|
| P-G | every above-ground pattern fixture in momwire and antennaknobs | bit-identical (no buried segment → the new term is never evaluated) |
| P-H | NEC-5 x13 on the buried dipole (`bhd10`, d = 0.15 m, soil A, 7 MHz): momwire's pattern from momwire's own currents | shape within 0.5 dB over directions within 20 dB of the peak; absolute gain within 1.0 dB (the impedance difference of momwire#1027, −1.2 % of R, enters through P_in) |
| P-I | NEC-5 x13 on the buried-radial vertical (connected spelling): the pattern with the radials placed honestly vs imaged | the honest pattern moves off the imaged one by less than the #1341 note's 0.46 dB at the peak, and toward NEC-5's |

## Amendments (each written before the run it governs)

**A1, after P-A/B/C/F ran (2026-09-16).** P-A, P-B and P-F hit. **P-C
missed:** 4.5e-9 at θ = 89.99° (1e-14 below 85°), the same number on all
three factors and all six media. Mechanism, confirmed by substitution: the
saddle spelling evaluates γ_p = √(λ_s² − k_p²) from λ_s − k_p = k_p(sinθ − 1),
which loses eight digits at grazing; with γ_p = j·k_p·cosθ the two spellings
agree to 7.6e-16 everywhere. The miss stands as registered. Consequence for
the build: the kernel is written in the Fresnel spelling (k_pz = k_p cosθ,
k_mz = √(k_m² − k_p² sin²θ) on the decaying root), never from λ_s − k_p.

**A2, after P-D ran.** The halving ratios are 2.000 on every rung and angle
(an exact 1/R approach of momwire's numerical transmitted integrals to the
closed form; each integral took milliseconds on the C++ contour), and the
radial component falls as 1/R at the same rate. **The absolute bar missed:**
7.3e-3 at 80 λ₀ for the VED at 70° against 2e-3; the constant in front of
1/R grows toward grazing, which the registration did not price. The miss
stands. P-D′ (`gate_richardson.py`) is registered as the decisive form:
extrapolate in 1/R over (40, 80) λ₀ and require rel < 1e-4 for θ ≤ 70°.

**A3, after `gate_empymod.py calib` ran.** empymod is NOT an oracle at
50–200 λ₀ over real soil with any of its three Hankel transforms at default
kernel sampling: O(1) disagreement with the direct + Fresnel-image far field
(which `_far_readout._far_moments` reproduces to 5e-11), and the three
transforms disagree with each other. On the trivial (air over air) interface
every transform reads the free-space 1/(k_pR) residual exactly. The qwe
transform with a 6000-points-per-decade kernel grid reaches the expected
1/(k_pR) residual at 20 λ₀ (9.0e-3 against 8.0e-3). P-E as registered is
therefore withdrawn as unmeasurable, not hit, and P-E′
(`gate_empymod_ladder.py`) replaces it: rungs 20/40/80 λ₀ at 21 MHz, qwe
6000 ppd, an above-soil CONTROL with the image formula as reference quoted
beside every buried number, Richardson over (40, 80), bar 3e-3.

**A4, after `nec5_pattern_gate.py` ran (no amendment, the record).** P-H HIT
(0.005 dB shape, 0.03 dB absolute on `buried_dipole`); P-I HIT (the honest
readout moves −0.186 dB at the peak, toward NEC-5, landing 0.005 dB from it).
Numbers in `RESULTS.md`.
