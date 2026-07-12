"""Distributed series wire impedance: conductor loss + insulation loading.

Real antenna wire is not PEC. Two per-unit-length series effects matter at
HF (stevenmburns/momwire#131):

* **Conductor loss** — the internal impedance of a round conductor,
  Z'_int(ω) [Ω/m]. Exact closed form for a solid cylinder of radius a and
  conductivity σ:

      k_c    = sqrt(jωμσ)                    (complex wavenumber in the metal)
      Z'_int = k_c / (2π a σ) · I₀(k_c a) / I₁(k_c a)

  which recovers both limits: DC (|k_c a| → 0) → 1/(σ π a²), and strong
  skin effect (|k_c a| ≫ 1) → (1+j)/(2π a σ δ) with δ = sqrt(2/(ωμσ)).
  The I₀/I₁ ratio is evaluated with scipy's exponentially *scaled* Bessels
  (`ive`), whose common scale factor cancels in the ratio — the unscaled
  I's overflow for a/δ ≳ 350.

* **Insulation loading** — a dielectric jacket (inner radius a, outer
  radius b, relative permittivity εr) stores extra E-field energy near the
  wire, which acts as a distributed series inductance (King's insulated-
  antenna theory, quasi-static limit):

      L'_ins = μ₀/(2π) · (1 − 1/εr) · ln(b/a)      [H/m]

  This slows the guided wave on the wire — the familiar few-percent
  "insulated wire tunes long" velocity-factor effect. Dielectric loss
  (tan δ) is deliberately out of scope for now.

Both enter the MoM identically: a per-wire series impedance
Z'(ω) = Z'_int + jωL'_ins loading the impedance matrix over same-wire
basis overlaps (see `BSplineSolver._loading_gram`).
"""

import numpy as np
from scipy.special import ive

MU0 = 1.25663706127e-6


def wire_internal_impedance(omega, radius, sigma):
    """Per-unit-length internal impedance of a round solid conductor [Ω/m].

    Exact for all skin depths (DC through strong skin effect). `omega` may
    be a scalar or an array; the result broadcasts accordingly.
    """
    omega = np.asarray(omega, dtype=float)
    if radius <= 0.0:
        raise ValueError(f"wire radius must be > 0, got {radius}")
    if sigma <= 0.0:
        raise ValueError(f"conductivity must be > 0 S/m, got {sigma}")
    kc = np.sqrt(1j * omega * MU0 * sigma)
    z = kc * radius
    # ive(v, z) = iv(v, z)·exp(-|Re z|): the scale factor is common to both
    # orders, so the ratio equals I0/I1 exactly without overflow.
    ratio = ive(0, z) / ive(1, z)
    return kc / (2.0 * np.pi * radius * sigma) * ratio


def insulation_inductance(radius, ins_radius, eps_r):
    """Distributed series inductance of a dielectric jacket [H/m]."""
    if ins_radius <= radius:
        raise ValueError(
            f"insulation_radius ({ins_radius}) must exceed the conductor "
            f"radius ({radius})"
        )
    if eps_r < 1.0:
        raise ValueError(f"insulation_eps_r must be >= 1, got {eps_r}")
    return MU0 / (2.0 * np.pi) * (1.0 - 1.0 / eps_r) * np.log(ins_radius / radius)


def normalize_per_wire(value, n_wires, name):
    """None | scalar | length-n_wires sequence → None | (n_wires,) float array.

    A scalar applies to every wire; None disables the effect. Entries of a
    sequence may be NaN to disable the effect on individual wires.
    """
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n_wires, float(arr))
    if arr.shape != (n_wires,):
        raise ValueError(
            f"{name}: expected a scalar or a length-{n_wires} sequence "
            f"(one entry per wire), got shape {arr.shape}"
        )
    return arr


def series_impedance_per_wire(
    omega, wire_radius, conductivity, insulation_radius, insulation_eps_r
):
    """Per-wire distributed series impedance Z'(ω) [Ω/m].

    `conductivity` / `insulation_radius` / `insulation_eps_r` are the
    normalized (n_wires,) arrays (or None) from `normalize_per_wire`; NaN
    entries switch the effect off for that wire. `omega` may be scalar or
    (n_k,); the result is (n_wires,) or (n_wires, n_k) complex.
    """
    omega = np.asarray(omega, dtype=float)
    n_w = (
        conductivity.shape[0]
        if conductivity is not None
        else insulation_radius.shape[0]
    )
    out = np.zeros((n_w,) + omega.shape, dtype=np.complex128)
    for w in range(n_w):
        if conductivity is not None and np.isfinite(conductivity[w]):
            out[w] += wire_internal_impedance(omega, wire_radius, conductivity[w])
        if insulation_radius is not None and np.isfinite(insulation_radius[w]):
            L = insulation_inductance(
                wire_radius, insulation_radius[w], insulation_eps_r[w]
            )
            out[w] += 1j * omega * L
    return out
