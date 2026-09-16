"""U8 prototype: the far-zone limit of the transmitted family (momwire#570).

Two derivations of the same three angular factors, gated against each other
(P-C), against the algebra they must satisfy (P-A transversality, P-B the
ε̃ = 1 collapse, P-F grazing), and against momwire's own numerical
transmitted integrals on a range ladder (P-D). empymod is `gate_empymod.py`.

Conventions are momwire's: e^{+jωt}, air above z = 0 (k_p real), ground
below (k_m, Im ≤ 0), γ = √(λ² − k²) on `_sommerfeld._gamma`'s cuts, source
at depth d = −z′ > 0. E = C₁·e^{−jk_pR}/R·M with C₁ = −jωμ₀/4π.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from momwire._sommerfeld import _gamma
from momwire._sommerfeld_below import k_medium
from momwire._sommerfeld_transmitted import _six_integrals_transmitted

HERE = Path(__file__).resolve().parent
C0 = 299792458.0
EPS0 = 8.8541878128e-12
MU0 = 4e-7 * math.pi
SOILS = {"A": (13.0, 0.005), "B": (20.0, 0.03), "C": (5.0, 0.001), "one": (1.0, 0.0)}


def medium(freq_hz, eps_r, sigma):
    w = 2.0 * math.pi * freq_hz
    eps_t = complex(eps_r) - 1j * sigma / (w * EPS0)
    k_p = w / C0
    k_m = k_medium(eps_t, k_p)
    return eps_t, k_p, k_m, w


# ---------------------------------------------------------------------------
# Derivation 1: the saddle point of the 7a–7e surfaces
# ---------------------------------------------------------------------------


def saddle(theta, k_p, k_m):
    """(γ_p, γ_m, λ_s) at the stationary point λ_s = k_p sinθ."""
    lam = k_p * np.sin(theta) + 0j
    return _gamma(lam, k_p), _gamma(lam, k_m), lam


def factors_saddle(theta, k_p, k_m):
    """(T_e, T_h, T_v) from the surfaces at the saddle, unit moment."""
    g_p, g_m, lam = saddle(theta, k_p, k_m)
    s, c = np.sin(theta), np.cos(theta)
    a_v = 2.0 * g_p / (k_m * k_m * g_p + k_p * k_p * g_m)  # V_T's amplitude
    a_u = 2.0 * g_p / (g_m + g_p)  # U_T's amplitude
    t_e = a_u
    t_v = -(k_p * k_p) * s * a_v
    t_h = c * a_u - 1j * k_p * s * s * (g_m - g_p) * a_v
    return t_e, t_h, t_v


def cyl_components_saddle(theta, phi, k_p, k_m, kind):
    """(E_ρ, E_φ, E_z)/(C₁·g) straight from 7a–7e with the saddle
    substitutions — BEFORE any projection, so transversality is a test."""
    g_p, g_m, lam = saddle(theta, k_p, k_m)
    a_v = 2.0 * g_p / (k_m * k_m * g_p + k_p * k_p * g_m)
    a_u = 2.0 * g_p / (g_m + g_p)
    d_rho = -1j * lam  # ∂/∂ρ
    d_z = -g_p  # ∂/∂z
    d_zp = +g_m  # ∂/∂z′
    if kind == "VED":
        e_rho = d_rho * d_z * a_v
        e_phi = 0.0 * e_rho
        e_z = lam * lam * a_v  # (∂²/∂z² + k_p²) → λ²
    else:
        e_rho = np.cos(phi) * (d_rho * d_rho * a_v + a_u)  # (1/ρ)∂/∂ρ → 0
        e_phi = -np.sin(phi) * a_u
        e_z = -np.cos(phi) * d_rho * d_zp * a_v
    return e_rho, e_phi, e_z


# ---------------------------------------------------------------------------
# Derivation 2: reciprocity — a plane wave from θ transmitted into the ground
# ---------------------------------------------------------------------------


def factors_fresnel(theta, k_p, k_m):
    """(T_e, T_h, T_v) from the Fresnel transmission coefficients (E-field,
    non-magnetic media) read at the buried source: t_s = 2k_pz/(k_pz + k_mz),
    t_p = 2k_pz k_p k_m/(k_m² k_pz + k_p² k_mz); the transmitted in-plane
    polarisation θ̂_t = (cosθ_t cosφ, cosθ_t sinφ, −sinθ_t) with Snell's
    sinθ_t = (k_p/k_m) sinθ, cosθ_t = k_mz/k_m."""
    s, c = np.sin(theta), np.cos(theta)
    k_pz = k_p * c + 0j
    k_mz = np.sqrt(k_m * k_m - (k_p * s) ** 2 + 0j)
    k_mz = np.where(k_mz.imag > 0, -k_mz, k_mz)  # the decaying root
    t_s = 2.0 * k_pz / (k_pz + k_mz)
    t_p = 2.0 * k_pz * k_p * k_m / (k_m * k_m * k_pz + k_p * k_p * k_mz)
    cos_t = k_mz / k_m
    sin_t = (k_p / k_m) * s
    return t_s, t_p * cos_t, -t_p * sin_t


def depth_factor(theta, k_p, k_m, d):
    """e^{−γ_m(λ_s)·d}: the in-medium leg of the transmitted wave."""
    _, g_m, _ = saddle(theta, k_p, k_m)
    return np.exp(-g_m * d)


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------


def transmitted_moments(mid, moment, k_p, k_m, theta, phi, ground_z, form="saddle"):
    """(M_θ, M_φ) on the θ×φ grid for segments BELOW the plane."""
    f = factors_saddle if form == "saddle" else factors_fresnel
    t_e, t_h, t_v = (x[:, None] for x in f(theta, k_p, k_m))
    s = np.sin(theta)[:, None]
    cp, sp = np.cos(phi)[None, :], np.sin(phi)[None, :]
    m_th = np.zeros((theta.size, phi.size), dtype=complex)
    m_ph = np.zeros_like(m_th)
    for r, m in zip(mid, moment):
        d = ground_z - r[2]
        assert d > 0.0
        ph = (
            np.exp(1j * k_p * s * (r[0] * cp + r[1] * sp))
            * depth_factor(theta, k_p, k_m, d)[:, None]
        )
        m_th += ph * (t_h * (m[0] * cp + m[1] * sp) + t_v * m[2])
        m_ph += ph * t_e * (-m[0] * sp + m[1] * cp)
    return m_th, m_ph


def free_space_moments(mid, moment, k, theta, phi):
    """The direct far-field moment, projected on θ̂ and φ̂."""
    s, c = np.sin(theta)[:, None], np.cos(theta)[:, None]
    cp, sp = np.cos(phi)[None, :], np.sin(phi)[None, :]
    rhat = np.stack([s * cp, s * sp, np.broadcast_to(c, (theta.size, phi.size))], -1)
    th = np.stack([c * cp, c * sp, -np.broadcast_to(s, rhat.shape[:2])], -1)
    ph = np.stack(
        [
            -np.broadcast_to(sp, rhat.shape[:2]),
            np.broadcast_to(cp, rhat.shape[:2]),
            np.zeros(rhat.shape[:2]),
        ],
        -1,
    )
    phase = k * np.einsum("ijc,nc->ijn", rhat, mid)
    M = np.einsum("ijn,nc->ijc", np.exp(1j * phase), moment)
    return np.sum(M * th, -1), np.sum(M * ph, -1)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

RESULTS = {}


def report(gid, ok, detail):
    RESULTS[gid] = {"ok": bool(ok), **detail}
    print(f"[{'HIT ' if ok else 'MISS'}] {gid}: {detail}")


def gate_A():
    """Transversality of the raw 7a–7e components at the saddle."""
    worst = 0.0
    theta = np.deg2rad(np.linspace(0.0, 89.9, 900))
    for soil in ("A", "B", "C"):
        for f in (7e6, 21e6):
            _, k_p, k_m, _ = medium(f, *SOILS[soil])
            for kind in ("HED", "VED"):
                for phi in (0.0, 0.7, 1.4):
                    e_rho, e_phi, e_z = cyl_components_saddle(
                        theta, phi, k_p, k_m, kind
                    )
                    e_r = e_rho * np.sin(theta) + e_z * np.cos(theta)
                    mag = np.sqrt(abs(e_rho) ** 2 + abs(e_phi) ** 2 + abs(e_z) ** 2)
                    worst = max(worst, float(np.max(abs(e_r) / mag)))
    report("P-A", worst < 1e-12, {"max |r.E|/|E|": worst})


def gate_B():
    """ε̃ = 1: the transmitted moment IS the free-space moment."""
    _, k_p, k_m, _ = medium(7e6, 1.0, 0.0)
    assert k_m == k_p
    rng = np.random.default_rng(570)
    mid = rng.uniform(-3, 3, (20, 3))
    mid[:, 2] = -rng.uniform(0.05, 2.0, 20)
    moment = rng.normal(size=(20, 3)) + 1j * rng.normal(size=(20, 3))
    theta = np.deg2rad(np.linspace(0.0, 89.0, 90))
    phi = np.deg2rad(np.linspace(0.0, 360.0, 73))
    a = transmitted_moments(mid, moment, k_p, k_m, theta, phi, 0.0)
    b = free_space_moments(mid, moment, k_p, theta, phi)
    scale = max(np.max(abs(b[0])), np.max(abs(b[1])))
    rel = max(np.max(abs(a[0] - b[0])), np.max(abs(a[1] - b[1]))) / scale
    t_e, t_h, t_v = factors_saddle(theta, k_p, k_m)
    fac = max(
        np.max(abs(t_e - 1.0)),
        np.max(abs(t_h - np.cos(theta))),
        np.max(abs(t_v + np.sin(theta))),
    )
    report(
        "P-B",
        rel < 1e-12 and fac < 1e-12,
        {"moment rel": float(rel), "factor abs": float(fac)},
    )


def gate_C():
    """Saddle form == reciprocity/Fresnel form."""
    worst = 0.0
    theta = np.deg2rad(np.linspace(0.0, 89.99, 2000))
    for soil in ("A", "B", "C"):
        for f in (7e6, 21e6):
            _, k_p, k_m, _ = medium(f, *SOILS[soil])
            a = factors_saddle(theta, k_p, k_m)
            b = factors_fresnel(theta, k_p, k_m)
            for x, y in zip(a, b):
                worst = max(
                    worst, float(np.max(abs(x - y) / np.maximum(abs(y), 1e-300)))
                )
    report("P-C", worst < 1e-12, {"max rel": worst})


def gate_F():
    """Grazing: every factor vanishes like cosθ."""
    _, k_p, k_m, _ = medium(7e6, *SOILS["A"])
    th = np.deg2rad(np.array([89.5, 89.9]))
    ratios = []
    for t in factors_saddle(th, k_p, k_m):
        ratios.append(float(abs(t[1]) / abs(t[0])))
    expected = math.cos(th[1]) / math.cos(th[0])
    ok = all(0.18 <= r <= 0.22 for r in ratios)
    report("P-F", ok, {"ratios 89.9/89.5": ratios, "cos ratio": expected})


def numeric_far(k_p, k_m, eps_t, R, theta, phi, d, kind, w):
    """R·e^{+jk_pR}·(E_θ, E_φ) from momwire's six numerical transmitted
    integrals, assembled through 7a–7e. Returns also the self-convergence."""
    rho, z = R * math.sin(theta), R * math.cos(theta)
    vals = _six_integrals_transmitted(
        eps_t, k_p, rho, z, -d, rtol=1e-11, selfconv=False
    )
    c1 = -1j * w * MU0 / (4.0 * math.pi)
    if kind == "VED":
        e_rho, e_phi, e_z = c1 * vals[0], 0j, c1 * vals[1]
    else:
        e_rho = c1 * math.cos(phi) * (vals[2] + vals[5])
        e_phi = -c1 * math.sin(phi) * (vals[3] + vals[5])
        e_z = -c1 * math.cos(phi) * vals[4]
    e_th = e_rho * math.cos(theta) - e_z * math.sin(theta)
    e_r = e_rho * math.sin(theta) + e_z * math.cos(theta)
    g = np.exp(-1j * k_p * R) / R
    return e_th / g, e_phi / g, e_r / g


def gate_D():
    """Range ladder: numerical integral → closed form as 1/R."""
    _, k_p, k_m, w = medium(7e6, *SOILS["A"])
    eps_t = medium(7e6, *SOILS["A"])[0]
    lam0 = 2.0 * math.pi / k_p
    c1 = -1j * w * MU0 / (4.0 * math.pi)
    rows = []
    worst80, ratios = 0.0, []
    for kind, d in (("HED", 0.15), ("VED", 0.6)):
        phi = 0.6 if kind == "HED" else 0.0
        mid = np.array([[0.0, 0.0, -d]])
        mom = (
            np.array([[1.0, 0.0, 0.0]])
            if kind == "HED"
            else np.array([[0.0, 0.0, 1.0]])
        )
        for th_deg in (10.0, 30.0, 50.0, 70.0, 80.0):
            th = math.radians(th_deg)
            m_th, m_ph = transmitted_moments(
                mid, mom, k_p, k_m, np.array([th]), np.array([phi]), 0.0
            )
            closed = np.array([c1 * m_th[0, 0], c1 * m_ph[0, 0]])
            errs = []
            for n in (5, 10, 20, 40, 80):
                R = n * lam0
                t0 = time.time()
                e_th, e_ph, e_r = numeric_far(k_p, k_m, eps_t, R, th, phi, d, kind, w)
                num = np.array([e_th, e_ph])
                rel = float(np.linalg.norm(num - closed) / np.linalg.norm(closed))
                errs.append(rel)
                rows.append(
                    {
                        "kind": kind,
                        "theta": th_deg,
                        "R_lam0": n,
                        "rel": rel,
                        "r_over_E": float(abs(e_r) / np.linalg.norm(closed)),
                        "secs": time.time() - t0,
                    }
                )
                print(
                    f"  {kind} θ={th_deg:4.0f} R={n:3d}λ₀ rel={rel:.3e} |E_r|/|E|={abs(e_r) / np.linalg.norm(closed):.2e} ({time.time() - t0:.1f}s)"
                )
            if th_deg <= 70.0:
                worst80 = max(worst80, errs[-1])
            if th_deg <= 50.0:
                ratios += [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    ok = worst80 < 2e-3 and all(1.6 <= r <= 2.4 for r in ratios)
    report(
        "P-D",
        ok,
        {
            "worst rel at 80 lam0, theta<=70": worst80,
            "halving ratios theta<=50": ratios,
        },
    )
    (HERE / "p_d_ladder.json").write_text(json.dumps(rows, indent=1))


def main(argv=None):
    argv = argv or sys.argv[1:]
    gates = {"A": gate_A, "B": gate_B, "C": gate_C, "F": gate_F, "D": gate_D}
    for g in argv or ["A", "B", "C", "F", "D"]:
        gates[g]()
    (HERE / "proto_results.json").write_text(json.dumps(RESULTS, indent=1))


if __name__ == "__main__":
    main()
