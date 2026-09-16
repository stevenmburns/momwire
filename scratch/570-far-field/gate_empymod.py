"""P-E: empymod as the independent far-range oracle for the buried far field.

`calib` first: which Hankel transform empymod trusts at R = 50–200 λ₀ in the
wave regime, measured on two configurations with known answers — a trivial
interface (air over air) against the analytic Hertzian dipole, and a source
ABOVE real soil against the direct + Fresnel-image far field momwire's
readout already serves. Then `run`: the buried source against the closed
form of `proto_far.py`.

empymod frame (from scratch/524-phase0/empymod/harness.py, verified there):
z positive DOWN, depth = [0], layer 0 = air (res 2e14), layer 1 = soil;
x, y unchanged; HED along +x: (Ex, Ey, Ez)_spec = (+11, +21, −31);
VED along +z_spec (up): (−13, −23, +33). No conjugation.
"""

from __future__ import annotations

import json
import math
import sys
import time
import warnings
from pathlib import Path

import empymod
import numpy as np

import proto_far as p

HERE = Path(__file__).resolve().parent
RES_AIR = 2e14
COMPONENT_MAP = {
    "HED": {"Ex": (11, +1.0), "Ey": (21, +1.0), "Ez": (31, -1.0)},
    "VED": {"Ex": (13, -1.0), "Ey": (23, -1.0), "Ez": (33, +1.0)},
}
HT = {
    "quad": dict(
        ht="quad", htarg={"a": 1e-8, "b": 300.0, "limit": 8000, "pts_per_dec": 600}
    ),
    "qwe": dict(
        ht="qwe",
        htarg={
            "rtol": 1e-12,
            "atol": 1e-30,
            "nquad": 51,
            "maxint": 400,
            "pts_per_dec": 0,
        },
    ),
    "dlf": dict(ht="dlf", htarg={}),
}
THETAS = (5.0, 20.0, 40.0, 60.0, 75.0, 85.0)
PHIS = (0.0, 30.0, 90.0)


def e_spec(src_spec, src_type, obs_spec, freq, eps_r, sigma, settings):
    """E (n, 3) in SPEC coordinates at obs points for a unit dipole."""
    res = [RES_AIR, 1.0 / sigma if sigma > 0 else RES_AIR]
    eperm = [1.0, eps_r]
    obs = np.asarray(obs_spec, float)
    out = np.zeros((len(obs), 3), complex)
    src_emp = [src_spec[0], src_spec[1], -src_spec[2]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for z in sorted(set(obs[:, 2])):
            idx = np.where(obs[:, 2] == z)[0]
            for ci, comp in enumerate(("Ex", "Ey", "Ez")):
                ab, sign = COMPONENT_MAP[src_type][comp]
                v = empymod.dipole(
                    src=src_emp,
                    rec=[obs[idx, 0], obs[idx, 1], -z],
                    depth=[0.0],
                    res=res,
                    freqtime=freq,
                    ab=ab,
                    epermH=eperm,
                    xdirect=True,
                    verb=0,
                    **settings,
                )
                out[idx, ci] = sign * np.atleast_1d(np.asarray(v, complex))
    return out


def spherical(theta, phi):
    s, c = math.sin(theta), math.cos(theta)
    cp, sp = math.cos(phi), math.sin(phi)
    rhat = np.array([s * cp, s * sp, c])
    th = np.array([c * cp, c * sp, -s])
    ph = np.array([-sp, cp, 0.0])
    return rhat, th, ph


def directions():
    return [(math.radians(t), math.radians(f)) for t in THETAS for f in PHIS]


def closed_form_buried(kind, d, k_p, k_m, w, theta, phi):
    c1 = -1j * w * p.MU0 / (4.0 * math.pi)
    mid = np.array([[0.0, 0.0, -d]])
    mom = np.array([[1.0, 0.0, 0.0]]) if kind == "HED" else np.array([[0.0, 0.0, 1.0]])
    m_th, m_ph = p.transmitted_moments(
        mid, mom, k_p, k_m, np.array([theta]), np.array([phi]), 0.0, form="fresnel"
    )
    return c1 * m_th[0, 0], c1 * m_ph[0, 0]


def image_coeffs(eps_t, theta):
    """momwire's (rho_h, rho_v) image multipliers, `_far_readout._image_coeffs`."""
    rz = math.cos(theta)
    q = np.sqrt(eps_t - math.sin(theta) ** 2)
    return (rz - q) / (rz + q), (eps_t * rz - q) / (eps_t * rz + q)


def closed_form_above(kind, h, k_p, eps_t, w, theta, phi):
    """Direct + Fresnel image of a unit dipole at height h, momwire's readout."""
    c1 = -1j * w * p.MU0 / (4.0 * math.pi)
    rhat, th, ph = spherical(theta, phi)
    m = np.array([1.0, 0.0, 0.0]) if kind == "HED" else np.array([0.0, 0.0, 1.0])
    pos = np.array([0.0, 0.0, h])
    direct = m * np.exp(1j * k_p * rhat @ pos)
    img = m * np.array([-1.0, -1.0, 1.0]) * np.exp(1j * k_p * rhat @ (pos * [1, 1, -1]))
    rho_h, rho_v = image_coeffs(eps_t, theta)
    h_hat = ph
    v_hat = np.array(
        [
            -math.cos(phi) * math.cos(theta),
            -math.sin(phi) * math.cos(theta),
            math.sin(theta),
        ]
    )
    refl = rho_v * (img @ v_hat) * v_hat - rho_h * (img @ h_hat) * h_hat
    tot = direct + refl
    return c1 * (tot @ th), c1 * (tot @ ph)


def compare(E_cart, dirs, R, k_p, closed):
    """rel err of (E_θ, E_φ)·R·e^{+jk_pR} against the closed pair, per direction."""
    g = np.exp(-1j * k_p * R) / R
    rows = []
    for (theta, phi), e, (c_th, c_ph) in zip(dirs, E_cart, closed):
        rhat, th, ph = spherical(theta, phi)
        num = np.array([e @ th, e @ ph]) / g
        ref = np.array([c_th, c_ph])
        rel = float(np.linalg.norm(num - ref) / np.linalg.norm(ref))
        rows.append(
            {
                "theta": math.degrees(theta),
                "phi": math.degrees(phi),
                "rel": rel,
                "e_r_over_E": float(abs(e @ rhat) / g / np.linalg.norm(ref)),
            }
        )
    return rows


def calib(n_lam=(50, 200)):
    out = {}
    freq = 21e6
    for name, settings in HT.items():
        for cfg in ("air-air", "above-soilA"):
            eps_r, sigma = (1.0, 0.0) if cfg == "air-air" else p.SOILS["A"]
            eps_t, k_p, k_m, w = p.medium(freq, eps_r, sigma)
            lam0 = 2 * math.pi / k_p
            for n in n_lam:
                R = n * lam0
                dirs = directions()
                for kind in ("HED", "VED"):
                    if cfg == "air-air":
                        src = (0.0, 0.0, -0.15 if kind == "HED" else -0.6)
                        closed = [
                            closed_form_buried(kind, -src[2], k_p, k_m, w, t, f)
                            for t, f in dirs
                        ]
                    else:
                        src = (0.0, 0.0, 2.0)
                        closed = [
                            closed_form_above(kind, src[2], k_p, eps_t, w, t, f)
                            for t, f in dirs
                        ]
                    obs = [tuple(R * spherical(t, f)[0]) for t, f in dirs]
                    t0 = time.time()
                    try:
                        E = e_spec(src, kind, obs, freq, eps_r, sigma, settings)
                    except Exception as exc:  # noqa: BLE001 — a calibration row that errors is recorded as such
                        out[f"{name}/{cfg}/{n}/{kind}"] = {"error": repr(exc)}
                        print(name, cfg, n, kind, "ERROR", exc)
                        continue
                    rows = compare(E, dirs, R, k_p, closed)
                    worst = max(r["rel"] for r in rows if r["theta"] <= 75)
                    worst85 = max(r["rel"] for r in rows if r["theta"] > 75)
                    out[f"{name}/{cfg}/{n}/{kind}"] = {
                        "worst_le75": worst,
                        "worst_85": worst85,
                        "secs": time.time() - t0,
                        "rows": rows,
                    }
                    print(
                        f"{name:5s} {cfg:11s} R={n:3d}λ₀ {kind}: worst≤75° {worst:.2e}  85° {worst85:.2e}  ({time.time() - t0:.0f}s)"
                    )
    (HERE / "p_e_calib.json").write_text(json.dumps(out, indent=1))


def run(ht_name):
    settings = HT[ht_name]
    out = {}
    for soil in ("A", "B", "C"):
        for freq in (7e6, 21e6):
            eps_r, sigma = p.SOILS[soil]
            eps_t, k_p, k_m, w = p.medium(freq, eps_r, sigma)
            lam0 = 2 * math.pi / k_p
            for n in (50, 200):
                R = n * lam0
                dirs = directions()
                for kind, d in (("HED", 0.15), ("VED", 0.6)):
                    src = (0.0, 0.0, -d)
                    closed = [
                        closed_form_buried(kind, d, k_p, k_m, w, t, f) for t, f in dirs
                    ]
                    obs = [tuple(R * spherical(t, f)[0]) for t, f in dirs]
                    t0 = time.time()
                    E = e_spec(src, kind, obs, freq, eps_r, sigma, settings)
                    rows = compare(E, dirs, R, k_p, closed)
                    worst = max(r["rel"] for r in rows if r["theta"] <= 75)
                    worst85 = max(r["rel"] for r in rows if r["theta"] > 75)
                    key = f"{soil}/{freq / 1e6:g}/{n}/{kind}"
                    out[key] = {
                        "worst_le75": worst,
                        "worst_85": worst85,
                        "secs": time.time() - t0,
                        "rows": rows,
                    }
                    print(
                        f"{key:18s}: worst≤75° {worst:.2e}  85° {worst85:.2e}  ({time.time() - t0:.0f}s)",
                        flush=True,
                    )
    (HERE / "p_e_results.json").write_text(json.dumps(out, indent=1))
    bar = {50: 5e-3, 200: 2e-3}
    misses = [
        k
        for k, v in out.items()
        if v["worst_le75"] > bar[int(k.split("/")[2])] or v["worst_85"] > 2e-2
    ]
    print("P-E", "HIT" if not misses else f"MISS {misses}")


if __name__ == "__main__":
    if sys.argv[1] == "calib":
        calib()
    else:
        run(sys.argv[2] if len(sys.argv) > 2 else "quad")
