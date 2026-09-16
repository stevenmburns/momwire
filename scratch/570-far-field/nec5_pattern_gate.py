"""P-H / P-I: NEC-5 x13 against the transmitted far field on the catalog.

momwire's own currents (the antennaknobs MomwireEngine at each design's
defaults over soil (13, 0.005)) feed the far-field readout: above-plane
elements through momwire's `_far_moments` (direct + Fresnel image), below-
plane elements through the prototype's transmitted factors. Gain is
normalised by the engine's input power, as the app does. NEC-5 x13 solves
the same design through the antennaknobs NEC5Engine and prints its own RP.

Registered in PLAN.md (P-H, P-I). Coarse grid: θ every 5°, φ every 15°.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import proto_far as p  # noqa: E402

from antennaknobs.engines.momwire import ETA0, MomwireEngine  # noqa: E402
from antennaknobs.engines.nec5 import NEC5Engine  # noqa: E402
from antennaknobs.in_medium import below_surface_mask  # noqa: E402
from momwire._far_readout import Ground, _far_moments  # noqa: E402

SOIL = ("finite", 13.0, 0.005)
NEC5 = os.environ.get(
    "NEC5_EXE", str(Path.home() / "antennas/NEC5-downloads/nec5-linux/nec5cl")
)
GRID = dict(n_theta=18, n_phi=24, del_theta=5, del_phi=15)


def momwire_pattern(eng, *, honest=True):
    """dBi rings on GRID from momwire's currents. `honest=False` images every
    element (the pre-U8 readout, for P-I's movement)."""
    wavelength = eng._wavelength_for(eng.builder.freq)
    k = 2.0 * math.pi / wavelength
    freq_hz = eng.builder.freq * 1e6
    sim, coeffs, _z = eng._solved_excited(wavelength)
    mid, dr, i_mid = eng._segment_dipoles(sim, coeffs)
    moment = i_mid[:, None] * dr
    theta = np.deg2rad(np.linspace(0, 90 - GRID["del_theta"], GRID["n_theta"]))
    phi = np.deg2rad(np.linspace(0, 360, GRID["n_phi"] + 1))
    ground = Ground("sommerfeld", SOIL[1], SOIL[2])
    gz = float(eng._ground_z)
    below = below_surface_mask(mid, gz)
    if honest and below.any():
        m_th, m_ph = _far_moments(
            mid[~below], moment[~below], k, theta, phi, ground, gz, freq_hz
        )
        eps_t, k_p, k_m, _ = p.medium(freq_hz, SOIL[1], SOIL[2])
        t_th, t_ph = p.transmitted_moments(
            mid[below], moment[below], k_p, k_m, theta, phi, gz, form="fresnel"
        )
        m_th, m_ph = m_th + t_th, m_ph + t_ph
    else:
        m_th, m_ph = _far_moments(mid, moment, k, theta, phi, ground, gz, freq_hz)
    p_in = eng.input_power()
    norm = ETA0 * k * k / (8.0 * math.pi * p_in)
    d = norm * (np.abs(m_th) ** 2 + np.abs(m_ph) ** 2)
    return 10.0 * np.log10(np.maximum(d, 1e-30)), float(below.sum()) / len(below), _z


def compare(name, mw, n5, window_db=20.0):
    mw, n5 = np.asarray(mw), np.asarray(n5)
    lit = n5 >= n5.max() - window_db
    d = mw - n5
    i = np.unravel_index(int(np.argmax(n5)), n5.shape)
    shape = d[lit] - np.mean(d[lit])
    out = {
        "nec5_peak_dbi": float(n5.max()),
        "momwire_peak_dbi": float(mw.max()),
        "delta_at_nec5_peak_db": float(d[i]),
        "max_abs_delta_lit_db": float(np.max(np.abs(d[lit]))),
        "mean_delta_lit_db": float(np.mean(d[lit])),
        "max_abs_shape_delta_lit_db": float(np.max(np.abs(shape))),
        "lit_directions": int(lit.sum()),
    }
    print(
        f"{name}: "
        + ", ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in out.items()
        )
    )
    return out


def main():
    from antennaknobs.designs.specialty.buried_dipole import Builder as BD
    from antennaknobs.designs.verticals.buried_radial_vertical import Builder as BRV

    results = {}
    for name, B in (("buried_dipole", BD), ("buried_radial_vertical", BRV)):
        t0 = time.time()
        eng = MomwireEngine(B(), ground=SOIL)
        mw, frac, z = momwire_pattern(eng, honest=True)
        rec = {
            "momwire_z": [complex(v).real for v in np.atleast_1d(z)[:1]]
            + [complex(v).imag for v in np.atleast_1d(z)[:1]],
            "below_fraction": frac,
        }
        if frac < 1.0:
            imaged, _, _ = momwire_pattern(eng, honest=False)
            rec["imaged_peak_dbi"] = float(imaged.max())
            rec["honest_peak_dbi"] = float(mw.max())
            rec["honest_minus_imaged_at_imaged_peak_db"] = float(
                (mw - imaged)[np.unravel_index(int(np.argmax(imaged)), imaged.shape)]
            )
            rec["max_abs_honest_minus_imaged_lit_db"] = float(
                np.max(np.abs((mw - imaged)[imaged >= imaged.max() - 20]))
            )
            print(
                f"{name}: imaged peak {imaged.max():.3f} dBi, honest peak {mw.max():.3f} dBi, Δ at imaged peak {rec['honest_minus_imaged_at_imaged_peak_db']:+.3f} dB"
            )
        print(
            f"{name}: momwire done in {time.time() - t0:.0f}s; peak {mw.max():.3f} dBi; below fraction {frac:.3f}",
            flush=True,
        )
        t0 = time.time()
        n5 = NEC5Engine(B(), ground=SOIL, nec5_exe=NEC5, timeout=1800)
        ff = n5.far_field(**GRID)
        z5 = n5.impedance()
        rec["nec5_z"] = [complex(z5[0]).real, complex(z5[0]).imag]
        print(
            f"{name}: NEC-5 done in {time.time() - t0:.0f}s; Z {z5[0]:.3f} vs momwire {np.atleast_1d(z)[0]:.3f}",
            flush=True,
        )
        rec["compare"] = compare(name, mw, ff.rings)
        rec["momwire_rings"] = np.asarray(mw).round(3).tolist()
        rec["nec5_rings"] = np.asarray(ff.rings).round(3).tolist()
        results[name] = rec
    (HERE / "p_h_i_nec5.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
