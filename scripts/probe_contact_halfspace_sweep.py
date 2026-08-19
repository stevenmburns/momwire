"""momwire#282 stage 2: the contact gap as a function of the half-space,
swept against the binary rather than against four soils.

Study §5.4 named three candidates for the low-eps_r gap and stage 1 killed
the second. The first (the remainder's near-interface behaviour) is killed
by `probe_contact_direct_remainder.py`: with the interpolation grid bypassed
(the rtol axis is saturated — see that probe's docstring for what the bypass
does and does not exclude) the poor-soil residual moves from 3.3274 to
3.3305 ohm, and raising the remainder's spatial quadrature order 3 -> 12
moves it 0.03 ohm. Neither is 3 ohm.

The third candidate is "the missing base-loss resistance is real". This
script is its kill-or-confirm: sweep eps_r at a conductivity small enough
that the half-space cannot dissipate (loss tangent <~ 1e-2) and see where
the discrepancy goes. A LOSS term must vanish with sigma. If the gap is at
its largest over a lossless dielectric, it is not a loss term and all three
of the study's candidates are gone.

WHAT IT MEASURED (2026-08-19, N = 41 monopole, contact)
--------------------------------------------------------
`--mode epsr` at sigma = 1e-5. |discrepancy|, ohm:

  eps_r  1.02  1.05  1.1   1.2   1.35  1.5   2.0   2.5   3.0   4.0
         0.46  0.70  1.11  1.82  2.63  3.21  4.15  4.36  4.28  3.87
  eps_r  5.0   6.5   8.0   10    13    16    20    30    50    81
         3.42  2.84  2.40  1.97  1.53  1.24  0.98  0.67  0.47  0.41

A smooth single-peaked curve, vanishing at both physical limits (eps~ -> 1,
no ground; eps~ -> infinity, PEC) and PEAKING at 4.36 ohm over a dielectric
with no loss in it. Candidate 3 is dead. The low end MEASURES the
eps~ -> 1 zero rather than asserting it: monotone to zero down to 0.46 ohm
at eps_r = 1.02.

`--mode sigma` at eps_r = 5 walks the second axis and agrees: the
discrepancy is 3.42 ohm at sigma = 0 and falls monotonically as the ground
becomes conductive, reaching the 0.2-0.35 ohm floor by sigma = 0.03.

Along both paths the REAL part tracks the uncancelled image-charge fraction
`1 - C2 = 2/(eps~ + 1)`: K = Re(disc / (1 - C2)) sits at -10.3 +- 0.3 over
eps_r in [10, 30] and at -9.2 +- 0.2 along the whole sigma path from 0 to
3e-3. NOTE the spelling — with complex eps~, disc.real / (1 - C2) is a
different quantity and walks -9.19 -> -6.88 along the sigma path; K is what
reproduces. It is a good description, not a law: it drifts 40 % below
eps_r = 6, the constant is -8.6 on the bent quarter wave, and across the
wire radii whose decks actually pass the 1.5 ohm PEC-conditioning bar
(0.5-5 mm, a 10x range — see `probe_contact_deck_conditioning.py`) it is
-9.51 to -10.41. See the study's stage-2 record for what it does and does
not license.

Runs the binary, so: antennaknobs venv, NEC5_EXE set.

    NEC5_EXE=~/antennas/NEC5-downloads/nec5-linux/nec5cl \
        /home/smburns/antennas/antennaknobs/.venv/bin/python \
        scripts/probe_contact_halfspace_sweep.py --mode epsr
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

FREQ_MHZ = 14.0
C = 299792458.0
WL = C / (FREQ_MHZ * 1e6)
EPS0 = 8.8541878128e-12
RAD = 0.005
MONO_H = 5.3535

# The two sweeps. `epsr` holds sigma at a value whose loss tangent is <= 1e-2
# everywhere on the sweep; `sigma` holds eps_r at poor soil's 5.0 and walks
# the conductivity over five decades through it.
EPSR_SWEEP = [
    1.02,
    1.05,
    1.1,
    1.2,
    1.35,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.5,
    8.0,
    10.0,
    13.0,
    16.0,
    20.0,
    30.0,
    50.0,
    81.0,
]
SIGMA_SWEEP = [0.0, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def _num(x):
    return f"{float(x):.6E}"


def deck(n, eps_r, sigma):
    p0 = (0.0, 0.0, 0.0)
    p1 = (0.0, 0.0, MONO_H)
    gn = (
        "GN 1 0 0 0\n"
        if eps_r is None
        else f"GN 0 0 0 0 {_num(eps_r)} {_num(sigma)} {_num(1.0)} {_num(0.0)} NOFILE\n"
    )
    return (
        "CM momwire#282 stage 2 — contact monopole, half-space sweep\nCE\n"
        f"GW 1 {n} {_num(p0[0])} {_num(p0[1])} {_num(p0[2])} "
        f"{_num(p1[0])} {_num(p1[1])} {_num(p1[2])} {_num(RAD)}\n"
        "GE 1 0\n" + gn + f"EX 0 1 1 1 {_num(1.0)} {_num(0.0)}\n"
        f"FR 0 1 0 0 {_num(FREQ_MHZ)} {_num(0.0)}\n"
        "XQ 0\nEN\n"
    )


def momwire_z(n, eps_r, sigma):
    from momwire import BSplineSolver

    kw = dict(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, MONO_H]])],
        n_per_edge_per_wire=[[n]],
        wire_radius=RAD,
        wavelength=WL,
        degree=2,
        feed_model="segment",
        feed_wire_index=0,
        feed_arclength=0.0,
        ground_z=0.0,
    )
    if eps_r is not None:
        kw.update(ground_eps=(eps_r, sigma), ground_model="sommerfeld")
    z, _ = BSplineSolver(**kw).compute_impedance()
    return complex(z)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("epsr", "sigma"), default="epsr")
    p.add_argument("--n", type=int, nargs="+", default=[41])
    p.add_argument("--sigma", type=float, default=1e-5)
    p.add_argument("--eps-r", type=float, default=5.0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from antennaknobs.engines.nec5 import NEC5Engine

    sys.path.insert(0, str(Path.home() / "antennas/antennaknobs/scripts"))
    from bench_nec5_walk_why import make_dipole

    captures = Path(os.environ.get("STAGE2_CAPTURES", "/tmp/stage2-captures"))
    captures.mkdir(parents=True, exist_ok=True)
    eng = NEC5Engine(make_dipole(20), ground=None, capture_dir=captures)

    if args.mode == "epsr":
        cases = [(e, args.sigma) for e in EPSR_SWEEP]
        label = "eps_r"
    else:
        cases = [(args.eps_r, s) for s in SIGMA_SWEEP]
        label = "sigma"

    omega = 2.0 * np.pi * FREQ_MHZ * 1e6
    rows = []
    hdr = (
        f"{label:>8} {'tan_d':>9} {'N':>3}  {'d_mw':>19} {'d_n5':>19} "
        f"{'discrepancy':>19} {'|.|':>7} {'arg':>7} {'K':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for n in args.n:
        z_pec_n5 = complex(eng.run_deck(deck(n, None, 0.0))[0][0][2])
        z_pec_mw = momwire_z(n, None, 0.0)
        print(f"  PEC N={n}: nec5={z_pec_n5:.4f}  momwire={z_pec_mw:.4f}")
        for eps_r, sigma in cases:
            z_n5 = complex(eng.run_deck(deck(n, eps_r, sigma))[0][0][2])
            z_mw = momwire_z(n, eps_r, sigma)
            d_n5 = z_n5 - z_pec_n5
            d_mw = z_mw - z_pec_mw
            disc = d_mw - d_n5
            tan_d = sigma / (omega * EPS0 * eps_r)
            # NOTE the spelling: K = Re(disc/(1-C2)), NOT disc.real/(1-C2).
            # With complex eps~ the two differ, and only the first reproduces
            # the -10.3/-9.2 constants the stage-2 record quotes.
            eps_t = eps_r - 1j * sigma / (omega * EPS0)
            k_real = float(np.real(disc * (eps_t + 1.0) / 2.0))
            rows.append(
                dict(
                    n=n,
                    eps_r=eps_r,
                    sigma=sigma,
                    tan_delta=tan_d,
                    z_n5=[z_n5.real, z_n5.imag],
                    z_mw=[z_mw.real, z_mw.imag],
                    d_n5=[d_n5.real, d_n5.imag],
                    d_mw=[d_mw.real, d_mw.imag],
                    disc=[disc.real, disc.imag],
                    mag=abs(disc),
                    arg_deg=float(np.degrees(np.angle(disc))),
                    k_real=k_real,
                )
            )
            shown = eps_r if args.mode == "epsr" else sigma
            print(
                f"{shown:>8.4g} {tan_d:>9.3g} {n:>3}  "
                f"{d_mw.real:>9.3f}{d_mw.imag:>+9.3f}j  {d_n5.real:>9.3f}{d_n5.imag:>+9.3f}j  "
                f"{disc.real:>9.3f}{disc.imag:>+9.3f}j  {abs(disc):>7.3f} "
                f"{np.degrees(np.angle(disc)):>7.1f} {k_real:>8.2f}",
                flush=True,
            )

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
