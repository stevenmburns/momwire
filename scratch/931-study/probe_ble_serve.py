"""momwire#931: does momwire actually REFUSE the BLE geometry?

I told the fleet it does, citing #838's recorded blockers (the R1 cap and the
grazing floor). That was an inference from an issue's text, not a measurement,
and scratch/ble-1937/RESULTS.md records momwire SERVING the same geometry on
v0.47.0. Settling it by running it.

BLE 1937 Fig. 36: 3 MHz, 77-degree mast (21.4 m), No. 8 radials 0.152 m down,
sigma 2e-3. Hub spelling: radials to a buried hub, one rise to z=0, mast above.
"""

import math

import numpy as np

from momwire.bspline import BSplineSolver

C0 = 299792458.0
F = 3.0e6
WL = C0 / F
H_MAST = 77.0 / 360.0 * WL
DEPTH = 6 * 0.0254
A_WIRE = 0.001628
A_MAST = 2.5 * 0.0254 / 2
SIGMA = 2e-3


def deck(n_radials, r_len_ft, eps_r, n_rad_seg=10, n_mast_seg=12):
    R = r_len_ft * 0.3048
    wires, npe = [], []
    for i in range(n_radials):
        th = 2 * math.pi * i / n_radials
        wires.append(
            np.array([[0.0, 0.0, -DEPTH], [R * math.cos(th), R * math.sin(th), -DEPTH]])
        )
        npe.append([n_rad_seg])
    rise_i = len(wires)
    wires.append(np.array([[0.0, 0.0, -DEPTH], [0.0, 0.0, 0.0]]))
    npe.append([2])
    mast_i = rise_i + 1
    wires.append(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, H_MAST]]))
    npe.append([n_mast_seg])
    return dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=[
            [(i, "start") for i in range(n_radials)] + [(rise_i, "start")],
            [(rise_i, "end"), (mast_i, "start")],
        ],
        feeds=[(mast_i, H_MAST / n_mast_seg / 2, 1 + 0j)],
        wavelength=WL,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=(eps_r, SIGMA),
        ground_model="sommerfeld",
    )


for eps_r in (15.0, 5.0):
    for lft in (135.0, 45.0):
        for n in (2, 4):
            try:
                z, _ = BSplineSolver(**deck(n, lft, eps_r)).compute_impedance()
                print(
                    f"  eps {eps_r:4.1f}  {lft:5.0f} ft  N={n:2d}   SERVED   "
                    f"{complex(z).real:8.3f}{complex(z).imag:+8.3f}j"
                )
            except Exception as e:  # noqa: BLE001
                msg = " ".join(str(e).split())
                key = (
                    "R1 cap"
                    if "R1" in msg or "lambda" in msg.lower()
                    else "grazing"
                    if "grazing" in msg.lower()
                    else msg[:40]
                )
                print(f"  eps {eps_r:4.1f}  {lft:5.0f} ft  N={n:2d}   REFUSED  {key}")
