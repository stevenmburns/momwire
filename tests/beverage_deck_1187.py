"""A momwire-native Beverage with a ground rod at each end (momwire#1187).

The geometry and mesh of antennaknobs' `wire.beverage` (AK#1707) at its
defaults, as that design hands them to momwire's B-spline and
sinusoidal-Galerkin lanes — reproduced here because a momwire test must not
import antennaknobs. 245.73 m (1.5 free-space wavelengths at 1.83 MHz) of #14
copper 2.5 m up, a 0.25 m feed edge and a 0.25 m termination edge at the
bottom of each down-lead, each standing on a 1.8 m rod of the same wire that
crosses z = 0 into the soil. Every segment is horizontal or vertical.

The two ports are the feed edge (the design's `PortOnWire("feed")`) and the
termination edge (`"term"`, a 500 ohm resistor in the design). `feed_z`
closes the term port on that resistor, which is the "feed" row of the
design's docstring (the antenna side of its 9:1 transformer).
"""

from __future__ import annotations

import numpy as np

C0 = 299792458.0
FREQ_HZ = 1.83e6
WL = C0 / FREQ_HZ
LENGTH_M = 1.5 * WL  # 245.73 m
HEIGHT_M = 2.5
EDGE_M = 0.25
ROD_M = 1.8
RADIUS_M = 0.814e-3  # #14 AWG
COPPER = 5.8e7
TERM_R = 500.0

# The app's soil presets (ARRL Table 3.1).
AVERAGE = (13.0, 0.005)
POOR = (13.0, 0.002)


def beverage_deck(ground_eps=AVERAGE, *, length_m=LENGTH_M, rod_segments=7):
    """Solver kwargs for the rod Beverage. `length_m` shortens the run (the
    rods then stand closer); the mesh is the AK design's B-spline mesh at the
    defaults: 7 segments per rod, one per edge, 126 on the run."""
    L = float(length_m)
    n_run = max(1, int(round(126 * L / LENGTH_M)))
    wires = [
        np.array([(0.0, 0.0, -ROD_M), (0.0, 0.0, 0.0)]),
        np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, 0.0, EDGE_M),
                (0.0, 0.0, HEIGHT_M),
                (L, 0.0, HEIGHT_M),
                (L, 0.0, EDGE_M),
                (L, 0.0, 0.0),
            ]
        ),
        np.array([(L, 0.0, 0.0), (L, 0.0, -ROD_M)]),
    ]
    run_len = 2.0 * HEIGHT_M + L
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[rod_segments], [1, 1, n_run, 1, 1], [rod_segments]],
        feeds=[(1, 0.5 * EDGE_M, 1 + 0j), (1, run_len - 0.5 * EDGE_M, 1 + 0j)],
        wavelength=WL,
        wire_radius=RADIUS_M,
        wire_conductivity=[COPPER, COPPER, COPPER],
        ground_z=0.0,
        ground_eps=ground_eps,
        ground_model="sommerfeld",
    )


def feed_z(solver, term_r=TERM_R):
    """The feed port's impedance with the term port closed on `term_r`."""
    z = np.linalg.inv(np.asarray(solver.compute_y_matrix()))
    return complex(z[0, 0] - z[0, 1] * z[1, 0] / (z[1, 1] + term_r))
