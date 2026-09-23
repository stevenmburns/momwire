"""momwire#1156 probe 0: does SG's compute_port_solution enter the lower
medium on a fully-buried deck, as compute_impedance does?"""

import warnings

import numpy as np

from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

WL = 299792458.0 / 7.0e6
pts = np.array([(-2.5, 0.0, -0.5), (2.5, 0.0, -0.5)])
d = dict(
    wires=[pts],
    n_per_edge_per_wire=[[21]],
    feeds=[(0, 2.5, 1 + 0j)],
    wavelength=WL,
    wire_radius=1e-3,
    ground_z=0.0,
    ground_eps=(13.0, 0.005),
    ground_model="sommerfeld",
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    s = SinusoidalGalerkinSolver(**d)
    z_imp = complex(s.compute_impedance()[0])
    k = 2 * np.pi / WL
    zs = s.compute_impedance_swept([k])
    print("compute_impedance      ", z_imp)
    print("impedance_swept        ", complex(np.ravel(zs)[0]))
    for name, fn in (
        ("compute_port_solution", s.compute_port_solution),
        ("compute_y_matrix_swept", lambda: s.compute_y_matrix_swept([k])),
    ):
        try:
            fn()
            print(name, "served")
        except Exception as e:  # noqa: BLE001 - the probe reports what raised
            print(name, type(e).__name__, str(e)[:120])
