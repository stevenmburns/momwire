"""razor-2p's WIRE LOSS takes |I| from libm's hypot (momwire#1479).

`RazorSolver.wire_loss_power` integrates ½·Re Z'·∫|I|² dl from the knot
currents. Its |I| was numpy's complex absolute, whose AVX-512 and AVX2 loops
disagree in the last bit, so a printed WIRE LOSS could depend on the CPU and
no C++ replay could call the same function. It is now `abs(complex)` (libm's
hypot) element by element, as the pattern readout's E columns are; this pins
the watts to that arithmetic bit for bit.
"""

import numpy as np

from momwire import _wire_loading
from momwire.razor import RazorSolver


def _solver():
    polylines = [np.array([[0.0, 0.0, -2.5], [0.0, 0.0, 2.5]])]
    return RazorSolver(
        wires=polylines,
        n_per_edge_per_wire=[[801]],
        wire_radius=0.0005,
        wavelength=299.792458 / 28.5,
        feeds=[(0, 2.5, 1.0)],
        wire_conductivity=[5.8e7],
    )


def test_the_wire_loss_is_libm_hypots_integral():
    solver = _solver()
    coeffs = solver.compute_port_solution().coeffs[:, 0]
    total, per_wire = solver.wire_loss_power(coeffs)

    geom = solver._build_geometry()
    alpha = np.asarray(coeffs, dtype=np.complex128)[: geom["n_basis_total"]]
    ent_seg = geom["wing_seg"].reshape(-1)
    ent_rise = geom["wing_rise"].reshape(-1)
    ent_val = np.repeat(alpha, 2) * geom["wing_sigma"].reshape(-1)
    i_lo = np.zeros(geom["n_segs_total"], dtype=np.complex128)
    i_hi = np.zeros(geom["n_segs_total"], dtype=np.complex128)
    np.add.at(i_hi, ent_seg[ent_rise], ent_val[ent_rise])
    np.add.at(i_lo, ent_seg[~ent_rise], ent_val[~ent_rise])
    mag_lo = np.array([abs(v) for v in i_lo.tolist()])
    mag_hi = np.array([abs(v) for v in i_hi.tolist()])
    h = np.asarray(geom["seg_h"], dtype=np.float64)
    gram = h * (mag_lo**2 + np.real(i_lo * np.conj(i_hi)) + mag_hi**2) / 3.0
    r_w = np.real(_wire_loading.loading_for(solver, solver.omega, geom).z_wire)
    wire_of = solver._wire_of_seg(geom)
    want = np.zeros(1)
    np.add.at(want, wire_of, 0.5 * r_w[wire_of] * gram)
    assert per_wire.tobytes() == want.tobytes()
    assert total == float(want.sum()) and total > 0.0
