"""Compare TriangularPySim against NEC2 for a matched 2-element Yagi in free space.

Run from the project venv (needs PyNEC — see `scripts/build_pynec.sh`):

    .venv/bin/python scripts/compare_yagi_nec.py
"""

import numpy as np
import PyNEC as nec

from pysim.triangular import TriangularPySim


def _run_nec(c, n_seg, freq_mhz):
    c.geometry_complete(0)
    c.gn_card(-1, 0, 0, 0, 0, 0, 0, 0)  # free space

    feed_seg = (n_seg + 1) // 2
    c.ex_card(0, 1, feed_seg, 0, 1.0, 0.0, 0, 0, 0, 0)

    c.fr_card(0, 1, freq_mhz, 0)
    c.xq_card(0)

    sc = c.get_structure_currents(0)
    currents = sc.get_current()
    tags = sc.get_current_segment_tag()

    driver_indices = [i for i, t in enumerate(tags) if t == 1]
    feed_idx = driver_indices[feed_seg - 1]
    return 1.0 / currents[feed_idx]


def nec_free_space_dipole(*, freq_mhz, halfdriver, wire_radius, n_seg):
    c = nec.nec_context()
    geo = c.get_geometry()

    geo.wire(
        1,
        n_seg,
        0.0,
        -halfdriver,
        0.0,
        0.0,
        +halfdriver,
        0.0,
        wire_radius,
        1.0,
        1.0,
    )
    return _run_nec(c, n_seg, freq_mhz)


def nec_free_space_yagi(
    *, freq_mhz, halfdriver, reflector_factor, spacing, wire_radius, n_seg
):
    c = nec.nec_context()
    geo = c.get_geometry()

    geo.wire(
        1,
        n_seg,
        0.0,
        -halfdriver,
        0.0,
        0.0,
        +halfdriver,
        0.0,
        wire_radius,
        1.0,
        1.0,
    )

    refl_h = halfdriver * reflector_factor
    geo.wire(
        2,
        n_seg,
        -spacing,
        -refl_h,
        0.0,
        -spacing,
        +refl_h,
        0.0,
        wire_radius,
        1.0,
        1.0,
    )
    return _run_nec(c, n_seg, freq_mhz)


def main():
    wavelength = 22.0
    freq_mhz = 299.792458 / wavelength
    halfdriver = 0.962 * wavelength / 4
    spacing = halfdriver
    refl_factor = 1.05
    wire_radius = 0.0005

    print(f"Geometry: wavelength={wavelength:.3f} m  freq={freq_mhz:.4f} MHz")
    print(
        f"          halfdriver={halfdriver:.4f} m  "
        f"spacing={spacing:.4f} m ({spacing / wavelength:.3f} lambda)  "
        f"reflector_factor={refl_factor}"
    )
    print(f"          wire_radius={wire_radius:.4f} m")
    print()

    print("=== Single dipole (driver only) ===")
    print("TriangularPySim (tent-basis Galerkin, single straight wire):")
    dipole_polyline = np.array([[0.0, -halfdriver, 0.0], [0.0, halfdriver, 0.0]])
    for nsegs in [20, 40, 80, 160]:
        z, _ = TriangularPySim(
            wires=[dipole_polyline],
            n_per_edge_per_wire=[[nsegs]],
            nsegs=nsegs,
        ).compute_impedance()
        print(f"  nsegs={nsegs:3d}: Z = {z.real:8.3f} + j{z.imag:8.3f}")
    print()

    print("NEC2 free space (dipole only):")
    for n_seg in [21, 41, 101]:
        z = nec_free_space_dipole(
            freq_mhz=freq_mhz,
            halfdriver=halfdriver,
            wire_radius=wire_radius,
            n_seg=n_seg,
        )
        print(f"  n_seg={n_seg:3d}: Z = {z.real:8.3f} + j{z.imag:8.3f}")

    print()
    print("=== Two-element Yagi (driver + reflector) ===")
    print("TriangularPySim (tent-basis Galerkin, driver + reflector):")
    driver_polyline = np.array([[0.0, -halfdriver, 0.0], [0.0, halfdriver, 0.0]])
    refl_polyline = np.array(
        [
            [-spacing, -refl_factor * halfdriver, 0.0],
            [-spacing, refl_factor * halfdriver, 0.0],
        ]
    )
    for nsegs in [20, 40, 80, 160]:
        z, _ = TriangularPySim(
            wires=[driver_polyline, refl_polyline],
            n_per_edge_per_wire=[[nsegs], [nsegs]],
            nsegs=nsegs,
        ).compute_impedance()
        print(f"  nsegs={nsegs:3d}: Z = {z.real:8.3f} + j{z.imag:8.3f}")
    print()

    print("NEC2 free space:")
    for n_seg in [21, 41, 101]:
        z = nec_free_space_yagi(
            freq_mhz=freq_mhz,
            halfdriver=halfdriver,
            reflector_factor=refl_factor,
            spacing=spacing,
            wire_radius=wire_radius,
            n_seg=n_seg,
        )
        print(f"  n_seg={n_seg:3d}: Z = {z.real:8.3f} + j{z.imag:8.3f}")


if __name__ == "__main__":
    main()
