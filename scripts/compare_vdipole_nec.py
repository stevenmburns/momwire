"""Compare BentTriangularPySim against NEC2 for a V-dipole.

Geometry: two arms of length halfdriver meeting at the feed (origin).
Each arm bends away from the y-axis by angle alpha in the y-z plane:
  alpha = 0:   straight dipole along y-axis
  alpha > 0:   V-shape opening downward

NEC models this as two GW cards joined at the origin (segment 1's end =
segment 2's start). The feed is at the boundary segment of arm 1
(NEC fed via EX card at the last segment of arm 1, which abuts arm 2).

Run with antenna_designer's venv (which has PyNEC):

    PYTHONPATH=/home/smburns/antennas/pysim/src \\
        /home/smburns/antennas/antenna_designer/.venv/bin/python \\
        scripts/compare_vdipole_nec.py
"""
import numpy as np
import PyNEC as nec

from pysim.triangular_bent import BentTriangularPySim


def nec_v_dipole(*, freq_mhz, half_arm, alpha_rad, wire_radius, n_per_arm):
    c = nec.nec_context()
    geo = c.get_geometry()

    cos_a = np.cos(alpha_rad)
    sin_a = np.sin(alpha_rad)
    # Arm 1: from end1 -> origin
    end1 = (0.0, -half_arm * cos_a, -half_arm * sin_a)
    # Arm 2: from origin -> end2
    end2 = (0.0, +half_arm * cos_a, -half_arm * sin_a)

    geo.wire(
        1, n_per_arm,
        end1[0], end1[1], end1[2],
        0.0, 0.0, 0.0,
        wire_radius, 1.0, 1.0,
    )
    geo.wire(
        2, n_per_arm,
        0.0, 0.0, 0.0,
        end2[0], end2[1], end2[2],
        wire_radius, 1.0, 1.0,
    )
    c.geometry_complete(0)
    c.gn_card(-1, 0, 0, 0, 0, 0, 0, 0)  # free space

    # Feed at the last segment of arm 1 (the one abutting arm 2 at the origin).
    c.ex_card(0, 1, n_per_arm, 0, 1.0, 0.0, 0, 0, 0, 0)
    c.fr_card(0, 1, freq_mhz, 0)
    c.xq_card(0)

    sc = c.get_structure_currents(0)
    currents = sc.get_current()
    tags = sc.get_current_segment_tag()
    # Feed segment is tag=1, segment_no=n_per_arm (i.e. last on arm 1).
    feed_idx = [i for i, t in enumerate(tags) if t == 1][n_per_arm - 1]
    return 1.0 / currents[feed_idx]


def main():
    bt = BentTriangularPySim()
    wavelength = bt.wavelength
    freq_mhz = 299.792458 / wavelength
    half_arm = bt.halfdriver
    wire_radius = bt.wire_radius

    print(f"Geometry: wavelength={wavelength:.3f} m  freq={freq_mhz:.4f} MHz")
    print(f"          half_arm={half_arm:.4f} m  wire_radius={wire_radius:.4f} m")
    print()

    for alpha_deg in [0, 15, 30, 45, 60]:
        alpha = np.radians(alpha_deg)
        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)
        polyline = np.array([
            [0.0, -half_arm * cos_a, -half_arm * sin_a],
            [0.0, 0.0, 0.0],
            [0.0, +half_arm * cos_a, -half_arm * sin_a],
        ])
        print(f"=== alpha = {alpha_deg:3d} deg ===")
        print("BentTriangularPySim:")
        for n_per_edge in [20, 40, 80]:
            z, _ = BentTriangularPySim(
                polyline=polyline, n_per_edge=n_per_edge, nsegs=2 * n_per_edge,
            ).compute_impedance()
            print(f"  n_per_arm={n_per_edge:3d}: Z = {z.real:8.3f} + j{z.imag:8.3f}")

        print("NEC2:")
        for n_per_arm in [20, 40, 80]:
            z = nec_v_dipole(
                freq_mhz=freq_mhz, half_arm=half_arm, alpha_rad=alpha,
                wire_radius=wire_radius, n_per_arm=n_per_arm,
            )
            print(f"  n_per_arm={n_per_arm:3d}: Z = {z.real:8.3f} + j{z.imag:8.3f}")
        print()


if __name__ == "__main__":
    main()
