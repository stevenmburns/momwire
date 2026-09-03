"""#813 step 3 shape (a), probe 1: does #812's below fill take a wire whose
END sits AT the interface?

Shape (a) puts the crossing wire into a below-only sub-deck, where its plane
end is declared a grounded end carrying a half tent. That end sits at exactly
z = ground_z. The below/below remainder guards `d <= 0.0` on SOMETHING, and
which something decides whether the sub-geometry can be built at all or
whether the guard has to be told to read nodes instead of endpoints.

Asked by construction rather than by reading the guard.
"""

import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))

from momwire import razor as _razor  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402

C0 = 299792458.0
LAM = C0 / 7.0e6
SOIL_A = (13.0, 0.005)


def below_deck(top_z, n=8):
    """A vertical wire from -2 m up to `top_z`, wholly below when top_z < 0."""
    return dict(
        wires=[np.array([[0.0, 0.0, -2.0], [0.0, 0.0, top_z]])],
        n_per_edge_per_wire=[[n]],
        wire_radius=0.001,
        wavelength=LAM,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
        feeds=[(0, 1.0, 1 + 0j)],
    )


def main():
    warnings.filterwarnings("ignore")
    _razor._SERVE_BELOW_PLANE = True
    for label, top in (
        ("clear of the plane   (top = -0.05 m)", -0.05),
        ("one micron below     (top = -1e-6 m)", -1e-6),
        ("EXACTLY at the plane (top =  0.0  m)", 0.0),
    ):
        deck = below_deck(top)
        try:
            rs = RazorSolver(**deck, nec5_quadrature=False, n_qp_path=8)
            z = complex(np.asarray(rs.compute_impedance()[0]).ravel()[0])
            print(f"  {label}:  Z = {z:.6g}")
        except Exception as e:  # noqa: BLE001 -- the probe's whole subject
            print(f"  {label}:  {type(e).__name__}: {str(e)[:150]}")


if __name__ == "__main__":
    main()


def past_the_label_guard():
    """The same question with `_medium_spec.wire_media` stepped past, so the
    two guards are separated: the label scan refuses a below wire whose end
    touches the plane (no crossing exemption is available to a lone end,
    momwire#698 skips one-member groups), and underneath it the below/below
    remainder has its own `d <= 0` guard. Which one actually stops shape (a)?
    """
    import momwire._potential_ground as PG

    warnings.filterwarnings("ignore")
    _razor._SERVE_BELOW_PLANE = True
    orig = RazorSolver._refuse_buried_geometry

    def stub(self):
        self._below_plane = True
        self._crossing = False

    RazorSolver._refuse_buried_geometry = stub
    try:
        for label, top in (
            ("top = -0.05 m ", -0.05),
            ("top = -1e-9 m ", -1e-9),
            ("top =  0.0  m ", 0.0),
        ):
            rs = RazorSolver(**below_deck(top), nec5_quadrature=False, n_qp_path=8)
            geom = rs._build_geometry()
            prep = rs._assemble_Z_prepare(geom)
            try:
                Z = rs._assemble_Z_from_prepared(geom, prep, rs.k, rs.omega)
                print(f"  {label}: fill OK, |Z| max = {np.abs(Z).max():.6g}")
            except Exception as e:  # noqa: BLE001 -- the probe's subject
                print(f"  {label}: {type(e).__name__}: {str(e)[:140]}")
            # what the guard actually reads
            eps_t = PG._ground_refl.eps_tilde(rs.ground_eps, rs.omega, rs.eps)
            g = PG.BelowMediumGround(rs, geom, rs.k, rs.omega, eps_tilde=eps_t)
            src_l, src_r = (
                geom["seg_p0"],
                geom["seg_p0"] + geom["seg_h"][:, None] * geom["seg_t"],
            )
            pts, _t, _w = rs._testing_paths(geom)
            print(
                f"       min source-endpoint depth = "
                f"{min(-src_l[:, 2].max(), -src_r[:, 2].max()):.3e};  "
                f"min path-node depth = {-pts.reshape(-1, 3)[:, 2].max():.3e};  "
                f"k_m = {g.k_m:.6g}"
            )
    finally:
        RazorSolver._refuse_buried_geometry = orig


if __name__ == "__main__":
    print("\n--- past the label guard ---")
    past_the_label_guard()
