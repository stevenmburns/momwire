"""momwire#282 stage 2: the two INTERNAL structure probes behind the study's
"stated as a hypothesis" paragraph.

The stage-2 record says the gap is consistent with living in §2.2's third row
— what continues the current into the earth at the contact node — and then
says two things stop that being a finding rather than a hypothesis. Both were
measured, and this is what measured them. Neither needs the binary.

`--mode symmetry`
    A dropped testing-side bracket `[f_m Phi_n]` at the grounded end would be
    the obvious sharp check, because it would break `Z = Z^T`. It cannot
    fire: both sides of the Phi term carry a basis DERIVATIVE, so the weak
    form is symmetric whatever the bracket does. This prints the number that
    says so, at PEC and on every finite ground, contact and clearance alike.

`--mode phi-weight`
    The charge-conservation reading: the grounded basis integrates to 1 at
    the node and its image to `-w_Phi`, so the composite carries a net
    contact charge proportional to `1 - w_Phi = 1 - C2`. Forcing `w_Phi = 1`
    is the naive restoration. It moves the residual the WRONG way, and — the
    part that matters — it is not the local operation it looks like: the same
    patch moves the CLEARANCE deck by tens of ohms, because `w_Phi` is the
    ground's charge response for those segments and not a property of the
    junction. Whatever the right compensating term is, it is not a
    re-weighting of the existing table.

    That clearance control is also what shows study §5.4 candidate 2's stub
    ladder cannot discriminate: `w_Phi` is shared with the stubbed geometry,
    so a momwire-vs-momwire self-consistency test is blind to it.

This script exists because the numbers it produces are quoted in a permanent
design record. A number in the record with no script behind it is a number
the next reader cannot check.

    python scripts/probe_contact_node_structure.py --mode symmetry
    python scripts/probe_contact_node_structure.py --mode phi-weight
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from golden_contact_nec5 import CONTACT_LADDERS, GROUND_EPS

from momwire import BSplineSolver, _potential_ground

C = 299792458.0
FREQ_MHZ = 14.0
WL = C / (FREQ_MHZ * 1e6)
RAD = 0.005
MONO_H = 5.3535

# The grounds this probe reports on, coarse-to-fine in |eps~|. `diel` is
# stage 2's lossless dielectric; the rest are the lane's soils.
PROBE_GROUNDS = ("avg", "poor", "diel")


def _kw(n, ground, clearance=0.0):
    kw = dict(
        wires=[np.array([[0.0, 0.0, clearance], [0.0, 0.0, clearance + MONO_H]])],
        n_per_edge_per_wire=[[n]],
        wire_radius=RAD,
        wavelength=WL,
        degree=2,
        feed_model="segment",
        feed_wire_index=0,
        feed_arclength=0.0,
        ground_z=0.0,
    )
    if ground != "pec":
        kw.update(ground_eps=GROUND_EPS[ground], ground_model="sommerfeld")
    return kw


# --------------------------------------------------------------------------
# mode: symmetry
# --------------------------------------------------------------------------
def captured_Z(n, ground, clearance=0.0):
    """The loaded dense operator `_compute_Z_operator` built, kept."""
    holder = {}
    orig = BSplineSolver._compute_Z_operator

    def spy(self, geom, supp_seg, polys, same_edge_prep=None):
        Z = orig(self, geom, supp_seg, polys, same_edge_prep=same_edge_prep)
        holder["Z"] = np.array(Z)
        return Z

    BSplineSolver._compute_Z_operator = spy
    try:
        z_in, _ = BSplineSolver(**_kw(n, ground, clearance)).compute_impedance()
    finally:
        BSplineSolver._compute_Z_operator = orig
    return holder["Z"], complex(z_in)


def run_symmetry(n):
    print("max|Z - Z^T| / max|Z| — Galerkin symmetry, which a dropped")
    print("testing-side bracket would break if the weak form could see it\n")
    print(f"{'deck':>26} {'max|Z-Z^T|':>13} {'relative':>12}")
    print("-" * 53)
    for label, clearance in (("contact", 0.0), ("clearance 0.25 m", 0.25)):
        for ground in ("pec", *PROBE_GROUNDS):
            Z, _ = captured_Z(n, ground, clearance)
            A = Z - Z.T
            rel = np.max(np.abs(A)) / np.max(np.abs(Z))
            print(
                f"{label + ' / ' + ground:>26} {np.max(np.abs(A)):>13.4e} {rel:>12.3e}"
            )
    print(
        "\nSame order everywhere, contact and clearance alike: this test is a "
        "NULL\nby construction, not evidence that no bracket is missing."
    )


# --------------------------------------------------------------------------
# mode: phi-weight
# --------------------------------------------------------------------------
def solve_patched(n, ground, *, mode="off", k_base=3, clearance=0.0):
    """`mode="off"` is the shipped path; `"all"` forces `w_Phi = 1`
    everywhere; `"base"` forces it on pairs touching the bottom `k_base`
    segments, symmetrised over the pair so the weak form stays symmetric."""
    if mode == "off" or ground == "pec":
        z, _ = BSplineSolver(**_kw(n, ground, clearance)).compute_impedance()
        return complex(z)

    def force(w_Phi, i0=0):
        w_Phi = np.array(w_Phi, dtype=np.complex128)
        if mode == "all":
            w_Phi[...] = 1.0
        else:
            w_Phi[:, :k_base] = 1.0
            rows = [r for r in range(w_Phi.shape[0]) if i0 + r < k_base]
            if rows:
                w_Phi[rows, :] = 1.0
        return w_Phi

    orig_t = _potential_ground.PotentialGround.weight_tables
    orig_w = _potential_ground.PotentialGround.weight_windows

    def patched_tables(self, prep=None):
        tables = orig_t(self, prep=prep)
        if tables is None:
            return None
        w_A, w_Phi = tables
        return w_A, force(w_Phi)

    def patched_windows(self, observers=None, sources=None):
        producer = orig_w(self, observers=observers, sources=sources)

        def wrapped(i0, i1):
            w_A, w_Phi = producer(i0, i1)
            return w_A, force(w_Phi, i0)

        return wrapped

    _potential_ground.PotentialGround.weight_tables = patched_tables
    _potential_ground.PotentialGround.weight_windows = patched_windows
    try:
        z, _ = BSplineSolver(**_kw(n, ground, clearance)).compute_impedance()
    finally:
        _potential_ground.PotentialGround.weight_tables = orig_t
        _potential_ground.PotentialGround.weight_windows = orig_w
    return complex(z)


def run_phi_weight(n, k_bases):
    nec = {g: dict(CONTACT_LADDERS["monopole"][g]) for g in ("pec", *PROBE_GROUNDS)}
    z_pec = solve_patched(n, "pec")

    print(f"CONTACT, N = {n}: does restoring the image CHARGE close the gap?\n")
    hdr = f"{'ground':>7} {'variant':>15}  {'d_momwire':>19}  {'residual':>9}"
    print(hdr)
    print("-" * len(hdr))
    for ground in PROBE_GROUNDS:
        d_n5 = nec[ground][n] - nec["pec"][n]
        variants = [("shipped", dict(mode="off"))]
        variants += [(f"w_Phi=1 base{k}", dict(mode="base", k_base=k)) for k in k_bases]
        variants += [("w_Phi=1 all", dict(mode="all"))]
        for label, kw in variants:
            d = solve_patched(n, ground, **kw) - z_pec
            print(
                f"{ground:>7} {label:>15}  {d.real:>9.3f}{d.imag:>+9.3f}j  "
                f"{abs(d - d_n5):>9.4f}"
            )
        print()

    print("CLEARANCE control, base lifted 0.25 m — the patch is NOT local.\n")
    z_pec_l = solve_patched(n, "pec", clearance=0.25)
    hdr = f"{'ground':>7}  {'shipped':>19}  {'w_Phi=1 base3':>19}  {'moved':>9}"
    print(hdr)
    print("-" * len(hdr))
    moves = []
    for ground in PROBE_GROUNDS:
        base = solve_patched(n, ground, clearance=0.25) - z_pec_l
        loc = solve_patched(n, ground, mode="base", k_base=3, clearance=0.25) - z_pec_l
        moves.append(abs(loc - base))
        print(
            f"{ground:>7}  {base.real:>9.3f}{base.imag:>+9.3f}j  "
            f"{loc.real:>9.3f}{loc.imag:>+9.3f}j  {abs(loc - base):>9.4f}"
        )
    print(
        f"\nClearance movement spans {min(moves):.1f}-{max(moves):.1f} ohm on a deck "
        "with NO contact node,\nwhich is what makes `w_Phi` a property of the "
        "ground and not of the junction."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("symmetry", "phi-weight"), default="phi-weight")
    p.add_argument("--n", type=int, default=41)
    p.add_argument("--k-base", type=int, nargs="+", default=[1, 2, 3, 5])
    args = p.parse_args()
    if args.mode == "symmetry":
        run_symmetry(args.n)
    else:
        run_phi_weight(args.n, args.k_base)


if __name__ == "__main__":
    main()
