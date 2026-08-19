"""momwire#282 stage 2: which contact decks the difference-of-columns can
actually measure, and how the gap's size moves with the antenna.

Two questions, one sweep, because they turned out to be the same question.

**Is `K = disc / (1 - C2)` a law?** The stage-2 record reports that the real
part of the momwire-vs-binary contact discrepancy tracks the uncancelled
image-charge fraction at about -10.3 over eps_r in [10, 30]. If that constant
were a property of the CONTACT it would survive changing the antenna, and the
envelope pins could be replaced by a gate on the law. It does not survive, so
they cannot, and the record says so — this is what measured it.

**Which decks may be differenced at all?** A difference of columns cancels a
CONSTANT formulation offset. It cannot cancel two offsets that are both still
moving. Stage 1 found one deck where that fails (study finding 4, the 0.42
lambda inverted-L); this sweep found three more, and they are the same three
decks whose `K` comes out incoherent. The PEC column offset printed per case
is the diagnostic, and `test_the_pec_columns_agree_well_enough_to_difference`
is the gate that came out of it.

Read the two together: where the PEC columns agree, `K` is stable and the
residual means something; where they do not, `K` walks and it does not.

Runs the binary, so: antennaknobs venv, NEC5_EXE set.

    NEC5_EXE=~/antennas/NEC5-downloads/nec5-linux/nec5cl \
        /home/smburns/antennas/antennaknobs/.venv/bin/python \
        scripts/probe_contact_deck_conditioning.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

C = 299792458.0
EPS0 = 8.8541878128e-12
SIGMA = 1e-5  # loss tangent <= 1e-2 across the sweep: nothing to dissipate
EPSR = [2.5, 4.0, 6.5, 10.0, 16.0, 30.0]
QW = 5.3535  # the lane's quarter wave at 14 MHz


def mono(h):
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, h]])


def bent(riser, top):
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, riser], [top, 0.0, riser]])


def grounded_u(h, d):
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, h], [d, 0.0, h], [d, 0.0, 0.0]])


# (label, freq MHz, radius, polyline, per-edge split)
CASES = [
    ("mono qw    14MHz a=5mm", 14.0, 0.005, mono(QW), [41]),
    ("mono qw     7MHz a=5mm", 7.0, 0.005, mono(QW), [41]),
    ("mono qw    21MHz a=5mm", 21.0, 0.005, mono(QW), [41]),
    ("mono qw    14MHz a=0.5mm", 14.0, 0.0005, mono(QW), [41]),
    ("mono qw    14MHz a=50mm", 14.0, 0.05, mono(QW), [41]),
    ("mono h/2   14MHz a=5mm", 14.0, 0.005, mono(QW / 2), [41]),
    ("mono 2h    14MHz a=5mm", 14.0, 0.005, mono(2 * QW), [41]),
    ("bent qw    14MHz a=5mm", 14.0, 0.005, bent(QW / 2, QW / 2), [24, 24]),
    ("grounded U 14MHz a=5mm", 14.0, 0.005, grounded_u(QW / 2, QW / 2), [24, 24, 24]),
]

# Above this the two codes' PEC columns are too far apart for a difference of
# columns to isolate the ground. The lane's own gate uses 1.5 ohm at its
# finest rung; this sweep runs one mesh per deck, so it only LABELS.
CONDITIONING_BAR = 1.5


def _num(x):
    return f"{float(x):.6E}"


def deck(poly, split, radius, freq, eps_r, sigma):
    cards = []
    for i, n in enumerate(split):
        p0, p1 = poly[i], poly[i + 1]
        cards.append(
            f"GW {i + 1} {n} {_num(p0[0])} {_num(p0[1])} {_num(p0[2])} "
            f"{_num(p1[0])} {_num(p1[1])} {_num(p1[2])} {_num(radius)}\n"
        )
    gn = (
        "GN 1 0 0 0\n"
        if eps_r is None
        else f"GN 0 0 0 0 {_num(eps_r)} {_num(sigma)} {_num(1.0)} {_num(0.0)} NOFILE\n"
    )
    return (
        "CM momwire#282 stage 2 — contact deck conditioning\nCE\n"
        + "".join(cards)
        + "GE 1 0\n"
        + gn
        + f"EX 0 1 1 1 {_num(1.0)} {_num(0.0)}\n"
        + f"FR 0 1 0 0 {_num(freq)} {_num(0.0)}\n"
        + "XQ 0\nEN\n"
    )


def momwire_z(poly, split, radius, freq, eps_r, sigma):
    from momwire import BSplineSolver

    kw = dict(
        wires=[poly],
        n_per_edge_per_wire=[split],
        wire_radius=radius,
        wavelength=C / (freq * 1e6),
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
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from antennaknobs.engines.nec5 import NEC5Engine

    sys.path.insert(0, str(Path.home() / "antennas/antennaknobs/scripts"))
    from bench_nec5_walk_why import make_dipole

    captures = Path(os.environ.get("STAGE2_CAPTURES", "/tmp/stage2-captures"))
    captures.mkdir(parents=True, exist_ok=True)
    eng = NEC5Engine(make_dipole(20), ground=None, capture_dir=captures)

    rows, summary = [], []
    for label, freq, radius, poly, split in CASES:
        z_pec_n5 = complex(
            eng.run_deck(deck(poly, split, radius, freq, None, 0.0))[0][0][2]
        )
        z_pec_mw = momwire_z(poly, split, radius, freq, None, 0.0)
        offset = abs(z_pec_mw - z_pec_n5)
        d_r = z_pec_mw.real - z_pec_n5.real
        flag = "" if offset <= CONDITIONING_BAR else "   <-- NOT DIFFERENCEABLE"
        print(f"\n{label}")
        print(
            f"  PEC: nec5={z_pec_n5:.3f}  momwire={z_pec_mw:.3f}  "
            f"|offset|={offset:.3f} (dR={d_r:+.3f}){flag}"
        )
        print(f"  {'eps_r':>7}  {'disc':>19}  {'Re(disc/(1-C2))':>17}")
        ks = []
        for eps_r in EPSR:
            z_n5 = complex(
                eng.run_deck(deck(poly, split, radius, freq, eps_r, SIGMA))[0][0][2]
            )
            z_mw = momwire_z(poly, split, radius, freq, eps_r, SIGMA)
            disc = (z_mw - z_pec_mw) - (z_n5 - z_pec_n5)
            omega = 2.0 * np.pi * freq * 1e6
            eps_t = eps_r - 1j * SIGMA / (omega * EPS0)
            # NOTE the spelling: Re(disc/(1-C2)), NOT disc.real/(1-C2). With a
            # complex eps~ the two differ, and only this one reproduces.
            k = (disc / (2.0 / (eps_t + 1.0))).real
            ks.append(k)
            print(f"  {eps_r:>7.4g}  {disc.real:>9.3f}{disc.imag:>+9.3f}j  {k:>17.3f}")
            rows.append(
                dict(
                    case=label,
                    freq=freq,
                    radius=radius,
                    eps_r=eps_r,
                    pec_offset=offset,
                    disc=[disc.real, disc.imag],
                    k_real=k,
                )
            )
        kk = np.array(ks)
        spread = kk.max() - kk.min()
        print(
            f"  {'':>7}  {'K mean':>19}  {kk.mean():>17.3f}"
            f"   spread {spread:.3f} ({100 * spread / abs(kk.mean()):.0f}%)"
        )
        summary.append((label, offset, kk.mean(), spread))

    print("\n\n=== SUMMARY: conditioning decides whether K means anything ===")
    print(f"{'deck':>26} {'PEC offset':>11} {'K mean':>9} {'K spread':>9}  verdict")
    print("-" * 78)
    for label, offset, kmean, spread in summary:
        ok = offset <= CONDITIONING_BAR
        print(
            f"{label:>26} {offset:>11.3f} {kmean:>9.2f} {spread:>9.2f}  "
            f"{'differenceable' if ok else 'REJECT — columns still moving'}"
        )
    print(
        "\nK is NOT a universal constant even on the differenceable decks "
        "(-10.3 straight,\n-8.6 bent), which is why the record reports the "
        "(1 - C2) relation as a description\nand gates the envelope pins "
        "instead."
    )

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
