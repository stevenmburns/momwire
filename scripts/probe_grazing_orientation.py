"""momwire#510 experiment 3 — is the grazing floor #624's contact node?

The question the plan poses
--------------------------
As h -> 0 the radials of 0033 approach ground CONTACT, which is its own known
problem (momwire#624).  "These may be one mechanism seen from two sides" — and
if they are, #510 and #624's stage 3 collapse into one investigation.

The arithmetic already says the two OVERLAP.  #624's stub ladder
(``spike_contact_stub_ladder.py``) sweeps stub heights 0.1, 0.03, 0.01, 0.003,
0.001 m at 14 MHz, where λ = 21.414 m — so its rungs sit at

    h/λ = 4.67e-3, 1.40e-3, 4.67e-4, 1.40e-4, 4.67e-5

which straddles the floor experiment 1 measured (clean ≥ 1e-2 λ, broken
≤ 1e-3).  The whole #624 ladder lives inside #510's broken zone.  Yet #624
measured a contact node that is BOUNDED there — 0.21-0.55 Ω of ladder spread,
and razor at contact sitting at bspline's accuracy on all four grounds — while
#510 at the same h/λ is hundreds of ohms out with the reactance sign flipped.
Same regime, two very different magnitudes.  So overlap is not identity, and
something other than height alone separates them.

The discriminator
-----------------
The obvious candidate is ORIENTATION.  A vertical wire's image ADDS to it; a
horizontal wire's image CANCELS it, and the cancellation becomes total as
h -> 0.  #624's stub is vertical; 0033's four 39.6 m radials are horizontal
and carry most of the structure.  Two modes separate that from the alternative
— that what matters is the grazing FEEDPOINT, which both cases share.

``--mode tilt``   The feed junction is pinned at 0033's own 1.778 cm while the
                  radials are TILTED UP about it, far end walked from grazing
                  to 16 m.  Radial LENGTH is held at 39.624 m (the horizontal
                  reach shrinks to compensate), so the antenna keeps its
                  electrical size and only its proximity to the plane changes.
                  If the error survives the tilt, the mechanism is at the
                  grazing junction and #510 is #624 from above.  If it
                  collapses as the radials leave the grazing zone, the
                  mechanism is the horizontal wire near the interface and the
                  two are different animals.

``--mode lone``   The vertical ALONE, no radials, its base node driven and
                  swept in height — the same ``EX 0,1,-1`` card, so nothing
                  about the drive spelling changes.  A vertical grazing the
                  plane, with no horizontal wire anywhere in the model.

Both go through the seam route experiment 1 used, and that choice is earned:
razor-nec5 reproduced the binary to 0.00 % over ``GN 1`` at every rung of that
sweep, which is proof the seam's addressing, mesh and feed spelling are exact
when the ground is not in play.  The ``GN 1`` control is carried here for the
same reason.

WHAT IT MEASURED (2026-08-25)
-----------------------------
``--mode tilt`` — the feed junction pinned at 1.09e-4 λ throughout, error
against the binary in per cent of ``|Z|``:

  radial far end   0.018   0.05   0.164    0.5   1.64      5    16.4  m
  (/λ)            1.1e-4 3.1e-4   1e-3  3.1e-3   1e-2 3.1e-2   1e-1
  razor GN 0      175.88 229.66  413.70  60.67  10.04   1.04    0.01
  razor GN 1        0.00   0.00    0.00   0.00   0.00   0.00    0.00
  bspline GN 0    437.23 208.03   72.24  26.16  11.04   6.20    3.00

``--mode lone`` — the vertical alone, no radials, base swept:

  base h/λ        1.09e-4   5e-4   1e-3   3e-3   1e-2   3e-2   1e-1
  razor GN 0         0.00   0.00   0.00   0.00   0.01   0.00    0.00
  bspline GN 0       6.42   6.24   6.05   5.61   5.13   4.92    4.86

**#510 is NOT #624 seen from above.**  The grazing FEEDPOINT is innocent: the
junction never leaves 1.09e-4 λ across the tilt sweep and razor-nec5 still
lands at 0.01 % once the radials climb away, and a lone vertical whose bottom
segment grazes the plane is exact at 0.00 % at every height over ``GN 0``.
What breaks is the **horizontal wire lying in the grazing zone** — which is
what #624's vertical stub does not have and never did.  The two mechanisms do
not collapse into one investigation.

And the mechanism reads as **catastrophic cancellation in the numerical
ground, not cancellation itself**.  A horizontal wire's image current is
antiparallel, so as h -> 0 every coupling through the plane becomes the small
difference of two nearly equal large quantities.  Over ``GN 1`` that
difference is a closed-form image and razor holds 0.00 % at 1.09e-4 λ with the
radials flat — the SAME cancellation, exactly computed.  Over ``GN 0`` the
image is an integral evaluated numerically, its absolute error does not shrink
with h, and the relative error in the difference blows up.  That predicts
exactly what experiment 1 saw: error growing as refinement puts more unknowns
in the regime, sign non-monotone rung to rung, and a hard onset where the
cancellation gets deep enough to eat the integrator's accuracy.

Which reopens the outlook.  A formulation limit would be refuse-only; an
accuracy floor in the Sommerfeld evaluation near the interface is a FIXABLE
target, and the next lever is whether the onset moves with the integrator's
own accuracy setting (the rtol / interpolation-grid axis
``probe_contact_direct_remainder.py`` found saturated at CONTACT — a different
regime, so it has to be asked again here).

Runs the binary, so: antennaknobs venv, ``NEC5_EXE`` set.

    NEC5_EXE=~/antennas/NEC5-downloads/nec5-linux/nec5cl \
        /home/smburns/antennas/antennaknobs/.venv/bin/python \
        scripts/probe_grazing_orientation.py --mode tilt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from probe_grazing_height_floor import (
    N_SEG,
    NATIVE_H,
    RADIAL_LEN,
    RADIUS,
    SOIL,
    TRUNKS,
    VERT_LEN,
    WL,
    FREQ_MHZ,
    _num,
    momwire_z,
)

# The radials' far-end height, in metres.  The first rung is 0033 itself
# (flat), the last is 1e-1 λ — the rung experiment 1 found unambiguously
# clean.  1.6364 m is 1e-2 λ, the top of the floor.
TILT_SWEEP = (NATIVE_H, 0.05, 0.16364, 0.5, 1.6364, 5.0, 16.364)

# The lone vertical's base height, as h/λ — experiment 1's sweep, trimmed to
# the rungs that matter either side of the floor.
LONE_SWEEP = (1.09e-4, 5e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


def tilt_deck(h_far: float, *, pec: bool = False) -> str:
    """0033 with its radials hinged up about a feed junction pinned at
    :data:`NATIVE_H`.

    The radial keeps its 39.624 m LENGTH and gives up horizontal reach, so the
    sweep changes where the wire sits relative to the plane and not how much
    wire there is.  A tilt that changed length would move the resonance and
    the error column would be reading two things at once.
    """
    h = NATIVE_H
    rise = h_far - h
    if abs(rise) > RADIAL_LEN:
        raise ValueError("radial cannot rise further than it is long")
    reach = math.sqrt(RADIAL_LEN**2 - rise**2)
    top = h + VERT_LEN
    wires = [f"GW 1,{N_SEG},0.,0.,{_num(h)},0.,0.,{_num(top)},{_num(RADIUS)}"]
    for tag, (ux, uy) in enumerate(
        ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)), start=2
    ):
        wires.append(
            f"GW {tag},{N_SEG},0.,0.,{_num(h)},"
            f"{_num(ux * reach)},{_num(uy * reach)},{_num(h_far)},{_num(RADIUS)}"
        )
    gn = "GN 1,0,0,0" if pec else f"GN 0,0,0,0,{_num(SOIL[0])},{_num(SOIL[1])},1.,0."
    return (
        "CM momwire#510 — elevated radial system, radials tilted up\n"
        f"CM feed junction pinned at {h:.6g} m, radial far end {h_far:.6g} m\nCE\n"
        + "\n".join(wires)
        + "\nGE 1,-1\n"
        + f"FR 0,1,0,0,{_num(FREQ_MHZ)}\n"
        + gn
        + "\nEX 0,1,-1,0,1.414214,0.\nXQ 0\nEN\n"
    )


def lone_deck(h: float, *, pec: bool = False) -> str:
    """0033's vertical alone, lifted to ``h``, driven at its first interior
    node.

    No horizontal wire anywhere in the model and no junction — the grazing
    element is the bottom segment of a vertical.  That is #624's stub geometry
    scaled up, asked through #510's seam and reference.

    The drive moves off the base node that the tilt mode uses, and it has to:
    a wire END standing free carries no basis function, and the binary says so
    rather than guessing (``SORVT1: ERROR - Voltage source specified where
    there is no basis function``).  0033 can drive its base because five wires
    meet there; a lone wire cannot.  Node 1 — the first interior node, one
    segment up — is the nearest thing to it that exists.
    """
    top = h + VERT_LEN
    gn = "GN 1,0,0,0" if pec else f"GN 0,0,0,0,{_num(SOIL[0])},{_num(SOIL[1])},1.,0."
    return (
        "CM momwire#510 — the vertical alone, base grazing\n"
        f"CM base at {h:.6g} m = {h / WL:.4g} wavelengths\nCE\n"
        f"GW 1,{N_SEG},0.,0.,{_num(h)},0.,0.,{_num(top)},{_num(RADIUS)}\n"
        "GE 1,-1\n"
        f"FR 0,1,0,0,{_num(FREQ_MHZ)}\n" + gn + "\nEX 0,1,1,0,1.414214,0.\nXQ 0\nEN\n"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("tilt", "lone"), default="tilt")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    exe = Path(os.path.expanduser(os.environ.get("NEC5_EXE", "")))
    if not exe.is_file():
        raise SystemExit(f"NEC5_EXE not found: {exe}")

    sys.path.insert(0, str(Path.home() / "antennas/antennaknobs/scripts"))
    from antennaknobs.engines.nec5 import NEC5Engine
    from bench_nec5_walk_why import make_dipole

    captures = Path(
        os.environ.get("GRAZING_CAPTURES", "/tmp/claude-1000/510-grazing-captures")
    )
    captures.mkdir(parents=True, exist_ok=True)
    eng = NEC5Engine(make_dipole(20), ground=None, capture_dir=captures)

    if args.mode == "tilt":
        print(f"0033 with its radials tilted up about a junction at {NATIVE_H} m")
        print(f"radial length held at {RADIAL_LEN} m; lambda = {WL:.2f} m")
        cases = [(v, tilt_deck) for v in TILT_SWEEP]
        col, unit = "far end", "m"
    else:
        print("0033's vertical alone, base node driven, no radials")
        print(f"vertical {VERT_LEN:.5f} m; lambda = {WL:.2f} m")
        cases = [(v * WL, lone_deck) for v in LONE_SWEEP]
        col, unit = "base h", "m"
    print(f"soil: eps_r = {SOIL[0]}, sigma = {SOIL[1]}  (GN 0, average)\n")

    rows = []
    for pec in (True, False):
        print(f"\n{'GN 1 PERFECT GROUND (control)' if pec else 'GN 0 AVERAGE SOIL'}\n")
        hdr = (
            f"{col + ' (' + unit + ')':>13} {'/lambda':>9} | {'nec5cl':>21} | "
            + " | ".join(f"{t:>21} {'err%':>8}" for t in TRUNKS)
        )
        print(hdr)
        print("-" * len(hdr))
        for value, builder in cases:
            text = builder(value, pec=pec)
            z_ref = complex(eng.run_deck(text)[0][0][2])
            line = (
                f"{value:>13.5g} {value / WL:>9.3g} | "
                f"{z_ref.real:>10.4f}{z_ref.imag:>+10.4f}j |"
            )
            row = dict(
                mode=args.mode,
                pec=pec,
                value=value,
                over_wl=value / WL,
                z_ref=[z_ref.real, z_ref.imag],
            )
            for trunk in TRUNKS:
                z = momwire_z(text, trunk)
                if z is None:
                    line += f" {'refused':>21} {'--':>8} |"
                    row[trunk] = None
                    continue
                err = 100.0 * abs(z - z_ref) / abs(z_ref)
                line += f" {z.real:>10.4f}{z.imag:>+10.4f}j {err:>8.2f} |"
                row[trunk] = [z.real, z.imag]
                row[f"{trunk}_err_pct"] = err
            print(line, flush=True)
            rows.append(row)

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
