"""momwire#510 experiment 1 — where does the grazing-height floor start?

The finding this answers
------------------------
Captures 0033 and 0034 are one elevated radial system — a 35.56 m vertical
over four 39.624 m radials at 1.832 MHz, everything but the vertical standing
1.778 cm over ``GN 0`` average soil, which is **h/λ = 1.09e-4** of the 163.6 m
wavelength.  Every basis answers 176-437 % away from the capture with the
reactance sign FLIPPED, and the seam serves it silently
(``test_eznec_serve.DIVERGENT_IDS`` excuses the two from the accuracy gates
and says nothing to a user).

"Wrong at 1.09e-4 λ" is not yet a fact anyone can act on.  A refusal needs a
threshold, a documented limit needs a number, and a bug needs a signature.
All three want the same thing: the error as a FUNCTION of height, with an
onset.

The instrument
--------------
The antenna is lifted RIGIDLY — vertical length, radial length, radius, mesh
and drive card all bit-identical from rung to rung, with only the whole
structure's z translated.  So the only variable on the sweep is h, and the
comparison at each rung is against the licensed binary running the same deck
text.  Z genuinely moves with height (an elevated radial system is height
dependent physics, not a flat ladder), which is exactly why the reference has
to be the binary at every rung rather than the smallest rung's own answer.

Two controls, both free, both load-bearing:

* **``GN 1`` perfect ground at every rung.**  A perfect image has no
  Sommerfeld integral in it.  If the PEC ladder tracks the binary down to
  1e-4 λ and the finite one does not, the floor is in the ground model and
  not in the mesh, the junction, the drive card or the seam.
* **both shipped trunks** — ``bspline`` (degree 2) and ``razor-nec5``.  #593
  ships two executables and the standing user ruling is "both serve or don't
  serve", so a threshold that fits one trunk and not the other is not a
  threshold.  They disagree with each other by 100 Ω on this deck at the
  native height, which is itself part of the signature.

``--mode segments`` walks the second axis at a FIXED height: the native deck
meshes a 39.624 m radial in five segments, so Δ = 7.92 m sits 445× above the
1.78 cm height.  Whether the floor is set by h/λ or by h/Δ decides whether
this is a ground-model limit or a mesh rule, and the two axes cannot be read
apart from the h sweep alone.

WHAT IT MEASURED (2026-08-25)
-----------------------------
``--mode height``, error against the binary in per cent of ``|Z|``:

  h/λ        1e-1  3e-2  2e-2  1e-2  7e-3  5e-3  3e-3   2e-3   1e-3   5e-4   2e-4   1.09e-4
  razor GN0  0.01  0.00  0.00  0.06  0.25  0.87  3.91  10.31  44.18 230.91 239.54  171.86
  razor GN1  0.00  0.00  0.00  0.00  0.00  0.00  0.00   0.00   0.00   0.01   0.01    0.00
  bspl  GN0  4.68  5.06  5.30  5.92  6.40  7.14  9.72  14.84  45.66  37.23 167.08  435.18
  bspl  GN1  4.72  5.17  5.39  5.86  ----  ----  6.87   ----   8.23  10.13  15.78   24.17

**The floor is between 1e-2 and 1e-3 λ, and it is in the GROUND.**  Over a
perfect image razor-nec5 holds 0.00 % at every rung down to 1.09e-4 λ — same
mesh, same five-wire junction, same ``EX 0,1,-1`` card, same seam — so the
grazing failure is not the mesh, not the junction, not the drive spelling and
not this seam's addressing.  Only the ``GN 0`` card changes.

Both trunks leave their own baseline at the same place (razor's baseline is
0.00 %, bspline's is the ~5 % basis difference it carries at every height
including over PEC), which is what makes ONE threshold serve both and the
"both serve or don't serve" ruling satisfiable:

    clean ≥ 1e-2 λ · ~1 % at 5e-3 · ~10 % at 2e-3 · broken ≤ 1e-3

``--mode segments`` — native height, mesh refined 5 → 41 segments a wire —
says the controlling variable is **h/λ and not h/Δ**:

  N           5      9     15     25     41
  h/delta   0.0022 0.0040 0.0067 0.0112 0.0184
  nec5cl    38.79  40.69  41.30  41.54  41.63    -49.58 -42.16 -39.80 -38.81 -38.36j
  razor%   175.88 158.64 358.73 703.13 276.32
  bspl%    437.23 189.29  47.42  68.22 217.04

The binary CONVERGES monotonically under refinement; both trunks diverge, and
erratically.  N=41 puts h/Δ at 0.0184 — the same h/Δ the height sweep reaches
at h/λ = 1e-3, where razor is 44 % out — yet here it is 276 %.  So refining
the mesh does not buy the answer back, it costs more of it.

That is a **breakdown signature, not a model gap**: a bounded formulation
error is monotone in the mesh and keeps its sign, and this is neither (razor
walks −50j → +113j → −231j → +34j across four adjacent rungs).  D3's category,
which is the reading the release decision needs.

Runs the binary, so: antennaknobs venv, ``NEC5_EXE`` set.

    NEC5_EXE=~/antennas/NEC5-downloads/nec5-linux/nec5cl \
        /home/smburns/antennas/antennaknobs/.venv/bin/python \
        scripts/probe_grazing_height_floor.py --mode height
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

C = 299792458.0

# 0033's operating point and geometry, read off
# `tests/fixtures/eznec/decks/0033_elevated-radial-system.nec`.
FREQ_MHZ = 1.832
WL = C / (FREQ_MHZ * 1e6)  # 163.6 m
NATIVE_H = 0.01778001  # the radial plane's height, m
VERT_LEN = 35.56 - NATIVE_H  # the vertical, measured from the radial plane
RADIAL_LEN = 39.624
RADIUS = 0.001294
N_SEG = 5  # segments per wire, 0033's own mesh
SOIL = (13.0, 0.005)  # `GN 0 0 0 0 13. .005` — average

# The sweep, as h/λ.  The first rung IS the captured deck; the last is a
# tenth of a wavelength, well inside refl-coef's own documented 0.1-0.5 λ
# validity window and so a rung the ground model is not in dispute at.
HEIGHT_SWEEP = (
    1.09e-4,
    2e-4,
    5e-4,
    1e-3,
    2e-3,
    3e-3,
    5e-3,
    7e-3,
    1e-2,
    2e-2,
    3e-2,
    1e-1,
)

# The mesh axis, at the native height.  Five is 0033's own.
SEGMENT_SWEEP = (5, 9, 15, 25, 41)

TRUNKS = ("bspline", "razor-nec5")


def _num(x: float) -> str:
    return f"{float(x):.8E}"


def deck(
    h: float,
    *,
    n_seg: int = N_SEG,
    pec: bool = False,
    soil: tuple[float, float] = SOIL,
) -> str:
    """0033, translated to radial height ``h``.

    The structure above the radial plane is unchanged: the vertical keeps its
    length rather than its tip height, so lifting the antenna does not also
    shorten it.  The drive card is the capture's own ``EX 0,1,-1`` — node
    addressing, the ``-1`` spelling of node 0 — so this walks the same route
    through the seam that the captured deck does.
    """
    top = h + VERT_LEN
    wires = [f"GW 1,{n_seg},0.,0.,{_num(h)},0.,0.,{_num(top)},{_num(RADIUS)}"]
    for tag, (dx, dy) in enumerate(
        ((RADIAL_LEN, 0.0), (0.0, RADIAL_LEN), (-RADIAL_LEN, 0.0), (0.0, -RADIAL_LEN)),
        start=2,
    ):
        wires.append(
            f"GW {tag},{n_seg},0.,0.,{_num(h)},"
            f"{_num(dx)},{_num(dy)},{_num(h)},{_num(RADIUS)}"
        )
    gn = "GN 1,0,0,0" if pec else f"GN 0,0,0,0,{_num(soil[0])},{_num(soil[1])},1.,0."
    return (
        "CM momwire#510 — elevated radial system, grazing-height sweep\n"
        f"CM h = {h:.6g} m = {h / WL:.4g} wavelengths\nCE\n"
        + "\n".join(wires)
        + "\nGE 1,-1\n"
        + f"FR 0,1,0,0,{_num(FREQ_MHZ)}\n"
        + gn
        + "\nEX 0,1,-1,0,1.414214,0.\nXQ 0\nEN\n"
    )


def momwire_z(text: str, basis: str) -> complex | None:
    """What one trunk answers for this deck, or ``None`` if it refuses."""
    from momwire.deck._nec5 import parse_nec5
    from momwire.eznec import _serve

    try:
        data = _serve.serve(parse_nec5(text), basis=basis)
    except _serve.ServeRefusal as exc:
        print(f"      REFUSED ({basis}): {exc}")
        return None
    return complex(data.sources[0].impedance)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("height", "segments"), default="height")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    exe = Path(os.path.expanduser(os.environ.get("NEC5_EXE", "")))
    if not exe.is_file():
        raise SystemExit(f"NEC5_EXE not found: {exe}")

    import sys

    sys.path.insert(0, str(Path.home() / "antennas/antennaknobs/scripts"))
    from antennaknobs.engines.nec5 import NEC5Engine
    from bench_nec5_walk_why import make_dipole

    captures = Path(
        os.environ.get("GRAZING_CAPTURES", "/tmp/claude-1000/510-grazing-captures")
    )
    captures.mkdir(parents=True, exist_ok=True)
    # The engine's own geometry is never consulted by `run_deck`; it exists
    # only to construct the wrapper.
    eng = NEC5Engine(make_dipole(20), ground=None, capture_dir=captures)

    if args.mode == "height":
        cases = [(h * WL, N_SEG) for h in HEIGHT_SWEEP]
    else:
        cases = [(NATIVE_H, n) for n in SEGMENT_SWEEP]

    print(f"0033 lifted rigidly — {FREQ_MHZ} MHz, lambda = {WL:.2f} m")
    print(f"vertical {VERT_LEN:.5f} m over four {RADIAL_LEN} m radials, a = {RADIUS} m")
    print(f"soil: eps_r = {SOIL[0]}, sigma = {SOIL[1]}  (GN 0, average)\n")

    rows = []
    for pec in (True, False):
        label = (
            "GN 1 PERFECT GROUND (control — no Sommerfeld integral)"
            if pec
            else ("GN 0 AVERAGE SOIL (the deck's own card)")
        )
        print(f"\n{label}\n")
        hdr = (
            f"{'h/lambda':>10} {'h (m)':>10} {'N':>3} {'h/delta':>9} | "
            f"{'nec5cl':>21} | " + " | ".join(f"{t:>21} {'err%':>7}" for t in TRUNKS)
        )
        print(hdr)
        print("-" * len(hdr))
        for h, n_seg in cases:
            text = deck(h, n_seg=n_seg, pec=pec)
            z_ref = complex(eng.run_deck(text)[0][0][2])
            delta = RADIAL_LEN / n_seg
            line = (
                f"{h / WL:>10.3g} {h:>10.5g} {n_seg:>3} {h / delta:>9.3g} | "
                f"{z_ref.real:>10.4f}{z_ref.imag:>+10.4f}j |"
            )
            row = dict(
                mode=args.mode,
                pec=pec,
                h=h,
                h_over_wl=h / WL,
                n_seg=n_seg,
                h_over_delta=h / delta,
                z_ref=[z_ref.real, z_ref.imag],
            )
            for trunk in TRUNKS:
                z = momwire_z(text, trunk)
                if z is None:
                    line += f" {'refused':>21} {'--':>7} |"
                    row[trunk] = None
                    continue
                err = 100.0 * abs(z - z_ref) / abs(z_ref)
                line += f" {z.real:>10.4f}{z.imag:>+10.4f}j {err:>7.2f} |"
                row[trunk] = [z.real, z.imag]
                row[f"{trunk}_err_pct"] = err
            print(line, flush=True)
            rows.append(row)

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
