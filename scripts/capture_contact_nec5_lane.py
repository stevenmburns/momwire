"""Capture the GROUND-CONTACT-over-a-FINITE-GROUND lane against the NEC-5
binary, and regenerate `tests/golden_contact_nec5.py`.

momwire#282 stage 1. The sibling of `scripts/capture_razor_contact_nec5_lane.py`
— same run recipe, same generated-literals discipline — and the instrument
`docs/design/contact-over-finite-ground.md` asks for: the study measured a
capability that had shipped ungated since momwire#151 and found a reference
for it, and this script is that reference turned into a lane the momwire
test suite can run with no binary and no antennaknobs dependency.

Must run under the antennaknobs venv (that is where the NEC-5 engine wrapper
lives), with the binary on `NEC5_EXE`:

    NEC5_EXE=~/antennas/NEC5-downloads/nec5-linux/nec5cl \
        /home/smburns/antennas/antennaknobs/.venv/bin/python \
        scripts/capture_contact_nec5_lane.py

Only the binary's PRINTED impedances are recorded; nothing about NEC-5's
internals is read, quoted or inferred here.

WHAT IS GATED, AND WHY IT IS A DIFFERENCE OF COLUMNS
----------------------------------------------------
Not `Z`. At PEC contact, N = 41, momwire and the binary already sit 1.26 Ω
apart in X (40.643+22.018j printed against `BSplineSolver(degree=2)`'s
40.662+23.278j) and every bit of that is basis difference — two
formulations discretizing the same antenna. Gating `Z` would gate the
formulation gap and drown the ground in it.

What is gated is the GROUND-INDUCED SHIFT

    delta = Z(soil) - Z(PEC)          at matched geometry and matched N

which cancels the formulation's own offset and leaves the thing under test:
what the ground did. This is the pattern `docs/design/solver-architecture.md`
§6.5 used for the clearance case, applied to contact.

TWO BAR SHAPES, BECAUSE THE TABLE HAS TWO BEHAVIOURS
-----------------------------------------------------
The study's §3.5 measured `|delta_momwire - delta_nec5|` down the ladder and
found the rows do two different things, so they get two different claims
(maintainer decision D2, 2026-08-18):

  * HIGH-|eps~| grounds — sea water and "very good" — CONVERGE. The residual
    shrinks with mesh (sea 0.70 -> 0.31, very good 0.98 -> 0.005/0.12). Those
    rows are gated by DECAY: the finest rung at its measured level + 25 %,
    and the ladder monotone into it. That is a real agreement claim.

  * LOW-eps_r grounds — average and poor soil — DIVERGE, then SATURATE. The
    residual GROWS with mesh and flattens (average 0.67 -> 1.24, poor
    2.92 -> 3.31). A gap that grows under refinement and then stops is a
    difference of LIMITS, not a discretization artifact, and there is no
    honest tight bar to put on it. Those rows are gated by an ENVELOPE PIN
    at the measured saturation, plus a check that it really has saturated —
    which is what makes the envelope a claim rather than a shrug.

The asymmetry is the honest outcome and it is written into the gate rather
than averaged away. The envelope rows are momwire#282 STAGE 2's subject:
momwire under-predicts the ground-loss resistance of a grounded vertical
over poor soil by ~2.7 Ω, the cause is not known, and study §5.4 names three
candidates and the experiment that separates them. If stage 2 closes it,
these pins are what should fail.

THE DECKS
---------
Two geometries at 14 MHz (lambda = 21.414 m), radius 5 mm, base in the
plane, driven THERE (`EX 0 1 1 1`, momwire's `feed_arclength=0.0`), with
`GE 1 0`:

  monopole    a 5.3535 m vertical — the study's headline deck, on the
              study's own N = 11/21/41/61/81 ladder so this capture can be
              read directly against §3.5's published table
  inverted-L  a 2.675 m riser plus a 2.675 m top wire — the same quarter
              wave, bent, so the contact unknown is exercised on a wire
              that also has an interior bend and a second edge. It is here
              to answer the one question the headline table cannot: whether
              the low-eps_r gap is a property of the ground or of a
              vertical.

              NOT the study's 3 m + 6 m inverted-L (§3.2 / S6), and the
              reason is measured. That deck is 0.42 lambda — near the
              grounded half-wave antiresonance, where |Z| ~ 1050 Ω and
              nothing is converged: over N = 12 -> 96 the binary's own PEC
              answer walks 264+911j -> 328+1004j (+24 % in R) and
              `BSplineSolver`'s walks 133+659j -> 269+917j (+103 %). The
              difference of columns cancels the formulation offset, but it
              cannot cancel two offsets that are both still moving, and the
              residuals come out at 30-95 Ω on every ground including the
              high-|eps~| ones the monopole closes to 0.27 Ω. That is a
              statement about an antiresonant deck at coarse mesh, not
              about ground contact, and gating it would gate the wrong
              thing. Captured for the record it would still be honest;
              gated it would not be, so this lane takes the bent quarter
              wave instead — which is also the geometry the sibling razor
              contact lane already uses.

Five grounds per geometry: PEC (`GN 1`) and the four finite ones through
the binary's native Sommerfeld solution (`GN 0 ... NOFILE` — NEC-5 has no
reflection-coefficient ground at all, which is a large part of why
momwire's was withdrawn at contact by this same stage).

WHAT IS NOT HERE
----------------
* The reflection-coefficient ground. Withdrawn at contact by this stage;
  there is nothing left to gate.
* `SinusoidalSolver`. Its contact answer has no mesh limit — it walks as
  ~N^0.4 (momwire#291), which is a property of collocating that basis at a
  contact node and not of the ground. Gating it against a converged binary
  ladder would gate the walk. `test_291_*` owns it.
* The eps~ -> infinity recovery through the Sommerfeld path, as a
  CONVERGENCE claim. It does not converge: the interpolation grid's error
  stops scaling with the ~1/sqrt(eps~) surfaces at the small R1 that only a
  contact deck queries, so the residual floors at 0.33-0.59 Ω and RISES
  with mesh (study §3.7). That is momwire#443, it is an instrument defect
  in shared machinery rather than a contact defect, and the lane pins the
  floor where it is instead of promising something the grid cannot deliver.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "tests" / "golden_contact_nec5.py"

FREQ_MHZ = 14.0
C = 299792458.0
WL = C / (FREQ_MHZ * 1e6)

RAD = 0.005
MONO_H = 5.3535
INVL_RISER = MONO_H / 2.0
INVL_TOP = MONO_H / 2.0

# Per-geometry ladders, in TOTAL segments. The monopole keeps the study's
# own rungs so this capture is readable against §3.5's published table; the
# inverted-L takes even rungs so its 1:1 length split meshes uniformly.
MONO_LADDER = (11, 21, 41, 61, 81)
INVL_LADDER = (12, 24, 48, 72, 96)

# The study's ground constants, in the study's own order of |eps~|. The two
# leading rows are the DECAY class and the two trailing rows the ENVELOPE
# class; `test_contact_nec5_lane.py` reads that split from here.
GROUND_EPS = {
    "sea": (81.0, 5.0),
    "vgood": (20.0, 0.0303),
    "avg": (13.0, 0.005),
    "poor": (5.0, 0.001),
}


# --------------------------------------------------------------------------
# geometry, in both spellings (momwire polylines and NEC deck cards)
# --------------------------------------------------------------------------
def monopole_wires():
    return [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, MONO_H]])]


def invl_wires():
    return [
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, INVL_RISER],
                [INVL_TOP, 0.0, INVL_RISER],
            ]
        )
    ]


def _num(x):
    return f"{float(x):.6E}"


def _gw(tag, n, p0, p1):
    return (
        f"GW {tag} {n} {_num(p0[0])} {_num(p0[1])} {_num(p0[2])} "
        f"{_num(p1[0])} {_num(p1[1])} {_num(p1[2])} {_num(RAD)}\n"
    )


def _gn(ground):
    """`GN 1` for PEC, else NEC-5's native Sommerfeld solution.

    `NOFILE` is the binary's own spelling for "solve the half-space here
    rather than reading a tabulated grid", and the two trailing zeros are
    the permeability fields — NOT NEC-2's radial-screen slots, which live at
    the same positions in that dialect and produce a silently MAGNETIC
    ground when written here (study §3.9). Anyone extending this script
    must leave them alone.
    """
    if ground == "pec":
        return "GN 1 0 0 0\n"
    eps_r, sigma = GROUND_EPS[ground]
    return f"GN 0 0 0 0 {_num(eps_r)} {_num(sigma)} {_num(1.0)} {_num(0.0)} NOFILE\n"


def monopole_deck(n, ground):
    p = monopole_wires()[0]
    return (
        "CM momwire#282 contact lane — monopole\nCE\n"
        + _gw(1, n, p[0], p[1])
        + "GE 1 0\n"
        + _gn(ground)
        + f"EX 0 1 1 1 {_num(1.0)} {_num(0.0)}\n"
        + f"FR 0 1 0 0 {_num(FREQ_MHZ)} {_num(0.0)}\n"
        + "XQ 0\nEN\n"
    )


def invl_deck(n, ground):
    p = invl_wires()[0]
    n_riser = n // 2
    return (
        "CM momwire#282 contact lane — inverted-L\nCE\n"
        + _gw(1, n_riser, p[0], p[1])
        + _gw(2, n - n_riser, p[1], p[2])
        + "GE 1 0\n"
        + _gn(ground)
        + f"EX 0 1 1 1 {_num(1.0)} {_num(0.0)}\n"
        + f"FR 0 1 0 0 {_num(FREQ_MHZ)} {_num(0.0)}\n"
        + "XQ 0\nEN\n"
    )


GEOMS = {
    "monopole": {
        "wires": monopole_wires,
        "deck": monopole_deck,
        "ladder": MONO_LADDER,
        "split": lambda n: [[n]],
    },
    "invl": {
        "wires": invl_wires,
        "deck": invl_deck,
        "ladder": INVL_LADDER,
        "split": lambda n: [[n // 2, n - n // 2]],
    },
}


def momwire_z(name, n, ground):
    """`BSplineSolver(degree=2, feed_model="segment")` on the same deck.

    Recomputed here only for the capture-time report — the golden module
    holds the BINARY's literals and nothing else, so the lane never pins a
    momwire float across machines (see the momwire#249 lesson: cross-build
    bit equality is not a property this tree has).
    """
    from momwire import BSplineSolver

    g = GEOMS[name]
    kw = dict(
        wires=g["wires"](),
        n_per_edge_per_wire=g["split"](n),
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
    z, _ = BSplineSolver(**kw).compute_impedance()
    return complex(z)


# --------------------------------------------------------------------------
def main() -> None:
    from antennaknobs.engines.nec5 import NEC5Engine

    sys.path.insert(0, str(Path.home() / "antennas/antennaknobs/scripts"))
    from bench_nec5_walk_why import make_dipole

    captures = Path(
        os.environ.get("CONTACT_LANE_CAPTURES", "/tmp/contact-lane-captures")
    )
    captures.mkdir(parents=True, exist_ok=True)
    eng = NEC5Engine(make_dipole(20), ground=None, capture_dir=captures)

    rows: dict[str, dict[str, list[tuple[int, complex]]]] = {}
    mw: dict[str, dict[str, dict[int, complex]]] = {}
    for name, g in GEOMS.items():
        rows[name] = {}
        mw[name] = {}
        for ground in ("pec", *GROUND_EPS):
            out = []
            mw[name][ground] = {}
            for n in g["ladder"]:
                zn = complex(eng.run_deck(g["deck"](n, ground))[0][0][2])
                out.append((n, zn))
                mw[name][ground][n] = momwire_z(name, n, ground)
                print(
                    f"{name:>9} {ground:>6} N={n:<3} nec5={zn:.4f} "
                    f"momwire={mw[name][ground][n]:.4f}"
                )
            rows[name][ground] = out

    _write_golden(rows)
    _report(rows, mw)


def _write_golden(rows) -> None:
    lines = [
        '"""NEC-5 printed impedances for GROUND-CONTACT decks over the',
        "binary's native Sommerfeld ground, and over PEC.",
        "",
        "GENERATED by scripts/capture_contact_nec5_lane.py — do not edit by",
        "hand. See that script for the decks, the dimensions, the run recipe",
        "and the two bar shapes these numbers are gated against",
        "(momwire#282 stage 1). Citation: NEC-5 (LLNL-CODE-746721),",
        "ground-contact ladder decks over four soils, 2026-08-18.",
        "",
        "`CONTACT_LADDERS[geometry][ground]` is a tuple of `(N_total, Z)`,",
        "the impedance the binary PRINTED for that deck at that mesh. The",
        "`pec` row of each geometry is the control the ground-induced shift",
        "`delta = Z(ground) - Z(pec)` is taken against, at matched N — the",
        "difference of columns that cancels the formulation's own offset and",
        "leaves the ground.",
        "",
        "`GROUND_EPS` is the (eps_r, sigma) the corresponding `GN 0` card",
        "carried, so a consumer builds the momwire side from the same",
        "constants rather than a transcription of them.",
        "",
        "No momwire float is recorded here on purpose: the momwire column is",
        "recomputed at test time. Pinning it would pin a cross-build float",
        "equality this tree does not have.",
        '"""',
        "",
        "GROUND_EPS = {",
    ]
    for g, (eps_r, sigma) in GROUND_EPS.items():
        lines.append(f'    "{g}": ({eps_r!r}, {sigma!r}),')
    lines.append("}")
    lines.append("")
    lines.append("CONTACT_LADDERS = {")

    def _lit(z):
        sign = "-" if z.imag < 0 else "+"
        return f"{z.real:.4f} {sign} {abs(z.imag):.4f}j"

    for name, per_ground in rows.items():
        lines.append(f'    "{name}": {{')
        for ground, out in per_ground.items():
            lines.append(f'        "{ground}": (')
            for n, zn in out:
                lines.append(f"            ({n}, {_lit(zn)}),")
            lines.append("        ),")
        lines.append("    },")
    lines.append("}")
    GOLDEN.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {GOLDEN}")


def _report(rows, mw) -> None:
    """The difference-of-columns table, per geometry — the shape §3.5
    published, so a re-capture is directly comparable to the study."""
    for name, per_ground in rows.items():
        print(f"\n===== {name}: delta = Z(ground) - Z(pec) =====")
        print(
            f"{'ground':>7} {'N':>4} {'delta momwire':>22} "
            f"{'delta nec5':>22} {'|diff|':>9}"
        )
        pec = dict(per_ground["pec"])
        for ground in GROUND_EPS:
            for n, zn in per_ground[ground]:
                d_n5 = zn - pec[n]
                d_mw = mw[name][ground][n] - mw[name]["pec"][n]
                print(
                    f"{ground:>7} {n:>4} {d_mw:>22.4f} {d_n5:>22.4f} "
                    f"{abs(d_mw - d_n5):9.4f}"
                )


if __name__ == "__main__":
    main()
