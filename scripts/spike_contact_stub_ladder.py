"""momwire#624 — the refl-coef discriminator, on a self-consistency instrument.

Why not the discriminator §5.5 / `scratch/624-spike/README.md` proposed
--------------------------------------------------------------------
That one said: sweep the restored term's coefficient under `refl-coef`, which
has no Sommerfeld remainder, and see whether the argmin moves to 1.0. It
cannot be run as written, because it needs the licensed binary as the
reference and **refl-coef at contact sits ~26 Ω from it** — 52.006+21.505j
against 26.643+10.767j on average soil at N=21. That is the model error D3
withdrew refl-coef at contact for, and it dwarfs the ~3 Ω the term is worth.
Fitting a 5 Ω knob to close a 26 Ω model gap measures the gap, not the knob.

So the reference has to go, and the instrument has to be self-consistency.

Half the question needs no experiment
-------------------------------------
Read from `razor.py`: T2's `M0c` is the reduced-kernel static moments times
`w_Φ` and nothing else; `rem_fn` — the Sommerfeld remainder — is built at
:2558 and used only at :2653/:2661, in the Q FIELD term, after T2 is already
assembled. So the folded scalar potential at the plane is

    refl-coef    (1 − w_Φ)·M0(plane)                  ← the term, complete
    sommerfeld   (1 − w_Φ)·M0(plane) + Φ_Q(plane)     ← the term, incomplete

by construction, not by measurement.

But that predicts a **soil-dependent** deficit, because the remainder's
weight in the composing ground varies strongly with ε̃ — and the measured
argmin did NOT vary: ≈0.4 on very good, average and poor alike
(`scratch/624-spike/`). A single coefficient across grounds with very
different remainders looks like a model-INDEPENDENT scale factor, which is
the row-halving's signature, not the remainder's.

What this script measures
-------------------------
The **stubbed ladder** (§3.8 ladder B), which is momwire against momwire and
needs no binary: the same antenna with its contacting element replaced by a
vanishing grounded stub. Shrinking the stub does not change the antenna, so a
self-consistent contact node must give an h-independent answer.

Two corrections from the stage-2 record are built in:

* **the feed goes on the RADIATOR, not the stub's grounded base.** Fed at the
  base, the feed segment shrinks with the stub and the ladder measures the
  delta-gap source model instead — momwire's own PEC ladder is 53 Ω out at a
  0.1 mm stub that way, and 3.2e-4 Ω with the feed moved;
* the ladder is a **self-consistency** instrument, so what it can exclude is
  an internally inconsistent contact node — not a formulation difference the
  contact deck shares with its own stubbed limit.

PEC runs first as the control: the term is identically zero there, so a flat
PEC ladder is the harness certifying itself before any finite-ground row is
read.

Reading it
----------
If the coefficient that flattens the refl-coef ladder is ≈1.0, the term is
complete where there is no remainder, and Sommerfeld's deficit is Φ_Q.
If it is ≈0.4-0.5 under refl-coef too, the deficit is model-independent and
the row halving is the suspect.
"""

from __future__ import annotations

import numpy as np

FREQ_MHZ = 14.0
WL = 299792458.0 / (FREQ_MHZ * 1e6)
RAD = 0.005
MONO_H = 5.3535
N_RAD = 20

# The stub heights, in metres. The study's ladder B goes to 0.1 mm.
STUBS = (0.1, 0.03, 0.01, 0.003, 0.001)
SOILS = {"average": (13.0, 0.005), "poor": (5.0, 0.001)}
COEFFS = (0.0, 0.25, 0.4, 0.5, 0.75, 1.0)


def stub_z(h, soil, *, model, coeff):
    """The stubbed monopole: a 1-segment grounded stub plus the radiator.

    The mesh above the stub is HELD FIXED, and that is not a detail. Spelling
    the radiator as one edge from z = h to the tip on a fixed segment COUNT
    re-meshes the whole antenna every rung — a 1.9 % segment-length change
    across this ladder — and the PEC control then drifts 2.5 Ω, which is a
    mesh artefact with no contact node in it. So the radiator is spelled as a
    polyline whose knots sit at the ORIGINAL uniform mesh's z values, and only
    its first edge (from the stub's top to the first fixed knot) changes
    length. Everything above the contact is bit-identical from rung to rung.

    The feed sits on a FIXED PHYSICAL KNOT of that mesh, not at a fixed
    fraction of the radiator — the stage-2 correction, without which the feed
    segment shrinks with the stub and the ladder measures the delta-gap source
    model instead (53 Ω out at a 0.1 mm stub, against 3.2e-4 Ω with it).

    A one-segment wire junctioned at one end and standing in the plane at the
    other is exactly what momwire#608 stopped refusing, so razor can host this
    instrument at all only since that landed.
    """
    from momwire import RazorSolver

    d = MONO_H / N_RAD  # the fixed mesh this ladder never disturbs
    knots = [h] + [(i + 1) * d for i in range(N_RAD)]
    radiator = np.array([[0.0, 0.0, z] for z in knots])
    feed_knot = (N_RAD // 2) * d  # a fixed physical height
    kw = dict(
        wires=[
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, h]]),
            radiator,
        ],
        n_per_edge_per_wire=[[1], [1] * N_RAD],
        wire_radius=RAD,
        wavelength=WL,
        ground_z=0.0,
        nec5_quadrature=True,
        feeds=[(1, feed_knot - h, 1.0 + 0j)],
    )
    if soil is not None:
        kw.update(ground_eps=soil, ground_model=model)

    RazorSolver._spike_contact = True
    RazorSolver._spike_plane_reference = float(coeff)
    try:
        z, _ = RazorSolver(**kw).compute_impedance()
    finally:
        RazorSolver._spike_contact = False
        RazorSolver._spike_plane_reference = 0.0
    return complex(z)


def ladder(soil, *, model, coeff):
    return [stub_z(h, soil, model=model, coeff=coeff) for h in STUBS]


def spread(zs):
    """How far the ladder drifts: worst rung against the smallest stub."""
    return max(abs(z - zs[-1]) for z in zs[:-1])


def main():
    print(f"stubbed monopole {MONO_H} m, r={RAD} m, {FREQ_MHZ} MHz")
    print(f"stub heights (m): {', '.join(str(h) for h in STUBS)}")
    print("feed on the RADIATOR's midpoint (stage-2 correction)\n")

    print("CONTROL — PEC, where the restored term is identically zero.")
    print("A flat ladder here is the harness certifying itself.\n")
    zs = ladder(None, model=None, coeff=0.0)
    for h, z in zip(STUBS, zs):
        print(f"   h={h:<7g} {z.real:10.5f}{z.imag:+10.5f}j")
    print(f"   spread vs smallest stub: {spread(zs):.2e} ohm\n")

    for name, soil in SOILS.items():
        for model in ("refl-coef", "sommerfeld"):
            print(f"\n{name} soil, {model} — ladder spread vs the term's coefficient\n")
            print(
                f"   {'coeff':>6s} | {'spread (ohm)':>13s} | {'Z at smallest stub':>26s}"
            )
            best, best_c = None, None
            for c in COEFFS:
                zs = ladder(soil, model=model, coeff=c)
                s = spread(zs)
                if best is None or s < best:
                    best, best_c = s, c
                print(
                    f"   {c:>6.2f} | {s:>13.4f} | "
                    f"{zs[-1].real:12.5f}{zs[-1].imag:+12.5f}j"
                )
            print(f"   -> flattest at coefficient {best_c}")


if __name__ == "__main__":
    main()
