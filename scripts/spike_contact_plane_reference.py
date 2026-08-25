"""momwire#624 spike — razor's grounded row over a FINITE ground.

`docs/design/contact-over-finite-ground.md` §5.5, which says in full:

> implement the restored plane-reference term `(1 − w_Φ)·M0(plane)` behind a
> flag, keep the image wing at coefficient 1, and measure razor against
> bspline on the §3.5 decks under the difference-of-columns bar, plus PEC
> bit-identity. Also measure the row-halving assumption separately. **Until
> that experiment runs, Stage 3 has no schedule.**

Run under the antennaknobs venv, with the binary on `NEC5_EXE`:

    NEC5_EXE=~/antennas/NEC5-downloads/nec5-linux/nec5cl \
        /home/smburns/antennas/antennaknobs/.venv/bin/python \
        scripts/spike_contact_plane_reference.py

Only the binary's PRINTED impedance is read; nothing about its internals is
quoted or inferred. Citation: NEC-5 (LLNL-CODE-746721).

What is measured, and why each one is here
------------------------------------------
**1. PEC bit-identity.** The restored term is identically zero at PEC, so
turning the spike on must not move a PEC contact answer by one ULP. This is
implemented as untouched arithmetic rather than an added zero — at PEC there
is no `w_Φ` table at all and the term's whole branch is skipped — so the
gate is `==`, not `approx`.

**2. Does the residual SATURATE or WALK?** This, not accuracy, is the parity
gate momwire#624 turns on. bspline's contact residual opens with mesh and
then flattens (a difference of limits); the direct-field trunk's DIVERGED
(momwire#282). §4.3 predicts razor is on bspline's side — its contact charge
is a bounded doublet, not a 1/Δ point charge — and this ladder is that
prediction's test. A solver whose answer walks without bound cannot serve at
any bar.

**3. What the term is worth.** razor with the term OFF and ON, both against
the binary's printed shift, on the same difference-of-columns bar §3.5 used.
OFF is what razor would do today if the refusal were simply lifted; ON is
§4.3's fix. The gap between them is the term's whole contribution.

**4. bspline alongside**, because the question is not only "is razor good"
but "is razor DIFFERENT" — if razor lands on the binary where bspline sits
3.3 Ω away, that localizes a gap currently unexplained for both trunks.

Not measured here
-----------------
The **row-halving** assumption. §5.5 asks for it separately: razor's grounded
row is the real half of the testing path only, halved by the self-image
invariance `E(M·r) = −M·E(r)`, which is a PEC identity that a weighted image
does not satisfy. Testing it means filling the row both ways — halved, and
with the image half integrated explicitly — which is a fill-level change
rather than a flag, and it is scoped as its own step. If the ladder below
saturates and the term closes the gap, the halving is the next suspect for
whatever is left; if the ladder walks, the halving is where to look first.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

FREQ_MHZ = 14.0
WL = 299792458.0 / (FREQ_MHZ * 1e6)
RAD = 0.005
MONO_H = 5.3535
EXE = Path(os.path.expanduser(os.environ.get("NEC5_EXE", "")))

# §3.5's four soils, as (eps_r, sigma).
SOILS = {
    "sea": (81.0, 5.0),
    "v.good": (20.0, 0.0303),
    "average": (13.0, 0.005),
    "poor": (5.0, 0.001),
}
LADDER = (11, 21, 41, 61)

WIRE = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, MONO_H]])


def _num(x):
    return f"{float(x):.6E}"


def deck(n, soil):
    gn = (
        "GN 1 0 0 0\n"
        if soil is None
        else f"GN 0 0 0 0 {_num(soil[0])} {_num(soil[1])} {_num(1.0)} {_num(0.0)} NOFILE\n"
    )
    return (
        "CM momwire#624 spike — contact monopole\nCE\n"
        f"GW 1 {n} 0.0 0.0 0.0 0.0 0.0 {_num(MONO_H)} {_num(RAD)}\n"
        "GE 1 0\n" + gn + f"EX 0 1 1 1 {_num(1.0)} {_num(0.0)}\n"
        f"FR 0 1 0 0 {_num(FREQ_MHZ)} {_num(0.0)}\n"
        "XQ 0\nEN\n"
    )


def binary_z(n, soil):
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "m.nec").write_text(deck(n, soil))
        subprocess.run(
            [str(EXE)],
            input="m.nec\nm.out\n\n",
            capture_output=True,
            text=True,
            cwd=td,
            timeout=900,
        )
        out = (Path(td) / "m.out").read_text(errors="replace")
    row = [
        ln for ln in out.splitlines() if re.match(r"\s*1\s+\d+ \d+\s+1\.0000E\+00", ln)
    ]
    nums = re.findall(r"[-+]?\d\.\d+E[-+]\d+", row[0])
    return complex(float(nums[4]), float(nums[5]))


def _kw(n, soil):
    kw = dict(
        wires=[WIRE],
        n_per_edge_per_wire=[[n]],
        wire_radius=RAD,
        wavelength=WL,
        ground_z=0.0,
    )
    if soil is not None:
        kw.update(ground_eps=soil, ground_model="sommerfeld")
    return kw


def razor_z(n, soil, *, plane_reference):
    """RazorSolver's contact impedance, with the restored term off or on."""
    from momwire import RazorSolver

    RazorSolver._spike_contact = True
    # NOT bool() — the flag doubles as the term's coefficient so the sign and
    # scale can be swept, which is how an implementation sign error is told
    # apart from a wrong hypothesis. 0.0 is off.
    RazorSolver._spike_plane_reference = float(plane_reference)
    try:
        z, _ = RazorSolver(
            **_kw(n, soil), feeds=[(0, 0.0, 1.0 + 0j)], nec5_quadrature=True
        ).compute_impedance()
    finally:
        RazorSolver._spike_contact = False
        RazorSolver._spike_plane_reference = 0.0
    return complex(z)


def bspline_z(n, soil):
    from momwire import BSplineSolver

    z, _ = BSplineSolver(
        **_kw(n, soil), degree=2, feed_model="segment", feed_arclength=0.0
    ).compute_impedance()
    return complex(z)


def gate_pec_bit_identity():
    """The restored term is identically zero at PEC — so nothing may move."""
    print("1. PEC contact bit-identity (the term is zero at PEC)\n")
    ok = True
    for n in LADDER:
        off = razor_z(n, None, plane_reference=0.0)
        on = razor_z(n, None, plane_reference=1.0)
        same = off == on
        ok &= same
        print(
            f"   N={n:>3d}  off {off.real:12.6f}{off.imag:+12.6f}j   "
            f"on {on.real:12.6f}{on.imag:+12.6f}j   {'BIT-IDENTICAL' if same else 'MOVED'}"
        )
    print(f"\n   verdict: {'PASS' if ok else 'FAIL'}\n")
    return ok


def ladder():
    """The go/no-go table: does the residual saturate, and what is the term worth?"""
    print("2-4. difference-of-columns  delta = Z(soil) - Z(PEC)")
    print("     |diff| = |delta_solver - delta_binary|, the §3.5 bar\n")
    print(
        f"   {'soil':<8s}{'N':>4s} | {'razor OFF':>10s} {'razor ON':>10s} "
        f"{'bspline':>10s} | {'term moved':>11s}"
    )
    rows = {}
    for name, soil in SOILS.items():
        print("   " + "-" * 62)
        for n in LADDER:
            d_b = binary_z(n, soil) - binary_z(n, None)
            pec_r = razor_z(n, None, plane_reference=0.0)
            off = razor_z(n, soil, plane_reference=0.0) - pec_r
            on = razor_z(n, soil, plane_reference=1.0) - pec_r
            bs = bspline_z(n, soil) - bspline_z(n, None)
            rows.setdefault(name, []).append(
                (n, abs(off - d_b), abs(on - d_b), abs(bs - d_b))
            )
            print(
                f"   {name:<8s}{n:>4d} | {abs(off - d_b):>10.3f} {abs(on - d_b):>10.3f} "
                f"{abs(bs - d_b):>10.3f} | {abs(on - off):>11.3f}"
            )
    return rows


def verdict(rows):
    """The parity gate: BOUNDED or DIVERGING — not "saturating or not".

    The distinction that decides whether razor can serve at all is whether
    the residual stays bounded under refinement (momwire#282's direct-field
    trunk did not; bspline does). "Saturating" is a finer question about the
    limit's shape and a single ratio makes a poor test of it — a row that is
    flat at 0.005 has increments of pure noise. So the increments are printed
    and the bounded/diverging call is made on their TREND.
    """
    print("\n\n5. BOUNDED or DIVERGING under refinement? (the parity gate)\n")
    # Below this the whole ladder is quadrature noise and no trend in it
    # means anything; labelling such a row either way would be a fiction.
    FLAT = 0.05
    print(f"   {'row':<13s} | {'ladder (N = 11, 21, 41, 61)':<36s} | increments")
    for name, rs in rows.items():
        for label, col in (("OFF", 1), ("ON", 2)):
            vals = [r[col] for r in rs]
            incs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
            if max(vals) < FLAT:
                tag = f"flat (< {FLAT} ohm, noise)"
            elif abs(incs[-1]) > abs(incs[0]) and vals[-1] > vals[0]:
                tag = "DIVERGING"
            else:
                tag = "bounded"
            print(
                f"   {name + ' ' + label:<13s} | "
                f"{' → '.join(f'{v:.3f}' for v in vals):<36s} "
                f"| {' '.join(f'{v:+.3f}' for v in incs)}  {tag}"
            )
    print(
        "\n   Bounded = servable under an envelope pin, which is what D1 already\n"
        "   decided for bspline's contact row. Diverging is the direct-field\n"
        "   trunk's shape (momwire#282) and is not servable at any bar."
    )


COEFFS = (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0, -1.0)


def coefficient_scan():
    """Sweep the term's coefficient — the sign check, and the magnitude check.

    A fitted coefficient is NOT a derivation, and nothing here should be read
    as one. What the sweep can settle is narrower and worth having:

    * **the sign**, by whether −1 is worse than +1 (an implementation sign
      error and a wrong hypothesis look identical at one coefficient);
    * **whether the term's SHAPE is right**, by whether one coefficient
      reduces the residual on every ground and every mesh at once. A term of
      the wrong shape helps one soil and hurts another; a term of the right
      shape at the wrong scale does what this one does.
    """
    print("\n\n6. the term's coefficient, swept\n")
    print("   |delta_razor - delta_binary| vs the coefficient on the restored term\n")
    print(f"   {'soil':<8s}{'N':>4s} | " + " ".join(f"{c:>7.2f}" for c in COEFFS))
    for name, soil in SOILS.items():
        print("   " + "-" * 70)
        for n in (21, 41):
            d_b = binary_z(n, soil) - binary_z(n, None)
            pec = razor_z(n, None, plane_reference=0.0)
            vals = [
                abs((razor_z(n, soil, plane_reference=c) - pec) - d_b) for c in COEFFS
            ]
            print(f"   {name:<8s}{n:>4d} | " + " ".join(f"{v:>7.3f}" for v in vals))


def main():
    if not EXE.is_file():
        raise SystemExit(f"NEC5_EXE not found: {EXE}")
    print(f"contact monopole {MONO_H} m, r={RAD} m, {FREQ_MHZ} MHz, base-fed\n")
    pec_ok = gate_pec_bit_identity()
    rows = ladder()
    verdict(rows)
    coefficient_scan()
    if not pec_ok:
        print("\n*** PEC BIT-IDENTITY FAILED — the term is not zero at PEC ***")


if __name__ == "__main__":
    main()
