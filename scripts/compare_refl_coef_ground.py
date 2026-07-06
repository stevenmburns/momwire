"""Phase 1 evaluation: momwire ground_eps candidates vs golden PyNEC gn 0.

Runs the full validation matrix (tests/fixtures_refl_coef_geoms.py against
tests/golden_refl_coef_ground.py) for every Φ-term weighting candidate and
reports |ΔZ| vs the NEC reflection-coefficient oracle, alongside the PEC-image
solve everyone is trying to beat.

Acceptance (docs/refl-coef-ground-plan.md Phase 1): |ΔZ| ≤ ~2 Ω across the
dipole 0.1–0.5λ heights, and strictly better than the PEC-image solve
everywhere in that window. Below 0.1λ is reported, not gated.

Run from the momwire repo:  .venv/bin/python scripts/compare_refl_coef_ground.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)) or ".", "tests"))

from fixtures_refl_coef_geoms import GEOMS  # noqa: E402
from golden_refl_coef_ground import GOLDEN  # noqa: E402

from momwire import BSplineSolver  # noqa: E402
from momwire._ground_refl import PHI_MODES  # noqa: E402

WINDOW = (0.1, 0.2, 0.35, 0.5)  # acceptance window height fractions


def main():
    # momwire solves: PEC once per geometry, each candidate per (geom, ground)
    z_mom = {}  # (name, frac, eps, sigma, mode) -> complex; mode "pec" too
    for (name, frac), kw in GEOMS.items():
        z_pec, _ = BSplineSolver(**kw, ground_z=0.0).compute_impedance()
        grounds = sorted({(e, s) for (n, f, e, s) in GOLDEN if n == name and f == frac})
        for eps_r, sigma in grounds:
            z_mom[(name, frac, eps_r, sigma, "pec")] = z_pec
            for mode in PHI_MODES:
                sim = BSplineSolver(
                    **kw,
                    ground_z=0.0,
                    ground_eps=(eps_r, sigma),
                    ground_phi_mode=mode,
                )
                z, _ = sim.compute_impedance()
                z_mom[(name, frac, eps_r, sigma, mode)] = z

    cols = ("pec",) + PHI_MODES
    print(f"{'case':38s}" + "".join(f"{c:>10s}" for c in cols) + f"{'gn0':>20s}")
    stats = {c: [] for c in cols}  # (in_window, name, |dZ|)
    for key in sorted(GOLDEN):
        name, frac, eps_r, sigma = key
        gn0 = GOLDEN[key]["finite-fast"]
        row = f"{name:11s} h={frac:4.2f} eps={eps_r:4.1f} s={sigma:5.3f} "
        for c in cols:
            dz = abs(z_mom[(name, frac, eps_r, sigma, c)] - gn0)
            row += f"{dz:10.2f}"
            stats[c].append((frac in WINDOW, name, dz))
        row += f"    {gn0.real:7.2f}{gn0.imag:+7.2f}j"
        print(row)

    print("\n--- summary: |dZ| vs gn0 (ohms) ---")
    for scope, pred in (
        ("dipole 0.1-0.5wl (acceptance)", lambda w, n: w and n == "dipole"),
        ("inverted_l 0.1-0.5wl", lambda w, n: w and n == "inverted_l"),
        ("all h=0.05 (report only)", lambda w, n: not w),
    ):
        print(scope)
        for c in cols:
            vals = [d for w, n, d in stats[c] if pred(w, n)]
            print(f"  {c:7s} max={max(vals):6.2f}  mean={sum(vals)/len(vals):6.2f}")

    # strictly-better-than-PEC check in the window
    print("\n--- cases in window where candidate is NOT better than PEC ---")
    any_bad = False
    for key in sorted(GOLDEN):
        name, frac, eps_r, sigma = key
        if frac not in WINDOW:
            continue
        gn0 = GOLDEN[key]["finite-fast"]
        d_pec = abs(z_mom[(name, frac, eps_r, sigma, "pec")] - gn0)
        for mode in PHI_MODES:
            d = abs(z_mom[(name, frac, eps_r, sigma, mode)] - gn0)
            if d >= d_pec:
                print(f"  {mode}: {name} h={frac} eps={eps_r} s={sigma}: "
                      f"{d:.2f} >= pec {d_pec:.2f}")
                any_bad = True
    if not any_bad:
        print("  (none — every candidate beats PEC on every window case)")


if __name__ == "__main__":
    main()
