"""Phase 1 + Phase 6 evaluation: momwire ground_eps vs golden PyNEC gn 0.

Runs the full validation matrix (tests/fixtures_refl_coef_geoms.py against
tests/golden_refl_coef_ground.py):

- BSplineSolver (Phase 1): every Φ-term weighting candidate, |ΔZ| vs the NEC
  reflection-coefficient oracle, alongside the PEC-image solve everyone is
  trying to beat.
- SinusoidalSolver (Phase 6): PEC image vs `ground_eps` Fresnel-weighted
  image (no `ground_phi_mode` — the field-based formulation has no Φ split),
  same residual table + summary stats.
- Cross-solver PEC floor (Phase 6 acceptance input): |Z_solver_PEC − PyNEC
  PEC| per case for both solvers — the solver-vs-NEC discretization floor the
  finite-ground gates are set relative to.
- BSplineSolver `ground_model="sommerfeld"` (sommerfeld plan Phase 4):
  |ΔZ| vs the nec2c gn 2 oracle across ALL heights (0.02–0.5λ) and the
  yagi, with the refl-coef and PEC solves' gn 2 residuals alongside for
  scale — this is the table the sommerfeld acceptance gates are set from.

Acceptance (docs/refl-coef-ground-plan.md Phase 1): |ΔZ| ≤ ~2 Ω across the
dipole 0.1–0.5λ heights, and strictly better than the PEC-image solve
everywhere in that window. Below 0.1λ is reported, not gated.
Phase 6 gates at the sinusoidal cross-solver floor + ~1 Ω in that window.

Run from the momwire repo:  .venv/bin/python scripts/compare_refl_coef_ground.py
"""

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(__file__)) or ".", "tests")
)

from fixtures_refl_coef_geoms import GEOMS  # noqa: E402
from golden_refl_coef_ground import GOLDEN  # noqa: E402

from momwire import BSplineSolver, SinusoidalSolver  # noqa: E402
from momwire._ground_refl import PHI_MODES  # noqa: E402

WINDOW = (0.1, 0.2, 0.35, 0.5)  # acceptance window height fractions


def grounds_for(name, frac):
    return sorted({(e, s) for (n, f, e, s) in GOLDEN if n == name and f == frac})


def print_residual_table(label, z_solver, cols):
    """|ΔZ| vs gn0 per case + the summary stats / strictly-better blocks.

    z_solver: (name, frac, eps_r, sigma, col) -> complex, col "pec" included.
    """
    print(f"\n=== {label}: |dZ| vs gn0 per case (ohms) ===")
    print(f"{'case':38s}" + "".join(f"{c:>10s}" for c in cols) + f"{'gn0':>20s}")
    stats = {c: [] for c in cols}  # (in_window, name, |dZ|)
    for key in sorted(GOLDEN):
        name, frac, eps_r, sigma = key
        gn0 = GOLDEN[key]["finite-fast"]
        row = f"{name:11s} h={frac:4.2f} eps={eps_r:4.1f} s={sigma:5.3f} "
        for c in cols:
            dz = abs(z_solver[(name, frac, eps_r, sigma, c)] - gn0)
            row += f"{dz:10.2f}"
            stats[c].append((frac in WINDOW, name, dz))
        row += f"    {gn0.real:7.2f}{gn0.imag:+7.2f}j"
        print(row)

    print(f"\n--- {label} summary: |dZ| vs gn0 (ohms) ---")
    for scope, pred in (
        ("dipole 0.1-0.5wl (acceptance)", lambda w, n: w and n == "dipole"),
        ("inverted_l 0.1-0.5wl", lambda w, n: w and n == "inverted_l"),
        ("all h=0.05 (report only)", lambda w, n: not w),
    ):
        print(scope)
        for c in cols:
            vals = [d for w, n, d in stats[c] if pred(w, n)]
            print(f"  {c:7s} max={max(vals):6.2f}  mean={sum(vals) / len(vals):6.2f}")

    # strictly-better-than-PEC check in the window
    print(f"\n--- {label}: cases in window where candidate is NOT better than PEC ---")
    any_bad = False
    for key in sorted(GOLDEN):
        name, frac, eps_r, sigma = key
        if frac not in WINDOW:
            continue
        gn0 = GOLDEN[key]["finite-fast"]
        d_pec = abs(z_solver[(name, frac, eps_r, sigma, "pec")] - gn0)
        for c in cols:
            if c == "pec":
                continue
            d = abs(z_solver[(name, frac, eps_r, sigma, c)] - gn0)
            if d >= d_pec:
                print(
                    f"  {c}: {name} h={frac} eps={eps_r} s={sigma}: "
                    f"{d:.2f} >= pec {d_pec:.2f}"
                )
                any_bad = True
    if not any_bad:
        print("  (none — every candidate beats PEC on every window case)")


def main():
    # ------- BSplineSolver (Phase 1): PEC once per geometry, each Φ
    # candidate per (geom, ground) -------
    z_bsp = {}  # (name, frac, eps, sigma, mode) -> complex; mode "pec" too
    pec_by_case = {"bspline": {}, "sinusoidal": {}}  # (name, frac) -> Z_pec
    for (name, frac), kw in GEOMS.items():
        z_pec, _ = BSplineSolver(**kw, ground_z=0.0).compute_impedance()
        pec_by_case["bspline"][(name, frac)] = z_pec
        for eps_r, sigma in grounds_for(name, frac):
            z_bsp[(name, frac, eps_r, sigma, "pec")] = z_pec
            for mode in PHI_MODES:
                sim = BSplineSolver(
                    **kw,
                    ground_z=0.0,
                    ground_eps=(eps_r, sigma),
                    ground_phi_mode=mode,
                )
                z, _ = sim.compute_impedance()
                z_bsp[(name, frac, eps_r, sigma, mode)] = z

    print_residual_table("bspline", z_bsp, ("pec",) + PHI_MODES)

    # ------- SinusoidalSolver (Phase 6): field-based dyad, no Φ modes -------
    z_sin = {}  # (name, frac, eps, sigma, "pec"|"finite") -> complex
    for (name, frac), kw in GEOMS.items():
        z_pec, _ = SinusoidalSolver(**kw, ground_z=0.0).compute_impedance()
        pec_by_case["sinusoidal"][(name, frac)] = z_pec
        for eps_r, sigma in grounds_for(name, frac):
            z_sin[(name, frac, eps_r, sigma, "pec")] = z_pec
            sim = SinusoidalSolver(**kw, ground_z=0.0, ground_eps=(eps_r, sigma))
            z, _ = sim.compute_impedance()
            z_sin[(name, frac, eps_r, sigma, "finite")] = z

    print_residual_table("sinusoidal", z_sin, ("pec", "finite"))

    # ------- Cross-solver PEC floor: |Z_solver_PEC - PyNEC PEC| -------
    # PyNEC's PEC Z is ground-independent, so one number per (geom, height).
    # This is the discretization floor the Phase 6 acceptance gates against
    # (plan doc quotes ~1.4 ohm for bspline on the dipole window).
    golden_pec = {}  # (name, frac) -> PyNEC pec Z
    for (name, frac, _e, _s), models in GOLDEN.items():
        golden_pec[(name, frac)] = models["pec"]

    solvers = ("bspline", "sinusoidal")
    print("\n=== cross-solver PEC floor: |Z_solver_PEC - PyNEC PEC| (ohms) ===")
    print(
        f"{'case':22s}" + "".join(f"{s:>12s}" for s in solvers) + f"{'PyNEC pec':>22s}"
    )
    floor = {s: [] for s in solvers}  # (in_window, name, |dZ|)
    for name, frac in sorted(golden_pec):
        z_nec = golden_pec[(name, frac)]
        row = f"{name:11s} h={frac:4.2f}      "
        for s in solvers:
            d = abs(pec_by_case[s][(name, frac)] - z_nec)
            row += f"{d:12.2f}"
            floor[s].append((frac in WINDOW, name, d))
        row += f"      {z_nec.real:7.2f}{z_nec.imag:+7.2f}j"
        print(row)

    print("\n--- cross-solver PEC floor summary (ohms) ---")
    for scope, pred in (
        ("dipole 0.1-0.5wl (acceptance window)", lambda w, n: w and n == "dipole"),
        ("inverted_l 0.1-0.5wl", lambda w, n: w and n == "inverted_l"),
        ("all h=0.05 (report only)", lambda w, n: not w),
    ):
        print(scope)
        for s in solvers:
            vals = [d for w, n, d in floor[s] if pred(w, n)]
            print(f"  {s:10s} max={max(vals):6.2f}  mean={sum(vals) / len(vals):6.2f}")

    # ------- BSplineSolver ground_model="sommerfeld" vs gn 2 (nec2c) -------
    # gn 2 residuals for the refl-coef and PEC solves are context: below
    # 0.1wl they are the error the sommerfeld model exists to remove.
    print("\n=== bspline sommerfeld: |dZ| vs gn2 per case (ohms) ===")
    print(f"{'case':38s}{'somm':>10s}{'refl':>10s}{'pec':>10s}{'gn2':>20s}")
    somm_stats = {"somm": [], "refl": [], "pec": []}
    for key in sorted(GOLDEN):
        name, frac, eps_r, sigma = key
        gn2 = GOLDEN[key]["finite"]
        kw = GEOMS[(name, frac)]
        z_s, _ = BSplineSolver(
            **kw,
            ground_z=0.0,
            ground_eps=(eps_r, sigma),
            ground_model="sommerfeld",
        ).compute_impedance()
        d = {
            "somm": abs(z_s - gn2),
            "refl": abs(z_bsp[(name, frac, eps_r, sigma, "normal")] - gn2),
            "pec": abs(z_bsp[(name, frac, eps_r, sigma, "pec")] - gn2),
        }
        for c, v in d.items():
            somm_stats[c].append((name, v))
        print(
            f"{name:11s} h={frac:4.2f} eps={eps_r:4.1f} s={sigma:5.3f} "
            f"{d['somm']:10.2f}{d['refl']:10.2f}{d['pec']:10.2f}"
            f"    {gn2.real:7.2f}{gn2.imag:+7.2f}j"
        )
    print("\n--- sommerfeld summary: |dZ| vs gn2 (ohms), ALL heights ---")
    for name in ("dipole", "inverted_l", "yagi"):
        row = f"  {name:11s}"
        for c in ("somm", "refl", "pec"):
            vals = [v for n, v in somm_stats[c] if n == name]
            row += f"  {c} max={max(vals):6.2f} mean={sum(vals) / len(vals):6.2f}"
        print(row)


if __name__ == "__main__":
    main()
