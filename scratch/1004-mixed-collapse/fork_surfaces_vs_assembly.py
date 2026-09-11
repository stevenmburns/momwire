"""Surfaces or assembly? — momwire#1004, the fork experiment.

Phase 1 established: the CROSS block carries all of the ε̃ = 1 collapse error
(9.6e-05 at a 0.5 m gap to 5.7e-02 at 5 mm, vertical deck; 6.4e-01 on the
ρ-spanning one), and it is NOT the grid's sampling — refining theta 15×, R 4×
and the z′ ladder moves the answer in the sixth or seventh digit.

That leaves two possibilities, and this separates them WITHOUT a solver:

    SURFACES   what the grid tabulates is wrong near the plane
    ASSEMBLY   the surfaces are right and the cross block consumes them wrongly

The repo already owns the oracle. `test_gu3_5_eps_one_is_the_free_space_dipole_exactly`
asserts that at ε̃ = 1 the composed transmitted field equals the free-space
dyadic dipole to **1e-9** — at k_m = k_p the Sommerfeld identity turns V_T into
G/k_p² and U_T into G, so (7a)-(7e) reproduce the closed form term for term.
There is nothing to derive here; `fs_field` is that closed form, transcribed in
the test module, and `DirectSurfaces` drives the composition off
`t_surfaces_direct` instead of a tabulation.

**What that gate does not cover is the near plane.** Its three geometries put
the source at z′ = −0.15 / −1.0 / −0.05 and the observer at z = +1.0 / +3.0 /
+2.0 — the closest approach is |z| + |z′| = 1.05 m. #1004's deck sits at
|z| = |z′| = 0.005 m. So the collapse is gated a metre from the plane and
untested five millimetres from it, which is the whole regime in question.

REGISTERED PREDICTION, before running (two of mine were burned earlier; the
answer is to keep registering them):

  * **I predict the SURFACES are clean** — `t_surfaces_direct` at ε̃ = 1
    reproduces the free-space dipole to **<= 1e-8** at every gap down to 5 mm,
    so the fork points at the ASSEMBLY or at how the block queries the grid.
  * **Falsifier:** if the direct-surface error reaches **>= 1e-3** at 5 mm, the
    surfaces carry it and the fix is in the tabulated quantity.
  * Reasoning, stated so it can be wrong: a formulation error in the surfaces
    would be a smooth function of geometry and ought to have shown SOME
    sensitivity to how finely that geometry is sampled; phase 1 found none on
    any axis. But the R axis got slightly WORSE under refinement, which is the
    signature of a domain/clamping problem rather than a resolution one — and
    that would live on the query side, not in the surfaces.

Scratch only; nothing in `src/momwire` is touched.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")

HERE = Path(__file__).resolve()
TESTS = HERE.parents[2] / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import momwire._sommerfeld_transmitted as trans  # noqa: E402
from test_sommerfeld_transmitted import (  # noqa: E402
    C0,
    GROUND_Z,
    MU0,
    DirectSurfaces,
    fs_field,
    rel_vec,
)

F = 7e6
KP = 2.0 * np.pi * F / C0
OM = 2.0 * np.pi * F
C1 = -1j * OM * MU0 / (4.0 * np.pi)
GAPS = (0.5, 0.1, 0.02, 0.005)

# The ρ offsets the #1004 deck actually spans: two 0.6 m wires, so pairs run
# from coincident-in-ρ out to the full length. ρ = 0 is included because the
# vertical deck is entirely there — and it is where theta is degenerate.
RHOS = (0.0, 0.05, 0.3, 0.6)

P_HATS = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))


def field_from(surfaces, obs, src, p_hat):
    return trans.transmitted_field_below_to_above(
        np.array([obs], dtype=float),
        np.array([src], dtype=float),
        p_hat[None, :].astype(complex),
        GROUND_Z,
        KP,
        KP + 0j,
        surfaces,
    )[0]


def worst_over_geometry(surfaces, gap):
    """Worst relative miss against the closed form, over ρ and polarisation."""
    worst = 0.0
    where = None
    for rho in RHOS:
        src = [0.0, 0.0, -gap]
        obs = [rho, 0.0, +gap]
        if rho == 0.0 and gap == 0.0:
            continue
        for p_hat in P_HATS:
            got = field_from(surfaces, obs, src, p_hat)
            ref = fs_field(C1, KP, p_hat, np.asarray(obs) - np.asarray(src))
            r = rel_vec(got, ref)
            if r > worst:
                worst, where = r, (rho, "x" if p_hat[0] else "z")
    return worst, where


def main() -> int:
    print("=" * 78)
    print("GATE — the oracle must reproduce its own published number first")
    print("=" * 78)
    # The existing gate's own geometry, re-run here: if this does not land at
    # 1e-9 the harness is wrong and nothing below is evidence.
    surfaces = DirectSurfaces(1.0 + 0j, KP, OM, rtol=1e-11)
    check = 0.0
    for src_z, obs in ((-0.15, [6.0, 0.0, 1.0]), (-0.05, [0.0, 0.0, 2.0])):
        for p_hat in P_HATS:
            got = field_from(surfaces, obs, [0.0, 0.0, src_z], p_hat)
            ref = fs_field(C1, KP, p_hat, np.asarray(obs) - np.array([0, 0, src_z]))
            check = max(check, rel_vec(got, ref))
    print(f"  test_gu3_5's geometry reproduces at {check:.3e}  (its bar: 1e-9)")
    assert check < 1e-9, "the harness does not reproduce the repo's own gate"
    print("  PASS\n")

    print("=" * 78)
    print("THE FORK — direct surfaces at eps-tilde=1 vs the free-space dipole")
    print("=" * 78)
    print(f"{'gap':>8} {'|z|+|zp|':>10} {'worst rel':>13}  {'at':>12}")
    direct = {}
    for gap in GAPS:
        w, where = worst_over_geometry(surfaces, gap)
        direct[gap] = w
        loc = f"rho={where[0]}, {where[1]}" if where else "-"
        print(f"{gap:8.3f} {2 * gap:10.3f} {w:13.3e}  {loc:>12}")

    print()
    print("=" * 78)
    print("READ — against the registered prediction")
    print("=" * 78)
    worst = max(direct.values())
    print(f"  worst direct-surface miss over the ladder : {worst:.3e}")
    print(
        f"  predicted <= 1e-8 (surfaces clean)        : "
        f"{'CONFIRMED' if worst <= 1e-8 else 'NOT CONFIRMED'}"
    )
    print(
        f"  falsifier >= 1e-3 (surfaces carry it)     : "
        f"{'FIRED' if worst >= 1e-3 else 'not fired'}"
    )
    print()
    print("  The assembled cross block at the 5 mm gap is wrong by 6.4e-01")
    print("  (phase 1, rho-spanning deck). Compare that with the row above:")
    print("  if the surfaces are clean, that error is made downstream of them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
