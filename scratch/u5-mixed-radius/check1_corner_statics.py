"""U5 (a), identity check 1: does a per-pair corner radius rule keep the node's
O(1/a) corner singularities cancelling at a two-radius crossing node?

At the crossing node the four corner contributions the fill adds are

    Z[na,na] += c_aa(a_aa)      self completion, above family, closed form
    Z[nb,nb] += c_bb(a_bb)      self completion, below family, closed form
    Z[na,nb] -= c1 V(a_x1)      cross corner, above-observer row
    Z[nb,na] -= c1 V(a_x2)      cross corner, below-observer row (transpose)

(signs: `Z += self_completions`, `Z -= t_ab`, `Z -= t_ab.T`, and the corner in
t_ab is -sigma_a sigma_b c1 V = +c1 V with sigma_a = -1, sigma_b = +1). With
current continuous through the node the node tents share one coefficient, so
the corner energy is S = c_aa + c_bb - c1 V(a_x1) - c1 V(a_x2). Each piece has
a static part ~ 1/a with equal weights (DERIVATION-SAME-MEDIUM section 3). A
rule that keeps S bounded as both radii shrink together preserves the
cancellation; one that does not leaves O(1/a) residue.

Self corners use each family's own radius (a_aa = a_A, a_bb = a_B). Rules for
the two cross corners: observer (a_A, a_B), source (a_B, a_A), harmonic mean,
geometric mean, arithmetic mean, max, min. Equal radii is the control.

Everything is momwire's own: `_buried_medium`, `_c1_moment`, `six_point`,
`_g_of_r`, on a soil-A crossing deck at 7 MHz. No src change.
"""

from __future__ import annotations

import math

import numpy as np

from momwire import _crossing_fill, _near_interface
from momwire._sommerfeld_transmitted import _c1_moment
from momwire.bspline import BSplineSolver

C0 = 299792458.0
WL7 = C0 / 7e6
SOIL_A = (13.0, 0.005)


def solver():
    below = np.array([(0.0, 0.0, -2.0), (0.0, 0.0, 0.0)])
    above = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 10.0)])
    return BSplineSolver(
        wires=[below, above],
        n_per_edge_per_wire=[[8], [20]],
        junctions=[[(0, "end"), (1, "start")]],
        feeds=[(1, 4.3333333333, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )


def pieces(s):
    eps_t, eps_m, k_p, k_m, c2, a_m = s._buried_medium()
    omega, mu, eps0 = s.omega, s.mu, s.eps
    c1 = _c1_moment(omega, mu)

    def c_self(k, wgt, eps, a):
        beta_dir = 1.0 / (1j * omega * eps * 4 * math.pi)
        beta_img = wgt / (1j * omega * eps * 4 * math.pi)
        g = complex(_crossing_fill._g_of_r(k, np.array([a]))[0])
        return (beta_dir - beta_img) * g  # the node's image end IS the node: R = a both

    def c_aa(a):
        return c_self(k_p, c2, eps0, a)

    def c_bb(a):
        return c_self(k_m, a_m, eps_m, a)

    def c_x(a):
        return c1 * complex(
            _near_interface.six_point(eps_t, k_p, a, 0.0, 0.0, rtol=1e-10)[1]
        )

    return c_aa, c_bb, c_x


RULES = {
    "observer": lambda aA, aB: (aA, aB),
    "source": lambda aA, aB: (aB, aA),
    "harmonic": lambda aA, aB: (2 * aA * aB / (aA + aB),) * 2,
    "geometric": lambda aA, aB: (math.sqrt(aA * aB),) * 2,
    "arithmetic": lambda aA, aB: ((aA + aB) / 2,) * 2,
    "max": lambda aA, aB: (max(aA, aB),) * 2,
    "min": lambda aA, aB: (min(aA, aB),) * 2,
}


def main():
    s = solver()
    c_aa, c_bb, c_x = pieces(s)
    print(f"medium eps_t={s._buried_medium()[0]:.4f}")
    # static weights, measured: a * piece at tiny a
    a0 = 1e-6
    print(
        f"static weights (a*piece at a=1e-6): aa {a0 * c_aa(a0):.6g}  "
        f"bb {a0 * c_bb(a0):.6g}  cross {a0 * c_x(a0):.6g}"
    )
    for ratio in (1.0, 2.0, 4.0, 10.0):
        print(f"\n== radius ratio a_A / a_B = {ratio:g}")
        print(f"   {'scale a_A':>10} " + " ".join(f"{r:>22}" for r in RULES))
        for aA in (1e-3, 1e-4, 1e-5, 1e-6):
            aB = aA / ratio
            base = c_aa(aA) + c_bb(aB)
            row = []
            for rule in RULES.values():
                x1, x2 = rule(aA, aB)
                S = base - c_x(x1) - c_x(x2)
                row.append(abs(S) * aA)  # a * |S|: O(1/a) residue shows as a constant
            print(f"   {aA:10.0e} " + " ".join(f"{v:22.6e}" for v in row))
    print(
        "\ncolumns are a_A * |S|: bounded rules fall toward 0 as a_A shrinks; O(1/a) residue stays constant"
    )


if __name__ == "__main__":
    main()
