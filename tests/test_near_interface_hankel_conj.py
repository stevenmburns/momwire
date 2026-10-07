"""The column tail's down-ray Hankel is the up-ray's conjugate, bit for bit.

`column_tail` (the near-interface column rule) takes H2_0(dn rho) as
conj(H1_0(up rho)) instead of computing it: dn rho is exactly conj(up rho),
and xsf's Amos pair is conjugate-equivariant in IEEE arithmetic. That is an
identity about THIS build's arithmetic, so it is checked here, on every build
CI makes, over the arguments the rule produces (the tail nodes of columns
across rho and s) and a random sweep of the first quadrant, as uint64.
"""

import numpy as np
import pytest

from momwire import _near_interface as NI

nia = NI._nia

pytestmark = pytest.mark.skipif(
    nia is None or not hasattr(nia, "near_interface_hankel_conj_mismatches"),
    reason="needs the near-interface C++ twin",
)


def _tail_args(rho, s_min, lam_top):
    """`column_tail`'s up-ray arguments up * rho for one column."""
    scale = np.sqrt(2.0) / (s_min + rho)
    step = min(0.25 * scale, lam_top)
    edges = [0.0]
    while edges[-1] < 60.0 * scale:
        edges.append(edges[-1] + step)
        step *= 2.0
    gx = np.asarray(NI._GX)
    t = np.concatenate(
        [0.5 * (a + b) + 0.5 * (b - a) * gx for a, b in zip(edges[:-1], edges[1:])]
    )
    ray = np.exp(0.25j * np.pi)
    return (lam_top + t * ray) * rho


def test_the_rule_tail_arguments():
    lam_top = 8.0 * 0.627
    w = np.concatenate(
        [
            _tail_args(rho, s, lam_top)
            for rho in np.geomspace(1e-3, 40.0, 60)
            for s in np.geomspace(1e-4, 20.0, 25)
        ]
    )
    assert w.size > 100_000
    assert nia.near_interface_hankel_conj_mismatches(w) == 0


def test_a_random_sweep_of_the_first_quadrant():
    rng = np.random.default_rng(1)
    w = 10.0 ** rng.uniform(-3, 2, 200_000) + 1j * 10.0 ** rng.uniform(-3, 2, 200_000)
    assert nia.near_interface_hankel_conj_mismatches(w) == 0
