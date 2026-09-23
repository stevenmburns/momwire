"""momwire#1159 probe 8: above ground and in free space, SG is BIT-identical
to main. Dumps every public readout of several decks to an .npz; run once
against main's source (PYTHONPATH=<main>/src) and once against the branch,
then compare with `--compare a.npz b.npz`. Prints which momwire ran.

Decks: an L (two wires, one junction) fed at a KNOT with the point gap, the
same with the segment gap, with a node port at the junction, and with
junction ports; each in free space, over PEC, refl-coef and Sommerfeld
ground (wires above the plane).
"""

import sys
import warnings

import numpy as np

WL = 299792458.0 / 7e6


def decks():
    wires = [
        np.array([(0.0, 0.0, 1.0), (0.0, 0.0, 6.0)]),
        np.array([(0.0, 0.0, 6.0), (4.0, 0.0, 6.0)]),
    ]
    base = dict(
        wires=wires,
        n_per_edge_per_wire=[[10], [8]],
        junctions=[[(0, "end"), (1, "start")]],
        wavelength=WL,
        wire_radius=1e-3,
    )
    grounds = {
        "free": {},
        "pec": dict(ground_z=0.0),
        "refl": dict(ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="refl-coef"),
        "somm": dict(ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld"),
    }
    variants = {
        "knot-point": dict(feeds=[(0, 2.5, 1 + 0j), (1, 2.0, 1 + 0j)]),
        "knot-segment": dict(
            feeds=[(0, 2.5, 1 + 0j), (1, 2.0, 1 + 0j)], feed_model="segment"
        ),
        "node-port": dict(feeds=[(0, 2.5, 1 + 0j)], node_ports=[(0, (0,), 1 + 0j)]),
        "junction-port": dict(feeds=[], junction_ports=[0]),
    }
    for g, gk in grounds.items():
        for v, vk in variants.items():
            if v == "junction-port" and g in ("refl", "somm"):
                continue  # refused on a finite ground
            yield f"{g}/{v}", dict(base, **gk, **vk)


def dump(path):
    import momwire
    from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

    print("momwire from", momwire.__file__)
    out = {}
    for name, d in decks():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = SinusoidalGalerkinSolver(**d)
            z, alpha = s.compute_impedance()
            sol = s.compute_port_solution()
            c = sol.coeffs.sum(axis=1)
            out[f"{name}/z"] = np.asarray(z)
            out[f"{name}/alpha"] = np.asarray(alpha)
            out[f"{name}/y"] = np.asarray(sol.y)
            out[f"{name}/coeffs"] = sol.coeffs
            out[f"{name}/knots"] = np.concatenate(s.currents_at_knots(c))
            out[f"{name}/slopes"] = np.concatenate(s.current_slopes(c))
            arcs = [np.linspace(0, 5.0, 37), np.linspace(0, 4.0, 29)]
            out[f"{name}/sampled"] = np.concatenate(s.currents_at_knots(c, arcs))
            out[f"{name}/moments"] = s.element_currents(c, subdiv=3)[1]
    np.savez(path, **out)
    print(len(out), "arrays ->", path)


def compare(a, b):
    A, B = np.load(a), np.load(b)
    assert sorted(A.files) == sorted(B.files)
    bad = [k for k in A.files if not np.array_equal(A[k], B[k], equal_nan=True)]
    print(f"{len(A.files)} arrays, {len(bad)} differ")
    for k in bad:
        print("  DIFFER", k, np.abs(A[k] - B[k]).max())
    return not bad


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(0 if compare(sys.argv[2], sys.argv[3]) else 1)
    dump(sys.argv[1])
