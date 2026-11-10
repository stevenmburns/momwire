"""momwire#1897: the sector route against the dense solve on C_N decks with
vertical wires off the axis, free space and over a Sommerfeld half-space.
One JSON line per (deck, ground), flushed, to argv[1].

Decks:
* cage4 / cage6: a mast inside N off-axis verticals, top ring, spokes; fed
  at the mast foot (compute_impedance, harmonic 0).
* cage4-posts: the same with a 0 V port on every vertical, loaded 1 pF
  across each (compute_y_matrix + port algebra, every harmonic).
* parasitic4: a fed centre vertical with four parasitic verticals at 0.25 wl
  and N radials under each (a 4-square-like ring, off-axis verticals each
  with its own junction to its radials).
"""

import json
import math
import sys
import time
import warnings

import numpy as np

from momwire import BSplineSolver

WL = 2.0
SOMM = dict(ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld")


def cage(n, lift):
    H, FOOT, R = 0.5, 0.05, 0.08
    z0, z1, z2 = lift, lift + FOOT, lift + H
    foot = [
        (R * math.cos(2 * math.pi * i / n), R * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]
    w = [
        np.array([(0, 0, z0), (0, 0, z1)], float),
        np.array([(0, 0, z1), (0, 0, z2)], float),
    ]
    npe = [[3], [10]]
    for x, y in foot:
        w.append(np.array([(x, y, z1), (x, y, z2)]))
        npe.append([8])
    for i in range(n):
        (ax, ay), (bx, by) = foot[i], foot[(i + 1) % n]
        w.append(np.array([(ax, ay, z2), (bx, by, z2)]))
        npe.append([3])
    for x, y in foot:
        w.append(np.array([(0, 0, z1), (x, y, z1)]))
        npe.append([3])
    return w, npe, [(0, 0.5 * FOOT, 1 + 0j)]


def parasitic(n, lift):
    """Centre vertical fed at its base; n parasitic verticals at radius 0.5 m
    (0.25 wl); each element stands on 4 short radials at z = lift."""
    H, RR, ARM = 0.48, 0.5, 0.25
    w, npe = [], []
    centres = [(0.0, 0.0)] + [
        (RR * math.cos(2 * math.pi * i / n), RR * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]
    feeds = []
    for k, (cx, cy) in enumerate(centres):
        base = len(w)
        w.append(np.array([(cx, cy, lift), (cx, cy, lift + 0.04)]))
        npe.append([3])
        w.append(np.array([(cx, cy, lift + 0.04), (cx, cy, lift + H)]))
        npe.append([12])
        if k == 0:
            feeds.append((base, 0.02, 1 + 0j))
        for j in range(4):
            # radials point along the element's own azimuth frame so the ring is C_n
            phi = (math.atan2(cy, cx) if k else 0.0) + 2 * math.pi * j / 4
            w.append(
                np.array(
                    [
                        (cx, cy, lift),
                        (cx + ARM * math.cos(phi), cy + ARM * math.sin(phi), lift),
                    ]
                )
            )
            npe.append([5])
    return w, npe, feeds


def solve(w, npe, feeds, rot, ground, ports=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = BSplineSolver(
            wires=w,
            n_per_edge_per_wire=npe,
            degree=2,
            wavelength=WL,
            wire_radius=0.002,
            feeds=feeds,
            rotational_symmetry=rot,
            **(SOMM if ground else {}),
        )
        t = time.perf_counter()
        if ports:
            Y = s.compute_y_matrix()
            omega = 2 * math.pi * 299792458.0 / WL
            zl = 1 / (1j * omega * 1e-12)
            Zm = np.linalg.inv(Y)
            k = Y.shape[0] - 1
            z = Zm[0, 0] - Zm[0, 1:] @ np.linalg.solve(
                Zm[1:, 1:] + zl * np.eye(k), Zm[1:, 0]
            )
        else:
            z = complex(np.atleast_1d(s.compute_impedance()[0])[0])
        dt = time.perf_counter() - t
    m = s._rotational_map
    return complex(z), dt, (m.n_sectors if m else None)


out = open(sys.argv[1], "a")
for ground in (False, True):
    lift = 0.3 if ground else 0.0
    cases = [
        ("cage4", *cage(4, lift), False),
        ("cage6", *cage(6, lift), False),
        (
            "cage4-posts",
            *cage(4, lift)[:2],
            [(0, 0.025, 1 + 0j)] + [(2 + i, 0.2, 0j) for i in range(4)],
            True,
        ),
        ("parasitic4", *parasitic(4, lift), False),
    ]
    for name, w, npe, feeds, ports in cases:
        zr, tr, n = solve(w, npe, feeds, True, ground, ports)
        zd, td, _ = solve(w, npe, feeds, False, ground, ports)
        rec = dict(
            deck=name,
            ground="sommerfeld" if ground else "free",
            N=n,
            z_route=[zr.real, zr.imag],
            z_dense=[zd.real, zd.imag],
            rel=abs(zr - zd) / abs(zd),
            t_route=tr,
            t_dense=td,
        )
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(rec, flush=True)
