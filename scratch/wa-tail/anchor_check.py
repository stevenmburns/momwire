"""Do the sweep anchors (momwire#1471) stay noiseless under the WA tail?

    python anchor_check.py OUT.json [a|b|c ...]

Run with the venv of the tree under test (main or the branch). Three parts:

  a  feed Z from blended tables vs the exact fill: Ward's deck (the #1471
     test's `_DECK`) 9-12 MHz and Dan's buried-radial vertical
     (tests/fixtures/eznec_crossing_source_1282/dan_end1.nec) 13-15 MHz;
     worst |dZ|/|Z| per deck (the anchors comment records 2.8e-12 and
     5.1e-12 on the old engine)
  b  worst blended-vs-exact surface error per region at a mid-ladder target,
     `test_per_surface_interpolation_error_over_the_issues_regions`' own
     configuration (eps_r 13, target mid-rung near 13 - 8.66j, 3.05 lam_m,
     theta >= 5 deg)
  c  smoothness of single table nodes along eps_t: 50 points across two
     ladder steps, at the anchors' reference k2, grazing and steep nodes;
     residual of a degree-9 least-squares fit in ln Im(eps_t), relative to
     the node's scale (a smooth function reads ~1e-14; a panel-count flip in
     the stop rule shows as an isolated residual at the rtol scale)
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

from momwire import _sommerfeld as sm
from momwire import _sommerfeld_below as sb
from momwire._sommerfeld import _SURF_KEYS

EPS0 = 8.8541878128e-12
ROOT = Path(sb.__file__).resolve().parents[2]

WARD = """CM momwire#1471
CE
GW 1 26 -10 0 10 10 0 10 6.45426e-4
GW 2 26 -10 0 -1 10 0 -1 6.45426e-4
GW 3 2 -10 0 -1 -10 0 0 6.45426e-4
GW 4 13 -10 0 10 -10 0 0 6.45426e-4
GE -1
GN 2 0 0 0 13.0 0.005
FR 0 1 0 0 {f:.9f} 1
EX 0 1 13 2 1.0
XQ
EN
"""


def dan_deck():
    t = (ROOT / "tests/fixtures/eznec_crossing_source_1282/dan_end1.nec").read_text(
        encoding="latin-1"
    )
    lines = []
    for ln in t.splitlines():
        if re.match(r"\s*FR\b", ln.upper()):
            lines.append("FR 0 1 0 0 {f:.9f} 1")
        else:
            lines.append(ln.replace("{", "{{").replace("}", "}}"))
    return "\n".join(lines) + "\n"


def feed_z(deck, f_mhz, basis="bspline"):
    from momwire.eznec import _shell

    answered = _shell._answer(deck.format(f=f_mhz), basis=basis)
    assert not isinstance(answered, str), answered[-800:]
    port = answered[1][0].sources[0]
    return complex(port.voltage) / complex(port.current)


def reset():
    sm._GRID_CACHE.clear()
    sb.reset_sweep_state()


def part_a():
    out = {}
    for name, deck, freqs in (
        ("ward", WARD, [9.0 + 0.1 * i for i in range(31)]),
        ("dan", dan_deck(), [13.0 + 0.1 * i for i in range(21)]),
    ):
        os.environ[sb.ENV_ANCHORS] = "on"
        reset()
        blended = [feed_z(deck, f) for f in freqs]
        nb = dict(sb.SWEEP_STATS)
        os.environ[sb.ENV_ANCHORS] = "off"
        reset()
        exact = [feed_z(deck, f) for f in freqs]
        rel = [abs(a - b) / abs(b) for a, b in zip(blended, exact)]
        out[name] = dict(
            worst=max(rel),
            median=float(np.median(rel)),
            rel=rel,
            freqs=freqs,
            exact=[[z.real, z.imag] for z in exact],
            stats=nb,
        )
        print(name, f"worst {max(rel):.2e} median {np.median(rel):.2e}", flush=True)
    os.environ.pop(sb.ENV_ANCHORS, None)
    return out


def part_b():
    eps_r, sigma = 13.0, 0.005
    plan0 = sb._anchor_plan(complex(eps_r, -8.66))
    n = sb._SOMM_BELOW_ANCHOR_POINTS
    s_mid = math.sqrt(plan0.rungs[n // 2 - 1].imag * plan0.rungs[n // 2].imag)
    f_mid = sigma / (2 * math.pi * EPS0 * s_mid)
    omega = 2 * math.pi * f_mid
    eps_t, k2 = complex(eps_r, -sigma / (omega * EPS0)), omega / sm._C_LIGHT

    def grid():
        return sb.get_grid_below(eps_t, k2, 3.05 * sb._lambda_m(eps_t, k2), omega)

    os.environ[sb.ENV_ANCHORS] = "off"
    reset()
    exact = grid()
    os.environ[sb.ENV_ANCHORS] = "on"
    reset()
    blended = grid()
    for g in (exact, blended):
        g._ensure_for(g.r1_max, math.radians(5.0), 0.5 * math.pi)
    out = {}
    for idx, (b, e) in enumerate(zip(blended._regions, exact._regions)):
        if not (b["filled"] and e["filled"]):
            continue
        rows = b["n_r"]
        ev = e["vals"][:, :rows, :]
        scale = np.abs(ev).reshape(len(_SURF_KEYS), -1).max(axis=1)
        err = np.abs(b["vals"] - ev).reshape(len(_SURF_KEYS), -1).max(axis=1)
        out[str(idx)] = float((err / scale).max())
    os.environ.pop(sb.ENV_ANCHORS, None)
    print("b", {k: f"{v:.2e}" for k, v in out.items()}, flush=True)
    return out


def part_c():
    k2 = sb._anchor_k2()
    omega = k2 * sm._C_LIGHT
    eps_r = 13.0
    plan0 = sb._anchor_plan(complex(eps_r, -8.66))
    s0 = -plan0.rungs[3].imag
    s = s0 * np.exp(np.linspace(0.0, 2.0 * sb._SOMM_BELOW_ANCHOR_STEP, 50))
    nodes = [
        (r1, th)
        for r1 in (0.05, 0.5, 2.0)
        for th in (0.0175, 0.05, 0.2, 1.0, 5.0, 45.0)
    ]
    out = {}
    for r1, th in nodes:
        vals = []
        for si in s:
            eps_t = complex(eps_r, -si)
            lam_m = sb._lambda_m(eps_t, k2)
            v = sb.iv_surfaces_direct_below(
                eps_t,
                k2,
                np.array([r1 * lam_m]),
                np.radians([th]),
                rtol=1e-9,
                omega=omega,
            )
            vals.append([complex(v[k][0]) for k in _SURF_KEYS])
        vals = np.array(vals)  # (50, 4)
        x = np.log(s / s0)
        x = (x - x.mean()) / (x.max() - x.mean())
        V = np.vander(x, 10)
        worst = 0.0
        for j in range(vals.shape[1]):
            y = vals[:, j]
            sc = float(np.max(np.abs(y)))
            if sc == 0:
                continue
            coef, *_ = np.linalg.lstsq(V, y, rcond=None)
            res = float(np.max(np.abs(V @ coef - y))) / sc
            worst = max(worst, res)
        out[f"{r1}/{th}"] = worst
        print("c", r1, th, f"{worst:.2e}", flush=True)
    return out


def main():
    dst = sys.argv[1]
    parts = sys.argv[2:] or ["a", "b", "c"]
    res = {}
    if Path(dst).exists():
        res = json.loads(Path(dst).read_text())
    for p in parts:
        res[p] = {"a": part_a, "b": part_b, "c": part_c}[p]()
        Path(dst).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
