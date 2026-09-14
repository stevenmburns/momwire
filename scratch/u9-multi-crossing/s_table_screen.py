"""U9 route 2 screen (PLAN.md Amendment 2): low-band lattices under 0.05 deg.

No Z. Run against the worktree under test (`run_s.sh` runs all of it):

  PYTHONPATH=<worktree>/src python s_table_screen.py panels --out F
  PYTHONPATH=<worktree>/src python s_table_screen.py interp --lattice L2 --out F

The candidates change three module constants before a grid is built: the
floor (the low band's first node), the low band's dtheta, and the tail budget.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

import momwire
from momwire import _ground_refl
from momwire import _sommerfeld_below as below
from momwire._sommerfeld import _SURF_KEYS
from test_buried_serve_553 import F7, SOIL_A

C0 = 299792458.0
EPS0 = 8.8541878128e-12
SPEC = {"A": (13.0, 0.005), "B": (20.0, 0.03), "C": (5.0, 0.001)}
LPDA = (
    Path.home()
    / "antennas/nec-wild/community/cebik-w4rnl/models/LPDAs/nec"
    / "lpma3r5-4-6el86ft75o-buriedradials.nec"
)
LATTICES = {
    "L2": (0.05 - 2 * 0.05 / 3, 0.05 / 3),
    "L2p": (0.05 - 4 * 0.05 / 6, 0.05 / 6),
    "L3": (0.05 - 7 * 0.05 / 9, 0.05 / 9),
}
SCREEN_BUDGET = 48000
BAR = 4.7e-4
THETAS = (
    0.04,
    0.05 - 0.05 / 3,
    0.025,
    0.02,
    0.05 - 2 * 0.05 / 3,
    0.0125,
    0.05 - 7 * 0.05 / 9,
)
PANEL_R1S = (0.0, 0.02, 0.05, 0.2, 1.0, 2.0)
INTERP_R1S = (0.2, 1.0, 1.9, 3.0, 3.9)


def lpda_medium():
    """The LPDA deck's own GN 2 soil and first FR frequency."""
    fr = gn = None
    for line in LPDA.read_text().splitlines():
        f = [x for x in re.split(r"[,\s]+", line.strip()) if x]
        if f and f[0] == "FR" and fr is None:
            fr = float(f[5]) * 1e6
        if f and f[0] == "GN" and gn is None:
            gn = (float(f[5]), float(f[6]))
    return gn, fr


def media():
    out = [(f"{s}/{f / 1e6:.0f}MHz", SPEC[s], f) for s in SPEC for f in (7e6, 21e6)]
    out.append(("A/F7", tuple(SOIL_A), F7))
    gn, fr = lpda_medium()
    out.append(("LPDA", gn, fr))
    return out


def medium(soil, f):
    om = 2.0 * math.pi * f
    k2 = om / C0
    eps_t = _ground_refl.eps_tilde(soil, om, EPS0)
    return eps_t, k2, om, below.lambda_medium(eps_t, k2)


def set_lattice(name):
    floor, dth = LATTICES[name]
    below._SOMM_BELOW_TH_MIN_DEG = floor
    below._SOMM_BELOW_DTH_BAND_LO_DEG = dth
    below._MAX_TAIL_PANELS = SCREEN_BUDGET


def panels():
    below._MAX_TAIL_PANELS = SCREEN_BUDGET
    assert below._use_below_accel(), "the screen measures the accelerated contour"
    rows = []
    for th in THETAS:
        law = 6.4 / math.tan(math.radians(th))
        worst, where, noncon, errors = 0, None, 0, []
        for name, soil, f in media():
            eps_t, k2, om, lam_m = medium(soil, f)
            for r1l in PANEL_R1S:
                h = below.Health()
                try:
                    below.iv_surfaces_direct_below(
                        eps_t,
                        k2,
                        np.array([r1l * lam_m]),
                        np.radians([th]),
                        rtol=1e-9,
                        omega=om,
                        health=h,
                    )
                except ValueError as exc:
                    errors.append([name, r1l, str(exc)[:120]])
                    continue
                noncon += h.nonconvergent
                if h.max_tail_panels > worst:
                    worst, where = h.max_tail_panels, [name, r1l]
        rows.append(
            dict(
                theta_deg=th,
                law=law,
                worst=worst,
                worst_over_law=worst / law,
                where=where,
                nonconvergent=noncon,
                errors=errors,
            )
        )
        print(json.dumps(rows[-1]), flush=True)
    return dict(
        mode="panels", momwire=momwire.__file__, budget=SCREEN_BUDGET, rows=rows
    )


def interp(lattice):
    set_lattice(lattice)
    assert below._use_below_accel(), "the screen measures the accelerated contour"
    floor, dth = LATTICES[lattice]
    lim = math.radians(0.05 + 0.05 / 3) * (1 + 1e-9)
    out = []
    worst, where = 0.0, None
    for name, soil, f in media():
        eps_t, k2, om, lam_m = medium(soil, f)
        t0 = time.time()
        g = below.SommerfeldGridBelow(
            eps_t, k2, below._SOMM_BELOW_R1_CAP_LAMBDA_M * lam_m, omega=om
        )
        t_build = time.time() - t0
        t0 = time.time()
        g._ensure_band_lo()
        t_fill = time.time() - t0
        n_lo = sum(g._regions[i]["n_r"] * g._regions[i]["n_th"] for i in g._band_lo_idx)
        reg = g._regions[below.region_index(below._ZONE_INNER, below._BAND_LO)]
        nodes = reg["th0"] + reg["dth"] * np.arange(reg["n_th"])
        cells = [(a, b) for a, b in zip(nodes[:-1], nodes[1:]) if b <= lim]
        th_q = np.array([a + fr * (b - a) for a, b in cells for fr in (0.5, 0.33)])
        m_worst, m_where, noncon = 0.0, None, 0
        for r1l in INTERP_R1S:
            r1 = np.full(th_q.shape, r1l * lam_m)
            got = g.eval(r1, th_q)
            h = below.Health()
            ref = below.iv_surfaces_direct_below(
                eps_t, k2, r1, th_q, rtol=1e-9, omega=om, health=h
            )
            noncon += h.nonconvergent
            for key in _SURF_KEYS:
                a = np.asarray(got[key], dtype=complex)
                b = np.asarray(ref[key], dtype=complex)
                rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
                i = int(np.argmax(rel))
                if rel[i] > m_worst:
                    m_worst = float(rel[i])
                    m_where = [r1l, key, float(np.degrees(th_q[i]))]
        if m_worst > worst:
            worst, where = m_worst, [name, *m_where]
        out.append(
            dict(
                medium=name,
                soil=list(soil),
                f_hz=f,
                lam_m=lam_m,
                th0_deg=float(np.degrees(nodes[0])),
                n_th_lo=int(reg["n_th"]),
                n_lo_nodes_all_zones=int(n_lo),
                cells_queried=len(cells),
                worst_rel=m_worst,
                worst_at=m_where,
                ref_nonconvergent=noncon,
                build_s=round(t_build, 2),
                lo_fill_s=round(t_fill, 2),
            )
        )
        print(json.dumps(out[-1]), flush=True)
    return dict(
        mode="interp",
        momwire=momwire.__file__,
        lattice=lattice,
        floor_deg=floor,
        dth_lo_deg=dth,
        budget=SCREEN_BUDGET,
        bar=BAR,
        worst_rel=worst,
        worst_at=where,
        within_bar=worst <= BAR,
        media=out,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("panels", "interp"))
    ap.add_argument("--lattice", choices=sorted(LATTICES))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rec = panels() if args.what == "panels" else interp(args.lattice)
    args.out.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
