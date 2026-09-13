"""momwire#1027 step 2(f), P2f.4: where along the rod the two engines' currents
differ.

Free space at f * sqrt(13) = 25.599414 MHz, refine 16, L = 0.15 and 0.60 m;
one deck mesh for both engines (the step 2(a) rod). NEC-5's printed
segment-centre currents and segment lengths define the sample points; momwire's
spline current is evaluated at the same arc positions. Each profile is
normalised by its own engine's feed current, and m = sum(I * dl) / I_feed is the
current moment a short dipole's far field depends on.

  P2f.4a  R_NEC-5 / R_momwire against |m_NEC-5 / m_momwire|^2
  P2f.4b  the share of m_momwire - m_NEC-5 that comes from within 15 mm of the
          feed gap, read two ways: the share of its magnitude, and the share of
          its projection on m_momwire (the component that changes |m|, hence R)

Run from the momwire repo root with antennaknobs installed alongside:
  NEC5_EXE=<nec5cl> python scratch/1027-rod/step2f_currents.py [--out F]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from step2f_power_balance import free_space_builder

from antennaknobs.engines.momwire import MomwireEngine
from antennaknobs.engines.nec5 import NEC5Engine

NEAR_FEED = 0.015
TOP = -0.2
_CUR = "- - - Wire Currents - - -"
_AIP = "ANTENNA INPUT PARAMETERS"


def nec5_side(b):
    """(z centres, lengths, currents, feed current, Z) from one NEC-5 printout."""
    eng = NEC5Engine(b, ground="free")
    text = eng._run(eng.deck([b.freq]))
    rows = []
    for line in text.split(_CUR, 1)[1].splitlines():
        toks = line.split()
        if len(toks) != 10:
            if rows:
                break
            continue
        try:
            rows.append([float(t) for t in toks[2:8]])
        except ValueError:
            if rows:
                break
    arr = np.array(rows)
    z_n, len_n = arr[:, 2], arr[:, 3]
    cur = arr[:, 4] + 1j * arr[:, 5]
    feed = None
    for line in text.split(_AIP, 1)[1].splitlines():
        toks = line.split()
        if len(toks) == 12:
            try:
                feed = complex(float(toks[5]), float(toks[6]))
                z_in = complex(float(toks[7]), float(toks[8]))
                break
            except ValueError:
                continue
    if feed is None:
        raise RuntimeError("no input-parameter row")
    return z_n, len_n, cur, feed, z_in


def momwire_at(b, z_points):
    """momwire's current at vertical positions z, and its feed current (V = 1)."""
    eng = MomwireEngine(b, ground="free")
    sim, coeffs, z_in = eng._solved_excited(eng._wavelength_for(b.freq))
    polys = [np.asarray(p, dtype=float) for p in sim.wires_polylines]
    s_arrays = [[] for _ in polys]
    where = []
    for z in z_points:
        hit = None
        for w, poly in enumerate(polys):
            arc0 = 0.0
            for a, c in zip(poly[:-1], poly[1:], strict=True):
                lo, hi = sorted((a[2], c[2]))
                if lo - 1e-12 <= z <= hi + 1e-12:
                    hit = (w, arc0 + abs(z - a[2]), np.sign(c[2] - a[2]))
                    break
                arc0 += float(np.linalg.norm(c - a))
            if hit:
                break
        if hit is None:
            raise RuntimeError(f"z = {z} is on no polyline")
        w, s, sign = hit
        where.append((w, len(s_arrays[w]), sign))
        s_arrays[w].append(s)
    vals = sim.currents_at_knots(coeffs, s_array=[np.array(s) for s in s_arrays])
    out = np.array([sign * vals[w][i] for (w, i, sign) in where])
    return out, 1.0 / complex(z_in), complex(z_in)


def one_length(L):
    b = free_space_builder(L, 16)
    z_norm, len_norm, i_n5, feed_n5, z_n5 = nec5_side(b)
    # NEC-5 prints lengths normalised; recover its scale from the rod itself
    scale = L / float(np.sum(len_norm))
    z = z_norm * scale
    dl = len_norm * scale
    i_mw, feed_mw, z_mw = momwire_at(free_space_builder(L, 16), z)

    p_n5 = i_n5 / feed_n5
    p_mw = i_mw / feed_mw
    m_n5 = complex(np.sum(p_n5 * dl))
    m_mw = complex(np.sum(p_mw * dl))
    dm_i = (p_mw - p_n5) * dl
    dm = complex(np.sum(dm_i))
    z_feed = TOP - L / 2.0
    near = np.abs(z - z_feed) < NEAR_FEED
    ends = np.abs(np.abs(z - z_feed) - L / 2.0) < NEAR_FEED
    dm_near = complex(np.sum(dm_i[near]))
    dm_ends = complex(np.sum(dm_i[ends]))
    unit = np.conj(m_mw) / abs(m_mw)

    def proj(x):
        return float((x * unit).real)

    rec = dict(
        L=L,
        n_samples=int(z.size),
        scale_m_per_unit=scale,
        z_momwire=str(z_mw),
        z_nec5=str(z_n5),
        R_ratio=z_n5.real / z_mw.real,
        m_momwire=str(m_mw),
        m_nec5=str(m_n5),
        moment_ratio_sq=abs(m_n5 / m_mw) ** 2,
        dm=str(dm),
        share_near_abs=abs(dm_near) / abs(dm),
        share_near_proj=proj(dm_near) / proj(dm),
        share_ends_abs=abs(dm_ends) / abs(dm),
        share_ends_proj=proj(dm_ends) / proj(dm),
        profile=[
            dict(z=float(zi), dl=float(li), momwire=str(a), nec5=str(c))
            for zi, li, a, c in zip(z, dl, p_mw, p_n5, strict=True)
        ],
    )
    print(
        f"L={L:4.2f}  samples {z.size}  scale {scale:.6g} m/unit\n"
        f"  Z momwire {z_mw:.6g}   Z NEC-5 {z_n5:.6g}   R ratio {rec['R_ratio']:.5f}\n"
        f"  m momwire {m_mw:.6g}   m NEC-5 {m_n5:.6g}   |m ratio|^2 "
        f"{rec['moment_ratio_sq']:.5f}\n"
        f"  dm {dm:.4g}: within {NEAR_FEED * 1000:.0f} mm of feed "
        f"{rec['share_near_abs']:.3f} (|.|) / {rec['share_near_proj']:.3f} (proj);  "
        f"within {NEAR_FEED * 1000:.0f} mm of the ends {rec['share_ends_abs']:.3f} / "
        f"{rec['share_ends_proj']:.3f}",
        flush=True,
    )
    for dist in (0.0025, 0.0075, 0.02, 0.05, L / 2.0 - 0.01):
        k = int(np.argmin(np.abs(np.abs(z - z_feed) - dist)))
        print(
            f"    |z - z_feed| = {abs(z[k] - z_feed) * 1000:7.2f} mm  "
            f"I/I0 momwire {p_mw[k]:.5f}  NEC-5 {p_n5[k]:.5f}",
            flush=True,
        )
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    exe = os.environ["NEC5_EXE"]
    meta = dict(
        nec5_exe=exe, nec5_sha256=hashlib.sha256(Path(exe).read_bytes()).hexdigest()
    )
    print(meta, flush=True)
    out = [one_length(L) for L in (0.15, 0.60)]
    if args.out is not None:
        args.out.write_text(json.dumps(dict(meta=meta, rows=out), indent=1))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
