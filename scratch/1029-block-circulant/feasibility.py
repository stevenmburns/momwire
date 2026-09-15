"""momwire#1029 phase 0: is bs2's dense Z of `verticals.buried_radial_vertical`
block-circulant in the sector index, hub and mast included, and does a
per-harmonic solve reproduce the dense one? No new fill code: Z, the feed
vector and the KCL rows come from the same solver calls `compute_impedance`
makes.

  PYTHONPATH=<momwire src>:<antennaknobs src> \\
      python scratch/1029-block-circulant/feasibility.py --radials 4 12 --out feasibility.jsonl
  ... --structure-only    # sector grouping and sizes only; no Z is assembled

The deck and settings are #1067's momwire arm and the fea1ad0 profile's:
BSplineSolver degree 2, ground ("finite", 13.0, 0.005), the design's defaults with
`n_radials` set on the builder.

The unknowns are grouped as sectors (the basis functions on radial s, in the
order the solver lists them along the wire) and a mast group (everything else,
including any basis function a radial shares with another wire). Sectors are
ordered by the azimuth of each radial's far end.

Block-circulance is measured as max |Z[s,t] - Z[s+1,t+1]| (sector indices mod N)
over max |Z| of the block, reported separately for the radial-radial blocks, the
radial rows against the mast columns, and the mast rows against the radial
columns. The same shift test is applied to the feed vector, the port vector and
the KCL columns.

The per-harmonic solve uses the sector-averaged operator (C_d = mean_s Z[s,s+d],
B = mean_s Z[s,M], C_M = mean_t Z[M,t]):

  Lambda_h = sum_d C_d exp(+2 pi j h d / N);
  harmonic 0 carries the mast, K_0 = [[Lambda_0, sqrt(N) B], [sqrt(N) C_M, Z_MM]];
  harmonics h != 0 are Lambda_h alone.

Z^-1 is applied to the feed vector and to the KCL columns per harmonic, and the
constraint system is closed with the same Schur step `_solve_with_kcl` uses.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.linalg

GROUND = ("finite", 13.0, 0.005)


def _head(path):
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_solver(n_radials, structure_only=False):
    from antennaknobs.designs.verticals.buried_radial_vertical import Builder
    from antennaknobs.engines.momwire import MomwireEngine

    from momwire import BSplineSolver

    b = Builder()
    b.n_radials = n_radials
    eng = MomwireEngine(
        b, solver=BSplineSolver, solver_kwargs={"degree": 2}, ground=GROUND
    )
    if structure_only:
        # No solve before the registration: the solver the engine would make.
        return (
            eng._make_solver(wavelength=eng._wavelength_for(float(b.freq))),
            None,
            0.0,
        )
    made = []
    make = eng._make_solver

    def spy(*args, **kwargs):
        s = make(*args, **kwargs)
        made.append(s)
        return s

    eng._make_solver = spy
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z_engine = complex(np.atleast_1d(eng.impedance())[0])
    return made[-1], z_engine, time.perf_counter() - t0


def polylines_of(solver, geom):
    """Each wire's vertices, (n_i, 3). The solver keeps them under different
    names across versions, so look in the likely places and say which was used."""
    per_wire = geom.get("per_wire")
    if per_wire:
        first = per_wire[0]
        if isinstance(first, dict):
            for key in (
                "polyline",
                "pts",
                "points",
                "vertices",
                "verts",
                "xyz",
                "nodes",
            ):
                if key in first:
                    polylines_of.source = f"geom['per_wire'][w]['{key}']"
                    return [np.asarray(w[key], dtype=float) for w in per_wire]
    for attr in (
        "wires_polylines",
        "wires",
        "_wires",
        "_polylines",
        "polylines",
        "_wire_pts",
        "_wire_polylines",
    ):
        cand = getattr(solver, attr, None)
        if cand is not None and len(cand) and np.asarray(cand[0]).ndim == 2:
            polylines_of.source = f"solver.{attr}"
            return [np.asarray(pl, dtype=float) for pl in cand]
    detail = (
        sorted(per_wire[0])
        if per_wire and isinstance(per_wire[0], dict)
        else type(per_wire)
    )
    attrs = [a for a in dir(solver) if "wire" in a.lower() or "poly" in a.lower()]
    raise RuntimeError(
        f"no polylines found; per_wire[0]: {detail}; solver attrs: {attrs}"
    )


def sector_groups(solver, geom, wire_basis_global):
    polys = polylines_of(solver, geom)
    # wire_basis_global[w] is (kept, {kept_idx: global basis index}); the global
    # indices in kept order run along the wire.
    lists = [
        np.array([l2g[k] for k in range(len(kept))], dtype=np.int64)
        for kept, l2g in wire_basis_global
    ]
    counts = {}
    for ix in lists:
        for i in ix.tolist():
            counts[i] = counts.get(i, 0) + 1
    radial = []
    for w, pl in enumerate(polys):
        far = pl[np.argmax(np.hypot(pl[:, 0], pl[:, 1]))]
        if np.all(pl[:, 2] < 0.0) and math.hypot(far[0], far[1]) > 1.0:
            radial.append((math.atan2(far[1], far[0]) % (2 * math.pi), w))
    radial.sort()
    wires = [w for _ang, w in radial]
    sectors = [
        np.array([i for i in lists[w].tolist() if counts[i] == 1]) for w in wires
    ]
    in_sector = set(np.concatenate(sectors).tolist()) if sectors else set()
    n_total = int(max(max(ix.max() for ix in lists if ix.size), -1) + 1)
    mast = np.array(sorted(set(range(n_total)) - in_sector), dtype=np.int64)
    angles = [a for a, _w in radial]
    return {
        "wires": wires,
        "sectors": sectors,
        "mast": mast,
        "angles": angles,
        "polylines": polys,
        "shared_basis": sorted(i for i, c in counts.items() if c > 1),
        "n_polylines": len(polys),
    }


def rotation_mismatch(groups):
    """Max vertex distance between radial s rotated by 2 pi / N and radial s+1."""
    polys, wires = groups["polylines"], groups["wires"]
    n = len(wires)
    th = 2 * math.pi / n
    rot = np.array(
        [[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]]
    )
    worst = 0.0
    for s in range(n):
        a, b = polys[wires[s]], polys[wires[(s + 1) % n]]
        if a.shape != b.shape:
            return float("inf")
        worst = max(worst, float(np.max(np.linalg.norm(a @ rot.T - b, axis=1))))
    return worst


def shift_eps(blocks):
    """max |X[s] - X[s+1]| over max |X| for a stack indexed by sector."""
    x = np.asarray(blocks)
    return float(
        np.max(np.abs(x - np.roll(x, -1, axis=0))) / max(np.max(np.abs(x)), 1e-300)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radials", type=int, nargs="+", default=[4, 12])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--structure-only", action="store_true")
    args = ap.parse_args()
    import antennaknobs

    import momwire

    meta = {
        "_meta": True,
        "momwire_head": _head(Path(momwire.__file__).parent),
        "antennaknobs_head": _head(Path(antennaknobs.__file__).parent),
        "structure_only": args.structure_only,
    }
    out = open(args.out, "w") if args.out else None  # noqa: SIM115
    if out:
        out.write(json.dumps(meta) + "\n")
    print(json.dumps(meta))
    for n in args.radials:
        s, z_engine, t_engine = build_solver(n, args.structure_only)
        geom = s._build_geometry()
        supp_seg, polys, kcl_A, wire_knots, wire_basis_global = (
            s._build_basis_polynomials(geom)
        )
        n_b = supp_seg.shape[0]
        g = sector_groups(s, geom, wire_basis_global)
        sizes = [len(x) for x in g["sectors"]]
        rec = {
            "radials": n,
            "n_basis": int(n_b),
            "n_polylines": g["n_polylines"],
            "n_sectors_found": len(g["sectors"]),
            "sector_sizes": sorted(set(sizes)),
            "mast_size": int(len(g["mast"])),
            "shared_basis": len(g["shared_basis"]),
            "n_kcl_rows": int(kcl_A.shape[0]),
            "rotation_mismatch_m": rotation_mismatch(g),
            "polylines_from": getattr(polylines_of, "source", None),
            "angles_deg": [round(math.degrees(a), 9) for a in g["angles"]],
            "use_singular_enrichment": bool(
                getattr(s, "use_singular_enrichment", False)
            ),
            "enrichment_variant": str(getattr(s, "enrichment_variant", None)),
            "geom_keys": sorted(geom) if args.structure_only else None,
            "z_engine": None if z_engine is None else [z_engine.real, z_engine.imag],
            "t_engine_s": round(t_engine, 3),
        }
        if not args.structure_only and len(set(sizes)) == 1 and len(sizes) == n:
            m = sizes[0]
            order = np.concatenate([*g["sectors"], g["mast"]])
            t0 = time.perf_counter()
            Z = s._compute_Z_operator(geom, supp_seg, polys)
            v, port_vectors, _vpf, all_voltages, kcl_con = s._feed_drive_and_readout(
                geom, wire_knots, wire_basis_global, n_b, kcl_A
            )
            rec["t_fill_s"] = round(time.perf_counter() - t0, 3)
            u = port_vectors[0]
            t0 = time.perf_counter()
            c_dense = s._solve_with_kcl(Z.copy(), v, kcl_con)
            rec["t_dense_solve_s"] = round(time.perf_counter() - t0, 4)
            z_dense = complex(
                np.atleast_1d(all_voltages)[0] / (u @ c_dense[: u.shape[0]])
            )

            P = Z[np.ix_(order, order)]
            nr = n * m
            R = P[:nr, :nr].reshape(n, m, n, m)
            RM = P[:nr, nr:].reshape(n, m, -1)
            MR = P[nr:, :nr].reshape(-1, n, m)
            ZMM = P[nr:, nr:]
            diag_shift = np.stack(
                [R[s_, :, (s_ + np.arange(n)) % n, :] for s_ in range(n)]
            )
            # diag_shift[s, d] = Z[s, s+d] (m x m); circulant iff independent of s
            rec["eps_radial_blocks"] = shift_eps(diag_shift)
            rec["eps_radial_rows_mast_cols"] = shift_eps(RM) if RM.size else 0.0
            rec["eps_mast_rows_radial_cols"] = (
                shift_eps(np.moveaxis(MR, 1, 0)) if MR.size else 0.0
            )
            vo, uo = v[order], u[order]
            rec["eps_feed_vector"] = (
                shift_eps(vo[:nr].reshape(n, m)) if np.any(vo[:nr]) else 0.0
            )
            rec["feed_on_radials_max"] = float(np.max(np.abs(vo[:nr])))
            rec["port_on_radials_max"] = float(np.max(np.abs(uo[:nr])))
            Ao = kcl_con[:, order]
            rec["eps_kcl_columns"] = shift_eps(
                np.moveaxis(Ao[:, :nr].reshape(-1, n, m), 1, 0)
            )
            rec["kcl_rows_touching_radials"] = int(
                np.sum(np.any(Ao[:, :nr] != 0, axis=1))
            )
            rec["max_abs_Z"] = float(np.max(np.abs(P)))

            t0 = time.perf_counter()
            C = diag_shift.mean(axis=0)  # (n, m, m): C_d
            B = RM.mean(axis=0)  # (m, p)
            CM = MR.mean(axis=1)  # (p, m)
            lam = n * np.fft.ifft(C, axis=0)  # Lambda_h = sum_d C_d e^{+2 pi j h d / N}
            sq = math.sqrt(n)
            p_ = ZMM.shape[0]
            K0 = np.block([[lam[0], sq * B], [sq * CM, ZMM]])
            lu0 = scipy.linalg.lu_factor(K0)
            lus = [scipy.linalg.lu_factor(lam[h]) for h in range(1, n)]

            def zinv(rhs, order=order, nr=nr, n=n, m=m, sq=sq, lu0=lu0, lus=lus):
                rhs = rhs.reshape(rhs.shape[0], -1)
                ro = rhs[order]
                Rr = ro[:nr].reshape(n, m, -1)
                Rh = np.fft.fft(Rr, axis=0) / sq
                xh = np.empty_like(Rh)
                top = scipy.linalg.lu_solve(lu0, np.vstack([Rh[0], ro[nr:]]))
                xh[0] = top[:m]
                xm = top[m:]
                for h in range(1, n):
                    xh[h] = scipy.linalg.lu_solve(lus[h - 1], Rh[h])
                xr = (sq * np.fft.ifft(xh, axis=0)).reshape(nr, -1)
                xo = np.vstack([xr, xm])
                x = np.empty_like(xo)
                x[order] = xo
                return x

            w = zinv(v.astype(np.complex128))[:, 0]
            X = zinv(kcl_con.T.astype(np.complex128))
            lam_k = scipy.linalg.solve(kcl_con @ X, kcl_con @ w)
            c_sec = w - X @ lam_k
            rec["t_sector_solve_s"] = round(time.perf_counter() - t0, 4)
            z_sec = complex(np.atleast_1d(all_voltages)[0] / (u @ c_sec[: u.shape[0]]))
            ch = np.fft.fft(c_sec[order][:nr].reshape(n, m), axis=0) / sq
            h0 = np.linalg.norm(ch[0])
            rec.update(
                m=m,
                p=p_,
                z_dense=[z_dense.real, z_dense.imag],
                z_sector=[z_sec.real, z_sec.imag],
                z_engine_vs_dense_rel=abs(z_engine - z_dense) / abs(z_dense),
                z_sector_vs_dense_rel=abs(z_sec - z_dense) / abs(z_dense),
                currents_sector_vs_dense_rel=float(
                    np.max(np.abs(c_sec - c_dense)) / np.max(np.abs(c_dense))
                ),
                harmonics_nonzero_over_zero=float(
                    max((np.linalg.norm(ch[h]) for h in range(1, n)), default=0.0) / h0
                ),
                cond_K0=float(np.linalg.cond(K0)),
            )
        line = json.dumps(rec)
        print(line, flush=True)
        if out:
            out.write(line + "\n")
            out.flush()
    if out:
        out.close()


if __name__ == "__main__":
    main()
