"""momwire#1517 step 1: the C++ bspline front half
(`_bspline_front_core.cpp`), unit by unit, against `BSplineSolver`'s own.

    python scripts/core_split/bspline_front_compare.py DECK_OR_DIR...

For every deck the C++ seam prepares for bspline, `BSplineSolver` is built
from the seam's own keyword arguments (no fill) and each ported unit's
arrays are compared byte for byte (`front_compare.differences`). Units:
`_wire_spec.drop_duplicated_wires` (the dedup and its refusals),
`_build_geometry`, `_build_basis_polynomials` (supp_seg, polys, kcl_A,
the wire knots), the port spec (`junctions`, `node_gaps`, `feeds` as the
constructor normalises them), `_node_gap_columns` and `_port_columns`
(the delta-gap source vectors and the constraint rows). Exit 1 on any mismatch.
"""

from __future__ import annotations

import sys
import warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import front_compare as fc
from seam_compare import _decks


def _py(value):
    """numpy arrays as arrays, lists of ints as lists, scalars as Python."""
    import numpy as np

    if isinstance(value, dict):
        return {k: _py(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_py(v) for v in value]
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value, dtype=np.float64)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def main():
    import numpy as np

    from momwire import BSplineSolver, _accel

    acc = _accel.acc
    if acc is None or not hasattr(acc, "bspline_front_geometry"):
        sys.exit("no accelerator with the bspline front hook loaded")
    warnings.simplefilter("ignore")
    # numpy's own ILP64 dsyrk, for `tangents @ tangents.T` (numpy's syrk case)
    import ctypes
    import glob
    import os

    lib = glob.glob(
        os.path.join(
            os.path.dirname(np.__file__), "..", "numpy.libs", "libscipy_openblas64_*"
        )
    )
    npl = ctypes.CDLL(lib[0])
    syrk = ctypes.cast(npl.scipy_cblas_dsyrk64_, ctypes.c_void_p).value
    blas = [
        ctypes.cast(getattr(npl, f), ctypes.c_void_p).value
        for f in (
            "scipy_cblas_zgemm64_",
            "scipy_cblas_zgemv64_",
            "scipy_cblas_zdotu_sub64_",
            "scipy_zgesv_64_",
        )
    ]
    args = list(sys.argv[1:])
    # `--chunk-mb X`: also the chunked fill (`_compute_Z_dense_chunked`) at a
    # budget of X MB (a float cuts small decks into several windows)
    chunk_mb = 0.0
    if "--chunk-mb" in args:
        i = args.index("--chunk-mb")
        chunk_mb = float(args[i + 1])
        del args[i : i + 2]
    counts, mismatches = Counter(), 0
    for path in _decks(args):
        text = path.read_bytes().decode("latin-1")
        raw = text.encode("latin-1")
        first = acc.bspline_front_geometry(raw)
        samples = []
        if first and "kwargs" in first[0]:
            for wire, npe in zip(
                first[0]["kwargs"]["wires"], first[0]["kwargs"]["n_per_edge_per_wire"]
            ):
                length = float(np.linalg.norm(wire[1] - wire[0]))
                n = int(npe[0])
                samples.append(
                    [(i + 0.5) * length / n for i in range(n)]
                    + [0.0, length, 0.37 * length]
                )
        for k, entry in enumerate(
            acc.bspline_front_geometry(raw, syrk, blas, samples, chunk_mb)
        ):
            try:
                solver = BSplineSolver(**entry["kwargs"])
                geom = solver._build_geometry()
            except Exception as exc:  # noqa: BLE001 -- counted by type
                want = (type(exc).__name__, str(exc))
                if "raised" in entry:
                    counts["raised-alike"] += 1
                    if tuple(entry["raised"]) != want:
                        mismatches += 1
                        print(f"MISMATCH {path}[{k}]: {entry['raised']} vs {want}")
                else:
                    counts[f"python-raised:{type(exc).__name__}"] += 1
                continue
            if "raised" in entry or "declined" in entry:
                mismatches += 1
                print(
                    f"MISMATCH {path}[{k}]: C++ {entry.get('raised') or entry['declined']}"
                )
                continue
            dd = solver._wire_dedup
            want = (
                None if dd is None else (dd.n_input, tuple(dd.kept), tuple(dd.removed))
            )
            d = fc.differences(want, entry["dedup"], "dedup")
            counts["deduped"] += dd is not None
            d += fc.differences(_py(dict(geom)), entry["geom"], "geom")
            if "basis_declined" in entry:
                counts[f"basis-declined:{entry['basis_declined']}"] += 1
            else:
                try:
                    supp, polys, kcl, knots, _wbg = solver._build_basis_polynomials(
                        geom
                    )
                    want = (supp, polys, kcl, [np.asarray(k) for k in knots])
                    got = entry.get("basis")
                    if got is None:
                        d.append(f"basis: C++ raised {entry['basis_raised']}")
                    else:
                        d += fc.differences(want, tuple(got), "basis")
                        counts["basis-compared"] += 1
                        spec = (
                            [list(g) for g in solver.junctions],
                            list(solver.node_gaps),
                            list(solver.feeds),
                        )
                        d += fc.differences(spec, tuple(entry["spec"]), "spec")
                        B, kcl_con = solver._port_columns(
                            geom, knots, _wbg, supp.shape[0], kcl
                        )
                        d += fc.differences(B, entry["port_cols"], "port_cols")
                        d += fc.differences(kcl_con, entry["basis"][2], "kcl_con")
                        if "J" in entry:
                            J = solver._build_J_blocks(geom, solver.k)
                            dj = fc.differences(J, entry["J"], "J")
                            if dj and J.shape == entry["J"].shape:
                                diff = np.argwhere(J != entry["J"])
                                dj.append(
                                    f"J: {len(diff)} entries differ, first {diff[0].tolist()}"
                                )
                            d += dj
                            counts["J-compared"] += 1
                            Z = solver._assemble_Z(J, supp, polys, geom)
                            d += fc.differences(Z, entry["Z"], "Z")
                            counts["Z-compared"] += 1
                            if "Z_chunked" in entry:
                                solver.swept_mem_mb = chunk_mb
                                Zc = solver._compute_Z_dense_chunked(
                                    geom, solver.k, supp, polys
                                )
                                d += fc.differences(Zc, entry["Z_chunked"], "Z_chunked")
                                counts["Z-chunked-compared"] += 1
                                counts["Z-chunked-equals-dense"] += bool(
                                    np.array_equal(Zc, Z)
                                )
                            from momwire import _lapack

                            Bp, kcon = solver._port_columns(
                                geom, knots, _wbg, supp.shape[0], kcl
                            )
                            try:
                                Xp = solver._solve_with_kcl_ports(Z.copy(), Bp, kcon)
                                Yp = _lapack.matmul(Bp.T, Xp)
                                d += fc.differences(
                                    np.ascontiguousarray(Xp), entry["X"], "X"
                                )
                                d += fc.differences(
                                    np.ascontiguousarray(Yp), entry["Y"], "Y"
                                )
                                counts["solve-compared"] += 1
                                c0 = np.ascontiguousarray(Xp[:, 0])
                                d += fc.differences(
                                    [
                                        np.asarray(v)
                                        for v in solver.currents_at_knots(c0)
                                    ],
                                    list(entry["knot_currents"]),
                                    "knot_currents",
                                )
                                if k == 0 and "sample_currents" in entry:
                                    d += fc.differences(
                                        [
                                            np.asarray(v)
                                            for v in solver.currents_at_knots(
                                                c0, samples
                                            )
                                        ],
                                        list(entry["sample_currents"]),
                                        "sample_currents",
                                    )
                                    d += fc.differences(
                                        [
                                            np.asarray(v)
                                            for v in solver.current_slopes(c0, samples)
                                        ],
                                        list(entry["sample_slopes"]),
                                        "sample_slopes",
                                    )
                                    counts["readouts-compared"] += 1
                            except np.linalg.LinAlgError:
                                if entry.get("solve_info", 0) == 0:
                                    d.append("solve: Python singular, C++ not")
                        cols, _v = solver._node_gap_columns(_wbg, supp.shape[0])
                        d += fc.differences(
                            cols, entry["node_gap_cols"], "node_gap_cols"
                        )
                except Exception as exc:  # noqa: BLE001 -- compared by message
                    want = ("ValueError", str(exc))
                    if tuple(entry.get("basis_raised", ())) != want:
                        d.append(
                            f"basis: Python raised {want}, C++ {entry.get('basis_raised')}"
                        )
                    counts["basis-raised-alike"] += 1
            counts["compared"] += 1
            if d:
                mismatches += 1
                print(f"MISMATCH {path}[{k}]: {d[:4]}")
    print(dict(counts), "mismatches", mismatches)
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
