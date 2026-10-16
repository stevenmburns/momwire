"""The Python front half's tables, keyed as the C++ front half returns them,
and a byte-for-byte comparison of the two.

`python_front(solver)` runs the Python front half on a constructed
`RazorSolver` -- `_build_geometry`, `_assemble_Z_prepare`, `_port_columns` --
and lays out every array `_razor_front_core.cpp` builds under the core's own
names (`RazorSolveContext.front_port_solution(..., want_front=True)` /
`_accelerators.razor_front_build`). `differences(a, b)` lists every key whose
value is not the same bytes, dtype, shape and Python type on both sides.

Used by tests/test_razor_cxx_front.py and scripts/core_split/cxx_solve_corpus.py.
"""

import numpy as np


def python_front(solver):
    """``(geom, front, cols)`` from the Python front half of `solver`."""
    from momwire import razor

    geom = solver._build_geometry()
    prepared = solver._assemble_Z_prepare(geom)
    cols = solver._port_columns(geom)
    t2 = prepared["t2_chunks"]
    n = prepared["n_basis"]
    fall_a = np.zeros(n, dtype=bool)
    fall_b = np.zeros(n, dtype=bool)
    fall_a[prepared["fall_a"]] = True
    fall_b[prepared["fall_b"]] = True
    omega = solver.c * solver.k
    front = {
        "s_a": prepared["s_a"],
        "s_b": prepared["s_b"],
        "h_a": prepared["h_a"],
        "h_b": prepared["h_b"],
        "q_a": prepared["q_a"],
        "q_b": prepared["q_b"],
        "cent": t2.obs,
        "tans": prepared["tans"].reshape(-1, 3),
        "wts": prepared["wts"].reshape(-1),
        "td_a": np.ascontiguousarray(prepared["td_a"]),
        "td_b": np.ascontiguousarray(prepared["td_b"]),
        "fall_a": fall_a,
        "fall_b": fall_b,
        "a": t2.a,
        "group": t2.group_i,
        "a_ek": t2.a_ek,
        "b_obs": t2.b_obs,
        "xg": np.asarray(t2.xg),
        "wg": np.asarray(t2.wg),
        "window_rows": razor._schedule.elem_rows(
            razor._WEIGHTED_CHUNK_ELEMS
            if prepared["image"] is not None and prepared["image"]["weighted"]
            else razor._CHUNK_ELEMS,
            prepared["n_path"] * n,
        ),
        "k": float(solver.k),
        "c": float(solver.c),
        "c_a_imag": complex(1j * omega * solver.mu).imag,
        "c_phi_imag": complex(1j * omega * solver.eps).imag,
        "radius_per_wire": solver._radius_per_wire,
        "uniform_radius": solver._uniform_radius,
        "n_per_edge_per_wire": solver.n_per_edge_per_wire,
        "feeds": solver.feeds,
        "node_gaps": solver.node_gaps,
        "feed_basis": np.asarray(solver._feed_basis_indices(geom), dtype=np.int64),
    }
    image = prepared["image"]
    if image is not None:
        ti = image["t2_chunks"]
        front["img_seg_p0"] = ti.seg_p0
        front["img_seg_t"] = ti.seg_t
        front["img_td_a"] = np.ascontiguousarray(image["td_a"])
        front["img_td_b"] = np.ascontiguousarray(image["td_b"])
        # The core shares ONE label column between the source sets and the
        # observers; the Python route's image token carries its own copy.
        assert np.array_equal(ti.group_j, t2.group_j)
        assert np.array_equal(ti.obs, t2.obs) and np.array_equal(ti.a_ek, t2.a_ek)
    if t2.group_j.size:
        assert np.array_equal(t2.group_j, t2.group_i)
    if image is not None and "src_l" in image:
        # The composing ground (momwire#1479 S3): what `_source_block_rows`
        # reads off the ground at the fill's omega and off the prepare.
        ground = razor._potential_ground.potential_ground_for(
            solver, geom, solver.k, omega
        )
        front["eps_t"] = complex(ground.eps_tilde)
        front["c2"] = complex(ground.image_coefficient)
        front["obs_pts"] = image["obs_pts"]
        front["sig_a"] = geom["wing_sigma"][:, 0]
        front["sig_b"] = geom["wing_sigma"][:, 1]
        front["r1_max"] = razor._sommerfeld.max_image_distance(
            image["src_l"], image["src_r"], solver.ground_z
        )
        front["n_qp_rem"] = razor._remainder_qp(
            image["obs_pts"],
            image["src_l"],
            image["src_r"],
            solver.ground_z,
            solver.n_qp_sommerfeld,
        )
    return geom, front, cols


def differences(a, b, path=""):
    """Every place `a` and `b` are not the same bytes / dtype / shape /
    Python type, as a list of strings (empty: identical)."""
    out = []
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return [f"{path}: {type(a).__name__} vs {type(b).__name__}"]
        if a.dtype != b.dtype or a.shape != b.shape:
            return [f"{path}: {a.dtype}{a.shape} vs {b.dtype}{b.shape}"]
        if np.ascontiguousarray(a).tobytes() != np.ascontiguousarray(b).tobytes():
            return [f"{path}: bytes differ"]
        return []
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out.append(f"{path}: keys {sorted(set(a) ^ set(b))}")
        for k in sorted(set(a) & set(b), key=str):
            out += differences(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return [
                f"{path}: {type(a).__name__}[{len(a)}] vs {type(b).__name__}[{len(b)}]"
            ]
        for i, (x, y) in enumerate(zip(a, b)):
            out += differences(x, y, f"{path}[{i}]")
        return out
    if type(a) is not type(b):
        # A Python int against a numpy integer, or float against np.float64,
        # is the same value read through a different type: name it.
        return [f"{path}: {type(a).__name__} {a!r} vs {type(b).__name__} {b!r}"]
    if isinstance(a, float):
        if np.float64(a).tobytes() != np.float64(b).tobytes():
            return [f"{path}: {a!r} vs {b!r}"]
        return []
    if isinstance(a, complex):
        if np.complex128(a).tobytes() != np.complex128(b).tobytes():
            return [f"{path}: {a!r} vs {b!r}"]
        return []
    if a != b:
        return [f"{path}: {a!r} vs {b!r}"]
    return []
