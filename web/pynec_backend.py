"""PyNEC drop-in backend for the web UI.

Mirrors the response shape of `web.server`'s pysim solver paths so the
frontend can swap between solvers via a `solver` field on every request.

PyNEC is optional: `HAVE_PYNEC` is False if the import fails, and the
server falls back to pysim with a one-time warning.
"""

from __future__ import annotations

import math
import time

import numpy as np

try:
    import PyNEC as nec  # type: ignore

    HAVE_PYNEC = True
except ImportError:
    HAVE_PYNEC = False
    nec = None

from .examples import REGISTRY as EXAMPLES


C_LIGHT = 299_792_458.0

# Typical "average" earth, matching antenna_designer/sim.py.
GROUND_DIELECTRIC = 10.0
GROUND_CONDUCTIVITY = 0.002


def _segment_centers_to_knot_currents(
    cur_per_seg: np.ndarray,
    n_knots: int,
    junction_at_start: bool = False,
    junction_at_end: bool = False,
) -> np.ndarray:
    """Map NEC's per-segment-center currents onto the (n_knots,)-knot array
    the UI expects.

    Interior knot k sits between segments k-1 and k, so we average. The
    boundary knots default to zero (open-wire BC), but at a junction with
    another wire the current is continuous through the endpoint — pass
    junction_at_start/end=True to carry the adjacent segment-center value
    onto the boundary knot instead.
    """
    full = np.zeros(n_knots, dtype=np.complex128)
    if cur_per_seg.shape[0] != n_knots - 1:
        raise RuntimeError(
            f"segment-current length {cur_per_seg.shape[0]} doesn't match "
            f"n_knots-1 = {n_knots - 1}"
        )
    full[1:-1] = 0.5 * (cur_per_seg[:-1] + cur_per_seg[1:])
    if junction_at_start:
        full[0] = cur_per_seg[0]
    if junction_at_end:
        full[-1] = cur_per_seg[-1]
    return full


def _run_solve(
    c,
    n_seg_total: int,
    feed_seg: int,
    freq_mhz: float,
    ground: bool = False,
    feed_tag: int = 1,
    ground_fast: bool = False,
):
    if ground:
        # ITYPE=2: Sommerfeld-Norton, full kernel integration — accurate for
        # antennas a fraction of a wavelength above ground, ~100x slower per
        # solve than free space.
        # ITYPE=0: reflection-coefficient approximation — applies a Fresnel
        # reflection on the image current rather than evaluating Sommerfeld
        # integrals. Cheap (~10x faster than ITYPE=2) and accurate enough
        # away from grazing angles; degrades for very low antennas.
        itype = 0 if ground_fast else 2
        c.gn_card(itype, 0, GROUND_DIELECTRIC, GROUND_CONDUCTIVITY, 0, 0, 0, 0)
    else:
        c.gn_card(-1, 0, 0, 0, 0, 0, 0, 0)  # free space
    c.ex_card(0, feed_tag, feed_seg, 0, 1.0, 0.0, 0, 0, 0, 0)
    c.fr_card(0, 1, freq_mhz, 0)
    c.xq_card(0)
    sc = c.get_structure_currents(0)
    cur_arr = np.asarray(sc.get_current(), dtype=np.complex128)
    tag_arr = np.asarray(sc.get_current_segment_tag())
    return cur_arr, tag_arr




def _polyline_knots(path, npe_list) -> np.ndarray:
    """Concatenated per-edge knot positions, deduping shared corners."""
    parts = []
    for i, n_e in enumerate(npe_list):
        seg = np.linspace(path[i], path[i + 1], n_e + 1)
        parts.append(seg if i == 0 else seg[1:])
    return np.vstack(parts)





_FANDIPOLE_FEED_GAP = 0.01  # meters; half-gap, matches antenna_designer eps


def _fandipole_ring(k_bands):
    """K cone-direction ring positions evenly distributed at 360°/K around
    the cone axis. K=2 places the two bands at opposite ends of a diameter
    (180° apart), K=3 at the vertices of an equilateral triangle, etc.
    Matches a physical K-spreader fan dipole where the bands fan
    symmetrically around the central feed axis.
    """
    step = 360.0 / k_bands
    return [
        (
            math.cos(math.radians(i * step)),
            math.sin(math.radians(i * step)),
        )
        for i in range(k_bands)
    ]


def _build_fandipole(req: dict):
    """Cone-arrangement fan dipole. Up to 5 bands, each a two-edge arm on
    each side (S->A_i->B_i, mirrored T->Ay_i->By_i), all bands sharing the
    T->S feed gap. Mirrors antenna_designer/.../fandipole.py.
    """
    n_per_wire = int(req.get("n_per_wire", 21))
    design_freq_mhz = float(req.get("design_freq_mhz", 14.3))
    n_bands = int(req.get("n_bands", 2))
    if not 1 <= n_bands <= 5:
        raise ValueError(f"n_bands must be in [1, 5], got {n_bands}")
    band_lengths_m = list(req.get("band_lengths_m", [10.2551, 5.2691]))
    if len(band_lengths_m) < n_bands:
        raise ValueError(
            f"band_lengths_m has {len(band_lengths_m)} entries, need {n_bands}"
        )
    band_lengths_m = band_lengths_m[:n_bands]
    band_freqs_mhz = list(req.get("band_freqs_mhz", []))[:n_bands]
    slope = float(req.get("slope", 0.5))
    cone_radius_m = float(req.get("cone_radius_m", 0.12))
    t0_factor = float(req.get("t0_factor", math.sqrt(2.0)))
    wire_radius = float(req.get("wire_radius", 0.0005))
    ground = bool(req.get("ground", False))
    ground_fast = bool(req.get("ground_fast", False))
    height_m = float(req.get("height_m", 0.0))

    eps_feed = _FANDIPOLE_FEED_GAP
    t0 = cone_radius_m * t0_factor
    Zc = 1.0 / math.sqrt(1.0 + slope * slope)
    Zs = slope * Zc
    z_offset = height_m if ground else 0.0

    def ry(p):
        return (p[0], -p[1], p[2])

    S = (0.0, eps_feed, z_offset)
    T = ry(S)
    C = (S[0], S[1] + t0 * Zc, S[2] - t0 * Zs)
    lst = _fandipole_ring(n_bands)

    A_pos = [
        (
            C[0] + cone_radius_m * x,
            C[1] + cone_radius_m * y * Zs,
            C[2] + cone_radius_m * y * Zc,
        )
        for (x, y) in lst
    ]
    # dist(S, A_i) is independent of i: sqrt(radius^2 + t0^2). Each arm's
    # axial leg makes up the remainder of the half-band-length.
    ls = []
    for i, (q, a) in enumerate(zip(band_lengths_m, A_pos)):
        dsa = math.sqrt(sum((s_ - a_) ** 2 for s_, a_ in zip(S, a)))
        l_i = q / 2.0 - dsa
        if l_i <= 0:
            raise ValueError(
                f"band {i}: cone geometry leaves no axial leg "
                f"(band_length={q:.3f} m, radial leg={dsa:.3f} m)"
            )
        ls.append(l_i)
    B_pos = [(a[0], a[1] + l * Zc, a[2] - l * Zs) for (l, a) in zip(ls, A_pos)]
    A_neg = [ry(a) for a in A_pos]
    B_neg = [ry(b) for b in B_pos]

    c = nec.nec_context()
    geo = c.get_geometry()
    # Tag 1: feed gap T->S (1 segment, delta-gap source location).
    geo.wire(1, 1, *T, *S, wire_radius, 1.0, 1.0)
    next_tag = 2
    band_tags = []
    for i in range(n_bands):
        t_sr, t_sa, t_tr, t_ta = next_tag, next_tag + 1, next_tag + 2, next_tag + 3
        geo.wire(t_sr, n_per_wire, *S, *A_pos[i], wire_radius, 1.0, 1.0)
        geo.wire(t_sa, n_per_wire, *A_pos[i], *B_pos[i], wire_radius, 1.0, 1.0)
        geo.wire(t_tr, n_per_wire, *T, *A_neg[i], wire_radius, 1.0, 1.0)
        geo.wire(t_ta, n_per_wire, *A_neg[i], *B_neg[i], wire_radius, 1.0, 1.0)
        band_tags.append((t_sr, t_sa, t_tr, t_ta))
        next_tag += 4
    c.geometry_complete(0)

    wavelength_design = C_LIGHT / (design_freq_mhz * 1e6)
    return {
        "context": c,
        "feed_tag": 1,
        "feed_seg": 1,
        "n_per_wire": n_per_wire,
        "n_bands": n_bands,
        "band_tags": band_tags,
        "band_lengths_m": band_lengths_m,
        "band_freqs_mhz": band_freqs_mhz,
        "S": S,
        "T": T,
        "A_pos": A_pos,
        "B_pos": B_pos,
        "A_neg": A_neg,
        "B_neg": B_neg,
        "slope": slope,
        "cone_radius_m": cone_radius_m,
        "t0_m": t0,
        "wavelength_design": wavelength_design,
        "design_freq_mhz": design_freq_mhz,
        "ground": ground,
        "ground_fast": ground_fast,
        "z_offset": z_offset,
    }


def _fandipole_band_label(i: int, freqs_mhz: list[float], length_m: float) -> str:
    if i < len(freqs_mhz):
        return f"{freqs_mhz[i]:.2f} MHz"
    return f"band {i} ({length_m:.2f} m)"


def solve_fandipole(req: dict) -> dict:
    """Multi-band fan dipole via PyNEC. All band wires share the T->S feed
    gap; PyNEC's segment-endpoint junctions stitch the geometry together
    automatically.
    """
    meas_freq_mhz = float(
        req.get("measurement_freq_mhz", req.get("design_freq_mhz", 14.3))
    )
    b = _build_fandipole(req)
    c = b["context"]
    n_per = b["n_per_wire"]
    n_total = 1 + 4 * b["n_bands"] * n_per

    t0_clock = time.perf_counter()
    cur_arr, tag_arr = _run_solve(
        c,
        n_total,
        b["feed_seg"],
        meas_freq_mhz,
        ground=b["ground"],
        feed_tag=b["feed_tag"],
        ground_fast=b["ground_fast"],
    )
    solve_ms = (time.perf_counter() - t0_clock) * 1e3

    feed_idx_in_tag = np.where(tag_arr == b["feed_tag"])[0]
    fed_global = feed_idx_in_tag[b["feed_seg"] - 1]
    z_in = complex(1.0 / cur_arr[fed_global])

    wires = []
    # Synthetic 3-knot feed record (T, midpoint, S) — gives the UI an
    # interior knot to anchor feed_knot_index on. NEC's per-segment
    # current goes onto the midpoint; the two end-knots are zero so the
    # render's open-wire convention holds.
    T, S = b["T"], b["S"]
    mid = tuple(0.5 * (a + s_) for a, s_ in zip(T, S))
    feed_cur = complex(cur_arr[fed_global])
    feed_knots = np.array([T, mid, S], dtype=float)
    feed_currents = np.array([0.0 + 0.0j, feed_cur, 0.0 + 0.0j], dtype=np.complex128)
    wires.append(
        {
            "label": "feed",
            "knot_positions": feed_knots.tolist(),
            "knot_currents_re": feed_currents.real.tolist(),
            "knot_currents_im": feed_currents.imag.tolist(),
        }
    )

    for i, (t_sr, t_sa, t_tr, t_ta) in enumerate(b["band_tags"]):
        label = _fandipole_band_label(i, b["band_freqs_mhz"], b["band_lengths_m"][i])
        path_pos = [S, b["A_pos"][i], b["B_pos"][i]]
        knots_pos = _polyline_knots(path_pos, [n_per, n_per])
        cur_pos = np.concatenate(
            [
                cur_arr[np.where(tag_arr == t_sr)[0]],
                cur_arr[np.where(tag_arr == t_sa)[0]],
            ]
        )
        knot_cur_pos = _segment_centers_to_knot_currents(
            cur_pos, knots_pos.shape[0], junction_at_start=True
        )
        wires.append(
            {
                "label": f"{label} +y",
                "knot_positions": knots_pos.tolist(),
                "knot_currents_re": knot_cur_pos.real.tolist(),
                "knot_currents_im": knot_cur_pos.imag.tolist(),
            }
        )
        path_neg = [T, b["A_neg"][i], b["B_neg"][i]]
        knots_neg = _polyline_knots(path_neg, [n_per, n_per])
        cur_neg = np.concatenate(
            [
                cur_arr[np.where(tag_arr == t_tr)[0]],
                cur_arr[np.where(tag_arr == t_ta)[0]],
            ]
        )
        knot_cur_neg = _segment_centers_to_knot_currents(
            cur_neg, knots_neg.shape[0], junction_at_start=True
        )
        wires.append(
            {
                "label": f"{label} -y",
                "knot_positions": knots_neg.tolist(),
                "knot_currents_re": knot_cur_neg.real.tolist(),
                "knot_currents_im": knot_cur_neg.imag.tolist(),
            }
        )

    return {
        "geometry": "fan_dipole",
        "wires": wires,
        "feed_wire_index": 0,
        "feed_knot_index": 1,  # midpoint of the synthetic 3-knot feed wire
        "z_in_re": float(z_in.real),
        "z_in_im": float(z_in.imag),
        "design_freq_mhz": b["design_freq_mhz"],
        "measurement_freq_mhz": meas_freq_mhz,
        "lambda_design_m": b["wavelength_design"],
        "n_bands": b["n_bands"],
        "band_lengths_m": list(b["band_lengths_m"]),
        "band_freqs_mhz": list(b["band_freqs_mhz"]),
        "slope": b["slope"],
        "cone_radius_m": b["cone_radius_m"],
        "t0_m": b["t0_m"],
        "solve_ms": solve_ms,
        "solver": "pynec",
        "ground": b["ground"],
        "ground_fast": b["ground_fast"],
        "height_m": b["z_offset"],
        "ground_eps_r": GROUND_DIELECTRIC,
        "ground_sigma": GROUND_CONDUCTIVITY,
    }


def solve(req: dict) -> dict:
    geometry = req.get("geometry", "inverted_v")
    if geometry == "fan_dipole":
        return solve_fandipole(req)
    ex = EXAMPLES.get(geometry) or EXAMPLES["inverted_v"]
    if ex.pynec_solve is None:
        raise ValueError(f"PyNEC solve not implemented for geometry {ex.name!r}")
    return ex.pynec_solve(req)


def pattern(req: dict) -> dict:
    """NEC's `rp_card`-computed gain pattern over the upper hemisphere.

    Returns a (n_theta × n_phi) gain grid in dBi at θ ∈ [0°, 90°], full φ.
    With ground off, the lower hemisphere is symmetric to the upper for the
    flat geometries supported here, but we only ship the upper half — the
    UI mirrors as needed.
    """
    geometry = req.get("geometry", "inverted_v")
    if geometry == "fan_dipole":
        b = _build_fandipole(req)
    else:
        ex = EXAMPLES.get(geometry) or EXAMPLES["inverted_v"]
        if ex.pynec_build is None:
            raise ValueError(
                f"PyNEC pattern not implemented for geometry {ex.name!r}"
            )
        b = ex.pynec_build(req)
    c = b["context"]
    feed_seg = b["feed_seg"]
    n_per_wire = b["n_per_wire"]
    feed_tag = b.get("feed_tag", 1)
    meas_freq_mhz = float(
        req.get("measurement_freq_mhz", req.get("design_freq_mhz", 14.3))
    )

    t0 = time.perf_counter()
    _run_solve(
        c,
        2 * n_per_wire,
        feed_seg,
        meas_freq_mhz,
        ground=b["ground"],
        feed_tag=feed_tag,
        ground_fast=b["ground_fast"],
    )

    # 2°×5° grid: 46 thetas (0..90), 73 phis (0..360 inclusive). At ~3.4k
    # rays this runs in tens of ms — fine for a debounced overlay request.
    n_theta = 46
    n_phi = 73
    del_theta = 90.0 / (n_theta - 1)
    del_phi = 360.0 / (n_phi - 1)
    c.rp_card(0, n_theta, n_phi, 0, 5, 0, 0, 0.0, 0.0, del_theta, del_phi, 0.0, 0.0)
    gains = [
        [float(c.get_gain(0, ti, pi)) for pi in range(n_phi)] for ti in range(n_theta)
    ]
    pattern_ms = (time.perf_counter() - t0) * 1e3

    return {
        "available": True,
        "geometry": geometry,
        "ground": b["ground"],
        "ground_fast": b["ground_fast"],
        "height_m": b["z_offset"],
        "measurement_freq_mhz": meas_freq_mhz,
        "theta_deg": [ti * del_theta for ti in range(n_theta)],
        "phi_deg": [pi * del_phi for pi in range(n_phi)],
        "gain_dbi": gains,
        "pattern_ms": pattern_ms,
    }


def _sweep_at(req: dict, freq_mhz: float) -> complex:
    """Single-frequency Z via PyNEC, used to build the swept Z array."""
    req2 = dict(req)
    req2["measurement_freq_mhz"] = freq_mhz
    res = solve(req2)
    return complex(res["z_in_re"], res["z_in_im"])


def sweep(req: dict, freqs_mhz: list[float]) -> tuple[list[float], list[float]]:
    """Loop-based sweep. PyNEC has no batched API, so we run one solve per
    frequency. At N=30 each solve is ~1.5 ms — 41 points * 1.5 ms = ~60 ms,
    fine for an interactive sweep."""
    z_re, z_im = [], []
    for f in freqs_mhz:
        z = _sweep_at(req, f)
        z_re.append(float(z.real))
        z_im.append(float(z.imag))
    return z_re, z_im
