"""momwire solve/validation backend for the antenna cross-check harness.

All geometries live in validation.examples — each registered antenna bundles
its momwire solve/sweep and pynec build/solve into one file. The `solve()`
dispatcher here looks the geometry up in EXAMPLES and calls its callables;
adding or removing an antenna doesn't touch this file. It also carries the
shared response-packing / directivity-normalisation helpers the per-geometry
example modules import.

The response shape is uniform across geometries — each wire is a sequence of
knots with per-knot complex currents and the feed lives on one of the wires.

This used to also host a FastAPI app (the interactive UI backend); that server
was retired and only the solver/validation code remains, consumed by the test
suite (tests/test_web_server.py, tests/test_pynec_backend.py).
"""

from __future__ import annotations

# Configure BLAS/OpenMP thread counts BEFORE numpy/scipy/PyNEC import — each
# library snapshots the env at its own import time and ignores later changes.
#
# OPENBLAS_NUM_THREADS=1: numpy/scipy bring their own OpenBLAS thread pool
#   that sits idle most of the request lifetime but contends with PyNEC's
#   MKL/OpenBLAS-LAPACKE pool for cores. vtune confirmed this was costing
#   ~8% wall time at NP=4 on the gather-scatter fill.
#
# OMP_NUM_THREADS / MKL_NUM_THREADS: with the gather-scatter matrix fill
#   (see PR #21) the per-source parallel-for inside cmset() and MKL/OpenBLAS'
#   zgetrf both want available cores. Default to the physical-core count
#   (not logical / HT count) — see _physical_cpu_count(); the FP-vector-
#   saturated quadrature inner loops gain nothing from HT siblings and
#   actually slow down ~15% from execution-unit contention on KBL-class
#   chips. An operator can override via the env to share with other
#   workloads on the same host.
#
# Older comment explaining why we used to pin everything to 1: the interactive
# workload is many small solves (≤ 250×250 dense complex matrices), and on
# the pre-gather-scatter code path thread orchestration costs dwarfed the
# per-call work — a 2-director live solve went from 220 ms (8 threads) to
# 67 ms (1 thread) on an 8-core box. That regression is no longer reproducible
# with the current build: matrix fill itself parallelizes, the OMP team is
# spawned once per cmset(), and OpenBLAS contention is removed by the
# OPENBLAS_NUM_THREADS=1 pin above.
import os


def _physical_cpu_count() -> int:
    """Number of physical cores (not logical / HT siblings).

    Our quadrature kernels are FP-vector-saturated (libmvec AVX2 sin/cos
    inner loops, no spare FU bandwidth), so two HT siblings on one physical
    core contend for execution units rather than overlap. Ad-hoc bench on
    KBL-R 4C/8T showed 4-thread runs ~15% faster than 8-thread runs of the
    swept-ground hot path. Pin to physical-core count to skip that loss.
    """
    try:
        cores = set()
        phys, coreid = None, None
        with open("/proc/cpuinfo") as f:
            for line in f:
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip()
                if key == "physical id":
                    phys = val
                elif key == "core id":
                    coreid = val
                elif not line.strip() and phys is not None and coreid is not None:
                    cores.add((phys, coreid))
                    phys, coreid = None, None
        if phys is not None and coreid is not None:
            cores.add((phys, coreid))
        if cores:
            return len(cores)
    except OSError:
        pass
    # Fallback: assume 2 HT siblings per core on x86. Wrong on chips without
    # HT, but in that case the caller can override via the env var.
    return max(1, (os.cpu_count() or 1) // 2)


_NPROC = str(_physical_cpu_count())
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", _NPROC)
os.environ.setdefault("MKL_NUM_THREADS", _NPROC)
# Between OMP parallel regions the workers default to busy-spinning on the
# team barrier (GOMP_SPINCOUNT ~300k, ~80 ms on KBL). Each momwire solve runs
# only ~1 ms of C++ kernel then ~10–20 ms of Python serial work (basis-coef
# build, sparse matmul, LU solve); the workers spin through all of that
# Python time on every solve. On the N=21 hentenna width-sweep harness
# (`scripts/vtune_hentenna_width_sweep.py`) VTune attributed ~63% of sin's
# CPU and ~32% of pynec's CPU to `libgomp` barrier-wait under this default.
# Make workers park immediately so the spin time goes away — wall-clock
# drops ~4× on both solvers at N=21 (sin 78 → 19 ms/step, pynec 67 → 9 ms).
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("GOMP_SPINCOUNT", "0")

# ruff: noqa: E402 — imports below must follow the env-var setup above so
# OpenBLAS picks up the thread count at its own import time.
import numpy as np

from momwire.bspline import BSplineSolver
from momwire.hmatrix import HMatrixSolver
from momwire.sinusoidal import SinusoidalSolver
from momwire.triangular import TriangularSolver

from . import pynec_backend
from .examples import REGISTRY as EXAMPLES


# Per-model option allowlist. Frontend sends `momwire_model` + `model_options`
# (a flat dict); we forward only the kwargs each class accepts, so an
# unrecognised option from a stale client never raises. Defaults match the
# class signatures so unset options behave identically to the old code.
_PYSIM_MODEL_KEYS = {
    "triangular": ("n_qp_reg", "n_qp_off"),
    "sinusoidal": ("n_qp_const",),
    "bspline": (
        "degree",
        "n_qp_pair",
        "n_qp_source",
        "feed_smoothing_factor",
        "use_singular_enrichment",
        "n_qp_sing",
        "enrichment_min_k",
        "enrichment_variant",
        "tikhonov_lambda",
        "auto_tap_ratio_threshold",
    ),
    # Hierarchical (H-matrix / ACA) accelerator: same B-spline basis as
    # bspline, plus the clustering / compression / iterative-solve knobs.
    "hmatrix": (
        "degree",
        "n_qp_pair",
        "n_qp_source",
        "feed_smoothing_factor",
        "aca_eta",
        "aca_leaf_size",
        "aca_tol",
        "solve_tol",
    ),
}
_PYSIM_MODELS = {
    "triangular": TriangularSolver,
    "sinusoidal": SinusoidalSolver,
    "bspline": BSplineSolver,
    "hmatrix": HMatrixSolver,
}


# hmatrix supports ground through its dense fallback path.
_PYSIM_MODELS_WITH_GROUND = {"triangular", "bspline", "sinusoidal", "hmatrix"}


def _make_momwire_sim(req: dict, **base_kwargs):
    """Instantiate the Momwire model the request selected.

    base_kwargs are the geometry-derived constructor kwargs every model
    accepts (wires, n_per_edge_per_wire, feed_*, wavelength, halfdriver_factor,
    nsegs, junctions). All three momwire models now accept ground_z; the set
    is kept as an allowlist so future models that don't support it can be
    excluded by name. model_options entries are filtered through the
    per-model allowlist.
    """
    model = req.get("momwire_model", "triangular")
    if model not in _PYSIM_MODELS:
        model = "triangular"
    cls = _PYSIM_MODELS[model]
    allowed = _PYSIM_MODEL_KEYS[model]

    opts = req.get("model_options") or {}
    extra = {k: opts[k] for k in allowed if k in opts}

    if model not in _PYSIM_MODELS_WITH_GROUND:
        base_kwargs.pop("ground_z", None)

    return cls(**base_kwargs, **extra)


C_LIGHT = 299_792_458.0  # m/s, matches TriangularSolver's eps*mu derivation to ~1e-9
_EPS0 = 8.854187817e-12  # F/m


def _attach_derived_em_fields(out: dict) -> None:
    """Augment the solve response with frequency-derived EM scalars the
    frontend would otherwise compute from raw physics constants.

    Sets:
      - `k_meas_m_inv`: wavenumber 2π f / c at measurement freq (rad/m)
      - `ground_eps_im`: imaginary part of the complex relative permittivity
        of the ground, -σ / (ω ε₀); 0 when ground is off or σ=0.

    The frontend reads these directly so it doesn't need to carry C_LIGHT
    or ε₀ literals. `lambda_design_m` is already shipped by each example.
    """
    f_hz = float(out["measurement_freq_mhz"]) * 1e6
    omega = 2.0 * np.pi * f_hz
    out["k_meas_m_inv"] = omega / C_LIGHT
    sigma = float(out.get("ground_sigma", 0.0) or 0.0)
    out["ground_eps_im"] = -sigma / (omega * _EPS0) if omega > 0 else 0.0


def _compute_directivity_norm(out: dict, n_theta: int = 45, n_phi: int = 90) -> None:
    """Attach `directivity_norm` = 4π / ∫|M_perp|² dΩ to the response.

    Multiplying this by the frontend's azimuth-cut |M_perp(π/2, φ)|² yields
    absolute directivity D(φ) (linear); 10·log10(D) is dBi.

    With ground enabled, integrates only the upper hemisphere and adds the
    Fresnel-reflected contribution from the geometric image so the
    normalization matches what the JS far-field code displays.
    """
    k = float(out["k_meas_m_inv"])
    ground_on = bool(out.get("ground", False))

    mids, drs, i_mids = [], [], []
    for w in out["wires"]:
        # Prefer the finer-grained sample arrays (knot + segment-midpoint)
        # when the model produced them, so non-tent bases get their intra-
        # segment curvature integrated. Falls back to knot arrays for any
        # backend that only ships knot data (PyNEC).
        if "sample_positions" in w:
            pts = np.asarray(w["sample_positions"], dtype=np.float64)
            cur = np.asarray(
                w["sample_currents_re"], dtype=np.float64
            ) + 1j * np.asarray(w["sample_currents_im"], dtype=np.float64)
        else:
            pts = np.asarray(w["knot_positions"], dtype=np.float64)
            cur = np.asarray(w["knot_currents_re"], dtype=np.float64) + 1j * np.asarray(
                w["knot_currents_im"], dtype=np.float64
            )
        drs.append(pts[1:] - pts[:-1])
        mids.append(0.5 * (pts[1:] + pts[:-1]))
        i_mids.append(0.5 * (cur[1:] + cur[:-1]))
    mid = np.concatenate(mids, axis=0)  # (Nseg, 3)
    dr = np.concatenate(drs, axis=0)  # (Nseg, 3)
    i_mid = np.concatenate(i_mids, axis=0)  # (Nseg,)

    # Cell-centered grid. With ground, sample only the upper hemisphere so
    # the integral is over the half-space the antenna actually radiates into.
    if ground_on:
        theta = (np.arange(n_theta) + 0.5) * (np.pi / 2 / n_theta)
        dtheta = np.pi / 2 / n_theta
    else:
        theta = (np.arange(n_theta) + 0.5) * (np.pi / n_theta)
        dtheta = np.pi / n_theta
    phi = np.arange(n_phi) * (2 * np.pi / n_phi)
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    cos_p, sin_p = np.cos(phi), np.sin(phi)

    rx = sin_t[:, None] * cos_p[None, :]
    ry = sin_t[:, None] * sin_p[None, :]
    rz = np.broadcast_to(cos_t[:, None], (n_theta, n_phi))
    rhat = np.stack([rx, ry, rz], axis=-1)  # (nθ, nφ, 3)

    phase = k * np.einsum("ijc,nc->ijn", rhat, mid)  # (nθ, nφ, Nseg)
    expp = np.exp(1j * phase)
    weighted = i_mid[:, None] * dr  # (Nseg, 3)
    M = np.einsum("ijn,nc->ijc", expp, weighted)  # (nθ, nφ, 3)
    m_dot_r = np.sum(M * rhat, axis=-1)
    M_perp = M - m_dot_r[..., None] * rhat

    if ground_on:
        # PEC-image method, then Fresnel-correct the reflected wave per-ray.
        # Image current: horizontal components flipped, vertical preserved.
        # This reproduces PEC reflection when ρ_h=-1, ρ_v=+1, and lets us
        # apply the actual finite-ground coefficients to that same image.
        mid_img = mid * np.array([1.0, 1.0, -1.0])
        dr_img = dr * np.array([-1.0, -1.0, 1.0])
        weighted_img = i_mid[:, None] * dr_img
        phase_img = k * np.einsum("ijc,nc->ijn", rhat, mid_img)
        expp_img = np.exp(1j * phase_img)
        M_img = np.einsum("ijn,nc->ijc", expp_img, weighted_img)
        m_img_dot_r = np.sum(M_img * rhat, axis=-1)
        M_img_perp = M_img - m_img_dot_r[..., None] * rhat

        # Polarization basis at each ray: ĥ = ẑ × r̂ (perp to plane of
        # incidence), v̂ = r̂ × ĥ (in plane of incidence, perp to r̂).
        s = np.sqrt(rx * rx + ry * ry)
        s_safe = np.where(s > 1e-12, s, 1.0)
        h_hat = np.stack([-ry / s_safe, rx / s_safe, np.zeros_like(rx)], axis=-1)
        v_hat = np.stack([-rx * rz / s_safe, -ry * rz / s_safe, s], axis=-1)

        M_img_h = np.sum(M_img_perp * h_hat, axis=-1)  # complex (nθ, nφ)
        M_img_v = np.sum(M_img_perp * v_hat, axis=-1)

        eps_c = out["ground_eps_r"] + 1j * out["ground_eps_im"]
        cos_ti = rz
        sin2_ti = s * s
        Q = np.sqrt(eps_c - sin2_ti)
        rho_h = (cos_ti - Q) / (cos_ti + Q)
        rho_v = (eps_c * cos_ti - Q) / (eps_c * cos_ti + Q)

        # Reflected: ρ_v on the v-pol component, −ρ_h on the h-pol component
        # (the minus sign folds the PEC image's pre-applied horizontal flip
        # back out so ρ_h=−1 recovers the PEC limit exactly).
        M_refl = (rho_v * M_img_v)[..., None] * v_hat - (rho_h * M_img_h)[
            ..., None
        ] * h_hat
        M_perp = M_perp + M_refl

    mag2 = np.sum((M_perp.real**2 + M_perp.imag**2), axis=-1)  # (nθ, nφ)

    dphi = 2 * np.pi / n_phi
    p_rad = float(np.sum(mag2 * sin_t[:, None]) * dtheta * dphi)
    out["directivity_norm"] = (4 * np.pi / p_rad) if p_rad > 0 else 0.0


def _wire_record(
    knots: np.ndarray,
    currents: np.ndarray,
    label: str,
    sample_currents: np.ndarray | None = None,
) -> dict:
    """Package one wire's record for the JSON response. `currents` is a
    length-M_w complex array (one per mesh knot) as produced by each
    model's `currents_at_knots(coeffs)` method.

    When `sample_currents` is provided, additional `sample_positions` /
    `sample_currents_re` / `sample_currents_im` arrays are attached at
    knots-and-midpoints interleaved (2*N_seg + 1 entries per wire). This is
    what `_compute_directivity_norm` and the frontend renderers consume to
    resolve intra-segment basis curvature (B-spline d=2, sinusoidal three-
    term) and the B-spline enrichment shape that vanishes at every knot.
    """
    currents = np.asarray(currents, dtype=np.complex128)
    if currents.shape[0] != knots.shape[0]:
        raise ValueError(
            f"_wire_record: currents/knots length mismatch "
            f"({currents.shape[0]} vs {knots.shape[0]})"
        )
    out = {
        "label": label,
        "knot_positions": knots.tolist(),
        "knot_currents_re": currents.real.tolist(),
        "knot_currents_im": currents.imag.tolist(),
    }
    if sample_currents is not None:
        sample_currents = np.asarray(sample_currents, dtype=np.complex128)
        n_seg = knots.shape[0] - 1
        expected = 2 * n_seg + 1
        if sample_currents.shape[0] != expected:
            raise ValueError(
                f"_wire_record: sample_currents length {sample_currents.shape[0]} "
                f"!= expected 2*N_seg+1 = {expected}"
            )
        sample_positions = np.empty((expected, 3), dtype=np.float64)
        sample_positions[0::2] = knots
        sample_positions[1::2] = 0.5 * (knots[:-1] + knots[1:])
        out["sample_positions"] = sample_positions.tolist()
        out["sample_currents_re"] = sample_currents.real.tolist()
        out["sample_currents_im"] = sample_currents.imag.tolist()
    return out


def _sample_arc_for_wire(knots: np.ndarray) -> np.ndarray:
    """Build interleaved (knot_arc, midpoint_arc, knot_arc, ...) array from a
    wire's 3D knot positions. Segment lengths come from successive-knot
    distances along the polyline.
    """
    knots = np.asarray(knots, dtype=np.float64)
    h_seg = np.linalg.norm(knots[1:] - knots[:-1], axis=1)
    arc_at_knot = np.concatenate([[0.0], np.cumsum(h_seg)])
    mid_arc = 0.5 * (arc_at_knot[:-1] + arc_at_knot[1:])
    sample_arc = np.empty(2 * h_seg.shape[0] + 1, dtype=np.float64)
    sample_arc[0::2] = arc_at_knot
    sample_arc[1::2] = mid_arc
    return sample_arc


def _pack_momwire_wires(sim, coeffs, knot_arrays, labels) -> list[dict]:
    """Build wire records for every momwire wire with both knot-level currents
    AND finer-grained mid-segment samples (one extra sample per segment).

    Calls `sim.currents_at_knots(coeffs)` once for the knot values and once
    more with an `s_array` of per-wire interleaved knot-and-midpoint arcs.
    The model's basis is then evaluated exactly at the midpoints — including
    the B-spline enrichment basis Φ_sing, which is zero at the knots but
    non-zero in the interior.
    """
    sample_arcs = [_sample_arc_for_wire(k) for k in knot_arrays]
    knot_currents = sim.currents_at_knots(coeffs)
    sample_currents = sim.currents_at_knots(coeffs, s_array=sample_arcs)
    return [
        _wire_record(
            np.asarray(knot_arrays[i]),
            knot_currents[i],
            labels[i],
            sample_currents=sample_currents[i],
        )
        for i in range(len(knot_arrays))
    ]


# Momwire PEC ground: pass these to the response so the frontend's Fresnel
# far-field code treats the surface as a perfect electric conductor
# (ρ_h → −1, ρ_v → +1 in the eps_r → ∞ limit).
_PEC_GROUND_EPS_R = 1.0e10
_PEC_GROUND_SIGMA = 0.0


def _read_ground(req: dict) -> tuple[bool, float, float]:
    """Common request parsing: returns (ground_on, height_m, z_offset).

    height_m is the antenna height above ground when ground_on=True; z_offset
    is what each geometry helper adds to its native (z=0) coordinates.
    """
    ground_on = bool(req.get("ground", False))
    height_m = float(req.get("height_m", 0.0))
    z_offset = height_m if ground_on else 0.0
    return ground_on, height_m, z_offset


def _polyline_knots(polyline: np.ndarray, npe_list: list[int]) -> np.ndarray:
    """Concatenated per-edge knot positions, with shared corners deduped."""
    parts = []
    for i, n_e in enumerate(npe_list):
        seg = np.linspace(polyline[i], polyline[i + 1], n_e + 1)
        parts.append(seg if i == 0 else seg[1:])
    return np.vstack(parts)


def solve(req: dict) -> dict:
    geometry = req.get("geometry", "inverted_v")
    use_pynec = req.get("solver") == "pynec" and pynec_backend.HAVE_PYNEC
    if use_pynec:
        out = pynec_backend.solve(req)
        _attach_derived_em_fields(out)
        _compute_directivity_norm(out)
        return out
    ex = EXAMPLES.get(geometry) or EXAMPLES["inverted_v"]
    out = ex.momwire_solve(req)
    out["solver"] = "momwire"
    _attach_derived_em_fields(out)
    _compute_directivity_norm(out)
    return out
