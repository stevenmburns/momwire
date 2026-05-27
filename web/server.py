"""FastAPI server exposing BentTriangularPySim over a WebSocket.

The frontend opens /ws, sends {"angle_deg": ..., "n_per_arm": ...} messages,
and receives the per-knot current vector + feedpoint impedance.

Run: uvicorn web.server:app --reload
"""
from __future__ import annotations

import json
import time

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from pysim.triangular_bent import BentTriangularPySim


app = FastAPI(title="pysim interactive")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _inverted_v_polyline(arm_len: float, angle_deg: float) -> np.ndarray:
    """Inverted-V with apex at the origin and arms drooping in the XZ plane.

    angle_deg is the droop of each arm from horizontal: 0 = flat dipole,
    larger = more closed V. Capped below 90 to avoid arm coincidence.
    """
    alpha = np.deg2rad(angle_deg)
    cos_a, sin_a = float(np.cos(alpha)), float(np.sin(alpha))
    left = np.array([-arm_len * cos_a, 0.0, -arm_len * sin_a])
    apex = np.array([0.0, 0.0, 0.0])
    right = np.array([arm_len * cos_a, 0.0, -arm_len * sin_a])
    return np.vstack([left, apex, right])


def solve(
    *,
    angle_deg: float,
    n_per_arm: int,
    wavelength: float,
    halfdriver_factor: float,
    wire_radius: float,
) -> dict:
    sim = BentTriangularPySim(
        wavelength=wavelength,
        halfdriver_factor=halfdriver_factor,
        nsegs=n_per_arm,
    )
    sim.wire_radius = wire_radius
    polyline = _inverted_v_polyline(sim.halfdriver, angle_deg)
    sim.polyline = polyline
    sim.n_per_edge = [n_per_arm, n_per_arm]

    t0 = time.perf_counter()
    z_in, coeffs = sim.compute_impedance()
    solve_ms = (time.perf_counter() - t0) * 1e3

    # Build knot positions along the polyline and pad coefficients with zeros
    # at the two endpoints (open-wire boundary condition for tent basis).
    seg_l = np.vstack([
        np.linspace(polyline[0], polyline[1], n_per_arm + 1)[:-1],
        np.linspace(polyline[1], polyline[2], n_per_arm + 1)[:-1],
    ])
    seg_r = np.vstack([
        np.linspace(polyline[0], polyline[1], n_per_arm + 1)[1:],
        np.linspace(polyline[1], polyline[2], n_per_arm + 1)[1:],
    ])
    knots = np.vstack([seg_l, seg_r[-1:]])
    n_knots = knots.shape[0]
    assert n_knots == 2 * n_per_arm + 1

    knot_currents = np.zeros(n_knots, dtype=np.complex128)
    knot_currents[1:-1] = coeffs

    # Feed knot: by default the solver picks the interior knot closest to
    # midpoint arclength, which is the apex (index n_per_arm in knot list).
    feed_knot_index = n_per_arm

    return {
        "knot_positions": knots.tolist(),
        "knot_currents_re": knot_currents.real.tolist(),
        "knot_currents_im": knot_currents.imag.tolist(),
        "feed_knot_index": feed_knot_index,
        "z_in_re": float(z_in.real),
        "z_in_im": float(z_in.imag),
        "angle_deg": angle_deg,
        "n_per_arm": n_per_arm,
        "solve_ms": solve_ms,
    }


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            req = json.loads(raw)
            result = solve(
                angle_deg=float(req.get("angle_deg", 30.0)),
                n_per_arm=int(req.get("n_per_arm", 40)),
                wavelength=float(req.get("wavelength", 22.0)),
                halfdriver_factor=float(req.get("halfdriver_factor", 0.962)),
                wire_radius=float(req.get("wire_radius", 0.0005)),
            )
            await ws.send_text(json.dumps(result))
    except WebSocketDisconnect:
        return
