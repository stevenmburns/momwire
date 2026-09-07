"""momwire#887 — reproduce the fine-mesh instability, then bisect it.

The issue reports that a COATED deck (a'+L loading, eps_r >= 9) moves its
conductance peak by ~11 % between lambda/240 and lambda/320 while the BARE
control on the same geometry stays flat to 0.001 over four rungs. Its stated
suspicion is the equivalent radius: a' replaces the KERNEL radius, so the fill
sees Delta/a' rather than Delta/a, and at a'/a = 1.85 the thin-wire ratio
degrades ~2x faster with refinement.

That suspicion is testable directly, because a'+L is a PAIR of independent
edits to the deck and each half can be applied alone:

    bare      radius a,  no L        the control
    a-only    radius a', no L        the kernel half, alone
    L-only    radius a,  with L      the inductance half, alone
    a'+L      radius a', with L      what a coated deck actually solves

`L-only` is not reachable through the constructor -- `insulation_radius`
drives BOTH halves -- so it is built by constructing the coated solver and
putting `_radius_per_wire` back to a afterwards. `a-only` is the reverse: pass
a' as `wire_radius` with no jacket kwargs at all.

The four arms MUST differ in the deck before any output is worth reading, so
`_describe` prints each arm's kernel radius and its per-metre L and the script
refuses to go on if two arms agree.

Geometry is the issue's: Lamensdorf 1967, a = 3.175 mm, b = 6.35 mm (b/a = 2),
600 MHz, the monopole read as its image dipole. The observable is the location
of the conductance peak in h/lambda0, found by sweeping h at a FIXED segment
length per rung -- refining the mesh means shortening the segment, not holding
the segment count.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from momwire._wire_loading import equivalent_radius, insulation_inductance  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.sinusoidal import SinusoidalSolver  # noqa: E402

C0 = 299792458.0
FREQ = 600e6
LAM = C0 / FREQ
A = 3.175e-3
B = 6.35e-3

MU0 = 1.25663706127e-6


def _arms(eps_r):
    """name -> (wire_radius passed to the constructor, jacket kwargs, restore_a)."""
    a_eq = equivalent_radius(A, B, eps_r)
    return {
        "bare": (A, {}, False),
        "a-only": (a_eq, {}, False),
        "L-only": (A, {"insulation_radius": B, "insulation_eps_r": eps_r}, True),
        "a'+L": (A, {"insulation_radius": B, "insulation_eps_r": eps_r}, False),
    }


def _build(cls, arm, h, n_seg, eps_r):
    radius, jacket, restore = _arms(eps_r)[arm]
    # Monopole of height h read as its image dipole: a wire from -h to +h fed
    # at the centre, in free space.
    wire = np.array([(0.0, 0.0, -h), (0.0, 0.0, h)])
    s = cls(
        wires=[wire],
        n_per_edge_per_wire=[[n_seg]],
        wavelength=LAM,
        wire_radius=radius,
        feeds=[(0, h, 1 + 0j)],
        **jacket,
    )
    if restore:
        s._radius_per_wire = np.full_like(s._radius_per_wire, A)
        s._uniform_radius = float(A)
    return s


def _loading_l(s):
    """Per-metre insulation inductance this solver will actually apply, or 0."""
    if getattr(s, "insulation_radius", None) is None:
        return 0.0
    return float(
        insulation_inductance(
            s._conductor_radius_per_wire[0],
            s.insulation_radius[0],
            s.insulation_eps_r[0],
        )
    )


def _describe(cls, eps_r):
    """Print each arm's (kernel radius, per-metre L) and refuse duplicates."""
    print(f"  arm table for {cls.__name__}, eps_r = {eps_r}:")
    seen = {}
    for arm in _arms(eps_r):
        s = _build(cls, arm, 0.1, 40, eps_r)
        key = (round(float(s._radius_per_wire[0]), 15), round(_loading_l(s), 24))
        print(
            f"    {arm:8s} kernel a = {s._radius_per_wire[0] * 1e3:.6f} mm   "
            f"L' = {_loading_l(s) * 1e9:.6f} nH/m"
        )
        if key in seen:
            raise SystemExit(
                f"ARMS DO NOT DIFFER: {arm!r} and {seen[key]!r} build the same "
                f"deck. Nothing below this line would mean anything."
            )
        seen[key] = arm
    print()


def _g_peak(cls, arm, seg_len, eps_r, h_lo, h_hi, n_h):
    """h/lambda at which the input conductance peaks, at a FIXED segment length."""
    best = (-np.inf, None)
    for h in np.linspace(h_lo, h_hi, n_h):
        n_seg = max(4, int(round(2.0 * h / seg_len)))
        if n_seg % 2:  # keep the feed on a node/centre
            n_seg += 1
        s = _build(cls, arm, h, n_seg, eps_r)
        z = complex(s.compute_impedance()[0])
        g = (1.0 / z).real
        if g > best[0]:
            best = (g, h / LAM)
    return best[1]


def main():
    eps_r = float(sys.argv[1]) if len(sys.argv) > 1 else 9.0
    rungs = [120, 160, 240, 320]
    print(
        f"momwire#887 — coated fine-mesh study, eps_r = {eps_r}, {FREQ / 1e6:.0f} MHz"
    )
    print(
        f"lambda = {LAM * 1e3:.3f} mm, a = {A * 1e3} mm, b = {B * 1e3} mm, "
        f"a'/a = {equivalent_radius(A, B, eps_r) / A:.4f}\n"
    )

    for cls in (SinusoidalSolver, BSplineSolver):
        _describe(cls, eps_r)
        print(f"  conductance-peak h/lambda, {cls.__name__}:")
        print("    arm      " + "".join(f"  lam/{r:<5d}" for r in rungs))
        for arm in _arms(eps_r):
            row = []
            for r in rungs:
                row.append(_g_peak(cls, arm, LAM / r, eps_r, 0.06, 0.16, 101))
            drift = max(row) - min(row)
            print(
                f"    {arm:8s} "
                + "".join(f"  {v:.4f}   " for v in row)
                + f"   drift {drift:.4f}"
            )
        print()


if __name__ == "__main__":
    main()
