"""How the threaded Φ → Z reduction (momwire#1905) is held against scipy's.

On x86-64 the kernel is scipy's arithmetic to the bit, so the gates compare
bytes. On other machines (macOS arm64) scipy's own build contracts its
complex multiply-add to FMA while the kernel, compiled with
-ffp-contract=off, does not, so the two differ at round-off: there the
gates hold a tolerance and REPORT the measured drift as a `PhiReduceDrift`
warning, which the CI log's warnings summary carries (Steve, 2026-11-11:
measured on the change, judged usable).
"""

from __future__ import annotations

import platform
import warnings

import numpy as np

EXACT = platform.machine().lower() in ("x86_64", "amd64")

# Measured on the macOS arm64 lane (PR #1909, 2026-11-11): free space 0
# (bit-identical); the buried inverted-L's mixed fill 1.5e-13 of max|Z|
# (worst single element 7.8e-12 relative, the cancellation-heavy ground
# terms) and its Zin 1.0e-11 relative; synthetic CSC operands <= 2.3e-16.
# Accepted by Steve 2026-11-11 as usable; the bounds below keep ~7x and ~10x
# headroom over those, so a regression that moves them shows.
#: Bound on the fill's drift, max|got - ref| / max|ref|, off x86-64.
TOL_FILL = 1e-12
#: Bound on an impedance's relative drift, off x86-64.
TOL_Z = 1e-10


class PhiReduceDrift(UserWarning):
    """The measured kernel-vs-scipy drift on a machine where it is not 0."""


def matches(got, ref, label, tol=TOL_FILL):
    """Bytes equal on x86-64; elsewhere within `tol`, reporting the drift."""
    got, ref = np.ascontiguousarray(got), np.ascontiguousarray(ref)
    if got.shape != ref.shape or got.dtype != ref.dtype:
        return False
    if EXACT:
        return got.tobytes() == ref.tobytes()
    scale = float(np.max(np.abs(ref))) or 1.0
    d = float(np.max(np.abs(got - ref))) / scale
    nz = np.abs(ref) > 0
    elem = float(np.max(np.abs(got - ref)[nz] / np.abs(ref)[nz])) if nz.any() else 0.0
    warnings.warn(
        f"{label} on {platform.machine()}: max|d|/max|ref| = {d:.3e}, "
        f"worst elementwise relative = {elem:.3e}",
        PhiReduceDrift,
        stacklevel=2,
    )
    return d <= tol
