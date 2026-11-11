"""How the threaded Φ → Z reduction (momwire#1905) is held against scipy's.

On x86-64 the kernel is scipy's arithmetic to the bit, so the gates compare
bytes. On other machines (macOS arm64) scipy's own build contracts its
complex multiply-add to FMA while the kernel, compiled with
-ffp-contract=off, does not, so the two differ at round-off: there the
gates hold a tolerance and REPORT the measured drift as a `PhiReduceDrift`
warning, which the CI log's warnings summary carries (Steve, 2026-11-11:
measure on the change, then decide whether the drift is usable).
"""

from __future__ import annotations

import platform
import warnings

import numpy as np

EXACT = platform.machine().lower() in ("x86_64", "amd64")

#: Provisional bound on the fill's element drift, max|got - ref| / max|ref|,
#: off x86-64. The measured value is what the warning reports.
TOL_FILL = 1e-12
#: The same for an impedance, relative.
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
