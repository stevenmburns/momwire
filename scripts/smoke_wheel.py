"""Smoke a BUILT momwire wheel in a fresh interpreter — momwire#1014.

Run this against an installed wheel, not a source checkout:

    pip install wheelhouse/momwire-*-cp314-cp314-win_amd64.whl
    python scripts/smoke_wheel.py

What it checks, and why each one is here rather than assumed:

  * **the import came from the WHEEL.** This script is run from inside the
    repository, where `src/momwire/` exists and can shadow the installed
    package depending on cwd and sys.path. A smoke that imported the checkout
    would pass on a wheel that cannot even be unpacked. So it asserts the
    module resolved into site-packages and NOT into this tree — the whole
    point is to exercise what a user gets.
  * **the accelerator loaded.** `momwire.accelerated` is False whenever the
    extension fails to import, and `setup.py` deliberately falls back to pure
    Python rather than failing, so a wheel with a broken or unvendored runtime
    imports fine and is silently 50x slower. On Windows that means the
    delvewheel-vendored libomp140/msvcp140; on macOS, loading against
    Homebrew's libomp without a private copy.
  * **it solves.** Importing proves the linker was satisfied; it does not prove
    the kernels run. One tiny fill exercises the accelerated path end to end.

Deliberately has NO test dependencies (no pytest, no scikit-rf) and reads no
repository fixtures: it must run in the same bare environment a user has.
"""

from __future__ import annotations

import pathlib
import sys


def main() -> int:
    import momwire

    here = pathlib.Path(__file__).resolve().parent.parent
    mod = pathlib.Path(momwire.__file__).resolve()
    print(f"python      {sys.version.split()[0]} ({sys.implementation.name})")
    try:
        from importlib.metadata import version as _dist_version

        dist = _dist_version("momwire")
    except Exception:  # noqa: BLE001 — a version we cannot read is not a failure
        dist = "?"
    print(f"momwire     {dist}")
    print(f"imported    {mod}")
    print(f"accelerated {momwire.accelerated}")

    if here in mod.parents:
        raise SystemExit(
            f"FAIL: imported the source checkout at {mod}, not an installed "
            f"wheel. This smoke is meaningless unless it exercises the built "
            f"artefact; run it from outside {here} or with the checkout off "
            f"sys.path."
        )
    if not momwire.accelerated:
        raise SystemExit(
            "FAIL: momwire.accelerated is False — the C++ extension did not "
            "load. The wheel imports (setup.py falls back to pure Python by "
            "design), so this is exactly the failure that stays silent: a "
            "vendored OpenMP/MSVC runtime is missing or the ABI does not match "
            "this interpreter."
        )

    import numpy as np

    from momwire.bspline import BSplineSolver

    # A bare half-wave dipole, coarse. Small enough to be instant on any
    # runner; real enough that a broken kernel cannot return a finite number
    # by accident.
    hd = 0.962 * 22 / 4
    wire = np.array([[0.0, -hd, 0.0], [0.0, hd, 0.0]])
    z, currents = BSplineSolver(
        wires=[wire], n_per_edge_per_wire=[[8]], nsegs=8, degree=1
    ).compute_impedance()

    print(f"solve       Z = {z.real:.3f} {z.imag:+.3f}j  ({currents.size} dof)")
    if not (np.isfinite(z.real) and np.isfinite(z.imag)):
        raise SystemExit(f"FAIL: impedance is not finite: {z!r}")
    # A resonant half-wave dipole is ~70 ohm resistive. The band is wide on
    # purpose -- this is a "the kernels ran" check, not an accuracy gate, and
    # a coarse mesh at degree 1 is legitimately off.
    if not (20.0 < z.real < 200.0):
        raise SystemExit(f"FAIL: Re(Z) = {z.real} is not a plausible dipole")

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
