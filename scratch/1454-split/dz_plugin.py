"""pytest plugin: max |dZ|/|Z| between the series and the direct rule over
whatever exact-kernel corrections a test run performs (momwire#1454 step 3).

    PYTHONPATH=scratch/1454-split DZ_OUT=/tmp/dz pytest -p dz_plugin tests/ -m slow ...

Every correction razor runs during the session is applied both ways, as
`dz.py` does, and each worker appends one JSON line to ``$DZ_OUT.<pid>`` at
exit with its worst ratios and its counts. The tests see the direct rule's
matrix, so their own verdicts are untouched.
"""

import json
import os

import numpy as np

STATE = {"corrections": 0, "dz": 0.0, "dz_el": 0.0, "where": ""}


def pytest_configure(config):
    from momwire import RazorSolver, _exact_kernel_series as ES

    orig = RazorSolver._add_exact_kernel_correction

    def both(self, Z, geom, k, omega):
        Z0 = Z.copy(order="F")
        saved = ES.ENABLED
        try:
            ES.ENABLED = False
            orig(self, Z, geom, k, omega)
            ES.ENABLED = True
            orig(self, Z0, geom, k, omega)
        finally:
            ES.ENABLED = saved
        d = np.abs(Z0 - Z)
        dz = float(d.max() / np.abs(Z).max())
        nz = np.abs(Z) > 0
        STATE["corrections"] += 1
        STATE["dz_el"] = max(STATE["dz_el"], float((d[nz] / np.abs(Z)[nz]).max()))
        if dz >= STATE["dz"]:
            STATE["dz"] = dz
            STATE["where"] = os.environ.get("PYTEST_CURRENT_TEST", "")

    RazorSolver._add_exact_kernel_correction = both


def pytest_unconfigure(config):
    out = os.environ.get("DZ_OUT")
    if out and STATE["corrections"]:
        with open(f"{out}.{os.getpid()}", "w") as fh:
            fh.write(json.dumps(STATE) + "\n")
