"""momwire sets the thread-pool wait variables before NumPy loads.

The block at the top of `momwire/__init__.py` must run before the package's
first import of NumPy or the accelerator (the pools read their environment
once, at library load), so this reads the variables at the moment NumPy is
first imported, in a fresh interpreter.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

_KEYS = ("OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "OPENBLAS_THREAD_TIMEOUT")

_PROBE = """
import builtins, json, os
seen = {}
_real = builtins.__import__
def _hook(name, *a, **k):
    if name.split(".")[0] == "numpy" and "numpy" not in seen:
        seen["numpy"] = {k: os.environ.get(k) for k in %r}
    return _real(name, *a, **k)
builtins.__import__ = _hook
import momwire.%s
print(json.dumps(seen.get("numpy")))
"""


def _env_at_numpy_import(module: str = "bspline", **set_vars) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in _KEYS}
    env.update(set_vars)
    out = subprocess.run(
        [sys.executable, "-c", _PROBE % (_KEYS, module)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_wait_variables_are_set_before_numpy_loads():
    assert _env_at_numpy_import() == {
        "OMP_WAIT_POLICY": "PASSIVE",
        "GOMP_SPINCOUNT": "0",
        "OPENBLAS_THREAD_TIMEOUT": "1",
    }


def test_the_simnec_portal_import_path_gets_them_too():
    assert _env_at_numpy_import("portal")["OPENBLAS_THREAD_TIMEOUT"] == "1"


def test_a_callers_own_value_is_kept():
    seen = _env_at_numpy_import(OPENBLAS_THREAD_TIMEOUT="7")
    assert seen["OPENBLAS_THREAD_TIMEOUT"] == "7"
    assert seen["OMP_WAIT_POLICY"] == "PASSIVE"
