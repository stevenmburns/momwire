"""The plain ``momwire-nec2c*`` launchers pin AVX-512 off before NumPy.

Every other entry point momwire owns pins (``tests/test_avx512_off.py``): the
thin clients at the top of their ``main``, the pool's workers, the standalone
exe. The plain launchers have no client in front of them, and their console
script used to be ``momwire.dropin.nec2c:main``, whose import of the
``momwire`` package reads NumPy in before any line of ``main`` runs. So a pin
inside ``main`` would have been too late, and the script had none at all. The
entry is now :func:`momwire_serve_client.stock_main`, which pins first.

The check is made in a fresh interpreter that fakes an AVX-512 CPU (this box
may have none) and records the environment at the instant NumPy is first
imported.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import momwire_serve_client as mech

ROOT = Path(__file__).resolve().parent.parent

_SCRIPT = r"""
import json, os, sys
import momwire_serve_client as m

m.cpu_has_avx512 = lambda: True  # pretend this CPU has AVX-512
seen = {}

class Spy:
    def find_spec(self, name, path=None, target=None):
        if name == "numpy" and "env" not in seen:
            seen["env"] = [
                os.environ.get("NPY_DISABLE_CPU_FEATURES"),
                os.environ.get("OPENBLAS_CORETYPE"),
            ]
        return None

sys.meta_path.insert(0, Spy())
sys.argv[0] = "momwire-nec2c-sinusoidal"
if sys.argv[1] == "entry":
    rc = m.stock_main(["-version"])
else:  # the old target, for the control
    from momwire.dropin.nec2c import main
    rc = main(["-version"])
print(json.dumps({"rc": rc, "env": seen.get("env")}), file=sys.stderr)
"""


def _run(which: str) -> dict:
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in (mech.NPY_DISABLE_ENV, mech.NPY_ENABLE_ENV, mech.OPENBLAS_CORETYPE_ENV)
    }
    done = subprocess.run(
        [sys.executable, "-c", _SCRIPT, which],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stderr.strip().splitlines()[-1])


def test_the_entry_pins_before_numpy_is_imported():
    seen = _run("entry")
    assert seen["rc"] == 0
    disable, coretype = seen["env"]
    assert disable, "NumPy was imported before the AVX-512 pin"
    assert "AVX512" in disable or "X86_V4" in disable
    assert coretype == mech.OPENBLAS_CORETYPE


def test_the_old_entry_could_not_have_pinned_in_time():
    """The control that makes the test above mean something: reached through
    the package's own ``main``, NumPy is already in when ``main`` begins, so
    nothing it set could reach NumPy. If this ever pins, the premise (and the
    reason for a separate entry) is gone and the entry can be put back."""
    disable, coretype = _run("package")["env"]
    assert disable is None and coretype is None


def _scripts() -> dict[str, str]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]


def test_every_plain_launcher_goes_through_the_pin():
    plain = {
        name: target
        for name, target in _scripts().items()
        if name.startswith("momwire-nec2c")
        and name != "momwire-nec2c-shared"
        and target != "momwire_nec2c_client:main"
    }
    assert plain, "no plain launcher found"
    assert {"momwire-nec2c", "momwire-nec2c-sinusoidal"} <= plain.keys()
    assert set(plain.values()) <= {"momwire_serve_client:stock_main"}
    # Nothing is left on the unpinned target.
    assert "momwire.dropin.nec2c:main" not in _scripts().values()


def test_the_entry_is_the_portals_main(monkeypatch):
    import momwire.dropin.nec2c as portal

    calls = []
    monkeypatch.setattr(mech, "pin_cpu_env", lambda *a: calls.append("pin") or {})
    monkeypatch.setattr(
        portal, "main", lambda argv=None: calls.append(("main", argv)) or 7
    )
    assert mech.stock_main(["-version"]) == 7
    assert calls == ["pin", ("main", ["-version"])]
