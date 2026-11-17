"""PA3(b): the below table's sweep anchors at their SHIPPED default, `auto`,
executable against Python, each a fresh process with its own empty cache.

Every executable gate pins MOMWIRE_BELOW_ANCHORS=off (standalone.yml's env,
the Skylake nightly's audit.py, the twin tests), on purpose: with `auto`, a
buried sweep's table digits depend on which frequencies the seen list holds
(`_sommerfeld_below._SOMM_BELOW_ANCHOR_TRIGGER`), and that list is shared
through the disk cache, so two processes on one cache directory see
different histories and a byte gate between them reads noise. What ships is
`auto` (`anchor_mode`), so `auto` itself was never held exe == Python.

Here each side gets its own empty MOMWIRE_SOMM_CACHE_DIR and the same deck: a
lossy buried FR sweep of `_POINTS` frequencies inside the trigger's window
(`_SOMM_BELOW_ANCHOR_WINDOW_STEPS` ladder steps of log(1.1)), so the trigger
fires part-way and the tail of the sweep reads blended tables. The test
proves blending ran on both sides before it believes the comparison:
`SWEEP_STATS["blended_grids"]` in the Python process, and the executable's
own count (the printout's ANCHOR-BLENDED under MOMWIRE_TIMES=1 for NEC-5, the
`times:` line's ground_blended on stderr for every build), and the two counts
must agree. A green byte comparison over a sweep that blended nothing would
gate nothing.

The printout alone is too coarse to see the blend: on this deck it is the
same bytes with the anchors off (measured). So for the two NEC-5 builds the
gate is also each finished run's arrays (Y, coeffs, the port vectors;
`exe_arrays_hook.py`), executable against Python bit for bit, and Python with
the anchors off must move exactly the blended runs' arrays, which shows the
comparison could have failed. The NEC-4.2 builds write no arrays, so for them
it is the printout and the counts.

The executables are the Skylake nightly's layout (scripts/nightly/audit.py
EXES): MOMWIRE_STANDALONE_EXES=<root> holding `<cmake basis>-<variant>/<target>`,
the variant from MOMWIRE_FORCE_VARIANT (default avx2). Unset, the module
skips: only a box that builds the four no-Python executables runs it (the
nightly's `anchors-auto` phase). The whole module takes 17 s on Skylake at
one thread (measured 2026-11-17: 3 to 5 s per executable).
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "nightly"))

import audit  # noqa: E402

from momwire import _sommerfeld_below as sb  # noqa: E402

EXES_ROOT = os.environ.get("MOMWIRE_STANDALONE_EXES")
VARIANT = os.environ.get("MOMWIRE_FORCE_VARIANT", "avx2")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not EXES_ROOT,
        reason="MOMWIRE_STANDALONE_EXES does not name a no-Python executable build root",
    ),
]

# A synthetic all-below dipole under a lossy GN 2 soil, swept in frequency.
_POINTS = sb._SOMM_BELOW_ANCHOR_TRIGGER + 2
_F0_MHZ, _DF_MHZ = 6.8, 0.04
_DECK = f"""CM PA3(b): a buried dipole swept in frequency, anchors at their default
CE
GW 1,17,-4.,0.,-.4,4.,0.,-.4,.0012
GE -1,-1
FR 0,{_POINTS},0,0,{_F0_MHZ},{_DF_MHZ}
GN 2,0,0,0,11.,.008
EX 0,1,9,0,1.,0.
XQ 0
EN
"""

# The environment either side may read that would change what this measures.
_DROP = (
    "MOMWIRE_BELOW_ANCHORS",
    "MOMWIRE_SOMM_CACHE",
    "MOMWIRE_SOMM_CACHE_DIR",
    "MOMWIRE_SOMM_CACHE_MAX_MB",
    "MOMWIRE_TIMES",
    "MW_EXE_COPY",
    "MW_EXE_FORCE_FALLBACK",
    "MW_EXE_ROUTE_LOG",
    "MW_EXE_ARRAYS",
    "MW_EXE_PYARRAYS",
)

_PY = """
import json, os, sys
if os.environ.get("MW_EXE_PYARRAYS"):
    sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "core_split"))
    import exe_arrays_hook  # noqa: F401 - each finished run's arrays, as the exe writes them
from momwire.dropin.nec5 import _shell
from momwire import _sommerfeld_below as sb
deck, out, basis, dialect = sys.argv[1:5]
kw = dict(basis=basis) if dialect == "nec5" else dict(basis=basis, dialect="nec4")
text = open(deck, "rb").read().decode("latin-1")
printout = _shell.render_files(text, **kw)[0]
open(out, "wb").write(_shell.printout_bytes(printout))
print(json.dumps(dict(sb.SWEEP_STATS)))
"""


def test_the_sweep_sits_inside_the_trigger_window():
    """The deck's span is inside the window, so the trigger can fire."""
    span = math.log((_F0_MHZ + (_POINTS - 1) * _DF_MHZ) / _F0_MHZ)
    assert span < sb._SOMM_BELOW_ANCHOR_WINDOW_STEPS * sb._SOMM_BELOW_ANCHOR_STEP
    assert _POINTS > sb._SOMM_BELOW_ANCHOR_TRIGGER


def _env(cache: Path, **extra) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in _DROP}
    env.update(
        MOMWIRE_SOMM_CACHE_DIR=str(cache),
        MOMWIRE_THREADS="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MOMWIRE_FORCE_VARIANT=VARIANT,
        PYTHONWARNINGS="ignore",
    )
    env.update(extra)
    return env


def _python(deck: Path, out: Path, basis: str, dialect: str, cache: Path, **extra):
    """(printout bytes, SWEEP_STATS) from a fresh interpreter."""
    p = subprocess.run(
        [sys.executable, "-c", _PY, str(deck), str(out), basis, dialect],
        env=_env(cache, **extra),
        capture_output=True,
        text=True,
        timeout=1800,
        cwd=ROOT,
    )
    assert p.returncode == 0, p.stderr[-2000:]
    return out.read_bytes(), json.loads(p.stdout.strip().splitlines()[-1])


def _arrays(d: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(d.glob("*.bin"))}


def audit_arrays() -> tuple[str, ...]:
    sys.path.insert(0, str(ROOT / "scripts" / "core_split"))
    import printout_compare as pc

    return pc.ARRAYS


def _fresh(tmp_path: Path, name: str) -> Path:
    d = tmp_path / name
    d.mkdir()
    assert not any(d.iterdir())
    return d


@pytest.mark.parametrize("name", sorted(audit.EXES))
def test_anchors_auto_exe_matches_python_in_fresh_processes(name, tmp_path):
    cmake_basis, target, basis, dialect = audit.EXES[name]
    exe = Path(EXES_ROOT, f"{cmake_basis}-{VARIANT}", target)
    # Named a build root, so a missing executable is a failure, not a skip:
    # an all-skipped run exits 0 and would read as a green nightly phase.
    assert exe.is_file(), f"no {exe}"
    deck = tmp_path / "anchors_auto.nec"
    deck.write_text(_DECK)

    # Python: a fresh interpreter, its own empty cache. A NEC-5 build also
    # writes each finished run's arrays (scripts/core_split/exe_arrays_hook.py,
    # the arrays exe_compare.py holds byte for byte); the NEC-4.2 builds have
    # no such hook.
    arrays = dialect == "nec5"
    py_arr, exe_arr = _fresh(tmp_path, "arr-py"), _fresh(tmp_path, "arr-exe")
    py_bytes, stats = _python(
        deck,
        tmp_path / "py.out",
        basis,
        dialect,
        _fresh(tmp_path, "cache-py"),
        **({"MW_EXE_PYARRAYS": str(py_arr)} if arrays else {}),
    )
    assert audit.ERR not in py_bytes, py_bytes[-1500:]
    # Blending ran in the Python process, or the comparison measures nothing.
    assert stats["blended_grids"] > 0, stats

    # The executable as shipped: its own empty cache, anchors on their default.
    exe_out = tmp_path / "exe.out"
    route = tmp_path / "route"
    e = subprocess.run(
        [str(exe), str(deck), str(exe_out)],
        env=_env(
            _fresh(tmp_path, "cache-exe"),
            MW_EXE_ROUTE_LOG=str(route),
            **({"MW_EXE_ARRAYS": str(exe_arr)} if arrays else {}),
        ),
        capture_output=True,
        timeout=1800,
    )
    assert e.returncode == 0, e.stderr[-2000:]
    assert route.read_text().split()[0].startswith("cxx:"), route.read_text()
    exe_bytes = exe_out.read_bytes()

    # The executable's own count, from a second fresh run under MOMWIRE_TIMES.
    timed_out = tmp_path / "exe-timed.out"
    t = subprocess.run(
        [str(exe), str(deck), str(timed_out)],
        env=_env(_fresh(tmp_path, "cache-exe-timed"), MOMWIRE_TIMES="1"),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert t.returncode == 0, t.stderr[-2000:]
    blended = sum(int(n) for n in re.findall(r"ground_blended=(\d+)", t.stderr))
    assert blended > 0, t.stderr[-2000:]
    # The two sides blended the same number of tables (measured 3 and 3: the
    # trigger fires at the 8th frequency of 10).
    assert blended == stats["blended_grids"], (blended, stats)
    if dialect == "nec5":
        printed = timed_out.read_text(encoding="latin-1")
        in_printout = [int(n) for n in re.findall(r"ANCHOR-BLENDED (\d+)\)", printed)]
        assert len(in_printout) == _POINTS, in_printout
        assert sum(in_printout) > 0, in_printout

    # The house gate: byte for byte, the NEC-4.2 slot's two time lines masked.
    exe_m, n_exe = audit.mask(exe_bytes)
    py_m, n_py = audit.mask(py_bytes)
    assert n_exe == n_py
    assert exe_m == py_m, (
        f"{name}: first differing line {audit.first_difference(py_m, exe_m)}"
        f" (Python blended {stats['blended_grids']}, exe {blended})"
    )
    if not arrays:
        return
    # A five-figure printout cannot see the blend: this deck's printout is the
    # same bytes with the anchors off (measured). The arrays can, so they are
    # the gate that a blended table reached the answer alike on both sides.
    got, want = _arrays(exe_arr), _arrays(py_arr)
    assert len(want) == len(audit_arrays()) * _POINTS, sorted(want)
    assert sorted(got) == sorted(want)
    assert [k for k in want if got[k] != want[k]] == []
    # ... and that they would have shown it: Python with the anchors off moves
    # the blended runs' arrays (measured: runs 7-9, every array).
    off_arr = _fresh(tmp_path, "arr-off")
    _python(
        deck,
        tmp_path / "off.out",
        basis,
        dialect,
        _fresh(tmp_path, "cache-off"),
        MW_EXE_PYARRAYS=str(off_arr),
        MOMWIRE_BELOW_ANCHORS="off",
    )
    off = _arrays(off_arr)
    moved = sorted({k.split("_", 1)[0] for k in want if off.get(k) != want[k]})
    assert len(moved) == stats["blended_grids"], moved
