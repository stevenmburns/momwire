"""razor-2p's cross blocks across a sweep from anchors (`crossing_anchors`,
`_razor_crossing_core.h`'s `anchors`).

On a stepping FR of a deck whose wire crosses the ground, the above x below
cross blocks and the crossing node charges are computed exactly at n_a
Chebyshev-Lobatto wavenumbers spanning the sweep (its two ends among them)
and interpolated at the rest; the above and below blocks stay exact. One C++
core holds the plan, the entries and the interpolation, for the executable's
fill and for RazorSolver's (which fills its anchors with its own cross
terms), so the two print the same bytes.

The gates, and why each is not green by construction:

* every on-test first proves the branch RAN, from the trace
  (MOMWIRE_CROSSING_ANCHOR_TRACE=1: anchors built, fills interpolated) --
  a fill that silently fell back would agree to the bit and measure nothing;
* on against off within the registry's declared drift, |dZin|/|Zin|, at
  full precision on the executable (MW_EXE_ARRAYS) and from the printout on
  the Python route, AND on != off (the interpolant is not the exact fill);
* each decline (too few frequencies, too wide a band, one frequency) is the
  flag-off printout byte for byte but line 2, and its trace names why;
* the executable and the launcher agree byte for byte with it on and off.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from momwire import _features
from momwire._accel import acc

pytestmark = [
    pytest.mark.reach(7),
    pytest.mark.skipif(
        acc is None or not getattr(acc, "crossing_anchors_1117", False),
        reason="needs the crossing anchors core",
    ),
    pytest.mark.filterwarnings("ignore"),
]

EXE = os.environ.get("MOMWIRE_RAZOR2P_STANDALONE_EXE")
BASIS = os.environ.get("MOMWIRE_STANDALONE_BASIS", "razor-2p")
_needs_exe = pytest.mark.skipif(
    not EXE or BASIS != "razor-2p",
    reason="MOMWIRE_RAZOR2P_STANDALONE_EXE does not name a razor-2p executable",
)

ENV = "MOMWIRE_X_CROSSING_ANCHORS"
TRACE = "MOMWIRE_CROSSING_ANCHOR_TRACE"


def _deck(n_freq: int, f0: float, step: float) -> str:
    """A synthetic vertical over two short buried radials, its feed above the
    interface and a crossing node at z = 0, over Sommerfeld ground."""
    return (
        "CM crossing_anchors: vertical over two buried radials (synthetic)\nCE\n"
        "GW 1 8 0 0 -0.1 2.5 0 -0.1 5e-4\n"
        "GW 2 8 0 0 -0.1 -2.5 0 -0.1 5e-4\n"
        "GW 3 2 0 0 -0.1 0 0 0 5e-4\n"
        "GW 4 8 0 0 0 0 0 4.5 5e-4\n"
        "GE -1 0\nGN 0 0 0 0 13 0.005 1 0 NOFILE\nEX 0 4 1 0 1 0\n"
        f"FR 0 {n_freq} 0 0 {f0} {step}\nXQ 0\nEN\n"
    )


SWEEP = _deck(9, 14.0, 0.05)  # 9 points over 2.9 %: the plan engages
DECLINES = {
    "few-frequencies": _deck(6, 14.0, 0.05),  # n_freq <= n_a + 1
    "wide-band": _deck(9, 14.0, 0.25),  # 14 % > MAX_BAND
    "one-frequency": _deck(1, 14.0, 0.0),
}

_TRACE_LINE = re.compile(
    r"^crossing_anchors: (\S+) plan=(\S+) n_a=(\d+) built=(\d+) "
    r"interpolated=(\d+) copied=(\d+)",
    re.M,
)


def _clean_env(**extra):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(_features.ENV_PREFIX)
        and not k.startswith("MOMWIRE_CROSSING_ANCHOR")
        and k not in {a for f in _features.FEATURES.values() for a in f.aliases}
    }
    env.update(extra)
    return env


def _zin(printout: bytes) -> list[complex]:
    """Every run's input impedance, from the ANTENNA INPUT PARAMETERS
    tables (the R and X columns)."""
    lines = printout.decode("latin-1").splitlines()
    out = []
    for i, line in enumerate(lines):
        if "ANTENNA INPUT PARAMETERS" in line:
            f = lines[i + 4].split()
            out.append(complex(float(f[7]), float(f[8])))
    return out


def _launcher() -> str:
    name = "momwire-nec5-razor-2p"
    found = shutil.which(name) or str(Path(sys.executable).parent / name)
    assert Path(found).exists(), found
    return found


def _run(prog, deck_text, d: Path, env) -> tuple[bytes, str]:
    d.mkdir(parents=True)
    (d / "deck.nec").write_text(deck_text, newline="")
    r = subprocess.run(
        [prog, "deck.nec", "deck.out"],
        cwd=d,
        env=env,
        capture_output=True,
        check=False,
        timeout=600,
    )
    assert r.returncode == 0, r.stderr
    return (d / "deck.out").read_bytes(), r.stderr.decode("latin-1")


def _python(deck_text, d: Path, env) -> tuple[bytes, str]:
    """The cold Python route, `_shell.main` with basis razor-2p."""
    d.mkdir(parents=True)
    (d / "deck.nec").write_text(deck_text, newline="")
    code = (
        "import sys; from momwire.dropin.nec5._shell import main; "
        "sys.exit(main(sys.argv[1:], basis='razor-2p'))"
    )
    r = subprocess.run(
        [sys.executable, "-c", code, "deck.nec", "deck.out"],
        cwd=d,
        env=env,
        capture_output=True,
        check=False,
        timeout=600,
    )
    assert r.returncode == 0, r.stderr
    return (d / "deck.out").read_bytes(), r.stderr.decode("latin-1")


def _but_line2(a: bytes, b: bytes) -> bool:
    la, lb = a.split(b"\n"), b.split(b"\n")
    return la[:1] + la[2:] == lb[:1] + lb[2:]


def _last_trace(stderr: str):
    found = _TRACE_LINE.findall(stderr)
    assert found, stderr
    return found[-1]


# --- the registry ---------------------------------------------------------------


def test_the_registry_entry():
    f = _features.FEATURES["crossing_anchors"]
    assert (f.reach, f.kind, f.env, f.kwarg, f.default, f.offer) == (
        7,
        "native",
        ENV,
        "crossing_anchors",
        False,
        False,
    )
    assert f.label and f.summary
    # Approximate: never under the default bar's line would make it look
    # exact, and over 1e-4 it would move a printed digit routinely.
    assert 0 < f.drift <= 1e-4


def test_the_plan_declines_and_engages():
    wl = [299.8 / (14.0 + 0.05 * i) for i in range(9)]
    p = acc.crossing_anchor_plan(wl)
    assert p["ok"] and p["why"] == "ok" and p["n_a"] == 5
    # The two ends are the plan's own wavenumbers, so a fill there copies.
    assert p["anchor_k"][0] == max(p["k"]) and p["anchor_k"][-1] == min(p["k"])
    assert acc.crossing_anchor_plan(wl[:6])["why"] == "few-frequencies"
    wide = [299.8 / (14.0 + 0.25 * i) for i in range(9)]
    assert acc.crossing_anchor_plan(wide)["why"] == "wide-band"
    assert acc.crossing_anchor_plan([])["why"] == ""


def test_the_interpolant_is_exact_on_a_polynomial():
    # n_a nodes interpolate a degree n_a - 1 polynomial in k exactly (to
    # rounding): the barycentric weights are the Lobatto ones.
    wl = [299.8 / (14.0 + 0.05 * i) for i in range(9)]
    p = acc.crossing_anchor_plan(wl)
    ak = np.array(p["anchor_k"])
    n = 3
    idx = np.array([0, 4, 8], dtype=np.int64)
    coef = np.array([1 + 2j, -0.5 + 1j, 3 - 1j])
    raw = np.array([[c * (1 + kj + kj**2 + kj**4) for c in coef] for kj in ak])
    for k in p["k"]:
        z = np.zeros((n, n), dtype=complex, order="F")
        acc.crossing_anchor_apply(z, idx, raw, wl, k)
        want = coef * (1 + k + k**2 + k**4)
        np.testing.assert_allclose(z.ravel(order="F")[idx], want, rtol=1e-12)


# --- the Python route ------------------------------------------------------------


def test_python_off_and_unset_are_the_same_bytes(tmp_path):
    unset, err = _python(SWEEP, tmp_path / "unset", _clean_env(**{TRACE: "1"}))
    off, _ = _python(SWEEP, tmp_path / "off", _clean_env(**{ENV: "0", TRACE: "1"}))
    assert unset == off
    assert b"+crossing_anchors" not in unset.split(b"\n")[1]
    assert "crossing_anchors:" not in err  # off never enters the branch


def test_python_on_runs_and_stays_within_the_drift(tmp_path):
    off, _ = _python(SWEEP, tmp_path / "off", _clean_env())
    on, err = _python(SWEEP, tmp_path / "on", _clean_env(**{ENV: "1", TRACE: "1"}))
    what, plan, n_a, built, interp, copied = _last_trace(err)
    # Nine points and five Lobatto nodes: the two ends and the middle are
    # anchors (copied), the other six are interpolated.
    assert plan == "ok" and int(built) == int(n_a) == 5, err
    assert int(interp) == 6 and int(copied) == 3, err
    assert on.split(b"\n")[1].rstrip().endswith(b"+crossing_anchors")
    z_on, z_off = np.array(_zin(on)), np.array(_zin(off))
    assert len(z_on) == len(z_off) == 9
    # The printout's five figures: the drift plus half a last digit.
    rel = np.abs(z_on - z_off) / np.abs(z_off)
    assert rel.max() <= _features.FEATURES["crossing_anchors"].drift + 5e-5


@pytest.mark.parametrize("case", sorted(DECLINES))
def test_python_declines_fill_as_off(case, tmp_path):
    off, _ = _python(DECLINES[case], tmp_path / "off", _clean_env())
    on, err = _python(
        DECLINES[case], tmp_path / "on", _clean_env(**{ENV: "1", TRACE: "1"})
    )
    assert _but_line2(on, off)
    what, plan, *_ = _last_trace(err)
    assert what == "declined", err
    if case != "one-frequency":
        assert plan == case, err


# --- the executable --------------------------------------------------------------


def _zfull(d: Path) -> np.ndarray:
    n = len(list(d.glob("*_y.bin")))
    return np.array(
        [1.0 / np.fromfile(d / f"{k}_y.bin", dtype=np.complex128)[0] for k in range(n)]
    )


@_needs_exe
def test_exe_off_and_unset_are_the_same_bytes(tmp_path):
    unset, err = _run(EXE, SWEEP, tmp_path / "unset", _clean_env(**{TRACE: "1"}))
    off, _ = _run(EXE, SWEEP, tmp_path / "off", _clean_env(**{ENV: "0", TRACE: "1"}))
    assert unset == off
    assert "crossing_anchors:" not in err


@_needs_exe
def test_exe_on_runs_and_stays_within_the_drift(tmp_path):
    arr_off, arr_on = tmp_path / "arr_off", tmp_path / "arr_on"
    arr_off.mkdir()
    arr_on.mkdir()
    off, _ = _run(EXE, SWEEP, tmp_path / "off", _clean_env(MW_EXE_ARRAYS=str(arr_off)))
    on, err = _run(
        EXE,
        SWEEP,
        tmp_path / "on",
        _clean_env(**{ENV: "1", TRACE: "1", "MW_EXE_ARRAYS": str(arr_on)}),
    )
    what, plan, n_a, built, interp, copied = _last_trace(err)
    assert plan == "ok" and int(built) == 5 and int(interp) == 6 and int(copied) == 3
    z_on, z_off = _zfull(arr_on), _zfull(arr_off)
    assert len(z_on) == len(z_off) == 9
    rel = np.abs(z_on - z_off) / np.abs(z_off)
    assert 0 < rel.max() <= _features.FEATURES["crossing_anchors"].drift
    # The plan's ends and middle copy their anchors: there the cross terms
    # are the exact ones, and only the fold's rounding can move Zin.
    assert max(rel[0], rel[4], rel[-1]) < 1e-13, rel


@_needs_exe
@pytest.mark.parametrize("case", sorted(DECLINES))
def test_exe_declines_fill_as_off(case, tmp_path):
    off, _ = _run(EXE, DECLINES[case], tmp_path / "off", _clean_env())
    on, err = _run(
        EXE, DECLINES[case], tmp_path / "on", _clean_env(**{ENV: "1", TRACE: "1"})
    )
    assert _but_line2(on, off)
    what, plan, *_ = _last_trace(err)
    assert what == "declined", err
    if case != "one-frequency":
        assert plan == case, err


@_needs_exe
@pytest.mark.parametrize("state", ["1", "0"])
def test_exe_is_the_launcher_byte_for_byte(state, tmp_path):
    env = _clean_env(**{ENV: state})
    exe, _ = _run(EXE, SWEEP, tmp_path / "exe", env)
    ref, _ = _run(_launcher(), SWEEP, tmp_path / "launcher", env)
    assert exe == ref
    assert (b"+crossing_anchors" in exe.split(b"\n")[1]) == (state == "1")
