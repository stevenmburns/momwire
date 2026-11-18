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
* each decline (too few frequencies, too wide a band, one frequency, a deck
  too large electrically for the most anchors; anchors past their memory
  budget decline the same way, checked on the budget itself) is the
  flag-off printout
  byte for byte but line 2, and its trace names why;
* the anchor count is the error rule's: the fewest from 5 that bring the
  a-priori bound under a third of the declared drift, more on a larger
  deck, a decline past the most;
* the executable and the launcher agree byte for byte with it on and off,
  and composed with `crossing_table` and with `crossing_sheet_retune`.
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


def _big_deck(n_freq: int, f0: float, step: float) -> str:
    """A long, low-loss synthetic deck: a half-wave vertical over four 40 m
    radials at 28 MHz on eps_r 13, sigma 1e-3, where the cross entries' phase
    turns too far across a 9.5 % band for even the most anchors to bound."""
    lines = ["CM crossing_anchors: electrically large (synthetic)", "CE"]
    for t, (x, y) in enumerate(((40, 0), (0, 40), (-40, 0), (0, -40)), start=1):
        lines.append(f"GW {t} 12 0 0 -0.15 {x} {y} -0.15 5e-4")
    lines += [
        "GW 5 2 0 0 -0.15 0 0 0 5e-4",
        "GW 6 8 0 0 0 0 0 5.353 5e-4",
        "GE -1 0",
        "GN 0 0 0 0 13 0.001 1 0 NOFILE",
        "EX 0 6 1 0 1 0",
        f"FR 0 {n_freq} 0 0 {f0} {step}",
        "XQ 0",
        "EN",
    ]
    return "\n".join(lines) + "\n"


SWEEP = _deck(9, 14.0, 0.05)  # 9 points over 2.9 %: the plan engages
DECLINES = {
    "few-frequencies": _deck(6, 14.0, 0.05),  # n_freq <= n_a + 1
    "wide-band": _deck(9, 14.0, 0.25),  # 14 % > MAX_BAND
    "one-frequency": _deck(1, 14.0, 0.0),
    "electrically-large": _big_deck(9, 28.0, 0.3325),  # 9.5 %
}

# The lower medium the synthetic decks share, and the solver's c.
GROUND = (13.0, 0.005)
C0 = 299792458.0

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


def _plan(wl, span=5.0, ground=GROUND):
    return acc.crossing_anchor_plan(wl, span, ground[0], ground[1], C0)


def test_the_plan_declines_and_engages():
    wl = [299.8 / (14.0 + 0.05 * i) for i in range(9)]
    p = _plan(wl)
    assert p["ok"] and p["why"] == "ok" and p["n_a"] == 5
    # The two ends are the plan's own wavenumbers, so a fill there copies.
    assert p["anchor_k"][0] == max(p["k"]) and p["anchor_k"][-1] == min(p["k"])
    assert _plan(wl[:6])["why"] == "few-frequencies"
    wide = [299.8 / (14.0 + 0.25 * i) for i in range(9)]
    assert _plan(wide)["why"] == "wide-band"
    assert _plan([])["why"] == ""


def test_the_target_is_a_third_of_the_declared_drift():
    # The bound's target and the registry's drift are one number.
    drift = _features.FEATURES["crossing_anchors"].drift
    assert acc.crossing_anchor_bound_target == pytest.approx(drift / 3, rel=1e-15)


def test_the_anchor_count_follows_the_error_bound():
    """More anchors as the deck grows electrically, each plan the fewest
    (from 5) whose bound meets the target, then a decline."""
    target = acc.crossing_anchor_bound_target
    wl = [299.8 / (28.0 + 0.133 * i) for i in range(21)]  # 9.5 % at 28 MHz
    seen = []
    for span in (1.0, 4.0, 8.0, 12.0, 16.0, 24.0, 400.0):
        p = _plan(wl, span, (13.0, 0.001))
        if not p["ok"]:
            assert p["why"] == "electrically-large", p
            seen.append(None)
            continue
        n = p["n_a"]
        assert p["est"] == acc.crossing_anchor_bound(n, p["x"], p["rho"])
        assert p["est"] <= target
        if n > 5:
            assert acc.crossing_anchor_bound(n - 1, p["x"], p["rho"]) > target
        assert len(p["anchor_k"]) == n
        seen.append(n)
    counts = [n for n in seen if n is not None]
    assert counts == sorted(counts) and counts[-1] > counts[0], seen
    assert seen[-1] is None, seen
    # A small deck over a narrow band takes the floor, five.
    narrow = [299.8 / (14.0 + 0.02 * i) for i in range(21)]  # 2.9 %
    assert _plan(narrow, 1.0, (13.0, 0.001))["n_a"] == 5
    # The phase turn scales with the span and with how fast k_m turns: a
    # low-loss ground turns it faster than a lossy one.
    lo = _plan(wl, 10.0, (13.0, 0.001))
    assert lo["x"] == pytest.approx(2 * _plan(wl, 5.0, (13.0, 0.001))["x"], rel=1e-12)
    assert _plan(wl, 10.0, (1.0, 0.0))["x"] < lo["x"]


def test_the_study_knob_forces_the_count(monkeypatch):
    wl = [299.8 / (28.0 + 0.133 * i) for i in range(21)]
    monkeypatch.setenv("MOMWIRE_CROSSING_ANCHOR_NODES", "7")
    p = _plan(wl, 400.0, (13.0, 0.001))
    assert p["ok"] and p["n_a"] == 7


def test_the_anchors_hold_a_memory_budget():
    # Their values (n_a x entries) plus one n x n scratch, 16 bytes each,
    # within 256 MiB: the target deck's size fits, a 3000-basis crossing
    # deck with half its bases below does not, even at the floor of five.
    assert acc.crossing_anchor_fits(5, 2 * 125 * 126 + 2 * 251, 251)
    assert acc.crossing_anchor_fits(12, 2 * 125 * 126 + 2 * 251, 251)
    assert not acc.crossing_anchor_fits(5, 2 * 1500 * 1500, 3000)
    budget = 256 * 1024 * 1024 // 16
    assert acc.crossing_anchor_fits(1, budget - 100 * 100, 100)
    assert not acc.crossing_anchor_fits(1, budget - 100 * 100 + 1, 100)


def test_the_span_is_above_to_below():
    # A segment's side is its midpoint's; the span the farthest pair of ends.
    p0 = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [5.0, 0.0, 2.0]])
    t = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    h = np.array([3.0, 2.0, 1.0])
    # Below: (0,0,-1)-(3,0,-1). Above: (0,0,0)-(0,0,2), (5,0,2)-(5,0,3).
    want = np.sqrt(5.0**2 + 4.0**2)  # (5,0,3) to (0,0,-1)
    assert acc.crossing_anchor_span(p0, t, h, 0.0) == pytest.approx(want, rel=1e-15)


def test_the_interpolant_is_exact_on_a_polynomial():
    # n_a nodes interpolate a degree n_a - 1 polynomial in k exactly (to
    # rounding): the barycentric weights are the Lobatto ones.
    wl = [299.8 / (14.0 + 0.05 * i) for i in range(9)]
    p = _plan(wl)
    ak = np.array(p["anchor_k"])
    n = 3
    idx = np.array([0, 4, 8], dtype=np.int64)
    coef = np.array([1 + 2j, -0.5 + 1j, 3 - 1j])
    raw = np.array([[c * (1 + kj + kj**2 + kj**4) for c in coef] for kj in ak])
    for k in p["k"]:
        z = np.zeros((n, n), dtype=complex, order="F")
        acc.crossing_anchor_apply(z, idx, raw, wl, k, *p["size"])
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


COMPOSED = {
    "on": {ENV: "1"},
    "off": {ENV: "0"},
    "with-table": {ENV: "1", "MOMWIRE_X_CROSSING_TABLE": "1"},
    "table-alone": {ENV: "0", "MOMWIRE_X_CROSSING_TABLE": "1"},
    "with-retune": {ENV: "1", "MOMWIRE_X_CROSSING_SHEET_RETUNE": "1"},
}


@_needs_exe
@pytest.mark.parametrize("case", sorted(COMPOSED))
def test_exe_is_the_launcher_byte_for_byte(case, tmp_path):
    env = _clean_env(**COMPOSED[case], **{TRACE: "1"})
    exe, err_exe = _run(EXE, SWEEP, tmp_path / "exe", env)
    ref, _ = _run(_launcher(), SWEEP, tmp_path / "launcher", env)
    assert exe == ref
    on = COMPOSED[case][ENV] == "1"
    assert (b"+crossing_anchors" in exe.split(b"\n")[1]) == on
    if on:
        # The executable ran the anchors (not a fallback that agrees by
        # construction). The launcher's solve runs in its drop-in server,
        # whose stderr is not this process's, so its side is the bytes.
        what, plan, n_a, built, interp, copied = _last_trace(err_exe)
        assert plan == "ok" and int(built) == int(n_a) and int(interp) > 0, err_exe


@_needs_exe
def test_exe_composes_with_the_sheet_retune(tmp_path):
    """`crossing_sheet_retune` moves only the census's build weights, a
    constant inside the fill, so the anchors are its own exact cross blocks
    at the anchor wavenumbers: with both on against the retune alone is the
    anchors' drift, and the retune is still in force (its stamp)."""
    arr_r, arr_b = tmp_path / "arr_r", tmp_path / "arr_b"
    arr_r.mkdir()
    arr_b.mkdir()
    retune = {"MOMWIRE_X_CROSSING_SHEET_RETUNE": "1"}
    alone, _ = _run(
        EXE, SWEEP, tmp_path / "r", _clean_env(**retune, MW_EXE_ARRAYS=str(arr_r))
    )
    both, err = _run(
        EXE,
        SWEEP,
        tmp_path / "b",
        _clean_env(**retune, **{ENV: "1", TRACE: "1", "MW_EXE_ARRAYS": str(arr_b)}),
    )
    what, plan, n_a, built, interp, copied = _last_trace(err)
    assert plan == "ok" and int(interp) > 0, err
    line2 = both.split(b"\n")[1]
    assert b"+crossing_sheet_retune" in line2 and b"+crossing_anchors" in line2
    z_b, z_r = _zfull(arr_b), _zfull(arr_r)
    rel = np.abs(z_b - z_r) / np.abs(z_r)
    assert 0 < rel.max() <= _features.FEATURES["crossing_anchors"].drift
