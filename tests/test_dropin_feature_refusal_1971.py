"""A native feature outside its reach, on a drop-in: a clean refusal.

A native experimental feature (`momwire._features`, ``kind="native"``)
turned on by its ``MOMWIRE_X_<NAME>`` variable where its reach does not
cover the running route raises :class:`FeatureNotCertified` from inside the
solve. Every drop-in front end answers that as a refusal in the registry's
own sentence (the feature, the route, the platform, its reach) -- the NEC-5
and NEC-4.2 slots' `` ***** NEC ERROR - `` line, the nec2c portal's
``ERROR:`` frame -- and never as the last line of defence's ``INTERNAL ERROR
IN MOMWIRE ENGINE - FeatureNotCertified: ...`` frame, which reads as an
engine fault (momwire#1971, the drop-in half of razor_pair_reuse's
promotion).

Not vacuous: each front end is handed a deck it SOLVES (the control run,
with the variable off, prints an impedance table and no error), and the
feature is one its own solve asks for: razor_cxx (reach 1) in nec5-razor-2p's
razor route, point_cplx_staged (reach 1) in the sinusoidal solver that
nec42-sinusoidal and nec2c-sinusoidal build. The drop-in is level 4 (5 on
macOS), past both reaches.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from momwire import _features

pytestmark = pytest.mark.filterwarnings("ignore")

# A free-space half-wave dipole at 14 MHz: every front end below solves it.
DIPOLE = (
    "CM a free-space dipole, 21 segments\nCE\nGW 1 21 0 -5 8 0 5 8 0.001\nGE 0\n"
    "EX 0 1 11 0 1 0\nFR 0 1 0 0 14 0\nXQ\nEN\n"
)

# (launcher, feature, how the deck reaches it: "file" = argv deck + printout,
# "stdin" = the nec2c portal's stdin stream, its frame on stdout).
FRONT_ENDS = {
    "nec5-razor-2p": ("momwire-nec5-razor-2p", "razor_cxx", "file"),
    "nec42-sinusoidal": ("momwire-nec42-sinusoidal", "point_cplx_staged", "file"),
    "nec2c-sinusoidal": ("momwire-nec2c-sinusoidal", "point_cplx_staged", "stdin"),
}


def _launcher(name: str) -> str:
    here = Path(sys.executable).parent
    for found in (
        shutil.which(name, path=str(here)),
        shutil.which(name),
        str(here / name),
    ):
        if found and Path(found).exists():
            return found
    pytest.skip(f"{name} is not installed in this environment")


def _sentence(feature: str) -> str:
    """The registry's refusal for `feature` on the drop-in route here."""
    f = _features.FEATURES[feature]
    assert f.kind == "native" and not _features.certified(feature, "dropin"), f
    env = {f.env: "1"}
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        with pytest.raises(_features.FeatureNotCertified) as info:
            _features.check(feature, "dropin")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return str(info.value)


def _run(name: str, mode: str, tmp_path: Path, env: dict[str, str]) -> str:
    deck = tmp_path / "dipole.nec"
    deck.write_text(DIPOLE, newline="")
    full = {k: v for k, v in os.environ.items() if not k.startswith("MOMWIRE_X_")}
    full.update(env)
    prog = _launcher(name)
    if mode == "file":
        out = tmp_path / "dipole.out"
        out.unlink(missing_ok=True)
        r = subprocess.run(
            [prog, deck.name, out.name],
            cwd=tmp_path,
            env=full,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=300,
        )
        assert r.returncode == 0, r.stderr
        return out.read_text(encoding="latin-1")
    with open(deck, "rb") as stdin:
        r = subprocess.run(
            [prog],
            cwd=tmp_path,
            env=full,
            stdin=stdin,
            capture_output=True,
            check=False,
            timeout=300,
        )
    assert r.returncode == 0, r.stderr
    return r.stdout.decode("latin-1")


@pytest.mark.parametrize("front", sorted(FRONT_ENDS))
def test_a_feature_outside_its_reach_is_a_refusal_not_an_internal_error(
    front, tmp_path
):
    name, feature, mode = FRONT_ENDS[front]
    sentence = _sentence(feature)
    env_var = _features.FEATURES[feature].env

    # the control: this front end solves the deck
    (tmp_path / "off").mkdir()
    solved = _run(name, mode, tmp_path / "off", {})
    assert "ERROR" not in solved, solved
    assert "IMPEDANCE" in solved, solved

    # the feature on: the registry's sentence, as the front end's refusal
    (tmp_path / "on").mkdir()
    refused = _run(name, mode, tmp_path / "on", {env_var: "1"})
    assert "INTERNAL ERROR" not in refused, refused
    assert "FeatureNotCertified" not in refused, refused
    if mode == "file":
        assert f" ***** NEC ERROR - {sentence}" in refused, refused
    else:
        assert f"ERROR: {sentence}" in refused.splitlines(), refused
