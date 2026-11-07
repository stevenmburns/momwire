"""momwire#1831: the drop-in moved to ``momwire.dropin``; the old names stay.

``momwire.portal`` became :mod:`momwire.dropin.nec2c` and ``momwire.eznec``
became :mod:`momwire.dropin.nec5`.  antennaknobs and the scripts import the
old names, private submodules included, and monkeypatch some of what they
import, so the shims must hand back the SAME module objects, not copies.
The thin client gained ``momwire-dropin-client`` beside
``momwire-eznec-client``, which stays as an alias.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import momwire_eznec_client as client

REPO = Path(__file__).resolve().parent.parent

SUBMODULES = {
    ("momwire.portal", "momwire.dropin.nec2c"): ("_portal", "_shared"),
    ("momwire.eznec", "momwire.dropin.nec5"): (
        "_gmo",
        "_nec4",
        "_nec4_printout",
        "_printout",
        "_resident",
        "_serve",
        "_shell",
    ),
}


@pytest.mark.parametrize(("old", "new"), list(SUBMODULES))
def test_an_old_package_name_is_the_new_package(old, new):
    assert importlib.import_module(old) is importlib.import_module(new)


@pytest.mark.parametrize(
    ("old", "new", "sub"),
    [(old, new, sub) for (old, new), subs in SUBMODULES.items() for sub in subs],
)
def test_an_old_submodule_name_is_the_new_module(old, new, sub):
    """Same object by every spelling, and it keeps its own name and spec, so
    a monkeypatch through either name reaches the code that runs."""
    by_old = importlib.import_module(f"{old}.{sub}")
    by_new = importlib.import_module(f"{new}.{sub}")
    assert by_old is by_new
    assert by_new.__name__ == f"{new}.{sub}"
    assert by_new.__spec__.name == f"{new}.{sub}"


def test_the_old_names_resolve_in_a_fresh_interpreter():
    """Old name first, before anything imported the new one: the order a
    script that only knows the old name takes."""
    code = (
        "from momwire.eznec import _serve, serve\n"
        "from momwire.portal import _portal, run_deck\n"
        "import momwire.dropin.nec5._serve as s, momwire.dropin.nec2c._portal as p\n"
        "assert _serve is s and serve is s.serve\n"
        "assert _portal is p and run_deck is p.run_deck\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


@pytest.mark.parametrize(("old", "new"), [("momwire.portal", "momwire.dropin.nec2c")])
def test_python_dash_m_old_name_answers_as_the_new_one(old, new):
    """``python -m momwire.portal -version`` still runs the engine."""
    answers = [
        subprocess.run(
            [sys.executable, "-m", name, "-version"],
            capture_output=True,
            timeout=180,
        )
        for name in (old, new)
    ]
    assert answers[0].returncode == answers[1].returncode == 0
    assert answers[0].stdout == answers[1].stdout
    assert answers[0].stdout.startswith(b"NEC2 momwire-nec2c ")


def test_python_dash_m_momwire_eznec_renders_the_same_printout(tmp_path):
    deck = (
        REPO
        / "tests"
        / "fixtures"
        / "eznec"
        / "decks"
        / "0042_dipole-in-free-space.nec"
    )
    outputs = []
    for name in ("momwire.eznec", "momwire.dropin.nec5"):
        out = tmp_path / f"{name}.out"
        proc = subprocess.run(
            [sys.executable, "-m", name, str(deck), str(out)],
            capture_output=True,
            timeout=600,
        )
        assert proc.returncode == 0, proc.stderr
        outputs.append(out.read_bytes())
    assert outputs[0] == outputs[1]
    assert outputs[0]


def test_the_dropin_client_is_the_eznec_client_under_a_new_name():
    scripts = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["momwire-dropin-client"] == scripts["momwire-eznec-client"]


@pytest.mark.parametrize(
    ("prog", "name"),
    [
        ("momwire-dropin-client", "momwire-nec5"),
        ("momwire-eznec-client", "momwire-nec5"),
        ("momwire-dropin-client-razor-2p", "momwire-nec5-razor-2p"),
        ("momwire-eznec-client-razor-2p", "momwire-nec5-razor-2p"),
        ("Momwire-Dropin-Client-Bspline.exe", "momwire-nec5-bspline"),
    ],
)
def test_the_dropin_client_reads_its_basis_from_its_name(prog, name):
    assert client.launcher_name(prog) == name
