"""Two venvs on one base interpreter are two engines — momwire#1674.

A venv's ``bin/python`` is a symlink to its base interpreter, so the old key
element ``realpath(sys.executable)`` was the same for every venv built on it,
and two installs of one version shared a resident server. Inside a venv its
prefix now joins the key; outside one (a frozen bundle, a plain interpreter)
the key is the one it was, which the native C launcher spells too.
"""

from __future__ import annotations

import os
import sys

import pytest

import momwire_eznec_client as eznec
import momwire_nec2c_client as nec2c
import momwire_serve_client as mech

BASE = "/opt/python3.12/bin/python3.12"


@pytest.fixture
def interpreter(monkeypatch):
    """Point the process at BASE through ``prefix``'s ``bin/python``."""

    def use(prefix: str | None):
        monkeypatch.setattr(
            os.path,
            "realpath",
            lambda p: BASE if p.endswith("python") or p == BASE else p,
        )
        monkeypatch.setattr(sys, "base_prefix", "/opt/python3.12")
        monkeypatch.setattr(sys, "prefix", prefix or "/opt/python3.12")
        monkeypatch.setattr(
            sys, "executable", f"{prefix}/bin/python" if prefix else BASE
        )

    return use


def _keys():
    return (
        eznec.config_key(None, 600.0),
        eznec.config_key("razor-2p", 600.0, "nec5"),
        eznec.config_key(None, 600.0, "nec4"),
        nec2c.config_key([], 600.0),
    )


def test_two_venvs_on_one_interpreter_get_two_servers(interpreter):
    interpreter("/home/u/a/.venv")
    a = _keys()
    interpreter("/home/u/b/.venv")
    b = _keys()
    assert all(x != y for x, y in zip(a, b, strict=True))


def test_one_venv_keeps_one_key(interpreter):
    interpreter("/home/u/a/.venv")
    assert _keys() == _keys()


def test_outside_a_venv_the_key_is_the_one_it_was(interpreter):
    """No venv element: the key is the realpath-only digest the C launcher
    and every earlier release compute."""
    interpreter(None)
    assert mech.interpreter_identity() == [BASE]
    major, minor = mech.dist_version()
    assert eznec.config_key(None, 600.0) == mech.digest(
        [
            f"eznec.{major}.{minor}",
            BASE,
            repr(600.0),
            "default",
            *mech.junction_rule_identity(),
        ]
    )


def test_inside_a_venv_the_prefix_joins_the_key(interpreter):
    interpreter("/home/u/a/.venv")
    assert mech.interpreter_identity() == [BASE, "venv=/home/u/a/.venv"]
