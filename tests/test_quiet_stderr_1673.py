"""momwire#1673's switch: ``MOMWIRE_ADVISORIES=off`` writes nothing on stderr.

SimNEC's streaming call (``- -``) used to never drain the engine's stderr, so
once the advisories a long stream raises passed the pipe's size (about 4 KB
on Windows) the engine blocked on the write and the host waited forever for
a printout. The launchers then wrote nothing there. 0.97.0 wrote them by
default; since momwire#2119 ``off`` is the default again, and the writing
modes keep a per-process cap (tests/test_advisories_modes_2119.py). Off, the
warnings still reach anyone recording them.
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from momwire.dropin.nec5 import _shell

DECK = (
    Path(__file__).parent / "fixtures" / "buried_grazing_1672" / "shallow_s40.nec"
).read_text()
# Two structures, NX-separated, each raising CoarseCrossingNode and
# ExtendedKernelDefault (~1.9 KB of advisories per structure).
STREAM = DECK.replace("\nEN\n", "\nNX\n") + DECK

SHELL = "import sys; from momwire.dropin.nec5._shell import main; sys.exit(main(sys.argv[1:]))"


def _run(args, stdin, env_extra):
    env = {k: v for k, v in os.environ.items() if k != _shell.ADVISORIES_ENV}
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", SHELL, *args],
        input=stdin.encode(),
        capture_output=True,
        env=env,
        timeout=600,
        check=False,
    )


OFF = {_shell.ADVISORIES_ENV: "off"}


@pytest.mark.integration
def test_a_stream_of_advisory_decks_writes_nothing_on_stderr_when_off():
    quiet = _run(["-", "-"], STREAM, OFF)
    assert quiet.returncode == 0
    assert quiet.stdout.count(b"ANTENNA INPUT PARAMETERS") == 2
    assert quiet.stderr == b"", quiet.stderr[:400]
    # ...as by default (momwire#2119), and a writing mode carries them
    assert _run(["-", "-"], STREAM, {}).stderr == b""
    loud = _run(["-", "-"], STREAM, {_shell.ADVISORIES_ENV: "deck"})
    assert b"CoarseCrossingNode" in loud.stderr
    assert loud.stdout == quiet.stdout


@pytest.mark.integration
def test_a_one_shot_writes_nothing_on_stderr_when_off(tmp_path):
    deck = tmp_path / "d.nec"
    deck.write_text(DECK)
    out = tmp_path / "d.out"
    r = _run([str(deck), str(out)], "", OFF)
    assert r.returncode == 0 and r.stderr == b"", r.stderr[:400]
    assert "ANTENNA INPUT PARAMETERS" in out.read_text()


@pytest.mark.parametrize("setting", [None, "off", "deck", "all"])
def test_the_advisories_are_still_raised_in_process(tmp_path, monkeypatch, setting):
    """Only the display changes: a caller recording warnings around `main`
    sees every advisory, displayed or sunk."""
    if setting is None:
        monkeypatch.delenv(_shell.ADVISORIES_ENV, raising=False)
    else:
        monkeypatch.setenv(_shell.ADVISORIES_ENV, setting)
    deck = tmp_path / "d.nec"
    deck.write_text(DECK)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert _shell.main([str(deck), str(tmp_path / "d.out")]) == 0
    names = {w.category.__name__ for w in seen}
    assert "CoarseCrossingNode" in names, names


def test_main_puts_stderr_back_and_its_own_lines_still_reach_it(capsys):
    before = sys.stderr
    assert _shell.main(["-", "x"]) == 2  # not the streaming call: refused
    assert sys.stderr is before
    assert _shell.STREAMING_REFUSAL in capsys.readouterr().err
