"""A '-' that is not SimNEC's streaming call is refused cleanly (momwire#1395).

SimNEC 5.4 starts an ``NEC5CL_a5x`` engine once as ``<engine> - -`` and
streams decks over stdin/stdout, which the NEC-5 slot serves since
momwire#1512 (``tests/test_eznec_stream_1512.py``). Any other ``-`` -- one
of the two paths only, or ``- -`` in the NEC-4.2 slot -- used to be read as
a file name: the refusal printout went into a file called ``-`` and the
process exited 0 with nothing on stdout, so the caller waited forever. Both
entry points (the drop-in client and the shell the Windows executables run)
refuse it on stderr with status 2 and write no file.
"""

from __future__ import annotations

import pytest

import momwire_eznec_client
from momwire.eznec import _shell


@pytest.mark.parametrize("main", [momwire_eznec_client.main, _shell.main])
@pytest.mark.parametrize("args", [["-", "out.txt"], ["deck.nec", "-"]])
def test_a_lone_dash_is_refused_on_stderr(main, args, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(args) == 2
    captured = capsys.readouterr()
    assert "streaming mode" in captured.err
    assert captured.out == ""
    assert list(tmp_path.iterdir()) == []


def test_the_nec4_slot_refuses_the_streaming_call(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert _shell.main(["-", "-"], dialect="nec4") == 2
    captured = capsys.readouterr()
    assert "streaming mode" in captured.err
    assert captured.out == ""
    assert list(tmp_path.iterdir()) == []


def test_the_constant_is_one_sentence_in_both_entry_points():
    assert momwire_eznec_client.STREAMING_REFUSAL == _shell.STREAMING_REFUSAL
