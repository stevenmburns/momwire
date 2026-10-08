"""SimNEC 5.4's streaming call is refused cleanly (momwire#1395).

SimNEC 5.4 starts an ``NEC5CL_a5x`` engine once as ``<engine> - -`` and
streams decks over stdin/stdout. momwire does not serve that mode. It used
to read ``-`` as a file name, write its refusal printout into a file called
``-`` and exit 0 with nothing on stdout, so the caller waited forever. Both
entry points (the drop-in client and the shell the Windows executables run)
now refuse on stderr with status 2 and write no file.
"""

from __future__ import annotations

import pytest

import momwire_eznec_client
from momwire.eznec import _shell


@pytest.mark.parametrize("main", [momwire_eznec_client.main, _shell.main])
@pytest.mark.parametrize("args", [["-", "-"], ["-", "out.txt"], ["deck.nec", "-"]])
def test_streaming_is_refused_on_stderr(main, args, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(args) == 2
    captured = capsys.readouterr()
    assert "streaming mode" in captured.err
    assert captured.out == ""
    assert list(tmp_path.iterdir()) == []
