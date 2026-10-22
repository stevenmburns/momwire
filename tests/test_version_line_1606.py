"""The ``-version`` line every launcher answers (momwire#1606).

Four fields separated by spaces, proposed to AE6TY for SimNEC: the deck
dialect, the launcher's name (always starting ``momwire``), the interface
modes in square brackets, and the full momwire version::

    NEC5 momwire-nec5-razor-2p [sb] 0.82.0

Before this the NEC-4.2 launchers answered ``NEC5momwire.<major>.<minor>``,
the NEC-5 slot's line, so SimNEC read engine 5 for them.

The dialect leads for the SimNEC builds already deployed, whose
``Options.getEngine`` takes the engine number from the line's first digit
(``[a-zA-Z]*([0-9])+?(.*)``, ``matches`` over the whole line) and whose NEC-2
and NEC-4.2 insulation checks need 2 and 4 there. A line that starts with the
program name gives no number at all.
"""

import re

import pytest

import momwire_eznec_client as eznec_client
import momwire_nec2c_client as nec2c_client
import momwire_serve_client as mech
from momwire.portal import _portal as nec_portal

# SimNEC's own patterns (nec2/Options, nec2/Execute), read from its jar.
OPTIONS_ENGINE_PIECE = re.compile(r"[a-zA-Z]*([0-9])+?(.*)")
VERSION_NECD = re.compile(r"(NEC\d+\D.*)")

LINE = re.compile(r"(NEC5|NEC42|NEC2) (momwire\S*) \[([sb]*)\] (\S+)")


def _engine_number(line: str) -> str:
    match = OPTIONS_ENGINE_PIECE.fullmatch(line)
    assert match is not None, line
    return match.group(1)


@pytest.mark.parametrize(
    ("prog", "dialect", "name"),
    [
        ("momwire-nec5", "NEC5", "momwire-nec5"),
        ("momwire-nec5-razor-2p", "NEC5", "momwire-nec5-razor-2p"),
        ("momwire-nec5-bspline.exe", "NEC5", "momwire-nec5-bspline"),
        ("momwire-eznec-client", "NEC5", "momwire-nec5"),
        ("momwire-nec42", "NEC42", "momwire-nec42"),
        ("/opt/bin/momwire-nec42-sinusoidal", "NEC42", "momwire-nec42-sinusoidal"),
        ("C:\\SimNEC\\Momwire-Nec42-Pulse.exe", "NEC42", "momwire-nec42-pulse"),
    ],
)
def test_each_launcher_names_its_own_dialect(prog, dialect, name):
    line = eznec_client.probe_version(prog)
    assert line == f"{dialect} {name} [sb] {mech.momwire_version()}"


def test_a_nec42_launcher_no_longer_answers_as_nec5(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["momwire-nec42-sinusoidal", "-version"])
    assert eznec_client.main(None) == 0
    first = capsys.readouterr().out.splitlines()[0]
    assert first.startswith("NEC42 momwire-nec42-sinusoidal [sb] ")
    assert _engine_number(first) == "4"


def test_a_renamed_copy_still_names_itself_as_ours():
    """Ward's macOS setup copies the executable to ``NEC5momwire``; the line
    must still start the name field with ``momwire``."""
    line = eznec_client.probe_version("/Users/w/momwire/NEC5momwire")
    assert LINE.fullmatch(line), line
    assert line.split()[1] == "momwire-nec5"


@pytest.mark.parametrize(
    ("line", "number"),
    [
        (eznec_client.probe_version("momwire-nec5-razor-2p"), "5"),
        (eznec_client.probe_version("momwire-nec42"), "4"),
        (nec2c_client.probe_version(), "2"),
        (nec_portal.probe_version("sinusoidal"), "2"),
        (nec2c_client.probe_version(suffix="pulse"), "2"),
    ],
)
def test_deployed_simnec_reads_the_right_engine_number(line, number):
    assert LINE.fullmatch(line), line
    assert _engine_number(line) == number
    assert VERSION_NECD.match(line)


def test_the_version_is_the_full_one():
    version = mech.momwire_version()
    assert version is not None and version.count(".") == 2
    assert eznec_client.probe_version("momwire-nec5").endswith(f" {version}")


def test_the_nec2_lines_agree_between_the_portal_and_its_client():
    assert nec2c_client.probe_version() == nec_portal.PROBE_VERSION
    for suffix in (None, "bspline", "sinusoidal", "pulse"):
        assert nec2c_client.probe_version(suffix=suffix) == nec_portal.probe_version(
            suffix
        )
        assert nec2c_client.nec2_launcher_name(suffix) == nec_portal.nec2_launcher_name(
            suffix
        )


def _portal_probe(argv, prog):
    out = []

    class _Out:
        def write(self, text):
            out.append(text)

        def flush(self):
            pass

    with nec_portal.engine_scope():
        rc = nec_portal.main(argv, stdin=None, stdout=_Out(), stderr=None, prog=prog)
    assert rc == 0
    return "".join(out)


def test_the_nec2_probe_names_the_launcher_not_the_flag():
    """The line is the launcher's: a copy named for a basis says so, and
    ``--basis`` picks the engine without moving the line."""
    version = mech.momwire_version()
    named = _portal_probe(["-version"], "/usr/bin/momwire-nec2c-pulse")
    assert named == f"NEC2 momwire-nec2c-pulse [sb] {version}\n"
    flagged = _portal_probe(["--basis", "pulse", "-version"], "momwire-nec2c")
    assert flagged == f"NEC2 momwire-nec2c [sb] {version}\n"
