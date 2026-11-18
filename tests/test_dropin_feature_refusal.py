"""A native feature asked for outside its reach, on the Python drop-in
(momwire#1971 promotion).

`_features.check` raises :class:`~momwire._features.FeatureNotCertified`
when a NATIVE feature is on below its reach. On the drop-in (levels 4 and 5)
that escaped the solve and came back as the last-ditch frame,
``INTERNAL ERROR IN MOMWIRE ENGINE - FeatureNotCertified: ...``, which reads
as a crash. It is a refusal: the printout's ``NEC ERROR`` line is the
registry's own sentence, in every slot (NEC-5, NEC-4.2, NEC-2's SimNEC
portal), however the deck arrived.

The test is generic over the registry: every native feature is moved to
reach 1 (below the drop-in, wherever this runs) and turned on, so whichever
one the slot's solver reads first refuses. The deck is a plain dipole that
every slot serves with the features off.
"""

from __future__ import annotations

import dataclasses

import pytest

from momwire import _features

DIPOLE = """CM a dipole (synthetic)
CE
GW 1 21 0 -2.5 8 0 2.5 8 0.001
GE 0
GN -1
EX 0 1 11 0 1 0
FR 0 1 0 0 28.5 0
XQ
EN
"""


@pytest.fixture
def natives_past_reach(monkeypatch):
    """Every native feature at reach 1 and on in the environment; returns
    their names."""
    for name in list(_features.FEATURES):
        monkeypatch.delenv(_features.FEATURES[name].env, raising=False)
    names = []
    for name, f in list(_features.FEATURES.items()):
        if f.kind != "native":
            continue
        monkeypatch.setitem(
            _features.FEATURES, name, dataclasses.replace(f, reach=1, default=False)
        )
        monkeypatch.setenv(f.env, "1")
        names.append(name)
    assert names
    assert _features.current_level("dropin") > 1
    return names


def _refused_by_name(printout: str, names) -> str:
    assert "INTERNAL ERROR" not in printout, printout
    lines = [ln for ln in printout.splitlines() if "ERROR" in ln]
    assert lines, printout
    hits = [
        n
        for n in names
        if any(f"{n} is not certified for the dropin route" in ln for ln in lines)
    ]
    assert len(hits) == 1, lines
    return hits[0]


@pytest.mark.parametrize(
    ("basis", "dialect"),
    [
        ("razor-2p", "nec5"),  # nec5-razor-2p
        ("bspline", "nec5"),  # nec5-bspline
        ("bspline", "nec4"),  # nec42-bspline
        ("sinusoidal", "nec4"),  # nec42-sinusoidal
    ],
)
def test_the_eznec_slots_refuse_in_the_registrys_sentence(
    basis, dialect, natives_past_reach, monkeypatch
):
    from momwire.dropin.nec5 import _shell

    printout = _shell.render(DIPOLE, basis=basis, dialect=dialect)
    _refused_by_name(printout, natives_past_reach)
    # and with them off the same deck is answered, so the refusal is theirs
    for name in natives_past_reach:
        monkeypatch.setenv(_features.FEATURES[name].env, "0")
    served = _shell.render(DIPOLE, basis=basis, dialect=dialect)
    assert "ERROR" not in served, served


def test_an_escaped_one_is_still_the_registrys_sentence():
    """Wherever else it escapes from (the shell's and the seam's last line of
    defence), the frame is the sentence, not INTERNAL ERROR."""
    from momwire.dropin.nec5 import _shell

    exc = _features.FeatureNotCertified("x is not certified for the dropin route")
    assert _shell._internal_error_reason(exc) == str(exc)
    printout, _ = _shell.seam_failure(DIPOLE, exc)
    assert "INTERNAL ERROR" not in printout
    assert "x is not certified for the dropin route" in printout
    other = _shell._internal_error_reason(RuntimeError("boom"))
    assert other == "INTERNAL ERROR IN MOMWIRE ENGINE - RuntimeError: boom"


def test_the_simnec_portal_refuses_in_the_registrys_sentence(
    natives_past_reach, monkeypatch
):
    from momwire.dropin.nec2c import _portal

    out, _err = _portal.render_deck(DIPOLE.replace("GN -1\n", ""))
    _refused_by_name("\n".join(out), natives_past_reach)
    for name in natives_past_reach:
        monkeypatch.setenv(_features.FEATURES[name].env, "0")
    out, _err = _portal.render_deck(DIPOLE.replace("GN -1\n", ""))
    assert not any("ERROR" in ln for ln in out), out
