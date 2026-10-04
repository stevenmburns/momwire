"""EZNEC's External NEC-4.2 slot, phase 1: the name switch and the deck reader.

momwire#1295.  A program name carrying ``nec4`` answers the NEC-4.2 slot: its
decks are read by the ``nec4`` dialect (``momwire.deck._nec4``), which
addresses sources at segment centres, so the slot's roster is
``deck.NEC2_BASES`` and razor refuses by name.  The corpus is the seventeen
decks EZNEC wrote in the 2026-10-03 capture sitting
(``tests/fixtures/eznec_nec42/``, README there).

The target impedances below are the licensed NEC-4.2's, as antennaknobs
``scratch/eznec-capture/NEC42-SOLVE-NOTES-2026-10-03.md`` quotes them (black
box: the engine solved each captured deck and the figures came back).  The
licensee has since permitted its printouts as fixtures, and phase 2 keeps
them (``tests/fixtures/eznec_nec42/printouts/``); the same impedances are on
their ANTENNA INPUT PARAMETERS rows.
"""

from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path

import pytest

import momwire_serve_client as mech
from momwire.deck import NEC2_BASES, DeckError, parse
from momwire.deck._nec4 import parse_nec4
from momwire.deck._solver import (
    dialect_from_program_name,
    nec4_basis_from_program_name,
)
from momwire.eznec import _serve
from momwire.eznec._nec4 import basis_refusal
from momwire.eznec._shell import render

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "eznec_nec42"
ENTRY = Path(__file__).resolve().parent.parent / "scripts" / "eznec_freeze" / "entry.py"

# Licensed NEC-4.2's feed-point impedance per captured deck, ohms, as the
# solve notes quote it (provenance in the module docstring).
NEC42_Z = {
    "0223": 81.7499 + 46.0034j,
    "0224": 81.7499 + 46.0034j,
    "0225": 106.1310 + 40.2981j,
    "0226": 106.1310 + 40.2981j,
    "0227": 82.1757 + 46.3603j,
    "0228": 81.7499 + 46.0034j,
    "0229": 81.6310 + 44.9372j,
    "0230": 81.6310 + 44.9372j,
    "0231": 81.6784 + 45.4004j,
    "0232": 81.6784 + 45.4004j,
    "0233": 0.1118 - 9078.8200j,
    "0234": 31.3773 + 25.9276j,
    "0235": 13.6484 + 4.4250j,
    "0236": 81.7499 + 46.0034j,
    "0237": 81.7499 + 46.0034j,
    "0238": 81.7499 + 46.0034j,
    "0239": 149.7710 + 143.2080j,
}

CAPTURES = tuple(sorted(NEC42_Z))

# Captures this dialect refuses by name, with the sentence's anchor.  Empty:
# every deck the slot wrote is read.
REFUSED: dict[str, str] = {}


def deck_text(capture: str) -> str:
    (path,) = FIXTURES.glob(f"{capture}_*.nec")
    return path.read_bytes().decode("latin-1")


def _entry_module():
    spec = importlib.util.spec_from_file_location("eznec_freeze_entry_1295", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# the corpus is the captures, byte for byte


def test_the_fixtures_are_the_captures_as_written():
    """CRLF, as EZNEC wrote them: the files are a byte oracle (``-text`` in
    the directory's .gitattributes), and a normalised copy would no longer be
    what the slot sends."""
    assert sorted(p.name[:4] for p in FIXTURES.glob("*.nec")) == list(CAPTURES)
    for capture in CAPTURES:
        raw = (next(FIXTURES.glob(f"{capture}_*.nec"))).read_bytes()
        assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b""), capture
        assert b"in NEC-4.2 format." in raw, capture


# --------------------------------------------------------------------------
# the name switch


@pytest.mark.parametrize(
    "prog, dialect, basis",
    [
        ("momwire-nec4", "nec4", None),
        ("momwire-nec4.exe", "nec4", None),
        ("C:\\EZNEC 7.0\\Docs\\momwire-nec4.exe", "nec4", None),
        ("momwire-nec4-sinusoidal.exe", "nec4", "sinusoidal"),
        ("Momwire-NEC4-Sinusoidal-Galerkin.EXE", "nec4", "sinusoidal-galerkin"),
        ("/opt/bin/momwire-nec4-bspline-d1", "nec4", "bspline-d1"),
        # Copies the README cannot stop someone making: each NAMES its basis,
        # so each reaches the refusal rather than the default.
        ("momwire-nec4-razor-2p.exe", "nec4", "razor-2p"),
        ("momwire-nec4-", "nec4", ""),
        ("momwire-nec4-rzaor", "nec4", "rzaor"),
        ("momwire-nec42.exe", "nec4", "2"),
        # The capture shim's own name: a nec4 name with an unknown suffix.
        ("momwire-nec4-capture.exe", "nec4", "capture"),
        # The NEC-5 family is untouched.
        ("momwire-eznec.exe", "nec5", None),
        ("momwire-eznec-razor-2p.exe", "nec5", None),
        ("momwire-nec5-razor-2p", "nec5", None),
        ("python", "nec5", None),
    ],
)
def test_the_name_picks_the_slot_and_the_basis(prog, dialect, basis):
    assert dialect_from_program_name(prog) == dialect
    assert nec4_basis_from_program_name(prog) == basis
    # The stdlib copy a thin client reads is the same rule.
    assert mech.filename_nec4(prog) == (dialect == "nec4", basis)


@pytest.mark.parametrize(
    "prog, dialect, basis",
    [
        ("momwire-nec4.exe", "nec4", _serve.BASIS),
        ("momwire-nec4-sinusoidal.exe", "nec4", "sinusoidal"),
        ("momwire-nec4-razor-2p.exe", "nec4", "razor-2p"),
        ("momwire-nec4-.exe", "nec4", ""),
        ("momwire-eznec.exe", "nec5", _serve.BASIS),
        ("momwire-eznec-razor-2p.exe", "nec5", "razor-2p"),
        ("momwire-eznec-engine.exe", "nec5", _serve.BASIS),
    ],
)
def test_the_frozen_entry_reads_the_slot_off_its_name(prog, dialect, basis):
    entry = _entry_module()
    assert entry.dialect_for(prog) == dialect
    assert entry.basis_for(prog) == basis


def test_the_engine_one_shot_takes_the_slot_as_a_leading_flag(monkeypatch):
    """The native launcher runs the bundle's ONE engine exe, named for
    neither slot, so a NEC-4.2 launcher's fallback rung says ``--dialect``
    the way a twin says ``--basis``; either order is the same request."""
    entry = _entry_module()
    seen = {}
    monkeypatch.setattr(
        entry,
        "main",
        lambda argv, *, basis, dialect: (
            seen.update(argv=argv, basis=basis, dialect=dialect) or 0
        ),
    )
    for argv in (
        ["momwire-eznec-engine.exe", "--dialect", "nec4", "--basis", "pulse"],
        ["momwire-eznec-engine.exe", "--basis", "pulse", "--dialect", "nec4"],
    ):
        assert entry.run([*argv, "EZ.NEC", "NEC.OUT"]) == 0
        assert seen == {
            "argv": ["EZ.NEC", "NEC.OUT"],
            "basis": "pulse",
            "dialect": "nec4",
        }


def test_a_nec4_named_daemon_serves_the_nec4_slot(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "momwire.eznec._resident.serve_main",
        lambda argv: seen.update(argv=argv) or 0,
    )
    assert _entry_module().run(["momwire-nec4-pulse.exe", "--serve"]) == 0
    assert seen["argv"] == ["--serve", "--basis", "pulse", "--dialect", "nec4"]


def test_a_nec4_server_never_shares_a_nec5_servers_key():
    """The slot is engine identity: a NEC-5 daemon answering a NEC-4.2 deck
    would parse it in the wrong dialect.  A NEC-5 key is unchanged."""
    import momwire_eznec_client as client

    assert client.config_key(None, 900.0, "nec4") != client.config_key(None, 900.0)
    assert client.config_key(None, 900.0, "nec5") == client.config_key(None, 900.0)
    command = client._server_command("/tmp/x.sock", None, 900.0, "/tmp/x.log", "nec4")
    assert command[-2:] == ["--dialect", "nec4"]
    assert "--dialect" not in client._server_command(
        "/tmp/x.sock", None, 900.0, "/tmp/x.log"
    )


def test_the_slots_roster_is_the_centre_feeding_bases():
    for basis in NEC2_BASES:
        assert basis_refusal(basis) is None, basis
    for basis in ("razor-2p", "razor-nec5"):
        sentence = basis_refusal(basis)
        assert sentence is not None and repr(basis) in sentence
        assert "segment centres" in sentence
    assert "unknown basis 'rzaor'" in basis_refusal("rzaor")
    assert "unknown basis ''" in basis_refusal("")


def test_every_nec4_console_script_names_a_basis_the_slot_serves():
    import tomllib

    scripts = tomllib.loads((ENTRY.parents[2] / "pyproject.toml").read_text())[
        "project"
    ]["scripts"]
    family = {
        k: v for k, v in scripts.items() if dialect_from_program_name(k) == "nec4"
    }
    assert "momwire-nec4" in family and "momwire-nec4-sinusoidal" in family
    assert set(family.values()) == {"momwire_eznec_client:main"}
    bases = {nec4_basis_from_program_name(k) for k in family} - {None}
    assert bases == set(NEC2_BASES)


# --------------------------------------------------------------------------
# the deck reader


@pytest.mark.parametrize("capture", CAPTURES)
def test_every_captured_deck_parses_or_refuses_by_name(capture):
    if capture in REFUSED:
        with pytest.raises(DeckError, match=REFUSED[capture]):
            parse_nec4(deck_text(capture))
        return
    model = parse(deck_text(capture), dialect="nec4")
    assert model.feeds, capture
    assert len([g for g in model.groups if g is not None]) == 1
    (group,) = model.groups
    assert len(group.frequencies) == 1


def test_ex6_is_a_current_source_and_ex0_a_voltage_source():
    current = parse_nec4(deck_text("0223"))
    voltage = parse_nec4(deck_text("0224"))
    assert current.current_feeds == (0,)
    assert voltage.current_feeds == ()
    # Same port, same amplitude; only the kind differs.
    assert current.feeds == voltage.feeds
    assert current.feeds[0][2] == pytest.approx(1.414214)


def test_each_ground_card_maps_to_its_momwire_ground():
    assert parse_nec4(deck_text("0228")).ground is None
    assert parse_nec4(deck_text("0229")).ground == "pec"
    # Both spellings of GN 1 are one ground.
    assert parse_nec4(deck_text("0230")).ground == "pec"
    # GD rides as the second medium a cliff pattern alone reads.
    second = parse_nec4(deck_text("0230")).second_medium
    assert (second.eps_r, second.sigma) == (13.0, 0.005)
    assert parse_nec4(deck_text("0231")).ground == ("finite", 13.0, 0.005)
    # GN 3: the same Sommerfeld half-space as GN 2, one momwire evaluation.
    assert parse_nec4(deck_text("0232")).ground == ("finite", 13.0, 0.005)
    buried = parse_nec4(deck_text("0239"))
    assert buried.ground == ("finite", 13.0, 0.005)
    assert buried.ground_plane_flag and not buried.ground_plane_interpolates
    assert max(v[2] for w in buried.wires for v in w.vertices) < 0.0


def test_the_networks_attach_by_tag_and_segment():
    model = parse_nec4(deck_text("0234"))
    assert [n.kind for n in model.networks] == ["TL", "TL", "NT"]
    assert [n.address_a for n in model.networks] == [(3, 1), (3, 2), (3, 1)]
    assert model.current_feeds == (0,)


def test_the_pattern_requests_are_read_as_written():
    elevation = parse_nec4(deck_text("0237")).groups[0].request
    assert (elevation.n_theta, elevation.n_phi, elevation.d_theta_deg) == (361, 1, -1.0)
    grid = parse_nec4(deck_text("0238")).groups[0].request
    assert (grid.n_theta, grid.n_phi) == (37, 73)
    assert parse_nec4(deck_text("0233")).groups[0].request is None


def _edited(capture: str, old: str, new: str) -> str:
    text = deck_text(capture)
    assert old in text
    return text.replace(old, new)


@pytest.mark.parametrize(
    "text, match",
    [
        pytest.param(
            _edited("0223", "EX 6,1,6,0,1.414214,0.", "EX 1,1,6,0,1.414214,0."),
            r"EX type 1 is not served by this engine's nec4 dialect",
            id="ex-type",
        ),
        pytest.param(
            _edited("0223", "EX 6,1,6,0,1.414214,0.", "EX 6,1,6,0,0.,0."),
            r"sets a current of zero",
            id="zero-current",
        ),
        pytest.param(
            _edited(
                "0223",
                "EX 6,1,6,0,1.414214,0.",
                "EX 6,1,6,0,1.414214,0.\r\nEX 0,1,2,0,1.,0.",
            ),
            r"2 sources in one run with an EX 6 current source",
            id="current-in-a-phased-drive",
        ),
        pytest.param(
            _edited(
                "0223",
                "EX 6,1,6,0,1.414214,0.",
                "EX 6,1,6,0,1.414214,0.\r\nXQ 0\r\nEX 0,1,6,0,1.,0.",
            ),
            r"a port is either a voltage source or a current source",
            id="one-port-two-kinds",
        ),
        pytest.param(
            _edited("0223", "GE 0,-1", "GE 2,-1"),
            r"GE 2 is not a NEC-4.2 ground flag",
            id="ge-flag",
        ),
        pytest.param(
            _edited("0223", "GE 0,-1", "GE 0,1"),
            r"GE with a second field of 1",
            id="ge-second-field",
        ),
        pytest.param(
            _edited("0232", "GN 3,0,0,0,13.,.005", "GN 3,4,0,0,13.,.005,1.,.001"),
            r"GN 3 with a 4-wire radial ground screen",
            id="gn3-screen",
        ),
    ],
)
def test_what_no_capture_writes_refuses_by_name(text, match):
    with pytest.raises(DeckError, match=match):
        parse_nec4(text)


def test_the_nec2_dialect_still_refuses_what_nec4_adds():
    """The deltas are the nec4 dialect's alone: nec2 decks mean what they did."""
    with pytest.raises(DeckError, match="this engine drives EX 0 only"):
        parse(deck_text("0223"), dialect="nec2")
    with pytest.raises(DeckError, match="GN type 3 is not supported"):
        parse(deck_text("0232"), dialect="nec2")


# momwire#1317: GN 2 naming a Sommerfeld-table file, the way the licensed
# NEC-4.2 was run on it (spaced fields), against the same card without one.
# On probe p7, the 7 MHz GN 2 dipole: 0231's 10-wavelength height makes a
# cold Sommerfeld solve several times dearer for the same question.
_GN2_WRITTEN = "GN 2,0,0,0,13.,.005"
_GN2_BARE = "GN 2 0 0 0 13. .005 0 0 0 0"


def _gn2(trailer: str | None) -> str:
    card = _GN2_BARE if trailer is None else f"{_GN2_BARE} {trailer}"
    (path,) = (FIXTURES / "probes").glob("p7_*.nec")
    text = path.read_bytes().decode("latin-1")
    assert _GN2_WRITTEN in text
    return text.replace(_GN2_WRITTEN, card)


@pytest.mark.parametrize("name", ["NOFILE", "SOMEX10.NEC"])
def test_nec4_reads_gn2_without_its_sommerfeld_table_file(name):
    """The file is a cache the licensed engine computes when it cannot open
    it, so the card means what it means without the name."""
    assert parse_nec4(_gn2(name)) == parse_nec4(_gn2(None))
    assert parse_nec4(_gn2(name)).ground == ("finite", 13.0, 0.005)


@pytest.mark.parametrize("name", ["NOFILE", "SOMEX10.NEC"])
def test_nec2_still_refuses_a_gn_file_name(name):
    with pytest.raises(DeckError) as exc:
        parse(_gn2(name), dialect="nec2")
    assert str(exc.value) == (
        f"GN carries a trailing token {name!r}; this engine's nec2 dialect "
        f"has no Sommerfeld-table file field on GN (momwire#1084 adds that "
        f"field to the nec5 dialect only)"
    )


@pytest.mark.parametrize(
    "old, new",
    [
        ("GN 3,0,0,0,13.,.005", "GN 3,0,0,0,13.,.005,0,0,0,0,NOFILE"),
        ("GN 1", "GN 1,NOFILE"),
    ],
    ids=["gn3", "gn1"],
)
def test_nec4_refuses_a_file_name_on_any_other_gn_type(old, new):
    capture = "0232" if old.startswith("GN 3") else "0229"
    text = deck_text(capture)
    line = next(ln for ln in text.splitlines() if ln.startswith(old))
    with pytest.raises(DeckError, match=r"reads a Sommerfeld-table file name on GN 2"):
        parse_nec4(text.replace(line, new))


# --------------------------------------------------------------------------
# the solve


def _aip_row(capture: str, basis: str):
    """The feed-point row through the portal's DeckSolver — the solve the
    slot's printout is rendered from, without the printout."""
    return _aip_row_of(deck_text(capture), basis)


def _aip_row_of(text: str, basis: str):
    from momwire.portal._portal import DeckSolver, _run_records, parse_deck

    deck = parse_deck(text, dialect="nec4")
    solver = DeckSolver(deck, basis=basis)
    (group,) = [g for g in deck.groups if g is not None]
    records = _run_records(solver, group, group.freqs_mhz[0], 0)
    (row,) = records.aip_rows
    return row


def test_a_gn2_file_name_moves_no_bit_of_the_solve():
    bare = _aip_row_of(_gn2(None), "bspline")
    for name in ("NOFILE", "SOMEX10.NEC"):
        named = _aip_row_of(_gn2(name), "bspline")
        assert named.impedance == bare.impedance, name
        assert named.current == bare.current, name


def test_ex6_and_ex0_are_two_drives_of_one_problem():
    """0223/0224: one antenna, the two source cards.  The licensed engine gave
    the same impedance to every printed digit; here the current source pins
    the source current at the card's value and leaves the impedance the
    voltage drive's own."""
    current = _aip_row("0223", "bspline")
    voltage = _aip_row("0224", "bspline")
    assert current.current == 1.414214 + 0j
    assert voltage.volts == 1.414214 + 0j
    assert current.impedance == pytest.approx(voltage.impedance, rel=1e-12)
    assert current.volts == pytest.approx(current.impedance * 1.414214, rel=1e-12)
    assert current.power == pytest.approx(
        0.5 * 1.414214**2 * voltage.impedance.real, rel=1e-12
    )


@pytest.mark.integration
@pytest.mark.parametrize("basis", ["razor-2p", "razor-nec5", "rzaor", ""])
def test_a_basis_the_slot_cannot_serve_refuses_in_the_printout(basis):
    text = render(deck_text("0224"), basis=basis, dialect="nec4")
    assert " ***** NEC ERROR - " in text
    assert basis_refusal(basis) in text
    assert "ANTENNA INPUT PARAMETERS" not in text
    # The comment box still echoes, so EZNEC shows the refusal rather than
    # discarding the file as stale.
    assert "Written by EZNEC/Pro+ v. 7.0 in NEC-4.2 format." in text


@pytest.mark.integration
def test_a_served_printout_is_the_nec42_layout():
    """Phase 2: the NEC-4.2 banner and sections, not the portal's NEC-2 body
    (the layout itself is gated in test_eznec_nec42_printout_1295.py)."""
    text = render(deck_text("0224"), basis="bspline", dialect="nec4")
    assert "NEC ERROR" not in text
    assert "NUMERICAL ELECTROMAGNETICS CODE (NEC-4.2)" in text
    assert "- - - SEGMENTATION DATA - - -" in text
    assert "- - - ANTENNA INPUT PARAMETERS - - -" in text
    assert "DATA CARD No:" not in text
    assert "Written by EZNEC/Pro+ v. 7.0 in NEC-4.2 format." in text


@pytest.mark.integration
def test_the_nec5_slot_is_the_default_and_unchanged():
    nec5_deck = (Path(__file__).resolve().parent / "fixtures" / "eznec" / "decks").glob(
        "0010_*"
    )
    text = next(nec5_deck).read_bytes().decode("latin-1")
    assert render(text) == render(text, dialect="nec5")
    assert "NUMERICAL ELECTROMAGNETICS CODE (NEC-5)" in render(text)
    # A NEC-5 deck is not a NEC-4.2 one: the nec4 slot reads it in its own
    # dialect, which has no node-addressed source.
    assert "NEC ERROR" in render(text, dialect="nec4")


# --------------------------------------------------------------------------
# the numbers, against the licensed engine's

# Per-deck impedance for the slot's default basis and for point-matched
# sinusoidal, its natural one.  Reported (``-s`` prints the table); the
# assertion is a sanity bar only where a capture is a well-posed comparison —
# see `_SANITY_EXEMPT` for the two that are not.
_SANITY_REL = 0.05

_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:E[-+]\d+)?")

# 0233 is the 300 MHz dipole driven at 13.9 MHz: R is 0.11 ohm under a
# 9 kohm reactance, so a 5 % bar on |Z| is all reactance and R is noise; the
# solve notes call it a poor tolerance comparison.  0239, the buried dipole,
# is not converged at the deck's 11 segments in either code: under refinement
# (Haswell, 2026-10-03) bspline reads 181.8+131.3j, 221.2+110.4j,
# 246.4+85.2j, 261.4+61.0j at 11/21/41/81 segments and sinusoidal
# 148.3+142.7j, 197.8+126.2j, 233.1+100.6j, 254.6+73.9j, the two closing on
# each other.  NEC-4.2's 149.8+143.2j sits on sinusoidal's 11-segment value,
# a shared-basis coincidence at an unconverged mesh, not a target to hold
# bspline to.
_SANITY_EXEMPT = {"0233", "0239"}


def test_the_quoted_targets_are_the_licensed_printouts_own_rows():
    """The notes' figures and the committed printouts are one source: each
    target is the ANTENNA INPUT PARAMETERS impedance of its capture's
    printout, to the printed digits."""
    for capture in CAPTURES:
        (path,) = (FIXTURES / "printouts").glob(f"{capture}_*.out")
        lines = path.read_text(encoding="latin-1").splitlines()
        start = next(
            i for i, ln in enumerate(lines) if "ANTENNA INPUT PARAMETERS" in ln
        )
        row = next(
            cells
            for ln in lines[start + 1 :]
            if len(cells := _NUMBER.findall(ln)) == 11
        )
        z = complex(float(row[6]), float(row[7]))
        assert z == pytest.approx(NEC42_Z[capture], rel=1e-5, abs=1e-4), capture


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("basis", ["bspline", "sinusoidal"])
def test_the_impedance_against_the_licensed_engine(basis, capsys):
    rows = []
    for capture in CAPTURES:
        text = render(deck_text(capture), basis=basis, dialect="nec4")
        if "NEC ERROR" in text:
            refusal = text.split(" ***** NEC ERROR - ", 1)[1].strip()
            rows.append((capture, None, refusal))
            continue
        lines = text.splitlines()
        start = next(
            i for i, ln in enumerate(lines) if "ANTENNA INPUT PARAMETERS" in ln
        )
        # By number, not by whitespace: a negative E12.5 cell fills its field
        # and runs into its neighbour (``9.29044E-03-5.22804E-03``).
        row = next(
            cells
            for ln in lines[start + 1 :]
            if len(cells := _NUMBER.findall(ln)) == 11
        )
        rows.append((capture, complex(float(row[6]), float(row[7])), None))
    with capsys.disabled():
        print(f"\n{basis}: capture, NEC-4.2 Z, momwire Z, |dZ|/|Z|")
        for capture, z, refusal in rows:
            target = NEC42_Z[capture]
            if z is None:
                print(f"  {capture}  {target:.4f}  REFUSED: {refusal}")
            else:
                rel = abs(z - target) / abs(target)
                print(f"  {capture}  {target:.4f}  {z:.4f}  {rel:.4f}")
    for capture, z, refusal in rows:
        if z is None or capture in _SANITY_EXEMPT:
            continue
        target = NEC42_Z[capture]
        assert abs(z - target) / abs(target) < _SANITY_REL, (capture, z, target)
        assert math.isfinite(z.real) and math.isfinite(z.imag)
