"""NX-chained NEC-5 streams — what SimNEC sends when it batches runs.

SimNEC drives a NEC-5-class engine one queued run at a time, until enough
runs are queued that it batches several into one engine call: each run's deck
body, ``NX`` between them, ``EN`` after the last. It then reads the printout
run by run, each run's response ending at the ``INPUT LINE .. NX`` echo. A
multi-frequency or swept model fills the queue, so it is those models that
reach this path (AC6LA's tri-band, 2026-10-07: one frequency worked, three
refused at the ``NX``).

The licensed NEC-5 accepts ``NX`` (observed black-box on a two-structure
deck; layout only is pinned here, no NEC-5 numbers):

  * structure 1 prints in full and ends on ``***** INPUT LINE  n  NX``, ``n``
    numbered on from its own post-``GE`` cards, where a lone deck's ``EN``
    echo stands, with no ``RUN TIME`` after it;
  * the next line is the ``1`` form feed opening the next structure's own
    full header, comment box included, its card echoes numbered from 1;
  * every structure starts from scratch (no frequency, ground or load carries
    across ``NX``), so each is the standalone deck its body makes;
  * the last structure ends on the ``EN`` echo and the stream's one
    ``RUN TIME``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from momwire.deck import DeckError, parse_nec5
from momwire.deck._nec5 import split_structures
from momwire.eznec import _shell
from momwire.eznec._shell import render

EZNEC_DECKS = Path(__file__).parent / "fixtures" / "eznec" / "decks"
BASES = ("bspline", "razor-2p")

_NX_ECHO = re.compile(r"^ \*\*\*\*\* INPUT LINE +(\d+) +NX\b")
_EN_ECHO = re.compile(r"^ \*\*\*\*\* INPUT LINE +(\d+) +EN\b")
_ECHO = re.compile(r"^ \*\*\*\*\* INPUT LINE +(\d+) +([A-Z]{2})\b")
_RUN_TIME = " RUN TIME ="


def _dipole(title: str, mhz: float, *, ground: str = "GE 0", load: str = "") -> str:
    return (
        f"CM {title}\n"
        "CE\n"
        "GW 1 21 0. -2.5 10. 0. 2.5 10. .001\n"
        f"{ground}\n"
        f"FR 0 1 0 0 {mhz}\n"
        f"{load}"
        "EX 0 1 11 0 1. 0.\n"
        "XQ\n"
    )


def _chain(*bodies: str) -> str:
    return "".join(
        b + ("NX\n" if i < len(bodies) - 1 else "EN\n") for i, b in enumerate(bodies)
    )


def _simnec_runs(printout: str) -> list[list[str]]:
    """The printout cut the way SimNEC's NEC-5 reader cuts it: a run ends at
    a line holding INPUT, LINE and NX in that order (or at end of file)."""
    runs: list[list[str]] = [[]]
    for line in printout.splitlines():
        up = line.upper()
        i, j, k = up.find("INPUT"), up.find("LINE"), up.find("NX")
        if 0 <= i < j < k:
            runs.append([])
            continue
        runs[-1].append(line)
    return runs


def _input_rows(lines: list[str]) -> list[str]:
    """Every ANTENNA INPUT PARAMETERS row, verbatim."""
    rows: list[str] = []
    take, in_table = False, 0
    for line in lines:
        if "ANTENNA INPUT PARAMETERS" in line:
            take, in_table = True, 0
            continue
        if take:
            if re.match(r"^\s+\d+\s+\d+\s+\d+\s", line):
                rows.append(line)
                in_table += 1
            elif in_table and not line.strip():
                take = False
    return rows


def _without_run_time(printout: str) -> str:
    return "\n".join(ln for ln in printout.splitlines() if not ln.startswith(_RUN_TIME))


# -- splitting ---------------------------------------------------------------


def test_a_deck_without_nx_is_not_a_chain():
    assert split_structures(_dipole("one", 28.0) + "EN\n") is None


def test_every_eznec_capture_deck_takes_the_one_structure_path():
    # EZNEC terminates every deck with EN and writes no NX, so the chain path
    # must never see one of its decks: the one-structure path, and its bytes,
    # are what EZNEC keeps getting.
    decks = sorted(EZNEC_DECKS.glob("*.nec"))
    assert decks
    for path in decks:
        assert split_structures(path.read_text(encoding="latin-1")) is None, path.name


def test_a_chain_splits_at_each_nx_and_stops_at_en():
    one, two = _dipole("one", 28.0), _dipole("two", 29.0)
    structures = split_structures(_chain(one, two) + "GW junk after EN\n")
    assert structures is not None
    assert [s.terminator for s in structures] == ["NX", "EN"]
    assert structures[0].deck_text == one + "EN\n"
    assert structures[0].echo_text == one + "NX\n"
    assert structures[1].deck_text == two + "EN\n"


def test_a_stream_that_ends_after_nx_is_truncated_and_says_so():
    lines = render(_dipole("one", 28.0) + "NX\n").splitlines()
    # The structure that arrived whole is answered; the one that never came
    # is refused as truncated, exactly as a lone deck without EN is.
    assert sum(bool(_NX_ECHO.match(ln)) for ln in lines) == 1
    assert "NEC ERROR" in lines[-1] and "deck has no EN card" in lines[-1]


def test_parse_nec5_still_reads_one_structure_and_names_the_split():
    with pytest.raises(DeckError, match="split_structures"):
        parse_nec5(_chain(_dipole("one", 28.0), _dipole("two", 29.0)))


# -- each structure is its own standalone deck ---------------------------------


@pytest.mark.parametrize("basis", BASES)
def test_each_chained_structure_answers_as_it_does_alone(basis):
    # Different frequency, ground and load per structure, so a value carried
    # across NX (which the licensed engine does not do) would show.
    bodies = (
        _dipole("one", 28.0),
        _dipole("two", 14.2, ground="GE 1\nGN 1", load="LD 4 1 11 0 10. 5.\n"),
        _dipole("three", 21.1),
    )
    chained = _simnec_runs(render(_chain(*bodies), basis=basis))
    assert len(chained) == len(bodies)
    for run, body in zip(chained, bodies, strict=True):
        alone = render(body + "EN\n", basis=basis).splitlines()
        assert _input_rows(run) == _input_rows(alone)
        assert _input_rows(run)


@pytest.mark.parametrize("basis", BASES)
def test_each_structure_prints_what_it_prints_alone_up_to_its_terminator(basis):
    bodies = (_dipole("one", 28.0), _dipole("two", 29.0))
    chained = render(_chain(*bodies), basis=basis).splitlines()
    cut = next(i for i, ln in enumerate(chained) if _NX_ECHO.match(ln))
    alone = [render(b + "EN\n", basis=basis).splitlines() for b in bodies]
    # Structure 1: the lone printout with its EN echo (and the RUN TIME
    # after it) replaced by the NX echo at the same number.
    en = next(i for i, ln in enumerate(alone[0]) if _EN_ECHO.match(ln))
    assert chained[:cut] == alone[0][:en]
    assert _NX_ECHO.match(chained[cut]).group(1) == _EN_ECHO.match(alone[0][en]).group(
        1
    )
    # Structure 2 is the last: its lone printout, RUN TIME aside.
    assert _without_run_time("\n".join(chained[cut + 1 :])) == _without_run_time(
        "\n".join(alone[1])
    )


# -- the layout of structure 2+ (the licensed engine's, layout only) ----------


def test_the_chained_layout_is_the_licensed_one():
    printout = render(
        _chain(_dipole("dipole one", 28.0), _dipole("dipole two", 29.0))
    ).splitlines()
    nx = [i for i, ln in enumerate(printout) if _NX_ECHO.match(ln)]
    en = [i for i, ln in enumerate(printout) if _EN_ECHO.match(ln)]
    assert len(nx) == 1 and len(en) == 1
    (i,) = nx
    # NX numbered on from structure 1's own post-GE cards (FR, EX, XQ).
    assert _NX_ECHO.match(printout[i]).group(1) == "4"
    # No RUN TIME closes structure 1: the next line is the form feed that
    # opens structure 2's full header, whose comment box is its own.
    assert printout[i + 1] == "1"
    spec = next(
        j for j in range(i, len(printout)) if "STRUCTURE SPECIFICATION" in printout[j]
    )
    header2 = printout[i + 1 : spec]
    assert any(ln.strip() == "dipole two" for ln in header2)
    assert not any(ln.strip() == "dipole one" for ln in header2)
    assert any("NUMERICAL ELECTROMAGNETICS CODE (NEC-5)" in ln for ln in header2)
    # Structure 2's card echoes restart at 1, and EN is numbered from them.
    after = [_ECHO.match(ln) for ln in printout[i + 1 :]]
    numbers = [(m.group(2), int(m.group(1))) for m in after if m]
    assert numbers == [("FR", 1), ("EX", 2), ("XQ", 3), ("EN", 4)]
    # One RUN TIME, last, after the EN echo and one blank.
    run_times = [j for j, ln in enumerate(printout) if ln.startswith(_RUN_TIME)]
    assert run_times == [len(printout) - 1]
    assert printout[-2] == "" and _EN_ECHO.match(printout[-3])
    # Each structure prints its own frequency section.
    assert sum("FREQUENCY=" in ln for ln in printout) == 2


# -- AC6LA's tri-band, as SimNEC batches it ---------------------------------

_DAN_GEOMETRY = (
    "GW 1 2 0.000000e+00 0.000000e+00 1.000000e-01 0.000000e+00 0.000000e+00 0.000000e+00 2.000000e-03\n"
    "GW 2 2 0.000000e+00 0.000000e+00 1.000000e-01 -1.396000e-08 8.000000e-01 8.000000e-01 2.000000e-03\n"
    "GW 3 2 0.000000e+00 0.000000e+00 1.000000e-01 0.000000e+00 0.000000e+00 2.000000e-01 2.000000e-03\n"
    "GW 4 100 -1.396000e-08 8.000000e-01 8.000000e-01 -1.396000e-08 8.000000e-01 1.190000e+01 2.000000e-03\n"
    "GW 5 100 0.000000e+00 0.000000e+00 2.000000e-01 0.000000e+00 0.000000e+00 1.200000e+01 2.000000e-02\n"
    "GW 6 100 -1.396000e-08 8.000000e-01 1.190000e+01 -1.068000e-06 6.122000e+01 1.000000e+01 2.000000e-03\n"
)
# The feed and Dan's two NECSource capacitors, each driven in turn (one XQ
# block per port), as SimNEC measures an N-port.
_DAN_PORTS = ((1, 1), (3, 1), (2, 1))
DAN_MHZ = (1.85, 3.65, 7.15)


def _dan_body(mhz: float) -> str:
    lines = [
        "CM AE6TY wireCurrentsDataFile wireCurrentsDataFile16_",
        "CM useFiles:true",
        "CM UR0GT Tri-Band",
        "CM NEC5",
        "CE",
    ]
    text = "\n".join(lines) + "\n" + _DAN_GEOMETRY + "GE 1\nGN 1\n"
    text += f"FR 0 1 0 0 {mhz:.6f} 1\n"
    for driven in range(len(_DAN_PORTS)):
        for k, (tag, seg) in enumerate(_DAN_PORTS):
            text += f"EX 0 {tag} {seg} 0 {1.0 if k == driven else 1e-10:.6e}\n"
        text += "XQ\n"
    return text


@pytest.mark.parametrize("basis", BASES)
def test_dan_tri_band_returns_every_frequency(basis):
    bodies = [_dan_body(f) for f in DAN_MHZ]
    printout = render(_chain(*bodies), basis=basis)
    assert "NEC ERROR" not in printout
    runs = _simnec_runs(printout)
    assert len(runs) == len(DAN_MHZ)
    for run, body, mhz in zip(runs, bodies, DAN_MHZ, strict=True):
        assert any(f"FREQUENCY= {mhz:.4E} MHZ" in ln for ln in run)
        rows = _input_rows(run)
        assert len(rows) == 3 * len(_DAN_PORTS)  # every port row of every block
        assert rows == _input_rows(render(body + "EN\n", basis=basis).splitlines())


@pytest.mark.integration
def test_the_engine_process_answers_a_chain_through_its_files(tmp_path):
    deck, out = tmp_path / "in.nec", tmp_path / "out.txt"
    deck.write_text(_chain(_dipole("one", 28.0), _dipole("two", 29.0)))
    assert _shell.main([str(deck), str(out)]) == 0
    text = out.read_bytes().decode("latin-1").replace("\r\n", "\n")
    assert len(_simnec_runs(text)) == 2
    assert "NEC ERROR" not in text


# -- a refusal inside a chain ------------------------------------------------


def test_a_refused_structure_ends_the_stream_with_its_own_refusal():
    printout = render(
        _chain(
            _dipole("one", 28.0), _dipole("two", 29.0) + "GM 0 0 0. 0. 0. 0. 0. 0. 0\n"
        )
    )
    lines = printout.splitlines()
    i = next(i for i, ln in enumerate(lines) if _NX_ECHO.match(ln))
    # Structure 1 solved and printed; structure 2's refusal follows with its
    # own header, naming the card.
    assert _input_rows(lines[:i])
    assert lines[i + 1] == "1"
    assert "NEC ERROR" in lines[-1] and "GM" in lines[-1]
    assert not any(ln.startswith(_RUN_TIME) for ln in lines)
