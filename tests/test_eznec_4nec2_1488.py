"""momwire#1488: 4nec2 as a front end for the NEC-5 launchers (and, since
momwire#1592, for the NEC-4.2 ones).

4nec2 runs its engine as ``ENGINE.exe < nec2d.tmp``: no arguments, the deck
and printout paths on stdin (``..\\out\\EXAMPLE3.inp``, ``..\\out\\EXAMPLE3.out``),
a deck with no ``EN`` card, and, when the deck's ``GE`` carries a third field
of 1, a ``<printout stem>.GMO`` geometry file expected beside the printout.

The byte oracle is ONE sample: Ward AE6TY's razor-2p port (76.7) on Dan
AC6LA's 4nec2 Example 3 (``tests/fixtures/4nec2_example3/``). The other
geometry gates below pin the conventions this engine chose where that sample
cannot speak (first-appearance node order, no image nodes, the last structure
of a chain); the PR lists them as questions for Ward.

Everything here runs in process. The same contract over real processes and a
real daemon -- the Python thin client, the native launcher, the frozen entry
-- is gated in ``test_eznec_client_c.py`` and ``test_serve_resident.py``.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest

import momwire_eznec_client as eznec_client
import momwire_serve_client as mech
from momwire.deck._nec5 import parse_nec5
from momwire.eznec import _gmo, _serve, _shell

HERE = Path(__file__).resolve().parent
EXAMPLE3 = HERE / "fixtures" / "4nec2_example3"
FIXTURES = HERE / "fixtures"


def _example3_text() -> str:
    return (EXAMPLE3 / "EXAMPLE3.inp").read_bytes().decode("latin-1")


# Example 3 over NO ground: the same geometry -- so the same ``.GMO`` --
# with the Sommerfeld fill (~13 s on a Haswell) taken out, for the default
# lane. The deck as Dan sent it is solved in the slow lane below.
_SOMMERFELD_CARD = b"GN 2 0 0 0 14 .006"


def _free_space_example3() -> bytes:
    raw = (EXAMPLE3 / "EXAMPLE3.inp").read_bytes()
    assert raw.count(_SOMMERFELD_CARD) == 1
    return raw.replace(_SOMMERFELD_CARD, b"GN -1")


def _deck(body: str, ge: str = "GE 0 0 1") -> str:
    """A deck around ``body``'s GW cards, closed by ``EN``: free space, one
    drive on the first wire's first segment, one frequency, one ``XQ``."""
    return f"CM t\nCE\n{body}{ge}\nGN -1\nEX 0 1 1 0 1 0\nFR 0 1 0 0 14\nXQ\nEN\n"


def _gmo_lines(text: str) -> list[str]:
    gmo = _gmo.gmo_text(text)
    assert gmo is not None
    return gmo.splitlines()


def _parsed_gmo(
    text: str,
) -> tuple[list[tuple[float, ...]], list[tuple[int, int, float]]]:
    """``(nodes, elements)`` read back out of a ``.GMO`` file's text."""
    lines = _gmo_lines(text)
    count = int(lines[0])
    nodes = [tuple(float(v) for v in line.split()) for line in lines[1 : 1 + count]]
    elements = []
    for line in lines[2 + count :]:
        a, b, r = line.split()
        elements.append((int(a), int(b), float(r)))
    assert int(lines[1 + count]) == len(elements)
    return nodes, elements


# --------------------------------------------------------------------------
# the reader: GE's third field
# --------------------------------------------------------------------------


def test_ge_third_field_one_asks_for_the_geometry_file():
    assert parse_nec5(_deck("GW 1 3 0 0 0 0 0 1 .001\n")).geometry_file is True


@pytest.mark.parametrize("ge", ["GE 0", "GE 0,-1", "GE 1,-1", "GE 0 0 0"])
def test_ge_without_a_third_field_asks_for_nothing(ge):
    assert parse_nec5(_deck("GW 1 3 0 0 0 0 0 1 .001\n", ge)).geometry_file is False


@pytest.mark.parametrize("value", ["2", "-1", "7"])
def test_any_nonzero_third_field_asks_for_the_file(value):
    """Ward AE6TY's 76.7 writes the file and solves for every nonzero value
    (Windows-builder sitting, 2026-10-15), and this engine matches him."""
    text = _deck("GW 1 3 0 0 0 0 0 1 .001\n", f"GE 0 0 {value}")
    assert parse_nec5(text).geometry_file is True
    assert _gmo.gmo_text(text) == _gmo.gmo_text(
        text.replace(f"GE 0 0 {value}", "GE 0 0 1")
    )


def test_no_captured_deck_asks_for_a_geometry_file():
    """Why the two-argument form is untouched: no EZNEC or SimNEC deck in the
    committed corpus carries a third ``GE`` field, so the sniff that gates
    the whole ``.GMO`` path never fires on one."""
    decks = [
        p
        for p in FIXTURES.rglob("*")
        if p.suffix.lower() == ".nec"
        and "4nec2" not in p.relative_to(FIXTURES).parts[0]
    ]
    assert len(decks) > 80
    asking = [p for p in decks if _gmo.requests_gmo(p.read_bytes().decode("latin-1"))]
    assert asking == []


# --------------------------------------------------------------------------
# the .GMO file: the sample, byte for byte
# --------------------------------------------------------------------------


def test_example3_gmo_is_wards_sample_to_the_byte(tmp_path):
    gmo = _gmo.gmo_text(_shell.close_deck(_example3_text()))
    assert gmo is not None
    out = tmp_path / "EXAMPLE3.GMO"
    _shell.write_gmo(out, gmo)
    assert out.read_bytes() == (EXAMPLE3 / "EXAMPLE3.GMO").read_bytes()


def test_the_gmo_counts_are_the_printouts_counts():
    """The file numbers the nodes the printout's ``Number of nodes`` counts:
    the fused 1 um grid of ``structure_of``, so the two can never disagree."""
    deck = parse_nec5(_shell.close_deck(_example3_text()))
    structure = _serve.structure_of(deck)
    nodes, elements = _parsed_gmo(_shell.close_deck(_example3_text()))
    assert len(nodes) == structure.node_count == 42
    assert len(elements) == structure.wire_element_count == 41


# --------------------------------------------------------------------------
# the conventions the sample cannot pin (questions for Ward)
# --------------------------------------------------------------------------


def test_a_three_wire_junction_is_one_node_numbered_where_it_first_appears():
    """A T: wire 1 along x through the junction's end, wires 2 and 3 leave
    it. The junction is node 3 (wire 1's end 2); wires 2 and 3 start on it
    and add only their own new nodes, in deck order."""
    text = _deck(
        "GW 1 2 -1 0 5 0 0 5 .001\nGW 2 2 0 0 5 1 0 5 .002\nGW 3 2 0 0 5 0 0 4 .003\n"
    )
    nodes, elements = _parsed_gmo(text)
    assert nodes == [
        (-1.0, 0.0, 5.0),
        (-0.5, 0.0, 5.0),
        (0.0, 0.0, 5.0),
        (0.5, 0.0, 5.0),
        (1.0, 0.0, 5.0),
        (0.0, 0.0, 4.5),
        (0.0, 0.0, 4.0),
    ]
    assert elements == [
        (1, 2, 0.001),
        (2, 3, 0.001),
        (3, 4, 0.002),
        (4, 5, 0.002),
        (3, 6, 0.003),
        (6, 7, 0.003),
    ]


def test_a_wire_ending_on_an_earlier_node_numbers_its_new_nodes_first():
    """Wire 2 runs TOWARD wire 1's end: its end 1 and interior are new and
    take the next numbers, and its end 2 is wire 1's node. First appearance,
    end 1 to end 2 -- the convention a coordinate-sorted numbering would
    break (this deck's x falls along wire 2)."""
    text = _deck("GW 1 2 0 0 5 1 0 5 .001\nGW 2 2 3 0 5 1 0 5 .001\n")
    nodes, elements = _parsed_gmo(text)
    assert nodes == [
        (0.0, 0.0, 5.0),
        (0.5, 0.0, 5.0),
        (1.0, 0.0, 5.0),
        (3.0, 0.0, 5.0),
        (2.0, 0.0, 5.0),
    ]
    assert elements == [(1, 2, 0.001), (2, 3, 0.001), (4, 5, 0.001), (5, 3, 0.001)]


def test_a_wire_on_a_ground_plane_adds_no_image_nodes_or_elements():
    """A monopole standing on ``GE 1`` ground: its base is one node of the
    structure, and the image the solver sees is not in the file."""
    text = (
        "CM t\nCE\nGW 1 4 0 0 0 0 0 2 .005\nGE 1 0 1\nGN 1\n"
        "EX 0 1 1 0 1 0\nFR 0 1 0 0 30\nXQ\nEN\n"
    )
    nodes, elements = _parsed_gmo(text)
    assert len(nodes) == 5 and len(elements) == 4
    assert nodes[0] == (0.0, 0.0, 0.0)
    assert all(z >= 0.0 for _, _, z in nodes)


def test_each_element_carries_its_own_wires_radius():
    text = _deck("GW 1 1 0 0 1 0 0 2 1.5e-3\nGW 2 1 0 0 2 0 0 3 2.25e-2\n")
    _, elements = _parsed_gmo(text)
    assert [r for _, _, r in elements] == [1.5e-3, 2.25e-2]
    assert _gmo_lines(text)[-1] == "      2      3  2.25000E-02"


def test_an_nx_chain_writes_the_last_structure_that_asks():
    first = "CM a\nCE\nGW 1 2 0 0 0 0 0 1 .001\nGE 0 0 1\nGN -1\nEX 0 1 1 0 1 0\nFR 0 1 0 0 14\nXQ\nNX\n"
    second = "CM b\nCE\nGW 1 3 0 0 0 0 0 3 .002\nGE 0 0 1\nGN -1\nEX 0 1 1 0 1 0\nFR 0 1 0 0 14\nXQ\nEN\n"
    nodes, _ = _parsed_gmo(first + second)
    assert len(nodes) == 4 and nodes[-1] == (0.0, 0.0, 3.0)
    quiet_second = second.replace("GE 0 0 1", "GE 0 0 0")
    nodes, _ = _parsed_gmo(first + quiet_second)
    assert len(nodes) == 3


def test_a_deck_that_asks_for_nothing_gets_no_file():
    assert _gmo.gmo_text(_deck("GW 1 3 0 0 0 0 0 1 .001\n", "GE 0,-1")) is None


def test_a_deck_that_does_not_parse_gets_no_file():
    assert _gmo.gmo_text(_deck("GW 1 0 0 0 0 0 0 1 .001\n")) is None


# --------------------------------------------------------------------------
# EN: optional for the stdin form, and only there
# --------------------------------------------------------------------------


def test_close_deck_adds_en_to_a_deck_without_one():
    assert _shell.close_deck("CM\r\nCE\r\nXQ\r\n") == "CM\r\nCE\r\nXQ\r\nEN\n"
    assert _shell.close_deck("CM\nCE\nXQ") == "CM\nCE\nXQ\nEN\n"


@pytest.mark.parametrize(
    "text",
    [
        "CM\nCE\nXQ\nEN\n",
        "CM\nCE\nXQ\n en\n",
        # a chain that stops after a structure reads as the stream form reads
        # it, with no empty structure added after its NX (momwire#1698)
        "CM\nCE\nXQ\nNX\n",
        "CM\nCE\nXQ\nNX\n\n  \n",
    ],
)
def test_close_deck_leaves_a_deck_with_en_alone(text):
    assert _shell.close_deck(text) == text


def test_the_two_argument_form_closes_a_deck_without_en_at_end_of_file(tmp_path):
    # momwire#1698: the licensed NEC-5 closes a deck at end of file, so the
    # file form does too, whether its names came from stdin or the command line.
    deck, out = tmp_path / "EXAMPLE3.nec", tmp_path / "EXAMPLE3.out"
    deck.write_bytes(_free_space_example3())
    assert b"\nEN" not in deck.read_bytes()
    assert _shell.main([str(deck), str(out)]) == 0
    printout = out.read_text(encoding="latin-1")
    assert "NEC ERROR" not in printout
    assert "ANTENNA INPUT PARAMETERS" in printout


def test_the_deck_reader_itself_still_refuses_a_deck_without_en():
    # Only the file form closes the text; the stream form's decks (#1512) and
    # the reader stay strict, since EN is what ends one streamed deck.
    printout = _shell.render(_example3_text())
    assert "deck has no EN card" in printout


# --------------------------------------------------------------------------
# the shared mechanics: stdin, the stem rule, the wire frames
# --------------------------------------------------------------------------


class _Pipe(io.BytesIO):
    def isatty(self) -> bool:
        return False


class _Terminal(io.BytesIO):
    def isatty(self) -> bool:
        return True


def test_stdin_paths_strips_crlf_and_whitespace():
    stream = _Pipe(b"  ..\\out\\EXAMPLE3.inp\r\n..\\out\\EXAMPLE3.out \r\nextra\r\n")
    assert mech.stdin_paths(stream) == [
        "..\\out\\EXAMPLE3.inp",
        "..\\out\\EXAMPLE3.out",
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b"", []),
        (b"deck.inp\r\n", ["deck.inp"]),
        (b"deck.inp", ["deck.inp"]),
        (b"\r\nout.out\r\n", []),
    ],
)
def test_stdin_paths_returns_what_arrived(raw, expected):
    assert mech.stdin_paths(_Pipe(raw)) == expected


@pytest.mark.parametrize(
    ("line", "path"),
    [
        ('"..\\out\\x.inp"\r\n', "..\\out\\x.inp"),
        ('  "my files\\x y.inp"  \r\n', "my files\\x y.inp"),
        ('"x.inp\r\n', '"x.inp'),
        ('x.inp"\r\n', 'x.inp"'),
        ('"\r\n', '"'),
        ('""\r\n', ""),
        ('""x.inp""\r\n', '"x.inp"'),
    ],
)
def test_one_pair_of_surrounding_quotes_is_stripped(line, path):
    """Ward's port strips surrounding double quotes from a ``.tmp`` line; a
    quoted path made every momwire launcher write nothing at all before this
    (momwire#1488). One pair only, spaces inside kept, and a lone or
    mismatched quote left for the OS to refuse by name."""
    assert mech.strip_path_line(line) == path


def test_stdin_paths_reads_quoted_names():
    stream = _Pipe(b'"..\\out\\EX 3.inp"\r\n"..\\out\\EX 3.out"\r\n')
    assert mech.stdin_paths(stream) == ["..\\out\\EX 3.inp", "..\\out\\EX 3.out"]


def test_stdin_paths_never_reads_a_terminal():
    assert mech.stdin_paths(_Terminal(b"deck.inp\nout.out\n")) is None


@pytest.mark.parametrize(
    ("printout", "gmo"),
    [
        ("..\\out\\EXAMPLE3.out", "..\\out\\EXAMPLE3.GMO"),
        ("../out/EXAMPLE3.out", "../out/EXAMPLE3.GMO"),
        ("EXAMPLE3.OUT", "EXAMPLE3.GMO"),
        ("a.b.c", "a.b.GMO"),
        ("noext", "noext.GMO"),
        ("dir.d/noext", "dir.d/noext.GMO"),
        (".out", ".out.GMO"),
        ("..out", "..out.GMO"),
        ("dir/...a.b", "dir/...a.GMO"),
    ],
)
def test_the_gmo_path_is_the_printout_stem(printout, gmo):
    assert mech.gmo_path(printout) == gmo
    assert _gmo.gmo_path(printout) == gmo


def test_a_backslash_separates_only_where_the_os_says_so():
    expected = "dir.d\\noext.GMO" if os.name == "nt" else "dir.GMO"
    assert mech.gmo_path("dir.d\\noext") == expected


def test_the_answer_frame_round_trips_and_an_unframed_answer_is_the_printout():
    printout = " printout\r\n"
    gmo = "     1\r\n  0.00000E+00  0.00000E+00  0.00000E+00\r\n     0\r\n"
    framed = _shell.framed(printout.replace("\r\n", "\n"), gmo.replace("\r\n", "\n"))
    wire = framed.replace("\n", "\r\n").encode("latin-1")
    assert mech.split_answer(wire) == (gmo.encode(), printout.encode())
    assert mech.split_answer(printout.encode()) == (None, printout.encode())
    assert _shell.framed("x\n", None) == "x\n"
    with pytest.raises(ValueError, match="never closed"):
        mech.split_answer(mech.GMO_FRAME_START + b"     1\r\n")


def test_the_seam_reads_the_stdin_forms_request_line():
    text, en_optional = _shell._request(
        mech.EN_OPTIONAL_REQUEST.decode("latin-1") + "CM\nCE\n"
    )
    assert (text, en_optional) == ("CM\nCE\n", True)
    assert _shell._request("CM\nCE\n") == ("CM\nCE\n", False)


# --------------------------------------------------------------------------
# end to end, in process: the shell's stdin form and the thin client's
# --------------------------------------------------------------------------


def _room(tmp_path: Path, deck: bytes | None = None) -> tuple[Path, Path]:
    """4nec2's layout: the engine runs in ``bin``, the files sit in ``out``.
    The deck is Example 3 over no ground unless ``deck`` says otherwise."""
    engine_dir = tmp_path / "bin"
    out_dir = tmp_path / "out"
    engine_dir.mkdir()
    out_dir.mkdir()
    (out_dir / "EXAMPLE3.inp").write_bytes(
        _free_space_example3() if deck is None else deck
    )
    return engine_dir, out_dir


_NAMES = b"..\\out\\EXAMPLE3.inp\r\n..\\out\\EXAMPLE3.out\r\n"
if os.name != "nt":
    # The same two relative names as POSIX spells them: a backslash is a
    # filename character there, as the OS reads it.
    _NAMES = _NAMES.replace(b"\\", b"/")


def _stdin(monkeypatch, raw: bytes) -> None:
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(_Pipe(raw)))


def _impedance(printout: bytes) -> complex:
    """The ANTENNA INPUT PARAMETERS row's impedance, read off the printout:
    the row ends V, I, Z, Y (re/im pairs) and the power, so Z is the fifth
    and fourth numbers from the end."""
    lines = printout.decode("latin-1").splitlines()
    for k, line in enumerate(lines):
        if "ANTENNA INPUT PARAMETERS" in line:
            for row in lines[k + 1 :]:
                fields = row.split()
                if len(fields) >= 11 and fields[0].isdigit():
                    return complex(float(fields[-5]), float(fields[-4]))
    raise AssertionError("no ANTENNA INPUT PARAMETERS row")


@pytest.mark.parametrize("basis", [_serve.BASIS, "razor-2p"])
def test_the_shell_reads_4nec2s_names_from_stdin(tmp_path, monkeypatch, capsys, basis):
    """No arguments, the two relative names on stdin with CRLF, a deck with
    no EN: the printout and the sample's ``.GMO`` land beside the deck, and
    the printout is the one the deck closed by EN would have had."""
    engine_dir, out_dir = _room(tmp_path)
    monkeypatch.chdir(engine_dir)
    _stdin(monkeypatch, _NAMES)

    assert _shell.main([], basis=basis) == 0

    assert capsys.readouterr().out == ""
    assert (out_dir / "EXAMPLE3.GMO").read_bytes() == (
        EXAMPLE3 / "EXAMPLE3.GMO"
    ).read_bytes()
    printout = (out_dir / "EXAMPLE3.out").read_bytes()
    assert b"NEC ERROR" not in printout
    closed = tmp_path / "closed.nec"
    closed.write_bytes(_free_space_example3() + b"EN\r\n")
    _shell.run(closed, tmp_path / "closed.out", basis=basis)
    assert printout == (tmp_path / "closed.out").read_bytes()
    assert list(engine_dir.iterdir()) == []


@pytest.mark.slow
@pytest.mark.parametrize("basis", [_serve.BASIS, "razor-2p"])
def test_dans_example3_over_sommerfeld_ground_end_to_end(tmp_path, monkeypatch, basis):
    """The deck exactly as Dan sent it, Sommerfeld ground and all: the
    sample's ``.GMO``, and on razor-2p the impedance Ward's port of the same
    formulation printed, 61.409 + j18.768 ohms (measured here 61.406 +
    j18.765; his port builds its own Sommerfeld grid, ours blends a cached
    one, and the two are documented to differ at this level)."""
    engine_dir, out_dir = _room(tmp_path, (EXAMPLE3 / "EXAMPLE3.inp").read_bytes())
    monkeypatch.chdir(engine_dir)
    _stdin(monkeypatch, _NAMES)

    assert _shell.main([], basis=basis) == 0

    assert (out_dir / "EXAMPLE3.GMO").read_bytes() == (
        EXAMPLE3 / "EXAMPLE3.GMO"
    ).read_bytes()
    z = _impedance((out_dir / "EXAMPLE3.out").read_bytes())
    if basis == "razor-2p":
        assert abs(z - complex(61.409, 18.768)) < 0.01, z
    else:
        # bs2 answers in its own basis; pinned so a change is noticed.
        assert abs(z - complex(61.834, 21.469)) < 0.01, z


def test_the_shell_with_one_name_on_stdin_is_the_one_argument_fault(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    _stdin(monkeypatch, b"deck.inp\r\n")
    assert _shell.main([]) == 0
    assert capsys.readouterr().out.splitlines() == [_shell.ARGUMENT_ERROR_OUTPUT]
    assert list(tmp_path.iterdir()) == []


def test_the_shell_with_nothing_on_stdin_is_the_empty_command_line(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    _stdin(monkeypatch, b"")
    assert _shell.main([]) == 0
    assert capsys.readouterr().out.splitlines() == [_shell.ARGUMENT_ERROR_INPUT]


def test_the_shell_reads_quoted_names_with_spaces(tmp_path, monkeypatch, capsys):
    """The quoted form, with a space inside each name: the deck is found and
    both files land where the quotes said."""
    engine_dir = tmp_path / "bin"
    out_dir = tmp_path / "my out"
    engine_dir.mkdir()
    out_dir.mkdir()
    (out_dir / "EX 3.inp").write_bytes(_free_space_example3())
    monkeypatch.chdir(engine_dir)
    sep = os.sep.encode()
    names = b'"..' + sep + b"my out" + sep + b'EX 3.inp"\r\n'
    names += b'  "..' + sep + b"my out" + sep + b'EX 3.out"  \r\n'
    _stdin(monkeypatch, names)

    assert _shell.main([]) == 0

    assert capsys.readouterr().out == ""
    assert b"NEC ERROR" not in (out_dir / "EX 3.out").read_bytes()
    assert (out_dir / "EX 3.GMO").read_bytes() == (
        EXAMPLE3 / "EXAMPLE3.GMO"
    ).read_bytes()


def test_a_mismatched_quote_on_the_deck_is_refused_in_the_printout(
    tmp_path, monkeypatch, capsys
):
    """``"deck.inp`` names a file that is not there: the refusal names it in
    the printout, which is written where the second line says."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "deck.inp").write_bytes(_free_space_example3())
    _stdin(monkeypatch, b'"deck.inp\r\ndeck.out\r\n')
    assert _shell.main([]) == 0
    printout = (tmp_path / "deck.out").read_bytes().decode("latin-1")
    assert "NEC ERROR" in printout and 'UNABLE TO READ INPUT FILE "deck.inp' in printout


def test_an_unwritable_printout_is_named_on_stdout_never_silent(
    tmp_path, monkeypatch, capsys
):
    """The printout path cannot be opened (its folder does not exist, as a
    path with a stray quote does not on Windows): there is no file to refuse
    in, so the failure is named on stdout. Before momwire#1488's fix this
    exited 0 having said nothing anywhere."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "deck.inp").write_bytes(_free_space_example3())
    bad = os.path.join('"no such dir', "deck.out")
    _stdin(monkeypatch, b"deck.inp\r\n" + bad.encode() + b"\r\n")
    assert _shell.main([]) == 0
    assert capsys.readouterr().out.splitlines() == [f"{_shell.UNABLE_TO_OPEN} {bad}"]


def test_the_thin_clients_last_ditch_is_never_silent(tmp_path, capsys):
    bad = str(tmp_path / "no such dir" / "x.out")
    eznec_client._last_ditch(bad, "MOMWIRE ENGINE UNAVAILABLE - test")
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"{eznec_client.ARGUMENT_ERROR_INPUT} {bad}"
    assert "MOMWIRE ENGINE UNAVAILABLE - test" in out[1]


NEC42_DECK = FIXTURES / "eznec_nec42" / "0224_dipole-ex0.nec"


def test_the_nec42_slot_reads_4nec2s_names_from_stdin(tmp_path, monkeypatch, capsys):
    """4nec2 drives its NEC-4.2 engine as it drives NEC-5, ``ENGINE <
    nec2d.tmp`` with a deck that has no ``EN`` (momwire#1592, Dan AC6LA):
    the NEC-4.2 slot reads the names and closes the deck as the NEC-5 slot
    does, and writes no ``.GMO`` (4nec2 asks none of its NEC-4.2 engine)."""
    raw = NEC42_DECK.read_bytes()
    no_en = raw[: raw.rindex(b"EN")]
    engine_dir = tmp_path / "bin"
    out_dir = tmp_path / "out"
    engine_dir.mkdir()
    out_dir.mkdir()
    (out_dir / "EXAMPLE3.inp").write_bytes(no_en)
    monkeypatch.chdir(engine_dir)
    _stdin(monkeypatch, _NAMES)

    assert _shell.main([], basis="sinusoidal", dialect="nec4") == 0

    assert capsys.readouterr().out == ""
    printout = (out_dir / "EXAMPLE3.out").read_bytes().decode("latin-1")
    expected = _shell.render(
        raw.decode("latin-1"), basis="sinusoidal", dialect="nec4"
    ).replace("\n", "\r\n")
    assert "NEC ERROR" not in printout
    assert "NEC-4.2" in printout
    assert _unclocked(printout) == _unclocked(expected)
    assert sorted(p.name for p in out_dir.iterdir()) == ["EXAMPLE3.inp", "EXAMPLE3.out"]


def _unclocked(printout: str) -> str:
    """The printout without this engine's two measured times."""
    import re

    printout = re.sub(r"FILL= *\d+\.\d+ SEC\.", "FILL= - SEC.", printout)
    return re.sub(r"RUN TIME = *\d+\.\d+", "RUN TIME = -", printout)


def test_the_two_argument_form_writes_the_gmo_too(tmp_path):
    """The file follows the deck, not the form: the same deck with an EN,
    named on the command line, gets the same ``.GMO``."""
    deck = tmp_path / "EXAMPLE3.nec"
    deck.write_bytes(_free_space_example3() + b"EN\r\n")
    _shell.run(deck, tmp_path / "EXAMPLE3.out", basis="razor-2p")
    assert (tmp_path / "EXAMPLE3.GMO").read_bytes() == (
        EXAMPLE3 / "EXAMPLE3.GMO"
    ).read_bytes()


def _in_process_server(monkeypatch, basis: str) -> list[bytes]:
    """The thin client's resident rung answered by the seam itself, through
    the server's own codec (latin-1, CRLF): what a daemon would put on the
    wire, minus the socket."""
    sent: list[bytes] = []

    def served(deck_bytes, _basis, _idle, dialect="nec5"):
        sent.append(deck_bytes)
        body = deck_bytes.decode("latin-1").replace("\r\n", "\n")
        out, _ = _shell.seam(basis=basis, dialect=dialect).answer(body.rstrip("\n"), "")
        return out.replace("\n", "\r\n").encode("latin-1")

    monkeypatch.setattr(eznec_client, "_served_bytes", served)
    return sent


@pytest.mark.parametrize(
    ("name", "basis"),
    [("momwire-nec5", _serve.BASIS), ("momwire-nec5-razor-2p", "razor-2p")],
)
def test_the_thin_client_reads_4nec2s_names_from_stdin(
    tmp_path, monkeypatch, capsys, name, basis
):
    engine_dir, out_dir = _room(tmp_path)
    monkeypatch.chdir(engine_dir)
    _stdin(monkeypatch, _NAMES)
    monkeypatch.setattr("sys.argv", [str(engine_dir / name)])
    sent = _in_process_server(monkeypatch, basis)

    assert eznec_client.main(None) == 0

    assert capsys.readouterr().out == ""
    assert len(sent) == 1 and sent[0].startswith(mech.EN_OPTIONAL_REQUEST)
    assert (out_dir / "EXAMPLE3.GMO").read_bytes() == (
        EXAMPLE3 / "EXAMPLE3.GMO"
    ).read_bytes()
    oracle = tmp_path / "oracle.out"
    _shell.run(out_dir / "EXAMPLE3.inp", oracle, basis=basis, en_optional=True)
    assert (out_dir / "EXAMPLE3.out").read_bytes() == oracle.read_bytes()


def test_the_thin_client_reads_4nec2s_names_from_stdin_for_nec42(
    tmp_path, monkeypatch, capsys
):
    """The NEC-4.2 launcher name reads the same form (momwire#1592) and asks
    the NEC-4.2 slot, with the stdin allowance on the wire; no ``.GMO``."""
    engine_dir = tmp_path / "bin"
    out_dir = tmp_path / "out"
    engine_dir.mkdir()
    out_dir.mkdir()
    raw = NEC42_DECK.read_bytes()
    (out_dir / "EXAMPLE3.inp").write_bytes(raw[: raw.rindex(b"EN")])
    monkeypatch.chdir(engine_dir)
    _stdin(monkeypatch, _NAMES)
    monkeypatch.setattr("sys.argv", [str(engine_dir / "momwire-nec42-sinusoidal")])
    sent = _in_process_server(monkeypatch, "sinusoidal")

    assert eznec_client.main(None) == 0

    assert capsys.readouterr().out == ""
    assert len(sent) == 1 and sent[0].startswith(mech.EN_OPTIONAL_REQUEST)
    printout = (out_dir / "EXAMPLE3.out").read_bytes().decode("latin-1")
    assert "NEC ERROR" not in printout and "NEC-4.2" in printout
    assert sorted(p.name for p in out_dir.iterdir()) == ["EXAMPLE3.inp", "EXAMPLE3.out"]


def test_the_thin_clients_two_argument_form_sends_the_deck_verbatim(
    tmp_path, monkeypatch
):
    deck = tmp_path / "d.nec"
    deck.write_bytes(_deck("GW 1 3 0 0 0 0 0 1 .001\n", "GE 0,-1").encode())
    monkeypatch.setattr(
        "sys.argv", ["momwire-nec5", str(deck), str(tmp_path / "d.out")]
    )
    sent = _in_process_server(monkeypatch, _serve.BASIS)
    assert eznec_client.main(None) == 0
    assert sent == [deck.read_bytes()]
    assert not (tmp_path / "d.GMO").exists()
