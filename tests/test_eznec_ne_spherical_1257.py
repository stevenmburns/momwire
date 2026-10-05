"""``NE 1`` / ``NH 1`` — the near field on a spherical grid (momwire#1257).

Dan (AC6LA) asked for ``NE 1`` in the ``momwire-nec5-*`` engines so he can
read the near field at high elevation over a Sommerfeld ground. The nec5
dialect used to refuse it with a sentence that was wrong twice: EZNEC does
write ``NE 1`` (when its user enters the grid in spherical form), and so does
SimNEC's ``NearField("p", …)``.

``tests/fixtures/eznec_ne_spherical_1257/`` (README there) holds two decks we
wrote and the licensed NEC-5's printouts of them. The gates:

  * the CONVENTION — ``(R, θ, φ)``, θ from the zenith, φ from +x, R fastest
    then θ then φ — and the printed table's Cartesian point columns, digit
    for digit against the licensed printouts;
  * the LAYOUT — the whole printout from the structure heading down, numbers
    masked, is the licensed one on both decks and on two bases;
  * the VALUES — on razor-2p, NEC-5's formulation twin, every cell inside a
    stated envelope, with the rows that are excluded named and the reason
    for each asserted where the reason is the oracle's;
  * a point BELOW a finite ground refuses by name, over a perfect ground it
    does not;
  * an ``NE 1`` deck goes end to end through the engine's own file seam
    (and, in the integration lane, the resident server EZNEC and SimNEC call).
"""

from __future__ import annotations

import functools
import io
import math
import re
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import momwire_serve_client as mech
from momwire._near_readout import cos_sin_deg
from momwire.deck import DeckError, Nec5NearFieldRequest, parse_nec5
from momwire.eznec import _resident, _serve, _shell
from momwire.eznec._shell import render
from momwire.serve._server import Server
from test_serve_resident import _ask, short_room, transport  # noqa: F401 - fixtures

FIXTURES = Path(__file__).parent / "fixtures" / "eznec_ne_spherical_1257"
# The ground deck fills a Sommerfeld table, which is most of a second alone
# and several under a loaded xdist run, so its gates ride the integration lane
# (the deck->printout seam); the free-space deck's gate every PR.
FREE = "dipole-ne1-free"
GROUND = pytest.param("dipole-ne1-ground", marks=pytest.mark.integration)
DECKS = (FREE, GROUND)
TWIN = "razor-2p"
_NUMBER = re.compile(r"[-+]?\d+(\.\d+)?(E?[-+]\d+)?")


def _deck(name: str) -> str:
    return (FIXTURES / f"{name}.nec").read_text()


@functools.lru_cache(maxsize=None)
def _served(name: str, basis: str = TWIN) -> str:
    """One render per deck and basis per worker: the ground deck's Sommerfeld
    table is most of a second to fill."""
    return render(_deck(name), basis=basis)


def _oracle(name: str) -> list[str]:
    lines = (FIXTURES / f"{name}.out").read_text(encoding="latin-1").splitlines()
    # The licensed engine's own ground-table cache chatter (README): a blank,
    # then two lines, which momwire does not print.
    out, i = [], 0
    while i < len(lines):
        if not lines[i].strip() and i + 1 < len(lines) and "GMPINO" in lines[i + 1]:
            i += 3
            continue
        out.append(lines[i])
        i += 1
    return out


def _mask(line: str) -> str:
    return " ".join("#" if _NUMBER.fullmatch(t) else t for t in line.split())


def _from_structure(lines: list[str]) -> list[str]:
    i = next(i for i, ln in enumerate(lines) if "STRUCTURE SPECIFICATION" in ln)
    return [_mask(ln) for ln in lines[i:]]


def _near_block(lines: list[str]) -> tuple[list[str], list[str]]:
    """The five header lines and the rows of the one near-field table."""
    i = next(i for i, ln in enumerate(lines) if "- - - NEAR " in ln)
    rows = []
    for ln in lines[i + 5 :]:
        if not ln.strip():
            break
        rows.append(ln)
    return lines[i : i + 5], rows


def _row(line: str) -> tuple[np.ndarray, np.ndarray]:
    f = [float(t) for t in line.split()]
    field = np.array(
        [f[3 + 2 * c] * np.exp(1j * np.radians(f[4 + 2 * c])) for c in range(3)]
    )
    return np.array(f[:3]), field


# -- the convention -------------------------------------------------------


def test_ne_1_parses_every_field_and_nh_1_is_its_twin():
    head = _deck("dipole-ne1-free").split("\nNE ")[0] + "\n"
    (electric,) = parse_nec5(
        head + "NE 1 2 7 3 2.0 0.0 0.0 8.0 15.0 45.0\nEN\n"
    ).requests
    assert electric == Nec5NearFieldRequest(
        coordinates=1,
        counts=(2, 7, 3),
        origin=(2.0, 0.0, 0.0),
        step=(8.0, 15.0, 45.0),
    )
    (magnetic,) = parse_nec5(
        head + "NH 1 2 7 3 2.0 0.0 0.0 8.0 15.0 45.0\nEN\n"
    ).requests
    assert magnetic.coordinates == 1 and magnetic.magnetic


@pytest.mark.parametrize("mnemonic", ["NE", "NH"])
@pytest.mark.parametrize("system", [2, -1])
def test_any_other_coordinate_system_refuses_with_the_two_it_reads(mnemonic, system):
    head = _deck("dipole-ne1-free").split("\nNE ")[0] + "\n"
    with pytest.raises(DeckError) as excinfo:
        parse_nec5(head + f"{mnemonic} {system} 1 1 1 10.0 30.0 60.0 0 0 0\nEN\n")
    message = str(excinfo.value)
    assert message.startswith(f"{mnemonic} coordinate system {system} ")
    assert "rectangular (0" in message and "spherical (1" in message
    # The old sentence said EZNEC only writes the rectangular form. It does not.
    assert "EZNEC emits" not in message


@pytest.mark.parametrize(
    ("card", "point"),
    [
        # Single points read off a licensed printout of a deck we wrote: the
        # second field is theta FROM THE ZENITH, the third the azimuth from +x.
        ("NE 1 1 1 1 10.0 0.0 0.0 0 0 0", (0.0, 0.0, 10.0)),
        ("NE 1 1 1 1 10.0 90.0 0.0 0 0 0", (10.0, 0.0, 0.0)),
        ("NE 1 1 1 1 10.0 90.0 90.0 0 0 0", (0.0, 10.0, 0.0)),
        ("NE 1 1 1 1 10.0 30.0 60.0 0 0 0", (2.5, 4.3301, 8.6603)),
        ("NE 1 1 1 1 10.0 60.0 30.0 0 0 0", (7.5, 4.3301, 5.0)),
        ("NE 1 1 1 1 10.0 120.0 10.0 0 0 0", (8.5287, 1.5038, -5.0)),
        ("NE 1 1 1 1 10.0 -30.0 0.0 0 0 0", (-5.0, 0.0, 8.6603)),
        ("NE 1 1 1 1 10.0 200.0 0.0 0 0 0", (-3.4202, 0.0, -9.3969)),
        ("NE 1 1 1 1 10.0 10.0 400.0 0 0 0", (1.3302, 1.1162, 9.8481)),
    ],
)
def test_the_spherical_point_is_the_licensed_engines(card, point):
    head = _deck("dipole-ne1-free").split("\nNE ")[0] + "\n"
    (request,) = parse_nec5(head + card + "\nEN\n").requests
    (got,) = _serve._grid_points(request)
    np.testing.assert_allclose(got, point, atol=5e-5 * 10.0)


def test_the_walk_is_r_fastest_then_theta_then_phi():
    """``NE 1,2,3,2,10.,0.,0.,5.,30.,45.`` on the licensed engine: twelve
    rows, R flipping every row, theta every two, phi every six."""
    head = _deck("dipole-ne1-free").split("\nNE ")[0] + "\n"
    card = "NE 1 2 3 2 10.0 0.0 0.0 5.0 30.0 45.0"
    (request,) = parse_nec5(head + card + "\nEN\n").requests
    got = _serve._grid_points(request)
    want = []
    for phi in (0.0, 45.0):
        for theta in (0.0, 30.0, 60.0):
            for r in (10.0, 15.0):
                t, p = np.radians(theta), np.radians(phi)
                want.append(
                    (
                        r * np.sin(t) * np.cos(p),
                        r * np.sin(t) * np.sin(p),
                        r * np.cos(t),
                    )
                )
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_an_axis_angle_places_an_exact_zero():
    """90 degrees puts the point ON the horizon, z = 0.0 exactly; the licensed
    engine prints ~2.6e-14 of R there (see the coordinate gate below)."""
    for angle, want in ((0.0, (1.0, 0.0)), (90.0, (0.0, 1.0)), (180.0, (-1.0, 0.0))):
        assert cos_sin_deg(angle) == want
    assert cos_sin_deg(-90.0) == (0.0, -1.0)
    assert cos_sin_deg(450.0) == (0.0, 1.0)


# -- the printout ---------------------------------------------------------


@pytest.mark.parametrize("basis", [TWIN, _serve.BASIS])
@pytest.mark.parametrize("name", DECKS)
def test_the_layout_is_the_licensed_one(name, basis):
    ours = _from_structure(_served(name, basis).splitlines())
    assert ours == _from_structure(_oracle(name))


@pytest.mark.parametrize("name", DECKS)
def test_the_table_stays_cartesian_and_its_points_are_the_licensed_ones(name):
    """The five header lines byte for byte (X, Y, Z in METERS on a spherical
    grid too), and every LOCATION cell to the printed digit.

    The one exception is a coordinate the geometry makes exactly zero: we
    print ``0.0000E+00`` there and the licensed engine prints ~2.6e-14 of R
    (``2.6485E-13`` at R = 10 m), so those cells are held to 1e-12 of R."""
    ours_head, ours = _near_block(_served(name).splitlines())
    theirs_head, theirs = _near_block(_oracle(name))
    assert ours_head == theirs_head
    assert "      METERS      METERS      METERS" in ours_head[4]
    assert len(ours) == len(theirs)
    zeros = 0
    for a, b in zip(ours, theirs, strict=True):
        pa, pb = _row(a)[0], _row(b)[0]
        r = float(np.linalg.norm(pb))
        for ca, cb, va, vb in zip(a[:38].split(), b[:38].split(), pa, pb, strict=True):
            if ca == cb:
                continue
            assert va == 0.0 and abs(vb) <= 1e-12 * r, (a, b)
            zeros += 1
    assert zeros > 0  # the exception is exercised, not vacuous


def _range(point: np.ndarray) -> int:
    """Which R shell of the grid a printed point is on (10 m or 1000 m here);
    the printed coordinates carry five digits, so R itself is not a key."""
    return round(math.log10(float(np.linalg.norm(point))))


# The value envelopes, on razor-2p (NEC-5's formulation twin), as the vector
# difference over the table's own scale at that range. The scale is the
# largest licensed |E| among the rows the envelope covers.
_ENVELOPE = {"dipole-ne1-free": 1e-3, "dipole-ne1-ground": 1e-2}


def _excluded(point: np.ndarray) -> str | None:
    """Why a ground-deck row is left out of the envelope, or ``None``.

    One kind, and it is not a tolerance question: the licensed engine's own
    high-elevation defect. Within a few degrees of the zenith at 1000 m its
    near field is hundreds of times its own far field (asserted in the test
    below).

    There used to be a second: momwire's own near field at 1000 m within
    ~15 degrees of the horizon, off the broadside plane, where the ground
    remainder was frozen at the edge of the 15-wavelength Sommerfeld table.
    momwire#1258 continues the remainder past the edge and those four rows
    are in the envelope now (``test_the_grazing_rows_are_in_the_envelope``).
    """
    r = float(np.linalg.norm(point))
    if r < 100.0:
        return None
    elevation = np.degrees(np.arcsin(point[2] / r))
    if 82.0 < elevation < 90.0 - 1e-9:
        return "x13-zenith"
    return None


@pytest.mark.parametrize("name", DECKS)
def test_the_values_sit_inside_the_envelope(name):
    _, ours = _near_block(_served(name).splitlines())
    _, theirs = _near_block(_oracle(name))
    rows = [(_row(a), _row(b)) for a, b in zip(ours, theirs, strict=True)]
    kept = [(o, t) for o, t in rows if _excluded(t[0]) is None]
    scales: dict[int, float] = {}
    for _, (point, field) in kept:
        r = _range(point)
        scales[r] = max(scales.get(r, 0.0), float(np.linalg.norm(field)))
    worst = max(
        np.linalg.norm(of - tf) / scales[_range(tp)] for (_, of), (tp, tf) in kept
    )
    assert worst <= _ENVELOPE[name]
    if name == "dipole-ne1-free":
        assert len(kept) == len(rows)


def test_the_grazing_rows_are_in_the_envelope():
    """The four rows #1257 had to leave out (R = 1000 m, off the broadside
    plane, within 15 degrees of the horizon) are kept now, and each sits
    within 1 % of the licensed |E| at its own point, not just of the table's
    scale (momwire#1258; before it, the horizon row was 58x too large)."""
    _, ours = _near_block(_served("dipole-ne1-ground").splitlines())
    _, theirs = _near_block(_oracle("dipole-ne1-ground"))
    grazing = []
    for a, b in zip(ours, theirs, strict=True):
        (tp, tf), (_, of) = _row(b), _row(a)
        r = float(np.linalg.norm(tp))
        elevation = np.degrees(np.arcsin(tp[2] / r))
        if r > 100.0 and elevation < 16.0 and abs(tp[1]) > 1e-6 * r:
            assert _excluded(tp) is None
            grazing.append(np.linalg.norm(of - tf) / np.linalg.norm(tf))
    assert len(grazing) == 4
    assert max(grazing) < 1e-2


@pytest.mark.integration
def test_the_licensed_zenith_defect_is_what_the_exclusion_says():
    """At R = 1000 m, theta = 5 degrees, the licensed near field is ~290x
    the field at the zenith one row away; momwire's moves by 0.05 %."""
    _, ours = _near_block(_served("dipole-ne1-ground").splitlines())
    _, theirs = _near_block(_oracle("dipole-ne1-ground"))
    by_point = {}
    for a, b in zip(ours, theirs, strict=True):
        (tp, tf), (_, of) = _row(b), _row(a)
        by_point[tuple(np.round(tp, 1))] = (np.linalg.norm(of), np.linalg.norm(tf))
    zenith = by_point[(0.0, 0.0, 1000.0)]
    cone = by_point[(87.2, 0.0, 996.2)]
    assert cone[1] > 100.0 * zenith[1]
    assert abs(cone[0] / zenith[0] - 1.0) < 1e-2


# -- below a finite ground ------------------------------------------------


def _ground_deck(card: str, ground: str = "GN 2 0 0 0 13.0 0.005 0 0 NOFILE") -> str:
    head = _deck("dipole-ne1-ground").split("\nNE ")[0] + "\n"
    head = re.sub(r"(?m)^GN .*$", ground, head)
    return head + card + "\nEN\n"


@pytest.mark.parametrize(
    "card",
    [
        "NE 1 1 5 1 1000.0 90.0 0.0 0 22.5 0",  # theta past 90: into the soil
        "NH 1 1 3 1 10.0 170.0 0.0 0 5.0 0",
        "NE 0 1 1 1 5.0 0.0 -1.0 0 0 0",  # the rectangular form, same rule
    ],
)
def test_a_point_below_a_finite_ground_refuses_by_name(card):
    reason = _serve.refusal(parse_nec5(_ground_deck(card)))
    assert reason is not None
    assert "below this deck's finite ground" in reason
    assert reason.startswith(("NE (near electric field)", "NH (near magnetic field)"))


def test_a_point_below_a_perfect_ground_is_served():
    """Both engines print the same direct-plus-image sum there."""
    text = _ground_deck("NE 1 1 5 1 1000.0 90.0 0.0 0 22.5 0", ground="GN 1")
    assert _serve.refusal(parse_nec5(text)) is None
    assert "NEAR ELECTRIC FIELDS" in render(text, basis=TWIN)


def test_the_horizon_is_not_below_the_ground():
    """z = 0.0 exactly (the axis-exact trig) sits ON the interface, which is
    the limit from above and is served."""
    text = _ground_deck("NE 1 1 1 1 1000.0 90.0 0.0 0 0 0")
    assert _serve.refusal(parse_nec5(text)) is None


# -- end to end -----------------------------------------------------------


def test_an_ne_1_deck_goes_through_the_engines_file_seam(tmp_path):
    """``engine <deck> <printout>``, exit 0, the printout on disk carrying the
    spherical grid's Cartesian table."""
    deck = tmp_path / "EZN5.NEC"
    deck.write_text(_deck("dipole-ne1-free"))
    out = tmp_path / "NEC5.OUT"
    assert _shell.main([str(deck), str(out)], basis=TWIN) == 0
    written = out.read_bytes().decode("latin-1").replace("\r\n", "\n")
    assert "NEC ERROR" not in written
    head, rows = _near_block(written.splitlines())
    assert len(rows) == 2 * 7 * 3
    assert written == render(_deck("dipole-ne1-free"), basis=TWIN)


@pytest.fixture
def razor_server(short_room, transport):  # noqa: F811 - the imported fixtures
    """The resident server the ``momwire-nec5-razor-2p`` client talks to."""
    path = str(short_room / f"srv{mech.address_suffix(transport)}")
    log = io.StringIO()

    def connection(conn, number, log_stream, solve_lock):
        _resident._connection(conn, number, log_stream, solve_lock, basis=TWIN)

    server = Server(path, idle_timeout=3600.0, log=log, connection=connection)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        probe = mech.connect(path)
        if probe is None:
            time.sleep(0.01)
            continue
        probe.close()
        break
    yield path
    server.stop()
    thread.join(timeout=10)


@pytest.mark.integration
@pytest.mark.parametrize("name", DECKS)
def test_the_resident_server_answers_an_ne_1_deck(razor_server, tmp_path, name):
    deck = tmp_path / "in.nec"
    deck.write_text(_deck(name))
    oracle = tmp_path / "oracle.out"
    _shell.run(deck, oracle, basis=TWIN)
    answer = _ask(razor_server, deck.read_bytes())
    assert answer == oracle.read_bytes()
    assert b"NEAR ELECTRIC FIELDS" in answer and b"NEC ERROR" not in answer
