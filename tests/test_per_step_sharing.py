"""What a solver per FR step may share, and what must never leak.

The nec2 launchers build one solver per frequency from one prepared mesh. The
geometry tables, the junction normalisation and the far-field direction dots
do not depend on the frequency, so a step takes the first step's. The gates
here are the ones that make that safe:

* a hit is EQUAL to a fresh build, key by key, on every deck shape that can
  hit (free space, a PEC plane, loads, a TL, a junction with a radius step);
* the shared arrays are read-only, so a writer fails where it writes;
* a deck whose geometry reads something the key does not name (a finite
  ground, wire loading) never shares, and a writable array never hits;
* a second structure, or the same structure with one wire changed, in one
  process, cannot see the first's tables: each printout through the resident
  loop equals the one a fresh process gives.
"""

from __future__ import annotations

import io
import re
import subprocess
import sys

import numpy as np
import pytest

import momwire._far_readout as fr
import momwire._wire_spec as WS
import momwire.sinusoidal as S
from momwire import _nec2_end_cap
from momwire._identity_memo import IdentityMemo
from momwire.deck import build_solver, parse_nec2, prepare_mesh
from momwire.dropin.nec2c import _portal, engine_scope

DIPOLE = "GW 1 21 0 -5.1 10 0 5.1 10 0.001\n"
SWEEP = "EX 0 1 11 0 1 0\nFR 0 4 0 0 14 0.5\nXQ\nNX\n"

DECKS = {
    "free_space": f"CE a\n{DIPOLE}GE 0\n{SWEEP}",
    "pec_plane": f"CE a\n{DIPOLE}GE 1\nGN 1\n{SWEEP}",
    "series_load": f"CE a\n{DIPOLE}GE 0\nLD 0 1 3 3 30 5e-6 0\n{SWEEP}",
    "two_wires_tl": (
        f"CE a\n{DIPOLE}GW 2 21 0 -5.1 13 0 5.1 13 0.001\nGE 0\n"
        f"TL 1 11 2 11 450 0 0 0 0 0\n{SWEEP}"
    ),
    "radius_step_junction": (
        f"CE a\nGW 1 11 0 0 10 -5 0 5 0.001\nGW 2 11 0 0 10 5 0 5 0.002\nGE 0\n{SWEEP}"
    ),
}
# Geometry that reads what the memo key does not name: not shared at all.
UNSHARED = {
    "finite_ground": f"CE a\n{DIPOLE}GE 1\nGN 0 0 0 0 13 0.005\n{SWEEP}",
    "sommerfeld_ground": f"CE a\n{DIPOLE}GE 1\nGN 2 0 0 0 13 0.005\n{SWEEP}",
}

_TIMING = re.compile(r"FILL: +\d+ msec +FACTOR: +\d+ msec")


def _mask(text: str) -> str:
    return _TIMING.sub("FILL: 0 msec  FACTOR: 0 msec", text)


def _geometries_equal(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], np.ndarray):
            assert a[key].dtype == b[key].dtype and np.array_equal(a[key], b[key]), key
        else:
            assert a[key] == b[key], key


def _steps(text: str, basis="sinusoidal"):
    """One solver per frequency, built the portal's way from one mesh."""
    model = parse_nec2(text)
    mesh = prepare_mesh(model)
    with _nec2_end_cap.launcher_default():
        return [
            build_solver(model, basis=basis, frequency_mhz=f, mesh=mesh).solver
            for f in model.groups[0].frequencies
        ]


@pytest.fixture(autouse=True)
def _empty_memos():
    memos = (S._GEOMETRY_MEMO, WS._JUNCTIONS_MEMO)
    for memo in memos:
        memo.clear()
        memo.hits = memo.misses = 0
    yield
    for memo in memos:
        memo.clear()


@pytest.mark.parametrize("name", sorted(DECKS))
def test_a_shared_geometry_is_equal_to_a_fresh_build(name):
    solvers = _steps(DECKS[name])
    tables = [s._build_geometry() for s in solvers]
    # One build for the structure, the other steps served by it.
    assert S._GEOMETRY_MEMO.misses == 1
    assert S._GEOMETRY_MEMO.hits == len(solvers) - 1
    assert all(t is tables[0] for t in tables)
    for s in solvers:
        # `_build_geometry_fresh` is the unshared computation, whatever the
        # memo holds: every step's own answer is the shared one.
        shared = s._cached_geometry
        s._cached_geometry = None
        _geometries_equal(shared, s._build_geometry_fresh())


def test_the_shared_arrays_are_read_only():
    (first, *_rest) = _steps(DECKS["free_space"])
    geom = first._build_geometry()
    arrays = [v for v in geom.values() if isinstance(v, np.ndarray)]
    assert arrays
    for a in arrays:
        assert not a.flags.writeable
    with pytest.raises(ValueError):
        geom["seg_l"][0, 0] = 1.0


@pytest.mark.parametrize("name", sorted(UNSHARED))
def test_a_ground_the_key_does_not_name_never_shares(name):
    solvers = _steps(UNSHARED[name])
    tables = [s._build_geometry() for s in solvers]
    assert S._GEOMETRY_MEMO.hits == 0 and S._GEOMETRY_MEMO.misses == 0
    assert all(t is not tables[0] for t in tables[1:])


def test_wire_loading_never_shares():
    text = f"CE a\n{DIPOLE}GE 0\nLD 5 1 0 0 5.8e7\n{SWEEP}"
    tables = [s._build_geometry() for s in _steps(text)]
    assert S._GEOMETRY_MEMO.hits == 0 and S._GEOMETRY_MEMO.misses == 0
    assert tables[0] is not tables[1]


def test_a_writable_array_never_hits():
    """Only a prepared mesh's frozen polylines are recognised: any other
    caller's arrays are its own to change, and are never served from a memo."""
    wires = [np.array([[0.0, -5.0, 10.0], [0.0, 5.0, 10.0]])]
    a = S.SinusoidalSolver(wires=wires, n_per_edge_per_wire=[[11]], wavelength=20.0)
    b = S.SinusoidalSolver(wires=wires, n_per_edge_per_wire=[[11]], wavelength=21.0)
    assert a._build_geometry() is not b._build_geometry()
    assert S._GEOMETRY_MEMO.hits == 0 and S._GEOMETRY_MEMO.misses == 0


def test_another_structure_misses():
    for text in (
        DECKS["free_space"],
        DECKS["free_space"].replace("5.1 10 0 5.1", "5.1 10 0 5.2"),
    ):
        _steps(text)[0]._build_geometry()
    assert S._GEOMETRY_MEMO.hits == 0 and S._GEOMETRY_MEMO.misses == 2


def test_two_solvers_of_one_mesh_that_differ_in_the_feed_do_not_share():
    model = parse_nec2(DECKS["free_space"])
    mesh = prepare_mesh(model)
    s1 = build_solver(model, basis="sinusoidal", mesh=mesh).solver
    s2 = build_solver(model, basis="sinusoidal", mesh=mesh).solver
    s2.feeds = [(w, 0.25 * 10.2, v) for w, _a, v in s2.feeds]
    g1, g2 = s1._build_geometry(), s2._build_geometry()
    assert g1 is not g2
    assert g1["feed_segs"] != g2["feed_segs"]


def test_normalised_junctions_are_fresh_lists_and_a_refusal_is_not_kept():
    pls = [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    ]
    for pl in pls:
        pl.setflags(write=False)
    counts = [[3], [3]]
    first = WS.normalize_junctions([[(0, "end"), (1, "start")]], pls, counts)
    first[0].append("scribble")
    second = WS.normalize_junctions([[(0, "end"), (1, "start")]], pls, counts)
    assert second == [[(0, "end"), (1, "start")]]
    assert WS._JUNCTIONS_MEMO.hits == 1
    # A refusal (a member that does not sit on its group's point) raises each
    # time and is not stored.
    for _ in range(2):
        with pytest.raises(ValueError):
            WS.normalize_junctions([[(0, "start"), (1, "end")]], pls, counts)
    assert WS._JUNCTIONS_MEMO.hits == 1
    # An iterator is read once, so it is never keyed.
    gen = (g for g in [[(0, "end"), (1, "start")]])
    assert WS.normalize_junctions(gen, pls, counts) == second


def test_direction_dots_are_shared_across_k_and_leave_the_factor_alone():
    fr._PHASE_CACHE.clear()
    fr._DOT_CACHE.clear()
    rng = np.random.default_rng(3)
    mid = rng.normal(size=(40, 3)) + np.array([0.0, 0.0, 3.0])
    moment = rng.normal(size=(40, 3)) + 1j * rng.normal(size=(40, 3))
    theta = np.deg2rad(np.arange(0.0, 91.0, 5.0))
    phi = np.deg2rad(np.arange(0.0, 360.0, 5.0))
    ground = fr.Ground("free")
    ks = [2 * np.pi / 7.5, 2 * np.pi / 7.2]
    warm = [fr._far_moments(mid, moment, k, theta, phi, ground, 0.0, 14e6) for k in ks]
    # Two ks, one set of dots; two factors.
    assert len(fr._DOT_CACHE) == 1 and len(fr._PHASE_CACHE) == 2
    for k, got in zip(ks, warm, strict=True):
        fr._PHASE_CACHE.clear()
        fr._DOT_CACHE.clear()
        cold = fr._far_moments(mid, moment, k, theta, phi, ground, 0.0, 14e6)
        for a, b in zip(got, cold, strict=True):
            assert np.array_equal(a, b)


# -- the resident route -----------------------------------------------------


def _fresh_process(deck: str) -> str:
    done = subprocess.run(
        [
            sys.executable,
            "-W",
            "ignore",
            "-m",
            "momwire.dropin.nec2c",
            "--basis",
            "sinusoidal",
        ],
        input=deck,
        capture_output=True,
        text=True,
        check=True,
    )
    return _mask(done.stdout)


def _resident(*decks: str) -> str:
    """The decks through the resident loop of ONE process, in a single stream,
    as a SimNEC session sends them."""
    out = io.StringIO()
    with engine_scope():
        _portal.main(
            ["--basis", "sinusoidal"],
            stdin=io.StringIO("".join(decks)),
            stdout=out,
            stderr=io.StringIO(),
            prog="momwire-nec2c-sinusoidal",
        )
    return _mask(out.getvalue())


def _frames(text: str) -> list[str]:
    """A printout's deck frames. The banner's VERSION line is printed at
    start-up and again after every NX, so a stream of n decks has n + 1 of
    them and a process that served one deck has two: what lies between a
    consecutive pair is one deck's frame."""
    return text.split("VERSION:")[1:-1]


def test_decks_served_by_one_process_answer_as_a_fresh_process_does():
    """Two different structures, then the first with one wire's end moved, then
    the first again, in one resident process: none may see another's tables.
    Each deck's frame in the stream is the frame a process of its own prints,
    and the sharing really ran (three of every four steps were served)."""
    first = f"CE first\n{DIPOLE}GE 0\n{SWEEP}"
    second = "CE second\nGW 1 21 0 -3.1 10 0 3.1 10 0.001\nGE 0\n" + SWEEP
    edited = first.replace("CE first", "CE edited").replace(
        "0 5.1 10 0.001", "0 5.15 10 0.001"
    )
    assert edited != first
    decks = [first, second, edited, first]
    before = (S._GEOMETRY_MEMO.hits, S._GEOMETRY_MEMO.misses)
    stream = _resident(*decks)
    # 4 decks x 4 frequencies; every deck is its own mesh, so one miss each.
    assert S._GEOMETRY_MEMO.misses - before[1] == 4
    assert S._GEOMETRY_MEMO.hits - before[0] == 12
    frames = _frames(stream)
    assert len(frames) == 4
    for deck, frame in zip(decks, frames, strict=True):
        (alone,) = _frames(_fresh_process(deck))
        assert frame == alone
    # The edit is a different antenna, not a replay of the first.
    assert frames[0] != frames[2]
    assert frames[0] == frames[3]


def test_the_element_map_is_found_by_the_meshs_identity_and_not_over_shared(
    monkeypatch,
):
    """Every step of a deck asks for the same segment-to-element map, and a
    prepared mesh's solvers are recognised by their frozen polylines rather
    than by a copy of every coordinate. A different structure in the same
    DeckSolver (never produced by the portal, but the method must not assume
    it) recomputes."""
    deck = _portal.parse_deck(DECKS["free_space"])
    solver = _portal.DeckSolver(deck, basis="sinusoidal")
    calls = []
    real = solver._element_match_uncached
    monkeypatch.setattr(
        solver,
        "_element_match_uncached",
        lambda s, rows=None: calls.append(s) or real(s, rows),
    )
    model = parse_nec2(DECKS["free_space"])
    mesh = prepare_mesh(model)
    with _nec2_end_cap.launcher_default():
        steps = [
            build_solver(model, basis="sinusoidal", frequency_mhz=f, mesh=mesh).solver
            for f in (14.0, 14.5, 15.0)
        ]
    first = solver._element_match(steps[0])
    assert all(solver._element_match(s) is first for s in steps[1:])
    assert len(calls) == 1
    other = parse_nec2(DECKS["free_space"].replace("5.1 10 0 5.1", "5.1 10 0 5.4"))
    moved = build_solver(
        other, basis="sinusoidal", frequency_mhz=14.0, mesh=prepare_mesh(other)
    ).solver
    assert solver._element_match(moved) is not first
    assert len(calls) == 2


def test_the_memo_is_safe_to_hit_from_several_threads():
    import threading

    memo = IdentityMemo(size=2)
    arrays = [np.zeros(3) for _ in range(6)]
    for a in arrays:
        a.setflags(write=False)
    errors = []

    def work(seed):
        try:
            for i in range(400):
                a = arrays[(seed + i) % len(arrays)]
                assert memo.get((a,), None, lambda a=a: id(a)) == id(a)
        except Exception as exc:  # noqa: BLE001 - the test reports whatever broke
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
