"""The persistent Sommerfeld grid store (momwire#1224, `_somm_disk_cache`).

Two halves:

* **Bit identity through production** (`test_g1_*`): two FRESH processes solve
  the same buried decks through the real solver -- the first against an empty
  store (it fills and writes), the second against the warm one (it must read
  every grid and fill none). Every cached grid's whole state (`grid_digest`)
  and every impedance must be EQUAL, not close. The decks between them reach
  all three families: above/above masters, below/below (with its lazily
  filled band regions) and the below->above transmitted grid.
* **Robustness** (`test_g3_*`): corrupt, truncated, wrong-version and
  wrong-state files are never served and are rebuilt; an unwritable directory
  degrades to in-memory with one warning; racing writers never expose a
  partial file; the env switch turns it all off; eviction is LRU.

The session conftest turns the store OFF for every other test; each test here
turns it on against its own tmp_path.
"""

import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from momwire import _somm_disk_cache as dc
from momwire._sommerfeld import SommerfeldGrid
from momwire._sommerfeld_below import SommerfeldGridBelow

TESTS = Path(__file__).resolve().parent


@pytest.fixture
def store(tmp_path, monkeypatch):
    d = tmp_path / "somm"
    monkeypatch.setenv(dc.ENV_ENABLE, "1")
    monkeypatch.setenv(dc.ENV_DIR, str(d))
    monkeypatch.delenv(dc.ENV_MAX_MB, raising=False)
    monkeypatch.setattr(dc, "_WRITE_WARNED", set())
    for k in dc.STATS:
        monkeypatch.setitem(dc.STATS, k, 0)
    return d


def _synthetic(cls=SommerfeldGrid, n=64, seed=0):
    """A grid-shaped object (no fill): every attribute kind a real grid
    carries -- python float/int/complex, numpy scalar, bool, None, arrays,
    a list of region dicts."""
    rng = np.random.default_rng(seed)
    g = object.__new__(cls)
    g.eps_t = complex(13.0, -12.84)
    g.lattice_eps = complex(13.0, -12.84)
    g.k2 = 2.0 * np.pi
    g.r1_max = 1.5625
    g._r0_fill = 0.0
    g._health = None
    g.k_m = np.complex128(1.0 - 2.0j)
    g._regions = [
        {
            "r0": 0.0,
            "dr": 0.01,
            "n_r": n,
            "th0": np.radians(0.0),
            "dth": np.radians(10.0),
            "n_th": 4,
            "filled": bool(i),
            "vals": rng.normal(size=(4, n, 4)) + 1j * rng.normal(size=(4, n, 4)),
            "r_nodes": np.arange(n) * 0.01,
        }
        for i in range(2)
    ]
    return g


KEY = ("above", complex(13.0, -12.84), 1.5625, complex(13.0, -12.84))


def _path(key=KEY, extra=()):
    name, ident = dc._identity(key, extra)
    return dc.cache_dir() / name, ident


def _filler(obj, calls):
    def fill():
        calls.append(1)
        return obj

    return fill


# ---------------------------------------------------------------------------
# G1: bit identity, two fresh processes, through the production solver
# ---------------------------------------------------------------------------

_CHILD = textwrap.dedent(
    """
    import json, sys, warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, {tests!r})
    from momwire import _somm_disk_cache as dc
    from momwire import _sommerfeld as sm
    from momwire.bspline import BSplineSolver
    from test_buried_serve_553 import served_deck
    from test_crossing_serve_524 import hub_deck

    z = {{}}
    if "served" in {decks!r}:
        z["served"] = complex(served_deck().compute_impedance()[0])
    if "hub" in {decks!r}:
        z["hub"] = complex(
            BSplineSolver(**hub_deck(n_radials=4)).compute_impedance()[0]
        )
    grids = {{repr(k): dc.grid_digest(g) for k, g in sm._NORM_CACHE.items()}}
    grids.update({{
        repr(k): dc.grid_digest(g)
        for k, g in sm._GRID_CACHE.items()
        if k[0] in ("below", "below-above")
    }})
    print(json.dumps({{
        "z": {{k: [v.real.hex(), v.imag.hex()] for k, v in z.items()}},
        "grids": grids,
        "stats": dc.STATS,
    }}))
    """
)


def _child(store_dir, decks, enabled=True):
    env = dict(os.environ)
    env[dc.ENV_DIR] = str(store_dir)
    env[dc.ENV_ENABLE] = "1" if enabled else "0"
    out = subprocess.run(
        [sys.executable, "-c", _CHILD.format(tests=str(TESTS), decks=decks)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def _cold_then_warm(tmp_path, decks):
    d = tmp_path / "somm"
    return _child(d, decks), _child(d, decks)


def test_g1_a_warm_store_is_bit_identical_to_a_cold_fill(tmp_path):
    """The served deck (elevated monopole over a detached buried radial)
    reaches all three families in one solve."""
    cold, warm = _cold_then_warm(tmp_path, ("served",))
    families = {k.split("'")[1] for k in cold["grids"]}  # the regime tag
    assert families == {"above", "below", "below-above"}, families
    _assert_warm_is_cold(cold, warm)


@pytest.mark.integration
def test_g1_the_crossing_hub_deck_too(tmp_path):
    """The crossing serve (radials junctioned to the rise at a buried hub),
    the deck class the momwire#1224 benchmarks time."""
    cold, warm = _cold_then_warm(tmp_path, ("hub",))
    families = {k.split("'")[1] for k in cold["grids"]}
    assert {"above", "below"} <= families, families
    _assert_warm_is_cold(cold, warm)


def _assert_warm_is_cold(cold, warm):
    # The cold process filled every grid and wrote it; the warm one read
    # every grid and filled none -- so the comparison below really is
    # "disk hit vs fresh fill", not two fresh fills.
    assert cold["stats"]["hits"] == 0 and cold["stats"]["misses"] > 0, cold
    assert warm["stats"]["misses"] == 0, warm["stats"]
    assert warm["stats"]["hits"] == cold["stats"]["misses"], (cold, warm)
    assert warm["stats"]["rejected"] == 0

    assert warm["grids"] == cold["grids"]
    assert warm["z"] == cold["z"]


# ---------------------------------------------------------------------------
# G3: robustness
# ---------------------------------------------------------------------------


def test_g3_round_trip_is_exact_and_typed(store):
    g = _synthetic(SommerfeldGridBelow)
    calls = []
    first = dc.fetch_or_fill(KEY, _filler(g, calls))
    assert first is g and calls == [1]
    again = dc.fetch_or_fill(KEY, _filler(g, calls))
    assert calls == [1] and again is not g
    assert type(again) is SommerfeldGridBelow
    assert dc.grid_digest(again) == dc.grid_digest(g)
    # Types survive, not just values: np.float64 stays np.float64.
    assert type(again._regions[0]["th0"]) is type(g._regions[0]["th0"])
    assert type(again.k_m) is np.complex128 and type(again.k2) is float
    assert dc.STATS["hits"] == 1 and dc.STATS["writes"] == 1


@pytest.mark.parametrize("damage", ["flip", "truncate", "empty", "garbage"])
def test_g3_a_damaged_file_is_rebuilt_never_served(store, damage):
    g = _synthetic()
    dc.fetch_or_fill(KEY, _filler(g, []))
    path, _ = _path()
    data = bytearray(path.read_bytes())
    if damage == "flip":
        # Inside the first array's payload, well past the zip header.
        data[len(data) // 3] ^= 0x01
    elif damage == "truncate":
        data = data[: len(data) // 2]
    elif damage == "empty":
        data = bytearray()
    else:
        data = bytearray(b"not a zip file at all" * 10)
    path.write_bytes(bytes(data))

    calls = []
    got = dc.fetch_or_fill(KEY, _filler(g, calls))
    assert calls == [1], "a damaged file was served"
    assert got is g
    assert dc.STATS["rejected"] == 1
    # ... and the refill replaced it with a good one.
    calls = []
    again = dc.fetch_or_fill(KEY, _filler(g, calls))
    assert calls == [] and dc.grid_digest(again) == dc.grid_digest(g)


@pytest.mark.parametrize("field", ["format", "code", "state", "key"])
def test_g3_a_file_from_another_version_is_not_served(store, field):
    """A file at the right NAME whose manifest says another format / code /
    state / key (a hash collision, a hand-copied file) is a miss."""
    g = _synthetic()
    path, ident = _path()
    wrong = dict(ident)
    wrong[field] = 999 if field == "format" else "something else"
    path.parent.mkdir(parents=True)
    path.write_bytes(dc._serialize(g, wrong))
    calls = []
    dc.fetch_or_fill(KEY, _filler(g, calls))
    assert calls == [1] and dc.STATS["rejected"] == 1


def test_g3_new_code_or_state_keys_a_different_file(store, monkeypatch):
    g = _synthetic()
    dc.fetch_or_fill(KEY, _filler(g, []))
    p0, _ = _path()

    # A tuned constant (env override or monkeypatch) is a new state tag.
    from momwire import _sommerfeld

    monkeypatch.setattr(_sommerfeld, "_SOMM_DTH_FAR_DEG", 2.0)
    p1, _ = _path()
    assert p1 != p0
    monkeypatch.undo()

    # A different build of the fill is a new code fingerprint.
    monkeypatch.setattr(dc, "_CODE_FP", ["another build"])
    p2, _ = _path()
    assert p2 != p0
    calls = []
    dc.fetch_or_fill(KEY, _filler(g, calls))
    assert calls == [1]


def test_g3_extra_fill_arguments_are_part_of_the_key(store):
    assert _path(extra=(1e-9,))[0] != _path(extra=(1e-8,))[0]


def test_g3_disabled_by_env_touches_no_disk(store, monkeypatch):
    monkeypatch.setenv(dc.ENV_ENABLE, "0")
    calls = []
    g = _synthetic()
    assert dc.fetch_or_fill(KEY, _filler(g, calls)) is g
    assert dc.fetch_or_fill(KEY, _filler(g, calls)) is g
    assert calls == [1, 1]
    assert not store.exists()
    assert dc.STATS == {"hits": 0, "misses": 0, "writes": 0, "rejected": 0}


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="POSIX permission bits; root ignores them",
)
def test_g3_an_unwritable_dir_degrades_with_one_log_line(store, caplog):
    store.mkdir()
    store.chmod(0o500)
    try:
        caplog.set_level(logging.WARNING, logger="momwire.sommerfeld_cache")
        calls = []
        for seed in range(3):
            g = _synthetic(seed=seed)
            key = KEY[:2] + (float(seed),) + KEY[3:]
            assert dc.fetch_or_fill(key, _filler(g, calls)) is g
        assert calls == [1, 1, 1]
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, [r.getMessage() for r in warnings]
        assert "not writable" in warnings[0].getMessage()
        assert list(store.iterdir()) == []
    finally:
        store.chmod(0o700)


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="POSIX permission bits; root ignores them",
)
def test_g3_a_read_only_store_still_serves_what_it_holds(store):
    g = _synthetic()
    dc.fetch_or_fill(KEY, _filler(g, []))
    store.chmod(0o500)
    try:
        calls = []
        got = dc.fetch_or_fill(KEY, _filler(g, calls))
        assert calls == [] and dc.grid_digest(got) == dc.grid_digest(g)
    finally:
        store.chmod(0o700)


def test_g3_an_uncreatable_dir_degrades(store, monkeypatch, tmp_path):
    blocker = tmp_path / "a-file"
    blocker.write_text("x")
    monkeypatch.setenv(dc.ENV_DIR, str(blocker / "somm"))
    g = _synthetic()
    calls = []
    assert dc.fetch_or_fill(KEY, _filler(g, calls)) is g
    assert dc.fetch_or_fill(KEY, _filler(g, calls)) is g
    assert calls == [1, 1]


_RACER = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {tests!r})
    from momwire import _somm_disk_cache as dc
    import test_somm_disk_cache_1224 as t
    g = t._synthetic(n=4096)
    want = dc.grid_digest(g)
    path, ident = t._path()
    served = 0
    for i in range(40):
        dc._store(path, ident, g)
        got = dc._load(path, ident)
        if got is not None:
            assert dc.grid_digest(got) == want, "a racing writer was served torn"
            served += 1
    assert dc.STATS["rejected"] == 0, dc.STATS
    print(served)
    """
)


def test_g3_racing_writers_never_expose_a_partial_file(store):
    """Four processes write and read ONE key as fast as they can. Every read
    is a whole file: os.replace is atomic, and the temp files are unique."""
    env = dict(os.environ)
    env[dc.ENV_ENABLE] = "1"
    env[dc.ENV_DIR] = str(store)
    code = _RACER.format(tests=str(TESTS))
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", code],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(4)
    ]
    for p in procs:
        out, err = p.communicate(timeout=300)
        assert p.returncode == 0, err[-3000:]
        assert int(out.strip()) > 0
    # Nothing left behind but the one grid (and the binary-digest memo).
    left = sorted(p.name for p in store.iterdir())
    assert left == sorted([_path()[0].name, dc._BINARY_SIDECAR]), left


def test_g3_eviction_is_least_recently_used(store, monkeypatch):
    keys = [KEY[:2] + (float(i),) + KEY[3:] for i in range(3)]
    grids = [_synthetic(seed=i) for i in range(3)]
    dc.fetch_or_fill(keys[0], _filler(grids[0], []))
    size = _path(keys[0])[0].stat().st_size
    dc.fetch_or_fill(keys[1], _filler(grids[1], []))
    p0, p1 = _path(keys[0])[0], _path(keys[1])[0]
    os.utime(p0, (1_000_000, 1_000_000))
    os.utime(p1, (2_000_000, 2_000_000))
    # A hit refreshes the clock: key 0 becomes the most recently used.
    calls = []
    dc.fetch_or_fill(keys[0], _filler(grids[0], calls))
    assert calls == []
    # Room for two files: the third write evicts key 1, the LRU one.
    monkeypatch.setenv(dc.ENV_MAX_MB, str(2.5 * size / (1 << 20)))
    dc.fetch_or_fill(keys[2], _filler(grids[2], []))
    assert p0.exists() and not p1.exists() and _path(keys[2])[0].exists()


def test_g3_the_file_just_written_survives_a_tiny_budget(store, monkeypatch):
    monkeypatch.setenv(dc.ENV_MAX_MB, "0")
    dc.fetch_or_fill(KEY, _filler(_synthetic(), []))
    assert _path()[0].exists()


def test_g3_stale_temp_files_are_swept(store):
    store.mkdir()
    stale = store / ".orphan.npz.123.abc.tmp"
    fresh = store / ".live.npz.456.def.tmp"
    stale.write_bytes(b"x")
    fresh.write_bytes(b"x")
    os.utime(stale, (1_000_000, 1_000_000))
    dc.fetch_or_fill(KEY, _filler(_synthetic(), []))
    assert not stale.exists() and fresh.exists()


def test_g3_a_lazily_filled_grid_is_rewritten(store):
    """The below family fills band regions after the getter returns; the
    file must follow, so the next process starts with what this one did."""
    g = _synthetic(SommerfeldGridBelow)
    got = dc.fetch_or_fill(KEY, _filler(g, []), track=True)
    reg = got._regions[0]
    reg["vals"] = reg["vals"] * 2.0
    reg["filled"] = True
    dc.note_mutated(got)
    path, ident = _path()
    loaded = dc._load(path, ident)
    assert loaded._regions[0]["filled"] is True
    assert dc.grid_digest(loaded) == dc.grid_digest(got)


def test_g3_an_unpersistable_grid_is_kept_in_memory(store):
    g = _synthetic()
    g.extra = object()  # nothing exact to write
    calls = []
    assert dc.fetch_or_fill(KEY, _filler(g, calls)) is g
    assert dc.fetch_or_fill(KEY, _filler(g, calls)) is g
    assert calls == [1, 1] and dc.STATS["writes"] == 0


def test_g3_the_binary_digest_memo_follows_the_file(store, tmp_path):
    import hashlib

    so = tmp_path / "fake.so"
    so.write_bytes(b"one build")
    want = hashlib.sha256(b"one build").hexdigest()
    assert dc._binary_digest(so) == want
    side = store / dc._BINARY_SIDECAR
    assert want in side.read_text()
    # The memo is what a second process reads ...
    assert dc._binary_digest(so) == want
    # ... a rebuild (new bytes, new mtime) is a new digest ...
    so.write_bytes(b"another build!")
    os.utime(so, ns=(10**18, 10**18))
    assert dc._binary_digest(so) == hashlib.sha256(b"another build!").hexdigest()
    # ... and a damaged memo is recomputed, never trusted.
    side.write_text("{not json")
    assert dc._binary_digest(so) == hashlib.sha256(b"another build!").hexdigest()


def test_g3_platform_default_locations(monkeypatch):
    monkeypatch.delenv(dc.ENV_DIR, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", "/la")
    assert dc.cache_dir() == Path("/la") / "momwire" / "Cache" / "sommerfeld"
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", "/xdg")
    assert dc.cache_dir() == Path("/xdg") / "momwire" / "sommerfeld"
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert dc.cache_dir() == Path.home() / ".cache" / "momwire" / "sommerfeld"
    monkeypatch.setenv(dc.ENV_DIR, "/elsewhere")
    assert dc.cache_dir() == Path("/elsewhere")


def test_g3_the_session_store_is_off_and_private():
    """The conftest contract every other test relies on."""
    assert not dc.enabled()
    d = Path(os.environ[dc.ENV_DIR])
    assert d.name.startswith("momwire-somm-cache-")
