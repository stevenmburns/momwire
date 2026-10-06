"""Suite-wide wiring for the OPT-IN parallel run (momwire#400).

The default `pytest tests` is serial and needs nothing from this file. The
fast spelling is

    pytest tests -n auto --dist loadgroup        (~1m52s vs ~4m41s serial)

and this conftest is what makes it correct as well as fast:

* **Per-worker thread pinning.** The C++ accelerator is built `-fopenmp`,
  so without pinning each xdist worker spawns a full-width OpenMP pool and
  the run measures 486 s — 74% SLOWER than serial (thread contention, not
  parallelism). `OMP_NUM_THREADS` is read at the first parallel region,
  which is safely after `pytest_configure`, so the pin below lands in time
  on every worker. `setdefault`, so an explicit environment wins.

* **Group markers.** Under `--dist loadgroup` everything in one group runs
  sequentially on one worker:
  - the portal tests share server sockets and warm caches across modules —
    measured colliding (4 failures) when a fast parallel run interleaves
    them, and passing serially under the same pinning;
  - the memgate residency gates each hold a deliberately large working set,
    so running them CONCURRENTLY would multiply peaks the budgets were
    never certified against (the 8 GB machine rule). One group ⇒ one at a
    time, whatever `-n` says.

Groups are inert outside xdist: a serial run ignores them entirely.
"""

import os
import sys

import pytest as _pytest


_SOMM_CACHE_TMP: list = []


_WIDTH_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _pin_worker_threads(environ=None, modules=None):
    """One OpenMP/BLAS thread per xdist worker, whatever the caller exported.

    momwire#1108: `OMP_NUM_THREADS=8 make test` used to cost 910 s and trip
    the per-test ceiling 14 times, against 236 s with nothing exported,
    because this pin was `setdefault` and an exported width survived into
    every worker (8 workers x 8 threads). Single-process study scripts still
    want a width exported; only the xdist lane is overridden, and loudly.

    The override works only if it lands BEFORE momwire (and so libgomp) is
    imported in the worker: the OpenMP pool reads its width when the runtime
    loads. If momwire is already imported we cannot fix it from here, so
    fail naming the trap rather than run 4x slow and flag unrelated tests.
    """
    environ = os.environ if environ is None else environ
    modules = sys.modules if modules is None else modules
    exported = {
        k: environ[k] for k in _WIDTH_VARS if environ.get(k, "1").strip() != "1"
    }
    if exported and any(m == "momwire" or m.startswith("momwire.") for m in modules):
        raise _pytest.UsageError(
            f"{exported} is exported and momwire was imported in this xdist "
            "worker before tests/conftest.py could pin one thread (momwire#1108): "
            "the OpenMP pool is already full-width. Unset the variable for the "
            "xdist lane, or run with `-p no:xdist -n0`."
        )
    for k in _WIDTH_VARS:
        environ[k] = "1"


def pytest_configure(config):
    # The time-budget report and duration log (tests/time_budget.py; policy
    # there). Registered here rather than via `pytest_plugins`, which a
    # non-root conftest may not use.
    import time_budget

    if not config.pluginmanager.is_registered(time_budget):
        config.pluginmanager.register(time_budget, "momwire_time_budget")

    # The persistent Sommerfeld grid store (momwire#1224) is OFF for the whole
    # session, and pointed at a throwaway directory besides, so no test reads
    # or writes a user's cache. Off rather than merely relocated, because
    # tests monkeypatch the fill itself (`iv_surfaces_direct_below`,
    # `_use_below_accel`, `_acc`) and clear the in-process caches expecting a
    # refill; a shared store would hand one test's patched fill to the next.
    # The store's own tests turn it on, each against its own tmp_path.
    # `os.environ`, not a fixture, so subprocesses a test spawns inherit it.
    import tempfile

    if not _SOMM_CACHE_TMP:
        _SOMM_CACHE_TMP.append(tempfile.mkdtemp(prefix="momwire-somm-cache-"))
    os.environ["MOMWIRE_SOMM_CACHE"] = "0"
    os.environ["MOMWIRE_SOMM_CACHE_DIR"] = _SOMM_CACHE_TMP[0]

    # xdist worker detection: only workers carry `workerinput`. The
    # controller and any serial run are left untouched, so the serial
    # lanes keep the threading they were certified with.
    if hasattr(config, "workerinput"):
        _pin_worker_threads()


def pytest_unconfigure(config):
    import shutil

    for d in _SOMM_CACHE_TMP:
        shutil.rmtree(d, ignore_errors=True)


# Modules whose tests share portal server state (sockets, cache warmth).
_PORTAL_GROUP_FILES = (
    "test_portal.py",
    "test_portal_shared.py",
    "test_portal_differential.py",
    "test_portal_fixtures.py",
)


# Modules whose SLOW tests share one expensive module-scoped fixture.
#
# Module scope is per-PROCESS, and every xdist worker is its own process — so
# when loadgroup scatters a module's tests across workers, each worker builds
# that module's fixture from scratch. These two build ladders whose own
# docstrings advertise the cost ("Measured ~75 s, which is why every test that
# reads it is `slow`"), and the duplication was visible in the durations: both
# parametrizations of `test_the_ground_adds_no_cross_formulation_gap` paying a
# full setup, 60.9 s + 60.5 s in one module and 38.3 s + 37.8 s in the other.
#
# Measured before this grouping: the two files together ran 102.4 s at `-n 2`
# against 97.6 s at `-n 1` — single-worker was FASTER despite zero parallelism,
# because the duplicated fixture build cost more than the parallelism saved.
#
# ONE GROUP PER MODULE, deliberately, keyed on the file stem rather than a
# single shared name: the point is to stop a module's fixture being built once
# per worker, NOT to serialise these modules against each other. A shared group
# would pin ~200 s of setup onto one worker and make it the critical path.
_FIXTURE_GROUP_FILES = (
    "test_razor_sommerfeld_ground.py",
    "test_razor_refl_coef_ground.py",
    # momwire#838: the sub-1 deg band costs ~2 s to fill, and this module
    # warms it in a module-scoped fixture, so scattering it pays that once per
    # WORKER. Same shape as the two above.
    #
    # `test_below_fills_568.py` joined them at momwire#838 part 2. Before the
    # far annulus existed its one band-touching gate cost ~8 s and was left
    # alone deliberately; with the cap at 4 lambda_m that same gate queries
    # R1 at the cap AND theta at the floor -- the single most expensive cell
    # in the grid -- and ran ~18 s, close enough to the 20 s HARD ceiling on
    # a slow runner to matter. Its module fixture now warms that corner, and
    # this entry is what stops the warm being paid once per worker.
    "test_grazing_band_838.py",
    "test_below_fills_568.py",
    # momwire#935 added a fourth theta band whose nodes cost about twice a
    # mid-band node (7610 tail panels against 3868), and its module fixture
    # warms BOTH grazing bands on two decks. Scattered, each worker pays that
    # fill again -- the same shape as the entry above, with a steeper price.
    "test_grazing_band_lo_935.py",
    # momwire#1064's floor band holds the most expensive nodes in the grid
    # (22,239 tail panels at the floor), and its module fixture warms the
    # inner zone's floor band once. Same shape as the entries above.
    "test_grazing_band_floor_1064.py",
    # momwire#865: the surface-radial anchor's decks are Sommerfeld solves of
    # 5.7 s (N = 4) to 35 s (N = 16), and its assertions share them through a
    # module-level cache. Scattered, each worker re-solved: measured 282 s
    # ungrouped against ~80 s of actual distinct work, with single tests
    # reading 281 s purely from duplication. Same shape as the entries above.
    "test_surface_radials_865.py",
    # momwire#1029 phase 2b: the sector route's swept gate sweeps three
    # wavenumbers of a buried Sommerfeld deck, and a Sommerfeld grid is built
    # per k. Measured on this file's own deck, one thread: 4.1-5.0 s per NEW
    # k, 0.12-0.15 s once cached. Scattered, each worker rebuilt the grids it
    # needed and three tests read 12.5-15.4 s of pure duplication; grouped,
    # the module fixture builds each grid once and every call phase is
    # milliseconds. Same shape as `test_surface_radials_865.py` above.
    "test_rotational_symmetry_1029.py",
    # momwire#1131: the above-ground route's Sommerfeld cases share one grid
    # through a module fixture (~6 s to build on the laptop, one thread).
    "test_route_above_ground_1131.py",
)


# Modules that render the 80-deck EZNEC capture corpus.
#
# `test_eznec_serve.corpus()` / `served()` are `lru_cache`d, and that cache is
# per-PROCESS — so when loadgroup scatters these modules across workers, each
# worker re-renders the decks it needs.  Unlike `_FIXTURE_GROUP_FILES` the
# expensive thing here is shared BETWEEN modules (`test_eznec_reproducibility`
# imports `corpus` from `test_eznec_serve`), so one group per module would
# still pay one full render per module.  Hence a single shared group.
#
# Measured: the two bar tests in `test_eznec_reproducibility` were charged
# 33.7 s + 32.6 s scattered, against 11.29 s + 0.04 s in one process — the
# second is a cache hit.  All ten files together run 39.7 s in ONE process
# against 178.4 s attributed across workers.
#
# The `_FIXTURE_GROUP_FILES` note warns that a shared group can pin ~200 s onto
# one worker and become the critical path.  Measured here it is 40 s, against
# a ~96 s ideal-parallel remainder for everything else, so it is not.
_EZNEC_CORPUS_GROUP_FILES = (
    "test_eznec_serve.py",
    "test_eznec_reproducibility.py",
    "test_eznec_drive_spelling.py",
    "test_eznec_networks.py",
    "test_eznec_one_segment_wire.py",
    "test_eznec_shell.py",
    "test_eznec_printout.py",
    "test_eznec_basis_choice.py",
    "test_eznec_buried_refusal.py",
    "test_razor_nec5_corpus.py",
)


# tryfirst is LOAD-BEARING (momwire#403). xdist's worker hook rewrites each
# grouped item's nodeid to "id@group" in its own pytest_collection_modifyitems,
# and that hook runs BEFORE a conftest's plain hookimpl — so a group marker
# added here without tryfirst is silently ignored and loadgroup degrades to
# plain load: portal tests spread across every worker (measured: all 4, and
# the cliff exactness tests reddened). A file-level pytestmark would also
# work for the portal files, but memgate grouping keys on a MARKER, which
# only a hook can see.
@_pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.get_closest_marker("memgate") is not None:
            item.add_marker(_pytest.mark.xdist_group("memgate"))
        elif item.path.name in _PORTAL_GROUP_FILES:
            item.add_marker(_pytest.mark.xdist_group("portal"))
        elif (
            item.path.name in _FIXTURE_GROUP_FILES
            and item.get_closest_marker("nightly") is None
        ):
            # A `nightly` sweep (momwire#1359) is parametrized one deck per
            # case and fills its own deck's bands, so the module fixture is
            # not what it costs; grouped, its six decks queued on one worker
            # (the floor band's sweep was a 27-minute single item that way).
            item.add_marker(_pytest.mark.xdist_group(item.path.stem))
        elif item.path.name in _EZNEC_CORPUS_GROUP_FILES:
            item.add_marker(_pytest.mark.xdist_group("eznec_corpus"))


# --------------------------------------------------------------------------
# Test time budget: REPORT only, plus a per-test duration log
# --------------------------------------------------------------------------
# The hooks and the policy live in `tests/time_budget.py` (registered in
# pytest_configure above), so a test can load them into a throwaway session.
# In one line: CI wall time never gates. Unmarked tests over 5 s (and over
# 20 s) are NAMED, never failed — runners spread ~3x on the same code — and
# lane moves are decided at release by the MEDIAN over many main runs
# (`scripts/test_durations.py`). Speed regressions are judged by paired
# same-machine runs and the pre-tag sweep, never here.


# --------------------------------------------------------------------------
# Plane sheets on the route gates (momwire#1173 Design E phase 2)
# --------------------------------------------------------------------------
# The modules that hold two routes of the crossing fill to the bit (tiled vs
# untiled, product vs grid, the array memo vs the dict reference, the
# inverted-L's product) run each gate twice through this fixture:
#
#   "exact"    the sheets off: the exact machinery's gate, as it always was.
#   "sheets"   the sheets ON, at a quarter of the shipped build weight (planes
#              of 64 or more rows), so the sheets that nearly pay are taken
#              too -- the radials' plane of hub_deck(16) x2 / x4 (which the
#              shipped rule takes as well) and of the inverted-L and its
#              leaning twin, WA7ARK's height sheets -- and the same routes are
#              held to the bit with most of their rows interpolated. The plan
#              is the FILL's, decided before any cut of its rows, which is
#              what makes that possible; phase 1 decided per call and ran
#              these modules with the sheets off. The smallest decks
#              (crossing_deck(1), the detached and fan decks, the sloped
#              radials) plan no sheet even so and are exact-route gates in
#              both modes; the height sheet's cuts are also gated on a
#              miniature Beverage in `test_plane_sheet_1173`. A fill that
#              planned a sheet and served no row from it fails at teardown,
#              so a green "sheets" row cannot be a sheet that never ran.
@_pytest.fixture(params=["exact", "sheets"])
def sheet_modes(request, monkeypatch):
    ni = _pytest.importorskip("momwire._near_interface")
    if request.param == "exact":
        monkeypatch.setattr(ni, "_SHEET", False)
    else:
        monkeypatch.setattr(ni, "_SHEET_BUILD_WEIGHT", ni._SHEET_BUILD_WEIGHT / 4)
        monkeypatch.setattr(ni, "_SHEET_HEIGHT_WEIGHT", ni._SHEET_HEIGHT_WEIGHT / 4)
        monkeypatch.setattr(ni, "_SHEET_MIN_ROWS", 64)
    stats = dict.fromkeys(ni._SHEET_STATS, 0)
    monkeypatch.setattr(ni, "_SHEET_STATS", stats)
    yield request.param
    planned = stats["planes_planned"] + stats["heights_planned"]
    if request.param == "sheets" and planned and ni._use_sheet():
        assert stats["sheet_rows"] > 0, "a fill planned a sheet and served no row"
