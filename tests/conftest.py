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

import pytest as _pytest


def pytest_configure(config):
    # xdist worker detection: only workers carry `workerinput`. The
    # controller and any serial run are left untouched, so the serial
    # lanes keep the threading they were certified with.
    if hasattr(config, "workerinput"):
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


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
        elif item.path.name in _FIXTURE_GROUP_FILES:
            item.add_marker(_pytest.mark.xdist_group(item.path.stem))
        elif item.path.name in _EZNEC_CORPUS_GROUP_FILES:
            item.add_marker(_pytest.mark.xdist_group("eznec_corpus"))


# --------------------------------------------------------------------------
# Test time-budget guardrail (ported from antennaknobs' #393 hook)
# --------------------------------------------------------------------------
# The PR lane decides how many edit->signal turns an hour this repo supports,
# and #692's marker pass got it to 2:31 by moving 403 integration tests to a
# push-only lane. Nothing kept it there: the count held across the next two
# merges because those authors marked their new tests, not because anything
# checked. This is the check.
#
# An unmarked test whose CALL phase breaches the ceiling gets named in a loud
# terminal section. It reports by default and does not fail, because absolute
# call times drift with hardware and a gate that reddens on a busy laptop is
# a gate people learn to ignore; set MOMWIRE_ENFORCE_TIME_BUDGET=1 (CI, once
# the numbers are calibrated) to also fail the run.
#
# The four exempt markers are the four lanes that are not the edit loop.
# `integration` is exempt for a reason worth stating: it is the one marker
# here that is NOT about duration — 306 of #692's 403 integration tests run
# under 3 s. A test can be exempt because it is slow OR because it is not an
# edit-loop guard, and those are different claims.
# TWO thresholds, because a report nobody reads stops working and a gate that
# reddens on a busy laptop is a gate people learn to ignore.
#
#   CEILING (5 s)      REPORTS. The suite carries 24 unmarked tests over it
#                      today — #692's convergence ladders, left in the lane on
#                      purpose. Naming them every run beats rediscovering them.
#   HARD_CEILING (20 s) FAILS, always, no opt-in. Nothing has to be remembered
#                      for the case that actually matters: a genuinely new slow
#                      test cannot land whether or not anyone read the report.
#
# 20 s and not 10 s, which is where this nearly landed. The worst unmarked test
# measures 9.44 s / 9.47 s on two runs here — reproducible, and a 10 s gate
# looks like it clears. It does not: this repo's own `test` job took 2m36s and
# 4m03s on two runs of the SAME commit, a 1.55x runner spread, which puts that
# test at ~14.6 s on a slow runner. A hard gate must sit above the worst case
# times the observed variance, not above the best measurement — otherwise it
# reddens when nothing is wrong, and a gate people learn to ignore is worse
# than no gate.
#
# The gap between the two is the point. Call times drift with load, so a 2 s
# test can touch 5 s on a busy machine and get REPORTED; reaching 20 s means
# something genuinely changed.
TIME_BUDGET_CEILING_S = float(os.environ.get("MOMWIRE_TIME_BUDGET_CEILING_S", "5.0"))
TIME_BUDGET_HARD_CEILING_S = float(
    os.environ.get("MOMWIRE_TIME_BUDGET_HARD_CEILING_S", "20.0")
)
_TIME_BUDGET_EXEMPT_MARKERS = ("slow", "memgate", "crossgate", "integration")
_time_budget_offenders: list[tuple[str, float]] = []


def _is_xdist_worker(config) -> bool:
    return hasattr(config, "workerinput")


def pytest_runtest_logreport(report):
    # Under xdist this fires on the workers AND on the controller, which is
    # handed every worker's reports — so the controller's list is the whole
    # run and the summary below (controller-only) sees all of it.
    # Collect above the LOWER of the two. Normally that is the soft ceiling
    # and this reads as you expect, but keying it to the soft one alone means
    # a tuned `HARD < SOFT` silently disables the hard gate — nothing reaches
    # the check because nothing was collected. Found by a malformed test of
    # this very hook, which is the good way to find it.
    if report.when != "call" or report.duration <= min(
        TIME_BUDGET_CEILING_S, TIME_BUDGET_HARD_CEILING_S
    ):
        return
    if any(m in report.keywords for m in _TIME_BUDGET_EXEMPT_MARKERS):
        return
    _time_budget_offenders.append((report.nodeid, report.duration))


def pytest_terminal_summary(terminalreporter):
    if not _time_budget_offenders:
        return
    tr = terminalreporter
    tr.section("test time-budget guardrail", sep="!", red=True, bold=True)
    tr.line(
        f"{len(_time_budget_offenders)} unmarked test(s) over the "
        f"{TIME_BUDGET_CEILING_S:.0f}s ceiling:"
    )
    for nodeid, dur in sorted(_time_budget_offenders, key=lambda x: -x[1]):
        over_hard = " <-- OVER HARD CEILING" if dur > TIME_BUDGET_HARD_CEILING_S else ""
        tr.line(f"  {dur:6.2f}s  {nodeid}{over_hard}")
    if _over_hard_ceiling():
        tr.line("")
        tr.line(
            f"FAILING: {len(_over_hard_ceiling())} test(s) over the "
            f"{TIME_BUDGET_HARD_CEILING_S:.0f}s HARD ceiling. This is not "
            f"advisory."
        )
    tr.line(
        "Fix: make it faster, or mark it — `slow` (push lane), `integration` "
        "(crosses a process/socket/CLI/printout seam), `crossgate`/`memgate` "
        "(certification). See the pyproject markers."
    )


def _over_hard_ceiling() -> list[tuple[str, float]]:
    return [x for x in _time_budget_offenders if x[1] > TIME_BUDGET_HARD_CEILING_S]


def pytest_sessionfinish(session, exitstatus):
    # Controller only: a worker's exitstatus does not reach the caller, and
    # the controller is where the whole run's offenders are.
    if _is_xdist_worker(session.config):
        return
    if _over_hard_ceiling():
        session.exitstatus = _pytest.ExitCode.TESTS_FAILED
    elif _time_budget_offenders and os.environ.get("MOMWIRE_ENFORCE_TIME_BUDGET"):
        # Opt-in strict mode: fail on the SOFT ceiling too, for whenever the
        # 24-test debt is paid down and the tighter standard can be held.
        session.exitstatus = _pytest.ExitCode.TESTS_FAILED
