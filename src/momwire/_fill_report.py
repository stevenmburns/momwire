"""What a solver's fill did: `fill_report()` (momwire#2016).

One read-only method on every dense solver family, returning plain data that
`json.dumps` takes. It keeps two savings apart, because they are different:

* **Pair reuse** (`pair_reuse`) copies matrix entries, so it shortens the fill
  only.
* **The sector route** (`rotational_symmetry=True`) fills one sector's rows and
  factors one block per harmonic, so it shortens the factor too.

THE SHAPE::

    {
      "version": 1,
      "solver": "RazorSolver",
      "fills": 1,                 # fills this solver has run
      "frequencies": 1,           # frequencies those fills covered
      "times": {"fill_s", "factor_s", "census_s"},
      "route": "dense",           # None before a fill; "sector", "hmatrix",
                                  # "array-block", "cxx"
      "reuse": None | {
        "state": "served" | "partial" | "declined",
        "declined": None | str,   # why it declined, or what a partial left out
        "pairs", "computed", "reused", "near", "kernel_evals",
        "families": {"direct": {...}, "image": {...}, "remainder": {...}},
        "uncounted": [...],       # families this fill has but cannot count
        "plan_bytes", "census_peak_bytes",
        "wires": None | {"labels", "pairs", "computed", "reused", "near"}
                      | {"unavailable": str},
      },
      "sector": None | {
        "n_sectors", "axial", "blocks", "computed_pairs", "pairs", "declined",
        "wire_sector",
      },
    }

WHAT THE NUMBERS MEAN.

`times` and `fills` are TOTALS since the solver was built, so a sweep's cost is
its sum and a second solve adds to the first. `fill_s` is the whole fill's wall
time and includes `census_s`, the part of it spent finding what repeats: a
census paid once and repaid only by later fills is why reuse can slow a single
solve. `factor_s` is the LU routines' wall time on the solving thread,
back-substitution included.

Everything else describes the LAST fill, which on a sweep is the last
frequency. `route`, `reuse` and `sector` are what that fill actually served,
never what a plan offered: a fill that read no plan reads "declined" here even
if an earlier fill was served. On a batched sweep, one fill covers a chunk of
frequencies and `frequencies` counts them.

`computed + reused + near == pairs` holds per family, per wire block and in
total. `computed` pairs were evaluated, `reused` were copied from an evaluated
one, and `near` are the pairs too close to share, evaluated one by one.
`kernel_evals` is what the kernel was actually called on per frequency, which
exceeds `computed + near` where a call evaluates a rectangle around the pairs
it needs. `reused` counts entries, so a ratio of `pairs / (computed + near)` is
the compression, not the time saved.

THE WIRE MAP (`fill_report(wires=False)` skips it) is built when it is read,
from the plans the process already holds, so a solve never pays for it. A plan
the process's cache has dropped reads `{"unavailable": reason}` rather than
being rebuilt: the report never runs a census.

Nothing here runs unless its option is on: no reuse analysis without
`pair_reuse`, and no symmetry probe without `rotational_symmetry=True`.

A deck refused by `rotational_symmetry=True` refuses at construction, so there
is no solver to ask: :func:`refused_report` gives the same shape from the
exception.
"""

from __future__ import annotations

import copy
import time
import weakref

from . import _lapack

REPORT_VERSION = 1

#: The reason given when a fill asked for reuse and its route read no plan.
NO_PLAN_READ = "this fill's route reads no reuse plan"

#: The wire map's answer once a plan has left the process's caches.
PLAN_GONE = "the reuse plan is no longer held; solve again to read the map"


class FillLedger:
    """One solver's running record. Small, and written only at the fill seams
    and the factor stage: never read by a fill."""

    __slots__ = (
        "fills",
        "frequencies",
        "fill_s",
        "factor_s",
        "census_s",
        "depth",
        "route",
        "reuse",
        "pending",
        "mark",
        "last",
        "sector_rows",
        "sector_segs",
        "sector_blocks",
    )

    def __init__(self):
        self.fills = 0
        self.frequencies = 0
        self.fill_s = 0.0
        self.factor_s = 0.0
        self.census_s = 0.0
        self.depth = 0
        self.route = None
        self.reuse = None  # the last fill's: what `fill_report` reads
        self.pending = None  # the running fill's notes
        self.mark = None
        self.last = None  # what the last served fill's wire map reads
        self.sector_rows = None
        self.sector_segs = None
        self.sector_blocks = []

    def __getstate__(self):
        # A pickled or deep-copied solver keeps its totals; the plans it
        # pointed at (weak references) belong to this process.
        return {
            "fills": self.fills,
            "frequencies": self.frequencies,
            "fill_s": self.fill_s,
            "factor_s": self.factor_s,
            "census_s": self.census_s,
            "route": self.route,
        }

    def __setstate__(self, state):
        self.__init__()
        for name, value in state.items():
            setattr(self, name, value)


def ledger_of(solver):
    """The solver's ledger, made on first use."""
    ledger = solver.__dict__.get("_fill_ledger")
    if ledger is None:
        ledger = solver.__dict__["_fill_ledger"] = FillLedger()
    return ledger


class fill_clock:
    """``with fill_clock(solver):`` around one fill. Fills nest (a restricted
    fill calls the dense one), and only the outermost counts.

    A context manager, so that a seam inside a method adds no stack frame."""

    __slots__ = ("solver", "ledger", "t0", "frequencies", "cancelled")

    def __init__(self, solver, frequencies=1):
        self.solver = solver
        self.ledger = ledger_of(solver)
        self.frequencies = frequencies
        self.cancelled = False

    def __enter__(self):
        ledger = self.ledger
        ledger.depth += 1
        if ledger.depth == 1:
            _lapack.set_factor_sink(ledger)
            ledger.pending = None
            ledger.sector_blocks = []
            ledger.mark = self.solver._fill_report_begin()
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        ledger = self.ledger
        ledger.depth -= 1
        if ledger.depth == 0:
            if exc_type is None and not self.cancelled:
                ledger.fill_s += time.perf_counter() - self.t0
                ledger.fills += 1
                ledger.frequencies += self.frequencies
                self.solver._fill_report_end(ledger)
            # The running fill's notes can hold a plan strongly: past the fill
            # only the process's caches may keep one alive.
            ledger.pending = None
        return False


class fill_time:
    """``with fill_time(solver):`` around work that belongs to the fill but
    runs ahead of it (razor2p's k-independent prepare): its seconds count
    toward `fill_s`, and it is not a fill of its own."""

    __slots__ = ("ledger", "t0")

    def __init__(self, solver):
        self.ledger = ledger_of(solver)

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.ledger.depth == 0:
            self.ledger.fill_s += time.perf_counter() - self.t0
        return False


def add_stage_times(solver, fill_s, factor_s, frequencies=1):
    """A fill and factor measured elsewhere (the razor2p C++ route's own
    clocks), counted as one fill."""
    ledger = ledger_of(solver)
    ledger.fill_s += float(fill_s)
    ledger.factor_s += float(factor_s)
    ledger.fills += 1
    ledger.frequencies += frequencies


def timed_census(solver, build):
    """`build` with its seconds counted as census time."""

    def timed():
        t0 = time.perf_counter()
        try:
            return build()
        finally:
            ledger_of(solver).census_s += time.perf_counter() - t0

    return timed


def note_sector_rows(solver, rows, n_segs):
    """The sector route's fill: `rows` observer segments against `n_segs`
    sources."""
    ledger = ledger_of(solver)
    ledger.sector_rows = int(rows)
    ledger.sector_segs = int(n_segs)


def note_sector_block(solver, size):
    """One block the sector route factored."""
    ledger_of(solver).sector_blocks.append(int(size))


def family(pairs, computed, near, kernel_evals):
    """One family's counts; `reused` is what the others leave."""
    pairs, computed, near = int(pairs), int(computed), int(near)
    return {
        "pairs": pairs,
        "computed": computed,
        "reused": pairs - computed - near,
        "near": near,
        "kernel_evals": int(kernel_evals),
    }


def reuse_entry(
    state,
    declined=None,
    families=None,
    uncounted=(),
    plan_bytes=None,
    census_peak_bytes=None,
):
    """The `reuse` block, its totals summed over `families`."""
    families = families or {}
    out = {"state": state, "declined": declined}
    for key in ("pairs", "computed", "reused", "near", "kernel_evals"):
        out[key] = sum(f[key] for f in families.values()) if families else None
    out["families"] = families
    out["uncounted"] = list(uncounted)
    out["plan_bytes"] = None if plan_bytes is None else int(plan_bytes)
    out["census_peak_bytes"] = (
        None if census_peak_bytes is None else int(census_peak_bytes)
    )
    out["wires"] = None
    return out


def declined_entry(reason):
    return reuse_entry("declined", declined=reason or NO_PLAN_READ)


def wire_labels(n_wires, kept=None):
    """Each solved wire's index among the wires the caller passed."""
    return [int(w) for w in (range(n_wires) if kept is None else kept)]


def wire_block(labels, counts):
    """The `wires` entry from a (W, W, 3) count of computed, reused and near
    pairs, every family's summed."""
    return {
        "labels": labels,
        "pairs": counts.sum(axis=2).tolist(),
        "computed": counts[:, :, 0].tolist(),
        "reused": counts[:, :, 1].tolist(),
        "near": counts[:, :, 2].tolist(),
    }


def plan_ref(plan):
    """A weak reference to a Python plan, or None for one that takes none."""
    try:
        return weakref.ref(plan)
    except TypeError:
        return None


class FillReport:
    """`fill_report()`, for a solver that implements the three hooks."""

    #: The route a dense fill of this class is called.
    _fill_report_route_name = "dense"

    def _fill_report_begin(self):
        """Anything the fill's end compares against."""
        return None

    def _fill_report_end(self, ledger):
        """Record what the fill that just ended served."""

    def _fill_report_wires(self, ledger):
        """The last fill's wire map, or `{"unavailable": reason}`."""
        return {"unavailable": "this solver has no wire map"}

    def _fill_report_sector(self, ledger):
        """The `sector` block, or None when the route is not on."""
        smap = getattr(self, "_rotational_map", None)
        if smap is None:
            return None
        wire_sector = {}
        for s, wires in enumerate(smap.sectors):
            for w in wires:
                wire_sector[int(w)] = s
        for w in smap.axial:
            wire_sector[int(w)] = -1
        return {
            "n_sectors": int(smap.n_sectors),
            "axial": len(smap.axial),
            "blocks": list(ledger.sector_blocks),
            "computed_pairs": (
                None
                if ledger.sector_rows is None
                else ledger.sector_rows * ledger.sector_segs
            ),
            "pairs": None if ledger.sector_segs is None else ledger.sector_segs**2,
            "declined": None,
            # Per wire: its sector, -1 on the axis. Sector 0 and the axis are
            # the rows the fill computed; the rest follow by rotation.
            "wire_sector": [wire_sector[w] for w in sorted(wire_sector)],
        }

    def fill_report(self, wires=True):
        """What this solver's fill did: see :mod:`momwire._fill_report`.

        Read-only, valid at any time (before a solve the counts are zero and
        `route` is None), and it never runs a census or changes a later
        solve. `wires=False` leaves out the wire-by-wire map."""
        ledger = ledger_of(self)
        reuse = None
        if ledger.reuse is not None:
            reuse = copy.deepcopy(ledger.reuse)
            if wires and reuse["state"] != "declined":
                reuse["wires"] = self._fill_report_wires(ledger)
        return {
            "version": REPORT_VERSION,
            "solver": type(self).__name__,
            "fills": ledger.fills,
            "frequencies": ledger.frequencies,
            "times": {
                "fill_s": ledger.fill_s,
                "factor_s": ledger.factor_s,
                "census_s": ledger.census_s,
            },
            "route": ledger.route,
            "reuse": reuse,
            "sector": self._fill_report_sector(ledger),
        }


def refused_report(exc, solver=None):
    """The report of a deck `rotational_symmetry=True` refused: the solver
    never came to exist, so this reads the exception. `sector.declined` is
    its sentence, which names the condition that failed."""
    return {
        "version": REPORT_VERSION,
        "solver": solver,
        "fills": 0,
        "frequencies": 0,
        "times": {"fill_s": 0.0, "factor_s": 0.0, "census_s": 0.0},
        "route": None,
        "reuse": None,
        "sector": {
            "n_sectors": None,
            "axial": None,
            "blocks": [],
            "computed_pairs": None,
            "pairs": None,
            "declined": str(exc),
            "wire_sector": [],
        },
    }
