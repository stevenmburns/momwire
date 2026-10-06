"""Stored-node fixtures for the grazing-band interpolation gates (momwire#1359).

The four "interpolates to the bar" sweeps (`test_grazing_band_floor_1064`,
`test_grazing_band_lo_935`, `test_grazing_band_838` x2) each cold-fill whole
theta bands on six decks to read a few dozen interpolated values. They test
two different things, and each is cheap on its own:

  * the STENCIL -- the bicubic on the band's lattice -- given the node values;
  * the FILL -- that each node holds the right integral.

This module is the shared machinery for both, used by the unit tests in
`test_grazing_interp_units_1359.py` and by the generator beside the fixtures
(`tests/fixtures/grazing_1359/regenerate.py`). Nothing here is a test.

What a fixture holds, per sweep, for ONE deck (the sweep's measured worst):

  * the query points the sweep asks about, and the direct reference
    (`iv_surfaces_direct_below`, rtol 1e-9) at each;
  * the node values of exactly the lattice nodes those queries' stencils read
    -- found by RECORDING the production `_interp`'s reads, not by re-deriving
    its index arithmetic -- computed by the production `_fill_region`;
  * the interpolated values at generation, as a pin;
  * a header: deck, rtol, the generating commit, and the provenance that
    decides staleness (`staleness`).

A grid built by `loaded_grid` holds those nodes and NaN everywhere else, and
its `_fill_region` raises, so a stencil that reads any node the fixture does
not hold returns NaN, and a query that reaches an unloaded region fails by
name. Nothing integrates at test time.
"""

import ast
import contextlib
import functools
import hashlib
import inspect
import io
import json
import re
import subprocess
import sys
import textwrap
import tokenize
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from momwire import _ground_refl  # noqa: E402
from momwire import _sommerfeld as somm  # noqa: E402
from momwire import _sommerfeld_below as below  # noqa: E402
from momwire._sommerfeld import _SURF_KEYS  # noqa: E402
from test_grazing_band_lo_935 import SOILS, _deck  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "grazing_1359"
REGENERATE = "python tests/fixtures/grazing_1359/regenerate.py"
SCHEMA = 1
RTOL = 1e-9  # the grid's own rtol, and the direct reference's

RI = below.region_index


# ---------------------------------------------------------------------------
# the sweeps, as the slow tests spell them
# ---------------------------------------------------------------------------
#
# Each entry is the query set of one nightly sweep, for one deck. `metric` is
# the sweep's own error normalisation: the floor and low bands divide by each
# surface's own magnitude, #838's two by the largest surface at the node.


def _thirds_and_midpoints(nodes):
    return np.concatenate(
        [nodes[:-1] + 0.5 * np.diff(nodes), nodes[:-1] + 0.33 * np.diff(nodes)]
    )


def _q_floor(g, lam_m):
    nodes = g._regions[RI(below._ZONE_INNER, below._BAND_FLOOR)]["th_nodes"][:3]
    th = _thirds_and_midpoints(nodes)
    return [(np.full(th.shape, r1l * lam_m), th) for r1l in (0.2, 1.0, 1.9)]


def _q_lo(g, lam_m):
    th = _thirds_and_midpoints(
        g._regions[RI(below._ZONE_INNER, below._BAND_LO)]["th_nodes"]
    )
    return [(np.full(th.shape, r1l * lam_m), th) for r1l in (0.2, 1.0, 1.9)]


def _q_mid(g, lam_m):
    th = np.radians(np.linspace(0.105, 0.995, 23))
    return [(np.full(th.shape, r1l * lam_m), th) for r1l in (0.2, 1.0, 1.9)]


def _q_far(g, lam_m):
    r1 = np.linspace(2.05, 3.95, 11) * lam_m
    return [
        (r1, np.radians(np.full(r1.shape, d))) for d in (0.3, 2.0, 17.0, 45.0, 80.0)
    ]


SWEEPS = {
    "floor": {
        "gate": "test_grazing_band_floor_1064.py::test_the_floor_band_interpolates_to_the_bar",
        "bar": ("test_grazing_band_floor_1064", "FLOOR_BAND_BAR"),
        "metric": "per_surface",
        "queries": _q_floor,
    },
    "lo": {
        "gate": "test_grazing_band_lo_935.py::test_the_low_band_interpolates_to_the_bar",
        "bar": ("test_grazing_band_lo_935", "LO_BAND_BAR"),
        "metric": "per_surface",
        "queries": _q_lo,
    },
    "mid": {
        "gate": "test_grazing_band_838.py::test_the_band_interpolates_to_the_bar",
        "bar": ("test_grazing_band_838", "BAND_BAR"),
        "metric": "node_scale",
        "queries": _q_mid,
    },
    "far": {
        "gate": "test_grazing_band_838.py::test_the_far_zone_interpolates_to_the_bar",
        "bar": ("test_grazing_band_838", "FAR_BAR"),
        "metric": "node_scale",
        "queries": _q_far,
    },
}


def bar(sweep):
    mod, name = SWEEPS[sweep]["bar"]
    return float(getattr(__import__(mod), name))


def rel_error(sweep, got, ref):
    """(4, n) relative error, in the sweep's own normalisation."""
    if SWEEPS[sweep]["metric"] == "per_surface":
        return np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)
    return np.abs(got - ref) / np.abs(ref).max(axis=0)[None, :]


def node_scale(vals):
    """The largest surface magnitude at each node: (4, n) -> (n,)."""
    return np.abs(vals).max(axis=0)


# ---------------------------------------------------------------------------
# grids: the production lattice, with nothing integrated
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _no_integration():
    """Construct a grid without evaluating its eager regions.

    The constructor fills the steep and grazing bands of the inner and near
    zones eagerly. Swapping the direct surfaces for a NaN source keeps every
    lattice decision -- node counts, origins, steps, padding -- the
    production constructor's own, at no cost.
    """
    real = below.iv_surfaces_direct_below

    def nan_surfaces(eps_t, k2, R1, theta, **_kw):
        shape = np.broadcast(np.asarray(R1), np.asarray(theta)).shape
        return {k: np.full(shape, np.nan + 0j) for k in _SURF_KEYS}

    below.iv_surfaces_direct_below = nan_surfaces
    try:
        yield
    finally:
        below.iv_surfaces_direct_below = real


def lattice_grid(soil, f, health=None):
    """A production `SommerfeldGridBelow` for one deck: every region unfilled,
    every table NaN, and no node integrated. Sized as the sweeps size theirs,
    to the R1 cap."""
    eps_t, k2, om, lam_m = _deck(soil, f)
    with _no_integration():
        g = below.SommerfeldGridBelow(
            eps_t,
            k2,
            below._SOMM_BELOW_R1_CAP_LAMBDA_M * lam_m,
            omega=om,
            health=health,
        )
    for reg in g._regions:
        reg["vals"] = np.full(
            (len(_SURF_KEYS), reg["n_r"], reg["n_th"]), np.nan, dtype=np.complex128
        )
        reg["filled"] = False
    return g


def fill_rows(g, idx, rows):
    """Run the PRODUCTION `_fill_region(idx)`, cold, restricted to R1 rows.

    The region (and, for a floor-band region, the low band it shares its top
    two columns with) is cut to `rows` for the duration of the call, so the
    production fill -- joint evaluation, ownership of the shared columns, the
    copy -- runs exactly as written, over a handful of rows instead of the
    whole band. The values land in `g`'s full tables at those rows; the
    regions stay marked unfilled. Returns {region: (rows, (4, len(rows), n_th))}
    for every region the call filled.
    """
    rows = np.unique(np.asarray(rows, dtype=int))
    group = [idx]
    if idx in g._band_floor_idx:
        group.append(RI(idx // below._N_BANDS, below._BAND_LO))
    saved = {}
    for i in group:
        reg = g._regions[i]
        saved[i] = {k: reg[k] for k in ("r_nodes", "n_r", "vals", "filled")}
        reg["r_nodes"] = reg["r_nodes"][rows]
        reg["n_r"] = rows.size
        reg["vals"] = np.full(
            (len(_SURF_KEYS), rows.size, reg["n_th"]), np.nan, dtype=np.complex128
        )
        reg["filled"] = False
    try:
        g._fill_region(idx)
        got = {i: g._regions[i]["vals"] for i in group if g._regions[i]["filled"]}
    finally:
        for i in group:
            g._regions[i].update(saved[i])
    for i, sub in got.items():
        g._regions[i]["vals"][:, rows, :] = sub
    return {i: (rows, sub) for i, sub in got.items()}


class _Reads(np.ndarray):
    """A table that records which (row, column) pairs are read from it."""

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 3:
            ii, jj = np.broadcast_arrays(np.asarray(key[1]), np.asarray(key[2]))
            self.reads.update(zip(ii.ravel().tolist(), jj.ravel().tolist()))
        return np.asarray(self)[key]


def stencil_nodes(g, queries):
    """{region: sorted [(i, j)]} -- every node the production `_interp` reads
    to answer `queries`. Recorded on a NaN grid, so it costs nothing."""
    for reg in g._regions:
        rec = reg["vals"].view(_Reads)
        rec.reads = set()
        reg["vals"] = rec
        reg["filled"] = True
    try:
        for r1, th in queries:
            g.eval(r1, th)
        return {
            i: sorted(reg["vals"].reads)
            for i, reg in enumerate(g._regions)
            if reg["vals"].reads
        }
    finally:
        for reg in g._regions:
            reg["vals"] = np.asarray(reg["vals"])
            reg["filled"] = False


# ---------------------------------------------------------------------------
# fixture I/O
# ---------------------------------------------------------------------------


def path(sweep):
    return FIXTURE_DIR / f"{sweep}.npz"


def save(sweep, header, arrays):
    np.savez_compressed(path(sweep), header=np.array(json.dumps(header)), **arrays)


def load(sweep):
    with np.load(path(sweep), allow_pickle=False) as d:
        fx = {k: d[k] for k in d.files if k != "header"}
        fx["header"] = json.loads(str(d["header"]))
    return fx


# Node coordinates are recomputed by each platform's constructor, from a
# wavelength that passes through complex sqrt/hypot -- libm, not IEEE-exact
# arithmetic. An ulp of drift across platforms is expected and harmless; a
# lattice that MOVED moves by a cell fraction, ~1e-3 at least.
LATTICE_RTOL = 1e-12


def loaded_grid(fx):
    """A lattice grid holding the fixture's nodes and NaN everywhere else,
    whose `_fill_region` raises. Fails by name if the production lattice no
    longer puts the fixture's nodes where they were."""
    soil, f = fx["header"]["deck"]
    g = lattice_grid(soil, f)
    reg_ids, ii, jj = fx["node_region"], fx["node_i"], fx["node_j"]
    shape = {
        r: (g._regions[r]["n_r"], g._regions[r]["n_th"]) for r in set(reg_ids.tolist())
    }
    for r, i, j in zip(reg_ids, ii, jj):
        if not (i < shape[r][0] and j < shape[r][1]):
            raise AssertionError(
                f"STALE FIXTURE: region {r} is now {shape[r]} nodes and the fixture "
                f"holds node ({i}, {j}): the lattice changed. Regenerate with "
                f"`{REGENERATE}`."
            )
    for name, got, want in (
        (
            "R1",
            [g._regions[r]["r_nodes"][i] for r, i in zip(reg_ids, ii)],
            fx["node_r"],
        ),
        (
            "theta",
            [g._regions[r]["th_nodes"][j] for r, j in zip(reg_ids, jj)],
            fx["node_th"],
        ),
    ):
        got = np.asarray(got)
        moved = np.abs(got - want) > LATTICE_RTOL * np.abs(want)
        if moved.any():
            k = int(np.argmax(moved))
            raise AssertionError(
                f"STALE FIXTURE: the production lattice moved. Node "
                f"(region {reg_ids[k]}, {ii[k]}, {jj[k]}) has {name} = {got[k]!r}, "
                f"the fixture {want[k]!r} ({int(moved.sum())} nodes moved). "
                f"Regenerate with `{REGENERATE}`."
            )
    for r in np.unique(reg_ids):
        sel = reg_ids == r
        g._regions[r]["vals"][:, ii[sel], jj[sel]] = fx["node_vals"][:, sel]
        g._regions[r]["filled"] = True

    def no_fill(idx):
        raise AssertionError(
            f"a query reached region {idx}, which the fixture does not hold: the "
            "routing or the zone/band layout no longer matches the fixture"
        )

    g._fill_region = no_fill
    return g


# ---------------------------------------------------------------------------
# provenance: what makes a fixture stale
# ---------------------------------------------------------------------------
#
# A fixture is STALE when the code or constants that produced its numbers
# have changed, whether or not the numbers did. Three things are recorded at
# generation and compared at test time:
#
#  1. `constants` -- every numeric module constant of `_sommerfeld_below`
#     (lattice steps, band edges, panel cap, Gauss orders, depths), plus the
#     two it borrows from `_sommerfeld`. Compared exactly: they are literals.
#  2. `code` -- one hash per function or C++ symbol on the path from a deck to
#     an interpolated value (`_PY_CODE`, `_CPP_CODE`). Each is a hash of the
#     TOKENS with comments and docstrings dropped, so a comment, docstring or
#     re-wrap does not make a fixture stale and a changed statement does.
#  3. the lattice itself -- every stored node's (R1, theta), checked in
#     `loaded_grid` against the production constructor.
#
# The list in (2) is the claim "these are the code paths the numbers depend
# on". It can miss one; the fill gate is the backstop, because it recomputes
# stored nodes cold through whatever code is there now.

_PY_CODE = {
    "below": (
        "iv_surfaces_direct_below",
        "_six_integrals_below_many",
        "_six_integrals_below",
        "_integrand_six_below",
        "_gauss_segment",
        "_adaptive_segment",
        "_wynn_epsilon",
        "_head",
        "_tail_below",
        "_run_contour",
        "_refuse_if_capped",
        "_limits_r1_zero_shared",
        "_limits_r1_zero_below",
        "divide_out_below",
        "_c1_moment",
        "k_medium",
        "lambda_medium",
        "_use_below_accel",
        "_six_below_accel",
        "region_index",
        "SommerfeldGridBelow.__init__",
        "SommerfeldGridBelow.eval",
        "SommerfeldGridBelow._interp",
        "SommerfeldGridBelow._fill_region",
    ),
    "somm": ("SommerfeldGrid._lagrange4",),
    "ground_refl": ("eps_tilde",),
}
_PY_MODULES = {"below": below, "somm": somm, "ground_refl": _ground_refl}

# None = the whole file (headers that are nothing but the engine).
_CPP_CODE = {
    "_accel_mw568.cpp": (
        "d12",
        "SixBelow",
        "SixResult",
        "six_below_one",
        "below_six_integrals_batch",
    ),
    "_contour_engine_inline.h": None,
    "_branch_cut_inline.h": None,
    "_fma_inline.h": None,
}

_SKIP_TOKENS = {
    tokenize.COMMENT,
    tokenize.NL,
    tokenize.ENCODING,
    tokenize.ENDMARKER,
}


def _docstring_lines(tree):
    """Start lines of every docstring in `tree`."""
    lines = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                lines.add(first.lineno)
    return lines


def _py_tokens(src):
    """The token stream of `src`, minus comments, docstrings, non-logical
    newlines and f-strings (messages; their tokenization also differs across
    Python versions). Plain strings stay: a swapped dict key is a real change."""
    src = textwrap.dedent(src)
    doc = _docstring_lines(ast.parse(src))
    out, fdepth = [], 0
    fstart = getattr(tokenize, "FSTRING_START", None)
    fend = getattr(tokenize, "FSTRING_END", None)
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if fstart is not None and tok.type == fstart:
            fdepth += 1
            continue
        if fend is not None and tok.type == fend:
            fdepth -= 1
            continue
        if fdepth or tok.type in _SKIP_TOKENS:
            continue
        if tok.type == tokenize.STRING:
            if tok.start[0] in doc or re.match(r"(?i)[rbu]*f", tok.string):
                continue
        out.append(tok.string if tok.type != tokenize.NEWLINE else "\n")
    return " ".join(out)


_CPP_LEX = re.compile(
    r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', re.S
)


def _cpp_strip(src):
    """Comments out, whitespace collapsed; string literals kept."""
    src = _CPP_LEX.sub(lambda m: " " if m.group(0)[0] == "/" else m.group(0), src)
    return re.sub(r"\s+", " ", src).strip()


def _cpp_symbol(src, name):
    """The definition of one struct or function in comment-stripped C++."""
    found = []
    for m in re.finditer(r"\b(?:struct|class)\s+" + name + r"\b[^;{(]*\{", src):
        found.append(m.end() - 1)
    for m in re.finditer(r"\b" + name + r"\s*\(", src):
        depth, k = 0, m.end() - 1
        while k < len(src):
            depth += {"(": 1, ")": -1}.get(src[k], 0)
            k += 1
            if depth == 0:
                break
        rest = re.match(r"\s*(?:const\s*)?\{", src[k:])
        if rest:
            found.append(k + rest.end() - 1)
    if len(found) != 1:
        raise LookupError(f"C++ symbol {name!r}: {len(found)} definitions found")
    start = found[0]
    depth = 0
    for k in range(start, len(src)):
        depth += {"{": 1, "}": -1}.get(src[k], 0)
        if depth == 0:
            return src[src.rfind(";", 0, start) + 1 : k + 1].strip()
    raise LookupError(f"C++ symbol {name!r}: unbalanced braces")


def _h(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@functools.cache
def code_hashes():
    """{component: hash} over `_PY_CODE` and `_CPP_CODE`."""
    out = {}
    for mod_key, names in _PY_CODE.items():
        mod = _PY_MODULES[mod_key]
        for qual in names:
            obj = mod
            for part in qual.split("."):
                obj = getattr(obj, part)
            out[f"py:{mod.__name__}.{qual}"] = _h(_py_tokens(inspect.getsource(obj)))
    src_dir = Path(below.__file__).resolve().parent
    for fname, symbols in _CPP_CODE.items():
        text = _cpp_strip((src_dir / fname).read_text())
        if symbols is None:
            out[f"cpp:{fname}"] = _h(text)
        else:
            for s in symbols:
                out[f"cpp:{fname}:{s}"] = _h(_cpp_symbol(text, s))
    return out


def constants():
    """Every numeric module constant of `_sommerfeld_below`, plus the two it
    borrows. Runtime switches (booleans) are not constants of the numbers."""
    out = {}
    for name, val in sorted(vars(below).items()):
        if not re.fullmatch(r"_[A-Z][A-Z0-9_]*", name) or isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            out[name] = val
        elif isinstance(val, (tuple, frozenset)) and all(
            isinstance(v, (int, float, tuple)) for v in val
        ):
            out[name] = json.loads(
                json.dumps(sorted(val) if isinstance(val, frozenset) else val)
            )
    out["_sommerfeld._SOMM_TH_SPLIT_DEG"] = somm._SOMM_TH_SPLIT_DEG
    out["_sommerfeld._C_LIGHT"] = somm._C_LIGHT
    return out


def provenance(soil, f):
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=FIXTURE_DIR,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "schema": SCHEMA,
        "deck": [soil, f],
        "soil": list(SOILS[soil]),
        "rtol": RTOL,
        "generated_at_commit": commit,
        "constants": constants(),
        "code": code_hashes(),
    }


def staleness(fx):
    """Human-readable reasons the fixture no longer describes this code; empty
    when it is current."""
    hdr = fx["header"]
    soil, f = hdr["deck"]
    problems = []
    if hdr.get("schema") != SCHEMA:
        problems.append(f"schema {hdr.get('schema')} != {SCHEMA}")
    if hdr.get("rtol") != RTOL:
        problems.append(f"rtol {hdr.get('rtol')} != {RTOL}")
    if list(SOILS[soil]) != hdr["soil"]:
        problems.append(f"soil {soil} is {SOILS[soil]}, fixture {hdr['soil']}")
    now = constants()
    then = hdr["constants"]
    for k in sorted(set(now) | set(then)):
        if json.loads(json.dumps(now.get(k))) != then.get(k):
            problems.append(f"constant {k}: {then.get(k)!r} -> {now.get(k)!r}")
    now = code_hashes()
    then = hdr["code"]
    for k in sorted(set(now) | set(then)):
        if now.get(k) != then.get(k):
            what = (
                "added" if k not in then else "removed" if k not in now else "changed"
            )
            problems.append(f"code {what}: {k}")
    return problems


# ---------------------------------------------------------------------------
# the fill, checked slot by slot
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def spy_direct():
    """Record every (R1, theta) -> surfaces the real `iv_surfaces_direct_below`
    computes while active: {(R1, theta): (4,) complex}. A pass-through, so the
    fill it watches runs unchanged and pays nothing extra."""
    real = below.iv_surfaces_direct_below
    seen = {}

    def spy(eps_t, k2, R1, theta, **kw):
        out = real(eps_t, k2, R1, theta, **kw)
        r, t = np.broadcast_arrays(np.asarray(R1, float), np.asarray(theta, float))
        v = np.stack([np.asarray(out[k]).reshape(r.shape) for k in _SURF_KEYS])
        for n, key in enumerate(zip(r.ravel().tolist(), t.ravel().tolist())):
            seen[key] = v.reshape(len(_SURF_KEYS), -1)[:, n]
        return out

    below.iv_surfaces_direct_below = spy
    try:
        yield seen
    finally:
        below.iv_surfaces_direct_below = real


def misplaced_slots(g, filled, seen):
    """Slots of the restricted fill whose value is not, bit for bit, the direct
    evaluation AT THAT SLOT'S NODE -- for a floor band's shared columns, at the
    low band's node, which owns them. Returns [(region, i, j, why)]."""
    shared = dict(below._SOMM_BELOW_BAND_FLOOR_SHARED)
    bad = []
    for idx, (rows, sub) in filled.items():
        reg = g._regions[idx]
        th_at = list(reg["th_nodes"])
        if idx in g._band_floor_idx:
            lo = g._regions[RI(idx // below._N_BANDS, below._BAND_LO)]
            for f_col, lo_col in shared.items():
                th_at[f_col] = lo["th_nodes"][lo_col]
        for a, i in enumerate(rows):
            for j in range(reg["n_th"]):
                key = (float(reg["r_nodes"][i]), float(th_at[j]))
                if key not in seen:
                    bad.append((idx, int(i), j, "no direct evaluation at its node"))
                elif not np.array_equal(sub[:, a, j], seen[key]):
                    bad.append((idx, int(i), j, "holds another node's value"))
    return bad
