"""The sinusoidal `pair_reuse` plan (`_sin_pair_reuse_core.cpp`, on the
common reuse core), which both routes run (momwire#2063 stage 2), and the C++
fill's use of it against the Python route's.

The nec42-sinusoidal executable's fill (`_sinusoidal_fill_core.cpp`) serves
the feature when it is requested (the in-process hook here; a certification
build of the executable, `MW_EXE_CERTIFY`, in tests/test_standalone_nec42.py).
Its reach is level 4, the drop-in, short of every executable level, so the
shipped executables refuse it.

Held:
- the per-segment types (`reversible_segment_types`, `reversible_ground_types`)
  against numpy's spelling of them, `np.unique` over the same columns;
- the plan, every array of it (the class map, representatives, codes, near
  pairs and their blocks, the canonical sources and the radius runs), for
  the direct block, the PEC image's and a finite ground's reflected term,
  at any OpenMP thread count (the census's parts) and any worker count (its
  bands), the classes numbered where they first occur in the pairs' order;
  and what the census counted (`census_peak_bytes`) at any thread count;
- the allowance's verdicts (momwire#2123): over a ladder of allowances
  around a deck's class map and its plan's fill, a deck declines from one
  allowance down and serves the same plan above it; and the class cap's, a
  deck short of 4/3x compression declining with the cap named;
- the NEC-4.2 slot's printout and its fill (Z, the port coefficients and
  Y, by row digest) with the feature requested, against the Python slot's
  with it on, on decks that serve it over free space, a PEC plane and the
  Sommerfeld ground; the fill differs from the C++ fill with it off (so the
  reuse is what ran) and the C++ route reports the census Python's does;
- every decline, with Python's reason: the NEC-4 junction kernel, a deck
  below the small-deck threshold, a plan past its allowance, and a census
  past the class cap.

Bit for bit, not within the drift: the two routes evaluate the same classes
with the same kernels on the same slices.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

from momwire import _accel
from momwire import _sin_pair_reuse as pr
from momwire import sinusoidal as S
from momwire.sinusoidal import SinusoidalSolver

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"

acc = _accel.acc
pytestmark = [
    pytest.mark.reach(4),
    pytest.mark.skipif(
        acc is None or not getattr(acc, "sin_pair_reuse_cxx", False),
        reason="needs the extension's C++ pair-reuse hooks",
    ),
]

LAM = 10.7


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("MOMWIRE_X_PAIR_REUSE", raising=False)
    pr.clear_shared_plans()
    acc.pair_reuse_cap_compression(-1)
    acc.sin_pair_reuse_configure(
        allowance=0, small_free=-1, small_ground=-1, workers=0, clear=True
    )
    yield
    pr.clear_shared_plans()
    acc.pair_reuse_cap_compression(-1)
    acc.sin_pair_reuse_configure(
        allowance=0, small_free=-1, small_ground=-1, workers=0, clear=True
    )


def _solver(wires, radii=None):
    """An extended-kernel solver (the NEC-4.2 slot's) over straight wires."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SinusoidalSolver(
            wires=[np.asarray(w, dtype=np.float64) for w, _ in wires],
            n_per_edge_per_wire=[[n] for _, n in wires],
            feed_wire_index=len(wires) // 2,
            wavelength=LAM,
            wire_radius=2e-3 if radii is None else radii,
            extended_kernel=True,
        )


def _curtain(n=10, segs=41, flip=(), tilt=0.0):
    out = []
    for i in range(n):
        a = np.array([0.1 * LAM * i, 0.0, 1.0])
        b = np.array([0.1 * LAM * i + tilt * (i % 3), 0.0, 1.0 + 0.5 * LAM])
        out.append(((b, a) if i in flip else (a, b), segs))
    return [(np.stack(p), s) for p, s in out]


def _random_wires(n=24, segs=17, seed=1911):
    """Straight wires in random directions: every orientation, no symmetry."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        c = rng.uniform(-3.0, 3.0, 3) + np.array([0.0, 0.0, 6.0])
        d = rng.normal(size=3)
        d *= 0.2 * LAM / np.linalg.norm(d)
        out.append((np.stack([c - d, c + d]), segs))
    return out


DECKS = {
    "curtain": (_curtain(), None),
    "flipped, tilted": (_curtain(flip=(1, 4, 7), tilt=0.3), None),
    "two radii": (_curtain(flip=(2,)), [2e-3, 3e-3] * 5),
    "random wires": (_random_wires(), None),
}


def _inputs(s):
    geom = s._build_geometry()
    N = geom["n_segs"]
    seg_c = np.ascontiguousarray(geom["seg_centers"], dtype=np.float64)
    seg_t = np.ascontiguousarray(geom["seg_tangents"], dtype=np.float64)
    seg_h = np.ascontiguousarray(geom["seg_h"], dtype=np.float64)
    src_a = np.ascontiguousarray(s._seg_radius(geom), dtype=np.float64)
    a_seg = (
        np.full(N, float(s._uniform_radius)) if s._uniform_radius is not None else src_a
    )
    ind1, ind2 = (np.ascontiguousarray(x, dtype=np.int8) for x in s._ek_gating(geom))
    # the image as the C++ fill mirrors it, in the NEC-4.2 slot's z = 0
    img_c = seg_c.copy()
    img_c[:, 2] = 2 * 0.0 - img_c[:, 2]
    img_t = seg_t * np.array([1.0, 1.0, -1.0])
    rows = max(1, int(256 * 1024 * 1024 // (20 * 8 * N)))
    return dict(
        N=N, seg_c=seg_c, seg_t=seg_t, seg_h=seg_h, src_a=src_a, a_seg=a_seg,
        ind1=ind1, ind2=ind2, img_c=img_c, img_t=img_t, rows=rows,
    )  # fmt: skip


def _same(a, b, what):
    a, b = np.asarray(a), np.asarray(b)
    assert (a.shape, a.dtype) == (b.shape, b.dtype), (
        what,
        a.shape,
        b.shape,
        a.dtype,
        b.dtype,
    )
    assert a.tobytes() == b.tobytes(), what


def _blocks_same(py, cx, what):
    assert len(py) == len(cx), what
    for x, y in zip(py, cx, strict=True):
        assert (x[0], x[1], x[5], x[6]) == (y[0], y[1], y[5], y[6]), what
        for k in (2, 3, 4):
            _same(x[k], y[k], f"{what}[{k}]")


def _plan_same(a, b, ground, peak=True):
    if isinstance(a, pr.Declined) or isinstance(b, pr.Declined):
        assert a == b
        return
    _same(a.cls, b.cls, "cls")
    _same(
        a.rep_flat if ground else a.rep_src, b.rep_flat if ground else b.rep_src, "rep"
    )
    _same(a.rep_code, b.rep_code, "rep_code")
    assert a.code_bits == b.code_bits
    _same(a.near_flat, b.near_flat, "near_flat")
    _blocks_same(a.near_blocks, b.near_blocks, "near_blocks")
    if ground:
        _blocks_same(a.rep_blocks, b.rep_blocks, "rep_blocks")
    else:
        _same(a.src_c, b.src_c, "src_c")
        _same(a.src_t, b.src_t, "src_t")
        assert [tuple(r) for r in a.runs] == [tuple(r) for r in b.runs]
    assert pr.fill_nbytes(a) == b.handle.fill_nbytes == pr.fill_nbytes(b)
    if peak:
        assert a.census_peak_bytes == b.census_peak_bytes


def _plan(block, g, types, gtypes, allowance, workers, monkeypatch):
    """The plan for `block` of deck inputs `g`, at `workers` census workers."""
    monkeypatch.setattr(pr, "_workers", lambda: workers)
    src_type, obs_type, src_rev = types
    common = (g["seg_h"], g["a_seg"])
    if block == "reflected":
        gs, go, gsr, gor = gtypes
        return pr.build_ground_plan(
            g["seg_c"], g["seg_t"], g["seg_c"], g["seg_t"], 0.0, *common, gs, go,
            g["rows"], src_rev=gsr, obs_rev=gor, allowance=allowance,
        )  # fmt: skip
    sc, st = (g["seg_c"], g["seg_t"]) if block == "direct" else (g["img_c"], g["img_t"])
    return pr.build_plan(
        g["seg_c"], g["seg_t"], sc, st, *common, src_type, obs_type, g["rows"],
        src_rev=src_rev, allowance=allowance,
    )  # fmt: skip


def _types(g):
    py = pr.reversible_segment_types(
        g["seg_h"], g["a_seg"], g["src_a"], g["ind1"], g["ind2"]
    )
    gpy = pr.reversible_ground_types(*py, g["seg_t"])
    return py, gpy


def _ids(*cols):
    """numpy's dense ids for the distinct rows of per-segment columns."""
    _, ids = np.unique(np.stack(cols, axis=1), axis=0, return_inverse=True)
    return np.asarray(ids, dtype=np.int64).reshape(-1)


def _numpy_types(g):
    """`reversible_segment_types` and `reversible_ground_types` in numpy:
    the half-length at 48 relative bits, the radius and end codes exact,
    both orientations in one id space; then the vertical tangent component
    at 48 absolute bits."""
    n = g["N"]
    m, e = np.frexp(np.concatenate([g["seg_h"]] * 2))
    cols = [
        e.astype(np.int64),
        np.rint(np.ldexp(m, 48)).astype(np.int64),
        np.concatenate([g["src_a"]] * 2).view(np.int64),
        np.concatenate([g["ind1"], g["ind2"]]).astype(np.int64),
        np.concatenate([g["ind2"], g["ind1"]]).astype(np.int64),
    ]
    src = _ids(*cols)
    obs = _ids(g["a_seg"].view(np.int64))
    tz = np.rint(np.concatenate([g["seg_t"][:, 2], -g["seg_t"][:, 2]]) * 2.0**48)
    tz = tz.astype(np.int64)
    g_src = _ids(src, tz)
    g_obs = _ids(np.concatenate([obs, obs]), tz)
    return (src[:n], obs, src[n:]), (g_src[:n], g_obs[:n], g_src[n:], g_obs[n:])


@pytest.mark.parametrize("deck", list(DECKS))
def test_the_types_are_numpys(deck):
    g = _inputs(_solver(*DECKS[deck]))
    cx, gcx = _types(g)
    py, gpy = _numpy_types(g)
    for a, b in zip((*py, *gpy), (*cx, *gcx), strict=True):
        _same(a, b, "types")
    # the reduced kernel's (half-length only)
    m, e = np.frexp(g["seg_h"])
    rpy = _ids(e.astype(np.int64), np.rint(np.ldexp(m, 48)).astype(np.int64))
    rcx = pr.reversible_segment_types(g["seg_h"], g["a_seg"])
    for a, b in zip((rpy, py[1], rpy), rcx, strict=True):
        _same(a, b, "reduced types")
    # not vacuous: the end codes split the sources into types, reversal moves them
    assert np.unique(py[0]).size > 1 and np.any(py[0] != py[2])


@pytest.mark.parametrize("block", ["direct", "image", "reflected"])
@pytest.mark.parametrize("deck", list(DECKS))
def test_the_plan_is_one_whatever_the_threads(deck, block, monkeypatch):
    """The census hashes its pairs into one part per OpenMP thread and
    numbers the classes by their first pair whatever the parts, and its
    bands follow the workers: the plan is the same bytes at any of either,
    and the census's count at any thread count. The random wires compress
    less than 4/3x: the cap is lifted (`pair_reuse_cap_compression(1)`), this
    test being the plan's, not the cap's (nor the allowance's, lifted too)."""
    from threadpoolctl import threadpool_limits

    acc.pair_reuse_cap_compression(1)
    big = 1 << 40
    g = _inputs(_solver(*DECKS[deck]))
    types, gtypes = _types(g)
    with threadpool_limits(limits=1, user_api="openmp"):
        one = _plan(block, g, types, gtypes, big, 3, monkeypatch)
    for threads in (2, 5, 8):
        with threadpool_limits(limits=threads, user_api="openmp"):
            got = _plan(block, g, types, gtypes, big, 3, monkeypatch)
        _plan_same(one, got, block == "reflected")
    for workers in (1, 8):
        got = _plan(block, g, types, gtypes, big, workers, monkeypatch)
        _plan_same(one, got, block == "reflected", peak=False)
    # not vacuous: coded orientations, reused pairs where the deck repeats
    # (random wires repeat little: they are the orientation code's test),
    # near pairs where they lie
    assert np.any(one.rep_code != 0)
    if deck != "random wires":
        assert one.n_classes < one.n_pairs // 2
    if block == "direct":
        assert one.n_near > 0


@pytest.mark.parametrize("block", ["direct", "reflected"])
@pytest.mark.parametrize("deck", ["curtain", "random wires"])
def test_the_allowance_holds_the_plan(deck, block, monkeypatch):
    """Every allowance of a ladder around the deck's own numbers: its class
    map (declined before any census) and the plan's fill (declined at the
    end), and between them. Where it serves, the plan is the one an
    allowance that never binds finds, its fill within it. The census has no
    byte bound (its bands follow the allowance, so its count moves). Under
    the cap lifted, as the random wires need."""
    acc.pair_reuse_cap_compression(1)
    g = _inputs(_solver(*DECKS[deck]))
    types, gtypes = _types(g)
    workers = 4
    lifted = _plan(block, g, types, gtypes, 1 << 40, workers, monkeypatch)
    cls_bytes = int(lifted.cls.nbytes)
    fill = pr.fill_nbytes(lifted)
    marks = [cls_bytes, fill]
    ladder = sorted({max(1, m + d) for m in marks for d in (-1, 0, 1)})
    ladder += [int(x) for x in np.linspace(cls_bytes, 2 * fill, 13)]
    verdicts = []
    for allowance in sorted(set(ladder)):
        got = _plan(block, g, types, gtypes, allowance, workers, monkeypatch)
        declined = isinstance(got, pr.Declined)
        verdicts.append(declined)
        assert declined == (allowance < fill), allowance
        if declined:
            assert got.reason == pr.allowance_reason(allowance)
        else:
            _plan_same(lifted, got, block == "reflected", peak=False)
    assert verdicts[0] and not verdicts[-1]


@pytest.mark.parametrize("block", ["direct", "image", "reflected"])
def test_the_cap_holds_the_plan(block, monkeypatch):
    """The random wires keep more than three quarters of their pairs as
    classes: under the cap (4/3x compression) every block's census stops and declines with
    the cap named, at any worker count; the curtain, far under it, serves."""
    for deck, served in (("random wires", False), ("curtain", True)):
        g = _inputs(_solver(*DECKS[deck]))
        types, gtypes = _types(g)
        for workers in (1, 4):
            got = _plan(block, g, types, gtypes, None, workers, monkeypatch)
            if served:
                assert not isinstance(got, pr.Declined)
                assert got.n_classes <= pr.class_cap(got.n_pairs)
            else:
                assert got == pr.Declined(pr.cap_reason(g["N"] ** 2)), got


# --- the NEC-4.2 slot's printout and fill, through the in-process hook ------


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tools():
    rc = _load("core_split_nec4_run_compare", "scripts/core_split/nec4_run_compare.py")
    fc = _load(
        "core_split_sinusoidal_fill_compare",
        "scripts/core_split/sinusoidal_fill_compare.py",
    )
    try:
        blas = rc.blas_addresses("sinusoidal")
        fill_blas = fc.blas_addresses()
    except (IndexError, AttributeError, OSError):
        pytest.skip("no bundled OpenBLAS beside numpy / SciPy (not a wheel install)")
    if not hasattr(acc, "sinusoidal_nec4_printout"):
        pytest.skip("no compiled accelerator with the NEC-4.2 run half")
    return rc, fc, blas, fill_blas


def curtain_deck(n=10, segs=41, ground="free", flip=(), tilt=0.0, radii=None, nf=3):
    """`n` parallel wires with no junction (the NEC-4.2 slot's NEC-4 kernel
    serves any junction, and the reuse steps aside from it)."""
    lines = ["CM pair reuse curtain", "CE"]
    lam = 299.792458 / 28.0
    for i in range(n):
        x = 0.1 * lam * i
        a, b = (x, 0.0, 1.0), (x + tilt * (i % 3), 0.0, 1.0 + 0.5 * lam)
        if i in flip:
            a, b = b, a
        r = radii[i % len(radii)] if radii else 0.002
        lines.append(
            f"GW {i + 1} {segs} {a[0]:.6f} {a[1]:.6f} {a[2]:.6f} "
            f"{b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {r}"
        )
    if ground == "free":
        lines.append("GE 0")
    else:
        lines += ["GE 1", "GN 1" if ground == "pec" else "GN 2 0 0 0 13 .005"]
    lines += [
        f"FR 0 {nf} 0 0 28 0.1",
        f"EX 0 {n // 2 + 1} {segs // 2 + 1} 0 1 0",
        "XQ",
        "EN",
    ]
    return "\n".join(lines) + "\n"


@pytest.fixture
def py_outcome(monkeypatch):
    """Each Python fill's `pair_reuse_declined` and census, in fill order."""
    seen = []
    orig = S.SinusoidalSolver._fill_report_end

    def wrap(self, ledger):
        seen.append((self.pair_reuse_declined, dict(self.pair_reuse_census)))
        return orig(self, ledger)

    monkeypatch.setattr(S.SinusoidalSolver, "_fill_report_end", wrap)
    return seen


def _runs(tools, text, monkeypatch, py_outcome):
    """(C++ printout on, C++ stats, C++ fill on, C++ fill off, Python's
    printout on, Python's fill on)."""
    from threadpoolctl import threadpool_limits

    rc, fc, blas, fill_blas = tools
    version, variant, marker = rc.pc.stamp()
    data = text.encode("latin-1")
    with threadpool_limits(limits=1):
        acc.sin_pair_reuse_stats(True)
        ok, on = acc.sinusoidal_nec4_printout(
            data, blas, version, variant, marker, ["pair_reuse"]
        )
        assert ok, on
        stats = acc.sin_pair_reuse_stats()
        fill_on = acc.sinusoidal_fill_dump(data, fill_blas, fc.WHOLE, ["pair_reuse"])
        fill_off = acc.sinusoidal_fill_dump(data, fill_blas, fc.WHOLE, [])
        monkeypatch.setenv("MOMWIRE_X_PAIR_REUSE", "1")
        pr.clear_shared_plans()
        py_outcome.clear()
        py = rc.py_printout(text, "sinusoidal")
        outcome = list(py_outcome)
        pr.clear_shared_plans()
        fill_py = fc.py_fill(text)
        monkeypatch.delenv("MOMWIRE_X_PAIR_REUSE")
    body = [rc.mask(t)[0].split("\n")[2:] for t in (py, on)]
    assert body[0] == body[1], rc.first_difference(
        "\n".join(body[0]), "\n".join(body[1])
    )
    assert fill_on == fill_py, rc.first_difference(fill_py, fill_on)
    return stats, outcome, fill_on, fill_off


SERVED = {
    "free space": curtain_deck(),
    "PEC": curtain_deck(ground="pec"),
    "Sommerfeld": curtain_deck(ground="somm"),
    "flipped, tilted": curtain_deck(flip=(1, 4, 7), tilt=0.3),
    "two radii over PEC": curtain_deck(ground="pec", flip=(0, 3), radii=[0.002, 0.003]),
    "tilted, two radii, Sommerfeld": curtain_deck(
        ground="somm", flip=(0, 3), tilt=0.2, radii=[0.002, 0.004]
    ),
    # a two-wire junction at one radius keeps NEC-2's kernel and its uniform
    # end factors (`_sinusoidal_nec4.solved_nodes`), so the reuse serves it;
    # the junction's ends carry the extended kernel's code 2
    "a two-wire junction": (
        "CM vee\nCE\nGW 1 250 0 0 1 0 0 11 .002\nGW 2 250 0 0 11 5 0 16 .002\n"
        "GE 0\nFR 0 1 0 0 28 0\nEX 0 1 125 0 1 0\nXQ\nEN\n"
    ),
}


@pytest.mark.parametrize("deck", list(SERVED))
def test_the_reused_printout_and_fill_are_pythons(deck, tools, monkeypatch, py_outcome):
    """Two of these decks compress less than 2x (the tilted Sommerfeld
    curtain, the junction): the class cap is lifted on both routes (the
    hook), this test being the served plan's parity."""
    acc.pair_reuse_cap_compression(1)
    stats, outcome, fill_on, fill_off = _runs(
        tools, SERVED[deck], monkeypatch, py_outcome
    )
    assert stats["last"] == "served"
    assert outcome and all(o == (None, outcome[0][1]) for o in outcome)
    assert stats["census"] == outcome[0][1]
    assert stats["built"] - stats["declined"] >= 1
    # the reuse ran: the fill is not the fill without it
    assert fill_on != fill_off


DECLINED = {
    # a junction: the NEC-4.2 slot's NEC-4 kernel, which the reuse never serves
    "a three-wire junction": (
        "CM tee\nCE\nGW 1 150 0 0 1 0 0 11 .002\nGW 2 150 0 0 11 5 0 16 .002\n"
        "GW 3 150 0 0 11 -5 0 16 .002\nGE 0\nFR 0 1 0 0 28 0\nEX 0 1 75 0 1 0\nXQ\nEN\n",
        "the NEC-4 junction kernel",
    ),
    "small deck": (
        curtain_deck(n=4, segs=41),
        "too few segments to repay the plan (N < 400)",
    ),
    "small deck over the ground": (
        curtain_deck(n=4, segs=41, ground="somm"),
        "too few segments to repay the plan (N < 200)",
    ),
    "transmitted_mixed m2": (
        (FIX / "transmitted_mixed" / "m2_dipole_over_buried.nec").read_text(
            encoding="latin-1"
        ),
        None,
    ),
}


# The buried deck runs in the slow lane (0.96.0 release: its default-lane
# median passed the 5 s budget, scripts/test_durations.py).
@pytest.mark.parametrize(
    "deck",
    [
        pytest.param(d, marks=pytest.mark.slow) if d == "transmitted_mixed m2" else d
        for d in DECLINED
    ],
)
def test_a_declined_deck_declines_as_pythons(deck, tools, monkeypatch, py_outcome):
    text, reason = DECLINED[deck]
    stats, outcome, fill_on, fill_off = _runs(tools, text, monkeypatch, py_outcome)
    assert fill_on == fill_off
    assert stats["built"] == 0
    if reason is None:
        # a mixed deck takes `_assemble_Z_mixed`, which no reuse sees
        assert stats["last"] == "" and all(o[0] is None and not o[1] for o in outcome)
    else:
        assert stats["last"] == reason
        assert outcome and {o[0] for o in outcome} == {reason}


@pytest.mark.parametrize("ground", ["free", "pec", "somm"])
def test_a_deck_past_the_allowance_declines_as_pythons(
    ground, tools, monkeypatch, py_outcome
):
    """An allowance the curtain's plan passes (0.7 MB: its class map is 0.64
    MB) on both routes: the same sentence, and the fill as with the reuse
    off."""
    allowance = 700_000
    monkeypatch.setattr(pr, "ALLOWANCE_BYTES", allowance)
    acc.sin_pair_reuse_configure(allowance=allowance, clear=True)
    stats, outcome, fill_on, fill_off = _runs(
        tools, curtain_deck(ground=ground), monkeypatch, py_outcome
    )
    reason = "the kept plan would exceed its allowance of 0.667572 MB"
    assert pr.allowance_reason(allowance) == reason
    assert stats["last"] == reason
    assert {o[0] for o in outcome} == {reason}
    assert fill_on == fill_off
    # declined once, then found declined at the next frequencies
    assert stats["built"] == stats["declined"] == 1 and stats["found"] >= 1


def random_deck(ground="free", n=24, segs=17, seed=1911, nf=2):
    """`_random_wires` as a deck: no junction, nearly every far pair its own
    class, 408 segments."""
    lam = 299.792458 / 28.0
    rng = np.random.default_rng(seed)
    lines = ["CM pair reuse random wires", "CE"]
    for i in range(n):
        c = rng.uniform(-3.0, 3.0, 3) + np.array([0.0, 0.0, 6.0])
        d = rng.normal(size=3)
        d *= 0.2 * lam / np.linalg.norm(d)
        a, b = c - d, c + d
        lines.append(
            f"GW {i + 1} {segs} {a[0]:.6f} {a[1]:.6f} {a[2]:.6f} "
            f"{b[0]:.6f} {b[1]:.6f} {b[2]:.6f} 0.002"
        )
    if ground == "free":
        lines.append("GE 0")
    else:
        lines += ["GE 1", "GN 1" if ground == "pec" else "GN 2 0 0 0 13 .005"]
    lines += [f"FR 0 {nf} 0 0 28 0.1", f"EX 0 {n // 2 + 1} {segs // 2 + 1} 0 1 0"]
    return "\n".join([*lines, "XQ", "EN"]) + "\n"


@pytest.mark.parametrize("ground", ["free", "pec", "somm"])
def test_a_deck_short_of_the_cap_declines_as_pythons(
    ground, tools, monkeypatch, py_outcome
):
    """The random wires pass the class cap at the default allowance on both
    routes: the same sentence (the cap, not the count the census stopped
    at, which follows each route's bands), and the fill as off."""
    reason = pr.cap_reason(408 * 408)
    assert reason == (
        "more than 124848 classes for 166464 pairs (the cap is 1.333x compression)"
    )
    stats, outcome, fill_on, fill_off = _runs(
        tools, random_deck(ground=ground), monkeypatch, py_outcome
    )
    assert stats["last"] == reason
    assert {o[0] for o in outcome} == {reason}
    assert fill_on == fill_off
    assert stats["built"] == stats["declined"] == 1


def test_the_reuse_is_requested_per_run(tools, monkeypatch, py_outcome):
    """Off is off: the hook without the feature leaves no outcome and builds
    nothing, after a run that served it."""
    rc, fc, blas, fill_blas = tools
    version, variant, marker = rc.pc.stamp()
    data = SERVED["free space"].encode("latin-1")
    acc.sin_pair_reuse_stats(True)
    ok, _ = acc.sinusoidal_nec4_printout(
        data, blas, version, variant, marker, ["pair_reuse"]
    )
    assert ok and acc.sin_pair_reuse_stats()["last"] == "served"
    acc.sin_pair_reuse_stats(True)
    ok, _ = acc.sinusoidal_nec4_printout(data, blas, version, variant, marker, [])
    st = acc.sin_pair_reuse_stats()
    assert ok and st["last"] == "" and st["built"] == st["found"] == 0
