"""`pair_reuse` on the point-matched sinusoidal fill (momwire#1911).

The fill's field tensors Φ (one value per shape: constant, sin, cos) at an
(observer point, source segment) pair depend on the pair's geometry only
through four scalars, measured in the SOURCE's frame (origin at its centre,
z along its natural tangent):

* ``z``   -- the observer centre's coordinate along the source axis;
* ``rho`` -- its distance from that axis;
* ``td``  -- the cosine between the observer and source tangents;
* ``rdt`` -- the radial vector dotted with the observer tangent;

and, beyond those, on the source's half-length, the observer's radius and,
under the extended kernel, the source's radius and its two end codes. A
straight source's current is axisymmetric, so its field has no azimuthal
part and the projection onto the observer's tangent needs nothing more.
Both C++ kernels (`sinusoidal_field_tensor`, `_ek`) read exactly these
quantities (`_sinusoidal_core.cpp`, stage A), so two pairs agreeing on all
of them have the same Φ: every rigid motion, mirrored pairs included.

THE KEY quantises the four scalars with a step of 2^-KEY_BITS times the
pair's singular distance (the observer centre to the source segment, within
sqrt(2), floored at the observer radius), rounded down to a power of two so
the step is a function of the value alone; the source's half-length is
quantised against ITSELF at KEY_BITS relative bits (`_quantise_length`, a
few ulp, so lengths that differ by more than rounding never share a
class), and the radii and end codes are exact. A member
differs from its representative by at most 2^-(KEY_BITS+1) relative to the
scale at which Φ varies (k r times that in phase), and takes its Φ. Pairs
that straddle a quantum land in neighbouring classes, which costs
compression, never accuracy. The classes are exact-equality groups of the
quantised integers (the `factorize_ints` hash kernel, staged over its three
columns), so no two different keys can share a class.

EXCLUDED: a pair whose observer lies within `NEAR_LENGTHS` kernel scales of
the source's centre (in the max norm of |z| and rho: the self term and its
neighbours) is not shared, and is evaluated by the shipped kernel in the
ORIGINAL frame, exactly as the fill would (`near_blocks`). Those pairs carry
the near-singular field that the basis sums over neighbouring sources
cancel, so any rounding of their geometry other than the structure's own,
including the canonical frame's, is amplified into the answer.

The kernel scale is the longer of the source's length and the observer's
radius, max(2H, a), razor-2p's rule (momwire#1971, ported by #1982). Where
a segment is shorter than its radius the kernel varies over a, not h, and
a rule in source lengths alone shared pairs a few h apart but inside a
radius. Measured on a hairpin (two 10 m wires 2a apart joined by a 2a
bridge of 21 segments, a = 6 cm at lambda 20 m, so h/a = 0.095 and k h is
inside the served range; free space, reduced kernel), at the origin: Zin
moved 1.4e-3 against the flag-off fill with the rule in h alone, 1.6e-9
with max(2H, a); 11 bridge segments (h/a 0.18), 2.0e-9 -> 7e-12. Where
2H >= a, the usual mesh, the rule and the plan are unchanged.

EVALUATION is the shipped kernel in the OBSERVER's frame: one observer row
at the origin along +z, and each class's source placed so the four scalars
come out as the representative's, its half-length, radius and end codes
taken from the representative's own source. A row against many sources is
the kernel's efficient shape; the rows are split across a thread pool whose
workers each run the kernel on one OpenMP thread. Classes are found once per
geometry (the key holds no k), Φ per class once per frequency, then gathered
into each observer band, so the reduction (`_reduce_phi_band`) is
unchanged. Memory: an (N_obs, N_src) int32 class map, a quarter of Z, plus
the class table; no Φ-sized copy. The census folds each band's classes into
one table as it goes, so it holds each class once (THE CENSUS, below),
and both are held to `PLAN_BUDGET_BYTES` (momwire#2019).

OVER A FINITE GROUND (momwire#1970) the image block is not a free-space
field: the Sommerfeld fill subtracts C2 Φ_img - S, S the interpolated
remainder, and the reflection-coefficient fill a Fresnel-weighted Φ_img.
Both are invariant only under the isometries that keep the ground plane:
horizontal translations, rotations about a vertical axis and reflections in
vertical planes. A 3-D key would merge a tilted pair with a horizontal one
and two congruent horizontal pairs at different heights, so the REFLECTED
term takes its own classes (`build_ground_plan`) and the direct block keeps
the 3-D ones, which are coarser for any structure not at one height.

What the reflected term reads of a pair, observer m over source segment n
with heights z above the plane: the image source is n mirrored, so the
observer-to-image vector is (d, z_m + z_n), d the horizontal separation;
every quadrature node of n is at z_n + t_nz s; the remainder grid, the
specular ray and the image kernel see nothing else. So the heights enter
ONLY as their sum, and the rest is the horizontal configuration of d and
the two tangents' horizontal parts a = t_n,xy and b = t_m,xy, plus their
vertical parts. The key holds the sum, |d|, d.a, d.b, a.b (the Gram matrix
of d, a, b, with |a| and |b| fixed by t_nz, t_mz), and t_nz, t_mz in the
segment types: complete invariants of the ground group, reflections
included. The reflected term is smooth and even under each reflection, so
it is a smooth function of those dot products (Schwarz) and a quantum in
d.a or d.b moves it by a quantum's worth, also where d, a or b vanish. The
quantum is the existing one, against the pair's distance to the IMAGE
segment, the scale at which the reflected term varies.

Each class's value is the shipped evaluator on its representative pair, in
the ORIGINAL frame (`evaluate_pairs`): the reflected term needs no
canonical placement, and the representatives come in runs of observer rows
with many sources each, the evaluators' efficient shape. Over ground the
fill holds two class maps, the direct and the reflected, as over PEC.

ORIENTATION (momwire#1969). Both keys are measured along the segments'
stored tangents, and a tangent is a convention: reversing one (s -> -s
along it) is the same physical pair. So a mirror image of a closed loop,
which reverses the loop's direction and so both tangents of a pair, or the
offsets +k and -k along one straight run, would key apart. Reversing a
tangent moves the key and the value by known signs:

* the SOURCE's: z, td (direct) and d.a, a.b (reflected) change sign, and
  its two end codes swap (the source type, under the extended kernel; the
  reflected term's t_nz too). Its basis shapes, as vectors along the new
  tangent, are -const, +sin, -cos of the old: the odd shape's own sign
  cancels the current's;
* the OBSERVER's: rdt, td (direct) and d.b, a.b (reflected) change sign,
  with t_mz in the reflected term's observer type, and the tested component
  E.t_obs changes sign in every shape.

Each pair is keyed in the orientation whose key is the largest of its four
(lexicographically: the source-odd column, the observer-odd column, the
both-odd one, then the types), and its flips ride in the class map's two
low bits (`CODE_BITS`): `CODE_SIGNS` is the sign each flip combination puts
on the three shapes, applied at the gather. A class's stored value is the
canonical orientation's, its representative's own value times its code's
signs (`evaluate`, `evaluate_classes`). A pair whose key is unchanged by a
flip has the shapes that flip negates equal to zero up to rounding, so
which of its orientations is taken is immaterial. Near pairs keep code 0.
"""

from __future__ import annotations

import collections
import functools
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from ._accel import acc as _acc

#: Relative resolution of the key, in bits.
KEY_BITS = 48

#: Pairs whose observer lies within this many kernel scales (`_near_scale`:
#: the source's length, or the observer's radius where that is longer) of
#: the source's centre are not shared: each is evaluated directly, in the
#: original frame. Their field is near-singular, and the basis sums over neighbouring
#: sources cancel most of it, so a geometry rounding other than the
#: structure's own is amplified there (measured: canonical-frame near pairs
#: moved a 16-bowtie array's input impedance 7.7e-11, original-frame ones
#: 6e-14).
NEAR_LENGTHS = 4.0

#: Observer rows per direct near-pair kernel call (their near sources'
#: union is the call's source list).
NEAR_ROWS = 16

#: Observer rows per representative-pair call in the ground plan
#: (`build_ground_plan`): a call evaluates its rows against the union of
#: their representatives' sources. Measured on the ground skyloop at 2097
#: segments (Skylake, 8 threads, momwire#1970), the calls' summed wall per
#: frequency: 16 rows 0.20 s (1.9 pairs evaluated per class), 4 rows 0.18 s,
#: 1 row 0.56 s (the Python around each call).
REP_ROWS = 16

_NEAR_SENTINEL = -1  # below every packed exponent-and-type column

#: Sources per kernel call in `evaluate` (its per-row scratch is ~360 bytes
#: per source). At 8192 each worker's scratch spilled out of its core's
#: cache: on the ELT whip at 8 workers (Rocketlake) the kernel's summed
#: seconds were 3.7x one worker's. Measured per frequency there, wall:
#: 8192 0.086 s, 4096 0.059, 2048 0.048, 1024 0.080 (the Python per call
#: holds the GIL), 512 0.135 (momwire#1916). A speed knob, but not a
#: bit-free one: the kernel's batched sincos is a vector loop with a scalar
#: tail, and which a source falls in can move its Φ by an ulp.
EVAL_CHUNK = 2048

#: Classes per `rep_scalars` + `canonical_sources` call in `build_plan`: the
#: gathered geometry, the scalars and the placement are ~300 B a class, 20 MB
#: a call here, beside a plan that holds the budget's rest. Elementwise, so
#: the chunk moves no value.
_CANON_CHUNK = 1 << 16

#: `pair_reuse="verify"`: per-pair agreement bound, |Φ_reused - Φ_direct|
#: over the pair's largest |Φ_direct| among the three shapes.
#:
#: It sits above the direct kernel's own reproducibility, not at the
#: typical pair's agreement (~1e-12). A far pair whose observer lies close
#: to the source's AXIS (a vertical arm's image under a horizontal one, say)
#: has a shape formed as a difference of much larger end terms, which the
#: kernel rounds at a few 1e-10 of the pair's largest |Φ|: the same
#: pair spelled from a mirror image, or with its observer moved 4 ulps,
#: reads 4e-10 apart (the jack of momwire#1969's second finding, sin shape,
#: PEC image, extended kernel). The canonical key merges mirror images, so
#: more sampled pairs meet a congruent representative spelled differently.
#: Measured (Skylake), verify's largest per-pair difference, with every
#: matrix and feed impedance correct (max |dZ| / max |Z| <= 5.4e-14):
#:
#:   probe                                         fills   largest
#:   adversarial decks, 4 grounds x EK x 3 freqs     354   8.1e-11
#:   six-arm jack, the stored key (97262e36)         432   4.6e-9
#:   six-arm jack, the canonical key                 864   1.0e-8
#:
#: and on deliberate defects, the smallest any test or probe injects: a
#: source keyed without its end-code swap 5.4e-5 (feed impedance 540x the
#: drift), a reflected code map with its flips swapped 2.3e-4, a wrong sign
#: entry 8.2e-3. The bound has 10x over the noise and 500x under the
#: smallest defect. That defect read 5.4e-5 here and 2.7e-7 in the feed
#: impedance: at that ratio, 1e-7 here is the drift bound (5e-10) there.
VERIFY_TOL = 1e-7

#: `pair_reuse="verify"`: pairs checked per observer band.
VERIFY_SAMPLES = 512

#: Whether the sector route (`rotational_symmetry=True`, momwire#1912) takes
#: `pair_reuse`; off, it declines and fills as without it. Its fill holds
#: sector 0's rows and the axial rows only, and the plan then covers just
#: those (momwire#1969, first finding: a plan over every row made the route
#: up to 1.8x slower). What is left to reuse is small, because the rotation
#: was most of it. Measured on Rocketlake, 8 threads, 13 frequencies, free
#: space, sector-row plan against the route alone:
#:
#:   deck     pairs   reuse  census  fill/freq (reuse vs direct)  run
#:   sky_c48  5.86 M  1.49x  0.71 s  0.081 vs 0.129 s             4.22 vs 4.12 s
#:   pc_m40   3.58 M  1.25x  0.47 s  0.057 vs 0.096 s             3.08 vs 2.86 s
#:
#: Per frequency the reuse fill is ~40 % faster, but the census, paid once
#: per geometry at ~120 ns of wall a pair (8 threads; the kernel is ~19 ns),
#: takes 12 to 15 frequencies to repay, so a one-deck run comes out even or
#: a few percent slower. A cheaper census, or a sweep long enough to repay
#: it, is what turning this on needs.
SECTOR_ROUTE = False

#: Segments below which `pair_reuse` steps aside (`pair_reuse_declined`),
#: the census costing more than a small deck's fill saves: in free space and
#: over the PEC image, and over a finite ground (Sommerfeld or reflection
#: coefficient), whose image term is dearer per pair and so repays sooner.
#: `pair_reuse="verify"` ignores them, so a test can still check the reuse
#: on a small deck. Measured, wall seconds per deck, warm, one resident
#: process. Free space, Skylake, 4 threads, 101 frequencies (reuse off /
#: the stored-tangent key / the canonical key):
#:
#:   deck     segs   off    stored  canonical
#:   sfd_h5    203   0.91   1.09    1.10       (20 % slower)
#:   sfd_h9    365   1.71   1.67    1.68
#:   sfd_h13   529   2.62   2.18    2.24
#:
#: Over Sommerfeld, Skylake (momwire#1970's ladder), off -> on, 8 and 4
#: threads:
#:
#:   deck     segs   8 threads      4 threads
#:   gsfd_h1    41   1.12 -> 1.28   1.04 -> 1.29
#:   gsfd_h3   123   1.73 -> 1.89   1.70 -> 1.71
#:   gsfd_h5   203   2.50 -> 2.54   2.67 -> 2.30
#:   gsfd_h9   365   5.22 -> 4.39   5.24 -> 4.01
SMALL_DECK_SEGS_FREE = 400
SMALL_DECK_SEGS_GROUND = 200

#: Whether the plans key pairs in the orientation-canonical form
#: (momwire#1969, module docstring). Off, every pair keys along its stored
#: tangents and every code is 0, the classes as before.
CANONICAL = True

#: Bits of a coded class map entry that hold the flips: entry = class << 2 |
#: code, code bit 0 the source tangent reversed, bit 1 the observer's.
CODE_BITS = 2

#: Pairs above which a plan is not coded (its class numbers would not fit
#: an int32 beside the code) and keys along the stored tangents.
CODED_PAIRS = 1 << (31 - CODE_BITS)

#: The sign each flip combination puts on the (const, sin, cos) shapes,
#: indexed by code. Source reversed: -const, +sin, -cos (the current's sign
#: times each shape's parity); observer reversed: every shape negated.
CODE_SIGNS = np.array(
    [
        [1.0, 1.0, 1.0],  # 0: as stored
        [-1.0, 1.0, -1.0],  # 1: source reversed
        [-1.0, -1.0, -1.0],  # 2: observer reversed
        [1.0, -1.0, 1.0],  # 3: both
    ]
)

_HAVE_FACTORIZE = _acc is not None and bool(
    getattr(_acc, "exact_factorize_1224", False)
)
_HAVE_INT_GROUPS = _acc is not None and bool(getattr(_acc, "int_groups", False))


#: The byte budget of one fill's plans AND their census, razor-2p's
#: (`_razor_pair_reuse_core.h`, PLAN_BUDGET_BYTES) and for its reason: the
#: 512 MB hosted bound is this, one dense Z (144 MB at the 3000-segment cap)
#: and the interpreter. Past it the fill declines (`over_budget_reason`) and
#: runs as with reuse off (momwire#2019). Measured, Skylake, 4 threads, free
#: space unless named, peak RSS of the solving process in MB, reuse off / on
#: before the budget / on with it:
#:
#:   deck                            segs   off   before  with
#:   20 elements, each its own h     3220   421   1663    451  declines
#:   20 elements of 5 lengths        3020   382   1277    397  declines
#:   20 equal elements               3020   382    521    447  reuses
#:   the same, unequal spacing       3020   382    503    433  reuses
#:   the same, equal, Sommerfeld     3020   588    523    487  reuses
#:   uniform dipole                  4001   597    765    670  reuses
#:   11 elements, each its own h     1111   164    296    292  reuses
PLAN_BUDGET_BYTES = 256 << 20

#: Bytes of plans the process keeps between solvers (`shared_plan`), one
#: fill's budget (momwire#2026): a long-lived server holds no more than a
#: single fill may use. A plan is mostly its int32 class map, 4 bytes per
#: pair: the ELT whip's direct block is ~35 MB.
SHARED_PLAN_BYTES = PLAN_BUDGET_BYTES

#: Bytes per class and per near pair of one wavenumber's values, the three
#: complex shapes (`evaluate`, `evaluate_near`).
VALUE_BYTES = 48

#: THE CENSUS folds each band's classes, in band order, into one running
#: table (`_Fold`), so what it holds grows with the deck's classes, not with
#: each band's: a class seen in many bands is held once. Holding every band's
#: classes until one merge at the end, as before, cost ~240 B per BAND class,
#: and on a deck whose rows repeat that is the deck's classes several times
#: over: the ELT whip (4,392 segments, 94 bands of 47 rows at 8 threads) has
#: 0.92 M classes in 5.0 M band classes, a census counted at 1.22 GB (peak
#: RSS over its entry, Rocketlake: 1.09 GB) for a plan of 201 MB, so its
#: reuse declined on the 256 MB budget. A class is numbered where it first
#: occurs in the pairs' flat order either way, so the plan is the same bytes.
#:
#: What the census holds is counted (`_Fold.held`) from the class map, the
#: table's keys and slots (`IntGroups.nbytes`), each class's representative
#: and the near pairs, plus `BAND_BYTES_PER_PAIR` for every band in flight or
#: finished and not yet folded. A band starts only where it fits beside that
#: under the budget, and the census stops once the count, or the plan's
#: least possible fill (`PLAN_FLOOR_BYTES_PER_CLASS`), passes it: the classes
#: only grow, so either verdict is final, and a deck that declines has held
#: no more than the budget on the way.
#:
#: Measured on the whip at 8 threads (Rocketlake): counted 200 MB, and peak
#: RSS over the census's entry 210 MB to the last fold, 228 MB to the plan.
#:
#: Peak transient of one band per pair: its scalars, key columns, codes and
#: grouping (the hash table, at twice the pairs, and its inverses). Measured
#: (Rocketlake, one worker, so one band at a time, peak RSS over the band's
#: entry): the direct term at most 130 B (the whip and the LogYag, 33 and 4
#: bands), the reflected term at most 156 B (curtains of 8 and 20 wires over
#: a reflection-coefficient ground), each taken here with 5 to 8 % over.
BAND_BYTES_PER_PAIR = 136
GROUND_BAND_BYTES_PER_PAIR = 168

#: The share of the budget the bands in flight may take: each band is cut to
#: at most this over a pool's worth (`_band_rows`), so the rest is the
#: classes'. Smaller bands move no class (the numbering is the flat order's).
BAND_SHARE = 0.25

#: The plan's fill holds at least this much per class beside its class map:
#: one wavenumber's values (`VALUE_BYTES`) and an int64 index (`rep_src`, or
#: the ground plan's `rep_flat`).
PLAN_FLOOR_BYTES_PER_CLASS = VALUE_BYTES + 8

_shared_plans: OrderedDict = OrderedDict()
_shared_lock = threading.Lock()


class _OverBudget:
    """A census that stopped at `PLAN_BUDGET_BYTES`: kept in the plan cache
    like a plan, so that a sweep does not repeat it per frequency."""


OVER_BUDGET = _OverBudget()


def over_budget_reason() -> str:
    """`pair_reuse_declined` for a deck past the budget, razor-2p's words."""
    return (
        "the plan and its census would exceed the "
        f"{PLAN_BUDGET_BYTES / 2**20:.6g} MB budget"
    )


def kernel_evals(plan) -> int:
    """The (observer, source) pairs one frequency's evaluation hands the
    kernel: a class's canonical row each, and every direct call's rectangle
    of rows by sources, which exceeds the pairs it needs (`near_blocks`).
    Kept on the plan, which never changes once built."""
    held = plan.__dict__.get("_kernel_evals")
    if held is None:
        blocks = (*plan.near_blocks, *getattr(plan, "rep_blocks", ()))
        held = sum((b[1] - b[0]) * b[2].size for b in blocks)
        if not hasattr(plan, "rep_blocks"):
            held += plan.n_classes
        plan.__dict__["_kernel_evals"] = held
    return int(held)


def plan_nbytes(plan) -> int:
    """The arrays a plan holds, in bytes (its lists of slices are small)."""
    if plan is OVER_BUDGET:
        return 0
    return sum(
        v.nbytes for v in vars(plan).values() if isinstance(v, np.ndarray)
    ) + sum(
        x.nbytes
        for blk in (*plan.near_blocks, *getattr(plan, "rep_blocks", ()))
        for x in blk
        if isinstance(x, np.ndarray)
    )


def fill_nbytes(plan) -> int:
    """What a plan costs a fill: its arrays and one wavenumber's values."""
    return plan_nbytes(plan) + VALUE_BYTES * (plan.n_classes + plan.n_near)


def _within_budget(plan, budget):
    """`plan`, or `OVER_BUDGET` where its fill would pass `budget`."""
    return plan if budget is None or fill_nbytes(plan) <= budget else OVER_BUDGET


def shared_plan(key, build):
    """The plan for `key` from the process's cache, else `build()`'s, kept.

    A plan holds no k, and its key is the geometry's bytes, so it outlives
    the solver that found it: the portal builds one solver per frequency and
    a design server one per knob turn back to a value, and each would pay the
    census again with a per-instance cache (momwire#1916 part C). Plans are
    never written after `build_plan` returns, so solvers share one object.
    Least recently used first out past `SHARED_PLAN_BYTES`, the newest kept
    whatever its size. Two threads missing the same key both build; the
    second's plan replaces the first's, and either is the same classes.
    """
    with _shared_lock:
        held = _shared_plans.get(key)
        if held is not None:
            _shared_plans.move_to_end(key)
            return held[0]
    plan = build()
    # The census's pages go back before the fill: they sit free in the pool's
    # arenas, where Z and the fill's arrays would not reuse them.
    from .razor import _trim_heap

    _trim_heap()
    # A declined census holds no arrays, but its key is the deck's geometry.
    size = plan_nbytes(plan) or sum(len(x) for x in key if isinstance(x, bytes))
    with _shared_lock:
        _shared_plans[key] = (plan, size)
        _shared_plans.move_to_end(key)
        total = sum(v[1] for v in _shared_plans.values())
        while total > SHARED_PLAN_BYTES and len(_shared_plans) > 1:
            _, (_, dropped) = _shared_plans.popitem(last=False)
            total -= dropped
    return plan


def clear_shared_plans() -> None:
    """Drop every kept plan (tests, and a caller done with its decks)."""
    with _shared_lock:
        _shared_plans.clear()


class PairReuseMismatch(AssertionError):
    """`pair_reuse="verify"` found a reused Φ that the direct kernel does not
    reproduce."""


@dataclass
class Plan:
    """Which class each (observer, source) pair belongs to, and how to
    evaluate each class once."""

    cls: np.ndarray  # (N_obs, N_src) int32 class of each pair
    rep_src: np.ndarray  # (n_cls,) int64 the representative's source segment
    near_flat: np.ndarray  # (n_near,) int64 sorted flat index m * N_src + n
    near_blocks: list  # (m0, m1, sources, row, col) per direct kernel call
    # The canonical evaluation's inputs, k-independent, so built with the
    # classes (momwire#1916): each class's source in the observer frame
    # (`canonical_sources` of the representative's rho, z, rdt, td), and the
    # runs of classes sharing an observer radius, as (c0, c1, observer
    # segment). Classes are numbered so that each run is contiguous.
    src_c: np.ndarray  # (n_cls, 3) float64
    src_t: np.ndarray  # (n_cls, 3) float64
    runs: list
    # momwire#1969: with code_bits, cls holds class << code_bits | code, and
    # rep_code is each representative's own code (its value, evaluated as
    # it stands, times CODE_SIGNS[rep_code] is the canonical one).
    rep_code: np.ndarray | None = None  # (n_cls,) int8
    code_bits: int = 0
    # The census's own peak by `CENSUS_BYTES_PER_BAND_CLASS`, for the fill
    # report (momwire#2016).
    census_peak_bytes: int | None = None

    @property
    def n_near(self) -> int:
        return int(self.near_flat.size)

    @property
    def n_pairs(self) -> int:
        return int(self.cls.size)

    @property
    def n_classes(self) -> int:
        return int(self.rep_src.shape[0])


@dataclass
class GroundPlan:
    """The reflected term's classes (`build_ground_plan`): each class's
    value is its representative pair's, evaluated where it stands."""

    cls: np.ndarray  # (N_obs, N_src) int32 class of each pair
    rep_flat: np.ndarray  # (n_cls,) int64 sorted flat index of each representative
    rep_blocks: list  # `near_blocks` over rep_flat
    near_flat: np.ndarray  # (n_near,) int64 sorted flat index m * N_src + n
    near_blocks: list
    rep_code: np.ndarray | None = None  # as `Plan`'s (momwire#1969)
    code_bits: int = 0
    census_peak_bytes: int | None = None  # as `Plan`'s (momwire#2016)

    @property
    def n_near(self) -> int:
        return int(self.near_flat.size)

    @property
    def n_pairs(self) -> int:
        return int(self.cls.size)

    @property
    def n_classes(self) -> int:
        return int(self.rep_flat.size)


def _types(cols):
    """Dense ids for the distinct rows of a few per-segment columns."""
    stacked = np.stack([np.asarray(c) for c in cols], axis=1)
    _, ids = np.unique(stacked, axis=0, return_inverse=True)
    return np.asarray(ids, dtype=np.int64).reshape(-1)


def _quantise_length(h):
    """(exponent, mantissa) of a positive length at KEY_BITS relative bits."""
    m, e = np.frexp(h)
    return e.astype(np.int64), np.rint(np.ldexp(m, KEY_BITS)).astype(np.int64)


def segment_types(seg_h, a_obs, src_a=None, ind1=None, ind2=None):
    """Per-segment source-type and observer-type ids.

    The source type holds what Φ reads of the source besides its pose: its
    half-length (quantised like the key) and, under the extended kernel, its
    radius and end codes (exact). The observer type is its radius (exact),
    the scalar each kernel call takes."""
    e, q = _quantise_length(np.asarray(seg_h, dtype=np.float64))
    cols = [e, q]
    if src_a is not None:
        cols += [
            np.asarray(src_a, dtype=np.float64).view(np.int64),
            np.asarray(ind1, dtype=np.int64),
            np.asarray(ind2, dtype=np.int64),
        ]
    src_type = _types(cols)
    obs_type = _types([np.asarray(a_obs, dtype=np.float64).view(np.int64)])
    return src_type, obs_type


def _quantise_unit(x):
    """A direction cosine at KEY_BITS absolute bits."""
    return np.rint(np.asarray(x, dtype=np.float64) * float(1 << KEY_BITS)).astype(
        np.int64
    )


def ground_segment_types(src_type, obs_type, seg_t):
    """`segment_types`' ids refined by each segment's vertical tangent
    component, which the reflected term reads and a horizontal isometry
    keeps (`build_ground_plan`)."""
    tz = _quantise_unit(np.asarray(seg_t, dtype=np.float64)[:, 2])
    return _types([src_type, tz]), _types([obs_type, tz])


def reversible_segment_types(seg_h, a_obs, src_a=None, ind1=None, ind2=None):
    """`segment_types`, and each source's type with its tangent reversed, in
    one id space: (src_type, obs_type, src_rev). Reversal swaps the two end
    codes (the extended kernel's); nothing else the type holds has a
    direction, so without them src_rev is src_type."""
    if src_a is None:
        src_type, obs_type = segment_types(seg_h, a_obs)
        return src_type, obs_type, src_type
    n = len(seg_h)

    def twice(x):
        x = np.asarray(x)
        return np.concatenate([x, x])

    src_type, obs_type = segment_types(
        twice(seg_h),
        twice(a_obs),
        twice(src_a),
        np.concatenate([np.asarray(ind1), np.asarray(ind2)]),
        np.concatenate([np.asarray(ind2), np.asarray(ind1)]),
    )
    return src_type[:n], obs_type[:n], src_type[n:]


def reversible_ground_types(src_type, obs_type, src_rev, seg_t):
    """`ground_segment_types` for both orientations of every segment, in one
    id space: (g_src, g_obs, g_src_rev, g_obs_rev). Reversing a tangent
    negates its vertical component, and a source's end codes swap."""
    n = len(src_type)
    seg_t = np.asarray(seg_t, dtype=np.float64)
    g_src, g_obs = ground_segment_types(
        np.concatenate([src_type, src_rev]),
        np.concatenate([obs_type, obs_type]),
        np.concatenate([seg_t, -seg_t]),
    )
    return g_src[:n], g_obs[:n], g_src[n:], g_obs[n:]


def pair_scalars(obs_c, obs_t, src_c, src_t):
    """z, rho, td, rdt for every (observer, source) pair, (R, N) each, in
    the kernels' own order of operations."""
    return _scalars(obs_c[:, None], obs_t[:, None], src_c[None, :], src_t[None, :])


def rep_scalars(obs_c, obs_t, src_c, src_t, obs, src):
    """`pair_scalars` of the pairs (obs[i], src[i]) only, (G,) each: the same
    operations element by element, so the same bits as the pair's entry in
    its band's (R, N) arrays."""
    return _scalars(obs_c[obs], obs_t[obs], src_c[src], src_t[src])


def _scalars(oc, ot, sc, st):
    # Elementwise in a fixed order: the operands' shapes decide which pairs,
    # never a value.
    rvx = oc[..., 0] - sc[..., 0]
    rvy = oc[..., 1] - sc[..., 1]
    rvz = oc[..., 2] - sc[..., 2]
    tnx, tny, tnz = st[..., 0], st[..., 1], st[..., 2]
    z = rvx * tnx + rvy * tny + rvz * tnz
    rvx -= z * tnx
    rvy -= z * tny
    rvz -= z * tnz
    rho = np.sqrt(rvx * rvx + rvy * rvy + rvz * rvz)
    tmx, tmy, tmz = ot[..., 0], ot[..., 1], ot[..., 2]
    td = tmx * tnx + tmy * tny + tmz * tnz
    rdt = rvx * tmx + rvy * tmy + rvz * tmz
    return z, rho, td, rdt


def _near_scale(H, a_obs):
    """The near rule's kernel scale per (observer, source) pair, (R, N):
    max(2H, a), the source's length or the observer's radius, whichever is
    longer (see the module docstring)."""
    return np.maximum(2.0 * H[None, :], np.asarray(a_obs)[:, None])


def _band_keys(z, rho, td, rdt, H, a_obs, type_id, near=False):
    """The quantised key columns of one band (int64, (R*N,) each).

    With `near=True`, a NEAR pair (`NEAR_LENGTHS`) takes the one sentinel
    key: its value is the direct kernel's, written over the gather. Returns
    the columns and the near mask (or None)."""
    # The pair's singular distance, from the observer centre to the source
    # SEGMENT (its axis beside the span, its nearer end beyond it), within
    # a factor sqrt(2) as max(rho, |z| - H), floored at the observer radius.
    az = np.abs(z)
    near = (
        (np.maximum(az, rho) < NEAR_LENGTHS * _near_scale(H, a_obs)).reshape(-1)
        if near and NEAR_LENGTHS > 0
        else None
    )
    az -= H[None, :]
    scale = np.maximum(az, rho, out=az)
    np.maximum(scale, a_obs[:, None], out=scale)
    # frexp's exponent and 2^(KEY_BITS - e), off the bits (scale is a
    # positive normal number): the same integers, without two slow ufuncs.
    e = (scale.view(np.int64) >> 52) - 1022
    del scale, az
    step_inv = ((1023 + KEY_BITS - e) << 52).view(np.float64)
    zq = np.rint(z * step_inv).astype(np.int64)
    rq = np.rint(rho * step_inv).astype(np.int64)
    dq = np.rint(rdt * step_inv).astype(np.int64)
    del step_inv
    tq = np.rint(td * float(1 << KEY_BITS)).astype(np.int64)
    # The exponent and the type share a column: |e| < 2^11 always, and the
    # type id is below N^2 < 2^40.
    e += 1 << 11
    e <<= 40
    e += type_id
    cols = [c.reshape(-1) for c in (e, zq, rq, dq, tq)]
    if near is not None and near.any():
        idx = np.flatnonzero(near)
        cols[0][idx] = _NEAR_SENTINEL
        for c in cols[1:]:
            c[idx] = 0
    return cols, near


def _canonical_codes(cols, ia, ib, it, near, n_src, ds=None, do=None):
    """Key one band's pairs in their canonical orientation, in place, and
    return each pair's flips (int8 codes, `CODE_BITS`).

    `cols` are a band's key columns, (R*N,) each: `cols[ia]` is odd under a
    source reversal only, `cols[ib]` under an observer reversal only,
    `cols[it]` under either, and `cols[0]` packs the types, which a source
    reversal moves by `ds` (per source, (N,)) and an observer reversal by
    `do` (per row, (R,)); None where a reversal leaves them. Of the four
    orientations the largest key is taken, in the order (ia, ib, it, the
    types): the sign of ia decides the source flip and that of ib the
    observer's, and only where one is zero do the rest decide. Near pairs
    (their key the sentinel) keep code 0."""
    a, b, t, e = cols[ia], cols[ib], cols[it], cols[0]
    fs = a < 0
    fo = b < 0
    za = a == 0
    zb = b == 0
    if near is not None:
        za &= ~near
        zb &= ~near

    def ds_of(idx):
        return None if ds is None else ds[idx % n_src]

    def do_of(idx):
        return None if do is None else do[idx // n_src]

    # Only the observer free: the both-odd column, then the observer types.
    idx = np.flatnonzero(zb & ~za)
    if idx.size:
        tp = np.where(fs[idx], -t[idx], t[idx])
        flip = tp < 0
        d = do_of(idx)
        if d is not None:
            flip |= (tp == 0) & (d > 0)
        fo[idx] = flip
    # Only the source free: likewise, then the source types.
    idx = np.flatnonzero(za & ~zb)
    if idx.size:
        tp = np.where(fo[idx], -t[idx], t[idx])
        flip = tp < 0
        d = ds_of(idx)
        if d is not None:
            flip |= (tp == 0) & (d > 0)
        fs[idx] = flip
    # Both free: the largest of the four (t, types).
    idx = np.flatnonzero(za & zb)
    if idx.size:
        t_i = t[idx]
        d_s = ds_of(idx)
        d_o = do_of(idx)
        valid = np.zeros(idx.size, dtype=bool)
        best_t = np.zeros(idx.size, dtype=np.int64)
        best_e = np.zeros(idx.size, dtype=np.int64)
        best_s = np.zeros(idx.size, dtype=bool)
        best_o = np.zeros(idx.size, dtype=bool)
        for cs in (False, True):
            for co in (False, True):
                tc = -t_i if cs != co else t_i
                ec = np.zeros(idx.size, dtype=np.int64)
                if cs and d_s is not None:
                    ec += d_s
                if co and d_o is not None:
                    ec += d_o
                better = ~valid | (tc > best_t) | ((tc == best_t) & (ec > best_e))
                best_t = np.where(better, tc, best_t)
                best_e = np.where(better, ec, best_e)
                best_s[better] = cs
                best_o[better] = co
                valid[:] = True
        fs[idx] = best_s
        fo[idx] = best_o
    np.negative(a, out=a, where=fs)
    np.negative(b, out=b, where=fo)
    np.negative(t, out=t, where=fs ^ fo)
    R = a.size // n_src
    if ds is not None:
        e.reshape(R, n_src)[...] += ds[None, :] * fs.reshape(R, n_src)
    if do is not None:
        e.reshape(R, n_src)[...] += do[:, None] * fo.reshape(R, n_src)
    code = fs.view(np.int8).copy()
    code |= fo.view(np.int8) << 1
    return code


def _coded_band(inv, code):
    """A band's class map entries: class << CODE_BITS | code."""
    out = inv.astype(np.int32)
    out <<= CODE_BITS
    out |= code
    return out


def _renumber(cls_rows, ginv, coded):
    """A band's class map, its band-local classes replaced by the merged
    ones, the codes (if any) kept."""
    g = ginv.astype(np.int32)
    if not coded:
        return g[cls_rows]
    out = g[cls_rows >> CODE_BITS]
    out <<= CODE_BITS
    out |= cls_rows & ((1 << CODE_BITS) - 1)
    return out


def _group(cols):
    """`(first, inverse)`: exact-equality groups of the rows of int64
    columns. The hash kernel takes three columns, so five are grouped in
    two stages, the first stage's ids a column of the second; without it, numpy's
    sorted spelling (other group numbers, the same groups)."""
    if _HAVE_FACTORIZE and cols[0].shape[0] < 2**31 - 2:
        first, ids = _acc.factorize_ints(list(cols[:3]))
        rest = list(cols[3:])
        while rest:
            take, rest = rest[:2], rest[2:]
            first, ids = _acc.factorize_ints([np.asarray(ids, np.int64), *take])
        return np.asarray(first, np.int64), np.asarray(ids, np.int64).reshape(-1)
    _, first, inv = np.unique(
        np.stack(cols, axis=1), axis=0, return_index=True, return_inverse=True
    )
    return np.asarray(first, np.int64), np.asarray(inv, np.int64).reshape(-1)


def _limit_openmp():
    """A pool worker's kernels run on one OpenMP thread: the pool is the
    parallelism. libgomp keeps the thread count per calling thread, so this
    leaves every other thread's as it was."""
    from threadpoolctl import ThreadpoolController

    ThreadpoolController().select(user_api="openmp").limit(limits=1)


@functools.lru_cache(maxsize=1)
def _pool():
    """The census's, the evaluation's and the gather's pool: as many workers
    as this process may run on, held to OMP_NUM_THREADS where the caller
    pinned one."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    try:
        n = min(n, int(os.environ.get("OMP_NUM_THREADS", "") or n))
    except ValueError:
        pass
    return ThreadPoolExecutor(max_workers=max(1, n), initializer=_limit_openmp)


def _workers():
    return _pool()._max_workers


#: Tasks per worker `_run` cuts a long item list into. One future per item
#: cost ~1,300 submits per frequency on the ELT whip (1,032 near blocks and
#: ~280 evaluation pieces), each a submit, a lock and a wake-up; a few
#: contiguous runs per worker keep the load balanced at a fraction of that
#: (momwire#1916 part D). Each item is still its own `fn` call on its own
#: slice, so no value moves.
RUN_TASKS_PER_WORKER = 4


def _run(fn, items):
    """`fn` over `items`, on the pool when it has more than one worker:
    contiguous runs of items per task once there are more items than
    `RUN_TASKS_PER_WORKER` tasks per worker. Results in `items`' order."""
    workers = _workers()
    if workers == 1 or len(items) == 1:
        return [fn(x) for x in items]
    items = list(items)
    n_tasks = workers * RUN_TASKS_PER_WORKER
    if len(items) <= n_tasks:
        return list(_pool().map(fn, items))
    cuts = [len(items) * t // n_tasks for t in range(n_tasks + 1)]
    runs = [items[a:b] for a, b in zip(cuts[:-1], cuts[1:])]

    def run(chunk):
        return [fn(x) for x in chunk]

    return [r for chunk in _pool().map(run, runs) for r in chunk]


class _PyGroups:
    """`IntGroups` without the extension: the held keys and a band's grouped
    again at each add (`_group`), renumbered in first-appearance order, so
    the groups and their numbers are the kernel's."""

    def __init__(self, width):
        self._keys = [np.zeros(0, np.int64) for _ in range(width)]

    def __len__(self):
        return int(self._keys[0].size)

    def nbytes(self):
        return sum(k.nbytes for k in self._keys)

    def add(self, cols):
        g0 = len(self)
        joined = [
            np.concatenate([k, np.asarray(c, np.int64)])
            for k, c in zip(self._keys, cols)
        ]
        first, inv = _group(joined)
        order = np.argsort(first, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        self._keys = [j[first[order]] for j in joined]
        return rank[inv[g0:]]


def _nbytes(x):
    """The bytes of the arrays in a (nested) band result."""
    if isinstance(x, np.ndarray):
        return x.nbytes
    if isinstance(x, (list, tuple)):
        return sum(_nbytes(y) for y in x)
    return 0


class _Fold:
    """The census's classes, each band's folded in band order into one table
    (THE CENSUS, above). `add(keys, kept, near)` takes a band's distinct
    keys, its per-class payload `kept` (arrays over the same rows) and its
    near pairs, and returns each key's class, numbered where the class first
    occurs in the bands' order: one `_group` over every band's keys in turn
    numbers them so. Only the payload of the classes a band was first to see
    is kept."""

    def __init__(self, width, map_bytes):
        self.table = _acc.IntGroups(width) if _HAVE_INT_GROUPS else _PyGroups(width)
        self.map_bytes = map_bytes
        self.kept = []
        self.near = []
        self._kept_bytes = 0

    def __len__(self):
        return len(self.table)

    def add(self, keys, kept, near):
        g0 = len(self.table)
        ids = np.asarray(self.table.add(list(keys)), dtype=np.int64)
        # A band's keys are distinct, so each new class is one row, and the
        # rows of the new classes come in their number order.
        new = np.flatnonzero(ids >= g0)
        part = tuple(np.ascontiguousarray(p[new]) for p in kept)
        self.kept.append(part)
        self.near.append(near)
        self._kept_bytes += _nbytes(part) + near.nbytes
        return ids

    def held(self):
        """The bytes the census holds, beside the bands in flight."""
        return self.map_bytes + int(self.table.nbytes()) + self._kept_bytes

    def floor(self):
        """The least the plan's fill can hold (`PLAN_FLOOR_BYTES_PER_CLASS`)."""
        return self.map_bytes + PLAN_FLOOR_BYTES_PER_CLASS * len(self)

    def joined(self, j):
        """Payload column j over every class, in class order."""
        return np.concatenate([p[j] for p in self.kept])

    def near_flat(self):
        return np.concatenate(self.near) if self.near else np.zeros(0, np.int64)

    def release(self):
        """Drop the payload once the plan holds its joined copies."""
        self.kept.clear()
        self.near.clear()


def _band_rows(rows, M, N, budget, per_pair):
    """Observer rows per census band: `rows` (the swept budget's) shared out
    over the pool, none so small that its per-band cost shows, and under a
    byte budget at most `BAND_SHARE` of it over a pool's worth of bands."""
    workers = _workers()
    rows = max(1, int(rows) // workers)
    # Enough bands to share out, none so small that its per-band cost shows.
    rows = min(rows, max(16, -(-M // (4 * workers))))
    if budget is not None and per_pair > 0:
        rows = min(rows, max(1, int(BAND_SHARE * budget) // (workers * N * per_pair)))
    return rows


def _trim_census():
    # The table's pages and the bands' (freed in the pool's arenas) go back
    # before the plan is built beside the map: freed, they still count in
    # the process's resident set.
    from .razor import _trim_heap

    _trim_heap()


def _census_bands(band, starts, census, fold, budget, band_bytes):
    """`band(start)` over `starts` on the pool, each result handed to
    `fold(i, result)` in band order, which folds it into `census` (a
    `_Fold`). Returns the most the census held (`_Fold.held` and
    `band_bytes` per band started and not yet folded), or None once that or
    the plan's floor passes `budget` bytes; a band starts only where it fits.
    `budget` None runs a pool's worth at once throughout."""
    n = len(starts)
    workers = _workers()
    peak = 0
    pending = collections.deque()
    nxt = 0

    def fits(k):
        return budget is None or census.held() + band_bytes * k <= budget

    try:
        for i in range(n):
            while nxt < n and len(pending) < workers and fits(len(pending) + 1):
                if workers == 1:
                    pending.append(band(starts[nxt]))
                else:
                    pending.append(_pool().submit(band, starts[nxt]))
                nxt += 1
            if not pending:  # not one band fits beside what is held
                return None
            peak = max(peak, census.held() + band_bytes * len(pending))
            r = pending.popleft()
            fold(i, r if workers == 1 else r.result())
            del r
            peak = max(peak, census.held() + band_bytes * len(pending))
            if budget is not None and max(peak, census.floor()) > budget:
                return None
        return peak
    finally:
        # A census stopped early: the bands not started never run, and those
        # running finish before the map they write is dropped.
        if workers > 1:
            for p in pending:
                p.cancel()
            for p in pending:
                if not p.cancelled():
                    p.exception()


def build_plan(
    obs_c,
    obs_t,
    src_c,
    src_t,
    seg_h,
    a_obs,
    src_type,
    obs_type,
    rows,
    src_rev=None,
    budget=None,
):
    """The class map over every (observer, source) pair, found in observer
    bands whose transients together hold `rows` rows (O(rows * N_src)).

    With `src_rev` (each source's type reversed, `reversible_segment_types`)
    and `CANONICAL`, the pairs are keyed in their canonical orientation and
    the map is coded (momwire#1969, module docstring).

    With `budget` (bytes), returns `OVER_BUDGET` where the census or the
    plan's fill would pass it (momwire#2019); the map is counted before it is
    allocated."""
    obs_c = np.asarray(obs_c, dtype=np.float64)
    obs_t = np.asarray(obs_t, dtype=np.float64)
    src_c = np.asarray(src_c, dtype=np.float64)
    src_t = np.asarray(src_t, dtype=np.float64)
    a_obs = np.asarray(a_obs, dtype=np.float64)
    H = 0.5 * np.asarray(seg_h, dtype=np.float64)
    M, N = obs_c.shape[0], src_c.shape[0]
    n_obs_types = int(obs_type.max()) + 1
    coded = src_rev is not None and CANONICAL and M * N <= CODED_PAIRS
    ds = None
    if coded:
        ds = (np.asarray(src_rev, np.int64) - src_type) * n_obs_types
        if not ds.any():
            ds = None
    if budget is not None and 4 * M * N > budget:
        return OVER_BUDGET
    cls = np.empty((M, N), dtype=np.int32)
    rows = _band_rows(rows, M, N, budget, BAND_BYTES_PER_PAIR)
    starts = list(range(0, M, rows))

    def band(i0):
        i1 = min(M, i0 + rows)
        z, rho, td, rdt = pair_scalars(obs_c[i0:i1], obs_t[i0:i1], src_c, src_t)
        type_id = src_type[None, :] * n_obs_types + obs_type[i0:i1, None]
        cols, near = _band_keys(z, rho, td, rdt, H, a_obs[i0:i1], type_id, True)
        del type_id
        # (e, z, rho, rdt, td): z odd under the source's reversal, rdt under
        # the observer's, td under either.
        code = _canonical_codes(cols, 1, 3, 4, near, N, ds=ds) if coded else None
        near = (
            np.zeros(0, np.int64)
            if near is None
            else np.flatnonzero(near).astype(np.int64) + i0 * N
        )
        first, inv = _group(cols)
        if coded:
            cls[i0:i1] = _coded_band(inv, code).reshape(i1 - i0, N)
            rep_code = code[first]
        else:
            cls[i0:i1] = inv.reshape(i1 - i0, N)
            rep_code = np.zeros(first.size, np.int8)
        del z, rho, td, rdt
        keys = [c[first] for c in cols]
        r_first, n_first = np.divmod(first, N)
        return keys, r_first + i0, n_first, near, rep_code

    census = _Fold(5, cls.nbytes)

    def fold(i, result):
        # The band's classes numbered as the deck's (`_Fold`), in its rows of
        # the map, codes kept.
        keys, r_first, n_first, near, rep_code = result
        i0 = starts[i]
        kept = (r_first.astype(np.int32), n_first.astype(np.int32), rep_code)
        ids = census.add(keys, kept, near)
        cls[i0 : i0 + rows] = _renumber(cls[i0 : i0 + rows], ids, coded)

    band_bytes = BAND_BYTES_PER_PAIR * rows * N
    census_peak = _census_bands(band, starts, census, fold, budget, band_bytes)
    if census_peak is None:
        return OVER_BUDGET
    census.table = None
    _trim_census()
    # Classes renumbered so those sharing an observer radius are contiguous
    # (stable within a radius): `evaluate` then runs on slices. A class's
    # members and value are unchanged; only its number moves.
    rep_obs = census.joined(0).astype(np.int64)
    perm = np.argsort(obs_type[rep_obs], kind="stable")
    if not np.array_equal(perm, np.arange(perm.size)):
        renumber = np.empty(perm.size, dtype=np.int64)
        renumber[perm] = np.arange(perm.size, dtype=np.int64)
        for i0 in starts:
            cls[i0 : i0 + rows] = _renumber(cls[i0 : i0 + rows], renumber, coded)
        del renumber
    rep_obs = rep_obs[perm]
    rep_src = census.joined(1)[perm].astype(np.int64)
    rep_code = census.joined(2)[perm]
    near_flat = census.near_flat()  # sorted: bands in order
    census.release()
    del perm
    # Each representative's (rho, z, rdt, td), the bits its band computed
    # (`rep_scalars`), placed in the observer frame.
    n_cls = rep_src.size
    can_c = np.empty((n_cls, 3))
    can_t = np.empty((n_cls, 3))
    for c0 in range(0, n_cls, _CANON_CHUNK):  # bounded transients
        c1 = min(n_cls, c0 + _CANON_CHUNK)
        z, rho, td, rdt = rep_scalars(
            obs_c, obs_t, src_c, src_t, rep_obs[c0:c1], rep_src[c0:c1]
        )
        canon = np.stack([rho, z, rdt, td], axis=1)
        can_c[c0:c1], can_t[c0:c1] = canonical_sources(canon)
    otype = obs_type[rep_obs]
    cuts = np.flatnonzero(np.diff(otype)) + 1
    edges = [0, *cuts.tolist(), int(otype.size)]
    runs = [
        (c0, c1, int(rep_obs[c0])) for c0, c1 in zip(edges[:-1], edges[1:]) if c1 > c0
    ]
    plan = Plan(
        cls=cls,
        rep_src=rep_src,
        near_flat=near_flat,
        near_blocks=near_blocks(near_flat, N, a_obs),
        src_c=can_c,
        src_t=can_t,
        runs=runs,
        rep_code=rep_code if coded else None,
        code_bits=CODE_BITS if coded else 0,
        census_peak_bytes=census_peak,
    )
    return _within_budget(plan, budget)


def near_blocks(near_flat, n_src, a_obs, rows=None):
    """The direct kernel calls that evaluate the near pairs: per run of up to
    NEAR_ROWS observer rows of one radius, the union of their near sources,
    and where each near pair sits in that call's (rows, sources) block."""
    rows = NEAR_ROWS if rows is None else rows
    m_all, n_all = np.divmod(near_flat, n_src)
    out = []
    if near_flat.size == 0:
        return out
    # Cut at every NEAR_ROWS boundary and at every change of observer radius.
    cut = np.flatnonzero(np.diff(a_obs)) + 1
    edges = np.union1d(np.arange(0, a_obs.shape[0] + 1, rows), cut)
    edges = np.union1d(edges, [a_obs.shape[0]])
    lo = np.searchsorted(m_all, edges[:-1], side="left")
    hi = np.searchsorted(m_all, edges[1:], side="left")
    for m0, m1, p0, p1 in zip(edges[:-1], edges[1:], lo, hi):
        if p1 == p0:
            continue
        ns, col = np.unique(n_all[p0:p1], return_inverse=True)
        out.append(
            (int(m0), int(m1), ns, m_all[p0:p1] - m0, col.reshape(-1), int(p0), int(p1))
        )
    return out


def evaluate_near(plan, direct):
    """The near pairs' Φ, (n_near, 3): `direct(m0, m1, ns)` is the shipped
    kernel on observer rows m0:m1 against source segments `ns`, in the
    original frame."""
    return evaluate_pairs(plan.n_near, plan.near_blocks, direct)


def evaluate_pairs(n, blocks, direct):
    """(n, 3) values of the pairs `blocks` (`near_blocks`) lay out, each
    block one `direct(m0, m1, ns)` call on its rows and sources."""
    out = np.empty((n, 3), dtype=np.complex128)

    def one(block):
        m0, m1, ns, r, c, p0, p1 = block
        phis = direct(m0, m1, ns)
        for sh in range(3):
            out[p0:p1, sh] = np.asarray(phis[sh])[r, c]

    _run(one, blocks)
    return out


def ground_pair_scalars(obs_c, obs_t, src_c, src_t, ground_z):
    """What the reflected term reads of every (observer, source) pair, (R, N)
    each: the height sum hs, the horizontal separation |d|, d.a, d.b and
    a.b (module docstring); then the observer's axial and radial
    coordinates in the IMAGE segment's frame, the scale and the near test."""
    dx = obs_c[:, None, 0] - src_c[None, :, 0]
    dy = obs_c[:, None, 1] - src_c[None, :, 1]
    hs = (obs_c[:, None, 2] - ground_z) + (src_c[None, :, 2] - ground_z)
    ax, ay, az = src_t[None, :, 0], src_t[None, :, 1], src_t[None, :, 2]
    bx, by = obs_t[:, None, 0], obs_t[:, None, 1]
    dn = np.hypot(dx, dy)
    da = dx * ax + dy * ay
    db = dx * bx + dy * by
    del dx, dy
    ab = bx * ax + by * ay
    # The image segment's tangent is (a, -t_nz) and the observer sits at
    # (d, hs) from its centre. rho by difference is inaccurate only near the
    # image axis within its span, a near pair.
    zi = da - hs * az
    rho = dn * dn
    rho += hs * hs
    rho -= zi * zi
    np.maximum(rho, 0.0, out=rho)
    np.sqrt(rho, out=rho)
    return hs, dn, da, db, ab, zi, rho


def _ground_band_keys(scalars, H, a_obs, type_id):
    """`_band_keys` for the reflected term: the quantised columns of one
    band (int64, (R*N,) each) and the near mask, from `ground_pair_scalars`.

    The step is 2^-KEY_BITS of the pair's distance to the image SEGMENT,
    rounded down to a power of two; a near pair (`NEAR_LENGTHS` against the
    image segment, the test the PEC image's plan applies) takes the
    sentinel key and its value from the direct evaluator."""
    hs, dn, da, db, ab, zi, rho = scalars
    az = np.abs(zi)
    near = (
        (np.maximum(az, rho) < NEAR_LENGTHS * _near_scale(H, a_obs)).reshape(-1)
        if NEAR_LENGTHS > 0
        else None
    )
    az -= H[None, :]
    scale = np.maximum(az, rho, out=az)
    np.maximum(scale, a_obs[:, None], out=scale)
    e = (scale.view(np.int64) >> 52) - 1022
    del scale, az
    step_inv = ((1023 + KEY_BITS - e) << 52).view(np.float64)
    lengths = [np.rint(x * step_inv).astype(np.int64) for x in (hs, dn, da, db)]
    del step_inv
    e += 1 << 11
    e <<= 40
    e += type_id
    cols = [c.reshape(-1) for c in (e, *lengths, _quantise_unit(ab))]
    if near is not None and near.any():
        idx = np.flatnonzero(near)
        cols[0][idx] = _NEAR_SENTINEL
        for c in cols[1:]:
            c[idx] = 0
    return cols, near


def build_ground_plan(
    obs_c,
    obs_t,
    src_c,
    src_t,
    ground_z,
    seg_h,
    a_obs,
    src_type,
    obs_type,
    rows,
    src_rev=None,
    obs_rev=None,
    budget=None,
):
    """The reflected term's class map (`GroundPlan`) over every (observer,
    source) pair, the REAL sources' (the image is the evaluator's), found in
    bands of `rows` rows like `build_plan`'s. `src_type` / `obs_type` are
    `ground_segment_types`'. Classes are numbered by their representative's
    flat index, the first pair of the class in the bands' order, so the
    representatives are row runs for `near_blocks`.

    With `src_rev` / `obs_rev` (`reversible_ground_types`) and `CANONICAL`,
    the pairs are keyed in their canonical orientation, as `build_plan`'s:
    d.a is odd under the source's reversal, d.b under the observer's, a.b
    under either, and the types carry t_nz, t_mz and the end codes. `budget`
    is `build_plan`'s."""
    obs_c = np.asarray(obs_c, dtype=np.float64)
    obs_t = np.asarray(obs_t, dtype=np.float64)
    src_c = np.asarray(src_c, dtype=np.float64)
    src_t = np.asarray(src_t, dtype=np.float64)
    a_obs = np.asarray(a_obs, dtype=np.float64)
    H = 0.5 * np.asarray(seg_h, dtype=np.float64)
    M, N = obs_c.shape[0], src_c.shape[0]
    coded = (
        src_rev is not None
        and obs_rev is not None
        and CANONICAL
        and M * N <= CODED_PAIRS
    )
    n_obs_types = int(obs_type.max()) + 1
    ds = do = None
    if coded:
        n_obs_types = max(n_obs_types, int(np.max(obs_rev)) + 1)
        ds = (np.asarray(src_rev, np.int64) - src_type) * n_obs_types
        do = np.asarray(obs_rev, np.int64) - obs_type
        ds = ds if ds.any() else None
        do = do if do.any() else None
    if budget is not None and 4 * M * N > budget:
        return OVER_BUDGET
    cls = np.empty((M, N), dtype=np.int32)
    rows = _band_rows(rows, M, N, budget, GROUND_BAND_BYTES_PER_PAIR)
    starts = list(range(0, M, rows))

    def band(i0):
        i1 = min(M, i0 + rows)
        scalars = ground_pair_scalars(
            obs_c[i0:i1], obs_t[i0:i1], src_c, src_t, ground_z
        )
        type_id = src_type[None, :] * n_obs_types + obs_type[i0:i1, None]
        cols, near = _ground_band_keys(scalars, H, a_obs[i0:i1], type_id)
        del scalars, type_id
        # (e, hs, |d|, d.a, d.b, a.b).
        code = None
        if coded:
            code = _canonical_codes(
                cols, 3, 4, 5, near, N, ds=ds, do=None if do is None else do[i0:i1]
            )
        near = (
            np.zeros(0, np.int64)
            if near is None
            else np.flatnonzero(near).astype(np.int64) + i0 * N
        )
        first, inv = _group(cols)
        if coded:
            cls[i0:i1] = _coded_band(inv, code).reshape(i1 - i0, N)
            rep_code = code[first]
        else:
            cls[i0:i1] = inv.reshape(i1 - i0, N)
            rep_code = np.zeros(first.size, np.int8)
        return [c[first] for c in cols], first + i0 * N, near, rep_code

    census = _Fold(6, cls.nbytes)

    def fold(i, result):
        keys, rep, near, rep_code = result
        i0 = starts[i]
        i1 = min(M, i0 + rows)
        ids = census.add(keys, (rep, rep_code), near)
        cls[i0:i1] = _renumber(cls[i0:i1], ids, coded)

    band_bytes = GROUND_BAND_BYTES_PER_PAIR * rows * N
    census_peak = _census_bands(band, starts, census, fold, budget, band_bytes)
    if census_peak is None:
        return OVER_BUDGET
    census.table = None
    _trim_census()
    rep_flat = census.joined(0)
    rep_code = census.joined(1)
    # Numbered by the representative's flat index. The bands run in row
    # order and the hash kernel numbers a class where it first occurs, so
    # there that is already the order and the map keeps its numbers.
    perm = np.argsort(rep_flat, kind="stable")
    if not np.array_equal(perm, np.arange(perm.size)):
        renumber = np.empty(perm.size, dtype=np.int64)
        renumber[perm] = np.arange(perm.size, dtype=np.int64)
        for i0 in starts:
            i1 = min(M, i0 + rows)
            cls[i0:i1] = _renumber(cls[i0:i1], renumber, coded)
        del renumber
    rep_flat = np.ascontiguousarray(rep_flat[perm], dtype=np.int64)
    rep_code = rep_code[perm]
    near_flat = census.near_flat()
    census.release()
    del perm
    plan = GroundPlan(
        cls=cls,
        rep_flat=rep_flat,
        rep_blocks=near_blocks(rep_flat, N, a_obs, rows=REP_ROWS),
        near_flat=near_flat,
        near_blocks=near_blocks(near_flat, N, a_obs),
        rep_code=rep_code if coded else None,
        code_bits=CODE_BITS if coded else 0,
        census_peak_bytes=census_peak,
    )
    return _within_budget(plan, budget)


def to_canonical(values, rep_code):
    """Class values, each its representative's as it stands, made the
    canonical orientation's in place: times CODE_SIGNS[rep_code] per shape,
    as exact negations. `rep_code` None (an uncoded plan) leaves them."""
    if rep_code is None:
        return values
    for sh in range(3):
        neg = CODE_SIGNS[:, sh][rep_code] < 0
        np.negative(values[:, sh], out=values[:, sh], where=neg)
    return values


def evaluate_classes(plan, direct):
    """The reflected classes' canonical values, (n_cls, 3): each
    representative pair evaluated where it stands (`evaluate_pairs`), then
    `to_canonical`."""
    vals = evaluate_pairs(plan.n_classes, plan.rep_blocks, direct)
    return to_canonical(vals, plan.rep_code)


def canonical_sources(canon):
    """Source centres and tangents placing each class in the canonical
    observer frame: the observer at the origin along +z.

    With the source tangent u = (s, 0, td), s = sqrt(1 - td^2), the radial
    vector is alpha p + beta q over p = (-td, 0, s) and q = (0, 1, 0), both
    normal to u: alpha s = rdt (its projection on the observer tangent) and
    alpha^2 + beta^2 = rho^2. The observer sits at rho_vec + z u from the
    source's centre."""
    rho, z, rdt, td = canon[:, 0], canon[:, 1], canon[:, 2], canon[:, 3]
    s = np.sqrt(np.maximum(0.0, 1.0 - td * td))
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(s > 0.0, rdt / s, 0.0)
    alpha = np.clip(alpha, -rho, rho)
    beta = np.sqrt(np.maximum(0.0, rho * rho - alpha * alpha))
    n = canon.shape[0]
    t = np.zeros((n, 3))
    t[:, 0] = s
    t[:, 2] = td
    rv = np.empty((n, 3))  # observer minus source
    rv[:, 0] = -alpha * td + z * s
    rv[:, 1] = beta
    rv[:, 2] = alpha * s + z * td
    return -rv, t


def evaluate(plan, call, obs_type=None):
    """Φ per class: (n_cls, 3) complex. `call(src_c, src_t, ns, g_obs)` runs
    the kernel for the canonical observer at observer segment `g_obs`'s
    radius against canonical sources standing for source segments `ns`,
    returning its (3, 1, len(ns)) tuple.

    The plan carries the canonical sources and the radius runs
    (`build_plan`), so each call is on slices of them and writes a slice:
    the Python around the kernel, which holds the GIL and so ran one worker
    at a time, is a few views per call (momwire#1916). `obs_type` is
    unused, the runs being the plan's. Calls are `EVAL_CHUNK` sources, small
    enough that a worker's kernel scratch stays in its core's cache."""
    del obs_type
    out = np.empty((plan.n_classes, 3), dtype=np.complex128)
    src_c, src_t, rep_src = plan.src_c, plan.src_t, plan.rep_src
    pieces = [
        (c0, min(c1, c0 + EVAL_CHUNK), g_obs)
        for r0, c1, g_obs in plan.runs
        for c0 in range(r0, c1, EVAL_CHUNK)
    ]

    def one(piece):
        c0, c1, g_obs = piece
        phis = call(src_c[c0:c1], src_t[c0:c1], rep_src[c0:c1], g_obs)
        for sh in range(3):
            out[c0:c1, sh] = phis[sh][0]

    _run(one, pieces)
    # Each representative was evaluated as it stands: the class's value is
    # the canonical orientation's (momwire#1969).
    return to_canonical(out, plan.rep_code)


def scatter(rep, plan, rows, near=None):
    """The band's three Φ, gathered from the class values (on the pool, by
    row pieces), each pair's code's signs applied (a coded plan), the near
    pairs then written from `near` (`evaluate_near`)."""
    i0, i1 = rows
    c = plan.cls[i0:i1]
    out = tuple(np.empty(c.shape, dtype=np.complex128) for _ in range(3))
    workers = _workers()
    step = max(1, -(-c.shape[0] // workers))
    pieces = [(s, r) for s in range(3) for r in range(0, c.shape[0], step)]
    bits = getattr(plan, "code_bits", 0)

    def take(piece):
        s, r = piece
        cr = c[r : r + step]
        dst = out[s][r : r + step]
        if not bits:
            np.take(rep[:, s], cr, out=dst, mode="clip")
            return
        np.take(rep[:, s], cr >> bits, out=dst, mode="clip")
        neg = (CODE_SIGNS[:, s] < 0)[cr & ((1 << bits) - 1)]
        np.negative(dst, out=dst, where=neg)

    _run(take, pieces)
    if near is not None and plan.n_near:
        local, vals = _band_near(plan, rows, near)
        for sh in range(3):
            out[sh].reshape(-1)[local] = vals[:, sh]
    return out


def _band_near(plan, rows, near):
    """The band's near pairs: band-local flat indices and their (k, 3) Φ."""
    i0, i1 = rows
    n_src = plan.cls.shape[1]
    p0, p1 = np.searchsorted(plan.near_flat, [i0 * n_src, i1 * n_src])
    return plan.near_flat[p0:p1] - i0 * n_src, near[p0:p1]


_HAVE_REDUCE = _acc is not None and bool(getattr(_acc, "pair_reuse_reduce_1916", False))
#: The fused kernel decodes a coded class map (momwire#1969).
_HAVE_REDUCE_CODED = _acc is not None and bool(
    getattr(_acc, "pair_reuse_reduce_1969", False)
)


def reduce_takes(Ms, out, coded=False):
    """Whether `reduce_band` can take this band: the fused kernel is built
    (one that decodes, for a coded plan), and the reduction is
    `reduce_phi_band`'s sparse regime (CSC complex128 Ms) into complex128
    rows."""
    import scipy.sparse

    if not _HAVE_REDUCE or out.ndim != 2 or out.dtype != np.complex128:
        return False
    if coded and not _HAVE_REDUCE_CODED:
        return False
    if out.shape[1] > 1 and out.strides[1] != 16:
        return False
    return all(
        isinstance(M, scipy.sparse.csc_matrix)
        and M.dtype == np.complex128
        and M.shape[1] == out.shape[1]
        for M in Ms
    )


def reduce_band(blocks, rows, Ms, out):
    """`out = Φ₀@M₀ + Φ₁@M₁ + Φ₂@M₂` for the band, Φ gathered by class
    inside the reduction and never materialised (`_accel_sin_pair_reuse.cpp`),
    the bits of `scatter` then `reduce_phi_band`. `blocks` is
    `[(plan, rep, near)]`: the direct block, then the PEC image's, which is
    subtracted. Call only where `reduce_takes`."""
    i0, i1 = rows
    args = []
    signs = []
    for plan, rep, near in blocks:
        local, vals = _band_near(plan, rows, near)
        args.append(
            (
                plan.cls[i0:i1],
                rep,
                np.ascontiguousarray(local, dtype=np.int64),
                np.ascontiguousarray(vals),
            )
        )
        # A coded plan's sign table (`CODE_SIGNS`), which the kernel decodes
        # by; None, its classes as stored.
        signs.append(
            np.ascontiguousarray(CODE_SIGNS, dtype=np.float64)
            if getattr(plan, "code_bits", 0)
            else None
        )
    m_args = []
    for M in Ms:
        m_args += [M.indptr, M.indices, M.data]
    # An older kernel takes neither (`reduce_takes` keeps coded plans off it).
    kw = {"signs": signs[0]} if _HAVE_REDUCE_CODED else {}
    image = None
    if len(args) > 1:
        image = (*args[1], signs[1]) if _HAVE_REDUCE_CODED else args[1]
    _acc.pair_reuse_reduce_band(out, *args[0], image, *m_args, **kw)
    return out


def verify_band(phis, rows, n_src, direct, samples=VERIFY_SAMPLES, tol=VERIFY_TOL):
    """Check `samples` pairs of the band against the direct kernel.

    `direct(m, ns)` returns the kernel's tuple for observer row `m` against
    source segments `ns`, in the original frame. Returns the largest
    relative difference seen; raises `PairReuseMismatch` above `tol`.

    Relative to the pair's largest |Φ| over the three shapes, floored at
    1e-6 of the sample's largest: a far pair whose Φ is itself a
    cancellation (perpendicular, say) carries the input rounding of any
    frame at a relative size no reuse could hold. Both sides are the same
    physical pair with the code's signs applied; what differs is how the
    kernel rounds two congruent spellings of it, which is why `tol` is set
    above the kernel's own reproducibility (`VERIFY_TOL`)."""
    i0, i1 = rows
    n_pairs = (i1 - i0) * n_src
    flat = np.unique(
        np.linspace(0, n_pairs - 1, min(samples, n_pairs)).astype(np.int64)
    )
    r, n = np.divmod(flat, n_src)
    diff, scale = [], []
    for m in np.unique(r):
        ns = n[r == m]
        d = direct(int(m) + i0, ns)
        want = [np.asarray(d[s])[0] for s in range(3)]
        diff.append(
            np.maximum.reduce([np.abs(phis[s][m, ns] - want[s]) for s in range(3)])
        )
        scale.append(np.maximum.reduce([np.abs(w) for w in want]))
    diff, scale = np.concatenate(diff), np.concatenate(scale)
    floor = 1e-6 * float(scale.max()) if scale.size else 0.0
    den = np.maximum(scale, floor)
    worst = float(np.max(diff / np.where(den > 0, den, 1.0))) if diff.size else 0.0
    if worst > tol:
        raise PairReuseMismatch(
            f"pair_reuse: a reused field tensor differs from the direct kernel "
            f"by {worst:.3g} relative (bound {tol:g})"
        )
    return worst
