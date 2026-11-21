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
quantised integers, so no two different keys can share a class.

THE PLAN IS C++ (momwire#2063): the keys, the census and the plan cache are
`_sin_pair_reuse_core.cpp`, the sinusoidal adapter to the common reuse core
(`_pair_reuse_core.h`), and the C++ fill (the nec42-sinusoidal executable)
runs the same plan. This module holds the plan's views (`Plan`,
`GroundPlan`), the evaluation through the shipped kernels, the gather and
the reduction, and `verify`.

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
the class table; no Φ-sized copy. The census holds each class once (THE
CENSUS, below), and both are held to `PLAN_BUDGET_BYTES` (momwire#2019).

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

import functools
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from ._accel import acc as _acc

#: Relative resolution of the key, in bits.
KEY_BITS = 48

#: Pairs whose observer lies within this many kernel scales (`near_scale`:
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

#: Sources per kernel call in `evaluate` (its per-row scratch is ~360 bytes
#: per source). At 8192 each worker's scratch spilled out of its core's
#: cache: on the ELT whip at 8 workers (Rocketlake) the kernel's summed
#: seconds were 3.7x one worker's. Measured per frequency there, wall:
#: 8192 0.086 s, 4096 0.059, 2048 0.048, 1024 0.080 (the Python per call
#: holds the GIL), 512 0.135 (momwire#1916). A speed knob, but not a
#: bit-free one: the kernel's batched sincos is a vector loop with a scalar
#: tail, and which a source falls in can move its Φ by an ulp.
EVAL_CHUNK = 2048

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

#: The extension's C++ plan (momwire#2063): without it the reuse declines.
HAVE_PLAN = _acc is not None and bool(getattr(_acc, "sin_pair_reuse_2063", False))


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

#: Bytes of plans the process keeps between solvers (`shared`), one fill's
#: budget (momwire#2026): a long-lived server holds no more than a single
#: fill may use. A plan is mostly its int32 class map, 4 bytes per pair: the
#: ELT whip's direct block is ~35 MB. The cache is the common core's
#: (`_pair_reuse_core.h`, SHARED_PLAN_BYTES), one for every basis.
SHARED_PLAN_BYTES = PLAN_BUDGET_BYTES

#: Bytes per class and per near pair of one wavenumber's values, the three
#: complex shapes (`evaluate`, `evaluate_near`).
VALUE_BYTES = 48

#: THE CENSUS (the C++ core's) holds each class once, so what it holds grows
#: with the deck's classes, not with each band's. Its budget counts what one
#: running table folding the bands in order would hold (`IntGroups`), as the
#: numpy census it replaced did (momwire#2063): a class seen in many bands is
#: held once. Holding every band's
#: classes until one merge at the end, as before, cost ~240 B per BAND class,
#: and on a deck whose rows repeat that is the deck's classes several times
#: over: the ELT whip (4,392 segments, 94 bands of 47 rows at 8 threads) has
#: 0.92 M classes in 5.0 M band classes, a census counted at 1.22 GB (peak
#: RSS over its entry, Rocketlake: 1.09 GB) for a plan of 201 MB, so its
#: reuse declined on the 256 MB budget. A class is numbered where it first
#: occurs in the pairs' flat order either way, so the plan is the same bytes.
#:
#: What the census holds is counted from the class map, the table's keys and
#: slots (`IntGroups.nbytes`), each class's representative and the near
#: pairs, plus `BAND_BYTES_PER_PAIR` for every band in flight or
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
#: at most this over a pool's worth (`band_rows` in C++), so the rest is the
#: classes'. Smaller bands move no class (the numbering is the flat order's).
BAND_SHARE = 0.25

#: The plan's fill holds at least this much per class beside its class map:
#: one wavenumber's values (`VALUE_BYTES`) and an int64 index (`rep_src`, or
#: the ground plan's `rep_flat`).
PLAN_FLOOR_BYTES_PER_CLASS = VALUE_BYTES + 8


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


def clear_shared_plans() -> None:
    """Drop every kept plan (tests, and a caller done with its decks)."""
    if HAVE_PLAN:
        _acc.sin_pair_reuse_configure(clear=True)


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
    # (`canonical_source` of the representative's rho, z, rdt, td), and the
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
    # What the census counted at its most (momwire#2016, #2019).
    census_peak_bytes: int | None = None
    # The C++ plan the arrays are views of (`_acc.SinReusePlan`).
    handle: object = None

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
    handle: object = None  # as `Plan`'s

    @property
    def n_near(self) -> int:
        return int(self.near_flat.size)

    @property
    def n_pairs(self) -> int:
        return int(self.cls.size)

    @property
    def n_classes(self) -> int:
        return int(self.rep_flat.size)


def _limit_openmp():
    """A pool worker's kernels run on one OpenMP thread: the pool is the
    parallelism. libgomp keeps the thread count per calling thread, so this
    leaves every other thread's as it was."""
    from threadpoolctl import ThreadpoolController

    ThreadpoolController().select(user_api="openmp").limit(limits=1)


@functools.lru_cache(maxsize=1)
def _pool():
    """The evaluation's and the gather's pool, whose size is also the
    census's worker count (its bands' budget): as many workers as this
    process may run on, held to OMP_NUM_THREADS where the caller pinned
    one."""
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


def reversible_segment_types(seg_h, a_obs, src_a=None, ind1=None, ind2=None):
    """Per-segment type ids: (src_type, obs_type, src_rev).

    The source type holds what Φ reads of the source besides its pose: its
    half-length (quantised at KEY_BITS relative bits) and, under the extended
    kernel, its radius and end codes (exact); `src_rev` is each source's type
    with its tangent reversed, in one id space (the end codes swap; without
    them it is src_type). The observer type is its radius (exact), the scalar
    each kernel call takes. The C++ census's own (`_sin_pair_reuse_core.h`)."""
    if src_a is None:
        return _acc.sin_pair_reuse_types(seg_h, a_obs)
    return _acc.sin_pair_reuse_types(seg_h, a_obs, src_a, ind1, ind2)


def reversible_ground_types(src_type, obs_type, src_rev, seg_t):
    """The types refined by each segment's vertical tangent component, which
    the reflected term reads and a horizontal isometry keeps
    (`build_ground_plan`), for both orientations of every segment, in one id
    space: (g_src, g_obs, g_src_rev, g_obs_rev). Reversing a tangent negates
    its vertical component, and a source's end codes swap. The C++ census's
    own."""
    i64 = functools.partial(np.ascontiguousarray, dtype=np.int64)
    return _acc.sin_pair_reuse_ground_types(
        i64(src_type),
        i64(obs_type),
        i64(src_rev),
        np.ascontiguousarray(seg_t, dtype=np.float64),
    )


def _as_plan(handle, ground):
    """`Plan` / `GroundPlan` over a C++ plan's arrays."""
    if handle is None:
        return OVER_BUDGET
    common = dict(
        cls=handle.cls,
        near_flat=handle.near_flat,
        near_blocks=handle.near_blocks,
        rep_code=handle.rep_code,
        code_bits=handle.code_bits,
        census_peak_bytes=handle.census_peak_bytes,
        handle=handle,
    )
    if ground:
        return GroundPlan(rep_flat=handle.rep, rep_blocks=handle.rep_blocks, **common)
    return Plan(
        rep_src=handle.rep,
        src_c=handle.src_c,
        src_t=handle.src_t,
        runs=handle.runs,
        **common,
    )


def _census(
    ground,
    obs_c,
    obs_t,
    src_c,
    src_t,
    seg_h,
    a_obs,
    src_type,
    obs_type,
    rows,
    src_rev,
    obs_rev,
    ground_z,
    budget,
    shared,
):
    f64 = functools.partial(np.ascontiguousarray, dtype=np.float64)
    i64 = functools.partial(np.ascontiguousarray, dtype=np.int64)
    if not CANONICAL:
        src_rev = obs_rev = None
    handle, built = _acc.sin_pair_reuse_plan(
        f64(obs_c), f64(obs_t), f64(src_c), f64(src_t), f64(seg_h), f64(a_obs),
        i64(src_type), i64(obs_type), int(rows),
        src_rev=None if src_rev is None else i64(src_rev),
        budget=budget,
        workers=_workers(),
        ground_z=None if not ground else float(ground_z),
        obs_rev=None if obs_rev is None else i64(obs_rev),
        band_bytes=GROUND_BAND_BYTES_PER_PAIR if ground else BAND_BYTES_PER_PAIR,
        shared=shared,
    )  # fmt: skip
    return _as_plan(handle, ground), built


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
    """The class map over every (observer, source) pair (`Plan`), the C++
    census's, uncached, found in observer bands whose transients together
    hold `rows` rows.

    With `src_rev` (each source's type reversed, `reversible_segment_types`)
    and `CANONICAL`, the pairs are keyed in their canonical orientation and
    the map is coded (momwire#1969, module docstring).

    With `budget` (bytes), returns `OVER_BUDGET` where the census or the
    plan's fill would pass it (momwire#2019); the map is counted before it is
    allocated."""
    return _census(
        False, obs_c, obs_t, src_c, src_t, seg_h, a_obs, src_type, obs_type, rows,
        src_rev, None, None, budget, False,
    )[0]  # fmt: skip


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
    source) pair, the REAL sources' (the image is the evaluator's), the C++
    census's, uncached. `src_type` / `obs_type` are
    `reversible_ground_types`'. Classes are numbered by their
    representative's flat index, the first pair of the class, so the
    representatives are row runs for the evaluation's calls. With `src_rev`
    / `obs_rev` and `CANONICAL`, the pairs are keyed in their canonical
    orientation; `budget` is `build_plan`'s."""
    return _census(
        True, obs_c, obs_t, src_c, src_t, seg_h, a_obs, src_type, obs_type, rows,
        src_rev, obs_rev, ground_z, budget, False,
    )[0]  # fmt: skip


def shared(
    block,
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
    obs_rev=None,
    ground_z=None,
    budget=None,
):
    """`build_plan` (`block` "direct" or "image") or `build_ground_plan`
    ("reflected") through the process's plan cache: (plan or OVER_BUDGET,
    whether a census ran).

    A plan holds no k, and its key is every array the census reads, so it
    outlives the solver that found it: the portal builds one solver per
    frequency and a design server one per knob turn back to a value, and
    each would pay the census again with a per-instance cache (momwire#1916
    part C). Plans are never written once built, so solvers share one. Least
    recently used first out past `SHARED_PLAN_BYTES`, the newest kept
    whatever its size; a declined census is kept too, so that a sweep does
    not repeat it per frequency."""
    return _census(
        block == "reflected", obs_c, obs_t, src_c, src_t, seg_h, a_obs, src_type,
        obs_type, rows, src_rev, obs_rev, ground_z, budget, True,
    )  # fmt: skip


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
