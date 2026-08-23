# The NEC-2-class coupled-loop pathology: mechanism, and why a testing-rule fix cannot save the basis

*Study of 2026-08-23. Scripts: `scripts/loop_pathology_dissect.py`,
`scripts/loop_pathology_residual_emf.py`,
`scripts/loop_pathology_pulse_ladder.py`,
`scripts/loop_pathology_half_razor.py` — every number below reproduces
from them. The driving example is W7EL's coupled-loop demonstration model
(QRZ thread 1000972: a 300 m vertical standing on the rim of a closed
80 × 80 m loop, driven at 500 Hz), on which NEC-2-class engines put
~160 A of spurious circulating current in the loop against a ~1 A source,
while NEC-5, momwire's B-spline, and momwire's pulse rows agree on
~0.48 A of legitimate charging current.*

## The mechanism, measured

Burke's account of the defect (quoted publicly by W7EL in the QRZ
thread): the line integral of ∇φ around a closed loop must vanish;
errors in evaluating it act as a small spurious voltage source, growing
as 1/f. This study takes that from an account to a measurement, and pins
*which* property of the discretization turns the error on.

**The functional.** Apply the assembled matrix to a fixed, smooth
current distribution and sum the loop's tested equations — the discrete
circulation of the tested field around the loop. For *any* current, at
any frequency, the scalar-potential part of the continuum integral is
zero; whatever survives is the discretization's own residual EMF.

**Result** (`loop_pathology_residual_emf.py`): the point-matched
sinusoidal scheme carries **2.1 × 10⁵ V per ampere of loop current at
500 Hz, scaling as f^(−1.007)** over two decades. The same functional on
the pulse rows reads ~10⁻⁷ V/A — machine zero.

**Where it comes from** (`loop_pathology_dissect.py`):

| experiment | residual (V per A of loop current) |
| --- | --- |
| isolated square loop, uniform current | ~0.5 |
| same, octagon / 16-gon (corner angle) | same order — corners are not the seat |
| same square + a 3-wire junction stub carrying **zero** current | **2.6 × 10⁴** — 50,000× |
| the stub *present but not junction-connected* | ~0.5 again |
| wire radius swept 100× (0.0005 → 0.05 m) | **identical** — not the kernel |

The seat is the **multi-wire junction**: its basis conditions (the
P-sums of the three-term expansion) force large sinusoidal-shape content
near the junction, making φ vary at sub-segment scale. Point matching
*samples* E at segment midpoints, so any loop sum of tested equations is
a midpoint-rule **quadrature** of ∮E·dl — and a quadrature's error is
O(local φ) where φ varies inside a segment. The φ scale rides 1/ω, hence
Burke's 1/f. A channel-split of the fill (const/sin/cos masked
assemblies) shows the residual surviving an eleven-orders-of-magnitude
analytic cancellation between the const and cos channels — it is
analytic, not floating-point (the folded well-scaled shapes of
`docs/sinusoidal_basis_design.md` do not remove it).

The clean schemes are clean **by structure, not by accuracy**: the pulse
rows difference φ across each segment's two ends (the four-point charge
stencil), razor/NEC-5-class testing integrates E·dl over intervals, and
the B-spline family's charge is the exact in-basis derivative — in each
case the loop sum telescopes, cancelling the numerical error along with
the value. `loop_pathology_pulse_ladder.py` shows both pulse rows
holding a frequency-flat loop/source ratio from 50 kHz to 5 Hz on Roy's
model — including the plain pulse row, whose *impedance* is badly wrong
there (its known reactance defect) while its loop physics stays clean.
Point matching itself is exonerated: both pulse rows are point-matched.

One more corollary worth recording: the erratic segmentation behavior of
NEC-2-class engines on loop structures ("instability") is the ratio of
two error-dominated quantities — the residual EMF injected by the real
current over the loop's discrete self-response — so it lurches with mesh
instead of converging (measured across a uniform k = 1..16 refinement:
245 / 0.4 / 0.6 / 32 / 2.2 A). And the *source* current of the stock
scheme in the deep-LF limit is not trustworthy either: the spurious loop
current feeds back through the junction (0.86 A at 50 Hz where the clean
lanes' I ∝ f law gives 0.109 A).

## The smallest-change candidate, and its honest failure

If the disease is quadrature of ∮∇φ, the minimal cure is to stop
quadraturing it: keep the NEC-2 basis and junction conditions, keep
midpoint collocation for the smooth −jωA part, and test the scalar
potential as an exact end-difference φ(right node) − φ(left node), with
φ evaluated once per unique node so every loop sum telescopes by
construction (`loop_pathology_half_razor.py` — "half-razor"). Two
ingredients are both required: the difference testing, and the folded
well-scaled channels {1, sin, cos − 1} with the exactly-summed `AC`
coefficient — with literal channels the giant A/C cancellation destroys
the operator below ~1 kHz.

It works — against the disease it targets. On Roy's model the
loop/source ratio comes out **flat at ~0.44 from 50 kHz to 5 Hz** (the
clean-class value; the spurious EMF is structurally dead), and at
moderate electrical size the scheme agrees with the stock solver and
bspline to a few percent.

And then it fails somewhere new. Below L/λ ≈ 10⁻³ the input capacitance
collapses ~200× (859 → 4.3 pF on Roy's model). Fifty-digit re-solves of
the same matrix and row/column equilibration change nothing — the defect
is in the operator. The diagnostic at 50 Hz shows what happened: the
solved **node** potentials form perfect equipotential plateaus with
exactly the drive-voltage jump at the gap, while the **midpoint**
potentials sit near zero. The three-term basis has intra-segment
quadratic charge freedom, and a node-only test cannot see charge
patterns whose potential cancels at the nodes — a checkerboard mode, in
the classic staggered-grid sense, which supplies the plateaus with ~200×
too little net charge.

## The conclusion

The two failures are one statement about the basis:

> With the NEC-2 three-term expansion, φ must be controlled both at the
> nodes and inside the segments, and N equations cannot do both. Point
> matching controls the midpoints and leaks a spurious loop EMF at the
> junctions; node differencing controls the nodes and admits the
> checkerboard. The basis is the disease.

Which is presumably why the historical fix was **modified basis
functions** rather than better integration. The tent/interval route
(RazorSolver; the NEC-5-class formulation it twins) has no intra-segment
charge freedom for a checkerboard to live in and tests E·dl over
intervals so nothing quadratures ∮∇φ; the B-spline route carries charge
as the exact derivative of its current under Galerkin testing. momwire
ships both, and they are the recommendation for loop-sensitive
structures — the point-matched sinusoidal solver's role remains
NEC-2-formulation fidelity, pathology included, as a cross-engine
reference.

*Provenance note: every claim about the reference engines here is from
our own runs of public models; the mechanism account credited to Burke
is quoted from W7EL's public forum post.*
