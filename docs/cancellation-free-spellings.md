# Cancellation-free spellings

*momwire#799. The sibling table itself lives in `src/momwire/_stable.py`'s
module docstring, with `src/momwire/_stable_inline.h` as its C++ twin; this
note says why it exists and what it is worth, so the next kernel starts from
the stable form instead of rediscovering the problem from a widened bar.*

Every kernel here integrates a **remainder** — the Green's function with its
static part taken analytically — and the subtraction that forms one is a
difference of near-equal numbers wherever the phase is small. Written
literally, `exp(-jkR) - 1` returns its real part to an *absolute* epsilon
rather than a relative one: the answer is `-(kR)²/2`, so the relative error is
`ε/((kR)²/2)` and it grows as the mesh refines. Measured against a 60-digit
reference:

| kR | `exp(-jkR) - 1`, real part | `expm1_neg_jkR` |
|---|---|---|
| 1e-4 | 5.2e-9 | 1.7e-16 |
| 1.26e-3 | 6.7e-11 | 1.6e-16 |
| 8e-3 | 1.1e-12 | 2.4e-16 |
| 3e-2 | 6.0e-14 | 2.1e-16 |

The same shape recurs in every difference these kernels form, and the fix is
never a tolerance. It is always the same move: **spell the difference so that
no term in it is larger than the answer.** `expm1`, `log1p`, `-2 sin²(y/2)`,
`tan(kΔ/2)`, a rationalised `(a² - b²)/(a + b)`, or a series where no closed
form exists. Consult the table in `_stable.py` before writing a new one.

Three consequences worth keeping in view.

**The loss is in the mesh, not in the physics.** Every entry above gets worse
as the segments get shorter, so a spelling that is invisible on a 24-segment
dipole is 1e-9 on an 801-segment one. `_static_axis_moments`' m1 is the sharp
case: its `r1 - r0` term is up to 1600× larger than m1 itself at 801 segments,
which made the literal form 1.9e-9 relative and moved the solved Z by 2.6e-11.

**Both lanes, always, in one change.** A kernel with a numpy twin and a C++
twin shares its spelling deliberately — the cross-lane tests bound the
difference between them, and rewriting one lane alone turns an arithmetic
improvement into a test failure. Rewriting both moves real-k output at the
1e-12..1e-13 level, so it is a re-pin under the momwire#762 protocol and never
a rider on a PR with another purpose.

**A widened cross-lane bar is a symptom, not a fix — and it will also tell
you when your diagnosis is wrong.** momwire#798 widened the razor complex-k
bars to 1e-10 because macOS read 8.4e-13 where Linux read 2.5e-16, and
attributed it to one ulp of libm disagreement amplified by the subtraction of
1. Removing that subtraction did not move the macOS number at all. The
deviation turned out to be *k-independent* — it reads 8.131e-13 at k = 1e-30 —
and it was `_axis_frame`'s `perp = |d|² − u_r²`, an exact cancellation on any
collinear pair that an FMA contraction resolves differently in two lanes,
worth 1e-7 relative in the `rho2` every static moment takes a logarithm of.

Two lessons, and the second is the expensive one. A bar set from the
platform that fails is a bar that can never contradict your model of why it
failed; set it from the platform that is clean and let the other one argue.
And when a cancellation is *exactly* zero in exact arithmetic — not merely
small — it is invisible in every check that compares one implementation
against itself, and shows up only where two implementations round it
differently.
