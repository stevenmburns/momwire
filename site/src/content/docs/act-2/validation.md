---
title: "7 · How do you know it's right?"
description: A number from a solver is a claim, not a fact. Two independent cross-examinations — does it stop moving, and does an unrelated engine agree — are what turn 69.6 − 18.3j from output into evidence.
---

Every chapter so far has quoted an impedance to a tenth of an ohm as if it were
handed down. It wasn't — it's the output of code *we* wrote, and there is no
closed-form dipole impedance to check it against. So how does anyone trust a
method-of-moments number? Two cross-examinations, and a solver has to pass both.

## Internal: does it stop moving?

The first is cheap and necessary: a converged answer **stops changing** as you
add unknowns. Refine the mesh; if the impedance keeps wandering, you haven't
found the antenna's number, you've found *a* number for *this* mesh.

![Two panels, R and X versus segment count N on a log axis. The sinusoidal and B-spline curves sit flat on the NEC-2 reference line from about five segments on — a sharp knee. The Act I pulse curve crawls slowly toward the line across hundreds of segments and never quite arrives, especially in reactance.](../../../assets/figures/ch7-knee.svg)

momwire's good bases hit **the knee** by five to seven segments and then sit
still — that flat line is what convergence looks like. The [Act I](/act-1/the-question/) pulse toy, on
the same axes, is still crawling at three hundred. That contrast is the whole of
Acts I–II in one picture: continuity buys you a knee.

But convergence is necessary, not sufficient. A solver can march confidently to
a stable, wrong answer — a consistent bug, a subtly mis-derived kernel, a basis
too crude to resolve the near field (we watched exactly that in [chapter 3](/act-1/the-feed/), where
the pulse's reactance converged about an ohm shy of a continuous basis). A
method agreeing *with itself* proves only that it's self-consistent.
For the truth you need a witness who shares none of your code.

## External: does an unrelated engine agree?

That witness is NEC-2 — the reference wire solver of the last forty years, via
PyNEC. It's an independent implementation with its own kernel, its own testing,
its own everything. When momwire and NEC agree on an antenna, the agreement
can't be a shared bug, because they share no code. Here is momwire's sinusoidal
*and* B-spline solver against NEC-2 on two antennas — the specimen dipole and a
two-element Yagi ([`docs/convergence_analysis.md`](https://github.com/stevenmburns/momwire/blob/v0.9.0/docs/convergence_analysis.md)):

![A horizontal bar chart of momwire-minus-NEC residuals for four quantities — dipole R, dipole X, Yagi R, Yagi X — with a bar each for the sinusoidal and B-spline solvers. Every bar sits within about a quarter of an ohm of zero, most within a tenth; the B-spline bars are all under 0.05 ohm. Dotted bands mark ±0.2 ohm.](../../../assets/figures/ch7-validation.svg)

Two engines, two antennas, two of momwire's own bases — everything lands inside
a few tenths of an ohm, and the B-spline solver inside a twentieth. The Yagi is
the sterner test: a *parasitic* element with no feed of its own, its current
induced entirely by the driver, coupling through the near field. Getting
`77.25 + 6.71j` against NEC's `77.28 + 6.74j` means the mutual coupling, not
just the self-impedance, is right.

## The honest part

The bars aren't all *zero*, and that's the honest part worth dwelling on. Two
different discretizations of the same continuous problem — a sinusoidal basis
with collocation, a B-spline basis with Galerkin — can converge to values a
fraction of an ohm apart, because they are different finite approximations, not
the exact integral. Neither is "wrong." The sinusoidal solver's slightly larger
reactance residual is real, and it shrinks as `N` grows past 41.

Which is exactly why the cross-check matters, and why momwire carries *both*
bases plus the NEC comparison rather than trusting one. The sinusoidal solver
sharing NEC's basis makes it a *tight* ruler — a tenth of an ohm — but a ruler
that agrees with a copy of itself is only so convincing. The real evidence is
the **B-spline** solver: a completely different basis, a completely different
testing scheme, landing on the same answer. When two methods that share nothing
but the physics agree, the thing they agree on is the physics.

## Act II, closed

You now have the craft, and a reason to trust it:

- continuous **bases** — sinusoidal (ch. 4) and B-spline (ch. 5) — converge in a
  handful of segments where the pulse crawled for hundreds;
- honest **quadrature** (ch. 6) fills the matrix accurately, splitting the
  smooth pairs from the singular ones;
- and **convergence plus cross-validation** (ch. 7) turn the output into
  evidence.

That is a trustworthy free-space wire solver. [Act III](/act-3/mirror-worlds/) raises the stakes to where
real antennas live: above the ground — first as a perfect mirror, then as real,
lossy dirt, and finally paying Sommerfeld's full price for getting it exactly
right.
