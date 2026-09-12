# momwire#956 re-adjudication of the #524 phase-2 ladder, on `956-wterms-transverse`

Haswell (i7-4770K), 2026-09-12. Scratch only; no `src/momwire` change in this
directory. The fix under test is `956-wterms-transverse` (abfe861): the ẑẑ
kernel `k²V − ∂zW` → `k²V + ∂z′W`, plus the missing test-end W term (TW).

## The experiment is branch-vs-main, and here is why

The ask was "re-run the ladder exactly as session 6 ran it, and report against
§6/§7". **The session-6 absolutes do not reproduce on this box on ANY commit**,
so that comparison cannot be made honestly. The A/B reported here is therefore
**branch against main on this machine**, which is immune to whatever drifted and
isolates exactly what the fix does. §6 absolutes are quoted beside each row and
marked *not reproducible here* rather than silently compared against.

### probe29 — the ε̃ = 1 adjudicator: identical on three commits, none of them §6

| momwire commit | complete+merged (ε̃ = 1) | vs truth |
|---|---|---|
| `956-wterms-transverse` (abfe861) | 11.9321−1020.8398j | 34.585 % |
| `origin/main` (0e72ab4) | 11.9321−1020.8398j | 34.585 % |
| `12584f2` (pre-0.53.0, pre-#1040) | 11.9321−1020.8398j | 34.585 % |
| §6 ledger, 2026-08-26 | 17.5619−758.1617j | 0.002 % — **not reproducible here** |

Bit-identical on all three, with the corner reading −29.9792−204344.7540j, which
matches §6's recorded "magnitude 204,345" exactly — so the deck and the setup ARE
session 6's. Two consequences: **the prediction "probe 29 cannot move" is
CONFIRMED**, and **the fix is exonerated on it, as is momwire#1040**.

**The retired-name hypothesis was tested and FALSIFIED.** A setattr on a retired
name is a silent no-op, so: snapshot every momwire module namespace before the
run, diff after, and log every `BSplineSolver.__setattr__` against the class's
own attributes. Result — **zero** new module attributes on `_crossing_fill`,
`bspline`, `_sommerfeld_transmitted` or `_sommerfeld_below`, and every instance
attribute set is an ordinary constructor parameter. Nothing was lost to a dead
name.

**What the non-reproduction is, as far as this unit can say.** momwire's own
ε̃ = 1 certification passes on main and on the branch, and probe29's failing
quantity is the PROBE's `complete+merged` composition rather than
`s.compute_impedance()`. probe31 does not reproduce §6 either (node current
0.0199 against §6's 1.19 per unit feed; slope ratio 1.223 at g1 against 0.0450).
So this is **suite-wide drift of the banked probes' own node-cell bookkeeping
against momwire's current API, with the serve's certification intact** — not a
physics regression, and not a dead knob. Left OPEN; no bisect was run.

## The headline: NEC-5 on a second machine, and the residual is closed

`buried_radial_vertical`, connected spelling, 4 radials, hub 0.15 m, soil A,
7.1 MHz. Both engines ladder through the SAME builder knob, so a rung is one
geometry meshed two ways; segment and GW counts are printed per rung because an
arm that silently does nothing looks exactly like an arm that does.

| nominal_nsegs | segs | GW | momwire | NEC-5 | dR | dX |
|---|---|---|---|---|---|---|
| 42 | 475 | 14 | 78.1425+46.3863j | 77.9370+45.2030j | −0.2055 | −1.1833 |
| 84 | 936 | 14 | 78.1495+46.4210j | 78.0060+45.5780j | −0.1435 | −0.8430 |
| 168 | 1850 | 14 | 78.1542+46.4448j | 78.0430+45.7760j | −0.1112 | −0.6688 |

**Richardson p=1: momwire 78.1590+46.4692j, NEC-5 78.0809+45.9788j.**
(p=2: momwire 78.1558+46.4530j, NEC-5 78.0557+45.8441j.)

Laptop-control's reading on a different machine: momwire 78.154+46.445j, NEC-5
78.080+45.974j. **Reproduced** — 0.005 Ω in R and 0.024 Ω in X for momwire,
0.0009 Ω and 0.005 Ω for NEC-5.

**ΔR = −0.078 Ω, ΔX = −0.49 Ω, against the pre-fix residual of +2.1+4.9j Ω** —
27× in R and 10× in X. The per-rung Δ shrinks monotonically on both axes, so the
extrapolation is not carrying the result.

## probe31 — junction currents: the registered prediction MISSED

AGARD |1/ε̃| = 0.054730. Magnitudes, which is §6's own like-for-like (the
quantity is complex and the phases differ; §6 compares "slope-ratio magnitude").

| rung | main | branch | delta | \|main−AGARD\| | \|branch−AGARD\| | verdict |
|---|---|---|---|---|---|---|
| g1 | 1.223085 | 1.231151 | +0.008066 | 1.168355 | 1.176422 | branch **FARTHER** |
| g2 | 0.063597 | 0.064236 | +0.000639 | 0.008867 | 0.009506 | branch **FARTHER** |

| rung | KCL deficit, main | KCL deficit, branch |
|---|---|---|
| g1 | 1.068e-05 | 1.078e-05 |
| g2 | 3.829e-05 | 3.830e-05 |

§6 (**not reproducible here**): slope ratio 0.0450 → 0.0532, KCL ~2e-7, node
current 1.19 per unit feed.

**Registered prediction: "probe 31's slope ratio should land closer to 0.0547
than 0.0532." It did not.** The branch moves the ratio AWAY from AGARD on both
rungs. The movement is small — +1.0 % on g2, on a quantity sitting ~16 % above
AGARD on BOTH commits — and KCL is unmoved to three digits, so continuity is
neither gained nor lost. The honest statement, and it is not reconciled here:
**the fix is neutral-to-slightly-adverse on probe31's slope ratio and decisive
on the driving-point residual.** Both facts stand in the record.

## Traps found in this unit

- **The probes WRITE INTO THE BANK.** `probe31_currents.py` overwrites
  `scratch/524-phase2/results/probe31-currents.json`, the session-6 record. I
  clobbered it once and restored it from git; every run in this unit now
  restores the bank afterwards and keeps its own output elsewhere. The banked
  results are not read-only and a re-run silently replaces the thing you are
  comparing against.
- **A code reading lost to a five-line experiment.** I read
  `probe25_node_cell.py:163` carrying its own `s_zz = (FA_w*tzA) @ (k2sq*V −
  dzW) @ (FB_w*tzB).T`, concluded the phase-2 probes reimplement the spelling
  and could not see the branch at all, and was about to report the whole unit as
  unrunnable. A tripwire on every `_crossing_fill` entry point proved otherwise:
  probe29 calls `cross_complete_block_split`, `_main_split`, `_ends_and_corner`,
  `self_completions` and `axis_data`. The probes DO exercise the branch.

## The derived bytes had to be regenerated first

Probes 33 and 34 both FAILED on the first attempt, and neither failure was
physics:

- probe33 — `ModuleNotFoundError: bench_nec5_walk_why`. It exists, at
  `antennaknobs:scripts/bench_nec5_walk_why.py`; it is simply not on the probe's
  `sys.path`.
- probe34 — `FileNotFoundError: results/probe23-blocks-g1.npz`. probe23 WRITES
  that file; the bank does not carry it.

Both are the banking commit's own warning made real — *"bank the four cited
studies, **minus the derived bytes**"*. **This is also the most likely
explanation of the suite-wide §6 non-reproduction**: the phase-2 bank was
committed without the artifacts its own probes need.

probe23's blocks are computed FROM momwire, so they were regenerated **per
commit**. Reusing branch-built blocks under main would have compared a commit
against itself and produced a clean "no change" that meant nothing.

## probe34 — the omission zoo: the registered prediction HITS

| | predicted by Laptop-control | measured here |
|---|---|---|
| branch, g1 deck | 169.7754−82.2803j | **169.7756−82.2800j** |
| main, g1 deck ("was") | 138.9619−102.6019j | **138.9609−102.6097j** |

Agreement to 2e-4 on the branch and 8e-3 on main — **reproduced on a second
machine, both sides of the change.**

| rung | main | branch | delta |
|---|---|---|---|
| g1 Z | 138.9609−102.6097j | 169.7756−82.2800j | +30.81+20.33j |
| g1 Δ vs mono | +67.3589−53.5703j | +98.1736−33.2406j | |
| g2 Z | 138.9625−102.6062j | 169.7772−82.2763j | +30.81+20.33j |

**main's Δ = +67.3589−53.5703j reproduces §6's soil-A +67.2−53.7j.** So §6 is
partly reachable after all — what fails to reproduce is the σ ladder and the
current-level observables, not this deck.

**A finding about the probe, not the fix: the zoo's knobs are inert here.** All
five spellings — M-only, M+SW, B(M+SW+SQ), A(M+SW+SQ+BT), A+corner-no-selfcomp —
return the SAME Z to every digit, on BOTH commits. §6 reports them as distinct
(M-only −2.40−6.64j; M+SW/B ≈ 0; A-class −1000j). A knob that does not move the
answer cannot adjudicate anything, so **probe34's §7 conclusion ("no omission
spelling reproduces engine Δ") is not re-derivable from this run** — not because
it is wrong, but because the instrument is no longer discriminating.

## probe33 — the high-σ ladder: §6's collapse does NOT reproduce, on either commit

| σ | main | branch | delta |
|---|---|---|---|
| 0.05 | +71.2591−909.4517j | +71.6895−909.7393j | +0.43−0.29j |
| 0.5 | +427.9195−381.6736j | +431.2184−381.7714j | +3.30−0.10j |
| 5 | +72.6271−10.7233j | +73.9140−8.6844j | +1.29+2.04j |

§6 (**not reproducible here**): +67.2−53.7j → +31.7−2.1j → +6.9−3.6j →
+1.1−2.6j, i.e. Δ → 0 as σ → ∞.

**Neither commit collapses.** Δ goes 71 → 431 → 74, not → 0. Since main behaves
the same way, this is the suite drift and not the fix. But it means **§7's
adjudicator 2 — "both conventions converge exactly where the contact fiction
becomes exact" — is not re-derivable from this run either.**

Registered prediction: *"probe 33's Δ at soil A moves from +67.2−53.7j and must
still collapse to ~0 at σ = 5."* **Split verdict**: the soil-A Δ does move
(+67.36−53.57j → +98.17−33.24j, via probe34's g1 deck) — **HIT**; the σ = 5
collapse does not happen on either commit — **MISSED**, and missed on main too,
so it is not attributable to the fix.

## Scoreboard against the registered predictions

| prediction | verdict |
|---|---|
| probe 29 cannot move (W ≡ 0 at ε̃ = 1) | **HIT** — bit-identical on three commits |
| probe 31 slope ratio closer to 0.0547 than 0.0532 | **MISSED** — branch moves AWAY on both rungs |
| probe 33 soil-A Δ moves from +67.2−53.7j | **HIT** |
| probe 33 Δ still collapses to ~0 at σ = 5 | **MISSED** — no collapse on either commit |
| g1 deck reads 169.7754−82.2803j (was 138.9619−102.6019j) | **HIT to 2e-4, both sides** |
| NEC-5: momwire 78.154+46.445j vs NEC-5 78.080+45.974j | **HIT** — 78.1590+46.4692j vs 78.0809+45.9788j |

Four hits, two misses, both misses reported as registered and neither
attributable to the fix in isolation.
