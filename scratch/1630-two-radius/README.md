# momwire#1630: the two-radius crossing node on the sinusoidal pair

Skylake, 2026-10-22. References are black boxes, deck in, impedance out:
NEC-4.2 (`nec42cl-static`, no EK card, so the engine's default kernel) and
NEC-5 (`nec5cl`, the x13 build; NEC-5 is permanently EK-on). momwire was run
through its NEC-4.2-dialect front door (`parse(..., "nec4")` +
`build_solver`), which leaves `extended_kernel` False on both sinusoidal
families (the deck has no EK card) and gives bs2 the roster's EK default
(`ek` True, `exact_kernel` auto resolving False). Each JSON row records the
solver's kernel and the warnings it raised.

## Decks (`decks.py`)

A 5 m vertical (19 mm), 14 MHz, over 16 buried 4 m radials (1 mm) at
-0.0254 m, GN 2 13 / 0.005 NOFILE, GE -1. Two spellings:

- **hub**: a drop from the base down to a buried hub, with the radials leaving
  the hub. The crossing node joins two members.
- **slope**: the radials meet the base at z = 0, each on a 45-degree leg down
  to the radial depth. The crossing node joins 17 members.

Refinement r = 1, 2, 4, 8 multiplies every count (vertical 8r, radial 8r,
drop and legs r). The source is EX 0 on the vertical:

- `mid` is segment 2r+1, whose lower end is 1.25 m up at every r;
- `bottom` is segment 1, next to the interface junction.

The NEC-5 decks put the source on the knot 1.25 m up, which is the limit of
`mid`. The feeds therefore sit h/2 apart, and that difference refines away.

The radius configurations (`ref_ladder.RADII`) are:

- `main`: 19 / 1 mm;
- `one`: 1 / 1 mm;
- `above_half`: 9.5 / 1 mm;
- `below_half`: 19 / 0.5 mm.

`n_rad` = 16, 4 and 1.

## Findings

1. **NEC-4.2 is not a reference here.** Three faults stack, each isolated by
   a probe (`nec42_form_probe*.py`, `nec42_step_probe.py`):
   - **The 16-radial hub junction fails even with ONE radius and even in FREE
     SPACE.** NEC-4.2 reads 23.0, 23.2, 14.4, 27.7 ohm over r = 1 to 8, while
     NEC-5 settles at 21.0 (`form2.jsonl`). One radial converges (191 vs
     NEC-5's 185); four drift by about 7 %.
   - **The crossing node is discontinuous in the radius ratio.** With the
     1-radial hub at r = 4, vertical radius over a 1 mm drop:
     - 1.0 mm: NEC-4.2 189 - 67j, NEC-5 185 - 68j;
     - 1.1 mm: NEC-4.2 203 - 408j, NEC-5 185 - 67j;
     - 1.5 mm: NEC-4.2 131 - 765j;
     - 19 mm: NEC-4.2 56 - 495j, NEC-5 179 - 60j (`step_gnd.jsonl`).

     The same 19-on-1 mm step in free space agrees with NEC-5 to ~2 ohm
     (`step.jsonl`).
   - **The `bottom` feed makes it worse**, running to -2773 - 5158j at r = 8.
     The "246 - j517 against ~62 ohm" hand translation is this
     radius-step-at-the-interface fault, not NEC-4.2's real answer for the
     antenna: it does not converge toward NEC-5 under refinement at any feed.
2. **SG serves the node, on the U5 rule.** On the radius RESPONSE, where the
   node convention's gap cancels (`resp_probe.py`, `resp.jsonl`, 1-radial hub,
   r = 4, each configuration minus the all-1 mm control):

   | change | NEC-5 | SG | bs2 | point-matched (observer rule, rejected) |
   |---|---|---|---|---|
   | vertical 2 mm | -0.74+3.25j | -0.79+3.34j | -0.79+3.33j | -12.93+7.98j |
   | vertical 4 mm | -1.79+5.93j | -1.89+6.12j | -1.89+6.10j | -25.82+16.00j |
   | vertical 19 mm | -6.81+7.89j | -7.20+8.30j | -7.13+8.22j | -54.69+34.44j |
   | buried 0.5 mm | +10.04-3.37j | +10.18-3.51j | +10.18-3.51j | -0.97+0.47j |
   | buried 0.25 mm | +19.85-6.72j | +20.11-7.02j | +20.12-7.02j | -2.41+1.17j |

   On the U5 rod (`rod_guard.py`) SG gives rise ×½ +7.862, rise ×¼ +15.627
   and top ×½ +0.493, against bs2's +7.861 / +15.624 / +0.492.
3. **Point-matched refuses the node.** The shipped observer-side rows swap the
   physics, and so do six other placements and weights (`rod_candidates.py`):
   all rows at a_below, at a_above, at the source radius, and two
   radius-aware two-k weights. The cause is below the crossing. The lane's
   ordinary radius-STEP junction misses both references in free space
   (`sin_step_free.py` against `nec42_buried_step.py`). For a dipole with
   0.5 / 1 mm halves, the step response at n = 20 / 40 per half is:

   | engine | n = 20 | n = 40 |
   |---|---|---|
   | sinusoidal | -0.59 - 8.86j | -0.76 - 1.16j (moving away) |
   | NEC-4.2 | -28.1j | -26.6j |
   | NEC-5 | -34.7j | -34.8j |
   | SG | -33.4j | -34.0j |

   Buried, the same deck reads sinusoidal +7.1 / +4.4 against NEC-4.2
   +12.9 / +12.7 and NEC-5 +16.4 / +16.5. Filed separately.
4. **SG `fill="direct"` refuses the node** (`sg_direct_probe.py`). It has no
   two-radius spelling for the node content its closed forms carry: on the rod
   it gives rise ×½ -163.7, and on the hub a 19 mm vertical moves it
   -161.8 - 471.1j.

`runs/tables.txt` is every ladder (`tables.py runs/ref.jsonl runs/mw.jsonl
runs/mw2.jsonl`): Z per rung and |dZ| per doubling. Its `sinusoidal` rows are
the REJECTED observer-side rule, run before the refusal landed. On the
16-radial hub, `main` / `mid`, SG minus NEC-5 is:

| r | SG - NEC-5 | \|gap\| |
|---|---|---|
| 1 | 5.50+5.92j | 8.08 |
| 2 | 2.63+3.68j | 4.52 |
| 4 | 1.34+2.48j | 2.82 |
| 8 | 0.73+1.77j | 1.91 |

The gap shrinks every doubling, and SG and bs2 agree to 0.3 ohm at every
rung.
