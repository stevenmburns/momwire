# Networks move into the engine: TL/NT through the portal

Design doc for momwire#456 workstream 2's central decision, taken 2026-08-19:
**the portal stays whole inside momwire, and momwire gains the network
solve.** The MNA core currently in antennaknobs moves down into a momwire
submodule; antennaknobs becomes its consumer; the nec2 dialect then serves
`TL`/`NT`. This document records why, what moves, what must be fixed before
the boundary hardens, and the landmines with names.

## The decision, and the case

Three seams want a drop-in engine (SimNEC shipped; 4nec2 and EZNEC captured
— antennaknobs `docs/status/2026-08-16-…` / `2026-08-18-…`), and two of the
three emit network cards: `NT` reaches the engine in 53 of the 457 bundled
4nec2 models (52 manufactured from `EX 6`), `TL` in 45; EZNEC's own
feed-system examples are TL/NT models, and its junction loads arrive as `NT`
cards. NEC-2 and NEC-5 both solve networks natively; an engine standing at
their seams has to as well.

The alternative both capture docs leaned toward — putting the frontend seam
in antennaknobs, which already had the pieces — answered "what is the
cheapest seam", not "what is the product". It loses on three counts:

* **Packaging.** A frozen Windows drop-in (workstream 5) built from momwire
  bundles a compute library; built from antennaknobs it bundles the app —
  server, plotting, catalogs — and every further seam doubles that surface.
* **Dependency direction.** antennaknobs → momwire is the only clean arrow.
  Option 3 makes the app repo the engine-emulation vehicle and fragments the
  "which binary is the momwire drop-in" identity.
* **The oracle flips in our favor.** Serving `TL`/`NT` makes the nec2
  dialect *more* byte-compatible with nec2c, not less: the three retired
  differential fixtures (`dipole_nt_network`, `dipole_tl_network`,
  `dipole_tl_shunt_crossed`) come off the refusal list and return to byte
  comparison against the engine being emulated.

## The inventory (what actually lives in antennaknobs)

Surveyed 2026-08-19, full detail in the survey; the load-bearing findings:

* The solver is `network_reduce.py` (1,519 lines): pure-numpy TL/chain math
  (lines 120–260, zero first-party imports), the MNA primitives and
  `MNASystem` (263–649, type-free — it never sees a `Network`, a port name
  or a branch dataclass), and `NetworkReducer` with its ~460-line
  `apply_branches` stamping switch. Its module-scope imports are stdlib +
  numpy + one first-party module (`.network`); scipy (`zgesvx`) is imported
  lazily inside `solve()`. **momwire's declared deps (numpy, scipy) already
  cover it exactly.**
* The spec layer is `network.py` (1,619 lines): port types (`PortOnWire`,
  `PortVirtual`, `PortAtEnd`, `PortAtVertex`, `PortOnWireFloating`), eleven
  branch dataclasses, sources, the `Network` container — **plus ~148 lines
  of wire/cable catalog geometry that has nothing to do with circuits** and
  is imported by the engine/builder/web layers.
* Nobody subclasses `NetworkReducer` or `MNASystem` anywhere; coupling is
  100 % composition. The engines own the port-index contract and hand the
  reducer a dict. The web layer's entire contact is one exception class.
* ~240 tests pin the core; the two strongest cross-implementation oracles
  (`test_tl_composition.py`, the #65 `nt_card` oracle in
  `test_momwire_engine.py`) are **PyNEC-gated** and cannot run inside
  momwire's dependency envelope.
* One live import cycle (`network_reduce → network → touchstone`, papered
  over with call-time imports) **dissolves** if the core moves and the
  Touchstone parser stays.

**Verdict: extraction is a go.** The feared #426-class entanglement is not
there. The seam falls at `apply_branches`: below it nothing is
app-specific; above it (instance flattening, Touchstone reading, anchor
virtualization, schematic labels) everything stays.

## What moves, what stays

Into a new **`momwire.networks`** submodule (numpy/scipy only):

| moves | notes |
|---|---|
| TL/chain/balanced math (`tl_abcd`, `tl_admittance_2x2`, `balanced_admittance_4x4`, …) | already dependency-free |
| `_Group2Element`, `_stamp_abcd`, `_series_group2`, `magnetizing_impedance`, `MNASystem`, `solve`, `poison_singular_sample`, `SingularNetworkError`, `RCOND_*` policy | the type-free inner core |
| `NetworkReducer` (`apply_branches` down: terminations, `excited_state`, `impedance_from_y`, `reflection_from_y`, …) | its one internal-type input is `Network` |
| The branch/port/source dataclasses and a **flat** `Network` container | `PortVirtual` moves (the reducer's node space needs it); RLC helper functions move |

Stays in antennaknobs:

| stays | why |
|---|---|
| `Composite`/`Instance`/flattening, `station.py` | design-authoring hierarchy, downstream of schematic; antennaknobs flattens **before** handing momwire a flat `Network`, so the private `_branch_port_refs`/`_rewrite_branch` reach-through never crosses the boundary |
| Touchstone parser | `TouchstoneLoad.data`/`TouchstoneTwoPort.data` retyped to a ~10-line `y_at(f_hz)`/`nports` protocol; the parser and its Builder-confined file reading stay |
| wire/cable catalogs, `validate_named_wires_referenced` | geometry and engine-contract code that only shares a file today |
| `_anchor_wires` + the whole `nec_import` translation layer, `plane.py`, `module.py`, schematic | app land, unambiguous |
| PyNEC-gated oracles (`test_tl_composition`, `nt_card`) | remain as antennaknobs integration tests guarding the moved core from above |

Compatibility: `antennaknobs.network` / `antennaknobs.network_reduce` become
re-export shims for one release so the ~40 design files and 52 test files
don't churn in the move PR.

## Pre-move chores (before the boundary hardens)

1. **The label-string contract.** `schematic.py` pattern-matches the
   reducer's power-budget label strings ("TL rig→feed", …) — a documented
   drift hazard *inside one repo*, an undeclared cross-repo contract after
   the move. Replace with a structured probe identity (branch kind +
   terminals as data, label rendered app-side) **first**.
2. **Split `network.py`.** The ~148 geometry lines out of the circuit
   module; mechanical but wide-blast-radius (engine/builder/web/designs all
   import from it).
3. **Touchstone protocol** as above.
4. Consolidate the `C_LIGHT` duplicates onto the momwire-side constant at
   move time (four owners today).

## The dialect: parse per-reader, semantics once

The split the geometry code already uses ("shared, not merely equivalent"):

* **Parsing stays per-dialect.** The importer keeps its permissive full
  grammar and skip-lists; the portal's nec2 dialect gets strict `TL`/`NT`
  sections written into `deck-grammar-nec2.md` first, tests citing anchors.
* **Card semantics live once, in momwire**: crossed lines via negative Z₀
  (normalised to a `transposed` polarity flag, *not* a negated Z₀),
  zero-length auto-distance from segment midpoints, the three shunt-stub
  forms, `NT` Y-triples vs the exact resistive-π reduction, signed
  addressing. Ported from `nec_import.py`'s translation layer
  (`_translate_network_cards`, `_end_shunt`), which is wild-corpus
  validated; antennaknobs' `NecDeck.network()` then delegates its card→
  branch mapping down so the two readers cannot drift on exactly the cards
  where a subtle divergence costs the most (the W7EL triple's config C is a
  70 % error from one canonicalization mistake).
* The abstraction gap named by the inventory — momwire's deck pipeline
  speaks `PortPlan`, the core speaks a named-port `Network` — is closed on
  the deck side: `deck/_solver` builds `port_to_idx` from the deck's union
  port set exactly the way `engines/momwire.py` does today, and hands the
  reducer a flat `Network`. momwire does not adopt hierarchy.

## Portal serve path (phase C)

`DeckModel` grows a network record (branches + connection addressing per
execute group); `build_solver` composes the antenna port Y with the reducer;
the printout renderer adds the `NETWORK DATA` and `STRUCTURE EXCITATION DATA
AT NETWORK CONNECTION POINTS` blocks in nec2c's format. Gates, in order of
strength:

1. The three retired differential fixtures return to **byte comparison**
   against nec2c, plus new fixtures for the shunt-stub and crossed-line
   variants.
2. The `EX 6` manufactured form (phantom wire + `EX 0` + `NT`) pinned
   specifically — 52 of the 457 bundled 4nec2 models arrive this way.
3. The corpus ladder re-run: serving NT/TL takes the bundle from 64 % to
   78 % serve; the census script is the measurement.
4. (Later, NEC-5 front:) the W7EL oracle triple gates favored-wire node
   addressing.

## Landmines, named

* **momwire#157 — remote network wires.** Portal-native networks put
  EZNEC's 100 λ virtual wires and 4nec2's z-equals-tag phantom wires into
  momwire's *geometry*. Over Sommerfeld ground that is the #157
  assembly-hang regime. Strategy decision belongs to phase C: virtualize
  network-only remote wires inside the deck pipeline (the engine-side
  analogue of `_anchor_wires`, but on the strict dialect where the idiom is
  identifiable), fix #157 outright, or ship with a named envelope. Not
  optional to ignore: EZNEC's feed-system decks are *exactly* this shape.
* **Symmetry × networks.** #415's cell model is oracle-verified for `LD`;
  whether `TL`/`NT` follow the same cell rule is measured-not: the probe
  decks extend in minutes and must run before symmetric decks with networks
  are served. Until then: a `GX`/`GR`-live deck with network cards refuses
  by name (the same pattern as #415's staged guard).
* **The spec's identity changes.** "Antenna-only" is load-bearing language
  across `deck-grammar-nec2.md`, the refusal messages, and #388's
  "deliberate split" row. Phase C includes the spec rewrite; the SimNEC
  seam's byte-exactness claims must be re-verified after the dialect widens
  (nec2c serves these cards, so the differential suite is the check).

## Phasing

Each phase is one stacked arc; no phase starts before the prior's gates are
green. No release is scheduled until there is a consumer for one.

* **Phase A — prep (antennaknobs):** label-string contract → structured
  probes; `network.py` split; Touchstone protocol. Gates: full antennaknobs
  suite, schematic tests, zero behavior change.
* **Phase B — the move:** `momwire.networks` lands with the core + its
  portable tests (the synthetic-Y oracle battery); antennaknobs rewires via
  shims; the PyNEC oracles stay behind and must stay green. Gates: both
  suites, plus the `nt_card` and TL-composition oracles unchanged to the
  digit.
* **Phase C — the portal serves:** dialect spec first, parser + semantics +
  reducer composition + printout blocks; the byte-differential gates above;
  the #157 and symmetry decisions taken and implemented or explicitly
  refused-by-name.
* **Phase D — consumers converge:** `NecDeck.network()` delegates card
  semantics to momwire; the census ladder re-measured; #456 checklist
  updated.

## Non-goals

The station-model authoring layer (`Composite`/`Instance`/`station.py`),
schematic, plane picker, VNA/Touchstone tooling, and anchor virtualization
stay antennaknobs. SimNEC's own external circuit solving is unaffected.
Nothing here implements radial screens, surface patches, or the MININEC
ground decision — separate #456 threads.
