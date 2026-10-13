# momwire#1170 stage 0: extended kernel above ground, reduced on buried pairs

The record behind the PR's numbers. 2026-10-13, change tree `7481b353`
against base `afe9e92e`, both fresh clones built with `make build` on Skylake.

## What is here

- `decks/` the decks every engine read, one file per deck and dialect.
  `g8hul_nt_*` is G8HUL's end-fed vertical on a 2 m ground stake as he wrote
  it (antennaknobs `scratch/g8hul-advisory`), fed through its 49:1 network;
  `g8hul_*` is the same antenna fed directly, so the antenna port is read;
  `brv_*` is the 16-radial buried-radial vertical (`hub_deck(16)`, the
  momwire#1173/#1224 bench decks); `fat_*` is a 25 mm vertical on a 1 m
  stake, at segment/radius 5 (x1) and 2.5 (x2), where the extended kernel
  is large enough to measure. A `.nec4` deck feeds a segment centre and a
  `.nec5` deck a node, so the two dialects' decks are not the same feed and
  are never compared with each other.
- `refrun.py`, `runset.sh` one deck through a momwire EZNEC slot, as served
  and with the default extended kernel forced off for the whole deck (what
  the fallback did before this change). Records the port impedance at full
  precision, each solver's per-wire kernel, and a census of what the moment
  kernel was handed.
- `run_engine.sh` an engine as a black box, `exe deck printout`, each deck
  in a fresh directory. NEC-4.2 for `.nec4`, clean-room NEC-5 for `.nec5`.
  The printouts are not committed; `table.py` holds the impedances printed.
- `gates_ab.py` the bit-identity gate, base tree against change tree.
- `table.py` prints the tables below from `results/`.

## Bit identity, base against change

`gates_ab.py`, 70 arrays (operator Z and feed impedance), 20542 complex
values, `np.array_equal`:

    g1 (no buried wire, extended_kernel False and True): 48/48 arrays, 7548 values
    g2 (buried wires, extended_kernel False, chunked and dense): 22/22 arrays, 12994 values

And through the front end, on all twelve thin-wire reference decks, the base
tree as it serves the deck is the change tree with the kernel forced off,
bit for bit on the solver's port impedance.

## Reference comparison at equal mesh

Ohms. "gap" is |Z - reference|. "moves" is how the gap changed.

| deck | reference | reference Z | all reduced (today) | extended above | shift | gap before | gap after | moves |
|---|---|---|---|---|---|---|---|---|
| g8hul_x1.nec4 | NEC-4.2 | 2925.33+257.284j | 2936.5096+157.1796j | 2936.4973+157.2172j | -0.0123+0.0376j | 100.7267 | 100.6880 | toward by 0.0388 |
| g8hul_x3.nec4 | NEC-4.2 | 2873.61-450.905j | 2855.7627-491.9355j | 2855.7896-491.8295j | +0.0269+0.1060j | 44.7441 | 44.6361 | toward by 0.1080 |
| g8hul_nt_x1.nec4 | NEC-4.2 | 59.8084+5.21897j | 60.0370+3.1508j | 60.0368+3.1517j | -0.0002+0.0009j | 2.0808 | 2.0798 | toward by 0.0009 |
| g8hul_nt_x3.nec4 | NEC-4.2 | 58.7375-9.23229j | 58.3645-10.0920j | 58.3651-10.0897j | +0.0006+0.0023j | 0.9371 | 0.9348 | toward by 0.0023 |
| brv_x1.nec4 | NEC-4.2 | 101.158-57.8815j | 100.8466-58.6446j | 100.8463-58.6495j | -0.0004-0.0049j | 0.8242 | 0.8288 | AWAY by 0.0047 |
| brv_x2.nec4 | NEC-4.2 | 102.117-60.0176j | 100.7151-58.4820j | 100.7147-58.4875j | -0.0004-0.0055j | 2.0793 | 2.0755 | toward by 0.0037 |
| brv_x4.nec4 | NEC-4.2 | 100.993-58.7559j | 100.6388-58.3628j | 100.6383-58.3694j | -0.0005-0.0065j | 0.5292 | 0.5246 | toward by 0.0045 |
| fat_x1.nec4 | NEC-4.2 | 111.239-66.1858j | 111.1800-66.8537j | 111.1922-67.0124j | +0.0122-0.1588j | 0.6705 | 0.8280 | AWAY by 0.1575 |
| fat_x2.nec4 | NEC-4.2 | 111.033-66.7439j | 110.8542-66.8431j | 110.9624-67.0256j | +0.1082-0.1826j | 0.2045 | 0.2905 | AWAY by 0.0860 |
| g8hul_x1.nec5 | NEC-5 | 2725.4+753.78j | 2635.0971+898.4921j | 2635.0451+898.5487j | -0.0520+0.0566j | 170.5761 | 170.6516 | AWAY by 0.0755 |
| g8hul_x3.nec5 | NEC-5 | 2920.7-250.1j | 2933.9494-170.2753j | 2933.9535-170.1571j | +0.0040+0.1183j | 80.9168 | 81.0341 | AWAY by 0.1173 |
| brv_x1.nec5 | NEC-5 | 112.25-76.157j | 113.9886-67.1340j | 113.9880-67.1402j | -0.0006-0.0062j | 9.1890 | 9.1828 | toward by 0.0062 |
| brv_x2.nec5 | NEC-5 | 100.19-62.253j | 100.8398-58.4462j | 100.8393-58.4519j | -0.0005-0.0057j | 3.8618 | 3.8562 | toward by 0.0057 |
| brv_x4.nec5 | NEC-5 | 100.42-60.122j | 100.7079-58.3638j | 100.7075-58.3704j | -0.0005-0.0066j | 1.7816 | 1.7751 | toward by 0.0065 |
| fat_x1.nec5 | NEC-5 | 111.77-69.726j | 112.0357-67.2706j | 112.0257-67.4436j | -0.0100-0.1731j | 2.4698 | 2.2967 | toward by 0.1731 |
| fat_x2.nec5 | NEC-5 | 111.28-68.855j | 111.4965-66.9456j | 111.5195-67.1755j | +0.0230-0.2299j | 1.9217 | 1.6965 | toward by 0.2252 |

Read plainly:

- **Buried-radial vertical**: the shift is 5 to 7 milliohm against gaps of
  0.5 to 9 ohm. Toward on five of six rows, away on one. It decides nothing.
- **G8HUL's vertical** (1 mm wire, segment/radius 500 at x1): at most
  0.12 ohm against gaps of 45 to 170 ohm on a feed whose reactance moves
  700 ohm between x1 and x3 in every engine. Toward NEC-4.2, away from
  NEC-5. It decides nothing either.
- **Fat vertical on a stake**: 0.09 to 0.23 ohm against gaps of 0.2 to
  2.5 ohm. Toward NEC-5 on both meshes, AWAY from NEC-4.2 on both.

So stage 0 is not an accuracy result. It makes the above-ground wires of a
buried deck solve with the kernel the dialect asked for; where that is
measurable it lands nearer NEC-5 and further from NEC-4.2.

## Which pairs took which kernel

Counted in the moment kernel's own call (`refrun.py` wraps
`_seg_seg_full_moments_offedge`): pairs handed in, and pairs the spec it was
handed extends. The counts are per kernel call, so a pair is counted once
for the direct block, once for the image block, and once more where a
same-edge block is refilled to be subtracted. The cross-medium pairs are a
field-form block and never reach this kernel.

| deck | wires extended / all | air pairs: handed, extended | soil pairs: handed, extended | forced reduced: air, soil |
|---|---|---|---|---|
| g8hul_x1.nec4 | 1 / 2 | 4722, 4722 | 300, 0 | [4722, 0], [300, 0] |
| g8hul_x3.nec4 | 1 / 2 | 42962, 42962 | 2700, 0 | [42962, 0], [2700, 0] |
| g8hul_nt_x1.nec4 | 2 / 3 | 4885, 4885 | 300, 0 | [4885, 0], [300, 0] |
| g8hul_nt_x3.nec4 | 2 / 3 | 43445, 43445 | 2700, 0 | [43445, 0], [2700, 0] |
| brv_x1.nec4 | 1 / 18 | 551, 551 | 54092, 0 | [551, 0], [54092, 0] |
| brv_x2.nec4 | 1 / 18 | 2234, 2234 | 213772, 0 | [2234, 0], [213772, 0] |
| brv_x4.nec4 | 1 / 18 | 8982, 8982 | 849932, 0 | [8982, 0], [849932, 0] |
| fat_x1.nec4 | 1 / 2 | 16762, 16762 | 192, 0 | [16762, 0], [192, 0] |
| fat_x2.nec4 | 1 / 2 | 67122, 67122 | 768, 0 | [67122, 0], [768, 0] |
| g8hul_x1.nec5 | 1 / 2 | 4800, 4800 | 300, 0 | [4800, 0], [300, 0] |
| g8hul_x3.nec5 | 1 / 2 | 43200, 43200 | 2700, 0 | [43200, 0], [2700, 0] |
| brv_x1.nec5 | 1 / 18 | 675, 675 | 54092, 0 | [675, 0], [54092, 0] |
| brv_x2.nec5 | 1 / 18 | 2700, 2700 | 213772, 0 | [2700, 0], [213772, 0] |
| brv_x4.nec5 | 1 / 18 | 10800, 10800 | 849932, 0 | [10800, 0], [849932, 0] |
| fat_x1.nec5 | 1 / 2 | 19200, 19200 | 192, 0 | [19200, 0], [192, 0] |
| fat_x2.nec5 | 1 / 2 | 76800, 76800 | 768, 0 | [76800, 0], [768, 0] |

