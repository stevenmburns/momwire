# momwire#2063 stage 0: survey (read-only), 2026-11-26

This survey reads the code at main ed479892 (v0.98.0). There are two parts, and each claim in them carries a file:line:

- [A-direct-offset.md](A-direct-offset.md): the direct and offset families on razor2p, sinusoidal and bs2; the switches and every reader site; and the proposed stage-A interface.
- [B-ground.md](B-ground.md): the PEC image, the refl-coef image and the Sommerfeld remainder on each basis; why `bs2-image-reuse` saved no time; and the proposed stage-B interface.

I spot-checked the load-bearing claims against the tree before writing this index:

- sinusoidal's `GroundKey` (`_sin_pair_reuse_core.cpp:245`) and razor's `q_key` (`_razor_pair_reuse_core.cpp:166`) both key on the height sum plus horizontal terms.
- bs2's remainder key (`_bspline_pair_reuse_core.cpp:367-392`) uses six distances and four separate heights.
- `_offset_rows` (`bspline.py:5796`) has no caller.
- The header comment at `_sin_pair_reuse_core.h:25-28` is stale: it still describes the M·N/2 cap.
- AK passes `offset_reuse` (`engines/momwire.py:2424`).

## What changes in the #2063 plan

1. **Stage B's key.** The re-plan's identity, |P−M(Q)|² = |P−Q|² + 4·h_P·h_Q, is true, but it is the wrong key. "Equal direct key plus equal h_P·h_Q" fixes neither ρ nor z+z′, and the refl-coef and Sommerfeld terms read those two separately. The key that works is the one sinusoidal and razor already use: the height sum and the horizontal offset directly. bs2's remainder map is narrower than that. It finds same-height horizontal translations only, never a vertical stack (B §3).
2. **The stage-B copy for bs2 must copy assembled entries, not moments.** `bs2-image-reuse` cut the pair-terms from 5.15M to 1.25M but left the weights window and the weighted assembly running for every pair. Its own numbers show the kernel 0.14 s faster but the fill only 0.016 s faster (B §5). The branch was never profiled, so do that before briefing B.
3. **Stage A is not one stage to port.** bs2's `offset_reuse` is four mechanisms (A §2):
   - R: the reg Toeplitz, the only exact one;
   - F: the free-space offset block;
   - I: the image offset block;
   - S: the Sommerfeld fold.

   F, I and S spread an edge's own first and last rows, which is where the registry's 5e-11 drift comes from.
4. **The offset stage may not pay on razor2p and sinusoidal.** Their censuses already share the far pairs on a straight run. An offset stage adds only about 7 near pairs per row, and it pays more only when segments are shorter than the radius. That is the #1971 hairpin, the case where sharing near pairs moved Zin by 1.5e-5 (A §3). The survey's recommendation: ship A0 (bs2's offset mechanics moved into the core, byte-identical) and only *measure* A1 (enabling it on the other two).
5. **There is a floor in bs2's direct family.** In free space with reuse on, bs2 fills 92k computed pairs in 0.195 s and razor fills 95k in 0.043 s. Razor avoids this floor with the fused `assemble_rows` (#2144). It is not in the plan yet (B §6).
6. **Coverage gaps outside the plan:**
   - The sinusoidal exe declines refl-coef reuse, which Python serves.
   - The bs2 exe has no refl-coef route, so exe == Python can only be gated on PEC and Sommerfeld.
   - Buried and transmitted fills have no reuse on any basis.
   - #1772's identical-edge copy is a fifth, ungated reuse.
7. **The cross-engine gate in `test_bs2_ground_reuse_report_2159.py` compares unlike families.** bs2's "image" is the image alone, sinusoidal's "reflected" is the image plus the remainder, and razor's "image" shares classes with its direct block. It needs a like-for-like definition before it can be the stage-B gate.
8. **Stage D:** the reader sites are listed in A §4–5, AK among them. A deprecation warning would fire on every AK bs2 fill, so AK must stop passing `offset_reuse` first. One switch also loses "census on, offset off" and "offset alone".

## Questions for Steve

Both files carry the full lists (A §6, B §6). These need answers before stages A or B are briefed:

1. **Stage A scope.** Should A be A0 only (bs2 byte-identical into the core) with A1 as a measurement, as the survey recommends?
2. **The stage-B copy for bs2.** Should it be a copy of assembled basis-pair entries from a budgeted side buffer, after the sweep and offset blocks and before the near-image fixup? That means profiling `bs2-image-reuse` first.
3. **Sommerfeld families.** One joint "ground" family (image plus remainder, sinusoidal's shape) or two (image and remainder)? This also decides what the cross-engine gate compares.
4. **The bs2 direct-family floor** (bs2 0.195 s against razor 0.043 s): does it go in stage A, stage B, or an issue of its own?
5. **Stage D:** is losing "offset without the census" (and the reverse) intended? What order should AK's change and momwire's deprecation go in?
6. **razor2p's acceptance deck for any offset stage.** Is it the #1971 hairpin? Should the offset stage decline under `exact_kernel`?
7. **Follow-ups to file**, each linked both ways on #2063: the bs2 direct floor, the sinusoidal exe's refl-coef decline, and the two pieces of dead or stale code.
