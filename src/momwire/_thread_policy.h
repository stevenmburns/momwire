// How many threads a deck of a given size runs on (momwire#2023): the one
// rule behind the standalone executables' `threads_for` and the Python
// drop-in's `momwire._thread_policy`. No platform header and no arithmetic on
// floats; `cores` is the caller's count of the physical cores it may use.
#pragma once

#include <algorithm>

namespace mw_threads {

// The size policy, from the largest solve's basis functions (unit 5's sweep,
// Rocketlake, 8 cores / 16 threads, one process per call): up to ~100 one
// thread is fastest (dip101 2.7 ms; 4 threads 2.9, 8 threads 3.5), so under
// 128 the deck runs on the serial copy and never starts OpenMP; from ~150
// to ~300 four threads win (dip301 8.2 ms against 14.6 on one and 8.4 on
// eight); from ~400 every core (dip801 36 ms on 8, 42 on 4, 99 on 1).
constexpr long long ONE_THREAD_BELOW = 128;
constexpr long long FOUR_THREADS_BELOW = 384;

// A buried deck with a crossing block (`crossing_blocks`: every above
// segment against every below one, Sommerfeld integrals per node pair) does
// far more work per basis function than the dipoles above were timed on
// (momwire#1956), so from CROSSING_PAIRS such segment pairs it gets every
// core whatever its basis count. Rocketlake, 8 cores, median of 5, ms at the
// old choice -> every core: slant_base_source (128 pairs, 23 basis) 37.6 ->
// 14.5; deep_s40 (324, 44) 50.1 -> 15.3; d3_hub (1312, 97, near field) 841
// -> 178; fat_hub (3000, 144) 48.7 -> 35.1; ground_rod_efhw (2895, 207)
// 104.9 -> 72.4; aca_vertical_screen (24684, 324) 92.7 -> 73.2. No crossing
// deck measured was faster on fewer, and the smallest has 128 pairs. Decks
// without one (free space, above ground) keep the basis-count rule; a deck
// wholly below the plane has its own measure, below.
constexpr long long CROSSING_PAIRS = 64;

// A wholly-buried deck (`below_pairs`: its segments squared, every pair a
// below/below Sommerfeld pair) does more fill work per basis function than
// the free-space dipoles too, but less than a crossing block, and it has no
// crossing pairs to count. Forced-thread sweep, Rocketlake, median of 15, ms
// at 4 -> 8 threads: under ~30,000 pairs it is flat (bd_s161, 25,921 pairs,
// 13.7 -> 12.6; bh_r16_s10, 25,600, 13.6 -> 13.8); from 40,000 pairs 8 wins
// on 8 of the 10 decks measured: bd_s201 (40,401) 16.4 -> 12.4, ab_long_wire
// (57,600) 21.8 -> 18.5, bd_s281 (78,961) 24.0 -> 17.1, bh_r12_s25 (90,000)
// 25.8 -> 21.1, ab_hub16 (102,400) 27.9 -> 22.0; the other two, hubs of
// 57,600 pairs, were within 0.8 ms either way. Bits do not depend on the
// count on this route (byte-identical at 1, 4 and 8), so only time moves.
constexpr long long BELOW_PAIRS = 40000;

inline int for_deck(int cores, long long n_basis, long long crossing_pairs = 0,
                    long long below_pairs = 0) {
    if (crossing_pairs >= CROSSING_PAIRS) return cores;
    if (below_pairs >= BELOW_PAIRS) return cores;
    if (n_basis < ONE_THREAD_BELOW) return 1;
    if (n_basis < FOUR_THREADS_BELOW) return std::min(4, cores);
    return cores;
}

}  // namespace mw_threads
