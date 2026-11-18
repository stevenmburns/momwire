# One bound across the executables' Sommerfeld stores

Decision record for momwire#2014.

## The problem

A standalone executable keeps its Sommerfeld tables in
`<cache>/sommerfeld/exe/<tag>`, where the tag is the version plus the build's
date and time. `MOMWIRE_SOMM_CACHE_MAX_MB` used to bound one such directory.
A replaced executable's directory was never read again and nothing deleted it,
so the total grew by one directory per build: 9.4 GB on one development machine
before a hand prune on 2026-11-18.

## The decision

**`MOMWIRE_SOMM_CACHE_MAX_MB` now bounds every build under `exe/` together.**
After each write, the eviction pass walks `exe/*/`, adds up the finished files
of all builds, and deletes `.grid` tables least recently used first until the
total is under the bound. The default stays 512 MB.

| Candidate | Verdict | Why |
|---|---|---|
| 1. One LRU bound across all builds | **Chosen** | It is the only candidate that bounds the total by construction. A replaced build stops refreshing its files, so its tables are the first to go, with no notion of "stale" to tune. |
| 2. Prune sibling builds at startup | Rejected | An age or count rule cannot tell an old build from one that is installed and in use, such as the MSVC and MinGW previews on one PC. It also bounds the number of directories rather than the bytes. |
| 3. Key the directory by content | Rejected | It needs a trustworthy C++ fingerprint of everything the tables depend on. The build timestamp exists because the version string is not one. It would also still need a bound. |
| 4. An explicit prune command | Rejected | Users will not run it, and the growth is invisible until the disk is full. |

## Answers to the issue's questions

**Do the executable and Python stores share one bound?** No, each has its own.
The Python store's `.npz` files sit directly in `<cache>/sommerfeld/` and its
eviction scans only that level. The two stores are used by different programs
and neither can read the other's files, so sharing a bound would let one evict
the other's tables for no gain. The most the two can hold together is twice the
bound: 1 GB by default.

**What is the default total?** 512 MB, unchanged. A lone executable keeps the
capacity it always had, so its hit rates do not move. Raising the default would
only make the total larger for users who have one build.

**How do two builds running at once behave?** They share the bound fairly by
use. If their tables fit together, neither evicts the other. If they do not,
the least recently used tables go whichever build owns them, and the owner
refills on its next miss. Each build keeps its own directory, so their files
never mix.

**Does `momwire-engine.exe` use this store?** No. The drop-in engine is the
frozen Python program (`scripts/eznec_freeze/entry.py`), and it uses the Python
store. Only the standalone routes call `configure_disk_cache` (`standalone/route.cpp`,
`route_bs2.cpp`, `route_sin.cpp`), and the Python extension never does.

## What the pass guarantees

- **A race is a miss, never an error.** Writers still use a unique temporary
  file renamed into place, so a reader sees a whole file or none. The directory
  walk uses the non-throwing iterator calls, because another process can delete
  an entry or a whole directory during the walk.
- **A file that cannot be deleted is skipped.** Windows refuses to delete a file
  open in another process. The pass moves on to the next table and a later pass
  retries.
- **Only the store's own files are deleted.** Those are `.grid` tables, `.tmp`
  files over an hour old, and the seen lists of a retired build. Symbolic links
  are never followed, and a directory holding anything else is left in place.
- **Seen lists are not evicted.** `seen-*.bin` files steer the below table's
  sweep anchors, so deleting one can change a later sweep's digits. They count
  toward the total. A build with no tables left keeps them for 30 days after
  their last update, and then its directory is removed.

## Limits of the bound

The bound holds when a pass finishes, not at every instant:

- Concurrent writers can overshoot briefly, between one writer's rename and its
  pass. With 11 concurrent processes under a 2 MB bound, the largest total
  sampled was 3.0 MB, and the store ended at 1.3 MB.
- The file just written is never evicted by its own pass, so a single table
  larger than the bound stays until something replaces it.
- A table held open on Windows stays until a later pass can delete it.
- Temporary files in flight are not counted.

Each write now scans every build's files, about 12 µs per file on the
development machine, or 61 ms per write for a store of 5,000 tables. The bound
limits how many files there can be.

## Verification

`tests/test_somm_store_bound_2014.py` holds the bound, concurrency and retirement
rules on a small driver (`scripts/core_split/somm_store_driver.cpp`), on Linux.
`tests/test_standalone_nopython.py` holds the shipped executable to the bound,
including each platform's default cache directory and a table held open by
another process. The `standalone` workflow runs that module on Linux, Windows
and macOS.
