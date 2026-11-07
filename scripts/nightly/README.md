# The Skylake nightly

Rollout step 5 of the dev-flow contract. On a pull request, momwire runs only
the Linux default lane and the stable-path byte tripwire
(`tests/test_stable_path_tripwire.py`). Executable parity, the sse2 variant,
thread counts and timing are not per-PR gates any more. This nightly is the
net behind them. It runs on Skylake (i7-6700K: 4 cores, 8 threads, no
AVX-512) against momwire `main`, once a night.

It is not `.github/workflows/nightly.yml`. That one is the GitHub-hosted
`pytest -m nightly` sweep, and this one leaves it alone.

## What runs

1. **Build.** `git fetch`, then `main`'s commit is checked out in
   `~/antennas/nightly/mw`. Python is built with `make build`, so
   `MOMWIRE_REQUIRE_ACCEL=1` and a failed compile fails the night. The four
   no-Python executables are built with the standalone CMake build
   (`standalone/CMakeLists.txt`, the flags `build_release_linux.sh` passes),
   ccache, and the two static OpenBLAS archives from
   `standalone/build_openblas.sh`. Those are built once per hash of that
   script.
2. **exe == Python, `avx2-T1`, every night.** Every deck in `corpus.txt` goes
   through each executable that serves its dialect, and through the Python
   route that executable replays (`_shell.render_files`, in-process, as
   `audit.py` says). The printouts must be byte-identical, with only the
   NEC-4.2 slot's two measured-time lines masked:

   | label | CMake target (today's name) | Python basis | dialect |
   |---|---|---|---|
   | nec5-bspline | `momwire-bs2-standalone` | bspline | NEC-5 |
   | nec5-razor-2p | `momwire-razor2p-standalone` | razor-2p | NEC-5 |
   | nec42-bspline | `momwire-nec42-standalone` | bspline | NEC-4.2 |
   | nec42-sinusoidal | `momwire-nec42-sinusoidal-standalone` | sinusoidal | NEC-4.2 |

   The SG executables were removed in #1828 and are not here. Environment:
   `momwire_serve_client.pin_cpu_env()`, which pins AVX-512 off only on a CPU
   that has it, so nothing on Skylake. Also `MOMWIRE_BELOW_ANCHORS=off` and
   `MOMWIRE_FORCE_VARIANT=<variant>`. Both sides get `MOMWIRE_THREADS`,
   `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` set to the cell's thread
   count.

   A deck the executable **declines** (route `refused:<why>`) is recorded,
   not failed. Python still answers it, though, because an executable that
   declines what Python serves is a finding.
3. **One rotating cell.** One more cell of {avx2, sse2} × {T1, T2, T4}
   (avx2-T1 excluded) runs the same audit. The cell is picked from the date:
   `ROTATION[date.toordinal() % 5]` in `nightly.py`, with `ROTATION` =
   avx2-T2, avx2-T4, sse2-T1, sse2-T2, sse2-T4. Five consecutive nights cover
   the whole space.

   An sse2 cell uses the executables' sse2 builds, and Python is forced to its
   sse2 accelerator. The harness checks `momwire._accel.VARIANT` in every
   worker, and a mismatch is a severe `variant` finding. **These cells are
   what certify support level 2** (`ak-linux-sse2` in
   `support/hierarchy.toml`), which has no PR job.
4. **The fast perf sweep.** This is antennaknobs' `scratch/pretag-sweep`
   (`driver.py fast`, 643 cases), run from a sparse checkout of antennaknobs
   at the fixed commit `AK_SHA` in `nightly.py`. The same checkout supplies
   the sweep's `--ak-src`. The sweep is too big to copy here (cases.json alone
   is 6000 lines), and its cases are antennaknobs' catalog, so it is invoked
   in place and its commit is recorded in every row (`ak_sha`). Moving
   `AK_SHA` is a deliberate PR, and it re-baselines the sweep: triage skips
   the sweep comparison on the night the pin moves.
5. **Triage** (`triage.py`) compares the night with the previous green night
   on this box. It does not compare with the release baseline.
6. **Bisect** runs on a red night. It searches the first-parent commits since
   the last green night, re-running only the failing decks and cases.
7. **Report** goes to one pinned issue, "Skylake nightly status".
8. **Prune.** The last 14 nights are kept, plus the last green one, with the
   results capped at 1 GB.

**Skip:** when `main` is the commit the last green night ran, the night does
nothing (it makes no `gh` call either). That matches `nightly.yml`. A red
night on an unmoved `main` does run again, picking up the next rotation
cell. It posts no second comment if the findings are the same.

The whole night is budgeted at about 2 h of wall time. The first night took
the time recorded in its issue body.

## The corpus

`corpus.txt` lists in-repo fixture directories, and every `*.nec` / `*.deck`
in them is a deck. Each line also says which dialects it feeds. A deck added
to one of those directories joins the nightly automatically.
`tests/test_skylake_nightly.py` fails if a line names an empty or missing
directory.

An optional `~/antennas/nightly/local_corpus.txt` adds more decks. It lives
on Skylake only and is never committed, one `<dialects>\t<absolute path>`
per line. Those decks are labelled `local#<n>`, and their paths stay in the
night's local files. **Anything posted names in-repo decks only. A local
finding is reported by count and index, never by path or file name.** Keep
private decks out of it (`~/private/`, users' models).

## Severity

The script proposes a severity and never acts on it. It does not revert, open
pull requests or file issues.

| severity | means | proposed action |
|---|---|---|
| severe | exe ≠ Python on a deck the C++ route served; a Python printout or sweep Z that moved more than `Z_REL` = 5e-12 relative, or changed its text; a new outcome regression (sweep ok → not ok, or an executable that served a deck last green night now declines it); a harness phase failed | revert within the day |
| moderate | smaller numeric drift; a sweep MEM or TIME flag (analyze.py's thresholds: +10 % and +30 MB, +10 % and +0.5 s on a timed case); a long-standing decline of a deck Python serves; a Python error or executable non-answer that is not new | open an issue, fix before the next release |
| minor | a printout whose numbers differ only by a last-digit flip (a rounding boundary) | add to the known list |
| known | matched `known.json` | none |

A night is **red** when any finding is severe or moderate. Python drift is
compared only in the avx2-T1 cell, the one that runs every night. The other
cells check exe == Python on their own.

### The known list

`known.json` holds findings that have been seen and accepted. Each entry has
a `kind`, plus fnmatch patterns `match` (deck label or sweep case id), `exe`
and `cell`. A missing pattern matches anything. Each entry also carries a
`note` saying why it is accepted and which issue will retire it. A matched
finding is listed as known and does not turn the night red. Adding to the
list is a PR like any other.

## Reporting

The issue's body is replaced with the latest night every night the nightly
runs. A comment is added only on a red night, or when a night turns green
after a red one. `gh` is always called with `-R stevenmburns/momwire`, and
it makes a handful of calls a night, never polling.

## On Skylake

```
~/antennas/nightly -> ~/storage/nightly   (the big disk)
  mw/          momwire clone, reused nightly (Python built in place)
  venv/        Python 3.12, momwire editable + [test] + matplotlib (the sweep)
  exes/        CMake build dirs, <basis>-<variant>
  openblas/    <hash of build_openblas.sh>/libmwxp.a, libmwnp.a
  ak/          sparse antennaknobs at AK_SHA (src/antennaknobs, scratch/pretag-sweep)
  harness/     this directory, copied from the ref under test
  results/<date>/  night.json, audit-<cell>.jsonl, sweep.jsonl, triage.json,
                   bisect.json, issue-body.md, logs, py-avx2-T1.tar.xz
  state.json   last green / last night / the issue number
  nightly.log  the service's output
```

The harness runs from `harness/`, not from `mw/`. A bisect checks out older
commits, which predate these files. Each night copies `scripts/nightly` from
`main` into `harness/` and re-executes itself if the copy changed.

The units are `~/.config/systemd/user/momwire-skylake-nightly.{service,timer}`
(`python nightly.py install` writes them). The timer fires at 01:30 local.
The service has `MemoryMax=24G` and `MemorySwapMax=0`, and the script takes
`flock ~/antennas/.nightly.lock` itself. Lingering is on for `smburns`, so
the timer fires without a login.

## Re-running a night by hand

```
systemctl --user start momwire-skylake-nightly.service     # as the timer does
tail -f ~/antennas/nightly/nightly.log
```

or, with options, under the same memory cap:

```
cd ~/antennas/nightly
systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0 \
    venv/bin/python harness/nightly.py run --force [--dry-run] \
    [--cells sse2-T2] [--no-sweep] [--no-bisect] [--date 2026-11-08]
```

- `--force` runs even when `main` has not moved.
- `--dry-run` prints the would-be issue body and comment. It posts nothing
  and leaves `state.json` as it was.
- `--cells` replaces avx2-T1 plus the rotation cell.

One cell alone, without the rest of the night:

```
venv/bin/python harness/audit.py --tree mw --exes exes --cell sse2-T1 --out /tmp/a.jsonl
venv/bin/python harness/triage.py results/<date> --base results/<green date>
```

## Bisecting by hand

The automatic bisect probes the last green commit first. A finding that
reproduces there is reported as "not a regression in this range": that
happens for a rotation cell that did not run on the green night. The
bisect then halves the range. Each probe rebuilds Python and only the
executables the findings name, and re-runs only their decks or sweep cases.
An unbuildable commit is skipped, and the search stops at a 100-minute
budget. To re-run it:

```
venv/bin/python harness/nightly.py bisect results/<red date> [--base results/<green date>]
```

To do it with git, check out a commit in `mw/` and run `make build
PYTHON=../venv/bin/python`. Then rebuild an executable with `cmake --build
exes/<basis>-<variant> -j6` and run `audit.py ... --only only.txt`, where
`only.txt` has `"<exe label>\t<deck label>"` lines. Run `triage.py` on that
directory against the green night. The executables' `-version` lines and
each night's commit are in `night.json`.
