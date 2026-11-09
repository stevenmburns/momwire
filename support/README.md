# Support hierarchy

`hierarchy.toml` orders the support levels and names the CI jobs that certify
each; `src/momwire/_features.py` gives each opt-in feature its reach (the
deepest level it is certified at). The header of `hierarchy.toml` has the
details.

## How a pull request's required checks are chosen

Branch protection on `main` names one check, `required`
(`.github/workflows/required.yml`). It runs `scripts/support_matrix.py select`
on the PR's diff (`base...head`), then waits for exactly the jobs that
selected, and passes only when every one of them concluded `success`. A
selected job that was skipped, cancelled or never reported fails it.

The rules, in order:

1. **Always:** `ci.yml:lint`, `ci.yml:test` (the Linux default lane, which runs
   the stable-path tripwire) and `ci.yml:test-pynec`.
2. **A feature the PR touches** (a changed file matches its `paths`, or the PR
   adds the feature or edits its registry entry): every job of levels
   1..reach that runs on this PR. A stable feature (reach = the deepest
   level) needs every job that runs.
3. **Stable path** (any changed file no feature claims): the jobs that would
   run for those files alone, i.e. today's path filters. The script evaluates
   the workflows themselves (`on.pull_request.paths`, the `changes` jobs'
   `dorny/paths-filter` rules, each job's `if:` and `needs:`), so a C++ change
   still selects the standalone lanes and `test-macos`, and a filter edited in
   a workflow is followed without a second copy here.
4. **Never** a job that does not run on a pull request (the slow, crossgate,
   memgate and integration lanes, wheels, the drop-in, releases). The nightly
   and main's push runs certify those. `not_required_level_jobs` in the
   output lists the level jobs a touched feature could not be held to.
5. **Never** a job listed under `[advisory]` in `hierarchy.toml`: it runs on
   the PR and shows its result, but is not waited on. Today that is
   `ci.yml:consumer-contract`.

To see what a branch would need: `python scripts/support_matrix.py select
--base origin/main` (the working tree) or `--head <sha>`.

After re-running a failed job, re-run `required` too: it gave its verdict
when that job failed.

## Adding a feature's paths

Give the `Feature` entry a `paths=` tuple of globs from the repo root (`**`
crosses directories, `*` does not). List only files that are the feature's
alone: its own module, its own C++ translation unit, its own tests. A file it
shares with stable code (`bspline.py`, `razor.py`, a C++ core the executables
also build) stays out, so a change to it is a stable-path change and keeps
the stable gates. `tests/test_support_matrix.py` fails if a glob matches no
file.

## Re-ranking

Edit `hierarchy.toml` (and `LEVELS` in `_features.py`, which
`tests/test_features_registry.py` holds equal). Moving a level or a job
changes which checks a feature PR needs from the next PR on, with no
change to branch protection.
