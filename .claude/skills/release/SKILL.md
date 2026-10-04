---
name: release
description: Cut a momwire release — version bump, tag, wheels/PyPI verification, and the antennaknobs follow-up checklist
---

# Cut a momwire release

Release flow for momwire (vX.Y.Z tags → wheels workflow → PyPI via Trusted
Publishing). Every step below was learned the hard way; do them in order.

## Preconditions

1. On `main`, clean tree, up to date with origin (`git fetch && git status`).
2. Latest main CI green (`gh run list --branch=main --limit 2`) — never tag a
   commit whose CI hasn't finished.
3. **The pre-tag sweep is clean** (added at 0.63.0, after #1189's 7 GiB
   transient reached a release candidate with no per-PR gate touching it).
   Run antennaknobs' `scratch/pretag-sweep/` in its `fast` mode on the
   Skylake box against the stored previous-release baseline, about 12 min
   (the harness README has the commands). Every OUTCOME, MEM, TIME or Z flag
   is attributed to a PR that says so, or fixed, before the tag. The run's
   results file becomes the next release's baseline.
4. **The default lane's cost is checked by MEDIAN, and moved before the
   tag** (added 2026-10-03, when CI wall time stopped gating anything —
   runners spread ~3x on the same code). Run
   `python scripts/test_durations.py` (needs `gh`; it pulls the per-test
   `durations-*` artifacts of the last 25 green main runs of `ci.yml` and
   `wheels.yml`). Every default-lane test it lists as a slow-lane candidate
   — median over the 5 s budget for some job/OS — moves to the `slow` lane
   (or is made faster) in **its own PR**, merged before the bump PR. A
   single run's long time is not a candidate; a median is. This step is
   hygiene only: speed regressions are the sweep above's job (and, for
   Windows, paired runs on the Windows box), never these numbers.
5. **Record the workflow-efficiency numbers.** Run
   `python scripts/ci_metrics.py --days <days since the last release>` and
   compare with the previous release's numbers (put them in the bump PR's
   body so the next release has them). Baseline, 2026-10-03, 30 days: 257
   merged PRs; lead time median 0.2 h, p90 2.7 h; CI+wheels runs per PR
   median 1, p90 3; 4 HARD-ceiling failures of 31 failed runs; default-lane
   `test` job median 11.8 min. Since the ceiling stopped failing, its count
   is zero by construction and measures nothing new — the guardrails that
   matter after the change are the default-lane `test` job time (hygiene;
   it should not creep while long tests are allowed to land) and the pre-tag
   sweep's flags (performance).

## Steps

1. **Bump the version BEFORE tagging.** Edit `version = "X.Y.Z"` in
   `pyproject.toml`. The wheels build reads it at build time — tagging first
   mislabels every wheel.
2. Commit on a branch as
   `chore: bump version to X.Y.Z (<one-line theme>)`, open a PR, and
   rebase-merge it once CI is green. No direct pushes to main, not even for
   the bump. Do NOT add a CI-skip marker (the tag build must run).
3. Rebase-merge rewrites the SHA, so tag the bump commit **as it landed on
   main**, never the branch commit:
   ```bash
   git checkout main && git pull --ff-only
   git log -1 --format=%s   # must be the bump commit
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```
4. The tag triggers the `wheels` workflow: it builds all wheels, publishes to
   PyPI, **and auto-creates the GitHub release with generated notes**. Do NOT
   run `gh release create` — it already exists once the workflow finishes
   (PR titles become the release notes, so they were written accordingly).
5. Watch it to completion (~8–9 min): `gh run watch <run-id> --exit-status`.
6. Verify PyPI actually serves it:
   ```bash
   curl -s https://pypi.org/pypi/momwire/json | python3 -c "import json,sys; print(json.load(sys.stdin)['info']['version'])"
   ```

## The signed drop-in (momwire#711)

The same tag also triggers `eznec-dropin`, which Authenticode-signs the
frozen engine and the compiled launcher `momwire-eznec.exe`, then copies
the launcher once per name in `launcher_stems()` in
`scripts/eznec_freeze/build.py` (today seven: `momwire-nec5[-razor-2p]`,
`momwire-nec4[-sinusoidal]`, and the deprecated `momwire-eznec-razor-2p` /
`-razor-nec5` beside `momwire-eznec` itself) with Azure Artifact Signing
before zipping them. Non-tag builds sign with a throwaway self-signed cert,
so a tag is the ONLY time the real certificate is exercised.

**Before tagging**, confirm the credential has not expired — an expired
`AZURE_CLIENT_SECRET` fails only on tags, i.e. mid-release:

```bash
gh secret list --repo stevenmburns/momwire   # AZURE_CLIENT_SECRET present
```

Presence is not freshness; if the release is near the secret's expiry date,
rotate it first. Prove the whole path without cutting anything:

```bash
gh workflow run eznec-dropin.yml --ref main -f azure_sign=true
```

That signs with the real certificate and uploads a `-SIGNTEST` bundle. No
tag, no GitHub release, repeatable.

**After the tag build**, the freeze job's log must contain both of:

```
chain verified
signature present on all N executables
```

where N is `len(SHIPPED_VARIANTS) + 2` — the engine, the default launcher,
and one copy per shipped variant — so **4** as of v0.48.0. The number moves
whenever a launcher variant is added (#817 took it from 2 to 4); a count
below the current one means a copy arrived unsigned and the build already
failed on it, so read the line for the count you expect rather than the
one written here.

`chain verified` is the hard gate — it means the signature validates to a
trusted root, not that the check was waived. The waiver
(`MOMWIRE_SIGN_ALLOW_UNTRUSTED`) is exported only by the self-signed
rehearsal step, which does not run on tags.

**If signing fails, suspect configuration, not code.** The two live failure
modes both surface as `403` plus `SignerSign() failed`, naming neither:

- the metadata `Endpoint` region not matching where the account and profile
  live (`wus2` / account `momwire` / profile `momwire-public-trust`);
- the service principal missing the **Artifact Signing Certificate Profile
  Signer** role.

Do not go reading `sign.py` first. Full setup and secret-rotation procedure:
`docs/code-signing.md`.

Leaf certificates are valid for THREE DAYS by design, so a fresh one is
issued per release and the RFC-3161 timestamp is what keeps shipped
signatures valid. A timestamping outage is therefore a release blocker, not
a warning.

## The antennaknobs follow-up (do not skip)

antennaknobs consumes momwire via an EXACT pin (`momwire==X.Y.Z`), so every
momwire release that antennaknobs should adopt needs a deliberate antennaknobs
PR touching **three** places in one commit:

1. `pyproject.toml` → `momwire==X.Y.Z`
2. `Dockerfile` → `pip install "momwire==X.Y.Z"`
3. The `momwire` **git submodule pointer** →
   `git -C momwire fetch && git -C momwire checkout <bump-commit>` then
   `git add momwire`

The submodule is the one everyone forgets (missed in antennaknobs PR #268,
fixed in #270). CI will NOT catch it (`test.yml` updates the submodule with
`--remote`); the symptom is silent — a fresh dev clone builds editable momwire
from the stale submodule, then `pip install -e ".[test]"` quietly replaces it
with the PyPI wheel and edits under `momwire/` stop taking effect. Verify with
`git -C momwire log --oneline -1` == the pinned release's bump commit.

Also in that PR, per the default-cost audit discipline: if the release changes
solver behavior reachable from an unqualified request, latency-smoke the
default path before merging, and keep expensive models opt-in.

PyPI publishes ~9 min after the tag; if the antennaknobs PR's wheel-smoke job
races the publish, wait and re-run it rather than merging on red.
