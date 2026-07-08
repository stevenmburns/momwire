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

## Steps

1. **Bump the version BEFORE tagging.** Edit `version = "X.Y.Z"` in
   `pyproject.toml`. The wheels build reads it at build time — tagging first
   mislabels every wheel.
2. Commit directly to main as
   `chore: bump version to X.Y.Z (<one-line theme>)` and push. This is the
   one sanctioned direct-to-main push; do NOT add a CI-skip marker (the tag
   build must run).
3. Tag the bump commit and push the tag:
   ```bash
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
