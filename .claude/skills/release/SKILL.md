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

## The signed drop-in (momwire#711)

The same tag also triggers `eznec-dropin`, which Authenticode-signs
`momwire-eznec.exe` and its `-razor-nec5` twin with Azure Artifact Signing
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
signature present on all 2 executables
```

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
