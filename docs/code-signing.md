# Code signing the EZNEC drop-in

How `momwire-eznec.exe` gets an Authenticode signature, how to set the Azure
side up from nothing, and how to rotate the credential before it strands a
release. Landed in momwire#711 on 2026-08-29.

The *mechanism* is documented in the code it lives in — `scripts/eznec_freeze/sign.py`,
`build.py`, and `.github/workflows/eznec-dropin.yml` all carry their reasoning
in comments. This file covers what the repo cannot: the Azure resources behind
those variables, and the portal procedure to recreate or repair them.

For the release-day checklist, see the `release` skill instead. This is the
long-tail document you open perhaps twice a decade.

## What signs, and when

The workflow decides `MOMWIRE_SIGN_MODE` (`azure` | `rehearsal`) **exactly
once**, in its own step; every signing step, the credential scoping, the
bundle name, and `build.py`'s own signed-or-fail post-condition key off that
single value (the momwire#711 retro-review retired the five independently
edited predicates this replaces).

| Trigger | Mode | Chain check | Bundle name | Release attach |
| --- | --- | --- | --- | --- |
| pushed tag `v*` | azure | hard gate | `momwire-eznec-windows.zip` | yes |
| dispatch, `azure_sign=true` | azure | hard gate | `…-SIGNTEST.zip` | no |
| dispatch **on a tag ref** | azure | hard gate | `…-SIGNTEST.zip` | **no** — the release job requires a push event, so a signing test can never replace a published asset |
| anything else | rehearsal | waived | `…-SELFSIGNED-rehearsal.zip` | no |

A tag build **hard-requires working Azure signing** — there is no unsigned
fallback, deliberately: the failure mode (expired client secret, Azure or TSA
outage) is a red freeze job and a release without the drop-in zip, which is
why the rotation ritual below ends with a proving dispatch.

The rehearsal exists so the signing path is exercised on every push rather
than once per release, and it timestamps against the same Microsoft TSA the
release path uses — a red canary on that endpoint is news about the next
release. It must never reach a tag: a self-signed signature on a public
download is worse than none, because users get "Unknown Publisher" either way
and an untrusted root reads as tampering to some AV heuristics.

Signing happens **before** the `SHIPPED_VARIANTS` copy loop in `build.py`.
Authenticode covers PE contents and not filenames, so one signing call ships
both executables signed; `build.py` then asserts the copies carry a signature
rather than trusting the ordering.

**Scope.** In one-dir mode this signs the *launcher*. The Python payload in
`_internal/` is outside the signed bytes, so the signature attests who shipped
the stub, not that the engine beside it is untouched. One-file would cover
both and is disqualified on the measured 17 s vs 1.3 s per-launch cost.

## The Azure resources

| Thing | Value |
| --- | --- |
| Artifact Signing account | `momwire` |
| Certificate profile | `momwire-public-trust` (Public Trust) |
| Region / endpoint | West US 2 → `https://wus2.codesigning.azure.net` |
| Identity validation | Individual, subject `Steve Burns` |
| Service principal | `momwire-ci-signing` |
| Role on the account | Artifact Signing Certificate Profile Signer |
| GitHub secrets | `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET` |

The service was renamed from **Trusted Signing** to **Artifact Signing**, but
the resource provider is still `Microsoft.CodeSigning` and the dlib is still
`Azure.CodeSigning.Dlib.dll`. If a portal search for "Artifact Signing" comes
up empty — RBAC role names especially — try "Trusted Signing".

## Rotating the client secret

**This is the procedure you will actually need.** `AZURE_CLIENT_SECRET`
expires on whatever date was chosen at creation. Because signing runs only on
tags, expiry announces itself mid-release.

1. Portal → **Microsoft Entra ID** → **App registrations** → `momwire-ci-signing`
2. **Certificates & secrets** → **Client secrets** → **New client secret**
3. Copy the **Value** column, not **Secret ID**. Secret ID is a GUID and looks
   like the other two credentials, which is why this is the usual mistake. The
   Value is masked permanently once you navigate away.
4. Update the `AZURE_CLIENT_SECRET` repository secret in momwire.
5. Prove it before relying on it:
   `gh workflow run eznec-dropin.yml --ref main -f azure_sign=true`
6. Delete the superseded secret in Azure once the run is green.

Do not route the value through a shell — `gh secret set --body` puts it in
history. Either paste it into the GitHub web form, or use
`gh secret set AZURE_CLIENT_SECRET --repo stevenmburns/momwire`, which prompts
and reads hidden.

## First-time setup, from nothing

Only needed if the account is lost, or someone else takes this over.

1. **A paid Azure subscription.** Free, trial, and sponsored subscriptions are
   explicitly unsupported; account creation fails outright on them. A personal
   Microsoft account with no Entra directory cannot even open the portal — it
   errors with *"does not exist in tenant 'Microsoft Services'"*, because the
   portal authenticates against the `/organizations` endpoint. Signing up via
   `azure.microsoft.com` provisions the directory; going straight to
   `portal.azure.com` does not.
2. **Fix the billing profile first.** For individual validation the identity
   form is populated read-only from the Azure billing account, and those values
   land on the certificate. Set the legal name and address to match the
   government ID *before* submitting anything — correcting it afterwards
   requires a brand-new validation request. The portal refuses some edits
   (letter casing, removing hyphens); those need
   `az billing account update --sold-to …`.
3. **Register the provider**: `az provider register --namespace "Microsoft.CodeSigning"`.
4. **Create the Artifact Signing account** in a supported region, Basic SKU
   (5,000 signatures/month against two per release). Resources **cannot** be
   migrated across subscriptions, tenants, or resource groups — getting this
   one right is what avoids rebuilding everything including validation.
5. **Assign yourself Artifact Signing Identity Verifier.** Without it the
   **New identity** button is greyed out in a way indistinguishable from being
   ineligible. Subscription Owner does not imply it.
6. **Identity validation** — portal only, the CLI cannot do this. Note the
   selector defaults to **Organization** on *every* visit, and an Individual
   request is invisible while it does; the list reads as empty rather than
   filtered, which looks exactly like a request that was never created.
7. **Certificate profile**, type Public Trust. Leave *Include street address*
   and *Include postal code* unchecked — for an individual that is a home
   address, published in every shipped binary. The CN/O dropdown being empty
   means validation has not completed.
8. **Service principal** with **Certificate Profile Signer** on the account,
   and nothing else. Those credentials live in GitHub secrets and should be
   able to sign builds and do nothing more; Identity Verifier on the same
   principal would let a leaked token interfere with the validation the whole
   trust story rests on.

Individual onboarding was reported paused from April 2025 in several Microsoft
Q&A threads while the quickstart still documented the individual path. On this
account, Individual was selectable on 2026-08-29. If a future attempt finds it
genuinely closed, the fallback is a commercial CA — SSL.com, DigiCert
KeyLocker, or an OV certificate on a hardware token.

## Renewing identity validation

Identity validation expires, and reminders start 60 days out. If it lapses,
certificate renewal stops and signing stops with it. Renewal is a new
validation request associated with the existing certificate profiles — the
same AU10TIX flow, needing a phone, Microsoft Authenticator, and a government
photo ID showing an address.

The verification link expires after **seven days** and cannot be resent on the
same request; missing it means starting over. Sign in to that link with the
same email address that is on the request, or it errors with "You don't have
permission to access this page".

## Failure modes

**`403` plus `SignerSign() failed`.** Configuration, not code, and the error
names neither cause. Either the metadata `Endpoint` region does not match
where the account and profile live, or the service principal lacks the
Certificate Profile Signer role. Check both before reading any Python.

**`chain NOT verified` on a tag build.** Should be impossible — the waiver is
exported only by the rehearsal step, which does not run on tags. If it appears,
the step conditions have been edited wrongly.

**The install step hangs.** Do not reintroduce `ArtifactSigningClientTools.msi`.
`msiexec /quiet` hung for nine minutes on `windows-latest` with no output and
no failure (run 33229631979); it wants an interactive installer context a
hosted runner does not provide. The dlib comes from the
`Microsoft.ArtifactSigning.Client` NuGet package — **version-pinned** in the
workflow, because the install runs at tag time on the release critical path;
bump the pin deliberately and re-prove with an `azure_sign` dispatch — and
.NET 8 from `actions/setup-dotnet`, with a five-minute step timeout so the
next thing that stalls fails fast.

**Signatures that were valid go invalid within days.** The timestamp is
missing. Artifact Signing leaf certificates are valid for **three days** by
design; RFC-3161 timestamping is what makes a signature outlive the cert, and
`sign.py` treats it as mandatory rather than optional for that reason.

## Smart App Control

SAC is the thing this signature genuinely unlocks, as opposed to SmartScreen,
which it does not. SAC trusts an app when Microsoft's cloud can confidently
classify it as safe **or** when it is correctly signed by a CA in the Microsoft
Trusted Root Program; unsigned or invalidly signed binaries are blocked
outright. A valid signature lets an app run even before cloud reputation
exists, which is exactly the position a low-volume release is in.

Our chain satisfies the stated criteria, verified on a real Windows 11 client
(build 26200) against the run-33230110713 bundle:

| Criterion | Observed |
| --- | --- |
| Signature valid on Win11 client | `Status: Valid`, "Signature verified" |
| CA in Microsoft Trusted Root Program | chains to Microsoft Identity Verification Root CA 2020 |
| **RSA, not ECC** | RSA, 3072-bit |
| Timestamped | Microsoft Public RSA Time Stamping Authority |

**The RSA requirement is the trap.** SAC's signature check does not support
ECC, so an otherwise perfectly valid ECC signature is silently untrusted.
Artifact Signing issues RSA today; if that ever changes, or a different
certificate profile is used, this is the thing to re-check first.

### Why this is not covered by CI

`windows-latest` is **Windows Server** (windows-2025-vs2026 at time of
writing). Smart App Control is a Windows 11 *client* feature and does not
exist on Server, so no CI run can exercise it — this is architectural, not a
gap to be closed by a better workflow. GitHub offers no Windows 11 client
runners.

A developer machine usually cannot test it either. SAC only initialises on a
**clean Windows 11 install**, starts in Evaluation mode, and Microsoft's
service decides whether to move it to enforcement. Turning it off is one-way:
re-enabling requires reinstalling Windows.

Check the state before drawing any conclusion from a machine:

```powershell
(Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy' `
  -Name VerifiedAndReputablePolicyState).VerifiedAndReputablePolicyState
# 0 = off   1 = enforcement   2 = evaluation
```

So the honest standing claim is that **the published criteria are verified,
not that enforcement was observed**. Empirical confirmation needs a throwaway
Windows 11 VM from a fresh ISO, with that registry value reading 1 or 2 before
the test means anything.

## What signing does not fix

Artifact Signing issues **no EV certificates**, and Microsoft states there is
no plan to. SmartScreen reputation accrues per file hash, so a 60 MB niche
download builds it slowly and each release partly restarts the climb. This
buys provenance, tamper-evidence, and enterprise WDAC/AppLocker acceptance —
not a clean download dialog on day one. Only an EV certificate from a
commercial CA does that.

The ~99 numpy and scipy `.pyd` files in the bundle ship unsigned. That is
normal for PyPI wheels and matters only under enterprise policy.
