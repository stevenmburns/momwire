"""The timestamp assertion in the drop-in's signing step (momwire#755).

Trusted Signing issues very short-lived certificates — v0.44.0's leaf was
valid for 72 hours — so the RFC-3161 countersignature is the only thing that
keeps a shipped signature valid past the end of the week. The failure this
gates is DELAYED and INVISIBLE: a build that stopped timestamping signs
successfully, prints ``chain verified``, passes every check on the build day,
and starts failing Smart App Control about three days later on end-user
machines, with nothing in the build log naming the cause.

These run on Linux by faking `signtool`, which is the only way to reach this
logic off Windows. What they pin is the DECISION, which is where the bug was:
one exit code was carrying both verdicts, so the waiver written for the chain
silently covered the timestamp too.

The fake's OUTPUT shape is taken from a real canary run (33395335907) rather
than imagined, and that matters — the first attempt at this check asked
PowerShell's `Get-AuthenticodeSignature` for `TimeStamperCertificate` and was
wrong, because PowerShell leaves that field empty when the signing chain is
untrusted. It called correctly-stamped rehearsal binaries unstamped. signtool
prints its timestamp line either way, which is the property this relies on.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "scripts" / "eznec_freeze")
)

import sign as _sign


# Two files, and signtool prints the timestamp line once per file it stamped.
_STAMPED = "The signature is timestamped: Mon Aug 31 13:11:03 2026\n"
_CHAIN_ERR = "SignTool Error: A certificate chain processed, but terminated...\n"


class _Fake:
    """Stand-in for `subprocess.run`, shaped like signtool's real output.

    The shape matters and is taken from a real canary run (33395335907): a
    self-signed rehearsal build prints the timestamp line for EVERY binary and
    THEN fails the chain. That combination — stamped but untrusted — is the
    case the waiver has to forgive without forgiving the timestamp.
    """

    def __init__(self, *, chain_ok: bool, stamped: int):
        self.chain_ok = chain_ok
        self.stamped = stamped
        self.calls = 0

    def __call__(self, cmd, *args, **kwargs):
        self.calls += 1
        out = _STAMPED * self.stamped + ("" if self.chain_ok else _CHAIN_ERR)
        return subprocess.CompletedProcess(
            cmd, 0 if self.chain_ok else 1, stdout=out, stderr=""
        )


@pytest.fixture
def faked(monkeypatch):
    def install(*, chain_ok, stamped):
        fake = _Fake(chain_ok=chain_ok, stamped=stamped)
        monkeypatch.setattr(_sign.subprocess, "run", fake)
        return fake

    return install


PATHS = [Path("momwire-eznec.exe"), Path("momwire-eznec-engine.exe")]


def test_a_signed_and_timestamped_build_passes(faked, monkeypatch):
    monkeypatch.delenv("MOMWIRE_SIGN_ALLOW_UNTRUSTED", raising=False)
    fake = faked(chain_ok=True, stamped=2)
    _sign._verify(Path("signtool.exe"), PATHS)
    assert fake.calls == 1, "both verdicts should come from ONE signtool run"


def test_a_missing_timestamp_fails_the_build(faked, monkeypatch):
    """The straightforward case: real certificate, good chain, no countersignature."""
    monkeypatch.delenv("MOMWIRE_SIGN_ALLOW_UNTRUSTED", raising=False)
    faked(chain_ok=True, stamped=0)
    with pytest.raises(_sign.SigningError, match="WITHOUT an RFC-3161 timestamp"):
        _sign._verify(Path("signtool.exe"), PATHS)


def test_the_rehearsal_waiver_does_NOT_cover_the_timestamp(faked, monkeypatch):
    """momwire#755's whole point, and the reason the two checks are separate.

    The rehearsal certificate is self-signed, so the chain ALWAYS fails and
    the waiver has to forgive that. Before the split, one `signtool verify`
    answered both questions, so `/tw` — the obvious one-flag fix — would have
    been swallowed by the same waiver on exactly the path that exists to catch
    signing regressions early.
    """
    monkeypatch.setenv("MOMWIRE_SIGN_ALLOW_UNTRUSTED", "1")
    faked(chain_ok=False, stamped=0)
    with pytest.raises(_sign.SigningError, match="WITHOUT an RFC-3161 timestamp"):
        _sign._verify(Path("signtool.exe"), PATHS)


def test_the_waiver_still_forgives_the_chain_it_was_written_for(faked, monkeypatch):
    """The rehearsal must still pass when only the chain is untrusted —
    otherwise this gate would break the canary that runs on every push."""
    monkeypatch.setenv("MOMWIRE_SIGN_ALLOW_UNTRUSTED", "1")
    faked(chain_ok=False, stamped=2)
    _sign._verify(Path("signtool.exe"), PATHS)


def test_a_PARTIALLY_timestamped_build_fails(faked, monkeypatch):
    """One of two stamped is a failure, not a pass.

    build.py signs several binaries in one signtool invocation, so a per-file
    count is the only reading that catches "the second one missed"."""
    monkeypatch.delenv("MOMWIRE_SIGN_ALLOW_UNTRUSTED", raising=False)
    faked(chain_ok=True, stamped=1)
    with pytest.raises(_sign.SigningError, match="reported 1 timestamped"):
        _sign._verify(Path("signtool.exe"), PATHS)
