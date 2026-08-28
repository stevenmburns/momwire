"""Authenticode-sign the frozen EZNEC drop-in.

Signing is OPT-IN and driven entirely by the environment, so an unconfigured
build behaves exactly as it always has — unsigned, no signtool required, and
runnable on a box (or a Linux runner) that has no Windows SDK at all:

``MOMWIRE_SIGN_SHA1``
    SHA-1 thumbprint of a code-signing certificate in the Windows certificate
    store.  UNSET IS THE OFF SWITCH: no thumbprint, no signing.
``MOMWIRE_SIGN_TIMESTAMP_URL``
    RFC-3161 timestamp authority.  Defaults to DigiCert's.
``MOMWIRE_SIGN_ALLOW_UNTRUSTED``
    ``1`` downgrades the ``signtool verify /pa`` chain check from a gate to a
    printed note.  This is for the SELF-SIGNED REHEARSAL only: a self-signed
    certificate cannot chain to a trusted root, so /pa always fails, and
    without this the rehearsal could never pass.  Leave it unset for a real
    certificate — then a broken chain correctly fails the build.
``SIGNTOOL``
    Explicit path to ``signtool.exe``, skipping the Windows Kits search.

Why the thumbprint seam and not a .pfx: since June 2023 the CA/Browser Forum
baseline requires code-signing keys to live on FIPS-140-2 Level 2 hardware,
so there is no importable key file to point at any more.  A store thumbprint
is what both a hardware token and a self-signed rehearsal cert present, which
is why the rehearsal exercises the real code path rather than a stand-in.
A cloud-HSM CA (Azure Trusted Signing, DigiCert KeyLocker) identifies the key
differently — ``/dlib`` plus a metadata file instead of ``/sha1`` — and that
is the ONLY part that changes: swap `_identity_args()` and the rest holds.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

DEFAULT_TIMESTAMP_URL = "http://timestamp.digicert.com"

# Newest first; the Kits layout gained the versioned `bin\<sdk>\` level in the
# Windows 10 SDK, and the flat `bin\x64\` form is the older fallback.
_KITS_GLOBS = (
    r"C:\Program Files (x86)\Windows Kits\10\bin\*\x64\signtool.exe",
    r"C:\Program Files\Windows Kits\10\bin\*\x64\signtool.exe",
    r"C:\Program Files (x86)\Windows Kits\10\bin\x64\signtool.exe",
)


class SigningError(RuntimeError):
    """Signing was asked for and could not be delivered."""


def _version_key(path: Path) -> tuple[int, ...]:
    """Sort key for a Kits path, so `10.0.26100.0` beats `10.0.19041.0`."""
    for part in path.parts:
        if part.replace(".", "").isdigit() and "." in part:
            return tuple(int(n) for n in part.split("."))
    return (0,)


def find_signtool() -> Path:
    """Locate ``signtool.exe``, or explain what to install."""
    explicit = os.environ.get("SIGNTOOL")
    if explicit:
        if not Path(explicit).is_file():
            raise SigningError(f"SIGNTOOL points at no file: {explicit}")
        return Path(explicit)

    on_path = shutil.which("signtool")
    if on_path:
        return Path(on_path)

    found: list[Path] = []
    for pattern in _KITS_GLOBS:
        root, _, tail = pattern.partition("*")
        base = Path(root)
        if not tail:
            if base.is_file():
                found.append(base)
            continue
        if base.is_dir():
            found.extend(base.glob("*" + tail))
    if found:
        return max(found, key=_version_key)

    raise SigningError(
        "signtool.exe not found. It ships with the Windows SDK component "
        '"Signing Tools for Desktop Apps"; set SIGNTOOL=<path> to override. '
        "(It is preinstalled on GitHub's windows-latest runners.)"
    )


def certificate_table_size(path: Path) -> int:
    """Bytes in the PE Attribute Certificate Table — 0 when unsigned.

    Parsed here rather than shelled out to ``signtool verify`` because this
    answers the one question a build can always assert: did a signature
    actually get ATTACHED?  ``verify`` additionally judges the trust chain,
    which a self-signed rehearsal cert must fail by construction.
    """
    data = path.read_bytes()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe : pe + 4] != b"PE\0\0":
        raise SigningError(f"not a PE image: {path}")
    magic = struct.unpack_from("<H", data, pe + 24)[0]
    # Optional header: 24 bytes of standard fields, then 112 (PE32+) or 96
    # (PE32) more before the data directories begin.
    ddir = pe + 24 + (112 if magic == 0x20B else 96)
    n_dirs = struct.unpack_from("<I", data, ddir - 4)[0]
    if n_dirs < 5:  # index 4 is the certificate table; absent entirely
        return 0
    _rva, size = struct.unpack_from("<II", data, ddir + 4 * 8)
    return size


def has_signature(path: Path) -> bool:
    return certificate_table_size(path) > 0


def _identity_args() -> list[str]:
    """How signtool should find the private key.

    The one CA-specific stanza.  A cloud-HSM provider replaces this with
    ``["/dlib", <provider dll>, "/dmdf", <metadata json>]``.
    """
    return ["/sha1", os.environ["MOMWIRE_SIGN_SHA1"]]


def sign(paths: list[Path]) -> None:
    """Sign ``paths`` in one signtool invocation, then assert the result."""
    if sys.platform != "win32":
        raise SigningError(
            "Authenticode signing was requested (MOMWIRE_SIGN_SHA1 is set) "
            f"but this is {sys.platform}, not Windows."
        )

    signtool = find_signtool()
    timestamp = os.environ.get("MOMWIRE_SIGN_TIMESTAMP_URL", DEFAULT_TIMESTAMP_URL)
    thumbprint = os.environ["MOMWIRE_SIGN_SHA1"]
    cmd = [
        str(signtool),
        "sign",
        "/fd",
        "SHA256",
        # Timestamping is not optional: without it every signature already in
        # the wild goes invalid the day the certificate expires.
        "/tr",
        timestamp,
        "/td",
        "SHA256",
        "/v",
        *_identity_args(),
        *(str(p) for p in paths),
    ]
    printable = ["<thumbprint>" if a == thumbprint else a for a in cmd]
    print("+", " ".join(printable), flush=True)
    if subprocess.run(cmd).returncode != 0:
        raise SigningError("signtool sign failed")

    for path in paths:
        if not has_signature(path):
            raise SigningError(
                f"signtool reported success but {path} carries no signature"
            )
        print(f"signed: {path.name} ({certificate_table_size(path)} cert bytes)")

    _verify(signtool, paths)


def _verify(signtool: Path, paths: list[Path]) -> None:
    """``signtool verify /pa`` — a gate, unless the rehearsal waives it."""
    lenient = os.environ.get("MOMWIRE_SIGN_ALLOW_UNTRUSTED") == "1"
    cmd = [str(signtool), "verify", "/pa", "/v", *(str(p) for p in paths)]
    print("+", " ".join(cmd), flush=True)
    if subprocess.run(cmd).returncode == 0:
        print("chain verified")
        return
    if lenient:
        print(
            "chain NOT verified — expected, MOMWIRE_SIGN_ALLOW_UNTRUSTED=1 "
            "(a self-signed rehearsal certificate has no trusted root). "
            "The signature itself is attached and was asserted above."
        )
        return
    raise SigningError(
        "signtool verify /pa failed. For a self-signed rehearsal set "
        "MOMWIRE_SIGN_ALLOW_UNTRUSTED=1; for a real certificate this is a "
        "genuine chain failure."
    )


def sign_if_configured(paths: list[Path]) -> bool:
    """Sign when a thumbprint is configured; otherwise say so and skip."""
    if not os.environ.get("MOMWIRE_SIGN_SHA1"):
        print("unsigned build (MOMWIRE_SIGN_SHA1 unset)")
        return False
    sign(paths)
    return True
