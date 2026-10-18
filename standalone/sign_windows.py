"""Authenticode-sign the Windows standalone executables (the preview release asset).

    python standalone/sign_windows.py EXE...

The signing itself is scripts/eznec_freeze/sign.py's, unchanged: signtool
found the same way, one invocation, the timestamp asserted on every path, and
the chain checked unless MOMWIRE_SIGN_ALLOW_UNTRUSTED (the self-signed
rehearsal) waives it. The workflow sets the identity exactly as
eznec-dropin.yml does, by one signing mode: the real Azure certificate on a
tag or an azure_sign dispatch, a self-signed rehearsal on every other run.

Unlike the freeze's `sign_if_configured`, there is no unsigned route here:
this script runs only in CI, where a missing identity is a broken workflow,
so it fails instead of skipping. Afterwards every file must carry an
attached signature (sign.has_signature), whatever signtool reported.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "eznec_freeze_sign", HERE.parent / "scripts" / "eznec_freeze" / "sign.py"
)
sign = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sign)


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv]
    if not paths:
        print("usage: sign_windows.py EXE...", file=sys.stderr)
        return 2
    if not (
        os.environ.get("MOMWIRE_SIGN_METADATA") or os.environ.get("MOMWIRE_SIGN_SHA1")
    ):
        print(
            "no signing identity (neither MOMWIRE_SIGN_METADATA nor "
            "MOMWIRE_SIGN_SHA1 is set): the workflow's signing mode did not run",
            file=sys.stderr,
        )
        return 1
    sign.sign(paths)
    unsigned = [p for p in paths if not sign.has_signature(p)]
    if unsigned:
        print(f"no signature attached: {unsigned}", file=sys.stderr)
        return 1
    for p in paths:
        print(f"signed {p} ({sign.certificate_table_size(p)} bytes of certificate)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
