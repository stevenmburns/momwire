"""The `-version` line a C++ standalone executable answers (momwire#1831).

It is its Python launcher's line (`momwire_serve_client.version_line`,
momwire#1606) with one difference, in NAME: the launcher's name, then
``-standalone`` and the C++ core's variant, so a user can tell the two apart::

    NEC5 momwire-nec5-razor-2p [sb] 0.87.1                  the launcher
    NEC5 momwire-nec5-razor-2p-standalone-avx2 [sb] 0.87.1  the executable

The variant is the one compiled in (``MW_EXE_VARIANT``): ``avx2`` or ``sse2``,
``legacy`` on macOS. Where ``MOMWIRE_FORCE_VARIANT`` names one, the lanes run
that build, and the line must say so.
"""

import os

VARIANTS = ("avx2", "sse2", "legacy")


def assert_standalone_line(exe_stdout: bytes, launcher_stdout: bytes) -> str:
    """The executable's line is the launcher's with NAME suffixed; returns the
    variant it names."""
    exe = exe_stdout.decode("ascii").splitlines()
    ref = launcher_stdout.decode("ascii").splitlines()
    assert len(exe) == len(ref) == 1, (exe_stdout, launcher_stdout)
    exe_fields, ref_fields = exe[0].split(" "), ref[0].split(" ")
    assert len(exe_fields) == len(ref_fields) == 4, (exe, ref)
    dialect, name, modes, version = exe_fields
    assert (dialect, modes, version) == (
        ref_fields[0],
        ref_fields[2],
        ref_fields[3],
    ), (exe, ref)
    prefix = f"{ref_fields[1]}-standalone-"
    assert name.startswith(prefix), (exe, ref)
    variant = name[len(prefix) :]
    assert variant in VARIANTS, (exe, ref)
    forced = os.environ.get("MOMWIRE_FORCE_VARIANT")
    if forced in ("avx2", "sse2"):
        assert variant == forced, (exe, forced)
    return variant
