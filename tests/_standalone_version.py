"""The `-version` line a C++ standalone executable answers (momwire#1831).

It is its Python launcher's line (`momwire_serve_client.version_line`,
momwire#1606) with one difference, in NAME: the launcher's name, then
``-standalone`` and the C++ core's variant, so a user can tell the two apart::

    NEC5 momwire-nec5-razor-2p [sb] 0.87.1                  the launcher
    NEC5 momwire-nec5-razor-2p-standalone-avx2 [sb] 0.87.1  the executable

The variant is the one compiled in (``MW_EXE_VARIANT``): ``avx2`` or ``sse2``,
``legacy`` on macOS. Where ``MOMWIRE_FORCE_VARIANT`` names one, the lanes run
that build, and the line must say so.

A MinGW-w64 build (windows-static.yml, the single-file experiment) names its
toolchain after the variant, ``...-standalone-avx2-ucrt64``, so a printout's
``CM version`` card says which of two Windows builds answered; the MSVC build
names none. ``MW_EXE_EXPECT_TOOLCHAIN`` (the workflow's gates) names the one
the executable under test must carry, ``none`` for none.
"""

import os

VARIANTS = ("avx2", "sse2", "legacy")
TOOLCHAINS = ("ucrt64", "mingw64", "clang64")


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
    variant, _, toolchain = name[len(prefix) :].partition("-")
    assert variant in VARIANTS, (exe, ref)
    assert toolchain in ("", *TOOLCHAINS), (exe, ref)
    expect = os.environ.get("MW_EXE_EXPECT_TOOLCHAIN")
    if expect:
        assert toolchain == ("" if expect == "none" else expect), (exe, expect)
    forced = os.environ.get("MOMWIRE_FORCE_VARIANT")
    if forced in ("avx2", "sse2"):
        assert variant == forced, (exe, forced)
    return variant
