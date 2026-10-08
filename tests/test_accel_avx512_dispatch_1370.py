"""The AVX-512 variant's dispatch (momwire#1370).

`_avx512` is the AVX2 build plus AVX-512 F/CD/BW/DQ/VL. Importing it on a CPU
without all five (or under an OS that does not save the opmask/zmm state) is
momwire#1032's silent SIGILL one level up, so the property under test is the
same as there: it is NEVER imported unless every required feature was read
and found. The decision is asserted on stubs (no second variant may enter this
interpreter: see test_accel_cpu_guard_1032's loader note); the real loads run
in subprocesses at the bottom.

The runtime half (that the variant, once chosen, is loaded and agrees with
`_avx2`) needs an AVX-512 CPU and lives in test_avx512_variant_1370.py.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

import momwire
from momwire import _accel

REQUIRED = _accel._AVX512_REQUIRED

# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------


@pytest.fixture
def unforced(monkeypatch):
    monkeypatch.delenv(_accel._FORCE_VARIANT_ENV, raising=False)
    monkeypatch.delenv("MOMWIRE_REQUIRE_ACCEL", raising=False)


def test_the_preference_order_is_avx512_avx2_sse2_legacy(unforced):
    assert _accel._variants_to_try(True, True) == (
        _accel._AVX512,
        _accel._AVX2,
        _accel._SSE2,
        _accel._LEGACY,
    )
    assert _accel._ALL_VARIANTS == _accel._variants_to_try(True, True)


@pytest.mark.parametrize("avx512", [False, None])
def test_avx512_is_never_tried_unless_checked_present(unforced, avx512):
    """False is a CPU without it; None is one we could not read. Same answer."""
    chain = _accel._variants_to_try(True, avx512)
    assert chain == (_accel._AVX2, _accel._SSE2, _accel._LEGACY)


@pytest.mark.parametrize("verdict", [False, None])
@pytest.mark.parametrize("avx512", [True, False, None])
def test_no_avx2_verdict_means_no_avx512_either(unforced, verdict, avx512):
    """Two checked yeses, or the baseline: a contradictory pair (AVX-512 yes,
    AVX2 unknown) breaks toward the baseline like every other unknown."""
    chain = _accel._variants_to_try(verdict, avx512)
    assert chain == (_accel._SSE2, _accel._LEGACY)


def test_the_default_signature_never_selects_avx512(unforced):
    """A caller that predates the second check gets the pre-#1370 chain."""
    assert _accel._AVX512 not in _accel._variants_to_try(True)


# ---------------------------------------------------------------------------
# The Linux check: /proc/cpuinfo
# ---------------------------------------------------------------------------

# An AVX-512 server line (Zen 4 / Ice Lake class), cut to what matters.
AVX512_LINE = (
    "flags\t\t: fpu sse2 ssse3 sse4_1 sse4_2 avx avx2 fma bmi1 bmi2 "
    "avx512f avx512dq avx512cd avx512bw avx512vl avx512vbmi\n"
)


def _cpuinfo(monkeypatch, tmp_path, text):
    path = tmp_path / "cpuinfo"
    path.write_text("processor\t: 0\nvendor_id\t: AuthenticAMD\n" + text)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(_accel.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(_accel, "_CPUINFO_PATH", str(path))


def test_a_full_avx512_line_is_supported(monkeypatch, tmp_path):
    _cpuinfo(monkeypatch, tmp_path, AVX512_LINE)
    assert _accel._avx512_missing() == ()
    assert _accel._cpu_supports_avx512() is True


@pytest.mark.parametrize("flag", REQUIRED)
def test_each_missing_flag_is_named_and_refused(monkeypatch, tmp_path, flag):
    """Every required flag, removed one at a time, is a "no" that names it.

    DQ and VL are the brief's examples; CD and BW are in the set because
    the build enables them (setup.py, MSVC /arch:AVX512), and AVX2/FMA
    because the AVX-512 build is the AVX2 one plus five."""
    _cpuinfo(monkeypatch, tmp_path, AVX512_LINE.replace(f" {flag} ", " "))
    assert _accel._avx512_missing() == (flag,)
    assert _accel._cpu_supports_avx512() is False


def test_skylake_client_is_avx2_not_avx512(monkeypatch, tmp_path):
    """The i7-6700K line this suite's fleet box reports, abridged."""
    _cpuinfo(
        monkeypatch,
        tmp_path,
        "flags\t\t: fpu sse2 ssse3 sse4_1 sse4_2 avx avx2 fma bmi1 bmi2 adx\n",
    )
    assert _accel._avx512_missing() == REQUIRED[2:]
    assert _accel._cpu_supports_extension() is True


def test_no_flags_line_cannot_tell(monkeypatch, tmp_path):
    _cpuinfo(monkeypatch, tmp_path, "model name\t: something\n")
    assert _accel._avx512_missing() is None
    assert _accel._cpu_supports_avx512() is None


def test_no_procfs_cannot_tell(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(_accel.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(_accel, "_CPUINFO_PATH", str(tmp_path / "absent"))
    assert _accel._avx512_missing() is None


def test_macos_never_gets_avx512(monkeypatch):
    """No variant is built on darwin (arm64 wheel; an Intel source build is
    the single unsuffixed extension)."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(_accel.platform, "machine", lambda: "x86_64")
    assert _accel._avx512_missing() is None


@pytest.mark.parametrize("machine", ["aarch64", "arm64", "ppc64le"])
def test_non_x86_never_gets_avx512(monkeypatch, machine):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(_accel.platform, "machine", lambda: machine)
    assert _accel._avx512_missing() is None


def test_the_avx512_check_never_raises(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")

    def explode():
        raise RuntimeError("boom")

    monkeypatch.setattr(_accel.platform, "machine", explode)
    with pytest.raises(RuntimeError):
        _accel._avx512_missing_uncaught()
    assert _accel._avx512_missing() is None


# ---------------------------------------------------------------------------
# The Windows check: IsProcessorFeaturePresent(41) AND the cpuid probe
# ---------------------------------------------------------------------------

# Register images. Leaf 1 ECX: FMA (12), OSXSAVE (27), AVX (28). Leaf 7 EBX:
# AVX2 (5), F (16), DQ (17), CD (28), BW (30), VL (31). XCR0: 0xE7 = x87, SSE,
# AVX, opmask, ZMM_Hi256, Hi16_ZMM.
L1 = (1 << 12) | (1 << 27) | (1 << 28)
L7_AVX2 = 1 << 5
L7_AVX512 = L7_AVX2 | (1 << 16) | (1 << 17) | (1 << 28) | (1 << 30) | (1 << 31)
# Knights Landing: F and CD, no DQ/BW/VL. The case PF_AVX512F alone gets wrong.
L7_KNL = L7_AVX2 | (1 << 16) | (1 << 28)
XCR0_ALL = 0xE7
XCR0_AVX = 0x07


def test_cpuid_decode_of_an_avx512_cpu():
    flags = _accel._flags_from_cpuid(0xD, L1, L7_AVX512, XCR0_ALL)
    assert set(REQUIRED) <= flags


def test_cpuid_decode_of_knights_landing_misses_dq_bw_vl():
    flags = _accel._flags_from_cpuid(0xD, L1, L7_KNL, XCR0_ALL)
    assert {"avx512f", "avx512cd"} <= flags
    assert not {"avx512dq", "avx512bw", "avx512vl"} & flags


def test_cpuid_decode_honours_the_os_zmm_state():
    """CPU has AVX-512, OS saves only the AVX state: Linux would list no
    avx512* flag, and neither may the decoder."""
    flags = _accel._flags_from_cpuid(0xD, L1, L7_AVX512, XCR0_AVX)
    assert "avx2" in flags
    assert not any(f.startswith("avx512") for f in flags)


def test_cpuid_decode_without_osxsave_is_empty():
    flags = _accel._flags_from_cpuid(0xD, L1 & ~(1 << 27), L7_AVX512, XCR0_ALL)
    assert flags == frozenset()


def test_cpuid_decode_never_reads_past_the_max_leaf():
    flags = _accel._flags_from_cpuid(6, L1, L7_AVX512, XCR0_ALL)
    assert flags == {"fma", "avx"}


def _windows(monkeypatch, cpuid_flags, pf):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(_accel.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(_accel, "_cpuid_flags", lambda: cpuid_flags)
    monkeypatch.setattr(_accel, "_windows_feature_present", lambda feature: pf)


FULL = frozenset(REQUIRED) | {"avx"}


def test_windows_with_both_sources_agreeing_is_supported(monkeypatch):
    _windows(monkeypatch, FULL, True)
    assert _accel._avx512_missing() == ()


@pytest.mark.parametrize("pf", [False, None])
def test_windows_without_the_os_vouching_for_f_is_refused(monkeypatch, pf):
    _windows(monkeypatch, FULL, pf)
    assert _accel._avx512_missing() == ("avx512f",)


@pytest.mark.parametrize("flag", REQUIRED)
def test_windows_with_the_os_saying_f_but_cpuid_missing_one(monkeypatch, flag):
    _windows(monkeypatch, FULL - {flag}, True)
    assert _accel._avx512_missing() == (flag,)


def test_windows_without_the_probe_cannot_tell(monkeypatch):
    _windows(monkeypatch, None, True)
    assert _accel._avx512_missing() is None


# ---------------------------------------------------------------------------
# _load
# ---------------------------------------------------------------------------


def _recording_loader(monkeypatch, present: str):
    """test_accel_cpu_guard_1032's spy: records the names asked for, imports
    nothing, and serves a stub only for `present`."""
    asked: list[str] = []

    class _Stub:
        def __init__(self, name):
            self.__name__ = name

    def spy(base_name, suffix):
        asked.append(f"{base_name}{suffix}")
        return _Stub(f"momwire.{base_name}{suffix}") if suffix == present else None

    monkeypatch.setattr(_accel, "_import_variant", spy)
    return asked


def _cpu(monkeypatch, *, avx2, missing):
    monkeypatch.setattr(_accel, "_cpu_supports_extension", lambda: avx2)
    monkeypatch.setattr(_accel, "_avx512_missing", lambda: missing)


def test_an_avx512_cpu_loads_the_avx512_build(monkeypatch, unforced):
    _cpu(monkeypatch, avx2=True, missing=())
    asked = _recording_loader(monkeypatch, "_avx512")
    _mod, loaded, variant = _accel._load()
    assert (loaded, variant) == (True, "avx512")
    assert asked == ["_accelerators_avx512"]


def test_an_avx512_cpu_without_the_build_takes_avx2(monkeypatch, unforced):
    """An install predating #1370 on an AVX-512 box: the chain steps down."""
    _cpu(monkeypatch, avx2=True, missing=())
    asked = _recording_loader(monkeypatch, "_avx2")
    _mod, loaded, variant = _accel._load()
    assert (loaded, variant) == (True, "avx2")
    assert asked == ["_accelerators_avx512", "_accelerators_avx2"]


@pytest.mark.parametrize("missing", [("avx512dq",), ("avx512vl",), None])
def test_a_cpu_short_of_avx512_never_asks_for_it(monkeypatch, unforced, missing):
    _cpu(monkeypatch, avx2=True, missing=missing)
    asked = _recording_loader(monkeypatch, "_avx2")
    _mod, loaded, variant = _accel._load()
    assert (loaded, variant) == (True, "avx2")
    assert "_accelerators_avx512" not in asked


def test_the_force_override_takes_avx512_on_a_cpu_that_has_it(monkeypatch):
    monkeypatch.setenv(_accel._FORCE_VARIANT_ENV, "avx512")
    assert _accel._variants_to_try(True, True) == (_accel._AVX512,)
    _cpu(monkeypatch, avx2=True, missing=())
    asked = _recording_loader(monkeypatch, "_avx512")
    assert _accel._load()[1:] == (True, "avx512")
    assert asked == ["_accelerators_avx512"]


@pytest.mark.parametrize("avx512", [False, None])
def test_the_force_override_cannot_load_avx512_on_a_cpu_without_it(monkeypatch, avx512):
    monkeypatch.setenv(_accel._FORCE_VARIANT_ENV, "avx512")
    assert _accel._variants_to_try(True, avx512) == ()


@pytest.mark.parametrize(
    "missing,named",
    [(("avx512dq", "avx512vl"), "lacks avx512dq, avx512vl"), (None, "could not be")],
)
def test_forced_avx512_on_the_wrong_cpu_names_why_and_imports_nothing(
    monkeypatch, missing, named
):
    """The #1038 shape: pure Python, with the reason named, never a crash."""
    monkeypatch.setenv(_accel._FORCE_VARIANT_ENV, "avx512")
    monkeypatch.delenv("MOMWIRE_REQUIRE_ACCEL", raising=False)
    _cpu(monkeypatch, avx2=True, missing=missing)
    asked = _recording_loader(monkeypatch, "_avx512")
    monkeypatch.setattr(_accel, "_extension_built", lambda suffix=None: True)

    with pytest.warns(RuntimeWarning) as record:
        mod, loaded, variant = _accel._load()

    assert (mod, loaded, variant) == (None, False, None)
    assert asked == [], "a variant was imported on a CPU that was refused"
    messages = [str(w.message) for w in record]
    assert any(named in m and "NOT loaded" in m for m in messages), messages


def test_forced_avx512_refusal_is_an_error_under_require_accel(monkeypatch):
    monkeypatch.setenv(_accel._FORCE_VARIANT_ENV, "avx512")
    monkeypatch.setenv("MOMWIRE_REQUIRE_ACCEL", "1")
    _cpu(monkeypatch, avx2=True, missing=("avx512bw",))
    _recording_loader(monkeypatch, "_avx512")
    with pytest.raises(RuntimeError, match="lacks avx512bw"):
        _accel._load()


def test_forced_avx512_not_in_this_install_says_so(monkeypatch):
    """A CPU that runs it, an install that lacks it: #1038's message."""
    monkeypatch.setenv(_accel._FORCE_VARIANT_ENV, "avx512")
    monkeypatch.delenv("MOMWIRE_REQUIRE_ACCEL", raising=False)
    _cpu(monkeypatch, avx2=True, missing=())
    monkeypatch.setattr(_accel, "_import_variant", lambda *a: None)
    monkeypatch.setattr(
        _accel, "_extension_built", lambda suffix=None: suffix in (None, "_avx2")
    )
    with pytest.warns(RuntimeWarning) as record:
        assert _accel._load() == (None, False, None)
    assert any(
        "variant 'avx512' is not in this install (present: avx2)" in str(w.message)
        for w in record
    )


# ---------------------------------------------------------------------------
# The build and the check name the same set
# ---------------------------------------------------------------------------

SETUP = Path(__file__).resolve().parents[1] / "setup.py"


def test_setup_builds_exactly_what_the_loader_checks():
    """setup.py's `_AVX512_GCC_FLAGS` and `_AVX512_REQUIRED` past its AVX2
    pair are one list spelled twice; a flag added to the build and not to the
    check is a CPU the loader would wrongly let through."""
    if not SETUP.exists():
        pytest.skip("no source tree (a wheel install): setup.py is not here")
    tree = ast.parse(SETUP.read_text())
    flags = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "_AVX512_GCC_FLAGS" for t in node.targets
        ):
            flags = ast.literal_eval(node.value)
    assert flags is not None, "setup.py no longer defines _AVX512_GCC_FLAGS"
    assert [f.removeprefix("-m") for f in flags] == list(REQUIRED[2:])
    assert REQUIRED[:2] == _accel._LINUX_REQUIRED_FLAGS


# ---------------------------------------------------------------------------
# This box, for real
# ---------------------------------------------------------------------------

_X86 = _accel.platform.machine().lower() in ("x86_64", "amd64")
_X86_LINUX = sys.platform == "linux" and _X86
_X86_WIN = sys.platform == "win32" and _X86


@pytest.mark.skipif(not _X86_LINUX, reason="/proc/cpuinfo is the oracle (x86 Linux)")
def test_the_cpuid_probe_agrees_with_proc_cpuinfo_here():
    """The Windows decoder, run on real registers and held to the kernel's own
    answer. On a box without AVX-512 this proves the "no" half; on an
    AVX-512 box (the Fly run) the "yes" half."""
    flags = _accel._cpuid_flags()
    if flags is None:
        pytest.skip("momwire._cpuid is not built in this install")
    proc = _accel._linux_cpu_flags()
    assert proc is not None
    names = set(REQUIRED) | {"avx"}
    assert {f for f in names if f in flags} == {f for f in names if f in proc}


def _probe(env_extra):
    return subprocess.run(
        [
            sys.executable,
            "-W",
            "always::RuntimeWarning",
            "-c",
            "import momwire; print(momwire.accelerator_variant)",
        ],
        env={**os.environ, **env_extra},
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_forcing_avx512_here_is_honoured_or_refused_by_name():
    """A real interpreter. With AVX-512 the forced build loads; without it the
    process survives, stays unaccelerated and says which flags it lacked."""
    if not _accel._extension_built("_avx512"):
        pytest.skip("no _avx512 build in this install")
    # REQUIRE_ACCEL off: the refusal is a warning here, not the error it is
    # under a build lane.
    out = _probe({_accel._FORCE_VARIANT_ENV: "avx512", "MOMWIRE_REQUIRE_ACCEL": "0"})
    assert out.returncode == 0, out.stderr[-2000:]
    if _accel._cpu_supports_avx512() is True:
        assert out.stdout.strip() == "avx512"
    else:
        assert out.stdout.strip() == "None"
        assert "'avx512'" in out.stderr and "NOT loaded" in out.stderr, out.stderr


def test_this_box_chooses_avx512_exactly_when_it_has_it():
    if os.environ.get(_accel._FORCE_VARIANT_ENV):
        pytest.skip("the variant is pinned by the environment, not by this CPU")
    if not _accel._extension_built("_avx512"):
        pytest.skip("no _avx512 build in this install")
    has = _accel._cpu_supports_avx512() is True
    assert (momwire.accelerator_variant == "avx512") == has, (
        momwire.accelerator_variant,
        _accel._avx512_missing(),
    )


@pytest.mark.skipif(
    sys.platform != "win32" or not _X86_WIN,
    reason="the Windows half of the check (x86-64 Windows)",
)
def test_the_cpuid_probe_loads_and_agrees_with_windows_here():
    """On a real Windows box: the probe built and loads, and its decoding
    agrees with IsProcessorFeaturePresent wherever both can answer (AVX2,
    and AVX-512F with the OS's state)."""
    if not _accel._extension_built("_avx2"):
        pytest.skip("a single-variant install")
    flags = _accel._cpuid_flags()
    assert flags is not None, "momwire._cpuid did not build or load"
    pf_avx2 = _accel._windows_feature_present(_accel._PF_AVX2_INSTRUCTIONS_AVAILABLE)
    pf_f = _accel._windows_feature_present(_accel._PF_AVX512F_INSTRUCTIONS_AVAILABLE)
    assert ("avx2" in flags) == (pf_avx2 is True), (sorted(flags), pf_avx2)
    assert ("avx512f" in flags) == (pf_f is True), (sorted(flags), pf_f)
