"""Single load point for the optional C++ accelerator (``_accelerators``).

Every solver module imports the extension through here instead of carrying its
own ``try/except ImportError`` guard, so the decision of *whether to warn* lives
in one place. The distinction that matters:

* **Extension never built** (unsupported platform, or a deliberate pure-Python
  install) — the pure-Python fallback is expected, so stay silent.
* **Extension built but failed to load** — something is wrong at *runtime*, and
  the fast path silently vanishes, so warn loudly. The Linux/macOS wheels link
  the *system* OpenMP runtime rather than bundling one (so they share a single
  runtime with pynec-accel instead of clashing), so the usual cause is that the
  runtime is missing: ``apt install libgomp1`` on Linux, ``brew install libomp``
  on macOS. Windows is the other shape — there the extensions link LLVM's
  ``libomp140.x86_64.dll`` (``/openmp:llvm`` in setup.py), which is no system
  DLL, so the wheel and the frozen bundle have to SHIP it (momwire#737). The
  older failure — a static-TLS clash from a vendored libgomp
  (momwire < 0.2.2 or pynec-accel < 1.7.4.post1) loaded after another, failing
  with "cannot allocate memory in static TLS block" — is the other cause.
  Either way the fallback used to be invisible — this module makes it audible.
* **Extension built, loadable, and fatal to run** — the CPU predates the
  instruction set the wheel was compiled for. This one cannot be expressed as
  an exception handler at all: the process dies with an illegal instruction
  and prints nothing (momwire#1032). It is caught BEFORE the import, by
  ``_cpu_supports_extension()``, and warns like the case above.

Public attributes:
    ``acc``          — the loaded ``_accelerators`` module, or ``None``.
    ``LOADED``       — ``True`` iff the accelerator imported successfully.
    ``MAX_N_QP``     — the off-edge pair kernels' quadrature ceiling.
    ``SAME_EDGE_MAX_N_QP`` — the same-edge reg kernel's, which is separate.
    ``serves_n_qp()``— routing predicate for that ceiling; warns on fallback.
"""

from __future__ import annotations

import importlib.machinery
import os
import pathlib
import platform
import sys
import warnings


def _extension_built(variant_suffix: str | None = None) -> bool:
    """True if a compiled ``_accelerators`` extension exists on disk.

    ``variant_suffix`` names one variant ("_avx512", "_avx2", "_sse2", "" for
    the unsuffixed legacy build); None asks about ANY of them, which is what
    "was this install built with an accelerator at all" means.

    Distinguishes "built but won't load" from "never built": the file's presence
    means the build succeeded, so a failed import is a runtime problem worth a
    warning rather than an expected pure-Python fallback.
    """
    pkg = pathlib.Path(__file__).parent
    names = (
        ("_avx512", "_avx2", "_sse2", "")
        if variant_suffix is None
        else (variant_suffix,)
    )
    return any(
        (pkg / f"_accelerators{name}{ext}").exists()
        for name in names
        for ext in importlib.machinery.EXTENSION_SUFFIXES
    )


# ---------------------------------------------------------------------------
# The CPU-feature guard (momwire#1032)
# ---------------------------------------------------------------------------
#
# `except ImportError` cannot catch an illegal instruction. The wheels are
# built for AVX2 + FMA (setup.py: `/arch:AVX2` on MSVC, `-mavx2 -mfma`
# otherwise), so on a CPU without them the extension LOADS and then faults the
# moment a vectorized loop runs: STATUS_ILLEGAL_INSTRUCTION (0xC000001D) on
# Windows, SIGILL on Linux. The interpreter dies with no traceback and no
# output — the pure-Python fallback that exists for a MISSING extension never
# runs, because the extension is not missing. So the check has to happen
# BEFORE the import, not around it.
#
# WHICH PLATFORMS CARRY THE REQUIREMENT, read off setup.py rather than assumed:
#
#   win32   /arch:AVX2                     -> requirement
#   darwin  no -mavx2/-mfma, BOTH arches   -> none. The darwin branch is the
#           "simple pragmas" port (Homebrew libomp + clang's own NEON
#           autovectorization); it passes no AVX flag on Intel Macs either, so
#           macOS is not affected at all — not merely arm64.
#   other   -mavx2 -mfma                   -> requirement
#
# The last branch is arch-BLIND — setup.py never inspects `platform.machine()`,
# so it would hand `-mavx2` to a Linux aarch64 build too. That is why the
# machine gate below is not redundant: on a non-x86 CPU there is no AVX2 flag
# to find in /proc/cpuinfo, and a guard that read the absence as "unsupported"
# would refuse an extension that is perfectly good.
#
# THE ASYMMETRY THAT MATTERS: a wrong False is worse than a wrong None. False
# turns a working accelerated install into a silently slow one, which is the
# harder failure to notice; None just preserves today's behaviour, which is
# correct everywhere except the CPUs this issue is about. So every path that
# cannot answer with certainty returns None.

# `IsProcessorFeaturePresent` — PF_AVX2_INSTRUCTIONS_AVAILABLE. (39 is
# PF_AVX_INSTRUCTIONS_AVAILABLE, if a future build drops to /arch:AVX.)
_PF_AVX2_INSTRUCTIONS_AVAILABLE = 40

# `platform.machine()` spellings for the x86 family, lowercased. Anything else
# (aarch64, arm64, ppc64le, s390x, riscv64) has no AVX2 to look for.
_X86_MACHINES = frozenset({"x86_64", "amd64", "x86", "i386", "i486", "i586", "i686"})

# What the Linux/POSIX branch compiles for. FMA is checked as well as AVX2
# because setup.py passes `-mfma` beside `-mavx2`: every shipping AVX2 CPU has
# FMA3, so this is belt-and-braces rather than a case anyone has hit.
_LINUX_REQUIRED_FLAGS = ("avx2", "fma")

# Named so a test can point the parser at a fixture instead of mocking `open`.
# The parser is the part with real behaviour (two possible key spellings, a
# missing file, a file with no flag line at all), so it is worth exercising for
# real on this box rather than through a stand-in.
_CPUINFO_PATH = "/proc/cpuinfo"


def _cpu_supports_extension() -> bool | None:
    """Does this CPU have the instruction-set extensions the wheel was built for?

    ``True``  — checked, present.
    ``False`` — checked, ABSENT. Importing the extension would fault the
                interpreter, so the caller must not import it.
    ``None``  — this platform's build carries no such requirement, or the CPU
                could not be interrogated. The caller proceeds exactly as it
                did before momwire#1032.

    Never raises — the whole body is wrapped, not just the interrogations. A
    guard that can itself break `import momwire` is worse than the crash it
    prevents, and the parts that look infallible (`platform.machine()`) are
    exactly the ones nobody writes a handler for.
    """
    try:
        return _cpu_supports_extension_uncaught()
    except Exception:  # noqa: BLE001 — see above; None means "proceed as before"
        return None


def _cpu_supports_extension_uncaught() -> bool | None:
    """`_cpu_supports_extension` without the blanket handler, so the tests can
    exercise the real failure modes rather than the handler's shadow."""
    if sys.platform == "darwin":
        return None  # no AVX flags on either macOS arch (see the note above)
    if platform.machine().lower() not in _X86_MACHINES:
        return None  # nothing to look for; setup.py's flags are x86's
    if sys.platform == "win32":
        return _windows_feature_present(_PF_AVX2_INSTRUCTIONS_AVAILABLE)
    flags = _linux_cpu_flags()
    if flags is None:
        return None
    return all(f in flags for f in _LINUX_REQUIRED_FLAGS)


def _windows_feature_present(feature: int) -> bool | None:
    """`IsProcessorFeaturePresent(feature)`; None when it cannot be asked."""
    try:
        import ctypes

        fn = ctypes.windll.kernel32.IsProcessorFeaturePresent
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_uint32]
        return bool(fn(feature))
    except Exception:  # noqa: BLE001 — a guard that can raise defeats its own purpose
        # No windll, no such export, a stub kernel32, a ctypes built without
        # windll on a Windows-like host: every one of them means "cannot
        # tell", and none of them may propagate out of `import momwire`.
        return None


def _linux_cpu_flags() -> frozenset[str] | None:
    """The first ``flags`` (x86) or ``Features`` (arm) line of /proc/cpuinfo,
    as a set; None when there is no procfs or no such line.

    One core's line answers for the CPU. The kernel lists a feature only when
    it is usable: an AVX or AVX-512 flag is cleared when the kernel does not
    enable that xstate in XCR0, so a flag here already means "the CPU has it
    AND the OS saves its registers".
    """
    try:
        with open(_CPUINFO_PATH, encoding="ascii", errors="replace") as fh:
            for line in fh:
                if not line.startswith("flags") and not line.startswith("Features"):
                    continue
                _, _, rest = line.partition(":")
                return frozenset(rest.split())
    except OSError:
        return None  # no procfs (a container, a BSD, a locked-down sandbox)
    return None  # a procfs with no flags line at all


# ---------------------------------------------------------------------------
# The AVX-512 check (momwire#1370)
# ---------------------------------------------------------------------------
#
# The `_avx512` variant is the AVX2 build plus AVX-512 F/CD/BW/DQ/VL (setup.py's
# `_AVX512_GCC_FLAGS`; MSVC's /arch:AVX512 enables the same five). It may be
# chosen only when every one of them is present AND the OS saves the opmask and
# zmm registers: anything less is momwire#1032's silent death one level up.
#
#   linux   /proc/cpuinfo. The kernel clears the avx512* flags when it has not
#           enabled their xstate in XCR0, so a listed flag answers both.
#   win32   IsProcessorFeaturePresent(PF_AVX512F_INSTRUCTIONS_AVAILABLE) speaks
#           for F alone, and F alone is not enough (Knights Landing/Mill have
#           F/CD/ER/PF without DQ/BW/VL). So the rest comes from
#           `momwire._cpuid`, a baseline-built probe: CPUID leaf 7 for the
#           feature bits, XCR0 for the state, decoded below by the kernel's
#           rule. Both must agree; a probe that did not build or load answers
#           None.
#   darwin  never: no variant is built there. The macOS wheel is arm64 only,
#           and a source build on an Intel Mac takes the single unsuffixed
#           extension (setup.py builds variants only on x86 off darwin).
#   other   never: no AVX-512 off x86.
#
# Same asymmetry as the AVX2 check. A wrong "no" costs the AVX-512 speedup and
# keeps `_avx2`; a wrong "yes" kills the interpreter. So every path that cannot
# read the answer returns None, and None never selects `_avx512`.

# IsProcessorFeaturePresent: AVX-512F, with the OS's xstate support.
_PF_AVX512F_INSTRUCTIONS_AVAILABLE = 41

# Everything the `_avx512` build was compiled to use, in /proc/cpuinfo's
# spelling: the AVX2 build's pair and the five AVX-512 subsets. The dispatch
# test holds the last five equal to setup.py's `_AVX512_GCC_FLAGS`.
_AVX512_REQUIRED = (
    "avx2",
    "fma",
    "avx512f",
    "avx512cd",
    "avx512bw",
    "avx512dq",
    "avx512vl",
)

# CPUID feature bits (Intel SDM vol. 2A, "CPUID"), by /proc/cpuinfo name.
_LEAF1_ECX_BITS = {"fma": 12, "avx": 28}
_LEAF1_ECX_OSXSAVE = 27
_LEAF7_EBX_BITS = {
    "avx2": 5,
    "avx512f": 16,
    "avx512dq": 17,
    "avx512cd": 28,
    "avx512bw": 30,
    "avx512vl": 31,
}
_LEAF7_AVX512 = frozenset(n for n in _LEAF7_EBX_BITS if n.startswith("avx512"))
# XCR0 components the OS must save: SSE (bit 1) and the AVX upper halves
# (bit 2) for any AVX instruction; plus the opmask (5), ZMM_Hi256 (6) and
# Hi16_ZMM (7) for AVX-512.
_XCR0_AVX = 0b0000_0110
_XCR0_AVX512 = 0b1110_0110


def _flags_from_cpuid(
    max_leaf: int, leaf1_ecx: int, leaf7_ebx: int, xcr0: int
) -> frozenset[str]:
    """The flags these registers vouch for, by the rule Linux applies before
    listing them in /proc/cpuinfo: an AVX-family feature counts only if the
    OS has OSXSAVE set and saves the AVX state in XCR0, and an AVX-512 one only
    if it also saves the opmask and zmm state. A leaf above `max_leaf` (CPUID
    leaf 0's eax) is not asked for, and reads as no bits."""
    flags: set[str] = set()
    if max_leaf < 1 or not (leaf1_ecx >> _LEAF1_ECX_OSXSAVE) & 1:
        return frozenset()
    os_avx = (xcr0 & _XCR0_AVX) == _XCR0_AVX
    os_avx512 = (xcr0 & _XCR0_AVX512) == _XCR0_AVX512
    if not os_avx:
        return frozenset()
    flags.update(n for n, b in _LEAF1_ECX_BITS.items() if (leaf1_ecx >> b) & 1)
    if max_leaf >= 7:
        for name, bit in _LEAF7_EBX_BITS.items():
            if name in _LEAF7_AVX512 and not os_avx512:
                continue
            if (leaf7_ebx >> bit) & 1:
                flags.add(name)
    return frozenset(flags)


def _cpuid_flags() -> frozenset[str] | None:
    """`_flags_from_cpuid` on this CPU's registers, through `momwire._cpuid`;
    None when the probe is not built or does not load."""
    try:
        probe = importlib.import_module(f"{__package__}._cpuid")
    except ImportError:
        return None
    max_leaf = probe.cpuid(0, 0)[0]
    leaf1_ecx = probe.cpuid(1, 0)[2] if max_leaf >= 1 else 0
    leaf7_ebx = probe.cpuid(7, 0)[1] if max_leaf >= 7 else 0
    return _flags_from_cpuid(max_leaf, leaf1_ecx, leaf7_ebx, probe.xcr0())


def _avx512_missing() -> tuple[str, ...] | None:
    """Which of `_AVX512_REQUIRED` this CPU (and OS) lacks.

    ``()``    — checked, nothing missing: the `_avx512` build runs here.
    a tuple   — checked, these are missing: importing it would fault.
    ``None``  — no `_avx512` build applies to this platform, or the CPU could
                not be interrogated. Never selects the variant.

    Never raises, for `_cpu_supports_extension`'s reason.
    """
    try:
        return _avx512_missing_uncaught()
    except Exception:  # noqa: BLE001 — None means "do not choose AVX-512"
        return None


def _avx512_missing_uncaught() -> tuple[str, ...] | None:
    if sys.platform == "darwin":
        return None
    if platform.machine().lower() not in _X86_MACHINES:
        return None
    if sys.platform == "win32":
        flags = _cpuid_flags()
        if flags is None:
            return None
        if _windows_feature_present(_PF_AVX512F_INSTRUCTIONS_AVAILABLE) is not True:
            # The OS does not vouch for F (or cannot be asked): whatever CPUID
            # says, F does not count.
            flags = flags - {"avx512f"}
    else:
        flags = _linux_cpu_flags()
        if flags is None:
            return None
    return tuple(f for f in _AVX512_REQUIRED if f not in flags)


def _cpu_supports_avx512() -> bool | None:
    """`_avx512_missing` as a verdict: True, False, or None (cannot tell)."""
    missing = _avx512_missing()
    return None if missing is None else not missing


def _forced_avx512_refused_message(missing: tuple[str, ...] | None) -> str:
    """Why `MOMWIRE_FORCE_VARIANT=avx512` loaded nothing on this CPU."""
    why = (
        "this CPU could not be checked for AVX-512"
        if missing is None
        else f"this CPU lacks {', '.join(missing)}"
    )
    return (
        f"momwire: {_FORCE_VARIANT_ENV}='avx512', but {why}, so the AVX-512 "
        "build was NOT loaded: here it would not raise, it would kill the "
        "interpreter with an illegal instruction (momwire#1032). Falling back "
        "to the slower pure-Python path. Unset it, or name a variant this CPU "
        "runs."
    )


def _no_avx2_message() -> str:
    """The sentence a user on one of these CPUs has to be able to act on."""
    return (
        "momwire: this CPU does not support the AVX2/FMA instructions the "
        "compiled accelerator was built for, so the accelerator was NOT "
        "loaded and momwire is running in pure Python. The answers are the "
        "same; the solver is slower. Loading it would not have raised an "
        "ImportError — it would have killed the interpreter with an illegal "
        "instruction (0xC000001D on Windows, SIGILL on Linux), which is the "
        "silent exit this check replaces. The accelerated build needs an "
        "Intel Core from 2013 (Haswell) or later, or an AMD from 2015 "
        "(Excavator) or later. See momwire#1032."
    )


def _platform_hint() -> str:
    """Why a BUILT extension might refuse to import, per OS.

    Unchanged from before momwire#1032 — these are the runtime-linkage causes
    (a missing OpenMP runtime, a static-TLS clash), which are a different
    failure from the CPU one and keep their own prose.
    """
    if sys.platform == "darwin":
        return (
            "On macOS the accelerator links Homebrew's OpenMP runtime, "
            "which the wheel does not bundle (so it can share one libomp "
            "with pynec-accel); install it with `brew install libomp`."
        )
    if sys.platform == "win32":
        # NOT vcomp140.dll, which is what MSVC usually means and what
        # Windows already redistributes: setup.py builds the extensions
        # with /openmp:llvm, so they link LLVM's runtime, which is
        # neither a system DLL nor something PyInstaller collects.
        # momwire#737 shipped two EZNEC bundles without it.
        return (
            "On Windows the accelerator links LLVM's OpenMP runtime, "
            "libomp140.x86_64.dll (the extensions are built with "
            "/openmp:llvm, so it is that and not vcomp140.dll); the "
            "wheel and the EZNEC bundle are expected to ship or find "
            "it. Reinstalling momwire is the first fix; failing that, "
            "the file comes with the Visual C++ LLVM OpenMP runtime "
            "(the LLVM/clang component of a Visual Studio install), "
            "and must sit beside the extension or on PATH."
        )
    return (
        "On Linux the accelerator links the system libgomp (the GCC "
        "OpenMP runtime), which the wheel does not bundle (so it "
        "shares one libgomp with pynec-accel); install it if missing "
        "(`apt install libgomp1`, or your distro's equivalent). A "
        "static-TLS clash from an older vendored-libgomp build "
        "(momwire < 0.2.2 or pynec-accel < 1.7.4.post1) is the other "
        "cause; the stopgap there is "
        "GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2097152."
    )


# The label reported as `momwire.accelerator_variant`, paired with the module
# -name suffix it maps to. "legacy" is the unsuffixed extension: what macOS and
# every non-x86 build still produce, and what an install predating momwire#1032
# has on disk. Keeping it in the chain is what lets a wheel built either way
# load in either loader.
_AVX512 = ("avx512", "_avx512")
_AVX2 = ("avx2", "_avx2")
_SSE2 = ("sse2", "_sse2")
_LEGACY = ("legacy", "")
# Most preferred first.
_ALL_VARIANTS = (_AVX512, _AVX2, _SSE2, _LEGACY)

# Test-only override, documented here because it has no other documentation:
# forces the chain to one variant so the accelerator suite can be run against
# one build on a box that would choose another. Not a supported user knob.
#
# 'avx2', 'sse2' and 'legacy' are taken on trust: forcing 'avx2' on a CPU
# without AVX2 is how you get the fault this whole issue is about, on purpose,
# and that is unchanged. 'avx512' is NOT: it is honoured only where the
# AVX-512 check passes, and anywhere else it loads nothing and says why
# (`_forced_avx512_refused_message`). A timing or gate run that asked for
# AVX-512 on the wrong box then fails loudly instead of dying silently, and a
# typo'd fleet host cannot be mistaken for an AVX-512 measurement.
_FORCE_VARIANT_ENV = "MOMWIRE_FORCE_VARIANT"


def _variants_to_try(
    verdict: bool | None, avx512: bool | None = None
) -> tuple[tuple[str, str], ...]:
    """The (label, suffix) chain to attempt, most preferred first.

    `verdict` is the AVX2 check (`_cpu_supports_extension`), `avx512` the
    AVX-512 one (`_cpu_supports_avx512`).

    UNKNOWN IS NOT OPTIMISTIC. A CPU we could not classify gets the baseline
    build, never the AVX2 one, and the AVX-512 build only on two checked yeses:
    guessing wrong upward kills the interpreter with no output, and guessing
    wrong downward costs speed. Those are not comparable mistakes, so the tie
    always breaks the same way.

    A forced 'avx512' on a CPU that did not pass the AVX-512 check returns the
    EMPTY chain, and `_load` says why (see `_FORCE_VARIANT_ENV`).
    """
    forced = os.environ.get(_FORCE_VARIANT_ENV)
    if forced:
        if forced == "avx512":
            return (_AVX512,) if avx512 is True else ()
        chain = {"avx2": (_AVX2,), "sse2": (_SSE2,), "legacy": (_LEGACY,)}.get(forced)
        if chain is not None:
            return chain
        warnings.warn(
            f"{_FORCE_VARIANT_ENV}={forced!r} is not one of 'avx512', 'avx2', "
            "'sse2', 'legacy'; ignoring it and choosing normally.",
            RuntimeWarning,
            stacklevel=3,
        )
    if verdict is True and avx512 is True:
        return (_AVX512, _AVX2, _SSE2, _LEGACY)
    if verdict is True:
        return (_AVX2, _SSE2, _LEGACY)
    return (_SSE2, _LEGACY)


def _import_variant(base_name: str, suffix: str):
    """Import one variant of one extension, or None if it is not there."""
    try:
        return importlib.import_module(f"{__package__}.{base_name}{suffix}")
    except ImportError:
        return None


def _load():
    """Import the accelerator, warning if a *built* extension fails to load.

    Returns ``(module_or_None, loaded_bool, variant_or_None)``. Kept as a
    function so the warn decision is unit-testable without reloading the whole
    package.

    The CPU check comes FIRST and decides WHICH extension is imported, not
    merely whether: on a pre-AVX2 CPU the import itself is what kills the
    process (momwire#1032), so there is no exception handler that could take
    its place.
    """
    verdict = _cpu_supports_extension()
    missing = _avx512_missing()
    chain = _variants_to_try(verdict, None if missing is None else not missing)
    for label, suffix in chain:
        mod = _import_variant("_accelerators", suffix)
        if mod is not None:
            _alias_historic_name("_accelerators", mod)
            return mod, True, label

    # Nothing in the chain imported. Several very different reasons, and the
    # user needs to be told which.
    tried = ", ".join(label for label, _ in chain) or "none"
    forced_suffix = dict(_ALL_VARIANTS).get(os.environ.get(_FORCE_VARIANT_ENV))
    present = [label for label, suffix in _ALL_VARIANTS if _extension_built(suffix)]
    if forced_suffix == _AVX512[1] and not chain:
        # Forced AVX-512 on a CPU that did not pass the check: refused BEFORE
        # any import, whether or not the build is on disk.
        reason = _forced_avx512_refused_message(missing)
    elif forced_suffix is not None and not _extension_built(forced_suffix) and present:
        # The test-only override named a build this install does not carry
        # (momwire#1038). Said plainly: without it, a caller guesses, and the
        # usual guess (a missing OpenMP runtime) sends the debugging elsewhere.
        forced = os.environ[_FORCE_VARIANT_ENV]
        reason = (
            f"momwire: {_FORCE_VARIANT_ENV}={forced!r}, but variant {forced!r} is "
            f"not in this install (present: {', '.join(present)}); falling back to "
            "the slower pure-Python path. Unset it, or name a present variant."
        )
    elif verdict is False and not any(_extension_built(s) for _, s in chain):
        # The CPU cannot run what IS built (an AVX2-only wheel predating the
        # double build) — the case that used to be a silent process death.
        reason = _no_avx2_message()
    elif any(_extension_built(s) for _, s in chain):
        reason = (
            "momwire: a compiled accelerator is installed but no variant "
            f"imported (tried: {tried}); falling back to the slower "
            f"pure-Python path. {_platform_hint()}"
        )
    else:
        # Genuinely not built for this platform — pure-Python is expected.
        return None, False, None

    if os.environ.get("MOMWIRE_REQUIRE_ACCEL") == "1":
        raise RuntimeError(
            f"{reason} MOMWIRE_REQUIRE_ACCEL=1 is set, so this is an error "
            f"rather than a fallback. Variants tried: {tried}."
        )
    warnings.warn(reason, RuntimeWarning, stacklevel=3)
    return None, False, None


def _alias_historic_name(base_name: str, mod) -> None:
    """Bind the chosen variant under the extension's ORIGINAL, unsuffixed name.

    Before momwire#1032 there was one `momwire._accelerators`, and plenty of
    code says so — `from momwire import _accelerators` appears across this
    repo's own tests, and anything outside it that reached for the extension
    did the same. Splitting the file into `_avx2`/`_sse2` would break every one
    of those with an ImportError that has nothing to do with the caller.

    So the historic name keeps working and now means "whichever variant this
    process loaded", which is the only thing it can honestly mean once there is
    more than one. Registered in `sys.modules` AND as a package attribute
    because `from momwire import _accelerators` checks the attribute first.
    """
    full = f"{__package__}.{base_name}"
    sys.modules.setdefault(full, mod)
    pkg = sys.modules.get(__package__)
    if pkg is not None and not hasattr(pkg, base_name):
        setattr(pkg, base_name, mod)


def import_companion(base_name: str):
    """Import `base_name` from the SAME variant `_accelerators` came from.

    `_near_interface` has its own optional extensions, and they must match:
    an AVX2 `_near_interface_accel` beside an SSE2 `_accelerators` faults on
    exactly the CPU the split exists for (an AVX-512 one beside an AVX2
    `_accelerators`, on the CPU `_avx512` was refused on), and it would fault
    from the module nobody was looking at. One decision, made once, read by
    all of them.
    """
    if VARIANT is None:
        return None
    suffix = dict(_ALL_VARIANTS)[VARIANT]
    mod = _import_variant(base_name, suffix)
    if mod is not None:
        _alias_historic_name(base_name, mod)
        _install_cancel_translation(mod)
    return mod


acc, LOADED, VARIANT = _load()


# Kernels that take a trailing ``cancel_flag`` and raise the C++ ``AcceleratorAborted``
# when it is tripped mid-fill (Phase 2). We remap that to the shared
# ``momwire.SolveAborted`` here — the one place the extension is loaded — so callers
# only ever catch a single abort type, whether it came from a Python-level
# checkpoint or from inside a native fill.
_CANCELLABLE_KERNELS = (
    "assemble_Z_bspline",
    "assemble_Z_bspline_windowed",
    "assemble_Z_bspline_weighted_windowed",
    "assemble_Z_sinusoidal_windowed",
    "bspline_assemble_offedge_block",
    "sinusoidal_field_tensor",
    "sinusoidal_field_tensor_refl",
    "sinusoidal_field_tensor_cplx",
    # The extended-kernel twins take and poll the same flag. `_ek` was left off
    # this tuple when momwire#245 added it, so a cancelled EK solve surfaced the
    # raw ``AcceleratorAborted`` instead of ``SolveAborted``; #259 adds both
    # rather than land its own kernel with the same hole.
    "sinusoidal_field_tensor_ek",
    "sinusoidal_field_tensor_ek_refl",
    # Same hole on the fused bspline block assembler's variants: only the
    # plain one was listed, so a cancelled H-matrix fill over finite ground
    # or under EK surfaced the raw ``AcceleratorAborted``. momwire#269 needs
    # `_refl_ek` (the path it opens) and lists its two siblings with it
    # rather than leave the tuple half-populated.
    "bspline_assemble_offedge_block_refl",
    "bspline_assemble_offedge_block_ek",
    "bspline_assemble_offedge_block_refl_ek",
    # Their laddered twins (momwire#1362), the entries an EK H-matrix far
    # block reaches by default; the same poll, so the same translation.
    "bspline_assemble_offedge_block_ek_tiered",
    "bspline_assemble_offedge_block_refl_ek_tiered",
    "sinusoidal_galerkin_far_fill",
    "somm_six_integrals_batch",
    # The razor-blade formulation's fused moment fill (momwire#742). It polls
    # between observer tiles, which is the only granularity that exists there:
    # the whole fill is one call per (observer set, source set, k), so a
    # cancelled razor solve reaches Python again only through this remap.
    "razor_seg_moments",
    # Its in-medium twin (momwire#796). Same kernel body, same tile-granular
    # poll, so an aborted complex-k fill must remap the same way; listed with
    # its sibling rather than left for the next hole-finding issue.
    "razor_seg_moments_cplx",
    # momwire#1006. The same-edge static-moment entries became cancellable when
    # their O(N^2) gather turned out to be the window a knob change waits out:
    # 85% of the call at N=801, 96% at N=3201, and the whole call grows
    # quadratically -- 107 ms at N=3201, 420 ms at N=6401. Registering them
    # here is what makes the C++ `AcceleratorAborted` surface as `SolveAborted`;
    # a kernel that polls but is not listed raises the wrong exception type and
    # every caller's `except SolveAborted` misses it.
    "seg_seg_static_moments_bspline_uniform",
    "seg_seg_static_moments_bspline_uniform_ek",
    # AK#1712 found the Sommerfeld remainder block raising the raw
    # `AcceleratorAborted` through a web cancel, and a census then found eight
    # more kernels that take `cancel_flag` but were never listed: the
    # complex-eps and weighted assembler variants, the Galerkin far fill's
    # complex and EK twins, and the remainder projection. Listed together,
    # and `tests/test_cancellable_kernel_census.py` now fails for any kernel
    # whose signature takes `cancel_flag` and is missing here, so the next
    # one cannot be added with the same hole.
    "sommerfeld_remainder_bspline_Q",
    "remainder_field_proj_batch",
    "assemble_Z_bspline_weighted",
    "assemble_Z_bspline_cplx_eps",
    "assemble_Z_bspline_weighted_cplx_eps",
    "assemble_Z_bspline_windowed_cplx_eps",
    "assemble_Z_bspline_weighted_windowed_cplx_eps",
    "sinusoidal_galerkin_far_fill_cplx",
    "sinusoidal_galerkin_far_fill_ek",
    # momwire#1201: the remainder at listed pairs, and the graded rule's
    # fused source-side moments.
    "remainder_field_proj_owned",
    "remainder_graded_inner",
    # momwire#1348: the below/below and below->above grid fills' contour
    # batches, polled per node.
    "below_six_integrals_batch",
    "transmitted_six_integrals_batch",
    # The crossing fill's and the near-interface tables' native calls that
    # run long enough to hold a cancel (measured: tests/
    # test_cancel_native_crossing_1348.py has the census). The column twin
    # lives in the `_near_interface_accel` companion, which registers its own
    # `AcceleratorAborted`; `import_companion` wraps it from this same list.
    "factorize_rows",
    "factorize_ints",
    "group_first_ranks",
    "merge_rows_by_z",
    "near_interface_six_columns",
    # momwire#1335: the plan's key ids (factorize_rows of the same rows).
    "factorize_line_keys",
    # momwire#1377: the tiles' key classes and class sums, row order and
    # late-row marks.
    "factorize_float_classes",
    "class_sums",
    "stable_tile_order",
    "late_sandwich_rows",
)

# Methods of the extension's classes that take `cancel_flag`, as (class,
# method): the census reads module-level kernels only, so these are listed by
# hand and wrapped on the class.
_CANCELLABLE_METHODS = (("RowIndex", "find"), ("RowGroups", "add"))


def _install_cancel_translation(mod) -> None:
    """Wrap each cancellable kernel on ``mod`` so ``AcceleratorAborted`` surfaces
    as ``momwire.SolveAborted``. No-op if the extension predates Phase 2."""
    import functools

    from ._cancel import SolveAborted

    aborted = getattr(mod, "AcceleratorAborted", None)
    if aborted is None:
        return

    def _wrap(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except aborted:
                raise SolveAborted() from None

        return wrapper

    for name in _CANCELLABLE_KERNELS:
        raw = getattr(mod, name, None)
        if raw is not None:
            setattr(mod, name, _wrap(raw))
    for cls_name, meth in _CANCELLABLE_METHODS:
        cls = getattr(mod, cls_name, None)
        raw = getattr(cls, meth, None) if cls is not None else None
        if raw is not None:
            setattr(cls, meth, _wrap(raw))


if acc is not None:
    _install_cancel_translation(acc)


# ---------------------------------------------------------------------------
# The quadrature ceiling (momwire#769)
# ---------------------------------------------------------------------------
#
# The B-spline pair kernels carry L1-sized stack scratch and refuse n_qp above
# what fits it. READ OFF THE EXTENSION rather than re-spelled here, so the
# Python routing guard cannot drift from the kernels' real limit and so
# momwire#762 — which tiles the qr loop and lifts the ceiling — changes one
# `constexpr` in _accel_common.h and nothing on this side.
#
# The fallback for an older extension that predates the export is the value
# every such build compiled in.
MAX_N_QP: int = int(getattr(acc, "BSPLINE_MAX_N_QP", 8)) if acc is not None else 8

# The same-edge reg-moment kernel keeps a ceiling that momwire#762 did not
# lift — it needs a different transformation from the six off-edge kernels.
# Separate constant because it is a separate kernel with a separate limit;
# collapsing them is how momwire#769 came to miss this path in the first
# place.
SAME_EDGE_MAX_N_QP: int = (
    int(getattr(acc, "BSPLINE_SAME_EDGE_MAX_N_QP", 8)) if acc is not None else 8
)


def serves_n_qp(
    n_qp: int, what: str, *, eligible: bool = True, cap: int | None = None
) -> bool:
    """True if the accelerated pair kernels can take this quadrature order.

    False means "route to numpy", not "fail": the numpy path has no ceiling
    and is what momwire#758's anchors were converged on. Before momwire#769
    the kernels were simply called and raised, which turned a slow-but-correct
    answer into an unhandled `RuntimeError` — on exactly the crossing/lossy-soil
    class that needs the order (momwire#760).

    `eligible` is what makes the warning mean something. A caller with complex
    k, or an EK spec with no C++ twin, or a monkeypatched-off accelerator was
    ALREADY taking numpy, and telling it about a quadrature ceiling it never
    reached is noise — the repo's own complex-k and Sommerfeld truth references
    call this with n_qp of 12, 64 and 256 for exactly that reason. So the
    warning fires only when this ceiling is the thing that moved the work.

    It does warn when it is: the cliff is real (numpy against threaded C++, on
    work that is O(n_qp^2)) and a silent 100x is its own bug report.
    `warnings` dedupes by call site, so a solve that falls back on every block
    says so once.
    """
    ceiling = MAX_N_QP if cap is None else cap
    if n_qp <= ceiling:
        return True
    if eligible:
        warnings.warn(
            f"n_qp={n_qp} exceeds the accelerated {what} kernel's ceiling of "
            f"{ceiling}, so this fill takes the numpy path — correct, but much "
            f"slower, and the cost grows as n_qp^2. Lifting the ceiling is "
            f"momwire#762.",
            RuntimeWarning,
            stacklevel=3,
        )
    return False
