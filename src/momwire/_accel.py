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

    ``variant_suffix`` names one variant ("_avx2", "_sse2", "" for the
    unsuffixed legacy build); None asks about ANY of them, which is what
    "was this install built with an accelerator at all" means.

    Distinguishes "built but won't load" from "never built": the file's presence
    means the build succeeded, so a failed import is a runtime problem worth a
    warning rather than an expected pure-Python fallback.
    """
    pkg = pathlib.Path(__file__).parent
    names = ("_avx2", "_sse2", "") if variant_suffix is None else (variant_suffix,)
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
        try:
            import ctypes

            fn = ctypes.windll.kernel32.IsProcessorFeaturePresent
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_uint32]
            return bool(fn(_PF_AVX2_INSTRUCTIONS_AVAILABLE))
        except Exception:  # noqa: BLE001 — a guard that can raise defeats its own purpose
            # No windll, no such export, a stub kernel32, a ctypes built
            # without windll on a Windows-like host: every one of them
            # means "cannot tell", and none of them may propagate out of
            # `import momwire`.
            return None
    try:
        with open(_CPUINFO_PATH, encoding="ascii", errors="replace") as fh:
            for line in fh:
                if not line.startswith("flags") and not line.startswith("Features"):
                    continue
                _, _, rest = line.partition(":")
                flags = set(rest.split())
                # One core's flag line answers for the CPU; stop at the first.
                return all(f in flags for f in _LINUX_REQUIRED_FLAGS)
    except OSError:
        return None  # no procfs (a container, a BSD, a locked-down sandbox)
    return None  # a procfs with no flags line at all


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
_AVX2 = ("avx2", "_avx2")
_SSE2 = ("sse2", "_sse2")
_LEGACY = ("legacy", "")

# Test-only override, documented here because it has no other documentation:
# forces the chain to one variant so the accelerator suite can be run against
# the BASELINE build on an AVX2 box. Not a supported user knob — a wrong value
# here is how you get the fault this whole issue is about, on purpose.
_FORCE_VARIANT_ENV = "MOMWIRE_FORCE_VARIANT"


def _variants_to_try(verdict: bool | None) -> tuple[tuple[str, str], ...]:
    """The (label, suffix) chain to attempt, most preferred first.

    UNKNOWN IS NOT OPTIMISTIC. A CPU we could not classify gets the baseline
    build, never the AVX2 one: guessing wrong upward kills the interpreter with
    no output, and guessing wrong downward costs speed. Those are not
    comparable mistakes, so the tie always breaks the same way.
    """
    forced = os.environ.get(_FORCE_VARIANT_ENV)
    if forced:
        chain = {"avx2": (_AVX2,), "sse2": (_SSE2,), "legacy": (_LEGACY,)}.get(forced)
        if chain is not None:
            return chain
        warnings.warn(
            f"{_FORCE_VARIANT_ENV}={forced!r} is not one of 'avx2', 'sse2', "
            "'legacy'; ignoring it and choosing normally.",
            RuntimeWarning,
            stacklevel=3,
        )
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
    chain = _variants_to_try(verdict)
    for label, suffix in chain:
        mod = _import_variant("_accelerators", suffix)
        if mod is not None:
            _alias_historic_name("_accelerators", mod)
            return mod, True, label

    # Nothing in the chain imported. Two very different reasons, and the user
    # needs to be told which.
    tried = ", ".join(label for label, _ in chain)
    if verdict is False and not any(_extension_built(s) for _, s in chain):
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

    `_near_interface` has its own optional extension, and the two must match:
    an AVX2 `_near_interface_accel` beside an SSE2 `_accelerators` faults on
    exactly the CPU the split exists for, and it would fault from the module
    nobody was looking at. One decision, made once, read by both.
    """
    if VARIANT is None:
        return None
    suffix = dict((label, sfx) for label, sfx in (_AVX2, _SSE2, _LEGACY))[VARIANT]
    mod = _import_variant(base_name, suffix)
    if mod is not None:
        _alias_historic_name(base_name, mod)
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
    "bspline_assemble_offedge_block",
    "sinusoidal_field_tensor",
    "sinusoidal_field_tensor_refl",
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
)


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
