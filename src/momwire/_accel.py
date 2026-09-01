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

Public attributes:
    ``acc``          — the loaded ``_accelerators`` module, or ``None``.
    ``LOADED``       — ``True`` iff the accelerator imported successfully.
    ``MAX_N_QP``     — the B-spline pair kernels' quadrature ceiling.
    ``serves_n_qp()``— routing predicate for that ceiling; warns on fallback.
"""

from __future__ import annotations

import importlib.machinery
import pathlib
import sys
import warnings


def _extension_built() -> bool:
    """True if a compiled ``_accelerators`` extension exists on disk.

    Distinguishes "built but won't load" from "never built": the file's presence
    means the build succeeded, so a failed import is a runtime problem worth a
    warning rather than an expected pure-Python fallback.
    """
    pkg = pathlib.Path(__file__).parent
    return any(
        (pkg / f"_accelerators{suffix}").exists()
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )


def _load():
    """Import the accelerator, warning if a *built* extension fails to load.

    Returns ``(module_or_None, loaded_bool)``. Kept as a function so the warn
    decision is unit-testable without reloading the whole package.
    """
    try:
        from . import _accelerators as mod
    except ImportError as exc:
        if _extension_built():
            if sys.platform == "darwin":
                hint = (
                    "On macOS the accelerator links Homebrew's OpenMP runtime, "
                    "which the wheel does not bundle (so it can share one libomp "
                    "with pynec-accel); install it with `brew install libomp`."
                )
            elif sys.platform == "win32":
                # NOT vcomp140.dll, which is what MSVC usually means and what
                # Windows already redistributes: setup.py builds the extensions
                # with /openmp:llvm, so they link LLVM's runtime, which is
                # neither a system DLL nor something PyInstaller collects.
                # momwire#737 shipped two EZNEC bundles without it.
                hint = (
                    "On Windows the accelerator links LLVM's OpenMP runtime, "
                    "libomp140.x86_64.dll (the extensions are built with "
                    "/openmp:llvm, so it is that and not vcomp140.dll); the "
                    "wheel and the EZNEC bundle are expected to ship or find "
                    "it. Reinstalling momwire is the first fix; failing that, "
                    "the file comes with the Visual C++ LLVM OpenMP runtime "
                    "(the LLVM/clang component of a Visual Studio install), "
                    "and must sit beside the extension or on PATH."
                )
            else:
                hint = (
                    "On Linux the accelerator links the system libgomp (the GCC "
                    "OpenMP runtime), which the wheel does not bundle (so it "
                    "shares one libgomp with pynec-accel); install it if missing "
                    "(`apt install libgomp1`, or your distro's equivalent). A "
                    "static-TLS clash from an older vendored-libgomp build "
                    "(momwire < 0.2.2 or pynec-accel < 1.7.4.post1) is the other "
                    "cause; the stopgap there is "
                    "GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2097152."
                )
            warnings.warn(
                "momwire: the compiled accelerator '_accelerators' is installed "
                f"but failed to import ({exc!r}); falling back to the slower "
                f"pure-Python path. {hint}",
                RuntimeWarning,
                stacklevel=3,
            )
        # else: genuinely not built for this platform — pure-Python is expected.
        return None, False
    return mod, True


acc, LOADED = _load()


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


def serves_n_qp(n_qp: int, what: str, *, eligible: bool = True) -> bool:
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
    if n_qp <= MAX_N_QP:
        return True
    if eligible:
        warnings.warn(
            f"n_qp={n_qp} exceeds the accelerated {what} kernel's ceiling of "
            f"{MAX_N_QP}, so this fill takes the numpy path — correct, but much "
            f"slower, and the cost grows as n_qp^2. Lifting the ceiling is "
            f"momwire#762.",
            RuntimeWarning,
            stacklevel=3,
        )
    return False
