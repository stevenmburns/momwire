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
  on macOS. The older failure — a static-TLS clash from a vendored libgomp
  (momwire < 0.2.2 or pynec-accel < 1.7.4.post1) loaded after another, failing
  with "cannot allocate memory in static TLS block" — is the other cause.
  Either way the fallback used to be invisible — this module makes it audible.

Public attributes:
    ``acc``     — the loaded ``_accelerators`` module, or ``None``.
    ``LOADED``  — ``True`` iff the accelerator imported successfully.
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
    "bspline_assemble_offedge_block",
    "sinusoidal_field_tensor",
    "somm_six_integrals_batch",
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
