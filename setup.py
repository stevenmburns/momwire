import glob
import os
import sys
import warnings

from pybind11.setup_helpers import ParallelCompile, Pybind11Extension
from setuptools import setup
from setuptools.command.build_ext import build_ext

# Build-time error classes. setuptools.errors is the modern home (distutils is
# removed in Python 3.12+); fall back to distutils for very old setuptools.
try:
    from setuptools.errors import CCompilerError, ExecError, FileError, PlatformError
except ImportError:  # pragma: no cover - ancient setuptools
    from distutils.errors import (  # type: ignore[no-redef]
        CCompilerError,
    )
    from distutils.errors import (
        DistutilsExecError as ExecError,
    )
    from distutils.errors import (
        DistutilsFileError as FileError,
    )
    from distutils.errors import (
        DistutilsPlatformError as PlatformError,
    )

# momwire._accelerators is an *optional* C++ speedup: every module that imports
# it (_bspline_kernels, bspline, hmatrix,
# sinusoidal) guards the import with `try/except ImportError` and falls back to
# a pure-Python/numpy path. So a platform with no working compiler / libmvec /
# libomp (musllinux, glibc < 2.28, an arch outside the wheel matrix, or no
# toolchain at all) should still get a usable install rather than a hard
# `pip install` failure. This cmdclass makes a failed extension build a warning
# instead of an error, leaving the package importable in pure-Python mode.
#
# This does NOT let a silently-degraded *wheel* ship: the cibuildwheel
# test-command asserts `import momwire._accelerators` succeeds, so any CI wheel
# that fails to compile the extension fails its tests. The graceful path is only
# for source (sdist) installs on unsupported platforms.
_OPTIONAL_BUILD_ERRORS = (
    CCompilerError,
    ExecError,
    PlatformError,
    FileError,  # inplace copy of an extension that never got built
    FileNotFoundError,  # compiler binary absent
)


class OptionalBuildExt(build_ext):
    def run(self):
        try:
            super().run()
        except _OPTIONAL_BUILD_ERRORS as exc:
            self._warn(exc)

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except _OPTIONAL_BUILD_ERRORS as exc:
            self._warn(exc)

    @staticmethod
    def _warn(exc):
        warnings.warn(
            f"a momwire C++ extension failed to build ({exc!r}); "
            "installing in pure-Python mode. The solver will work but run "
            "slower. Install a C++ toolchain (and on Linux, glibc>=2.28 with "
            "libmvec) for the accelerated path.",
            stacklevel=2,
        )


# The accelerator is built on all three platforms; the vectorization strategy
# differs per platform. Linux/GCC binds the inner sincos to glibc's libmvec
# (-lmvec) via the `omp declare simd` block in _accelerators.cpp; Windows/MSVC
# has no libmvec, so it relies on /arch:AVX2 autovectorization plus OpenMP
# parallelism; macOS Apple Silicon (arm64) has neither libmvec nor AVX2, so it
# relies on Homebrew libomp for OpenMP parallelism and lets clang autovectorize
# the inner loops for NEON. The .cpp guards the libmvec-specific declarations to
# non-MSVC, non-Apple compilers. If the extension fails to build/import,
# the solvers fall back to pure Python.
if sys.platform == "win32":
    # OpenMP on MSVC is a minefield for this code: /openmp:experimental rejects
    # unsigned loop indices (the kernels use size_t) and silently drops the
    # `reduction` clause from `omp simd` (a correctness hazard), while
    # /openmp:llvm rejects the `omp simd` directive outright. We use
    # /openmp:llvm — it supports the OpenMP 3.0 `collapse` clause and unsigned
    # loop indices, so the parallel-for loops need no changes — and the .cpp
    # neutralizes the `omp simd` directives under _MSC_VER, leaving /arch:AVX2
    # autovectorization to handle the inner loops. /arch:AVX2 matches the Linux
    # AVX2 baseline.
    # /MP: MSVC's own parallel compile across the five TUs. Needed because
    # pybind11's ParallelCompile below is a verified NO-OP here — it patches
    # the distutils base Compiler.compile, and MSVC's compiler class overrides
    # compile() in its own class dict, so the patch never runs. Without /MP
    # the split would make Windows wheel builds strictly SLOWER than the
    # monolith (five serial preamble parses) — the #710 review's finding.
    extra_compile_args = ["/O2", "/arch:AVX2", "/openmp:llvm", "/fp:fast", "/MP"]
    extra_link_args = []
elif sys.platform == "darwin":
    # Apple clang ships no OpenMP runtime and macOS has no libmvec, so this
    # branch is deliberately the "simple pragmas" port: Homebrew's libomp gives
    # us the OpenMP parallel-for + omp-simd directives (passed through Apple
    # clang via -Xpreprocessor -fopenmp), and clang autovectorizes the inner
    # sincos for NEON on its own. No -mavx2/-mfma (arm64 has no AVX2) and no
    # -lmvec (no vectorized libm on macOS); the libmvec `declare simd` block in
    # _accelerators.cpp is #ifdef'd off under __APPLE__. delocate vendors the
    # libomp dylib into the wheel (the -rpath below points the extension at it).
    _libomp = os.environ.get("LIBOMP_PREFIX", "/opt/homebrew/opt/libomp")
    extra_compile_args = [
        "-O3",
        "-Xpreprocessor",
        "-fopenmp",
        # Same errno rationale as the Linux branch: let the vectorizer run.
        "-fno-math-errno",
        "-std=gnu++11",
        f"-I{os.path.join(_libomp, 'include')}",
    ]
    extra_link_args = [
        f"-L{os.path.join(_libomp, 'lib')}",
        "-lomp",
        f"-Wl,-rpath,{os.path.join(_libomp, 'lib')}",
    ]
else:
    extra_compile_args = [
        # Force -O3 -- Debian's Python CFLAGS inject -O2 before our flags
        # and pybind11's default -O3 doesn't override that. Our -O3 here
        # comes after both and wins (gcc takes the last -O).
        "-O3",
        "-fopenmp",
        "-fopenmp-simd",
        # AVX2 + FMA: required for the SIMD inner-loop sincos in
        # _accelerators.cpp to use libmvec (vectorized libm). KBL/HSW
        # and newer Intel; matches what pybind11 release wheels can't
        # assume but a local pip install -e . can.
        "-mavx2",
        "-mfma",
        # `std::cos` / `std::sin` set errno on domain errors by default,
        # which is a global side effect that blocks auto-vectorization.
        # We don't care about errno from a deterministic-domain real input,
        # so disable the side effect to let the vectorizer kick in.
        "-fno-math-errno",
        "-g",
        "-fno-omit-frame-pointer",
        "-std=gnu++11",
    ]
    extra_link_args = ["-fopenmp", "-lpthread", "-lmvec"]

# The near-interface twin (momwire#680 U2) compiles against the vendored
# scipy/xsf headers (extern/xsf, header-only), which are C++17. The main
# `_accelerators` extension stays at gnu++11 untouched; only this second,
# equally-optional extension gets the newer standard.
if sys.platform == "win32":
    _near_compile_args = extra_compile_args + ["/std:c++17"]
else:
    _near_compile_args = [
        "-std=gnu++17" if a == "-std=gnu++11" else a for a in extra_compile_args
    ]

# The inline headers `_accelerators.cpp` pulls in. setuptools rebuilds an
# object file only when a listed source or DEPENDENCY is newer than it, and a
# header it has never been told about is neither: without this list, editing
# `_contour_engine_inline.h` and re-running `build_ext --inplace` prints
# "copying build/lib.../_accelerators...so" and silently re-installs the OLD
# binary. That is a measurement hazard, not just an inconvenience — momwire#568
# spent a benchmark round comparing a stale .so against numpy and read a 6x
# speedup where the current code had 19x. Every new inline header belongs here.
_ACCEL_HEADERS = [
    "src/momwire/_bspline_static_moments_inline.h",
    "src/momwire/_bspline_ek_moments_inline.h",
    "src/momwire/_contour_engine_inline.h",
    # The shared preamble of the split TUs (momwire#687): includes, the OpenMP
    # simd declarations and the cancellation machinery. Every accelerator TU
    # includes it, so an edit here must rebuild all of them -- the momwire#568
    # stale-.so lesson, which is exactly why this list exists.
    "src/momwire/_accel_common.h",
    "src/momwire/_accel_somm_proj_inline.h",
]

# The accelerator's translation units (momwire#687). The monolith was one
# ~8,000-line TU, so a one-line edit anywhere recompiled all of it; these are
# cut along the sections' own boundaries and compile independently. The thin
# `_accelerators.cpp` holds only PYBIND11_MODULE and calls each section's
# `register_*`, so every kernel keeps internal linkage inside its own TU.
_ACCEL_SOURCES = [
    "src/momwire/_accelerators.cpp",
    "src/momwire/_accel_bspline.cpp",
    "src/momwire/_accel_sinusoidal.cpp",
    "src/momwire/_accel_somm.cpp",
    "src/momwire/_accel_mw568.cpp",
]

# Same staleness rationale for the near-interface twin: the contour engine
# header AND every vendored xsf header are dependencies, or editing (or
# re-vendoring) one silently re-installs the old binary.
_NEAR_HEADERS = ["src/momwire/_contour_engine_inline.h"] + sorted(
    glob.glob("extern/xsf/include/xsf/**/*.h", recursive=True)
)

# Compile the accelerator's translation units concurrently (momwire#687). With
# the old single-TU monolith this bought nothing; with five TUs it is what
# makes a COLD build -- CI's first run, and every wheel-matrix job -- cheaper
# rather than merely no worse, since splitting a TU adds total compiler work
# (the shared preamble is parsed once per TU) even as it shrinks the
# incremental edit. Honours NPY_NUM_BUILD_JOBS; defaults to the CPU count, and
# `NPY_NUM_BUILD_JOBS=1` restores serial compilation for a constrained runner.
# GCC/clang only: MSVC's compiler class overrides the compile() this patches,
# so on Windows the parallelism comes from /MP in extra_compile_args above.
ParallelCompile("NPY_NUM_BUILD_JOBS").install()

ext_modules = [
    Pybind11Extension(
        "momwire._accelerators",
        _ACCEL_SOURCES,
        depends=_ACCEL_HEADERS,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Pybind11Extension(
        "momwire._near_interface_accel",
        ["src/momwire/_near_interface_accel.cpp"],
        depends=_NEAR_HEADERS,
        include_dirs=["extern/xsf/include"],
        extra_compile_args=_near_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": OptionalBuildExt},
    # Listed explicitly rather than discovered: the list is short, and an
    # explicit one cannot silently ship a stray directory under src/.
    packages=[
        "momwire",
        "momwire.deck",
        "momwire.eznec",
        "momwire.networks",
        "momwire.portal",
    ],
    # The `momwire-nec2c-shared` client (issue #379). A top-level MODULE rather
    # than part of the package on purpose: its whole value is that running it
    # imports neither `momwire` nor NumPy, and a module inside the package
    # would import `momwire/__init__.py` to get there.
    py_modules=["momwire_nec2c_client"],
    package_dir={"": "src/"},
)
