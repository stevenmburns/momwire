import os
import sys

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

# The accelerator is built on all three platforms; the vectorization strategy
# differs per platform. Linux/GCC binds the inner sincos to glibc's libmvec
# (-lmvec) via the `omp declare simd` block in _accelerators.cpp; Windows/MSVC
# has no libmvec, so it relies on /arch:AVX2 autovectorization plus OpenMP
# parallelism; macOS Apple Silicon (arm64) has neither libmvec nor AVX2, so it
# relies on Homebrew libomp for OpenMP parallelism and lets clang autovectorize
# the inner loops for NEON. The .cpp guards the libmvec-specific declarations to
# non-MSVC, non-Apple compilers. If the extension fails to build/import,
# triangular.py falls back to pure Python.
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
    extra_compile_args = ["/O2", "/arch:AVX2", "/openmp:llvm", "/fp:fast"]
    extra_link_args = []
    extra_objects = []
elif sys.platform == "darwin":
    # Apple clang ships no OpenMP runtime, so Homebrew's libomp gives us the
    # OpenMP parallel-for + omp-simd directives (passed through Apple clang via
    # -Xpreprocessor -fopenmp). macOS has no libmvec, but Apple Silicon's inner
    # sincos is vectorized for NEON via SLEEF (see the SLEEF block below); absent
    # SLEEF it falls back to Apple clang's scalar/NEON-autovectorized sincos. No
    # -mavx2/-mfma (arm64 has no AVX2). delocate vendors the libomp dylib into
    # the wheel (the -rpath below points the extension at it); SLEEF is
    # static-linked, so it adds nothing for delocate to vendor.
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
    # Vectorize the inner sincos via SLEEF's NEON 2-wide-double routines
    # (_ZGVnN2v_{sin,cos}) when a static SLEEF is available. -fveclib=SLEEF makes
    # clang substitute them into the existing `omp simd` loops (no declare-simd
    # block needed — that's the libmvec-specific trick, gated off under __APPLE__
    # in _accelerators.cpp). The gnuabi static archive is linked by full path so
    # the link is static (no dylib for delocate to vendor). Without SLEEF_PREFIX
    # set to a prefix that actually contains the archive (e.g. a plain local
    # `pip install -e .`), this stays scalar — same as before.
    _sleef = os.environ.get("SLEEF_PREFIX")
    _sleef_a = (
        os.path.join(_sleef, "lib", "libsleefgnuabi.a") if _sleef else None
    )
    if _sleef_a and os.path.exists(_sleef_a):
        extra_compile_args.append("-fveclib=SLEEF")
        extra_objects = [_sleef_a]
    else:
        extra_objects = []
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
    extra_objects = []

ext_modules = [
    Pybind11Extension(
        "momwire._accelerators",
        ["src/momwire/_accelerators.cpp"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        extra_objects=extra_objects,
    ),
]

setup(
    ext_modules=ext_modules,
    packages=["momwire"],
    package_dir={"": "src/"},
)
