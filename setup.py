import glob
import os
import platform
import subprocess
import sys
import warnings

from pybind11.setup_helpers import ParallelCompile, Pybind11Extension
from setuptools import setup
from pathlib import Path

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
        # PER-EXTENSION OBJECT TREE (momwire#1032). distutils names an object
        # file after its SOURCE path, not after the extension being built, so
        # two extensions compiled from the SAME sources with DIFFERENT flags
        # collide in `build/temp.../src/momwire/*.o`: the second link reuses
        # the first's objects and produces a second .so with the first's code.
        # That failure is silent and total — the "baseline" extension would
        # contain AVX2 instructions and fault on exactly the CPUs the double
        # build exists for, while importing cleanly on every box we own.
        base_temp = self.build_temp
        self.build_temp = os.path.join(base_temp, ext.name.rsplit(".", 1)[-1])
        try:
            super().build_extension(ext)
        except _OPTIONAL_BUILD_ERRORS as exc:
            self._warn(exc)
            return
        finally:
            self.build_temp = base_temp
        if os.environ.get("MOMWIRE_STRIP_SYMBOLS") == "1":
            self._split_debug(Path(self.get_ext_fullpath(ext.name)))

    @staticmethod
    def _split_debug(so: Path) -> None:
        """Move the DWARF out of a built extension, KEEP the symbol table.

        The released Linux wheel shipped both extensions unstripped and the
        debug info was 38 MB of it — while the Windows wheel ships the same
        `_accelerators` at 1.8 MB, because MSVC puts debug info in a separate
        .pdb that is not distributed (momwire#1030).

        `--strip-debug` rather than `-s`, and the difference is worth the
        bytes. Measured on this extension:

            published (unstripped)   34.8 MB
            strip --strip-debug       2.2 MB   .symtab kept, 1998 symbols
            strip -s                  1.9 MB   no symbols at all

        0.4 MB across both extensions buys a segfault backtrace that NAMES our
        functions instead of printing addresses. The DWARF goes to a sibling
        `<name>.debug` linked by build ID, so a debugger given both still has
        everything the unstripped build had.

        `-g` in the compile flags stays either way: it pairs with
        `-fno-omit-frame-pointer` so a profile can walk the Python/C++
        boundary, and this runs after the link.
        """
        if not so.exists() or sys.platform not in ("linux", "linux2"):
            return
        dbg = so.with_suffix(so.suffix + ".debug")
        try:
            subprocess.run(
                ["objcopy", "--only-keep-debug", str(so), str(dbg)], check=True
            )
            subprocess.run(["strip", "--strip-debug", str(so)], check=True)
            # Run from the extension's own directory and name both files
            # RELATIVELY: `--add-gnu-debuglink` resolves its argument against
            # the cwd, and objcopy writes its temp file there too, so an
            # absolute target with a changed cwd fails on the temp file.
            subprocess.run(
                ["objcopy", f"--add-gnu-debuglink={dbg.name}", so.name],
                check=True,
                cwd=so.parent,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            # binutils absent or refusing: ship the unstripped extension
            # rather than fail the build. Bigger, never broken.
            print(f"momwire: could not split debug info from {so.name}: {exc}")
            return
        # MOVE the sidecar out of the package directory. `build_py` packages
        # everything under it, so leaving `<name>.so.debug` there put the DWARF
        # straight back into the wheel — measured, the first version of this
        # shipped both. `--add-gnu-debuglink` has already recorded the basename
        # and its CRC, so a debugger still finds the file through the standard
        # search path once it is installed alongside; the release artefact is
        # built from `build/debug/`.
        out = Path("build") / "debug"
        try:
            out.mkdir(parents=True, exist_ok=True)
            dbg.replace(out / dbg.name)
        except OSError as exc:
            dbg.unlink(missing_ok=True)
            print(f"momwire: dropped {dbg.name} (could not move it aside: {exc})")
            return
        print(f"momwire: {so.name} DWARF -> {out / dbg.name}, symbols kept")

    @staticmethod
    def _warn(exc):
        # The graceful path exists for sdist installs on unsupported
        # platforms. A DEVELOPMENT build wants the opposite: `make build`
        # sets MOMWIRE_REQUIRE_ACCEL=1 so a broken toolchain fails the lane
        # instead of exiting 0 and leaving a stale in-place .so — the
        # repo's documented measurement hazard (#716 review).
        if os.environ.get("MOMWIRE_REQUIRE_ACCEL") == "1":
            raise exc
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
    # /MP: MSVC's own parallel compile across the accelerator's TUs. Needed because
    # pybind11's ParallelCompile below is a verified NO-OP here — it patches
    # the distutils base Compiler.compile, and MSVC's compiler class overrides
    # compile() in its own class dict, so the patch never runs. Without /MP
    # the split would make Windows wheel builds strictly SLOWER than the
    # monolith (one serial preamble parse per TU) — the #710 review's finding.
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
    "src/momwire/_branch_cut_inline.h",
    # Both of these were MISSING until momwire#824, and the omission is what
    # that issue actually was. The far static-moment header is included by
    # _accel_bspline.cpp; the cross-lane stable spellings (#799) by three TUs.
    # An edit to either recompiled nothing, `build_ext --inplace` reported
    # success, and the OLD object was relinked and re-copied -- so the .so
    # stopped matching its own source while every test that reads the source
    # kept passing. #824 was read as a libm `pow` divergence for exactly this
    # reason: it is reproducible, box-local (CI always builds cold, so CI
    # cannot see it), and survives a `git checkout main`, because it is the
    # BINARY that is stale and not the tree. tests/test_accel_header_deps.py
    # now enforces the whole list against the sources' own #includes.
    "src/momwire/_bspline_static_far_inline.h",
    "src/momwire/_stable_inline.h",
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
    "src/momwire/_accel_razor.cpp",
]

# Same staleness rationale for the near-interface twin: the contour engine
# header AND every vendored xsf header are dependencies, or editing (or
# re-vendoring) one silently re-installs the old binary.
_NEAR_HEADERS = [
    "src/momwire/_contour_engine_inline.h",
    # The shared branch cut (#714) -- this extension carries the third
    # call site, so an edit to it must rebuild this .so too.
    "src/momwire/_branch_cut_inline.h",
] + sorted(glob.glob("extern/xsf/include/xsf/**/*.h", recursive=True))

# Compile the accelerator's translation units concurrently (momwire#687). With
# the old single-TU monolith this bought nothing; with the split it is what
# makes a COLD build -- CI's first run, and every wheel-matrix job -- cheaper
# rather than merely no worse, since splitting a TU adds total compiler work
# (the shared preamble is parsed once per TU) even as it shrinks the
# incremental edit. Honours NPY_NUM_BUILD_JOBS; defaults to the CPU count, and
# `NPY_NUM_BUILD_JOBS=1` restores serial compilation for a constrained runner.
# GCC/clang only: MSVC's compiler class overrides the compile() this patches,
# so on Windows the parallelism comes from /MP in extra_compile_args above.
ParallelCompile("NPY_NUM_BUILD_JOBS").install()

# ---------------------------------------------------------------------------
# The double build (momwire#1032)
# ---------------------------------------------------------------------------
#
# A wheel built `/arch:AVX2` (MSVC) or `-mavx2 -mfma` (GCC) LOADS on a CPU that
# predates those instructions and then dies the first time a vectorized loop
# runs: STATUS_ILLEGAL_INSTRUCTION (0xC000001D) on Windows, SIGILL on Linux, no
# traceback. `except ImportError` cannot catch a hardware fault, so the
# pure-Python fallback never ran and the process simply vanished — three days
# of a user's time on the QRZ thread.
#
# So on x86 both extensions are compiled TWICE from the same sources: once with
# today's flags (`_avx2`) and once at the x86-64 baseline (`_sse2`), and
# `momwire._accel` picks by a CPU-feature check before importing either. The
# wheel roughly doubles in size; that is the whole cost.
#
# NOT on macOS or non-x86: setup.py's darwin branch passes no AVX flag on
# either Mac arch (it is the simple-pragmas port), and no other architecture
# has AVX2 to ask for. Those keep the single, unsuffixed extension they have
# always had — the loader falls back to that name, so an existing install and
# a cross-built wheel both keep working.
_X86_MACHINES = {"x86_64", "amd64", "x86", "i386", "i486", "i586", "i686"}
_DOUBLE_BUILD = platform.machine().lower() in _X86_MACHINES and sys.platform != "darwin"

# What makes a build AVX2. Removing these leaves each compiler at its own
# x86-64 default, which IS the SSE2 baseline: MSVC x64 has no /arch below
# AVX, and GCC's x86-64 target implies SSE2.
_AVX2_ONLY_FLAGS = {"/arch:AVX2", "-mavx2", "-mfma"}


def _baseline(args):
    """`args` with the AVX2-specific flags removed, order otherwise intact."""
    return [a for a in args if a not in _AVX2_ONLY_FLAGS]


def _variant(base_name, suffix, sources, *, depends, compile_args, include_dirs=None):
    """One Pybind11Extension for one (module, instruction-set) pair.

    `define_macros` is what makes two extensions from one source legal:
    `PYBIND11_MODULE(MOMWIRE_MODULE_NAME, m)` pastes this token into the init
    symbol, so each variant exports its own `PyInit_...`. Without it both
    builds export `PyInit__accelerators` and the second cannot be imported
    under its own name at all.
    """
    name = base_name + suffix
    return Pybind11Extension(
        f"momwire.{name}",
        sources,
        depends=depends,
        include_dirs=include_dirs or [],
        extra_compile_args=compile_args,
        extra_link_args=extra_link_args,
        define_macros=[("MOMWIRE_MODULE_NAME", name)],
    )


if _DOUBLE_BUILD:
    _VARIANTS = (
        ("_avx2", extra_compile_args, _near_compile_args),
        ("_sse2", _baseline(extra_compile_args), _baseline(_near_compile_args)),
    )
else:
    _VARIANTS = (("", extra_compile_args, _near_compile_args),)

ext_modules = []
for _suffix, _accel_args, _near_args in _VARIANTS:
    ext_modules.append(
        _variant(
            "_accelerators",
            _suffix,
            _ACCEL_SOURCES,
            depends=_ACCEL_HEADERS,
            compile_args=_accel_args,
        )
    )
    ext_modules.append(
        _variant(
            "_near_interface_accel",
            _suffix,
            ["src/momwire/_near_interface_accel.cpp"],
            depends=_NEAR_HEADERS,
            compile_args=_near_args,
            include_dirs=["extern/xsf/include"],
        )
    )

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": OptionalBuildExt},
    # Listed explicitly rather than discovered: the list is short, and an
    # explicit one cannot silently ship a stray directory under src/.
    # Enumerated, so tests/test_portal_shared.py::
    # test_every_momwire_subpackage_is_shipped can hold this list to the
    # tree: momwire.serve was created (#719 U4) without an entry here, every
    # non-editable install lost the subpackage, and the first thing to
    # notice was the Windows freeze canary two PRs later (#718 phase 2) —
    # the editable dev install never sees this class of breakage.
    packages=[
        "momwire",
        "momwire.deck",
        "momwire.eznec",
        "momwire.networks",
        "momwire.portal",
        "momwire.serve",
    ],
    # The `momwire-nec2c-shared` client (issue #379). A top-level MODULE rather
    # than part of the package on purpose: its whole value is that running it
    # imports neither `momwire` nor NumPy, and a module inside the package
    # would import `momwire/__init__.py` to get there.
    py_modules=[
        "momwire_nec2c_client",
        # Its mechanics (#718 phase 2): the shared finding-the-server module
        # both thin clients import — same stdlib-only rule, same reason.
        "momwire_serve_client",
        # The EZNEC leg's thin client (momwire#532), same top-level rule.
        "momwire_eznec_client",
    ],
    package_dir={"": "src/"},
)
