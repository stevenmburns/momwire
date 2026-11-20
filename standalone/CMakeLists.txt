# momwire-razor2p: `momwire-nec5-razor-2p` as one native executable
# (momwire_razor2p.cpp's header says what it does). -DMW_EXE_BASIS=bspline_nec5 with
# MW_EXE_NO_PYTHON builds `momwire-nec5-bspline-standalone`, `momwire-nec5-bspline`'s
# twin, from the same cores (momwire#1517). -DMW_EXE_BASIS=sin42 builds the
# NEC-4.2 slot's `momwire-nec42-sinusoidal` (momwire#1607 B4b): the hybrid,
# or with MW_EXE_NO_PYTHON `momwire-nec42-sinusoidal-standalone`.
# -DMW_EXE_BASIS=bspline_nec42 with
# MW_EXE_NO_PYTHON builds `momwire-nec42-bspline-standalone`, the slot's default
# launcher `momwire-nec42` (nec42-bspline, momwire#1607 Phase C2).
#
#   cmake -S standalone -B build-exe -G Ninja \
#         -DMW_OPENBLAS_LIB=/path/to/libmwxp.a \
#         -DMW_OPENBLAS_NUMPY_LIB=/path/to/libmwnp.a \
#         -DMW_DEFAULT_PYTHON_HOME=/path/to/momwire/.venv
#   cmake --build build-exe
#
# MW_OPENBLAS_LIB is a STATIC OpenBLAS whose every global symbol carries the
# `mwx_` prefix (standalone/build_openblas.sh makes one from the commit SciPy's
# wheel bundles), and MW_OPENBLAS_NUMPY_LIB the same from numpy's, prefixed
# `mwn_` (momwire#1751), so neither can bind to, or be bound by, the copies
# numpy and scipy load into the same process.
#
# CPython is never linked: its headers give PyConfig's layout, and the library
# is opened at run time only for a deck the C++ route declines. The Python
# found here must be the minor version the momwire environment runs.
#
# -DMW_EXE_NO_PYTHON=ON builds `momwire-nec5-razor-2p-standalone` instead
# (momwire#1512): no embedded interpreter, no Python headers, no version
# check against an installed momwire; the launcher layer is C++
# (launcher.cpp), and a deck the C++ route declines is a refusal printout
# naming why. No Python is needed to configure or build it.
#
# Windows, no-Python (momwire#1558): MSVC, against numpy's OpenBLAS DLL from
# the scipy-openblas32 wheel, named directly (an import library is made from
# its exports, below) and shipped beside the executable:
#
#   cmake -S standalone -B build -A x64 -DMW_EXE_NO_PYTHON=ON \
#         -DMW_EXE_BASIS=razor2p|bspline_nec5 -DMW_EXE_VARIANT=avx2|sse2 \
#         -DMW_OPENBLAS_LIB=.../scipy_openblas32/lib/libscipy_openblas.dll \
#         -DMW_BLAS_PREFIX=scipy_
#   cmake --build build --config Release
#
# macOS, no-Python (momwire#1559): Apple clang on arm64, against Accelerate
# (the library numpy's and SciPy's macOS wheels link; no MW_OPENBLAS_LIB) and
# Homebrew's libomp (`brew install libomp`; LIBOMP_PREFIX or
# MW_LIBOMP_PREFIX names another keg), one variant (`legacy`):
#
#   cmake -S standalone -B build -DMW_EXE_NO_PYTHON=ON -DMW_EXE_BASIS=razor2p|bspline_nec5
#   cmake --build build
cmake_minimum_required(VERSION 3.20)
project(momwire_razor2p LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)  # gnu++17, as setup.py builds the extension
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(MW_EXE_NO_PYTHON "build momwire-nec5-razor-2p-standalone, with no Python in it" OFF)
# Which formulation the executable answers for (momwire#1517 Phase A): razor2p
# (`momwire-nec5-razor-2p`, route.cpp) or bs2 (`momwire-nec5-bspline`,
# route_bs2.cpp, no-Python builds only). Both from the same cores.
set(MW_EXE_BASIS "razor2p" CACHE STRING "formulation: razor2p, bspline_nec5, bspline_nec42 or sin42")
set_property(CACHE MW_EXE_BASIS PROPERTY STRINGS razor2p bspline_nec5 bspline_nec42 sin42)
if(MW_EXE_BASIS STREQUAL "bspline_nec5")
  if(NOT MW_EXE_NO_PYTHON)
    message(FATAL_ERROR "MW_EXE_BASIS=bspline_nec5 builds the no-Python executable only (MW_EXE_NO_PYTHON=ON)")
  endif()
  set(MW_ROUTE route_bs2.cpp)
elseif(MW_EXE_BASIS STREQUAL "sin42")
  # `momwire-nec42-sinusoidal` (momwire#1607 B4b): the NEC-4.2 slot, point
  # matched; the hybrid (the embedded launcher answers what the C++ route
  # declines) or, with MW_EXE_NO_PYTHON, the no-Python build
  set(MW_ROUTE route_sin.cpp)
elseif(MW_EXE_BASIS STREQUAL "bspline_nec42")
  # `momwire-nec42` (momwire#1607 Phase C2): the NEC-4.2 slot's default
  # basis, bs2 on the slot's run half, over bs2's route (MW_ROUTE_NEC4);
  # no-Python only
  if(NOT MW_EXE_NO_PYTHON)
    message(FATAL_ERROR "MW_EXE_BASIS=bspline_nec42 builds the no-Python executable only (MW_EXE_NO_PYTHON=ON)")
  endif()
  set(MW_ROUTE route_bs2.cpp)
elseif(MW_EXE_BASIS STREQUAL "razor2p")
  set(MW_ROUTE route.cpp)
else()
  message(FATAL_ERROR "MW_EXE_BASIS must be razor2p, bspline_nec5, bspline_nec42 or sin42")
endif()
# Which of the extension's two x86-64 variants the C++ core is compiled as
# (setup.py `_VARIANTS`): avx2 (-mavx2 -mfma), or sse2, the x86-64 baseline
# the extension's `_sse2` modules are (its flags minus the AVX2 ones). The
# Sommerfeld lanes (momwire#1441 step 3) follow the flags, as in the
# extension: __m256d under avx2, two __m128d under sse2. One variant per
# build for now; the in-process choice comes with the Windows work (#1512).
# macOS (momwire#1559) has one: `legacy`, the unsuffixed extension setup.py's
# darwin branch builds (no AVX flag on either arch), and the label
# `momwire.accelerator_variant` stamps on its printouts.
if(APPLE)
  set(_mw_variant_default "legacy")
else()
  set(_mw_variant_default "avx2")
endif()
set(MW_EXE_VARIANT "${_mw_variant_default}" CACHE STRING "C++ core variant: avx2, sse2 or (macOS) legacy")
set_property(CACHE MW_EXE_VARIANT PROPERTY STRINGS avx2 sse2 legacy)
if(APPLE)
  if(NOT MW_EXE_VARIANT STREQUAL "legacy")
    message(FATAL_ERROR "MW_EXE_VARIANT on macOS is legacy (setup.py's darwin branch)")
  endif()
elseif(MW_EXE_VARIANT STREQUAL "fat")
  # Both x86-64 variants in one executable, picked at run time by CPU
  # (momwire_razor2p.cpp `pick_variant`): the cores compiled four times --
  # sse2 threaded / serial as `mw` / `mws`, the objects every CPU runs, and
  # avx2 threaded / serial as `mwa` / `mwas`, two static libraries linked
  # AFTER every object. Their functions outside the renamed namespaces (a
  # template or inline function every copy instantiates) are renamed
  # `mwa$<name>` (fat_private_comdats.py), so the avx2 code calls avx2 bodies
  # and the sse2 code can never reach one; a library member is pulled in only
  # for a symbol no object defines. Each avx2 TU's dynamic initializers are
  # moved out of the C runtime's start-up list (fat_avx2_init.h) and run by
  # main() only once it has picked avx2.
  # MinGW-w64 (the single-file experiment, windows-static.yml) instead links
  # the avx2 copies as one --whole-archive library whose `.ctors` sections
  # defer_ctors.cmake renames (PE grouped-section sort), and keeps the shared
  # names: its gate (fat_compare.py) holds fat == one-variant byte for byte.
  # Windows no-Python builds only.
  if(NOT (MSVC OR MINGW) OR NOT MW_EXE_NO_PYTHON)
    message(FATAL_ERROR "MW_EXE_VARIANT=fat is the Windows (MSVC or MinGW-w64) no-Python build only")
  endif()
  set(MW_EXE_FAT ON)
elseif(NOT MW_EXE_VARIANT STREQUAL "avx2" AND NOT MW_EXE_VARIANT STREQUAL "sse2")
  message(FATAL_ERROR "MW_EXE_VARIANT must be avx2, sse2 or (Windows no-Python) fat")
endif()
set(MW_OPENBLAS_LIB "" CACHE FILEPATH "static OpenBLAS, symbols prefixed mwx_")
# The release build (standalone/build_release_linux.sh): libgomp linked in
# from the toolchain's archive rather than taken from the system, so the
# executable needs nothing but glibc.
option(MW_EXE_STATIC_GOMP "link libgomp statically (Linux)" OFF)
set(MW_DEFAULT_PYTHON_HOME "" CACHE PATH "momwire environment used when MOMWIRE_PYTHON_HOME is unset")
set(MW_LIBPYTHON "" CACHE STRING "Python library opened for the fallback (default: this Python's)")
# macOS, no-Python (momwire#1559): the BLAS and LAPACK are Accelerate's, the
# library numpy and SciPy link on macOS 14+ arm64 (both wheels' __config__),
# reached through blas_accelerate_*.c under the prefix `mwa_`; nothing ships
# beside the executable for it. MW_OPENBLAS_LIB is then unused.
set(MW_ACCELERATE OFF)
if(APPLE AND MW_EXE_NO_PYTHON)
  set(MW_ACCELERATE ON)
  enable_language(C)
endif()
# The static OpenBLAS's symbol prefix (blas.h): build_openblas.sh's `mwx_` on
# Linux; a Windows static library needs none (no interposition there).
if(WIN32 AND NOT MINGW)
  set(_mw_prefix_default "")
elseif(MW_ACCELERATE)
  set(_mw_prefix_default "mwa_")
else()
  set(_mw_prefix_default "mwx_")
endif()
set(MW_BLAS_PREFIX "${_mw_prefix_default}" CACHE STRING "symbol prefix of MW_OPENBLAS_LIB")
if(NOT MW_OPENBLAS_LIB AND NOT MW_ACCELERATE)
  message(FATAL_ERROR "set MW_OPENBLAS_LIB to an mwx_-prefixed static OpenBLAS")
endif()
# numpy's OpenBLAS beside SciPy's (momwire#1751, blas.h): build_openblas.sh's
# libmwnp.a, every symbol `mwn_`. The products and numpy's solve (MWN) then
# run on numpy's release and the LAPACK (MWX) on SciPy's, as the launcher's
# do. Unset (Windows, macOS, a one-library build) MWN is MWX.
set(MW_OPENBLAS_NUMPY_LIB "" CACHE FILEPATH "numpy's static OpenBLAS, symbols prefixed mwn_")
set(MW_BLAS_NP_PREFIX "mwn_" CACHE STRING "symbol prefix of MW_OPENBLAS_NUMPY_LIB")
if(MW_OPENBLAS_NUMPY_LIB)
  if(MW_BLAS_NP_PREFIX STREQUAL MW_BLAS_PREFIX)
    message(FATAL_ERROR "MW_BLAS_NP_PREFIX must differ from MW_BLAS_PREFIX")
  endif()
  add_compile_definitions(MW_BLAS_NP_PREFIX=${MW_BLAS_NP_PREFIX} MW_BLAS_TWO_LIBS=1)
endif()
# Windows, no-Python (momwire#1558): MW_OPENBLAS_LIB may name the OpenBLAS DLL
# itself (numpy's release, from the scipy-openblas32 wheel). Its own `.lib` is a MinGW
# import library, which MSVC links but cannot delay-load, so the import
# library is made here from the DLL's export table, in MSVC's short-import
# form: `dumpbin /exports` to a .def, `lib /def` to the .lib.
if(MSVC AND MW_OPENBLAS_LIB MATCHES "\\.[dD][lL][lL]$")
  get_filename_component(_mw_tools "${CMAKE_LINKER}" DIRECTORY)
  find_program(MW_DUMPBIN dumpbin HINTS "${_mw_tools}" REQUIRED)
  get_filename_component(MW_OPENBLAS_DLL "${MW_OPENBLAS_LIB}" NAME)
  get_filename_component(_mw_blas_stem "${MW_OPENBLAS_LIB}" NAME_WE)
  execute_process(COMMAND "${MW_DUMPBIN}" /nologo /exports "${MW_OPENBLAS_LIB}"
                  OUTPUT_VARIABLE _mw_exports RESULT_VARIABLE _mw_rc)
  if(NOT _mw_rc EQUAL 0)
    message(FATAL_ERROR "dumpbin /exports ${MW_OPENBLAS_LIB} failed")
  endif()
  # "  ordinal hint RVA      name" rows
  string(REGEX MATCHALL "\n +[0-9]+ +[0-9A-F]+ [0-9A-F]+ [A-Za-z_][A-Za-z0-9_@]*" _mw_rows "${_mw_exports}")
  set(_mw_def "LIBRARY ${MW_OPENBLAS_DLL}\nEXPORTS\n")
  foreach(_row IN LISTS _mw_rows)
    string(REGEX REPLACE ".* ([^ ]+)$" "\\1" _sym "${_row}")
    string(APPEND _mw_def "    ${_sym}\n")
  endforeach()
  list(LENGTH _mw_rows _mw_n)
  if(_mw_n LESS 100)
    message(FATAL_ERROR "${MW_OPENBLAS_LIB}: only ${_mw_n} exports read")
  endif()
  set(_mw_implib "${CMAKE_CURRENT_BINARY_DIR}/${_mw_blas_stem}.lib")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${_mw_blas_stem}.def" "${_mw_def}")
  execute_process(COMMAND "${CMAKE_AR}" /nologo /machine:x64
                          "/def:${CMAKE_CURRENT_BINARY_DIR}/${_mw_blas_stem}.def" "/out:${_mw_implib}"
                  RESULT_VARIABLE _mw_rc OUTPUT_QUIET)
  if(NOT _mw_rc EQUAL 0)
    message(FATAL_ERROR "lib /def for ${MW_OPENBLAS_DLL} failed")
  endif()
  message(STATUS "${MW_OPENBLAS_DLL}: ${_mw_n} exports, import library ${_mw_implib}")
  set(MW_OPENBLAS_LIB "${_mw_implib}")
endif()

if(NOT MW_EXE_NO_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
endif()
if(APPLE)
  # Apple clang has no OpenMP runtime: Homebrew's libomp, as setup.py's
  # darwin branch takes it (LIBOMP_PREFIX), passed through the driver with
  # -Xpreprocessor. Linked from its static archive where the keg has one, so
  # nothing ships beside the executable for it.
  set(MW_LIBOMP_PREFIX "$ENV{LIBOMP_PREFIX}" CACHE PATH "Homebrew libomp prefix (macOS)")
  if(NOT MW_LIBOMP_PREFIX)
    set(MW_LIBOMP_PREFIX "/opt/homebrew/opt/libomp")
  endif()
  if(EXISTS "${MW_LIBOMP_PREFIX}/lib/libomp.a")
    set(MW_LIBOMP_LIB "${MW_LIBOMP_PREFIX}/lib/libomp.a")
  elseif(EXISTS "${MW_LIBOMP_PREFIX}/lib/libomp.dylib")
    set(MW_LIBOMP_LIB "${MW_LIBOMP_PREFIX}/lib/libomp.dylib")
  else()
    message(FATAL_ERROR "no libomp under ${MW_LIBOMP_PREFIX} (brew install libomp)")
  endif()
  message(STATUS "OpenMP: ${MW_LIBOMP_LIB}")
elseif(NOT MSVC)
  # MSVC takes /openmp:llvm on the compile line (setup.py's runtime); its
  # FindOpenMP answer is plain /openmp (vcomp), a second runtime.
  find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

# The release this executable is: pyproject.toml's version, read at configure
# time and compared at run time against the installed momwire's.
get_filename_component(MW_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
file(STRINGS "${MW_ROOT}/pyproject.toml" _mw_version_line REGEX "^version = \"[^\"]+\"" LIMIT_COUNT 1)
string(REGEX REPLACE "^version = \"([^\"]+)\"" "\\1" MW_VERSION "${_mw_version_line}")
if(NOT MW_VERSION)
  message(FATAL_ERROR "no version in ${MW_ROOT}/pyproject.toml")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${MW_ROOT}/pyproject.toml")

if(NOT MW_LIBPYTHON AND NOT MW_EXE_NO_PYTHON)
  if(WIN32)
    set(MW_LIBPYTHON "python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}.dll")
  elseif(APPLE)
    set(MW_LIBPYTHON "libpython${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}.dylib")
  else()
    set(MW_LIBPYTHON "libpython${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}.so.1.0")
  endif()
endif()

# The Python-free core: the TUs tests/test_razor_core_standalone.py builds,
# with setup.py's flags for this platform (the extension's own), so the C++
# route's bits are the extension's.
set(MW_SRC "${MW_ROOT}/src/momwire")
set(MW_CORE_SOURCES
  ${MW_SRC}/_razor_core.cpp
  ${MW_SRC}/_razor_t2_core.cpp
  ${MW_SRC}/_accel_razor_cplx.cpp
  ${MW_SRC}/_razor_solve_core.cpp
  ${MW_SRC}/_razor_pair_reuse_core.cpp  # momwire#1971: fill-pair reuse
  ${MW_SRC}/_pair_reuse_core.cpp  # momwire#2063: its common core
  ${MW_SRC}/_razor_front_core.cpp
  ${MW_SRC}/_razor_buried_core.cpp
  ${MW_SRC}/_razor_exact_core.cpp
  ${MW_SRC}/_exact_kernel_point_core.cpp
  ${MW_SRC}/_exact_kernel_series_core.cpp  # momwire#1454: `exact_kernel_series`
  ${MW_SRC}/_nec5_deck_core.cpp
  ${MW_SRC}/_nec5_seam_core.cpp
  ${MW_SRC}/_nec5_razor_seam_core.cpp
  ${MW_SRC}/_nec5_printout_core.cpp
  ${MW_SRC}/_nec5_razor_printout_core.cpp
  ${MW_SRC}/_razor_wire_loading_core.cpp
  # The composing Sommerfeld ground (momwire#1479 S3): the six-integral
  # kernel, the grid, and the remainder's projection and moments.
  ${MW_SRC}/_somm_kernel_core.cpp
  ${MW_SRC}/_somm_grid_core.cpp
  ${MW_SRC}/_cpu_count_core.cpp  # the teams' CPU counts (the affinity mask)
  ${MW_SRC}/_somm_proj_core.cpp
  ${MW_SRC}/_somm_below_proj_core.cpp
  ${MW_SRC}/_somm_below_grid_core.cpp
  ${MW_SRC}/_somm_below_fill_core.cpp
  ${MW_SRC}/_near_interface_core.cpp
  ${MW_SRC}/_near_interface_eval.cpp
  ${MW_SRC}/_crossing_table_core.cpp  # momwire#1860 L5's table (`crossing_table`)
  ${MW_SRC}/_near_interface_point_core.cpp
  ${MW_SRC}/_near_buried_core.cpp
  ${MW_SRC}/_razor_crossing_core.cpp
  ${MW_SRC}/_somm_field_core.cpp
)
if(MW_EXE_BASIS STREQUAL "bspline_nec5" OR MW_EXE_BASIS STREQUAL "bspline_nec42" OR MW_EXE_BASIS STREQUAL "sin42")
  # the transmitted family (momwire#1718 S12): the below->above table and its
  # projection, which a mixed deck's cross blocks read on every basis that
  # fills them
  list(APPEND MW_CORE_SOURCES ${MW_SRC}/_somm_transmitted_core.cpp)
endif()
if(MW_EXE_BASIS STREQUAL "bspline_nec5" OR MW_EXE_BASIS STREQUAL "bspline_nec42")
  # bs2 (momwire#1517): the B-spline kernels, the front half, the fill (its
  # exact kernel and Sommerfeld remainder), the NEC-5 seam and run half
  list(APPEND MW_CORE_SOURCES
    ${MW_SRC}/_bspline_core.cpp
    ${MW_SRC}/_exact_kernel_coaxial_core.cpp
    ${MW_SRC}/_somm_remainder_core.cpp
    ${MW_SRC}/_somm_pairs_core.cpp
    ${MW_SRC}/_bspline_front_core.cpp
    ${MW_SRC}/_bspline_fill_core.cpp
    ${MW_SRC}/_bspline_transmitted_core.cpp
    ${MW_SRC}/_bspline_crossing_core.cpp
    ${MW_SRC}/_nec5_bspline_seam_core.cpp
    ${MW_SRC}/_nec5_bspline_printout_core.cpp)
endif()
if(MW_EXE_BASIS STREQUAL "sin42" OR MW_EXE_BASIS STREQUAL "bspline_nec42")
  # the NEC-4.2 slot's reader, seam, build, printout writer and run half, and
  # the sinusoidal basis' point-matched kernels, front half, fill and answers
  # (momwire#1607 A1-B4a)
  list(APPEND MW_CORE_SOURCES
    ${MW_SRC}/_nec4_deck_core.cpp
    ${MW_SRC}/_nec4_seam_core.cpp
    ${MW_SRC}/_nec4_build_core.cpp
    ${MW_SRC}/_nec4_printout_core.cpp
    ${MW_SRC}/_nec4_run_core.cpp
    ${MW_SRC}/_sinusoidal_core.cpp
    ${MW_SRC}/_sinusoidal_nec4_core.cpp
    ${MW_SRC}/_sinusoidal_front_core.cpp
    ${MW_SRC}/_sinusoidal_fill_core.cpp
    ${MW_SRC}/_sin_pair_reuse_core.cpp  # momwire#1911: `pair_reuse` (requested only)
    ${MW_SRC}/_sinusoidal_transmitted_core.cpp
    ${MW_SRC}/_sinusoidal_crossing_core.cpp
    ${MW_SRC}/_sinusoidal_run_core.cpp)
endif()
if(MW_EXE_BASIS STREQUAL "sin42" OR MW_EXE_BASIS STREQUAL "bspline_nec42")
  # the point-matched fill's opt-in staged complex rows (momwire#1849,
  # `point_cplx_staged`), which `_sinusoidal_fill_core.cpp` above calls: the
  # complex Galerkin far fill's tile, and its libmvec sweep in its own TU
  # (its `omp declare simd` is TU-wide)
  list(APPEND MW_CORE_SOURCES
    ${MW_SRC}/_sinusoidal_galerkin_core.cpp
    ${MW_SRC}/_accel_sinusoidal_cplx.cpp)
endif()
if(MW_EXE_BASIS STREQUAL "bspline_nec42")
  # bs2's answers to the NEC-4.2 slot's run half (momwire#1607 Phase C1)
  list(APPEND MW_CORE_SOURCES ${MW_SRC}/_bspline_nec4_run_core.cpp)
endif()
# Two copies: `mw_core` (+ route.cpp) with OpenMP, the threaded fill; and
# `mw_core_serial`, the same sources with `mw` renamed `mws` and no OpenMP
# runtime at all (`omp simd` stays: -fopenmp-simd / MSVC's simd no-ops), so a
# one-thread answer never initialises the OpenMP runtime (route.h).
if(MSVC)
  # M_PI for every TU (as setup.py): a core's first standard header can come
  # before `_mw_core.h` defines _USE_MATH_DEFINES.
  set(MW_CORE_FLAGS /O2 /fp:fast /std:c++17 /D_USE_MATH_DEFINES=)
  if(MW_EXE_VARIANT STREQUAL "avx2")
    list(APPEND MW_CORE_FLAGS /arch:AVX2)
  endif()
  set(MW_OMP_FLAG /openmp:llvm)
else()
  if(APPLE)
    # Apple's driver takes OpenMP options only through -Xpreprocessor
    set(MW_CORE_FLAGS -O3 "SHELL:-Xpreprocessor -fopenmp-simd" -fno-math-errno -ffp-contract=off
        -fno-strict-overflow -fvisibility=hidden)
  else()
    set(MW_CORE_FLAGS -O3 -fopenmp-simd -fno-math-errno -ffp-contract=off
        -fno-strict-overflow -fvisibility=hidden)
  endif()
  if(MW_EXE_VARIANT STREQUAL "avx2")
    list(APPEND MW_CORE_FLAGS -mavx2 -mfma)
  endif()
  if(APPLE)
    set(MW_OMP_FLAG "SHELL:-Xpreprocessor -fopenmp" -I${MW_LIBOMP_PREFIX}/include)
  else()
    set(MW_OMP_FLAG -fopenmp)
  endif()
endif()
# Extra flags for the C++ cores, appended last so they win over the defaults
# above (gcc takes the last -ffp-contract=; cl the last /fp:). Empty in every
# shipped build: the experiments in windows-static.yml set it (an MSVC build
# at /fp:precise, a gcc one at -ffp-contract=fast) to measure which part of
# the code generation a fill-time gap comes from.
set(MW_EXE_EXTRA_CORE_FLAGS "" CACHE STRING "experiment only: flags appended to the C++ cores")
if(MW_EXE_EXTRA_CORE_FLAGS)
  separate_arguments(_mw_extra NATIVE_COMMAND "${MW_EXE_EXTRA_CORE_FLAGS}")
  list(APPEND MW_CORE_FLAGS ${_mw_extra})
endif()
add_library(mw_core OBJECT ${MW_CORE_SOURCES} ${MW_ROUTE})
target_include_directories(mw_core PUBLIC ${MW_SRC} ${CMAKE_CURRENT_SOURCE_DIR} ${MW_ROOT}/extern/xsf/include)
target_compile_options(mw_core PRIVATE ${MW_CORE_FLAGS} ${MW_OMP_FLAG})
target_compile_definitions(mw_core PRIVATE NDEBUG MW_BLAS_PREFIX=${MW_BLAS_PREFIX})
if(MW_EXE_BASIS STREQUAL "bspline_nec42")
  # route_bs2.cpp answers for the NEC-4.2 slot (both copies)
  set(MW_ROUTE_DEFS MW_ROUTE_NEC4=1)
endif()
target_compile_definitions(mw_core PRIVATE ${MW_ROUTE_DEFS})
add_library(mw_core_serial OBJECT ${MW_CORE_SOURCES} ${MW_ROUTE})
target_include_directories(mw_core_serial PUBLIC ${MW_SRC} ${CMAKE_CURRENT_SOURCE_DIR} ${MW_ROOT}/extern/xsf/include)
target_compile_options(mw_core_serial PRIVATE ${MW_CORE_FLAGS})
target_compile_definitions(mw_core_serial PRIVATE ${MW_ROUTE_DEFS})
target_compile_definitions(mw_core_serial PRIVATE NDEBUG MW_BLAS_PREFIX=${MW_BLAS_PREFIX} mw=mws
  xsf=mws_xsf  # the vendored xsf defines non-inline functions: one copy each
  Faddeeva=mws_Faddeeva  # xsf/faddeeva.h keeps its own namespace, outside xsf (momwire#1595 D1b)
  razor_cplx_brackets=mws_razor_cplx_brackets  # the one free function
  sg_cplx_phase_sweep=mws_sg_cplx_phase_sweep  # and the staged rows' sweep's (momwire#1849)
  somm_core=mws_somm_core  # the Sommerfeld cores' namespace
  somm_below=mws_somm_below ni_core=mws_ni_core ni_eval=mws_ni_eval  # the buried fill's cores (momwire#1512)
  somm_trans=mws_somm_trans  # the transmitted table's (momwire#1718 S12)
  ni_point=mws_ni_point near_buried=mws_near_buried  # the buried near-field readout (momwire#1653)
  offedge_lanes=mws_offedge_lanes somm_sym=mws_somm_sym  # bs2's (momwire#1517)
  mw_fg=mws_fg  # the field-form assembler's inline text: one copy per runtime
  mw1860=mws_mw1860)  # the crossing table's inline text (momwire#1860 L5): likewise

# The fat build's avx2 copies (MW_EXE_VARIANT=fat, above): the same sources
# and flags plus /arch:AVX2 (setup.py's avx2 variant), every namespace the
# serial copy renames renamed again, `mwa` / `mwas`; archived, after
# fat_private_comdats.py renames their functions outside those namespaces,
# as two static libraries linked after the sse2 objects. /FI fat_avx2_init.h
# moves each TU's dynamic initializers into `.mwav$m`, which the C runtime
# never runs (/wd4075: the warning that says so).
if(MW_EXE_FAT AND MINGW)
  # MinGW-w64: -mavx2 -mfma for /arch:AVX2; the constructors are deferred
  # at archive time (defer_ctors.cmake, below) rather than by a forced header.
  foreach(_mw_copy IN ITEMS mwa mwas)
    set(_mw_renames mw=${_mw_copy} xsf=${_mw_copy}_xsf Faddeeva=${_mw_copy}_Faddeeva
        razor_cplx_brackets=${_mw_copy}_razor_cplx_brackets somm_core=${_mw_copy}_somm_core
        somm_below=${_mw_copy}_somm_below ni_core=${_mw_copy}_ni_core ni_eval=${_mw_copy}_ni_eval
        somm_trans=${_mw_copy}_somm_trans ni_point=${_mw_copy}_ni_point
        near_buried=${_mw_copy}_near_buried offedge_lanes=${_mw_copy}_offedge_lanes
        somm_sym=${_mw_copy}_somm_sym mw_fg=${_mw_copy}_fg mw1860=${_mw_copy}_mw1860)
    add_library(mw_core_${_mw_copy} OBJECT ${MW_CORE_SOURCES} ${MW_ROUTE})
    target_include_directories(mw_core_${_mw_copy} PUBLIC ${MW_SRC} ${CMAKE_CURRENT_SOURCE_DIR} ${MW_ROOT}/extern/xsf/include)
    target_compile_options(mw_core_${_mw_copy} PRIVATE ${MW_CORE_FLAGS} -mavx2 -mfma)
    if(_mw_copy STREQUAL "mwa")
      target_compile_options(mw_core_${_mw_copy} PRIVATE ${MW_OMP_FLAG})
    endif()
    target_compile_definitions(mw_core_${_mw_copy} PRIVATE NDEBUG MW_BLAS_PREFIX=${MW_BLAS_PREFIX}
                               ${MW_ROUTE_DEFS} ${_mw_renames})
  endforeach()
elseif(MW_EXE_FAT)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  find_program(MW_LLVM_OBJCOPY llvm-objcopy HINTS "$ENV{ProgramFiles}/LLVM/bin" REQUIRED)
  find_program(MW_LLVM_NM llvm-nm HINTS "$ENV{ProgramFiles}/LLVM/bin" REQUIRED)
  foreach(_mw_copy IN ITEMS mwa mwas)
    set(_mw_renames mw=${_mw_copy} xsf=${_mw_copy}_xsf Faddeeva=${_mw_copy}_Faddeeva
        razor_cplx_brackets=${_mw_copy}_razor_cplx_brackets sg_cplx_phase_sweep=${_mw_copy}_sg_cplx_phase_sweep
        somm_core=${_mw_copy}_somm_core
        somm_below=${_mw_copy}_somm_below ni_core=${_mw_copy}_ni_core ni_eval=${_mw_copy}_ni_eval
        somm_trans=${_mw_copy}_somm_trans ni_point=${_mw_copy}_ni_point
        near_buried=${_mw_copy}_near_buried offedge_lanes=${_mw_copy}_offedge_lanes
        somm_sym=${_mw_copy}_somm_sym mw_fg=${_mw_copy}_fg mw1860=${_mw_copy}_mw1860)
    add_library(mw_core_${_mw_copy} OBJECT ${MW_CORE_SOURCES} ${MW_ROUTE})
    target_include_directories(mw_core_${_mw_copy} PUBLIC ${MW_SRC} ${CMAKE_CURRENT_SOURCE_DIR} ${MW_ROOT}/extern/xsf/include)
    target_compile_options(mw_core_${_mw_copy} PRIVATE ${MW_CORE_FLAGS} /arch:AVX2
                           "/FI${CMAKE_CURRENT_SOURCE_DIR}/fat_avx2_init.h" /wd4075)
    if(_mw_copy STREQUAL "mwa")
      target_compile_options(mw_core_${_mw_copy} PRIVATE ${MW_OMP_FLAG})
    endif()
    target_compile_definitions(mw_core_${_mw_copy} PRIVATE NDEBUG MW_BLAS_PREFIX=${MW_BLAS_PREFIX}
                               ${MW_ROUTE_DEFS} ${_mw_renames})
  endforeach()
  set(MW_FAT_LIBS "${CMAKE_CURRENT_BINARY_DIR}/fat/mw_core_mwa.lib" "${CMAKE_CURRENT_BINARY_DIR}/fat/mw_core_mwas.lib")
  add_custom_command(
    OUTPUT ${MW_FAT_LIBS}
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/fat_private_comdats.py"
            --objcopy "${MW_LLVM_OBJCOPY}" --nm "${MW_LLVM_NM}" --lib "${CMAKE_AR}"
            --out "${CMAKE_CURRENT_BINARY_DIR}/fat"
            "mw_core_mwa=${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/mw_core_mwa.dir"
            "mw_core_mwas=${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/mw_core_mwas.dir"
    DEPENDS mw_core_mwa mw_core_mwas $<TARGET_OBJECTS:mw_core_mwa> $<TARGET_OBJECTS:mw_core_mwas>
            "${CMAKE_CURRENT_SOURCE_DIR}/fat_private_comdats.py"
    COMMENT "fat: the avx2 copies' shared-name functions made their own"
    VERBATIM)
  add_custom_target(mw_core_fat_libs DEPENDS ${MW_FAT_LIBS})
endif()

# main(): no AVX2 here -- it is what asks the CPU before any AVX2 code runs.
if(MW_EXE_NO_PYTHON AND MW_EXE_BASIS STREQUAL "bspline_nec5")
  set(MW_EXE momwire-nec5-bspline-standalone)
  add_executable(${MW_EXE} momwire_razor2p.cpp launcher.cpp
                 $<TARGET_OBJECTS:mw_core> $<TARGET_OBJECTS:mw_core_serial>)
  target_include_directories(${MW_EXE} PRIVATE ${MW_SRC} ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${MW_EXE} PRIVATE
    NDEBUG
    MW_EXE_NO_PYTHON=1
    MW_EXE_VARIANT="${MW_EXE_VARIANT}"
    MW_EXE_BASIS_NAME="bspline"
    MW_EXE_PROGRAM="momwire-bs2"
    MW_EXE_LAUNCHER="momwire-nec5-bspline"
    MW_BLAS_PREFIX=${MW_BLAS_PREFIX}
    MW_VERSION="${MW_VERSION}")
elseif(MW_EXE_NO_PYTHON AND MW_EXE_BASIS STREQUAL "bspline_nec42")
  set(MW_EXE momwire-nec42-bspline-standalone)
  add_executable(${MW_EXE} momwire_razor2p.cpp launcher.cpp
                 $<TARGET_OBJECTS:mw_core> $<TARGET_OBJECTS:mw_core_serial>)
  target_include_directories(${MW_EXE} PRIVATE ${MW_SRC} ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${MW_EXE} PRIVATE
    NDEBUG
    MW_EXE_NO_PYTHON=1
    MW_EXE_DIALECT_NEC4=1
    MW_EXE_VARIANT="${MW_EXE_VARIANT}"
    MW_EXE_BASIS_NAME="bspline"
    MW_EXE_PROGRAM="momwire-nec42"
    MW_EXE_LAUNCHER="momwire-nec42"
    MW_BLAS_PREFIX=${MW_BLAS_PREFIX}
    MW_VERSION="${MW_VERSION}")
elseif(MW_EXE_NO_PYTHON AND MW_EXE_BASIS STREQUAL "sin42")
  set(MW_EXE momwire-nec42-sinusoidal-standalone)
  add_executable(${MW_EXE} momwire_razor2p.cpp launcher.cpp
                 $<TARGET_OBJECTS:mw_core> $<TARGET_OBJECTS:mw_core_serial>)
  target_include_directories(${MW_EXE} PRIVATE ${MW_SRC} ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${MW_EXE} PRIVATE
    NDEBUG
    MW_EXE_NO_PYTHON=1
    MW_EXE_DIALECT_NEC4=1
    MW_EXE_VARIANT="${MW_EXE_VARIANT}"
    MW_EXE_BASIS_NAME="sinusoidal"
    MW_EXE_PROGRAM="momwire-nec42-sinusoidal"
    MW_EXE_LAUNCHER="momwire-nec42-sinusoidal"
    MW_BLAS_PREFIX=${MW_BLAS_PREFIX}
    MW_VERSION="${MW_VERSION}")
elseif(MW_EXE_NO_PYTHON)
  set(MW_EXE momwire-nec5-razor-2p-standalone)
  add_executable(${MW_EXE} momwire_razor2p.cpp launcher.cpp
                 $<TARGET_OBJECTS:mw_core> $<TARGET_OBJECTS:mw_core_serial>)
  target_include_directories(${MW_EXE} PRIVATE ${MW_SRC} ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${MW_EXE} PRIVATE
    NDEBUG
    MW_EXE_NO_PYTHON=1
    MW_EXE_VARIANT="${MW_EXE_VARIANT}"
    MW_BLAS_PREFIX=${MW_BLAS_PREFIX}
    MW_VERSION="${MW_VERSION}")
elseif(MW_EXE_BASIS STREQUAL "sin42")
  set(MW_EXE momwire-nec42-sinusoidal)
  add_executable(${MW_EXE} momwire_razor2p.cpp $<TARGET_OBJECTS:mw_core> $<TARGET_OBJECTS:mw_core_serial>)
  target_include_directories(${MW_EXE} PRIVATE ${MW_SRC} ${Python3_INCLUDE_DIRS})
  target_compile_definitions(${MW_EXE} PRIVATE
    NDEBUG
    MW_EXE_DIALECT_NEC4=1
    MW_EXE_VARIANT="${MW_EXE_VARIANT}"
    MW_EXE_BASIS_NAME="sinusoidal"
    MW_EXE_PROGRAM="momwire-nec42-sinusoidal"
    MW_EXE_LAUNCHER="momwire-nec42-sinusoidal"
    MW_BLAS_PREFIX=${MW_BLAS_PREFIX}
    MW_VERSION="${MW_VERSION}"
    MW_LIBPYTHON="${MW_LIBPYTHON}")
  if(MW_DEFAULT_PYTHON_HOME)
    target_compile_definitions(${MW_EXE} PRIVATE MW_DEFAULT_PYTHON_HOME="${MW_DEFAULT_PYTHON_HOME}")
  endif()
else()
  set(MW_EXE momwire-razor2p)
  add_executable(${MW_EXE} momwire_razor2p.cpp $<TARGET_OBJECTS:mw_core> $<TARGET_OBJECTS:mw_core_serial>)
  target_include_directories(${MW_EXE} PRIVATE ${MW_SRC} ${Python3_INCLUDE_DIRS})
  target_compile_definitions(${MW_EXE} PRIVATE
    NDEBUG
    MW_EXE_VARIANT="${MW_EXE_VARIANT}"
    MW_BLAS_PREFIX=${MW_BLAS_PREFIX}
    MW_VERSION="${MW_VERSION}"
    MW_LIBPYTHON="${MW_LIBPYTHON}")
  if(MW_DEFAULT_PYTHON_HOME)
    target_compile_definitions(${MW_EXE} PRIVATE MW_DEFAULT_PYTHON_HOME="${MW_DEFAULT_PYTHON_HOME}")
  endif()
endif()
# A CERTIFICATION build (momwire#1911): the features named here also run at
# this executable's level, as a raised reach would let them, so the
# executable's own bytes can be held to Python's before Steve raises a reach
# (the route has to exist at the level first). Each name must have a C++
# route this basis carries (EXE_CERTIFYING in momwire_razor2p.cpp). No
# release build sets it: shipped, a feature past its reach is refused.
set(MW_EXE_CERTIFY "" CACHE STRING "certification build only: features run past their reach (pair_reuse, sin42)")
foreach(_mw_f IN LISTS MW_EXE_CERTIFY)
  if(_mw_f STREQUAL "pair_reuse" AND MW_EXE_BASIS STREQUAL "sin42")
    target_compile_definitions(${MW_EXE} PRIVATE MW_EXE_CERTIFY_PAIR_REUSE=1)
  else()
    message(FATAL_ERROR "MW_EXE_CERTIFY: ${_mw_f} has no C++ route on MW_EXE_BASIS=${MW_EXE_BASIS}")
  endif()
endforeach()
if(MW_EXE_FAT AND MINGW)
  # The avx2 copies as one archive whose constructors are moved out of the
  # start-up list (defer_ctors.cmake), linked whole, after the sse2 copies
  # (see MW_EXE_VARIANT=fat) and before OpenBLAS, which they call.
  set(_mw_avx2_lib "${CMAKE_CURRENT_BINARY_DIR}/libmw_avx2_deferred.a")
  add_custom_command(OUTPUT "${_mw_avx2_lib}"
    COMMAND ${CMAKE_COMMAND} -E rm -f "${_mw_avx2_lib}"
    COMMAND ${CMAKE_AR} qc "${_mw_avx2_lib}" $<TARGET_OBJECTS:mw_core_mwa> $<TARGET_OBJECTS:mw_core_mwas>
    COMMAND ${CMAKE_COMMAND} "-DARCHIVE=${_mw_avx2_lib}" "-DOBJCOPY=${CMAKE_OBJCOPY}" "-DOBJDUMP=${CMAKE_OBJDUMP}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/defer_ctors.cmake"
    COMMAND ${CMAKE_RANLIB} "${_mw_avx2_lib}"
    DEPENDS mw_core_mwa mw_core_mwas $<TARGET_OBJECTS:mw_core_mwa> $<TARGET_OBJECTS:mw_core_mwas>
            "${CMAKE_CURRENT_SOURCE_DIR}/defer_ctors.cmake"
    COMMAND_EXPAND_LISTS VERBATIM)
  add_custom_target(mw_avx2_deferred DEPENDS "${_mw_avx2_lib}")
  add_dependencies(${MW_EXE} mw_avx2_deferred)
  target_link_libraries(${MW_EXE} PRIVATE -Wl,--whole-archive "${_mw_avx2_lib}" -Wl,--no-whole-archive)
  target_compile_definitions(${MW_EXE} PRIVATE MW_EXE_FAT=1)
elseif(MW_EXE_FAT)
  # after the sse2 objects (see MW_EXE_VARIANT=fat); the map file is what
  # scripts/windows_fat/check_link.py reads to show which copy each symbol is
  add_dependencies(${MW_EXE} mw_core_fat_libs)
  target_link_libraries(${MW_EXE} PRIVATE ${MW_FAT_LIBS})
  target_compile_definitions(${MW_EXE} PRIVATE MW_EXE_FAT=1)
  target_link_options(${MW_EXE} PRIVATE "/MAP:${CMAKE_CURRENT_BINARY_DIR}/${MW_EXE}.map")
endif()
if(MSVC)
  target_compile_options(${MW_EXE} PRIVATE /O2 /std:c++17 /openmp:llvm /D_USE_MATH_DEFINES=)
else()
  # -ffp-contract=off: the launcher's .GMO grid is numpy's arithmetic
  target_compile_options(${MW_EXE} PRIVATE -O2 ${MW_OMP_FLAG} -ffp-contract=off)
endif()
if(MW_ACCELERATE)
  target_sources(${MW_EXE} PRIVATE blas_accelerate_lp64.c blas_accelerate_ilp64.c)
  target_link_libraries(${MW_EXE} PRIVATE "-framework Accelerate")
else()
  target_link_libraries(${MW_EXE} PRIVATE ${MW_OPENBLAS_LIB} ${MW_OPENBLAS_NUMPY_LIB})
endif()
if(MSVC AND MW_EXE_NO_PYTHON AND MW_OPENBLAS_DLL)
  # The Windows release (momwire#1558): the OpenBLAS DLL ships beside the
  # executable (numpy's release, scipy-openblas32; MW_BLAS_PREFIX=scipy_), and is
  # delay-loaded, so main() sets its thread count and core type before its
  # load-time constructor reads them (momwire_razor2p.cpp,
  # `blas_env_before_load`), and `-version` answers without loading it.
  target_link_options(${MW_EXE} PRIVATE /DELAYLOAD:${MW_OPENBLAS_DLL})
  target_link_libraries(${MW_EXE} PRIVATE delayimp)
elseif(MSVC)
  if(NOT MW_EXE_NO_PYTHON)
    # pyconfig.h asks MSVC to link pythonXY.lib on its own (#pragma comment);
    # this executable must never link it (it loads the DLL for the fallback).
    target_link_options(${MW_EXE} PRIVATE
      /NODEFAULTLIB:python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}.lib
      /NODEFAULTLIB:python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}_d.lib
      /NODEFAULTLIB:python3.lib)
  endif()
elseif(APPLE)
  target_link_libraries(${MW_EXE} PRIVATE "${MW_LIBOMP_LIB}")
  if(MW_LIBOMP_LIB MATCHES "\\.dylib$")
    # the dylib ships beside the executable
    set_target_properties(${MW_EXE} PROPERTIES BUILD_RPATH "@executable_path" INSTALL_RPATH "@executable_path")
  endif()
elseif(MW_EXE_STATIC_GOMP)
  # Compiled -fopenmp (above), linked without it: the archive, not -lgomp.
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libgomp.a
                  OUTPUT_VARIABLE MW_GOMP_ARCHIVE OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT EXISTS "${MW_GOMP_ARCHIVE}")
    message(FATAL_ERROR "MW_EXE_STATIC_GOMP: the compiler has no libgomp.a")
  endif()
  target_link_libraries(${MW_EXE} PRIVATE "${MW_GOMP_ARCHIVE}")
else()
  target_link_libraries(${MW_EXE} PRIVATE OpenMP::OpenMP_CXX)
endif()
if(UNIX AND NOT APPLE)
  # libstdc++ / libgcc static (the spike's cold-start measurement); libc,
  # libm, libmvec and (unless MW_EXE_STATIC_GOMP) libgomp stay shared, as
  # the extension has them.
  target_link_libraries(${MW_EXE} PRIVATE -static-libstdc++ -static-libgcc ${CMAKE_DL_LIBS} pthread m mvec)
endif()
if(MINGW)
  # One file (windows-static.yml): every runtime from its archive -- libgomp
  # (MW_EXE_STATIC_GOMP), libstdc++, libgcc and winpthreads, which libgomp and
  # the OpenBLAS archives need -- so nothing ships beside the executable.
  target_link_options(${MW_EXE} PRIVATE -static -static-libgcc -static-libstdc++)
  target_link_libraries(${MW_EXE} PRIVATE pthread)
  # MW_EXE_LIBM_SHIM (the libm experiment, WINDOWS-SINGLE-FILE.md): the
  # executable's exp/sin/cos/... forwarded to a system DLL's (msvcrt.dll or
  # ucrtbase.dll) instead of mingw-w64's own, which win the link otherwise.
  set(MW_EXE_LIBM_SHIM "" CACHE STRING "MinGW: forward libm to msvcrt or ucrtbase (empty: mingw-w64's own)")
  if(MW_EXE_LIBM_SHIM)
    enable_language(C)
    target_sources(${MW_EXE} PRIVATE mingw_libm_shim.c)
    target_compile_definitions(${MW_EXE} PRIVATE MW_LIBM_SHIM_DLL="${MW_EXE_LIBM_SHIM}.dll")
  endif()
  # MW_EXE_VMATH_SHIM (optimisation 4, WINDOWS-SINGLE-FILE.md): the avx2 cores'
  # `omp declare simd` blocks switched on, their vector sin / cos / exp calls
  # answered by mingw_vmath_shim.cpp. Off by default: an unset build compiles
  # what it compiled before. MW_EXE_VMATH_SHIM_COUNT adds the call counters
  # (the proof that it ran; never timed).
  option(MW_EXE_VMATH_SHIM "MinGW avx2: vector sin/cos/exp over mw_vmath" OFF)
  option(MW_EXE_VMATH_SHIM_COUNT "MinGW: count the shim's calls, report at exit" OFF)
  if(MW_EXE_VMATH_SHIM)
    if(MW_EXE_VARIANT STREQUAL "sse2")
      message(FATAL_ERROR "MW_EXE_VMATH_SHIM needs the avx2 cores (MW_EXE_VARIANT=avx2 or fat)")
    endif()
    add_library(mw_vmath_shim OBJECT mingw_vmath_shim.cpp)
    target_include_directories(mw_vmath_shim PRIVATE ${MW_SRC})
    target_compile_options(mw_vmath_shim PRIVATE ${MW_CORE_FLAGS} -mavx2 -mfma)
    if(MW_EXE_VMATH_SHIM_COUNT)
      target_compile_definitions(mw_vmath_shim PRIVATE MW_VMATH_SHIM_COUNT=1)
    endif()
    target_sources(${MW_EXE} PRIVATE $<TARGET_OBJECTS:mw_vmath_shim>)
    # The guards also ask for __AVX2__, so a fat build's sse2 copies are unchanged.
    foreach(_mw_t IN ITEMS mw_core mw_core_serial mw_core_mwa mw_core_mwas)
      if(TARGET ${_mw_t})
        target_compile_definitions(${_mw_t} PRIVATE MW_MINGW_VMATH_SHIM=1)
      endif()
    endforeach()
  endif()
endif()

if(MW_ACCELERATE)
  message(STATUS "${MW_EXE} ${MW_VERSION} (${MW_EXE_VARIANT}): no Python, Accelerate")
elseif(MW_EXE_NO_PYTHON)
  message(STATUS "${MW_EXE} ${MW_VERSION} (${MW_EXE_VARIANT}): no Python, OpenBLAS ${MW_OPENBLAS_LIB} ${MW_OPENBLAS_NUMPY_LIB}")
else()
  message(STATUS "${MW_EXE} ${MW_VERSION} (${MW_EXE_VARIANT}): Python ${Python3_VERSION} (${MW_LIBPYTHON}), OpenBLAS ${MW_OPENBLAS_LIB} ${MW_OPENBLAS_NUMPY_LIB}")
endif()
