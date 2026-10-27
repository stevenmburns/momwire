# momwire-razor2p: `momwire-nec5-razor-2p` as one native executable
# (momwire_razor2p.cpp's header says what it does). -DMW_EXE_BASIS=bs2 with
# MW_EXE_NO_PYTHON builds `momwire-bs2-standalone`, `momwire-nec5-bspline`'s
# twin, from the same cores (momwire#1517). -DMW_EXE_BASIS=sin42 builds the
# NEC-4.2 slot's `momwire-nec42-sinusoidal` (momwire#1607 B4b): the hybrid,
# or with MW_EXE_NO_PYTHON `momwire-nec42-sinusoidal-standalone`.
# -DMW_EXE_BASIS=sg42 with MW_EXE_NO_PYTHON builds
# `momwire-nec42-sinusoidal-galerkin-standalone`, the same slot's
# Sinusoidal-Galerkin launcher (momwire#1718 S4).
#
#   cmake -S standalone -B build-exe -G Ninja \
#         -DMW_OPENBLAS_LIB=/path/to/libmwxp.a \
#         -DMW_DEFAULT_PYTHON_HOME=/path/to/momwire/.venv
#   cmake --build build-exe
#
# MW_OPENBLAS_LIB is a STATIC OpenBLAS whose every global symbol carries the
# `mwx_` prefix (standalone/build_openblas.sh makes one from the commit SciPy's
# wheel bundles), so it can never bind to, or be bound by, the copies numpy and
# scipy load into the same process.
#
# CPython is never linked: its headers give PyConfig's layout, and the library
# is opened at run time only for a deck the C++ route declines. The Python
# found here must be the minor version the momwire environment runs.
#
# -DMW_EXE_NO_PYTHON=ON builds `momwire-razor2p-standalone` instead
# (momwire#1512): no embedded interpreter, no Python headers, no version
# check against an installed momwire; the launcher layer is C++
# (launcher.cpp), and a deck the C++ route declines is a refusal printout
# naming why. No Python is needed to configure or build it.
#
# Windows, no-Python (momwire#1558): MSVC, against SciPy's OpenBLAS DLL from
# the scipy-openblas32 wheel, named directly (an import library is made from
# its exports, below) and shipped beside the executable:
#
#   cmake -S standalone -B build -A x64 -DMW_EXE_NO_PYTHON=ON \
#         -DMW_EXE_BASIS=razor2p|bs2 -DMW_EXE_VARIANT=avx2|sse2 \
#         -DMW_OPENBLAS_LIB=.../scipy_openblas32/lib/libscipy_openblas.dll \
#         -DMW_BLAS_PREFIX=scipy_
#   cmake --build build --config Release
#
# macOS, no-Python (momwire#1559): Apple clang on arm64, against Accelerate
# (the library numpy's and SciPy's macOS wheels link; no MW_OPENBLAS_LIB) and
# Homebrew's libomp (`brew install libomp`; LIBOMP_PREFIX or
# MW_LIBOMP_PREFIX names another keg), one variant (`legacy`):
#
#   cmake -S standalone -B build -DMW_EXE_NO_PYTHON=ON -DMW_EXE_BASIS=razor2p|bs2
#   cmake --build build
cmake_minimum_required(VERSION 3.20)
project(momwire_razor2p LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)  # gnu++17, as setup.py builds the extension
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(MW_EXE_NO_PYTHON "build momwire-razor2p-standalone, with no Python in it" OFF)
# Which formulation the executable answers for (momwire#1517 Phase A): razor2p
# (`momwire-nec5-razor-2p`, route.cpp) or bs2 (`momwire-nec5-bspline`,
# route_bs2.cpp, no-Python builds only). Both from the same cores.
set(MW_EXE_BASIS "razor2p" CACHE STRING "formulation: razor2p, bs2, sin42 or sg42")
set_property(CACHE MW_EXE_BASIS PROPERTY STRINGS razor2p bs2 sin42 sg42)
if(MW_EXE_BASIS STREQUAL "bs2")
  if(NOT MW_EXE_NO_PYTHON)
    message(FATAL_ERROR "MW_EXE_BASIS=bs2 builds the no-Python executable only (MW_EXE_NO_PYTHON=ON)")
  endif()
  set(MW_ROUTE route_bs2.cpp)
elseif(MW_EXE_BASIS STREQUAL "sin42")
  # `momwire-nec42-sinusoidal` (momwire#1607 B4b): the NEC-4.2 slot, point
  # matched; the hybrid (the embedded launcher answers what the C++ route
  # declines) or, with MW_EXE_NO_PYTHON, the no-Python build
  set(MW_ROUTE route_sin.cpp)
elseif(MW_EXE_BASIS STREQUAL "sg42")
  # `momwire-nec42-sinusoidal-galerkin` (momwire#1718 S4): the same slot and
  # route, over the Sinusoidal-Galerkin engine (MW_ROUTE_SG); no-Python only
  if(NOT MW_EXE_NO_PYTHON)
    message(FATAL_ERROR "MW_EXE_BASIS=sg42 builds the no-Python executable only (MW_EXE_NO_PYTHON=ON)")
  endif()
  set(MW_ROUTE route_sin.cpp)
elseif(MW_EXE_BASIS STREQUAL "razor2p")
  set(MW_ROUTE route.cpp)
else()
  message(FATAL_ERROR "MW_EXE_BASIS must be razor2p, bs2, sin42 or sg42")
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
elseif(NOT MW_EXE_VARIANT STREQUAL "avx2" AND NOT MW_EXE_VARIANT STREQUAL "sse2")
  message(FATAL_ERROR "MW_EXE_VARIANT must be avx2 or sse2")
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
if(WIN32)
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
# Windows, no-Python (momwire#1558): MW_OPENBLAS_LIB may name the OpenBLAS DLL
# itself (SciPy's, from the scipy-openblas32 wheel). Its own `.lib` is a MinGW
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
  ${MW_SRC}/_razor_front_core.cpp
  ${MW_SRC}/_razor_buried_core.cpp
  ${MW_SRC}/_razor_exact_core.cpp
  ${MW_SRC}/_exact_kernel_point_core.cpp
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
  ${MW_SRC}/_somm_proj_core.cpp
  ${MW_SRC}/_somm_below_proj_core.cpp
  ${MW_SRC}/_somm_below_grid_core.cpp
  ${MW_SRC}/_somm_below_fill_core.cpp
  ${MW_SRC}/_near_interface_core.cpp
  ${MW_SRC}/_near_interface_eval.cpp
  ${MW_SRC}/_near_interface_point_core.cpp
  ${MW_SRC}/_near_buried_core.cpp
  ${MW_SRC}/_razor_crossing_core.cpp
  ${MW_SRC}/_somm_field_core.cpp
)
if(MW_EXE_BASIS STREQUAL "bs2")
  # bs2 (momwire#1517): the B-spline kernels, the front half, the fill (its
  # exact kernel and Sommerfeld remainder), the NEC-5 seam and run half
  list(APPEND MW_CORE_SOURCES
    ${MW_SRC}/_bspline_core.cpp
    ${MW_SRC}/_exact_kernel_coaxial_core.cpp
    ${MW_SRC}/_somm_remainder_core.cpp
    ${MW_SRC}/_somm_pairs_core.cpp
    ${MW_SRC}/_bspline_front_core.cpp
    ${MW_SRC}/_bspline_fill_core.cpp
    ${MW_SRC}/_bspline_crossing_core.cpp
    ${MW_SRC}/_nec5_bspline_seam_core.cpp
    ${MW_SRC}/_nec5_bspline_printout_core.cpp)
endif()
if(MW_EXE_BASIS STREQUAL "sin42" OR MW_EXE_BASIS STREQUAL "sg42")
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
    ${MW_SRC}/_sinusoidal_crossing_core.cpp
    ${MW_SRC}/_sinusoidal_run_core.cpp)
endif()
if(MW_EXE_BASIS STREQUAL "sg42")
  # the Sinusoidal-Galerkin fill on the B-spline core's sinusoidal kernels,
  # and its answers to the run half (momwire#1718 S2-S3)
  list(APPEND MW_CORE_SOURCES
    ${MW_SRC}/_bspline_core.cpp
    # the Sommerfeld remainder (S5b): bs2's fused block kernel, its graded
    # grazing pairs and their projection
    ${MW_SRC}/_exact_kernel_coaxial_core.cpp
    ${MW_SRC}/_somm_remainder_core.cpp
    ${MW_SRC}/_somm_pairs_core.cpp
    ${MW_SRC}/_somm_pairs_owned_core.cpp
    ${MW_SRC}/_bspline_front_core.cpp
    ${MW_SRC}/_bspline_fill_core.cpp
    ${MW_SRC}/_bspline_crossing_core.cpp
    ${MW_SRC}/_sinusoidal_galerkin_fill_core.cpp
    # the direct-field fill (S7): a wire end on the ground plane, over the
    # Galerkin far fill core (S1) and its complex twin's sweep TU
    ${MW_SRC}/_sinusoidal_galerkin_direct_core.cpp
    ${MW_SRC}/_sinusoidal_galerkin_core.cpp
    ${MW_SRC}/_accel_sinusoidal_cplx.cpp
    ${MW_SRC}/_sinusoidal_galerkin_run_core.cpp)
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
add_library(mw_core OBJECT ${MW_CORE_SOURCES} ${MW_ROUTE})
target_include_directories(mw_core PUBLIC ${MW_SRC} ${CMAKE_CURRENT_SOURCE_DIR} ${MW_ROOT}/extern/xsf/include)
target_compile_options(mw_core PRIVATE ${MW_CORE_FLAGS} ${MW_OMP_FLAG})
target_compile_definitions(mw_core PRIVATE NDEBUG MW_BLAS_PREFIX=${MW_BLAS_PREFIX})
if(MW_EXE_BASIS STREQUAL "sg42")
  # route_sin.cpp answers for the Sinusoidal-Galerkin engine (both copies)
  set(MW_ROUTE_DEFS MW_ROUTE_SG=1)
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
  sg_cplx_phase_sweep=mws_sg_cplx_phase_sweep  # the SG complex fill's (momwire#1718 S7)
  somm_core=mws_somm_core  # the Sommerfeld cores' namespace
  somm_below=mws_somm_below ni_core=mws_ni_core ni_eval=mws_ni_eval  # the buried fill's cores (momwire#1512)
  ni_point=mws_ni_point near_buried=mws_near_buried  # the buried near-field readout (momwire#1653)
  offedge_lanes=mws_offedge_lanes somm_sym=mws_somm_sym  # bs2's (momwire#1517)
  mw_fg=mws_fg)  # the field-form assembler's inline text: one copy per runtime

# main(): no AVX2 here -- it is what asks the CPU before any AVX2 code runs.
if(MW_EXE_NO_PYTHON AND MW_EXE_BASIS STREQUAL "bs2")
  set(MW_EXE momwire-bs2-standalone)
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
elseif(MW_EXE_NO_PYTHON AND MW_EXE_BASIS STREQUAL "sg42")
  set(MW_EXE momwire-nec42-sinusoidal-galerkin-standalone)
  add_executable(${MW_EXE} momwire_razor2p.cpp launcher.cpp
                 $<TARGET_OBJECTS:mw_core> $<TARGET_OBJECTS:mw_core_serial>)
  target_include_directories(${MW_EXE} PRIVATE ${MW_SRC} ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${MW_EXE} PRIVATE
    NDEBUG
    MW_EXE_NO_PYTHON=1
    MW_EXE_DIALECT_NEC4=1
    MW_EXE_VARIANT="${MW_EXE_VARIANT}"
    MW_EXE_BASIS_NAME="sinusoidal-galerkin"
    MW_EXE_PROGRAM="momwire-nec42-sinusoidal-galerkin"
    MW_EXE_LAUNCHER="momwire-nec42-sinusoidal-galerkin"
    MW_BLAS_PREFIX=${MW_BLAS_PREFIX}
    MW_VERSION="${MW_VERSION}")
elseif(MW_EXE_NO_PYTHON)
  set(MW_EXE momwire-razor2p-standalone)
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
  target_link_libraries(${MW_EXE} PRIVATE ${MW_OPENBLAS_LIB})
endif()
if(MSVC AND MW_EXE_NO_PYTHON AND MW_OPENBLAS_DLL)
  # The Windows release (momwire#1558): the OpenBLAS DLL ships beside the
  # executable (SciPy's, scipy-openblas32; MW_BLAS_PREFIX=scipy_), and is
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

if(MW_ACCELERATE)
  message(STATUS "${MW_EXE} ${MW_VERSION} (${MW_EXE_VARIANT}): no Python, Accelerate")
elseif(MW_EXE_NO_PYTHON)
  message(STATUS "${MW_EXE} ${MW_VERSION} (${MW_EXE_VARIANT}): no Python, OpenBLAS ${MW_OPENBLAS_LIB}")
else()
  message(STATUS "${MW_EXE} ${MW_VERSION} (${MW_EXE_VARIANT}): Python ${Python3_VERSION} (${MW_LIBPYTHON}), OpenBLAS ${MW_OPENBLAS_LIB}")
endif()
