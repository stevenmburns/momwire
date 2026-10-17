# momwire-razor2p: `momwire-nec5-razor-2p` as one native executable
# (momwire_razor2p.cpp's header says what it does).
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
cmake_minimum_required(VERSION 3.20)
project(momwire_razor2p LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)  # gnu++17, as setup.py builds the extension
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(MW_EXE_NO_PYTHON "build momwire-razor2p-standalone, with no Python in it" OFF)
# Which of the extension's two x86-64 variants the C++ core is compiled as
# (setup.py `_VARIANTS`): avx2 (-mavx2 -mfma), or sse2, the x86-64 baseline
# the extension's `_sse2` modules are (its flags minus the AVX2 ones). The
# Sommerfeld lanes (momwire#1441 step 3) follow the flags, as in the
# extension: __m256d under avx2, two __m128d under sse2. One variant per
# build for now; the in-process choice comes with the Windows work (#1512).
set(MW_EXE_VARIANT "avx2" CACHE STRING "C++ core variant: avx2 or sse2")
set_property(CACHE MW_EXE_VARIANT PROPERTY STRINGS avx2 sse2)
if(NOT MW_EXE_VARIANT STREQUAL "avx2" AND NOT MW_EXE_VARIANT STREQUAL "sse2")
  message(FATAL_ERROR "MW_EXE_VARIANT must be avx2 or sse2")
endif()
set(MW_OPENBLAS_LIB "" CACHE FILEPATH "static OpenBLAS, symbols prefixed mwx_")
# The release build (standalone/build_release_linux.sh): libgomp linked in
# from the toolchain's archive rather than taken from the system, so the
# executable needs nothing but glibc.
option(MW_EXE_STATIC_GOMP "link libgomp statically (Linux)" OFF)
set(MW_DEFAULT_PYTHON_HOME "" CACHE PATH "momwire environment used when MOMWIRE_PYTHON_HOME is unset")
set(MW_LIBPYTHON "" CACHE STRING "Python library opened for the fallback (default: this Python's)")
# The static OpenBLAS's symbol prefix (blas.h): build_openblas.sh's `mwx_` on
# Linux; a Windows static library needs none (no interposition there).
if(WIN32)
  set(_mw_prefix_default "")
else()
  set(_mw_prefix_default "mwx_")
endif()
set(MW_BLAS_PREFIX "${_mw_prefix_default}" CACHE STRING "symbol prefix of MW_OPENBLAS_LIB")
if(NOT MW_OPENBLAS_LIB)
  message(FATAL_ERROR "set MW_OPENBLAS_LIB to an mwx_-prefixed static OpenBLAS")
endif()

if(NOT MW_EXE_NO_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
endif()
if(NOT MSVC)
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
  ${MW_SRC}/_razor_crossing_core.cpp
  ${MW_SRC}/_somm_field_core.cpp
)
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
  set(MW_CORE_FLAGS -O3 -fopenmp-simd -fno-math-errno -ffp-contract=off
      -fno-strict-overflow -fvisibility=hidden)
  if(MW_EXE_VARIANT STREQUAL "avx2")
    list(APPEND MW_CORE_FLAGS -mavx2 -mfma)
  endif()
  set(MW_OMP_FLAG -fopenmp)
endif()
add_library(mw_core OBJECT ${MW_CORE_SOURCES} route.cpp)
target_include_directories(mw_core PUBLIC ${MW_SRC} ${CMAKE_CURRENT_SOURCE_DIR} ${MW_ROOT}/extern/xsf/include)
target_compile_options(mw_core PRIVATE ${MW_CORE_FLAGS} ${MW_OMP_FLAG})
target_compile_definitions(mw_core PRIVATE NDEBUG MW_BLAS_PREFIX=${MW_BLAS_PREFIX})
add_library(mw_core_serial OBJECT ${MW_CORE_SOURCES} route.cpp)
target_include_directories(mw_core_serial PUBLIC ${MW_SRC} ${CMAKE_CURRENT_SOURCE_DIR} ${MW_ROOT}/extern/xsf/include)
target_compile_options(mw_core_serial PRIVATE ${MW_CORE_FLAGS})
target_compile_definitions(mw_core_serial PRIVATE NDEBUG MW_BLAS_PREFIX=${MW_BLAS_PREFIX} mw=mws
  xsf=mws_xsf  # the vendored xsf defines non-inline functions: one copy each
  razor_cplx_brackets=mws_razor_cplx_brackets  # the one free function
  somm_core=mws_somm_core  # the Sommerfeld cores' namespace
  somm_below=mws_somm_below ni_core=mws_ni_core ni_eval=mws_ni_eval)  # the buried fill's cores (momwire#1512)

# main(): no AVX2 here -- it is what asks the CPU before any AVX2 code runs.
if(MW_EXE_NO_PYTHON)
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
  target_compile_options(${MW_EXE} PRIVATE -O2 -fopenmp -ffp-contract=off)
endif()
target_link_libraries(${MW_EXE} PRIVATE ${MW_OPENBLAS_LIB})
if(MSVC)
  if(NOT MW_EXE_NO_PYTHON)
    # pyconfig.h asks MSVC to link pythonXY.lib on its own (#pragma comment);
    # this executable must never link it (it loads the DLL for the fallback).
    target_link_options(${MW_EXE} PRIVATE
      /NODEFAULTLIB:python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}.lib
      /NODEFAULTLIB:python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}_d.lib
      /NODEFAULTLIB:python3.lib)
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

if(MW_EXE_NO_PYTHON)
  message(STATUS "${MW_EXE} ${MW_VERSION} (${MW_EXE_VARIANT}): no Python, OpenBLAS ${MW_OPENBLAS_LIB}")
else()
  message(STATUS "${MW_EXE} ${MW_VERSION} (${MW_EXE_VARIANT}): Python ${Python3_VERSION} (${MW_LIBPYTHON}), OpenBLAS ${MW_OPENBLAS_LIB}")
endif()
