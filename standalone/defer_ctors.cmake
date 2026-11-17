# The fat build's avx2 copies (MW_EXE_VARIANT=fat, MinGW): their static
# constructors moved out of the start-up list, run by main() only once it
# has picked avx2 (momwire_razor2p.cpp `run_avx2_constructors`).
#
#   cmake -DARCHIVE=libmw_avx2_deferred.a -DOBJCOPY=... -DOBJDUMP=... -P defer_ctors.cmake
#
# WHY. A constructor in an avx2 TU is compiled -mavx2 like the rest of it, and
# the C runtime runs every `.ctors` entry before main(): on a CPU without AVX
# the fat executable died there (Intel SDE as Westmere, windows-static.yml:
# `vmovups [rsp+0x20], xmm6`, a VEX-encoded prologue). Each `.ctors[.N]`
# section becomes `.data$mwactors_m[.N]`, which the PE linker sorts between
# the two markers main() defines (`.data$mwactors_a`, `.data$mwactors_z`); a
# `.dtors[.N]` section, should one appear, becomes `.data$mwadtors_m[.N]`,
# never run. Section names only: no code or data changes.
if(NOT ARCHIVE OR NOT OBJCOPY OR NOT OBJDUMP)
  message(FATAL_ERROR "defer_ctors.cmake: set ARCHIVE, OBJCOPY and OBJDUMP")
endif()
execute_process(COMMAND "${OBJDUMP}" -h "${ARCHIVE}" OUTPUT_VARIABLE _h RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "objdump -h ${ARCHIVE} failed")
endif()
string(REGEX MATCHALL "[ \t]\\.(c|d)tors(\\.[0-9]+)?[ \t]" _names "${_h}")
set(_seen "")
set(_args "")
set(_n 0)
foreach(_raw IN LISTS _names)
  string(STRIP "${_raw}" _s)
  math(EXPR _n "${_n} + 1")
  if(_s IN_LIST _seen)
    continue()
  endif()
  list(APPEND _seen "${_s}")
  string(REGEX REPLACE "^\\.(c|d)tors" "" _tail "${_s}")
  if(_s MATCHES "^\\.ctors")
    list(APPEND _args "--rename-section" "${_s}=.data$mwactors_m${_tail}")
  else()
    list(APPEND _args "--rename-section" "${_s}=.data$mwadtors_m${_tail}")
  endif()
endforeach()
if(_args)
  execute_process(COMMAND "${OBJCOPY}" ${_args} "${ARCHIVE}" RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "objcopy on ${ARCHIVE} failed")
  endif()
endif()
message(STATUS "defer_ctors: ${_n} constructor/destructor sections (${_seen}) moved out of start-up in ${ARCHIVE}")
