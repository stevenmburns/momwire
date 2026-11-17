// Forced (/FI) on every TU of the fat MSVC build's avx2 copies
// (MW_EXE_VARIANT=fat, CMakeLists.txt), before any of its own text.
//
// WHY. Those TUs are compiled /arch:AVX2, their dynamic initializers (a
// namespace-scope object with a constructor, or a value computed at start-up)
// included, and the C runtime runs every `.CRT$XCU` entry before main(): on a
// CPU without AVX the fat executable would fault before it could ask the CPU
// anything (the MinGW build did exactly that under Intel SDE as Westmere).
// `init_seg` puts this TU's initializer pointers in `.mwav$m` instead, which
// the C runtime does not know; the linker sorts it between the two markers
// momwire_razor2p.cpp defines (`.mwav$a`, `.mwav$z`), and `pick_variant` runs
// the entries (`run_avx2_initializers`) only once it has picked avx2.
// Placement only: no code changes. scripts/windows_fat/check_link.py holds
// the linked image to it (no start-up entry reaches an avx2 copy).
#pragma once
#pragma section(".mwav$m", read)
#pragma init_seg(".mwav$m")
