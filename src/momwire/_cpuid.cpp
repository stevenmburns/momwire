// The CPU probe the loader reads before it imports any accelerator variant
// (momwire#1370). Raw registers only: `momwire._accel` decodes the bits, so
// the decoding is plain Python a unit test can feed any register value.
//
// WHY IT EXISTS. On Windows the only OS answer for AVX-512 is
// IsProcessorFeaturePresent(PF_AVX512F_INSTRUCTIONS_AVAILABLE), which speaks
// for AVX-512F alone. The `_avx512` variant is built for F/CD/BW/DQ/VL, and a
// CPU with F but not the rest exists (Knights Landing/Mill: F/CD/ER/PF only),
// so the loader needs CPUID leaf 7 to check the rest and XCR0 to check that
// the OS saves the opmask and zmm state. On Linux /proc/cpuinfo already
// answers both (the kernel clears the avx512* flags when it does not enable
// their xstate); this probe is built there too so the suite can hold its
// decoding to /proc/cpuinfo on real hardware.
//
// BUILD CONTRACT. Compiled once, at each compiler's x86-64 default, with no
// /arch, no -m flag, no OpenMP and no libmvec (setup.py): it is imported on a
// CPU of unknown capability, so it may contain nothing a baseline CPU cannot
// execute. `xgetbv` faults (#UD) unless CPUID.1:ECX.OSXSAVE[bit 27] is set,
// so `xcr0()` checks that bit itself and answers 0 without it.
#include <pybind11/pybind11.h>

#include <cstdint>

#if defined(_MSC_VER)
#include <immintrin.h>
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace py = pybind11;

namespace {

void cpuid_raw(uint32_t leaf, uint32_t subleaf, uint32_t r[4]) {
#if defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    for (int i = 0; i < 4; ++i) r[i] = static_cast<uint32_t>(regs[i]);
#else
    __cpuid_count(leaf, subleaf, r[0], r[1], r[2], r[3]);
#endif
}

py::tuple cpuid(uint32_t leaf, uint32_t subleaf) {
    uint32_t r[4] = {0, 0, 0, 0};
    cpuid_raw(leaf, subleaf, r);
    return py::make_tuple(r[0], r[1], r[2], r[3]);
}

uint64_t xcr0() {
    uint32_t r[4] = {0, 0, 0, 0};
    cpuid_raw(0, 0, r);
    if (r[0] < 1) return 0;
    cpuid_raw(1, 0, r);
    if (!(r[2] & (1u << 27))) return 0;  // no OSXSAVE: xgetbv would fault
#if defined(_MSC_VER)
    return static_cast<uint64_t>(_xgetbv(0));
#else
    uint32_t lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (static_cast<uint64_t>(hi) << 32) | lo;
#endif
}

}  // namespace

PYBIND11_MODULE(_cpuid, m) {
    m.doc() = "CPUID and XCR0, raw, for momwire._accel's variant choice.";
    m.def("cpuid", &cpuid, py::arg("leaf"), py::arg("subleaf") = 0,
          "(eax, ebx, ecx, edx) of CPUID(leaf, subleaf). The caller checks "
          "leaf 0's eax (the highest basic leaf) before asking for more.");
    m.def("xcr0", &xcr0,
          "XCR0 (the OS-enabled xstate components), or 0 when the OS has not "
          "set OSXSAVE and xgetbv is not available.");
}
