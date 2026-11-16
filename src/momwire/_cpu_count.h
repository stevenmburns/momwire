// How many threads an in-library pool may use (momwire#1952's follow-up).
// Declared with no platform header: the definitions are `_cpu_count_core.cpp`,
// the one TU that includes <windows.h>, whose `near` / `far` / min / max
// macros would otherwise reach the kernels that size pools from these.
#pragma once

namespace mw {
namespace cpu {

// Logical CPUs the affinity mask allows (Linux sched_getaffinity, Windows
// GetProcessAffinityMask); hardware_concurrency() where there is no mask
// (macOS) or it cannot be read. At least 1.
int allowed_threads();

// Physical cores the affinity mask reaches, SMT siblings counted once
// (Linux: the CPUs' thread_siblings_list; Windows: the RelationProcessorCore
// records; macOS: hw.physicalcpu). allowed_threads() where the topology
// cannot be read. At least 1.
int allowed_cores();

// The pool size every in-library team resolves to: MOMWIRE_THREADS when set,
// clamped to allowed_threads(); else allowed_cores(). The crossing cores'
// team cap and the Sommerfeld grid's node pool read it.
int pool_threads();

}  // namespace cpu
}  // namespace mw
