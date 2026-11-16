// `_cpu_count.h`'s definitions. The one TU here that includes <windows.h>
// (its macros stay out of the kernels); no float arithmetic.
//
// The standalone executable's own `hardware_threads` / `physical_cores`
// (momwire#1952, standalone/momwire_razor2p.cpp) read the same masks for its
// OpenMP and OpenBLAS pools; these are the in-library pools' counterparts.
#include "_cpu_count.h"

#include <algorithm>
#include <cstdlib>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <vector>
#elif defined(__linux__)
#include <sched.h>

#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

namespace mw {
namespace cpu {

namespace {

int hardware() {
    const unsigned n = std::thread::hardware_concurrency();
    return n > 0 ? static_cast<int>(n) : 1;
}

#if defined(__linux__)
// A CPU's thread_siblings_list, read once per CPU per process: the topology
// does not change under a running process, the mask can (it is read on every
// call). "" when sysfs does not say.
std::string siblings(int cpu) {
    static std::mutex mu;
    static std::map<int, std::string> seen;
    const std::lock_guard<std::mutex> lock(mu);
    const auto it = seen.find(cpu);
    if (it != seen.end()) return it->second;
    std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
    std::string s;
    if (f.is_open()) s.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    seen.emplace(cpu, s);
    return s;
}
#endif

}  // namespace

int allowed_threads() {
#if defined(_WIN32)
    // The process mask covers one processor group and reads 0 for a process
    // spread over several (>64 CPUs): then every CPU.
    DWORD_PTR proc = 0, sys = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys) && proc != 0) {
        int n = 0;
        for (DWORD_PTR m = proc; m != 0; m &= m - 1) n++;
        return n;
    }
#elif defined(__linux__)
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof set, &set) == 0 && CPU_COUNT(&set) > 0) return CPU_COUNT(&set);
#endif
    return hardware();
}

int allowed_cores() {
#if defined(_WIN32)
    DWORD_PTR proc = 0, sys = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys)) proc = 0;
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<char> buf(len);
    if (len == 0 || !GetLogicalProcessorInformationEx(
                        RelationProcessorCore,
                        reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data()), &len))
        return allowed_threads();
    // A core counts when its group-0 mask meets the process mask (#1952's
    // reading); a mask of 0 (several groups) counts every core.
    int cores = 0;
    for (DWORD off = 0; off < len;) {
        auto *p = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data() + off);
        if (p->Relationship == RelationProcessorCore) {
            const GROUP_AFFINITY &g = p->Processor.GroupMask[0];
            if (proc == 0 || (g.Group == 0 && (g.Mask & proc) != 0)) cores++;
        }
        off += p->Size;
    }
    return cores > 0 ? cores : allowed_threads();
#elif defined(__linux__)
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof set, &set) != 0 || CPU_COUNT(&set) <= 0) return hardware();
    std::vector<std::string> cores;
    for (int c = 0; c < CPU_SETSIZE; c++) {
        if (!CPU_ISSET(c, &set)) continue;
        const std::string s = siblings(c);
        if (s.empty()) return CPU_COUNT(&set);  // no topology: every allowed CPU
        if (std::find(cores.begin(), cores.end(), s) == cores.end()) cores.push_back(s);
    }
    return cores.empty() ? CPU_COUNT(&set) : static_cast<int>(cores.size());
#elif defined(__APPLE__)
    // no affinity masks, and no SMT on Apple silicon
    int cores = 0;
    std::size_t len = sizeof cores;
    if (sysctlbyname("hw.physicalcpu", &cores, &len, nullptr, 0) != 0 || cores <= 0) return hardware();
    return cores;
#else
    return hardware();
#endif
}

int pool_threads() {
    // MOMWIRE_THREADS is the caller's count for every pool; a mask narrower
    // than it would only queue the extra threads behind the allowed CPUs.
    if (const char *mt = std::getenv("MOMWIRE_THREADS")) {
        const long v = std::strtol(mt, nullptr, 10);
        if (v > 0) return static_cast<int>(std::min<long>(v, allowed_threads()));
    }
    return std::max(1, allowed_cores());
}

}  // namespace cpu
}  // namespace mw
