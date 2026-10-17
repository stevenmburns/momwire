// TEMPORARY (proto/razor-sse2-study): per-stage wall-clock accumulators for
// the sse2-vs-avx2 study. Single-threaded use only (MOMWIRE_THREADS=1 runs
// the serial copy). Printed on stderr by study::dump() when MW_STUDY_TIMES=1.
#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace study {
struct Slot {
    const char *name;
    double sec;
    long calls;
};
inline Slot *slots() {
    static Slot s[64];
    return s;
}
inline int &n_slots() {
    static int n = 0;
    return n;
}
inline Slot &slot(const char *name) {
    Slot *s = slots();
    for (int i = 0; i < n_slots(); i++)
        if (s[i].name == name || std::strcmp(s[i].name, name) == 0) return s[i];
    Slot &x = s[n_slots()++];
    x.name = name;
    x.sec = 0;
    x.calls = 0;
    return x;
}
struct Scope {
    Slot &s;
    std::chrono::steady_clock::time_point t;
    explicit Scope(const char *name) : s(slot(name)), t(std::chrono::steady_clock::now()) {}
    ~Scope() {
        s.sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
        s.calls++;
    }
};
inline void dump() {
    const char *e = std::getenv("MW_STUDY_TIMES");
    if (e == nullptr || e[0] != '1') return;
    for (int i = 0; i < n_slots(); i++)
        std::fprintf(stderr, "STUDY %-28s %10.3f ms %8ld\n", slots()[i].name, slots()[i].sec * 1e3,
                     slots()[i].calls);
}
}  // namespace study
#define STUDY_CAT2(a, b) a##b
#define STUDY_CAT(a, b) STUDY_CAT2(a, b)
#define STUDY_T(name) study::Scope STUDY_CAT(study_scope_, __LINE__)(name)
