// momwire-razor2p: `momwire-nec5-razor-2p` as one native executable.
//
//     momwire-razor2p DECK PRINTOUT        (EZNEC's and SimNEC's form)
//     momwire-razor2p < PATHS              (4nec2's: the two names on stdin)
//     momwire-razor2p - -                  (SimNEC's streaming call, momwire#1512)
//     momwire-razor2p -version
//
// The contract is the launcher's (`momwire.eznec._shell`): the printout file
// is the channel, exit 0. Two routes answer it, and they write the same bytes:
//
//   * C++: the deck reader, the structure half, the front half, the fill, the
//     LU and the run half (`_nec5_razor_printout_core.h`), for every deck it
//     covers. No Python is loaded.
//   * the fallback: everything else -- a declined deck (`route == declined`),
//     a deck asking for a `.GMO` file, the stdin form, `-version`, an
//     argument error, a deck that cannot be read. CPython is LOADED here, not
//     linked (dlopen / LoadLibrary), from the momwire environment beside the
//     executable, and runs the launcher's own one-shot path, `_shell.main`,
//     in this process.
//
// ONE BLAS / LAPACK. The executable links its own OpenBLAS (static, its
// symbols prefixed `mwx_`, built from the commit SciPy's wheel bundles) and
// uses it for the C++ route's LU and products. Before the fallback imports
// momwire it publishes the same routines as `sys._mw_host_lapack` (ABI 2,
// `momwire._lapack`), so Python's LU and its run-path products run on them
// too, and the C++ route replays `_lapack`'s calls rather than numpy's.
//
// VERSION. The executable is one momwire release, compiled in. It reads the
// installed momwire's version the way `momwire_serve_client` does (the first
// `momwire-*.dist-info` / `.egg-info` in site-packages) before answering any
// deck, and REFUSES, in the printout, on a mismatch: it never runs one
// release's C++ beside another's Python.
//
// THREADS. The executable sets its own count on its OpenBLAS and on OpenMP:
// MOMWIRE_THREADS, else by the deck's size for the C++ route (`threads_for`:
// 1, 4, or one per physical core) and every hardware thread for the fallback
// (the launcher's default), exported as OPENBLAS_NUM_THREADS /
// OMP_NUM_THREADS before CPython starts so the embedded numpy and scipy read
// it.
//
// Gate hooks (environment; never set in use):
//   MW_EXE_ROUTE_LOG=FILE    append "<route> <deck>" (cxx | fallback:<why>)
//   MW_EXE_FORCE_FALLBACK=1  answer every deck through the fallback
//   MW_EXE_ARRAYS=DIR        the C++ route writes each finished run's arrays
//   MW_EXE_PYSTARTUP=FILE    the fallback execs FILE before answering
//   MW_EXE_TIMES=1           a timing line on stderr
//
// MOMWIRE_TIMES=1 (for users, MOMWIRE_TIMES): the printout's MATRIX TIMING
// (FILL, FACTOR) and RUN TIME carry measured seconds instead of zeros, and
// each answer writes one `times:` line on stderr, its stages in milliseconds
// (`_mw_times.h`). Off, nothing is timed and the printout is unchanged.
//   MW_EXE_COPY=threads      one thread through the OpenMP copy, not the serial one
//   MW_EXE_FAIL=alloc|runtime  (no-Python build) throw from the solve, to test
//                            the INTERNAL ERROR printout
#include "_mw_times.h"
#include "_nec5_deck_core.h"
#include "_nec5_razor_printout_core.h"
#include "route.h"
#define mw mws
#include "route.h"
#undef mw

#if !MW_EXE_NO_PYTHON
#define PY_SSIZE_T_CLEAN
#include <Python.h>  // declarations and PyConfig's layout only; NOT linked
#endif

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <intrin.h>
#else
#include <dirent.h>
#include <sched.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#if !MW_EXE_NO_PYTHON
#include <dlfcn.h>
#endif
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/time.h>
#endif
#include <time.h>
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

#ifndef MW_VERSION
#error "MW_VERSION (the momwire release this executable is) must be defined"
#endif

#include "blas.h"

extern "C" void omp_set_num_threads(int);

namespace {

// The formulation this executable answers for, as the printout's stamp
// names it (CMakeLists.txt's MW_EXE_BASIS), and the program's own name.
#ifndef MW_EXE_BASIS_NAME
#define MW_EXE_BASIS_NAME "razor-2p"
#endif
#ifndef MW_EXE_PROGRAM
#define MW_EXE_PROGRAM "momwire-razor2p"
#endif
const char *const BASIS = MW_EXE_BASIS_NAME;
const char *const VERSION = MW_VERSION;
// The C++ core's variant, as the printout's stamp names it (CMakeLists.txt).
#ifndef MW_EXE_VARIANT
#define MW_EXE_VARIANT "avx2"
#endif
const char *const EXE_VARIANT = MW_EXE_VARIANT;

using Clock = std::chrono::steady_clock;
const Clock::time_point T0 = Clock::now();

double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// --- MOMWIRE_TIMES ------------------------------------------------------------

bool read_file(const std::string &path, std::string &out);

// How long ago the OS created this process, in ms (-1: not known). Linux
// reads /proc (10 ms ticks), macOS its process table, Windows the process's
// creation time; a reader's `start_ms` is this less the time since main().
double process_age_ms() {
#if defined(_WIN32)
    FILETIME created, exited, kernel, user, now;
    if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) return -1.0;
    GetSystemTimeAsFileTime(&now);  // Windows 7 has no precise form
    const auto u = [](const FILETIME &t) {
        return (static_cast<unsigned long long>(t.dwHighDateTime) << 32) | t.dwLowDateTime;
    };
    const unsigned long long a = u(created), b = u(now);
    return b > a ? static_cast<double>(b - a) / 1e4 : 0.0;  // 100 ns units
#elif defined(__APPLE__)
    struct kinfo_proc info;
    std::size_t len = sizeof info;
    int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, static_cast<int>(getpid())};
    if (sysctl(mib, 4, &info, &len, nullptr, 0) != 0 || len == 0) return -1.0;
    struct timeval now;
    gettimeofday(&now, nullptr);
    const struct timeval &st = info.kp_proc.p_starttime;
    const double age = (now.tv_sec - st.tv_sec) * 1e3 + (now.tv_usec - st.tv_usec) / 1e3;
    return age > 0 ? age : 0.0;
#else
    std::string stat;
    if (!read_file("/proc/self/stat", stat)) return -1.0;
    const std::size_t paren = stat.rfind(')');
    if (paren == std::string::npos) return -1.0;
    std::istringstream in(stat.substr(paren + 1));
    std::string field;
    for (int k = 3; k <= 22 && (in >> field); k++)
        if (k == 22) {
            const double ticks = std::strtod(field.c_str(), nullptr);
            const long hz = sysconf(_SC_CLK_TCK);
            struct timespec boot;
            if (hz <= 0 || clock_gettime(CLOCK_BOOTTIME, &boot) != 0) return -1.0;
            const double age = (boot.tv_sec + boot.tv_nsec / 1e9 - ticks / hz) * 1e3;
            return age > 0 ? age : 0.0;
        }
    return -1.0;
#endif
}

// MOMWIRE_TIMES's clock: `on`, the process's start (RUN TIME's zero), main()'s
// entry, and where the next `times:` line's total starts.
struct Times {
    bool on = false;
    Clock::time_point origin, main_at, line_from;
    double start_ms = 0.0;  // the first line's; 0 after it
};
Times g_times;

void times_begin() {
    if (!mw_times::env_on()) return;
    g_times.on = true;
    g_times.main_at = Clock::now();
    const double age = process_age_ms();
    // unknown: the static initialisers' clock (T0), which is near enough
    const double before_main = age >= 0.0 ? std::max(age, ms_since(T0)) : ms_since(T0);
    g_times.origin = g_times.main_at - std::chrono::duration_cast<Clock::duration>(
                                           std::chrono::duration<double, std::milli>(before_main));
    g_times.start_ms = before_main;
    g_times.line_from = g_times.origin;
    mw_times::enable(g_times.origin);
}

// The threads the next answer's fill and factor run on (the printout's
// MOMWIRE_TIMES lines report them).
void times_threads(int fill, int factor) {
    mw_times::state().fill_threads = fill;
    mw_times::state().factor_threads = factor;
}

// One answer's `times:` line on stderr; the next one's total starts here.
void times_line(const char *route, int threads, double load_ms, double write_ms) {
    if (!g_times.on) return;
    const std::string line =
        mw_times::take_line(route, threads, g_times.start_ms, load_ms, write_ms, g_times.line_from);
    std::fprintf(stderr, "%s\n", line.c_str());
    std::fflush(stderr);
    g_times.start_ms = 0.0;
    g_times.line_from = Clock::now();
}

std::string env(const char *name) {
    const char *v = std::getenv(name);
    return v == nullptr ? std::string() : std::string(v);
}

void set_env(const char *name, const std::string &value) {
#if defined(_WIN32)
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

bool read_file(const std::string &path, std::string &out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return !f.bad();
}

void log_route(const std::string &route, const std::string &deck) {
    const std::string file = env("MW_EXE_ROUTE_LOG");
    if (file.empty()) return;
    if (std::FILE *w = std::fopen(file.c_str(), "ab")) {
        std::fprintf(w, "%s %s\n", route.c_str(), deck.c_str());
        std::fclose(w);
    }
}

// The fallback's thread count (MOMWIRE_THREADS, else every hardware thread:
// the launcher's own default, which its thread-sensitive printouts need).
int fallback_threads = 1;

int hardware_threads() {
    const int n = static_cast<int>(std::thread::hardware_concurrency());
    return n > 0 ? n : 1;
}

// Physical cores this process may run on (SMT siblings counted once), else
// the hardware threads. The C++ route never runs more threads than this: its
// two pools (the OpenMP fill, the OpenBLAS LU) each spin between their
// regions, and on SMT siblings they cost ~80-100 ms a call at every size
// (Rocketlake, 16 threads on 8 cores, unit 5).
int physical_cores() {
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<char> buf(len);
    auto *info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data());
    if (len == 0 || !GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len))
        return hardware_threads();
    int cores = 0;
    for (DWORD off = 0; off < len;) {
        auto *p = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data() + off);
        cores += p->Relationship == RelationProcessorCore;
        off += p->Size;
    }
    return cores > 0 ? cores : hardware_threads();
#elif defined(__APPLE__)
    // macOS (momwire#1559): no affinity masks and no SMT on Apple silicon;
    // the kernel's count of physical cores
    int cores = 0;
    std::size_t len = sizeof cores;
    if (sysctlbyname("hw.physicalcpu", &cores, &len, nullptr, 0) != 0 || cores <= 0)
        return hardware_threads();
    return cores;
#else
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof set, &set) != 0) return hardware_threads();
    std::vector<std::string> seen;
    for (int c = 0; c < CPU_SETSIZE; c++) {
        if (!CPU_ISSET(c, &set)) continue;
        std::string siblings;
        const std::string f = "/sys/devices/system/cpu/cpu" + std::to_string(c) +
                              "/topology/thread_siblings_list";
        if (!read_file(f, siblings)) return CPU_COUNT(&set);
        if (std::find(seen.begin(), seen.end(), siblings) == seen.end()) seen.push_back(siblings);
    }
    return seen.empty() ? hardware_threads() : static_cast<int>(seen.size());
#endif
}

// The size policy, from the largest solve's basis functions (unit 5's sweep,
// Rocketlake, 8 cores / 16 threads, one process per call): up to ~100 one
// thread is fastest (dip101 2.7 ms; 4 threads 2.9, 8 threads 3.5), so under
// 128 the deck runs on the serial copy and never starts OpenMP; from ~150
// to ~300 four threads win (dip301 8.2 ms against 14.6 on one and 8.4 on
// eight); from ~400 every core (dip801 36 ms on 8, 42 on 4, 99 on 1).
int threads_for(long long n_basis) {
    const int cores = physical_cores();
    if (n_basis < 128) return 1;
    if (n_basis < 384) return std::min(4, cores);
    return cores;
}

// The executable's OpenBLAS, OpenMP (only when `omp`: the serial route never
// enters the OpenMP runtime) and, for an embedded Python, the environment
// its numpy and scipy read.
void set_threads(int n, bool omp) {
    MWX(openblas_set_num_threads)(n);
    if (omp) omp_set_num_threads(n);
    set_env("OPENBLAS_NUM_THREADS", std::to_string(n));
    set_env("OMP_NUM_THREADS", std::to_string(n));
}

// OpenBLAS sizes its thread pool in a load-time constructor (gotoblas_init),
// from OPENBLAS_NUM_THREADS, else every CPU -- before main() can choose -- and
// a 16-thread pool started and torn down around a one-thread deck costs ~8
// busy CPUs for the life of the process (measured: Rocketlake). This runs
// first (a numbered constructor precedes the default ones) and has the pool
// start at one thread; `set_threads` grows it only on a route that wants
// more. The fallback's numpy and scipy still read the count set_threads
// exports before CPython starts.
#if defined(__GNUC__) && !defined(_WIN32)
__attribute__((constructor(101))) void one_blas_thread_at_load() {
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
}
#endif

#if !MW_EXE_NO_PYTHON
// --- where momwire's Python lives -------------------------------------------
// MOMWIRE_PYTHON_HOME names a venv (or an unpacked environment with the same
// shape); else the build's MW_DEFAULT_PYTHON_HOME. The interpreter library is
// MOMWIRE_LIBPYTHON, else the build's MW_LIBPYTHON.
std::string python_home() {
    std::string h = env("MOMWIRE_PYTHON_HOME");
#if defined(MW_DEFAULT_PYTHON_HOME)
    if (h.empty()) h = MW_DEFAULT_PYTHON_HOME;
#endif
    return h;
}

std::string site_packages(const std::string &home) {
#if defined(_WIN32)
    return home + "\\Lib\\site-packages";
#else
    return home + "/lib/python" + std::to_string(PY_MAJOR_VERSION) + "." +
           std::to_string(PY_MINOR_VERSION) + "/site-packages";
#endif
}

std::vector<std::string> list_dir(const std::string &dir) {
    std::vector<std::string> names;
#if defined(_WIN32)
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA((dir + "\\*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return names;
    do names.push_back(fd.cFileName);
    while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    if (DIR *d = opendir(dir.c_str())) {
        while (dirent *e = readdir(d)) names.push_back(e->d_name);
        closedir(d);
    }
#endif
    return names;
}

std::string lower(std::string s) {
    for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// `momwire_serve_client._scan_metadata_version` over one site-packages: the
// first momwire metadata directory in listing order, its METADATA (else
// PKG-INFO), the first unfolded `Version:` header. "" when not found.
std::string installed_version(const std::string &home) {
    const std::string sp = site_packages(home);
    for (const std::string &name : list_dir(sp)) {
        const std::string low = lower(name);
        const bool meta = (low.size() > 10 && low.compare(low.size() - 10, 10, ".dist-info") == 0) ||
                          (low.size() > 9 && low.compare(low.size() - 9, 9, ".egg-info") == 0);
        if (!meta) continue;
        const std::string stem = low.substr(0, low.rfind('.'));
        if (stem.substr(0, stem.find('-')) != "momwire") continue;
        for (const char *file : {"METADATA", "PKG-INFO"}) {
            std::string text;
            if (!read_file(sp + "/" + name + "/" + file, text) || text.empty()) continue;
            std::istringstream in(text);
            std::string line;
            while (std::getline(in, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty()) break;
                const std::size_t colon = line.find(':');
                if (colon == std::string::npos) break;
                if (lower(line.substr(0, colon)) == "version") {
                    std::size_t a = colon + 1;
                    while (a < line.size() && (line[a] == ' ' || line[a] == '\t')) a++;
                    return line.substr(a);
                }
            }
            return "";
        }
        return "";
    }
    return "";
}

#endif  // !MW_EXE_NO_PYTHON

// --- the C++ route's preconditions ------------------------------------------

#if defined(_WIN32) && MW_EXE_NO_PYTHON
// CPUID and XGETBV, asked directly (momwire#1558): IsProcessorFeaturePresent
// knows PF_AVX2_INSTRUCTIONS_AVAILABLE (40) and PF_AVX512F (41) only from
// Windows 10, and answers FALSE for an id it does not know, so on Windows 7
// an AVX2 CPU would read as baseline. `xcr0_bits` are the register states the
// OS saves (XCR0), which the instructions need as much as the CPU bits.
bool cpuid_has(int leaf7_ebx_bit, unsigned long long xcr0_bits, bool needs_fma) {
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuid(r, 1);
    const bool osxsave = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1, fma = (r[2] >> 12) & 1;
    if (!osxsave || !avx || (needs_fma && !fma)) return false;
    if ((_xgetbv(0) & xcr0_bits) != xcr0_bits) return false;
    __cpuidex(r, 7, 0);
    return (r[1] >> leaf7_ebx_bit) & 1;
}
#endif

// Whether this CPU has AVX2 and FMA (`momwire._accel`'s avx2 test).
bool cpu_has_avx2() {
#if defined(_WIN32) && MW_EXE_NO_PYTHON
    return cpuid_has(5 /* AVX2 */, 0x6 /* SSE, AVX state */, true);
#elif defined(_WIN32)
    // `momwire._accel` asks the same, and the hybrid build must agree with it
    return IsProcessorFeaturePresent(40 /* PF_AVX2_INSTRUCTIONS_AVAILABLE */) != 0;
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

// The C++ kernels are one variant's (MW_EXE_VARIANT, CMakeLists.txt); they
// answer only where Python would load that variant (`momwire._accel`: avx2
// on a CPU with AVX2 + FMA, sse2 on any other x86-64 CPU, and
// MOMWIRE_FORCE_VARIANT naming one overrides the CPU). The no-Python build
// has no Python to agree with: avx2 needs the CPU, sse2 runs anywhere.
bool cxx_variant_ok() {
    const bool avx2 = std::string(EXE_VARIANT) == "avx2";
#if !MW_EXE_NO_PYTHON
    const std::string forced = env("MOMWIRE_FORCE_VARIANT");
    if (!forced.empty()) return forced == EXE_VARIANT && (!avx2 || cpu_has_avx2());
    return avx2 == cpu_has_avx2();
#else
    return !avx2 || cpu_has_avx2();
#endif
}

#if !MW_EXE_NO_PYTHON
// `_gmo.requests_gmo`, conservatively: any `GE` line that does not read
// cleanly with a zero third field sends the deck to Python, which writes the
// `.GMO` file the C++ route does not.
bool may_request_gmo(const std::string &latin1) {
    for (const std::string &line : mw::nec5::latin1_splitlines(latin1)) {
        std::size_t a = 0;
        while (a < line.size() && mw::nec5::latin1_isspace(static_cast<unsigned char>(line[a]))) a++;
        if (line.size() - a < 2) continue;
        const char c0 = static_cast<char>(std::toupper(static_cast<unsigned char>(line[a])));
        const char c1 = static_cast<char>(std::toupper(static_cast<unsigned char>(line[a + 1])));
        if (c0 != 'G' || c1 != 'E') continue;
        try {
            mw::nec5::Card card;
            if (!mw::nec5::read_card(line, card)) continue;
            if (card.mnemonic == "GE" && card.f(2) == 0.0) continue;
            if (card.mnemonic != "GE") continue;
        } catch (...) {
        }
        return true;
    }
    return false;
}

// --- the embedded Python (the fallback) ----------------------------------------

#define PYFN(name) decltype(&name) p_##name = nullptr
struct Py {
    PYFN(PyConfig_InitPythonConfig);
    PYFN(PyConfig_SetBytesString);
    PYFN(Py_InitializeFromConfig);
    PYFN(PyConfig_Clear);
    PYFN(PyStatus_Exception);
    PYFN(PyRun_SimpleString);
    PYFN(PyImport_AddModule);
    PYFN(PyModule_GetDict);
    PYFN(PyDict_GetItemString);
    PYFN(PyLong_AsLong);
};
#undef PYFN

std::string hex(const std::string &s) {
    static const char *d = "0123456789abcdef";
    std::string out;
    for (unsigned char c : s) {
        out += d[c >> 4];
        out += d[c & 15];
    }
    return out;
}

// A Python expression for `s` (the OS's bytes, decoded as the interpreter
// decodes argv: the filesystem encoding with surrogateescape).
std::string py_str(const std::string &s) {
    return "bytes.fromhex('" + hex(s) + "').decode(sys.getfilesystemencoding(), 'surrogateescape')";
}

[[noreturn]] void die(const std::string &what) {
    std::fprintf(stderr, MW_EXE_PROGRAM ": %s\n", what.c_str());
    std::exit(3);
}

void *load_python(const std::string &lib) {
#if defined(_WIN32)
    HMODULE h = LoadLibraryA(lib.c_str());
    if (h == nullptr) die("cannot load " + lib);
    return reinterpret_cast<void *>(h);
#else
    void *h = dlopen(lib.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (h == nullptr) die(dlerror());
    return h;
#endif
}

void *py_sym(void *lib, const char *name) {
#if defined(_WIN32)
    void *p = reinterpret_cast<void *>(GetProcAddress(reinterpret_cast<HMODULE>(lib), name));
#else
    void *p = dlsym(lib, name);
#endif
    if (p == nullptr) die(std::string("no ") + name + " in the Python library");
    return p;
}

std::string libpython() {
    std::string l = env("MOMWIRE_LIBPYTHON");
#if defined(MW_LIBPYTHON)
    if (l.empty()) l = MW_LIBPYTHON;
#endif
    if (l.empty()) die("no Python library named (MOMWIRE_LIBPYTHON)");
    return l;
}

// Start CPython over the momwire environment, publish this executable's
// library, check the release, and run `code`; returns `_rc`.
int run_python(const std::string &home, const std::string &code) {
    const Clock::time_point t0 = Clock::now();
    void *lib = load_python(libpython());
    Py py;
#define GET(name) py.p_##name = reinterpret_cast<decltype(py.p_##name)>(py_sym(lib, #name))
    GET(PyConfig_InitPythonConfig);
    GET(PyConfig_SetBytesString);
    GET(Py_InitializeFromConfig);
    GET(PyConfig_Clear);
    GET(PyStatus_Exception);
    GET(PyRun_SimpleString);
    GET(PyImport_AddModule);
    GET(PyModule_GetDict);
    GET(PyDict_GetItemString);
    GET(PyLong_AsLong);
#undef GET
    PyConfig cfg;
    py.p_PyConfig_InitPythonConfig(&cfg);
    // The environment's python as the "executable": getpath finds pyvenv.cfg
    // beside it, and site adds its site-packages (an editable momwire's too).
#if defined(_WIN32)
    const std::string exe = home + "\\Scripts\\python.exe";
#else
    const std::string exe = home + "/bin/python";
#endif
    py.p_PyConfig_SetBytesString(&cfg, &cfg.executable, exe.c_str());
    py.p_PyConfig_SetBytesString(&cfg, &cfg.program_name, exe.c_str());
    cfg.install_signal_handlers = 0;
    cfg.parse_argv = 0;
    PyStatus st = py.p_Py_InitializeFromConfig(&cfg);
    py.p_PyConfig_Clear(&cfg);
    if (py.p_PyStatus_Exception(st))
        die(std::string("Python did not start: ") + (st.err_msg ? st.err_msg : ""));
    const double t_init = ms_since(t0);
    char table[1024];
    std::snprintf(table, sizeof table,
                  "import sys\n"
                  "sys._mw_host_lapack = {'abi': 2, 'zgetrf': %llu, 'zgetrs': %llu,\n"
                  "    'dgetrf': %llu, 'dgetrs': %llu, 'zgemm': %llu, 'zgemv': %llu}\n",
                  (unsigned long long)(uintptr_t)&MWX(zgetrf_),
                  (unsigned long long)(uintptr_t)&MWX(zgetrs_),
                  (unsigned long long)(uintptr_t)&MWX(dgetrf_),
                  (unsigned long long)(uintptr_t)&MWX(dgetrs_),
                  (unsigned long long)(uintptr_t)&MWX(zgemm_),
                  (unsigned long long)(uintptr_t)&MWX(zgemv_));
    if (py.p_PyRun_SimpleString(table) != 0) die("could not publish the library");
    // The release check, again, as the stamp reads it.
    const std::string check =
        std::string("import momwire_serve_client as _msc\n"
                    "_v = _msc.momwire_version()\n"
                    "if _v != '") +
        VERSION +
        "':\n"
        "    raise SystemExit('momwire-razor2p ' + '" +
        VERSION + "' + ': the momwire beside it is ' + str(_v))\n"
        // AVX-512 off for the embedded numpy and SciPy, before momwire imports
        // them: the launcher's own pin (momwire_serve_client.pin_cpu_env).
        // The static OpenBLAS build_openblas.sh makes carries no AVX-512
        // kernels at all; the Windows build's release DLL does, and reads
        // its core type at load, before this line can set one.
        "_msc.pin_cpu_env()\n";
    if (py.p_PyRun_SimpleString(check.c_str()) != 0) std::exit(3);
    const std::string startup = env("MW_EXE_PYSTARTUP");
    if (!startup.empty())
        py.p_PyRun_SimpleString(("exec(compile(open(" + py_str(startup) + ", 'rb').read(), " +
                                 py_str(startup) + ", 'exec'))\n")
                                    .c_str());
    const Clock::time_point t1 = Clock::now();
    if (g_times.on) {
        // MOMWIRE_TIMES: the Python answer writes the `times:` line, and
        // takes this process's start and the interpreter's load from here
        char pre[256];
        std::snprintf(pre, sizeof pre,
                      "import sys\nsys._mw_times_pre = {'route': 'fallback', 'start_ms': %.3f, "
                      "'load_ms': %.3f}\n",
                      std::chrono::duration<double, std::milli>(t0 - g_times.origin).count(),
                      std::chrono::duration<double, std::milli>(t1 - t0).count());
        py.p_PyRun_SimpleString(pre);
    }
    const std::string run = "import sys\n_rc = 0\n" + code +
                            "sys.stdout.flush()\nsys.stderr.flush()\n";
    if (py.p_PyRun_SimpleString(run.c_str()) != 0) return 1;
    PyObject *main = py.p_PyImport_AddModule("__main__");
    PyObject *rc = py.p_PyDict_GetItemString(py.p_PyModule_GetDict(main), "_rc");
    if (env("MW_EXE_TIMES") == "1")
        std::fprintf(stderr, "fallback: start %.2f ms, init %.2f ms, answer %.2f ms, total %.2f ms\n",
                     ms_since(T0) - ms_since(t0), t_init, ms_since(t1), ms_since(T0));
    return rc == nullptr ? 0 : static_cast<int>(py.p_PyLong_AsLong(rc));
}

// The launcher's own path for anything the C++ route does not answer:
// `momwire_eznec_client.main` (its argv rules, its constants, its last-ditch
// printout) with the resident-server rung skipped and the one-shot rung,
// `_shell.main(..., basis, en_optional)`, run in this process instead of a
// child interpreter.
const char *const FALLBACK_PY = R"PY(
import momwire_eznec_client as _c
def _launcher(args):
    en_optional = False
    if args == ["-version"]:
        print(_c.probe_version())
        return 0
    if not args:
        from_stdin = _c._mech.stdin_paths()
        if from_stdin is not None:
            args = from_stdin
            en_optional = True
    if len(args) < 1:
        print(_c.ARGUMENT_ERROR_INPUT)
        return 0
    if len(args) != 2:
        print(_c.ARGUMENT_ERROR_OUTPUT)
        return 0
    deck_path, printout_path = args
    if "-" in (deck_path, printout_path):
        if args == ["-", "-"]:
            from momwire.eznec._shell import main as _main
            return _main(args, basis="razor-2p")
        print(_c.STREAMING_REFUSAL, file=sys.stderr)
        return 2
    try:
        from momwire.eznec._shell import main as _main
        _main([deck_path, printout_path], basis="razor-2p", en_optional=en_optional)
    except Exception as exc:  # the client's rung 4: the file is the only channel
        _c._last_ditch(printout_path, f"MOMWIRE ENGINE UNAVAILABLE - {type(exc).__name__}: {exc}")
    return 0
)PY";

int fallback(const std::string &home, const std::vector<std::string> &args) {
    set_threads(fallback_threads, false);
    std::string argv = "[";
    for (const std::string &a : args) argv += py_str(a) + ", ";
    argv += "]";
    return run_python(home, std::string(FALLBACK_PY) + "_rc = _launcher(" + argv + ")\n");
}

// A refusal printout naming the mismatch (the deck's comment echo kept, as
// the launcher's own refusals keep it).
void refuse_version(const std::string &deck, const std::string &out, const std::string &found) {
    const std::string why = std::string("this momwire-razor2p is momwire ") + VERSION +
                            ", but the momwire installed beside it is " +
                            (found.empty() ? std::string("missing") : found) +
                            "; reinstall them together";
    std::fprintf(stderr, "momwire-razor2p: %s\n", why.c_str());
    if (out.empty()) return;
    std::string text;
    const bool readable = read_file(deck, text);
    mw::nec5::EngineStamp stamp{VERSION, EXE_VARIANT, ""};
    const std::string printout = mw::nec5::printout_bytes(
        mw::nec5::render_refusal(readable ? &text : nullptr, why, BASIS, stamp));
    if (std::FILE *w = std::fopen(out.c_str(), "wb")) {
        std::fwrite(printout.data(), 1, printout.size(), w);
        std::fclose(w);
    }
}

#endif  // !MW_EXE_NO_PYTHON

}  // namespace

#if !MW_EXE_NO_PYTHON
int main(int argc, char **argv) {
    times_begin();
    const std::vector<std::string> args(argv + 1, argv + argc);

    // MOMWIRE_THREADS fixes the count for every deck; unset, the deck's size
    // decides (below), and the fallback gets every hardware thread.
    const int forced_threads = std::atoi(env("MOMWIRE_THREADS").c_str());
    int threads = forced_threads > 0 ? forced_threads : hardware_threads();
    // Nothing is sized yet: OpenBLAS starts its pool lazily, and resizing it
    // here would cost a deck that then runs on one thread ~0.7 ms. Each route
    // sizes what it uses (`set_threads`).
    fallback_threads = threads;

    const std::string home = python_home();
    const bool two_paths = args.size() == 2 && args[0] != "-" && args[1] != "-";
    const std::string found = installed_version(home);
    if (found != VERSION) {
        refuse_version(two_paths ? args[0] : "", two_paths ? args[1] : "", found);
        return 0;
    }
    if (!two_paths) {
        log_route("fallback:argv", args.empty() ? "-" : args[0]);
        return fallback(home, args);
    }
    const std::string &deck = args[0], &out = args[1];
    std::string why;
    std::string text;
    if (env("MW_EXE_FORCE_FALLBACK") == "1")
        why = "forced";
    else if (!cxx_variant_ok())
        why = "variant";
    else if (!read_file(deck, text))
        why = "unreadable";
    else if (may_request_gmo(text))
        why = "gmo";
    if (why.empty()) {
        const Clock::time_point t = Clock::now();
        // The Sommerfeld masters' own store (momwire#1479 S4), keyed by this
        // build, before anything can fill a grid (the size policy's
        // seam builds the front): a rebuild at the same version starts a fresh directory.
        {
            const std::string tag = std::string(VERSION) + "-" + __DATE__ + "-" + __TIME__;
            mws::exe::configure_grid_cache(tag);
            mw::exe::configure_grid_cache(tag);
        }
        // The size policy (`threads_for`). MOMWIRE_THREADS overrides it: 1 is
        // the serial copy, more the threaded one.
        if (forced_threads <= 0) {
            const mw_times::Scope timed(mw_times::seam);
            const long long n = mws::exe::largest_solve(text);
            threads = n >= 0 ? threads_for(n) : 1;
        }
        const std::string arrays_dir = env("MW_EXE_ARRAYS");
        std::string printout;
        bool printed;
        if (threads == 1 && env("MW_EXE_COPY") != "threads") {
            // The fill on one thread with no OpenMP; OpenBLAS on two when the
            // size policy chose (not MOMWIRE_THREADS): its one-thread path
            // rounds differently from its threaded one, alike at 2, 4, 8 and
            // 16 (measured), and the launcher's default is threaded, so a
            // thread-sensitive printout cell (a 1e10-ohm network pin's
            // reactance) prints the launcher's digits.
            const int blas = forced_threads > 0 ? 1 : std::min(2, hardware_threads());
            set_threads(blas, false);
            times_threads(1, blas);
            printed = mws::exe::answer(text, VERSION, EXE_VARIANT, arrays_dir, printout, why);
        } else {
            set_threads(threads, true);
            times_threads(threads, threads);
            printed = mw::exe::answer(text, VERSION, EXE_VARIANT, arrays_dir, printout, why);
        }
        if (printed) {
            const Clock::time_point tw = Clock::now();
            std::FILE *w = std::fopen(out.c_str(), "wb");
            if (w != nullptr) {
                std::fwrite(printout.data(), 1, printout.size(), w);
                if (std::fclose(w) == 0) {
                    times_line("cxx", threads, 0.0, ms_since(tw));
                    log_route(threads == 1 && env("MW_EXE_COPY") != "threads" ? "cxx:serial"
                                                                    : "cxx:threads",
                              deck);
                    if (env("MW_EXE_TIMES") == "1")
                        std::fprintf(stderr, "cxx: threads %d, answer %.3f ms, total %.3f ms\n",
                                     threads, ms_since(t), ms_since(T0));
                    return 0;
                }
            }
            why = "unwritable";  // Python's words for it
        } else {
            why = "declined:" + why;
        }
    }
    for (char &c : why)
        if (c == ' ') c = '_';
    log_route("fallback:" + why, deck);
    return fallback(home, args);
}
#endif  // !MW_EXE_NO_PYTHON

#if MW_EXE_NO_PYTHON
// ============================================================================
// The no-Python build (momwire#1512 step 1): `momwire-razor2p-standalone`.
//
// Everything the launcher does, in this process, with nothing loaded: the
// client's argv rules (`momwire_eznec_client.main`: `-version`, 4nec2's
// stdin form, its argument errors, a refused '-') and the
// one-shot shell's frame (`_shell.main` / `run`: an unreadable deck, the
// deck closed by EN for the stdin form, the `.GMO` file, the printout
// written CRLF, and the INTERNAL ERROR printout for anything that escapes,
// re-reading the deck for its echo, with `UNABLE TO OPEN FILE <path>` on
// stdout when the printout itself cannot be written). A deck the C++ route
// declines -- the hybrid's fallback -- is a refusal printout naming why
// ("not served by this build: ..."), and so is a CPU without AVX2 / FMA.
// SimNEC's streaming call, `- -`, is `_shell.stream` (momwire#1512): each
// structure answered as its NX arrives, its printout flushed to stdout.
// ============================================================================
#include "launcher.h"

#include <cerrno>
#include <csignal>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <io.h>
#endif

namespace {

namespace ml = mw::launcher;

// `momwire_eznec_client`'s argument errors (its own, not the shell's) and
// its streaming refusal; `_shell.UNABLE_TO_OPEN` for an unwritable printout.
const char *const ARGUMENT_ERROR_INPUT = "UNABLE TO OPEN FILE";
const char *const ARGUMENT_ERROR_OUTPUT = "UNABLE TO OPEN SECOND FILE";
const char *const UNABLE_TO_OPEN = "UNABLE TO OPEN FILE";
const char *const STREAMING_REFUSAL =
    "momwire serves the streaming mode (decks on stdin, printouts on stdout) as '- -' only; "
    "give it '- -', or two file paths (momwire#1395, momwire#1512)";

const mw::nec5::EngineStamp STAMP{VERSION, EXE_VARIANT, ""};

// `probe_version`: NEC5momwire.<major>.<minor>, this build's release.
std::string probe_version() {
    const std::string v = VERSION;
    const std::size_t a = v.find('.');
    const std::size_t b = a == std::string::npos ? a : v.find('.', a + 1);
    if (a == std::string::npos) return "NEC5momwire.0.0";
    const std::string minor = v.substr(a + 1, b == std::string::npos ? std::string::npos : b - a - 1);
    return "NEC5momwire." + v.substr(0, a) + "." + minor;
}

// `_shell.read_deck`: the file's bytes, false for anything open/read refuses
// (a directory included, as Python's IsADirectoryError).
bool read_deck(const std::string &path, std::string &out) {
    out.clear();
#if !defined(_WIN32)
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) != 0 || S_ISDIR(st.st_mode)) {
        close(fd);
        return false;
    }
    char buf[65536];
    for (;;) {
        const ssize_t r = read(fd, buf, sizeof buf);
        if (r < 0) {
            if (errno == EINTR) continue;
            close(fd);
            return false;
        }
        if (r == 0) break;
        out.append(buf, static_cast<std::size_t>(r));
    }
    close(fd);
    return true;
#else
    return read_file(path, out);
#endif
}

// An `open(path, "w")` + write: 0, or the errno Python's OSError carries.
int write_file(const std::string &path, const std::string &bytes) {
    errno = 0;
    std::FILE *w = std::fopen(path.c_str(), "wb");
    if (w == nullptr) return errno ? errno : EIO;
    const std::size_t n = std::fwrite(bytes.data(), 1, bytes.size(), w);
    int err = n == bytes.size() ? 0 : (errno ? errno : EIO);
    if (std::fclose(w) != 0 && err == 0) err = errno ? errno : EIO;
    return err;
}

std::string refusal(const std::string *latin1, const std::string &utf8_reason) {
    return mw::nec5::printout_bytes(mw::nec5::render_refusal(latin1, utf8_reason, BASIS, STAMP));
}

// The sentence for a deck the C++ route declines: what is not served, by
// the route's own reason, in words where the reason is a short name.
std::string not_served(std::string why) {
    static const std::pair<const char *, const char *> words[] = {
        {"finite-ground-buried", "a wire below a finite ground (GN 0 / GN 2), which this build does not solve yet"},
        {"stepping-fr-count", "a stepping FR asking for more than a million frequencies"},
        {"duplicated-wire", "a deck in which two wires coincide"},
        {"no-wires", "a deck with no wire to solve"},
        {"segment-count", "a segment count past what this build sizes"},
        {"non-finite-geometry", "a wire coordinate that is not a finite number"},
        {"non-finite-radius", "a wire radius that is not a finite number"},
        {"non-finite-frequency", "a frequency that is not a finite number"},
        {"wire-loading-values", "a wire-loading value (LD 2 / LD 3 / LD 5) this build does not take"},
        {"load-values", "a load value this build does not take"},
        {"node-key-range", "a coordinate too large to place on the node grid"},
        {"node-range", "a node number too large for this build"},
        {"phantom-threshold", "a wire too close to the phantom-wire threshold to classify"},
        {"crossing-tolerance", "a wire crossing the ground plane too close to call"},
        {"source-text-encoding", "deck text this build does not read"},
        {"network-card-order", "network cards in an order this build does not read"},
    };
    std::string head = why, tail;
    const std::size_t colon = why.find(':');
    if (colon != std::string::npos && why.compare(0, colon, "front") == 0) {
        head = why.substr(colon + 1);
        tail = " (front half)";
    }
    for (const auto &w : words)
        if (head == w.first) return std::string("not served by this build: ") + w.second + tail;
    // a sentence with Python's or numpy's name for its exception: the
    // sentence alone (this build has neither)
    const std::size_t paren = why.find(" (");
    if (paren != std::string::npos) why = why.substr(0, paren);
    return "not served by this build: " + why;
}

// MOMWIRE_TIMES: the last render's thread count and printout write.
int g_render_threads = 1;
double g_write_ms = 0.0;

// What a failed allocation reports: libstdc++'s what(), spelled out so the
// MSVC build (whose what() is "bad allocation") prints the same line.
const char *const BAD_ALLOC = "std::bad_alloc";

std::string internal_error_reason(const std::string &kind, const std::string &message) {
    return "INTERNAL ERROR IN MOMWIRE ENGINE - " + kind + ": " + message;
}

// `_shell._report_internal_error`: the deck re-read for its echo; the
// printout path itself unwritable -> stdout.
void report_internal_error(const std::string &deck, const std::string &out,
                           const std::string &reason) {
    std::string text;
    const bool readable = read_deck(deck, text);
    if (write_file(out, refusal(readable ? &text : nullptr, reason)) != 0) {
#if defined(_WIN32)
        // Python's print(): CRLF on a Windows stdout. By here the OpenBLAS
        // DLL has loaded, and stdout has been found in binary mode after it
        // (the MinGW-built DLL shares this process's UCRT), so the mode is
        // set rather than assumed (momwire#1558).
        std::fflush(stdout);
        _setmode(_fileno(stdout), _O_TEXT);
#endif
        std::printf("%s %s\n", UNABLE_TO_OPEN, out.c_str());
    }
}

// `_shell.render` for a deck's text, through the C++ route: the printout's
// bytes. A route decline is a refusal; an escaped C++ exception is thrown
// as the PyError the INTERNAL ERROR frame reports.
std::string render(const std::string &deck_path, const std::string &text, int forced_threads,
                   bool &raised, bool open_end = false, bool *served = nullptr) {
    raised = false;
    if (served) *served = false;
    if (!cxx_variant_ok())
        return refusal(&text,
                       "not served by this build: this processor lacks AVX2 and FMA, which this "
                       "build's kernels need");
    const Clock::time_point t = Clock::now();
    {
        const std::string tag = std::string(VERSION) + "-" + __DATE__ + "-" + __TIME__;
        mws::exe::configure_grid_cache(tag);
        mw::exe::configure_grid_cache(tag);
    }
    int threads = forced_threads;
    if (forced_threads <= 0) {
        const mw_times::Scope timed(mw_times::seam);
        const long long n = mws::exe::largest_solve(text, open_end);
        threads = n >= 0 ? threads_for(n) : 1;
    }
    g_render_threads = threads;
    const std::string arrays_dir = env("MW_EXE_ARRAYS");
    std::string printout, why, kind, message, raised_kind;
    bool failed = false, printed;
    const bool serial = threads == 1 && env("MW_EXE_COPY") != "threads";
    if (serial) {
        const int blas = forced_threads > 0 ? 1 : std::min(2, hardware_threads());
        set_threads(blas, false);
        times_threads(1, blas);
        printed = mws::exe::answer(text, VERSION, EXE_VARIANT, arrays_dir, printout, why, &failed, &kind,
                                   &message, &raised_kind, open_end, served);
    } else {
        set_threads(threads, true);
        times_threads(threads, threads);
        printed = mw::exe::answer(text, VERSION, EXE_VARIANT, arrays_dir, printout, why, &failed, &kind,
                                  &message, &raised_kind, open_end, served);
    }
    if (printed) {
        // An exception Python's render raises (the INTERNAL ERROR printout
        // the route wrote): `run` then writes no .GMO.
        raised = !raised_kind.empty();
        log_route(serial ? "cxx:serial" : "cxx:threads", deck_path);
        if (env("MW_EXE_TIMES") == "1")
            std::fprintf(stderr, "cxx: threads %d, answer %.3f ms, total %.3f ms\n", threads,
                         ms_since(t), ms_since(T0));
        return printout;
    }
    if (failed) throw ml::PyError{kind, message};
    std::string logged = why;
    for (char &c : logged)
        if (c == ' ') c = '_';
    log_route("refused:" + logged, deck_path);
    return refusal(&text, not_served(why));
}

// `_shell.run` inside `_shell.main`'s last line of defence.
int run_deck_inner(const std::string &deck, const std::string &out, bool en_optional,
             int forced_threads) {
    try {
        std::string text, printout, gmo;
        bool has_gmo = false;
        if (!read_deck(deck, text)) {
            printout = refusal(nullptr, "UNABLE TO READ INPUT FILE " +
                                            ml::os_bytes_to_utf8(ml::path_str(deck)));
            log_route("unreadable", deck);
        } else {
            if (en_optional) text = ml::close_deck(text);
            // gate hook: an exception from the solve, as memory would raise
            const std::string inject = env("MW_EXE_FAIL");
            if (inject == "alloc") throw std::bad_alloc();
            if (inject == "runtime") throw std::runtime_error("injected by MW_EXE_FAIL");
            bool raised = false;
            printout = render(deck, text, forced_threads, raised);
            if (!raised) try {
                has_gmo = ml::gmo_text(text, gmo);
            } catch (const ml::PyError &e) {
                if (!e.kind.empty()) throw;
                // a geometry this build does not lay out: say so in the
                // printout rather than write a file Python would not
                printout = refusal(&text, not_served(e.message));
                has_gmo = false;
            }
        }
        if (has_gmo) {
            std::string crlf;
            for (char c : gmo) {
                if (c == '\n') crlf += '\r';
                crlf += c;
            }
            const std::string path = ml::gmo_path(out);
            if (const int err = write_file(path, crlf)) throw ml::os_error(err, path);
        }
        const Clock::time_point tw = Clock::now();
        const int err = write_file(out, printout);
        g_write_ms = ms_since(tw);
        if (err) throw ml::os_error(err, out);
        return 0;
    } catch (const ml::PyError &e) {
        report_internal_error(deck, out, internal_error_reason(e.kind, e.message));
    } catch (const std::bad_alloc &) {
        report_internal_error(deck, out, internal_error_reason("MemoryError", BAD_ALLOC));
    } catch (const std::exception &e) {
        report_internal_error(deck, out, internal_error_reason("RuntimeError", e.what()));
    } catch (...) {
        report_internal_error(deck, out,
                              internal_error_reason("RuntimeError", "an unknown C++ exception"));
    }
    return 0;
}

int run_deck(const std::string &deck, const std::string &out, bool en_optional, int forced_threads) {
    const int rc = run_deck_inner(deck, out, en_optional, forced_threads);
    times_line("cxx", g_render_threads, 0.0, g_write_ms);
    return rc;
}

// --- SimNEC's streaming call (`_shell.stream`, momwire#1512) -----------------

// `_shell._stream_piece`: one piece of the stream as its printout's bytes;
// `go_on` false once the stream is over (EN, the last piece, a refusal, a
// decline, an exception).
std::string stream_piece(const std::string &piece, bool cut_at_nx, int forced_threads,
                         bool &go_on) {
    go_on = false;
    try {
        const std::string inject = env("MW_EXE_FAIL");  // gate hook, as run_deck's
        if (inject == "alloc") throw std::bad_alloc();
        if (inject == "runtime") throw std::runtime_error("injected by MW_EXE_FAIL");
        bool raised = false, served = false;
        std::string printout = render("-", piece, forced_threads, raised, true, &served);
        go_on = cut_at_nx && served;
        return printout;
    } catch (const ml::PyError &e) {
        return refusal(&piece, internal_error_reason(e.kind, e.message));
    } catch (const std::bad_alloc &) {
        return refusal(&piece, internal_error_reason("MemoryError", BAD_ALLOC));
    } catch (const std::exception &e) {
        return refusal(&piece, internal_error_reason("RuntimeError", e.what()));
    } catch (...) {
        return refusal(&piece, internal_error_reason("RuntimeError", "an unknown C++ exception"));
    }
}

bool write_all(const std::string &bytes) {
    std::size_t done = 0;
    while (done < bytes.size()) {
#if defined(_WIN32)
        const int w = _write(1, bytes.data() + done, static_cast<unsigned>(bytes.size() - done));
#else
        const ssize_t w = write(1, bytes.data() + done, bytes.size() - done);
#endif
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;  // the caller stopped reading
        }
        done += static_cast<std::size_t>(w);
    }
    return true;
}

long read_some(char *buf, std::size_t n) {
    for (;;) {
#if defined(_WIN32)
        const int r = _read(0, buf, static_cast<unsigned>(n));
#else
        const ssize_t r = read(0, buf, n);
#endif
        if (r < 0 && errno == EINTR) continue;
        return r < 0 ? 0 : static_cast<long>(r);  // an unreadable stdin ends the stream
    }
}

int stream(int forced_threads) {
#if defined(_WIN32)
    _setmode(0, _O_BINARY);
    _setmode(1, _O_BINARY);
#else
    std::signal(SIGPIPE, SIG_IGN);  // a closed stdout is a write error, not a kill
#endif
    std::string pending;
    bool swallow_lf = false, final = false;
    std::vector<char> buf(65536);
    while (!final) {
        const long got = read_some(buf.data(), buf.size());
        final = got == 0;
        std::size_t skip = 0;
        if (swallow_lf && got > 0) {
            if (buf[0] == '\n') skip = 1;
            swallow_lf = false;
        }
        if (got > 0) pending.append(buf.data() + skip, static_cast<std::size_t>(got) - skip);
        for (;;) {
            std::size_t end = 0;
            bool is_en = false;
            if (!mw::nec5::next_structure(pending, final, end, is_en)) {
                if (final) {
                    bool blank = true;
                    for (char c : pending)
                        blank = blank && mw::nec5::latin1_isspace(static_cast<unsigned char>(c));
                    if (!blank) {
                        bool go_on = false;
                        const std::string last = stream_piece(pending, false, forced_threads, go_on);
                        const Clock::time_point tw = Clock::now();
                        write_all(last);
                        times_line("stream", g_render_threads, 0.0, ms_since(tw));
                    }
                }
                break;
            }
            const std::string piece = pending.substr(0, end);
            pending.erase(0, end);
            swallow_lf = piece.back() == '\r' && pending.empty() && !final;
            bool go_on = false;
            const std::string printout = stream_piece(piece, !is_en, forced_threads, go_on);
            log_route(go_on ? "stream:structure" : "stream:last", "-");
            const Clock::time_point tw = Clock::now();
            const bool wrote = write_all(printout);
            times_line("stream", g_render_threads, 0.0, ms_since(tw));
            if (!wrote || !go_on) return 0;
        }
    }
    return 0;
}

#if defined(_WIN32)
// The Windows build's OpenBLAS is the DLL SciPy's wheels bundle, beside the
// executable and DELAY-LOADED (CMakeLists.txt), so this runs before its load
// -time constructor reads its environment -- what the Linux build's numbered
// constructor (`one_blas_thread_at_load`) does for the static archive: the
// pool starts at one thread, and `set_threads` grows it only on a route that
// wants more. On an AVX-512 CPU it also holds the library to the Haswell
// kernels every AVX2 CPU runs, as `momwire_serve_client.pin_cpu_env` holds
// the launcher's (the Linux archive leaves SKYLAKEX out of its build; SciPy's
// DLL does not). A value the caller exported is never replaced.
void blas_env_before_load() {
    if (env("OPENBLAS_NUM_THREADS").empty()) set_env("OPENBLAS_NUM_THREADS", "1");
    if (env("OPENBLAS_CORETYPE").empty() && cpuid_has(16 /* AVX512F */, 0xe6, false))
        set_env("OPENBLAS_CORETYPE", "Haswell");
}
#endif

}  // namespace

int main(int argc, char **argv) {
    times_begin();
#if defined(_WIN32)
    blas_env_before_load();
#endif
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.size() == 1 && args[0] == "-version") {
        std::printf("%s\n", probe_version().c_str());
        return 0;
    }
    bool en_optional = false;
    if (args.empty()) {
        // 4nec2's form (momwire#1488): `ENGINE < nec2d.tmp`, the names on stdin
        std::vector<std::string> from_stdin;
        if (ml::stdin_paths(from_stdin)) {
            args = from_stdin;
            en_optional = true;
        }
    }
    if (args.size() < 1) {
        std::printf("%s\n", ARGUMENT_ERROR_INPUT);
        return 0;
    }
    if (args.size() != 2) {
        std::printf("%s\n", ARGUMENT_ERROR_OUTPUT);
        return 0;
    }
    const int forced_threads = std::atoi(env("MOMWIRE_THREADS").c_str());
    if (args[0] == "-" && args[1] == "-") return stream(forced_threads);
    if (args[0] == "-" || args[1] == "-") {
        std::fprintf(stderr, "%s\n", STREAMING_REFUSAL);
        return 2;
    }
    return run_deck(args[0], args[1], en_optional, forced_threads);
}
#endif  // MW_EXE_NO_PYTHON
