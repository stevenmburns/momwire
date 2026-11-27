// momwire-razor2p: `momwire-nec5-razor2p` as one native executable.
//
//     momwire-razor2p DECK PRINTOUT        (EZNEC's and SimNEC's form)
//     momwire-razor2p < PATHS              (4nec2's: the two names on stdin)
//     momwire-razor2p - -                  (SimNEC's streaming call, momwire#1512)
//     momwire-razor2p -version
//
// The contract is the launcher's (`momwire.dropin.nec5._shell`): the printout file
// is the channel, exit 0. Two routes answer it, and they write the same bytes:
//
//   * C++: the deck reader, the structure half, the front half, the fill, the
//     LU and the run half (`_nec5_razor_printout_core.h`), for every deck it
//     covers. No Python is loaded.
//   * the fallback: everything else -- a declined deck (`route == declined`),
//     a deck asking for a `.GMO` file, the stdin form, an
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
// 1, 4, or one per physical core; every core for a buried crossing deck) and
// every hardware thread for the fallback
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
// MOMWIRE_X_<NAME> (an experimental feature, `momwire._features`): one this
// executable carries (EXE_FEATURES) and whose reach covers its support level
// runs, and line 2 of the printout names it; any other one turned on is
// refused, one sentence on stderr and exit 2, before anything else runs
// (`feature_refusal`).
//
// MOMWIRE_TIMES=1 (for users, MOMWIRE_TIMES): the printout's MATRIX TIMING
// (FILL, FACTOR) and RUN TIME carry measured seconds instead of zeros, and
// each answer writes one `times:` line on stderr, its stages in milliseconds
// (`_mw_times.h`). Off, nothing is timed and the printout is unchanged.
//   MW_EXE_COPY=threads      one thread through the OpenMP copy, not the serial one
//   MOMWIRE_FORCE_VARIANT=avx2|sse2  (fat build) the path, instead of the CPU's pick;
//                            a deprecated `-avx2` / `-sse2` file name asks the
//                            same when the variable is unset (exe_name.h)
//   MW_EXE_FAIL=alloc|runtime  (no-Python build) throw from the solve, to test
//                            the INTERNAL ERROR printout
#include "_mw_times.h"
#include "_thread_policy.h"
#include "_nec5_deck_core.h"
#include "_nec5_razor_printout_core.h"
#if MW_EXE_DIALECT_NEC4
#include "_nec4_printout_core.h"
#endif
#if MW_EXE_DIALECT_NEC2
#include "_nec2_printout_core.h"
#endif
#include "exe_name.h"
#include "route.h"
#define mw mws
#include "route.h"
#undef mw
#if MW_EXE_FAT
// The fat build (MW_EXE_VARIANT=fat): the cores twice more, compiled for avx2,
// as `mwa` (threaded) and `mwas` (serial). `mw` / `mws` are then sse2's, the
// copies every CPU runs, and the ones this file's direct calls (the deck
// reader, the refusal printouts) reach.
#define mw mwa
#include "route.h"
#undef mw
#define mw mwas
#include "route.h"
#undef mw
namespace mw_fat {
bool avx2 = false;  // main() picks (`pick_variant`) before any route call
}
#define MW_PICK(sse, avx, call) (mw_fat::avx2 ? avx::exe::call : sse::exe::call)
#else
#define MW_PICK(sse, avx, call) (sse::exe::call)
#endif

// The route's two copies as this file calls them: the serial one and the
// threaded one, each the sse2 or (fat build, avx2 picked) the avx2 build's.
namespace rt_serial {
inline long long largest_solve(const std::string &t, bool open_end = false, long long *pairs = nullptr,
                               long long *below = nullptr) {
    return MW_PICK(mws, mwas, largest_solve(t, open_end, pairs, below));
}
inline void configure_grid_cache(const std::string &tag) { MW_PICK(mws, mwas, configure_grid_cache(tag)); }
inline bool configure_features(const std::vector<std::string> &names, bool stamped = true) {
    return MW_PICK(mws, mwas, configure_features(names, stamped));
}
inline bool withdraw_features(const std::vector<std::string> &names) {
    return MW_PICK(mws, mwas, withdraw_features(names));
}
inline void stamp_uncertified(bool on) { MW_PICK(mws, mwas, stamp_uncertified(on)); }
inline void stamp_notes(const std::string &pragma, const std::string &rename) {
    MW_PICK(mws, mwas, stamp_notes(pragma, rename));
}
inline void reset_features() { MW_PICK(mws, mwas, reset_features()); }
inline void reuse_plans(long long &built, long long &declined, long long &found) {
    MW_PICK(mws, mwas, reuse_plans(built, declined, found));
}
inline std::string reuse_outcome() { return MW_PICK(mws, mwas, reuse_outcome()); }
#if MW_EXE_DIALECT_NEC2
inline bool nec2_render(const std::string &t, int cap, std::vector<std::string> &out, std::vector<std::string> &err,
                        std::string &why, bool *failed, std::string *kind, std::string *message) {
    return MW_PICK(mws, mwas, nec2_render(t, cap, out, err, why, failed, kind, message));
}
inline long long nec2_largest_solve(const std::string &t) { return MW_PICK(mws, mwas, nec2_largest_solve(t)); }
#endif
inline bool answer(const std::string &t, const std::string &version, const std::string &variant,
                   const std::string &arrays_dir, std::string &printout, std::string &reason,
                   bool *failed = nullptr, std::string *kind = nullptr, std::string *message = nullptr,
                   std::string *raised_kind = nullptr, bool open_end = false, bool *served = nullptr) {
    return MW_PICK(mws, mwas, answer(t, version, variant, arrays_dir, printout, reason, failed, kind,
                                     message, raised_kind, open_end, served));
}
}  // namespace rt_serial
namespace rt_threads {
inline long long largest_solve(const std::string &t, bool open_end = false, long long *pairs = nullptr,
                               long long *below = nullptr) {
    return MW_PICK(mw, mwa, largest_solve(t, open_end, pairs, below));
}
inline void configure_grid_cache(const std::string &tag) { MW_PICK(mw, mwa, configure_grid_cache(tag)); }
inline bool configure_features(const std::vector<std::string> &names, bool stamped = true) {
    return MW_PICK(mw, mwa, configure_features(names, stamped));
}
inline bool withdraw_features(const std::vector<std::string> &names) {
    return MW_PICK(mw, mwa, withdraw_features(names));
}
inline void stamp_uncertified(bool on) { MW_PICK(mw, mwa, stamp_uncertified(on)); }
inline void stamp_notes(const std::string &pragma, const std::string &rename) {
    MW_PICK(mw, mwa, stamp_notes(pragma, rename));
}
inline void reset_features() { MW_PICK(mw, mwa, reset_features()); }
inline void reuse_plans(long long &built, long long &declined, long long &found) {
    MW_PICK(mw, mwa, reuse_plans(built, declined, found));
}
inline std::string reuse_outcome() { return MW_PICK(mw, mwa, reuse_outcome()); }
#if MW_EXE_DIALECT_NEC2
inline bool nec2_render(const std::string &t, int cap, std::vector<std::string> &out, std::vector<std::string> &err,
                        std::string &why, bool *failed, std::string *kind, std::string *message) {
    return MW_PICK(mw, mwa, nec2_render(t, cap, out, err, why, failed, kind, message));
}
#endif
inline bool answer(const std::string &t, const std::string &version, const std::string &variant,
                   const std::string &arrays_dir, std::string &printout, std::string &reason,
                   bool *failed = nullptr, std::string *kind = nullptr, std::string *message = nullptr,
                   std::string *raised_kind = nullptr, bool open_end = false, bool *served = nullptr) {
    return MW_PICK(mw, mwa, answer(t, version, variant, arrays_dir, printout, reason, failed, kind,
                                   message, raised_kind, open_end, served));
}
}  // namespace rt_threads

// MW_EXE_TIMES's reuse line (momwire#1971): the fill-pair reuse plans both
// copies built, how many of those built declined, and how many they found,
// process totals, so a gate can prove the reuse ran (built - declined >= 1
// with MOMWIRE_X_PAIR_REUSE on, 0 built with it off). A declined plan
// fills as off, so a plan built is not by itself a reuse that ran.
static void reuse_line() {
    long long built_s = 0, declined_s = 0, found_s = 0, built_t = 0, declined_t = 0, found_t = 0;
    rt_serial::reuse_plans(built_s, declined_s, found_s);
    rt_threads::reuse_plans(built_t, declined_t, found_t);
    std::fprintf(stderr, "cxx: reuse plans built %lld, declined %lld, found %lld\n", built_s + built_t,
                 declined_s + declined_t, found_s + found_t);
    // the sinusoidal route's last fill: "served" or why it declined (momwire#1911)
    const std::string last = rt_threads::reuse_outcome() + rt_serial::reuse_outcome();
    if (!last.empty()) std::fprintf(stderr, "cxx: reuse last fill %s\n", last.c_str());
}

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
#include <ctime>
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
#if defined(__GLIBC__)
#include <malloc.h>  // mallopt
#endif
#if defined(__linux__) || defined(__MINGW32__)
// The static OpenBLAS's pool teardown (a missing one fails the link); the
// numpy library's too where it is a second archive (blas.h, momwire#1751).
extern "C" int MWX(blas_thread_shutdown_)(void);
#if MW_BLAS_TWO_LIBS
extern "C" int MWN(blas_thread_shutdown_)(void);
#endif
#endif

#if !defined(_WIN32)
extern char **environ;  // POSIX; macOS declares it in no header
#endif

namespace {

// The formulation this executable answers for, as the printout's stamp
// names it (CMakeLists.txt's MW_EXE_BASIS), and the program's own name.
#ifndef MW_EXE_BASIS_NAME
#define MW_EXE_BASIS_NAME "razor2p"
#endif
#ifndef MW_EXE_PROGRAM
#define MW_EXE_PROGRAM "momwire-razor2p"
#endif
// The Python launcher this executable stands in for (CMakeLists.txt sets it
// per basis): its printouts are that launcher's, byte for byte.
#ifndef MW_EXE_LAUNCHER
#define MW_EXE_LAUNCHER "momwire-nec5-razor2p"
#endif
// The EZNEC slot whose printout this answers: 1 for the NEC-4.2 slot
// (`momwire-nec42-sinusoidal`, momwire#1607 B4b), else NEC-5. Or, with
// MW_EXE_DIALECT_NEC2, the NEC-2 portal (`momwire-nec2c-sinusoidal`,
// momwire#2065), whose command line is nec2c's (nec2c_cli.inc).
#ifndef MW_EXE_DIALECT_NEC4
#define MW_EXE_DIALECT_NEC4 0
#endif
#ifndef MW_EXE_DIALECT_NEC2
#define MW_EXE_DIALECT_NEC2 0
#endif
// The slot's version-line dialect, the line's first field.
#if MW_EXE_DIALECT_NEC4
#define MW_EXE_VERSION_DIALECT "NEC42"
#elif MW_EXE_DIALECT_NEC2
#define MW_EXE_VERSION_DIALECT "NEC2"
#else
#define MW_EXE_VERSION_DIALECT "NEC5"
#endif
const char *const BASIS = MW_EXE_BASIS_NAME;
const char *const VERSION = MW_VERSION;
// The C++ core's variant, as the printout's stamp names it (CMakeLists.txt).
#ifndef MW_EXE_VARIANT
#define MW_EXE_VARIANT "avx2"
#endif
#if MW_EXE_FAT
// The fat build carries both; main() names the one it picked (`pick_variant`).
const char *EXE_VARIANT = "sse2";
#else
const char *const EXE_VARIANT = MW_EXE_VARIANT;
#endif
// The NAME field of the `-version` line (momwire#1606), the one place every
// build gets it: the launcher's name, then `-standalone` and the C++ core's
// variant, e.g. `momwire-nec5-razor2p-standalone-avx2` (momwire#1831), so a
// SimNEC user (or the `CM version` card SimNEC echoes on every deck) can tell
// this executable from the Python launcher of the same name, and the avx2
// build from the sse2 one. Everything else in the line is the launcher's.
// A MinGW-w64 build (windows-static.yml, the single-file experiment) adds
// its toolchain after the variant, `...-standalone-avx2-ucrt64`, so a deck's
// `CM version` card or a bug report says which of two Windows builds of one
// release answered; the MSVC build (the shipped zip) adds nothing.
#if defined(__MINGW32__) && defined(__clang__)
#define MW_EXE_TOOLCHAIN_TAG "-clang64"
#elif defined(__MINGW32__) && defined(_UCRT)
#define MW_EXE_TOOLCHAIN_TAG "-ucrt64"
#elif defined(__MINGW32__)
#define MW_EXE_TOOLCHAIN_TAG "-mingw64"
#else
#define MW_EXE_TOOLCHAIN_TAG ""
#endif
#define MW_EXE_VERSION_NAME MW_EXE_LAUNCHER "-standalone-" MW_EXE_VARIANT MW_EXE_TOOLCHAIN_TAG

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

// Experimental features (`momwire._features`). This executable's C++ route
// carries a few (EXE_FEATURES, each with the reach the registry gives it;
// tests/test_features_registry.py holds the two equal): one turned on in the
// environment runs when this executable's support level is within its reach,
// as `_features.certified` would say. Every other feature -- one it does not
// carry, one whose reach stops short of it, or a name the registry does not
// know -- is refused here, before any deck is read, rather than silently
// ignored (or honoured by the embedded fallback). The scan is generic (any
// MOMWIRE_X_<NAME>) plus the older spellings some features kept as aliases;
// on is "1", "true", "yes" or "on" in any case, as there.
const char *const FEATURE_ALIASES[][2] = {
    {"MOMWIRE_EXACT_KERNEL_SERIES", "exact_kernel_series"},
    {"MOMWIRE_RAZOR_CXX", "razor_cxx"},
};

struct ExeFeature {
    const char *name;
    int reach;
    bool default_on;  // the registry's `default`: on within reach unless set off
};
// The one fill-reuse switch, `pair_reuse` (momwire#1982, #2063 step 4;
// reach 11, every executable), which each formulation's C++ route reads from
// `ni_eval::requested_features` as its own reuse: bs2's census and same-edge
// Toeplitz form (`_nec5_bspline_printout_core.cpp`), razor2p's fill-pair
// reuse (`_nec5_razor_seam_core.cpp`, `_razor_pair_reuse_core.h`) and the
// point-matched fill's class map (`_sin_pair_reuse_core.h`). Every route
// that reads it is within its family's reach (the registry's `families`:
// bs2 9, razor2p 8, sinusoidal 11). Then momwire#1860's levers, ported to
// the buried fills' C++ (`ni_eval::Features`), and momwire#1956's retuned
// sheet weights (reach 8: the nec5 executables).
const ExeFeature EXE_FEATURES[] = {
    {"pair_reuse", 11, false},
    {"factored_columns", 9, true},
    {"height_sheet_recal", 9, true},
    {"crossing_table", 9, false},
    {"crossing_above_rule", 9, false},
    {"crossing_sheet_retune", 8, false},
    {"crossing_anchors", 8, false},
};

// A carried feature that only one formulation's route runs: {name, basis}.
// crossing_anchors is razor2p's fill, but nec5-bs2 shares its levels (6 to
// 8) and its EXE_FEATURES, so without this it would accept the variable,
// name it on line 2 and fill as with it off. It refuses instead, in the
// registry's sentence.
const char *const EXE_FEATURE_BASES[][2] = {
    {"crossing_anchors", "razor2p"},
};

bool feature_basis_here(const std::string &name) {
    for (const auto &fb : EXE_FEATURE_BASES)
        if (name == fb[0]) return std::string(BASIS) == fb[1];
    return true;
}

// The deprecated spellings of `pair_reuse` (`_features.FamilyReach.alias`,
// momwire#2063 step 4), each its formulation's: {variable, basis}. On its
// own basis the variable reads as MOMWIRE_X_PAIR_REUSE, with a note on
// stderr (`reuse_alias_note`), unless that one is set, which wins; on any
// other basis it is a name this executable does not carry, refused as it
// always was there. tests/test_features_registry.py holds this to the
// registry's rows.
const char *const REUSE_ALIASES[][2] = {
    {"MOMWIRE_X_OFFSET_REUSE", "bs2"},
    {"MOMWIRE_X_RAZOR_PAIR_REUSE", "razor2p"},
};

bool env_set(const std::string &var) { return env(var.c_str()).find_first_not_of(" \t\r\n") != std::string::npos; }

// This basis's deprecated `pair_reuse` variable, or "" when it has none.
std::string reuse_alias_here() {
    for (const auto &a : REUSE_ALIASES)
        if (std::string(BASIS) == a[1]) return a[0];
    return std::string();
}

// The deprecated variable this run reads for `pair_reuse`, or "": set, on
// this basis, and MOMWIRE_X_PAIR_REUSE unset.
std::string reuse_alias_read() {
    const std::string alias = reuse_alias_here();
    if (alias.empty() || env_set("MOMWIRE_X_PAIR_REUSE") || !env_set(alias)) return std::string();
    return alias;
}

// A carried feature a BASIS accepts and whose formulation ignores it:
// {basis, name}. razor2p's route takes crossing_above_rule and, as Python's
// razor2p, never reads it, so it runs as before (it changes nothing) and is
// never named on line 2, as `_features.stamp` names only what the running
// solver honours (momwire#1980). A feature `EXE_FEATURE_BASES` refuses on a
// basis needs no row here: a refusal never reaches line 2.
// tests/test_feature_solvers_1980.py holds this to the registry.
const char *const EXE_IGNORED[][2] = {
    {"razor2p", "crossing_above_rule"},
};

bool feature_honoured_here(const std::string &name) {
    for (const auto &row : EXE_IGNORED)
        if (std::string(BASIS) == row[0] && name == row[1]) return false;
    return true;
}

// This executable's level in support/hierarchy.toml, as
// `_features.current_level("exe", basis=...)` places it.
int exe_level() {
#if MW_EXE_DIALECT_NEC2
    return 11;  // exe-nec2c-sinusoidal
#elif MW_EXE_DIALECT_NEC4
    return std::string(BASIS) == "sinusoidal" ? 10 : 9;  // exe-nec42-sinusoidal / exe-nec42-bs2
#elif defined(__MINGW32__)
    return 7;  // exe-nec5-bs2-razor2p-windows-mingw
#elif defined(_WIN32)
    return 6;  // exe-nec5-bs2-razor2p-windows
#else
    return 8;  // exe-nec5-bs2-razor2p-linux-macos
#endif
}

bool feature_value_on(std::string v) {
    const std::size_t b = v.find_first_not_of(" \t\r\n");
    const std::size_t e = v.find_last_not_of(" \t\r\n");
    v = b == std::string::npos ? std::string() : v.substr(b, e - b + 1);
    for (char &c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return v == "1" || v == "true" || v == "yes" || v == "on";
}

// The features the environment turns on, by registry name, in the
// environment's order, then the aliases'.
std::vector<std::string> features_requested() {
#if defined(_WIN32)
    char **envp = _environ;
#else
    char **envp = environ;
#endif
    static const std::string prefix = "MOMWIRE_X_";
    std::vector<std::string> names;
    for (char **p = envp; p != nullptr && *p != nullptr; ++p) {
        const std::string kv(*p);
        const std::size_t eq = kv.find('=');
        if (eq == std::string::npos || kv.compare(0, prefix.size(), prefix) != 0) continue;
        if (!feature_value_on(kv.substr(eq + 1))) continue;
        // this basis's deprecated `pair_reuse` variable: read below
        if (kv.substr(0, eq) == reuse_alias_here()) continue;
        std::string name = kv.substr(prefix.size(), eq - prefix.size());
        for (char &c : name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        names.push_back(name);
    }
    for (const auto &alias : FEATURE_ALIASES)
        if (feature_value_on(env(alias[0]))) names.push_back(alias[1]);
    const std::string reuse_alias = reuse_alias_read();
    if (!reuse_alias.empty() && feature_value_on(env(reuse_alias.c_str()))) names.emplace_back("pair_reuse");
    return names;
}

// A CERTIFICATION build's features (CMake MW_EXE_CERTIFY, momwire#1911):
// {name, basis}, each also run here, past its reach, on the basis whose C++
// route carries it, and named on line 2 as Python names it. That is how the
// executable's own bytes are gated against Python's before its reach is
// raised. No release build defines these, and today none does: shipped, a
// feature without an EXE_FEATURES row is refused.
const char *const EXE_CERTIFYING[][2] = {
    {nullptr, nullptr},
};

bool feature_certifying_here(const std::string &name) {
    for (const auto &c : EXE_CERTIFYING)
        if (c[0] != nullptr && name == c[0]) return std::string(BASIS) == c[1];
    return false;
}

bool feature_runs_here(const std::string &name) {
    if (feature_certifying_here(name)) return true;
#if MW_EXE_DIALECT_NEC2
    // NEC-2's free-end cap is the NEC-2 route's own launcher default, read
    // by the route (`nec2_end_cap_mode`, momwire#2033) as the Python
    // launcher reads it: MOMWIRE_X_NEC2_END_CAP=1 is the refusing request
    // and =0 the uncapped answer. The NEC-2 printout carries no line 2. A
    // named exception (Steve, 2026-11-20): `_features.EXE_LAUNCHER_DEFAULTS`.
    if (name == "nec2_end_cap") return true;
#endif
    for (const ExeFeature &f : EXE_FEATURES)
        if (name == f.name) return exe_level() <= f.reach && feature_basis_here(name);
    return false;
}

// Whether the environment sets feature `name` at all (a non-blank value, on
// or not), by its canonical variable or an alias: a default-on feature so
// set takes the value, as `_features._env_value` reads it; unset, its
// default.
bool feature_env_set(const std::string &name) {
    std::string var = "MOMWIRE_X_" + name;
    for (char &c : var) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    std::vector<std::string> vars{var};
    for (const auto &alias : FEATURE_ALIASES)
        if (name == alias[1]) vars.emplace_back(alias[0]);
    if (name == "pair_reuse" && !reuse_alias_here().empty()) vars.push_back(reuse_alias_here());
    for (const std::string &v : vars)
        if (env_set(v)) return true;
    return false;
}

// Whether `name` is on by default in this executable (`_features.default_on`):
// promoted, and its reach covers this executable's level.
bool feature_default_here(const std::string &name) {
    for (const ExeFeature &f : EXE_FEATURES)
        if (name == f.name) return f.default_on && exe_level() <= f.reach && feature_basis_here(name);
    return false;
}

// Line 2's feature tail for this run (the routes' `ni_eval::requested_stamp`,
// for the printouts this file writes itself).
std::string g_feature_stamp;
// Line 2's two notes, kept apart from the features: they are cut first when
// line 2 overflows (`engine_stamp`, momwire#2158).
std::string g_pragma_note, g_rename_note;
#if MW_EXE_FAT
// The variant a deprecated `-avx2` / `-sse2` name asks for, or "", and line
// 2's "(renamed from ...)" note for it (`pick_variant`, exe_name.h).
std::string g_name_variant;
std::string g_name_note;
#endif

// What the environment asked for (`features_requested`), once it passed
// `feature_refusal`: what every deck runs with, bar its own pragma cards.
std::vector<std::string> g_env_asked;

void configure_asked(const std::vector<std::string> &asked, const std::string &pragma_note);

// One sentence on stderr and exit 2 when a feature this executable does not
// run is on; 0 (go on) otherwise, with the ones it runs asked of both copies
// of the C++ route (`configure_asked`).
int feature_refusal() {
    // `_features`' DeprecationWarning for the variable, on stderr
    const std::string reuse_alias = reuse_alias_read();
    if (!reuse_alias.empty())
        std::fprintf(stderr, "%s is deprecated; use MOMWIRE_X_PAIR_REUSE (momwire#2063)\n", reuse_alias.c_str());
    const std::vector<std::string> asked = features_requested();
    for (const std::string &name : asked) {
        if (feature_runs_here(name)) continue;
        // "P%cthon", not the word: the no-Python build is held to carry no
        // "python" bytes at all (tests/test_standalone_nopython.py).
        std::fprintf(stderr, "%s is not certified for this executable; it runs only in the P%cthon route\n",
                     name.c_str(), 'y');
        return 2;
    }
    g_env_asked = asked;
    configure_asked(asked, "");
    return 0;
}

// The features `asked` asks of both copies of the C++ route, from their
// start (the default-on ones only). A default-on feature
// (`feature_default_here`) is asked too unless its variable is set, and, as
// in `_features.stamp`, line 2 names only the features on whose default here
// is off; `pragma_note` (`_features.pragma_note`) follows them there.
void configure_asked(const std::vector<std::string> &asked, const std::string &pragma_note) {
    g_feature_stamp.clear();
    g_pragma_note.clear();
    g_rename_note.clear();
    // The C++ route requests the default-on features from the start
    // (`ni_eval`); one set off here, or outside this level's reach, is
    // withdrawn. Asked on, a default-here one is already requested and
    // unstamped; any other is requested, and stamped where this formulation
    // honours it (`feature_honoured_here`).
    std::vector<std::string> stamped, unstamped, withdrawn;
    for (const std::string &name : asked)
        if (name == "nec2_end_cap") continue;  // the NEC-2 route's own (above)
        else if (!feature_default_here(name))
            (feature_honoured_here(name) ? stamped : unstamped).push_back(name);
    for (const ExeFeature &f : EXE_FEATURES) {
        if (!f.default_on) continue;
        const bool asked_on = std::find(asked.begin(), asked.end(), f.name) != asked.end();
        const bool set_off = feature_env_set(f.name) && !asked_on;
        if (!feature_default_here(f.name) || set_off) withdrawn.emplace_back(f.name);
    }
    if (!stamped.empty()) {
        rt_serial::configure_features(stamped, true);
        rt_threads::configure_features(stamped, true);
    }
    if (!unstamped.empty()) {
        rt_serial::configure_features(unstamped, false);
        rt_threads::configure_features(unstamped, false);
    }
    if (!withdrawn.empty()) {
        rt_serial::withdraw_features(withdrawn);
        rt_threads::withdraw_features(withdrawn);
    }
    // registry order: a certifying feature first, as its entry precedes there
    bool uncertified = false;
    for (const auto &c : EXE_CERTIFYING)
        if (c[0] != nullptr && std::find(stamped.begin(), stamped.end(), c[0]) != stamped.end()) {
            g_feature_stamp += std::string("+") + c[0];
            uncertified = true;
        }
    for (const ExeFeature &f : EXE_FEATURES)
        for (const std::string &name : stamped)
            if (name == f.name) {
                g_feature_stamp += std::string("+") + f.name;
                break;
            }
    // past its reach, line 2 says so after the names, as `_features.stamp`
    if (uncertified) {
        g_feature_stamp += " uncertified on this platform";
        rt_serial::stamp_uncertified(true);
        rt_threads::stamp_uncertified(true);
    }
    // line 2's notes after the features, as `_printout.engine_stamp` writes
    // them: the deck's ignored pragma cards, then (fat build) a deprecated
    // variant name's (`pick_variant`), as Python's basis rename note
    g_pragma_note = pragma_note;
#if MW_EXE_FAT
    g_rename_note = g_name_note;
#endif
    rt_serial::stamp_notes(g_pragma_note, g_rename_note);
    rt_threads::stamp_notes(g_pragma_note, g_rename_note);
}

// --- per-deck pragmas (`_features.deck_pragmas`, momwire's pragma cards) -----
//
// A deck's `CM momwire: <name>` card asks for feature <name> for that deck
// alone, as its variable set on would: the variable set either way wins (0 is
// the user's kill switch), and it goes the way `feature_refusal` sends the
// variable (`configure_asked`), so a whole whose part this basis lacks is
// withheld as there. Only the registry's pragma features (`Feature.pragma`,
// tests/test_features_registry.py holds this list to it) that this
// executable carries are taken; any other name is ignored, never refused,
// and named on line 2 ("(pragma ignored: ...)"). The reader is Python's,
// character for character (tests/test_deck_pragma.py).
const char *const PRAGMA_FEATURES[] = {"pair_reuse", "rotational_symmetry"};

bool pragma_blank(char c) { return c == ' ' || c == '\t'; }

std::vector<std::string> deck_pragmas(const std::string &text) {
    std::vector<std::string> names;
    std::size_t at = 0;
    while (at <= text.size()) {
        std::size_t end = text.find_first_of("\r\n", at);
        if (end == std::string::npos) end = text.size();
        std::size_t k = at;
        while (k < end && pragma_blank(text[k])) k++;
        const bool cm = end - k >= 2 && (text[k] == 'C' || text[k] == 'c') &&
                        (text[k + 1] == 'M' || text[k + 1] == 'm');
        if (cm) {
            k += 2;
            while (k < end && pragma_blank(text[k])) k++;
            static const char prefix[] = "momwire:";
            const std::size_t n = sizeof prefix - 1;
            bool match = end - k >= n;
            for (std::size_t j = 0; match && j < n; j++)
                match = std::tolower(static_cast<unsigned char>(text[k + j])) == prefix[j];
            if (match) {
                k += n;
                while (k < end && pragma_blank(text[k])) k++;
                std::string name;
                for (; k < end; k++) {
                    const unsigned char c = static_cast<unsigned char>(text[k]);
                    const bool word = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                                      (c >= '0' && c <= '9') || c == '_';
                    if (!word) break;
                    name += static_cast<char>(std::tolower(c));
                }
                if (!name.empty() && std::find(names.begin(), names.end(), name) == names.end())
                    names.push_back(name);
            }
        }
        at = end + 1;
    }
    return names;
}

bool pragma_runs_here(const std::string &name) {
    for (const char *p : PRAGMA_FEATURES)
        if (name == p) return feature_runs_here(name);
    return false;
}

#if !MW_EXE_NO_PYTHON
// The pragma features this executable does not carry: its embedded
// launcher (the fallback) withholds them (`_features.withhold_pragmas`), so
// its Python answer ignores them as the C++ route does.
std::vector<std::string> pragmas_withheld() {
    std::vector<std::string> out;
    for (const char *p : PRAGMA_FEATURES)
        if (!feature_runs_here(p)) out.emplace_back(p);
    return out;
}
#endif

// One deck answered under its own pragma cards (`_features.pragma_scope`):
// the route's requests and line 2 re-made for the deck, then back to the
// environment's after it. A scope inside another is the outer one's, and a
// deck with no pragma card changes nothing.
class DeckPragmas {
  public:
    explicit DeckPragmas(const std::string &text) {
        if (depth()++ > 0) return;
        std::vector<std::string> asked = g_env_asked;
        std::string ignored;
        for (const std::string &name : deck_pragmas(text)) {
            if (!pragma_runs_here(name)) {
                ignored += (ignored.empty() ? "" : ", ") + name;
                continue;
            }
            if (feature_env_set(name)) continue;  // the variable decides
            asked.push_back(name);
        }
        if (asked.size() == g_env_asked.size() && ignored.empty()) return;
        active_ = true;
        reconfigure(asked, ignored.empty() ? "" : "(pragma ignored: " + ignored + ")");
    }
    ~DeckPragmas() {
        depth()--;
        if (active_) reconfigure(g_env_asked, "");
    }
    DeckPragmas(const DeckPragmas &) = delete;
    DeckPragmas &operator=(const DeckPragmas &) = delete;

  private:
    static int &depth() {
        static int d = 0;
        return d;
    }
    static void reconfigure(const std::vector<std::string> &asked, const std::string &note) {
        rt_serial::reset_features();
        rt_threads::reset_features();
        configure_asked(asked, note);
    }
    bool active_ = false;
};

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

// Logical CPUs this process may run on (momwire#1952): the affinity mask's
// count where the OS has one, else every hardware thread. Unpinned, the mask
// is every CPU and this is hardware_concurrency(), as before.
int hardware_threads() {
#if defined(_WIN32)
    // The process mask covers one processor group; it reads 0 for a process
    // spread over several (>64 CPUs), which falls through to the full count.
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
    // Count only the cores the process affinity mask reaches (momwire#1952):
    // a pinned run (start /affinity 55: one thread on each of four cores)
    // otherwise sized both pools for every core on the machine, 12 spinning
    // threads on 4 CPUs, and ran ~100x slow. A mask of 0 (several processor
    // groups) or a failed query counts every core, as before.
    DWORD_PTR proc = 0, sys = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys)) proc = 0;
    int cores = 0;
    for (DWORD off = 0; off < len;) {
        auto *p = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data() + off);
        if (p->Relationship == RelationProcessorCore) {
            const GROUP_AFFINITY &g = p->Processor.GroupMask[0];
            if (proc == 0 || (g.Group == 0 && (g.Mask & proc) != 0)) cores++;
        }
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

// The size policy (`_thread_policy.h`, which the Python drop-in shares), on
// the physical cores this process may run on.
int threads_for(long long n_basis, long long crossing_pairs = 0, long long below_pairs = 0) {
    return mw_threads::for_deck(physical_cores(), n_basis, crossing_pairs, below_pairs);
}

// The executable's OpenBLAS, OpenMP (only when `omp`: the serial route never
// enters the OpenMP runtime) and, for an embedded Python, the environment
// its numpy and scipy read.
void set_threads(int n, bool omp) {
    MWX(openblas_set_num_threads)(n);
#if MW_BLAS_TWO_LIBS
    MWN(openblas_set_num_threads)(n);  // numpy's library, the same count
#endif
#if defined(__linux__)
    // Growing the pool starts its new server threads AWAKE: each spins for
    // OpenBLAS's thread timeout (2^28 cycles, ~70 ms) before it sleeps, BLAS
    // call or not, so on a short deck they compete with the OpenMP team for
    // the whole run (perf-1028: fx_d1_buried_dipole's near field at 4
    // threads, 60 ms against Python's 36, with blas_thread_server at 43 % of
    // the samples and no BLAS call made). Shut the pool down instead: the
    // first threaded BLAS call starts it again at the same count
    // (`exec_blas` -> `blas_thread_init`), which is how numpy's pool sits in
    // a warm Python process -- asleep until asked. The thread count, and so
    // every partition and every bit, is unchanged.
    if (n > 1) {
        MWX(blas_thread_shutdown_)();
#if MW_BLAS_TWO_LIBS
        MWN(blas_thread_shutdown_)();
#endif
    }
#endif
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
//
// The pool's threads also spin after each call for OpenBLAS's default
// THREAD_TIMEOUT, 2^28 cycles (~0.1 s), before they sleep (momwire#1772).
// Between the BLAS calls of a fill -- the image weights, the loading Gram --
// that spin took the CPUs the fill's OpenMP threads needed: at four threads
// on a four-core Haswell it cost the buried and screen fills 15-28 %, razor2p
// as much as bs2. 2^16 cycles still catches a call that follows at once; a
// pool that then sleeps wakes on the next call. It sets only how long an idle
// thread spins: what a call computes, and on how many threads, is unchanged.
// A caller's own OPENBLAS_THREAD_TIMEOUT is kept.
//
// The OpenMP team's idle threads have the same habit the other way round:
// libgomp's default spins 300,000 iterations at each wait before a thread
// sleeps, so a fill's last parallel region leaves its team spinning through
// the factor that follows, against OpenBLAS's threads (Linux: libgomp reads
// GOMP_SPINCOUNT in its own load-time constructor, after this one). Measured
// on Skylake (4 cores / 8 threads), exe wall at MOMWIRE_THREADS = 8:
// sd_gn2_dipole101 razor2p 63.7 -> 9.7 ms, ak_yagi_free 52.5 -> 7.8,
// fs_yagi820 72.9 -> 44.9, gn2_dipole401 sg5 164.8 -> 106.0, bs2 159.9 ->
// 124.9; at 4: fs_yagi820 razor2p 55.4 -> 47.2, sd_fs_yagi805 sg5 133.3 ->
// 124.4; at 2 within noise. The cost is a wake where a region follows a wait
// longer than the spin: the buried bs2 fills at 3-4 threads, +1-2 %. 30,000
// iterations is the knee (10,000 and passive cost those fills 5-13 %). Only
// how long an idle thread spins changes; a caller's own GOMP_SPINCOUNT or
// OMP_WAIT_POLICY is kept.
#if defined(__GNUC__) && !defined(_WIN32)
__attribute__((constructor(101))) void one_blas_thread_at_load() {
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
    setenv("OPENBLAS_THREAD_TIMEOUT", "16", 0);
#if defined(__linux__)
    if (std::getenv("OMP_WAIT_POLICY") == nullptr) setenv("GOMP_SPINCOUNT", "30000", 0);
#endif
}
#endif
#if defined(__MINGW32__)
// The static MinGW-w64 build (windows-static.yml; no OpenBLAS DLL to
// delay-load): OpenBLAS and libgomp are linked in, so their load-time
// constructors run before main() and `blas_env_before_load` is too late for
// them. The Linux constructor's settings, ahead of them; libgomp is GCC's
// here too, so the same spin count. _putenv_s cannot keep an existing value,
// hence the checks.
__attribute__((constructor(101))) void one_blas_thread_at_load() {
    _putenv_s("OPENBLAS_NUM_THREADS", "1");
    if (std::getenv("OPENBLAS_THREAD_TIMEOUT") == nullptr) _putenv_s("OPENBLAS_THREAD_TIMEOUT", "16");
    if (std::getenv("OMP_WAIT_POLICY") == nullptr && std::getenv("GOMP_SPINCOUNT") == nullptr)
        _putenv_s("GOMP_SPINCOUNT", "30000");
}
// Both OpenBLAS pools shut down before the C runtime's exit sequence
// (d35e80c1; registered from main(), so it runs before the handlers
// registered at start-up, OpenBLAS's own `gotoblas_quit` among them). Kept
// for the A/B only: the shutdown is itself what TerminateThread()s the pool
// threads (main's MinGW wrapper, below, says why that hangs), and the soak
// measured it hanging more often, not less (9 against 4 in 44,736 each).
// MW_EXE_EXIT_SHUTDOWN=0 leaves the pools alone.
void shutdown_blas_pools() {
    MWX(blas_thread_shutdown_)();
#if MW_BLAS_TWO_LIBS
    MWN(blas_thread_shutdown_)();
#endif
}
void shutdown_blas_pools_at_exit() {
    // only on the normal exit (main's MinGW wrapper, below); the default
    // ends the process before any pool shutdown runs
    const char *how = std::getenv("MW_EXE_EXIT");
    if (how == nullptr || std::string(how) != "normal") return;
    const char *v = std::getenv("MW_EXE_EXIT_SHUTDOWN");
    if (v == nullptr || std::string(v) != "0") std::atexit(shutdown_blas_pools);
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
#if defined(__GNUC__)
// MinGW-w64 gcc: _xgetbv is an xsave intrinsic, callable only from code
// compiled for it (it runs only once OSXSAVE, checked first, is set).
__attribute__((target("xsave")))
#endif
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

#if MW_EXE_FAT
// The fat build's choice, made once, first thing in main(): avx2's kernels
// where `cpu_has_avx2` (CPUID's AVX2 and FMA, and XGETBV's OS-saved AVX
// state) says the CPU runs them, else sse2's -- the extension's rule
// (`momwire._accel`: avx2 on a CPU with AVX2 + FMA, sse2 on any other).
// MOMWIRE_FORCE_VARIANT, the extension's override, names one: sse2 forces the
// baseline; avx2 on a CPU without it is the avx2 build's own refusal
// (`cxx_variant_ok`), and no avx2 code runs.
//
// The executable's own name asks too (momwire#2090, exe_name.h): a copy
// under a deprecated `-avx2` / `-sse2` name runs that variant, as the
// variable would, and the variable wins when both are given. Line 2 then
// carries the rename note (`g_name_note`, `feature_refusal`).
void run_avx2_initializers();
std::string own_path() {
#if defined(_WIN32)
    std::string buf(32768, '\0');
    const DWORD n = GetModuleFileNameA(nullptr, &buf[0], static_cast<DWORD>(buf.size()));
    buf.resize(n < buf.size() ? n : 0);
    return buf;
#else
    return "";
#endif
}
void pick_variant() {
    g_name_variant = mw_exe_name::variant_from_path(own_path());
    g_name_note = mw_exe_name::deprecation_note(MW_EXE_LAUNCHER, g_name_variant);
    std::string forced = env("MOMWIRE_FORCE_VARIANT");
    if (forced.empty()) forced = g_name_variant;
    const bool has = cpu_has_avx2();
    const bool avx2 = forced == "avx2" || (forced != "sse2" && has);
    EXE_VARIANT = avx2 ? "avx2" : "sse2";
    mw_fat::avx2 = avx2 && has;
    if (mw_fat::avx2) run_avx2_initializers();
}

#if defined(_MSC_VER)
// The avx2 copies' dynamic initializers, which the C runtime does NOT run:
// fat_avx2_init.h (forced on every avx2 TU) puts their pointers in
// `.mwav$m`, which the linker sorts between these two markers, in link order
// as it sorts `.CRT$XCU`. They are compiled /arch:AVX2 like the rest of their
// TU, so on a CPU without AVX2 they must never run. Run once, after the
// pick and before any avx2 code, first to last as the C runtime's
// `_initterm` runs its own table; the linker may pad the section with zeros.
using mw_init_fn = void(__cdecl *)();
#pragma section(".mwav$a", read)
#pragma section(".mwav$z", read)
__declspec(allocate(".mwav$a")) const mw_init_fn mw_avx2_init_begin = nullptr;
__declspec(allocate(".mwav$z")) const mw_init_fn mw_avx2_init_end = nullptr;
void run_avx2_initializers() {
    const auto first = reinterpret_cast<std::uintptr_t>(&mw_avx2_init_begin) + sizeof(mw_init_fn);
    const auto last = reinterpret_cast<std::uintptr_t>(&mw_avx2_init_end);
    int n = 0;
    for (std::uintptr_t a = first; a < last; a += sizeof(mw_init_fn)) {
        const mw_init_fn f = *reinterpret_cast<const volatile mw_init_fn *>(a);
        if (f != nullptr) {
            f();
            n++;
        }
    }
    if (env("MW_EXE_TIMES") == "1") std::fprintf(stderr, "cxx: avx2 initializers %d\n", n);
}
#else
// MinGW-w64: the avx2 copies' static constructors, which the C runtime does
// NOT run: defer_ctors.cmake moves each avx2 object's `.ctors` entries into
// `.data$mwactors_m*`, which the PE linker sorts between these two markers.
// They are compiled -mavx2 like the rest of their TU, so on a CPU without
// AVX they must never run (Intel SDE as Westmere faulted in one, at start-up).
// Run once, after the pick, before any avx2 code; in reverse order, as
// libgcc's __do_global_ctors runs `.ctors`.
using mw_ctor = void (*)();
__attribute__((section(".data$mwactors_a"), used)) mw_ctor mw_avx2_ctors_begin = nullptr;
__attribute__((section(".data$mwactors_z"), used)) mw_ctor mw_avx2_ctors_end = nullptr;
void run_avx2_initializers() {
    const auto first = reinterpret_cast<std::uintptr_t>(&mw_avx2_ctors_begin) + sizeof(mw_ctor);
    const auto last = reinterpret_cast<std::uintptr_t>(&mw_avx2_ctors_end);
    int n = 0;
    for (std::uintptr_t a = last; a > first;) {
        a -= sizeof(mw_ctor);
        const mw_ctor f = *reinterpret_cast<volatile mw_ctor *>(a);
        // `.ctors` lists may carry -1 / 0 sentinels; neither is a function
        if (f != nullptr && reinterpret_cast<std::intptr_t>(f) != -1) {
            f();
            n++;
        }
    }
    if (env("MW_EXE_TIMES") == "1") std::fprintf(stderr, "cxx: avx2 initializers %d\n", n);
}
#endif
#endif

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
                  (unsigned long long)(uintptr_t)&MWN(zgemm_),
                  (unsigned long long)(uintptr_t)&MWN(zgemv_));
    if (py.p_PyRun_SimpleString(table) != 0) die("could not publish the library");
    // The release check, again, as the stamp reads it.
    const std::string check =
        std::string("import momwire_serve_client as _msc\n"
                    "_v = _msc.momwire_version()\n"
                    "if _v != '") +
        VERSION +
        "':\n"
        "    raise SystemExit('" MW_EXE_PROGRAM " ' + '" +
        VERSION + "' + ': the momwire beside it is ' + str(_v))\n"
        // AVX-512 off for the embedded numpy and SciPy, before momwire imports
        // them: the launcher's own pin (momwire_serve_client.pin_cpu_env).
        // The static OpenBLAS build_openblas.sh makes carries no AVX-512
        // kernels at all; the Windows build's DLL (scipy-openblas32) does, and reads
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
// The slot's word for the fallback's text: the shell's keyword (the NEC-4.2
// slot's `dialect="nec4"`, momwire#1607 B4b).
#if MW_EXE_DIALECT_NEC4
#define MW_EXE_SHELL_KW ", dialect=\"nec4\""
#else
#define MW_EXE_SHELL_KW ""
#endif
const char *const FALLBACK_PY = R"PY(
import momwire_eznec_client as _c
def _launcher(args):
    en_optional = False
    if args == ["-version"]:
        print(_c._mech.version_line(")PY" MW_EXE_VERSION_DIALECT R"PY(", ")PY" MW_EXE_VERSION_NAME R"PY("))
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
            from momwire.dropin.nec5._shell import main as _main
            return _main(args, basis=")PY" MW_EXE_BASIS_NAME R"PY(")PY" MW_EXE_SHELL_KW R"PY()
        print(_c.STREAMING_REFUSAL, file=sys.stderr)
        return 2
    try:
        from momwire.dropin.nec5._shell import main as _main
        _main([deck_path, printout_path], basis=")PY" MW_EXE_BASIS_NAME R"PY(")PY" MW_EXE_SHELL_KW R"PY(, en_optional=en_optional)
    except Exception as exc:  # the client's rung 4: the file is the only channel
        _c._last_ditch(printout_path, f"MOMWIRE ENGINE UNAVAILABLE - {type(exc).__name__}: {exc}")
    return 0
)PY";

int fallback(const std::string &home, const std::vector<std::string> &args) {
    set_threads(fallback_threads, false);
    std::string argv = "[";
    for (const std::string &a : args) argv += py_str(a) + ", ";
    argv += "]";
    // the pragma features this executable does not carry, ignored by its
    // Python answer as by its C++ route (`DeckPragmas`)
    std::string withheld = "[";
    for (const std::string &p : pragmas_withheld()) withheld += py_str(p) + ", ";
    withheld += "]";
    return run_python(home, std::string(FALLBACK_PY) +
                                "import momwire._features as _mwf\n_mwf.withhold_pragmas(" + withheld +
                                ")\n_rc = _launcher(" + argv + ")\n");
}

// A refusal printout naming the mismatch (the deck's comment echo kept, as
// the launcher's own refusals keep it).
void refuse_version(const std::string &deck, const std::string &out, const std::string &found) {
    const std::string why = std::string("this ") + MW_EXE_PROGRAM + " is momwire " + VERSION +
                            ", but the momwire installed beside it is " +
                            (found.empty() ? std::string("missing") : found) +
                            "; reinstall them together";
    std::fprintf(stderr, "%s: %s\n", MW_EXE_PROGRAM, why.c_str());
    if (out.empty()) return;
    std::string text;
    const bool readable = read_file(deck, text);
    mw::nec5::EngineStamp stamp{VERSION, EXE_VARIANT, g_feature_stamp, g_pragma_note, g_rename_note};
#if MW_EXE_DIALECT_NEC4
    const std::string printout = mw::nec5::printout_bytes(
        mw::nec4::render_nec4_refusal(readable ? &text : nullptr, why, BASIS, stamp));
#else
    const std::string printout = mw::nec5::printout_bytes(
        mw::nec5::render_refusal(readable ? &text : nullptr, why, BASIS, stamp));
#endif
    if (std::FILE *w = std::fopen(out.c_str(), "wb")) {
        std::fwrite(printout.data(), 1, printout.size(), w);
        std::fclose(w);
    }
}

#endif  // !MW_EXE_NO_PYTHON

}  // namespace

#if !MW_EXE_NO_PYTHON
int main(int argc, char **argv) {
#if defined(__MINGW32__)
    shutdown_blas_pools_at_exit();
#endif
#if MW_EXE_FAT
    pick_variant();  // before anything reaches a route copy (feature_refusal does)
    if (env("MW_EXE_TIMES") == "1") {
        std::fprintf(stderr, "cxx: simd %s\n", EXE_VARIANT);
        if (!g_name_variant.empty()) std::fprintf(stderr, "cxx: the name asks %s\n", g_name_variant.c_str());
    }
#endif
    if (const int refused = feature_refusal()) return refused;
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
    else if (!MW_EXE_DIALECT_NEC4 && may_request_gmo(text))  // the NEC-4.2 slot writes none
        why = "gmo";
    const DeckPragmas pragmas(text);  // the fallback reads its own (`_shell.render`)
    if (why.empty()) {
        const Clock::time_point t = Clock::now();
        // The Sommerfeld masters' own store (momwire#1479 S4), keyed by this
        // build, before anything can fill a grid (the size policy's
        // seam builds the front): a rebuild at the same version starts a fresh directory.
        {
            const std::string tag = std::string(VERSION) + "-" + __DATE__ + "-" + __TIME__
#if MW_EXE_FAT
                                    + "-" + EXE_VARIANT  // one store per variant
#endif
            ;
            rt_serial::configure_grid_cache(tag);
            rt_threads::configure_grid_cache(tag);
        }
        // The size policy (`threads_for`). MOMWIRE_THREADS overrides it: 1 is
        // the serial copy, more the threaded one.
        if (forced_threads <= 0) {
            const mw_times::Scope timed(mw_times::seam);
            long long pairs = 0, below = 0;
            const long long n = rt_serial::largest_solve(text, false, &pairs, &below);
            threads = n >= 0 ? threads_for(n, pairs, below) : 1;
            if (env("MW_EXE_TIMES") == "1")
                std::fprintf(stderr, "cxx: size n_basis %lld, crossing pairs %lld, below pairs %lld\n", n,
                             pairs, below);
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
            printed = rt_serial::answer(text, VERSION, EXE_VARIANT, arrays_dir, printout, why);
        } else {
            set_threads(threads, true);
            times_threads(threads, threads);
            printed = rt_threads::answer(text, VERSION, EXE_VARIANT, arrays_dir, printout, why);
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
                    if (env("MW_EXE_TIMES") == "1") {
                        std::fprintf(stderr, "cxx: threads %d, answer %.3f ms, total %.3f ms\n",
                                     threads, ms_since(t), ms_since(T0));
                        reuse_line();
                    }
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
// The no-Python build (momwire#1512 step 1): `momwire-nec5-razor2p-standalone`.
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

// Line 2's fields for the printouts this file writes (read after
// `feature_refusal`, which sets the tail).
mw::nec5::EngineStamp refusal_stamp() {
    return mw::nec5::EngineStamp{VERSION, EXE_VARIANT, g_feature_stamp, g_pragma_note, g_rename_note};
}

// `probe_version`: `momwire_serve_client.version_line`'s line (momwire#1606),
// the dialect, this executable's name (MW_EXE_VERSION_NAME), its interface
// modes and this build's full release, e.g.
// `NEC5 momwire-nec5-razor2p-standalone-avx2 [sb] 0.82.0`.
std::string probe_version() {
#if MW_EXE_FAT
    // the variant this run picked, as a one-variant build names its own
    return std::string(MW_EXE_VERSION_DIALECT " ") + MW_EXE_LAUNCHER "-standalone-" + EXE_VARIANT +
           MW_EXE_TOOLCHAIN_TAG " [sb] " + VERSION;
#else
    return std::string(MW_EXE_VERSION_DIALECT " ") + MW_EXE_VERSION_NAME + " [sb] " + VERSION;
#endif
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
#if MW_EXE_DIALECT_NEC4
    // `_printout.render_refusal(..., dialect="nec4")`: the NEC-4.2 banner
    return mw::nec5::printout_bytes(mw::nec4::render_nec4_refusal(latin1, utf8_reason, BASIS, refusal_stamp()));
#else
    return mw::nec5::printout_bytes(mw::nec5::render_refusal(latin1, utf8_reason, BASIS, refusal_stamp()));
#endif
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
        {"drive-magnitude-overflow", "a source whose magnitude overflows a double"},
        {"degree-key", "a node missing from the table of wires meeting at each node"},
        {"near-field-count", "a near-field request past what this build sizes"},
        {"near-field-coordinates", "a near-field coordinate form other than rectangular or spherical"},
        {"sequence", "a deck with no run to answer"},
        // razor's front (`scope` / `build`)
        {"malformed-args", "a wire description this build cannot read"},
        {"exact-kernel", "the exact thin-wire kernel forced on"},
        {"refl-coef-ground", "a reflection-coefficient ground on this basis"},
        {"somm-environment", "a Sommerfeld grid tuned through the MOMWIRE_SOMM_* environment"},
        {"quadrature-order", "a quadrature order past the frozen rules"},
        {"remainder-order", "a Sommerfeld remainder order past the frozen rules"},
        {"loading-shape", "wire loading that does not give one value per wire"},
        {"loading-values", "a wire conductivity this build does not take"},
        {"grid-layout", "a Sommerfeld grid with more regions than this build lays out"},
        {"buried-fill", "a buried wire without the below-ground field table"},
        // bs2's front and fill (`bspline-` / `bspline: ` stripped below)
        {"degree", "a B-spline degree other than 1, 2 or 3"},
        {"junction-coincidence", "a junction whose wire ends do not meet at one point"},
        {"low-stand-off",
         "a wire graded along a junction arm below the h/a = 2 stand-off floor (momwire#865 / #926)"},
        {"below-table", "a buried wire without the below-ground field table"},
        {"buried-capped-plan", "a buried deck whose grazing pairs lie past the Sommerfeld range cap"},
        {"buried-grazing-refusal", "a buried deck with grazing pairs and none to back off"},
        {"buried-cross-grazing-refusal", "a deck whose cross-interface pairs graze, with none to back off"},
        {"buried-refusal-contact", "a buried wire touching the interface where no junction earns it"},
        {"buried-refusal-crossing", "a wire crossing the interface away from a crossing junction"},
        {"buried-refusal-medium", "a buried wire with no lower medium given"},
        {"buried-refusal-unearned", "a ground contact no junction of a buried wire earns"},
        {"buried-two-radius-spread", "a crossing deck whose radii differ within one side of the interface"},
        {"buried-two-radius-nodes", "a two-radius crossing deck with more than one crossing node"},
        // `somm_trans::Refused`, as bs2's buried fill wraps it
        {"buried-transmitted-r-min-floor", "transmitted-field pairs closer than the grid's radial floor"},
        {"buried-transmitted-ladder-range", "transmitted-field pairs with no depth range to tabulate"},
        {"buried-transmitted-deeper-than-ladder", "a buried wire deeper than the transmitted-field depth ladder"},
        {"buried-transmitted-grid-too-small", "a transmitted-field grid too small to interpolate"},
        {"buried-transmitted-empty-class", "transmitted-field pairs with no wire on one side"},
        {"buried-transmitted-plan-nan", "transmitted-field pairs whose separations are not numbers"},
        {"buried-transmitted-separation-nan", "transmitted-field segments whose lengths are not numbers"},
        {"buried-transmitted-eval-nan", "transmitted-field reads at coordinates that are not numbers"},
        {"buried-transmitted-eval-r-min", "transmitted-field reads closer than the grid's radial floor"},
        {"buried-transmitted-eval-r-max", "transmitted-field reads past the grid's radial reach"},
        {"buried-transmitted-eval-theta-range", "transmitted-field reads at an angle outside the grid"},
        {"buried-transmitted-eval-theta-floor", "transmitted-field reads more grazing than the grid's floor"},
        {"buried-transmitted-eval-ladder", "transmitted-field reads outside the grid's depth ladder"},
        {"buried-buried-cross-range", "cross-interface pairs past the Sommerfeld range cap"},
        {"buried-buried-depth", "a buried wire deeper than the transmitted-field depth ladder"},
        {"buried-buried-cross-grazing", "cross-interface pairs more grazing than the grid's floor"},
        {"buried-refusal: transmitted-empty-class", "cross-interface pairs with no wire on one side"},
        {"buried-refusal: transmitted-plan-nan", "cross-interface pairs whose separations are not numbers"},
        {"buried-refusal: buried-cross-range", "cross-interface pairs past the Sommerfeld range cap"},
        {"buried-refusal: buried-depth", "a buried wire deeper than the transmitted-field depth ladder"},
    };
    std::string head = why, tail;
    const std::size_t colon = why.find(':');
    if (colon != std::string::npos && why.compare(0, colon, "front") == 0) {
        head = why.substr(colon + 1);
        tail = " (front half)";
    }
    // bs2's route label: `bspline-` from its front, `bspline: ` from its fill
    for (const char *label : {"bspline-", "bspline: "}) {
        const std::size_t n = std::strlen(label);
        if (head.compare(0, n, label) == 0) {
            head = head.substr(n);
            break;
        }
    }
    for (const auto &w : words)
        if (head == w.first) return std::string("not served by this build: ") + w.second + tail;
    // a sentence with Python's or numpy's name for its exception: the
    // sentence alone (this build has neither), its seam and route labels
    // stripped as a token's are (census 11-15: bs2 printed
    // `front:bspline-crossing serve with per-wire radii ...`)
    const std::size_t paren = head.find(" (");
    if (paren != std::string::npos) head = head.substr(0, paren);
    return "not served by this build: " + head + tail;
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
    const DeckPragmas pragmas(text);
    raised = false;
    if (served) *served = false;
    if (!cxx_variant_ok())
        return refusal(&text,
                       "not served by this build: this processor lacks AVX2 and FMA, which this "
                       "build's kernels need");
    const Clock::time_point t = Clock::now();
    {
        const std::string tag = std::string(VERSION) + "-" + __DATE__ + "-" + __TIME__
#if MW_EXE_FAT
                                    + "-" + EXE_VARIANT  // one store per variant
#endif
            ;
        rt_serial::configure_grid_cache(tag);
        rt_threads::configure_grid_cache(tag);
    }
    int threads = forced_threads;
    if (forced_threads <= 0) {
        const mw_times::Scope timed(mw_times::seam);
        long long pairs = 0, below = 0;
        const long long n = rt_serial::largest_solve(text, open_end, &pairs, &below);
        threads = n >= 0 ? threads_for(n, pairs, below) : 1;
        if (env("MW_EXE_TIMES") == "1")
            std::fprintf(stderr, "cxx: size n_basis %lld, crossing pairs %lld, below pairs %lld\n", n,
                         pairs, below);
    }
    g_render_threads = threads;
    const std::string arrays_dir = env("MW_EXE_ARRAYS");
    std::string printout, why, kind, message, raised_kind;
    bool failed = false, printed;
    const bool serial = threads == 1 && env("MW_EXE_COPY") != "threads";
    if (threads == 1 && env("MW_EXE_COPY") != "threads") {
        const int blas = forced_threads > 0 ? 1 : std::min(2, hardware_threads());
        set_threads(blas, false);
        times_threads(1, blas);
        if (serial) {
            printed = rt_serial::answer(text, VERSION, EXE_VARIANT, arrays_dir, printout, why, &failed, &kind,
                                       &message, &raised_kind, open_end, served);
        } else {
            omp_set_num_threads(1);
            printed = rt_threads::answer(text, VERSION, EXE_VARIANT, arrays_dir, printout, why, &failed, &kind,
                                      &message, &raised_kind, open_end, served);
        }
    } else {
        set_threads(threads, true);
        times_threads(threads, threads);
        printed = rt_threads::answer(text, VERSION, EXE_VARIANT, arrays_dir, printout, why, &failed, &kind,
                                  &message, &raised_kind, open_end, served);
    }
    if (printed) {
        // An exception Python's render raises (the INTERNAL ERROR printout
        // the route wrote): `run` then writes no .GMO.
        raised = !raised_kind.empty();
        log_route(serial ? "cxx:serial" : "cxx:threads", deck_path);
        if (env("MW_EXE_TIMES") == "1") {
            std::fprintf(stderr, "cxx: threads %d, answer %.3f ms, total %.3f ms\n", threads,
                         ms_since(t), ms_since(T0));
            reuse_line();
        }
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
            // every file-form deck closes at end of file, as the licensed NEC-5
            // closes one (momwire#1698); en_optional, the stdin form's flag from
            // before that (momwire#1488), no longer changes anything
            (void)en_optional;
            text = ml::close_deck(text);
            // gate hook: an exception from the solve, as memory would raise
            const std::string inject = env("MW_EXE_FAIL");
            if (inject == "alloc") throw std::bad_alloc();
            if (inject == "runtime") throw std::runtime_error("injected by MW_EXE_FAIL");
            bool raised = false;
            printout = render(deck, text, forced_threads, raised);
            // the NEC-4.2 slot writes no .GMO (`_shell.render_files`)
            if (!raised && !MW_EXE_DIALECT_NEC4) try {
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
    // the piece's pragma cards hold for its refusals too (`_shell._stream_piece`)
    const DeckPragmas pragmas(piece);
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

#if MW_EXE_DIALECT_NEC2
#include "nec2c_cli.inc"
#endif

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

#if defined(__MINGW32__)
// The static MinGW-w64 build ends the process itself once main's work is
// done, before the static teardown, because of how OpenBLAS's Win32 thread
// server stops: `blas_thread_shutdown_` (blas_server_win32.c) waits 50 ms
// for each pool thread and then TerminateThread()s it. A thread killed that
// way while it holds the process heap lock or the loader lock (a thread on
// its way out takes the loader lock) leaves the lock held for ever, and the
// next free() or LoadLibraryExW in the exit sequence waits on it: the two
// shapes of the rare exit hang (standalone/WINDOWS-SINGLE-FILE.md has the
// backtraces and the soak). OpenBLAS calls it from its own destructor,
// `gotoblas_quit`, on every C-runtime exit, and d35e80c1's explicit shutdown
// called it once more.
//
// So `end_before_static_teardown` is registered with atexit at the top of
// main: the table runs last-in first-out, so it runs AFTER every handler the
// work registered (the diagnostics that report at exit,
// MOMWIRE_SOMM_LAZY_CHECK's among them) and BEFORE those registered before
// main (static destructors, `.dtors`, OpenBLAS's `gotoblas_quit`), and it
// ends the process there with main's status. Every printout is already
// closed (fclose) and stdio is flushed.
//
// MW_EXE_EXIT (the stress job's A/B): `cexit` runs the whole table and then
// TerminateProcess (b554dae3, which still hung in 1 of 44,736); `normal`
// returns from main into the C runtime's exit, with the OpenBLAS pools shut
// down at exit (`shutdown_blas_pools`) unless MW_EXE_EXIT_SHUTDOWN=0;
// `terminate` skips the table, and with it the exit-time diagnostics.
#include <process.h>  // _cexit
int mw_main(int argc, char **argv);
namespace {
// main's status; -1 until main's work is done, so an exit() from inside the
// work (none in this build) keeps the C runtime's own path
int g_exit_rc = -1;
void end_before_static_teardown() {
    if (g_exit_rc < 0) return;
    std::fflush(nullptr);
    TerminateProcess(GetCurrentProcess(), static_cast<UINT>(g_exit_rc));
}
}  // namespace
int main(int argc, char **argv) {
    const std::string how = env("MW_EXE_EXIT");
    if (how.empty()) std::atexit(end_before_static_teardown);
    const int rc = mw_main(argc, argv);
    std::fflush(nullptr);
    if (how == "normal") return rc;
    g_exit_rc = rc;
    if (how != "terminate") _cexit();
    TerminateProcess(GetCurrentProcess(), static_cast<UINT>(rc));
    return rc;
}
int mw_main(int argc, char **argv) {
#else
int main(int argc, char **argv) {
#endif
#if defined(__MINGW32__)
    shutdown_blas_pools_at_exit();
#endif
#if MW_EXE_FAT
    pick_variant();  // before anything reaches a route copy (feature_refusal does)
    if (env("MW_EXE_TIMES") == "1") {
        std::fprintf(stderr, "cxx: simd %s\n", EXE_VARIANT);
        if (!g_name_variant.empty()) std::fprintf(stderr, "cxx: the name asks %s\n", g_name_variant.c_str());
    }
#endif
    if (const int refused = feature_refusal()) return refused;
    times_begin();
#if defined(_WIN32)
    blas_env_before_load();
#endif
    std::vector<std::string> args(argv + 1, argv + argc);
#if MW_EXE_DIALECT_NEC2
    return nec2cli::nec2c_main(args);  // nec2c's command line (nec2c_cli.inc)
#endif
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

// --- large blocks on huge pages (Linux) ---------------------------------------
//
// A warm Python process works on pages it has touched before; every
// executable run is a fresh process whose matrices and window buffers are
// first touched in that run, at one page fault per 4 KiB. On a medium or
// large deck the faults are a visible share of the run (fs_yagi820 sg5 at
// one thread on Skylake: 31,000 faults, ~60 ms of system time in 460 ms,
// against Python's 380 ms warm). So a block of 4 MiB and up is its own
// 2 MiB-aligned anonymous map marked MADV_HUGEPAGE, which the kernel backs
// with 2 MiB pages where it can (`transparent_hugepage` "madvise" or
// "always"): one fault per 2 MiB. It is unmapped when freed, as glibc does
// with its own large blocks. The smaller threshold (1 MiB) and keeping freed
// blocks for re-use were measured too: no faster, and up to +40 % peak RSS
// on small decks from rounding to 2 MiB. Allocation is all this changes: no
// value moves.
#if defined(__linux__)
#include <sys/mman.h>

#include <atomic>
#include <map>
#include <mutex>

namespace {

constexpr std::size_t BIG_MIN = std::size_t(4) << 20;
constexpr std::size_t HUGE_PAGE = std::size_t(2) << 20;

// The bookkeeping's own nodes come from malloc and go back to free, never
// through this file's operator new / delete. A node freed while `mu` is held
// (`live.erase` in big_delete) used to re-enter big_delete, and a node that
// malloc happened to place on a 2 MiB boundary passed its alignment test and
// locked `mu` again: the thread deadlocked on itself, every other thread
// idle (seen at about one 30-point efhw_hi sweep in a hundred at four
// threads, any basis; a core showed the mutex's owner the waiting thread).
// tests/test_standalone_nopython.py forces such a node with a malloc shim.
template <class T>
struct MallocAllocator {
    using value_type = T;
    MallocAllocator() = default;
    template <class U>
    MallocAllocator(const MallocAllocator<U> &) noexcept {}
    T *allocate(std::size_t n) {
        if (void *p = std::malloc(n * sizeof(T))) return static_cast<T *>(p);
        throw std::bad_alloc();
    }
    void deallocate(T *p, std::size_t) noexcept { std::free(p); }
};
template <class T, class U>
bool operator==(const MallocAllocator<T> &, const MallocAllocator<U> &) {
    return true;
}
template <class T, class U>
bool operator!=(const MallocAllocator<T> &, const MallocAllocator<U> &) {
    return false;
}

struct BigBlocks {
    std::mutex mu;
    // block -> mapped length
    std::map<void *, std::size_t, std::less<void *>,
             MallocAllocator<std::pair<void *const, std::size_t>>>
        live;
};
BigBlocks &big_blocks() {
    static BigBlocks *b = new BigBlocks();  // never destroyed: a delete may follow exit's destructors
    return *b;
}

void *big_new(std::size_t n) {
    const std::size_t len = (n + HUGE_PAGE - 1) & ~(HUGE_PAGE - 1);
    void *m = mmap(nullptr, len + HUGE_PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) return nullptr;
    const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(m);
    const std::uintptr_t s = (a + HUGE_PAGE - 1) & ~(HUGE_PAGE - 1);
    if (s > a) munmap(m, s - a);
    if (a + len + HUGE_PAGE > s + len) munmap(reinterpret_cast<void *>(s + len), a + len + HUGE_PAGE - (s + len));
    void *p = reinterpret_cast<void *>(s);
    madvise(p, len, MADV_HUGEPAGE);
    BigBlocks &b = big_blocks();
    std::lock_guard<std::mutex> lock(b.mu);
    b.live[p] = len;
    return p;
}

// true: `p` was one of these blocks (now unmapped)
bool big_delete(void *p) {
    if ((reinterpret_cast<std::uintptr_t>(p) & (HUGE_PAGE - 1)) != 0) return false;
    std::size_t len = 0;
    {
        BigBlocks &b = big_blocks();
        std::lock_guard<std::mutex> lock(b.mu);
        auto it = b.live.find(p);
        if (it == b.live.end()) return false;
        len = it->second;
        b.live.erase(it);
    }
    munmap(p, len);
    return true;
}

}  // namespace

// glibc maps a block of 128 KiB and up on its own (the threshold then
// follows the largest block freed) and trims the heap's top whenever 2x that
// is free, so a fill that frees and asks again -- window buffers, per-pair
// tables -- faults its pages in afresh each time, where a warm Python process
// has long since grown its heap. Once this process has asked for 64 MiB in
// blocks of 64 KiB and up (a medium or large deck; a small one never does),
// blocks under 4 MiB come from the heap and its top is kept up to 16 MiB.
// Measured on Skylake, exe wall: sin42 fs_yagi820 248 -> 160 ms at one
// thread (Python 229), fs_yagi3030 4055 -> 2864 (3545). Applied from the
// start it also cost small decks up to +5 MB (+33 %) of peak RSS; a fixed
// 1 MiB threshold instead slowed bspline's buried fills 5-20 %. A caller's
// own GLIBC_TUNABLES naming either threshold is kept.
#if defined(__GLIBC__)
constexpr std::size_t HEAP_BLOCK_MIN = std::size_t(64) << 10, HEAP_CHURN_MIN = std::size_t(64) << 20;
void heap_churn(std::size_t n) {
    static std::atomic<std::size_t> churn{0};
    static std::atomic<bool> done{false};
    if (n < HEAP_BLOCK_MIN || done.load(std::memory_order_relaxed)) return;
    if (churn.fetch_add(n, std::memory_order_relaxed) + n < HEAP_CHURN_MIN) return;
    if (done.exchange(true)) return;
    const char *tun = std::getenv("GLIBC_TUNABLES");
    if (tun != nullptr && (std::strstr(tun, "mmap_threshold") || std::strstr(tun, "trim_threshold"))) return;
    mallopt(M_MMAP_THRESHOLD, 4 << 20);
    mallopt(M_TRIM_THRESHOLD, 16 << 20);
}
#endif

void *operator new(std::size_t n) {
#if defined(__GLIBC__)
    heap_churn(n);
#endif
    if (n >= BIG_MIN)
        if (void *p = big_new(n)) return p;
    if (void *p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void *p) noexcept {
    if (p != nullptr && !big_delete(p)) std::free(p);
}
void operator delete(void *p, std::size_t) noexcept { operator delete(p); }
#endif
