// momwire-razor2p: `momwire-nec5-razor-2p` as one native executable.
//
//     momwire-razor2p DECK PRINTOUT        (EZNEC's and SimNEC's form)
//     momwire-razor2p < PATHS              (4nec2's: the two names on stdin)
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
// THREADS. The executable sets its own count (MOMWIRE_THREADS, else every
// hardware thread) on its OpenBLAS and on OpenMP, and exports it as
// OPENBLAS_NUM_THREADS / OMP_NUM_THREADS before CPython starts, so the
// embedded numpy and scipy run on the same count.
//
// Gate hooks (environment; never set in use):
//   MW_EXE_ROUTE_LOG=FILE    append "<route> <deck>" (cxx | fallback:<why>)
//   MW_EXE_FORCE_FALLBACK=1  answer every deck through the fallback
//   MW_EXE_ARRAYS=DIR        the C++ route writes each finished run's arrays
//   MW_EXE_PYSTARTUP=FILE    the fallback execs FILE before answering
//   MW_EXE_TIMES=1           a timing line on stderr
//   MW_EXE_COPY=threads      one thread through the OpenMP copy, not the serial one
#include "_nec5_deck_core.h"
#include "_nec5_razor_printout_core.h"
#include "route.h"
#define mw mws
#include "route.h"
#undef mw

#define PY_SSIZE_T_CLEAN
#include <Python.h>  // declarations and PyConfig's layout only; NOT linked

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#else
#include <dirent.h>
#include <dlfcn.h>
#include <unistd.h>
#endif

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
#include <thread>
#include <vector>

#ifndef MW_VERSION
#error "MW_VERSION (the momwire release this executable is) must be defined"
#endif

#include "blas.h"

extern "C" void omp_set_num_threads(int);

namespace {

const char *const BASIS = "razor-2p";
const char *const VERSION = MW_VERSION;

using Clock = std::chrono::steady_clock;
const Clock::time_point T0 = Clock::now();

double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
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

// Below this many basis functions a deck's solve runs on one thread, through
// the serial copy (no OpenMP runtime): starting threads costs more than the
// parallel fill saves on a small deck (Rocketlake, dip21: ~0.4 ms answered on
// one thread, ~0.75 ms on eight). Unit 5 measures where the crossover is.
constexpr long long SMALL_SOLVE = 256;

// The fallback's thread count (MOMWIRE_THREADS, else every hardware thread).
int fallback_threads = 1;

int hardware_threads() {
    const int n = static_cast<int>(std::thread::hardware_concurrency());
    return n > 0 ? n : 1;
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

// --- the C++ route's preconditions ------------------------------------------

// The C++ kernels are the avx2 variant's; they answer only where Python would
// load that variant (`momwire._accel`: AVX2 + FMA, no MOMWIRE_FORCE_VARIANT
// naming another).
bool cxx_variant_ok() {
    const std::string forced = env("MOMWIRE_FORCE_VARIANT");
    if (!forced.empty() && forced != "avx2") return false;
#if defined(_WIN32)
    return IsProcessorFeaturePresent(40 /* PF_AVX2_INSTRUCTIONS_AVAILABLE */) != 0;
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

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
    std::fprintf(stderr, "momwire-razor2p: %s\n", what.c_str());
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
        VERSION + "' + ': the momwire beside it is ' + str(_v))\n";
    if (py.p_PyRun_SimpleString(check.c_str()) != 0) std::exit(3);
    const std::string startup = env("MW_EXE_PYSTARTUP");
    if (!startup.empty())
        py.p_PyRun_SimpleString(("exec(compile(open(" + py_str(startup) + ", 'rb').read(), " +
                                 py_str(startup) + ", 'exec'))\n")
                                    .c_str());
    const Clock::time_point t1 = Clock::now();
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
    mw::nec5::EngineStamp stamp{VERSION, "avx2", ""};
    const std::string printout = mw::nec5::printout_bytes(
        mw::nec5::render_refusal(readable ? &text : nullptr, why, BASIS, stamp));
    if (std::FILE *w = std::fopen(out.c_str(), "wb")) {
        std::fwrite(printout.data(), 1, printout.size(), w);
        std::fclose(w);
    }
}

}  // namespace

int main(int argc, char **argv) {
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
        // The size policy: a deck whose largest solve is under SMALL_SOLVE
        // basis functions runs on one thread, through the serial copy (no
        // OpenMP); a larger one on every hardware thread. MOMWIRE_THREADS
        // overrides it: 1 is the serial copy, more the threaded one.
        if (forced_threads <= 0) {
            const long long n = mws::exe::largest_solve(text);
            threads = n >= 0 && n < SMALL_SOLVE ? 1 : hardware_threads();
        }
        const std::string arrays_dir = env("MW_EXE_ARRAYS");
        std::string printout;
        bool printed;
        if (threads == 1 && env("MW_EXE_COPY") != "threads") {
            set_threads(1, false);
            printed = mws::exe::answer(text, VERSION, "avx2", arrays_dir, printout, why);
        } else {
            set_threads(threads, true);
            printed = mw::exe::answer(text, VERSION, "avx2", arrays_dir, printout, why);
        }
        if (printed) {
            std::FILE *w = std::fopen(out.c_str(), "wb");
            if (w != nullptr) {
                std::fwrite(printout.data(), 1, printout.size(), w);
                if (std::fclose(w) == 0) {
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
