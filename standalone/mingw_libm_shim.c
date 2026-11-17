/* The MinGW-w64 executables' libm, forwarded to a system DLL's
 * (MW_EXE_LIBM_SHIM=msvcrt|ucrtbase, CMakeLists.txt; an EXPERIMENT for
 * standalone/WINDOWS-SINGLE-FILE.md, not a release build).
 *
 * WHY. mingw-w64's libmingwex defines its own exp, log, pow, sin, cos,
 * sincos, atan2 and hypot (libm_bench.c, run 44057069874: 24-79 ns a call,
 * where ucrtbase.dll's, MSVC's, take 4-12), and it
 * is ahead of the CRT's import library on gcc's link line, so a MinGW build
 * never reaches the DLL's even when the DLL exports them. Defining the names
 * here, in an object, keeps libmingwex's members out of the link (an archive
 * member is pulled only for a symbol nothing defines yet), and libmingwex's
 * own users (its cexp among them) bind to these too.
 *
 * Each function is looked up once, at start-up, with GetProcAddress, so the
 * DLL is NOT in the import table: the import-table gate cannot see it, and
 * on a Windows 7 without the UCRT (KB2999226) MW_LIBM_SHIM_DLL="ucrtbase.dll"
 * fails at start-up with the message below. msvcrt.dll is on every Windows.
 * A product build would import the functions instead (a dlltool import
 * library), so the gates see the DLL; the lookup is only this experiment's
 * way to try both DLLs from one toolchain.
 *
 * The values are the DLL's: no bit-compatibility with the mingw-w64 build is
 * claimed or needed (PRINTOUT_RTOL_WIN32, windows-static.yml's corpus gate).
 */
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#ifndef MW_LIBM_SHIM_DLL
#error "MW_LIBM_SHIM_DLL: the DLL to forward to (msvcrt.dll or ucrtbase.dll)"
#endif

typedef double (*mw_d1)(double);
typedef double (*mw_d2)(double, double);

static mw_d1 p_exp, p_log, p_sin, p_cos;
static mw_d2 p_pow, p_atan2, p_hypot;

static FARPROC need(HMODULE h, const char *name) {
    FARPROC f = GetProcAddress(h, name);
    if (f == NULL) {
        fprintf(stderr, "momwire: %s has no %s (MW_EXE_LIBM_SHIM)\n", MW_LIBM_SHIM_DLL, name);
        ExitProcess(70);
    }
    return f;
}

/* With the earliest constructors (OpenBLAS's and libgomp's call no libm, but
 * a C++ dynamic initializer might). */
__attribute__((constructor(101))) static void mw_libm_shim_init(void) {
    HMODULE h = LoadLibraryExA(MW_LIBM_SHIM_DLL, NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (h == NULL) h = LoadLibraryA(MW_LIBM_SHIM_DLL);
    if (h == NULL) {
        fprintf(stderr, "momwire: this build needs %s (on Windows 7: the Universal C Runtime, KB2999226)\n",
                MW_LIBM_SHIM_DLL);
        ExitProcess(70);
    }
    p_exp = (mw_d1)need(h, "exp");
    p_log = (mw_d1)need(h, "log");
    p_sin = (mw_d1)need(h, "sin");
    p_cos = (mw_d1)need(h, "cos");
    p_pow = (mw_d2)need(h, "pow");
    p_atan2 = (mw_d2)need(h, "atan2");
    p_hypot = (mw_d2)need(h, "_hypot");  /* msvcrt.dll exports only the underscored name */
    if (getenv("MW_EXE_TIMES") != NULL) fprintf(stderr, "cxx: libm from %s\n", MW_LIBM_SHIM_DLL);
}

double exp(double x) { return p_exp(x); }
double log(double x) { return p_log(x); }
double sin(double x) { return p_sin(x); }
double cos(double x) { return p_cos(x); }
double pow(double x, double y) { return p_pow(x, y); }
double atan2(double y, double x) { return p_atan2(y, x); }
double hypot(double x, double y) { return p_hypot(x, y); }
double _hypot(double x, double y) { return p_hypot(x, y); }
/* gcc fuses sin(x) and cos(x) of one x into sincos, which neither DLL has */
void sincos(double x, double *s, double *c) {
    *s = p_sin(x);
    *c = p_cos(x);
}
