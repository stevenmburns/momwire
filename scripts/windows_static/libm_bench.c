/* ns per call of the libm functions the cores call, from three places, on one
 * machine (windows-static.yml; standalone/WINDOWS-SINGLE-FILE.md):
 *
 *   link     the function this toolchain's link resolves (`exp` as written:
 *            libmingwex's in MINGW64, whatever UCRT64/CLANG64 bind)
 *   msvcrt   msvcrt.dll's, through GetProcAddress
 *   ucrtbase ucrtbase.dll's (what an MSVC /MD build calls), likewise
 *
 *     gcc -O2 -fno-builtin libm_bench.c -o libm_bench.exe && libm_bench.exe
 *
 * Every call goes through a volatile function pointer, so the three are
 * timed alike (no inlining, no vectorizing); inputs are a 4096-entry table
 * in each function's working range, cycled. Also prints, per function, the
 * largest difference in ulps between the DLLs' results and the link's, on
 * the same inputs (how far a forwarded build's values can move).
 */
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <windows.h>

#define N 4096
#define REPS 400

typedef double (*d1)(double);
typedef double (*d2)(double, double);

static double now(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

static double xa[N], xb[N];
static volatile double sink;

static double time1(d1 volatile f) {
    double best = 1e30;
    for (int t = 0; t < 5; t++) {
        double acc = 0, t0 = now();
        for (int r = 0; r < REPS; r++)
            for (int i = 0; i < N; i++) acc += f(xa[i]);
        double dt = (now() - t0) / ((double)REPS * N) * 1e9;
        sink = acc;
        if (dt < best) best = dt;
    }
    return best;
}

static double time2(d2 volatile f) {
    double best = 1e30;
    for (int t = 0; t < 5; t++) {
        double acc = 0, t0 = now();
        for (int r = 0; r < REPS; r++)
            for (int i = 0; i < N; i++) acc += f(xa[i], xb[i]);
        double dt = (now() - t0) / ((double)REPS * N) * 1e9;
        sink = acc;
        if (dt < best) best = dt;
    }
    return best;
}

static int64_t ord(double x) {
    int64_t i;
    memcpy(&i, &x, 8);
    return i < 0 ? INT64_MIN - i : i;
}

static double ulps1(d1 a, d1 b) {
    int64_t m = 0;
    for (int i = 0; i < N; i++) {
        int64_t d = ord(a(xa[i])) - ord(b(xa[i]));
        if (d < 0) d = -d;
        if (d > m) m = d;
    }
    return (double)m;
}

static double ulps2(d2 a, d2 b) {
    int64_t m = 0;
    for (int i = 0; i < N; i++) {
        int64_t d = ord(a(xa[i], xb[i])) - ord(b(xa[i], xb[i]));
        if (d < 0) d = -d;
        if (d > m) m = d;
    }
    return (double)m;
}

static void fill(double lo, double hi, double lo2, double hi2) {
    for (int i = 0; i < N; i++) {
        double u = (i * 0.6180339887498949) - (int)(i * 0.6180339887498949);
        double v = (i * 0.7548776662466927) - (int)(i * 0.7548776662466927);
        xa[i] = lo + (hi - lo) * u;
        xb[i] = lo2 + (hi2 - lo2) * v;
    }
}

static double sincos_sum(double x) {
    double s, c;
    sincos(x, &s, &c);
    return s + c;
}

static double cexp_re(double x) {
    double complex z = cexp(CMPLX(-0.01 * x, x));
    return creal(z) + cimag(z);
}

int main(void) {
    HMODULE ms = LoadLibraryA("msvcrt.dll"), uc = LoadLibraryA("ucrtbase.dll");
    struct {
        const char *name;
        int arity;
        double lo, hi, lo2, hi2;
        void *link;
    } fns[] = {
        {"exp", 1, -30, 2, 0, 0, (void *)exp},
        {"log", 1, 1e-3, 1e3, 0, 0, (void *)log},
        {"sin", 1, -60, 60, 0, 0, (void *)sin},
        {"cos", 1, -60, 60, 0, 0, (void *)cos},
        {"pow", 2, 0.1, 10, -3, 3, (void *)pow},
        {"atan2", 2, -5, 5, -5, 5, (void *)atan2},
        {"_hypot", 2, -5, 5, -5, 5, (void *)hypot},
    };
    printf("libm ns/call (best of 5), max |ulp| vs link\n");
    printf("| function | link | msvcrt | ucrtbase | ulp msvcrt | ulp ucrtbase |\n|---|---|---|---|---|---|\n");
    for (size_t k = 0; k < sizeof fns / sizeof fns[0]; k++) {
        fill(fns[k].lo, fns[k].hi, fns[k].lo2, fns[k].hi2);
        void *m = ms ? (void *)GetProcAddress(ms, fns[k].name) : NULL;
        void *u = uc ? (void *)GetProcAddress(uc, fns[k].name) : NULL;
        double tl, tm = -1, tu = -1, um = -1, uu = -1;
        if (fns[k].arity == 1) {
            tl = time1((d1)fns[k].link);
            if (m) tm = time1((d1)m), um = ulps1((d1)m, (d1)fns[k].link);
            if (u) tu = time1((d1)u), uu = ulps1((d1)u, (d1)fns[k].link);
        } else {
            tl = time2((d2)fns[k].link);
            if (m) tm = time2((d2)m), um = ulps2((d2)m, (d2)fns[k].link);
            if (u) tu = time2((d2)u), uu = ulps2((d2)u, (d2)fns[k].link);
        }
        printf("| %s | %.1f | %.1f | %.1f | %.0f | %.0f |\n", fns[k].name, tl, tm, tu, um, uu);
    }
    fill(-60, 60, 0, 0);
    printf("| sincos (link) | %.1f | | | | |\n", time1(sincos_sum));
    printf("| cexp (link) | %.1f | | | | |\n", time1(cexp_re));
    return 0;
}
