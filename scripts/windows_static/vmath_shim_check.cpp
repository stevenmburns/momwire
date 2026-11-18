// The MinGW vector-math shim's ABI and edge cases, run on Windows by
// windows-static.yml (standalone/mingw_vmath_shim.cpp says what the shim is).
//
//   g++ -O2 -fno-tree-vectorize -fopenmp-simd -fno-math-errno -ffp-contract=off
//       -mavx2 -mfma -DMW_VMATH_SHIM_COUNT -Isrc/momwire
//       scripts/windows_static/vmath_shim_check.cpp standalone/mingw_vmath_shim.cpp
//
// 1. ABI, through a hand-written caller (`mw_abi_probe`, not gcc's): the
//    callee-saved XMM6-XMM15 and RBX RBP RDI RSI R12-R15 keep sentinel values
//    across every path, RAX returns the result pointer, RSP, DF and MXCSR are
//    kept, misaligned result and argument pointers and both stack alignments
//    work, and nothing outside the 32 result bytes is written.
// 2. gcc's own caller: an `omp simd` loop calls the shim (counted) and agrees
//    bit for bit with the direct calls.
// 3. Accuracy against the C runtime's scalar functions, in ulps.
// 4. The fallback edges: a lane beyond MW_VMATH_TRIG_MAX, NaN, infinities,
//    exp overflow, in every lane position, give the scalar answer in ALL FOUR
//    lanes; signed zeros, subnormals and exp underflow.
//
// Exit status 0 only if every check passed.
#include <immintrin.h>

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

extern "C" {
__m256d _ZGVdN4v_sin(__m256d);
__m256d _ZGVdN4v_cos(__m256d);
__m256d _ZGVdN4v_exp(__m256d);
void mw_vmath_shim_counts(uint64_t out[6]);
}

// gcc's caller, as the cores' declarations make it
#pragma omp declare simd notinbranch simdlen(4)
extern "C" double sin(double);
#pragma omp declare simd notinbranch simdlen(4)
extern "C" double cos(double);
#pragma omp declare simd notinbranch simdlen(4)
extern "C" double exp(double);

namespace {

// The reference: through volatile pointers, so never gcc's vector call.
double (*volatile ref_sin)(double) = ::sin;
double (*volatile ref_cos)(double) = ::cos;
double (*volatile ref_exp)(double) = ::exp;

int g_fail = 0;
void fail(const char *what) {
    if (g_fail < 40) std::printf("FAIL %s\n", what);
    ++g_fail;
}

uint64_t bits(double x) {
    uint64_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}
double from_bits(uint64_t u) {
    double x;
    std::memcpy(&x, &u, sizeof x);
    return x;
}
// the same value: the same bits, or both NaN (a NaN's payload is not promised)
bool same(double a, double b) { return (std::isnan(a) && std::isnan(b)) || bits(a) == bits(b); }

// distance in representable doubles
uint64_t ulps(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return (std::isnan(a) && std::isnan(b)) ? 0 : UINT64_MAX;
    const auto key = [](double x) {
        const uint64_t u = bits(x);
        return (u >> 63) ? static_cast<int64_t>(0) - static_cast<int64_t>(u & ~(1ULL << 63))
                         : static_cast<int64_t>(u);
    };
    const int64_t ka = key(a), kb = key(b);
    return ka > kb ? static_cast<uint64_t>(ka) - static_cast<uint64_t>(kb)
                   : static_cast<uint64_t>(kb) - static_cast<uint64_t>(ka);
}

// ---------------------------------------------------------------- 1. the ABI
struct AbiState {
    uint64_t xmm_in[20];    // 0: XMM6..XMM15
    uint64_t gpr_in[8];     // 160: RBX RBP RDI RSI R12 R13 R14 R15
    uint64_t stack_off;     // 224: 0 or 16, RSP mod 32 at the call
    uint64_t xmm_out[20];   // 232
    uint64_t gpr_out[8];    // 392
    uint64_t ret;           // 456: RAX
    uint64_t rsp_before;    // 464
    uint64_t rsp_after;     // 472
    uint64_t flags;         // 480
    uint32_t mxcsr_before;  // 488
    uint32_t mxcsr_after;   // 492
};

extern "C" void mw_abi_probe(void *fn, void *result, const void *arg, AbiState *st);
extern "C" {
AbiState *mw_abi_st;
uint64_t mw_abi_rsp;
}

// RCX = fn, RDX = result, R8 = arg, R9 = st. The host's callee-saved registers
// go on the stack, every callee-saved register takes its sentinel, and the
// callee is entered as the Microsoft ABI says: result in RCX, argument in RDX,
// 32 bytes of shadow space, RSP 16-byte aligned before the CALL.
asm(R"(
    .text
    .globl mw_abi_probe
    .def mw_abi_probe; .scl 2; .type 32; .endef
mw_abi_probe:
    pushq %rbp
    pushq %rbx
    pushq %rdi
    pushq %rsi
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    subq $168, %rsp
    movups %xmm6, 0(%rsp)
    movups %xmm7, 16(%rsp)
    movups %xmm8, 32(%rsp)
    movups %xmm9, 48(%rsp)
    movups %xmm10, 64(%rsp)
    movups %xmm11, 80(%rsp)
    movups %xmm12, 96(%rsp)
    movups %xmm13, 112(%rsp)
    movups %xmm14, 128(%rsp)
    movups %xmm15, 144(%rsp)
    movq %r9, mw_abi_st(%rip)
    movq %rsp, mw_abi_rsp(%rip)
    movq %rcx, %rax
    movq %rdx, %rcx
    movq %r8, %rdx
    movq %r9, %r11
    andq $-32, %rsp
    subq $64, %rsp
    subq 224(%r11), %rsp
    movq %rsp, 464(%r11)
    stmxcsr 488(%r11)
    movups 0(%r11), %xmm6
    movups 16(%r11), %xmm7
    movups 32(%r11), %xmm8
    movups 48(%r11), %xmm9
    movups 64(%r11), %xmm10
    movups 80(%r11), %xmm11
    movups 96(%r11), %xmm12
    movups 112(%r11), %xmm13
    movups 128(%r11), %xmm14
    movups 144(%r11), %xmm15
    movq 160(%r11), %rbx
    movq 168(%r11), %rbp
    movq 176(%r11), %rdi
    movq 184(%r11), %rsi
    movq 192(%r11), %r12
    movq 200(%r11), %r13
    movq 208(%r11), %r14
    movq 216(%r11), %r15
    cld
    call *%rax
    movq mw_abi_st(%rip), %r11
    movq %rax, 456(%r11)
    movq %rsp, 472(%r11)
    movups %xmm6, 232(%r11)
    movups %xmm7, 248(%r11)
    movups %xmm8, 264(%r11)
    movups %xmm9, 280(%r11)
    movups %xmm10, 296(%r11)
    movups %xmm11, 312(%r11)
    movups %xmm12, 328(%r11)
    movups %xmm13, 344(%r11)
    movups %xmm14, 360(%r11)
    movups %xmm15, 376(%r11)
    movq %rbx, 392(%r11)
    movq %rbp, 400(%r11)
    movq %rdi, 408(%r11)
    movq %rsi, 416(%r11)
    movq %r12, 424(%r11)
    movq %r13, 432(%r11)
    movq %r14, 440(%r11)
    movq %r15, 448(%r11)
    stmxcsr 492(%r11)
    pushfq
    popq %rax
    movq %rax, 480(%r11)
    movq mw_abi_rsp(%rip), %rsp
    movups 0(%rsp), %xmm6
    movups 16(%rsp), %xmm7
    movups 32(%rsp), %xmm8
    movups 48(%rsp), %xmm9
    movups 64(%rsp), %xmm10
    movups 80(%rsp), %xmm11
    movups 96(%rsp), %xmm12
    movups 112(%rsp), %xmm13
    movups 128(%rsp), %xmm14
    movups 144(%rsp), %xmm15
    addq $168, %rsp
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rsi
    popq %rdi
    popq %rbx
    popq %rbp
    ret
)");

typedef __m256d (*vfn)(__m256d);
struct Fn {
    const char *name;
    vfn vec;
    double (*volatile *scalar)(double);
};
const Fn FNS[3] = {{"sin", _ZGVdN4v_sin, &ref_sin},
                   {"cos", _ZGVdN4v_cos, &ref_cos},
                   {"exp", _ZGVdN4v_exp, &ref_exp}};

int abi_cases = 0;

void abi_case(const Fn &f, const double in[4], const char *label) {
    for (uint64_t stack_off : {0ULL, 16ULL}) {
        for (size_t mis_r : {0u, 8u, 16u, 24u}) {
            for (size_t mis_a : {0u, 8u, 16u, 24u}) {
                alignas(64) unsigned char rbuf[128], abuf[128];
                std::memset(rbuf, 0xA5, sizeof rbuf);
                std::memset(abuf, 0x5A, sizeof abuf);
                unsigned char *rp = rbuf + 32 + mis_r, *ap = abuf + 32 + mis_a;
                std::memcpy(ap, in, 32);
                AbiState st;
                std::memset(&st, 0, sizeof st);
                for (int i = 0; i < 20; ++i) st.xmm_in[i] = 0x1111111111111111ULL * (i % 15 + 1);
                for (int i = 0; i < 8; ++i) st.gpr_in[i] = 0xC0C0C0C000000000ULL + i;
                st.stack_off = stack_off;
                ++abi_cases;
                mw_abi_probe(reinterpret_cast<void *>(f.vec), rp, ap, &st);
                char where[160];
                std::snprintf(where, sizeof where, "%s %s stack+%llu result+%zu arg+%zu", f.name,
                              label, (unsigned long long)stack_off, mis_r, mis_a);
                char msg[220];
                const auto check = [&](bool ok, const char *what) {
                    if (ok) return;
                    std::snprintf(msg, sizeof msg, "abi %s: %s", where, what);
                    fail(msg);
                };
                check(std::memcmp(st.xmm_in, st.xmm_out, sizeof st.xmm_in) == 0,
                      "XMM6-XMM15 not preserved");
                check(std::memcmp(st.gpr_in, st.gpr_out, sizeof st.gpr_in) == 0,
                      "RBX RBP RDI RSI R12-R15 not preserved");
                check(st.ret == reinterpret_cast<uint64_t>(rp), "RAX is not the result pointer");
                check(st.rsp_before == st.rsp_after, "RSP moved");
                check((st.flags & 0x400) == 0, "DF set on return");
                check((st.mxcsr_before & 0xFFC0) == (st.mxcsr_after & 0xFFC0), "MXCSR control changed");
                check(std::memcmp(ap, in, 32) == 0, "the argument was written");
                bool guard = true;
                for (size_t i = 0; i < sizeof rbuf; ++i) {
                    const size_t off = 32 + mis_r;
                    if ((i < off || i >= off + 32) && rbuf[i] != 0xA5) guard = false;
                }
                check(guard, "wrote outside the 32 result bytes");
                double out[4];
                std::memcpy(out, rp, 32);
                const __m256d direct = f.vec(_mm256_loadu_pd(in));
                double want[4];
                _mm256_storeu_pd(want, direct);
                for (int l = 0; l < 4; ++l) check(same(out[l], want[l]), "the value differs from a direct call");
            }
        }
    }
}

// ---------------------------------------------- 2. gcc's vectorised caller
__attribute__((noinline)) void loop_sin(const double *a, double *b, int n) {
#pragma omp simd
    for (int i = 0; i < n; ++i) b[i] = sin(a[i]);
}
__attribute__((noinline)) void loop_cos(const double *a, double *b, int n) {
#pragma omp simd
    for (int i = 0; i < n; ++i) b[i] = cos(a[i]);
}
__attribute__((noinline)) void loop_exp(const double *a, double *b, int n) {
#pragma omp simd
    for (int i = 0; i < n; ++i) b[i] = exp(a[i]);
}

void gcc_caller() {
    uint64_t c0[6], c1[6];
    enum { N = 4000 };
    static double a[N], b[N];
    std::mt19937_64 rng(1441);
    std::uniform_real_distribution<double> u(-50.0, 50.0);
    for (int i = 0; i < N; ++i) a[i] = u(rng);
    void (*const loops[3])(const double *, double *, int) = {loop_sin, loop_cos, loop_exp};
    for (int f = 0; f < 3; ++f) {
        mw_vmath_shim_counts(c0);
        loops[f](a, b, N);
        mw_vmath_shim_counts(c1);
        const uint64_t calls = c1[2 * f] - c0[2 * f];
        std::printf("gcc's omp simd loop, %s: %" PRIu64 " vector calls for %d elements\n", FNS[f].name, calls, N);
        if (calls < N / 4 - 8) fail("gcc's loop did not reach the shim");
        for (int i = 0; i + 4 <= N; i += 4) {
            double want[4];
            _mm256_storeu_pd(want, FNS[f].vec(_mm256_loadu_pd(a + i)));
            for (int l = 0; l < 4; ++l)
                if (bits(want[l]) != bits(b[i + l])) {
                    fail("gcc's loop differs from the direct call");
                    break;
                }
        }
    }
}

// ------------------------------------------------------------ 3. accuracy
void accuracy(const Fn &f, double lo, double hi, uint64_t gate) {
    std::mt19937_64 rng(1441);
    std::uniform_real_distribution<double> u(lo, hi);
    uint64_t worst = 0;
    double worst_x = 0;
    const int n = 2000000;
    for (int i = 0; i < n; i += 4) {
        double x[4], y[4];
        for (int l = 0; l < 4; ++l) x[l] = u(rng);
        _mm256_storeu_pd(y, f.vec(_mm256_loadu_pd(x)));
        for (int l = 0; l < 4; ++l) {
            const uint64_t d = ulps(y[l], (*f.scalar)(x[l]));
            if (d > worst) {
                worst = d;
                worst_x = x[l];
            }
        }
    }
    std::printf("%s on [%g, %g]: worst %" PRIu64 " ulp from the scalar function (x = %.17g), %d lanes\n",
                f.name, lo, hi, worst, worst_x, n);
    if (worst > gate) fail("accuracy");
}

// ------------------------------------------------------- 4. fallback edges
// Every lane position: `bad` there and ordinary values elsewhere. The whole
// vector must be the scalar function's, bit for bit.
void fallback_lanes(const Fn &f, double bad, const double ordinary[3]) {
    for (int pos = 0; pos < 4; ++pos) {
        double x[4], y[4];
        for (int l = 0, o = 0; l < 4; ++l) x[l] = (l == pos) ? bad : ordinary[o++];
        _mm256_storeu_pd(y, f.vec(_mm256_loadu_pd(x)));
        for (int l = 0; l < 4; ++l)
            if (!same(y[l], (*f.scalar)(x[l]))) {
                char msg[160];
                std::snprintf(msg, sizeof msg, "%s fallback: bad %.17g at lane %d, lane %d = %.17g, scalar %.17g",
                              f.name, bad, pos, l, y[l], (*f.scalar)(x[l]));
                fail(msg);
            }
    }
}

// All four lanes `x`: the fast path's answer against the scalar function.
void exact_value(const Fn &f, double x, double want) {
    double y[4];
    _mm256_storeu_pd(y, f.vec(_mm256_set1_pd(x)));
    for (int l = 0; l < 4; ++l)
        if (!same(y[l], want)) {
            char msg[160];
            std::snprintf(msg, sizeof msg, "%s(%.17g) = %.17g, want %.17g", f.name, x, y[l], want);
            fail(msg);
        }
}

void near_value(const Fn &f, double x, uint64_t gate) {
    double y[4];
    _mm256_storeu_pd(y, f.vec(_mm256_set1_pd(x)));
    const double want = (*f.scalar)(x);
    for (int l = 0; l < 4; ++l)
        if (ulps(y[l], want) > gate) {
            char msg[160];
            std::snprintf(msg, sizeof msg, "%s(%.17g) = %.17g, scalar %.17g", f.name, x, y[l], want);
            fail(msg);
        }
}

void edges() {
    const double inf = HUGE_VAL, nan = std::nan("");
    const double trig_ok[3] = {0.5, -1.25, 3.0};
    const double exp_ok[3] = {0.5, -1.25, 3.0};
    for (int f = 0; f < 2; ++f) {
        for (double bad : {1.0e5 * (1 + 1e-15), -1.0e5 * (1 + 1e-15), 1.5e5, -2.0e6, 1.0e22, 1.0e300,
                           -1.7e308, inf, -inf, nan})
            fallback_lanes(FNS[f], bad, trig_ok);
    }
    for (double bad : {709.0000000001, 709.5, 709.78, 709.79, 710.0, 1000.0, 1.0e308, inf, nan})
        fallback_lanes(FNS[2], bad, exp_ok);
    // the boundary itself is the fast path: close to scalar, finite
    near_value(FNS[0], 1.0e5, 4);
    near_value(FNS[0], -1.0e5, 4);
    near_value(FNS[1], 1.0e5, 4);
    near_value(FNS[2], 709.0, 4);
    // signed zeros and the smallest arguments
    exact_value(FNS[0], 0.0, 0.0);
    exact_value(FNS[0], -0.0, -0.0);
    exact_value(FNS[1], 0.0, 1.0);
    exact_value(FNS[1], -0.0, 1.0);
    exact_value(FNS[2], 0.0, 1.0);
    exact_value(FNS[2], -0.0, 1.0);
    const double denorm = from_bits(1), tiny = 1.0e-300;
    exact_value(FNS[0], denorm, denorm);
    exact_value(FNS[0], -denorm, -denorm);
    exact_value(FNS[0], tiny, tiny);
    exact_value(FNS[1], denorm, 1.0);
    exact_value(FNS[2], denorm, 1.0);
    // exp's underflow: the gradual range, then zero; the clamp at -1000
    for (double x : {-700.0, -708.0, -708.4, -720.0, -740.0, -744.0, -745.0}) near_value(FNS[2], x, 2);
    for (double x : {-745.2, -746.0, -800.0, -1000.0, -1000.5, -1.0e4, -1.0e308, -inf}) exact_value(FNS[2], x, 0.0);
    // mixed: an underflowing lane is no reason to fall back
    uint64_t c0[6], c1[6];
    mw_vmath_shim_counts(c0);
    double x[4] = {-800.0, 1.0, -1.0e308, 2.0}, y[4];
    _mm256_storeu_pd(y, _ZGVdN4v_exp(_mm256_loadu_pd(x)));
    mw_vmath_shim_counts(c1);
    if (c1[4] != c0[4] + 1 || c1[5] != c0[5]) fail("exp underflow left the fast path");
    if (y[0] != 0.0 || y[2] != 0.0 || ulps(y[1], (*ref_exp)(1.0)) > 2 || ulps(y[3], (*ref_exp)(2.0)) > 2)
        fail("exp mixed underflow lanes");
    // the fallback counters move for a bad lane
    mw_vmath_shim_counts(c0);
    (void)_ZGVdN4v_sin(_mm256_set_pd(1.0, 2.0, nan, 3.0));
    (void)_ZGVdN4v_cos(_mm256_set_pd(1.0, 1.0e6, 2.0, 3.0));
    (void)_ZGVdN4v_exp(_mm256_set_pd(710.0, 1.0, 2.0, 3.0));
    mw_vmath_shim_counts(c1);
    if (c1[1] != c0[1] + 1 || c1[3] != c0[3] + 1 || c1[5] != c0[5] + 1) fail("a bad lane did not take the fallback");
}

}  // namespace

int main() {
    const double fast[4] = {0.5, -1.25, 3.0, 100.0};
    const double exp_fast[4] = {0.5, -1.25, 3.0, -20.0};
    const double big[4] = {0.5, 2.0e6, 3.0, 100.0};
    const double hasnan[4] = {0.5, std::nan(""), 3.0, 100.0};
    const double over[4] = {0.5, 800.0, 3.0, -20.0};
    for (int f = 0; f < 2; ++f) {
        abi_case(FNS[f], fast, "fast path");
        abi_case(FNS[f], big, "fallback (beyond the range)");
        abi_case(FNS[f], hasnan, "fallback (NaN)");
    }
    abi_case(FNS[2], exp_fast, "fast path");
    abi_case(FNS[2], over, "fallback (overflow)");
    abi_case(FNS[2], hasnan, "fallback (NaN)");
    std::printf("ABI: %d probed calls\n", abi_cases);

    gcc_caller();

    accuracy(FNS[0], -3.2, 3.2, 2);
    accuracy(FNS[1], -3.2, 3.2, 2);
    accuracy(FNS[0], -1.6e4, 1.6e4, 2);
    accuracy(FNS[1], -1.6e4, 1.6e4, 2);
    accuracy(FNS[0], -1.0e5, 1.0e5, 2);
    accuracy(FNS[1], -1.0e5, 1.0e5, 2);
    accuracy(FNS[2], -700.0, 700.0, 2);
    accuracy(FNS[2], -5.0, 5.0, 2);

    edges();

    if (g_fail) {
        std::printf("vmath_shim_check: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("vmath_shim_check: all passed\n");
    return 0;
}
