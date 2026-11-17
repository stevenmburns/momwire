/* A program with no momwire code in it, against the same VC++ runtime DLLs
   (windows-fat.yml): what the C runtime decided about the CPU, and a large
   memcpy (VCRUNTIME140.dll's, which picks its own kernel by that decision).
   Run natively and under Intel SDE as Westmere, it says whether an SDE fault
   inside VCRUNTIME140.dll belongs to the DLL set rather than to the
   executable that loaded it.

       cl /nologo /O2 /MD isa_probe.c
*/
#include <intrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int __isa_available; /* the executable's own copy, set at its start-up */

int main(void) {
    int r[4];
    __cpuid(r, 1);
    printf("cpuid.1 ecx avx=%d osxsave=%d fma=%d; exe __isa_available=%d\n", (r[2] >> 28) & 1,
           (r[2] >> 27) & 1, (r[2] >> 12) & 1, __isa_available);
    fflush(stdout);
    size_t n = 1 << 22;
    char *a = malloc(n), *b = malloc(n);
    memset(a, 7, n);
    memcpy(b, a, n);
    printf("memcpy ok: %d\n", b[n - 1]);
    return 0;
}
