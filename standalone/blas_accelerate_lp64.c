/* The macOS executable's LAPACK (momwire#1559): Accelerate, as the launcher's
 * SciPy has it. SciPy's macOS 14+ arm64 wheels link Accelerate's new LAPACK
 * with 32-bit integers (the `$NEWLAPACK` symbols), so the LU, the symmetric
 * factor and the network solve's expert driver go to the same entry points
 * here. blas.h's names with MW_BLAS_PREFIX=mwa_; the CBLAS face numpy uses is
 * blas_accelerate_ilp64.c. C, not C++: Accelerate's complex parameter type is
 * C's, and a void pointer converts to it implicitly. */
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <stddef.h>

/* Accelerate sizes its own threads; the launcher's numpy and scipy run on
 * that default whatever OPENBLAS_NUM_THREADS says, so the executable leaves
 * it alone too. */
void mwa_openblas_set_num_threads(int n) { (void)n; }

void mwa_zgetrf_(int *m, int *n, void *a, int *lda, int *ipiv, int *info) {
    zgetrf_(m, n, a, lda, ipiv, info);
}
void mwa_zgetrs_(char *trans, int *n, int *nrhs, void *a, int *lda, int *ipiv, void *b, int *ldb,
                 int *info) {
    zgetrs_(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
}
void mwa_dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info) {
    dgetrf_(m, n, a, lda, ipiv, info);
}
void mwa_dgetrs_(char *trans, int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb,
                 int *info) {
    dgetrs_(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
}
/* `scipy.linalg.qr(..., pivoting=True)` (the Sinusoidal-Galerkin crossing
 * node's constrained solve, momwire#1718 S11b) */
void mwa_zgeqp3_(int *m, int *n, void *a, int *lda, int *jpvt, void *tau, void *work, int *lwork, double *rwork,
                 int *info) {
    zgeqp3_(m, n, a, lda, jpvt, tau, work, lwork, rwork, info);
}
void mwa_zsytrf_(char *uplo, int *n, void *a, int *lda, int *ipiv, void *work, int *lwork, int *info) {
    zsytrf_(uplo, n, a, lda, ipiv, work, lwork, info);
}
void mwa_zsytrs_(char *uplo, int *n, int *nrhs, void *a, int *lda, int *ipiv, void *b, int *ldb,
                 int *info) {
    zsytrs_(uplo, n, nrhs, a, lda, ipiv, b, ldb, info);
}
/* The trailing string lengths are OpenBLAS's Fortran ABI; Accelerate's
 * prototype has none. */
void mwa_zgesvx_(const char *fact, const char *trans, const int *n, const int *nrhs, void *a,
                 const int *lda, void *af, const int *ldaf, int *ipiv, char *equed, double *r,
                 double *c, void *b, const int *ldb, void *x, const int *ldx, double *rcond,
                 double *ferr, double *berr, void *work, double *rwork, int *info, size_t l1,
                 size_t l2, size_t l3) {
    (void)l1;
    (void)l2;
    (void)l3;
    zgesvx_(fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c, b, ldb, x, ldx, rcond, ferr,
            berr, work, rwork, info);
}
