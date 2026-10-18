/* The macOS executable's BLAS products (momwire#1559): Accelerate, as the
 * launcher's numpy has it. numpy's macOS 14+ arm64 wheels link Accelerate's
 * new interface with 64-bit integers (the `$NEWLAPACK$ILP64` symbols), so
 * the CBLAS calls route_bs2.cpp replays, the products route.cpp makes in
 * place of `a @ b` and `numpy.linalg.solve`'s gesv go to those entry points
 * here, widening blas.h's 32-bit arguments. The LAPACK SciPy calls is
 * blas_accelerate_lp64.c. */
#define ACCELERATE_NEW_LAPACK
#define ACCELERATE_LAPACK_ILP64
#include <Accelerate/Accelerate.h>
#include <stdlib.h>

typedef __LAPACK_int li;

void mwa_cblas_dsyrk(int order, int uplo, int trans, int n, int k, double alpha, const double *a,
                     int lda, double beta, double *c, int ldc) {
    cblas_dsyrk(order, uplo, trans, (li)n, (li)k, alpha, a, (li)lda, beta, c, (li)ldc);
}
void mwa_cblas_dgemm(int order, int ta, int tb, int m, int n, int k, double alpha, const double *a,
                     int lda, const double *b, int ldb, double beta, double *c, int ldc) {
    cblas_dgemm(order, ta, tb, (li)m, (li)n, (li)k, alpha, a, (li)lda, b, (li)ldb, beta, c, (li)ldc);
}
void mwa_cblas_dgemv(int order, int trans, int m, int n, double alpha, const double *a, int lda,
                     const double *x, int incx, double beta, double *y, int incy) {
    cblas_dgemv(order, trans, (li)m, (li)n, alpha, a, (li)lda, x, (li)incx, beta, y, (li)incy);
}
double mwa_cblas_ddot(int n, const double *x, int incx, const double *y, int incy) {
    return cblas_ddot((li)n, x, (li)incx, y, (li)incy);
}
void mwa_cblas_zgemm(int order, int ta, int tb, int m, int n, int k, const void *alpha, const void *a,
                     int lda, const void *b, int ldb, const void *beta, void *c, int ldc) {
    cblas_zgemm(order, ta, tb, (li)m, (li)n, (li)k, alpha, a, (li)lda, b, (li)ldb, beta, c, (li)ldc);
}
void mwa_cblas_zgemv(int order, int trans, int m, int n, const void *alpha, const void *a, int lda,
                     const void *x, int incx, const void *beta, void *y, int incy) {
    cblas_zgemv(order, trans, (li)m, (li)n, alpha, a, (li)lda, x, (li)incx, beta, y, (li)incy);
}
void mwa_cblas_zdotu_sub(int n, const void *x, int incx, const void *y, int incy, void *out) {
    cblas_zdotu_sub((li)n, x, (li)incx, y, (li)incy, out);
}
void mwa_cblas_zdotc_sub(int n, const void *x, int incx, const void *y, int incy, void *out) {
    cblas_zdotc_sub((li)n, x, (li)incx, y, (li)incy, out);
}

static enum CBLAS_TRANSPOSE tr(char t) {
    return (t == 'N' || t == 'n') ? CblasNoTrans : (t == 'T' || t == 't') ? CblasTrans : CblasConjTrans;
}

/* The Fortran products (column-major), as the CBLAS call numpy would make. */
void mwa_zgemm_(const char *ta, const char *tb, const int *m, const int *n, const int *k,
                const void *alpha, const void *a, const int *lda, const void *b, const int *ldb,
                const void *beta, void *c, const int *ldc) {
    cblas_zgemm(CblasColMajor, tr(*ta), tr(*tb), (li)*m, (li)*n, (li)*k, alpha, a, (li)*lda, b,
                (li)*ldb, beta, c, (li)*ldc);
}
void mwa_zgemv_(const char *t, const int *m, const int *n, const void *alpha, const void *a,
                const int *lda, const void *x, const int *incx, const void *beta, void *y,
                const int *incy) {
    cblas_zgemv(CblasColMajor, tr(*t), (li)*m, (li)*n, alpha, a, (li)*lda, x, (li)*incx, beta, y,
                (li)*incy);
}

/* `numpy.linalg.solve`: gesv, 64-bit pivots narrowed back. */
void mwa_zgesv_(const int *n, const int *nrhs, void *a, const int *lda, int *ipiv, void *b,
                const int *ldb, int *info) {
    li nn = *n, nr = *nrhs, la = *lda, lb = *ldb, inf = 0;
    li *piv = (li *)malloc(sizeof(li) * (size_t)(nn > 0 ? nn : 1));
    if (!piv) {
        *info = -1000;
        return;
    }
    zgesv_(&nn, &nr, a, &la, piv, b, &lb, &inf);
    for (li i = 0; i < nn; i++) ipiv[i] = (int)piv[i];
    free(piv);
    *info = (int)inf;
}
