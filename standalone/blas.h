// The executable's own OpenBLAS: the LP64 Fortran routines the C++ route and
// `sys._mw_host_lapack` (ABI 2) use, under the prefix the build gave them.
//
// MW_BLAS_PREFIX is `mwx_` on Linux (build_openblas.sh renames every global
// symbol of the static archive so it cannot meet the copies numpy and scipy
// load into the same process). A Windows executable has no such interposition
// (a static library's symbols never bind to a DLL's), so its build may leave
// the names bare: -DMW_BLAS_PREFIX= (empty). On macOS (momwire#1559) the
// names are `mwa_`, defined by blas_accelerate_*.c over Accelerate.
#pragma once

#include <complex>
#include <cstddef>

#ifndef MW_BLAS_PREFIX
#define MW_BLAS_PREFIX mwx_
#endif
#define MW_BLAS_CAT_(a, b) a##b
#define MW_BLAS_CAT(a, b) MW_BLAS_CAT_(a, b)
#define MWX(name) MW_BLAS_CAT(MW_BLAS_PREFIX, name)

extern "C" {
void MWX(openblas_set_num_threads)(int);
void MWX(zgetrf_)(int *, int *, std::complex<double> *, int *, int *, int *);
void MWX(zgetrs_)(char *, int *, int *, std::complex<double> *, int *, int *,
                  std::complex<double> *, int *, int *);
void MWX(dgetrf_)(int *, int *, double *, int *, int *, int *);
void MWX(dgetrs_)(char *, int *, int *, double *, int *, int *, double *, int *, int *);
void MWX(zgemm_)(const char *, const char *, const int *, const int *, const int *, const void *,
                 const void *, const int *, const void *, const int *, const void *, void *,
                 const int *);
void MWX(zgesvx_)(const char *, const char *, const int *, const int *, void *, const int *, void *,
                  const int *, int *, char *, double *, double *, void *, const int *, void *,
                  const int *, double *, double *, double *, void *, double *, int *, std::size_t,
                  std::size_t, std::size_t);
void MWX(zgemv_)(const char *, const int *, const int *, const void *, const void *, const int *,
                 const void *, const int *, const void *, void *, const int *);
}
