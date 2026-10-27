// The Sinusoidal-Galerkin mixed-potential fill and solve with no Python
// (momwire#1718 S2): read deck files (latin-1 bytes) and print, for each,
// `dump_fill` -- the comparison format
// `scripts/core_split/sinusoidal_galerkin_fill_compare.py` writes for
// `SinusoidalGalerkinSolver.compute_port_solution`'s own. The LU comes from
// the OpenBLAS build Python itself uses for it, SciPy's LP64 zgetrf / zgetrs
// (`lu_factor` / `lu_solve`). `tests/test_razor_core_standalone.py` builds
// this from the core TUs alone and compares its output with Python's.
//
//     sinusoidal_galerkin_fill_driver SCIPY_OPENBLAS NUMPY_OPENBLAS DECK [DECK ...]
//
// Output per deck: "== <path>", then the fill dump.
#include "_sinusoidal_galerkin_fill_core.h"
#include "_somm_below_fill_core.h"

#include <dlfcn.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void *sym_of(void *lib, const std::vector<std::string> &names) {
    for (const std::string &n : names) {
        void *p = dlsym(lib, n.c_str());
        if (p != nullptr) return p;
    }
    std::cerr << "no " << names[0] << " in the library\n";
    std::exit(2);
}

void *open_lib(const char *path) {
    void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) {
        std::cerr << dlerror() << "\n";
        std::exit(2);
    }
    return lib;
}

}  // namespace

int main(int argc, char **argv) {
    // the below/below table a wholly-buried deck's fill reads (momwire#1718
    // S11a), filled in process as the executable's route does
    somm_below::set_source(somm_below::fill_source());
    if (argc < 3) {
        std::cerr << "usage: sinusoidal_galerkin_fill_driver SCIPY_OPENBLAS NUMPY_OPENBLAS DECK...\n";
        return 2;
    }
    void *scipy_lib = open_lib(argv[1]);
    void *numpy_lib = open_lib(argv[2]);
    mw::sinusoidal::FillLinalg la;
    la.lu.getrf = reinterpret_cast<mw::razor::zgetrf_fn>(sym_of(scipy_lib, {"scipy_zgetrf_", "zgetrf_"}));
    la.lu.getrs = reinterpret_cast<mw::razor::zgetrs_fn>(sym_of(scipy_lib, {"scipy_zgetrs_", "zgetrs_"}));
    la.np.zgemm = reinterpret_cast<mw::nec5::cblas_zgemm_fn>(
        sym_of(numpy_lib, {"scipy_cblas_zgemm64_", "cblas_zgemm64_"}));
    la.np.zgemv = reinterpret_cast<mw::nec5::cblas_zgemv_fn>(
        sym_of(numpy_lib, {"scipy_cblas_zgemv64_", "cblas_zgemv64_"}));
    la.np.zdotu_sub = reinterpret_cast<mw::nec5::cblas_zdotu_sub_fn>(
        sym_of(numpy_lib, {"scipy_cblas_zdotu_sub64_", "cblas_zdotu_sub64_"}));
    la.np.dgemv = reinterpret_cast<mw::nec5::cblas_dgemv_fn>(
        sym_of(numpy_lib, {"scipy_cblas_dgemv64_", "cblas_dgemv64_"}));
    la.np.ddot = reinterpret_cast<mw::nec5::cblas_ddot_fn>(
        sym_of(numpy_lib, {"scipy_cblas_ddot64_", "cblas_ddot64_"}));
    for (int k = 3; k < argc; ++k) {
        std::ifstream in(argv[k], std::ios::binary);
        if (!in) {
            std::cerr << "cannot read " << argv[k] << "\n";
            return 2;
        }
        std::stringstream buf;
        buf << in.rdbuf();
        std::cout << "== " << argv[k] << "\n" << mw::sinusoidal_galerkin::dump_fill(buf.str(), la);
    }
    return 0;
}
