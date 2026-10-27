// The NEC-4.2 slot's printout for the Sinusoidal-Galerkin basis with no
// Python (momwire#1718 S3): the reader, the seam, the build, the sinusoidal
// front half, the Galerkin fill and solve and the run half
// (`_nec4_run_core.h`), over the OpenBLAS builds numpy and SciPy bundle.
// `tests/test_razor_core_standalone.py` builds this from the core TUs alone
// and holds each printout to `_nec4.render(text,
// basis="sinusoidal-galerkin")`'s, the measured-time lines masked
// (`scripts/core_split/nec4_run_compare.py`).
//
//     nec4_sinusoidal_galerkin_printout_driver SCIPY_OPENBLAS NUMPY_OPENBLAS VERSION VARIANT MARKER DECK...
//
// Output per deck: "== <path>", then "printed <bytes>" and the printout, or
// "declined <hex of the reason>".
#include "_sinusoidal_galerkin_run_core.h"
#include "_somm_below_fill_core.h"

#include <dlfcn.h>

#include <cstdio>
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

std::string hex(const std::string &s) {
    static const char *d = "0123456789abcdef";
    std::string out;
    for (unsigned char c : s) {
        out += d[c >> 4];
        out += d[c & 15];
    }
    return out;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 6) {
        std::cerr << "usage: nec4_sinusoidal_galerkin_printout_driver SCIPY_OPENBLAS NUMPY_OPENBLAS VERSION VARIANT "
                     "MARKER DECK...\n";
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
    la.np.zgesv = reinterpret_cast<mw::nec5::zgesv_fn>(sym_of(numpy_lib, {"scipy_zgesv_64_", "zgesv_64_"}));
    la.np.zgesvx = reinterpret_cast<mw::nec5::zgesvx_fn>(sym_of(scipy_lib, {"scipy_zgesvx_", "zgesvx_"}));
    la.np.dgemm = reinterpret_cast<mw::nec5::cblas_dgemm_fn>(
        sym_of(numpy_lib, {"scipy_cblas_dgemm64_", "cblas_dgemm64_"}));
    // the below/below table a wholly-buried deck's fill reads (momwire#1718
    // S11a), filled in process as the executable's route does
    somm_below::set_source(somm_below::fill_source());
    mw::nec5::EngineStamp stamp;
    stamp.version = argv[3];
    stamp.variant = argv[4];
    stamp.marker = std::string(argv[5]) == "-" ? std::string() : std::string(argv[5]);
    for (int k = 6; k < argc; ++k) {
        std::ifstream in(argv[k], std::ios::binary);
        if (!in) {
            std::cerr << "cannot read " << argv[k] << "\n";
            return 2;
        }
        std::stringstream buf;
        buf << in.rdbuf();
        std::string out, why;
        const bool ok = mw::sinusoidal_galerkin::printout(buf.str(), stamp, la, out, why);
        std::cout << "== " << argv[k] << "\n";
        if (ok)
            std::cout << "printed " << out.size() << "\n" << out;
        else
            std::cout << "declined " << hex(why) << "\n";
    }
    return 0;
}
