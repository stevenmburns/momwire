// PROTOTYPE (momwire#1517 step 1, not merged): deck file in -> printout
// file out for bspline (bs2), with no Python: `nec5_razor_printout_driver`'s
// protocol (see there) over `_nec5_bspline_printout_core.h`. NUMPY_OPENBLAS
// also supplies `scipy_cblas_dsyrk64_` (the assembler's `tangents @
// tangents.T`). `scripts/core_split/printout_compare.py --basis bspline`
// requires every printed byte to be `_shell.render(text, basis="bspline")`'s.
#include "_nec5_bspline_printout_core.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
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
    if (argc == 2 && std::string(argv[1]) == "--transliterate") {
        const std::string text((std::istreambuf_iterator<char>(std::cin)),
                               std::istreambuf_iterator<char>());
        const std::string out = mw::nec5::printout_bytes(text);
        std::fwrite(out.data(), 1, out.size(), stdout);
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--format") {
        std::string line;
        while (std::getline(std::cin, line)) {
            const unsigned long long bits = std::stoull(line, nullptr, 16);
            double x;
            std::memcpy(&x, &bits, sizeof x);
            std::printf("%s|%s|%s|%s|%s|%s|%s\n", mw::nec5::py_format_e(x, 13, 5).c_str(),
                        mw::nec5::py_format_e(x, 11, 4).c_str(),
                        mw::nec5::py_format_e(x, 12, 4).c_str(),
                        mw::nec5::py_format_e(x, 10, 3).c_str(),
                        mw::nec5::py_format_f(x, 10, 4).c_str(),
                        mw::nec5::py_format_f(x, 7, 2).c_str(),
                        mw::nec5::py_format_f(x, 7, 3).c_str());
        }
        return 0;
    }
    if (argc < 8) {
        std::cerr << "usage: nec5_bspline_printout_driver SCIPY_OPENBLAS NUMPY_OPENBLAS BASIS "
                     "VERSION VARIANT MARKER OUTDIR DECK...\n";
        return 2;
    }
    void *scipy_lib = open_lib(argv[1]);
    void *numpy_lib = open_lib(argv[2]);
    mw::razor::Lapack lp;
    lp.getrf = reinterpret_cast<mw::razor::zgetrf_fn>(sym_of(scipy_lib, {"scipy_zgetrf_", "zgetrf_"}));
    lp.getrs = reinterpret_cast<mw::razor::zgetrs_fn>(sym_of(scipy_lib, {"scipy_zgetrs_", "zgetrs_"}));
    mw::bspline::SymLapack sym;
    sym.sytrf = reinterpret_cast<mw::bspline::zsytrf_fn>(sym_of(scipy_lib, {"scipy_zsytrf_", "zsytrf_"}));
    sym.sytrs = reinterpret_cast<mw::bspline::zsytrs_fn>(sym_of(scipy_lib, {"scipy_zsytrs_", "zsytrs_"}));
    mw::nec5::NumpyLinalg np;
    np.dsyrk = reinterpret_cast<mw::nec5::cblas_dsyrk_fn>(
        sym_of(numpy_lib, {"scipy_cblas_dsyrk64_", "cblas_dsyrk64_"}));
    np.zgemm = reinterpret_cast<mw::nec5::cblas_zgemm_fn>(
        sym_of(numpy_lib, {"scipy_cblas_zgemm64_", "cblas_zgemm64_"}));
    np.zgemv = reinterpret_cast<mw::nec5::cblas_zgemv_fn>(
        sym_of(numpy_lib, {"scipy_cblas_zgemv64_", "cblas_zgemv64_"}));
    np.zdotu_sub = reinterpret_cast<mw::nec5::cblas_zdotu_sub_fn>(
        sym_of(numpy_lib, {"scipy_cblas_zdotu_sub64_", "cblas_zdotu_sub64_"}));
    np.zgesv = reinterpret_cast<mw::nec5::zgesv_fn>(sym_of(numpy_lib, {"scipy_zgesv_64_", "zgesv_64_"}));
    // the network solve's expert driver: SciPy's (`networks.MNASystem.solve`)
    np.zgesvx = reinterpret_cast<mw::nec5::zgesvx_fn>(sym_of(scipy_lib, {"scipy_zgesvx_", "zgesvx_"}));
    const std::string basis = argv[3];
    mw::nec5::EngineStamp stamp;
    stamp.version = argv[4];
    stamp.variant = argv[5];
    stamp.marker = std::string(argv[6]) == "-" ? std::string() : std::string(argv[6]);
    const std::string out = argv[7];
    static const char *seam[] = {"served", "refused", "raised", "declined"};
    const char *want = std::getenv("MW_PRINTOUT_ARRAYS");
    const bool dump = want != nullptr && std::string(want) == "1";
    for (int i = 8; i < argc; i++) {
        std::ifstream f(argv[i], std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        std::vector<mw::nec5::RunArrays> arrays;
        const mw::nec5::PrintoutResult r =
            mw::nec5::bspline_printout(text, basis, stamp, lp, sym, np, dump ? &arrays : nullptr);
        const bool printed = r.route == mw::nec5::PrintoutResult::printed;
        std::printf("== %s\nroute %s\nseam %s\nraised %s\nreason %s\n", argv[i],
                    printed ? "printed" : "declined", seam[r.seam], r.raised_kind.c_str(),
                    hex(r.reason).c_str());
        if (printed) {
            std::ostringstream name;
            name << out << "/" << (i - 8) << ".out";
            std::FILE *w = std::fopen(name.str().c_str(), "wb");
            std::fwrite(r.printout.data(), 1, r.printout.size(), w);
            std::fclose(w);
        }
        if (dump) {
            std::printf("arrays %zu\n", arrays.size());
            for (std::size_t k = 0; k < arrays.size(); k++) {
                const mw::nec5::RunArrays &a = arrays[k];
                const std::pair<const char *, const std::vector<mw::cd> *> named[] = {
                    {"y", &a.y},         {"coeffs", &a.coeffs}, {"v_applied", &a.v_applied},
                    {"v_gap", &a.v_gap}, {"i_port", &a.i_port}, {"i_source", &a.i_source}};
                for (const auto &nv : named) {
                    std::ostringstream name;
                    name << out << "/" << (i - 8) << "_" << k << "_" << nv.first << ".bin";
                    std::FILE *w = std::fopen(name.str().c_str(), "wb");
                    std::fwrite(nv.second->data(), sizeof(mw::cd), nv.second->size(), w);
                    std::fclose(w);
                }
            }
        }
        std::fflush(stdout);
    }
    return 0;
}
