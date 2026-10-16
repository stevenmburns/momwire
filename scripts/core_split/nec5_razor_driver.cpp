// deck text -> currents with no Python: the NEC-5 deck reader
// (`_nec5_deck_core.h`), the seam's structure half (`_nec5_razor_seam_core.h`),
// razor-2p's front half (`_razor_front_core.h`), the fill and the LU
// (`_razor_solve_core.h`).
//
//     nec5_razor_driver LIBOPENBLAS.so OUTDIR DECK...
//
// LIBOPENBLAS.so supplies zgetrf / zgetrs (tried as `scipy_<name>_`, then
// `<name>_`); SciPy's bundled OpenBLAS makes them the routines the Python
// route reaches. For each deck (read as bytes: latin-1, as the seam reads
// it) it prints
//
//     == PATH
//     route served|refused|raised|declined
//     kind ERROR_KIND          (raised: the Python exception's type name)
//     message HEX              (the sentence / exception text / decline, UTF-8 hex)
//     prepared K N_BASIS N_PORTS INFO     (one per preparation)
//
// and writes each preparation's solved coefficients X (n_basis x n_ports,
// column-major complex128) to OUTDIR/<deck index>_<K>_X.bin.
// tests/test_razor_core_standalone.py builds it from the core TUs alone and
// requires every line and every byte to be the Python seam's.
#include "_nec5_razor_seam_core.h"
#include "_razor_solve_core.h"

#include <dlfcn.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace {

void *sym(void *lib, const char *name) {
    const std::string a = std::string("scipy_") + name + "_", b = std::string(name) + "_";
    void *p = dlsym(lib, a.c_str());
    if (p == nullptr) p = dlsym(lib, b.c_str());
    if (p == nullptr) {
        std::cerr << "no " << name << " in the library\n";
        std::exit(2);
    }
    return p;
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
    if (argc < 3) {
        std::cerr << "usage: nec5_razor_driver LIBOPENBLAS.so OUTDIR DECK...\n";
        return 2;
    }
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) {
        std::cerr << dlerror() << "\n";
        return 2;
    }
    mw::razor::Lapack lp;
    lp.getrf = reinterpret_cast<mw::razor::zgetrf_fn>(sym(lib, "zgetrf"));
    lp.getrs = reinterpret_cast<mw::razor::zgetrs_fn>(sym(lib, "zgetrs"));
    const std::string out = argv[2];
    static const char *route[] = {"served", "refused", "raised", "declined"};
    mw::razor::SolveContext ctx;
    for (int i = 3; i < argc; i++) {
        std::ifstream f(argv[i], std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const mw::nec5::SeamResult r = mw::nec5::razor_seam(text);
        std::printf("== %s\nroute %s\nkind %s\nmessage %s\n", argv[i], route[r.route],
                    r.error_kind.c_str(), hex(r.message).c_str());
        for (size_t k = 0; k < r.prepared.size(); k++) {
            // A buried deck the fill cannot finish here (no below table)
            // is kept unsolved, its route "declined": never solved.
            if (r.prepared[k].unsolved) continue;
            const mw::razor::Front &front = *r.prepared[k].front;
            const int64_t n = front.n_basis, m = front.n_ports;
            std::vector<mw::cd> X(static_cast<size_t>(n * m));
            const int info = mw::razor::solve_ports(front, ctx, lp, X.data(), 0);
            std::printf("prepared %zu %lld %lld %d\n", k, static_cast<long long>(n),
                        static_cast<long long>(m), info);
            std::ostringstream name;
            name << out << "/" << (i - 3) << "_" << k << "_X.bin";
            std::FILE *w = std::fopen(name.str().c_str(), "wb");
            std::fwrite(X.data(), sizeof(mw::cd), X.size(), w);
            std::fclose(w);
        }
    }
    return 0;
}
