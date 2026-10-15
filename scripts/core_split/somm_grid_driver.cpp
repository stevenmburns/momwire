// `_sommerfeld.get_grid` in C++ alone (`_somm_grid_core.h`, Sommerfeld S2):
//
//     somm_grid_driver OUTDIR < CALLS
//
// Each line of CALLS is `eps_re eps_im k2 r1_max omega mu`, hex floats. For
// call k it writes OUTDIR/k.txt (the grid's scalars and each region's lattice,
// hex) and OUTDIR/k.bin (every region's value table, complex128, in order).
// `scripts/core_split/somm_grid_compare.py` holds both to Python's grid.
#include "_somm_grid_core.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: somm_grid_driver OUTDIR < CALLS\n";
        return 2;
    }
    std::string line;
    int k = 0;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string f[6];
        for (auto &x : f) in >> x;
        double v[6];
        for (int i = 0; i < 6; i++) v[i] = std::strtod(f[i].c_str(), nullptr);
        const somm_core::Grid &g =
            somm_core::get_grid(somm_core::cd(v[0], v[1]), v[2], v[3], v[4], v[5]);
        const std::string base = std::string(argv[1]) + "/" + std::to_string(k);
        std::FILE *t = std::fopen((base + ".txt").c_str(), "w");
        std::fprintf(t, "%a %a %a %a %a %a %a %a %a %a %a\n", g.eps_t.real(), g.eps_t.imag(),
                     g.lattice_eps.real(), g.lattice_eps.imag(), g.k2, g.omega, g.mu, g.r1_max,
                     g.r_break, g.r_near, g.r0_fill);
        std::FILE *b = std::fopen((base + ".bin").c_str(), "wb");
        for (const auto &R : g.regions) {
            std::fprintf(t, "%a %a %lld %a %a %lld\n", R.r0, R.dr, (long long)R.n_r, R.th0, R.dth,
                         (long long)R.n_th);
            std::fwrite(R.vals.data(), sizeof(somm_core::cd), R.vals.size(), b);
        }
        std::fclose(t);
        std::fclose(b);
        k++;
    }
    return 0;
}
