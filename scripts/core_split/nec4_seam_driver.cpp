// The NEC-4.2 seam front with no Python (momwire#1607 A2): read deck files
// (latin-1 bytes) and print, for each, `dump_seam` -- the comparison format
// `scripts/core_split/nec4_seam_compare.py` writes for the slot's Python
// front. `tests/test_razor_core_standalone.py` builds this from the core TUs
// alone and compares its output with Python's.
//
//     nec4_seam_driver sinusoidal|bspline DECK [DECK ...]
//
// Output per deck: "== <path>", then the seam dump.
#include "_nec4_seam_core.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: nec4_seam_driver sinusoidal|bspline DECK [DECK ...]\n";
        return 2;
    }
    const std::string basis = argv[1];
    for (int k = 2; k < argc; ++k) {
        std::ifstream in(argv[k], std::ios::binary);
        if (!in) {
            std::cerr << "cannot read " << argv[k] << "\n";
            return 2;
        }
        std::stringstream buf;
        buf << in.rdbuf();
        std::cout << "== " << argv[k] << "\n" << mw::nec4::dump_seam(buf.str(), basis);
    }
    return 0;
}
