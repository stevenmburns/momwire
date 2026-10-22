// The sinusoidal front half with no Python (momwire#1607 B2): read deck files
// (latin-1 bytes) and print, for each, `dump_front` -- the comparison format
// `scripts/core_split/sinusoidal_front_compare.py` writes for
// `SinusoidalSolver`'s own. `tests/test_razor_core_standalone.py` builds this from the core TUs
// alone and compares its output with Python's.
//
//     sinusoidal_front_driver DECK [DECK ...]
//
// Output per deck: "== <path>", then the front dump.
#include "_sinusoidal_front_core.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char **argv) {
    for (int k = 1; k < argc; ++k) {
        std::ifstream in(argv[k], std::ios::binary);
        if (!in) {
            std::cerr << "cannot read " << argv[k] << "\n";
            return 2;
        }
        std::stringstream buf;
        buf << in.rdbuf();
        std::cout << "== " << argv[k] << "\n" << mw::sinusoidal::dump_front(buf.str());
    }
    return 0;
}
