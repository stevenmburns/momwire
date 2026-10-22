// The NEC-4.2 slot's refusal printouts with no Python (momwire#1607 A3):
// read deck files (latin-1 bytes) and print, for each deck the C++ seam or
// build refuses before any solve, the whole printout the slot writes for it
// (`_nec4.render`'s text, which `tests/test_razor_core_standalone.py`
// compares it with); a deck that needs a solve prints "? solve".
//
//     nec4_printout_driver BASIS VERSION VARIANT MARKER DECK [DECK ...]
//
// Output per deck: "== <path>", then the printout or "? solve". An empty
// MARKER is passed as "-".
#include "_nec4_printout_core.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr << "usage: nec4_printout_driver BASIS VERSION VARIANT MARKER DECK [DECK ...]\n";
        return 2;
    }
    mw::nec5::EngineStamp stamp;
    stamp.version = argv[2];
    stamp.variant = argv[3];
    stamp.marker = std::string(argv[4]) == "-" ? std::string() : std::string(argv[4]);
    for (int k = 5; k < argc; ++k) {
        std::ifstream in(argv[k], std::ios::binary);
        if (!in) {
            std::cerr << "cannot read " << argv[k] << "\n";
            return 2;
        }
        std::stringstream buf;
        buf << in.rdbuf();
        std::string out;
        std::cout << "== " << argv[k] << "\n";
        if (mw::nec4::refusal_printout(buf.str(), argv[1], stamp, out)) {
            std::cout << out;
        } else {
            std::cout << "? solve\n";
        }
    }
    return 0;
}
