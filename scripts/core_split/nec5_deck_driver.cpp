// The NEC-5 deck reader with no Python: read deck files (their bytes are
// latin-1, as the seam reads them) and print, for each, the reader's
// `dump_parse` and `dump_stream` -- the comparison format
// `scripts/core_split/deck_compare.py` writes for the Python reader.
// `tests/test_razor_core_standalone.py` builds this from
// `_nec5_deck_core.cpp` alone and compares its output with the Python
// reader's, deck by deck.
//
//     nec5_deck_driver DECK [DECK ...]
//
// Output per deck: "== <path>", the parse dump, "--", the stream dump.
#include "_nec5_deck_core.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

int main(int argc, char **argv) {
    for (int k = 1; k < argc; ++k) {
        std::ifstream in(argv[k], std::ios::binary);
        if (!in) {
            std::cerr << "cannot read " << argv[k] << "\n";
            return 2;
        }
        std::stringstream buf;
        buf << in.rdbuf();
        std::string text = buf.str();
        std::cout << "== " << argv[k] << "\n"
                  << mw::nec5::dump_parse(text) << "--\n"
                  << mw::nec5::dump_stream(text);
    }
    return 0;
}
