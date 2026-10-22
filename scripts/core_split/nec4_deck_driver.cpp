// The nec2 / nec4 deck reader with no Python (momwire#1607 A1): read deck
// files (latin-1 bytes) and print, for each, the reader's `dump_parse` -- the
// comparison format `scripts/core_split/nec4_deck_compare.py` writes for
// `momwire.deck.parse`. `tests/test_razor_core_standalone.py` builds this
// from the core TUs alone and compares its output with the Python reader's.
//
//     nec4_deck_driver nec2|nec4 DECK [DECK ...]
//
// Output per deck: "== <path>", then the parse dump.
#include "_nec4_deck_core.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: nec4_deck_driver nec2|nec4 DECK [DECK ...]\n";
        return 2;
    }
    const std::string name = argv[1];
    if (name != "nec2" && name != "nec4") {
        std::cerr << "dialect must be nec2 or nec4, got " << name << "\n";
        return 2;
    }
    const mw::nec4::Dialect dialect = name == "nec2" ? mw::nec4::Dialect::nec2 : mw::nec4::Dialect::nec4;
    for (int k = 2; k < argc; ++k) {
        std::ifstream in(argv[k], std::ios::binary);
        if (!in) {
            std::cerr << "cannot read " << argv[k] << "\n";
            return 2;
        }
        std::stringstream buf;
        buf << in.rdbuf();
        std::cout << "== " << argv[k] << "\n" << mw::nec4::dump_parse(buf.str(), dialect);
    }
    return 0;
}
