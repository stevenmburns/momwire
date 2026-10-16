// The NEC-5 deck reader's binding (`_nec5_deck_core.cpp` is the reader, with
// no Python in it). Test-facing only: the launchers still read decks with
// `deck/_nec5.py`, which stays the reference; these entry points exist so
// `tests/test_nec5_deck_core.py` and `scripts/core_split/deck_compare.py`
// can hold the C++ reader to it field for field.
#include "_accel_common.h"
#include "_nec5_deck_core.h"

#include <string>

void register_nec5_deck(py::module_ &m) {
    m.def(
        "nec5_deck_dump",
        [](py::bytes latin1, const std::string &view) -> std::string {
            std::string text = latin1;
            if (view == "parse") return mw::nec5::dump_parse(text);
            if (view == "stream") return mw::nec5::dump_stream(text);
            throw py::value_error("view must be 'parse' or 'stream', got '" + view + "'");
        },
        py::arg("latin1"), py::arg("view"),
        "The C++ NEC-5 reader's outcome on a deck's latin-1 bytes, one line per "
        "field: 'parse' is parse_nec5's, 'stream' is the EZNEC seam's front "
        "(split_structures, split_runs, parse_nec5_shared, _answer up to the "
        "solve). scripts/core_split/deck_compare.py writes the Python side.");
    m.def(
        "nec5_text_probe",
        [](const std::string &op, py::bytes latin1) -> std::string {
            return mw::nec5::text_probe(op, std::string(latin1));
        },
        py::arg("op"), py::arg("latin1"),
        "One of the reader's latin-1 str rules or number spellings, for the "
        "tests that pin each against Python.");
}
