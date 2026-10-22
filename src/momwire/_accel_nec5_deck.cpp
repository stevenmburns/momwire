// The NEC-5 deck reader's binding (`_nec5_deck_core.cpp` is the reader, with
// no Python in it). Test-facing only: the launchers still read decks with
// `deck/_nec5.py`, which stays the reference; these entry points exist so
// `tests/test_nec5_deck_core.py` and `scripts/core_split/deck_compare.py`
// can hold the C++ reader to it field for field.
#include "_accel_common.h"
#include "_nec4_deck_core.h"
#include "_nec4_build_core.h"
#include "_nec4_seam_core.h"
#include "_nec5_deck_core.h"

#include <string>
#include <vector>

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
    // The nec2 / nec4 reader (`_nec4_deck_core.cpp`, momwire#1607 A1): the
    // same test-facing shape, against `momwire.deck.parse(text, dialect)`.
    m.def(
        "nec4_deck_dump",
        [](py::bytes latin1, const std::string &dialect) -> std::string {
            std::string text = latin1;
            if (dialect == "nec2") return mw::nec4::dump_parse(text, mw::nec4::Dialect::nec2);
            if (dialect == "nec4") return mw::nec4::dump_parse(text, mw::nec4::Dialect::nec4);
            throw py::value_error("dialect must be 'nec2' or 'nec4', got '" + dialect + "'");
        },
        py::arg("latin1"), py::arg("dialect"),
        "The C++ nec2 / nec4 reader's outcome on a deck's latin-1 bytes, one "
        "line per field (scripts/core_split/nec4_deck_compare.py writes the "
        "Python side).");
    // The NEC-4.2 seam front (`_nec4_seam_core.cpp`, momwire#1607 A2).
    m.def(
        "nec4_seam_dump",
        [](py::bytes latin1, const std::string &basis) -> std::string {
            return mw::nec4::dump_seam(std::string(latin1), basis);
        },
        py::arg("latin1"), py::arg("basis"),
        "The C++ NEC-4.2 seam front's outcome on a deck's latin-1 bytes "
        "(scripts/core_split/nec4_seam_compare.py writes the Python side).");
    // The NEC-4.2 build (`_nec4_build_core.cpp`, momwire#1607 A2b).
    m.def(
        "nec4_build_dump",
        [](py::bytes latin1, const std::string &basis) -> std::string {
            return mw::nec4::dump_build(std::string(latin1), basis);
        },
        py::arg("latin1"), py::arg("basis"),
        "The C++ NEC-4.2 build's outcome on a deck's latin-1 bytes "
        "(scripts/core_split/nec4_build_compare.py writes the Python side).");
    m.def(
        "nec4_float_probe",
        [](const std::string &op, std::vector<double> v) -> py::object {
            if (op == "repr" && v.size() == 1) return py::str(mw::nec4::py_repr(v[0]));
            if (op == "dist" && v.size() == 6) return py::float_(mw::nec4::py_dist(&v[0], &v[3]));
            throw py::value_error("op must be 'repr' (one value) or 'dist' (six)");
        },
        py::arg("op"), py::arg("values"),
        "The reader's copies of Python's float repr and math.dist, for the "
        "tests that pin them against Python.");
}
