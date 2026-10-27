// The NEC-5 deck reader's binding (`_nec5_deck_core.cpp` is the reader, with
// no Python in it). Test-facing only: the launchers still read decks with
// `deck/_nec5.py`, which stays the reference; these entry points exist so
// `tests/test_nec5_deck_core.py` and `scripts/core_split/deck_compare.py`
// can hold the C++ reader to it field for field.
#include "_accel_common.h"
#include "_nec4_deck_core.h"
#include "_nec4_build_core.h"
#include "_sinusoidal_fill_core.h"
#include "_sinusoidal_galerkin_fill_core.h"
#include "_sinusoidal_front_core.h"
#include "_nec4_seam_core.h"
#include "_nec5_deck_core.h"
#include "_somm_below_fill_core.h"

#include <memory>
#include <string>
#include <vector>

namespace {
// The process's below/below grid source for a buried sinusoidal fill
// (momwire#1607 B6a), installed for the call when none is.
struct SinBelowSource {
    bool installed = false;
    SinBelowSource() {
        static std::shared_ptr<somm_below::Source> s = somm_below::fill_source();
        if (somm_below::source() == nullptr) {
            somm_below::set_source(s);
            installed = true;
        }
    }
    ~SinBelowSource() {
        if (installed) somm_below::set_source(nullptr);
    }
};
}  // namespace

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
    // The sinusoidal front half (`_sinusoidal_front_core.cpp`, momwire#1607 B2).
    m.def(
        "sinusoidal_front_dump",
        [](py::bytes latin1) -> std::string { return mw::sinusoidal::dump_front(std::string(latin1)); },
        py::arg("latin1"),
        "The C++ sinusoidal front half (constants, junctions, _build_geometry, "
        "_basis_coefs) at every operating point the NEC-4.2 build makes of a "
        "deck (scripts/core_split/sinusoidal_front_compare.py writes the "
        "Python side).");
    // The sinusoidal free-space fill and solve (`_sinusoidal_fill_core.cpp`,
    // momwire#1607 B3). `blas`: numpy's ILP64 cblas_zgemm / cblas_zgemv /
    // cblas_zdotu_sub, SciPy's LP64 zgetrf / zgetrs, then numpy's cblas_dgemv /
    // cblas_ddot (the contact correction's real products), as addresses.
    m.def(
        "sinusoidal_fill_dump",
        [](py::bytes latin1, std::vector<uintptr_t> blas, std::size_t whole) -> std::string {
            if (blas.size() != 7) throw py::value_error("blas: seven addresses");
            mw::sinusoidal::FillLinalg la;
            la.np.zgemm = reinterpret_cast<mw::nec5::cblas_zgemm_fn>(blas[0]);
            la.np.zgemv = reinterpret_cast<mw::nec5::cblas_zgemv_fn>(blas[1]);
            la.np.zdotu_sub = reinterpret_cast<mw::nec5::cblas_zdotu_sub_fn>(blas[2]);
            la.lu.getrf = reinterpret_cast<mw::razor::zgetrf_fn>(blas[3]);
            la.lu.getrs = reinterpret_cast<mw::razor::zgetrs_fn>(blas[4]);
            la.np.dgemv = reinterpret_cast<mw::nec5::cblas_dgemv_fn>(blas[5]);
            la.np.ddot = reinterpret_cast<mw::nec5::cblas_ddot_fn>(blas[6]);
            const std::string text = latin1;
            py::gil_scoped_release nogil;
            SinBelowSource below;
            return mw::sinusoidal::dump_fill(text, la, whole);
        },
        py::arg("latin1"), py::arg("blas"), py::arg("whole") = 64,
        "The C++ sinusoidal free-space fill and port solve (Z, coefficients, Y) "
        "at every operating point the NEC-4.2 build makes of a deck "
        "(scripts/core_split/sinusoidal_fill_compare.py writes the Python side).");
    // The Sinusoidal-Galerkin mixed-potential fill and port solve in free
    // space (`_sinusoidal_galerkin_fill_core.cpp`, momwire#1718 S2); `blas` as
    // `sinusoidal_fill_dump`'s.
    m.def(
        "sinusoidal_galerkin_fill_dump",
        [](py::bytes latin1, std::vector<uintptr_t> blas, std::size_t whole) -> std::string {
            mw::sinusoidal::FillLinalg la;
            if (blas.empty()) {  // the fill alone (G), on any platform
                const std::string text = latin1;
                py::gil_scoped_release nogil;
                return mw::sinusoidal_galerkin::dump_fill(text, la, whole, true);
            }
            if (blas.size() != 7) throw py::value_error("blas: seven addresses, or none for the fill alone");
            la.np.zgemm = reinterpret_cast<mw::nec5::cblas_zgemm_fn>(blas[0]);
            la.np.zgemv = reinterpret_cast<mw::nec5::cblas_zgemv_fn>(blas[1]);
            la.np.zdotu_sub = reinterpret_cast<mw::nec5::cblas_zdotu_sub_fn>(blas[2]);
            la.lu.getrf = reinterpret_cast<mw::razor::zgetrf_fn>(blas[3]);
            la.lu.getrs = reinterpret_cast<mw::razor::zgetrs_fn>(blas[4]);
            la.np.dgemv = reinterpret_cast<mw::nec5::cblas_dgemv_fn>(blas[5]);
            la.np.ddot = reinterpret_cast<mw::nec5::cblas_ddot_fn>(blas[6]);
            const std::string text = latin1;
            py::gil_scoped_release nogil;
            SinBelowSource below;  // a wholly-buried deck's table (momwire#1718 S11a)
            return mw::sinusoidal_galerkin::dump_fill(text, la, whole);
        },
        py::arg("latin1"), py::arg("blas"), py::arg("whole") = 64,
        "The C++ Sinusoidal-Galerkin mixed-potential fill and port solve (G, "
        "coefficients, Y; G alone when `blas` is empty) at every operating point the NEC-4.2 build makes of a "
        "deck (scripts/core_split/sinusoidal_galerkin_fill_compare.py writes the "
        "Python side).");
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
