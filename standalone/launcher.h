// The launcher layer of `momwire-nec5-razor-2p`, in C++ for the no-Python
// build (MW_EXE_NO_PYTHON, momwire#1512 step 1): what `momwire_eznec_client`
// and `momwire.eznec._shell` do around a deck before and after the solve --
// the stdin form's two names, the paths as the printout names them, the deck
// closed by EN, the `.GMO` file, and the OSError text an unwritable file
// gives. Each function names the Python it mirrors; tests/test_standalone_exe.py
// holds the executable that uses them to the launcher's bytes.
#pragma once

#include <string>
#include <vector>

namespace mw {
namespace launcher {

// An exception Python would raise at this point, by its Python type name and
// `str(exc)`; the caller writes it as `_shell`'s INTERNAL ERROR printout.
struct PyError {
    std::string kind;
    std::string message;  // UTF-8
};

// `momwire_serve_client.stdin_paths`: false is None (no stdin, or a
// terminal); otherwise up to two names, each `strip_path_line`'s, stopping at
// end of input or at a line that strips to nothing.
bool stdin_paths(std::vector<std::string> &out);

// `strip_path_line` on one line (bytes as the OS gave them; Python's
// whitespace on the UTF-8 decoding).
std::string strip_path_line(const std::string &line);

// `str(Path(p))`: pathlib's normalised spelling of a POSIX path (repeated
// separators collapsed, `.` components and a trailing separator dropped, a
// leading `//` kept).
std::string path_str(const std::string &p);

// `_shell.close_deck`: the latin-1 text closed by EN when no line is one.
std::string close_deck(const std::string &latin1);

// `momwire_serve_client.gmo_path`: the printout path's extension (as
// `os.path.splitext` finds it) replaced by `.GMO`.
std::string gmo_path(const std::string &printout_path);

// `_gmo.gmo_text`: true with the file's text (`\n` line ends, latin-1) when
// the deck asks for one; false when it does not. Throws PyError where
// Python raises (a coordinate `round` cannot take; a reader IndexError).
bool gmo_text(const std::string &latin1, std::string &out);

// The PyError an `open(path, "w")` failing with `err` raises:
// `FileNotFoundError: [Errno 2] No such file or directory: 'x'` and kin.
PyError os_error(int err, const std::string &path);

// The OS's bytes for a path, as UTF-8 text a printout can carry: valid UTF-8
// as it is, any other byte as U+FFFD (Python's surrogate escape, which the
// printout's writer spells `?` either way).
std::string os_bytes_to_utf8(const std::string &bytes);

}  // namespace launcher
}  // namespace mw
