// nec2c's command line for `momwire-nec2c-sinusoidal-standalone` (momwire#2065):
// `momwire.dropin.nec2c._portal.main`, in C++, for the no-Python build with
// MW_EXE_DIALECT_NEC2. Included once, inside momwire_razor2p.cpp's anonymous
// namespace, after the launcher layer it shares (the thread policy, the deck
// reader and writer, the not-served sentences).
//
//     momwire-nec2c-sinusoidal-standalone              SimNEC: decks on stdin,
//                                                      printouts on stdout, NX framed
//     momwire-nec2c-sinusoidal-standalone < names      4nec2: two file names on stdin
//     momwire-nec2c-sinusoidal-standalone IN OUT       the file form (a CRLF printout)
//     momwire-nec2c-sinusoidal-standalone - -          the stream form, as bare
//     momwire-nec2c-sinusoidal-standalone -version
//
// Every byte is the Python launcher's (`tests/test_standalone_nec2c.py`):
// the banner, each deck's frame (`deck_frame`: the printout, the NX or EN
// echo, the banner again after an NX), EOF closing an open deck as EN does,
// the argument errors of the shared launcher shell, `--legacy-probe`'s
// refusal and `--save-decks`. What the C++ route does not answer is a
// refusal frame naming why ("not served by this build: ..."), and the
// engine flags this build does not carry (`--basis` another basis,
// `--cache-stats`, `--selftest`) are refused at the probe, exit 3.

namespace nec2cli {

const char *const ARG_ERROR_INPUT = "ERROR getting input file from command line";    // _shell
const char *const ARG_ERROR_OUTPUT = "ERROR getting output file from command line";  // _shell
const char *const REMOVED_FLAG_REFUSAL =
    "--legacy-probe was removed in momwire 0.83.0; drop it from the portal dialog command line";
// `deck.NEC2_BASES`, sorted: `_basis_refusal`'s choices
const char *const ROSTER = "arrayblock, bs1, bs2, hmatrix, pulse, sg, sinusoidal";
const char *const BANNER_SUFFIX = "+sin";  // `_BANNER_SUFFIXES["sinusoidal"]`

int g_end_cap = 2;
int g_forced_threads = 0;
std::string g_save_dir;
long long g_decks_saved = 0;

// MOMWIRE_X_NEC2_END_CAP as the launcher reads it (`_features.requested`):
// unset or blank, the launcher's default (2); on, the refusing request (1);
// anything else, off (0).
int end_cap_mode() {
    const std::string v = env("MOMWIRE_X_NEC2_END_CAP");
    if (v.find_first_not_of(" \t\r\n") == std::string::npos) return 2;
    return feature_value_on(v) ? 1 : 0;
}

std::string join(const std::vector<std::string> &lines) {
    std::string s;
    for (std::size_t k = 0; k < lines.size(); k++) {
        if (k) s += '\n';
        s += lines[k];
    }
    return s;
}

// Python's repr of a str with no quote or control character in it; any
// other is spelled the same way with its escapes, as far as this needs.
std::string py_repr(const std::string &s) {
    std::string out = "'";
    for (unsigned char c : s) {
        if (c == '\\' || c == '\'') {
            out += '\\';
            out += static_cast<char>(c);
        } else if (c < 0x20 || c == 0x7f) {
            char b[8];
            std::snprintf(b, sizeof b, c == '\n' ? "\\n" : c == '\t' ? "\\t" : c == '\r' ? "\\r" : "\\x%02x", c);
            out += b;
        } else {
            out += static_cast<char>(c);
        }
    }
    return out + "'";
}

// `_save_deck(body)`: the deck as received, its terminator EN
void save_deck(const std::string &latin1_body) {
    if (g_save_dir.empty()) return;
    g_decks_saved++;
    char stamp[32];
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#if defined(_WIN32)
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    std::strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", &local);
    char name[64];
    std::snprintf(name, sizeof name, "%s-%04lld.nec", stamp, g_decks_saved);
    const std::string path = g_save_dir + "/" + name;
    const std::string nl = !latin1_body.empty() && latin1_body.back() == '\r' ? "\r\n" : "\n";
    std::string text = latin1_body;
    while (!text.empty() && (text.back() == '\r' || text.back() == '\n')) text.pop_back();
    const std::string bytes = mw::nec5::latin1_to_utf8(text + nl + "EN" + nl);
#if !defined(_WIN32)
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);  // "x"
    if (fd < 0) return;
    std::size_t done = 0;
    while (done < bytes.size()) {
        const ssize_t w = write(fd, bytes.data() + done, bytes.size() - done);
        if (w <= 0) break;
        done += static_cast<std::size_t>(w);
    }
    close(fd);
#else
    std::FILE *w = std::fopen(path.c_str(), "wbx");
    if (w == nullptr) return;
    std::fwrite(bytes.data(), 1, bytes.size(), w);
    std::fclose(w);
#endif
}

// `deck_frame(body, terminator)`: the frame's stdout text and stderr text
// (UTF-8, `\n`), each as `_frame_chunks` joins them.
void deck_frame(const std::string &body, const std::string &terminator, std::string &out_text,
                std::string &err_text) {
    save_deck(body);
    std::vector<std::string> out, err;
    std::string why, kind, message;
    bool failed = false, printed = false;
    if (!cxx_variant_ok()) {
        why = "this processor lacks AVX2 and FMA, which this build's kernels need";
    } else {
        int threads = g_forced_threads;
        if (threads <= 0) {
            const long long n = rt_serial::nec2_largest_solve(body);
            threads = n >= 0 ? threads_for(n) : 1;
        }
        g_render_threads = threads;
        if (threads == 1 && env("MW_EXE_COPY") != "threads") {
            set_threads(g_forced_threads > 0 ? 1 : std::min(2, hardware_threads()), false);
            printed = rt_serial::nec2_render(body, g_end_cap, out, err, why, &failed, &kind, &message);
        } else {
            set_threads(threads, true);
            printed = rt_threads::nec2_render(body, g_end_cap, out, err, why, &failed, &kind, &message);
        }
    }
    if (printed) {
        log_route(g_render_threads == 1 && env("MW_EXE_COPY") != "threads" ? "cxx:serial" : "cxx:threads", "-");
    } else {
        std::string logged = why;
        for (char &c : logged)
            if (c == ' ') c = '_';
        log_route("refused:" + logged, "-");
        out.assign(3, std::string());
        err.clear();
        mw::nec2::append_error(out, failed ? internal_error_reason(kind, message)
                                           : (why.compare(0, 23, "this processor lacks AV") == 0
                                                  ? "not served by this build: " + why
                                                  : not_served(why)));
    }
    long long echoed = 0;
    for (const std::string &l : out)
        if (l.compare(0, 15, "  DATA CARD No:") == 0) echoed++;
    out.push_back(mw::nec2::sentinel_card(echoed + 1, terminator));
    if (terminator == "NX") {
        const std::vector<std::string> banner = mw::nec2::banner_lines(BANNER_SUFFIX);
        out.insert(out.end(), banner.begin() + 1, banner.end());
    }
    out_text = join(out) + "\n";
    err_text = err.empty() ? std::string() : join(err) + "\n";
}

// A source of lines, each with its line end (the last may have none).
class Lines {
  public:
    virtual ~Lines() = default;
    virtual bool next(std::string &line) = 0;
};

// sys.stdin on Linux: split at `\n` only (CPython's `newline="\n"` there),
// decoded as UTF-8 -- carried here as the latin-1 text the deck reader
// takes when every character fits, else declined by the frame (`utf8_ok`).
class StdinLines : public Lines {
  public:
    StdinLines() = default;
    bool next(std::string &line) override {
        for (;;) {
            const std::size_t at = buf_.find('\n', scan_);
            if (at != std::string::npos) {
                line = buf_.substr(0, at + 1);
                buf_.erase(0, at + 1);
                scan_ = 0;
                return true;
            }
            scan_ = buf_.size();
            if (eof_) {
                if (buf_.empty()) return false;
                line.swap(buf_);
                buf_.clear();
                scan_ = 0;
                return true;
            }
            char chunk[65536];
            const long got = read_some(chunk, sizeof chunk);
            if (got <= 0)
                eof_ = true;
            else
                buf_.append(chunk, static_cast<std::size_t>(got));
        }
    }
    // the first line, or false at end of input (`_first_line`)
    bool first(std::string &line) { return next(line); }
    // a line read back in front of the rest
    void unread(const std::string &line) {
        buf_.insert(0, line);
        scan_ = 0;
    }

  private:
    std::string buf_;
    std::size_t scan_ = 0;
    bool eof_ = false;
};

// `io.StringIO(text, newline=None)`: universal newlines, every line end `\n`
class TextLines : public Lines {
  public:
    explicit TextLines(const std::string &text) {
        std::string cur;
        for (std::size_t i = 0; i < text.size(); i++) {
            const char c = text[i];
            if (c == '\r' || c == '\n') {
                if (c == '\r' && i + 1 < text.size() && text[i + 1] == '\n') i++;
                lines_.push_back(cur + "\n");
                cur.clear();
            } else {
                cur += c;
            }
        }
        if (!cur.empty()) lines_.push_back(cur);
    }
    bool next(std::string &line) override {
        if (k_ >= lines_.size()) return false;
        line = lines_[k_++];
        return true;
    }

  private:
    std::vector<std::string> lines_;
    std::size_t k_ = 0;
};

// UTF-8 bytes as latin-1 text: false when they are not UTF-8 or carry a
// character past U+00FF (the reader's text is latin-1).
bool utf8_to_latin1(const std::string &in, std::string &out) {
    out.clear();
    for (std::size_t i = 0; i < in.size(); i++) {
        const unsigned char c = static_cast<unsigned char>(in[i]);
        if (c < 0x80) {
            out += static_cast<char>(c);
        } else if ((c == 0xC2 || c == 0xC3) && i + 1 < in.size() &&
                   (static_cast<unsigned char>(in[i + 1]) & 0xC0) == 0x80) {
            out += static_cast<char>(((c & 0x1F) << 6) | (static_cast<unsigned char>(in[i + 1]) & 0x3F));
            i++;
        } else {
            return false;
        }
    }
    return true;
}

// The first whitespace token of `line.strip().upper()`, or "".
std::string head_token(const std::string &latin1_line) {
    const std::vector<std::string> tokens = mw::nec5::latin1_split(latin1_line);
    if (tokens.empty()) return std::string();
    return mw::nec5::latin1_upper(tokens[0]);
}

bool blank(const std::string &latin1) {
    for (char c : latin1)
        if (!mw::nec5::latin1_isspace(static_cast<unsigned char>(c))) return false;
    return true;
}

// `run_session(portal_seam(), ...)`: the banner, then each frame as its
// terminator arrives. `sink` takes each stdout chunk; false stops (a closed
// stdout). `utf8_in`: the lines are UTF-8 (stdin), else latin-1 (a file).
template <class Sink>
int session(Lines &lines, bool utf8_in, Sink sink) {
    if (!sink(join(mw::nec2::banner_lines(BANNER_SUFFIX)) + "\n")) return 0;
    std::vector<std::string> body;
    bool body_ok = true;
    auto emit = [&](const std::string &terminator) {
        std::string text = join(body), out, err;
        if (!body_ok) {
            std::vector<std::string> frame(3, std::string());
            mw::nec2::append_error(frame, "not served by this build: deck text that is not UTF-8 or not latin-1");
            frame.push_back(mw::nec2::sentinel_card(1, terminator));
            if (terminator == "NX") {
                const std::vector<std::string> banner = mw::nec2::banner_lines(BANNER_SUFFIX);
                frame.insert(frame.end(), banner.begin() + 1, banner.end());
            }
            out = join(frame) + "\n";
        } else {
            deck_frame(text, terminator, out, err);
        }
        const bool wrote = sink(out);
        if (!err.empty()) {
            std::fwrite(err.data(), 1, err.size(), stderr);
            std::fflush(stderr);
        }
        return wrote;
    };
    std::string raw;
    while (lines.next(raw)) {
        std::string line;
        if (utf8_in) {
            if (!utf8_to_latin1(raw, line)) {
                body_ok = false;
                line = raw;
            }
        } else {
            line = raw;
        }
        const std::string head = head_token(line);
        if (head != "NX" && head != "EN") {
            if (!line.empty() && line.back() == '\n') line.pop_back();
            body.push_back(line);
            continue;
        }
        if (!emit(head)) return 0;
        if (head == "EN") return 0;
        body.clear();
        body_ok = true;
    }
    bool any = false;
    for (const std::string &l : body) any = any || !blank(l);
    if (any) emit("EN");
    return 0;
}

int resident(StdinLines &in) {
#if defined(_WIN32)
    _setmode(0, _O_BINARY);
    _setmode(1, _O_BINARY);
#else
    std::signal(SIGPIPE, SIG_IGN);
#endif
    return session(in, true, [](const std::string &chunk) { return write_all(chunk); });
}

// `file_printout(text)`: the stdout a deck file's text gives on stdin
std::string file_printout(const std::string &latin1_text) {
    TextLines lines(latin1_text);
    std::string out;
    session(lines, false, [&](const std::string &chunk) {
        out += chunk;
        return true;
    });
    return out;
}

// `file_refusal(text, reason)`
std::string file_refusal(const std::string &reason) {
    std::vector<std::string> out = mw::nec2::banner_lines(BANNER_SUFFIX);
    out.insert(out.end(), 3, std::string());
    mw::nec2::append_error(out, reason);
    out.push_back(mw::nec2::sentinel_card(1, "EN"));
    return join(out) + "\n";
}

// `_shell.run` for the nec2 front end, inside `_shell.main`'s last line of
// defence (`_report_internal_error`)
int run_file(const std::string &deck, const std::string &out, bool en_optional) {
    std::string text, printout;
    if (!read_deck(deck, text)) {
        printout = file_refusal("UNABLE TO READ INPUT FILE " + ml::os_bytes_to_utf8(ml::path_str(deck)));
        log_route("unreadable", deck);
    } else {
        if (en_optional) text = ml::close_deck(text);
        printout = file_printout(text);
    }
    const int err = write_file(out, mw::nec5::printout_bytes(printout));
    if (err) {
        const ml::PyError e = ml::os_error(err, out);
        if (write_file(out, mw::nec5::printout_bytes(file_refusal(internal_error_reason(e.kind, e.message)))) != 0)
            std::printf("%s %s\n", UNABLE_TO_OPEN, out.c_str());
    }
    return 0;
}

// `_shell_forms(args, ...)`: `_shell.main(args, dialect="nec2")`
int shell_forms(const std::vector<std::string> &args, StdinLines &in, bool en_optional) {
    if (args.empty()) {
        std::printf("%s\n", ARG_ERROR_INPUT);
        return 0;
    }
    if (args.size() != 2) {
        std::printf("%s\n", ARG_ERROR_OUTPUT);
        return 0;
    }
    if (args[0] == "-" || args[1] == "-") {
        if (args[0] == "-" && args[1] == "-") return resident(in);
        std::fprintf(stderr, "%s\n", STREAMING_REFUSAL);
        return 2;
    }
    return run_file(args[0], args[1], en_optional);
}

// `momwire_serve_client.is_names_line(line)`
bool is_names_line(const std::string &line) {
    const std::string stripped = mw::nec5::latin1_strip(line);
    if (stripped.empty()) return false;
    std::string spaced = stripped;
    for (char &c : spaced)
        if (c == ',') c = ' ';
    const std::vector<std::string> tokens = mw::nec5::latin1_split(spaced);
    if (tokens.empty()) return false;
    const std::string &head = tokens[0];
    const auto alpha = [](unsigned char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); };
    bool card = head.size() >= 2 && alpha(static_cast<unsigned char>(head[0])) &&
                alpha(static_cast<unsigned char>(head[1])) &&
                (head.size() == 2 || std::strchr("0123456789.+-", head[2]) != nullptr);
    if (!card) return true;
    const std::string path = ml::strip_path_line(line);
#if !defined(_WIN32)
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
#else
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (f) std::fclose(f);
    return f != nullptr;
#endif
}

// A flag's value: `--flag VALUE` or `--flag=VALUE`, every occurrence, the
// last winning (`configure_engine`'s while loops). `seen`: it was there.
bool take_value(std::vector<std::string> &rest, const std::string &flag, std::string &value) {
    bool seen = false;
    for (;;) {
        auto it = std::find(rest.begin(), rest.end(), flag);
        if (it != rest.end()) {
            seen = true;
            value = it + 1 != rest.end() ? *(it + 1) : std::string();
            rest.erase(it, it + 1 != rest.end() ? it + 2 : it + 1);
            continue;
        }
        auto eq = std::find_if(rest.begin(), rest.end(),
                               [&](const std::string &a) { return a.compare(0, flag.size() + 1, flag + "=") == 0; });
        if (eq != rest.end()) {
            seen = true;
            value = eq->substr(flag.size() + 1);
            rest.erase(eq);
            continue;
        }
        return seen;
    }
}

std::string lstrip_dashes_lower(const std::string &a) {
    std::size_t k = 0;
    while (k < a.size() && a[k] == '-') k++;
    std::string s = a.substr(k);
    for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string expanduser(const std::string &p) {
    if (p.empty() || p[0] != '~') return p;
    if (p.size() > 1 && p[1] != '/') return p;  // ~user: left as written
    const std::string home = env("HOME");
    return home.empty() ? p : home + p.substr(1);
}

bool is_dir(const std::string &p) {
#if !defined(_WIN32)
    struct stat st;
    return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
#else
    DWORD a = GetFileAttributesA(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
#endif
}

// `main(argv)`
int nec2c_main(std::vector<std::string> rest) {
    // `configure_engine`
    if (std::find(rest.begin(), rest.end(), "--legacy-probe") != rest.end()) {
        std::printf("%s\n", REMOVED_FLAG_REFUSAL);
        return 3;
    }
    std::string basis;
    if (take_value(rest, "--basis", basis) && basis != BASIS) {
        // the others in the roster, and their deprecated spellings
        // (`_basis_names.BASIS_ALIASES`, momwire#2091)
        static const char *const roster[] = {"arrayblock", "bs1", "bs2", "hmatrix", "pulse", "sg",
                                             "bspline", "bspline-d1", "sinusoidal-galerkin"};
        bool known = false;
        for (const char *r : roster) known = known || basis == r;
        if (known)
            std::printf("--basis %s is not this executable's: it answers the sinusoidal basis only, and the "
                        "P%cthon launcher momwire-nec2c answers the others\n",
                        py_repr(basis).c_str(), 'y');
        else
            std::printf("unknown --basis %s; choices: %s\n", py_repr(basis).c_str(), ROSTER);
        return 3;
    }
    std::string stats;
    if (take_value(rest, "--cache-stats", stats)) {
        if (stats.empty() || stats[0] == '-') {
            std::printf("--cache-stats needs a file path\n");
            return 3;
        }
        std::printf("--cache-stats is not served by this executable, which keeps no cross-deck cache\n");
        return 3;
    }
    std::string save;
    const bool save_flag = take_value(rest, "--save-decks", save);
    if (save_flag && (save.empty() || save[0] == '-')) {
        std::printf("--save-decks needs a directory\n");
        return 3;
    }
    if (!save_flag) save = env("MOMWIRE_NEC2C_SAVE_DECKS");
    if (!save.empty()) {
        save = expanduser(save);
        if (!is_dir(save)) {
            std::printf("--save-decks: %s is not a directory; create it first\n", py_repr(save).c_str());
            return 3;
        }
        g_save_dir = save;
    }
    rest.erase(std::remove(rest.begin(), rest.end(), std::string("--cache")), rest.end());

    for (const std::string &a : rest)
        if (lstrip_dashes_lower(a) == "version") {
            std::printf("%s\n", probe_version().c_str());
            return 0;
        }
    for (const std::string &a : rest)
        if (lstrip_dashes_lower(a) == "selftest") {
            std::printf("--selftest is not served by this executable; the P%cthon launcher runs it\n", 'y');
            return 3;
        }
    g_end_cap = end_cap_mode();
    g_forced_threads = std::atoi(env("MOMWIRE_THREADS").c_str());
    {
        const std::string tag = std::string(VERSION) + "-" + __DATE__ + "-" + __TIME__;
        rt_serial::configure_grid_cache(tag);
        rt_threads::configure_grid_cache(tag);
    }
    std::vector<std::string> positional;
    for (const std::string &a : rest)
        if (a == "-" || a.empty() || a[0] != '-') positional.push_back(a);
    StdinLines in;
    if (!positional.empty()) return shell_forms(positional, in, false);
    // no paths: SimNEC's decks, or 4nec2's two names (`_first_line`)
#if !defined(_WIN32)
    if (isatty(0)) return resident(in);
#endif
    std::string first;
    if (!in.first(first)) return resident(in);
    if (is_names_line(first)) {
        std::vector<std::string> names{ml::strip_path_line(first)};
        std::string second;
        if (in.next(second)) {
            second = ml::strip_path_line(second);
            if (!second.empty()) names.push_back(second);
        }
        return shell_forms(names, in, true);
    }
    in.unread(first);  // `itertools.chain([first], stdin)`
    return resident(in);
}

}  // namespace nec2cli
