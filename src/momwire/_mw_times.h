// Opt-in per-stage timing (MOMWIRE_TIMES=1): where a deck's time goes.
//
// One process-wide tally of EXCLUSIVE wall time per stage. A `Scope` names
// the stage the code inside it is in; nested scopes switch the clock to the
// inner stage and back, so a ground table filled in the middle of the matrix
// fill is charged to `ground` and not to `fill` as well, and the stages add
// up to the wall time they cover.
//
// OFF by default, and then it is one relaxed atomic load per scope: no clock
// is read and nothing is written. It never touches a number the solve
// computes; the only output it changes is what the caller asks for (the
// printout's MATRIX TIMING / RUN TIME lines, and the `times:` line on
// stderr), and only when it is on.
//
// Only the thread that switched it on is timed: a scope entered on another
// thread (an OpenMP worker, a second request in the resident server) is not
// counted, so the stages stay one thread's wall time.
//
// Header-only, in its own namespace (not `mw`), so the executable's two copies
// of the cores (`mw` and `mws`, standalone/CMakeLists.txt) share ONE tally.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

namespace mw_times {

using Clock = std::chrono::steady_clock;

enum Stage : int {
    none = 0,
    seam,        // the deck read and the structure half (mesh, front)
    ground,      // the Sommerfeld ground tables: built, read from disk, blended
    fill,        // the matrix fill
    factor,      // the LU factor and solve
    far_field,   // RP: the radiation pattern
    near_field,  // NE / NH
    run,         // the rest of the run half (port table, currents, power)
    render,      // the printout's text
    n_stages
};

// A copy of the tally at one moment (per-run deltas for the printout).
struct Snap {
    double ms[n_stages] = {};
    uint64_t built = 0, disk = 0, blended = 0;
};

struct State {
    std::atomic<bool> on{false};
    std::thread::id owner;
    Stage cur = none;
    Clock::time_point last;
    double ms[n_stages] = {};
    uint64_t ground_built = 0, ground_disk = 0, ground_blended = 0;
    uint64_t runs = 0;  // runs finished (an XQ block, a stepping FR's frequency)
    // the threads the caller runs the fill and the factor on (0: not known)
    int fill_threads = 0, factor_threads = 0;
    // where the last run's ground tables were counted to
    Snap mark;
    // RUN TIME's zero: the process's start where the caller knows it
    Clock::time_point origin;
};

inline State &state() {
    static State s;
    return s;
}

// MOMWIRE_TIMES set to anything but "" / 0 / off / false / no.
inline bool env_on() {
    const char *v = std::getenv("MOMWIRE_TIMES");
    if (v == nullptr || v[0] == '\0') return false;
    for (const char *off : {"0", "off", "false", "no", "OFF", "False", "FALSE", "No", "NO", "Off"})
        if (std::strcmp(v, off) == 0) return false;
    return true;
}

inline bool active() {
    State &s = state();
    return s.on.load(std::memory_order_relaxed) && s.owner == std::this_thread::get_id();
}

// Charge the time since the last switch to the current stage, and move to `to`.
inline Stage switch_to(Stage to) {
    State &s = state();
    const Clock::time_point now = Clock::now();
    s.ms[s.cur] += std::chrono::duration<double, std::milli>(now - s.last).count();
    s.last = now;
    const Stage prev = s.cur;
    s.cur = to;
    return prev;
}

// Start (or restart) the tally on this thread; `origin` is RUN TIME's zero.
inline void enable(Clock::time_point origin) {
    State &s = state();
    s.on.store(false, std::memory_order_relaxed);
    s.owner = std::this_thread::get_id();
    s.cur = none;
    s.last = Clock::now();
    for (double &m : s.ms) m = 0.0;
    s.ground_built = s.ground_disk = s.ground_blended = 0;
    s.runs = 0;
    s.mark = Snap();
    s.origin = origin;
    s.on.store(true, std::memory_order_relaxed);
}
inline void enable() { enable(Clock::now()); }
inline void disable() { state().on.store(false, std::memory_order_relaxed); }

// The stages so far, the current one brought up to now.
inline void settle() {
    if (active()) switch_to(state().cur);
}

inline Snap snap() {
    settle();
    const State &s = state();
    Snap o;
    for (int k = 0; k < n_stages; k++) o.ms[k] = s.ms[k];
    o.built = s.ground_built;
    o.disk = s.ground_disk;
    o.blended = s.ground_blended;
    return o;
}

inline double ms_of(Stage st) {
    settle();
    return state().ms[st];
}

// Seconds since `origin` (0 when off): the printout's RUN TIME.
inline double run_seconds() {
    if (!active()) return 0.0;
    return std::chrono::duration<double>(Clock::now() - state().origin).count();
}

class Scope {
  public:
    explicit Scope(Stage st) : act_(active()) {
        if (act_) prev_ = switch_to(st);
    }
    ~Scope() {
        if (act_) switch_to(prev_);
    }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

  private:
    bool act_;
    Stage prev_ = none;
};

// How the ground tables were obtained (counts of grids / fills).
inline void count_built() {
    if (active()) state().ground_built++;
}
inline void count_disk() {
    if (active()) state().ground_disk++;
}
inline void count_blended() {
    if (active()) state().ground_blended++;
}
inline void count_run() {
    if (active()) state().runs++;
}

// The `times:` line, one per answer, on stderr. Every key is always there, in
// this order, so a reader can split on spaces and `=`; `*_ms` values are
// milliseconds. `total_ms` is the wall time from `since` (the process's start
// for a process's first answer) and `other_ms` is what no stage claimed, so
// the stages and `other_ms` add up to `total_ms`. `start_ms` (process start to
// main) and `load_ms` (an interpreter / library the answer loaded) and
// `write_ms` (the printout written) are the caller's; the rest is the tally,
// which this restarts for the next answer.
inline std::string take_line(const char *route, int threads, double start_ms, double load_ms,
                             double write_ms, Clock::time_point since) {
    State &s = state();
    settle();
    const double total = std::chrono::duration<double, std::milli>(Clock::now() - since).count();
    const struct {
        const char *key;
        double ms;
    } cells[] = {{"start_ms", start_ms},          {"load_ms", load_ms},
                 {"seam_ms", s.ms[seam]},         {"ground_ms", s.ms[ground]},
                 {"fill_ms", s.ms[fill]},         {"factor_ms", s.ms[factor]},
                 {"rp_ms", s.ms[far_field]},      {"near_ms", s.ms[near_field]},
                 {"run_ms", s.ms[run]},           {"printout_ms", s.ms[render]},
                 {"write_ms", write_ms}};
    double claimed = 0.0;
    char buf[160];
    std::snprintf(buf, sizeof buf, "times: route=%s threads=%d runs=%llu", route, threads,
                  static_cast<unsigned long long>(s.runs));
    std::string line = buf;
    for (const auto &c : cells) {
        claimed += c.ms;
        std::snprintf(buf, sizeof buf, " %s=%.3f", c.key, c.ms);
        line += buf;
    }
    std::snprintf(buf, sizeof buf,
                  " other_ms=%.3f total_ms=%.3f ground_built=%llu ground_disk=%llu ground_blended=%llu",
                  total > claimed ? total - claimed : 0.0, total,
                  static_cast<unsigned long long>(s.ground_built),
                  static_cast<unsigned long long>(s.ground_disk),
                  static_cast<unsigned long long>(s.ground_blended));
    line += buf;
    const Clock::time_point origin = s.origin;
    enable(origin);  // the next answer's tally, RUN TIME's zero kept
    return line;
}

}  // namespace mw_times
