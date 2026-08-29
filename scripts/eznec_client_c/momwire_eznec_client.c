/*
 * momwire-eznec[-<basis>] -- the EZNEC drop-in engine at native launch speed.
 *
 * momwire#718 phase 3.  EZNEC Pro+ v7 launches its external NEC-5 console
 * engine once per calculation as `engine.exe <deck> <printout>`, and the
 * frozen Python one-shot pays ~1 s of interpreter and NumPy import on every
 * launch against a licensed engine that launches in 18-37 ms.  This program
 * is that same argv contract with the import deleted: forward the deck bytes
 * to the resident daemon over the phase-2 wire protocol (spawning it when
 * nobody is listening) and write the answered bytes to the printout path.
 *
 * The shell's three rules bind this client exactly as they bind the engine:
 *
 *   1. exit 0 always -- the exit code is not a channel EZNEC reads;
 *   2. always write the printout -- the file is the only channel there is;
 *   3. never read stdin -- EZNEC never writes one.
 *
 * THE WIRE PROTOCOL (momwire.eznec._resident, pinned by
 * tests/test_serve_resident.py): connect, send the deck bytes verbatim, shut
 * the write side down, read to EOF.  EOF is the frame; there are no length
 * prefixes and no framing of our own.  The server transcodes (latin-1, CRLF),
 * so what comes back IS the file -- this client never inspects or decodes
 * anything, and an EMPTY answer is a server that died mid-solve, never a
 * printout.
 *
 * THE LADDER, designed in from the start so the worst case of the feature is
 * "as slow as today" and never "the engine is broken":
 *
 *   1. a warm daemon answers -- milliseconds;
 *   2. nobody listening -- spawn `momwire-eznec-engine --serve` under the
 *      lock and poll-connect while it pays the cold import once;
 *   3. the resident path fails ANYWHERE (spawn, timeout, socket death, an
 *      empty answer) -- run `momwire-eznec-engine` as a one-shot child,
 *      which is correct at today's speed;
 *   4. even that fails -- write a printout carrying a named NEC ERROR line,
 *      and still exit 0.
 *
 * C rather than a fifth frozen Python stub because rung 1 is the whole point:
 * a ~9.5 MB PyInstaller stub per variant spends the launch it exists to save.
 * One source, two toolchains -- cc on POSIX, cl.exe on windows-latest -- and
 * the per-OS spellings below are of ONE property each, never of two designs.
 */

#ifndef _WIN32
#define _DEFAULT_SOURCE 1
#define _POSIX_C_SOURCE 200809L
#endif

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS 1
#define WIN32_LEAN_AND_MEAN 1
#include <winsock2.h>
#include <ws2tcpip.h>
#include <afunix.h>
#include <windows.h>
typedef SOCKET sock_t;
typedef int sock_len_t;
typedef HANDLE proc_t;
typedef HANDLE lock_t;
#define BAD_SOCK INVALID_SOCKET
#define NO_PROC ((HANDLE)NULL)
#define NO_LOCK INVALID_HANDLE_VALUE
#define DIR_SEP '\\'
#define EXE_SUFFIX ".exe"
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
typedef int sock_t;
typedef socklen_t sock_len_t;
typedef pid_t proc_t;
typedef int lock_t;
#define BAD_SOCK (-1)
#define NO_PROC ((pid_t)-1)
#define NO_LOCK (-1)
#define DIR_SEP '/'
#define EXE_SUFFIX ""
#endif

#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The distribution version the key is scoped by, injected by the build
 * script so this file has no second source of truth for it.  Stringified
 * rather than formatted with %d because the key must carry the version's own
 * TEXT: `importlib.metadata.version("momwire").split(".")[:2]` is what the
 * Python side hashes, and "09" is not 9. */
#ifndef MOMWIRE_VERSION_MAJOR
#define MOMWIRE_VERSION_MAJOR 0
#endif
#ifndef MOMWIRE_VERSION_MINOR
#define MOMWIRE_VERSION_MINOR 0
#endif
#define MW_STR2(x) #x
#define MW_STR(x) MW_STR2(x)
#define ENGINE_TAG "eznec." MW_STR(MOMWIRE_VERSION_MAJOR) "." MW_STR(MOMWIRE_VERSION_MINOR)

/* momwire#528/#643: the basis rides on this program's own filename, and the
 * `client` segment itself selects nothing -- `momwire_serve_client
 * .filename_basis`'s rule, which a COPY of that client is held to as well. */
#define FILENAME_MARKER "eznec-"
#define CLIENT_SEGMENT "client"

/* The one spawn target a deployed bundle has: there is no system Python on a
 * user's box, so the daemon is the bundle's own frozen exe (#718 D3).  The
 * `engine` segment is consumed by its own entry the way `client` is here, so
 * this plain name selects the default basis and `--basis` is the only thing
 * that moves it. */
#define ENGINE_NAME "momwire-eznec-engine" EXE_SUFFIX

/* One warm server per formulation per install, and the idle policy is part
 * of that identity -- so the number below is not a tunable, it is a hash
 * input whose TEXT must equal `repr(900.0)` on the Python side. */
#define IDLE_TIMEOUT_REPR "900.0"

/* How long a client waits for a server it spawned: the wait IS the cold
 * NumPy import, the thing the whole split exists to pay once, on a box that
 * may already be solving.  Generous, and bounded only so a wedged spawn
 * reaches rung 3 instead of hanging EZNEC forever. */
#define SPAWN_TIMEOUT 120.0
#define POLL_MIN_MS 5
#define POLL_MAX_MS 50

/* Rung 3 is a whole cold solve, not a launch. */
#define ONE_SHOT_TIMEOUT 600.0

#define RECV_CHUNK 65536
#define PATH_CAP 4096
#define REASON_CAP 200

/* EZNEC's own fault paths print these and write nothing (there is no output
 * path to write to) -- captured behavior, the same strings as the shell's. */
#define ARGUMENT_ERROR_INPUT "UNABLE TO OPEN FILE"
#define ARGUMENT_ERROR_OUTPUT "UNABLE TO OPEN SECOND FILE"

#define TRANSPORT_ENV "MOMWIRE_SERVE_TRANSPORT"
#define UNIX_SUFFIX ".sock"
#define TCP_SUFFIX ".port"

/* --------------------------------------------------------------------------
 * the reason
 *
 * Rung 4 is the only place a failure is ever spoken, so every rung records
 * WHY in one place and the last writer wins -- the reason a user needs is
 * the one that stopped the last rung that could still have worked.
 * ------------------------------------------------------------------------*/

static char g_reason[REASON_CAP] = "no reason recorded";

static void note(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_reason, sizeof g_reason, fmt, args);
    va_end(args);
}

static const char *os_error(void)
{
#ifdef _WIN32
    static char text[64];
    snprintf(text, sizeof text, "win32 error %lu", (unsigned long)GetLastError());
    return text;
#else
    return strerror(errno);
#endif
}

/* --------------------------------------------------------------------------
 * SHA-256 (public domain, FIPS 180-4)
 *
 * Vendored rather than linked: this program's reason to exist is that it
 * starts in single-digit milliseconds and depends on nothing, and the hash
 * is used for exactly one thing -- a 16-hex name short enough to fit in
 * sun_path with a directory in front of it.
 * ------------------------------------------------------------------------*/

typedef struct {
    uint32_t state[8];
    uint64_t bits;
    unsigned char block[64];
    size_t fill;
} sha256_t;

static const uint32_t SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
    0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
    0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
    0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

#define ROR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

static void sha256_compress(sha256_t *h, const unsigned char *p)
{
    uint32_t w[64];
    uint32_t a, b, c, d, e, f, g, s;
    int i;

    for (i = 0; i < 16; i++) {
        w[i] = ((uint32_t)p[i * 4] << 24) | ((uint32_t)p[i * 4 + 1] << 16) |
               ((uint32_t)p[i * 4 + 2] << 8) | (uint32_t)p[i * 4 + 3];
    }
    for (i = 16; i < 64; i++) {
        uint32_t s0 = ROR32(w[i - 15], 7) ^ ROR32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = ROR32(w[i - 2], 17) ^ ROR32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    a = h->state[0];
    b = h->state[1];
    c = h->state[2];
    d = h->state[3];
    e = h->state[4];
    f = h->state[5];
    g = h->state[6];
    s = h->state[7];

    for (i = 0; i < 64; i++) {
        uint32_t t1 = s + (ROR32(e, 6) ^ ROR32(e, 11) ^ ROR32(e, 25)) +
                      ((e & f) ^ (~e & g)) + SHA256_K[i] + w[i];
        uint32_t t2 = (ROR32(a, 2) ^ ROR32(a, 13) ^ ROR32(a, 22)) +
                      ((a & b) ^ (a & c) ^ (b & c));
        s = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    h->state[0] += a;
    h->state[1] += b;
    h->state[2] += c;
    h->state[3] += d;
    h->state[4] += e;
    h->state[5] += f;
    h->state[6] += g;
    h->state[7] += s;
}

static void sha256_init(sha256_t *h)
{
    h->state[0] = 0x6a09e667u;
    h->state[1] = 0xbb67ae85u;
    h->state[2] = 0x3c6ef372u;
    h->state[3] = 0xa54ff53au;
    h->state[4] = 0x510e527fu;
    h->state[5] = 0x9b05688cu;
    h->state[6] = 0x1f83d9abu;
    h->state[7] = 0x5be0cd19u;
    h->bits = 0;
    h->fill = 0;
}

static void sha256_update(sha256_t *h, const void *data, size_t len)
{
    const unsigned char *p = (const unsigned char *)data;
    h->bits += (uint64_t)len * 8u;
    while (len > 0) {
        size_t room = 64 - h->fill;
        size_t take = len < room ? len : room;
        memcpy(h->block + h->fill, p, take);
        h->fill += take;
        p += take;
        len -= take;
        if (h->fill == 64) {
            sha256_compress(h, h->block);
            h->fill = 0;
        }
    }
}

static void sha256_final(sha256_t *h, unsigned char out[32])
{
    uint64_t bits = h->bits;
    int i;

    h->block[h->fill++] = 0x80;
    if (h->fill > 56) {
        memset(h->block + h->fill, 0, 64 - h->fill);
        sha256_compress(h, h->block);
        h->fill = 0;
    }
    memset(h->block + h->fill, 0, 56 - h->fill);
    for (i = 0; i < 8; i++) {
        h->block[56 + i] = (unsigned char)(bits >> (56 - 8 * i));
    }
    sha256_compress(h, h->block);
    for (i = 0; i < 8; i++) {
        out[i * 4] = (unsigned char)(h->state[i] >> 24);
        out[i * 4 + 1] = (unsigned char)(h->state[i] >> 16);
        out[i * 4 + 2] = (unsigned char)(h->state[i] >> 8);
        out[i * 4 + 3] = (unsigned char)h->state[i];
    }
}

/* --------------------------------------------------------------------------
 * growable bytes
 * ------------------------------------------------------------------------*/

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} buf_t;

static int buf_room(buf_t *b, size_t extra)
{
    size_t want = b->len + extra;
    char *grown;
    if (want <= b->cap) {
        return 0;
    }
    if (want < 4096) {
        want = 4096;
    }
    if (want < b->cap * 2) {
        want = b->cap * 2;
    }
    grown = (char *)realloc(b->data, want);
    if (grown == NULL) {
        return -1;
    }
    b->data = grown;
    b->cap = want;
    return 0;
}

static int buf_add(buf_t *b, const void *data, size_t len)
{
    if (buf_room(b, len) != 0) {
        return -1;
    }
    memcpy(b->data + b->len, data, len);
    b->len += len;
    return 0;
}

static void buf_free(buf_t *b)
{
    free(b->data);
    b->data = NULL;
    b->len = b->cap = 0;
}

/* --------------------------------------------------------------------------
 * small platform spellings
 * ------------------------------------------------------------------------*/

static double now_seconds(void)
{
#ifdef _WIN32
    return (double)GetTickCount64() / 1000.0;
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0.0;
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
#endif
}

static void sleep_ms(unsigned ms)
{
#ifdef _WIN32
    Sleep(ms);
#else
    struct timespec ts;
    ts.tv_sec = (time_t)(ms / 1000u);
    ts.tv_nsec = (long)(ms % 1000u) * 1000000L;
    nanosleep(&ts, NULL);
#endif
}

static int copy_str(char *out, size_t cap, const char *src)
{
    size_t len = strlen(src);
    if (len + 1 > cap) {
        return -1;
    }
    memcpy(out, src, len + 1);
    return 0;
}

static int is_sep(char c)
{
#ifdef _WIN32
    return c == '/' || c == '\\';
#else
    return c == '/';
#endif
}

static int is_dir(const char *path)
{
#ifdef _WIN32
    DWORD attrs = GetFileAttributesA(path);
    return attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat info;
    return stat(path, &info) == 0 && S_ISDIR(info.st_mode);
#endif
}

static int file_exists(const char *path)
{
#ifdef _WIN32
    return GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES;
#else
    struct stat info;
    return stat(path, &info) == 0;
#endif
}

static void remove_file(const char *path)
{
#ifdef _WIN32
    DeleteFileA(path);
#else
    unlink(path);
#endif
}

/* 0700 where the OS has modes: the runtime directory holds sockets, and a
 * socket in a world-writable directory is a socket anyone can pre-empt. */
static int make_one_dir(const char *path)
{
#ifdef _WIN32
    if (CreateDirectoryA(path, NULL)) {
        return 0;
    }
    return GetLastError() == ERROR_ALREADY_EXISTS ? 0 : -1;
#else
    if (mkdir(path, 0700) == 0) {
        return 0;
    }
    return errno == EEXIST ? 0 : -1;
#endif
}

/* `os.makedirs(..., exist_ok=True)`: the override may name a directory whose
 * parent does not exist yet, and refusing that would be a rung-3 fallback
 * over a typo the user could not see. */
static int make_dirs(const char *path)
{
    char work[PATH_CAP];
    size_t i;

    if (copy_str(work, sizeof work, path) != 0) {
        return -1;
    }
    for (i = 1; work[i] != '\0'; i++) {
        if (!is_sep(work[i])) {
            continue;
        }
        work[i] = '\0';
        if (work[0] != '\0' && !(i == 2 && work[1] == ':')) {
            make_one_dir(work);
        }
        work[i] = path[i];
    }
    if (make_one_dir(work) != 0 && !is_dir(work)) {
        return -1;
    }
    return 0;
}

static int join_path(char *out, size_t cap, const char *dir, const char *leaf)
{
    int written = snprintf(out, cap, "%s%c%s", dir, DIR_SEP, leaf);
    return (written < 0 || (size_t)written >= cap) ? -1 : 0;
}

/* The absolute path a name resolves to.  Two clients of one bundle must
 * agree on it byte for byte -- it is a hash input (#718 D2) -- and an
 * unresolvable name (no engine at all) keeps its spelling rather than
 * failing here, so the key still exists and the ladder is what refuses. */
static void resolve_path(const char *path, char *out, size_t cap)
{
#ifdef _WIN32
    DWORD written = GetFullPathNameA(path, (DWORD)cap, out, NULL);
    if (written == 0 || written >= cap) {
        copy_str(out, cap, path);
    }
#else
    char *resolved = realpath(path, NULL);
    if (resolved == NULL) {
        copy_str(out, cap, path);
        return;
    }
    if (copy_str(out, cap, resolved) != 0) {
        copy_str(out, cap, path);
    }
    free(resolved);
#endif
}

/* --------------------------------------------------------------------------
 * identity: the name, the directory, the runtime directory, the key
 * ------------------------------------------------------------------------*/

/* `momwire_serve_client.filename_basis`, spelt in C for the same reason that
 * function exists at all: a COPY of the client is the thing that drifts, so
 * what it copies is the whole rule.  Everything after the marker names the
 * basis; the client's own `client` segment is swallowed when it is the whole
 * suffix and stripped when it leads one; and the EMPTY suffix of a name
 * ending at the marker travels as "" rather than as the default, because it
 * asked for a basis and named none -- that must reach the engine's refusal,
 * never round to bs2 silently (momwire#628's shape).
 *
 * Returns 1 when the name selected a basis (`out` may then be empty), 0 when
 * it named none.  ASCII lowering is the whole of the casefold here: the
 * names are ours and a Windows rename is the only thing that recases them. */
static int basis_of(const char *argv0, char *out, size_t cap)
{
    char name[PATH_CAP];
    const char *base = argv0;
    const char *marker;
    const char *suffix;
    size_t i, len;

    /* Both separators on every platform: `filename_basis` normalises
     * backslashes before it splits, so a Windows-shaped argv[0] read on a
     * POSIX box names the same file. */
    for (i = 0; argv0[i] != '\0'; i++) {
        if (argv0[i] == '/' || argv0[i] == '\\') {
            base = argv0 + i + 1;
        }
    }
    if (copy_str(name, sizeof name, base) != 0) {
        return 0;
    }
    for (i = 0; name[i] != '\0'; i++) {
        if (name[i] >= 'A' && name[i] <= 'Z') {
            name[i] = (char)(name[i] - 'A' + 'a');
        }
    }
    len = strlen(name);
    if (len >= 4 && strcmp(name + len - 4, ".exe") == 0) {
        name[len - 4] = '\0';
    }

    marker = strstr(name, FILENAME_MARKER);
    if (marker == NULL) {
        out[0] = '\0';
        return 0;
    }
    suffix = marker + strlen(FILENAME_MARKER);
    if (strcmp(suffix, CLIENT_SEGMENT) == 0) {
        out[0] = '\0';
        return 0;
    }
    if (strncmp(suffix, CLIENT_SEGMENT "-", strlen(CLIENT_SEGMENT) + 1) == 0) {
        suffix += strlen(CLIENT_SEGMENT) + 1;
    }
    if (copy_str(out, cap, suffix) != 0) {
        out[0] = '\0';
        return 1;
    }
    return 1;
}

/* The bundle directory: the engine exe sits beside this client, always.
 * `argv[0]` is what names the BASIS (a copy is the whole point), but the OS's
 * own answer is what names the FILE, because a client invoked through PATH or
 * a shell alias has an argv[0] that is not a path at all. */
static void own_directory(const char *argv0, char *out, size_t cap)
{
    char self[PATH_CAP];
    size_t i, cut = 0;

    self[0] = '\0';
#ifdef _WIN32
    {
        DWORD written = GetModuleFileNameA(NULL, self, (DWORD)sizeof self);
        if (written == 0 || written >= sizeof self) {
            self[0] = '\0';
        }
    }
#else
    {
        ssize_t written = readlink("/proc/self/exe", self, sizeof self - 1);
        if (written > 0) {
            self[written] = '\0';
        } else {
            self[0] = '\0';
        }
    }
#endif
    if (self[0] == '\0' && copy_str(self, sizeof self, argv0) != 0) {
        copy_str(out, cap, ".");
        return;
    }
    for (i = 0; self[i] != '\0'; i++) {
        if (is_sep(self[i])) {
            cut = i;
        }
    }
    if (cut == 0) {
        copy_str(out, cap, ".");
        return;
    }
    self[cut] = '\0';
    if (copy_str(out, cap, self) != 0) {
        copy_str(out, cap, ".");
    }
}

static const char *temp_dir(void)
{
#ifdef _WIN32
    static char path[PATH_CAP];
    DWORD written = GetTempPathA((DWORD)sizeof path, path);
    if (written == 0 || written >= sizeof path) {
        return ".";
    }
    /* GetTempPathA always ends in a separator; the join adds its own. */
    while (written > 1 && is_sep(path[written - 1])) {
        path[--written] = '\0';
    }
    return path;
#else
    const char *names[3] = {"TMPDIR", "TEMP", "TMP"};
    int i;
    for (i = 0; i < 3; i++) {
        const char *value = getenv(names[i]);
        if (value != NULL && value[0] != '\0') {
            return value;
        }
    }
    return "/tmp";
#endif
}

/* `momwire_serve_client.runtime_dir`, name for name.  Only the NAME has to
 * match: the ownership audit and the 0700 repair stay on the daemon's side,
 * which is the process that binds.
 *
 * $XDG_RUNTIME_DIR first because it is the one directory the OS promises is
 * private to this user and cleaned at logout; %LOCALAPPDATA% is that OS's
 * spelling of the same promise; and the tmp fallbacks carry the uid (or the
 * username) for the reason the directory exists at all. */
static int runtime_dir(char *out, size_t cap)
{
    const char *override_dir = getenv("MOMWIRE_PORTAL_RUNTIME_DIR");
    char leaf[128];

    if (override_dir != NULL && override_dir[0] != '\0') {
        if (copy_str(out, cap, override_dir) != 0) {
            note("runtime directory name is too long");
            return -1;
        }
    } else {
#ifdef _WIN32
        const char *local = getenv("LOCALAPPDATA");
        if (local != NULL && local[0] != '\0' && is_dir(local)) {
            if (join_path(out, cap, local, "momwire-portal") != 0) {
                note("runtime directory name is too long");
                return -1;
            }
        } else {
            const char *user = getenv("USERNAME");
            char named[256];
            if (user == NULL || user[0] == '\0') {
                DWORD size = (DWORD)sizeof named;
                if (!GetUserNameA(named, &size)) {
                    copy_str(named, sizeof named, "unknown");
                }
                user = named;
            }
            snprintf(leaf, sizeof leaf, "momwire-portal-%s", user);
            if (join_path(out, cap, temp_dir(), leaf) != 0) {
                note("runtime directory name is too long");
                return -1;
            }
        }
#else
        const char *xdg = getenv("XDG_RUNTIME_DIR");
        if (xdg != NULL && xdg[0] != '\0' && is_dir(xdg)) {
            if (join_path(out, cap, xdg, "momwire-portal") != 0) {
                note("runtime directory name is too long");
                return -1;
            }
        } else {
            snprintf(leaf, sizeof leaf, "momwire-portal-%ld", (long)getuid());
            if (join_path(out, cap, temp_dir(), leaf) != 0) {
                note("runtime directory name is too long");
                return -1;
            }
        }
#endif
    }
    if (make_dirs(out) != 0) {
        note("cannot create %s: %s", out, os_error());
        return -1;
    }
    return 0;
}

/* #718 D2: the same shape as `momwire_eznec_client.config_key` -- momwire
 * version, engine identity, idle policy, basis -- keyed on the DAEMON EXE's
 * resolved path rather than on an interpreter this program does not have.
 * Parity with the Python dev client is explicitly not wanted: they run
 * different engines and must never share a warm server.  Two C clients of
 * ONE bundle do share, and that is what this key is for. */
static void config_key(const char *engine_resolved, const char *basis, char out[17])
{
    static const char nul = '\0';
    sha256_t hash;
    unsigned char digest[32];
    int i;

    sha256_init(&hash);
    sha256_update(&hash, ENGINE_TAG, strlen(ENGINE_TAG));
    sha256_update(&hash, &nul, 1);
    sha256_update(&hash, engine_resolved, strlen(engine_resolved));
    sha256_update(&hash, &nul, 1);
    sha256_update(&hash, IDLE_TIMEOUT_REPR, strlen(IDLE_TIMEOUT_REPR));
    sha256_update(&hash, &nul, 1);
    sha256_update(&hash, basis, strlen(basis));
    sha256_final(&hash, digest);

    for (i = 0; i < 8; i++) {
        snprintf(out + i * 2, 3, "%02x", digest[i]);
    }
    out[16] = '\0';
}

/* --------------------------------------------------------------------------
 * the transports
 *
 * Two names for one question, "where is the warm server".  The client tries
 * both on CONNECT because a bundle's daemon may have been started by either
 * side; it PINS one on SPAWN, because the daemon binds and publishes at
 * exactly the `--socket` name it is given, interpreting it under ITS OWN
 * transport -- a `.sock` name handed to a TCP-transport daemon becomes a
 * rendezvous file with the wrong suffix, which no client would ever find.
 * ------------------------------------------------------------------------*/

static void close_sock(sock_t sock)
{
#ifdef _WIN32
    closesocket(sock);
#else
    close(sock);
#endif
}

/* Nothing listening is ONE answer for every unusable shape, because the
 * caller does one thing about all of them. */
static sock_t connect_unix(const char *path)
{
    struct sockaddr_un address;
    sock_t sock;
    size_t len = strlen(path);

    if (len + 1 > sizeof address.sun_path) {
        return BAD_SOCK;
    }
    sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock == BAD_SOCK) {
        /* A build without AF_UNIX at RUNTIME (a Windows older than 10 1803)
         * is not an error here -- it is the .port probe's turn. */
        return BAD_SOCK;
    }
    memset(&address, 0, sizeof address);
    address.sun_family = AF_UNIX;
    memcpy(address.sun_path, path, len + 1);
    if (connect(sock, (struct sockaddr *)&address, (sock_len_t)sizeof address) != 0) {
        close_sock(sock);
        return BAD_SOCK;
    }
    return sock;
}

/* The rendezvous file: `127.0.0.1:<port>\n`, read whole and small.  No file,
 * an empty one, a truncated one, bytes that are not a port -- all "nobody
 * listening", never an error (`read_rendezvous`'s rule). */
static int read_rendezvous(const char *path)
{
    char text[65];
    FILE *handle;
    size_t got;
    char *colon;
    char *cursor;
    long port;

    handle = fopen(path, "rb");
    if (handle == NULL) {
        return -1;
    }
    got = fread(text, 1, sizeof text - 1, handle);
    fclose(handle);
    text[got] = '\0';

    while (got > 0 && (unsigned char)text[got - 1] <= ' ') {
        text[--got] = '\0';
    }
    colon = strrchr(text, ':');
    if (colon == NULL || colon == text || colon[1] == '\0') {
        return -1;
    }
    for (cursor = colon + 1; *cursor != '\0'; cursor++) {
        if (*cursor < '0' || *cursor > '9') {
            return -1;
        }
    }
    port = strtol(colon + 1, NULL, 10);
    if (port <= 0 || port > 65535) {
        return -1;
    }
    return (int)port;
}

static sock_t connect_tcp(const char *rendezvous_path)
{
    struct sockaddr_in address;
    sock_t sock;
    int port = read_rendezvous(rendezvous_path);

    if (port < 0) {
        return BAD_SOCK;
    }
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == BAD_SOCK) {
        return BAD_SOCK;
    }
    memset(&address, 0, sizeof address);
    address.sin_family = AF_INET;
    address.sin_port = htons((unsigned short)port);
    /* Loopback, and only loopback: the rendezvous is a substitute for a
     * filesystem path, not a network service, and a published host that was
     * not 127.0.0.1 would be this client reaching a service that is not
     * ours (`publish_rendezvous` writes no other). */
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(sock, (struct sockaddr *)&address, (sock_len_t)sizeof address) != 0) {
        close_sock(sock);
        return BAD_SOCK;
    }
    return sock;
}

static sock_t connect_any(const char *unix_path, const char *tcp_path)
{
    sock_t sock = connect_unix(unix_path);
    if (sock != BAD_SOCK) {
        return sock;
    }
    return connect_tcp(tcp_path);
}

/* --------------------------------------------------------------------------
 * the lock
 *
 * Two spellings of ONE property -- the OS releases it when the holder dies --
 * which is why the obtain dance locks rather than creating an O_EXCL
 * lockfile: a client killed mid-spawn costs one retry, not a wedged directory
 * every future client has to age out with a heuristic.
 * ------------------------------------------------------------------------*/

static lock_t take_lock(const char *path)
{
#ifdef _WIN32
    HANDLE handle = CreateFileA(path, GENERIC_READ | GENERIC_WRITE,
                                FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                                OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    OVERLAPPED region;
    if (handle == INVALID_HANDLE_VALUE) {
        return NO_LOCK;
    }
    memset(&region, 0, sizeof region);
    /* Blocking and exclusive: the losers of a 16-client race must wait for
     * the winner's server rather than start a second one. */
    if (!LockFileEx(handle, LOCKFILE_EXCLUSIVE_LOCK, 0, 1, 0, &region)) {
        CloseHandle(handle);
        return NO_LOCK;
    }
    return handle;
#else
    /* O_CLOEXEC because a flock is held by the open FILE DESCRIPTION and
     * survives fork: a daemon that inherited this descriptor would hold the
     * lock for its whole 900 s life and every later client would block on
     * it.  (`spawn_server`'s close_fds=True is the Python side of this.) */
    int fd = open(path, O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) {
        return NO_LOCK;
    }
    while (flock(fd, LOCK_EX) != 0) {
        if (errno != EINTR) {
            close(fd);
            return NO_LOCK;
        }
    }
    return fd;
#endif
}

static void release_lock(lock_t lock)
{
#ifdef _WIN32
    OVERLAPPED region;
    memset(&region, 0, sizeof region);
    UnlockFileEx(lock, 0, 1, 0, &region);
    CloseHandle(lock);
#else
    flock(lock, LOCK_UN);
    close(lock);
#endif
}

/* --------------------------------------------------------------------------
 * launching children
 *
 * `log_path` non-NULL routes stdout and stderr to that file, appending:
 * a server writing to an inherited pipe would corrupt some other client's
 * transcript the day a deck warns.  `detached` is the daemon's promise that
 * it outlives the client that started it and never sees that console's
 * Ctrl-C -- start_new_session on POSIX, DETACHED_PROCESS |
 * CREATE_NEW_PROCESS_GROUP on the OS that has no sessions.
 * ------------------------------------------------------------------------*/

#ifdef _WIN32
/* CreateProcess takes a COMMAND LINE, not an argv, and the child's CRT
 * splits it back apart under the MSVC rules.  Every argument is quoted
 * unconditionally: bundle paths carry spaces, and an empty argument (the
 * `--basis ""` of a name that asked for a basis and named none) only
 * survives as a literal pair of quotes. */
static int quote_argument(char *out, size_t cap, size_t *len, const char *arg)
{
    size_t i = 0;
    size_t backslashes;
    size_t n;

#define PUT(ch)                                                                \
    do {                                                                       \
        if (*len + 1 >= cap) {                                                 \
            return -1;                                                         \
        }                                                                      \
        out[(*len)++] = (ch);                                                  \
    } while (0)

    PUT('"');
    for (;;) {
        backslashes = 0;
        while (arg[i] == '\\') {
            backslashes++;
            i++;
        }
        if (arg[i] == '\0') {
            /* Trailing backslashes would escape the closing quote. */
            for (n = 0; n < backslashes * 2; n++) {
                PUT('\\');
            }
            break;
        }
        if (arg[i] == '"') {
            for (n = 0; n < backslashes * 2 + 1; n++) {
                PUT('\\');
            }
            PUT('"');
        } else {
            for (n = 0; n < backslashes; n++) {
                PUT('\\');
            }
            PUT(arg[i]);
        }
        i++;
    }
    PUT('"');
    out[*len] = '\0';
    return 0;
#undef PUT
}

static int compose_command_line(char *out, size_t cap, char *const args[])
{
    size_t len = 0;
    int i;

    for (i = 0; args[i] != NULL; i++) {
        if (i > 0) {
            if (len + 1 >= cap) {
                return -1;
            }
            out[len++] = ' ';
        }
        if (quote_argument(out, cap, &len, args[i]) != 0) {
            return -1;
        }
    }
    out[len] = '\0';
    return 0;
}
#endif

static proc_t launch(const char *exe, char *const args[], const char *log_path,
                     int detached)
{
#ifdef _WIN32
    char command[PATH_CAP * 4];
    SECURITY_ATTRIBUTES inheritable;
    STARTUPINFOA startup;
    PROCESS_INFORMATION info;
    HANDLE null_in = INVALID_HANDLE_VALUE;
    HANDLE log = INVALID_HANDLE_VALUE;
    DWORD flags = CREATE_NO_WINDOW;

    if (compose_command_line(command, sizeof command, args) != 0) {
        note("engine command line is too long");
        return NO_PROC;
    }

    memset(&inheritable, 0, sizeof inheritable);
    inheritable.nLength = sizeof inheritable;
    inheritable.bInheritHandle = TRUE;

    /* Read AND write: this handle is the child's stdin always, and stands in
     * for a missing console on the output side below. */
    null_in = CreateFileA("NUL", GENERIC_READ | GENERIC_WRITE,
                          FILE_SHARE_READ | FILE_SHARE_WRITE, &inheritable,
                          OPEN_EXISTING, 0, NULL);
    if (null_in == INVALID_HANDLE_VALUE) {
        note("cannot open NUL: %s", os_error());
        return NO_PROC;
    }
    if (log_path != NULL) {
        log = CreateFileA(log_path, FILE_APPEND_DATA,
                          FILE_SHARE_READ | FILE_SHARE_WRITE, &inheritable,
                          OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        if (log == INVALID_HANDLE_VALUE) {
            CloseHandle(null_in);
            note("cannot open %s: %s", log_path, os_error());
            return NO_PROC;
        }
    }

    memset(&startup, 0, sizeof startup);
    startup.cb = sizeof startup;
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = null_in;
    if (log != INVALID_HANDLE_VALUE) {
        startup.hStdOutput = log;
        startup.hStdError = log;
    } else {
        /* No log means rung 3, whose transcript belongs to whoever launched
         * this client.  A parent with no console has NULL there, and
         * STARTF_USESTDHANDLES refuses NULL, so NUL stands in. */
        startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
        startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
        if (startup.hStdOutput == NULL || startup.hStdOutput == INVALID_HANDLE_VALUE) {
            startup.hStdOutput = null_in;
        }
        if (startup.hStdError == NULL || startup.hStdError == INVALID_HANDLE_VALUE) {
            startup.hStdError = null_in;
        }
    }
    memset(&info, 0, sizeof info);
    if (detached) {
        /* The daemon must not inherit ONE handle beyond the three std slots.
         * `bInheritHandles = TRUE` alone duplicates EVERY inheritable handle
         * in this process into the child -- including whatever pipes stand
         * behind this launcher's own stdout/stderr -- and a 900 s daemon
         * holding a duplicate of a pipe's write end means whoever launched
         * THIS process reads EOF only when the daemon dies.  The first
         * Windows canary hung its harness exactly there (the POSIX branch is
         * immune: dup2 REPLACES fds 1 and 2, it does not add copies).
         * PROC_THREAD_ATTRIBUTE_HANDLE_LIST is CreateProcess's spelling of
         * `close_fds=True` -- inheritance restricted to the listed handles.
         * The one-shot child stays on the plain path below: it inherits this
         * launcher's std handles BY DESIGN, and it exits, so its extra
         * duplicates die with it. */
        HANDLE keep[3];
        STARTUPINFOEXA extended;
        SIZE_T attr_size = 0;
        LPPROC_THREAD_ATTRIBUTE_LIST attrs;
        int kept = 0;
        int started;

        keep[kept++] = startup.hStdInput;
        keep[kept++] = startup.hStdOutput;
        keep[kept++] = startup.hStdError;

        InitializeProcThreadAttributeList(NULL, 1, 0, &attr_size);
        attrs = (LPPROC_THREAD_ATTRIBUTE_LIST)malloc(attr_size);
        if (attrs == NULL ||
            !InitializeProcThreadAttributeList(attrs, 1, 0, &attr_size) ||
            !UpdateProcThreadAttribute(attrs, 0,
                                       PROC_THREAD_ATTRIBUTE_HANDLE_LIST, keep,
                                       (SIZE_T)kept * sizeof(HANDLE), NULL,
                                       NULL)) {
            note("cannot restrict the daemon's handles: %s", os_error());
            free(attrs);
            CloseHandle(null_in);
            if (log != INVALID_HANDLE_VALUE) {
                CloseHandle(log);
            }
            return NO_PROC;
        }
        memset(&extended, 0, sizeof extended);
        extended.StartupInfo = startup;
        extended.StartupInfo.cb = sizeof extended;
        extended.lpAttributeList = attrs;
        flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP |
                EXTENDED_STARTUPINFO_PRESENT;

        started = CreateProcessA(exe, command, NULL, NULL, TRUE, flags, NULL,
                                 NULL, &extended.StartupInfo, &info);
        if (!started) {
            /* Before the attribute-list teardown, which may clobber it. */
            note("cannot start %s: %s", exe, os_error());
        }
        DeleteProcThreadAttributeList(attrs);
        free(attrs);
        if (!started) {
            CloseHandle(null_in);
            if (log != INVALID_HANDLE_VALUE) {
                CloseHandle(log);
            }
            return NO_PROC;
        }
    } else if (!CreateProcessA(exe, command, NULL, NULL, TRUE, flags, NULL,
                               NULL, &startup, &info)) {
        note("cannot start %s: %s", exe, os_error());
        CloseHandle(null_in);
        if (log != INVALID_HANDLE_VALUE) {
            CloseHandle(log);
        }
        return NO_PROC;
    }
    CloseHandle(info.hThread);
    CloseHandle(null_in);
    if (log != INVALID_HANDLE_VALUE) {
        CloseHandle(log);
    }
    return info.hProcess;
#else
    pid_t child = fork();
    int null_in;
    int log;

    if (child < 0) {
        note("cannot fork: %s", strerror(errno));
        return NO_PROC;
    }
    if (child > 0) {
        return child;
    }

    if (detached) {
        setsid();
    }
    null_in = open("/dev/null", O_RDONLY);
    if (null_in >= 0) {
        dup2(null_in, 0);
        if (null_in > 2) {
            close(null_in);
        }
    }
    if (log_path != NULL) {
        log = open(log_path, O_WRONLY | O_CREAT | O_APPEND, 0600);
        if (log >= 0) {
            dup2(log, 1);
            dup2(log, 2);
            if (log > 2) {
                close(log);
            }
        }
    }
    execv(exe, args);
    /* Only reachable when exec failed; 127 is the shell's own spelling of
     * "command not found", and the parent reads it as rung 3 failing. */
    _exit(127);
#endif
}

/* Still importing NumPy, or dead at start-up?  Without this a
 * misconfiguration is a client that waits out the whole spawn timeout in
 * silence. */
static int process_alive(proc_t child)
{
#ifdef _WIN32
    DWORD status = 0;
    if (!GetExitCodeProcess(child, &status)) {
        return 0;
    }
    return status == STILL_ACTIVE;
#else
    int status = 0;
    pid_t seen = waitpid(child, &status, WNOHANG);
    return seen == 0;
#endif
}

/* The child's exit status, or -1 if it outlived `timeout`. */
static int wait_for(proc_t child, double timeout)
{
#ifdef _WIN32
    DWORD status = 0;
    if (WaitForSingleObject(child, (DWORD)(timeout * 1000.0)) != WAIT_OBJECT_0) {
        return -1;
    }
    if (!GetExitCodeProcess(child, &status)) {
        return -1;
    }
    return (int)status;
#else
    double deadline = now_seconds() + timeout;
    for (;;) {
        int status = 0;
        pid_t seen = waitpid(child, &status, WNOHANG);
        if (seen == child) {
            if (WIFEXITED(status)) {
                return WEXITSTATUS(status);
            }
            return -1;
        }
        if (seen < 0 && errno != EINTR) {
            return -1;
        }
        if (now_seconds() >= deadline) {
            return -1;
        }
        sleep_ms(POLL_MIN_MS);
    }
#endif
}

static void forget_process(proc_t child)
{
#ifdef _WIN32
    CloseHandle(child);
#else
    (void)child;
#endif
}

/* --------------------------------------------------------------------------
 * files and the wire
 * ------------------------------------------------------------------------*/

static int read_whole_file(const char *path, buf_t *out)
{
    FILE *handle = fopen(path, "rb");
    char chunk[RECV_CHUNK];

    if (handle == NULL) {
        note("cannot read %s: %s", path, strerror(errno));
        return -1;
    }
    for (;;) {
        size_t got = fread(chunk, 1, sizeof chunk, handle);
        if (got > 0 && buf_add(out, chunk, got) != 0) {
            fclose(handle);
            note("out of memory reading %s", path);
            return -1;
        }
        if (got < sizeof chunk) {
            break;
        }
    }
    if (ferror(handle)) {
        fclose(handle);
        note("cannot read %s", path);
        return -1;
    }
    fclose(handle);
    return 0;
}

/* Binary mode, verbatim: the bytes on the wire ARE the file, CRLF and
 * latin-1 included, and a text-mode write would translate them twice. */
static int write_whole_file(const char *path, const char *data, size_t len)
{
    FILE *handle = fopen(path, "wb");
    if (handle == NULL) {
        note("cannot write %s: %s", path, strerror(errno));
        return -1;
    }
    if (len > 0 && fwrite(data, 1, len, handle) != len) {
        fclose(handle);
        note("cannot write %s", path);
        return -1;
    }
    if (fclose(handle) != 0) {
        note("cannot write %s", path);
        return -1;
    }
    return 0;
}

static int send_all(sock_t sock, const char *data, size_t len)
{
    size_t sent = 0;
    while (sent < len) {
        int chunk = (int)(len - sent > RECV_CHUNK ? RECV_CHUNK : len - sent);
#ifdef _WIN32
        int wrote = send(sock, data + sent, chunk, 0);
        if (wrote <= 0) {
            return -1;
        }
#else
        ssize_t wrote = send(sock, data + sent, (size_t)chunk, 0);
        if (wrote < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        if (wrote == 0) {
            return -1;
        }
#endif
        sent += (size_t)wrote;
    }
    return 0;
}

static int recv_to_eof(sock_t sock, buf_t *out)
{
    char chunk[RECV_CHUNK];
    for (;;) {
#ifdef _WIN32
        int got = recv(sock, chunk, (int)sizeof chunk, 0);
#else
        ssize_t got = recv(sock, chunk, sizeof chunk, 0);
        if (got < 0 && errno == EINTR) {
            continue;
        }
#endif
        if (got < 0) {
            return -1;
        }
        if (got == 0) {
            return 0;
        }
        if (buf_add(out, chunk, (size_t)got) != 0) {
            return -1;
        }
    }
}

/* --------------------------------------------------------------------------
 * the ladder
 * ------------------------------------------------------------------------*/

typedef struct {
    char engine[PATH_CAP];      /* the bundle's frozen exe, as launched */
    char engine_key[PATH_CAP];  /* its resolved path -- a hash input */
    char basis[256];            /* the suffix this name selected */
    int named_basis;            /* whether the name selected one at all */
    char unix_path[PATH_CAP];   /* <runtime>/<key>.sock */
    char tcp_path[PATH_CAP];    /* <runtime>/<key>.port */
    char address[PATH_CAP];     /* the one this client would SPAWN on */
    char log_path[PATH_CAP];    /* <address>.log */
    char lock_path[PATH_CAP];   /* <runtime>/<key>.lock */
    const char *pin;            /* "unix" or "tcp" -- the spawn's transport */
    int pin_is_ours;            /* the child needs it set only if we chose it */
} setup_t;

/* The pin: the environment's word when it has one -- a box running the whole
 * TCP path on purpose must have its daemon follow -- and otherwise the
 * platform's own answer.  On Windows that is TCP unconditionally, because
 * the frozen CPython there has no socket.AF_UNIX (the first canary proved
 * it) and a daemon cannot bind a transport its interpreter lacks. */
static void choose_pin(setup_t *setup)
{
    char chosen[16];
    const char *env = getenv(TRANSPORT_ENV);
    size_t i;

#ifdef _WIN32
    setup->pin = "tcp";
#else
    setup->pin = "unix";
#endif
    setup->pin_is_ours = 1;
    if (env == NULL || env[0] == '\0') {
        return;
    }
    /* Whatever it says, the child inherits it; we only have to read it well
     * enough to name the matching address file.  A value that is neither
     * transport is the daemon's to refuse by name, and the ladder is what
     * catches the refusal. */
    setup->pin_is_ours = 0;
    chosen[0] = '\0';
    for (i = 0; env[i] != '\0' && i < sizeof chosen - 1; i++) {
        char c = env[i];
        chosen[i] = (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c;
        chosen[i + 1] = '\0';
    }
    if (strcmp(chosen, "tcp") == 0) {
        setup->pin = "tcp";
    } else if (strcmp(chosen, "unix") == 0) {
        setup->pin = "unix";
    }
}

static int plan(setup_t *setup, const char *argv0)
{
    char directory[PATH_CAP];
    char runtime[PATH_CAP];
    char key[17];

    own_directory(argv0, directory, sizeof directory);
    if (join_path(setup->engine, sizeof setup->engine, directory, ENGINE_NAME) != 0) {
        note("bundle path is too long");
        return -1;
    }
    resolve_path(setup->engine, setup->engine_key, sizeof setup->engine_key);
    setup->named_basis = basis_of(argv0, setup->basis, sizeof setup->basis);

    if (runtime_dir(runtime, sizeof runtime) != 0) {
        return -1;
    }
    config_key(setup->engine_key, setup->basis, key);

    if (snprintf(setup->unix_path, sizeof setup->unix_path, "%s%c%s%s", runtime,
                 DIR_SEP, key, UNIX_SUFFIX) >= (int)sizeof setup->unix_path ||
        snprintf(setup->tcp_path, sizeof setup->tcp_path, "%s%c%s%s", runtime,
                 DIR_SEP, key, TCP_SUFFIX) >= (int)sizeof setup->tcp_path ||
        snprintf(setup->lock_path, sizeof setup->lock_path, "%s%c%s.lock", runtime,
                 DIR_SEP, key) >= (int)sizeof setup->lock_path) {
        note("runtime directory name is too long");
        return -1;
    }

    choose_pin(setup);
    copy_str(setup->address, sizeof setup->address,
             strcmp(setup->pin, "tcp") == 0 ? setup->tcp_path : setup->unix_path);
    if (snprintf(setup->log_path, sizeof setup->log_path, "%s.log", setup->address) >=
        (int)sizeof setup->log_path) {
        note("runtime directory name is too long");
        return -1;
    }
    return 0;
}

/* Rung 2's spawn.  The daemon binds and publishes at exactly the `--socket`
 * name it is handed, so the pin and the name are set together or not at all. */
static proc_t spawn_daemon(const setup_t *setup)
{
    char *args[12];
    int n = 0;

    if (!file_exists(setup->engine)) {
        note("no engine at %s", setup->engine);
        return NO_PROC;
    }
    if (setup->pin_is_ours) {
#ifdef _WIN32
        SetEnvironmentVariableA(TRANSPORT_ENV, setup->pin);
#else
        setenv(TRANSPORT_ENV, setup->pin, 1);
#endif
    }

    args[n++] = (char *)setup->engine;
    args[n++] = (char *)"--serve";
    args[n++] = (char *)"--socket";
    args[n++] = (char *)setup->address;
    args[n++] = (char *)"--idle-timeout";
    args[n++] = (char *)IDLE_TIMEOUT_REPR;
    args[n++] = (char *)"--log";
    args[n++] = (char *)setup->log_path;
    if (setup->named_basis) {
        /* Verbatim, empty included: a name that asked for a basis and spelt
         * it wrong must reach the engine's refusal by that name. */
        args[n++] = (char *)"--basis";
        args[n++] = (char *)setup->basis;
    }
    args[n] = NULL;
    return launch(setup->engine, args, setup->log_path, 1);
}

/* Rungs 1 and 2: a connected socket, starting the server if nobody answers.
 *
 * The fast path takes no lock at all -- a live server answers connect and
 * that is the whole story.  Only a failed connect enters the slow path, and
 * there the lock serialises the decision so a crew firing sixteen clients at
 * once starts ONE server: the losers block, and by the time they hold the
 * lock the winner's server is listening, so their retry is the fast path. */
static sock_t obtain(const setup_t *setup)
{
    lock_t lock;
    proc_t child;
    sock_t sock;
    double deadline;
    unsigned delay = POLL_MIN_MS;

    sock = connect_any(setup->unix_path, setup->tcp_path);
    if (sock != BAD_SOCK) {
        return sock;
    }

    lock = take_lock(setup->lock_path);
    if (lock == NO_LOCK) {
        note("cannot lock %s: %s", setup->lock_path, os_error());
        return BAD_SOCK;
    }

    sock = connect_any(setup->unix_path, setup->tcp_path);
    if (sock != BAD_SOCK) {
        release_lock(lock);
        return sock;
    }
    /* Whatever is on disk answered nothing, and under the lock is the only
     * place removing it is safe: a socket with nothing behind it, or a
     * rendezvous naming a port nobody answers, is litter the next bind must
     * not trip over. */
    if (file_exists(setup->unix_path)) {
        remove_file(setup->unix_path);
    }
    if (file_exists(setup->tcp_path)) {
        remove_file(setup->tcp_path);
    }

    child = spawn_daemon(setup);
    if (child == NO_PROC) {
        release_lock(lock);
        return BAD_SOCK;
    }

    deadline = now_seconds() + SPAWN_TIMEOUT;
    for (;;) {
        sock = connect_any(setup->unix_path, setup->tcp_path);
        if (sock != BAD_SOCK) {
            forget_process(child);
            release_lock(lock);
            return sock;
        }
        if (!process_alive(child)) {
            /* One more try before believing it: a server that bound, was
             * answered and exited between the connect above and this probe
             * is a lost race, not a failure. */
            sock = connect_any(setup->unix_path, setup->tcp_path);
            if (sock != BAD_SOCK) {
                forget_process(child);
                release_lock(lock);
                return sock;
            }
            note("the engine daemon exited without answering %s", setup->address);
            forget_process(child);
            release_lock(lock);
            return BAD_SOCK;
        }
        if (now_seconds() >= deadline) {
            note("the engine daemon did not answer %s in %gs", setup->address,
                 SPAWN_TIMEOUT);
            forget_process(child);
            release_lock(lock);
            return BAD_SOCK;
        }
        sleep_ms(delay);
        delay = delay * 3u / 2u;
        if (delay > POLL_MAX_MS) {
            delay = POLL_MAX_MS;
        }
    }
}

static int served(const setup_t *setup, const char *deck_path,
                  const char *printout_path)
{
    buf_t deck = {NULL, 0, 0};
    buf_t answer = {NULL, 0, 0};
    sock_t sock;
    int status = -1;

    if (read_whole_file(deck_path, &deck) != 0) {
        buf_free(&deck);
        return -1;
    }
    sock = obtain(setup);
    if (sock == BAD_SOCK) {
        buf_free(&deck);
        return -1;
    }
    if (send_all(sock, deck.data, deck.len) == 0) {
        /* EOF is the frame: the server answers once and closes, so the write
         * side has to go down before anything comes back. */
#ifdef _WIN32
        shutdown(sock, SD_SEND);
#else
        shutdown(sock, SHUT_WR);
#endif
        status = recv_to_eof(sock, &answer);
        if (status != 0) {
            note("the resident engine closed mid-answer");
        }
    } else {
        note("the resident engine closed while the deck was being sent");
    }
    close_sock(sock);
    buf_free(&deck);

    if (status != 0) {
        buf_free(&answer);
        return -1;
    }
    if (answer.len == 0) {
        /* A server that closed without a byte is a server that died
         * mid-solve; an empty printout must never reach EZNEC as an answer. */
        note("the resident engine closed without answering");
        buf_free(&answer);
        return -1;
    }
    status = write_whole_file(printout_path, answer.data, answer.len);
    buf_free(&answer);
    return status;
}

/* Rung 3: the bundle's own one-shot, correct at today's speed.  Exit 0 is
 * required rather than assumed even though the engine's contract is
 * exit-0-always: an exec that never found the engine also "completes", and
 * that must fall through to rung 4 rather than pass as an answer. */
static int one_shot(const setup_t *setup, const char *deck_path,
                    const char *printout_path)
{
    char *args[6];
    int n = 0;
    proc_t child;
    int status;

    if (!file_exists(setup->engine)) {
        note("no engine at %s", setup->engine);
        return -1;
    }
    args[n++] = (char *)setup->engine;
    if (setup->named_basis) {
        args[n++] = (char *)"--basis";
        args[n++] = (char *)setup->basis;
    }
    args[n++] = (char *)deck_path;
    args[n++] = (char *)printout_path;
    args[n] = NULL;

    child = launch(setup->engine, args, NULL, 0);
    if (child == NO_PROC) {
        return -1;
    }
    status = wait_for(child, ONE_SHOT_TIMEOUT);
    forget_process(child);
    if (status != 0) {
        note("the one-shot engine exited %d", status);
        return -1;
    }
    return 0;
}

/* Rung 4: the file is the only channel there is.
 *
 * Without the engine there is no header echo, and EZNEC may reject a
 * stampless printout as stale -- but a named refusal that MIGHT reach the
 * user beats an absent file, which certainly reads as a broken install.
 * CRLF because every other printout this bundle writes has it. */
static void last_ditch(const char *printout_path, const char *reason)
{
    char line[REASON_CAP + 96];
    int written = snprintf(line, sizeof line,
                           " ***** NEC ERROR - MOMWIRE ENGINE UNAVAILABLE - %s\r\n",
                           reason);
    if (written < 0) {
        return;
    }
    write_whole_file(printout_path, line, strlen(line));
}

int main(int argc, char **argv)
{
    setup_t setup;
    const char *deck_path;
    const char *printout_path;

    /* Rule 1 in both directions: EZNEC reads the printout and nothing else,
     * so every path below returns 0 -- including the two argument faults,
     * which print what the real engine prints and write no file because
     * there is no output path to write one to. */
    if (argc < 2) {
        printf("%s\n", ARGUMENT_ERROR_INPUT);
        return 0;
    }
    if (argc != 3) {
        printf("%s\n", ARGUMENT_ERROR_OUTPUT);
        return 0;
    }
    deck_path = argv[1];
    printout_path = argv[2];

#ifdef _WIN32
    {
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
            note("winsock is unavailable");
        }
    }
#else
    /* A daemon that dies mid-send must reach the ladder, not kill this
     * process: without this the write to a half-closed socket is a SIGPIPE
     * and rung 3 never runs. */
    signal(SIGPIPE, SIG_IGN);
#endif

    memset(&setup, 0, sizeof setup);
    if (plan(&setup, argv[0]) == 0 && served(&setup, deck_path, printout_path) == 0) {
        return 0;
    }
    if (one_shot(&setup, deck_path, printout_path) == 0) {
        return 0;
    }
    last_ditch(printout_path, g_reason);
    return 0;
}
