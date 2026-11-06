"""momwire#1822 minimal repro: the resident server's wire shape, no momwire.

    python repro_1822.py server PATH LOG [--accept-delay S] [--blocking]
    python repro_1822.py client PATH ID [--pre-shutdown-delay S]
    python repro_1822.py drive PATH LOG N CONC proc|thread [client opts]

The server is serve/_server.py's accept loop (a 0.25 s accept timeout, a
thread per connection) and eznec/_resident.py's framing (latin-1 text over
makefile, read to EOF, answer, close). The client is
momwire_eznec_client._served_bytes (connect, sendall, shutdown(SHUT_WR), recv
to EOF). DIAG ONLY: the client's recv has a timeout here so a hang is counted
rather than waited out.
"""

from __future__ import annotations

import io
import os
import socket
import subprocess
import sys
import threading
import time

DECK = ("CM deck\r\n" + "GW 1 21 0 0 -5 0 0 5 0.001\r\n" * 60 + "EN\r\n").encode()
ANSWER_LINES = 1500


def _arg(argv, flag, default=None, cast=str):
    if flag in argv:
        i = argv.index(flag)
        v = argv[i + 1]
        del argv[i : i + 2]
        return cast(v)
    return default


def server(argv):
    path, log_path = argv[0], argv[1]
    accept_delay = _arg(argv, "--accept-delay", 0.0, float)
    blocking = "--blocking" in argv
    log = open(log_path, "a", buffering=1)
    lock = threading.Lock()

    def note(text):
        with lock:
            log.write(f"{time.monotonic():.6f} {text}\n")

    try:
        os.unlink(path)
    except OSError:
        pass
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(path)
    listener.listen(64)
    if not blocking:
        listener.settimeout(0.25)
    note(f"listening {os.getpid()}")

    def serve(conn, number):
        try:
            rx = io.TextIOWrapper(conn.makefile("rb"), encoding="latin-1")
            tx = io.TextIOWrapper(conn.makefile("wb"), encoding="latin-1", newline="\r\n")
            ident = None
            nbytes = 0
            for line in rx:
                if ident is None:
                    ident = line.strip()
                    note(f"first {number} {ident} timeout={conn.gettimeout()}")
                nbytes += len(line)
            note(f"eof {number} {ident} {nbytes}")
            tx.write(("X" * 70 + "\n") * ANSWER_LINES)
            tx.flush()
            tx.close()
            rx.close()
        except Exception as exc:  # noqa: BLE001 - diag
            note(f"error {number} {exc!r}")
        finally:
            conn.close()
            note(f"closed {number}")

    served = 0
    while True:
        if accept_delay:
            time.sleep(accept_delay)
        try:
            conn, _ = listener.accept()
        except TimeoutError:
            continue
        served += 1
        note(f"accept {served} fd={conn.fileno()}")
        threading.Thread(target=serve, args=(conn, served), daemon=True).start()


def client_once(path, ident, pre_shutdown_delay=0.0, timeout=20.0):
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.connect(path)
    try:
        conn.sendall(f"ID {ident}\r\n".encode() + DECK)
        if pre_shutdown_delay:
            time.sleep(pre_shutdown_delay)
        conn.shutdown(socket.SHUT_WR)
        conn.settimeout(timeout)
        got = 0
        while True:
            try:
                chunk = conn.recv(65536)
            except TimeoutError:
                return f"HANG {ident} after {got} bytes"
            if not chunk:
                break
            got += len(chunk)
    finally:
        conn.close()
    return "ok" if got == 71 * ANSWER_LINES + ANSWER_LINES else f"SHORT {ident} {got}"


def client(argv):
    path, ident = argv[0], argv[1]
    d = _arg(argv, "--pre-shutdown-delay", 0.0, float)
    r = client_once(path, ident, d)
    print(r)
    return 0 if r == "ok" else 1


def drive(argv):
    path, log_path, n, conc, mode = argv[0], argv[1], int(argv[2]), int(argv[3]), argv[4]
    rest = argv[5:]
    d = _arg(list(rest), "--pre-shutdown-delay", 0.0, float)
    seconds = _arg(rest, "--seconds", 1e9, float)
    stop = time.monotonic() + seconds
    counter = iter(range(n))
    clock = threading.Lock()
    bad: list[str] = []
    done = [0]

    def worker():
        while time.monotonic() < stop:
            with clock:
                i = next(counter, None)
            if i is None:
                return
            if mode == "thread":
                r = client_once(path, str(i), d)
            else:
                p = subprocess.run(
                    [sys.executable, "-I", __file__, "client", path, str(i), *rest],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                r = p.stdout.strip() or f"EXIT {i} {p.returncode} {p.stderr[-300:]}"
            with clock:
                done[0] += 1
                if r != "ok":
                    bad.append(r)
                    print(r, flush=True)

    threads = [threading.Thread(target=worker) for _ in range(conc)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    dt = time.monotonic() - t0
    print(f"RESULT mode={mode} conc={conc} done={done[0]} bad={len(bad)} {dt:.1f}s", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    sys.exit({"server": server, "client": client, "drive": drive}[cmd](rest))
