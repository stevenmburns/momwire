# SimNECDriver — SimNEC's engine launcher, without SimNEC

A single-file Java program that does to a NEC-2 engine exactly what SimNEC
does, and reports everything SimNEC would have hidden: the exit code number,
stderr, and a hang. It exists for one report: *"the command works from my
terminal, and SimNEC says NEC Failure Code."*

The two calls it reproduces are read off `nec2/Execute.testCommand` and
`nec2/NEC2Daemon` in SimNEC.jar (see `docs/2026-08-08-simnec-execute-grammar.md`):

| step | SimNEC's call | what decides pass/fail |
|---|---|---|
| dialog | `new File(cmd).exists() && canExecute()`; engine class by the substrings `nec2c` / `nec5` / `nec42` in the lowercased path | one file, no arguments, no PATH search |
| probe | `sh -c '"<cmd>" -version'` (`cmd.exe /c` on Windows), stdin closed, `waitFor()` **before any stdout is read** | exit code must be 0 (24 allowed for nec5); then the first line against four version regexes |
| daemon | `sh -c '"<cmd>"'`, cwd = `user.home`, one process for the session; per deck, `"CM FF 2\n" + deck + "\n"` then `"NX\n\n"` down stdin; stdout read line by line until the `DATA CARD No: n NX` echo | end of stream is `BAD NEC RESPONSE`; stderr is never read |

On Unix the probe does **not** touch the child's environment. The
`Examples/Utils` PATH prepend happens only on Windows.

## The short one, for sending to someone

`SimNECProbe.java` is the same two launches in under 50 lines, with no
threads, no timeouts and no options, so a recipient can read the whole thing
before running it. It prints the probe's exit code and stderr, then submits
one dipole deck to the resident daemon and prints how that ended:

```
java SimNECProbe.java /absolute/path/to/momwire-nec2c-bspline
```

Everything below is the full driver, which exists for CI.

## Run it

Java 11 or newer, no build step:

```
java scripts/simnec_driver/SimNECDriver.java /abs/path/momwire-nec2c-bspline [deck ...]
```

Options: `--home DIR` (the daemon's cwd), `--repeat N` (each deck N times
down the one process), `--probe-only`, `--timeout S` (default 120; SimNEC
itself has no timeout and blocks forever). A deck's trailing `NX`/`EN` card
is stripped; SimNEC frames decks itself.

`run_shapes.sh <engine>` runs the driver through the launch shapes that
differ between a terminal and a GUI app: the terminal's environment, a
scrubbed GUI-launch environment, a path with a space, a name-selected basis,
a name that selects no basis (must fail, exit 3), and an entry point whose
interpreter is gone (must fail, exit 127). `.github/workflows/simnec-driver.yml`
runs it on macOS, Linux and Windows against the wheel on PyPI.

## Reading a failure

A non-zero probe exit is reported with SimNEC's own wording and what the
number means:

| exit | meaning |
|---|---|
| 1 | the engine raised; the traceback is on stderr, which SimNEC never shows |
| 3 | momwire refused at configure time and said why on **stdout**: an unknown basis from the file name, or a bad `--cache-stats` path |
| 126 | found but not executable: permission bits, or macOS "Operation not permitted" (folder privacy protection under a GUI app) |
| 127 | not found, or the `#!` interpreter is gone (a moved or deleted venv) |
| 137 | killed: the macOS quarantine shape |

The daemon stage reports end-of-stream with the process's exit code and
stderr, and a silent engine as a timeout with the byte count sitting in the
stderr pipe that nothing drains.
