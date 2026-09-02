// SimNECDriver — a stand-in for SimNEC's engine launcher, built to reproduce
// "works from the terminal, fails from Java" reports without SimNEC itself.
//
// It performs the two things SimNEC does to an external NEC-2 engine, the
// way SimNEC does them (read off `nec2/Execute.testCommand` and
// `nec2/NEC2Daemon` in SimNEC.jar, documented in
// docs/2026-08-08-simnec-execute-grammar.md):
//
//   1. THE PROBE.  `sh -c '"<cmd>" -version'` (cmd.exe /c on Windows), stdin
//      closed at once, then `Process.waitFor()` BEFORE a byte of stdout is
//      read.  A non-zero exit is the whole verdict ("NEC Failure Code <n>");
//      only then is the first stdout line matched against the version
//      regexes.  On Unix the child's environment is NOT touched (the
//      Examples/Utils PATH prepend is Windows-only).
//
//   2. THE DAEMON.  `sh -c '"<cmd>"'` with cwd = user.home, one process kept
//      for the whole session.  Each deck goes down stdin as
//      "CM FF 2\n" + deck + "\n" then "NX\n\n", and stdout is read line by line
//      until the NX data-card echo.  Stderr is never read.  End of stream is
//      "BAD NEC RESPONSE".  Teardown is Process.destroy().
//
// Everything SimNEC would hide — the exit code number, stderr, a hang — is
// reported here, labelled with what SimNEC would have shown instead.
//
// Usage (Java 11+, no build step):
//
//     java scripts/simnec_driver/SimNECDriver.java /abs/path/momwire-nec2c-bspline [deck ...]
//
// Options: --home DIR (cwd for the daemon, default user.home), --repeat N
// (submit each deck N times down the one process), --probe-only,
// --timeout SECONDS (default 120; SimNEC has none, it blocks forever).
// A deck file's trailing NX/EN card is stripped; SimNEC frames decks itself.
// With no deck given, a free-space dipole is used.

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class SimNECDriver {

    static final boolean IS_WINDOWS =
        System.getProperty("os.name").toLowerCase(Locale.ROOT).startsWith("windows");

    // nec2/Execute: the four version probes, tried in this order with lookingAt().
    static final Pattern VERSION_A = Pattern.compile("nec2c\\.ae6ty\\.(.*)");
    static final Pattern VERSION_B = Pattern.compile("5b4az\\.ae6ty\\.(.*)");
    static final Pattern VERSION_C = Pattern.compile("necpp\\.nec2c\\.(.*)");
    static final Pattern VERSION_NECD = Pattern.compile("(NEC\\d+\\D.*)");

    static final String DEFAULT_DECK =
        "CE dipole free space\n"
        + "GW 1 9 0. 0. -2.5 0. 0. 2.5 0.001\n"
        + "GE 0\n"
        + "EX 0 1 5 0 1.\n"
        + "FR 0 1 0 0 30. 0\n"
        + "XQ\n";

    static int failures = 0;
    static long timeoutSeconds = 120;

    static void ok(String what) { System.out.println("  ok    " + what); }
    static void fail(String what) { System.out.println("  FAIL  " + what); failures++; }
    static void note(String what) { System.out.println("        " + what); }

    public static void main(String[] args) throws Exception {
        String cmd = null;
        String home = System.getProperty("user.home");
        int repeat = 1;
        boolean probeOnly = false;
        List<Path> deckFiles = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--home": home = args[++i]; break;
                case "--repeat": repeat = Integer.parseInt(args[++i]); break;
                case "--timeout": timeoutSeconds = Long.parseLong(args[++i]); break;
                case "--probe-only": probeOnly = true; break;
                default:
                    if (cmd == null) cmd = args[i]; else deckFiles.add(Paths.get(args[i]));
            }
        }
        if (cmd == null) {
            System.err.println("usage: java SimNECDriver.java <engine path> [deck ...] [--home DIR] [--repeat N] [--probe-only] [--timeout S]");
            System.exit(2);
        }

        System.out.println("SimNECDriver  java " + System.getProperty("java.version")
            + "  os " + System.getProperty("os.name") + " " + System.getProperty("os.arch"));
        System.out.println("engine: " + cmd);
        System.out.println("PATH:   " + System.getenv("PATH"));
        System.out.println("HOME:   " + System.getProperty("user.home"));
        System.out.println();

        List<String> decks = new ArrayList<>();
        if (deckFiles.isEmpty()) decks.add(DEFAULT_DECK);
        for (Path p : deckFiles) decks.add(stripFraming(new String(Files.readAllBytes(p), StandardCharsets.UTF_8)));

        // ---- NEC2PortalDialog.safeNecCommand(): the dialog string is ONE file.
        System.out.println("dialog validation (NEC2PortalDialog.safeNecCommand)");
        File f = new File(cmd);
        if (f.exists() && f.canExecute()) ok("exists and is executable");
        else fail("NO NEC SERVICE FOUND (nec2c not yet configured?)  exists=" + f.exists() + " canExecute=" + f.canExecute());
        String lower = cmd.toLowerCase(Locale.ROOT);
        String engine = lower.contains("nec2c") ? "NEC2C" : lower.contains("nec5") ? "NEC5" : lower.contains("nec42") ? "NEC42" : null;
        if (engine == null) fail("NEC Error: path names no known engine (needs nec2c / nec5 / nec42 in it)");
        else ok("classified " + engine + " by path substring");
        if (!"NEC2C".equals(engine)) note("only the NEC2C route (stdin/stdout daemon) is driven here");
        System.out.println();

        // ---- Execute.testCommand(cmd, quiet)
        System.out.println("version probe (Execute.testCommand)");
        boolean probeOk = testCommand(cmd, lower.contains("nec5"));
        System.out.println();
        if (probeOnly || !probeOk) {
            summary();
            return;
        }

        // ---- NEC2Daemon + submit()
        System.out.println("resident daemon (NEC2Daemon), cwd=" + home + ", " + decks.size() + " deck(s) x " + repeat);
        daemon(cmd, home, decks, repeat);
        summary();
    }

    static void summary() {
        System.out.println();
        if (failures == 0) System.out.println("PASS");
        else System.out.println("FAIL  " + failures + " check(s); see above");
        System.exit(failures == 0 ? 0 : 1);
    }

    // The exact ProcessBuilder shapes.  The command string is double-quoted in
    // both, so a path with spaces survives the shell on this route.
    static ProcessBuilder shellCommand(String cmd, String tail) {
        String quoted = "\"" + cmd + "\"" + tail;
        ProcessBuilder pb = new ProcessBuilder();
        if (IS_WINDOWS) {
            // Execute.enhancePath: SimNEC prepends its own Examples/Utils to
            // PATH here (Windows only). Stand in with a harmless directory.
            String path = pb.environment().get("PATH");
            pb.environment().put("PATH", System.getProperty("java.io.tmpdir") + ";" + path);
            pb.command("cmd.exe", "/c", quoted);
        } else {
            pb.command("sh", "-c", quoted);
        }
        return pb;
    }

    static boolean testCommand(String cmd, boolean isNec5) throws Exception {
        // Execute.testCommand's own pre-checks.
        String base = cmd.substring(cmd.lastIndexOf(File.separator) + 1);
        int dot = base.lastIndexOf(".");
        if (dot > 0) base = base.substring(0, dot);
        if (!new File(cmd).exists()) { fail("Command not found: '" + cmd + "'"); return false; }
        if (base.contains("out")) { fail("Can't execute 'out' file:" + cmd); return false; }
        if (!new File(cmd).canExecute()) { fail("Bad Execute Permission: '" + cmd + "'"); return false; }

        ProcessBuilder pb = shellCommand(cmd, " -version");
        note("launch: " + pb.command());
        long t0 = System.nanoTime();
        Process p = pb.start();
        p.getOutputStream().close();
        BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
        // SimNEC: p.waitFor() with no timeout and nothing draining stdout or
        // stderr. A child that writes more than a pipe holds before exiting
        // would deadlock SimNEC; here it shows up as a timeout.
        if (!p.waitFor(timeoutSeconds, TimeUnit.SECONDS)) {
            fail("probe did not exit in " + timeoutSeconds + " s (SimNEC would block here forever); stderr pipe holds "
                + p.getErrorStream().available() + " bytes, stdout pipe holds " + p.getInputStream().available());
            p.destroyForcibly();
            return false;
        }
        int exit = p.exitValue();
        long ms = (System.nanoTime() - t0) / 1_000_000;
        String stderr = drain(p.getErrorStream());
        if (!(isNec5 && exit == 24) && exit != 0) {
            fail("exit " + exit + " after " + ms + " ms  -> SimNEC shows the warning 'NEC Failure Code " + exit + "' and stops here");
            note("meaning: " + exitMeaning(exit));
            showStderr(stderr);
            String out = drain(p.getInputStream());
            if (!out.isEmpty()) note("stdout (SimNEC never reads it on this path): " + out.trim());
            return false;
        }
        ok("exit " + exit + " after " + ms + " ms");
        String line = br.readLine();
        if (line == null) line = "";
        line = line.trim();
        p.destroy();
        Matcher m;
        if ((m = VERSION_A.matcher(line)).lookingAt() || (m = VERSION_B.matcher(line)).lookingAt()
            || (m = VERSION_C.matcher(line)).lookingAt()) {
            String tail = m.group(1);
            try {
                Double.valueOf(tail);
                ok("first line matches a legacy nec2c shape, version tail " + tail + "  (" + line + ")");
            } catch (NumberFormatException e) {
                fail("legacy shape but tail '" + tail + "' is not a Double -> 'nec2c version too old:'  (" + line + ")");
            }
        } else if (VERSION_NECD.matcher(line).lookingAt()) {
            ok("first line matches versionNECd  (" + line + ")");
        } else {
            fail("first line matches no version probe -> 'Unable to run'  (got: '" + line + "')");
        }
        showStderr(stderr);
        return failures == 0;
    }

    static void daemon(String cmd, String home, List<String> decks, int repeat) throws Exception {
        ProcessBuilder pb = shellCommand(cmd, "");
        pb.directory(new File(home));
        note("launch: " + pb.command());
        Process p = pb.start();
        // utilities.StreamPump: a PrintStream fed line-wise, flushed after each write.
        PrintStream ps = new PrintStream(p.getOutputStream());
        InputStream is = p.getInputStream();
        LineReader reader = new LineReader(is);

        int n = 0;
        for (int r = 0; r < repeat; r++) {
            for (String deck : decks) {
                n++;
                String text = "CM FF 2\n" + deck;    // NEC2Daemon.submit's prefix (no SOMNEC cache file here)
                ps.print(text + "\n"); ps.flush();
                ps.print("NX\n" + "\n"); ps.flush();
                long t0 = System.nanoTime();
                int lines = 0, inputTables = 0, errors = 0;
                String end = null;
                while (true) {
                    String line = reader.next(timeoutSeconds);
                    if (line == LineReader.TIMEOUT) {
                        fail("deck " + n + ": no line for " + timeoutSeconds + " s after " + lines + " line(s) (SimNEC would block forever)"
                            + "; alive=" + p.isAlive() + "; stderr pipe holds " + p.getErrorStream().available() + " bytes (SimNEC never drains it)");
                        showStderr(drain(p.getErrorStream()));
                        p.destroyForcibly();
                        return;
                    }
                    if (line == null) {
                        // Execute.getLine() returned null -> NECException("BAD NEC RESPONSE"),
                        // which surfaces as a stack trace inside SimNEC.
                        boolean dead = p.waitFor(5, TimeUnit.SECONDS);
                        fail("deck " + n + ": end of stream after " + lines + " line(s) -> 'BAD NEC RESPONSE'"
                            + (dead ? "; process exited " + p.exitValue() + " (" + exitMeaning(p.exitValue()) + ")" : "; process still alive"));
                        showStderr(drain(p.getErrorStream()));
                        p.destroyForcibly();
                        return;
                    }
                    lines++;
                    String[] parts = line.trim().split("\\s+");
                    if (partsMatch(parts, "ERROR:")) errors++;
                    if (partsMatch(parts, "---------", "ANTENNA", "INPUT", "PARAMETERS")
                        || partsMatch(parts, "-", "-", "-", "ANTENNA", "INPUT", "PARAMETERS")) inputTables++;
                    if (partsMatch(parts, "DATA", "CARD", "No:", "", "NX")) { end = "NX data-card echo"; break; }
                    if (stringsInOrder(line, "INPUT", "LINE", "NX")) { end = "NX input echo"; break; }
                    if (partsMatch(parts, "RUN", "TIME", "=")) { end = "RUN TIME ="; break; }
                    if (partsMatch(parts, "TOTAL", "RUN", "TIME:")) { end = "TOTAL RUN TIME:"; break; }
                }
                long ms = (System.nanoTime() - t0) / 1_000_000;
                String verdict = "deck " + n + ": " + lines + " lines, " + inputTables + " input table(s), ended on " + end + ", " + ms + " ms";
                if (inputTables == 0) fail(verdict + " -> no ANTENNA INPUT PARAMETERS table");
                else if (errors > 0) fail(verdict + " -> " + errors + " ERROR: line(s) (SimNEC warns 'NEC ERROR (1)')");
                else ok(verdict);
            }
        }
        if (!p.isAlive()) fail("engine exited " + p.exitValue() + " while SimNEC would still be holding it");
        else ok("engine still resident after " + n + " deck(s)");

        // Read what stderr holds before teardown: Process.destroy() closes
        // the streams, and SimNEC would never have looked.
        String err = drain(p.getErrorStream());
        if (!err.isEmpty()) note("stderr over the session (SimNEC never reads it; " + err.length() + " chars):\n" + indent(err));

        // NEC2Daemon.destroy(): kill, then close the three streams. SimNEC's
        // order is destroy, stdout, stdin. On Windows destroy() kills only
        // cmd.exe and the engine underneath survives until its stdin closes,
        // so here stdin is closed FIRST: this driver reads stdout on a thread,
        // and a Windows close of a stream with a read pending waits for that
        // read, which only ends when the engine does. SimNEC reads on the
        // calling thread and does not have that pending read.
        p.destroy();
        ps.close();
        String tail = reader.next(timeoutSeconds);
        while (tail != null && tail != LineReader.TIMEOUT) tail = reader.next(timeoutSeconds);
        if (tail == null) ok("engine ended once stdin closed");
        else fail("engine still holding stdout " + timeoutSeconds + " s after stdin closed");
        is.close();
        boolean gone = p.waitFor(10, TimeUnit.SECONDS);
        note("after destroy(): shell " + (gone ? "exited " + p.exitValue() + " (" + exitMeaning(p.exitValue()) + ")" : "still alive after 10 s"));
    }

    // nec2/Execute.partsMatch: "" is a one-field wildcard; parts may be longer than args.
    static boolean partsMatch(String[] parts, String... args) {
        if (parts.length < args.length) return false;
        for (int i = 0; i < args.length; i++) if (!args[i].isEmpty() && !args[i].equals(parts[i])) return false;
        return true;
    }

    // nec2/Execute.stringsInOrder: uppercased substrings at increasing indexes.
    static boolean stringsInOrder(String line, String... args) {
        String u = line.toUpperCase(Locale.ROOT);
        int at = 0;
        for (String a : args) {
            int i = u.indexOf(a.toUpperCase(Locale.ROOT), at);
            if (i < 0) return false;
            at = i + a.length();
        }
        return true;
    }

    static String stripFraming(String deck) {
        List<String> lines = new ArrayList<>(Arrays.asList(deck.split("\r?\n")));
        while (!lines.isEmpty()) {
            String last = lines.get(lines.size() - 1).trim();
            if (last.isEmpty() || last.equals("NX") || last.equals("EN") || last.startsWith("NX ") || last.startsWith("EN ")) lines.remove(lines.size() - 1);
            else break;
        }
        return String.join("\n", lines) + "\n";
    }

    static String exitMeaning(int exit) {
        switch (exit) {
            case 0: return "success";
            case 1: return "the engine itself failed, typically an uncaught exception (traceback on stderr)";
            case 2: return "argument error";
            case 3: return "momwire refused at configure time and said why on stdout (unknown basis from the file name, or a bad --cache-stats path)";
            case 126: return "the shell found it but could not execute it: permission bits, or macOS 'Operation not permitted' (folder privacy protection for Desktop/Documents/Downloads under a GUI app)";
            case 127: return "the shell could not find it, or its #! interpreter is missing";
            case 134: return "abort (SIGABRT), e.g. a native library failing to initialise";
            case 137: return "killed (SIGKILL), the macOS quarantine/Gatekeeper shape";
            case 139: return "segmentation fault in native code";
            case 143: return "terminated (SIGTERM), what Process.destroy() sends";
            case 253: return "nec2c's own 'Error reading input file' exit";
            default: return exit > 128 ? "killed by signal " + (exit - 128) : "engine-specific";
        }
    }

    static String drain(InputStream in) throws IOException {
        StringBuilder sb = new StringBuilder();
        byte[] buf = new byte[8192];
        while (in.available() > 0) {
            int k = in.read(buf, 0, Math.min(buf.length, in.available()));
            if (k < 0) break;
            sb.append(new String(buf, 0, k, StandardCharsets.UTF_8));
        }
        return sb.toString();
    }

    static void showStderr(String stderr) {
        if (stderr.isEmpty()) note("stderr: empty");
        else note("stderr (SimNEC never shows this):\n" + indent(stderr));
    }

    static String indent(String s) {
        StringBuilder sb = new StringBuilder();
        for (String l : s.split("\r?\n")) sb.append("          | ").append(l).append('\n');
        return sb.toString().replaceAll("\\s+$", "");
    }

    // BufferedReader.readLine on a thread, so a silent engine is a reported
    // timeout here rather than the forever-block it is in SimNEC.
    static final class LineReader {
        static final String TIMEOUT = new String("<timeout>");
        static final String EOF = new String("<eof>");
        final LinkedBlockingQueue<String> q = new LinkedBlockingQueue<>();

        LineReader(InputStream is) {
            BufferedReader br = new BufferedReader(new InputStreamReader(is));
            Thread t = new Thread(() -> {
                try {
                    String l;
                    while ((l = br.readLine()) != null) q.put(l);
                } catch (Exception e) {
                    // fall through to EOF
                }
                try { q.put(EOF); } catch (InterruptedException ignored) { }
            }, "SimNECDriver-reader");
            t.setDaemon(true);
            t.start();
        }

        String next(long seconds) throws InterruptedException {
            String l = q.poll(seconds, TimeUnit.SECONDS);
            if (l == null) return TIMEOUT;
            if (l == EOF) return null;
            return l;
        }
    }
}
