// SimNECProbe: launch a NEC-2 engine the way SimNEC does and print what SimNEC hides
// (exit code, stderr).       java SimNECProbe.java /absolute/path/to/momwire-nec2c-bspline
// 1. probe (Execute.testCommand): sh -c '"<cmd>" -version', stdin closed, exit code first.
// 2. daemon (NEC2Daemon): sh -c '"<cmd>"', cwd=home, deck+NX to stdin, stdout to the NX echo.
import java.io.*;

public class SimNECProbe {
  static final String DECK = "CE dipole\nGW 1 9 0. 0. -2.5 0. 0. 2.5 0.001\nGE 0\nEX 0 1 5 0 1.\nFR 0 1 0 0 30. 0\nXQ\n";

  public static void main(String[] a) throws Exception {
    String cmd = a[0];
    File f = new File(cmd);
    System.out.println("exists=" + f.exists() + " executable=" + f.canExecute() + "   (SimNEC's dialog check)");

    Process p = shell("\"" + cmd + "\" -version", null);
    p.getOutputStream().close();
    int exit = p.waitFor();
    System.out.println("probe exit " + exit + (exit == 0 ? "" : "   <- SimNEC shows 'NEC Failure Code " + exit + "' and stops here"));
    System.out.println("probe stdout: " + read(p.getInputStream()));
    System.out.println("probe stderr: " + read(p.getErrorStream()));
    if (exit != 0) return;

    p = shell("\"" + cmd + "\"", new File(System.getProperty("user.home")));
    PrintStream in = new PrintStream(p.getOutputStream());
    in.print("CM FF 2\n" + DECK + "\n" + "NX\n\n");
    in.flush();
    BufferedReader out = new BufferedReader(new InputStreamReader(p.getInputStream()));
    int lines = 0;
    String line;
    while ((line = out.readLine()) != null) {
      lines++;
      if (line.trim().matches("DATA CARD No:\\s+\\d+\\s+NX\\b.*")) break;
    }
    if (line == null) System.out.println("stdout ended after " + lines + " lines   <- SimNEC's 'BAD NEC RESPONSE'");
    else System.out.println("deck answered: " + lines + " lines, NX echo seen, engine still resident");
    in.close();
    System.out.println("daemon exit " + p.waitFor() + " after stdin closed");
    System.out.println("daemon stderr: " + read(p.getErrorStream()));
  }

  static Process shell(String command, File dir) throws IOException {
    boolean win = System.getProperty("os.name").toLowerCase().startsWith("windows");
    ProcessBuilder pb = win ? new ProcessBuilder("cmd.exe", "/c", command) : new ProcessBuilder("sh", "-c", command);
    if (dir != null) pb.directory(dir);
    return pb.start();
  }

  static String read(InputStream s) throws IOException {
    return new String(s.readAllBytes()).trim();
  }
}
