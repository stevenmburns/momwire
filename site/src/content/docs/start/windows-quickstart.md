---
title: "Quick start: momwire in EZNEC, SimNEC or 4nec2"
description: Use momwire as the calculating engine of the antenna program you already have on Windows. One download, no Python, and the program's own setting.
---

This page is for using momwire inside the antenna modelling program you already
run on Windows: EZNEC Pro+, SimNEC or 4nec2. You keep that program's screens,
models and plots. momwire replaces only the engine that does the calculation.
You don't need Python, and you don't need to type commands.

## 1. Download and unpack

1. Download
   [`momwire-windows.zip`](https://github.com/stevenmburns/momwire/releases/latest/download/momwire-windows.zip).
   The same file serves all three programs.
2. Before unpacking, right-click the zip, choose **Properties**, tick
   **Unblock** and click **OK**. Windows otherwise marks every file inside as
   downloaded from the internet.
3. Unpack it into a folder of its own. The zip contains a `momwire` folder, so
   unpacking it into `C:\` gives you `C:\momwire\`. The rest of this page
   assumes that location.
   - Don't unpack it into your antenna program's folder.
   - Keep the folder together. Each `.exe` needs the `_internal` folder beside
     it, so copying one `.exe` somewhere else won't work.

## 2. Pick the file for your program's engine setting

Each program offers one or more NEC engine settings: NEC-2, NEC-4.2 or NEC-5.
Each setting has two momwire files to choose from:

| Engine setting | momwire's default | Matches the original engine |
|---|---|---|
| NEC-5 (EZNEC, SimNEC, 4nec2) | `momwire-nec5.exe` | `momwire-nec5-razor2p.exe` |
| NEC-4.2 (EZNEC, SimNEC, 4nec2) | `momwire-nec42.exe` | `momwire-nec42-sinusoidal.exe` |
| NEC-2 in SimNEC | `momwire-nec2c.exe` | `momwire-nec2c-sinusoidal.exe` |
| NEC-2 in 4nec2 | `momwire-nec2d.exe` | `momwire-nec2d-sinusoidal.exe` |

- **The default** is momwire's own way of calculating. Start with it.
- **The other column** calculates the way that NEC engine does, so its answers
  sit close to what the original engine gives. Use it when you are comparing
  with results you already have.
- For 4nec2's NEC-2 setting use the `nec2d` files. 4nec2 can't read the
  printout the `nec2c` files write.

[Choosing an engine](/reference/choosing-an-engine/) explains how the two
differ.

## 3. Point your program at it

**EZNEC Pro+.** Set the external engine path to the file from the table:
`C:\momwire\momwire-nec5.exe` for External NEC-5, or
`C:\momwire\momwire-nec42.exe` for External NEC-4.2. EZNEC has no NEC-2
setting for an outside engine.

**SimNEC.** In the engine setting for NEC-2, NEC-4.2 or NEC-5, give the full
path to the matching file, for example `C:\momwire\momwire-nec5.exe`. SimNEC
decides which kind of deck to write from the file name, so keep the names as
they are. It also refuses any path containing the letters `out` (for example
under `C:\Users\scout\`), and `C:\momwire\` avoids that.

**4nec2.** 4nec2 runs its engines through a batch file, `4nec2.bat`, so momwire
goes in by changing one line in it. Make the change in a copy of the 4nec2
folder and keep your original 4nec2 as it was. NEC-5 also needs Dan AC6LA's
NEC-5 package for 4nec2. The steps are in
[In 4nec2](/reference/eznec-nec5/#in-4nec2).

## 4. Check that it worked

Run any model and look at the top of the printout: it names momwire as the
engine that answered.

With the NEC-4.2 and NEC-5 files, the first calculation can take a few seconds
while the engine loads, and the program shows nothing while it waits. The
engine then stays loaded for fifteen idle minutes, so each calculation after
that starts almost at once. The NEC-2 files load the engine for every run.

## When momwire says no

If momwire can't calculate a model exactly as written, it doesn't guess. The
printout says what it can't do and why, in place of the results. Change the
model, or run that one with your original engine.
[What refuses, and why](/reference/eznec-nec5/#what-refuses-and-why) lists the
cases.

## Updating to a new release

Delete the whole `C:\momwire\` folder, then unpack the new zip in its place.
Your program's setting doesn't change, because the path is the same.

Don't unpack a new release over an old one: files from the old release stay
behind, and the printout then reports the version as `(stale: …)`. If Windows
says a file is in use while you delete the folder, the engine is still loaded
from your last run. Wait fifteen minutes, or end `momwire-engine.exe` in Task
Manager.

## More detail

- [Which momwire to run on Windows](/reference/windows/) covers the faster
  standalone NEC-5 executable and when to use it.
- [Running momwire as EZNEC's NEC-5 engine](/reference/eznec-nec5/) is the full
  reference for the zip.
