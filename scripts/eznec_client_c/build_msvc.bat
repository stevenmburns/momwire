@echo off
rem Build the native EZNEC thin client with MSVC (momwire#718 phase 3).
rem
rem     scripts\eznec_client_c\build_msvc.bat <output-path>
rem
rem Run from a Developer Command Prompt (or after vcvarsall.bat) so cl.exe and
rem the SDK are on PATH.  The counterpart of build_cc.sh, argument for
rem argument, and the same single translation unit.
rem
rem /MT, not /MD: the exe ships in a bundle a user unzips beside EZNEC, and a
rem missing VC++ redistributable would be an engine that "does not open" on
rem exactly the machines this arc exists to be fast on.
rem
rem ws2_32 for the sockets (AF_UNIX included -- Winsock has spoken it since
rem Win10 1803, which is why the .sock probe is tried on Windows at all) and
rem advapi32 for GetUserNameA, the runtime directory's last-resort name.
rem
rem The version is a compile-time define because it is a HASH INPUT: the
rem server key is scoped by eznec.<major>.<minor>, so the exe carries the
rem version of the tree that built it.
setlocal enabledelayedexpansion

set "HERE=%~dp0"
set "OUT=%~1"
if "%OUT%"=="" set "OUT=momwire-eznec.exe"
if "%PYTHON%"=="" set "PYTHON=python"

set "VERSION="
for /f "usebackq delims=" %%v in (`%PYTHON% -c "from importlib.metadata import version; print(version('momwire'))" 2^>nul`) do set "VERSION=%%v"
if "%VERSION%"=="" (
    for /f "usebackq tokens=3 delims= " %%v in (`findstr /b /c:"version = " "%HERE%..\..\pyproject.toml"`) do (
        if "!VERSION!"=="" set "VERSION=%%~v"
    )
)
if "%VERSION%"=="" (
    echo build_msvc.bat: cannot determine the momwire version 1>&2
    exit /b 1
)

for /f "tokens=1,2 delims=." %%a in ("%VERSION%") do (
    set "MAJOR=%%a"
    set "MINOR=%%b"
)
if "%MAJOR%"=="" set "MAJOR=0"
if "%MINOR%"=="" set "MINOR=0"

cl /nologo /W3 /WX /O2 /MT /TC ^
    /DMOMWIRE_VERSION_MAJOR=%MAJOR% ^
    /DMOMWIRE_VERSION_MINOR=%MINOR% ^
    /Fe:"%OUT%" "%HERE%momwire_eznec_client.c" ^
    /link ws2_32.lib advapi32.lib
exit /b %ERRORLEVEL%
