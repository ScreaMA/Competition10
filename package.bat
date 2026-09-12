@echo off
REM package.bat - build a timestamped release archive on Windows
REM NOTE: keep this file ASCII-only; cmd.exe parses .bat with the OEM codepage
REM (GBK on zh-CN Windows) and non-ASCII comments break command parsing.

REM --- build timestamp (yyyyMMdd_HHmmss) ---
for /f "usebackq tokens=*" %%I in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"`) do set TIMESTAMP=%%I
if not "%TIMESTAMP%"=="" goto stamp_ok
REM fallback for machines without PowerShell
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%
:stamp_ok

REM --- package name ---
set PACKAGE_NAME=CoreGeek_%TIMESTAMP%.tar.gz

REM --- clean temp files ---
cd /d "%~dp0"
cd CoreGeek
del /s /q *.pyc 2>nul
for /d /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
del /q debug.log 2>nul
cd ..

REM --- create archive ---
tar -czf "%PACKAGE_NAME%" CoreGeek/

echo Package created: %PACKAGE_NAME%
dir "%PACKAGE_NAME%"
