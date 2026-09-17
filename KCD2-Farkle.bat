@echo off
rem Open the KCD2 dice calculator.
rem It is one self-contained HTML page - you can double-click that directly - but
rem Chrome and Edge only allow the parallel search on a served page, so this
rem starts a local server first. Close this window when you are done.
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (set "PY=.venv\Scripts\python.exe" & goto run)
py -3 --version >nul 2>&1
if not errorlevel 1 (set "PY=py -3" & goto run)
python --version >nul 2>&1
if not errorlevel 1 (set "PY=python" & goto run)

echo No Python found, so opening the page directly.
echo The search will use one core instead of all of them. Install Python from
echo https://www.python.org/downloads/ to get the parallel version.
echo.
start "" "index.html"
timeout /t 4 >nul
exit /b

:run
%PY% serve.py
