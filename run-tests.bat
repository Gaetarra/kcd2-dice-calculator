@echo off
rem Verify the engine: scoring, dice weights, the roll tables against brute
rem force, the compiled kernel against plain Python, and 40,000 simulated turns
rem against the DP's own predicted value. Takes a couple of minutes.
setlocal
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo Run "KCD2-Farkle.bat" first - it sets up the environment.
    pause
    exit /b 1
)

cd kcd2farkle
"..\%PY%" test_engine.py
echo.
pause
endlocal
