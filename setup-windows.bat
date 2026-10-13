@echo off
rem CSIDetector setup app for Windows: double-click to run.
rem First run: creates a Python environment in host\.venv and installs the
rem requirements (a few minutes); later runs just start the app.
setlocal
cd /d "%~dp0host" || goto :nofolder

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY where python >nul 2>&1 && set "PY=python"
if not defined PY goto :nopython
%PY% -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1 || goto :oldpython

if not exist ".venv\Scripts\python.exe" (
    echo Creating a Python environment in host\.venv ...
    %PY% -m venv .venv || goto :venvfail
)
echo Checking the Python packages (the first time this downloads them) ...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt || goto :pipfail

".venv\Scripts\python.exe" run.py setup %*
if errorlevel 1 pause
exit /b %errorlevel%

:nofolder
echo Can't find the host folder next to this file. Keep setup-windows.bat in the CSIDetector folder.
goto :fail
:nopython
echo Python isn't installed (or isn't on PATH).
echo Install Python 3.11 or newer from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" in the installer, then run this again.
goto :fail
:oldpython
echo Python 3.11 or newer wasn't found. Install it from https://www.python.org/downloads/
echo (if "python" opens the Microsoft Store, that's Windows' placeholder, not Python;
echo install it from python.org and tick "Add python.exe to PATH").
goto :fail
:venvfail
echo Couldn't create the Python environment in host\.venv (see the message above).
goto :fail
:pipfail
echo Installing the Python packages failed (see above). Check the internet
echo connection and run this again.
goto :fail
:fail
pause
exit /b 1
