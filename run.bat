@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "VOICE_TEXT_LOG=%~dp0startup.log"
echo [%date% %time%] Starting voice input... > "%VOICE_TEXT_LOG%"
echo [%date% %time%] Resolving Python interpreter... >> "%VOICE_TEXT_LOG%"

set "PYTHON_EXE="

rem ==== 1. Project-local virtualenv has the highest priority ====
call :TRY_INTERPRETER "%~dp0.venv\Scripts\python.exe"

rem ==== 2. An environment that is already activated ====
if not defined PYTHON_EXE if defined CONDA_PREFIX call :TRY_INTERPRETER "%CONDA_PREFIX%\python.exe"
if not defined PYTHON_EXE if defined VIRTUAL_ENV call :TRY_INTERPRETER "%VIRTUAL_ENV%\Scripts\python.exe"

rem ==== 3. Any conda env on this machine that already has deps installed ====
if not defined PYTHON_EXE call :SCAN_CONDA_ENVS

rem ==== 4. Any python on PATH that already has deps installed ====
if not defined PYTHON_EXE call :SCAN_PATH_INTERPRETERS

rem ==== 5. Fall back to the first python available, even if deps are missing ====
if not defined PYTHON_EXE (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        if not defined PYTHON_EXE set "PYTHON_EXE=%%P"
    )
)

if not defined PYTHON_EXE goto :NO_INTERPRETER

echo [%date% %time%] Interpreter: !PYTHON_EXE! >> "%VOICE_TEXT_LOG%"
echo Interpreter: !PYTHON_EXE!

"!PYTHON_EXE!" -c "import keyboard, sounddevice, numpy, pyperclip, PIL, pystray, requests" >nul 2>&1
if errorlevel 1 (
    echo [%date% %time%] WARN: dependencies missing in this interpreter. >> "%VOICE_TEXT_LOG%"
    echo.
    echo   WARNING: required dependencies are missing in this interpreter.
    echo            run:  pip install -r requirements.txt
    echo            See README.md for details.
    echo.
)

"!PYTHON_EXE!" main.py >> "%VOICE_TEXT_LOG%" 2>&1
echo [%date% %time%] Process exited with code %errorlevel%. >> "%VOICE_TEXT_LOG%"
endlocal
goto :EOF


rem ===== Queued interpreter if it exists AND can import runtime deps =====
:TRY_INTERPRETER
if "%~1"=="" goto :EOF
if not exist "%~1" goto :EOF
"%~1" -c "import keyboard, sounddevice, numpy, pyperclip, PIL, pystray, requests" >nul 2>&1
if not errorlevel 1 set "PYTHON_EXE=%~1"
goto :EOF

rem ===== Locate conda installation root(s), then walk envs/ =====
:SCAN_CONDA_ENVS
for /f "delims=" %%C in ('where conda 2^>nul') do (
    if not defined PYTHON_EXE (
        set "CONDA_BIN=%%~dpC"
        rem %%~dpC ends with a backslash; strip it, then take its parent => conda root
        for %%Q in ("!CONDA_BIN:~0,-1!") do set "CONDA_ROOT=%%~dpQ"
        if exist "!CONDA_ROOT!envs" (
            for /d %%E in ("!CONDA_ROOT!envs\*") do (
                call :TRY_INTERPRETER "%%~E\python.exe"
            )
        )
    )
)
goto :EOF

rem ===== Same check for every python found on PATH =====
:SCAN_PATH_INTERPRETERS
for /f "delims=" %%P in ('where python 2^>nul') do (
    call :TRY_INTERPRETER "%%P"
)
goto :EOF

:NO_INTERPRETER
echo [%date% %time%] ERROR: no Python interpreter found. >> "%VOICE_TEXT_LOG%"
echo.
echo   ERROR: no usable Python interpreter found.
echo          Install Python 3.9+ and add it to PATH, then run this script again.
echo          See README.md for details.
echo.
pause
endlocal
exit /b 1
