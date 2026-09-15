@echo off
setlocal
cd /d "%~dp0"

set "VOICE_TEXT_PYTHON=G:\anaconda\envs\utools\python.exe"
set "VOICE_TEXT_LOG=%~dp0startup.log"
echo [%date% %time%] Starting voice input... > "%VOICE_TEXT_LOG%"

if exist "%VOICE_TEXT_PYTHON%" (
    "%VOICE_TEXT_PYTHON%" main.py >> "%VOICE_TEXT_LOG%" 2>&1
) else (
    python main.py >> "%VOICE_TEXT_LOG%" 2>&1
)

echo [%date% %time%] Process exited with code %errorlevel%. >> "%VOICE_TEXT_LOG%"
