@echo off
echo ========================================
echo  Multimodal Agent - Windows Setup
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.10+
    exit /b 1
)

REM Check ffmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo WARNING: ffmpeg not found in PATH
    echo Please install ffmpeg and add to PATH, or place at C:\ffmpeg\bin\
    echo Download: https://www.gyan.dev/ffmpeg/builds/
    echo.
)

REM Navigate to project root
cd /d "%~dp0.."

REM Create venv if not exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate and install
echo Installing dependencies...
call .venv\Scripts\activate.bat
pip install -e ".[dev]"

echo.
echo ========================================
echo  Setup complete!
echo ========================================
echo.
echo To run the app: scripts\windows_run.bat
echo.
pause
