@echo off
echo ========================================
echo  Multimodal Agent - Starting...
echo ========================================
echo.

REM Navigate to project root
cd /d "%~dp0.."

REM Pull latest from GitHub (if remote configured)
git remote -v >nul 2>&1
if not errorlevel 1 (
    echo Pulling latest changes from GitHub...
    git pull origin master 2>nul || git pull origin main 2>nul || echo No remote or branch found, skipping pull
    echo.
)

REM Check if venv exists
if not exist ".venv" (
    echo ERROR: Virtual environment not found!
    echo Please run scripts\windows_setup.bat first
    pause
    exit /b 1
)

REM Activate venv
call .venv\Scripts\activate.bat

REM Check Ollama connection (optional)
echo Checking Ollama connection...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    curl -s http://localhost:11435/api/tags >nul 2>&1
    if errorlevel 1 (
        echo WARNING: Ollama not detected on localhost:11434 or :11435
        echo Make sure Ollama is running or SSH tunnel is active
        echo.
    ) else (
        echo Ollama detected on port 11435 (SSH tunnel)
    )
) else (
    echo Ollama detected on port 11434 (local)
)
echo.

REM Run Chainlit app
echo Starting Chainlit app...
echo Access at: http://localhost:8000
echo.
chainlit run app.py --host 0.0.0.0 --port 8000
