@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  AlgoTrader Setup
echo ============================================
echo.

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

REM --- Check Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=2 delims=. " %%a in ('python --version 2^>^&1') do set "PY_MAJOR=%%a"
for /f "tokens=3 delims=. " %%a in ('python --version 2^>^&1') do set "PY_MINOR=%%a"

if "%PY_MAJOR%"=="" set PY_MAJOR=3
if "%PY_MINOR%"=="" set PY_MINOR=0

if %PY_MAJOR% LSS 3 (
    echo [ERROR] Python 3.10+ is required. Found %PY_MAJOR%.%PY_MINOR%
    pause
    exit /b 1
)
if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 10 (
    echo [ERROR] Python 3.10+ is required. Found 3.%PY_MINOR%
    pause
    exit /b 1
)

echo [OK] Python %PY_MAJOR%.%PY_MINOR% found.

REM --- Check Node.js / npm ---
npm --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Node.js/npm is not in PATH.
    echo The Python backend will work, but the dashboard cannot be built.
    echo Install Node.js 18+ from https://nodejs.org/ if you want the UI.
    echo.
    set "HAS_NPM=0"
) else (
    echo [OK] npm found.
    set "HAS_NPM=1"
)

REM --- Create virtual environment ---
if not exist ".venv\Scripts\python.exe" (
    echo [*] Creating Python virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [OK] Virtual environment already exists.
)

REM --- Install Python dependencies ---
echo [*] Installing Python packages from requirements.txt ...
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed.
    pause
    exit /b 1
)

echo [OK] Python dependencies installed.

REM --- Create .env if missing ---
if not exist ".env" (
    echo [*] Creating .env from .env.example ...
    copy .env.example .env >nul
    echo [OK] .env created. Edit it if you add API keys later.
) else (
    echo [OK] .env already exists.
)

REM --- Install and build dashboard ---
if "%HAS_NPM%"=="1" (
    if not exist "web\package.json" (
        echo [WARNING] web/package.json not found. Skipping dashboard build.
    ) else (
        echo [*] Installing dashboard Node packages ...
        cd web
        call npm install
        if errorlevel 1 (
            echo [ERROR] npm install failed.
            cd ..
            pause
            exit /b 1
        )
        echo [*] Building dashboard production bundle ...
        call npm run build
        if errorlevel 1 (
            echo [ERROR] dashboard build failed.
            cd ..
            pause
            exit /b 1
        )
        cd ..
        echo [OK] Dashboard built.
    )
) else (
    echo [WARNING] Skipping dashboard build because npm is not available.
)

REM --- Run tests ---
echo [*] Running test suite ...
.venv\Scripts\python -m pytest tests -q
if errorlevel 1 (
    echo [WARNING] Some tests failed. Check the output above.
) else (
    echo [OK] All tests passed.
)

echo.
echo ============================================
echo  Setup complete.
echo ============================================
echo.
echo Next steps:
echo   - Run Start.bat to launch the dashboard.
echo   - Or run: python run_scan.py --offline
echo   - Or run: python paper_trade.py --offline
echo.
pause
