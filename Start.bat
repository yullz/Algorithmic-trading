@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  AlgoTrader Launcher
echo ============================================
echo.

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

REM --- Check virtual environment ---
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo Please run Setup.bat first.
    pause
    exit /b 1
)

REM --- Check / build dashboard if npm is available ---
if not exist "web\dist\index.html" (
    echo [*] Dashboard bundle not found.
    npm --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] npm is not available and web/dist is missing.
        echo Please run Setup.bat first, or install Node.js.
        pause
        exit /b 1
    )
    echo [*] Building dashboard ...
    cd web
    call npm install
    if errorlevel 1 (
        echo [ERROR] npm install failed.
        cd ..
        pause
        exit /b 1
    )
    call npm run build
    if errorlevel 1 (
        echo [ERROR] Dashboard build failed.
        cd ..
        pause
        exit /b 1
    )
    cd ..
    echo [OK] Dashboard built.
)

REM --- Start the server ---
echo [*] Starting AlgoTrader server on http://127.0.0.1:8777 ...
echo [*] Press Ctrl+C here to stop.
echo.

REM Open browser after a short delay
start /b cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:8777"

.venv\Scripts\python server\main.py

if errorlevel 1 (
    echo.
    echo [ERROR] Server exited with an error.
    pause
    exit /b 1
)

pause
