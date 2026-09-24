@echo off
cd /d "%~dp0"
title TCG-AI Web Console

echo ========================================================
echo   TCG-AI Dual-Card System - Web Console
echo ========================================================
echo.

:: 1. Check portable environment first
if exist "python_env\python.exe" (
    set "PY_CMD=python_env\python.exe"
    echo [1/3] Using portable Python environment: %CD%\python_env
    goto :check_port
)

:: 2. Fallback to system Python if portable package is not found
echo [1/3] Portable environment not found. Checking system Python...
set "PY_CMD="

where py >nul 2>&1
if not errorlevel 1 set "PY_CMD=py"
if defined PY_CMD goto :found_sys_python

where python >nul 2>&1
if not errorlevel 1 set "PY_CMD=python"
if defined PY_CMD goto :found_sys_python

echo.
echo [ERROR] Neither portable python_env nor system Python was found!
echo Please make sure python_env is unzipped or install Python 3.10+.
echo.
pause
exit /b 1

:found_sys_python
echo [1/3] Using system Python: %PY_CMD%

:check_port
echo [2/3] Checking port 8000...
netstat -ano | findstr /R /C:":8000 " >nul 2>&1
if errorlevel 1 goto :launch_app

echo [WARNING] Port 8000 is already in use!

:launch_app
echo [3/3] Launching Web Service: http://127.0.0.1:8000
echo   Press Ctrl + C in this window to stop the server.
echo.

start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:8000"

%PY_CMD% -m uvicorn web_app.main:app --host 127.0.0.1 --port 8000 --reload

echo.
echo [INFO] Server stopped.
pause
