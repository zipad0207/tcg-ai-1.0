@echo off
title TCG-AI Web Console

echo ========================================================
echo   TCG-AI Dual-Card System - Web Console
echo ========================================================
echo.

cd /d "%~dp0"
echo [1/3] Working Directory: %CD%

:: 1. Detect Python Interpreter
set "PY_CMD="

where py >nul 2>&1
if not errorlevel 1 set "PY_CMD=py"
if defined PY_CMD goto :found_python

where python >nul 2>&1
if not errorlevel 1 set "PY_CMD=python"
if defined PY_CMD goto :found_python

echo.
echo [ERROR] Python not found in system PATH!
echo Please install Python 3.10+ (https://www.python.org/downloads/)
echo Make sure to check "Add python.exe to PATH" during installation.
echo.
goto :exit_pause

:found_python
echo [2/3] Python detected: %PY_CMD%
%PY_CMD% --version

:: 2. Check Dependencies
echo [3/3] Checking dependencies...
%PY_CMD% -c "import torch, numpy, fastapi, uvicorn, websockets, openai, pydantic, requests, matplotlib, dotenv, PIL" >nul 2>&1
if errorlevel 1 goto :install_deps
echo [OK] All dependencies ready.
goto :check_port

:install_deps
echo.
echo [INFO] Missing dependencies, installing from requirements.txt...
%PY_CMD% -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if not errorlevel 1 goto :deps_ok

echo.
echo [INFO] Tsinghua mirror failed, trying Aliyun mirror...
%PY_CMD% -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
if not errorlevel 1 goto :deps_ok

echo.
echo [ERROR] Dependency installation failed!
goto :exit_pause

:deps_ok
echo [OK] Dependencies installed successfully!

:check_port
netstat -ano | findstr /R /C:":8000 " >nul 2>&1
if errorlevel 1 goto :launch_app

echo.
echo [WARNING] Port 8000 is already in use!

:launch_app
echo.
echo ========================================================
echo   Launching Web Service: http://127.0.0.1:8000
echo   Press Ctrl + C in this window to stop the server.
echo ========================================================
echo.

start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000"

%PY_CMD% -m uvicorn web_app.main:app --host 127.0.0.1 --port 8000 --reload

echo.
echo [INFO] Server stopped.

:exit_pause
echo.
pause
