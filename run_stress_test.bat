@echo off
cd /d "%~dp0"
title TCG-AI Stress Test Launcher

:: 1. Search for Python environment
set "PY_CMD="
if exist "python_env\python.exe" (
    set "PY_CMD=python_env\python.exe"
    goto :run_test
)

where py >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=py"
    goto :run_test
)

where python >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=python"
    goto :run_test
)

echo [ERROR] No Python environment found!
pause
exit /b 1

:run_test
%PY_CMD% stress_test_balancer.py %*

if errorlevel 1 (
    echo.
    echo [INFO] Process completed with non-zero exit code.
)
pause
