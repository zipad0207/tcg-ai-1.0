@echo off
title TCG-AI Web Console

echo ========================================================
echo   TCG-AI Dual-Card System - Web Console
echo ========================================================
echo.

cd /d "%~dp0"
echo [1/4] Working Directory: %CD%

:: 1. Check existing virtual environment first
if exist "venv\Scripts\python.exe" (
    set "PY_CMD=%CD%\venv\Scripts\python.exe"
    echo [2/4] Found local project virtualenv: %CD%\venv
    goto :check_deps
)

:: 2. Search for system Python >= 3.10
echo [2/4] Detecting Python (requires Python 3.10+)...
set "PY_CMD="

:: Candidate 1: py -3.11
py -3.11 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=py -3.11"
    goto :found_python
)

:: Candidate 2: py -3.12
py -3.12 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=py -3.12"
    goto :found_python
)

:: Candidate 3: py -3.10
py -3.10 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=py -3.10"
    goto :found_python
)

:: Candidate 4: LocalAppData Python 3.11
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD="%LOCALAPPDATA%\Programs\Python\Python311\python.exe""
        goto :found_python
    )
)

:: Candidate 5: LocalAppData Python 3.12
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD="%LOCALAPPDATA%\Programs\Python\Python312\python.exe""
        goto :found_python
    )
)

:: Candidate 6: default 'python' in PATH (ensure >= 3.10)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=python"
    goto :found_python
)

:: Candidate 7: default 'py' launcher (ensure >= 3.10)
py -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=py"
    goto :found_python
)

:: 3. No valid Python >= 3.10 found: AUTO-INSTALL Python 3.11
echo.
echo ========================================================
echo   [!] No Python 3.10+ found on this computer.
echo   [!] Starting automatic download and installation...
echo ========================================================
echo.

set "PY_INSTALLER=%TEMP%\python-3.11.9-installer.exe"
if exist "%PY_INSTALLER%" del /f /q "%PY_INSTALLER%" >nul 2>&1

echo [INFO] Downloading Python 3.11.9 installer...
curl.exe -k -L -f --progress-bar -o "%PY_INSTALLER%" "https://repo.huaweicloud.com/python/3.11.9/python-3.11.9-amd64.exe"
if not exist "%PY_INSTALLER%" (
    echo [INFO] Mirror download failed, trying official python.org...
    curl.exe -k -L -f --progress-bar -o "%PY_INSTALLER%" "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
)
if not exist "%PY_INSTALLER%" (
    echo [INFO] curl failed, trying PowerShell download...
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object Net.WebClient).DownloadFile('https://repo.huaweicloud.com/python/3.11.9/python-3.11.9-amd64.exe', '$env:TEMP\python-3.11.9-installer.exe')"
)

if not exist "%PY_INSTALLER%" (
    echo.
    echo [ERROR] Failed to download Python installer automatically.
    echo Please install Python 3.11 manually from https://www.python.org/downloads/
    goto :exit_pause
)

echo.
echo [INFO] Installing Python 3.11 (User mode, no administrator privileges needed)...
start /wait "" "%PY_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0 Include_pip=1 Include_launcher=1 SimpleInstall=1
del /f /q "%PY_INSTALLER%" >nul 2>&1

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PY_CMD="%LOCALAPPDATA%\Programs\Python\Python311\python.exe""
    echo [OK] Python 3.11 installed successfully!
    goto :found_python
)

echo.
echo [ERROR] Python installation did not complete as expected.
echo Please run installer manually or check https://www.python.org/downloads/
goto :exit_pause

:found_python
echo [2/4] Selected Python: %PY_CMD%
%PY_CMD% --version

:check_deps
:: 4. Check Dependencies
echo [3/4] Checking Python libraries...
%PY_CMD% -c "import torch, numpy, fastapi, uvicorn, websockets, openai, pydantic, requests, matplotlib, dotenv, PIL" >nul 2>&1
if not errorlevel 1 (
    echo [OK] All dependencies are ready.
    goto :check_port
)

:: Need installation: create isolated project venv to prevent permission errors
echo.
echo [INFO] Missing dependencies. Preparing local virtual environment...
if not exist "venv\Scripts\python.exe" (
    echo [INFO] Creating isolated virtual environment in .\venv ...
    %PY_CMD% -m venv venv
    if exist "venv\Scripts\python.exe" (
        set "PY_CMD=%CD%\venv\Scripts\python.exe"
        echo [OK] Virtual environment created: %CD%\venv
    )
) else (
    set "PY_CMD=%CD%\venv\Scripts\python.exe"
)

echo.
echo [INFO] Upgrading pip...
%PY_CMD% -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple >nul 2>&1

echo [INFO] Installing dependencies from requirements.txt...
%PY_CMD% -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if not errorlevel 1 goto :deps_ok

echo.
echo [INFO] Primary mirror failed, trying Aliyun mirror...
%PY_CMD% -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
if not errorlevel 1 goto :deps_ok

echo.
echo [ERROR] Dependency installation failed!
goto :exit_pause

:deps_ok
echo [OK] Dependencies installed successfully!

:check_port
echo [4/4] Checking port 8000...
netstat -ano | findstr /R /C:":8000 " >nul 2>&1
if errorlevel 1 goto :launch_app

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
