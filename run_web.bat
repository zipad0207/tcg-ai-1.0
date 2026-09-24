@echo off
chcp 65001 >nul
title TCG-AI Web 控制台 (一键启动)

echo ========================================================
echo   TCG-AI 双路集换式卡牌系统 - Web 可视化控制台
echo ========================================================
echo.

cd /d "%~dp0"
echo [1/3] 锁定工作目录: %CD%

:: 1. 检测 Python 解释器
set "PY_CMD="
where py >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=py"
    goto :python_found
)

where python >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=python"
    goto :python_found
)

:python_not_found
echo.
echo [错误] 系统中未检测到 Python 环境！
echo 请先安装 Python 3.10 或更高版本：
echo 1. 访问官网下载: https://www.python.org/downloads/
echo 2. 安装时请务必勾选: "Add python.exe to PATH"
echo.
pause
exit /b 1

:python_found
for /f "tokens=*" %%i in ('%PY_CMD% --version 2^>^&1') do set "PY_VER=%%i"
echo [2/3] 检测到 Python 解释器: %PY_CMD% (%PY_VER%)

:: 2. 检测运行依赖库
echo [3/3] 检查 Python 运行依赖库...
%PY_CMD% -c "import torch, numpy, fastapi, uvicorn, websockets, openai, pydantic, requests, matplotlib, dotenv, PIL" >nul 2>&1
if errorlevel 1 goto :install_deps
echo [OK] 依赖库完整齐全，无需重复安装。
goto :launch_app

:install_deps
echo.
echo [提示] 检测到缺少部分必要依赖库，正在自动为您安装...
echo 正在使用清华镜像源加速下载: -r requirements.txt ...
echo --------------------------------------------------------
%PY_CMD% -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if not errorlevel 1 goto :deps_installed

echo.
echo [提示] 主镜像安装异常，正在尝试备用镜像 (阿里云)...
%PY_CMD% -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
if errorlevel 1 goto :deps_failed

:deps_installed
echo.
echo [OK] 依赖环境安装配置完成！
goto :launch_app

:deps_failed
echo.
echo [错误] 依赖安装失败，请检查网络或手动执行: %PY_CMD% -m pip install -r requirements.txt
echo.
pause
exit /b 1

:launch_app
echo.
echo ========================================================
echo   正在启动 Web 服务: http://127.0.0.1:8000
echo   [提示] 服务就绪后，将自动在浏览器中打开前端界面
echo   [提示] 在此窗口按 Ctrl + C 可终止 Web 服务
echo ========================================================
echo.

start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000"

%PY_CMD% -m uvicorn web_app.main:app --host 127.0.0.1 --port 8000 --reload
if errorlevel 1 (
    echo.
    echo [警告] Web 服务退出
    pause
)
