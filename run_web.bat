@echo off
chcp 65001 >nul
title TCG-AI Web 控制台

echo ========================================================
echo   TCG-AI 双路集换式卡牌系统 - Web 可视化控制台
echo ========================================================
echo.

cd /d "%~dp0"
echo [1/3] 锁定工作目录: %CD%

:: 1. 检测 Python 解释器
set "PY_CMD="

where py >nul 2>&1
if not errorlevel 1 set "PY_CMD=py"
if defined PY_CMD goto :found_python

where python >nul 2>&1
if not errorlevel 1 set "PY_CMD=python"
if defined PY_CMD goto :found_python

echo.
echo [错误] 系统中未检测到 Python 环境！
echo 请先安装 Python 3.10 或更高版本 (https://www.python.org/downloads/)
echo 安装时请务必勾选 "Add python.exe to PATH"
echo.
goto :exit_pause

:found_python
echo [2/3] 检测到 Python 解释器: %PY_CMD%
%PY_CMD% --version

:: 2. 检测运行依赖库
echo [3/3] 检查 Python 运行依赖库...
%PY_CMD% -c "import torch, numpy, fastapi, uvicorn, websockets, openai, pydantic, requests, matplotlib, dotenv, PIL" >nul 2>&1
if errorlevel 1 goto :install_deps
echo [OK] 依赖库完整齐全，无需重复安装。
goto :check_port

:install_deps
echo.
echo [提示] 检测到缺少部分依赖库，正在自动安装...
echo 正在使用清华镜像源加速下载: -r requirements.txt ...
echo --------------------------------------------------------
%PY_CMD% -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if not errorlevel 1 goto :deps_ok

echo.
echo [提示] 主镜像安装异常，正在尝试备用镜像 (阿里云)...
%PY_CMD% -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
if not errorlevel 1 goto :deps_ok

echo.
echo [错误] 依赖安装失败，请检查网络或手动执行: %PY_CMD% -m pip install -r requirements.txt
goto :exit_pause

:deps_ok
echo.
echo [OK] 依赖环境配置完成！

:check_port
:: 检查 8000 端口占用
netstat -ano | findstr /R /C:":8000 " >nul 2>&1
if errorlevel 1 goto :launch_app

echo.
echo [警告] 端口 8000 当前已被占用！
echo 如果您之前已经启动过 Web 服务，请先关闭原窗口或终止相关进程。
echo 正在尝试启动...

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

echo.
echo [提示] Web 服务已停止运行。

:exit_pause
echo.
echo 按任意键关闭窗口...
pause >nul
