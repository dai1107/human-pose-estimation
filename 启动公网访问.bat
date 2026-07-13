@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)

if not exist "tools\cloudflared.exe" (
  echo 缺少 tools\cloudflared.exe，无法创建公网链接。
  echo 请参照《网页版使用说明》安装 Cloudflare Tunnel。
  pause
  exit /b 1
)

echo 正在创建匿名临时公网链接，请稍候……
echo 如需访问口令，请在命令行运行：.venv\Scripts\python.exe start_public_web.py --protected
%PYTHON% start_public_web.py
if errorlevel 1 pause
