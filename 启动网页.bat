@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)

echo 正在启动本机网页版：http://127.0.0.1:5000
echo 浏览器会在服务就绪后自动打开。关闭此窗口即可停止服务。
%PYTHON% start_web.py
if errorlevel 1 pause
