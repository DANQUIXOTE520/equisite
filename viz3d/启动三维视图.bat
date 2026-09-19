@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在启动本地服务器并打开三维视图...
python start_3d.py
pause
