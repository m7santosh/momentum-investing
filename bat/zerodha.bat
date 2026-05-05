@echo off
cd /d "%~dp0.."
python "client\zerodha.py" %*
pause
