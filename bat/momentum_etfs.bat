@echo off
cd /d "%~dp0.."
python "momentum\momentum_etfs.py" %*
pause
