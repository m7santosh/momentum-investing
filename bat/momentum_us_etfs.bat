@echo off
cd /d "%~dp0.."
python "momentum\momentum_us_etfs.py" %*
pause
