@echo off
cd /d "%~dp0..\.."
python "momentum\etf\momentum_etfs.py" %*
pause
