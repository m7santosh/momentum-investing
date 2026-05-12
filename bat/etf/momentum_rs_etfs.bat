@echo off
cd /d "%~dp0..\.."
python "momentum\etf\momentum_rs_etfs.py" %*
pause
