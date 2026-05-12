@echo off
cd /d "%~dp0..\.."
python "momentum/etf/momentum_us_etfs.py" %*
pause
