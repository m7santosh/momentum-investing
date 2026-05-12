@echo off
cd /d "%~dp0..\.."
python "momentum\stock\momentum_rs_stocks.py" %*
pause
