@echo off
cd /d "%~dp0..\.."
python "momentum\stock\momentum_stocks.py" %*
pause
