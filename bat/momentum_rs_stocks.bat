@echo off
cd /d "%~dp0.."
python "momentum\momentum_rs_stocks.py" %*
pause
