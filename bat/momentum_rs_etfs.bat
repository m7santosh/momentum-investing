@echo off
cd /d "%~dp0.."
python "momentum\momentum_rs_etfs.py" %*
pause
