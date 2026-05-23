@echo off
cd /d "%~dp0.."
python "momentum\RRGIndicator.py" %*
pause
