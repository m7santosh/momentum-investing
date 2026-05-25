@echo off
cd /d "%~dp0..\.."
python "momentum\etf\RRGIndicatorEtfs.py" %*
pause
