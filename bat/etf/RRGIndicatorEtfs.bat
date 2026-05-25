@echo off
REM Default: 3-month ETF RRG (tactical). Optional: RRGIndicatorEtfs.bat --period 6m
cd /d "%~dp0..\.."
python "momentum\etf\RRGIndicatorEtfs.py" %*
pause
