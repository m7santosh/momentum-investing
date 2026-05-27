@echo off
REM NSE indices & ETFs RRG — fixed 3-month (10w window). Optional: --window 14
cd /d "%~dp0..\.."
python "momentum\etf\RRGIndicatorEtfs3m.py" %*
pause
