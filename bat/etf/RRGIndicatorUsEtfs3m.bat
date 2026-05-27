@echo off
REM US ETF RRG vs ^GSPC — fixed 3-month (10w window). Optional: --window 14
cd /d "%~dp0..\.."
python "momentum\etf\RRGIndicatorUsEtfs3m.py" %*
pause
