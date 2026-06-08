@echo off
REM US ETF RRG — us.py list (3m). Optional: --screen-only for ADV$ table
cd /d "%~dp0..\.."
python "momentum\etf\RRGIndicatorUsEtfsLiquid3m.py" %*
pause
