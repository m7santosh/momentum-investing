@echo off
REM US ETF RRG — core us.py + ADV$ discoveries (3m). Optional: --screen-only --min-adv 5000000
cd /d "%~dp0..\.."
python "momentum\etf\RRGIndicatorUsEtfsLiquid3m.py" %*
pause
