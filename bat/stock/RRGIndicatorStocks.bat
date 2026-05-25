@echo off
REM Stock RRG: 6-month analysis (26 weekly points). Optional universe via STOCK_UNIVERSE or --universe
REM Example: RRGIndicatorStocks.bat --universe nifty_largemidcap
cd /d "%~dp0..\.."
python "momentum\stock\RRGIndicatorStocks.py" %*
pause