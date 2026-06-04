@echo off
REM Stock RRG: default 3-month tactical analysis. Optional universe via STOCK_UNIVERSE or --universe
REM Examples:
REM   RRGIndicatorStocks.bat
REM   RRGIndicatorStocks.bat --universe quality --period 6m
cd /d "%~dp0..\.."
python "momentum\stock\RRGIndicatorStocks.py" %*
pause