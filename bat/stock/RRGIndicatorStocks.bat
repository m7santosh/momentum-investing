@echo off
REM Optional: set STOCK_UNIVERSE=quality, n500, bse_largemidcap, or nifty_largemidcap
REM Or pass: RRGIndicatorStocks.bat --universe nifty_largemidcap
cd /d "%~dp0..\.."
python "momentum\stock\RRGIndicatorStocks.py" %*
pause