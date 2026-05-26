@echo off
REM US ETF RRG vs ^GSPC (Yahoo). Default: 3m. Optional: RRGIndicatorUsEtfs.bat --period 6m
cd /d "%~dp0..\.."
python "momentum\etf\RRGIndicatorUsEtfs.py" %*
pause
