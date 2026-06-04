@echo off
REM Stock RRG — 3-month tactical analysis (default universe: quality).
cd /d "%~dp0..\..\..\"
python "%CD%\momentum\stock\RRGIndicatorStocks3m.py" %*
exit /b %ERRORLEVEL%
