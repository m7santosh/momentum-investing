@echo off
REM Standalone stock RRG backtest (not linked to main RRG chart).
REM Example: backtest_rrg_stocks.bat --universe quality --start 13-03-2026 --end 01-05-2026
cd /d "%~dp0..\..\..\"
python "%CD%\momentum\rrg_backtest_standalone.py" --profile stock %*
exit /b %ERRORLEVEL%
