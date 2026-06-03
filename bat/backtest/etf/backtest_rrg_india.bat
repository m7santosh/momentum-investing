@echo off
REM Standalone India ETF RRG backtest (not linked to main RRG chart).
REM Console stays open until the backtest window appears, then closes.
REM Example: backtest_rrg_india.bat --start 13-03-2026 --end 01-05-2026 --top-n 7 --tail 1
cd /d "%~dp0..\..\..\"
python "%CD%\momentum\rrg_backtest_standalone.py" --profile india %*
exit /b %ERRORLEVEL%
