@echo off
REM Standalone US ETF RRG backtest (Yahoo vs ^GSPC; not linked to main RRG chart).
REM Console stays open until the backtest window appears, then closes.
REM Example: backtest_rrg_us.bat --start 2024-01-01 --end 2025-03-31 --top-n 7 --tail 2
REM CLI (no window): python backtest\etf\backtest_rrg_us.py --start ... --end ...
cd /d "%~dp0..\..\..\"
python "%CD%\momentum\rrg_backtest_standalone.py" --profile us %*
exit /b %ERRORLEVEL%
