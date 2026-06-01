@echo off
REM US ETF RRG swing backtest (Yahoo vs ^GSPC). Default: core us.py + liquid ADV$ ETFs.
REM Example: backtest_rrg_us.bat --start 2024-01-01 --end 2025-03-31 --top-n 7 --tail 2
REM Example: --pick-strategy leading_only --hold-until-rank-exit --max-hold-rank 15
cd /d "%~dp0..\..\..\"
python "backtest\etf\backtest_rrg_us.py" %*
pause
