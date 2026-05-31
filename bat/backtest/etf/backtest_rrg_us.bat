@echo off
REM US ETF RRG swing backtest (Yahoo vs ^GSPC). Default: core us.py + liquid ADV$ ETFs.
REM Example: backtest_rrg_us.bat --start 2024-01-01 --end 2025-03-31 --top-n 7 --tail 2
REM Pick strategies: --pick-strategy recommend|leading_improved|top_n|top_n_rank_exit
cd /d "%~dp0..\..\..\"
python "backtest\etf\backtest_rrg_us.py" %*
pause
