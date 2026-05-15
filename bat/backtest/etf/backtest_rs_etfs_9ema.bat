@echo off
cd /d "%~dp0..\..\..\"
python "backtest/etf/backtest_momentum_rs_etfs_9ema.py" %*
pause
