@echo off
REM Staggered dip-buying backtest UI (N lots, X%% profit, Y%% dip)
cd /d "%~dp0..\..\.."
python "momentum\staggered_dip_backtest_ui.py" %*
pause
