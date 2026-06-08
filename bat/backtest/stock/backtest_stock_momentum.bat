@echo off
cd /d "%~dp0..\..\.."
python "momentum\stock_momentum_backtest_ui.py" %*
pause
