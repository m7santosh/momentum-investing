@echo off
cd /d "%~dp0..\..\.."
python "backtest\stock\backtest_quality_momentum_rs_lv.py" %*
pause
