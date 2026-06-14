@echo off
REM Nifty index candlestick / Heikin Ashi backtest UI
cd /d "%~dp0..\..\.."
python "momentum\nifty_candlestick_backtest_ui.py" %*
pause
