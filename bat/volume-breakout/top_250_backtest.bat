@echo off
REM Volume Breakout top-250 backtest (Tkinter UI)
cd /d "%~dp0..\.."
python "volume-breakout\top_250_backtest_ui.py" %*
pause
