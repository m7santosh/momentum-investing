@echo off
REM Top 250 NSE stocks by volume / turnover (Tkinter viewer)
cd /d "%~dp0..\.."
python "volume-breakout\top_250_viewer.py" %*
pause
