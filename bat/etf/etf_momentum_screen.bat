@echo off
REM India NSE ETF momentum rankers — on-screen viewer (Abs / RS Blended / RS Adaptive)
cd /d "%~dp0..\.."
python "momentum\etf\etf_momentum_screen.py" %*
pause
