@echo off
REM US ETF momentum rankers — on-screen viewer (Abs / RS Blended / RS Adaptive)
cd /d "%~dp0..\.."
python "momentum\etf\etf_us_momentum_screen.py" %*
pause
