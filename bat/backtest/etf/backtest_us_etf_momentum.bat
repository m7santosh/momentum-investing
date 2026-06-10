@echo off
REM US ETF momentum backtest UI (Abs / RS Blended / RS Adaptive)
cd /d "%~dp0..\..\.."
python "momentum\etf_us_momentum_backtest_ui.py" %*
pause
