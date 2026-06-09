@echo off
REM India ETF momentum backtest UI (Abs / RS Blended / RS Adaptive)
cd /d "%~dp0..\..\.."
python "momentum\etf_momentum_backtest_ui.py" %*
pause
