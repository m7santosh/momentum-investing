@echo off
REM India NSE ETF momentum rankers — on-screen viewer (Abs / RS Blended / RS Adaptive)
cd /d "%~dp0..\.."
python -m pip install --upgrade pip
python -m pip install dotenv matplotlib pandas pyotp requests rich scipy yfinance
python "momentum\etf\etf_momentum_screen.py" %*
pause
