@echo off
cd /d "%~dp0.."
python "momentum\quality_momentum_rs.py" %*
pause
