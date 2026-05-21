@echo off
REM Optional: set QUALITY_RS_RUN_AS_OF=2026-04-01 and QUALITY_RS_REBALANCE=monthly
REM Or pass args: quality_momentum_rs_lv.bat --as-of 2026-04-01 --rebalance weekly
cd /d "%~dp0..\.."
python "momentum\stock\quality_momentum_rs_lv.py" %*
pause
