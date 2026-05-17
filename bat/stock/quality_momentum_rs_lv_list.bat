@echo off
REM Ranked list only — no rebalance / portfolio state
REM Optional: --as-of 2026-05-15  --limit 30
cd /d "%~dp0..\.."
python "momentum\stock\quality_momentum_rs_lv_list.py" %*
pause
