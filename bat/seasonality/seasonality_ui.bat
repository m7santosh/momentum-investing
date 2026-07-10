@echo off
REM Launch the seasonality GUI directly.
setlocal
cd /d "%~dp0\..\.."
python -m seasonality.seasonality_ui
