@echo off
REM Repo root = parent of this bat folder. Do not use ..\.. here (that leaves the repo).
REM Close scripts\_tickers_fragment.txt in other apps if you see a file-lock error.

cd /d "%~dp0.."
python "scripts\gen_tickers_from_csvs.py" --source Nifty50 "%USERPROFILE%\Downloads\ind_nifty50list.csv" --source Next50 "%USERPROFILE%\Downloads\ind_niftynext50list.csv" --source Midcap "%USERPROFILE%\Downloads\ind_niftymidcap150list (1).csv" --source Smallcap "%USERPROFILE%\Downloads\ind_niftysmallcap250list (2).csv" -o scripts\tickers_list.txt
pause
