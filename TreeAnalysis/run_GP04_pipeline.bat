@echo off
setlocal
cd /d "%~dp0"
echo Running TreeAnalysis pipeline for GP04 ...
python run_tree_analysis.py config/GP04.json
echo.
echo Done. Check results folder and run_report.txt
pause
