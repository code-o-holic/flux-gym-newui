@echo off
REM Set UTF-8 encoding for Python
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=utf-8
chcp 65001 >nul

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run the UI
python ui_new\app.py

pause
