@echo off
REM Double-click this file to start the AI Travel Planner.
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo The virtual environment ".venv" was not found. Follow the setup steps in README.md first.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
streamlit run app.py
pause
