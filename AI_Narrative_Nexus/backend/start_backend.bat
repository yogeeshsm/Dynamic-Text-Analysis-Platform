@echo off
echo ================================================
echo AI Narrative Nexus - Backend Server Startup
echo ================================================
echo.

REM Check if virtual environment exists
if not exist "..\venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup first:
    echo   1. cd ..
    echo   2. python -m venv venv
    echo   3. .\venv\Scripts\Activate.ps1
    echo   4. pip install -r backend\requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
call ..\venv\Scripts\activate.bat

REM Set Flask environment variables
set FLASK_APP=app.py
set FLASK_ENV=development

echo Starting Flask API server...
echo Backend will be available at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start Flask server
python app.py

pause
