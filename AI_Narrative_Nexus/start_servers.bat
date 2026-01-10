@echo off
echo Starting AI Narrative Nexus Servers...
echo.

REM Start Backend
start "Backend Server" cmd /k "cd /d "%~dp0" && .\venv\Scripts\activate && cd backend && python app.py"

REM Wait a bit for backend to start
timeout /t 5 /nobreak

REM Start Frontend  
start "Frontend Server" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo Both servers are starting in separate windows...
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
echo.
pause
