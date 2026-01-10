@echo off
echo ================================================
echo AI Narrative Nexus - Frontend Server Startup
echo ================================================
echo.

REM Check if node_modules exists
if not exist "node_modules" (
    echo Node modules not found. Installing dependencies...
    call npm install
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install dependencies
        echo Please make sure Node.js and npm are installed
        pause
        exit /b 1
    )
)

echo Starting Vite development server...
echo Frontend will be available at: http://localhost:3000
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start Vite dev server
call npm run dev

pause
