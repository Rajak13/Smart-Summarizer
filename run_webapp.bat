@echo off
REM Smart Summarizer - Quick Start Script (Windows)

echo ==================================
echo Smart Summarizer Web Application
echo ==================================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo Virtual environment not found. Please run install.bat first.
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check if dependencies are installed
echo Checking dependencies...
python -c "import flask" 2>nul
if errorlevel 1 (
    echo Flask not found. Installing dependencies...
    pip install -r requirements.txt
)

REM Start the Flask application
echo.
echo Starting Flask server...
echo Application will be available at: http://localhost:5001
echo.
echo Press Ctrl+C to stop the server
echo.

cd webapp
python app.py
