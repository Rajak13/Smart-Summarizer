@echo off
REM Smart Summarizer - Windows Installation Script

echo 📚 Smart Summarizer - Installation Script
echo ==========================================

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python 3 is required but not installed.
    echo Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo 🔧 Creating virtual environment...
    python -m venv .venv
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo 🔧 Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 🔧 Installing dependencies...
pip install -r requirements.txt

REM Download NLTK data
echo 🔧 Downloading NLTK data...
python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True)"

REM Create necessary directories
echo 🔧 Creating directories...
if not exist "data\samples" mkdir data\samples
if not exist "data\results" mkdir data\results
if not exist "logs" mkdir logs

REM Run tests
echo 🧪 Running system tests...
python test_app.py

echo.
echo 🎉 Installation complete!
echo.
echo To start the application:
echo   1. Activate the virtual environment: .venv\Scripts\activate
echo   2. Run the app: streamlit run app/main.py
echo   3. Open your browser to: http://localhost:8501
echo.
echo 📚 Happy summarizing!
pause