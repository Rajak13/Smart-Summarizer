#!/bin/bash

# Smart Summarizer - Quick Start Script

echo "=================================="
echo "Smart Summarizer Web Application"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Please run install.sh first."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if dependencies are installed
echo "Checking dependencies..."
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Flask not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Start the Flask application
echo ""
echo "Starting Flask server..."
echo "Application will be available at: http://localhost:5001"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd webapp
python app.py
