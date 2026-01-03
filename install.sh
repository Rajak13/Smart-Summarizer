#!/bin/bash

# Smart Summarizer - Installation Script
# Automated setup for the Smart Summarizer application

echo "📚 Smart Summarizer - Installation Script"
echo "=========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.8+ and try again."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "🔧 Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo "🔧 Downloading NLTK data..."
python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True)"

# Create necessary directories
echo "🔧 Creating directories..."
mkdir -p data/samples data/results logs

# Run tests
echo "🧪 Running system tests..."
python test_app.py

echo ""
echo "🎉 Installation complete!"
echo ""
echo "To start the application:"
echo "  1. Activate the virtual environment: source .venv/bin/activate"
echo "  2. Run the app: streamlit run app/main.py"
echo "  3. Open your browser to: http://localhost:8501"
echo ""
echo "For Windows users:"
echo "  1. Activate: .venv\\Scripts\\activate"
echo "  2. Run: streamlit run app/main.py"
echo ""
echo "📚 Happy summarizing!"