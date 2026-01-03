# 📚 Smart Summarizer

**AI-Powered Text Summarization with Multiple Model Comparison**

A professional web application that compares three state-of-the-art AI models: TextRank (extractive), BART (abstractive), and PEGASUS (abstractive). Built with Flask and featuring a modern, responsive UI.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

Smart Summarizer addresses the challenge of information overload by providing automated, high-quality text summarization. The application allows users to:

- Generate summaries using three different AI approaches
- Compare model performance side-by-side
- Process multiple documents in batches
- Evaluate summaries using ROUGE metrics
- Upload files (.txt, .md, .pdf, .docx)

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Rajak13/Smart-Summarizer.git
cd Smart-Summarizer

# Install dependencies
./install.sh  # or install.bat on Windows

# Run the application
./run_webapp.sh  # or run_webapp.bat on Windows

# Access at http://localhost:5001
```

For detailed instructions, see [QUICK_START.md](QUICK_START.md)

## 🌐 Deployment

Deploy to the cloud in minutes:

```bash
# Quick deploy script
./deploy.sh

# Or manually:
# Railway (Recommended)
railway up

# Render
# Push to GitHub and connect via dashboard

# Heroku
heroku create && git push heroku main

# Docker
docker build -t smart-summarizer . && docker run -p 5001:5001 smart-summarizer
```

For comprehensive deployment guide, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

## 🤖 Available Models

### 1. TextRank (Extractive)
- **Type**: Graph-based extractive summarization
- **Speed**: Very fast (~0.03 seconds)
- **Best for**: Quick summaries, keyword extraction

### 2. BART (Abstractive)
- **Model**: facebook/bart-large-cnn
- **Speed**: Moderate (~9 seconds on CPU)
- **Best for**: Fluent, human-like summaries

### 3. PEGASUS (Abstractive)
- **Model**: google/pegasus-cnn_dailymail
- **Speed**: Moderate (~6 seconds on CPU)
- **Best for**: High-quality abstractive summaries

## ✨ Features

- **🏠 Home**: Overview and model descriptions
- **📄 Single Summary**: Summarize individual documents
- **⚖️ Comparison**: Compare all three models side-by-side
- **📚 Batch Processing**: Process multiple documents simultaneously
- **📊 Evaluation**: ROUGE metrics visualization with Chart.js
- **📁 File Upload**: Support for .txt, .md, .pdf, .docx files
- **💾 Export**: Download batch results as CSV

## 📋 Requirements

- Python 3.9+
- 8GB+ RAM (16GB recommended for BART/PEGASUS)
- Internet connection (for downloading pre-trained models)

## 🛠️ Installation

### Automatic Installation

```bash
# macOS/Linux
./install.sh

# Windows
install.bat
```

### Manual Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

## 🎮 Usage

### Web Application

```bash
# Start the server
./run_webapp.sh  # or run_webapp.bat

# Access at http://localhost:5001
```

### Python API

```python
from models.textrank import TextRankSummarizer
from models.bart import BARTSummarizer
from models.pegasus import PEGASUSSummarizer

# Initialize models
textrank = TextRankSummarizer()
bart = BARTSummarizer(device='cpu')
pegasus = PEGASUSSummarizer(device='cpu')

# Generate summaries
text = "Your long text here..."
summary = textrank.summarize(text)
print(summary)
```

## 📊 Performance Benchmarks

Based on CNN/DailyMail dataset evaluation:

| Model    | ROUGE-1 | ROUGE-2 | ROUGE-L | Speed    | Memory |
|----------|---------|---------|---------|----------|--------|
| TextRank | 0.43    | 0.18    | 0.35    | Very Fast| Low    |
| BART     | 0.51    | 0.34    | 0.48    | Moderate | High   |
| PEGASUS  | 0.55    | 0.30    | 0.52    | Moderate | High   |

## 🏗️ Project Structure

```
smart-summarizer/
├── webapp/                 # Flask web application
│   ├── app.py             # Main Flask app
│   ├── templates/         # HTML templates
│   └── static/            # CSS, JS, assets
├── models/                # Summarization models
│   ├── textrank.py
│   ├── bart.py
│   └── pegasus.py
├── utils/                 # Utility functions
├── data/                  # Data files
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── Dockerfile             # Docker configuration
├── Procfile              # Heroku configuration
├── railway.json          # Railway configuration
├── render.yaml           # Render configuration
└── requirements.txt      # Python dependencies
```

## 🧪 Testing

```bash
# Test all routes
python test_webapp.py

# Run unit tests
pytest tests/

# Run with coverage
pytest tests/ --cov=models --cov=utils
```

## 🎨 Design

**Color Palette (Ink Wash):**
- Charcoal: #4A4A4A
- Cool Gray: #CBCBCB
- Soft Ivory: #FFFFE3
- Slate Blue: #6D8196

**Features:**
- Responsive design
- Font Awesome icons
- Professional typography
- Smooth animations

## 🚢 Deployment Options

### Recommended Platforms

1. **Railway** - Best for ML apps, generous free tier
2. **Render** - Free tier with good performance
3. **Heroku** - Easy deployment, limited free tier
4. **Docker** - Full control, deploy anywhere
5. **AWS EC2** - Production-grade hosting

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Hugging Face** for pre-trained BART and PEGASUS models
- **NetworkX** for graph algorithms in TextRank
- **Flask** for the web application framework
- **Chart.js** for data visualization

## 👨‍💻 Author

**Abdul Razzaq Ansari**

## 🔗 Links

- **GitHub**: https://github.com/Rajak13/Smart-Summarizer
- **Quick Start**: [QUICK_START.md](QUICK_START.md)
- **Deployment**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

## 📚 References

1. Mihalcea, R., & Tarau, P. (2004). TextRank: Bringing order into text.
2. Lewis, M., et al. (2019). BART: Denoising sequence-to-sequence pre-training.
3. Zhang, J., et al. (2020). PEGASUS: Pre-training with extracted gap-sentences.

---

© 2025 Smart Summarizer. Abdul Razzaq Ansari