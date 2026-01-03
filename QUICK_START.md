# Smart Summarizer - Quick Start Guide

## 🚀 Getting Started

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Rajak13/Smart-Summarizer.git
cd Smart-Summarizer

# Run installation script
# For macOS/Linux:
./install.sh

# For Windows:
install.bat
```

### 2. Running the Web Application

```bash
# Quick start (recommended)
# For macOS/Linux:
./run_webapp.sh

# For Windows:
run_webapp.bat

# Or manually:
cd webapp
python app.py
```

### 3. Access the Application

Open your browser and navigate to:
```
http://localhost:5001
```

## 📱 Features

### 🏠 Home Page
- Overview of three summarization models
- Model comparison cards
- Quick navigation

### 📄 Single Summary
- Generate summaries with individual models
- Upload files (.txt, .md, .pdf, .docx)
- Real-time processing metrics

### ⚖️ Comparison
- Compare all three models side-by-side
- Synchronized input
- Performance metrics for each model

### 📚 Batch Processing
- Process multiple documents simultaneously
- Load sample documents
- Export results to CSV
- Track processing status

### 📊 Evaluation
- ROUGE metrics visualization
- Benchmark data comparison
- Model performance insights

## 🤖 Models

### TextRank (Extractive)
- **Speed**: Very fast (~0.03s)
- **Type**: Graph-based PageRank
- **Best for**: Quick summaries, keyword extraction

### BART (Abstractive)
- **Speed**: Moderate (~9s on CPU)
- **Type**: Transformer encoder-decoder
- **Best for**: Fluent, human-like summaries

### PEGASUS (Abstractive)
- **Speed**: Moderate (~6s on CPU)
- **Type**: Gap Sentence Generation
- **Best for**: High-quality abstractive summaries

## 📝 Supported File Types

- Plain text (`.txt`, `.md`)
- PDF documents (`.pdf`)
- Word documents (`.docx`, `.doc`)

## 🔧 API Endpoints

### POST /api/summarize
Generate summary with a single model
```json
{
  "text": "Your text here...",
  "model": "bart"
}
```

### POST /api/compare
Compare all three models
```json
{
  "text": "Your text here..."
}
```

### POST /api/upload
Upload and extract text from file
```
multipart/form-data with file
```

## 🧪 Testing

```bash
# Test all routes
python test_webapp.py
```

## 📦 Project Structure

```
smart-summarizer/
├── webapp/              # Flask web application
│   ├── app.py          # Main application
│   ├── templates/      # HTML templates
│   └── static/         # CSS, JS, assets
├── models/             # Summarization models
├── utils/              # Utility functions
├── data/               # Data files
├── notebooks/          # Jupyter notebooks
└── tests/              # Test files
```

## 🎨 Design

**Color Palette (Ink Wash):**
- Charcoal: #4A4A4A
- Cool Gray: #CBCBCB
- Soft Ivory: #FFFFE3
- Slate Blue: #6D8196

## 🐛 Troubleshooting

### Models not loading?
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Port 5001 already in use?
Edit `webapp/app.py` and change:
```python
app.run(debug=True, port=5002)  # Use different port
```

### File upload not working?
Check file size (max 16MB) and format (.txt, .md, .pdf, .docx)

## 👨‍💻 Author

**Abdul Razzaq Ansari**

## 🔗 Links

- GitHub: https://github.com/Rajak13/Smart-Summarizer
- Documentation: See `webapp/README.md`

## 📄 License

© 2025 Smart Summarizer. Abdul Razzaq Ansari

---

**Need help?** Check the documentation or open an issue on GitHub.
