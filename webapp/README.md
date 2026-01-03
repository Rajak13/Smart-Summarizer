# Smart Summarizer Web Application

Professional web interface for comparing TextRank, BART, and PEGASUS summarization models.

## Features

- **Home**: Overview of the three summarization models
- **Single Summary**: Generate summaries with individual models
- **Comparison**: Compare all three models side-by-side
- **Batch Processing**: Process multiple documents simultaneously
- **Evaluation**: View ROUGE metric benchmarks and model performance

## Design

The UI follows the "Ink Wash" color palette:
- Charcoal (#4A4A4A)
- Cool Gray (#CBCBCB)
- Soft Ivory (#FFFFE3)
- Slate Blue (#6D8196)

## Running the Application

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Server

```bash
cd webapp
python app.py
```

The application will be available at: `http://localhost:5001`

### 3. Test the Routes

```bash
python test_webapp.py
```

## File Structure

```
webapp/
├── app.py                      # Flask application
├── templates/
│   ├── home.html              # Home page
│   ├── single_summary.html    # Single summary page
│   ├── comparison.html        # Model comparison page
│   ├── batch.html             # Batch processing page
│   └── evaluation.html        # Evaluation metrics page
├── static/
│   ├── css/
│   │   └── style.css          # Main stylesheet
│   └── js/
│       ├── evaluation.js      # Evaluation page logic
│       └── batch.js           # Batch processing logic
└── uploads/                    # Temporary file uploads
```

## API Endpoints

### POST /api/summarize
Generate a summary with a single model.

**Request:**
```json
{
  "text": "Your text here...",
  "model": "bart"  // or "textrank", "pegasus"
}
```

**Response:**
```json
{
  "success": true,
  "summary": "Generated summary...",
  "metadata": {
    "model_name": "BART",
    "processing_time": 2.34,
    "compression_ratio": 0.22
  }
}
```

### POST /api/compare
Compare all three models on the same text.

**Request:**
```json
{
  "text": "Your text here..."
}
```

**Response:**
```json
{
  "success": true,
  "results": {
    "textrank": { "summary": "...", "metadata": {...} },
    "bart": { "summary": "...", "metadata": {...} },
    "pegasus": { "summary": "...", "metadata": {...} }
  }
}
```

### POST /api/upload
Upload a file (.txt, .md, .pdf, .docx) and extract text.

**Request:** multipart/form-data with file

**Response:**
```json
{
  "success": true,
  "text": "Extracted text...",
  "filename": "document.pdf",
  "word_count": 1234
}
```

## Supported File Types

- Plain text (.txt, .md)
- PDF documents (.pdf)
- Word documents (.docx, .doc)

## Model Information

### TextRank
- Type: Extractive
- Algorithm: Graph-based PageRank
- Speed: Very fast (~0.03s)
- Best for: Quick summaries, keyword extraction

### BART
- Type: Abstractive
- Algorithm: Transformer encoder-decoder
- Speed: Moderate (~9s on CPU)
- Best for: Fluent, human-like summaries

### PEGASUS
- Type: Abstractive
- Algorithm: Gap Sentence Generation
- Speed: Moderate (~6s on CPU)
- Best for: High-quality abstractive summaries

## Notes

- Models are loaded lazily (on first use) to reduce startup time
- GPU acceleration is supported if CUDA is available
- All models generate similar compression ratios (~22%) for fair comparison
- File uploads are limited to 16MB
