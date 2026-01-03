---
title: Smart Summarizer
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# Smart Summarizer

Professional text summarization using three state-of-the-art models:

- **TextRank**: Fast extractive summarization (graph-based)
- **BART**: High-quality abstractive summarization  
- **PEGASUS**: Specialized abstractive model for summarization

## Features

- 📄 **Single Summary**: Generate summaries with individual models
- ⚖️ **Comparison**: Compare all three models side-by-side
- 📚 **Batch Processing**: Process multiple documents simultaneously
- 📊 **Evaluation**: ROUGE metrics and performance insights
- 📁 **File Support**: Upload .txt, .md, .pdf, .docx files

## Models

### TextRank (Extractive)
- **Speed**: Very fast (~0.03s)
- **Type**: Graph-based PageRank algorithm
- **Best for**: Quick summaries, keyword extraction

### BART (Abstractive)
- **Speed**: Moderate (~9s on CPU)
- **Type**: Transformer encoder-decoder
- **Best for**: Fluent, human-like summaries

### PEGASUS (Abstractive)
- **Speed**: Moderate (~6s on CPU)
- **Type**: Gap Sentence Generation pre-training
- **Best for**: High-quality abstractive summaries

## Usage

1. Navigate to the web interface
2. Choose between single summary or model comparison
3. Input text directly or upload a supported file
4. Select your preferred model(s)
5. Generate and compare summaries

## Supported File Types

- Plain text (`.txt`, `.md`)
- PDF documents (`.pdf`)
- Word documents (`.docx`, `.doc`)

## Author

**Abdul Razzaq Ansari**

## Links

- [GitHub Repository](https://github.com/Rajak13/Smart-Summarizer)
- [Documentation](https://github.com/Rajak13/Smart-Summarizer/blob/main/QUICK_START.md)