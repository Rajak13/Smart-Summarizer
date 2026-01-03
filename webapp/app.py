"""
Smart Summarizer - Flask Web Application
Professional UI matching Figma design
"""

from flask import Flask, render_template, request, jsonify
import sys
from pathlib import Path
import os
from werkzeug.utils import secure_filename
import PyPDF2
from docx import Document as DocxDocument

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.textrank import TextRankSummarizer
from models.bart import BARTSummarizer
from models.pegasus import PEGASUSSummarizer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'md', 'text', 'pdf', 'docx', 'doc'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize models (lazy loading)
models = {}

def get_model(model_name):
    """Load and cache models"""
    if model_name not in models:
        if model_name == "textrank":
            models[model_name] = TextRankSummarizer()
        elif model_name == "bart":
            models[model_name] = BARTSummarizer(device='cpu')
        elif model_name == "pegasus":
            models[model_name] = PEGASUSSummarizer(device='cpu')
    return models[model_name]

@app.route('/')
def home():
    """Home page"""
    return render_template('home.html')

@app.route('/single-summary')
def single_summary():
    """Single summary page"""
    return render_template('single_summary.html')

@app.route('/comparison')
def comparison():
    """Model comparison page"""
    return render_template('comparison.html')

@app.route('/batch')
def batch():
    """Batch processing page"""
    return render_template('batch.html')

@app.route('/evaluation')
def evaluation():
    """Evaluation page"""
    return render_template('evaluation.html')

@app.route('/api/summarize', methods=['POST'])
def summarize():
    """API endpoint for summarization"""
    try:
        data = request.json
        text = data.get('text', '')
        model_name = data.get('model', 'bart').lower()
        
        if not text or len(text.split()) < 10:
            return jsonify({
                'success': False,
                'error': 'Please provide at least 10 words of text'
            }), 400
        
        # Get model
        model = get_model(model_name)
        
        # Calculate target summary length (approximately 20-25% of original)
        input_words = len(text.split())
        target_length = max(30, min(150, int(input_words * 0.22)))  # 22% compression
        
        # Generate summary based on model type
        if model_name == 'textrank':
            # For TextRank, calculate number of sentences to achieve similar compression
            sentences = text.count('.') + text.count('!') + text.count('?')
            num_sentences = max(2, int(sentences * 0.3))  # ~30% of sentences
            result = model.summarize_with_metrics(text, num_sentences=num_sentences)
        else:
            # For BART and PEGASUS, use word-based limits
            result = model.summarize_with_metrics(
                text,
                max_length=target_length,
                min_length=max(20, int(target_length * 0.5))
            )
        
        return jsonify({
            'success': True,
            'summary': result['summary'],
            'metadata': result['metadata']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/compare', methods=['POST'])
def compare():
    """API endpoint for comparing all three models"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text or len(text.split()) < 10:
            return jsonify({
                'success': False,
                'error': 'Please provide at least 10 words of text'
            }), 400
        
        results = {}
        
        # Calculate consistent target length for all models
        input_words = len(text.split())
        target_length = max(30, min(150, int(input_words * 0.22)))
        sentences = text.count('.') + text.count('!') + text.count('?')
        num_sentences = max(2, int(sentences * 0.3))
        
        # Run all three models
        for model_name in ['textrank', 'bart', 'pegasus']:
            try:
                model = get_model(model_name)
                
                if model_name == 'textrank':
                    result = model.summarize_with_metrics(text, num_sentences=num_sentences)
                else:
                    result = model.summarize_with_metrics(
                        text,
                        max_length=target_length,
                        min_length=max(20, int(target_length * 0.5))
                    )
                
                results[model_name] = {
                    'summary': result['summary'],
                    'metadata': result['metadata']
                }
            except Exception as e:
                results[model_name] = {
                    'error': str(e)
                }
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """API endpoint for file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Please upload .txt, .md, .pdf, .docx, or .doc files'
            }), 400
        
        # Extract text based on file type
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        try:
            if file_ext in ['txt', 'md', 'text']:
                # Plain text files
                text = file.read().decode('utf-8')
            
            elif file_ext == 'pdf':
                # PDF files
                pdf_reader = PyPDF2.PdfReader(file)
                text = ''
                for page in pdf_reader.pages:
                    text += page.extract_text() + '\n'
            
            elif file_ext in ['docx', 'doc']:
                # Word documents
                doc = DocxDocument(file)
                text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            
            else:
                return jsonify({
                    'success': False,
                    'error': 'Unsupported file format'
                }), 400
                
        except UnicodeDecodeError:
            return jsonify({
                'success': False,
                'error': 'File encoding not supported. Please use UTF-8 encoded files'
            }), 400
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Error reading file: {str(e)}'
            }), 400
        
        if not text or len(text.split()) < 10:
            return jsonify({
                'success': False,
                'error': 'File content is too short. Please provide at least 10 words'
            }), 400
        
        return jsonify({
            'success': True,
            'text': text,
            'filename': filename,
            'word_count': len(text.split())
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    import os
    
    # Get port from environment variable (for cloud deployment)
    port = int(os.environ.get('PORT', 5001))
    
    # Check if running in production
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    # Bind to 0.0.0.0 for cloud deployment
    app.run(host='0.0.0.0', port=port, debug=debug)
