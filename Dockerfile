FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn==21.2.0

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads logs

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Set environment variables for Hugging Face Spaces
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Run the application on port 7860 for Hugging Face Spaces
CMD ["gunicorn", "--chdir", "webapp", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "2"]