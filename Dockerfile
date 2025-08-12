# Use Python 3.11 slim image for efficiency
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed chromadb_data logs web_ui/static

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=web_ui/app.py
ENV FLASK_ENV=production
ENV FLASK_DEBUG=False

# Expose port
EXPOSE 8000

# Health check  
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD sh -c 'curl -fsS http://localhost:${PORT:-8000}/health || exit 1'

# Run the application
# Default to Gunicorn; falls back to Flask if needed
ENV PORT=8000
CMD gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --keep-alive 60 web_ui.app:app
