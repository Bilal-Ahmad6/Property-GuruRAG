#!/bin/bash

# PropertyGuru Production Startup Script

# Set default values
export FLASK_ENV=${FLASK_ENV:-production}
export FLASK_DEBUG=${FLASK_DEBUG:-false}
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8000}
export WORKERS=${WORKERS:-1}

# Create necessary directories
mkdir -p data/raw data/processed chromadb_data logs

# Check for required environment variables
if [ -z "$COHERE_API_KEY" ] && [ -z "$ZAMEEN_COHERE_API_KEY" ]; then
    echo "❌ Error: COHERE_API_KEY or ZAMEEN_COHERE_API_KEY environment variable is required"
    exit 1
fi

# Check for vector database
if [ ! -d "chromadb_data" ] || [ -z "$(ls -A chromadb_data)" ]; then
    echo "⚠️  Warning: Vector database not found. You may need to run embedding process first:"
    echo "   python scripts/embed_and_store.py --input data/processed/zameen_phase7_chunks.jsonl"
fi

# Check for processed data
if [ ! -f "data/processed/zameen_phase7_processed.json" ]; then
    echo "⚠️  Warning: Processed data not found. You may need to process raw data first:"
    echo "   python scripts/clean_and_enrich.py"
fi

echo "🚀 Starting PropertyGuru in production mode..."
echo "📍 Host: $HOST"
echo "🔌 Port: $PORT"
echo "👥 Workers: $WORKERS"

# Start with Gunicorn in production, fallback to Flask dev server
if command -v gunicorn >/dev/null 2>&1; then
    echo "🔄 Using Gunicorn production server..."
    exec gunicorn \
        --bind $HOST:$PORT \
        --workers $WORKERS \
        --timeout 120 \
        --keep-alive 60 \
        --max-requests 1000 \
        --max-requests-jitter 100 \
        --access-logfile logs/access.log \
        --error-logfile logs/error.log \
        --log-level info \
        --preload \
        web_ui.app:app
else
    echo "🔄 Using Flask development server..."
    echo "⚠️  Consider installing Gunicorn for production: pip install gunicorn"
    exec python web_ui/app.py
fi
