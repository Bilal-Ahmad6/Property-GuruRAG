# PropertyGuru RAG API

FastAPI-based real estate RAG (Retrieval-Augmented Generation) system with lazy loading optimized for deployment on Render.

## Key Features

- **Lazy Loading**: ChromaDB and SentenceTransformer models are only loaded when first requested, not at startup
- **Fast Health Checks**: `/health` endpoint responds instantly for load balancer health checks
- **Production Ready**: Works with Gunicorn + UvicornWorker for deployment
- **Render Optimized**: Addresses "No open ports detected" and "WORKER TIMEOUT" issues

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

The app will start on `http://localhost:8000`

### Production Deployment (Render)

The app is configured to work with the Procfile:

```
web: gunicorn -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --keep-alive 60 app:app
```

## API Endpoints

### Health Check
```http
GET /health
```

Fast response for health checks (no heavy dependencies loaded).

**Response:**
```json
{
  "status": "healthy", 
  "message": "Service is running",
  "version": "1.0.0"
}
```

### Root Information
```http
GET /
```

Basic API information and available endpoints.

### Query Properties
```http
POST /query
```

Query the property database using natural language.

**Request Body:**
```json
{
  "question": "What properties are available in Bahria Town?",
  "k": 5,
  "collection": "zameen_listings"
}
```

**Response:**
```json
{
  "question": "What properties are available in Bahria Town?",
  "k": 5,
  "results": [
    {
      "id": "property_123",
      "text": "Property description...",
      "metadata": {
        "location": "Bahria Town",
        "price": "50 Lakh"
      }
    }
  ],
  "collection_used": "zameen_listings",
  "processing_time_ms": 245.67
}
```

### List Collections
```http
GET /collections
```

List available ChromaDB collections.

### Collection Info
```http
GET /collection/{collection_name}/info
```

Get information about a specific collection.

## Configuration

The app uses environment variables with the `ZAMEEN_` prefix:

- `ZAMEEN_EMBEDDING_MODEL`: Sentence transformer model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `ZAMEEN_COLLECTION_NAME`: Default collection name (default: `zameen_listings`)
- `ZAMEEN_CHROMA_PERSIST_DIR`: ChromaDB data directory (default: `chromadb_data`)
- `PORT`: Server port (set automatically by Render)

## Testing

Run the test script to verify functionality:

```bash
# Start the server
python app.py

# In another terminal, run tests
python test_api.py
```

## Deployment Benefits

### Before (Problems)
- Heavy dependencies loaded at import time
- Slow startup causing Render timeouts
- "No open ports detected" errors
- Worker timeouts during initialization

### After (Solutions)
- ✅ Fast startup - heavy dependencies loaded lazily
- ✅ Instant health checks for load balancers
- ✅ PORT binding happens immediately
- ✅ Stable production deployment

## Architecture

```
FastAPI App (app.py)
├── Instant startup (no heavy imports)
├── /health endpoint (always fast)
├── Lazy loading on first request:
│   ├── ChromaDB client
│   ├── SentenceTransformer model
│   └── Vector collections
└── Graceful error handling
```

## Production Deployment

1. **Render**: Uses the Procfile automatically
2. **Manual**: `gunicorn -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --workers 2 app:app`
3. **Local**: `python app.py`

The app automatically detects the environment and binds to the correct port.
