# PropertyGuru RAG — Bahria Town Phase 7 Real‑Estate Assistant

A practical, end‑to‑end Retrieval‑Augmented Generation (RAG) stack for real‑estate listings in Bahria Town Phase 7 (Rawalpindi).

It scrapes listings, cleans and enriches data, creates embeddings with Sentence-Transformers, stores vectors in ChromaDB, and serves a conversational web UI powered by Cohere (or other LLMs) to help users find relevant properties quickly.


## Highlights

- Data pipeline: scrape → clean/enrich → chunk → embed → query
- Vector DB: ChromaDB (persistent on disk)
- Embeddings: Sentence-Transformers (configurable model)
- LLMs: Cohere by default; Groq/others available in CLI
- Web UI: Flask app with chat interface and health endpoint
- API: Minimal FastAPI app for programmatic queries (optional)
- Windows-friendly (PowerShell) with start scripts and Docker support


## Project Structure

```
PropertyGuru/
├─ web_ui/
│  ├─ app.py                # Flask web app (UI + JSON APIs)
│  └─ templates/chat.html   # Chat UI
├─ rag_app/
│  └─ app.py                # Optional FastAPI service (/query)
├─ scripts/
│  ├─ scrape_listings.py    # Scrapes Zameen listing + detail pages
│  ├─ clean_and_enrich.py   # Cleans, canonicalizes, chunks text
│  ├─ embed_and_store.py    # Embeds chunks and stores in ChromaDB
│  ├─ query_rag.py          # RAG querying + CLI helpers
│  └─ utils.py              # IO/HTTP helpers
├─ data/
│  ├─ raw/                  # Scraped JSON + progress/template
│  └─ processed/            # Processed listings + chunks.jsonl
├─ chromadb_data/           # ChromaDB persistence (auto-created)
├─ tests/                   # Basic sanity tests
├─ config.py                # Central settings via pydantic-settings
├─ requirements.txt         # Dependencies (pinned/compatible)
├─ Dockerfile               # Container image for web UI
├─ docker-compose.yml       # Local container run (optional)
├─ start.bat / start.sh     # Convenience scripts to run UI
└─ README.md                # This file
```


## Prerequisites

- Python 3.10+ (3.11 recommended)
- Windows PowerShell (project scripts are Windows-friendly)
- Optional: CUDA‑enabled PyTorch for GPU embeddings


## Quick Start

1) Clone and create a virtual environment

```powershell
# From PowerShell in the project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

2) Configure environment

Create a .env file in the project root (values shown are examples):

```
# App + data
ZAMEEN_START_URL=https://www.zameen.com/Homes/Rawalpindi_Bahria_Town_Rawalpindi_Bahria_Town_Phase_7-3047-1.html
ZAMEEN_COLLECTION_NAME=zameen_listings
ZAMEEN_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

# LLM keys (UI uses Cohere by default)
COHERE_API_KEY=your_cohere_api_key_here
# Or via settings module prefix
ZAMEEN_COHERE_API_KEY=your_cohere_api_key_here
# Optional: for CLI modes in scripts/query_rag.py
ZAMEEN_GROQ_API_KEY=your_groq_api_key_here

# Web UI server
PORT=8000
HOST=0.0.0.0
```

3) Get data

Option A — Scrape from source (be considerate; see Legal/Ethics below):

```powershell
python scripts/scrape_listings.py --start-url $env:ZAMEEN_START_URL --max-pages 8
```

This writes:
- data/raw/zameen_phase7_progress.json (incremental)
- data/raw/zameen_phase7_raw.json (final JSON array)
- data/raw/template_max_fields.json (example with most fields)

Option B — Use an existing raw file

If you already have data/raw/zameen_phase7_raw.json, skip scraping.

4) Clean, canonicalize, and chunk

```powershell
python scripts/clean_and_enrich.py --input data/raw/zameen_phase7_raw.json --processed-out data/processed/zameen_phase7_processed.json --chunks-out data/processed/zameen_phase7_chunks.jsonl
```

5) Embed and store vectors in ChromaDB

```powershell
python scripts/embed_and_store.py --input data/processed/zameen_phase7_chunks.jsonl --processed data/processed/zameen_phase7_processed.json --collection zameen_listings --model sentence-transformers/all-mpnet-base-v2
```

This creates/updates a persistent DB at chromadb_data/ and prints a few verification queries.

6) Launch the Web UI

```powershell
python web_ui/app.py
# or
./start.bat
```

- UI: http://localhost:8000/
- Health: http://localhost:8000/health

If you see an error about missing Cohere key, ensure your .env contains COHERE_API_KEY (or ZAMEEN_COHERE_API_KEY).


## Web UI — What it does

- POST /api/message accepts { chat_id, message, store_history } and returns an assistant reply.
- The server builds a short conversation history and uses RAG over ChromaDB to find matching listings.
- The answer text is intentionally brief; the UI separately renders a neat table of matching properties (title, price, location, link), honoring requested counts (e.g., “show me 1 property”).
- GET /health exposes status for monitoring (API key present, vector DB folder exists, etc.).

Defaults
- Port: 8000 (env PORT)
- Host: 0.0.0.0 (env HOST)
- LLM engine: Cohere (model: command-r-plus)


## Programmatic API (optional)

There’s a minimal FastAPI app at rag_app/app.py:

- Run: uvicorn rag_app.app:app --host 0.0.0.0 --port 9000
- POST /query with body: { "question": "...", "k": 5, "collection": "zameen_listings" }
- Returns top‑k matches with text + metadata

This API uses Chroma’s built‑in embedding function (SentenceTransformerEmbeddingFunction) with the configured model.


## Configuration

All configuration is centralized in config.py via pydantic‑settings and .env support.

Key settings (env prefix ZAMEEN_):
- start_url: Seed listing page for scraping
- embedding_model: Sentence-Transformers model (e.g., sentence-transformers/all-mpnet-base-v2)
- collection_name: ChromaDB collection (default: zameen_listings)
- chroma_persist_dir: Persistence path (default: chromadb_data/)
- user_agent, requests_timeout: Scraper HTTP behavior
- cohere_api_key, groq_api_key: LLM API keys

Note: The Flask UI also looks for COHERE_API_KEY and ZAMEEN_COHERE_API_KEY.


## CLI: Query via RAG

You can test RAG from the terminal without the UI:

```powershell
python scripts/query_rag.py --collection zameen_listings --query "3 bed apartment near park in Phase 7" --k 5 --embedding-model sentence-transformers/all-mpnet-base-v2 --llm-engine cohere --cohere-model command-r-plus --cohere-api-key $env:COHERE_API_KEY --explain
```

The script:
- Normalizes location terms (e.g., “River Hills”)
- Extracts a requested number (“show me 3 ...”)
- Builds a compact property context from retrieved chunks
- Calls the chosen LLM to craft a short, helpful reply


## Docker

Build and run locally:

```powershell
# Build image
docker build -t propertyguru:local .

# Run (reads .env if provided)
docker run --rm -p 8000:8000 --env-file .env propertyguru:local
```

docker-compose is also provided for convenience.


## Testing

Basic tests live under tests/. To run:

```powershell
pytest -q
```


## Troubleshooting

- UI says “Cohere API key missing”
  - Add COHERE_API_KEY (or ZAMEEN_COHERE_API_KEY) to .env and restart
- Retrieval returns empty or irrelevant results
  - Ensure you ran the full pipeline: clean_and_enrich → embed_and_store
  - Verify chromadb_data/ exists and contains a collection named zameen_listings
  - Confirm the embedding model used during query matches the one used to embed
- ChromaDB schema or compatibility errors
  - Stop the app, delete chromadb_data/, and re‑run embeddings
- Slow embeddings
  - Install a CUDA build of PyTorch; scripts auto‑detect GPU if available


## Legal & Ethics

- This project includes a scraper for educational/demo purposes.
- Before scraping any website, always review and comply with its Terms of Service and robots.txt rules. Rate‑limit and cache responsibly.
- The scraper’s robots checks are disabled in code per user requirements; if you intend to use this beyond local demos, re‑enable checks and adopt polite crawling.


## Notes

- The web UI routes are in web_ui/app.py (Flask).
- The RAG logic (retrieval, prompting, helpers) is in scripts/query_rag.py.
- You can adjust prompt behavior and filters there if you want stricter type/price handling.
- For production, run the Flask app with a production WSGI (e.g., gunicorn) and a reverse proxy.


---

If you want this README tailored further (screenshots, branding, CI badges), let me know the details and I’ll refine it. 
