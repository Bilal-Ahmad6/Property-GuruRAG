import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)
from fastapi import FastAPI
from pydantic import BaseModel

# Import our local config module
import config
settings = config.settings


app = FastAPI(title="Zameen RAG API", version="0.1.0")


class QueryRequest(BaseModel):
    question: str
    k: int = 5
    collection: Optional[str] = None


class QueryResponse(BaseModel):
    question: str
    k: int
    results: List[Dict[str, Any]]


def get_collection(name: str):
    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    emb_fn = SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
    return client.get_or_create_collection(name=name, embedding_function=emb_fn)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    collection_name = req.collection or settings.collection_name
    collection = get_collection(collection_name)
    res = collection.query(query_texts=[req.question], n_results=req.k)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]

    results: List[Dict[str, Any]] = []
    for rid, doc, meta in zip(ids, docs, metas):
        results.append({
            "id": rid,
            "text": doc,
            "metadata": meta,
        })

    return QueryResponse(question=req.question, k=req.k, results=results)

