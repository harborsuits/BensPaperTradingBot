#!/usr/bin/env python3
"""
Gemma Embedding Service

FastAPI service for EmbeddingGemma-300M text embeddings.
Provides semantic similarity search for trading decisions.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager
import os
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "google/embeddinggemma-300m"
VECTOR_DIM = 256  # Use smaller dimension for speed
CACHE_FILE = "embeddings_cache.pkl"
INDEX_FILE = "faiss_index.idx"

# Global state
model = None
faiss_index = None
embeddings_cache = {}

class EmbeddingRequest(BaseModel):
    text: str
    id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class EmbeddingResponse(BaseModel):
    id: Optional[str] = None
    vector: List[float]
    dimension: int
    processing_ms: float
    cached: bool

class SimilarityRequest(BaseModel):
    query_text: str
    k: int = 5
    metadata_filters: Optional[Dict[str, Any]] = None
    min_similarity: float = 0.1

class SimilarityResponse(BaseModel):
    query_text: str
    results: List[Dict[str, Any]]
    processing_ms: float
    total_candidates: int

class HeadlineData(BaseModel):
    id: str
    text: str
    timestamp: str
    tickers: List[str]
    sector: Optional[str] = None
    sentiment: Optional[float] = None
    outcome_1d: Optional[float] = None
    outcome_5d: Optional[float] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and data on startup"""
    global model, faiss_index, embeddings_cache

    logger.info("Loading EmbeddingGemma model...")
    model = SentenceTransformer(MODEL_NAME, truncate_dim=VECTOR_DIM)

    # Load or create FAISS index
    if os.path.exists(INDEX_FILE):
        logger.info("Loading existing FAISS index...")
        faiss_index = faiss.read_index(INDEX_FILE)
    else:
        logger.info("Creating new FAISS index...")
        faiss_index = faiss.IndexFlatIP(VECTOR_DIM)  # Inner product for cosine similarity

    # Load embeddings cache
    if os.path.exists(CACHE_FILE):
        logger.info("Loading embeddings cache...")
        with open(CACHE_FILE, 'rb') as f:
            embeddings_cache = pickle.load(f)

    logger.info(f"Embedder service ready. Model: {MODEL_NAME}, Dimension: {VECTOR_DIM}")
    yield

    # Save on shutdown
    logger.info("Saving embeddings cache and index...")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(embeddings_cache, f)
    faiss.write_index(faiss_index, INDEX_FILE)

app = FastAPI(title="Gemma Embedder Service", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "dimension": VECTOR_DIM,
        "cached_embeddings": len(embeddings_cache),
        "index_size": faiss_index.ntotal if faiss_index else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/embed", response_model=EmbeddingResponse)
async def embed_text(request: EmbeddingRequest):
    """Generate embedding for text"""
    start_time = datetime.now()

    # Check cache first
    cache_key = hash(request.text)
    if cache_key in embeddings_cache:
        cached_vector = embeddings_cache[cache_key]
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        return EmbeddingResponse(
            id=request.id,
            vector=cached_vector.tolist(),
            dimension=VECTOR_DIM,
            processing_ms=processing_time,
            cached=True
        )

    try:
        # Generate embedding
        embedding = model.encode([request.text], normalize_embeddings=True)[0]

        # Cache the result
        embeddings_cache[cache_key] = embedding

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        return EmbeddingResponse(
            id=request.id,
            vector=embedding.tolist(),
            dimension=VECTOR_DIM,
            processing_ms=processing_time,
            cached=False
        )

    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

@app.post("/store")
async def store_headline(headline: HeadlineData):
    """Store headline with embedding for similarity search"""
    try:
        # Generate embedding
        embedding = model.encode([headline.text], normalize_embeddings=True)[0]

        # Add to FAISS index
        faiss_index.add(np.array([embedding], dtype=np.float32))

        # Cache embedding
        cache_key = f"headline_{headline.id}"
        embeddings_cache[cache_key] = embedding

        # Store metadata (in a simple in-memory dict for now)
        metadata_key = f"meta_{headline.id}"
        embeddings_cache[metadata_key] = {
            "id": headline.id,
            "text": headline.text,
            "timestamp": headline.timestamp,
            "tickers": headline.tickers,
            "sector": headline.sector,
            "sentiment": headline.sentiment,
            "outcome_1d": headline.outcome_1d,
            "outcome_5d": headline.outcome_5d,
            "faiss_index": faiss_index.ntotal - 1  # Index in FAISS
        }

        logger.info(f"Stored headline: {headline.id} (index {faiss_index.ntotal - 1})")
        return {"status": "stored", "index": faiss_index.ntotal - 1}

    except Exception as e:
        logger.error(f"Store failed: {e}")
        raise HTTPException(status_code=500, detail=f"Store failed: {str(e)}")

@app.post("/similar", response_model=SimilarityResponse)
async def find_similar(request: SimilarityRequest):
    """Find similar headlines using vector similarity"""
    start_time = datetime.now()

    try:
        # Generate embedding for query
        query_embedding = model.encode([request.query_text], normalize_embeddings=True)[0]
        query_vector = np.array([query_embedding], dtype=np.float32)

        # Search FAISS index
        if faiss_index.ntotal == 0:
            return SimilarityResponse(
                query_text=request.query_text,
                results=[],
                processing_ms=(datetime.now() - start_time).total_seconds() * 1000,
                total_candidates=0
            )

        # k-NN search (request.k + some buffer for filtering)
        search_k = min(request.k * 2, faiss_index.ntotal)
        scores, indices = faiss_index.search(query_vector, search_k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # No more results
                break

            # Get metadata
            metadata_key = None
            metadata = None

            # Find metadata for this index
            for key, value in embeddings_cache.items():
                if key.startswith("meta_") and isinstance(value, dict) and value.get("faiss_index") == idx:
                    metadata = value
                    break

            if metadata:
                # Calculate additional stats
                similar_cases = []
                total_return_1d = 0
                total_return_5d = 0
                valid_cases_1d = 0
                valid_cases_5d = 0

                # Simple aggregation of outcomes from similar cases
                for other_key, other_value in embeddings_cache.items():
                    if (other_key.startswith("meta_") and
                        isinstance(other_value, dict) and
                        other_value.get("sector") == metadata.get("sector")):

                        if other_value.get("outcome_1d") is not None:
                            total_return_1d += other_value["outcome_1d"]
                            valid_cases_1d += 1

                        if other_value.get("outcome_5d") is not None:
                            total_return_5d += other_value["outcome_5d"]
                            valid_cases_5d += 1

                avg_return_1d = total_return_1d / valid_cases_1d if valid_cases_1d > 0 else None
                avg_return_5d = total_return_5d / valid_cases_5d if valid_cases_5d > 0 else None

                # Apply shrinkage if insufficient samples
                if valid_cases_1d < 100:  # Minimum sample requirement
                    avg_return_1d = avg_return_1d * min(1.0, valid_cases_1d / 100) if avg_return_1d else None

                if valid_cases_5d < 100:
                    avg_return_5d = avg_return_5d * min(1.0, valid_cases_5d / 100) if avg_return_5d else None

                if score >= request.min_similarity:
                    results.append({
                        "id": metadata["id"],
                        "text": metadata["text"],
                        "timestamp": metadata["timestamp"],
                        "tickers": metadata["tickers"],
                        "sector": metadata["sector"],
                        "sentiment": metadata["sentiment"],
                        "similarity_score": float(score),
                        "avg_return_1d": avg_return_1d,
                        "avg_return_5d": avg_return_5d,
                        "sample_size_1d": valid_cases_1d,
                        "sample_size_5d": valid_cases_5d,
                        "outcome_1d": metadata.get("outcome_1d"),
                        "outcome_5d": metadata.get("outcome_5d")
                    })

        # Sort by similarity and take top k
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        results = results[:request.k]

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        return SimilarityResponse(
            query_text=request.query_text,
            results=results,
            processing_ms=processing_time,
            total_candidates=faiss_index.ntotal
        )

    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {str(e)}")

@app.delete("/clear")
async def clear_index():
    """Clear all stored embeddings (for testing)"""
    global faiss_index, embeddings_cache

    faiss_index = faiss.IndexFlatIP(VECTOR_DIM)
    embeddings_cache = {}

    # Save empty state
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(embeddings_cache, f)
    faiss.write_index(faiss_index, INDEX_FILE)

    return {"status": "cleared"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
