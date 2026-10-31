import os
import sys
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import settings
from shared.logging import get_logger
from shared.schemas import RetrievalRequest
from core.vector_db import vector_store

logger = get_logger("retrieval-service")

app = FastAPI(title="AlexandrAI Retrieval Service (RAG)", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "healthy", "service": "retrieval-service"}

@app.get("/health/live")
def live():
    return {"status": "alive", "service": "retrieval-service"}

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post(f"{settings.API_PREFIX}/retrieve")
def retrieve_candidates(req: RetrievalRequest) -> List[Dict[str, Any]]:
    candidates = vector_store.search_similar(query=req.query, top_k=req.top_k)
    return candidates

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
