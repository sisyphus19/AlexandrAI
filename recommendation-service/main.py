import os
import sys
from fastapi import FastAPI
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import settings
from shared.logging import get_logger
from shared.schemas import RankedCandidate
from core.recommendation_engine import recommendation_engine

logger = get_logger("recommendation-service")

app = FastAPI(title="AlexandrAI Recommendation Engine Service", version="1.0.0")

class RankRequest(BaseModel):
    candidates: List[Dict[str, Any]]
    favorite_genres: Optional[List[str]] = None
    max_page_count: Optional[int] = None
    top_k: int = 5

@app.get("/health")
def health():
    return {"status": "healthy", "service": "recommendation-service"}

@app.get("/health/live")
def live():
    return {"status": "alive", "service": "recommendation-service"}

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post(f"{settings.API_PREFIX}/rank", response_model=List[RankedCandidate])
def rank_candidates(req: RankRequest):
    ranked = recommendation_engine.process_recommendations(
        retrieved_candidates=req.candidates,
        favorite_genres=req.favorite_genres,
        max_page_count=req.max_page_count,
        top_k=req.top_k
    )
    return ranked

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
