import os
import sys
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import settings
from shared.logging import get_logger

logger = get_logger("analytics-service")

app = FastAPI(title="AlexandrAI Analytics & Business Metrics Service", version="1.0.0")

# Business Metrics Gauges & Counters
POPULAR_GENRES = Counter("business_genre_clicks_total", "Genre clicks count", ["genre"])
USER_FEEDBACK_RATIO = Gauge("business_user_feedback_satisfaction_ratio", "Percentage of positive feedback")

@app.get("/health")
def health():
    return {"status": "healthy", "service": "analytics-service"}

@app.get("/health/live")
def live():
    return {"status": "alive", "service": "analytics-service"}

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/analytics/track")
def track_event(event_type: str, details: dict):
    if event_type == "genre_click":
        genre = details.get("genre", "Unknown")
        POPULAR_GENRES.labels(genre=genre).inc()
    logger.info(f"Analytics tracked event '{event_type}': {details}")
    return {"status": "tracked"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
