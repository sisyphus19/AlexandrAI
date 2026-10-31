import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, Header, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from sqlalchemy.orm import Session
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY

from shared.config import settings
from shared.logging import get_logger
from shared.schemas import (
    UserRegisterRequest, UserLoginRequest, TokenResponse, UserPreferencesUpdate, UserProfileResponse,
    BookResponse, RecommendationRequest, RecommendationResponse, FeedbackRequest, FeedbackResponse,
    HealthCheckResponse
)
from core.database import get_db, UserModel, BookModel, FeedbackModel, init_db
from core.auth import hash_password, verify_password, create_access_token, get_current_user
from fastapi.security import OAuth2PasswordBearer

# Optional auth: returns user if valid Bearer token provided, else None (allows anonymous requests)
_optional_oauth2 = OAuth2PasswordBearer(tokenUrl=f"{settings.API_PREFIX}/users/login", auto_error=False)

async def get_current_user_optional(token: Optional[str] = Depends(_optional_oauth2), db: Session = Depends(get_db)) -> Optional[UserModel]:
    """Returns authenticated user or None for anonymous access."""
    if not token:
        return None
    try:
        return get_current_user(token=token, db=db)
    except HTTPException:
        return None
from core.vector_db import vector_store
from core.recommendation_engine import recommendation_engine
from core.llm_provider import llm_service

logger = get_logger("alexandrai-api-monolith")

app = FastAPI(
    title="AlexandrAI Platform",
    description="Distributed AI Book Recommendation Platform featuring LLM Reasoning, RAG Vector Search, and Multi-Factor Hybrid Reranking.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus Metrics Definitions
# Safe factory helpers prevent ValueError on uvicorn --reload (subprocess re-imports the same module,
# trying to re-register already-registered metrics into the global REGISTRY).
def _safe_counter(name: str, description: str, labels: list) -> Counter:
    try:
        return Counter(name, description, labels)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)

def _safe_histogram(name: str, description: str, labels: list) -> Histogram:
    try:
        return Histogram(name, description, labels)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)

REQUEST_COUNT = _safe_counter("http_requests_total", "Total HTTP Requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = _safe_histogram("http_request_duration_seconds", "HTTP Request Duration in Seconds", ["endpoint"])
RECOMMENDATION_COUNTER = _safe_counter("recommendations_generated_total", "Total Recommendations Generated", ["fallback_used"])
FEEDBACK_COUNTER = _safe_counter("recommendation_feedback_total", "Total Recommendation Feedback Entries", ["rating"])

# --- Request Tracing & Logging Middleware ---

@app.middleware("http")
async def request_tracing_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    start_time = time.time()

    response: Response = await call_next(request)

    latency_ms = (time.time() - start_time) * 1000
    response.headers["X-Request-ID"] = request_id

    # Record Prometheus metrics
    endpoint = request.url.path
    REQUEST_COUNT.labels(method=request.method, endpoint=endpoint, status=response.status_code).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency_ms / 1000.0)

    # Structured JSON log
    logger.info(
        f"{request.method} {endpoint} -> HTTP {response.status_code}",
        extra={"request_id": request_id, "latency_ms": latency_ms}
    )
    return response


# --- Startup & Health Endpoints ---

@app.on_event("startup")
def startup_event():
    init_db()
    logger.info("AlexandrAI Monolithic Core API started successfully.")


@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
def health_check():
    """Kubernetes Readiness Probe Endpoint."""
    return HealthCheckResponse(
        status="healthy",
        service=settings.SERVICE_NAME,
        timestamp=datetime.now(timezone.utc),
        version="1.0.0",
        dependencies={
            "database": "connected",
            "vector_store": "ready",
            "llm_provider": settings.LLM_PROVIDER
        }
    )


@app.get("/health/live", response_model=HealthCheckResponse, tags=["System"])
def liveness_check():
    """Kubernetes Liveness Probe Endpoint."""
    return HealthCheckResponse(
        status="alive",
        service=settings.SERVICE_NAME,
        timestamp=datetime.now(timezone.utc),
        version="1.0.0"
    )


@app.get("/metrics", tags=["System"])
def metrics():
    """Prometheus Scrape Endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# --- User Management Endpoints (/api/v1/users) ---

@app.post(f"{settings.API_PREFIX}/users/register", response_model=UserProfileResponse, status_code=status.HTTP_201_CREATED, tags=["Users"])
def register_user(req: UserRegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(UserModel).filter((UserModel.username == req.username) | (UserModel.email == req.email)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username or email already registered.")

    hashed_pw = hash_password(req.password)
    user = UserModel(
        username=req.username,
        email=req.email,
        hashed_password=hashed_pw,
        favorite_genres=[],
        preferred_book_length=None,
        preferred_mood=None
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post(f"{settings.API_PREFIX}/users/login", response_model=TokenResponse, tags=["Users"])
def login_user(req: UserLoginRequest, db: Session = Depends(get_db)):
    user = db.query(UserModel).filter(UserModel.username == req.username).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = create_access_token(user_id=user.id, username=user.username)
    return TokenResponse(access_token=token, user_id=user.id, username=user.username)


@app.get(f"{settings.API_PREFIX}/users/me", response_model=UserProfileResponse, tags=["Users"])
def get_user_profile(current_user: UserModel = Depends(get_current_user)):
    return current_user


@app.put(f"{settings.API_PREFIX}/users/preferences", response_model=UserProfileResponse, tags=["Users"])
def update_user_preferences(
    prefs: UserPreferencesUpdate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    current_user.favorite_genres = prefs.favorite_genres
    if prefs.preferred_book_length:
        current_user.preferred_book_length = prefs.preferred_book_length
    if prefs.preferred_mood:
        current_user.preferred_mood = prefs.preferred_mood

    db.commit()
    db.refresh(current_user)
    return current_user


# --- Book Catalog Endpoints (/api/v1/books) ---

@app.get(f"{settings.API_PREFIX}/books", response_model=List[BookResponse], tags=["Books"])
def list_books(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    books = db.query(BookModel).offset(skip).limit(limit).all()
    return books


@app.get(f"{settings.API_PREFIX}/books/{{book_id}}", response_model=BookResponse, tags=["Books"])
def get_book_details(book_id: int, db: Session = Depends(get_db)):
    book = db.query(BookModel).filter(BookModel.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found.")
    return book


# --- Core Recommendation Engine Endpoint (/api/v1/recommend) ---

@app.post(f"{settings.API_PREFIX}/recommend", response_model=RecommendationResponse, tags=["Recommendation Engine"])
async def generate_recommendations(
    req: RecommendationRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    start_time = time.time()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    # 1. Retrieve Candidate Books via Vector RAG Search
    retrieved_candidates = vector_store.search_similar(query=req.query, top_k=20)
    
    # If DB has books but vector search returns empty, load from DB fallback
    if not retrieved_candidates:
        db_books = db.query(BookModel).limit(20).all()
        retrieved_candidates = [
            {
                "id": b.id, "title": b.title, "author": b.author, "genres": b.genres,
                "description": b.description, "rating": b.rating, "page_count": b.page_count,
                "publication_year": b.publication_year, "semantic_similarity": 0.65
            }
            for b in db_books
        ]

    # 2. Extract User Context & Preferences
    user_fav_genres = req.favorite_genres or []
    user_mood = req.preferred_mood
    max_pages = req.max_page_count

    if req.user_id:
        user = db.query(UserModel).filter(UserModel.id == req.user_id).first()
        if user:
            if not user_fav_genres:
                user_fav_genres = user.favorite_genres or []
            if not user_mood:
                user_mood = user.preferred_mood

    # 3. Hybrid Reranking (CandidateFilter -> FeatureExtractor -> Scoring -> Ranker)
    ranked_candidates = recommendation_engine.process_recommendations(
        retrieved_candidates=retrieved_candidates,
        favorite_genres=user_fav_genres,
        max_page_count=max_pages,
        top_k=req.top_k
    )

    # 4. LLM Reasoning & Structured Feature Explainability
    explained_recommendations, fallback_used = await llm_service.get_explained_recommendations(
        query=req.query,
        ranked_candidates=ranked_candidates,
        genres=user_fav_genres,
        mood=user_mood
    )

    execution_time_ms = (time.time() - start_time) * 1000
    RECOMMENDATION_COUNTER.labels(fallback_used=str(fallback_used)).inc()

    return RecommendationResponse(
        query=req.query,
        request_id=request_id,
        recommendations=explained_recommendations,
        total_candidates_retrieved=len(retrieved_candidates),
        execution_time_ms=round(execution_time_ms, 2),
        llm_fallback_used=fallback_used
    )


# --- Recommendation Feedback Loop (/api/v1/feedback) ---

@app.post(f"{settings.API_PREFIX}/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED, tags=["Feedback"])
def record_feedback(
    req: FeedbackRequest,
    request: Request,
    current_user: Optional[UserModel] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    request_id = getattr(request.state, "request_id", None)
    
    feedback_entry = FeedbackModel(
        user_id=current_user.id if current_user else None,
        book_id=req.book_id,
        rating=req.rating,
        is_helpful=req.is_helpful,
        comment=req.comment,
        request_id=request_id
    )
    db.add(feedback_entry)
    db.commit()
    db.refresh(feedback_entry)

    FEEDBACK_COUNTER.labels(rating=str(req.rating)).inc()

    return FeedbackResponse(
        status="success",
        message="Thank you for your feedback! It will help improve future recommendations.",
        feedback_id=feedback_entry.id
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
