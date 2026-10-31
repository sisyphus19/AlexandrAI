from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

# --- Auth & User Schemas ---

class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, json_schema_extra={"example": "bookworm42"})
    email: EmailStr = Field(..., json_schema_extra={"example": "user@example.com"})
    password: str = Field(..., min_length=6, json_schema_extra={"example": "SecurePass123!"})

class UserLoginRequest(BaseModel):
    username: str = Field(..., json_schema_extra={"example": "bookworm42"})
    password: str = Field(..., json_schema_extra={"example": "SecurePass123!"})

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str

class UserPreferencesUpdate(BaseModel):
    favorite_genres: list[str] = Field(default_factory=list, json_schema_extra={"example": ["Sci-Fi", "Philosophy"]})
    preferred_book_length: str | None = Field(None, json_schema_extra={"example": "medium"})  # short (<300), medium (300-500), long (>500)
    preferred_mood: str | None = Field(None, json_schema_extra={"example": "thought-provoking"})

class UserProfileResponse(BaseModel):
    id: int
    username: str
    email: str
    favorite_genres: list[str]
    preferred_book_length: str | None
    preferred_mood: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

# --- Book Schemas ---

class BookBase(BaseModel):
    title: str
    author: str
    genres: list[str]
    description: str
    rating: float = Field(..., ge=0.0, le=5.0)
    page_count: int
    publication_year: int
    cover_image_url: str | None = None

class BookResponse(BookBase):
    id: int
    reviews_summary: str | None = None
    created_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)

# --- Retrieval Schemas ---

class RetrievalRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "philosophical sci-fi books like Dune"})
    top_k: int = Field(default=20, ge=1, le=50)
    genres: list[str] | None = None
    max_page_count: int | None = None

class CandidateBook(BaseModel):
    id: int
    title: str
    author: str
    genres: list[str]
    description: str
    rating: float
    page_count: int
    publication_year: int
    semantic_similarity: float

# --- Reranking & Recommendation Schemas ---

class ScoreBreakdown(BaseModel):
    semantic_score: float
    genre_affinity: float
    popularity_prior: float
    recency_factor: float
    final_score: float

class RankedCandidate(BaseModel):
    id: int
    title: str
    author: str
    genres: list[str]
    description: str
    rating: float
    page_count: int
    publication_year: int
    score_breakdown: ScoreBreakdown
    explainability_badges: list[str] = Field(default_factory=list)

class RecommendationRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "Suggest fantasy books with political intrigue but less violence"})
    user_id: int | None = None
    favorite_genres: list[str] | None = None
    preferred_mood: str | None = None
    max_page_count: int | None = None
    top_k: int = Field(default=5, ge=1, le=20)

class ExplainedRecommendation(BaseModel):
    book: BookResponse
    final_score: float
    score_breakdown: ScoreBreakdown
    reasoning_bullets: list[str]  # e.g., ["✓ shares philosophical sci-fi themes", "✓ medium-length novel (~280 pages)"]
    llm_explanation: str

class RecommendationResponse(BaseModel):
    query: str
    request_id: str
    recommendations: list[ExplainedRecommendation]
    total_candidates_retrieved: int
    execution_time_ms: float
    llm_fallback_used: bool = False
    message: str | None = None  # human-readable status when results are empty or degraded

# --- Feedback Schema ---

class FeedbackRequest(BaseModel):
    recommendation_id: str | None = None
    book_id: int
    rating: int = Field(..., ge=1, le=5, json_schema_extra={"example": 5})  # 1 to 5 stars or thumbs up/down mapped
    is_helpful: bool = True
    comment: str | None = None

class FeedbackResponse(BaseModel):
    status: str = "success"
    message: str = "Feedback recorded successfully"
    feedback_id: int

# --- Health & Metrics Schemas ---

class HealthCheckResponse(BaseModel):
    status: str = "healthy"
    service: str
    timestamp: datetime
    version: str = "1.0.0"
    dependencies: dict[str, str] = Field(default_factory=dict)

# --- Catalog Q&A (metadata-grounded) ---

class CatalogAskRequest(BaseModel):
    question: str = Field(..., min_length=1, json_schema_extra={"example": "Which books are written by Suzanne Collins?"})
    top_k: int = Field(default=5, ge=1, le=20)

class CatalogSource(BaseModel):
    book_id: int
    title: str
    author: str
    rating: float
    publication_year: int

class CatalogAskResponse(BaseModel):
    question: str
    answer: str
    sources: list[CatalogSource]
