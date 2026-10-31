import asyncio
import os
import sys
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import cache
from core.catalog_qa import answer_catalog_question, source_records
from llm_worker.worker import LLMQueueProducer
from shared.config import settings
from shared.logging import get_logger
from shared.schemas import (
    CatalogAskRequest,
    CatalogAskResponse,
    CatalogSource,
    HealthCheckResponse,
    RecommendationRequest,
    RecommendationResponse,
)

logger = get_logger("api-gateway")

app = FastAPI(
    title="AlexandrAI Recommendation API Gateway",
    description="Distributed API Gateway orchestrating microservices with end-to-end request tracing and resilient fallbacks.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GATEWAY_REQUESTS = Counter("gateway_requests_total", "Gateway requests count", ["method", "endpoint", "status"])
GATEWAY_LATENCY = Histogram("gateway_request_duration_seconds", "Gateway latency in seconds", ["endpoint"])

RECOMMEND_CACHE_TTL = 300  # seconds; only fully LLM-explained, non-degraded responses are cached

MSG_EMPTY_INDEX = ("No candidates were retrieved because the vector index is empty. "
                   "Load the catalog first (run `python etl.py`) and try again.")
MSG_RETRIEVAL_DOWN = "The retrieval service is unavailable right now, so no recommendations could be generated. Please try again shortly."
MSG_RANKING_DOWN = "The ranking service is unavailable right now, so no recommendations could be generated. Please try again shortly."
MSG_LLM_DOWN = "AI explanation unavailable right now (the LLM worker did not respond in time); showing ranked results without generated text."


@app.middleware("http")
async def tracing_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    start_time = time.time()

    response: Response = await call_next(request)
    latency_ms = (time.time() - start_time) * 1000
    response.headers["X-Request-ID"] = request_id

    GATEWAY_REQUESTS.labels(method=request.method, endpoint=request.url.path, status=response.status_code).inc()
    GATEWAY_LATENCY.labels(endpoint=request.url.path).observe(latency_ms / 1000.0)

    logger.info(f"Gateway {request.method} {request.url.path} -> {response.status_code}", extra={"request_id": request_id, "latency_ms": latency_ms})
    return response

@app.get("/health")
def health():
    return HealthCheckResponse(
        status="healthy",
        service="api-gateway",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        dependencies={"downstream_services": "ready"}
    )

@app.get("/health/live")
def live():
    return HealthCheckResponse(status="alive", service="api-gateway", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"))

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def _post_json(client: httpx.AsyncClient, url: str, payload: dict, headers: dict):
    """POST and return (json_body, error). Exactly one of the two is None."""
    try:
        res = await client.post(url, json=payload, headers=headers)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    if res.status_code != 200:
        return None, f"HTTP {res.status_code}"
    try:
        return res.json(), None
    except ValueError as e:
        return None, f"invalid JSON: {e}"


async def _wait_for_job(job_id: str, max_wait: float) -> dict | None:
    """Poll for the worker's result for up to `max_wait` seconds. Returns None on timeout."""
    deadline = time.monotonic() + max_wait
    while True:
        result = await asyncio.to_thread(LLMQueueProducer.get_result, job_id)
        if result:
            return result
        if time.monotonic() >= deadline:
            return None
        await asyncio.sleep(0.05)


def _degraded_recommendations(ranked: list[dict], top_k: int) -> list[dict]:
    """Ranked results without LLM text, used when the worker is down/slow/failed."""
    recs = []
    for c in ranked[:top_k]:
        sb = c["score_breakdown"]
        recs.append({
            "book": c,
            "final_score": sb["final_score"],
            "score_breakdown": sb,
            "reasoning_bullets": c.get("explainability_badges") or ["✓ Ranked by semantic match, rating and recency"],
            "llm_explanation": MSG_LLM_DOWN,
        })
    return recs


@app.post(f"{settings.API_PREFIX}/recommend", response_model=RecommendationResponse)
async def orchestrate_recommendation(req: RecommendationRequest, request: Request):
    start_time = time.time()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    headers = {"X-Request-ID": request_id}

    def elapsed_ms() -> float:
        return round((time.time() - start_time) * 1000, 2)

    # 0. Response cache (only successful, fully explained responses are ever stored)
    cache_key = cache.make_key(
        "recommend", query=req.query, genres=req.favorite_genres, mood=req.preferred_mood,
        max_pages=req.max_page_count, top_k=req.top_k,
    )
    cached = cache.get_json(cache_key)
    if cached:
        cached.update(request_id=request_id, execution_time_ms=elapsed_ms())
        return RecommendationResponse(**cached)

    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1. Retrieval Service
        candidates, err = await _post_json(
            client, f"{settings.RETRIEVAL_SERVICE_URL}{settings.API_PREFIX}/retrieve",
            {"query": req.query, "top_k": 20}, headers)
        if err:
            logger.warning(f"Retrieval Service call failed ({err}).")
            return RecommendationResponse(query=req.query, request_id=request_id, recommendations=[],
                                          total_candidates_retrieved=0, execution_time_ms=elapsed_ms(),
                                          message=MSG_RETRIEVAL_DOWN)
        if not candidates:
            # A 200 with nothing in it is not the same as an outage: the index simply has no books.
            logger.warning("Retrieval returned 0 candidates; vector index is probably empty.")
            return RecommendationResponse(query=req.query, request_id=request_id, recommendations=[],
                                          total_candidates_retrieved=0, execution_time_ms=elapsed_ms(),
                                          message=MSG_EMPTY_INDEX)

        # 2. Recommendation / Ranking Service
        ranked, err = await _post_json(
            client, f"{settings.RECOMMENDATION_SERVICE_URL}{settings.API_PREFIX}/rank",
            {"candidates": candidates, "favorite_genres": req.favorite_genres,
             "max_page_count": req.max_page_count, "top_k": req.top_k}, headers)
        if err or not ranked:
            logger.warning(f"Ranking Service call failed ({err or 'empty result'}).")
            return RecommendationResponse(query=req.query, request_id=request_id, recommendations=[],
                                          total_candidates_retrieved=len(candidates), execution_time_ms=elapsed_ms(),
                                          message=MSG_RANKING_DOWN)

    # 3. LLM explanations via the queue worker, with a bounded wait
    job_id = f"job-{uuid.uuid4().hex[:16]}"
    LLMQueueProducer.enqueue_job({
        "job_id": job_id,
        "query": req.query,
        "candidates": ranked,
        "genres": req.favorite_genres,
        "mood": req.preferred_mood,
    })
    result = await _wait_for_job(job_id, settings.LLM_JOB_WAIT_SECONDS)

    if result and result.get("status") == "completed":
        response = RecommendationResponse(
            query=req.query,
            request_id=request_id,
            recommendations=result["explained_recommendations"],
            total_candidates_retrieved=len(candidates),
            execution_time_ms=elapsed_ms(),
            llm_fallback_used=result.get("fallback_used", False),
        )
        if not response.llm_fallback_used:
            cache.set_json(cache_key, response.model_dump(mode="json"), ttl=RECOMMEND_CACHE_TTL)
        return response

    # Worker slow, down, or the job failed: still return the ranked books, clearly labelled.
    reason = f"job failed: {result.get('error')}" if result else "timed out waiting for worker"
    logger.warning(f"LLM job {job_id} degraded ({reason}).")
    return RecommendationResponse(
        query=req.query,
        request_id=request_id,
        recommendations=_degraded_recommendations(ranked, req.top_k),
        total_candidates_retrieved=len(candidates),
        execution_time_ms=elapsed_ms(),
        llm_fallback_used=True,
        message=MSG_LLM_DOWN,
    )


@app.post(f"{settings.API_PREFIX}/catalog/ask", response_model=CatalogAskResponse)
async def catalog_ask(req: CatalogAskRequest, request: Request):
    """Metadata-grounded catalog Q&A (title / author / rating / year only; no book text is indexed)."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    async with httpx.AsyncClient(timeout=10.0) as client:
        retrieved, err = await _post_json(
            client, f"{settings.RETRIEVAL_SERVICE_URL}{settings.API_PREFIX}/retrieve",
            {"query": req.question, "top_k": req.top_k}, {"X-Request-ID": request_id})
    if err:
        raise HTTPException(status_code=503, detail=MSG_RETRIEVAL_DOWN)

    answer = await answer_catalog_question(req.question, retrieved)
    return CatalogAskResponse(
        question=req.question,
        answer=answer,
        sources=[CatalogSource(**s) for s in source_records(retrieved)],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
