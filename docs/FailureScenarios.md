# AlexandrAI - Failure Scenarios & Resilient Fallbacks

## 1. Failure Scenario Matrix

| Component Failure | Consequence | Fallback Mechanism | User Experience Impact |
| :--- | :--- | :--- | :--- |
| **LLM Provider Outage / Timeout** | Asynchronous explanation generation times out or fails (>5.0s). | API Gateway intercepts timeout and generates structured explainability badges locally. | **Zero Downtime**: Top recommendations delivered instantly with template explanations (`✓ Recommended book`). |
| **Qdrant Vector DB Down** | Semantic ANN vector search fails or times out. | `Retrieval Service` catches exception and queries top rated books directly from PostgreSQL relational store. | **Degraded Recommendations**: Reverts to metadata & popularity-based matching instead of deep semantic match. |
| **Redis Queue / Cache Down** | Task queue or caching layer is unreachable. | Microservices bypass cache reads/writes and call downstream recommendation engine synchronously. | **Slight Latency Increase**: Recommendations delivered without caching benefits. |
| **PostgreSQL Outage** | User profiles or feedback writes fail. | In-memory cache returns anonymous user defaults; feedback events buffered into secondary log files. | **Read-Only Mode**: Users can still get recommendations without profile personalization updates. |

---

## 2. Structured Template Fallback Generator

When LLM service calls fail or exceed the 5.0 second deadline, `core/llm_provider.py` activates `_generate_fallback_explanation`:

```python
reasoning_bullets = [
    f"✓ Popular title in {candidate['genres'][0]}" if candidate.get('genres') else "✓ Highly rated book",
    f"✓ Fits your preferred page count ({candidate['page_count']} pages)" if candidate.get('page_count') else "✓ High user rating"
]
```

This guarantees that every API response maintains the structured JSON schema expectations of client applications without raising HTTP 500 errors.
