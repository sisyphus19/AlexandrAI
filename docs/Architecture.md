# AlexandrAI - System Architecture Specifications

## 1. Overview & Capability-Driven Design

AlexandrAI is structured as a distributed microservice architecture designed to handle semantic vector search, high-throughput feature scoring, and decoupled asynchronous LLM reasoning.

```
+-----------------------------------------------------------------------------------+
|                                 CLIENT DASHBOARD                                  |
+------------------------------------------+----------------------------------------+
                                           |
                                           | HTTP Requests (with X-Request-ID)
                                           v
+-----------------------------------------------------------------------------------+
|                                   API GATEWAY                                     |
|  - Middleware: Request ID generation, Prometheus Metrics, Rate Limiting           |
|  - Prefix: /api/v1/                                                               |
+--------+------------------+-----------------------+----------------------+--------+
         |                  |                       |                      |
         v                  v                       v                      v
+------------------+ +---------------+   +--------------------+   +-------------------+
|  USER SERVICE    | | BOOK SERVICE  |   | RETRIEVAL SERVICE  |   | RECOMMENDATION SVC|
| - Auth JWT       | | - Metadata    |   | - Qdrant Vector    |   | - Candidate Filter|
| - Preferences    | | - Catalog CRUD|   |   Retrieval        |   | - Feature Extract |
| - Feedback Loop  | | - Ratings     |   | - Embeddings Cache |   | - Scoring Engine  |
+--------+---------+ +-------+-------+   +---------+----------+   +---------+---------+
         |                   |                     |                        |
         v                   v                     v                        v
+------------------+ +---------------+   +--------------------+   +-------------------+
|  PostgreSQL DB   | | PostgreSQL DB |   |  Qdrant Vector DB  |   | Redis Task Queue  |
+------------------+ +---------------+   +--------------------+   +---------+---------+
                                                                            |
                                                                            v
                                                                  +-------------------+
                                                                  |  LLM ASYNC WORKER |
                                                                  | - Prompt Build    |
                                                                  | - Structured Badges|
                                                                  | - Instant Fallback|
                                                                  +-------------------+
```

---

## 2. Distributed Request Tracing & Observability

Every request entering the platform is assigned a unique `UUIDv4` `X-Request-ID` at the API Gateway layer if not present. This header is passed down to all downstream HTTP requests.

All services format log entries into structured JSON:

```json
{
  "timestamp": "2026-07-29T12:00:00Z",
  "service": "retrieval-service",
  "severity": "INFO",
  "request_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "logger": "retrieval-service",
  "message": "Retrieved 20 candidate vectors in 12.4ms",
  "latency_ms": 12.4
}
```

---

## 3. End-to-End Recommendation Dataflow

1. **User Request**: User sends query `POST /api/v1/recommend` with optional filters (genres, max page count, user_id).
2. **Retrieval Phase**: `API Gateway` proxies query to `Retrieval Service`, which computes sentence embeddings (`all-MiniLM-L6-v2`) and executes ANN vector search against `Qdrant` collection.
3. **Hybrid Reranking**: Candidates are sent to `Recommendation Service` where candidate filtering, feature extraction (genre affinity, publication year, popularity), and 4-factor formula scoring take place.
4. **Async LLM Queue**: `API Gateway` pushes the top candidate list to `Redis Task Queue` (`llm_explanation_jobs`).
5. **Reasoning & Explanation**: `LLM Worker` pops the job, generates bullet-point reasoning badges (`✓ shares philosophical sci-fi themes`) and LLM explanation summaries, saving the result back to Redis cache.
6. **Response Delivery**: If worker finishes within timeout, full reasoning is returned; otherwise, structured template fallback is returned instantly.
