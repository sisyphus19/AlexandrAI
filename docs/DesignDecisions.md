# AlexandrAI - Design Decisions & Tradeoffs

## 1. Multi-Factor Hybrid Reranking Formula

Pure semantic similarity often fails to capture individual user constraints such as length preference, author popularity, or recency. AlexandrAI implements a weighted hybrid scoring model:

$$\text{Final Score} = w_1 \cdot S_{\text{semantic}} + w_2 \cdot S_{\text{genre}} + w_3 \cdot S_{\text{popularity}} + w_4 \cdot S_{\text{recency}}$$

Where weights are empirically tuned for precision and diversity:
- $w_1 = 0.45$: High-dimensional semantic vector similarity (cosine distance from Qdrant embeddings).
- $w_2 = 0.25$: Genre affinity overlap between user preferences and candidate metadata.
- $w_3 = 0.15$: Prior popularity score derived from global rating averages and review counts normalized to $[0, 1]$.
- $w_4 = 0.15$: Recency factor favoring newer publications or timeless classics via exponential decay curve.

---

## 2. Redis Dual-Role Architecture

Instead of deploying separate operational systems for queue management (e.g. RabbitMQ / Celery) and key-value caching (e.g. Memcached), single-node/clustered **Redis** is leveraged in a dual-role:
1. **Async Job Queue**: Using Redis lists (`LPUSH` / `BRPOP`) for low-latency job distribution to `llm-worker` pods.
2. **Distributed Cache**: Storing computed vector search results, user profile state, and generated LLM badge summaries with automatic TTL expiration (1 hour).

---

## 3. Database Selection & Relational Fallback

- **PostgreSQL**: Chosen for transactional metadata integrity (`users`, `books`, `user_preferences`, `feedback`, `interaction_history`).
- **Qdrant**: Selected as vector store due to native HNSW indexing speed, payload metadata filtering, and low memory overhead.
- **SQLite In-Memory Fallback**: When external PostgreSQL or Qdrant cluster connections are absent (e.g., during local developer testing or offline test suites), `core/database.py` and `core/vector_db.py` fallback gracefully to SQLite and Python in-memory vector indexing.

---

## 4. API Versioning (`/api/v1/`)

To prevent breaking mobile or frontend client contracts during microservice iterations, all endpoints enforce strict prefixing (`/api/v1/`). Pydantic V2 models validate input schemas and guarantee backward compatibility.
