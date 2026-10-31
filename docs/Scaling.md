# AlexandrAI - Scaling & Performance Strategy

## 1. Horizontal Pod Autoscaler (HPA)

Retrieval and LLM execution are computationally heavier than standard CRUD operations. Horizontal Pod Autoscalers monitor CPU and memory utilization to dynamically scale workloads:

- **Retrieval Service HPA**: Scales between 2 and 10 replicas based on target CPU utilization (75%).
- **LLM Worker HPA**: Scales between 2 and 15 replicas based on queue depth and worker CPU load.

---

## 2. Qdrant High-Dimensional Vector Search Indexing

To maintain sub-20ms vector retrieval latencies as candidate catalogs scale beyond 1,000,000 books:
- **HNSW Indexing**: `m=16`, `ef_construct=100` for optimal index build vs search latency tradeoff.
- **Payload Metadata Indexing**: Indexes constructed on `genres`, `publication_year`, and `page_count` allowing Qdrant to perform pre-filtering before cosine similarity calculation.

---

## 3. Redis Queue Bottleneck Mitigation

When recommendation traffic spikes:
1. **Parallel Worker Pools**: Multiple `llm-worker` pods consume from `llm_explanation_jobs` using non-blocking Redis `BRPOP`.
2. **Job Deduplication**: Repeated queries derive explainability badges directly from Redis key-value cache (`recommendation:cache:<query_hash>`).
3. **Queue TTL Expiration**: Stale LLM jobs expire automatically after 30 seconds to prevent worker queue backlog.
