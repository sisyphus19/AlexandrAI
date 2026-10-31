# Changes in this revision (test-suite fixes + rename)

- **Renamed** the project from BookWise AI / `bookwiseAI` to **AlexandrAI** (docs, UI titles, service names, Docker container and image names, Helm chart `infra/helm/alexandrai`, Kubernetes namespace `alexandrai`, database `alexandrai_db`, SQLite file `alexandrai_local.db`).
- `core/llm_provider.py`: `_fmt_book_line` is a module-level function (it was nested inside `OllamaProvider`); prompt braces are escaped; the OpenAI path used an undefined variable; providers now raise on failure so `fallback_used` is reported correctly; mock explanations no longer invent page counts or themes; `LLMReasoningService.provider_name` added (the worker reads it).
- `core/recommendation_engine.py`: unknown page count / year (0) no longer produce badges, defaults or a fake recency score.
- `core/vector_db.py`: `points_count()` and `clear()`; embedding size is validated against `VECTOR_DIMENSION`; searches use `query_points`; offline fallback embedder is deterministic across processes (it used the per-process-salted `hash()`).
- `etl.py`: required-column check, de-duplication, blank-title rows dropped, clipping to column limits, rating clamp, idempotent upsert (no more delete-all), CLI: `--limit`, `--all`, `--reset --yes`.
- `api-gateway/main.py`: distinguishes empty index / retrieval outage / ranking outage, waits `LLM_JOB_WAIT_SECONDS` for the worker, degrades to ranked results when the worker is down, response cache, `POST /catalog/ask`.
- `shared`: `LLM_JOB_WAIT_SECONDS` setting, placeholder `OPENAI_API_KEY` values (`your_...`) are treated as unset, `message` field on `RecommendationResponse`, Pydantic v2 `json_schema_extra` / `ConfigDict`.
- `pytest.ini` added; `qdrant-client>=1.10.0` (needed for `query_points`).

---

# Changes in this revision

**Root causes found**
1. `postgresql://` URLs silently failed under SQLAlchemy 2.1 (default driver became psycopg v3; only psycopg2 installed) -> every service/ETL quietly fell back to private SQLite. Fixed: URL normalised to `postgresql+psycopg2://`, `ALLOW_SQLITE_FALLBACK=false` in containers, `<2.2` pin.
2. `PROMPT_TEMPLATE.format()` crashed on unescaped JSON braces, so the OpenAI/Ollama path always fell back to the mock. Fixed.
3. Gateway waited ~1 s for the worker; any real LLM call lost the race. Now waits `LLM_JOB_WAIT_SECONDS` (25 s) and worker publishes failures.
4. Compose never passed `OPENAI_API_KEY`/`LLM_PROVIDER` through (worker hard-coded `local`; gateway runs catalog QA but had no key).
5. ETL deleted all books *before* inserting; 6 authors exceed VARCHAR(255) so Postgres would have failed mid-run leaving an empty table. ETL is now an idempotent upsert with clipping, `--reset --yes`, `--dry-run`, pre-flight and post-run verification.
6. Mock explanations and ranker badges fabricated content ("rich narrative depth", "Quick read (0 pages)"). Now metadata-only and honest.
7. In local mode the in-memory vector store was empty on API start, so `/recommend` returned the first 20 rows. Now rebuilt from the DB at startup.

**Added**: `etl` compose service, `.dockerignore`, `.env.example`, Redis response cache, gateway `/catalog/ask` + `/stats`, retrieval `/health` with `points_count`, `scripts/eval_pipeline.py`, `tests/test_rag_pipeline.py` (19 new tests).
