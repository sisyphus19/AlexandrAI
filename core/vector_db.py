import hashlib
import re
from typing import Any

import numpy as np

from shared.config import settings
from shared.logging import get_logger

logger = get_logger("vector-db-module")

# Attempt Qdrant import
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("qdrant-client not installed. Vector DB will run in local memory mode.")

# Attempt sentence-transformers import
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Using TF-IDF/fallback vectorizer.")


class EmbeddingManager:
    """Handles text embedding generation using sentence-transformers with graceful fallback."""
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded embedding model: {model_name}")
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer '{model_name}': {e}. Using fallback vectorizer.")

    def embed_text(self, text: str) -> list[float]:
        if self.model is not None:
            vector = self.model.encode(text, convert_to_numpy=True)
            return vector.tolist()
        return self._fallback_embed(text)

    @staticmethod
    def _fallback_embed(text: str) -> list[float]:
        """
        Offline fallback: deterministic hashed bag-of-words (feature hashing) vector.

        Uses md5 rather than Python's built-in hash(), which is salted per process: with hash() the ETL
        process and the API process produced *different* vectors for the same text, so retrieval silently
        returned noise whenever the real model could not be loaded. Texts sharing words get similar vectors.
        """
        dim = settings.VECTOR_DIMENSION
        vec = np.zeros(dim, dtype=np.float64)
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm == 0:
            vec[0] = 1.0
            norm = 1.0
        return (vec / norm).tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        if self.model is not None and texts:
            vectors = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
            return [v.tolist() for v in vectors]
        return [self.embed_text(t) for t in texts]


class VectorStoreManager:
    """
    Manages Qdrant vector database storage and retrieval,
    with an in-memory fallback for lightweight standalone execution.
    """
    def __init__(self, collection_name: str = settings.QDRANT_COLLECTION_NAME):
        self.collection_name = collection_name
        self.embedding_mgr = EmbeddingManager()
        self.qdrant_client = None
        self.memory_store: dict[int, dict[str, Any]] = {}  # Fallback: book_id -> {vector, payload}
        
        self._init_qdrant()

    def _init_qdrant(self):
        if not QDRANT_AVAILABLE:
            logger.info("Running VectorStoreManager in Local Memory mode.")
            return

        try:
            if settings.QDRANT_URL:
                self.qdrant_client = QdrantClient(url=settings.QDRANT_URL)
            else:
                self.qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
            
            # Check/create collection
            collections = [c.name for c in self.qdrant_client.get_collections().collections]
            if self.collection_name not in collections:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=settings.VECTOR_DIMENSION,
                        distance=qmodels.Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Connected to Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant server ({e}). Falling back to Local Memory Vector Store.")
            self.qdrant_client = None

    def points_count(self) -> int:
        """Number of indexed books (Qdrant collection size, or the in-memory store size)."""
        if self.qdrant_client is not None:
            return int(self.qdrant_client.count(collection_name=self.collection_name, exact=True).count)
        return len(self.memory_store)

    def clear(self) -> None:
        """Remove every indexed vector (used by `etl.py --reset`)."""
        self.memory_store.clear()
        if self.qdrant_client is not None:
            self.qdrant_client.delete_collection(collection_name=self.collection_name)
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(size=settings.VECTOR_DIMENSION, distance=qmodels.Distance.COSINE),
            )

    def upsert_books(self, books_data: list[dict[str, Any]]):
        """
        Upserts book metadata and text embeddings into Qdrant or Local Memory Store (idempotent by book id).
        Each book dict needs: id, title, author. Optional: genres, description, rating, page_count, publication_year.
        Missing numeric fields are stored as 0 (= unknown) rather than invented defaults.
        Raises ValueError if the embedding size does not match settings.VECTOR_DIMENSION.
        """
        if not books_data:
            return

        texts = [
            f"{b['title']} by {b['author']}. Genres: {', '.join(b.get('genres') or [])}. Description: {b.get('description') or ''}"
            for b in books_data
        ]
        vectors = self.embedding_mgr.embed_batch(texts)

        expected = settings.VECTOR_DIMENSION
        for vec in vectors:
            if len(vec) != expected:
                raise ValueError(
                    f"Embedding dimension {len(vec)} does not match VECTOR_DIMENSION={expected}. "
                    f"Check EMBEDDING_MODEL_NAME ('{self.embedding_mgr.model_name}') and VECTOR_DIMENSION in the settings."
                )

        points = []
        for book, vector in zip(books_data, vectors):
            payload = {
                "id": book["id"],
                "title": book["title"],
                "author": book["author"],
                "genres": book.get("genres") or [],
                "description": book.get("description") or "",
                "rating": book.get("rating") or 0.0,
                "page_count": book.get("page_count") or 0,
                "publication_year": book.get("publication_year") or 0,
            }
            if self.qdrant_client is not None:
                points.append(qmodels.PointStruct(id=book["id"], vector=vector, payload=payload))
            else:
                self.memory_store[book["id"]] = {"vector": np.array(vector, dtype=np.float32), "payload": payload}

        if self.qdrant_client is not None:
            self.qdrant_client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Upserted {len(points)} books into Qdrant collection '{self.collection_name}'.")
        else:
            logger.info(f"Upserted {len(books_data)} books into Local Memory Vector Store.")

    def search_similar(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """
        Searches for candidate books similar to the natural language query.
        Returns list of payload dicts containing metadata and 'semantic_similarity' score.
        """
        query_vector = self.embedding_mgr.embed_text(query)
        
        if self.qdrant_client is not None:
            try:
                response = self.qdrant_client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                    with_payload=True,
                )
                candidates = []
                for res in response.points:
                    item = dict(res.payload or {})
                    # cosine score is in [-1, 1]; clamp so downstream scoring always sees 0..1
                    item["semantic_similarity"] = max(0.0, min(1.0, float(res.score)))
                    candidates.append(item)
                
                logger.info(f"Search Query: {query}")
                
                for c in candidates[:10]:
                    logger.info(
                        f"{c['title']} | score={c['semantic_similarity']:.4f}"
                    )
                
                return candidates
            except Exception as e:
                logger.error(f"Qdrant search error ({e}). Falling back to memory store search.")

        # Local memory fallback cosine similarity search
        if not self.memory_store:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        norm_q = np.linalg.norm(q_vec)
        
        candidates = []
        for book_id, item in self.memory_store.items():
            b_vec = item["vector"]
            norm_b = np.linalg.norm(b_vec)
            if norm_q > 0 and norm_b > 0:
                sim = float(np.dot(q_vec, b_vec) / (norm_q * norm_b))
            else:
                sim = 0.0
            
            cand = dict(item["payload"])
            cand["semantic_similarity"] = max(0.0, min(1.0, (sim + 1.0) / 2.0))  # normalize to 0..1
            candidates.append(cand)

        candidates.sort(key=lambda x: x["semantic_similarity"], reverse=True)
        return candidates[:top_k]

vector_store = VectorStoreManager()
