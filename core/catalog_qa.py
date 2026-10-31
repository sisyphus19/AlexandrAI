"""Metadata-grounded catalog QA. This is intentionally not full-text book RAG."""
import json
from typing import Any

import httpx

from shared.config import settings
from shared.logging import get_logger

logger = get_logger("catalog-qa")


def _source_record(book: dict[str, Any]) -> dict[str, Any]:
    return {
        "book_id": int(book.get("id", 0)),
        "title": str(book.get("title") or "Unknown title"),
        "author": str(book.get("author") or "Unknown author"),
        "rating": float(book.get("rating") or 0.0),
        "publication_year": int(book.get("publication_year") or 0),
    }


def source_records(retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Public helper: the citation records (id/title/author/rating/year) for a set of retrieved books."""
    return [_source_record(book) for book in retrieved]


async def answer_catalog_question(question: str, retrieved: list[dict[str, Any]]) -> str:
    """Answer only from retrieved metadata; fall back safely when OpenAI is unavailable."""
    sources = [_source_record(book) for book in retrieved]
    if not sources:
        return "I couldn't find matching books in the current catalog. Try a title, author, rating, or publication-year query."

    if not settings.OPENAI_API_KEY:
        titles = "; ".join(f"{s['title']} by {s['author']} (rating {s['rating']:.2f}/5)" for s in sources[:5])
        return ("OpenAI is not configured, so this is a metadata-only retrieval result rather than a generated answer. "
                f"Closest catalog matches: {titles}. The dataset does not contain book descriptions or full text.")

    prompt = {
        "question": question,
        "retrieved_catalog_records": sources,
        "constraints": [
            "Use only the supplied catalog records.",
            "The records contain title, author, average reader rating, and publication year only.",
            "Do not invent genres, plot details, themes, characters, page counts, or summaries.",
            "If the question requires book content or information absent from metadata, clearly say the catalog cannot answer it.",
            "Refer to supporting books by their exact title and author. Be concise."
        ]
    }
    try:
        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
                json={
                    "model": settings.OPENAI_MODEL,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "system", "content": "You answer questions about a book catalog. Treat catalog data as the only evidence; never claim to have read the books."},
                        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
                    ]
                }
            )
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"].strip()
            if answer:
                return answer
    except Exception as exc:
        logger.warning("Catalog QA OpenAI request failed; using metadata fallback: %s", exc)

    titles = "; ".join(f"{s['title']} by {s['author']} (rating {s['rating']:.2f}/5)" for s in sources[:5])
    return f"I couldn't generate an LLM answer. Retrieved catalog matches: {titles}. Only metadata is available; plot details cannot be verified."
