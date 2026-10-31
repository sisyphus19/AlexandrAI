import json
from abc import ABC, abstractmethod
from typing import Any

import httpx

from shared.config import settings
from shared.logging import get_logger
from shared.schemas import BookResponse, ExplainedRecommendation, RankedCandidate

logger = get_logger("llm-provider")

# NOTE: literal JSON braces are doubled ({{ }}) so str.format() only substitutes
# {query}, {genres}, {mood} and {books_summary}.
PROMPT_TEMPLATE = """
System: You are AlexandrAI, an expert literary recommendation assistant.
User Query: "{query}"
User Preferences: Genres: {genres}, Preferred Mood: {mood}

Candidate Books:
{books_summary}

Task:
Analyze the user's intent and explain why each book is recommended.
Use ONLY the facts given for each book above.
Do NOT invent or guess any detail marked "unknown" or "none available" (no made-up page counts, years, plot details or themes).
Provide a concise, engaging explanation paragraph and 3 bullet points starting with '✓' highlighting why this book matches.

Return JSON in format:
{{
  "explanations": [
    {{
      "book_id": <id>,
      "reasoning_bullets": ["✓ ...", "✓ ...", "✓ ..."],
      "explanation": "..."
    }}
  ]
}}
"""


def _fmt_book_line(c) -> str:
    """One prompt line per candidate. Missing fields are marked unknown so the LLM can't invent them."""
    pages = c.page_count if getattr(c, "page_count", 0) and c.page_count > 0 else "unknown"
    year = c.publication_year if getattr(c, "publication_year", 0) and c.publication_year > 0 else "unknown"
    desc = (getattr(c, "description", "") or "").strip()
    desc = desc[:300] if desc else "none available"
    genres = ", ".join(c.genres) if getattr(c, "genres", None) else "unknown"
    return (
        f"ID: {c.id} | Title: {c.title} | Author: {c.author} | Genres: {genres} | "
        f"Rating: {c.rating} | Pages: {pages} | Year: {year} | Description: {desc}"
    )


class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate_explanations(
        self,
        query: str,
        ranked_candidates: list[RankedCandidate],
        genres: list[str] | None = None,
        mood: str | None = None
    ) -> list[dict[str, Any]]:
        pass


class LocalMockProvider(BaseLLMProvider):
    """
    Offline / zero-cost explanation engine.
    Only states facts that are actually present on the candidate (no fabricated pages, themes or plot).
    """
    async def generate_explanations(
        self,
        query: str,
        ranked_candidates: list[RankedCandidate],
        genres: list[str] | None = None,
        mood: str | None = None
    ) -> list[dict[str, Any]]:
        results = []
        for item in ranked_candidates:
            bullets = list(item.explainability_badges)
            if not bullets:
                bullets = [f"✓ Matches search intent for '{query}'"]
                if item.rating:
                    bullets.append(f"✓ Highly rated at {item.rating:.1f} stars")
                if item.genres:
                    bullets.append(f"✓ Falls under {', '.join(item.genres[:2])}")

            parts = [f"'{item.title}' by {item.author} is a match for your request '{query}'."]
            if item.genres:
                parts.append(f"It is categorised under {', '.join(item.genres[:2])}.")

            desc = (item.description or "").strip()
            if desc:
                parts.append(f"About the book: {desc[:200]}")
            else:
                parts.append(
                    "No description is available for this book, so this pick is based on its rating "
                    "and how closely it matches your search."
                )

            if item.page_count and item.page_count > 0:
                parts.append(f"It runs {item.page_count} pages.")

            results.append({
                "book_id": item.id,
                "reasoning_bullets": bullets[:4],
                "explanation": " ".join(parts)
            })
        return results


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = settings.OPENAI_MODEL):
        self.api_key = api_key
        self.model = model
        self.mock = LocalMockProvider()

    async def generate_explanations(
        self,
        query: str,
        ranked_candidates: list[RankedCandidate],
        genres: list[str] | None = None,
        mood: str | None = None
    ) -> list[dict[str, Any]]:
        if not self.api_key:
            logger.info("OpenAI API key missing. Using local mock generator.")
            return await self.mock.generate_explanations(query, ranked_candidates, genres, mood)

        books_summary = "\n".join(_fmt_book_line(c) for c in ranked_candidates)

        prompt = PROMPT_TEMPLATE.format(
            query=query,
            genres=", ".join(genres) if genres else "None specified",
            mood=mood if mood else "Any",
            books_summary=books_summary
        )

        # Failures (timeout, non-200, invalid JSON) are deliberately NOT swallowed here:
        # LLMReasoningService catches them, switches to the fallback provider and sets fallback_used=True.
        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful JSON-only literary assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "response_format": {"type": "json_object"}
                }
            )
        if response.status_code != 200:
            raise RuntimeError(f"OpenAI returned HTTP {response.status_code}")
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content).get("explanations", [])


class OllamaProvider(BaseLLMProvider):
    def __init__(self, base_url: str = settings.OLLAMA_BASE_URL, model: str = settings.OLLAMA_MODEL):
        self.base_url = base_url
        self.model = model
        self.mock = LocalMockProvider()

    async def generate_explanations(
        self,
        query: str,
        ranked_candidates: list[RankedCandidate],
        genres: list[str] | None = None,
        mood: str | None = None
    ) -> list[dict[str, Any]]:
        books_summary = "\n".join(_fmt_book_line(b) for b in ranked_candidates)
        prompt = PROMPT_TEMPLATE.format(
            query=query,
            genres=", ".join(genres) if genres else "None",
            mood=mood if mood else "Any",
            books_summary=books_summary
        )

        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
            res = await client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "format": "json", "stream": False}
            )
        if res.status_code != 200:
            raise RuntimeError(f"Ollama returned HTTP {res.status_code}")
        parsed = json.loads(res.json().get("response", "{}"))
        return parsed.get("explanations", [])


class LLMReasoningService:
    """Manager class instantiating appropriate provider and executing explanation pipeline with fallbacks."""
    def __init__(self):
        provider_type = settings.LLM_PROVIDER.lower()
        if provider_type == "openai" and settings.OPENAI_API_KEY:
            self.provider = OpenAIProvider(api_key=settings.OPENAI_API_KEY)
            self.provider_name = "openai"
        elif provider_type == "ollama":
            self.provider = OllamaProvider()
            self.provider_name = "ollama"
        else:
            self.provider = LocalMockProvider()
            self.provider_name = "mock"

        self.fallback_provider = LocalMockProvider()

    async def get_explained_recommendations(
        self,
        query: str,
        ranked_candidates: list[RankedCandidate],
        genres: list[str] | None = None,
        mood: str | None = None
    ) -> tuple[list[ExplainedRecommendation], bool]:
        fallback_used = False
        try:
            raw_explanations = await self.provider.generate_explanations(query, ranked_candidates, genres, mood)
            expl_map = {item["book_id"]: item for item in raw_explanations if "book_id" in item}
        except Exception as e:
            logger.error(f"LLM Provider execution exception ({e}). Activating fallback generator.")
            fallback_used = True
            raw_explanations = await self.fallback_provider.generate_explanations(query, ranked_candidates, genres, mood)
            expl_map = {item["book_id"]: item for item in raw_explanations if "book_id" in item}

        explained_results = []
        for cand in ranked_candidates:
            expl_data = expl_map.get(cand.id)
            if not expl_data:
                fallback_used = True
                fallback_items = await self.fallback_provider.generate_explanations(query, [cand], genres, mood)
                expl_data = fallback_items[0]

            book_resp = BookResponse(
                id=cand.id,
                title=cand.title,
                author=cand.author,
                genres=cand.genres,
                description=cand.description,
                rating=cand.rating,
                page_count=cand.page_count,
                publication_year=cand.publication_year
            )

            explained_results.append(ExplainedRecommendation(
                book=book_resp,
                final_score=cand.score_breakdown.final_score,
                score_breakdown=cand.score_breakdown,
                reasoning_bullets=expl_data.get("reasoning_bullets", cand.explainability_badges),
                llm_explanation=expl_data.get("explanation", f"Recommended for query: '{query}'")
            ))

        return explained_results, fallback_used

llm_service = LLMReasoningService()
