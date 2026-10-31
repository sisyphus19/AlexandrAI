import argparse
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.database import BookModel, SessionLocal, init_db
from core.vector_db import vector_store
from shared.logging import get_logger

logger = get_logger("etl-pipeline")

BOOKS_CSV = Path("data/raw/Books.csv")
DEV_LIMIT = 1000  # Default number of rows to ingest; use `--all` to ingest the full dataset

# Column limits of the `books` table (core/database.py). Longer values would make Postgres reject the row.
MAX_TITLE = 255
MAX_AUTHOR = 255
MAX_COVER_URL = 500

REQUIRED_COLUMNS = ("book_id", "Title")


def _clean_text(value: Any, max_len: int) -> str | None:
    """Strip + clip a CSV cell; returns None for NaN/blank."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text[:max_len] if text else None


def _build_description(title: str, author: str, rating: float, year: int) -> str:
    """Synthetic text used for embeddings. Only states facts the dataset actually has."""
    parts = [f"Title: {title}", f"Author: {author}"]
    if rating > 0:
        parts.append(f"Average reader rating: {rating:.2f} out of 5.")
    if year > 0:
        parts.append(f"First published in {year}.")
    return "\n".join(parts)


def load_books_from_csv(limit: int | None = None) -> list[dict[str, Any]]:
    """
    Load the dataset from CSV and convert it into the internal Book schema.

    - Validates required columns (raises ValueError otherwise).
    - Drops rows with a missing/blank title or an invalid / duplicate book_id (first occurrence wins).
    - Clips text to the DB column limits and clamps ratings to 0..5.
    - `limit` caps the number of *valid* books returned.
    """
    df = pd.read_csv(BOOKS_CSV)

    # Remove leading/trailing spaces from column names
    df.columns = df.columns.str.strip()
    logger.info(f"CSV Columns: {list(df.columns)}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{BOOKS_CSV} is missing required columns: {missing}")

    books: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    skipped = 0

    for _, row in df.iterrows():
        if limit is not None and len(books) >= limit:
            break

        try:
            book_id = int(row["book_id"])
        except (TypeError, ValueError):
            skipped += 1
            continue

        title = _clean_text(row["Title"], MAX_TITLE)
        if title is None or book_id in seen_ids:
            skipped += 1
            continue
        seen_ids.add(book_id)

        author = _clean_text(row.get("Author"), MAX_AUTHOR) or "Unknown"

        rating = row.get("AvgRating")
        rating = 0.0 if rating is None or pd.isna(rating) else float(rating)
        rating = max(0.0, min(5.0, rating))

        year = row.get("Publication Year")
        year = 0 if year is None or pd.isna(year) else int(year)

        books.append(
            {
                "id": book_id,
                "title": title,
                "author": author,
                "genres": [],
                "description": _build_description(title, author, rating, year),
                "rating": rating,
                "page_count": 0,  # dataset has no page counts; 0 means "unknown"
                "publication_year": year,
                "cover_image_url": _clean_text(row.get("Image-URL"), MAX_COVER_URL),
            }
        )

    logger.info(f"Loaded {len(books)} books from CSV ({skipped} rows skipped: blank title, bad or duplicate id).")
    return books


def run_etl(limit: int | None = DEV_LIMIT, reset: bool = False) -> int:
    """
    Executes the AlexandrAI ETL pipeline. Safe to re-run: books are upserted by id and nothing is deleted
    unless `reset=True`.

    1. Initialize database
    2. (optional) Clear existing books + vectors
    3. Upsert book metadata
    4. Generate embeddings and upsert vectors into Qdrant

    Returns the number of books processed.
    """
    logger.info("Starting AlexandrAI ETL Pipeline...")
    init_db()

    books = load_books_from_csv(limit=limit)

    db = SessionLocal()
    try:
        if reset:
            logger.warning("--reset: deleting ALL existing books and vectors.")
            db.query(BookModel).delete()
            db.commit()
            vector_store.clear()

        logger.info(f"Upserting {len(books)} books into database...")
        for item in books:
            db.merge(
                BookModel(
                    id=item["id"],
                    title=item["title"],
                    author=item["author"],
                    genres=item["genres"],
                    description=item["description"],
                    rating=item["rating"],
                    page_count=item["page_count"],
                    publication_year=item["publication_year"],
                    cover_image_url=item["cover_image_url"],
                )
            )
        db.commit()
        logger.info(f"Successfully upserted {len(books)} books into the database.")

        logger.info("Generating embeddings and indexing Vector DB...")
        vector_store.upsert_books(books)
        logger.info("Vector database indexing completed.")

    except Exception as e:
        db.rollback()
        logger.exception(f"ETL failed: {e}")
        raise
    finally:
        db.close()

    logger.info("ETL pipeline completed successfully.")
    return len(books)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Load the Books.csv dataset into the database and vector index.")
    parser.add_argument("--limit", type=int, default=DEV_LIMIT, help=f"max books to ingest (default {DEV_LIMIT})")
    parser.add_argument("--all", action="store_true", help="ingest the full dataset (ignores --limit)")
    parser.add_argument("--reset", action="store_true", help="DELETE all existing books and vectors before loading")
    parser.add_argument("--yes", action="store_true", help="confirm destructive actions such as --reset")
    args = parser.parse_args(argv)

    if args.reset and not args.yes:
        parser.error("--reset deletes ALL books and vectors; re-run with --yes to confirm.")

    run_etl(limit=None if args.all else args.limit, reset=args.reset)


if __name__ == "__main__":
    main()
