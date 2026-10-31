import os
import sys
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import settings
from shared.logging import get_logger
from shared.schemas import BookResponse
from core.database import get_db, BookModel, init_db

logger = get_logger("book-service")

app = FastAPI(title="AlexandrAI Book Metadata Service", version="1.0.0")

@app.on_event("startup")
def startup():
    init_db()

@app.get("/health")
def health():
    return {"status": "healthy", "service": "book-service"}

@app.get("/health/live")
def live():
    return {"status": "alive", "service": "book-service"}

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get(f"{settings.API_PREFIX}/books", response_model=List[BookResponse])
def list_books(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    return db.query(BookModel).offset(skip).limit(limit).all()

@app.get(f"{settings.API_PREFIX}/books/{{book_id}}", response_model=BookResponse)
def get_book(book_id: int, db: Session = Depends(get_db)):
    book = db.query(BookModel).filter(BookModel.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return book

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
