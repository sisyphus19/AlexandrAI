import os
import sys
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import settings
from shared.logging import get_logger
from shared.schemas import UserRegisterRequest, UserLoginRequest, TokenResponse, UserPreferencesUpdate, UserProfileResponse
from core.database import get_db, UserModel, init_db
from core.auth import hash_password, verify_password, create_access_token, get_current_user

logger = get_logger("user-service")

app = FastAPI(title="AlexandrAI User Service", version="1.0.0")

@app.on_event("startup")
def startup():
    init_db()

@app.get("/health")
def health():
    return {"status": "healthy", "service": "user-service"}

@app.get("/health/live")
def live():
    return {"status": "alive", "service": "user-service"}

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post(f"{settings.API_PREFIX}/users/register", response_model=UserProfileResponse, status_code=201)
def register(req: UserRegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(UserModel).filter((UserModel.username == req.username) | (UserModel.email == req.email)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username or email already registered.")
    user = UserModel(
        username=req.username,
        email=req.email,
        hashed_password=hash_password(req.password),
        favorite_genres=[]
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@app.post(f"{settings.API_PREFIX}/users/login", response_model=TokenResponse)
def login(req: UserLoginRequest, db: Session = Depends(get_db)):
    user = db.query(UserModel).filter(UserModel.username == req.username).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    token = create_access_token(user_id=user.id, username=user.username)
    return TokenResponse(access_token=token, user_id=user.id, username=user.username)

@app.get(f"{settings.API_PREFIX}/users/me", response_model=UserProfileResponse)
def get_profile(current_user: UserModel = Depends(get_current_user)):
    return current_user

@app.put(f"{settings.API_PREFIX}/users/preferences", response_model=UserProfileResponse)
def update_preferences(prefs: UserPreferencesUpdate, current_user: UserModel = Depends(get_current_user), db: Session = Depends(get_db)):
    current_user.favorite_genres = prefs.favorite_genres
    if prefs.preferred_book_length:
        current_user.preferred_book_length = prefs.preferred_book_length
    if prefs.preferred_mood:
        current_user.preferred_mood = prefs.preferred_mood
    db.commit()
    db.refresh(current_user)
    return current_user

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
