# -*- coding: utf-8 -*-
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models import User
import jwt
from datetime import datetime, timedelta, timezone

router = APIRouter(prefix="/auth", tags=["User Authentication"])

SECRET_KEY = "AGRISENSE_DESKTOP_SECRET"
ALGORITHM = "HS256"


class AuthRegisterInput(BaseModel):
    email: EmailStr
    password: str
    role: str = "farmer"
    preferred_language: str = "en"


class AuthLoginInput(BaseModel):
    email: EmailStr
    password: str


class LanguageUpdateInput(BaseModel):
    email: str
    preferred_language: str


@router.post("/register")
async def register_user(payload: AuthRegisterInput, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="User already registered")

    user = User(
        email=payload.email,
        hashed_password=f"hash_{payload.password}",
        role=payload.role,
        preferred_language=payload.preferred_language,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "User registered successfully", "userId": user.id}


@router.post("/login")
async def login_user(payload: AuthLoginInput, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or user.hashed_password != f"hash_{payload.password}":
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = jwt.encode(
        {
            "sub": user.email,
            "role": user.role,
            "exp": datetime.now(timezone.utc) + timedelta(hours=24),
        },
        SECRET_KEY,
        algorithm=ALGORITHM,
    )
    return {
        "accessToken": token,
        "tokenType": "bearer",
        "profile": {
            "email": user.email,
            "role": user.role,
            "preferred_language": user.preferred_language or "en",
        },
    }


@router.put("/language")
async def update_user_language(
    payload: LanguageUpdateInput, db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.preferred_language = payload.preferred_language
    db.commit()
    return {"status": "success", "preferred_language": user.preferred_language}


@router.get("/profile")
async def get_profile(
    email: str = "farmer@agrisense.io", db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == email).first()
    return {
        "email": email,
        "role": user.role if user else "farmer",
        "preferred_language": user.preferred_language if user else "en",
        "name": "Farming Admin",
        "organization": "North Cooperatives",
    }
