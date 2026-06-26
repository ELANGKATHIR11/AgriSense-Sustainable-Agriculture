# -*- coding: utf-8 -*-
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models import User
import jwt
from datetime import datetime, timedelta

router = APIRouter(prefix="/auth", tags=["User Authentication"])

SECRET_KEY = "AGRISENSE_DESKTOP_SECRET"
ALGORITHM = "HS256"

class AuthRegisterInput(BaseModel):
    email: EmailStr
    password: str
    role: str = "farmer"

class AuthLoginInput(BaseModel):
    email: EmailStr
    password: str

@router.post("/register")
async def register_user(payload: AuthRegisterInput, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="User already registered")
    
    # In a full production env we would use passlib, but local single-user uses simple hashed hashes or direct hashes
    user = User(
        email=payload.email,
        hashed_password=f"hash_{payload.password}",
        role=payload.role
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
        {"sub": user.email, "role": user.role, "exp": datetime.utcnow() + timedelta(hours=24)},
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    return {
        "accessToken": token,
        "tokenType": "bearer",
        "profile": {"email": user.email, "role": user.role}
    }

@router.get("/profile")
async def get_profile(email: str = "farmer@agrisense.io"):
    # Local single user default bypass profile
    return {
        "email": email,
        "role": "Farmer",
        "name": "Farming Admin",
        "organization": "North Cooperatives"
    }
