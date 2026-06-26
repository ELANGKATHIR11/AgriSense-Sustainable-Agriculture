# -*- coding: utf-8 -*-
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Dict, Any

from backend.database import get_db
from backend.market_intelligence.models import MarketPrice, GovernmentUpdate, AgricultureNews
from backend.market_intelligence.scheduler import start_scheduler
from backend.market_intelligence.websocket import router as ws_router

router = APIRouter(prefix="/market", tags=["Market Intelligence"])

# Include the websocket router
router.include_router(ws_router)

@router.get("/prices")
async def get_all_prices(db: Session = Depends(get_db)):
    """
    Get the latest price records for all crops.
    """
    # Group by crop and get the latest record
    subquery = db.query(
        MarketPrice.crop,
        func.max(MarketPrice.timestamp).label("max_ts")
    ).group_by(MarketPrice.crop).subquery()
    
    latest_prices = db.query(MarketPrice).join(
        subquery,
        (MarketPrice.crop == subquery.c.crop) & (MarketPrice.timestamp == subquery.c.max_ts)
    ).all()
    
    return latest_prices

@router.get("/prices/{crop}")
async def get_crop_details(crop: str, db: Session = Depends(get_db)):
    """
    Get latest price, history, and best market for a specific crop.
    """
    # 1. Fetch latest price
    latest = db.query(MarketPrice).filter(
        MarketPrice.crop.ilike(crop)
    ).order_by(MarketPrice.timestamp.desc()).first()
    
    if not latest:
        raise HTTPException(status_code=404, detail=f"No price data found for crop '{crop}'")
        
    # 2. Fetch history
    history = db.query(MarketPrice).filter(
        MarketPrice.crop.ilike(crop)
    ).order_by(MarketPrice.timestamp.desc()).limit(30).all()
    
    # 3. Find best market (highest price is best for farmers to sell)
    best_market_record = db.query(MarketPrice).filter(
        MarketPrice.crop.ilike(crop)
    ).order_by(MarketPrice.price.desc()).first()
    
    best_market = {
        "market": best_market_record.market if best_market_record else "N/A",
        "district": best_market_record.district if best_market_record else "N/A",
        "state": best_market_record.state if best_market_record else "N/A",
        "price": best_market_record.price if best_market_record else 0.0,
        "source": best_market_record.source if best_market_record else "N/A"
    }
    
    return {
        "crop": crop,
        "latest_price": latest,
        "best_market": best_market,
        "history": history
    }

@router.get("/updates")
async def get_government_updates(db: Session = Depends(get_db)):
    """
    Get all government updates.
    """
    return db.query(GovernmentUpdate).order_by(GovernmentUpdate.id.desc()).all()

@router.get("/news")
async def get_agriculture_news(db: Session = Depends(get_db)):
    """
    Get all agriculture news.
    """
    return db.query(AgricultureNews).order_by(AgricultureNews.id.desc()).all()
