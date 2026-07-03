# -*- coding: utf-8 -*-
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models import MarketProduct

router = APIRouter(prefix="/marketplace", tags=["Agri Marketplace"])


def init_marketplace(db: Session):
    if db.query(MarketProduct).count() == 0:
        p1 = MarketProduct(
            name="Organic Copper Fungicide",
            category="pesticide",
            price=24.50,
            supplier="BioGreen Solutions",
            buy_url="https://amazon.com",
            description="Premium liquid copper fungicide. Excellent control of Tomato Leaf Mold and Powdery Mildew.",
        )
        p2 = MarketProduct(
            name="Bio-NPK Soil Booster",
            category="fertilizer",
            price=38.99,
            supplier="EcoFarm Supplements",
            buy_url="https://amazon.com",
            description="Slow release organic NPK pellets to correct soil deficiencies.",
        )
        p3 = MarketProduct(
            name="Hybrid Rice Seeds (IR-64)",
            category="seed",
            price=15.00,
            supplier="Punjab Seed Corp",
            buy_url="https://amazon.com",
            description="High-yield disease resistant hybrid rice seeds.",
        )
        db.add_all([p1, p2, p3])
        db.commit()


@router.get("/products")
async def get_products(db: Session = Depends(get_db)):
    init_marketplace(db)
    return db.query(MarketProduct).all()


@router.get("/recommendations")
async def get_recommendations(
    disease: str = "Tomato Leaf Mold", db: Session = Depends(get_db)
):
    init_marketplace(db)
    if "mold" in disease.lower() or "blight" in disease.lower():
        # Recommend copper fungicide
        return (
            db.query(MarketProduct).filter(MarketProduct.category == "pesticide").all()
        )
    # Default return fertilizer
    return db.query(MarketProduct).filter(MarketProduct.category == "fertilizer").all()
