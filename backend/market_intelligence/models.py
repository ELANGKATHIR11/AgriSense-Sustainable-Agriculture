from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from sqlalchemy.sql import func
from backend.database import Base

class MarketPrice(Base):
    __tablename__ = "market_prices"
    id = Column(Integer, primary_key=True, index=True)
    crop = Column(String, index=True)
    market = Column(String, index=True)
    district = Column(String)
    state = Column(String)
    price = Column(Float)
    min_price = Column(Float, nullable=True)
    max_price = Column(Float, nullable=True)
    arrival = Column(String, nullable=True)
    unit = Column(String, default="Quintal")
    source = Column(String)
    url = Column(String)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class GovernmentUpdate(Base):
    __tablename__ = "government_updates"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    summary = Column(Text)
    source = Column(String)
    url = Column(String)
    date = Column(String)
    category = Column(String)

class AgricultureNews(Base):
    __tablename__ = "agriculture_news"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    summary = Column(Text)
    source = Column(String)
    url = Column(String)
    published = Column(String)

class ScrapeCache(Base):
    __tablename__ = "scrape_cache"
    url = Column(String, primary_key=True, index=True)
    scraped_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

