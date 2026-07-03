from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from sqlalchemy.sql import func
from backend.database import Base


class MarketPrice(Base):
    __tablename__ = "market_prices"
    id = Column(Integer, primary_key=True, index=True)
    crop = Column(String, index=True)
    market = Column(String, index=True)
    district = Column(String)
    state = Column(String, index=True)
    price = Column(Float)
    min_price = Column(Float, nullable=True)
    max_price = Column(Float, nullable=True)
    arrival = Column(String, nullable=True)
    unit = Column(String, default="Quintal")
    source = Column(String)
    url = Column(String)
    timestamp = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        index=True,
    )

    # New fields for confidence and source ranking
    confidence = Column(Float, default=1.0)
    source_rank = Column(Integer, default=50)
    verification_count = Column(Integer, default=1)
    freshness_score = Column(Float, default=1.0)
    confidence_label = Column(
        String, default="Estimated"
    )  # Verified, Likely, Estimated, Low confidence


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
    scraped_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class KnownSource(Base):
    __tablename__ = "known_sources"
    id = Column(Integer, primary_key=True, index=True)
    crop = Column(String, index=True)
    url = Column(String, unique=True, index=True)
    domain = Column(String)
    confidence = Column(Integer, default=40)
    last_checked = Column(DateTime(timezone=True), nullable=True)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    next_refresh = Column(DateTime(timezone=True), nullable=True)


class MarketIntelligenceMetric(Base):
    __tablename__ = "market_intelligence_metrics"
    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String, index=True)
    value = Column(Float)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())


class CacheEntry(Base):
    __tablename__ = "cache_entries"
    key = Column(String, primary_key=True, index=True)
    value = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True))
