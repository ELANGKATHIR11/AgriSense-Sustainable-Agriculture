from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from backend.market_intelligence.models import ScrapeCache

def is_cached(url: str, db: Session, expiry_hours: int = 6) -> bool:
    """
    Check if a URL was scraped in the last `expiry_hours` hours.
    """
    record = db.query(ScrapeCache).filter(ScrapeCache.url == url).first()
    if record:
        time_elapsed = datetime.utcnow() - record.scraped_at.replace(tzinfo=None)
        if time_elapsed < timedelta(hours=expiry_hours):
            return True
    return False

def set_cache(url: str, db: Session):
    """
    Mark a URL as scraped today (now).
    """
    record = db.query(ScrapeCache).filter(ScrapeCache.url == url).first()
    if record:
        record.scraped_at = datetime.utcnow()
    else:
        record = ScrapeCache(url=url, scraped_at=datetime.utcnow())
        db.add(record)
    try:
        db.commit()
    except Exception:
        db.rollback()
