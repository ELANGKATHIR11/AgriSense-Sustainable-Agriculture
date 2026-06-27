from datetime import datetime, timedelta
import json
from sqlalchemy.orm import Session
from backend.market_intelligence.models import ScrapeCache, CacheEntry

# In-memory Cache dictionary
_memory_cache = {}

def get_cached_value(key: str, db: Session) -> str:
    """
    Get cache value looking at memory first, then SQLite table cache_entries.
    """
    now = datetime.utcnow()
    
    # 1. Check Memory Cache
    if key in _memory_cache:
        val, expiry = _memory_cache[key]
        if expiry > now:
            return val
        else:
            del _memory_cache[key]
            
    # 2. Check SQLite DB Cache
    record = db.query(CacheEntry).filter(CacheEntry.key == key).first()
    if record:
        if record.expires_at > now:
            # Re-fill memory cache
            _memory_cache[key] = (record.value, record.expires_at)
            return record.value
        else:
            db.delete(record)
            try:
                db.commit()
            except Exception:
                db.rollback()
                
    return None

def set_cached_value(key: str, value: str, expiry_hours: float, db: Session):
    """
    Set cache value in both memory and SQLite table cache_entries.
    """
    now = datetime.utcnow()
    expires_at = now + timedelta(hours=expiry_hours)
    
    # Update Memory
    _memory_cache[key] = (value, expires_at)
    
    # Update SQLite DB
    record = db.query(CacheEntry).filter(CacheEntry.key == key).first()
    if record:
        record.value = value
        record.expires_at = expires_at
    else:
        record = CacheEntry(key=key, value=value, expires_at=expires_at)
        db.add(record)
        
    try:
        db.commit()
    except Exception:
        db.rollback()

def is_cached(url: str, db: Session, expiry_hours: int = 6) -> bool:
    """
    Check if a URL was scraped in the last `expiry_hours` hours (legacy support).
    """
    record = db.query(ScrapeCache).filter(ScrapeCache.url == url).first()
    if record:
        time_elapsed = datetime.utcnow() - record.scraped_at.replace(tzinfo=None)
        if time_elapsed < timedelta(hours=expiry_hours):
            return True
    return False

def set_cache(url: str, db: Session):
    """
    Mark a URL as scraped today (now) (legacy support).
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

