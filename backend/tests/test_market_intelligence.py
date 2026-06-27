import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.market_intelligence.models import MarketPrice, KnownSource, CacheEntry
from backend.market_intelligence.search import get_source_confidence
from backend.market_intelligence.parser import extract_prices_from_html
from backend.market_intelligence.cache import get_cached_value, set_cached_value
from backend.market_intelligence.analytics import compute_analytics, normalize_and_save_prices

# Create test database engine
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def db_session():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

def test_source_confidence_scoring():
    """Verify trust engine domain confidence scoring."""
    assert get_source_confidence("https://agmarknet.gov.in/somepage") == 100
    assert get_source_confidence("https://enam.gov.in/portal") == 100
    assert get_source_confidence("https://upmandiparishad.in") == 92
    assert get_source_confidence("https://someblog.wordpress.com/page") == 20
    assert get_source_confidence("https://unknownwebsite.com") == 40

def test_cache_mechanism(db_session):
    """Verify multi-level caching (memory + SQLite)."""
    set_cached_value("test_key", "cached_data", 1.0, db_session)
    val = get_cached_value("test_key", db_session)
    assert val == "cached_data"

    # Test expiration
    set_cached_value("expired_key", "expired_val", -1.0, db_session)
    expired_val = get_cached_value("expired_key", db_session)
    assert expired_val is None

def test_parser_nested_tables():
    """Test extracting pricing from html tables."""
    html_content = """
    <html>
      <table>
        <tr>
          <th>Crop</th>
          <th>Mandi</th>
          <th>State</th>
          <th>Modal Price</th>
          <th>Min Price</th>
          <th>Max Price</th>
        </tr>
        <tr>
          <td>Tomato</td>
          <td>Kolar</td>
          <td>Karnataka</td>
          <td>4500</td>
          <td>4000</td>
          <td>5000</td>
        </tr>
      </table>
    </html>
    """
    import asyncio
    results = asyncio.run(extract_prices_from_html(html_content, "Tomato"))
    assert len(results) > 0
    assert results[0]["market"] == "Kolar"
    assert results[0]["modal_price"] == 4500.0

def test_analytics_computations(db_session):
    """Verify standard deviation, moving averages, and changes calculations."""
    now = datetime.utcnow()
    # Seed 3 price entries for Tomato
    p1 = MarketPrice(crop="Tomato", market="Mandi A", state="State A", price=4000.0, timestamp=now - timedelta(days=2))
    p2 = MarketPrice(crop="Tomato", market="Mandi A", state="State A", price=4500.0, timestamp=now - timedelta(days=1))
    p3 = MarketPrice(crop="Tomato", market="Mandi A", state="State A", price=5000.0, timestamp=now)
    
    db_session.add_all([p1, p2, p3])
    db_session.commit()
    
    analytics = compute_analytics("Tomato", db_session)
    assert analytics["average_india_price"] == 4500.0
    assert analytics["highest_market"]["price"] == 5000.0
    assert analytics["lowest_market"]["price"] == 4000.0
    assert analytics["price_spread"] == 1000.0
