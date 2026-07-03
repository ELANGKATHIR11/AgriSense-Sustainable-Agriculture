import logging
from datetime import datetime, timedelta
from sqlalchemy import func
from sqlalchemy.orm import Session
from backend.market_intelligence.models import MarketPrice
from backend.market_intelligence.search import get_source_confidence

logger = logging.getLogger("MarketAnalytics")


def compute_freshness_score(timestamp: datetime) -> float:
    """
    Compute freshness score from 0.0 to 1.0 based on elapsed time.
    """
    if not timestamp:
        return 0.1
    ts_naive = timestamp.replace(tzinfo=None) if timestamp.tzinfo else timestamp
    elapsed = datetime.utcnow() - ts_naive
    if elapsed <= timedelta(hours=12):
        return 1.0
    elif elapsed <= timedelta(days=1):
        return 0.9
    elif elapsed <= timedelta(days=3):
        return 0.7
    elif elapsed <= timedelta(days=7):
        return 0.4
    else:
        return 0.1


def compute_confidence_label(score: float) -> str:
    """
    Return confidence label based on calculated confidence score.
    """
    if score >= 0.85:
        return "Verified"
    elif score >= 0.65:
        return "Likely"
    elif score >= 0.40:
        return "Estimated"
    else:
        return "Low confidence"


def compute_analytics(crop: str, db: Session) -> dict:
    """
    Automatically compute:
    - Daily change
    - Weekly change
    - Monthly change
    - Moving average (7-day and 30-day)
    - Volatility (stddev)
    - Highest market
    - Lowest market
    - Average India price
    - Price spread (max - min)
    """
    now = datetime.utcnow()  # naive UTC — matches PostgreSQL TIMESTAMP columns

    # 1. Fetch prices in last 30 days
    thirty_days_ago = now - timedelta(days=30)
    # Compare timezone-aware now against DB timestamps (PostgreSQL handles tz-aware comparisons)
    prices_30d = (
        db.query(MarketPrice)
        .filter(MarketPrice.crop.ilike(crop), MarketPrice.timestamp >= thirty_days_ago)
        .all()
    )

    if not prices_30d:
        return {}

    prices_raw = [p.price for p in prices_30d if p.price > 0]
    if not prices_raw:
        return {}

    avg_price = sum(prices_raw) / len(prices_raw)
    max_price = max(prices_raw)
    min_price = min(prices_raw)
    spread = max_price - min_price

    # Volatility (Standard Deviation)
    import math

    variance = sum((x - avg_price) ** 2 for x in prices_raw) / len(prices_raw)
    volatility = math.sqrt(variance)

    # Highest & Lowest Markets
    highest_market_rec = max(prices_30d, key=lambda x: x.price)
    lowest_market_rec = min(prices_30d, key=lambda x: x.price)

    # Moving Averages
    prices_7d = [
        p.price
        for p in prices_30d
        if p.timestamp.replace(tzinfo=None) >= (now - timedelta(days=7))
    ]
    moving_average_7d = sum(prices_7d) / len(prices_7d) if prices_7d else avg_price
    moving_average_30d = avg_price

    # Changes (Daily, Weekly, Monthly)
    # Get average price today, 1 day ago, 7 days ago, 30 days ago
    def get_avg_at_period(start_date, end_date):
        res = (
            db.query(func.avg(MarketPrice.price))
            .filter(
                MarketPrice.crop.ilike(crop),
                MarketPrice.timestamp >= start_date,
                MarketPrice.timestamp <= end_date,
            )
            .scalar()
        )
        return float(res) if res else None

    avg_today = get_avg_at_period(now - timedelta(days=1), now) or avg_price
    avg_yesterday = get_avg_at_period(now - timedelta(days=2), now - timedelta(days=1))
    avg_last_week = get_avg_at_period(now - timedelta(days=8), now - timedelta(days=7))
    avg_last_month = get_avg_at_period(
        now - timedelta(days=31), now - timedelta(days=30)
    )

    daily_change = (
        ((avg_today - avg_yesterday) / avg_yesterday * 100) if avg_yesterday else 0.0
    )
    weekly_change = (
        ((avg_today - avg_last_week) / avg_last_week * 100) if avg_last_week else 0.0
    )
    monthly_change = (
        ((avg_today - avg_last_month) / avg_last_month * 100) if avg_last_month else 0.0
    )

    return {
        "crop": crop,
        "daily_change_pct": round(daily_change, 2),
        "weekly_change_pct": round(weekly_change, 2),
        "monthly_change_pct": round(monthly_change, 2),
        "moving_average_7d": round(moving_average_7d, 2),
        "moving_average_30d": round(moving_average_30d, 2),
        "volatility": round(volatility, 2),
        "highest_market": {
            "market": highest_market_rec.market,
            "state": highest_market_rec.state,
            "price": highest_market_rec.price,
        },
        "lowest_market": {
            "market": lowest_market_rec.market,
            "state": lowest_market_rec.state,
            "price": lowest_market_rec.price,
        },
        "average_india_price": round(avg_price, 2),
        "price_spread": round(spread, 2),
    }


def normalize_and_save_prices(db: Session, records: list[dict], url: str):
    """
    Step 9: Normalize and merge price details.
    Step 10: Never overwrite. Create history (always add new row).
    Step 12: Compute confidence, source_rank, verification_count, freshness_score.
    """
    for r in records:
        crop = r["crop"]
        market = r["market"]
        state = r["state"]
        r["date"]

        # Calculate source rank and initial confidence
        source_rank = get_source_confidence(url)
        initial_confidence = float(r.get("confidence", 0.5))

        # Freshness initially 1.0 for new inserts
        freshness = 1.0

        # Verify if an identical crop/market/state/date entry exists
        existing_list = (
            db.query(MarketPrice)
            .filter(
                MarketPrice.crop == crop,
                MarketPrice.market == market,
                MarketPrice.state == state,
                MarketPrice.price == float(r["modal_price"]),
            )
            .all()
        )

        # Verification count increment if we see it from multiple places or occurrences
        verification_count = len(existing_list) + 1

        # Calculate final confidence score
        calculated_confidence = (
            (initial_confidence * 0.4)
            + (source_rank / 100.0 * 0.4)
            + (min(verification_count, 5) / 5.0 * 0.2)
        )
        confidence_lbl = compute_confidence_label(calculated_confidence)

        new_price = MarketPrice(
            crop=crop,
            market=market,
            district=r["district"],
            state=state,
            price=float(r["modal_price"]),
            min_price=float(r["min_price"]),
            max_price=float(r["max_price"]),
            arrival=r["arrival"],
            unit=r.get("unit", "Quintal"),
            source=market + " Scraper",
            url=url,
            confidence=round(calculated_confidence, 2),
            source_rank=source_rank,
            verification_count=verification_count,
            freshness_score=freshness,
            confidence_label=confidence_lbl,
        )
        db.add(new_price)

    try:
        db.commit()
    except Exception as e:
        logger.error(f"Failed to save normalized prices: {e}")
        db.rollback()
