import os
import csv
import logging
import asyncio
import random
from datetime import datetime
from sqlalchemy.orm import Session
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from backend.database import SessionLocal, get_db
from backend.market_intelligence.models import MarketPrice, GovernmentUpdate, AgricultureNews
from backend.market_intelligence.search import search_duckduckgo
from backend.market_intelligence.scraper import scrape_url
from backend.market_intelligence.parser import extract_prices_from_html, parse_article_page
from backend.market_intelligence.summarizer import summarize_update
from backend.market_intelligence.websocket import manager as ws_manager
from backend.market_intelligence.cache import is_cached, set_cache

logger = logging.getLogger("MarketScheduler")
scheduler = AsyncIOScheduler()

# CSV file paths
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "AgriSense-Dataset")
CSV_200_PATH = os.path.join(DATASET_DIR, "top 200 indian crops.csv")

def get_all_crops() -> list[str]:
    """
    Load crop names from `top 200 indian crops.csv` and merge with other datasets to remove duplicates.
    """
    crops = set()
    
    # 1. Load from top 200 indian crops.csv
    if os.path.exists(CSV_200_PATH):
        try:
            with open(CSV_200_PATH, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    crop = row.get("Crop", "").strip()
                    if crop:
                        crops.add(crop)
        except Exception as e:
            logger.error(f"Error loading {CSV_200_PATH}: {e}")
            
    # 2. Try to merge from 48_crops_chatbot.csv if exists
    chatbot_csv = os.path.join(DATASET_DIR, "48_crops_chatbot.csv")
    if os.path.exists(chatbot_csv):
        try:
            with open(chatbot_csv, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0].strip():
                        crops.add(row[0].strip())
        except Exception as e:
            logger.warning(f"Could not merge chatbot crops: {e}")
            
    # Fallback to standard crops if empty
    if not crops:
        crops = {"Rice", "Wheat", "Maize", "Tomato", "Potato", "Onion", "Cotton", "Sugarcane", "Mustard", "Soybean"}
        
    # Clean list of crops, remove header-like or empty entries
    cleaned_crops = sorted([c for c in crops if c.lower() != "crop" and len(c) > 1])
    logger.info(f"Loaded {len(cleaned_crops)} crops for market intelligence analysis.")
    return cleaned_crops

def seed_initial_data():
    """
    Seed initial data if database tables are empty so the dashboard has instant, rich content.
    """
    db = SessionLocal()
    try:
        # Seeding Market Prices
        if db.query(MarketPrice).count() == 0:
            logger.info("Seeding initial market prices...")
            initial_prices = [
                MarketPrice(crop="Tomato", market="Kolar Mandi", district="Kolar", state="Karnataka", price=4500.0, min_price=4000.0, max_price=5000.0, arrival="45 Tons", unit="Quintal", source="Agmarknet", url="https://agmarknet.gov.in"),
                MarketPrice(crop="Onion", market="Lasalgaon Mandi", district="Nashik", state="Maharashtra", price=2500.0, min_price=2200.0, max_price=2800.0, arrival="120 Tons", unit="Quintal", source="eNAM", url="https://enam.gov.in"),
                MarketPrice(crop="Potato", market="Agra Mandi", district="Agra", state="Uttar Pradesh", price=1800.0, min_price=1600.0, max_price=2000.0, arrival="95 Tons", unit="Quintal", source="UP State Board", url="http://upmandiparishad.in"),
                MarketPrice(crop="Rice", market="Karnal Mandi", district="Karnal", state="Haryana", price=3600.0, min_price=3400.0, max_price=3800.0, arrival="250 Tons", unit="Quintal", source="eNAM", url="https://enam.gov.in"),
                MarketPrice(crop="Wheat", market="Khanna Mandi", district="Ludhiana", state="Punjab", price=2275.0, min_price=2275.0, max_price=2350.0, arrival="310 Tons", unit="Quintal", source="Agmarknet", url="https://agmarknet.gov.in"),
                MarketPrice(crop="Cotton", market="Adoni Mandi", district="Kurnool", state="Andhra Pradesh", price=7200.0, min_price=6800.0, max_price=7500.0, arrival="80 Tons", unit="Quintal", source="eNAM", url="https://enam.gov.in"),
                MarketPrice(crop="Mustard", market="Bharatpur Mandi", district="Bharatpur", state="Rajasthan", price=5400.0, min_price=5200.0, max_price=5650.0, arrival="110 Tons", unit="Quintal", source="Raj Mandi Board", url="https://rajmandiboard.in"),
                MarketPrice(crop="Soybean", market="Indore Mandi", district="Indore", state="Madhya Pradesh", price=4600.0, min_price=4400.0, max_price=4800.0, arrival="150 Tons", unit="Quintal", source="eNAM", url="https://enam.gov.in"),
            ]
            db.add_all(initial_prices)
            db.commit()

        # Seeding Government Updates
        if db.query(GovernmentUpdate).count() == 0:
            logger.info("Seeding initial government updates...")
            initial_updates = [
                GovernmentUpdate(title="PM Kisan 17th Installment Released", summary="Prime Minister Narendra Modi released the 17th installment of PM Kisan Samman Nidhi Yojana, transferring Rs 20,000 crores to over 9.2 crore farmers. Farmers can check status on the official PM-Kisan portal using Aadhaar.", source="Ministry of Agriculture", url="https://pmkisan.gov.in", date=datetime.utcnow().strftime("%Y-%m-%d"), category="PM Kisan"),
                GovernmentUpdate(title="Fertilizer Subsidy Allocation Updated", summary="Union cabinet approves Rs 1.08 lakh crore subsidy for Urea and NPK fertilizers for Rabi season to ensure easy availability at subsidized rates to farmers.", source="ICAR News", url="https://icar.org.in", date=datetime.utcnow().strftime("%Y-%m-%d"), category="fertilizer subsidy"),
                GovernmentUpdate(title="ICAR Introduces High-Yield Maize Seed Varieties", summary="Indian Council of Agricultural Research (ICAR) has launched three new drought-resistant hybrid maize seeds suited for dryland farming regions in Western India.", source="ICAR", url="https://icar.org.in", date=datetime.utcnow().strftime("%Y-%m-%d"), category="ICAR")
            ]
            db.add_all(initial_updates)
            db.commit()

        # Seeding Agriculture News
        if db.query(AgricultureNews).count() == 0:
            logger.info("Seeding initial agriculture news...")
            initial_news = [
                AgricultureNews(title="Monsoon Rainfall Forecast Predicts Normal Showers", summary="India Meteorological Department (IMD) forecasts normal monsoon showers for central and northern agricultural zones, easing crop deficit concerns.", source="Krishi Jagran", url="https://krishijagran.com", published=datetime.utcnow().strftime("%Y-%m-%d")),
                AgricultureNews(title="Tomato Leaf Mold Alert in Southern Districts", summary="Agricultural officials warn farmers in Kolar and Madanapalle of leaf mold outbreaks due to high morning humidity. Recommends copper fungicide.", source="AgriNews India", url="https://agrinews.in", published=datetime.utcnow().strftime("%Y-%m-%d")),
                AgricultureNews(title="Record Wheat Arrivals Registered in Punjab Mandis", summary="Wheat arrivals across Punjab procurement markets breach the 100-lakh-tonne mark. Government agencies speed up purchase and transport operations.", source="Financial Express", url="https://financialexpress.com", published=datetime.utcnow().strftime("%Y-%m-%d"))
            ]
            db.add_all(initial_news)
            db.commit()
    except Exception as e:
        logger.error(f"Error seeding market intelligence: {e}")
        db.rollback()
    finally:
        db.close()

async def job_update_market_prices():
    """
    Background job to run every 30 minutes to fetch live prices.
    """
    logger.info("Starting background market price scraper job...")
    crops = get_all_crops()
    db = SessionLocal()
    
    try:
        # Pick a randomized sample of 5 crops in each iteration to avoid overloading DDG
        # and stay within execution boundaries.
        sample_crops = random.sample(crops, min(len(crops), 5))
        
        for crop in sample_crops:
            # Query variations
            queries = [
                f"{crop} mandi price today India",
                f"{crop} market price India",
                f"{crop} modal price"
            ]
            
            # Select a random query to discover urls
            query = random.choice(queries)
            urls = await search_duckduckgo(query, max_results=3)
            
            for url in urls:
                if is_cached(url, db, expiry_hours=6):
                    logger.info(f"URL {url} cached (scraped within 6h). Skipping.")
                    continue
                    
                html = await scrape_url(url)
                if not html:
                    continue
                    
                records = extract_prices_from_html(html, crop)
                set_cache(url, db)
                
                for record in records:
                    # Save to DB (only keep latest per crop+market combination)
                    existing = db.query(MarketPrice).filter(
                        MarketPrice.crop == crop,
                        MarketPrice.market == record["market"]
                    ).first()
                    
                    if existing:
                        existing.price = record["modal_price"]
                        existing.min_price = record["min_price"]
                        existing.max_price = record["max_price"]
                        existing.arrival = record["arrival"]
                        existing.source = record["market"] + " Scraper"
                        existing.url = url
                    else:
                        new_price = MarketPrice(
                            crop=crop,
                            market=record["market"],
                            district=record["district"],
                            state=record["state"],
                            price=record["modal_price"],
                            min_price=record["min_price"],
                            max_price=record["max_price"],
                            arrival=record["arrival"],
                            unit="Quintal",
                            source=record["market"] + " Scraper",
                            url=url
                        )
                        db.add(new_price)
                        
                db.commit()
                
        # Broadcast the latest update to WebSockets
        latest_prices = db.query(MarketPrice).order_by(MarketPrice.timestamp.desc()).limit(10).all()
        await ws_manager.broadcast({
            "type": "market_prices_update",
            "data": [{
                "crop": p.crop,
                "market": p.market,
                "district": p.district,
                "state": p.state,
                "price": p.price,
                "arrival": p.arrival,
                "source": p.source,
                "timestamp": p.timestamp.isoformat() if p.timestamp else None
            } for p in latest_prices]
        })
        
    except Exception as e:
        logger.error(f"Error in job_update_market_prices: {e}")
        db.rollback()
    finally:
        db.close()

async def job_update_government_updates():
    """
    Background job to run every hour to collect government updates and schemes.
    """
    logger.info("Starting background government schemes job...")
    db = SessionLocal()
    
    queries = [
        ("latest agriculture schemes India", "Government Scheme"),
        ("PM Kisan updates", "PM Kisan"),
        ("fertilizer subsidy ministry of agriculture", "fertilizer subsidy"),
        ("ICAR technology transfer farmers", "ICAR")
    ]
    
    try:
        # Pick a random query
        query_text, category = random.choice(queries)
        urls = await search_duckduckgo(query_text, max_results=3)
        
        for url in urls:
            if is_cached(url, db, expiry_hours=6):
                continue
                
            html = await scrape_url(url)
            if not html:
                continue
                
            article = parse_article_page(html, url)
            set_cache(url, db)
            
            if article and article.get("title"):
                # Use AgriGPT summarization
                ai_summary = await summarize_update(article["title"], article["summary"], category)
                
                # Deduplicate by url/title
                exists = db.query(GovernmentUpdate).filter(
                    (GovernmentUpdate.url == url) | (GovernmentUpdate.title == article["title"])
                ).first()
                
                if not exists:
                    new_scheme = GovernmentUpdate(
                        title=article["title"],
                        summary=ai_summary,
                        source=article["source"],
                        url=url,
                        date=article["date"],
                        category=category
                    )
                    db.add(new_scheme)
                    db.commit()
                    
    except Exception as e:
        logger.error(f"Error in job_update_government_updates: {e}")
        db.rollback()
    finally:
        db.close()

async def job_update_news():
    """
    Background job to run every hour to collect agricultural news.
    """
    logger.info("Starting background agricultural news job...")
    db = SessionLocal()
    
    queries = [
        "Indian agriculture news today",
        "crop disease outbreak India",
        "rainfall agriculture updates India"
    ]
    
    try:
        query_text = random.choice(queries)
        urls = await search_duckduckgo(query_text, max_results=3)
        
        for url in urls:
            if is_cached(url, db, expiry_hours=6):
                continue
                
            html = await scrape_url(url)
            if not html:
                continue
                
            article = parse_article_page(html, url)
            set_cache(url, db)
            
            if article and article.get("title"):
                ai_summary = await summarize_update(article["title"], article["summary"], "News")
                
                exists = db.query(AgricultureNews).filter(
                    (AgricultureNews.url == url) | (AgricultureNews.title == article["title"])
                ).first()
                
                if not exists:
                    new_news = AgricultureNews(
                        title=article["title"],
                        summary=ai_summary,
                        source=article["source"],
                        url=url,
                        published=article["date"]
                    )
                    db.add(new_news)
                    db.commit()
                    
    except Exception as e:
        logger.error(f"Error in job_update_news: {e}")
        db.rollback()
    finally:
        db.close()

def start_scheduler():
    """
    Start background jobs using APScheduler.
    """
    # Seed data first to guarantee direct availability
    seed_initial_data()
    
    # Schedule market updates every 30 minutes
    scheduler.add_job(job_update_market_prices, 'interval', minutes=30, id="market_prices_job")
    
    # Schedule government updates every 60 minutes
    scheduler.add_job(job_update_government_updates, 'interval', minutes=60, id="gov_updates_job")
    
    # Schedule news updates every 60 minutes
    scheduler.add_job(job_update_news, 'interval', minutes=60, id="news_job")
    
    # Start scheduler
    scheduler.start()
    logger.info("Market Intelligence APScheduler started successfully.")
