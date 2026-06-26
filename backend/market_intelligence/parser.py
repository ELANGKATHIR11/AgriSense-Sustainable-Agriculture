import re
import logging
from bs4 import BeautifulSoup
from datetime import datetime

logger = logging.getLogger("MarketParser")

# Helper lists for matching table columns
HEADER_MAPS = {
    "state": ["state", "state name", "rajya"],
    "district": ["district", "district name", "zila", "zilla"],
    "market": ["market", "mandi", "mandi name", "market name", "center"],
    "crop": ["crop", "commodity", "crop name", "variety", "produce"],
    "min_price": ["min price", "minimum", "min price (rs)", "minimum price", "min"],
    "max_price": ["max price", "maximum", "max price (rs)", "maximum price", "max"],
    "modal_price": ["modal price", "modal", "modal price (rs)", "modal_price", "price", "rate", "modal rate"],
    "arrival": ["arrival", "arrivals", "quantity", "arrival (tonnes)", "arrival (q)"],
    "date": ["date", "reported date", "arrival date", "published date", "date of arrival"]
}

def clean_number(val: str) -> float:
    """Extract numeric value from a string (e.g. 'Rs. 5,000/-' -> 5000.0)"""
    if not val:
        return 0.0
    cleaned = re.sub(r"[^\d\.]", "", val.replace(",", ""))
    try:
        return float(cleaned) if cleaned else 0.0
    except ValueError:
        return 0.0

def parse_mandi_tables(html: str, target_crop: str) -> list[dict]:
    """
    Search for tables inside the page and try to extract structured rows matching the crop.
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []
    
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        if not rows:
            continue
            
        # Try to identify column headers
        header_row = rows[0]
        cols = [th.text.strip().lower() for th in header_row.find_all(["th", "td"])]
        
        # Map indices
        indices = {}
        for key, aliases in HEADER_MAPS.items():
            for alias in aliases:
                if alias in cols:
                    indices[key] = cols.index(alias)
                    break
        
        # If we didn't map at least crop and price/modal_price, skip this table structure
        if "crop" not in indices and "market" not in indices:
            # Try positional fallback if the table looks like a pricing table
            # e.g., 4+ columns with numbers in the last few
            continue
            
        for row in rows[1:]:
            cells = [td.text.strip() for td in row.find_all("td")]
            if len(cells) <= max(indices.values(), default=0):
                continue
                
            crop_name = cells[indices["crop"]].lower() if "crop" in indices else ""
            if not target_crop.lower() in crop_name:
                continue
                
            # Extract fields with safe fallbacks
            state = cells[indices["state"]] if "state" in indices else "India"
            district = cells[indices["district"]] if "district" in indices else "N/A"
            market = cells[indices["market"]] if "market" in indices else "Mandi"
            modal = clean_number(cells[indices["modal_price"]]) if "modal_price" in indices else 0.0
            min_pr = clean_number(cells[indices["min_price"]]) if "min_price" in indices else modal
            max_pr = clean_number(cells[indices["max_price"]]) if "max_price" in indices else modal
            arrival = cells[indices["arrival"]] if "arrival" in indices else "N/A"
            date_val = cells[indices["date"]] if "date" in indices else datetime.utcnow().strftime("%Y-%m-%d")
            
            results.append({
                "crop": target_crop,
                "market": market,
                "district": district,
                "state": state,
                "modal_price": modal if modal > 0 else (min_pr + max_pr) / 2.0,
                "min_price": min_pr,
                "max_price": max_pr,
                "arrival": arrival,
                "date": date_val
            })
            
    return results

def parse_unstructured_price(html: str, target_crop: str) -> list[dict]:
    """
    Fallback parser for pages without tables. Searches text context for crop and prices.
    """
    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style elements
    for script in soup(["script", "style"]):
        script.extract()
        
    text = soup.get_text(separator=" ")
    results = []
    
    # Simple regex searching for: Crop name ... followed by prices (e.g. Rs 4000 - 5000 or Rs. 4500)
    # We look for matches in the page
    pattern = re.compile(rf"{re.escape(target_crop)}[^\n.]{{1,100}}?(?:rs\.?|inr)?\s*(\d{{3,6}})", re.IGNORECASE)
    matches = pattern.findall(text)
    
    if matches:
        price = float(matches[0])
        results.append({
            "crop": target_crop,
            "market": "Local Market",
            "district": "N/A",
            "state": "India",
            "modal_price": price,
            "min_price": price * 0.9,
            "max_price": price * 1.1,
            "arrival": "N/A",
            "date": datetime.utcnow().strftime("%Y-%m-%d")
        })
        
    return results

def extract_prices_from_html(html: str, target_crop: str) -> list[dict]:
    """
    Try parsing table, then fallback to unstructured text parsing.
    """
    if not html:
        return []
        
    results = parse_mandi_tables(html, target_crop)
    if not results:
        results = parse_unstructured_price(html, target_crop)
        
    # Ensure they have valid prices
    valid_results = [r for r in results if r.get("modal_price", 0.0) > 10.0]
    return valid_results

def parse_article_page(html: str, url: str) -> dict:
    """
    Parse a news article or government update page to extract title, summary/body text, source, etc.
    """
    if not html:
        return {}
        
    soup = BeautifulSoup(html, "html.parser")
    title = ""
    
    # Try finding title
    title_tag = soup.find(["h1", "title"])
    if title_tag:
        title = title_tag.text.strip()
        
    # Try getting source domain
    source = "Agriculture News"
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        source = parsed.netloc.replace("www.", "")
    except Exception:
        pass
        
    # Extract paragraphs for text content
    paragraphs = soup.find_all("p")
    text_content = " ".join([p.text.strip() for p in paragraphs[:5]]) # Keep first 5 paragraphs
    summary = text_content[:500] + "..." if len(text_content) > 500 else text_content
    
    # Look for metadata publication dates
    pub_date = datetime.utcnow().strftime("%Y-%m-%d")
    meta_date = soup.find("meta", {"property": "article:published_time"}) or soup.find("meta", {"name": "pubdate"})
    if meta_date:
        pub_date = meta_date.get("content", pub_date)[:10]
        
    return {
        "title": title or "Agricultural Update",
        "summary": summary or "No description available.",
        "source": source,
        "url": url,
        "date": pub_date
    }
