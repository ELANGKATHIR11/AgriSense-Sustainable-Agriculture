import logging
import random
import asyncio
import urllib.parse
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("MarketSearch")

# List of common headers to mimic a browser
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0"
]

def clean_ddg_url(url: str) -> str:
    """
    Extract target URL from DuckDuckGo redirect URLs.
    Example: /l/?kh=-1&uddg=https%3A%2F%2Fexample.com%2Fpage
    """
    if "uddg=" in url:
        parsed = urllib.parse.urlparse(url)
        queries = urllib.parse.parse_qs(parsed.query)
        if "uddg" in queries and len(queries["uddg"]) > 0:
            return queries["uddg"][0]
    return url

async def search_duckduckgo(query: str, max_results: int = 10) -> list[str]:
    """
    Search DuckDuckGo using the html/lite version and discover target page URLs.
    """
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote_plus(query)}"
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://duckduckgo.com/"
    }
    
    # Retry configuration
    retries = 3
    delay = 2.0
    
    for attempt in range(retries):
        try:
            # Respect rate limit of 1 request per 2 seconds (with random delay 1-3 seconds)
            await asyncio.sleep(random.uniform(1.0, 3.0))
            
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    links = []
                    
                    # Search results are typically in a.result__url or links inside h2.result__title
                    for a in soup.select("a.result__url"):
                        href = a.get("href")
                        if href:
                            cleaned = clean_ddg_url(href)
                            if cleaned and cleaned.startswith("http"):
                                links.append(cleaned)
                                
                    # Try fallback result__snippet or general links if no result__url found
                    if not links:
                        for a in soup.select("a.result__snippet"):
                            href = a.get("href")
                            if href:
                                cleaned = clean_ddg_url(href)
                                if cleaned and cleaned.startswith("http"):
                                    links.append(cleaned)
                                    
                    # Try general parsing of links within the search result boxes
                    if not links:
                        for a in soup.find_all("a"):
                            href = a.get("href", "")
                            if "uddg=" in href:
                                cleaned = clean_ddg_url(href)
                                if cleaned and cleaned.startswith("http"):
                                    links.append(cleaned)
                    
                    # Deduplicate and filter out unwanted links
                    ignored_domains = [
                        "youtube.com", "facebook.com", "instagram.com", "pinterest.com", 
                        "amazon.in", "amazon.com", "flipkart.com", "twitter.com", 
                        "duckduckgo.com", "google.com", "bing.com"
                    ]
                    
                    filtered_links = []
                    for link in links:
                        domain = urllib.parse.urlparse(link).netloc.lower()
                        if not any(ignored in domain for ignored in ignored_domains):
                            if link not in filtered_links:
                                filtered_links.append(link)
                                if len(filtered_links) >= max_results:
                                    break
                    
                    logger.info(f"DDG search for '{query}' returned {len(filtered_links)} URLs.")
                    return filtered_links
                
                elif response.status_code == 429:
                    logger.warning(f"DDG rate limited (429). Retrying in {delay}s...")
                else:
                    logger.warning(f"DDG returned status {response.status_code}. Retrying...")
                    
        except Exception as e:
            logger.error(f"DDG search error: {e}")
            
        await asyncio.sleep(delay)
        delay *= 2  # Exponential backoff
        
    return []
