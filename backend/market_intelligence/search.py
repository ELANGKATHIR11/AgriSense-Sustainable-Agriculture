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
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
]

def clean_ddg_url(url: str) -> str:
    """
    Extract target URL from DuckDuckGo redirect URLs.
    """
    if "uddg=" in url:
        parsed = urllib.parse.urlparse(url)
        queries = urllib.parse.parse_qs(parsed.query)
        if "uddg" in queries and len(queries["uddg"]) > 0:
            return queries["uddg"][0]
    return url

def get_source_confidence(url: str) -> int:
    """
    Calculate confidence score based on domain/URL.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
    except Exception:
        return 40

    # Government & Official agriculture portals
    if "agmarknet.gov.in" in domain or "enam.gov.in" in domain:
        return 100
    if domain.endswith(".gov.in") or domain.endswith(".nic.in"):
        return 100
    if "gov" in domain or "nic" in domain:
        return 98
    if "icar" in domain:
        return 95
    if "mandiboard" in domain or "upmandiparishad" in domain or "msamb" in domain:
        return 92
    
    # Universities
    if domain.endswith(".edu.in") or domain.endswith(".edu") or "univ" in domain or "sau" in domain:
        return 90

    # Trusted News
    trusted_news = [
        "krishijagran.com", "financialexpress.com", "economictimes.indiatimes.com",
        "thehindubusinessline.com", "moneycontrol.com", "business-standard.com",
        "reuters.com", "bloomberg.com", "indianexpress.com", "thehindu.com"
    ]
    if any(tn in domain for tn in trusted_news):
        return 70

    # Blogs and low confidence
    blogs_and_spam = [
        "blog", "wordpress", "blogspot", "medium.com", "tumblr.com", "wixsite.com", "github.io"
    ]
    if any(bs in domain or bs in path for bs in blogs_and_spam):
        return 20

    # Generic or unknown domains
    return 40

async def search_duckduckgo_package(query: str, max_results: int = 10) -> list[str]:
    """
    Primary search using the official duckduckgo-search package in a threadpool to avoid blocking.
    """
    try:
        from duckduckgo_search import DDGS
        
        def run_search():
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
                return [r["href"] for r in results if "href" in r]
                
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, run_search)
    except Exception as e:
        logger.warning(f"duckduckgo-search package execution failed: {e}. Falling back to HTML.")
        return []

async def search_duckduckgo_html(query: str, max_results: int = 10) -> list[str]:
    """
    Fallback HTML DuckDuckGo search parser.
    """
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote_plus(query)}"
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://duckduckgo.com/"
    }
    
    retries = 3
    delay = 2.0
    
    for attempt in range(retries):
        try:
            # Randomize delay
            await asyncio.sleep(random.uniform(1.5, 3.5))
            
            async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    links = []
                    
                    for a in soup.select("a.result__url"):
                        href = a.get("href")
                        if href:
                            cleaned = clean_ddg_url(href)
                            if cleaned and cleaned.startswith("http"):
                                links.append(cleaned)
                                
                    if not links:
                        for a in soup.select("a.result__snippet"):
                            href = a.get("href")
                            if href:
                                cleaned = clean_ddg_url(href)
                                if cleaned and cleaned.startswith("http"):
                                    links.append(cleaned)
                                    
                    if not links:
                        for a in soup.find_all("a"):
                            href = a.get("href", "")
                            if "uddg=" in href:
                                cleaned = clean_ddg_url(href)
                                if cleaned and cleaned.startswith("http"):
                                    links.append(cleaned)
                    
                    return links
                elif response.status_code == 429:
                    logger.warning(f"DDG HTML endpoint rate limited (429). Retrying...")
                else:
                    logger.warning(f"DDG HTML returned {response.status_code}. Retrying...")
        except Exception as e:
            logger.error(f"DDG HTML endpoint error: {e}")
            
        await asyncio.sleep(delay)
        delay *= 2.0  # Exponential backoff
        
    return []

async def search_duckduckgo(query: str, max_results: int = 10) -> list[str]:
    """
    Main search entry point that runs package search, falls back to HTML parsing,
    filters domains, ranks them by confidence, and returns high confidence urls first.
    """
    links = await search_duckduckgo_package(query, max_results)
    if not links:
        links = await search_duckduckgo_html(query, max_results)
        
    # Filter out unwanted domains
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
                
    # Sort discovered URLs by confidence score (Source Trust Engine)
    ranked_links = sorted(filtered_links, key=lambda x: get_source_confidence(x), reverse=True)
    
    # Cap results
    ranked_links = ranked_links[:max_results]
    logger.info(f"DDG search for '{query}' returned {len(ranked_links)} ranked URLs.")
    return ranked_links

