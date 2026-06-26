import logging
import random
import asyncio
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("MarketScraper")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0"
]

async def scrape_url(url: str) -> str:
    """
    Download a page's HTML content using httpx.
    Implements rate limiting (1 request per 2s, random delay 1-3s), timeouts,
    retries with exponential backoff, and logging.
    """
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    retries = 3
    delay = 2.0
    
    for attempt in range(retries):
        # Enforce rate limit delay
        await asyncio.sleep(random.uniform(1.0, 3.0))
        
        try:
            logger.info(f"Scraping URL: {url} (Attempt {attempt + 1})")
            async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)
                
                if response.status_code == 200:
                    return response.text
                elif response.status_code in [404, 410]:
                    logger.warning(f"Page broken (status {response.status_code}): {url}. Skipping.")
                    break
                else:
                    logger.warning(f"Failed to fetch {url}. Status: {response.status_code}. Retrying...")
                    
        except httpx.TimeoutException:
            logger.warning(f"Timeout occurred while fetching {url}. Retrying...")
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {e}. Skipping.")
            break
            
        await asyncio.sleep(delay)
        delay *= 2  # Exponential backoff
        
    return ""
