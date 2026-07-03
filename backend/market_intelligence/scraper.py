import logging
import random
import asyncio
import socket
import ipaddress
import urllib.parse
import urllib.robotparser
import httpx

logger = logging.getLogger("MarketScraper")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# Global semaphore to limit maximum scraping concurrency to 10
scrape_semaphore = asyncio.Semaphore(10)

# In-memory robots.txt cache
robots_cache = {}


def is_safe_url(url: str) -> bool:
    """
    SSRF Prevention: validates URL scheme and ensures resolved IP is not private/local.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ["http", "https"]:
            return False

        hostname = parsed.hostname
        if not hostname:
            return False

        if hostname.lower() in ["localhost", "127.0.0.1", "::1"]:
            return False

        # Resolve hostname
        ip = socket.gethostbyname(hostname)
        ip_obj = ipaddress.ip_address(ip)

        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
            return False

        return True
    except Exception as e:
        logger.warning(f"SSRF safety check failed for {url}: {e}")
        return False


async def respect_robots_txt(url: str, user_agent: str) -> bool:
    """
    Checks if a URL is permitted under robots.txt.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        robots_url = f"{base_url}/robots.txt"

        # Check cache
        if base_url in robots_cache:
            rp = robots_cache[base_url]
        else:
            rp = urllib.robotparser.RobotFileParser()

            # Run fetch in threadpool to avoid blocking
            def fetch_robots():
                rp.set_url(robots_url)
                rp.read()

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, fetch_robots)
            robots_cache[base_url] = rp

        return rp.can_fetch(user_agent, url)
    except Exception:
        # Fallback to true if robots.txt check fails or is missing
        return True


async def scrape_url(url: str) -> str:
    """
    Download page's HTML content safely under a semaphore.
    """
    if not is_safe_url(url):
        logger.warning(f"Blocked unsafe/SSRF URL attempt: {url}")
        return ""

    user_agent = random.choice(USER_AGENTS)

    # Check robots.txt compliance
    if not await respect_robots_txt(url, user_agent):
        logger.info(f"Access to {url} blocked by robots.txt")
        return ""

    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
    }

    retries = 3
    delay = 2.0

    # Enter semaphore zone
    async with scrape_semaphore:
        for attempt in range(retries):
            # Enforce random delay
            await asyncio.sleep(random.uniform(1.0, 3.0))

            try:
                logger.info(f"Scraping URL: {url} (Attempt {attempt + 1})")
                async with httpx.AsyncClient(
                    timeout=12.0, follow_redirects=True
                ) as client:
                    response = await client.get(url, headers=headers)

                    if response.status_code == 200:
                        # Success
                        return response.text
                    elif response.status_code in [404, 410]:
                        logger.warning(
                            f"Page broken (status {response.status_code}): {url}."
                        )
                        break
                    elif response.status_code == 429:
                        logger.warning(f"Rate limited (429) on {url}. Backing off.")
                    else:
                        logger.warning(
                            f"Failed to fetch {url}. Status: {response.status_code}. Retrying..."
                        )
            except httpx.TimeoutException:
                logger.warning(f"Timeout occurred while fetching {url}. Retrying...")
            except Exception as e:
                logger.error(f"Error fetching URL {url}: {e}.")
                break

            await asyncio.sleep(delay)
            delay *= 2.0  # Exponential backoff

    return ""
