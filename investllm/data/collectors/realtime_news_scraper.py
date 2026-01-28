"""
Real-time News Scraper for InvestLLM
====================================
Fetches financial news in real-time using Firecrawl API.

Features:
- Real-time news monitoring from multiple sources
- Full article extraction with Firecrawl
- Stock symbol detection and tagging
- Sentiment-ready text preprocessing

Setup:
1. Get API key from https://www.firecrawl.dev/
2. Set FIRECRAWL_API_KEY in .env
"""

import os
import re
import json
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field, asdict
import hashlib
import aiohttp
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Structured news article"""
    url: str
    title: str
    content: str
    source: str
    published_at: datetime
    scraped_at: datetime = field(default_factory=datetime.now)
    symbols: List[str] = field(default_factory=list)
    sentiment_score: Optional[float] = None
    summary: Optional[str] = None

    @property
    def article_id(self) -> str:
        """Generate unique ID from URL"""
        return hashlib.md5(self.url.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        d = asdict(self)
        d['published_at'] = self.published_at.isoformat()
        d['scraped_at'] = self.scraped_at.isoformat()
        d['article_id'] = self.article_id
        return d


class RealTimeNewsScraper:
    """
    Real-time news scraper using Firecrawl API.

    Monitors Indian financial news sources and extracts full articles
    for sentiment analysis.
    """

    # News source configurations
    SOURCES = {
        "moneycontrol": {
            "base_url": "https://www.moneycontrol.com",
            "news_urls": [
                "https://www.moneycontrol.com/news/business/markets/",
                "https://www.moneycontrol.com/news/business/stocks/",
                "https://www.moneycontrol.com/news/business/earnings/"
            ],
            "priority": 1
        },
        "economic_times": {
            "base_url": "https://economictimes.indiatimes.com",
            "news_urls": [
                "https://economictimes.indiatimes.com/markets/stocks/news",
                "https://economictimes.indiatimes.com/markets/stocks/earnings"
            ],
            "priority": 1
        },
        "livemint": {
            "base_url": "https://www.livemint.com",
            "news_urls": [
                "https://www.livemint.com/market/stock-market-news",
                "https://www.livemint.com/companies"
            ],
            "priority": 2
        },
        "business_standard": {
            "base_url": "https://www.business-standard.com",
            "news_urls": [
                "https://www.business-standard.com/markets",
                "https://www.business-standard.com/companies"
            ],
            "priority": 2
        }
    }

    # Stock symbol patterns for Indian markets
    SYMBOL_PATTERNS = [
        r'\b(NSE|BSE):\s*([A-Z]+)\b',
        r'\b([A-Z]{3,10})\s+(?:shares?|stock|scrip)\b',
        r'\b([A-Z&]{3,10})\s+(?:Ltd|Limited)\b'
    ]

    # Common company name to symbol mapping
    COMPANY_SYMBOLS = {
        "reliance": "RELIANCE",
        "tata motors": "TATAMOTORS",
        "tata steel": "TATASTEEL",
        "tata consultancy": "TCS",
        "tcs": "TCS",
        "infosys": "INFY",
        "wipro": "WIPRO",
        "hdfc bank": "HDFCBANK",
        "icici bank": "ICICIBANK",
        "axis bank": "AXISBANK",
        "kotak": "KOTAKBANK",
        "sbi": "SBIN",
        "state bank": "SBIN",
        "bharti airtel": "BHARTIARTL",
        "airtel": "BHARTIARTL",
        "maruti": "MARUTI",
        "mahindra": "M&M",
        "asian paints": "ASIANPAINT",
        "hindustan unilever": "HINDUNILVR",
        "hul": "HINDUNILVR",
        "itc": "ITC",
        "larsen": "LT",
        "l&t": "LT",
        "adani": "ADANIENT",
        "adani ports": "ADANIPORTS",
        "bajaj finance": "BAJFINANCE",
        "bajaj finserv": "BAJAJFINSV",
        "titan": "TITAN",
        "nestle": "NESTLEIND",
        "ultratech": "ULTRACEMCO",
        "power grid": "POWERGRID",
        "ntpc": "NTPC",
        "ongc": "ONGC",
        "coal india": "COALINDIA",
        "sun pharma": "SUNPHARMA",
        "dr reddy": "DRREDDY",
        "cipla": "CIPLA",
        "hcl tech": "HCLTECH",
        "tech mahindra": "TECHM"
    }

    def __init__(
        self,
        firecrawl_api_key: str = None,
        cache_dir: str = "data/news_cache"
    ):
        """
        Initialize scraper.

        Args:
            firecrawl_api_key: Firecrawl API key
            cache_dir: Directory to cache articles
        """
        self.api_key = firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY", "")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Track scraped URLs to avoid duplicates
        self._scraped_urls: Set[str] = set()
        self._load_scraped_cache()

        # Rate limiting
        self._last_request_time = datetime.min
        self._min_request_interval = 0.5  # seconds

        if not self.api_key:
            logger.warning("FIRECRAWL_API_KEY not set - using fallback scraping")

    def _load_scraped_cache(self):
        """Load previously scraped URLs"""
        cache_file = self.cache_dir / "scraped_urls.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    self._scraped_urls = set(json.load(f))
                logger.info(f"Loaded {len(self._scraped_urls)} cached URLs")
            except Exception:
                self._scraped_urls = set()

    def _save_scraped_cache(self):
        """Save scraped URLs to cache"""
        cache_file = self.cache_dir / "scraped_urls.json"
        # Keep only last 10000 URLs
        urls_to_save = list(self._scraped_urls)[-10000:]
        with open(cache_file, 'w') as f:
            json.dump(urls_to_save, f)

    async def _rate_limit(self):
        """Apply rate limiting"""
        now = datetime.now()
        elapsed = (now - self._last_request_time).total_seconds()
        if elapsed < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - elapsed)
        self._last_request_time = datetime.now()

    # =========================================================================
    # FIRECRAWL SCRAPING
    # =========================================================================

    async def scrape_url_firecrawl(
        self,
        url: str,
        session: aiohttp.ClientSession
    ) -> Optional[Dict]:
        """
        Scrape a URL using Firecrawl API.

        Returns:
            Dict with markdown content and metadata
        """
        await self._rate_limit()

        if not self.api_key:
            return None

        try:
            async with session.post(
                "https://api.firecrawl.dev/v1/scrape",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "url": url,
                    "formats": ["markdown"],
                    "onlyMainContent": True
                }
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("success"):
                        return data.get("data", {})
                else:
                    logger.warning(f"Firecrawl error {response.status}: {url}")

        except Exception as e:
            logger.error(f"Firecrawl scrape failed: {e}")

        return None

    async def crawl_source_firecrawl(
        self,
        source_url: str,
        session: aiohttp.ClientSession,
        max_pages: int = 10
    ) -> List[str]:
        """
        Crawl a news source to discover article URLs.

        Returns:
            List of article URLs
        """
        await self._rate_limit()

        if not self.api_key:
            return []

        try:
            async with session.post(
                "https://api.firecrawl.dev/v1/map",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "url": source_url,
                    "limit": max_pages
                }
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("success"):
                        return data.get("links", [])

        except Exception as e:
            logger.error(f"Firecrawl crawl failed: {e}")

        return []

    # =========================================================================
    # TEXT PROCESSING
    # =========================================================================

    def extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text"""
        symbols = set()
        text_lower = text.lower()

        # Pattern matching
        for pattern in self.SYMBOL_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    symbol = match[-1].upper()
                else:
                    symbol = match.upper()
                if 2 < len(symbol) <= 12:
                    symbols.add(symbol)

        # Company name matching
        for company, symbol in self.COMPANY_SYMBOLS.items():
            if company in text_lower:
                symbols.add(symbol)

        return list(symbols)

    def clean_content(self, content: str) -> str:
        """Clean and preprocess article content"""
        if not content:
            return ""

        # Remove markdown formatting
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)  # Links
        content = re.sub(r'#+\s*', '', content)  # Headers
        content = re.sub(r'\*+', '', content)  # Bold/italic
        content = re.sub(r'`+', '', content)  # Code

        # Remove HTML remnants
        content = re.sub(r'<[^>]+>', '', content)

        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()

        return content

    def extract_title_from_markdown(self, markdown: str) -> str:
        """Extract title from markdown content"""
        lines = markdown.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                return line.lstrip('#').strip()
            elif line and len(line) > 10:
                return line[:200]
        return "Untitled"

    # =========================================================================
    # MAIN SCRAPING METHODS
    # =========================================================================

    async def scrape_article(
        self,
        url: str,
        source: str,
        session: aiohttp.ClientSession
    ) -> Optional[NewsArticle]:
        """
        Scrape a single article.

        Args:
            url: Article URL
            source: News source name
            session: aiohttp session

        Returns:
            NewsArticle or None if failed
        """
        if url in self._scraped_urls:
            return None

        # Use Firecrawl for scraping
        data = await self.scrape_url_firecrawl(url, session)

        if not data:
            return None

        markdown = data.get("markdown", "")
        metadata = data.get("metadata", {})

        if not markdown or len(markdown) < 100:
            return None

        # Extract article data
        title = metadata.get("title") or self.extract_title_from_markdown(markdown)
        content = self.clean_content(markdown)
        symbols = self.extract_symbols(content + " " + title)

        # Parse published date
        pub_date_str = metadata.get("publishedTime") or metadata.get("article:published_time")
        try:
            published_at = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
        except:
            published_at = datetime.now()

        article = NewsArticle(
            url=url,
            title=title,
            content=content,
            source=source,
            published_at=published_at,
            symbols=symbols
        )

        # Mark as scraped
        self._scraped_urls.add(url)

        return article

    async def fetch_latest_news(
        self,
        sources: List[str] = None,
        max_articles_per_source: int = 20,
        hours_back: int = 24
    ) -> List[NewsArticle]:
        """
        Fetch latest news articles from all sources.

        Args:
            sources: List of source names (None = all)
            max_articles_per_source: Max articles per source
            hours_back: Only fetch articles from last N hours

        Returns:
            List of NewsArticle objects
        """
        if sources is None:
            sources = list(self.SOURCES.keys())

        all_articles = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        async with aiohttp.ClientSession() as session:
            for source_name in sources:
                if source_name not in self.SOURCES:
                    continue

                source_config = self.SOURCES[source_name]
                logger.info(f"Fetching from {source_name}...")

                # Discover article URLs
                article_urls = []
                for news_url in source_config["news_urls"]:
                    urls = await self.crawl_source_firecrawl(
                        news_url, session, max_pages=max_articles_per_source
                    )
                    article_urls.extend(urls)

                # Filter to article URLs only
                article_urls = [
                    u for u in article_urls
                    if u and source_config["base_url"] in u
                    and u not in self._scraped_urls
                ][:max_articles_per_source]

                logger.info(f"Found {len(article_urls)} new articles from {source_name}")

                # Scrape articles
                for url in article_urls:
                    try:
                        article = await self.scrape_article(url, source_name, session)
                        if article and article.published_at >= cutoff_time:
                            all_articles.append(article)
                            logger.info(f"Scraped: {article.title[:50]}...")
                    except Exception as e:
                        logger.warning(f"Failed to scrape {url}: {e}")

        # Save cache
        self._save_scraped_cache()

        # Sort by published date (newest first)
        all_articles.sort(key=lambda x: x.published_at, reverse=True)

        logger.info(f"Total articles fetched: {len(all_articles)}")
        return all_articles

    async def monitor_news(
        self,
        callback,
        interval_minutes: int = 5,
        sources: List[str] = None
    ):
        """
        Continuously monitor news sources.

        Args:
            callback: Async function to call with new articles
            interval_minutes: Check interval in minutes
            sources: Sources to monitor
        """
        logger.info(f"Starting news monitor (interval: {interval_minutes} min)")

        while True:
            try:
                articles = await self.fetch_latest_news(
                    sources=sources,
                    max_articles_per_source=10,
                    hours_back=1
                )

                if articles:
                    await callback(articles)

            except Exception as e:
                logger.error(f"Monitor error: {e}")

            await asyncio.sleep(interval_minutes * 60)

    # =========================================================================
    # DATA EXPORT
    # =========================================================================

    def save_articles(
        self,
        articles: List[NewsArticle],
        filename: str = None
    ) -> Path:
        """Save articles to JSON file"""
        if filename is None:
            filename = f"news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.cache_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([a.to_dict() for a in articles], f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(articles)} articles to {filepath}")
        return filepath

    def load_articles(self, filename: str) -> List[NewsArticle]:
        """Load articles from JSON file"""
        filepath = self.cache_dir / filename
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)

        articles = []
        for d in data:
            d['published_at'] = datetime.fromisoformat(d['published_at'])
            d['scraped_at'] = datetime.fromisoformat(d['scraped_at'])
            del d['article_id']  # Computed property
            articles.append(NewsArticle(**d))

        return articles

    def get_articles_for_symbol(
        self,
        symbol: str,
        articles: List[NewsArticle]
    ) -> List[NewsArticle]:
        """Filter articles mentioning a specific symbol"""
        return [a for a in articles if symbol in a.symbols]


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def example_callback(articles: List[NewsArticle]):
    """Example callback for news monitoring"""
    print(f"\n{'='*60}")
    print(f"Received {len(articles)} new articles")
    print("=" * 60)
    for article in articles[:5]:
        print(f"\n[{article.source}] {article.title}")
        print(f"  Symbols: {article.symbols}")
        print(f"  Published: {article.published_at}")


async def main():
    """Example usage"""
    scraper = RealTimeNewsScraper()

    print("=" * 60)
    print("Real-Time News Scraper Demo")
    print("=" * 60)

    # Fetch latest news
    articles = await scraper.fetch_latest_news(
        sources=["moneycontrol"],
        max_articles_per_source=5,
        hours_back=24
    )

    print(f"\nFetched {len(articles)} articles")
    for article in articles[:3]:
        print(f"\n[{article.source}] {article.title}")
        print(f"  Symbols: {article.symbols}")
        print(f"  Content preview: {article.content[:200]}...")

    # Save articles
    if articles:
        scraper.save_articles(articles)


if __name__ == "__main__":
    asyncio.run(main())
