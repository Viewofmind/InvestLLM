"""
News Collector for InvestLLM
Builds a corpus of Indian financial news for sentiment training
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import hashlib
import re
from urllib.parse import urljoin, urlparse
import structlog
from bs4 import BeautifulSoup
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
import time

from investllm.config import settings, NEWS_SOURCES, STOCK_UNIVERSE

logger = structlog.get_logger()


@dataclass
class Article:
    """Represents a news article"""
    url: str
    title: str
    content: str
    source: str
    published_at: Optional[datetime]
    scraped_at: datetime
    author: Optional[str] = None
    categories: List[str] = None
    tags: List[str] = None
    stocks_mentioned: List[str] = None
    
    def __post_init__(self):
        self.categories = self.categories or []
        self.tags = self.tags or []
        self.stocks_mentioned = self.stocks_mentioned or []
    
    @property
    def id(self) -> str:
        """Generate unique ID from URL"""
        return hashlib.md5(self.url.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['id'] = self.id
        if self.published_at:
            data['published_at'] = self.published_at.isoformat()
        data['scraped_at'] = self.scraped_at.isoformat()
        return data


class NewsCollector:
    """
    Collects financial news from Indian sources
    
    Sources:
    - Moneycontrol
    - Economic Times
    - LiveMint
    - Business Standard
    - NDTV Profit
    
    Usage:
        collector = NewsCollector()
        
        # Collect recent news
        articles = await collector.collect_recent(days=7)
        
        # Build historical corpus
        await collector.build_corpus(years=5)
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.data_dir / "news"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Track collected URLs
        self.collected_urls: Set[str] = set()
        self._load_collected_urls()
        
        # Stock symbols for entity detection
        self.stock_symbols = set(STOCK_UNIVERSE)
        
        # Rate limiting
        self.request_delay = 1.0  # seconds between requests
        
        logger.info("news_collector_initialized", data_dir=str(self.data_dir))
    
    def _load_collected_urls(self):
        """Load previously collected URLs"""
        url_file = self.data_dir / "collected_urls.txt"
        if url_file.exists():
            with open(url_file) as f:
                self.collected_urls = set(line.strip() for line in f)
            logger.info("loaded_collected_urls", count=len(self.collected_urls))
    
    def _save_url(self, url: str):
        """Save URL to collected list"""
        url_file = self.data_dir / "collected_urls.txt"
        with open(url_file, "a") as f:
            f.write(url + "\n")
        self.collected_urls.add(url)
    
    # ===========================================
    # MONEYCONTROL SCRAPER
    # ===========================================
    
    async def scrape_moneycontrol(
        self,
        session: aiohttp.ClientSession,
        pages: int = 10
    ) -> List[Article]:
        """Scrape Moneycontrol stock news"""
        articles = []
        base_url = "https://www.moneycontrol.com/news/business/stocks/"
        
        for page in range(1, pages + 1):
            try:
                url = f"{base_url}page-{page}/" if page > 1 else base_url
                
                async with session.get(url, timeout=30) as response:
                    if response.status != 200:
                        logger.warning("moneycontrol_page_failed", page=page, status=response.status)
                        continue
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'lxml')
                    
                    # Find article links
                    article_links = soup.select('h2 a[href*="/news/"]')
                    
                    for link in article_links:
                        article_url = link.get('href')
                        
                        if article_url in self.collected_urls:
                            continue
                        
                        # Fetch full article
                        article = await self._fetch_moneycontrol_article(session, article_url)
                        
                        if article:
                            articles.append(article)
                            self._save_url(article_url)
                        
                        await asyncio.sleep(self.request_delay)
                
                logger.info("moneycontrol_page_scraped", page=page, articles=len(articles))
                await asyncio.sleep(2)  # Between pages
                
            except Exception as e:
                logger.error("moneycontrol_page_error", page=page, error=str(e))
        
        return articles
    
    async def _fetch_moneycontrol_article(
        self,
        session: aiohttp.ClientSession,
        url: str
    ) -> Optional[Article]:
        """Fetch and parse a single Moneycontrol article"""
        try:
            async with session.get(url, timeout=30) as response:
                if response.status != 200:
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'lxml')
                
                # Extract title
                title_elem = soup.select_one('h1.article_title, h1.artTitle')
                title = title_elem.get_text(strip=True) if title_elem else ""
                
                # Extract content
                content_elem = soup.select_one('div.article_content, div.content_wrapper')
                content = ""
                if content_elem:
                    paragraphs = content_elem.find_all('p')
                    content = "\n".join(p.get_text(strip=True) for p in paragraphs)
                
                # Extract date
                date_elem = soup.select_one('div.article_schedule, span.article_date')
                published_at = None
                if date_elem:
                    date_text = date_elem.get_text(strip=True)
                    published_at = self._parse_date(date_text)
                
                # Extract author
                author_elem = soup.select_one('span.article_author a')
                author = author_elem.get_text(strip=True) if author_elem else None
                
                # Find stock mentions
                stocks_mentioned = self._find_stock_mentions(title + " " + content)
                
                if not title or not content:
                    return None
                
                return Article(
                    url=url,
                    title=title,
                    content=content,
                    source="moneycontrol",
                    published_at=published_at,
                    scraped_at=datetime.now(),
                    author=author,
                    stocks_mentioned=stocks_mentioned
                )
                
        except Exception as e:
            logger.error("article_fetch_error", url=url, error=str(e))
            return None
    
    # ===========================================
    # ECONOMIC TIMES SCRAPER
    # ===========================================
    
    async def scrape_economic_times(
        self,
        session: aiohttp.ClientSession,
        pages: int = 10
    ) -> List[Article]:
        """Scrape Economic Times market news"""
        articles = []
        base_url = "https://economictimes.indiatimes.com/markets/stocks/news"
        
        for page in range(1, pages + 1):
            try:
                url = f"{base_url}/articlelist/{page}.cms" if page > 1 else base_url
                
                async with session.get(url, timeout=30) as response:
                    if response.status != 200:
                        continue
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'lxml')
                    
                    # Find article links
                    article_links = soup.select('div.eachStory a[href*="/markets/stocks/"]')
                    
                    for link in article_links:
                        article_url = link.get('href')
                        
                        if not article_url.startswith('http'):
                            article_url = f"https://economictimes.indiatimes.com{article_url}"
                        
                        if article_url in self.collected_urls:
                            continue
                        
                        article = await self._fetch_et_article(session, article_url)
                        
                        if article:
                            articles.append(article)
                            self._save_url(article_url)
                        
                        await asyncio.sleep(self.request_delay)
                
                logger.info("et_page_scraped", page=page, articles=len(articles))
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error("et_page_error", page=page, error=str(e))
        
        return articles
    
    async def _fetch_et_article(
        self,
        session: aiohttp.ClientSession,
        url: str
    ) -> Optional[Article]:
        """Fetch and parse Economic Times article"""
        try:
            async with session.get(url, timeout=30) as response:
                if response.status != 200:
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'lxml')
                
                # Extract title
                title_elem = soup.select_one('h1.artTitle')
                title = title_elem.get_text(strip=True) if title_elem else ""
                
                # Extract content
                content_elem = soup.select_one('div.artText')
                content = ""
                if content_elem:
                    paragraphs = content_elem.find_all('p')
                    content = "\n".join(p.get_text(strip=True) for p in paragraphs)
                
                # Extract date
                date_elem = soup.select_one('time')
                published_at = None
                if date_elem:
                    date_text = date_elem.get('datetime') or date_elem.get_text()
                    published_at = self._parse_date(date_text)
                
                stocks_mentioned = self._find_stock_mentions(title + " " + content)
                
                if not title or not content:
                    return None
                
                return Article(
                    url=url,
                    title=title,
                    content=content,
                    source="economic_times",
                    published_at=published_at,
                    scraped_at=datetime.now(),
                    stocks_mentioned=stocks_mentioned
                )
                
        except Exception as e:
            logger.error("et_article_error", url=url, error=str(e))
            return None
    
    # ===========================================
    # FIRECRAWL INTEGRATION (For Scale)
    # ===========================================
    
    async def scrape_with_firecrawl(
        self,
        source: str,
        max_pages: int = 100
    ) -> List[Article]:
        """
        Use Firecrawl API for efficient scraping
        Better for large-scale collection
        """
        if not settings.firecrawl_api_key:
            logger.warning("firecrawl_api_key_not_set")
            return []
        
        from firecrawl import FirecrawlApp
        
        app = FirecrawlApp(api_key=settings.firecrawl_api_key)
        source_config = NEWS_SOURCES.get(source)
        
        if not source_config:
            return []
        
        try:
            result = app.crawl_url(
                source_config['news_url'],
                params={
                    'limit': max_pages,
                    'scrapeOptions': {
                        'formats': ['markdown', 'html']
                    }
                }
            )
            
            articles = []
            if result and 'data' in result:
                for page in result['data']:
                    # Extract article from Firecrawl response
                    content = page.get('markdown', '')
                    metadata = page.get('metadata', {})
                    
                    article = Article(
                        url=metadata.get('sourceURL', ''),
                        title=metadata.get('title', ''),
                        content=content[:10000],  # Limit content size
                        source=source,
                        published_at=self._parse_date(metadata.get('publishedTime')),
                        scraped_at=datetime.now(),
                        stocks_mentioned=self._find_stock_mentions(content)
                    )
                    
                    if article.title and article.content:
                        articles.append(article)
            
            logger.info("firecrawl_complete", source=source, articles=len(articles))
            return articles
            
        except Exception as e:
            logger.error("firecrawl_error", source=source, error=str(e))
            return []
    
    # ===========================================
    # UTILITY METHODS
    # ===========================================
    
    def _find_stock_mentions(self, text: str) -> List[str]:
        """Find stock symbols mentioned in text"""
        mentioned = []
        
        # Direct symbol matching
        words = set(re.findall(r'\b[A-Z][A-Z0-9&-]+\b', text))
        
        for word in words:
            if word in self.stock_symbols:
                mentioned.append(word)
        
        # Company name matching (basic)
        company_patterns = {
            "Reliance": "RELIANCE",
            "TCS": "TCS",
            "Tata Consultancy": "TCS",
            "Infosys": "INFY",
            "HDFC Bank": "HDFCBANK",
            "ICICI Bank": "ICICIBANK",
            "State Bank": "SBIN",
            "SBI": "SBIN",
            "Wipro": "WIPRO",
            "HCL Tech": "HCLTECH",
            "Bharti Airtel": "BHARTIARTL",
            "ITC": "ITC",
            "Asian Paints": "ASIANPAINT",
            "Maruti": "MARUTI",
            "Bajaj Finance": "BAJFINANCE",
        }
        
        text_lower = text.lower()
        for pattern, symbol in company_patterns.items():
            if pattern.lower() in text_lower and symbol not in mentioned:
                mentioned.append(symbol)
        
        return mentioned
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse various date formats"""
        if not date_str:
            return None
        
        formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%B %d, %Y, %H:%M",
            "%b %d, %Y, %H:%M",
            "%d %b %Y, %H:%M",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        
        return None
    
    # ===========================================
    # SAVE/LOAD METHODS
    # ===========================================
    
    def save_articles(self, articles: List[Article], batch_name: str = None):
        """Save articles to JSON file"""
        if not articles:
            return
        
        batch_name = batch_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.data_dir / f"articles_{batch_name}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(
                [a.to_dict() for a in articles],
                f,
                ensure_ascii=False,
                indent=2
            )
        
        logger.info("articles_saved", filepath=str(filepath), count=len(articles))
    
    def load_all_articles(self) -> List[Dict]:
        """Load all saved articles"""
        articles = []
        
        for filepath in self.data_dir.glob("articles_*.json"):
            with open(filepath, encoding='utf-8') as f:
                articles.extend(json.load(f))
        
        logger.info("articles_loaded", count=len(articles))
        return articles
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export all articles to DataFrame"""
        articles = self.load_all_articles()
        return pd.DataFrame(articles)
    
    # ===========================================
    # MAIN COLLECTION METHODS
    # ===========================================
    
    async def collect_recent(self, pages: int = 10) -> List[Article]:
        """Collect recent news from all sources"""
        all_articles = []
        
        async with aiohttp.ClientSession(
            headers={'User-Agent': 'Mozilla/5.0'}
        ) as session:
            
            # Moneycontrol
            logger.info("scraping_moneycontrol")
            mc_articles = await self.scrape_moneycontrol(session, pages)
            all_articles.extend(mc_articles)
            
            # Economic Times
            logger.info("scraping_economic_times")
            et_articles = await self.scrape_economic_times(session, pages)
            all_articles.extend(et_articles)
        
        # Save
        self.save_articles(all_articles)
        
        logger.info("collection_complete", total=len(all_articles))
        return all_articles
    
    async def build_corpus(
        self,
        target_articles: int = 100000,
        use_firecrawl: bool = True
    ):
        """
        Build a large corpus of financial news
        Target: 100K+ articles for training
        """
        logger.info("starting_corpus_build", target=target_articles)
        
        total_collected = len(self.collected_urls)
        
        while total_collected < target_articles:
            if use_firecrawl and settings.firecrawl_api_key:
                # Use Firecrawl for efficiency
                for source in NEWS_SOURCES.keys():
                    articles = await self.scrape_with_firecrawl(source, max_pages=500)
                    self.save_articles(articles, f"{source}_{datetime.now().strftime('%Y%m%d')}")
                    total_collected += len(articles)
                    
                    if total_collected >= target_articles:
                        break
            else:
                # Manual scraping
                articles = await self.collect_recent(pages=50)
                total_collected += len(articles)
            
            logger.info("corpus_progress", collected=total_collected, target=target_articles)
            await asyncio.sleep(60)  # Pause between batches
        
        logger.info("corpus_build_complete", total=total_collected)


# ===========================================
# CONVENIENCE FUNCTIONS
# ===========================================

async def collect_news(pages: int = 10):
    """Convenience function to collect news"""
    collector = NewsCollector()
    return await collector.collect_recent(pages=pages)


def run_news_collection(pages: int = 10):
    """Run news collection synchronously"""
    return asyncio.run(collect_news(pages))


if __name__ == "__main__":
    # Test collection
    print("Starting news collection...")
    articles = run_news_collection(pages=5)
    print(f"Collected {len(articles)} articles")
