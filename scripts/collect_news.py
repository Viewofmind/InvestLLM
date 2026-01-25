#!/usr/bin/env python3
"""
Firecrawl News Collection Script
================================

Collects Indian financial news using Firecrawl API.

Your Credits: 500,000 credits
Strategy: ~100,000 quality articles

Credit Usage:
- 1 credit per page scraped
- Crawl uses 1 credit per discovered page
- We'll use targeted scraping for efficiency

Target Sources:
1. Moneycontrol (highest priority)
2. Economic Times
3. Business Standard
4. LiveMint
5. NDTV Profit
6. BSE Announcements
7. NSE Announcements

Usage:
    python scripts/collect_news.py
    python scripts/collect_news.py --target 50000 --source moneycontrol

Cost: Uses Firecrawl credits (500K available)
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Set
import time
import json
import hashlib
import argparse
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
from rich.panel import Panel
from rich.table import Table

console = Console()

# Try to import firecrawl
try:
    from firecrawl import FirecrawlApp
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False
    console.print("[yellow]firecrawl not installed. Run: pip install firecrawl-py[/yellow]")


# ===========================================
# NEWS SOURCES CONFIGURATION
# ===========================================

NEWS_SOURCES = {
    "moneycontrol": {
        "name": "Moneycontrol",
        "base_url": "https://www.moneycontrol.com",
        "news_sections": [
            "/news/business/stocks/",
            "/news/business/markets/",
            "/news/business/earnings/",
            "/news/business/economy/",
            "/news/business/companies/",
            "/news/business/ipo/"
        ],
        "priority": 1,
        "estimated_articles": 50000,
        "credits_per_100_articles": 150  # Some pages have multiple articles
    },
    "economic_times": {
        "name": "Economic Times",
        "base_url": "https://economictimes.indiatimes.com",
        "news_sections": [
            "/markets/stocks/news",
            "/markets/stocks/earnings",
            "/markets/ipos/news",
            "/industry/banking/finance/news",
            "/industry/auto/news"
        ],
        "priority": 1,
        "estimated_articles": 40000,
        "credits_per_100_articles": 120
    },
    "business_standard": {
        "name": "Business Standard",
        "base_url": "https://www.business-standard.com",
        "news_sections": [
            "/markets/news",
            "/companies/news",
            "/economy/news"
        ],
        "priority": 2,
        "estimated_articles": 20000,
        "credits_per_100_articles": 100
    },
    "livemint": {
        "name": "LiveMint",
        "base_url": "https://www.livemint.com",
        "news_sections": [
            "/market/stock-market-news",
            "/market/ipo-news",
            "/companies/news"
        ],
        "priority": 2,
        "estimated_articles": 20000,
        "credits_per_100_articles": 100
    },
    "ndtv_profit": {
        "name": "NDTV Profit",
        "base_url": "https://www.ndtvprofit.com",
        "news_sections": [
            "/markets",
            "/business"
        ],
        "priority": 3,
        "estimated_articles": 10000,
        "credits_per_100_articles": 80
    }
}

# Stock symbols for entity detection
STOCK_SYMBOLS = {
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL",
    "HINDUNILVR", "ITC", "KOTAKBANK", "LT", "AXISBANK", "BAJFINANCE",
    "ASIANPAINT", "MARUTI", "TITAN", "SUNPHARMA", "TATAMOTORS", "WIPRO",
    "ONGC", "NTPC", "POWERGRID", "M&M", "ULTRACEMCO", "NESTLEIND",
    "BAJAJFINSV", "TATASTEEL", "JSWSTEEL", "TECHM", "HCLTECH", "ADANIENT",
    "ADANIPORTS", "COALINDIA", "DRREDDY", "CIPLA", "BRITANNIA", "GRASIM",
    "EICHERMOT", "DIVISLAB", "HEROMOTOCO", "BPCL", "HINDALCO", "TATACONSUM",
    "SBILIFE", "INDUSINDBK", "APOLLOHOSP", "BAJAJ-AUTO", "HDFCLIFE", "UPL"
}

# Company name to symbol mapping
COMPANY_TO_SYMBOL = {
    "reliance": "RELIANCE",
    "tata consultancy": "TCS",
    "tcs": "TCS",
    "infosys": "INFY",
    "hdfc bank": "HDFCBANK",
    "icici bank": "ICICIBANK",
    "state bank": "SBIN",
    "sbi": "SBIN",
    "bharti airtel": "BHARTIARTL",
    "airtel": "BHARTIARTL",
    "hindustan unilever": "HINDUNILVR",
    "hul": "HINDUNILVR",
    "itc": "ITC",
    "kotak": "KOTAKBANK",
    "larsen": "LT",
    "l&t": "LT",
    "axis bank": "AXISBANK",
    "bajaj finance": "BAJFINANCE",
    "asian paints": "ASIANPAINT",
    "maruti": "MARUTI",
    "titan": "TITAN",
    "sun pharma": "SUNPHARMA",
    "tata motors": "TATAMOTORS",
    "wipro": "WIPRO",
    "ongc": "ONGC",
    "ntpc": "NTPC",
    "mahindra": "M&M",
    "ultratech": "ULTRACEMCO",
    "nestle": "NESTLEIND",
    "tata steel": "TATASTEEL",
    "jsw steel": "JSWSTEEL",
    "tech mahindra": "TECHM",
    "hcl tech": "HCLTECH",
    "adani": "ADANIENT",
    "coal india": "COALINDIA",
    "dr reddy": "DRREDDY",
    "cipla": "CIPLA",
    "britannia": "BRITANNIA",
    "hero motocorp": "HEROMOTOCO",
    "hindalco": "HINDALCO",
}


class FirecrawlNewsCollector:
    """
    Collects financial news using Firecrawl
    
    Optimization strategies:
    1. Use crawl mode for discovery
    2. Scrape individual articles for content
    3. Track collected URLs to avoid duplicates
    4. Entity extraction for stock mentions
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('FIRECRAWL_API_KEY')
        
        if not self.api_key:
            console.print("[red]FIRECRAWL_API_KEY not set![/red]")
            console.print("Set in .env file or environment")
            return
        
        self.app = FirecrawlApp(api_key=self.api_key)
        
        # Data directory
        self.data_dir = Path("data/news")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Track collected URLs
        self.collected_urls: Set[str] = set()
        self._load_collected_urls()
        
        # Statistics
        self.total_credits_used = 0
        self.total_articles = 0
    
    def _load_collected_urls(self):
        """Load previously collected URLs"""
        url_file = self.data_dir / "collected_urls.txt"
        if url_file.exists():
            with open(url_file) as f:
                self.collected_urls = set(line.strip() for line in f if line.strip())
            console.print(f"[dim]Loaded {len(self.collected_urls)} existing URLs[/dim]")
    
    def _save_url(self, url: str):
        """Save URL to collected list"""
        url_file = self.data_dir / "collected_urls.txt"
        with open(url_file, "a") as f:
            f.write(url + "\n")
        self.collected_urls.add(url)
    
    def _extract_stocks(self, text: str) -> List[str]:
        """Extract stock mentions from text"""
        if not text:
            return []
        
        mentioned = []
        text_lower = text.lower()
        
        # Direct symbol matching
        words = set(re.findall(r'\b[A-Z][A-Z0-9&-]+\b', text))
        for word in words:
            if word in STOCK_SYMBOLS:
                mentioned.append(word)
        
        # Company name matching
        for pattern, symbol in COMPANY_TO_SYMBOL.items():
            if pattern in text_lower and symbol not in mentioned:
                mentioned.append(symbol)
        
        return list(set(mentioned))
    
    def _generate_article_id(self, url: str) -> str:
        """Generate unique ID for article"""
        return hashlib.md5(url.encode()).hexdigest()[:16]
    
    def scrape_article(self, url: str) -> Optional[Dict]:
        """
        Scrape a single article
        
        Cost: 1 credit
        """
        if url in self.collected_urls:
            return None
        
        try:
            result = self.app.scrape_url(
                url,
                params={
                    'formats': ['markdown'],
                    'onlyMainContent': True
                }
            )
            
            if not result:
                return None
            
            self.total_credits_used += 1
            
            # Extract metadata
            metadata = result.get('metadata', {})
            content = result.get('markdown', '')
            
            article = {
                'id': self._generate_article_id(url),
                'url': url,
                'title': metadata.get('title', ''),
                'description': metadata.get('description', ''),
                'content': content[:50000],  # Limit content size
                'published_at': metadata.get('publishedTime'),
                'author': metadata.get('author'),
                'source': self._get_source_from_url(url),
                'scraped_at': datetime.now().isoformat(),
                'stocks_mentioned': self._extract_stocks(content),
                'word_count': len(content.split())
            }
            
            # Save URL
            self._save_url(url)
            self.total_articles += 1
            
            return article
            
        except Exception as e:
            console.print(f"[red]Error scraping {url}: {e}[/red]")
            return None
    
    def _get_source_from_url(self, url: str) -> str:
        """Extract source name from URL"""
        if 'moneycontrol' in url:
            return 'moneycontrol'
        elif 'economictimes' in url:
            return 'economic_times'
        elif 'business-standard' in url:
            return 'business_standard'
        elif 'livemint' in url:
            return 'livemint'
        elif 'ndtvprofit' in url or 'ndtv' in url:
            return 'ndtv_profit'
        return 'unknown'
    
    def crawl_source(
        self,
        source_key: str,
        max_pages: int = 1000,
        max_articles: int = 10000
    ) -> List[Dict]:
        """
        Crawl a news source to discover and scrape articles
        
        Cost: ~1 credit per page discovered + 1 per article scraped
        
        Strategy:
        1. Crawl section pages to get article URLs
        2. Filter for news articles only
        3. Scrape each article
        """
        source = NEWS_SOURCES.get(source_key)
        if not source:
            console.print(f"[red]Unknown source: {source_key}[/red]")
            return []
        
        articles = []
        discovered_urls = set()
        
        console.print(f"\n[bold]Crawling {source['name']}[/bold]")
        
        # Crawl each news section
        for section in source['news_sections']:
            url = source['base_url'] + section
            
            console.print(f"[dim]  Discovering articles in {section}...[/dim]")
            
            try:
                # Use map for bulk URL processing
                result = self.app.crawl_url(
                    url,
                    params={
                        'limit': min(max_pages, 500),
                        'scrapeOptions': {
                            'formats': ['markdown'],
                            'onlyMainContent': True
                        }
                    },
                    poll_interval=5
                )
                
                if result and 'data' in result:
                    for page in result['data']:
                        page_url = page.get('metadata', {}).get('sourceURL', '')
                        
                        # Filter for article URLs (contain news/article patterns)
                        if self._is_article_url(page_url):
                            discovered_urls.add(page_url)
                        
                        # Also extract article from crawl result
                        if page.get('markdown'):
                            metadata = page.get('metadata', {})
                            content = page.get('markdown', '')
                            
                            if self._is_article_content(content):
                                article = {
                                    'id': self._generate_article_id(page_url),
                                    'url': page_url,
                                    'title': metadata.get('title', ''),
                                    'content': content[:50000],
                                    'published_at': metadata.get('publishedTime'),
                                    'source': source_key,
                                    'scraped_at': datetime.now().isoformat(),
                                    'stocks_mentioned': self._extract_stocks(content),
                                    'word_count': len(content.split())
                                }
                                
                                if article['title'] and page_url not in self.collected_urls:
                                    articles.append(article)
                                    self._save_url(page_url)
                                    self.total_articles += 1
                
                # Credit tracking
                self.total_credits_used += min(max_pages, 500)
                
            except Exception as e:
                console.print(f"[red]Error crawling {url}: {e}[/red]")
            
            # Check if we have enough articles
            if len(articles) >= max_articles:
                break
            
            time.sleep(2)  # Rate limiting
        
        console.print(f"[green]  Found {len(articles)} articles from {source['name']}[/green]")
        
        return articles
    
    def _is_article_url(self, url: str) -> bool:
        """Check if URL is likely an article"""
        if not url:
            return False
        
        # Patterns that indicate article URLs
        article_patterns = [
            r'/news/',
            r'/article/',
            r'/\d{4}/\d{2}/',  # Date pattern
            r'/markets/',
            r'/stocks/',
            r'/companies/',
            r'-\d+\.cms$',  # ET article pattern
            r'\.html$'
        ]
        
        # Patterns to exclude
        exclude_patterns = [
            r'/video/',
            r'/slideshow/',
            r'/photo/',
            r'/author/',
            r'/topic/',
            r'/tag/',
            r'page=',
            r'/login',
            r'/subscribe'
        ]
        
        url_lower = url.lower()
        
        # Check exclusions first
        for pattern in exclude_patterns:
            if re.search(pattern, url_lower):
                return False
        
        # Check for article patterns
        for pattern in article_patterns:
            if re.search(pattern, url_lower):
                return True
        
        return False
    
    def _is_article_content(self, content: str) -> bool:
        """Check if content looks like a news article"""
        if not content:
            return False
        
        word_count = len(content.split())
        
        # Articles should have reasonable length
        if word_count < 100 or word_count > 10000:
            return False
        
        return True
    
    def collect_from_all_sources(
        self,
        target_articles: int = 100000,
        max_credits: int = 500000
    ) -> List[Dict]:
        """
        Collect articles from all sources
        
        Strategy: Prioritize sources by quality and efficiency
        """
        all_articles = []
        
        # Sort sources by priority
        sorted_sources = sorted(
            NEWS_SOURCES.items(),
            key=lambda x: x[1]['priority']
        )
        
        # Calculate allocation
        remaining_credits = max_credits
        remaining_articles = target_articles
        
        for source_key, source_config in sorted_sources:
            if remaining_articles <= 0 or remaining_credits <= 1000:
                break
            
            # Allocate credits to this source
            source_allocation = min(
                remaining_credits // 2,  # Don't use more than half remaining
                source_config['estimated_articles'] * source_config['credits_per_100_articles'] // 100
            )
            
            max_articles_for_source = min(
                remaining_articles // (len(sorted_sources) - sorted_sources.index((source_key, source_config))),
                source_config['estimated_articles']
            )
            
            articles = self.crawl_source(
                source_key,
                max_pages=source_allocation,
                max_articles=max_articles_for_source
            )
            
            all_articles.extend(articles)
            remaining_articles -= len(articles)
            remaining_credits -= self.total_credits_used
            
            # Save batch
            self._save_batch(articles, source_key)
            
            console.print(f"[dim]Credits used so far: {self.total_credits_used}[/dim]")
        
        return all_articles
    
    def _save_batch(self, articles: List[Dict], batch_name: str):
        """Save batch of articles to JSON"""
        if not articles:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.data_dir / f"articles_{batch_name}_{timestamp}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        
        console.print(f"[dim]Saved {len(articles)} articles to {filepath.name}[/dim]")
    
    def get_statistics(self) -> Dict:
        """Get collection statistics"""
        # Count articles from files
        total_articles = 0
        total_size = 0
        sources_count = {}
        
        for filepath in self.data_dir.glob("articles_*.json"):
            total_size += filepath.stat().st_size
            
            with open(filepath) as f:
                articles = json.load(f)
                total_articles += len(articles)
                
                for article in articles:
                    source = article.get('source', 'unknown')
                    sources_count[source] = sources_count.get(source, 0) + 1
        
        return {
            'total_articles': total_articles,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'sources': sources_count,
            'credits_used': self.total_credits_used,
            'collected_urls': len(self.collected_urls)
        }


# ===========================================
# MAIN
# ===========================================

def main(target: int = 100000, source: str = None, max_credits: int = 500000):
    """Main collection function"""
    
    if not FIRECRAWL_AVAILABLE:
        console.print("[red]firecrawl package required. Install with:[/red]")
        console.print("pip install firecrawl-py")
        return
    
    console.print(Panel.fit(
        f"[bold blue]Firecrawl News Collection[/bold blue]\n"
        f"Target: {target:,} articles | Credits: {max_credits:,}",
        border_style="blue"
    ))
    
    # Check API key
    if not os.getenv('FIRECRAWL_API_KEY'):
        console.print("\n[yellow]FIRECRAWL_API_KEY not set![/yellow]")
        console.print("Set in your .env file:")
        console.print("  FIRECRAWL_API_KEY=your_api_key")
        console.print("\nGet your API key from: https://firecrawl.dev/")
        return
    
    collector = FirecrawlNewsCollector()
    
    if source:
        # Collect from specific source
        articles = collector.crawl_source(source, max_articles=target)
    else:
        # Collect from all sources
        articles = collector.collect_from_all_sources(
            target_articles=target,
            max_credits=max_credits
        )
    
    # Show statistics
    stats = collector.get_statistics()
    
    console.print("\n" + "="*50)
    console.print("[bold green]Collection Complete![/bold green]")
    
    table = Table(title="Collection Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Articles", f"{stats['total_articles']:,}")
    table.add_row("Total Size", f"{stats['total_size_mb']:.1f} MB")
    table.add_row("Credits Used", f"{stats['credits_used']:,}")
    table.add_row("Remaining Credits", f"{max_credits - stats['credits_used']:,}")
    
    console.print(table)
    
    if stats['sources']:
        console.print("\n[bold]Articles by Source:[/bold]")
        for src, count in sorted(stats['sources'].items(), key=lambda x: -x[1]):
            console.print(f"  {src}: {count:,}")
    
    console.print(f"\n  Data saved to: {collector.data_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect news with Firecrawl")
    parser.add_argument("--target", type=int, default=100000, help="Target articles")
    parser.add_argument("--source", type=str, help="Specific source to crawl")
    parser.add_argument("--credits", type=int, default=500000, help="Max credits to use")
    
    args = parser.parse_args()
    
    main(target=args.target, source=args.source, max_credits=args.credits)
