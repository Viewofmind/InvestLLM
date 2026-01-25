#!/usr/bin/env python3
"""
InvestLLM Data Collection Master Script
=======================================

This script collects:
1. 20 years of price data (FREE via yfinance)
2. 5 years of minute data (Zerodha Kite)
3. News corpus (Firecrawl - 500K credits)

Run: python scripts/collect_all_data.py

Author: InvestLLM Team
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

console = Console()


def print_banner():
    console.print(Panel.fit(
        "[bold blue]InvestLLM Data Collection[/bold blue]\n"
        "Collecting 20 years of Indian market data",
        border_style="blue"
    ))


def collect_price_data(years: int = 20, symbols: list = None):
    """Collect historical price data"""
    from scripts.collect_prices import main as collect_prices
    collect_prices(years=years, symbols=symbols)


def collect_minute_data():
    """Collect minute-level data from Zerodha"""
    from scripts.collect_minute_data import main as collect_minute
    collect_minute()


def collect_news(target_articles: int = 100000):
    """Collect news articles"""
    from scripts.collect_news import main as collect_news_main
    collect_news_main(target=target_articles)


def collect_fundamentals():
    """Collect fundamental data"""
    from scripts.collect_fundamentals import main as collect_fund
    collect_fund()


def show_status():
    """Show data collection status"""
    data_dir = Path("data")
    
    table = Table(title="Data Collection Status")
    table.add_column("Data Type", style="cyan")
    table.add_column("Files", style="magenta")
    table.add_column("Size", style="green")
    table.add_column("Status", style="yellow")
    
    # Check price data
    price_dir = data_dir / "raw" / "prices"
    if price_dir.exists():
        files = list(price_dir.glob("*.parquet"))
        size = sum(f.stat().st_size for f in files) / (1024*1024)
        status = "✅ Complete" if len(files) >= 100 else f"🟡 {len(files)}/100"
        table.add_row("Price Data (EOD)", str(len(files)), f"{size:.1f} MB", status)
    else:
        table.add_row("Price Data (EOD)", "0", "0 MB", "❌ Not started")
    
    # Check minute data
    minute_dir = data_dir / "raw" / "minute"
    if minute_dir.exists():
        files = list(minute_dir.glob("*.parquet"))
        size = sum(f.stat().st_size for f in files) / (1024*1024)
        table.add_row("Minute Data", str(len(files)), f"{size:.1f} MB", "✅" if files else "❌")
    else:
        table.add_row("Minute Data", "0", "0 MB", "❌ Not started")
    
    # Check news
    news_dir = data_dir / "news"
    if news_dir.exists():
        files = list(news_dir.glob("*.json"))
        size = sum(f.stat().st_size for f in files) / (1024*1024)
        # Count articles
        article_count = 0
        for f in files:
            try:
                import json
                with open(f) as fp:
                    article_count += len(json.load(fp))
            except:
                pass
        status = "✅" if article_count >= 100000 else f"🟡 {article_count:,}"
        table.add_row("News Articles", f"{article_count:,}", f"{size:.1f} MB", status)
    else:
        table.add_row("News Articles", "0", "0 MB", "❌ Not started")
    
    # Check fundamentals
    fund_dir = data_dir / "fundamentals"
    if fund_dir.exists():
        files = list(fund_dir.glob("*.parquet")) + list(fund_dir.glob("*.csv"))
        size = sum(f.stat().st_size for f in files) / (1024*1024)
        table.add_row("Fundamentals", str(len(files)), f"{size:.1f} MB", "✅" if files else "❌")
    else:
        table.add_row("Fundamentals", "0", "0 MB", "❌ Not started")
    
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="InvestLLM Data Collection")
    parser.add_argument("--prices", action="store_true", help="Collect price data")
    parser.add_argument("--minute", action="store_true", help="Collect minute data (Zerodha)")
    parser.add_argument("--news", action="store_true", help="Collect news articles")
    parser.add_argument("--fundamentals", action="store_true", help="Collect fundamentals")
    parser.add_argument("--all", action="store_true", help="Collect everything")
    parser.add_argument("--status", action="store_true", help="Show collection status")
    parser.add_argument("--years", type=int, default=20, help="Years of price history")
    parser.add_argument("--articles", type=int, default=100000, help="Target news articles")
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.status:
        show_status()
        return
    
    if args.all or args.prices:
        console.print("\n[bold]📈 Collecting Price Data[/bold]")
        collect_price_data(years=args.years)
    
    if args.all or args.minute:
        console.print("\n[bold]⏱️ Collecting Minute Data[/bold]")
        collect_minute_data()
    
    if args.all or args.news:
        console.print("\n[bold]📰 Collecting News[/bold]")
        collect_news(target_articles=args.articles)
    
    if args.all or args.fundamentals:
        console.print("\n[bold]📊 Collecting Fundamentals[/bold]")
        collect_fundamentals()
    
    if not any([args.all, args.prices, args.minute, args.news, args.fundamentals, args.status]):
        console.print("[yellow]No action specified. Use --help for options.[/yellow]")
        console.print("\nQuick start:")
        console.print("  python scripts/collect_all_data.py --status    # Check status")
        console.print("  python scripts/collect_all_data.py --prices    # Collect prices")
        console.print("  python scripts/collect_all_data.py --all       # Collect everything")
    
    console.print("\n[bold green]Done![/bold green]")
    show_status()


if __name__ == "__main__":
    main()
