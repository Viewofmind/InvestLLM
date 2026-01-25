#!/usr/bin/env python3
"""
Zerodha Kite Minute Data Collection
===================================

Collects minute-level historical data using Zerodha Kite API.

Zerodha Limits:
- Minute data: Max 60 days
- 5-minute data: Max 100 days
- 15-minute data: Max 200 days
- Day data: Max 2000 days (~5.5 years)

Requirements:
- Zerodha Kite API subscription (₹2000/month)
- API Key and Secret from https://kite.trade/

Usage:
    python scripts/collect_minute_data.py
    python scripts/collect_minute_data.py --interval 5minute --days 100

Cost: ₹2000/month (Zerodha Kite API)
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import time
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

# Try to import kiteconnect
try:
    from kiteconnect import KiteConnect
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False
    console.print("[yellow]kiteconnect not installed. Run: pip install kiteconnect[/yellow]")


# ===========================================
# STOCK LISTS
# ===========================================

NIFTY_50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BPCL", "BHARTIARTL",
    "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC",
    "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LT",
    "M&M", "MARUTI", "NTPC", "NESTLEIND", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SUNPHARMA",
    "TCS", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TECHM",
    "TITAN", "ULTRACEMCO", "UPL", "WIPRO", "LTIM"
]


class ZerodhaDataCollector:
    """
    Collects minute-level data from Zerodha Kite API
    
    Authentication flow:
    1. Generate login URL
    2. User logs in and gets request_token
    3. Exchange for access_token
    4. Use access_token for API calls
    """
    
    INTERVAL_LIMITS = {
        'minute': 60,      # Max 60 days
        '3minute': 100,
        '5minute': 100,
        '10minute': 100,
        '15minute': 200,
        '30minute': 200,
        '60minute': 400,
        'day': 2000        # ~5.5 years
    }
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key or os.getenv('ZERODHA_API_KEY')
        self.api_secret = api_secret or os.getenv('ZERODHA_API_SECRET')
        self.access_token = os.getenv('ZERODHA_ACCESS_TOKEN')
        
        if not self.api_key or not self.api_secret:
            console.print("[red]Missing ZERODHA_API_KEY or ZERODHA_API_SECRET[/red]")
            console.print("Set in .env file or environment variables")
            return
        
        self.kite = KiteConnect(api_key=self.api_key)
        self.instruments_cache = {}
        
        # Data directory
        self.data_dir = Path("data/raw/minute")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def get_login_url(self) -> str:
        """Get Zerodha login URL"""
        return self.kite.login_url()
    
    def authenticate(self, request_token: str = None):
        """
        Complete authentication with request_token
        
        After user logs in via login_url, they get redirected with request_token
        """
        if self.access_token:
            self.kite.set_access_token(self.access_token)
            console.print("[green]Using existing access token[/green]")
            return True
        
        if not request_token:
            # Interactive authentication
            login_url = self.get_login_url()
            console.print(f"\n[bold]Step 1: Open this URL in browser:[/bold]")
            console.print(f"[blue]{login_url}[/blue]")
            console.print("\n[bold]Step 2: Log in with your Zerodha credentials[/bold]")
            console.print("\n[bold]Step 3: After login, copy the 'request_token' from the URL[/bold]")
            console.print("(URL will look like: https://yourapp.com/?request_token=XXXXX&...)")
            
            request_token = Prompt.ask("\nEnter request_token")
        
        try:
            data = self.kite.generate_session(
                request_token,
                api_secret=self.api_secret
            )
            
            self.access_token = data['access_token']
            self.kite.set_access_token(self.access_token)
            
            console.print(f"\n[green]Authentication successful![/green]")
            console.print(f"[dim]Save this access_token in .env for future use:[/dim]")
            console.print(f"ZERODHA_ACCESS_TOKEN={self.access_token}")
            
            return True
            
        except Exception as e:
            console.print(f"[red]Authentication failed: {e}[/red]")
            return False
    
    def get_instrument_token(self, symbol: str, exchange: str = "NSE") -> Optional[int]:
        """Get instrument token for a symbol"""
        cache_key = f"{exchange}:{symbol}"
        
        if cache_key in self.instruments_cache:
            return self.instruments_cache[cache_key]
        
        try:
            # Fetch instruments list (cache for efficiency)
            if not hasattr(self, '_instruments_df'):
                instruments = self.kite.instruments(exchange)
                self._instruments_df = pd.DataFrame(instruments)
            
            # Find instrument
            match = self._instruments_df[
                self._instruments_df['tradingsymbol'] == symbol
            ]
            
            if not match.empty:
                token = match.iloc[0]['instrument_token']
                self.instruments_cache[cache_key] = token
                return token
            
            return None
            
        except Exception as e:
            console.print(f"[red]Error getting instrument token for {symbol}: {e}[/red]")
            return None
    
    def download_historical(
        self,
        symbol: str,
        interval: str = 'minute',
        days: int = None,
        exchange: str = "NSE"
    ) -> Optional[pd.DataFrame]:
        """
        Download historical data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., RELIANCE)
            interval: minute, 5minute, 15minute, 60minute, day
            days: Number of days (max depends on interval)
            exchange: NSE or BSE
            
        Returns:
            DataFrame with OHLCV data
        """
        # Get max days for interval
        max_days = self.INTERVAL_LIMITS.get(interval, 60)
        days = min(days or max_days, max_days)
        
        # Get instrument token
        token = self.get_instrument_token(symbol, exchange)
        if not token:
            return None
        
        # Calculate dates
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        try:
            data = self.kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            if not data:
                return None
            
            df = pd.DataFrame(data)
            df['symbol'] = symbol
            df['exchange'] = exchange
            df['interval'] = interval
            
            # Rename columns
            df = df.rename(columns={'date': 'timestamp'})
            
            return df
            
        except Exception as e:
            console.print(f"[red]Error downloading {symbol}: {e}[/red]")
            return None
    
    def collect_all(
        self,
        symbols: List[str],
        interval: str = 'minute',
        days: int = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect data for all symbols
        
        Note: Zerodha has rate limits, so we add delays
        """
        results = {}
        failed = []
        
        console.print(f"\n[bold]Downloading {len(symbols)} stocks ({interval} data)[/bold]\n")
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Downloading", total=len(symbols))
            
            for symbol in symbols:
                df = self.download_historical(symbol, interval, days)
                
                if df is not None and not df.empty:
                    # Save to file
                    filepath = self.data_dir / f"{symbol}_{interval}.parquet"
                    df.to_parquet(filepath, index=False)
                    results[symbol] = df
                else:
                    failed.append(symbol)
                
                progress.update(task, advance=1)
                time.sleep(0.5)  # Rate limiting (Zerodha allows ~3 req/sec)
        
        console.print(f"\n[green]Success: {len(results)}/{len(symbols)}[/green]")
        if failed:
            console.print(f"[red]Failed: {', '.join(failed[:10])}[/red]")
        
        return results


# ===========================================
# MAIN
# ===========================================

def main(interval: str = 'minute', days: int = None, symbols: List[str] = None):
    """Main collection function"""
    
    if not KITE_AVAILABLE:
        console.print("[red]kiteconnect package required. Install with:[/red]")
        console.print("pip install kiteconnect")
        return
    
    console.print(Panel.fit(
        f"[bold blue]Zerodha Minute Data Collection[/bold blue]\n"
        f"Interval: {interval} | Source: Zerodha Kite API",
        border_style="blue"
    ))
    
    # Check environment variables
    if not os.getenv('ZERODHA_API_KEY'):
        console.print("\n[yellow]ZERODHA_API_KEY not set in environment[/yellow]")
        console.print("Set it in your .env file:")
        console.print("  ZERODHA_API_KEY=your_api_key")
        console.print("  ZERODHA_API_SECRET=your_api_secret")
        console.print("\nGet your API key from: https://kite.trade/")
        return
    
    symbols = symbols or NIFTY_50
    
    collector = ZerodhaDataCollector()
    
    # Authenticate
    if not collector.authenticate():
        return
    
    # Collect data
    results = collector.collect_all(symbols, interval, days)
    
    # Summary
    console.print("\n" + "="*50)
    console.print("[bold green]Collection Complete![/bold green]")
    console.print(f"  Data saved to: {collector.data_dir}")
    
    # Show sample
    if results:
        sample_symbol = list(results.keys())[0]
        sample_df = results[sample_symbol]
        console.print(f"\n[bold]Sample data ({sample_symbol}):[/bold]")
        console.print(f"  Rows: {len(sample_df)}")
        console.print(f"  Date range: {sample_df['timestamp'].min()} to {sample_df['timestamp'].max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect minute data from Zerodha")
    parser.add_argument("--interval", type=str, default="minute",
                       choices=['minute', '3minute', '5minute', '10minute', '15minute', '30minute', '60minute', 'day'],
                       help="Data interval")
    parser.add_argument("--days", type=int, help="Number of days (max depends on interval)")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    
    args = parser.parse_args()
    
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    
    main(interval=args.interval, days=args.days, symbols=symbols)
