#!/usr/bin/env python3
"""
Price Data Collection Script
============================

Collects 20 years of FREE historical price data using yfinance.

Features:
- Parallel downloading (10 workers)
- Progress tracking
- Automatic retry on failure
- Data validation
- Parquet storage (efficient)

Usage:
    python scripts/collect_prices.py
    python scripts/collect_prices.py --years 10 --symbols RELIANCE,TCS,INFY

Cost: FREE (yfinance uses Yahoo Finance API)
Time: ~15-20 minutes for 100 stocks, 20 years
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import time
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

console = Console()

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

NIFTY_NEXT_50 = [
    "ABB", "ADANIGREEN", "AMBUJACEM", "ATGL", "BAJAJHLDNG",
    "BANKBARODA", "BEL", "BERGEPAINT", "BOSCHLTD", "CANBK",
    "CHOLAFIN", "COLPAL", "DLF", "DABUR", "DMART",
    "GAIL", "GODREJCP", "HAVELLS", "HINDPETRO", "ICICIPRULI",
    "ICICIGI", "INDIGO", "IOC", "IRCTC", "JINDALSTEL",
    "LICI", "LUPIN", "MARICO", "NAUKRI", "NHPC",
    "OFSS", "PAGEIND", "PIIND", "PFC", "PIDILITIND",
    "PNB", "RECLTD", "SAIL", "SBICARD", "SHREECEM",
    "SIEMENS", "SRF", "TATAPOWER", "TORNTPHARM", "TRENT",
    "TVSMOTOR", "UNIONBANK", "VBL", "VEDL", "ZOMATO"
]

INDICES = {
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "NIFTYIT": "^CNXIT",
    "NIFTYPHARMA": "^CNXPHARMA",
    "NIFTYAUTO": "^CNXAUTO",
    "NIFTYFMCG": "^CNXFMCG",
    "NIFTYMETAL": "^CNXMETAL",
    "NIFTYREALTY": "^CNXREALTY",
    "NIFTYENERGY": "^CNXENERGY",
    "INDIAVIX": "^INDIAVIX"
}


# ===========================================
# DATA COLLECTION
# ===========================================

class PriceDataCollector:
    """
    Collects price data from yfinance (FREE)
    
    yfinance provides:
    - 20+ years of daily data
    - Adjusted prices (for splits/dividends)
    - Volume data
    - Free, no API key required
    """
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data/raw/prices")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.failed_symbols = []
        self.success_count = 0
    
    def download_stock(
        self,
        symbol: str,
        years: int = 20,
        exchange: str = "NSE"
    ) -> Optional[pd.DataFrame]:
        """
        Download historical data for a single stock
        
        Args:
            symbol: Stock symbol (e.g., RELIANCE)
            years: Years of history
            exchange: NSE or BSE
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        # Build yfinance symbol
        suffix = ".NS" if exchange == "NSE" else ".BO"
        yf_symbol = f"{symbol}{suffix}"
        
        # Calculate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                auto_adjust=True  # Adjust for splits/dividends
            )
            
            if df.empty:
                # Try BSE if NSE fails
                if exchange == "NSE":
                    return self.download_stock(symbol, years, "BSE")
                return None
            
            # Clean up columns
            df = df.reset_index()
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            
            # Rename 'date' to 'timestamp'
            if 'date' in df.columns:
                df = df.rename(columns={'date': 'timestamp'})
            
            # Add metadata
            df['symbol'] = symbol
            df['exchange'] = exchange
            
            # Remove unnecessary columns
            keep_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'exchange']
            df = df[[c for c in keep_cols if c in df.columns]]
            
            return df
            
        except Exception as e:
            console.print(f"[red]Error downloading {symbol}: {e}[/red]")
            return None
    
    def download_index(self, name: str, yf_symbol: str, years: int = 20) -> Optional[pd.DataFrame]:
        """Download index data"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                return None
            
            df = df.reset_index()
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            
            if 'date' in df.columns:
                df = df.rename(columns={'date': 'timestamp'})
            
            df['symbol'] = name
            df['exchange'] = 'INDEX'
            
            return df
            
        except Exception as e:
            console.print(f"[red]Error downloading index {name}: {e}[/red]")
            return None
    
    def save_data(self, symbol: str, df: pd.DataFrame):
        """Save data to parquet file"""
        filepath = self.data_dir / f"{symbol}.parquet"
        df.to_parquet(filepath, index=False)
    
    def load_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data from parquet file"""
        filepath = self.data_dir / f"{symbol}.parquet"
        if filepath.exists():
            return pd.read_parquet(filepath)
        return None
    
    def collect_all(
        self,
        symbols: List[str],
        years: int = 20,
        max_workers: int = 10,
        skip_existing: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect data for all symbols in parallel
        
        Args:
            symbols: List of stock symbols
            years: Years of history
            max_workers: Parallel download threads
            skip_existing: Skip if file already exists
            
        Returns:
            Dictionary of symbol -> DataFrame
        """
        results = {}
        to_download = []
        
        # Check existing files
        for symbol in symbols:
            filepath = self.data_dir / f"{symbol}.parquet"
            if skip_existing and filepath.exists():
                console.print(f"[dim]Skipping {symbol} (exists)[/dim]")
                results[symbol] = pd.read_parquet(filepath)
            else:
                to_download.append(symbol)
        
        if not to_download:
            console.print("[green]All data already collected![/green]")
            return results
        
        console.print(f"\n[bold]Downloading {len(to_download)} stocks...[/bold]\n")
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Downloading", total=len(to_download))
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.download_stock, symbol, years): symbol
                    for symbol in to_download
                }
                
                for future in as_completed(futures):
                    symbol = futures[future]
                    
                    try:
                        df = future.result()
                        
                        if df is not None and not df.empty:
                            self.save_data(symbol, df)
                            results[symbol] = df
                            self.success_count += 1
                        else:
                            self.failed_symbols.append(symbol)
                            
                    except Exception as e:
                        self.failed_symbols.append(symbol)
                        console.print(f"[red]Failed {symbol}: {e}[/red]")
                    
                    progress.update(task, advance=1)
                    time.sleep(0.1)  # Small delay to avoid rate limiting
        
        return results
    
    def collect_indices(self, years: int = 20) -> Dict[str, pd.DataFrame]:
        """Collect all index data"""
        results = {}
        
        console.print(f"\n[bold]Downloading {len(INDICES)} indices...[/bold]\n")
        
        for name, yf_symbol in INDICES.items():
            df = self.download_index(name, yf_symbol, years)
            
            if df is not None and not df.empty:
                # Save to indices subdirectory
                index_dir = self.data_dir.parent / "indices"
                index_dir.mkdir(exist_ok=True)
                df.to_parquet(index_dir / f"{name}.parquet", index=False)
                results[name] = df
                console.print(f"[green]✓ {name}[/green] ({len(df)} rows)")
            else:
                console.print(f"[red]✗ {name}[/red]")
        
        return results
    
    def validate_data(self) -> pd.DataFrame:
        """Validate collected data and generate report"""
        report_data = []
        
        for filepath in self.data_dir.glob("*.parquet"):
            symbol = filepath.stem
            df = pd.read_parquet(filepath)
            
            # Calculate metrics
            start_date = df['timestamp'].min()
            end_date = df['timestamp'].max()
            days = (end_date - start_date).days
            rows = len(df)
            
            # Check for gaps (more than 5 consecutive missing days)
            df_sorted = df.sort_values('timestamp')
            df_sorted['gap'] = df_sorted['timestamp'].diff()
            large_gaps = len(df_sorted[df_sorted['gap'] > pd.Timedelta(days=5)])
            
            # Check for nulls
            null_count = df.isnull().sum().sum()
            
            report_data.append({
                'symbol': symbol,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'days': days,
                'rows': rows,
                'coverage': round(rows / (days * 0.7) * 100, 1),  # ~70% are trading days
                'large_gaps': large_gaps,
                'null_values': null_count,
                'status': '✓' if large_gaps == 0 and null_count == 0 else '⚠'
            })
        
        return pd.DataFrame(report_data)


# ===========================================
# MAIN
# ===========================================

def main(years: int = 20, symbols: List[str] = None):
    """Main collection function"""
    
    console.print(Panel.fit(
        f"[bold blue]Price Data Collection[/bold blue]\n"
        f"Years: {years} | Source: yfinance (FREE)",
        border_style="blue"
    ))
    
    # Default to NIFTY 100
    if symbols is None:
        symbols = NIFTY_50 + NIFTY_NEXT_50
    
    collector = PriceDataCollector()
    
    # Collect stock data
    console.print(f"\n[bold]Phase 1: Collecting {len(symbols)} stocks[/bold]")
    results = collector.collect_all(symbols, years=years)
    
    # Collect index data
    console.print(f"\n[bold]Phase 2: Collecting indices[/bold]")
    indices = collector.collect_indices(years=years)
    
    # Validation report
    console.print(f"\n[bold]Phase 3: Validating data[/bold]")
    report = collector.validate_data()
    
    # Save report
    report_path = Path("data/price_collection_report.csv")
    report.to_csv(report_path, index=False)
    
    # Summary
    console.print("\n" + "="*50)
    console.print("[bold green]Collection Complete![/bold green]")
    console.print(f"  Stocks: {collector.success_count}/{len(symbols)}")
    console.print(f"  Indices: {len(indices)}/{len(INDICES)}")
    console.print(f"  Failed: {len(collector.failed_symbols)}")
    
    if collector.failed_symbols:
        console.print(f"  Failed symbols: {', '.join(collector.failed_symbols[:10])}")
    
    console.print(f"\n  Data saved to: {collector.data_dir}")
    console.print(f"  Report saved to: {report_path}")
    
    # Show sample data
    if results:
        sample_symbol = list(results.keys())[0]
        sample_df = results[sample_symbol]
        console.print(f"\n[bold]Sample data ({sample_symbol}):[/bold]")
        console.print(sample_df.head().to_string())


if __name__ == "__main__":
    from rich.panel import Panel
    
    parser = argparse.ArgumentParser(description="Collect price data")
    parser.add_argument("--years", type=int, default=20, help="Years of history")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    
    args = parser.parse_args()
    
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    
    main(years=args.years, symbols=symbols)
