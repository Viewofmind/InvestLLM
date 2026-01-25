#!/usr/bin/env python3
"""
Fundamental Data Collection Script
==================================

Collects fundamental data using yfinance (FREE)

Data includes:
- Key ratios (PE, PB, ROE, etc.)
- Financial statements
- Company info
- Analyst recommendations

Usage:
    python scripts/collect_fundamentals.py
    python scripts/collect_fundamentals.py --symbols RELIANCE,TCS

Cost: FREE (yfinance)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import time
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table

console = Console()


# ===========================================
# STOCK LISTS
# ===========================================

NIFTY_100 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BPCL", "BHARTIARTL",
    "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC",
    "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LT",
    "M&M", "MARUTI", "NTPC", "NESTLEIND", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SUNPHARMA",
    "TCS", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TECHM",
    "TITAN", "ULTRACEMCO", "UPL", "WIPRO", "LTIM",
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


class FundamentalCollector:
    """Collects fundamental data from yfinance"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data/fundamentals")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.statements_dir = self.data_dir / "statements"
        self.statements_dir.mkdir(exist_ok=True)
    
    def get_fundamentals(self, symbol: str) -> Dict:
        """Get key fundamentals for a stock"""
        yf_symbol = f"{symbol}.NS"
        
        try:
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                
                # Market data
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'current_price': info.get('currentPrice'),
                'week_52_high': info.get('fiftyTwoWeekHigh'),
                'week_52_low': info.get('fiftyTwoWeekLow'),
                
                # Valuation
                'pe_trailing': info.get('trailingPE'),
                'pe_forward': info.get('forwardPE'),
                'pb_ratio': info.get('priceToBook'),
                'ps_ratio': info.get('priceToSalesTrailing12Months'),
                'peg_ratio': info.get('pegRatio'),
                'ev_to_ebitda': info.get('enterpriseToEbitda'),
                
                # Profitability
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'gross_margin': info.get('grossMargins'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                
                # Growth
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                
                # Financial health
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                
                # Dividends
                'dividend_yield': info.get('dividendYield'),
                'payout_ratio': info.get('payoutRatio'),
                
                # Per share
                'eps_trailing': info.get('trailingEps'),
                'eps_forward': info.get('forwardEps'),
                'book_value': info.get('bookValue'),
                
                # Analyst
                'analyst_count': info.get('numberOfAnalystOpinions'),
                'recommendation': info.get('recommendationKey'),
                'target_mean': info.get('targetMeanPrice'),
                
                # Beta
                'beta': info.get('beta'),
                
                'collected_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            console.print(f"[red]Error getting fundamentals for {symbol}: {e}[/red]")
            return {'symbol': symbol, 'error': str(e)}
    
    def get_financial_statements(self, symbol: str) -> Dict:
        """Get financial statements"""
        yf_symbol = f"{symbol}.NS"
        
        try:
            ticker = yf.Ticker(yf_symbol)
            
            return {
                'income_statement': ticker.income_stmt,
                'quarterly_income': ticker.quarterly_income_stmt,
                'balance_sheet': ticker.balance_sheet,
                'quarterly_balance': ticker.quarterly_balance_sheet,
                'cashflow': ticker.cashflow,
                'quarterly_cashflow': ticker.quarterly_cashflow
            }
            
        except Exception as e:
            console.print(f"[red]Error getting statements for {symbol}: {e}[/red]")
            return {}
    
    def collect_all(
        self,
        symbols: List[str],
        include_statements: bool = True,
        max_workers: int = 5
    ) -> pd.DataFrame:
        """Collect fundamentals for all symbols"""
        results = []
        failed = []
        
        console.print(f"\n[bold]Collecting fundamentals for {len(symbols)} stocks[/bold]\n")
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Collecting", total=len(symbols))
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.get_fundamentals, symbol): symbol
                    for symbol in symbols
                }
                
                for future in as_completed(futures):
                    symbol = futures[future]
                    
                    try:
                        data = future.result()
                        if 'error' not in data:
                            results.append(data)
                            
                            # Get statements too
                            if include_statements:
                                statements = self.get_financial_statements(symbol)
                                self._save_statements(symbol, statements)
                        else:
                            failed.append(symbol)
                            
                    except Exception as e:
                        failed.append(symbol)
                    
                    progress.update(task, advance=1)
                    time.sleep(0.5)
        
        # Save fundamentals
        df = pd.DataFrame(results)
        df.to_parquet(self.data_dir / "fundamentals.parquet", index=False)
        df.to_csv(self.data_dir / "fundamentals.csv", index=False)
        
        console.print(f"\n[green]Success: {len(results)}/{len(symbols)}[/green]")
        if failed:
            console.print(f"[red]Failed: {', '.join(failed[:10])}[/red]")
        
        return df
    
    def _save_statements(self, symbol: str, statements: Dict):
        """Save financial statements"""
        symbol_dir = self.statements_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        
        for name, df in statements.items():
            if df is not None and not df.empty:
                try:
                    df.to_parquet(symbol_dir / f"{name}.parquet")
                except:
                    pass  # Some dataframes may have issues


def main(symbols: List[str] = None, include_statements: bool = True):
    """Main function"""
    
    console.print(Panel.fit(
        "[bold blue]Fundamental Data Collection[/bold blue]\n"
        "Source: yfinance (FREE)",
        border_style="blue"
    ))
    
    symbols = symbols or NIFTY_100
    collector = FundamentalCollector()
    
    df = collector.collect_all(symbols, include_statements=include_statements)
    
    console.print("\n" + "="*50)
    console.print("[bold green]Collection Complete![/bold green]")
    console.print(f"  Saved to: {collector.data_dir}")
    
    # Show sample
    if not df.empty:
        console.print(f"\n[bold]Sample data:[/bold]")
        console.print(df[['symbol', 'name', 'market_cap', 'pe_trailing', 'roe']].head(10).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect fundamental data")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    parser.add_argument("--no-statements", action="store_true", help="Skip financial statements")
    
    args = parser.parse_args()
    
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    
    main(symbols=symbols, include_statements=not args.no_statements)
