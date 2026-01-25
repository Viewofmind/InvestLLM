"""
Price Data Collector
Collects historical and real-time price data from multiple sources
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential
import time

from investllm.config import settings, STOCK_UNIVERSE, NIFTY_50, INDICES

logger = structlog.get_logger()


class PriceCollector:
    """
    Collects price data from multiple sources:
    - yfinance: Free historical data (primary for EOD)
    - NSE Bhavcopy: Official daily data
    - Zerodha Kite: Real-time and minute data (paid)
    
    Usage:
        collector = PriceCollector()
        
        # Collect all historical data
        collector.collect_all_historical()
        
        # Daily update
        collector.update_daily()
        
        # Get single stock
        df = collector.get_stock_history("RELIANCE", years=10)
    """
    
    NSE_SUFFIX = ".NS"
    BSE_SUFFIX = ".BO"
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.data_dir / "raw" / "prices"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Processed data directory
        self.processed_dir = settings.data_dir / "processed" / "prices"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("price_collector_initialized", data_dir=str(self.data_dir))
    
    # ===========================================
    # SINGLE STOCK METHODS
    # ===========================================
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def get_stock_history(
        self,
        symbol: str,
        years: int = 20,
        interval: str = "1d",
        exchange: str = "NSE"
    ) -> pd.DataFrame:
        """
        Get historical data for a single stock
        
        Args:
            symbol: Stock symbol (e.g., RELIANCE)
            years: Years of history (max ~20 for daily)
            interval: 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo
            exchange: NSE or BSE
            
        Returns:
            DataFrame with OHLCV data
        """
        # Build yfinance symbol
        suffix = self.NSE_SUFFIX if exchange == "NSE" else self.BSE_SUFFIX
        yf_symbol = f"{symbol}{suffix}"
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True
            )
            
            if df.empty:
                logger.warning("no_data_found", symbol=symbol)
                return pd.DataFrame()
            
            # Clean up
            df = df.reset_index()
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            
            # Add metadata
            df['symbol'] = symbol
            df['exchange'] = exchange
            
            # Rename 'date' to 'timestamp' if present
            if 'date' in df.columns:
                df = df.rename(columns={'date': 'timestamp'})
            
            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            logger.info(
                "stock_history_fetched",
                symbol=symbol,
                rows=len(df),
                start=df['timestamp'].min().strftime('%Y-%m-%d'),
                end=df['timestamp'].max().strftime('%Y-%m-%d')
            )
            
            return df
            
        except Exception as e:
            logger.error("stock_history_error", symbol=symbol, error=str(e))
            raise
    
    def get_intraday_history(
        self,
        symbol: str,
        days: int = 60,
        interval: str = "5m"
    ) -> pd.DataFrame:
        """
        Get intraday data (max 60 days for minute data)
        
        Args:
            symbol: Stock symbol
            days: Days of history (max 60 for minute data)
            interval: 1m, 5m, 15m, 30m, 1h
            
        Returns:
            DataFrame with intraday OHLCV
        """
        yf_symbol = f"{symbol}{self.NSE_SUFFIX}"
        
        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(
                period=f"{min(days, 60)}d",
                interval=interval
            )
            
            if df.empty:
                return pd.DataFrame()
            
            df = df.reset_index()
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            df['symbol'] = symbol
            
            if 'datetime' in df.columns:
                df = df.rename(columns={'datetime': 'timestamp'})
            
            logger.info("intraday_fetched", symbol=symbol, rows=len(df), interval=interval)
            return df
            
        except Exception as e:
            logger.error("intraday_error", symbol=symbol, error=str(e))
            return pd.DataFrame()
    
    # ===========================================
    # BULK COLLECTION METHODS
    # ===========================================
    
    def collect_all_historical(
        self,
        symbols: List[str] = None,
        years: int = 20,
        save: bool = True,
        max_workers: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect historical data for all stocks in parallel
        
        Args:
            symbols: List of symbols (default: STOCK_UNIVERSE)
            years: Years of history
            save: Whether to save to disk
            max_workers: Parallel workers
            
        Returns:
            Dictionary of symbol -> DataFrame
        """
        symbols = symbols or STOCK_UNIVERSE
        
        logger.info("starting_bulk_collection", count=len(symbols), years=years)
        
        results = {}
        failed = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.get_stock_history, symbol, years): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                
                try:
                    df = future.result()
                    
                    if not df.empty:
                        results[symbol] = df
                        
                        if save:
                            self._save_stock_data(symbol, df)
                            
                except Exception as e:
                    logger.error("bulk_fetch_failed", symbol=symbol, error=str(e))
                    failed.append(symbol)
                
                # Rate limiting
                time.sleep(0.1)
        
        logger.info(
            "bulk_collection_complete",
            success=len(results),
            failed=len(failed),
            failed_symbols=failed[:10]  # Log first 10
        )
        
        return results
    
    def collect_indices(self, years: int = 20, save: bool = True) -> Dict[str, pd.DataFrame]:
        """Collect index data"""
        results = {}
        
        for index_symbol in INDICES:
            try:
                ticker = yf.Ticker(index_symbol)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=years * 365)
                
                df = ticker.history(start=start_date, end=end_date)
                
                if not df.empty:
                    df = df.reset_index()
                    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
                    
                    # Clean symbol name
                    clean_name = index_symbol.replace("^", "").replace("NSE", "")
                    df['symbol'] = clean_name
                    
                    results[clean_name] = df
                    
                    if save:
                        filepath = self.data_dir / f"index_{clean_name}.parquet"
                        df.to_parquet(filepath)
                        logger.info("index_saved", symbol=clean_name, rows=len(df))
                        
            except Exception as e:
                logger.error("index_fetch_failed", symbol=index_symbol, error=str(e))
        
        return results
    
    def update_daily(self, symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Update with latest daily data
        Only fetches data from last available date
        """
        symbols = symbols or STOCK_UNIVERSE
        
        logger.info("starting_daily_update", count=len(symbols))
        
        results = {}
        
        for symbol in symbols:
            try:
                # Load existing data
                existing_file = self.data_dir / f"{symbol}.parquet"
                
                if existing_file.exists():
                    existing = pd.read_parquet(existing_file)
                    last_date = existing['timestamp'].max()
                    start_date = last_date + timedelta(days=1)
                else:
                    start_date = datetime.now() - timedelta(days=30)
                
                # Fetch new data
                yf_symbol = f"{symbol}{self.NSE_SUFFIX}"
                ticker = yf.Ticker(yf_symbol)
                
                new_data = ticker.history(start=start_date)
                
                if not new_data.empty:
                    new_data = new_data.reset_index()
                    new_data.columns = [c.lower().replace(' ', '_') for c in new_data.columns]
                    new_data['symbol'] = symbol
                    
                    if 'date' in new_data.columns:
                        new_data = new_data.rename(columns={'date': 'timestamp'})
                    
                    # Append to existing
                    if existing_file.exists():
                        combined = pd.concat([existing, new_data])
                        combined = combined.drop_duplicates(subset=['timestamp'])
                        combined = combined.sort_values('timestamp')
                    else:
                        combined = new_data
                    
                    # Save
                    combined.to_parquet(existing_file)
                    results[symbol] = new_data
                    
                    logger.info("daily_update", symbol=symbol, new_rows=len(new_data))
                    
            except Exception as e:
                logger.error("daily_update_failed", symbol=symbol, error=str(e))
            
            time.sleep(0.05)  # Rate limit
        
        return results
    
    # ===========================================
    # SAVE/LOAD METHODS
    # ===========================================
    
    def _save_stock_data(self, symbol: str, df: pd.DataFrame):
        """Save stock data to parquet"""
        filepath = self.data_dir / f"{symbol}.parquet"
        df.to_parquet(filepath, index=False)
        logger.debug("data_saved", symbol=symbol, filepath=str(filepath))
    
    def load_stock_data(self, symbol: str) -> pd.DataFrame:
        """Load stock data from parquet"""
        filepath = self.data_dir / f"{symbol}.parquet"
        
        if not filepath.exists():
            logger.warning("file_not_found", symbol=symbol)
            return pd.DataFrame()
        
        return pd.read_parquet(filepath)
    
    def load_all_data(self, symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Load all available stock data"""
        symbols = symbols or STOCK_UNIVERSE
        
        results = {}
        for symbol in symbols:
            df = self.load_stock_data(symbol)
            if not df.empty:
                results[symbol] = df
        
        return results
    
    # ===========================================
    # DATA QUALITY CHECKS
    # ===========================================
    
    def check_data_quality(self, symbol: str) -> Dict:
        """Check data quality for a stock"""
        df = self.load_stock_data(symbol)
        
        if df.empty:
            return {"symbol": symbol, "status": "no_data"}
        
        # Calculate metrics
        total_days = (df['timestamp'].max() - df['timestamp'].min()).days
        actual_rows = len(df)
        expected_rows = total_days * 0.7  # ~70% are trading days
        
        # Check for gaps
        df = df.sort_values('timestamp')
        df['gap'] = df['timestamp'].diff()
        large_gaps = df[df['gap'] > timedelta(days=5)]
        
        return {
            "symbol": symbol,
            "status": "ok",
            "start_date": df['timestamp'].min().strftime('%Y-%m-%d'),
            "end_date": df['timestamp'].max().strftime('%Y-%m-%d'),
            "total_rows": actual_rows,
            "coverage_ratio": actual_rows / expected_rows if expected_rows > 0 else 0,
            "large_gaps": len(large_gaps),
            "null_values": df.isnull().sum().to_dict()
        }
    
    def generate_quality_report(self) -> pd.DataFrame:
        """Generate quality report for all stocks"""
        reports = []
        
        for symbol in STOCK_UNIVERSE:
            report = self.check_data_quality(symbol)
            reports.append(report)
        
        return pd.DataFrame(reports)


# ===========================================
# NSE BHAVCOPY COLLECTOR (Bonus - Official Data)
# ===========================================

class NSEBhavcopyCollector:
    """
    Collects official Bhavcopy data from NSE
    This is the official end-of-day data
    """
    
    NSE_BHAVCOPY_URL = "https://www.nseindia.com/api/historical/cm/equity"
    
    def __init__(self):
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        }
    
    async def get_bhavcopy(self, date: datetime) -> pd.DataFrame:
        """
        Get Bhavcopy for a specific date
        
        Note: NSE has strict rate limiting and anti-scraping
        Use yfinance as primary, this as verification
        """
        # NSE requires session cookies
        # Implementation would need selenium or playwright for cookie handling
        # Keeping as placeholder
        raise NotImplementedError("NSE Bhavcopy requires authenticated session")


# ===========================================
# CONVENIENCE FUNCTIONS
# ===========================================

def collect_all_price_data(years: int = 20):
    """Convenience function to collect all price data"""
    collector = PriceCollector()
    
    print(f"Collecting {years} years of price data for {len(STOCK_UNIVERSE)} stocks...")
    print("This may take 10-15 minutes...")
    
    # Collect stock data
    results = collector.collect_all_historical(years=years, save=True)
    
    # Collect indices
    print("\nCollecting index data...")
    collector.collect_indices(years=years, save=True)
    
    # Generate quality report
    print("\nGenerating quality report...")
    report = collector.generate_quality_report()
    report.to_csv(settings.data_dir / "price_quality_report.csv", index=False)
    
    print(f"\n✅ Complete! Collected data for {len(results)} stocks")
    print(f"Data saved to: {collector.data_dir}")
    
    return results


if __name__ == "__main__":
    # Run collection
    collect_all_price_data(years=20)
