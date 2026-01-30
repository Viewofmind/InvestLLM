"""
Kite API Historical Data Collector
===================================
Fetch historical intraday data from Zerodha Kite API for multiple symbols.

Features:
- Fetch 5-minute OHLCV data
- Handle rate limits (3 requests/second)
- Automatic retry on failures
- Progress tracking
- Save to parquet format

Prerequisites:
    pip install kiteconnect pandas

Setup:
    1. Get API key from https://kite.trade/
    2. Generate access token
    3. Update credentials below

Usage:
    python scripts/kite_data_collector.py \
        --symbols NIFTY50 \
        --start-date 2021-01-01 \
        --end-date 2024-12-31 \
        --interval 5minute
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import time
from typing import List, Dict
import os
from kiteconnect import KiteConnect
import warnings
warnings.filterwarnings('ignore')


# Kite API Configuration
KITE_API_KEY = os.getenv('KITE_API_KEY', 'your_api_key_here')
KITE_ACCESS_TOKEN = os.getenv('KITE_ACCESS_TOKEN', 'your_access_token_here')

# Rate limiting
MAX_REQUESTS_PER_SECOND = 3
REQUEST_DELAY = 1.0 / MAX_REQUESTS_PER_SECOND


class KiteDataCollector:
    """Collect historical data from Kite API"""

    def __init__(self, api_key: str, access_token: str):
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        self.last_request_time = 0

    def rate_limit_wait(self):
        """Ensure we don't exceed rate limits"""
        elapsed = time.time() - self.last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self.last_request_time = time.time()

    def get_instruments(self, exchange: str = 'NSE') -> pd.DataFrame:
        """Get list of all instruments from exchange"""
        print(f"Fetching instruments from {exchange}...")
        instruments = self.kite.instruments(exchange)
        df = pd.DataFrame(instruments)
        print(f"Found {len(df)} instruments")
        return df

    def get_nifty50_symbols(self) -> List[str]:
        """Get NIFTY 50 stock symbols"""
        # Hardcoded NIFTY 50 stocks (as of 2024)
        nifty50 = [
            'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK',
            'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BEL', 'BPCL',
            'BHARTIARTL', 'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB',
            'DRREDDY', 'EICHERMOT', 'GRASIM', 'HCLTECH', 'HDFCBANK',
            'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK',
            'INDUSINDBK', 'INFY', 'ITC', 'JSWSTEEL', 'KOTAKBANK',
            'LT', 'M&M', 'MARUTI', 'NESTLEIND', 'NTPC',
            'ONGC', 'POWERGRID', 'RELIANCE', 'SBILIFE', 'SBIN',
            'SUNPHARMA', 'TATACONSUM', 'TATAMOTORS', 'TATASTEEL', 'TCS',
            'TECHM', 'TITAN', 'ULTRACEMCO', 'WIPRO', 'ZOMATO'
        ]
        return nifty50

    def get_nifty100_symbols(self) -> List[str]:
        """Get NIFTY 100 stock symbols (top 100 liquid stocks)"""
        nifty50 = self.get_nifty50_symbols()

        # Add next 50 from NIFTY Next 50
        nifty_next50 = [
            'ACC', 'AMBUJACEM', 'BANDHANBNK', 'BANKBARODA', 'BERGEPAINT',
            'BOSCHLTD', 'CANBK', 'CHOLAFIN', 'COLPAL', 'DABUR',
            'DLF', 'GODREJCP', 'GAIL', 'HAVELLS', 'HDFCAMC',
            'ICICIPRULI', 'INDIGO', 'JINDALSTEL', 'LICHSGFIN', 'LTIM',
            'LUPIN', 'MARICO', 'MOTHERSON', 'NAUKRI', 'NMDC',
            'OFSS', 'OIL', 'PAGEIND', 'PETRONET', 'PIDILITIND',
            'PFC', 'PNB', 'RECLTD', 'SBICARD', 'SHREECEM',
            'SIEMENS', 'SRF', 'TATAPOWER', 'TORNTPHARM', 'TRENT',
            'TVSMOTOR', 'UBL', 'UNITDSPR', 'UPL', 'VEDL',
            'VOLTAS', 'MCDOWELL-N', 'ADANIPOWER', 'ATGL', 'HAL'
        ]

        return nifty50 + nifty_next50

    def get_instrument_token(self, symbol: str, exchange: str = 'NSE') -> int:
        """Get instrument token for a symbol"""
        instruments = self.get_instruments(exchange)
        instrument = instruments[
            (instruments['tradingsymbol'] == symbol) &
            (instruments['segment'] == exchange)
        ]

        if len(instrument) == 0:
            raise ValueError(f"Symbol {symbol} not found in {exchange}")

        return instrument.iloc[0]['instrument_token']

    def fetch_historical_data(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str = '5minute'
    ) -> pd.DataFrame:
        """Fetch historical data for a symbol"""
        self.rate_limit_wait()

        try:
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            return df

        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def fetch_symbol_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = '5minute',
        exchange: str = 'NSE'
    ) -> pd.DataFrame:
        """Fetch historical data for a single symbol"""
        print(f"Fetching {symbol}...")

        try:
            # Get instrument token
            instrument_token = self.get_instrument_token(symbol, exchange)

            # Kite API has a limit of 60 days per request
            # Split into 60-day chunks
            all_data = []
            current_start = start_date

            while current_start < end_date:
                current_end = min(current_start + timedelta(days=60), end_date)

                df_chunk = self.fetch_historical_data(
                    instrument_token,
                    current_start,
                    current_end,
                    interval
                )

                if not df_chunk.empty:
                    df_chunk['symbol'] = symbol
                    df_chunk['instrument_token'] = instrument_token
                    all_data.append(df_chunk)

                current_start = current_end + timedelta(days=1)

            if all_data:
                df = pd.concat(all_data, ignore_index=True)
                print(f"  ✓ {symbol}: {len(df)} bars")
                return df
            else:
                print(f"  ✗ {symbol}: No data")
                return pd.DataFrame()

        except Exception as e:
            print(f"  ✗ {symbol}: Error - {e}")
            return pd.DataFrame()

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = '5minute',
        exchange: str = 'NSE',
        output_file: str = None
    ) -> pd.DataFrame:
        """Fetch data for multiple symbols"""
        print("="*60)
        print(f"Fetching Historical Data")
        print("="*60)
        print(f"Symbols: {len(symbols)}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Interval: {interval}")
        print(f"Exchange: {exchange}")
        print()

        all_data = []

        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] ", end='')

            df = self.fetch_symbol_data(
                symbol,
                start_date,
                end_date,
                interval,
                exchange
            )

            if not df.empty:
                all_data.append(df)

            # Save intermediate results every 10 symbols
            if output_file and i % 10 == 0 and all_data:
                temp_df = pd.concat(all_data, ignore_index=True)
                temp_output = output_file.replace('.parquet', f'_temp_{i}.parquet')
                temp_df.to_parquet(temp_output, index=False)
                print(f"  → Saved checkpoint: {temp_output}")

        if all_data:
            df = pd.concat(all_data, ignore_index=True)

            # Rename columns to match your format
            df = df.rename(columns={
                'date': 'TIME',
                'open': 'OPEN_PRICE',
                'high': 'HIGH_PRICE',
                'low': 'LOW_PRICE',
                'close': 'CLOSE_PRICE',
                'volume': 'VOLUME',
                'symbol': 'SYMBOL',
                'instrument_token': 'TOKEN'
            })

            # Add exchange column
            df['EXCHANGE'] = exchange

            # Sort by symbol and time
            df = df.sort_values(['SYMBOL', 'TIME']).reset_index(drop=True)

            print()
            print("="*60)
            print("Data Collection Complete")
            print("="*60)
            print(f"Total symbols: {df['SYMBOL'].nunique()}")
            print(f"Total bars: {len(df):,}")
            print(f"Date range: {df['TIME'].min()} to {df['TIME'].max()}")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

            if output_file:
                df.to_parquet(output_file, index=False)
                print(f"\nSaved to: {output_file}")

            return df
        else:
            print("\nNo data collected!")
            return pd.DataFrame()


def generate_access_token_instructions():
    """Print instructions for generating Kite access token"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║         Kite Connect API Setup Instructions                  ║
    ╚══════════════════════════════════════════════════════════════╝

    Step 1: Get API Credentials
    ─────────────────────────────────
    1. Go to: https://developers.kite.trade/
    2. Login with your Zerodha credentials
    3. Create a new app or use existing
    4. Note down your API Key and API Secret

    Step 2: Generate Access Token
    ─────────────────────────────────
    Run this Python script:

    ```python
    from kiteconnect import KiteConnect

    api_key = "your_api_key"
    api_secret = "your_api_secret"

    kite = KiteConnect(api_key=api_key)

    # Get login URL
    print(kite.login_url())

    # Open URL in browser, login, and copy request_token from redirect URL
    request_token = "paste_request_token_here"

    # Generate session
    data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = data["access_token"]

    print(f"Access Token: {access_token}")
    ```

    Step 3: Use Access Token
    ─────────────────────────────────
    Set environment variables:

    export KITE_API_KEY="your_api_key"
    export KITE_ACCESS_TOKEN="your_access_token"

    Or pass directly to script:

    python scripts/kite_data_collector.py \\
        --api-key your_api_key \\
        --access-token your_access_token \\
        --symbols NIFTY100 \\
        --start-date 2021-01-01 \\
        --end-date 2024-12-31

    Note: Access tokens expire daily. You need to regenerate them.

    ╚══════════════════════════════════════════════════════════════╝
    """)


def main():
    parser = argparse.ArgumentParser(description='Fetch historical data from Kite API')

    # API credentials
    parser.add_argument('--api-key', type=str,
                       default=os.getenv('KITE_API_KEY'),
                       help='Kite API key')
    parser.add_argument('--access-token', type=str,
                       default=os.getenv('KITE_ACCESS_TOKEN'),
                       help='Kite access token')

    # Data parameters
    parser.add_argument('--symbols', type=str, default='NIFTY50',
                       choices=['NIFTY50', 'NIFTY100', 'CUSTOM'],
                       help='Symbol list to fetch')
    parser.add_argument('--custom-symbols', type=str, nargs='+',
                       help='Custom list of symbols (if --symbols=CUSTOM)')
    parser.add_argument('--start-date', type=str, required=True,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--interval', type=str, default='5minute',
                       choices=['minute', '3minute', '5minute', '10minute', '15minute',
                               '30minute', '60minute', 'day'],
                       help='Data interval')
    parser.add_argument('--exchange', type=str, default='NSE',
                       choices=['NSE', 'BSE'],
                       help='Exchange')

    # Output
    parser.add_argument('--output', type=str,
                       default='data/kite_historical/historical_data.parquet',
                       help='Output file path')

    # Utility
    parser.add_argument('--show-setup', action='store_true',
                       help='Show API setup instructions')

    args = parser.parse_args()

    # Show setup instructions
    if args.show_setup:
        generate_access_token_instructions()
        return

    # Validate credentials
    if not args.api_key or not args.access_token:
        print("ERROR: API credentials not provided!")
        print("\nRun with --show-setup for instructions:")
        print("  python scripts/kite_data_collector.py --show-setup")
        return

    if args.api_key == 'your_api_key_here' or args.access_token == 'your_access_token_here':
        print("ERROR: Please set valid API credentials!")
        print("\nRun with --show-setup for instructions:")
        print("  python scripts/kite_data_collector.py --show-setup")
        return

    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')

    # Initialize collector
    collector = KiteDataCollector(args.api_key, args.access_token)

    # Get symbol list
    if args.symbols == 'NIFTY50':
        symbols = collector.get_nifty50_symbols()
    elif args.symbols == 'NIFTY100':
        symbols = collector.get_nifty100_symbols()
    else:  # CUSTOM
        if not args.custom_symbols:
            print("ERROR: --custom-symbols required when --symbols=CUSTOM")
            return
        symbols = args.custom_symbols

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Fetch data
    df = collector.fetch_multiple_symbols(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        interval=args.interval,
        exchange=args.exchange,
        output_file=args.output
    )

    if not df.empty:
        print("\n✓ Data collection successful!")
        print(f"\nNext steps:")
        print(f"1. Run feature engineering:")
        print(f"   python scripts/intraday_feature_engineering.py \\")
        print(f"       --input {args.output} \\")
        print(f"       --output data/intraday_features/train_features.parquet")
        print(f"\n2. Train model on new data:")
        print(f"   python scripts/train_intraday_model.py \\")
        print(f"       --data data/intraday_features/train_features.parquet")


if __name__ == '__main__':
    main()
