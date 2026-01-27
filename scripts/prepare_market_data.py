#!/usr/bin/env python3
"""
Prepare Market Data for Price Prediction
========================================

This script processes historical price data and fundamental data to create 
datasets for training deep learning models (LSTM/GRU).

Features:
1.  **Technical Indicators**: Adds RSI, MACD, Bollinger Bands, ATR, ADX, SMA, EMA.
2.  **Fundamental Data**: Merges quarterly financial metrics (P/E, EPS, etc.) with daily price data (forward fill).
3.  **Target Generation**: Creates target variables (Next Day Return, Direction).
4.  **Scaling**: Normalizes features using MinMax or StandardScaler.
5.  **Sequencing**: Creates sliding window sequences for time-series models.
"""

import os
import sys
from pathlib import Path
import glob
import argparse
import logging
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from rich.console import Console
from rich.progress import Progress

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed" / "price_prediction"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


class MarketDataProcessor:
    def __init__(self, tickers: List[str] = None):
        self.tickers = tickers
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_price_data(self, ticker: str) -> pd.DataFrame:
        """Load historical price data for a ticker"""
        file_path = DATA_DIR / "raw" / "prices" / f"{ticker}.parquet"
        if not file_path.exists():
            logger.warning(f"Price data not found for {ticker}")
            return pd.DataFrame()
            
        df = pd.read_parquet(file_path)
        
        # Ensure DateTime index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df.index.name = 'Date'
        
        # Standardize column names
        col_map = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        df = df.rename(columns=col_map)
        
        # Sort by date
        df = df.sort_index()
        
        # Handle duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators"""
        if len(df) < 50:
            return df
            
        # 1. Trend Indicators
        # SMA
        df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        df['SMA_200'] = SMAIndicator(close=df['Close'], window=200).sma_indicator()
        
        # EMA
        df['EMA_12'] = EMAIndicator(close=df['Close'], window=12).ema_indicator()
        df['EMA_26'] = EMAIndicator(close=df['Close'], window=26).ema_indicator()
        
        # MACD
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()
        
        # ADX (Trend Strength)
        adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'])
        df['ADX'] = adx.adx()
        
        # 2. Momentum Indicators
        # RSI
        df['RSI'] = RSIIndicator(close=df['Close']).rsi()
        
        # 3. Volatility Indicators
        # Bollinger Bands
        bb = BollingerBands(close=df['Close'])
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['Close']
        
        # ATR
        df['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()
        
        # 4. Volume Indicators
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA_20'] + 1e-8)
        
        # 5. Returns
        df['Log_Return'] = np.log((df['Close'] / df['Close'].shift(1)) + 1e-8)
        
        return df

    def add_fundamental_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Merge fundamental data (quarterly/annual) to daily prices using forward fill.
        Note: This is a simplified version. Real implementation would need structured fundamental tables.
        """
        # Placeholder for fundamental loading logic
        # For now, we will skip this or assume we have a processed fundamentals file
        # If we had a table like: Date | PE | EPS for each quarter
        # We would merge_asof
        
        # TODO: Implement full fundamental data merging
        # Currently just returning df as is to keep prototype simple
        return df

    def create_sequences(self, data: np.ndarray, seq_length: int, forecast_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create (X, y) sequences for LSTM
        X: (samples, seq_length, features)
        y: (samples, target)
        """
        xs, ys = [], []
        
        # Target is Next Day Return (Log Return shifted by -1)
        # Or simply the Close price shifted
        
        # Let's predict Next Day Log Return for now (Regression)
        # Column index for 'Log_Return' needs to be known or we pass it separately
        
        for i in range(len(data) - seq_length - forecast_horizon):
            x = data[i : i + seq_length]
            y = data[i + seq_length + forecast_horizon - 1, -1] # Assuming target is the last column
            xs.append(x)
            ys.append(y)
            
        return np.array(xs), np.array(ys)

    def process_ticker(self, ticker: str, seq_length: int = 60) -> Optional[Dict]:
        """Process a single ticker pipeline"""
        
        # 1. Load Data
        df = self.load_price_data(ticker)
        if df.empty:
            return None
            
        # 2. Add Technicals
        df = self.add_technical_indicators(df)
        
        # 3. Add Fundamentals (TODO)
        df = self.add_fundamental_data(df, ticker)
        
        # 4. Clean NaN (dropped due to rolling windows)
        df = df.dropna()
        
        if len(df) < seq_length * 2:
            return None
            
        # 5. Define Target
        # Target: Next Day Log Return
        # Shift Log Return back by 1 to align "Target" with "Today's Features"
        df['Target'] = df['Log_Return'].shift(-1)
        df = df.dropna()
        
        # 6. Select Features
        features = [
            'Open', 'High', 'Low', 'Close', 'Volume', 
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff', 'ADX',
            'SMA_20', 'SMA_50', 'SMA_200', 
            'EMA_12', 'EMA_26',
            'BB_High', 'BB_Low', 'BB_Width', 'ATR',
            'Volume_Ratio', 'Log_Return'
        ]
        
        # Available features
        available_features = [f for f in features if f in df.columns]
        self.feature_columns = available_features
        
        data = df[available_features + ['Target']].values
        
        return {
            'ticker': ticker,
            'data': data,
            'index': df.index,
            'feature_names': available_features
        }

    def process_all(self, seq_length: int = 60):
        """Process all tickers and save"""
        all_data = []
        
        files = list(DATA_DIR.glob("raw/prices/*.parquet"))
        if self.tickers:
            # Filter valid files
            files = [f for f in files if f.stem in self.tickers]
            
        console.print(f"Processing {len(files)} tickers...")
        
        for file in files:
            ticker = file.stem
            try:
                result = self.process_ticker(ticker, seq_length=seq_length)
                if result:
                    all_data.append(result)
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                
        if not all_data:
            console.print("[red]No data processed successfully[/red]")
            return
            
        # Combine all data to fit scaler? 
        # CAUTION: Fitting scaler on ALL data leaks future info if not careful.
        # Ideally: Train/Test split by TIME (e.g. 2000-2018 Train, 2019-2020 Test)
        # Then fit scaler on Train.
        
        # For simplicity in this script, we will just save the processed DataFrames
        # The training script will handle scaling and splitting correctly to prevent leakage.
        
        console.print(f"Saving processed data for {len(all_data)} tickers...")
        
        for item in all_data:
            ticker = item['ticker']
            data = item['data']
            cols = item['feature_names'] + ['Target']
            idx = item['index']
            
            df_processed = pd.DataFrame(data, index=idx, columns=cols)
            df_processed.to_parquet(PROCESSED_DIR / f"{ticker}_processed.parquet")
            
        console.print(f"[green]Done. Saved to {PROCESSED_DIR}[/green]")
        
        # Save feature list
        pd.Series(all_data[0]['feature_names']).to_csv(PROCESSED_DIR / "feature_names.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to process")
    parser.add_argument("--seq_length", type=int, default=60, help="Sequence length for LSTM")
    args = parser.parse_args()
    
    processor = MarketDataProcessor(tickers=args.tickers)
    processor.process_all(seq_length=args.seq_length)
