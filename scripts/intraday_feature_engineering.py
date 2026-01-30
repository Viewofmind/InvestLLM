"""
InvestLLM Intraday Feature Engineering
======================================
Generate ML features for intraday trading model.

Features:
- Technical indicators (RSI, MACD, Bollinger, ATR, etc.)
- Volume features (OBV, VWAP, volume profile)
- Time-based features (session, hour, day)
- Price action features (patterns, support/resistance)
- Rolling statistics (returns, volatility)

Usage:
    python scripts/intraday_feature_engineering.py \
        --input data/intraday_processed/train_5min.parquet \
        --output data/intraday_features/train_features.parquet
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class IntradayFeatureEngineer:
    """Generate features for intraday trading ML model"""

    def __init__(self, lookback_periods: List[int] = None):
        self.lookback_periods = lookback_periods or [5, 10, 20, 40, 60]  # 5-min bars
        self.feature_names = []

    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all features for the dataset"""
        print("Generating features...")

        # Sort data
        df = df.sort_values(['SYMBOL', 'TIME']).reset_index(drop=True)

        # Generate features per symbol
        result_dfs = []
        symbols = df['SYMBOL'].unique()

        for i, symbol in enumerate(symbols):
            if (i + 1) % 20 == 0:
                print(f"  Processing {i+1}/{len(symbols)} symbols...")

            symbol_df = df[df['SYMBOL'] == symbol].copy()
            symbol_df = self._generate_symbol_features(symbol_df)
            result_dfs.append(symbol_df)

        result = pd.concat(result_dfs, ignore_index=True)
        print(f"\nGenerated {len(self.feature_names)} features")

        return result

    def _generate_symbol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features for a single symbol"""
        # Basic price features
        df = self._add_price_features(df)

        # Technical indicators
        df = self._add_technical_indicators(df)

        # Volume features
        df = self._add_volume_features(df)

        # Time features
        df = self._add_time_features(df)

        # Rolling statistics
        df = self._add_rolling_stats(df)

        # Lag features
        df = self._add_lag_features(df)

        # Target variable
        df = self._add_target(df)

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic price-derived features"""
        # Returns
        df['returns'] = df['CLOSE_PRICE'].pct_change()
        df['log_returns'] = np.log(df['CLOSE_PRICE'] / df['CLOSE_PRICE'].shift(1))

        # Price ranges
        df['high_low_pct'] = (df['HIGH_PRICE'] - df['LOW_PRICE']) / df['CLOSE_PRICE']
        df['close_open_pct'] = (df['CLOSE_PRICE'] - df['OPEN_PRICE']) / df['OPEN_PRICE']
        df['high_close_pct'] = (df['HIGH_PRICE'] - df['CLOSE_PRICE']) / df['CLOSE_PRICE']
        df['low_close_pct'] = (df['CLOSE_PRICE'] - df['LOW_PRICE']) / df['CLOSE_PRICE']

        # Price position within range
        df['price_position'] = (df['CLOSE_PRICE'] - df['LOW_PRICE']) / (df['HIGH_PRICE'] - df['LOW_PRICE'] + 1e-10)

        # Gap features
        df['gap_pct'] = (df['OPEN_PRICE'] - df['CLOSE_PRICE'].shift(1)) / df['CLOSE_PRICE'].shift(1)

        self.feature_names.extend([
            'returns', 'log_returns', 'high_low_pct', 'close_open_pct',
            'high_close_pct', 'low_close_pct', 'price_position', 'gap_pct'
        ])

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Technical analysis indicators"""
        close = df['CLOSE_PRICE']
        high = df['HIGH_PRICE']
        low = df['LOW_PRICE']
        volume = df['VOLUME']

        # RSI (multiple periods)
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = self._calculate_rsi(close, period)
            self.feature_names.append(f'rsi_{period}')

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        self.feature_names.extend(['macd', 'macd_signal', 'macd_hist'])

        # Bollinger Bands
        for period in [10, 20]:
            sma = close.rolling(period).mean()
            std = close.rolling(period).std()
            df[f'bb_upper_{period}'] = sma + 2 * std
            df[f'bb_lower_{period}'] = sma - 2 * std
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
            df[f'bb_position_{period}'] = (close - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10)
            self.feature_names.extend([f'bb_width_{period}', f'bb_position_{period}'])

        # ATR (Average True Range)
        for period in [7, 14]:
            tr = pd.concat([
                high - low,
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            ], axis=1).max(axis=1)
            df[f'atr_{period}'] = tr.rolling(period).mean()
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / close
            self.feature_names.extend([f'atr_{period}', f'atr_pct_{period}'])

        # Stochastic Oscillator
        for period in [7, 14]:
            lowest_low = low.rolling(period).min()
            highest_high = high.rolling(period).max()
            df[f'stoch_k_{period}'] = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
            df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(3).mean()
            self.feature_names.extend([f'stoch_k_{period}', f'stoch_d_{period}'])

        # CCI (Commodity Channel Index)
        for period in [14, 20]:
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(period).mean()
            mad = typical_price.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
            df[f'cci_{period}'] = (typical_price - sma_tp) / (0.015 * mad + 1e-10)
            self.feature_names.append(f'cci_{period}')

        # ADX (Average Directional Index)
        df['adx_14'] = self._calculate_adx(high, low, close, 14)
        self.feature_names.append('adx_14')

        # Moving Averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = close.rolling(period).mean()
            df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
            df[f'price_vs_sma_{period}'] = (close - df[f'sma_{period}']) / df[f'sma_{period}']
            df[f'price_vs_ema_{period}'] = (close - df[f'ema_{period}']) / df[f'ema_{period}']
            self.feature_names.extend([f'price_vs_sma_{period}', f'price_vs_ema_{period}'])

        # VWAP
        df['vwap'] = (df['CLOSE_PRICE'] * df['VOLUME']).cumsum() / df['VOLUME'].cumsum()
        df['price_vs_vwap'] = (close - df['vwap']) / df['vwap']
        self.feature_names.append('price_vs_vwap')

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate ADX"""
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()

        return adx

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based features"""
        volume = df['VOLUME']
        close = df['CLOSE_PRICE']

        # Volume moving averages
        for period in [5, 10, 20]:
            df[f'volume_sma_{period}'] = volume.rolling(period).mean()
            df[f'volume_ratio_{period}'] = volume / (df[f'volume_sma_{period}'] + 1)
            self.feature_names.extend([f'volume_ratio_{period}'])

        # Volume change
        df['volume_change'] = volume.pct_change()
        df['volume_change_5'] = volume.pct_change(5)
        self.feature_names.extend(['volume_change', 'volume_change_5'])

        # OBV (On-Balance Volume)
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        df['obv'] = obv
        df['obv_change'] = obv.pct_change(5)
        self.feature_names.append('obv_change')

        # Volume-price trend
        df['volume_price_trend'] = ((close.diff() / close.shift(1)) * volume).rolling(10).sum()
        self.feature_names.append('volume_price_trend')

        # Money Flow Index
        typical_price = (df['HIGH_PRICE'] + df['LOW_PRICE'] + close) / 3
        money_flow = typical_price * volume
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        df['mfi_14'] = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-10)))
        self.feature_names.append('mfi_14')

        # Accumulation/Distribution
        clv = ((close - df['LOW_PRICE']) - (df['HIGH_PRICE'] - close)) / (df['HIGH_PRICE'] - df['LOW_PRICE'] + 1e-10)
        df['ad_line'] = (clv * volume).cumsum()
        df['ad_change'] = df['ad_line'].pct_change(5)
        self.feature_names.append('ad_change')

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based features"""
        df['hour'] = df['TIME'].dt.hour
        df['minute'] = df['TIME'].dt.minute
        df['day_of_week'] = df['TIME'].dt.dayofweek

        # Session indicators
        df['is_opening'] = ((df['hour'] == 9) & (df['minute'] < 30)).astype(int)
        df['is_first_hour'] = (df['hour'] == 9).astype(int)
        df['is_last_hour'] = (df['hour'] >= 14).astype(int)
        df['is_closing'] = ((df['hour'] == 15) & (df['minute'] >= 15)).astype(int)
        df['is_lunch'] = ((df['hour'] >= 12) & (df['hour'] < 13)).astype(int)

        # Time encoding (cyclical)
        minutes_since_open = (df['hour'] - 9) * 60 + df['minute'] - 15
        total_minutes = 375  # Trading day length
        df['time_sin'] = np.sin(2 * np.pi * minutes_since_open / total_minutes)
        df['time_cos'] = np.cos(2 * np.pi * minutes_since_open / total_minutes)

        # Day of week encoding
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)

        self.feature_names.extend([
            'is_opening', 'is_first_hour', 'is_last_hour', 'is_closing', 'is_lunch',
            'time_sin', 'time_cos', 'dow_sin', 'dow_cos'
        ])

        return df

    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling window statistics"""
        close = df['CLOSE_PRICE']
        returns = df['returns']

        for window in self.lookback_periods:
            # Returns statistics
            df[f'returns_mean_{window}'] = returns.rolling(window).mean()
            df[f'returns_std_{window}'] = returns.rolling(window).std()
            df[f'returns_skew_{window}'] = returns.rolling(window).skew()
            df[f'returns_kurt_{window}'] = returns.rolling(window).kurt()

            # Price statistics
            df[f'price_min_{window}'] = close.rolling(window).min()
            df[f'price_max_{window}'] = close.rolling(window).max()
            df[f'price_range_{window}'] = (df[f'price_max_{window}'] - df[f'price_min_{window}']) / close

            # Position in range
            df[f'price_position_{window}'] = (close - df[f'price_min_{window}']) / (
                df[f'price_max_{window}'] - df[f'price_min_{window}'] + 1e-10
            )

            # Volatility
            df[f'volatility_{window}'] = returns.rolling(window).std() * np.sqrt(252 * 75)  # Annualized

            self.feature_names.extend([
                f'returns_mean_{window}', f'returns_std_{window}',
                f'price_range_{window}', f'price_position_{window}', f'volatility_{window}'
            ])

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lagged features for sequence modeling"""
        for lag in [1, 2, 3, 5, 10]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_change_lag_{lag}'] = df['volume_change'].shift(lag)
            self.feature_names.extend([f'returns_lag_{lag}', f'volume_change_lag_{lag}'])

        return df

    def _add_target(self, df: pd.DataFrame, horizon: int = 6) -> pd.DataFrame:
        """
        Add target variable for prediction

        Target: Direction of price movement in next `horizon` bars (30 min for 5-min data)
        - 1: Price up by > 0.3%
        - 0: Price flat (-0.3% to +0.3%)
        - -1: Price down by > 0.3%
        """
        future_return = df['CLOSE_PRICE'].shift(-horizon) / df['CLOSE_PRICE'] - 1

        # Multi-class target
        threshold = 0.003  # 0.3%
        df['target_direction'] = 0
        df.loc[future_return > threshold, 'target_direction'] = 1
        df.loc[future_return < -threshold, 'target_direction'] = -1

        # Binary target (up or not)
        df['target_up'] = (future_return > threshold).astype(int)

        # Regression target
        df['target_return'] = future_return

        return df


def main():
    parser = argparse.ArgumentParser(description='Generate intraday features')
    parser.add_argument('--input', type=str,
                       default='data/intraday_processed/train_5min.parquet',
                       help='Input parquet file')
    parser.add_argument('--output', type=str,
                       default='data/intraday_features/train_features.parquet',
                       help='Output parquet file')
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} rows, {df['SYMBOL'].nunique()} symbols")

    # Generate features
    engineer = IntradayFeatureEngineer()
    df_features = engineer.generate_all_features(df)

    # Remove rows with NaN (from rolling calculations)
    initial_rows = len(df_features)
    df_features = df_features.dropna()
    print(f"\nRemoved {initial_rows - len(df_features):,} rows with NaN values")
    print(f"Final dataset: {len(df_features):,} rows")

    # Feature statistics
    feature_cols = [c for c in df_features.columns if c not in [
        'SYMBOL', 'TOKEN', 'EXCHANGE', 'MARKET CAP', 'TIME',
        'OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE', 'VOLUME',
        'target_direction', 'target_up', 'target_return', 'date', 'vwap',
        'obv', 'ad_line', 'hour', 'minute', 'day_of_week'
    ]]

    print(f"\nFeature columns: {len(feature_cols)}")

    # Target distribution
    print(f"\nTarget distribution:")
    print(df_features['target_direction'].value_counts(normalize=True))

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Save feature list
    feature_file = output_path.parent / 'feature_columns.txt'
    with open(feature_file, 'w') as f:
        f.write('\n'.join(sorted(feature_cols)))
    print(f"Feature list saved to {feature_file}")

    # Summary
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Rows: {len(df_features):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Symbols: {df_features['SYMBOL'].nunique()}")
    print(f"\nReady for model training!")


if __name__ == '__main__':
    main()
