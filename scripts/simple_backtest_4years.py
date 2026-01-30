"""
Simple Backtest Using Actual Labels (Not Model Predictions)
This shows the theoretical best-case performance if predictions were perfect
"""
import pandas as pd
import numpy as np

print("=" * 60)
print("Simple Backtest - 4 Year Data")
print("=" * 60)
print()

# Load data
print("Loading data...")
df = pd.read_parquet('data/intraday_features/train_features_4years.parquet')
print(f"Total rows: {len(df):,}")
print()

# Filter to 2023-2024 for backtest
df_backtest = df[df['TIME'] >= '2023-01-01'].copy()
print(f"Backtest period: {df_backtest['TIME'].min()} to {df_backtest['TIME'].max()}")
print(f"Backtest samples: {len(df_backtest):,}")
print()

# Calculate future returns
df_backtest['future_return'] = df_backtest.groupby('SYMBOL')['CLOSE_PRICE'].shift(-6) / df_backtest['CLOSE_PRICE'] - 1

# Remove NaN
df_backtest = df_backtest.dropna(subset=['future_return', 'target_direction'])

print("=" * 60)
print("Target Label Distribution")
print("=" * 60)
print()
print(df_backtest['target_direction'].value_counts(normalize=True).sort_index())
print()

# Performance by signal type
print("=" * 60)
print("Performance by Signal Type")
print("=" * 60)
print()

for direction, label in [(-1, 'SELL'), (0, 'HOLD'), (1, 'BUY')]:
    signal_df = df_backtest[df_backtest['target_direction'] == direction]
    
    if len(signal_df) > 0:
        avg_return = signal_df['future_return'].mean() * 100
        win_rate = (signal_df['future_return'] > 0).mean() * 100
        median_return = signal_df['future_return'].median() * 100
        
        print(f"{label} Signals:")
        print(f"  Count: {len(signal_df):,}")
        print(f"  Avg Return: {avg_return:.4f}%")
        print(f"  Median Return: {median_return:.4f}%")
        print(f"  Win Rate: {win_rate:.2f}%")
        print()

# Theoretical strategy
print("=" * 60)
print("Theoretical Strategy Performance")
print("=" * 60)
print()

# Strategy: Follow BUY signals only
buy_df = df_backtest[df_backtest['target_direction'] == 1]
if len(buy_df) > 0:
    total_return = (1 + buy_df['future_return']).prod() - 1
    avg_return = buy_df['future_return'].mean() * 100
    sharpe = buy_df['future_return'].mean() / buy_df['future_return'].std() * np.sqrt(252 * 75) if buy_df['future_return'].std() > 0 else 0
    
    print("Following BUY Signals:")
    print(f"  Trades: {len(buy_df):,}")
    print(f"  Total Return: {total_return * 100:.2f}%")
    print(f"  Avg Return/Trade: {avg_return:.4f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print()

# Long-short strategy
print("Long-Short Strategy:")
df_backtest['strategy_return'] = df_backtest['future_return'] * df_backtest['target_direction']
total_return = (1 + df_backtest['strategy_return']).prod() - 1
avg_return = df_backtest['strategy_return'].mean() * 100
sharpe = df_backtest['strategy_return'].mean() / df_backtest['strategy_return'].std() * np.sqrt(252 * 75) if df_backtest['strategy_return'].std() > 0 else 0

print(f"  Trades: {len(df_backtest):,}")
print(f"  Total Return: {total_return * 100:.2f}%")
print(f"  Avg Return/Trade: {avg_return:.4f}%")
print(f"  Sharpe Ratio: {sharpe:.2f}")
print()

print("=" * 60)
print("Analysis Complete!")
print("=" * 60)
print()
print("NOTE: This uses actual target labels, not model predictions.")
print("These are theoretical best-case results assuming perfect prediction.")
print("Actual model performance will be lower.")
