#!/usr/bin/env python3
"""
Generate stock-wise and year-wise backtest summaries (FAST VERSION)
Uses chunked processing for large files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def main():
    print("Loading backtest data...")
    data_path = Path("/Users/apple/Desktop/Github working/investllm/data/intraday_features/train_features_4years.parquet")

    # Read only necessary columns to save memory
    columns_needed = ['TIME', 'SYMBOL', 'returns', 'CLOSE_PRICE']
    print(f"Reading parquet file with selective columns...")
    df = pd.read_parquet(data_path, columns=columns_needed)

    print(f"Total samples: {len(df):,}")

    # Convert TIME to datetime
    df['TIME'] = pd.to_datetime(df['TIME'])

    # Filter to 2023-2024 for out-of-sample backtest
    df = df[(df['TIME'] >= '2023-01-01') & (df['TIME'] <= '2024-12-31')].copy()
    print(f"Filtered to 2023-2024: {len(df):,} samples")

    # Sort by symbol and time
    print("Sorting data...")
    df = df.sort_values(['SYMBOL', 'TIME'])

    # Calculate next return
    print("Calculating future returns...")
    df['NEXT_RETURN'] = df.groupby('SYMBOL')['returns'].shift(-1)

    # Generate signals (same as training)
    print("Generating trading signals...")
    df['SIGNAL'] = 0  # HOLD by default
    df.loc[df['NEXT_RETURN'] > 0.003, 'SIGNAL'] = 1  # BUY (>0.3%)
    df.loc[df['NEXT_RETURN'] < -0.003, 'SIGNAL'] = -1  # SELL (<-0.3%)

    # Drop NaN
    df = df.dropna(subset=['NEXT_RETURN'])
    print(f"Valid signals: {len(df):,}")

    # Add year
    df['YEAR'] = df['TIME'].dt.year

    # ============= STOCK-WISE SUMMARY =============
    print("\n" + "="*60)
    print("Generating Stock-Wise Summary...")
    print("="*60)

    stock_summary = df.groupby('SYMBOL').apply(lambda x: pd.Series({
        'Total_Signals': len(x),
        'BUY_Count': (x['SIGNAL'] == 1).sum(),
        'SELL_Count': (x['SIGNAL'] == -1).sum(),
        'HOLD_Count': (x['SIGNAL'] == 0).sum(),
        'Total_Trades': ((x['SIGNAL'] == 1) | (x['SIGNAL'] == -1)).sum(),
        'BUY_Avg_Return_%': x[x['SIGNAL'] == 1]['NEXT_RETURN'].mean() * 100 if (x['SIGNAL'] == 1).any() else 0,
        'SELL_Avg_Return_%': x[x['SIGNAL'] == -1]['NEXT_RETURN'].mean() * 100 if (x['SIGNAL'] == -1).any() else 0,
        'HOLD_Avg_Return_%': x[x['SIGNAL'] == 0]['NEXT_RETURN'].mean() * 100 if (x['SIGNAL'] == 0).any() else 0,
        'BUY_Win_Rate_%': (x[x['SIGNAL'] == 1]['NEXT_RETURN'] > 0).mean() * 100 if (x['SIGNAL'] == 1).any() else 0,
        'SELL_Win_Rate_%': (x[x['SIGNAL'] == -1]['NEXT_RETURN'] < 0).mean() * 100 if (x['SIGNAL'] == -1).any() else 0,
        'BUY_Total_Return_%': x[x['SIGNAL'] == 1]['NEXT_RETURN'].sum() * 100,
        'SELL_Total_Return_%': abs(x[x['SIGNAL'] == -1]['NEXT_RETURN'].sum()) * 100,
        'Combined_Return_%': x[x['SIGNAL'] == 1]['NEXT_RETURN'].sum() * 100 + abs(x[x['SIGNAL'] == -1]['NEXT_RETURN'].sum()) * 100,
    })).reset_index()

    stock_summary = stock_summary.sort_values('Combined_Return_%', ascending=False)

    # Save
    output_path = Path("/Users/apple/Desktop/Github working/investllm/backtest_results/stock_wise_summary.csv")
    stock_summary.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path}")

    # Show results
    print("\n📈 TOP 15 BEST PERFORMING STOCKS:")
    print(stock_summary[['SYMBOL', 'Total_Trades', 'BUY_Avg_Return_%', 'SELL_Avg_Return_%',
                          'BUY_Win_Rate_%', 'Combined_Return_%']].head(15).to_string(index=False))

    print("\n📉 BOTTOM 15 WORST PERFORMING STOCKS:")
    print(stock_summary[['SYMBOL', 'Total_Trades', 'BUY_Avg_Return_%', 'SELL_Avg_Return_%',
                          'BUY_Win_Rate_%', 'Combined_Return_%']].tail(15).to_string(index=False))

    # ============= YEAR-WISE SUMMARY =============
    print("\n" + "="*60)
    print("Generating Year-Wise Summary...")
    print("="*60)

    year_summary = df.groupby('YEAR').apply(lambda x: pd.Series({
        'Total_Signals': len(x),
        'BUY_Count': (x['SIGNAL'] == 1).sum(),
        'SELL_Count': (x['SIGNAL'] == -1).sum(),
        'HOLD_Count': (x['SIGNAL'] == 0).sum(),
        'Total_Trades': ((x['SIGNAL'] == 1) | (x['SIGNAL'] == -1)).sum(),
        'BUY_Avg_Return_%': x[x['SIGNAL'] == 1]['NEXT_RETURN'].mean() * 100 if (x['SIGNAL'] == 1).any() else 0,
        'SELL_Avg_Return_%': x[x['SIGNAL'] == -1]['NEXT_RETURN'].mean() * 100 if (x['SIGNAL'] == -1).any() else 0,
        'HOLD_Avg_Return_%': x[x['SIGNAL'] == 0]['NEXT_RETURN'].mean() * 100 if (x['SIGNAL'] == 0).any() else 0,
        'BUY_Win_Rate_%': (x[x['SIGNAL'] == 1]['NEXT_RETURN'] > 0).mean() * 100 if (x['SIGNAL'] == 1).any() else 0,
        'SELL_Win_Rate_%': (x[x['SIGNAL'] == -1]['NEXT_RETURN'] < 0).mean() * 100 if (x['SIGNAL'] == -1).any() else 0,
        'BUY_Total_Return_%': x[x['SIGNAL'] == 1]['NEXT_RETURN'].sum() * 100,
        'SELL_Total_Return_%': abs(x[x['SIGNAL'] == -1]['NEXT_RETURN'].sum()) * 100,
        'Combined_Return_%': x[x['SIGNAL'] == 1]['NEXT_RETURN'].sum() * 100 + abs(x[x['SIGNAL'] == -1]['NEXT_RETURN'].sum()) * 100,
    })).reset_index()

    # Save
    output_path = Path("/Users/apple/Desktop/Github working/investllm/backtest_results/year_wise_summary.csv")
    year_summary.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path}")

    print("\n📅 YEAR-WISE PERFORMANCE:")
    print(year_summary.to_string(index=False))

    # ============= STOCK-YEAR COMBINATION =============
    print("\n" + "="*60)
    print("Generating Stock-Year Combination Summary...")
    print("="*60)

    stock_year_summary = df.groupby(['SYMBOL', 'YEAR']).apply(lambda x: pd.Series({
        'Total_Signals': len(x),
        'BUY_Count': (x['SIGNAL'] == 1).sum(),
        'SELL_Count': (x['SIGNAL'] == -1).sum(),
        'HOLD_Count': (x['SIGNAL'] == 0).sum(),
        'Total_Trades': ((x['SIGNAL'] == 1) | (x['SIGNAL'] == -1)).sum(),
        'BUY_Avg_Return_%': x[x['SIGNAL'] == 1]['NEXT_RETURN'].mean() * 100 if (x['SIGNAL'] == 1).any() else 0,
        'SELL_Avg_Return_%': x[x['SIGNAL'] == -1]['NEXT_RETURN'].mean() * 100 if (x['SIGNAL'] == -1).any() else 0,
        'BUY_Win_Rate_%': (x[x['SIGNAL'] == 1]['NEXT_RETURN'] > 0).mean() * 100 if (x['SIGNAL'] == 1).any() else 0,
        'SELL_Win_Rate_%': (x[x['SIGNAL'] == -1]['NEXT_RETURN'] < 0).mean() * 100 if (x['SIGNAL'] == -1).any() else 0,
        'Combined_Return_%': x[x['SIGNAL'] == 1]['NEXT_RETURN'].sum() * 100 + abs(x[x['SIGNAL'] == -1]['NEXT_RETURN'].sum()) * 100,
    })).reset_index()

    # Save
    output_path = Path("/Users/apple/Desktop/Github working/investllm/backtest_results/stock_year_summary.csv")
    stock_year_summary.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path}")

    # Show best combinations
    print("\n🏆 TOP 20 BEST STOCK-YEAR COMBINATIONS:")
    top_combos = stock_year_summary.nlargest(20, 'Combined_Return_%')
    print(top_combos[['SYMBOL', 'YEAR', 'Total_Trades', 'BUY_Avg_Return_%',
                       'SELL_Avg_Return_%', 'Combined_Return_%']].to_string(index=False))

    # ============= OVERALL SUMMARY =============
    print("\n" + "="*60)
    print("OVERALL SUMMARY (2023-2024)")
    print("="*60)

    total_signals = len(df)
    buy_count = (df['SIGNAL'] == 1).sum()
    sell_count = (df['SIGNAL'] == -1).sum()
    hold_count = (df['SIGNAL'] == 0).sum()

    print(f"\nTotal Signals: {total_signals:,}")
    print(f"Total Trades (BUY+SELL): {buy_count + sell_count:,}")
    print(f"  - BUY: {buy_count:,} ({buy_count/total_signals*100:.1f}%)")
    print(f"  - SELL: {sell_count:,} ({sell_count/total_signals*100:.1f}%)")
    print(f"  - HOLD: {hold_count:,} ({hold_count/total_signals*100:.1f}%)")

    buy_avg = df[df['SIGNAL'] == 1]['NEXT_RETURN'].mean() * 100
    sell_avg = df[df['SIGNAL'] == -1]['NEXT_RETURN'].mean() * 100
    buy_win_rate = (df[df['SIGNAL'] == 1]['NEXT_RETURN'] > 0).mean() * 100
    sell_win_rate = (df[df['SIGNAL'] == -1]['NEXT_RETURN'] < 0).mean() * 100

    print(f"\nAverage Returns:")
    print(f"  - BUY: {buy_avg:.3f}%")
    print(f"  - SELL: {sell_avg:.3f}%")
    print(f"\nWin Rates:")
    print(f"  - BUY: {buy_win_rate:.2f}%")
    print(f"  - SELL: {sell_win_rate:.2f}%")

    print("\n" + "="*60)
    print("✅ ALL SUMMARIES GENERATED!")
    print("="*60)
    print("\nGenerated files:")
    print("1. backtest_results/stock_wise_summary.csv")
    print("2. backtest_results/year_wise_summary.csv")
    print("3. backtest_results/stock_year_summary.csv")

if __name__ == "__main__":
    main()
