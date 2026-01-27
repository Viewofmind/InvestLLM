#!/usr/bin/env python3
"""
InvestLLM - Data Exploration (EDA)
Run this script to explore the collected data.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Settings
DATA_DIR = Path("data")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

print("=" * 60)
print("InvestLLM - Data Exploration Report")
print("=" * 60)

# 1. Price Data
print("\n📈 PRICE DATA ANALYSIS")
print("-" * 40)

price_dir = DATA_DIR / "raw/prices"
if price_dir.exists():
    price_files = list(price_dir.glob("*.parquet"))
    print(f"Total stocks with price data: {len(price_files)}")
    
    # Load sample stock (RELIANCE)
    sample_file = price_dir / "RELIANCE.parquet"
    if sample_file.exists():
        df_rel = pd.read_parquet(sample_file)
        print(f"\nSample: RELIANCE.NS")
        print(f"  Date Range: {df_rel['timestamp'].min()} to {df_rel['timestamp'].max()}")
        print(f"  Total Rows: {len(df_rel):,}")
        print(f"  Columns: {list(df_rel.columns)}")
        print(f"\n  First 5 rows:")
        print(df_rel[['timestamp', 'open', 'high', 'low', 'close', 'volume']].head().to_string(index=False))
else:
    print("  Price directory not found!")

# 2. Index Data
print("\n📊 INDEX DATA ANALYSIS") 
print("-" * 40)

index_dir = DATA_DIR / "raw/indices"
if index_dir.exists():
    index_files = list(index_dir.glob("*.parquet"))
    print(f"Total indices: {len(index_files)}")
    for f in index_files[:5]:
        df = pd.read_parquet(f)
        print(f"  {f.stem}: {len(df):,} rows")
else:
    print("  Index directory not found!")

# 3. Fundamental Data
print("\n💰 FUNDAMENTAL DATA ANALYSIS")
print("-" * 40)

fund_dir = DATA_DIR / "fundamentals"
if fund_dir.exists():
    fund_files = list(fund_dir.glob("*.csv"))
    if fund_files:
        df_fund = pd.read_csv(fund_files[0])
        print(f"File: {fund_files[0].name}")
        print(f"Total companies: {len(df_fund)}")
        print(f"Columns: {list(df_fund.columns)}")
        print(f"\nTop 10 by Market Cap:")
        if 'market_cap' in df_fund.columns:
            top10 = df_fund.nlargest(10, 'market_cap')[['symbol', 'name', 'market_cap', 'pe_trailing']].copy()
            top10['market_cap'] = top10['market_cap'].apply(lambda x: f"₹{x/1e12:.2f}T" if pd.notna(x) else "N/A")
            print(top10.to_string(index=False))
else:
    print("  Fundamentals directory not found!")

# 4. News/Sentiment Data
print("\n📰 NEWS & SENTIMENT DATA ANALYSIS")
print("-" * 40)

fingpt_dir = DATA_DIR / "fingpt/datasets"
if fingpt_dir.exists():
    fingpt_files = list(fingpt_dir.glob("*.parquet"))
    print(f"FinGPT Datasets: {len(fingpt_files)}")
    total_samples = 0
    for f in fingpt_files:
        df = pd.read_parquet(f)
        print(f"  {f.stem}: {len(df):,} samples")
        total_samples += len(df)
    print(f"\nTotal sentiment samples: {total_samples:,}")
    
    # Show sample from sentiment data
    if fingpt_files:
        df_sample = pd.read_parquet(fingpt_files[0])
        print(f"\nSample from {fingpt_files[0].stem}:")
        if 'input' in df_sample.columns:
            print(f"  Input: {df_sample['input'].iloc[0][:100]}...")
        if 'output' in df_sample.columns:
            print(f"  Output: {df_sample['output'].iloc[0]}")
else:
    print("  FinGPT directory not found!")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"✅ Price Data: {len(list((DATA_DIR / 'raw/prices').glob('*.parquet'))) if (DATA_DIR / 'raw/prices').exists() else 0} stocks")
print(f"✅ Index Data: {len(list((DATA_DIR / 'raw/indices').glob('*.parquet'))) if (DATA_DIR / 'raw/indices').exists() else 0} indices")
print(f"✅ Fundamentals: {len(df_fund) if 'df_fund' in dir() else 0} companies")
print(f"✅ Sentiment Data: {total_samples if 'total_samples' in dir() else 0} samples")
print("\nPhase 1 Data Collection: COMPLETE ✓")
print("=" * 60)
