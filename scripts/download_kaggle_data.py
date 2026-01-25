#!/usr/bin/env python3
"""
Kaggle Indian Stock Market Datasets
====================================

FREE historical data from Kaggle!

Useful Datasets:
1. NIFTY 50 Stock Market Data (2000-2023)
2. Indian Stock Market Historical Data
3. NSE India Stock Data

Requirements:
1. Kaggle account (free)
2. API credentials from kaggle.com/account

Setup:
1. Go to kaggle.com/account
2. Click "Create New API Token"
3. Save kaggle.json to ~/.kaggle/kaggle.json
4. chmod 600 ~/.kaggle/kaggle.json

Usage:
    python scripts/download_kaggle_data.py
    python scripts/download_kaggle_data.py --list
    python scripts/download_kaggle_data.py --dataset rohanrao/nifty50-stock-market-data

Cost: FREE
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict
import argparse
import shutil
import zipfile

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Try to import kaggle
try:
    import kaggle
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    console.print("[yellow]kaggle not installed. Run: pip install kaggle[/yellow]")


# ===========================================
# USEFUL KAGGLE DATASETS
# ===========================================

INDIAN_STOCK_DATASETS = {
    "rohanrao/nifty50-stock-market-data": {
        "name": "NIFTY 50 Stock Market Data (2000-2021)",
        "description": "Historical data for all NIFTY 50 stocks",
        "size": "~50 MB",
        "files": ["*.csv"],
        "priority": 1,
        "use_for": "Primary historical data"
    },
    "debashis74017/stock-market-data-nifty-50-stocks": {
        "name": "Stock Market Data - NIFTY 50",
        "description": "Daily OHLCV for NIFTY 50",
        "size": "~30 MB",
        "files": ["*.csv"],
        "priority": 2,
        "use_for": "Backup/validation"
    },
    "iamsouravbanerjee/nifty50-stocks-dataset": {
        "name": "NIFTY 50 Stocks Dataset",
        "description": "NIFTY 50 with company details",
        "size": "~20 MB",
        "files": ["*.csv"],
        "priority": 3,
        "use_for": "Company metadata"
    },
    "rohanrao/nifty-indices-dataset": {
        "name": "NIFTY Indices Dataset",
        "description": "All NIFTY indices historical data",
        "size": "~10 MB",
        "files": ["*.csv"],
        "priority": 4,
        "use_for": "Index data"
    },
    "sudalairajkumar/indian-stock-market-data": {
        "name": "Indian Stock Market Data",
        "description": "Broader Indian market data",
        "size": "~100 MB",
        "files": ["*.csv"],
        "priority": 5,
        "use_for": "Extended coverage"
    }
}


class KaggleDataCollector:
    """
    Downloads datasets from Kaggle
    """
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data/kaggle")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.api = None
        self._init_api()
    
    def _init_api(self):
        """Initialize Kaggle API"""
        if not KAGGLE_AVAILABLE:
            return
        
        # Check for credentials
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        
        if not kaggle_json.exists():
            console.print("[yellow]Kaggle credentials not found![/yellow]")
            console.print("\nSetup instructions:")
            console.print("1. Go to kaggle.com/account")
            console.print("2. Click 'Create New API Token'")
            console.print("3. Save kaggle.json to ~/.kaggle/kaggle.json")
            console.print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
            return
        
        try:
            self.api = KaggleApi()
            self.api.authenticate()
            console.print("[green]✓ Kaggle API authenticated[/green]")
        except Exception as e:
            console.print(f"[red]Kaggle authentication failed: {e}[/red]")
    
    def list_datasets(self):
        """List available Indian stock datasets"""
        table = Table(title="Available Kaggle Datasets")
        table.add_column("Dataset ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Size", style="green")
        table.add_column("Use For", style="yellow")
        
        for dataset_id, info in INDIAN_STOCK_DATASETS.items():
            table.add_row(
                dataset_id,
                info['name'],
                info['size'],
                info['use_for']
            )
        
        console.print(table)
    
    def download_dataset(
        self,
        dataset_id: str,
        unzip: bool = True
    ) -> Optional[Path]:
        """
        Download a dataset from Kaggle
        
        Args:
            dataset_id: Kaggle dataset ID (e.g., "rohanrao/nifty50-stock-market-data")
            unzip: Whether to unzip the downloaded files
            
        Returns:
            Path to downloaded data
        """
        if not self.api:
            console.print("[red]Kaggle API not initialized![/red]")
            return None
        
        # Create dataset directory
        dataset_name = dataset_id.split("/")[-1]
        dataset_dir = self.data_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"\n[bold]Downloading: {dataset_id}[/bold]")
        
        try:
            # Download
            self.api.dataset_download_files(
                dataset_id,
                path=str(dataset_dir),
                unzip=unzip
            )
            
            console.print(f"[green]✓ Downloaded to: {dataset_dir}[/green]")
            
            # List files
            files = list(dataset_dir.glob("*"))
            console.print(f"  Files: {len(files)}")
            for f in files[:10]:
                size_mb = f.stat().st_size / (1024 * 1024)
                console.print(f"    - {f.name} ({size_mb:.1f} MB)")
            
            return dataset_dir
            
        except Exception as e:
            console.print(f"[red]Error downloading: {e}[/red]")
            return None
    
    def download_nifty50_data(self) -> pd.DataFrame:
        """
        Download and process NIFTY 50 historical data
        
        This is the primary dataset for historical price data
        """
        dataset_id = "rohanrao/nifty50-stock-market-data"
        dataset_dir = self.download_dataset(dataset_id)
        
        if not dataset_dir:
            return pd.DataFrame()
        
        # Process all CSV files
        all_data = []
        
        for csv_file in dataset_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                
                # Extract symbol from filename
                symbol = csv_file.stem.upper()
                df['symbol'] = symbol
                
                all_data.append(df)
                console.print(f"  [dim]Processed: {csv_file.name} ({len(df)} rows)[/dim]")
                
            except Exception as e:
                console.print(f"[yellow]  Error processing {csv_file.name}: {e}[/yellow]")
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        combined = pd.concat(all_data, ignore_index=True)
        
        # Standardize columns
        column_map = {
            'Date': 'timestamp',
            'date': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume',
            'Symbol': 'symbol'
        }
        combined = combined.rename(columns=column_map)
        
        # Convert date
        if 'timestamp' in combined.columns:
            combined['timestamp'] = pd.to_datetime(combined['timestamp'])
        
        # Save processed data
        processed_dir = Path("data/processed/prices")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = processed_dir / "kaggle_nifty50_historical.parquet"
        combined.to_parquet(filepath, index=False)
        
        console.print(f"\n[green]Processed data saved to: {filepath}[/green]")
        console.print(f"  Total rows: {len(combined):,}")
        console.print(f"  Stocks: {combined['symbol'].nunique()}")
        console.print(f"  Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
        
        return combined
    
    def download_all_datasets(self) -> Dict[str, Path]:
        """Download all priority datasets"""
        results = {}
        
        console.print(Panel.fit(
            "[bold blue]Downloading Kaggle Datasets[/bold blue]\n"
            "FREE historical market data!",
            border_style="blue"
        ))
        
        # Sort by priority
        sorted_datasets = sorted(
            INDIAN_STOCK_DATASETS.items(),
            key=lambda x: x[1]['priority']
        )
        
        for dataset_id, info in sorted_datasets:
            console.print(f"\n[cyan]{info['priority']}. {info['name']}[/cyan]")
            
            try:
                path = self.download_dataset(dataset_id)
                if path:
                    results[dataset_id] = path
            except Exception as e:
                console.print(f"[yellow]  Skipped: {e}[/yellow]")
        
        return results
    
    def merge_with_yfinance(self, kaggle_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge Kaggle data with yfinance data
        
        Strategy:
        - Use Kaggle for older data (pre-2020)
        - Use yfinance for recent data (2020+)
        - Handle symbol mismatches
        """
        yfinance_dir = Path("data/raw/prices")
        
        if not yfinance_dir.exists():
            console.print("[yellow]yfinance data not found. Run collect_prices.py first.[/yellow]")
            return kaggle_df
        
        merged_data = []
        
        for symbol in kaggle_df['symbol'].unique():
            kaggle_symbol_data = kaggle_df[kaggle_df['symbol'] == symbol].copy()
            
            # Try to load yfinance data
            yf_file = yfinance_dir / f"{symbol}.parquet"
            
            if yf_file.exists():
                yf_data = pd.read_parquet(yf_file)
                
                # Get cutoff date (use Kaggle before, yfinance after)
                kaggle_max_date = kaggle_symbol_data['timestamp'].max()
                
                # Filter yfinance for dates after Kaggle
                yf_data = yf_data[yf_data['timestamp'] > kaggle_max_date]
                
                # Combine
                combined = pd.concat([kaggle_symbol_data, yf_data], ignore_index=True)
                combined = combined.drop_duplicates(subset=['timestamp'])
                combined = combined.sort_values('timestamp')
                
                merged_data.append(combined)
            else:
                merged_data.append(kaggle_symbol_data)
        
        if merged_data:
            result = pd.concat(merged_data, ignore_index=True)
            
            # Save merged data
            merged_dir = Path("data/processed/prices")
            merged_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = merged_dir / "merged_historical.parquet"
            result.to_parquet(filepath, index=False)
            
            console.print(f"\n[green]Merged data saved to: {filepath}[/green]")
            console.print(f"  Total rows: {len(result):,}")
            
            return result
        
        return kaggle_df


# ===========================================
# MANUAL DOWNLOAD INSTRUCTIONS
# ===========================================

def print_manual_instructions():
    """Print instructions for manual download if API fails"""
    console.print(Panel.fit(
        "[bold yellow]Manual Download Instructions[/bold yellow]\n"
        "If Kaggle API setup is too complex",
        border_style="yellow"
    ))
    
    console.print("""
[bold]Option 1: Download via Browser[/bold]

1. Go to: https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data
2. Click "Download" button
3. Extract zip to: data/kaggle/nifty50-stock-market-data/
4. Run: python scripts/download_kaggle_data.py --process-local

[bold]Option 2: Use Kaggle CLI[/bold]

1. Install: pip install kaggle
2. Setup credentials (see above)
3. Download:
   kaggle datasets download -d rohanrao/nifty50-stock-market-data
   unzip nifty50-stock-market-data.zip -d data/kaggle/

[bold]Direct Links:[/bold]
- NIFTY 50: https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data
- NIFTY Indices: https://www.kaggle.com/datasets/rohanrao/nifty-indices-dataset
- Indian Stocks: https://www.kaggle.com/datasets/sudalairajkumar/indian-stock-market-data
""")


# ===========================================
# MAIN
# ===========================================

def main(
    dataset: str = None,
    all_datasets: bool = False,
    list_only: bool = False,
    process_local: bool = False
):
    """Main function"""
    
    console.print(Panel.fit(
        "[bold blue]Kaggle Indian Stock Market Data[/bold blue]\n"
        "FREE historical price data!",
        border_style="blue"
    ))
    
    if list_only:
        collector = KaggleDataCollector()
        collector.list_datasets()
        return
    
    if not KAGGLE_AVAILABLE:
        console.print("[red]kaggle package required. Install with:[/red]")
        console.print("pip install kaggle")
        print_manual_instructions()
        return
    
    collector = KaggleDataCollector()
    
    if not collector.api:
        print_manual_instructions()
        return
    
    if process_local:
        # Process already downloaded data
        console.print("[bold]Processing local Kaggle data...[/bold]")
        kaggle_dir = Path("data/kaggle/nifty50-stock-market-data")
        
        if kaggle_dir.exists():
            all_data = []
            for csv_file in kaggle_dir.glob("*.csv"):
                df = pd.read_csv(csv_file)
                df['symbol'] = csv_file.stem.upper()
                all_data.append(df)
            
            if all_data:
                combined = pd.concat(all_data, ignore_index=True)
                combined.to_parquet("data/processed/prices/kaggle_nifty50.parquet")
                console.print(f"[green]Processed {len(combined):,} rows[/green]")
        else:
            console.print(f"[red]Directory not found: {kaggle_dir}[/red]")
        return
    
    if all_datasets:
        collector.download_all_datasets()
    elif dataset:
        collector.download_dataset(dataset)
    else:
        # Default: Download NIFTY 50 data
        df = collector.download_nifty50_data()
        
        if not df.empty:
            console.print("\n[bold]Sample data:[/bold]")
            console.print(df.head(10).to_string())
    
    console.print("\n" + "="*50)
    console.print("[bold green]Download Complete![/bold green]")
    console.print(f"  Data saved to: {collector.data_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Kaggle datasets")
    parser.add_argument("--dataset", type=str, help="Specific dataset to download")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--process-local", action="store_true", help="Process already downloaded data")
    
    args = parser.parse_args()
    
    main(
        dataset=args.dataset,
        all_datasets=args.all,
        list_only=args.list,
        process_local=args.process_local
    )
