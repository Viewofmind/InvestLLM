#!/usr/bin/env python3
"""
HuggingFace Indian Financial News Dataset
==========================================

This dataset is GOLD - it's already labeled for sentiment!

Dataset: kdave/Indian_Financial_News
- 10,000+ Indian financial news articles
- Pre-labeled sentiment (positive, negative, neutral)
- Ready for training!

Other useful datasets:
- soumikrakshit/indian-financial-news
- FinGPT/fingpt-sentiment-train (global, but useful)

Usage:
    python scripts/download_hf_news.py
    python scripts/download_hf_news.py --dataset kdave/Indian_Financial_News

Cost: FREE
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress

console = Console()

# Try to import datasets
try:
    from datasets import load_dataset, Dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    console.print("[yellow]datasets not installed. Run: pip install datasets[/yellow]")


# ===========================================
# AVAILABLE DATASETS
# ===========================================

INDIAN_FINANCE_DATASETS = {
    "kdave/Indian_Financial_News": {
        "description": "Indian financial news with sentiment labels",
        "size": "~10K articles",
        "labels": ["positive", "negative", "neutral"],
        "columns": ["text", "label"],
        "priority": 1,
        "use_for": "Sentiment training"
    },
    "soumikrakshit/indian-financial-news": {
        "description": "Indian financial news headlines",
        "size": "~5K articles",
        "labels": None,
        "columns": ["headline", "description"],
        "priority": 2,
        "use_for": "Additional corpus"
    },
    "FinGPT/fingpt-sentiment-train": {
        "description": "Global financial sentiment (includes some Indian)",
        "size": "~50K articles",
        "labels": ["positive", "negative", "neutral"],
        "columns": ["input", "output"],
        "priority": 3,
        "use_for": "Pre-training / augmentation"
    },
    "zeroshot/twitter-financial-news-sentiment": {
        "description": "Twitter financial sentiment",
        "size": "~10K tweets",
        "labels": ["Bearish", "Bullish", "Neutral"],
        "columns": ["text", "label"],
        "priority": 4,
        "use_for": "Social sentiment"
    },
    "financial_phrasebank": {
        "description": "Financial phrase sentiment (academic)",
        "size": "~5K phrases",
        "labels": ["positive", "negative", "neutral"],
        "columns": ["sentence", "label"],
        "priority": 5,
        "use_for": "Fine-grained sentiment"
    }
}


class HuggingFaceDataCollector:
    """
    Downloads and processes financial datasets from HuggingFace
    """
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data/huggingface")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Processed data directory
        self.processed_dir = Path("data/processed/sentiment")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download_dataset(
        self,
        dataset_name: str,
        split: str = None
    ) -> Optional[Dataset]:
        """
        Download a dataset from HuggingFace
        
        Args:
            dataset_name: HuggingFace dataset name
            split: Optional split (train, test, validation)
            
        Returns:
            HuggingFace Dataset object
        """
        console.print(f"\n[bold]Downloading: {dataset_name}[/bold]")
        
        try:
            if split:
                dataset = load_dataset(dataset_name, split=split)
            else:
                dataset = load_dataset(dataset_name)
            
            console.print(f"[green]✓ Downloaded successfully[/green]")
            
            # Show info
            console.print(f"  Type: {type(dataset)}")
            if hasattr(dataset, 'num_rows'):
                console.print(f"  Rows: {dataset.num_rows}")
            elif isinstance(dataset, dict):
                console.print(f"  Keys: {list(dataset.keys())}")
            
            return dataset
            
        except Exception as e:
            console.print(f"[red]Error downloading {dataset_name}: {e}[/red]")
            return None
    
    def download_indian_financial_news(self) -> pd.DataFrame:
        """
        Download the primary Indian Financial News dataset
        
        This is the most valuable dataset for our use case:
        - Indian market specific
        - Pre-labeled sentiment
        - Good size for fine-tuning
        """
        dataset = self.download_dataset("kdave/Indian_Financial_News")
        
        if dataset is None:
            return pd.DataFrame()
        
        # Convert to DataFrame
        if isinstance(dataset, dict):
            # Combine all splits
            dfs = []
            for split_name, split_data in dataset.items():
                df = split_data.to_pandas()
                df['split'] = split_name
                dfs.append(df)
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = dataset.to_pandas()
        
        # Standardize columns
        if 'Sentiment' in df.columns:
            df = df.rename(columns={'Sentiment': 'label'})
        if 'News' in df.columns:
            df = df.rename(columns={'News': 'text'})
        if 'news' in df.columns:
            df = df.rename(columns={'news': 'text'})
        if 'sentiment' in df.columns:
            df = df.rename(columns={'sentiment': 'label'})
        
        # Clean labels
        if 'label' in df.columns:
            df['label'] = df['label'].str.lower().str.strip()
            
            # Map variations
            label_map = {
                'pos': 'positive',
                'neg': 'negative',
                'neu': 'neutral',
                '1': 'positive',
                '0': 'neutral',
                '-1': 'negative',
                '2': 'positive',
            }
            df['label'] = df['label'].replace(label_map)
        
        # Add source
        df['source'] = 'hf_indian_financial_news'
        
        # Save
        filepath = self.data_dir / "indian_financial_news.parquet"
        df.to_parquet(filepath, index=False)
        console.print(f"[green]Saved to: {filepath}[/green]")
        
        return df
    
    def download_fingpt_sentiment(self) -> pd.DataFrame:
        """
        Download FinGPT sentiment dataset
        
        Larger dataset with global financial sentiment
        Good for pre-training before fine-tuning on Indian data
        """
        dataset = self.download_dataset("FinGPT/fingpt-sentiment-train", split="train")
        
        if dataset is None:
            return pd.DataFrame()
        
        df = dataset.to_pandas()
        
        # Standardize columns
        if 'input' in df.columns:
            df = df.rename(columns={'input': 'text'})
        if 'output' in df.columns:
            df = df.rename(columns={'output': 'label'})
        
        # Clean labels
        if 'label' in df.columns:
            df['label'] = df['label'].str.lower().str.strip()
        
        df['source'] = 'hf_fingpt_sentiment'
        
        # Save
        filepath = self.data_dir / "fingpt_sentiment.parquet"
        df.to_parquet(filepath, index=False)
        console.print(f"[green]Saved to: {filepath}[/green]")
        
        return df
    
    def download_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Download all useful financial datasets
        """
        results = {}
        
        console.print(Panel.fit(
            "[bold blue]Downloading HuggingFace Financial Datasets[/bold blue]\n"
            "These are FREE and pre-labeled!",
            border_style="blue"
        ))
        
        # Priority 1: Indian Financial News (MOST IMPORTANT)
        console.print("\n[bold cyan]1. Indian Financial News (PRIMARY)[/bold cyan]")
        df = self.download_indian_financial_news()
        if not df.empty:
            results['indian_financial_news'] = df
        
        # Priority 2: FinGPT Sentiment
        console.print("\n[bold cyan]2. FinGPT Sentiment (AUGMENTATION)[/bold cyan]")
        df = self.download_fingpt_sentiment()
        if not df.empty:
            results['fingpt_sentiment'] = df
        
        # Priority 3: Twitter Financial Sentiment
        console.print("\n[bold cyan]3. Twitter Financial Sentiment[/bold cyan]")
        try:
            dataset = self.download_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
            if dataset:
                df = dataset.to_pandas()
                df['source'] = 'hf_twitter_financial'
                df.to_parquet(self.data_dir / "twitter_financial.parquet", index=False)
                results['twitter_financial'] = df
        except Exception as e:
            console.print(f"[yellow]Skipped: {e}[/yellow]")
        
        # Priority 4: Financial Phrasebank
        console.print("\n[bold cyan]4. Financial Phrasebank[/bold cyan]")
        try:
            dataset = self.download_dataset("financial_phrasebank", split="sentences_allagree")
            if dataset:
                df = dataset.to_pandas()
                df = df.rename(columns={'sentence': 'text'})
                df['source'] = 'hf_phrasebank'
                df.to_parquet(self.data_dir / "financial_phrasebank.parquet", index=False)
                results['financial_phrasebank'] = df
        except Exception as e:
            console.print(f"[yellow]Skipped: {e}[/yellow]")
        
        return results
    
    def create_combined_training_set(self) -> pd.DataFrame:
        """
        Combine all datasets into one training set
        
        Strategy:
        1. Use all Indian news (highest priority)
        2. Add subset of global data for variety
        3. Balance classes
        """
        all_data = []
        
        # Load all downloaded datasets
        for filepath in self.data_dir.glob("*.parquet"):
            df = pd.read_parquet(filepath)
            
            # Ensure required columns
            if 'text' in df.columns and 'label' in df.columns:
                all_data.append(df[['text', 'label', 'source']])
        
        if not all_data:
            console.print("[red]No datasets found![/red]")
            return pd.DataFrame()
        
        # Combine
        combined = pd.concat(all_data, ignore_index=True)
        
        # Clean
        combined = combined.dropna(subset=['text', 'label'])
        combined = combined[combined['text'].str.len() > 20]  # Remove too short
        
        # Standardize labels
        label_map = {
            'positive': 'positive',
            'negative': 'negative',
            'neutral': 'neutral',
            'bullish': 'positive',
            'bearish': 'negative',
            '1': 'positive',
            '0': 'neutral',
            '-1': 'negative',
            '2': 'positive',
        }
        combined['label'] = combined['label'].astype(str).str.lower().str.strip()
        combined['label'] = combined['label'].map(lambda x: label_map.get(x, x))
        
        # Keep only valid labels
        valid_labels = ['positive', 'negative', 'neutral']
        combined = combined[combined['label'].isin(valid_labels)]
        
        # Remove duplicates
        combined = combined.drop_duplicates(subset=['text'])
        
        # Save combined dataset
        filepath = self.processed_dir / "sentiment_training_data.parquet"
        combined.to_parquet(filepath, index=False)
        
        # Also save as CSV for easy viewing
        combined.to_csv(self.processed_dir / "sentiment_training_data.csv", index=False)
        
        console.print(f"\n[green]Combined dataset saved to: {filepath}[/green]")
        
        return combined
    
    def show_statistics(self):
        """Show statistics of downloaded data"""
        table = Table(title="Downloaded Datasets Statistics")
        table.add_column("Dataset", style="cyan")
        table.add_column("Rows", style="magenta")
        table.add_column("Positive", style="green")
        table.add_column("Negative", style="red")
        table.add_column("Neutral", style="yellow")
        
        total_rows = 0
        
        for filepath in self.data_dir.glob("*.parquet"):
            df = pd.read_parquet(filepath)
            
            if 'label' in df.columns:
                label_counts = df['label'].value_counts()
                pos = label_counts.get('positive', 0)
                neg = label_counts.get('negative', 0)
                neu = label_counts.get('neutral', 0)
            else:
                pos = neg = neu = '-'
            
            table.add_row(
                filepath.stem,
                f"{len(df):,}",
                str(pos),
                str(neg),
                str(neu)
            )
            total_rows += len(df)
        
        console.print(table)
        console.print(f"\n[bold]Total: {total_rows:,} records[/bold]")
        
        # Show combined stats if exists
        combined_path = self.processed_dir / "sentiment_training_data.parquet"
        if combined_path.exists():
            df = pd.read_parquet(combined_path)
            console.print(f"\n[bold cyan]Combined Training Set:[/bold cyan]")
            console.print(f"  Total: {len(df):,}")
            console.print(f"  Labels: {df['label'].value_counts().to_dict()}")


# ===========================================
# MAIN
# ===========================================

def main(dataset: str = None, all_datasets: bool = False, stats: bool = False):
    """Main function"""
    
    if not HF_AVAILABLE:
        console.print("[red]datasets package required. Install with:[/red]")
        console.print("pip install datasets")
        return
    
    console.print(Panel.fit(
        "[bold blue]HuggingFace Financial News Datasets[/bold blue]\n"
        "FREE pre-labeled sentiment data!",
        border_style="blue"
    ))
    
    collector = HuggingFaceDataCollector()
    
    if stats:
        collector.show_statistics()
        return
    
    if all_datasets:
        # Download all
        results = collector.download_all_datasets()
        
        # Create combined training set
        console.print("\n[bold]Creating combined training set...[/bold]")
        combined = collector.create_combined_training_set()
        
        # Show statistics
        collector.show_statistics()
        
    elif dataset:
        # Download specific dataset
        ds = collector.download_dataset(dataset)
        if ds:
            console.print(f"[green]Downloaded {dataset}[/green]")
    else:
        # Default: Download the primary dataset
        df = collector.download_indian_financial_news()
        
        if not df.empty:
            console.print("\n[bold]Dataset Statistics:[/bold]")
            console.print(f"  Total rows: {len(df):,}")
            if 'label' in df.columns:
                console.print(f"  Labels: {df['label'].value_counts().to_dict()}")
            
            console.print(f"\n[bold]Sample data:[/bold]")
            console.print(df[['text', 'label']].head(5).to_string())
    
    console.print("\n" + "="*50)
    console.print("[bold green]Download Complete![/bold green]")
    console.print(f"  Data saved to: {collector.data_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HuggingFace financial datasets")
    parser.add_argument("--dataset", type=str, help="Specific dataset to download")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    
    args = parser.parse_args()
    
    main(dataset=args.dataset, all_datasets=args.all, stats=args.stats)
