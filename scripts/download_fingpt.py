#!/usr/bin/env python3
"""
FinGPT Integration for InvestLLM
=================================

FinGPT provides PRE-TRAINED financial models and datasets!

This is HUGE - instead of training from scratch:
- Use FinGPT sentiment models as BASE
- Fine-tune on Indian market data
- Save 2-3 months of work!

Available Resources:
1. MODELS (Pre-trained):
   - fingpt-sentiment (Llama2/Llama3 fine-tuned for sentiment)
   - fingpt-forecaster (Price prediction)
   
2. DATASETS:
   - fingpt-sentiment-train (Sentiment labeled data)
   - fingpt-fiqa_qa (Financial Q&A)
   - fingpt-convfinqa (Conversational Q&A)
   - fingpt-finred (Relation extraction)
   - fingpt-ner-cls (Named Entity Recognition)

Usage:
    python scripts/download_fingpt.py --all
    python scripts/download_fingpt.py --models
    python scripts/download_fingpt.py --datasets

Cost: FREE (open source)
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

# Try imports
try:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download, hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# ===========================================
# FINGPT RESOURCES CATALOG
# ===========================================

FINGPT_MODELS = {
    # Sentiment Models (MOST USEFUL)
    "FinGPT/fingpt-sentiment_llama2-13b_lora": {
        "name": "FinGPT Sentiment (Llama2-13B)",
        "type": "sentiment",
        "base_model": "meta-llama/Llama-2-13b-hf",
        "description": "Sentiment analysis for financial text",
        "use_for": "News/earnings sentiment scoring",
        "priority": 1,
        "vram_required": "24GB+",
        "quantized": False
    },
    "FinGPT/fingpt-sentiment_llama2-7b_lora": {
        "name": "FinGPT Sentiment (Llama2-7B)",
        "type": "sentiment",
        "base_model": "meta-llama/Llama-2-7b-hf",
        "description": "Lighter sentiment model",
        "use_for": "Sentiment on consumer GPU",
        "priority": 2,
        "vram_required": "16GB",
        "quantized": False
    },
    "FinGPT/fingpt-mt_llama2-7b_lora": {
        "name": "FinGPT Multi-Task (Llama2-7B)",
        "type": "multi-task",
        "base_model": "meta-llama/Llama-2-7b-hf",
        "description": "Multi-task: sentiment, NER, QA",
        "use_for": "General financial NLP",
        "priority": 3,
        "vram_required": "16GB",
        "quantized": False
    },
    # Forecaster Models
    "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora": {
        "name": "FinGPT Forecaster (Dow30)",
        "type": "forecaster",
        "base_model": "meta-llama/Llama-2-7b-hf",
        "description": "Stock movement prediction",
        "use_for": "Reference architecture for price prediction",
        "priority": 4,
        "vram_required": "16GB",
        "quantized": False
    },
    # Newer Models (Llama3 based)
    "FinGPT/fingpt-sentiment-llama3-8b-lora": {
        "name": "FinGPT Sentiment (Llama3-8B)",
        "type": "sentiment",
        "base_model": "meta-llama/Meta-Llama-3-8B",
        "description": "Latest sentiment model",
        "use_for": "Best quality sentiment",
        "priority": 1,
        "vram_required": "16GB",
        "quantized": False
    }
}

FINGPT_DATASETS = {
    # Sentiment Datasets
    "FinGPT/fingpt-sentiment-train": {
        "name": "FinGPT Sentiment Training Data",
        "type": "sentiment",
        "size": "~76K samples",
        "description": "Financial sentiment with instructions",
        "use_for": "Sentiment model fine-tuning",
        "priority": 1,
        "columns": ["instruction", "input", "output"]
    },
    "FinGPT/fingpt-sentiment-cls": {
        "name": "FinGPT Sentiment Classification",
        "type": "sentiment",
        "size": "~50K samples",
        "description": "Classification format sentiment",
        "use_for": "Sentiment classification training",
        "priority": 2,
        "columns": ["text", "label"]
    },
    # Q&A Datasets
    "FinGPT/fingpt-fiqa_qa": {
        "name": "FinGPT Financial QA",
        "type": "qa",
        "size": "~17K samples",
        "description": "Financial question-answering",
        "use_for": "Building investment Q&A assistant",
        "priority": 1,
        "columns": ["question", "answer"]
    },
    "FinGPT/fingpt-convfinqa": {
        "name": "FinGPT Conversational QA",
        "type": "qa",
        "size": "~14K samples",
        "description": "Multi-turn financial conversations",
        "use_for": "Conversational investment analysis",
        "priority": 2,
        "columns": ["conversation"]
    },
    # NER & Relation Extraction
    "FinGPT/fingpt-ner": {
        "name": "FinGPT Named Entity Recognition",
        "type": "ner",
        "size": "~14K samples",
        "description": "Financial entity recognition",
        "use_for": "Extracting companies, metrics, people",
        "priority": 3,
        "columns": ["text", "entities"]
    },
    "FinGPT/fingpt-finred": {
        "name": "FinGPT Relation Extraction",
        "type": "relation",
        "size": "~8K samples",
        "description": "Financial relationship extraction",
        "use_for": "Understanding entity relationships",
        "priority": 4,
        "columns": ["text", "relations"]
    },
    # Headline Dataset
    "FinGPT/fingpt-headline": {
        "name": "FinGPT Headlines",
        "type": "sentiment",
        "size": "~100K samples",
        "description": "Financial news headlines with sentiment",
        "use_for": "Quick sentiment training",
        "priority": 2,
        "columns": ["headline", "sentiment"]
    },
    # Forecaster Training Data
    "FinGPT/fingpt-forecaster-dow30": {
        "name": "FinGPT Forecaster Data (Dow30)",
        "type": "forecaster",
        "size": "~50K samples",
        "description": "Price movement with news context",
        "use_for": "Reference for price prediction",
        "priority": 3,
        "columns": ["news", "price_movement"]
    }
}


class FinGPTIntegration:
    """
    Downloads and integrates FinGPT models and datasets
    """
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path("data/fingpt")
        self.models_dir = self.base_dir / "models"
        self.datasets_dir = self.base_dir / "datasets"
        
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.datasets_dir.mkdir(exist_ok=True)
    
    # ===========================================
    # DATASET DOWNLOADS
    # ===========================================
    
    def download_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Download a FinGPT dataset"""
        info = FINGPT_DATASETS.get(dataset_id)
        
        if not info:
            console.print(f"[red]Unknown dataset: {dataset_id}[/red]")
            return None
        
        console.print(f"\n[bold]Downloading: {info['name']}[/bold]")
        console.print(f"[dim]{info['description']}[/dim]")
        
        try:
            # Load dataset
            dataset = load_dataset(dataset_id, trust_remote_code=True)
            
            # Convert to DataFrame
            if isinstance(dataset, dict):
                # Has splits
                dfs = []
                for split_name, split_data in dataset.items():
                    df = split_data.to_pandas()
                    df['split'] = split_name
                    dfs.append(df)
                    console.print(f"  {split_name}: {len(df):,} rows")
                df = pd.concat(dfs, ignore_index=True)
            else:
                df = dataset.to_pandas()
                console.print(f"  Rows: {len(df):,}")
            
            # Save
            dataset_name = dataset_id.split("/")[-1]
            filepath = self.datasets_dir / f"{dataset_name}.parquet"
            df.to_parquet(filepath, index=False)
            console.print(f"[green]✓ Saved to: {filepath}[/green]")
            
            return df
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return None
    
    def download_all_datasets(self, types: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Download all or filtered datasets"""
        results = {}
        
        console.print(Panel.fit(
            "[bold blue]Downloading FinGPT Datasets[/bold blue]\n"
            "Pre-processed financial NLP data!",
            border_style="blue"
        ))
        
        # Sort by priority
        sorted_datasets = sorted(
            FINGPT_DATASETS.items(),
            key=lambda x: x[1]['priority']
        )
        
        for dataset_id, info in sorted_datasets:
            # Filter by type if specified
            if types and info['type'] not in types:
                continue
            
            df = self.download_dataset(dataset_id)
            if df is not None:
                results[dataset_id] = df
        
        return results
    
    # ===========================================
    # MODEL DOWNLOADS
    # ===========================================
    
    def download_model(self, model_id: str) -> Optional[Path]:
        """
        Download a FinGPT model (LoRA adapter)
        
        Note: These are LoRA adapters, not full models.
        You need the base model (Llama2/3) to use them.
        """
        info = FINGPT_MODELS.get(model_id)
        
        if not info:
            console.print(f"[red]Unknown model: {model_id}[/red]")
            return None
        
        console.print(f"\n[bold]Downloading: {info['name']}[/bold]")
        console.print(f"[dim]Base model: {info['base_model']}[/dim]")
        console.print(f"[dim]VRAM required: {info['vram_required']}[/dim]")
        
        model_name = model_id.split("/")[-1]
        model_path = self.models_dir / model_name
        
        try:
            # Download model files
            snapshot_download(
                repo_id=model_id,
                local_dir=str(model_path),
                local_dir_use_symlinks=False
            )
            
            console.print(f"[green]✓ Downloaded to: {model_path}[/green]")
            return model_path
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("[yellow]Note: Some models require Llama license acceptance[/yellow]")
            console.print("Visit: https://huggingface.co/meta-llama to accept license")
            return None
    
    def download_recommended_models(self) -> Dict[str, Path]:
        """Download recommended models for InvestLLM"""
        results = {}
        
        console.print(Panel.fit(
            "[bold blue]Downloading FinGPT Models[/bold blue]\n"
            "Pre-trained financial LLM adapters!",
            border_style="blue"
        ))
        
        # Recommended models
        recommended = [
            "FinGPT/fingpt-sentiment_llama2-7b_lora",  # Sentiment (fits most GPUs)
            "FinGPT/fingpt-mt_llama2-7b_lora",         # Multi-task
        ]
        
        for model_id in recommended:
            path = self.download_model(model_id)
            if path:
                results[model_id] = path
        
        return results
    
    # ===========================================
    # USAGE EXAMPLES
    # ===========================================
    
    def create_sentiment_training_set(self) -> pd.DataFrame:
        """
        Create combined sentiment training set from FinGPT data
        
        Combines:
        - fingpt-sentiment-train
        - fingpt-sentiment-cls
        - fingpt-headline
        """
        all_data = []
        
        # Load sentiment datasets
        sentiment_datasets = [
            "fingpt-sentiment-train",
            "fingpt-sentiment-cls", 
            "fingpt-headline"
        ]
        
        for dataset_name in sentiment_datasets:
            filepath = self.datasets_dir / f"{dataset_name}.parquet"
            
            if filepath.exists():
                df = pd.read_parquet(filepath)
                
                # Standardize columns
                if 'output' in df.columns:
                    df = df.rename(columns={'output': 'label', 'input': 'text'})
                if 'headline' in df.columns:
                    df = df.rename(columns={'headline': 'text', 'sentiment': 'label'})
                
                if 'text' in df.columns and 'label' in df.columns:
                    df['source'] = f'fingpt_{dataset_name}'
                    all_data.append(df[['text', 'label', 'source']])
                    console.print(f"  Added {len(df):,} from {dataset_name}")
        
        if not all_data:
            console.print("[yellow]No FinGPT data found. Run download first.[/yellow]")
            return pd.DataFrame()
        
        combined = pd.concat(all_data, ignore_index=True)
        
        # Clean labels
        label_map = {
            'positive': 'positive',
            'negative': 'negative', 
            'neutral': 'neutral',
            'mildly positive': 'positive',
            'mildly negative': 'negative',
            'moderately positive': 'positive',
            'moderately negative': 'negative',
            'strong positive': 'positive',
            'strong negative': 'negative',
        }
        combined['label'] = combined['label'].str.lower().str.strip()
        combined['label'] = combined['label'].map(lambda x: label_map.get(x, x))
        
        # Keep valid labels
        valid_labels = ['positive', 'negative', 'neutral']
        combined = combined[combined['label'].isin(valid_labels)]
        
        # Save
        output_path = Path("data/processed/sentiment/fingpt_sentiment_combined.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(output_path, index=False)
        
        console.print(f"\n[green]Combined FinGPT sentiment data:[/green]")
        console.print(f"  Total: {len(combined):,}")
        console.print(f"  Labels: {combined['label'].value_counts().to_dict()}")
        console.print(f"  Saved to: {output_path}")
        
        return combined
    
    def show_catalog(self):
        """Show available FinGPT resources"""
        # Models table
        console.print("\n[bold cyan]Available FinGPT Models:[/bold cyan]")
        table = Table()
        table.add_column("Model", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("VRAM", style="green")
        table.add_column("Use For")
        
        for model_id, info in sorted(FINGPT_MODELS.items(), key=lambda x: x[1]['priority']):
            table.add_row(
                info['name'],
                info['type'],
                info['vram_required'],
                info['use_for']
            )
        console.print(table)
        
        # Datasets table
        console.print("\n[bold cyan]Available FinGPT Datasets:[/bold cyan]")
        table = Table()
        table.add_column("Dataset", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Size", style="green")
        table.add_column("Use For")
        
        for dataset_id, info in sorted(FINGPT_DATASETS.items(), key=lambda x: x[1]['priority']):
            table.add_row(
                info['name'],
                info['type'],
                info['size'],
                info['use_for']
            )
        console.print(table)
    
    def show_status(self):
        """Show download status"""
        console.print("\n[bold]Downloaded Resources:[/bold]")
        
        # Check datasets
        console.print("\n[cyan]Datasets:[/cyan]")
        for filepath in self.datasets_dir.glob("*.parquet"):
            df = pd.read_parquet(filepath)
            console.print(f"  ✓ {filepath.stem}: {len(df):,} rows")
        
        if not list(self.datasets_dir.glob("*.parquet")):
            console.print("  [dim]No datasets downloaded yet[/dim]")
        
        # Check models
        console.print("\n[cyan]Models:[/cyan]")
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                console.print(f"  ✓ {model_dir.name}")
        
        if not list(self.models_dir.iterdir()):
            console.print("  [dim]No models downloaded yet[/dim]")


# ===========================================
# USAGE WITH INVESTLLM
# ===========================================

def create_usage_example():
    """Create example code for using FinGPT in InvestLLM"""
    
    example_code = '''
# ===========================================
# Using FinGPT Sentiment Model in InvestLLM
# ===========================================

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 1. Load base model (requires Llama access from HuggingFace)
base_model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. Load FinGPT LoRA adapter
adapter_path = "data/fingpt/models/fingpt-sentiment_llama2-7b_lora"
model = PeftModel.from_pretrained(base_model, adapter_path)

# 3. Use for sentiment analysis
def analyze_sentiment(text: str) -> str:
    prompt = f"""Instruction: What is the sentiment of this news? Please choose an answer from {{negative/neutral/positive}}.
Input: {text}
Answer: """
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract sentiment from response
    if "positive" in response.lower():
        return "positive"
    elif "negative" in response.lower():
        return "negative"
    else:
        return "neutral"

# 4. Example usage
news = "TCS Q3 profit rises 15% to Rs 12,380 crore, beats analyst estimates"
sentiment = analyze_sentiment(news)
print(f"Sentiment: {sentiment}")  # Output: positive


# ===========================================
# Fine-tuning FinGPT on Indian Market Data
# ===========================================

from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig

# 1. Load Indian financial news (labeled)
indian_data = load_dataset("parquet", 
    data_files="data/processed/sentiment/sentiment_training_data.parquet"
)

# 2. Format for training
def format_prompt(example):
    return f"""Instruction: What is the sentiment of this Indian market news?
Input: {example['text']}
Answer: {example['label']}"""

# 3. Configure LoRA for fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 4. Train
training_args = TrainingArguments(
    output_dir="models/fingpt-indian-sentiment",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    fp16=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=indian_data["train"],
    peft_config=lora_config,
    formatting_func=format_prompt,
    args=training_args,
)

trainer.train()
trainer.save_model("models/fingpt-indian-sentiment")
'''
    
    return example_code


# ===========================================
# MAIN
# ===========================================

def main(
    datasets: bool = False,
    models: bool = False,
    all_resources: bool = False,
    catalog: bool = False,
    status: bool = False,
    combine: bool = False
):
    """Main function"""
    
    if not HF_AVAILABLE:
        console.print("[red]Required packages not installed:[/red]")
        console.print("pip install datasets huggingface_hub peft transformers")
        return
    
    console.print(Panel.fit(
        "[bold blue]FinGPT Integration for InvestLLM[/bold blue]\n"
        "Pre-trained financial AI models & datasets!",
        border_style="blue"
    ))
    
    integration = FinGPTIntegration()
    
    if catalog:
        integration.show_catalog()
        return
    
    if status:
        integration.show_status()
        return
    
    if combine:
        integration.create_sentiment_training_set()
        return
    
    if all_resources or datasets:
        console.print("\n[bold]Downloading FinGPT Datasets...[/bold]")
        integration.download_all_datasets()
    
    if all_resources or models:
        console.print("\n[bold]Downloading FinGPT Models...[/bold]")
        console.print("[yellow]Note: Models require HuggingFace login and Llama license[/yellow]")
        console.print("Run: huggingface-cli login")
        integration.download_recommended_models()
    
    if not any([datasets, models, all_resources, catalog, status, combine]):
        # Default: show catalog and status
        integration.show_catalog()
        console.print("\n[bold]Usage:[/bold]")
        console.print("  python scripts/download_fingpt.py --datasets   # Download datasets")
        console.print("  python scripts/download_fingpt.py --models     # Download models")
        console.print("  python scripts/download_fingpt.py --all        # Download everything")
        console.print("  python scripts/download_fingpt.py --combine    # Create training set")
    
    console.print("\n[bold green]Done![/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FinGPT resources")
    parser.add_argument("--datasets", action="store_true", help="Download datasets")
    parser.add_argument("--models", action="store_true", help="Download models")
    parser.add_argument("--all", action="store_true", help="Download everything")
    parser.add_argument("--catalog", action="store_true", help="Show available resources")
    parser.add_argument("--status", action="store_true", help="Show download status")
    parser.add_argument("--combine", action="store_true", help="Create combined training set")
    
    args = parser.parse_args()
    
    main(
        datasets=args.datasets,
        models=args.models,
        all_resources=args.all,
        catalog=args.catalog,
        status=args.status,
        combine=args.combine
    )
