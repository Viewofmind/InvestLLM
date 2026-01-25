"""
InvestLLM Configuration
Centralized configuration management
"""

from typing import Optional, List
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # Application
    app_name: str = "InvestLLM"
    app_env: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    
    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "investllm"
    postgres_user: str = "investllm"
    postgres_password: str = ""
    database_url: Optional[str] = None
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    
    # Data Sources
    zerodha_api_key: Optional[str] = None
    zerodha_api_secret: Optional[str] = None
    zerodha_access_token: Optional[str] = None
    firecrawl_api_key: Optional[str] = None
    
    # AI Providers
    hf_token: Optional[str] = None
    wandb_api_key: Optional[str] = None
    wandb_project: str = "investllm"
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    
    # Storage
    data_dir: Path = Path("./data")
    models_dir: Path = Path("./models")
    cache_dir: Path = Path("./cache")
    
    # Training
    default_batch_size: int = 32
    default_learning_rate: float = 2e-5
    default_epochs: int = 3
    
    # Backtesting
    initial_capital: float = 1000000
    max_position_size: float = 0.05
    stop_loss_percent: float = 0.05
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def db_url(self) -> str:
        """Get database URL"""
        if self.database_url:
            return self.database_url
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    def ensure_directories(self):
        """Create necessary directories"""
        for dir_path in [self.data_dir, self.models_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Create subdirectories
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "features").mkdir(exist_ok=True)
        (self.data_dir / "news").mkdir(exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    settings = Settings()
    settings.ensure_directories()
    return settings


# Global settings instance
settings = get_settings()


# ===========================================
# Stock Universe Configuration
# ===========================================

# NIFTY 50 Stocks
NIFTY_50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BPCL", "BHARTIARTL",
    "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC",
    "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LT",
    "M&M", "MARUTI", "NTPC", "NESTLEIND", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SUNPHARMA",
    "TCS", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TECHM",
    "TITAN", "ULTRACEMCO", "UPL", "WIPRO", "LTIM"
]

# NIFTY NEXT 50
NIFTY_NEXT_50 = [
    "ABB", "ADANIGREEN", "AMBUJACEM", "ATGL", "BAJAJHLDNG",
    "BANKBARODA", "BEL", "BERGEPAINT", "BOSCHLTD", "CANBK",
    "CHOLAFIN", "COLPAL", "DLF", "DABUR", "DMART",
    "GAIL", "GODREJCP", "HAVELLS", "HINDPETRO", "ICICIPRULI",
    "ICICIGI", "INDIGO", "IOC", "IRCTC", "JINDALSTEL",
    "LICI", "LUPIN", "MARICO", "NAUKRI", "NHPC",
    "OFSS", "PAGEIND", "PIIND", "PFC", "PIDILITIND",
    "PNB", "RECLTD", "SAIL", "SBICARD", "SHREECEM",
    "SIEMENS", "SRF", "TATAPOWER", "TORNTPHARM", "TRENT",
    "TVSMOTOR", "UNIONBANK", "VBL", "VEDL", "ZOMATO"
]

# All stocks to track
STOCK_UNIVERSE = NIFTY_50 + NIFTY_NEXT_50

# Indices
INDICES = [
    "^NSEI",      # NIFTY 50
    "^NSEBANK",   # BANK NIFTY
    "^NSMIDCP",   # NIFTY MIDCAP
    "^NSEIT",     # NIFTY IT
    "^CNXPHARMA", # NIFTY PHARMA
    "^CNXAUTO",   # NIFTY AUTO
    "^CNXFMCG",   # NIFTY FMCG
    "^CNXMETAL",  # NIFTY METAL
    "^CNXREALTY", # NIFTY REALTY
    "^CNXENERGY", # NIFTY ENERGY
]

# News Sources
NEWS_SOURCES = {
    "moneycontrol": {
        "base_url": "https://www.moneycontrol.com",
        "news_url": "https://www.moneycontrol.com/news/business/stocks/",
        "priority": 1
    },
    "economic_times": {
        "base_url": "https://economictimes.indiatimes.com",
        "news_url": "https://economictimes.indiatimes.com/markets/stocks/news",
        "priority": 1
    },
    "livemint": {
        "base_url": "https://www.livemint.com",
        "news_url": "https://www.livemint.com/market/stock-market-news",
        "priority": 2
    },
    "business_standard": {
        "base_url": "https://www.business-standard.com",
        "news_url": "https://www.business-standard.com/markets",
        "priority": 2
    },
    "ndtv_profit": {
        "base_url": "https://www.ndtvprofit.com",
        "news_url": "https://www.ndtvprofit.com/markets",
        "priority": 3
    }
}

# Sector Mapping
SECTOR_MAP = {
    "banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK", "BANKBARODA", "PNB", "CANBK"],
    "it": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "OFSS", "NAUKRI"],
    "auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "TVSMOTOR"],
    "pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP", "LUPIN", "TORNTPHARM"],
    "fmcg": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM", "DABUR", "MARICO", "COLPAL", "GODREJCP"],
    "energy": ["RELIANCE", "ONGC", "BPCL", "IOC", "GAIL", "NTPC", "POWERGRID", "TATAPOWER", "ADANIGREEN"],
    "metals": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "COALINDIA", "SAIL", "JINDALSTEL"],
    "finance": ["BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE", "ICICIPRULI", "ICICIGI", "CHOLAFIN", "SBICARD", "PFC", "RECLTD"],
    "infrastructure": ["LT", "ADANIPORTS", "ADANIENT", "DLF", "ULTRACEMCO", "GRASIM", "SHREECEM", "AMBUJACEM"],
    "telecom": ["BHARTIARTL", "INDIGO"],
    "consumer": ["TITAN", "TRENT", "DMART", "PAGEIND", "PIDILITIND", "BERGEPAINT", "ASIANPAINT", "HAVELLS"],
}
