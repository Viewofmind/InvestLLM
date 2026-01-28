"""
Configuration settings for the trading platform
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # App
    APP_NAME: str = "InvestLLM Trading Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Trading
    INITIAL_CAPITAL: float = 100000.0
    DEFAULT_MODE: str = "paper"  # paper or live

    # Risk limits
    MAX_POSITION_PCT: float = 0.05
    MAX_SECTOR_PCT: float = 0.20
    MAX_DRAWDOWN_PCT: float = 0.15
    DAILY_LOSS_LIMIT_PCT: float = 0.03

    # API Keys
    ZERODHA_API_KEY: Optional[str] = None
    ZERODHA_API_SECRET: Optional[str] = None
    ZERODHA_ACCESS_TOKEN: Optional[str] = None
    FIRECRAWL_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    # Database (optional - for persistence)
    DATABASE_URL: Optional[str] = None

    # Redis (optional - for caching)
    REDIS_URL: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
