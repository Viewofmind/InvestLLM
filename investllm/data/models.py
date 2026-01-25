"""
Database Models for InvestLLM
Uses SQLAlchemy with TimescaleDB hypertables for time-series data
"""

from datetime import datetime, date
from typing import Optional, List
from decimal import Decimal
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Date, Boolean,
    Text, ForeignKey, Index, UniqueConstraint, BigInteger,
    Numeric, JSON, Enum as SQLEnum
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
import enum

Base = declarative_base()


# ===========================================
# ENUMS
# ===========================================

class Exchange(str, enum.Enum):
    NSE = "NSE"
    BSE = "BSE"


class Interval(str, enum.Enum):
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    DAY = "1d"
    WEEK = "1w"
    MONTH = "1M"


class SentimentLabel(str, enum.Enum):
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class EventType(str, enum.Enum):
    EARNINGS = "earnings"
    DIVIDEND = "dividend"
    SPLIT = "split"
    BONUS = "bonus"
    AGM = "agm"
    BOARD_MEETING = "board_meeting"
    POLICY = "policy"
    REGULATORY = "regulatory"
    MERGER = "merger"
    BUYBACK = "buyback"
    OTHER = "other"


# ===========================================
# STOCK MASTER DATA
# ===========================================

class Stock(Base):
    """Master table for stocks"""
    __tablename__ = "stocks"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(255))
    isin = Column(String(20), unique=True)
    
    # Exchange info
    nse_symbol = Column(String(50))
    bse_code = Column(String(20))
    
    # Classification
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap_category = Column(String(20))  # large, mid, small
    
    # Index membership
    is_nifty50 = Column(Boolean, default=False)
    is_nifty100 = Column(Boolean, default=False)
    is_niftynext50 = Column(Boolean, default=False)
    
    # Status
    is_active = Column(Boolean, default=True)
    listing_date = Column(Date)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    prices = relationship("PriceData", back_populates="stock")
    fundamentals = relationship("Fundamental", back_populates="stock")
    news_mentions = relationship("NewsMention", back_populates="stock")


# ===========================================
# PRICE DATA (TimescaleDB Hypertable)
# ===========================================

class PriceData(Base):
    """
    OHLCV price data - converted to TimescaleDB hypertable
    Partitioned by time for efficient time-series queries
    """
    __tablename__ = "price_data"
    
    # Composite primary key: time + stock_id + interval
    timestamp = Column(DateTime, primary_key=True, nullable=False)
    stock_id = Column(Integer, ForeignKey("stocks.id"), primary_key=True)
    interval = Column(String(10), primary_key=True, default="1d")
    
    # OHLCV
    open = Column(Numeric(12, 2), nullable=False)
    high = Column(Numeric(12, 2), nullable=False)
    low = Column(Numeric(12, 2), nullable=False)
    close = Column(Numeric(12, 2), nullable=False)
    volume = Column(BigInteger, nullable=False)
    
    # Additional
    vwap = Column(Numeric(12, 2))
    trades = Column(Integer)
    deliverable_volume = Column(BigInteger)
    deliverable_percent = Column(Float)
    
    # Adjusted prices (for splits/dividends)
    adj_close = Column(Numeric(12, 2))
    
    # Relationships
    stock = relationship("Stock", back_populates="prices")
    
    __table_args__ = (
        Index("idx_price_stock_time", "stock_id", "timestamp"),
        Index("idx_price_time", "timestamp"),
    )


class PriceMinute(Base):
    """
    Minute-level price data - separate table for performance
    """
    __tablename__ = "price_minute"
    
    timestamp = Column(DateTime, primary_key=True, nullable=False)
    stock_id = Column(Integer, ForeignKey("stocks.id"), primary_key=True)
    
    open = Column(Numeric(12, 2), nullable=False)
    high = Column(Numeric(12, 2), nullable=False)
    low = Column(Numeric(12, 2), nullable=False)
    close = Column(Numeric(12, 2), nullable=False)
    volume = Column(BigInteger, nullable=False)
    
    __table_args__ = (
        Index("idx_minute_stock_time", "stock_id", "timestamp"),
    )


# ===========================================
# FUNDAMENTAL DATA
# ===========================================

class Fundamental(Base):
    """Quarterly fundamental data"""
    __tablename__ = "fundamentals"
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    
    # Period
    fiscal_year = Column(Integer, nullable=False)
    quarter = Column(Integer, nullable=False)  # 1, 2, 3, 4
    period_end = Column(Date, nullable=False)
    
    # Income Statement
    revenue = Column(Numeric(18, 2))
    operating_profit = Column(Numeric(18, 2))
    net_profit = Column(Numeric(18, 2))
    eps = Column(Numeric(10, 2))
    
    # Margins
    gross_margin = Column(Float)
    operating_margin = Column(Float)
    net_margin = Column(Float)
    
    # Balance Sheet
    total_assets = Column(Numeric(18, 2))
    total_liabilities = Column(Numeric(18, 2))
    total_equity = Column(Numeric(18, 2))
    total_debt = Column(Numeric(18, 2))
    cash = Column(Numeric(18, 2))
    
    # Ratios
    pe_ratio = Column(Float)
    pb_ratio = Column(Float)
    roe = Column(Float)
    roce = Column(Float)
    debt_to_equity = Column(Float)
    current_ratio = Column(Float)
    
    # Growth (YoY)
    revenue_growth = Column(Float)
    profit_growth = Column(Float)
    
    # Metadata
    reported_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    stock = relationship("Stock", back_populates="fundamentals")
    
    __table_args__ = (
        UniqueConstraint("stock_id", "fiscal_year", "quarter", name="uq_fundamental_period"),
        Index("idx_fundamental_stock", "stock_id"),
    )


# ===========================================
# NEWS & SENTIMENT
# ===========================================

class NewsArticle(Base):
    """News articles corpus"""
    __tablename__ = "news_articles"
    
    id = Column(Integer, primary_key=True)
    
    # Source info
    source = Column(String(50), nullable=False)  # moneycontrol, et, etc.
    url = Column(Text, unique=True, nullable=False)
    
    # Content
    title = Column(Text, nullable=False)
    content = Column(Text)
    summary = Column(Text)
    
    # Dates
    published_at = Column(DateTime, nullable=False, index=True)
    scraped_at = Column(DateTime, default=datetime.utcnow)
    
    # Categories/Tags
    categories = Column(ARRAY(String))
    tags = Column(ARRAY(String))
    
    # AI Analysis
    sentiment_score = Column(Float)  # -1 to 1
    sentiment_label = Column(String(20))
    sentiment_confidence = Column(Float)
    
    # Embeddings stored in vector DB (Qdrant)
    embedding_id = Column(String(100))
    
    # Metadata
    author = Column(String(255))
    
    # Relationships
    mentions = relationship("NewsMention", back_populates="article")
    
    __table_args__ = (
        Index("idx_news_published", "published_at"),
        Index("idx_news_source", "source"),
    )


class NewsMention(Base):
    """Stock mentions in news articles"""
    __tablename__ = "news_mentions"
    
    id = Column(Integer, primary_key=True)
    article_id = Column(Integer, ForeignKey("news_articles.id"), nullable=False)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    
    # Mention details
    mention_count = Column(Integer, default=1)
    is_primary = Column(Boolean, default=False)  # Main subject of article
    
    # Stock-specific sentiment
    sentiment_score = Column(Float)
    
    # Relationships
    article = relationship("NewsArticle", back_populates="mentions")
    stock = relationship("Stock", back_populates="news_mentions")
    
    __table_args__ = (
        UniqueConstraint("article_id", "stock_id", name="uq_mention"),
        Index("idx_mention_stock", "stock_id"),
    )


# ===========================================
# CORPORATE EVENTS
# ===========================================

class CorporateEvent(Base):
    """Corporate actions and events"""
    __tablename__ = "corporate_events"
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    
    # Event details
    event_type = Column(String(50), nullable=False)
    event_date = Column(Date, nullable=False)
    
    # Event-specific data (JSON for flexibility)
    details = Column(JSONB)
    
    # For dividends
    dividend_amount = Column(Numeric(10, 2))
    dividend_type = Column(String(20))  # interim, final
    
    # For splits/bonus
    ratio_from = Column(Integer)
    ratio_to = Column(Integer)
    
    # For earnings
    eps_actual = Column(Numeric(10, 2))
    eps_expected = Column(Numeric(10, 2))
    
    # Dates
    announcement_date = Column(Date)
    record_date = Column(Date)
    ex_date = Column(Date)
    
    # Metadata
    source = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_event_stock_date", "stock_id", "event_date"),
        Index("idx_event_type", "event_type"),
    )


# ===========================================
# GLOBAL FACTORS
# ===========================================

class GlobalFactor(Base):
    """Global market factors (FII/DII, forex, crude, etc.)"""
    __tablename__ = "global_factors"
    
    timestamp = Column(DateTime, primary_key=True)
    
    # FII/DII flows (in crores)
    fii_buy = Column(Numeric(14, 2))
    fii_sell = Column(Numeric(14, 2))
    fii_net = Column(Numeric(14, 2))
    dii_buy = Column(Numeric(14, 2))
    dii_sell = Column(Numeric(14, 2))
    dii_net = Column(Numeric(14, 2))
    
    # Forex
    usdinr = Column(Numeric(10, 4))
    usdinr_change = Column(Float)
    
    # Commodities
    crude_oil = Column(Numeric(10, 2))
    gold = Column(Numeric(10, 2))
    
    # Global indices
    sp500 = Column(Numeric(10, 2))
    nasdaq = Column(Numeric(10, 2))
    dow = Column(Numeric(10, 2))
    sgx_nifty = Column(Numeric(10, 2))
    
    # VIX
    india_vix = Column(Float)
    
    # Interest rates
    repo_rate = Column(Float)
    reverse_repo = Column(Float)
    us_10y_yield = Column(Float)
    
    __table_args__ = (
        Index("idx_global_time", "timestamp"),
    )


# ===========================================
# MODEL TRAINING DATA
# ===========================================

class SentimentLabel(Base):
    """Labeled sentiment data for training"""
    __tablename__ = "sentiment_labels"
    
    id = Column(Integer, primary_key=True)
    
    # Source
    article_id = Column(Integer, ForeignKey("news_articles.id"))
    text = Column(Text, nullable=False)
    
    # Labels
    sentiment = Column(String(20), nullable=False)  # positive, negative, neutral
    confidence = Column(Float)
    
    # Multi-label
    has_earnings = Column(Boolean, default=False)
    has_policy = Column(Boolean, default=False)
    has_management = Column(Boolean, default=False)
    has_sector = Column(Boolean, default=False)
    
    # Stocks mentioned
    stocks_mentioned = Column(ARRAY(String))
    
    # Labeling info
    labeled_by = Column(String(50))  # human, gpt4, claude, gemini
    verified = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionRecord(Base):
    """Record of model predictions for tracking"""
    __tablename__ = "prediction_records"
    
    id = Column(Integer, primary_key=True)
    
    # Prediction details
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    prediction_date = Column(DateTime, nullable=False)
    target_date = Column(DateTime, nullable=False)
    
    # Model info
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50))
    
    # Prediction
    predicted_direction = Column(String(10))  # up, down, neutral
    predicted_change = Column(Float)  # % change
    confidence = Column(Float)
    
    # Actual outcome
    actual_direction = Column(String(10))
    actual_change = Column(Float)
    
    # Performance
    is_correct = Column(Boolean)
    
    # Features used (for analysis)
    features_snapshot = Column(JSONB)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_prediction_stock", "stock_id"),
        Index("idx_prediction_date", "prediction_date"),
    )


# ===========================================
# TRADING SIGNALS & STRATEGIES
# ===========================================

class TradingSignal(Base):
    """Generated trading signals"""
    __tablename__ = "trading_signals"
    
    id = Column(Integer, primary_key=True)
    
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    generated_at = Column(DateTime, nullable=False)
    
    # Signal
    signal_type = Column(String(10), nullable=False)  # buy, sell, hold
    strength = Column(Float)  # 0 to 1
    
    # Prices
    entry_price = Column(Numeric(12, 2))
    target_price = Column(Numeric(12, 2))
    stop_loss = Column(Numeric(12, 2))
    
    # Position sizing
    recommended_size = Column(Float)  # % of capital
    
    # Reasoning
    reasons = Column(JSONB)  # List of reasons
    
    # Model attribution
    sentiment_contribution = Column(Float)
    technical_contribution = Column(Float)
    fundamental_contribution = Column(Float)
    
    # Outcome (filled later)
    executed = Column(Boolean, default=False)
    exit_date = Column(DateTime)
    exit_price = Column(Numeric(12, 2))
    pnl_percent = Column(Float)
    
    __table_args__ = (
        Index("idx_signal_stock_time", "stock_id", "generated_at"),
    )


# ===========================================
# HELPER FUNCTIONS
# ===========================================

def create_hypertables(engine):
    """
    Create TimescaleDB hypertables for time-series data
    Run this after creating tables
    """
    from sqlalchemy import text
    
    hypertables = [
        ("price_data", "timestamp"),
        ("price_minute", "timestamp"),
        ("global_factors", "timestamp"),
    ]
    
    with engine.connect() as conn:
        # Enable TimescaleDB extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
        
        for table, time_column in hypertables:
            try:
                conn.execute(text(f"""
                    SELECT create_hypertable('{table}', '{time_column}', 
                        if_not_exists => TRUE,
                        migrate_data => TRUE
                    );
                """))
                print(f"Created hypertable: {table}")
            except Exception as e:
                print(f"Hypertable {table} may already exist: {e}")
        
        conn.commit()
