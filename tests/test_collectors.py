"""
Tests for Data Collectors
"""

import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path


class TestPriceCollector:
    """Tests for PriceCollector"""
    
    def test_import(self):
        """Test that PriceCollector can be imported"""
        from investllm.data.collectors.price_collector import PriceCollector
        assert PriceCollector is not None
    
    def test_init(self, temp_data_dir):
        """Test PriceCollector initialization"""
        from investllm.data.collectors.price_collector import PriceCollector
        collector = PriceCollector(data_dir=temp_data_dir / "prices")
        assert collector.data_dir.exists()
    
    @pytest.mark.slow
    def test_get_stock_history(self, temp_data_dir):
        """Test fetching stock history (requires network)"""
        from investllm.data.collectors.price_collector import PriceCollector
        collector = PriceCollector(data_dir=temp_data_dir / "prices")
        
        df = collector.get_stock_history("RELIANCE", years=1)
        
        assert not df.empty
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns
        assert 'symbol' in df.columns
    
    def test_save_load_stock_data(self, temp_data_dir, sample_price_data):
        """Test saving and loading stock data"""
        from investllm.data.collectors.price_collector import PriceCollector
        collector = PriceCollector(data_dir=temp_data_dir / "prices")
        
        # Save
        collector._save_stock_data("TEST", sample_price_data)
        
        # Load
        loaded = collector.load_stock_data("TEST")
        
        assert not loaded.empty
        assert len(loaded) == len(sample_price_data)
    
    def test_check_data_quality(self, temp_data_dir, sample_price_data):
        """Test data quality check"""
        from investllm.data.collectors.price_collector import PriceCollector
        collector = PriceCollector(data_dir=temp_data_dir / "prices")
        
        # Save test data
        collector._save_stock_data("TEST", sample_price_data)
        
        # Check quality
        report = collector.check_data_quality("TEST")
        
        assert report['status'] == 'ok'
        assert report['total_rows'] == 100


class TestFundamentalCollector:
    """Tests for FundamentalCollector"""
    
    def test_import(self):
        """Test that FundamentalCollector can be imported"""
        from investllm.data.collectors.fundamental_collector import FundamentalCollector
        assert FundamentalCollector is not None
    
    def test_init(self, temp_data_dir):
        """Test FundamentalCollector initialization"""
        from investllm.data.collectors.fundamental_collector import FundamentalCollector
        collector = FundamentalCollector(data_dir=temp_data_dir / "fundamentals")
        assert collector.data_dir.exists()
    
    @pytest.mark.slow
    def test_get_fundamentals(self, temp_data_dir):
        """Test fetching fundamentals (requires network)"""
        from investllm.data.collectors.fundamental_collector import FundamentalCollector
        collector = FundamentalCollector(data_dir=temp_data_dir / "fundamentals")
        
        data = collector.get_fundamentals("RELIANCE")
        
        assert 'symbol' in data
        assert data['symbol'] == 'RELIANCE'
        assert 'market_cap' in data
        assert 'pe_ratio' in data


class TestNewsCollector:
    """Tests for NewsCollector"""
    
    def test_import(self):
        """Test that NewsCollector can be imported"""
        from investllm.data.collectors.news_collector import NewsCollector
        assert NewsCollector is not None
    
    def test_init(self, temp_data_dir):
        """Test NewsCollector initialization"""
        from investllm.data.collectors.news_collector import NewsCollector
        collector = NewsCollector(data_dir=temp_data_dir / "news")
        assert collector.data_dir.exists()
    
    def test_find_stock_mentions(self, temp_data_dir):
        """Test stock mention detection"""
        from investllm.data.collectors.news_collector import NewsCollector
        collector = NewsCollector(data_dir=temp_data_dir / "news")
        
        text = "Reliance Industries and TCS reported strong results. HDFC Bank also performed well."
        mentions = collector._find_stock_mentions(text)
        
        assert 'RELIANCE' in mentions
        assert 'TCS' in mentions
        assert 'HDFCBANK' in mentions


class TestConfig:
    """Tests for configuration"""
    
    def test_config_import(self):
        """Test that config can be imported"""
        from investllm.config import settings, NIFTY_50, STOCK_UNIVERSE
        assert settings is not None
        assert len(NIFTY_50) == 50
        assert len(STOCK_UNIVERSE) >= 100
    
    def test_sector_map(self):
        """Test sector mapping"""
        from investllm.config import SECTOR_MAP
        
        assert 'banking' in SECTOR_MAP
        assert 'HDFCBANK' in SECTOR_MAP['banking']
        assert 'it' in SECTOR_MAP
        assert 'TCS' in SECTOR_MAP['it']
