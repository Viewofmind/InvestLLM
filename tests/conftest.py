"""
Pytest Configuration and Fixtures
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing"""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=100),
        periods=100,
        freq='D'
    )
    
    return pd.DataFrame({
        'timestamp': dates,
        'symbol': 'TEST',
        'open': [100 + i * 0.1 for i in range(100)],
        'high': [101 + i * 0.1 for i in range(100)],
        'low': [99 + i * 0.1 for i in range(100)],
        'close': [100.5 + i * 0.1 for i in range(100)],
        'volume': [1000000 + i * 1000 for i in range(100)]
    })


@pytest.fixture
def sample_news_article():
    """Generate sample news article for testing"""
    return {
        'url': 'https://example.com/news/1',
        'title': 'Reliance Industries Q3 Results: Net Profit Rises 10%',
        'content': '''Reliance Industries reported strong Q3 results with 
        net profit rising 10% year-over-year. The retail segment showed 
        exceptional growth with same-store sales increasing by 12%. 
        Management remains optimistic about future growth prospects.''',
        'source': 'test_source',
        'published_at': datetime.now(),
        'author': 'Test Author'
    }


@pytest.fixture
def sample_fundamentals():
    """Generate sample fundamental data for testing"""
    return {
        'symbol': 'TEST',
        'name': 'Test Company Ltd',
        'sector': 'Technology',
        'industry': 'Software',
        'market_cap': 1000000000000,
        'pe_ratio': 25.5,
        'pb_ratio': 3.2,
        'roe': 0.18,
        'debt_to_equity': 0.5,
        'revenue_growth': 0.15,
        'profit_margin': 0.12
    }


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create subdirectories
        for subdir in ['raw', 'processed', 'features', 'news']:
            os.makedirs(os.path.join(tmpdir, subdir))
        yield Path(tmpdir)


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set mock environment variables for testing"""
    monkeypatch.setenv('APP_ENV', 'test')
    monkeypatch.setenv('POSTGRES_HOST', 'localhost')
    monkeypatch.setenv('POSTGRES_PORT', '5432')
    monkeypatch.setenv('POSTGRES_DB', 'investllm_test')
    monkeypatch.setenv('POSTGRES_USER', 'test')
    monkeypatch.setenv('POSTGRES_PASSWORD', 'test')
