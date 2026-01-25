"""
Fundamental Data Collector
Collects quarterly results, financial ratios, and company metrics
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import structlog
import time

from investllm.config import settings, STOCK_UNIVERSE, NIFTY_50

logger = structlog.get_logger()


class FundamentalCollector:
    """
    Collects fundamental data from multiple sources:
    - yfinance: Free, good for basic fundamentals
    - Screener.in: Better for Indian specifics (if API available)
    - BSE/NSE: Official filings
    
    Usage:
        collector = FundamentalCollector()
        
        # Get single stock fundamentals
        data = collector.get_fundamentals("RELIANCE")
        
        # Collect all
        collector.collect_all_fundamentals()
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.data_dir / "fundamentals"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("fundamental_collector_initialized")
    
    def get_fundamentals(self, symbol: str) -> Dict:
        """
        Get comprehensive fundamentals for a stock
        
        Returns:
            Dictionary with all available fundamental data
        """
        yf_symbol = f"{symbol}.NS"
        
        try:
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info
            
            # Basic info
            fundamentals = {
                'symbol': symbol,
                'name': info.get('longName'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'website': info.get('website'),
                'description': info.get('longBusinessSummary'),
                
                # Market data
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'shares_outstanding': info.get('sharesOutstanding'),
                'float_shares': info.get('floatShares'),
                
                # Valuation ratios
                'pe_trailing': info.get('trailingPE'),
                'pe_forward': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'pb_ratio': info.get('priceToBook'),
                'ps_ratio': info.get('priceToSalesTrailing12Months'),
                'ev_to_revenue': info.get('enterpriseToRevenue'),
                'ev_to_ebitda': info.get('enterpriseToEbitda'),
                
                # Profitability
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'gross_margin': info.get('grossMargins'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                
                # Growth
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                
                # Financial health
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                'debt_to_equity': info.get('debtToEquity'),
                'total_debt': info.get('totalDebt'),
                'total_cash': info.get('totalCash'),
                
                # Dividends
                'dividend_yield': info.get('dividendYield'),
                'dividend_rate': info.get('dividendRate'),
                'payout_ratio': info.get('payoutRatio'),
                'ex_dividend_date': info.get('exDividendDate'),
                
                # Per share data
                'eps_trailing': info.get('trailingEps'),
                'eps_forward': info.get('forwardEps'),
                'book_value': info.get('bookValue'),
                
                # Price data
                'current_price': info.get('currentPrice'),
                'target_high': info.get('targetHighPrice'),
                'target_low': info.get('targetLowPrice'),
                'target_mean': info.get('targetMeanPrice'),
                'week_52_high': info.get('fiftyTwoWeekHigh'),
                'week_52_low': info.get('fiftyTwoWeekLow'),
                
                # Analyst data
                'analyst_count': info.get('numberOfAnalystOpinions'),
                'recommendation': info.get('recommendationKey'),
                'recommendation_mean': info.get('recommendationMean'),
                
                # Beta
                'beta': info.get('beta'),
                
                # Metadata
                'collected_at': datetime.now().isoformat()
            }
            
            logger.info("fundamentals_fetched", symbol=symbol)
            return fundamentals
            
        except Exception as e:
            logger.error("fundamentals_error", symbol=symbol, error=str(e))
            return {'symbol': symbol, 'error': str(e)}
    
    def get_financial_statements(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Get financial statements (income, balance sheet, cash flow)
        """
        yf_symbol = f"{symbol}.NS"
        
        try:
            ticker = yf.Ticker(yf_symbol)
            
            return {
                'income_statement': ticker.income_stmt,
                'quarterly_income': ticker.quarterly_income_stmt,
                'balance_sheet': ticker.balance_sheet,
                'quarterly_balance': ticker.quarterly_balance_sheet,
                'cashflow': ticker.cashflow,
                'quarterly_cashflow': ticker.quarterly_cashflow
            }
            
        except Exception as e:
            logger.error("financial_statements_error", symbol=symbol, error=str(e))
            return {}
    
    def get_key_metrics_history(self, symbol: str) -> pd.DataFrame:
        """
        Get historical key metrics (from quarterly data)
        """
        statements = self.get_financial_statements(symbol)
        
        if not statements or 'quarterly_income' not in statements:
            return pd.DataFrame()
        
        try:
            income = statements['quarterly_income']
            balance = statements['quarterly_balance']
            
            metrics = pd.DataFrame()
            
            if not income.empty:
                metrics['revenue'] = income.loc['Total Revenue'] if 'Total Revenue' in income.index else None
                metrics['net_income'] = income.loc['Net Income'] if 'Net Income' in income.index else None
                metrics['operating_income'] = income.loc['Operating Income'] if 'Operating Income' in income.index else None
                metrics['gross_profit'] = income.loc['Gross Profit'] if 'Gross Profit' in income.index else None
            
            if not balance.empty:
                metrics['total_assets'] = balance.loc['Total Assets'] if 'Total Assets' in balance.index else None
                metrics['total_debt'] = balance.loc['Total Debt'] if 'Total Debt' in balance.index else None
                metrics['stockholder_equity'] = balance.loc['Stockholders Equity'] if 'Stockholders Equity' in balance.index else None
            
            # Calculate ratios
            if 'net_income' in metrics and 'stockholder_equity' in metrics:
                metrics['roe'] = metrics['net_income'] / metrics['stockholder_equity']
            
            if 'net_income' in metrics and 'revenue' in metrics:
                metrics['profit_margin'] = metrics['net_income'] / metrics['revenue']
            
            metrics['symbol'] = symbol
            
            return metrics
            
        except Exception as e:
            logger.error("metrics_history_error", symbol=symbol, error=str(e))
            return pd.DataFrame()
    
    def collect_all_fundamentals(
        self,
        symbols: List[str] = None,
        save: bool = True,
        max_workers: int = 5
    ) -> Dict[str, Dict]:
        """
        Collect fundamentals for all stocks
        """
        symbols = symbols or STOCK_UNIVERSE
        
        logger.info("starting_fundamental_collection", count=len(symbols))
        
        results = {}
        failed = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.get_fundamentals, symbol): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                
                try:
                    data = future.result()
                    if 'error' not in data:
                        results[symbol] = data
                    else:
                        failed.append(symbol)
                except Exception as e:
                    logger.error("fundamental_fetch_failed", symbol=symbol, error=str(e))
                    failed.append(symbol)
                
                time.sleep(0.5)  # Rate limiting
        
        # Save to file
        if save and results:
            df = pd.DataFrame(list(results.values()))
            filepath = self.data_dir / "fundamentals_snapshot.parquet"
            df.to_parquet(filepath)
            
            # Also save as CSV for easy viewing
            df.to_csv(self.data_dir / "fundamentals_snapshot.csv", index=False)
            
            logger.info("fundamentals_saved", filepath=str(filepath))
        
        logger.info(
            "fundamental_collection_complete",
            success=len(results),
            failed=len(failed)
        )
        
        return results
    
    def collect_financial_statements(
        self,
        symbols: List[str] = None,
        max_workers: int = 5
    ):
        """
        Collect financial statements for all stocks
        """
        symbols = symbols or NIFTY_50  # Start with NIFTY 50
        
        logger.info("collecting_financial_statements", count=len(symbols))
        
        for symbol in symbols:
            try:
                statements = self.get_financial_statements(symbol)
                
                # Save each statement
                symbol_dir = self.data_dir / "statements" / symbol
                symbol_dir.mkdir(parents=True, exist_ok=True)
                
                for name, df in statements.items():
                    if df is not None and not df.empty:
                        df.to_parquet(symbol_dir / f"{name}.parquet")
                
                logger.info("statements_saved", symbol=symbol)
                
            except Exception as e:
                logger.error("statements_error", symbol=symbol, error=str(e))
            
            time.sleep(1)  # Rate limiting
    
    def get_peer_comparison(self, symbol: str, sector: str = None) -> pd.DataFrame:
        """
        Compare stock with sector peers
        """
        from investllm.config import SECTOR_MAP
        
        # Find peers
        if sector:
            peers = SECTOR_MAP.get(sector.lower(), [])
        else:
            # Get sector from stock
            fundamentals = self.get_fundamentals(symbol)
            sector = fundamentals.get('sector', '').lower()
            
            # Map to our sectors
            sector_mapping = {
                'financial services': 'banking',
                'technology': 'it',
                'healthcare': 'pharma',
                'consumer defensive': 'fmcg',
                'energy': 'energy',
                'basic materials': 'metals',
                'industrials': 'infrastructure',
                'consumer cyclical': 'auto',
            }
            
            mapped_sector = sector_mapping.get(sector, sector)
            peers = SECTOR_MAP.get(mapped_sector, [])
        
        if not peers:
            return pd.DataFrame()
        
        # Collect fundamentals for all peers
        peer_data = []
        for peer in peers[:10]:  # Limit to 10 peers
            data = self.get_fundamentals(peer)
            if 'error' not in data:
                peer_data.append(data)
            time.sleep(0.3)
        
        return pd.DataFrame(peer_data)
    
    def load_fundamentals(self) -> pd.DataFrame:
        """Load saved fundamentals"""
        filepath = self.data_dir / "fundamentals_snapshot.parquet"
        
        if not filepath.exists():
            return pd.DataFrame()
        
        return pd.read_parquet(filepath)


# ===========================================
# CONVENIENCE FUNCTIONS
# ===========================================

def collect_all_fundamentals():
    """Convenience function to collect all fundamentals"""
    collector = FundamentalCollector()
    
    print("Collecting fundamentals for all stocks...")
    results = collector.collect_all_fundamentals()
    
    print(f"\n✅ Collected fundamentals for {len(results)} stocks")
    print(f"Data saved to: {collector.data_dir}")
    
    return results


if __name__ == "__main__":
    collect_all_fundamentals()
