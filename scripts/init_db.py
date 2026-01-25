#!/usr/bin/env python3
"""
Database Initialization Script
Sets up PostgreSQL with TimescaleDB for InvestLLM
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import structlog

from investllm.config import settings, STOCK_UNIVERSE, NIFTY_50, NIFTY_NEXT_50, SECTOR_MAP
from investllm.data.models import Base, Stock, create_hypertables

logger = structlog.get_logger()


def create_database():
    """Create the database and all tables"""
    
    # Use sync engine for setup
    sync_url = settings.db_url.replace('+asyncpg', '')
    
    print(f"Connecting to: {sync_url.replace(settings.postgres_password, '***')}")
    
    try:
        engine = create_engine(sync_url)
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"PostgreSQL version: {version}")
        
        # Create all tables
        print("\nCreating tables...")
        Base.metadata.create_all(engine)
        print("✅ Tables created")
        
        # Create hypertables (TimescaleDB)
        print("\nCreating TimescaleDB hypertables...")
        try:
            create_hypertables(engine)
            print("✅ Hypertables created")
        except Exception as e:
            print(f"⚠️  Hypertable creation skipped (TimescaleDB may not be installed): {e}")
        
        # Seed stock master data
        print("\nSeeding stock master data...")
        seed_stocks(engine)
        print("✅ Stock data seeded")
        
        print("\n" + "="*50)
        print("✅ Database initialization complete!")
        print("="*50)
        
        return engine
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure PostgreSQL is running and the database exists.")
        print("You can create it with:")
        print(f"  createdb {settings.postgres_db}")
        raise


def seed_stocks(engine):
    """Seed the stocks table with NIFTY 100 stocks"""
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Check if already seeded
        existing = session.query(Stock).count()
        if existing > 0:
            print(f"  Stocks table already has {existing} records")
            return
        
        # Build sector lookup
        symbol_to_sector = {}
        for sector, symbols in SECTOR_MAP.items():
            for symbol in symbols:
                symbol_to_sector[symbol] = sector
        
        # Add NIFTY 50 stocks
        for symbol in NIFTY_50:
            stock = Stock(
                symbol=symbol,
                nse_symbol=symbol,
                is_nifty50=True,
                is_nifty100=True,
                sector=symbol_to_sector.get(symbol),
                is_active=True
            )
            session.add(stock)
        
        # Add NIFTY NEXT 50 stocks
        for symbol in NIFTY_NEXT_50:
            if symbol not in NIFTY_50:  # Avoid duplicates
                stock = Stock(
                    symbol=symbol,
                    nse_symbol=symbol,
                    is_nifty50=False,
                    is_niftynext50=True,
                    is_nifty100=True,
                    sector=symbol_to_sector.get(symbol),
                    is_active=True
                )
                session.add(stock)
        
        session.commit()
        print(f"  Added {len(NIFTY_50) + len(NIFTY_NEXT_50)} stocks")
        
    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()


def check_database_status():
    """Check the status of the database"""
    
    sync_url = settings.db_url.replace('+asyncpg', '')
    engine = create_engine(sync_url)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        print("\n📊 Database Status")
        print("="*50)
        
        # Check tables
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"\nTables: {len(tables)}")
        for table in tables:
            print(f"  - {table}")
        
        # Check record counts
        print("\nRecord counts:")
        
        tables_to_check = ['stocks', 'price_data', 'news_articles', 'fundamentals']
        for table in tables_to_check:
            if table in tables:
                result = session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.scalar()
                print(f"  - {table}: {count:,}")
        
        # Check TimescaleDB
        print("\nTimescaleDB status:")
        try:
            result = session.execute(text("""
                SELECT hypertable_name, num_chunks 
                FROM timescaledb_information.hypertables
            """))
            hypertables = result.fetchall()
            for ht in hypertables:
                print(f"  - {ht[0]}: {ht[1]} chunks")
        except Exception as e:
            print(f"  TimescaleDB not available: {e}")
        
    finally:
        session.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize InvestLLM database")
    parser.add_argument("--check", action="store_true", help="Check database status")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate all tables")
    
    args = parser.parse_args()
    
    if args.check:
        check_database_status()
    elif args.reset:
        print("⚠️  This will delete all data. Are you sure? (yes/no)")
        if input().lower() == 'yes':
            sync_url = settings.db_url.replace('+asyncpg', '')
            engine = create_engine(sync_url)
            Base.metadata.drop_all(engine)
            print("Tables dropped")
            create_database()
    else:
        create_database()
