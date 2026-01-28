"""
API Routes
"""

from app.routes.trading import router as trading_router
from app.routes.portfolio import router as portfolio_router
from app.routes.risk import router as risk_router
from app.routes.signals import router as signals_router
from app.routes.settings import router as settings_router

__all__ = [
    "trading_router",
    "portfolio_router",
    "risk_router",
    "signals_router",
    "settings_router"
]
