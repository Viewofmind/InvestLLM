"""
Settings API routes
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional

from app.core.config import settings

router = APIRouter()


class ModeRequest(BaseModel):
    mode: str = Field(..., description="paper or live")


class CapitalRequest(BaseModel):
    capital: float = Field(..., gt=0, description="Initial capital")


@router.get("/")
async def get_settings(request: Request):
    """Get current settings"""
    trader = request.app.state.trader_service

    return {
        "mode": trader.mode.value,
        "initial_capital": trader.initial_capital,
        "is_running": trader.is_running,
        "risk_limits": {
            "max_position_pct": settings.MAX_POSITION_PCT * 100,
            "max_sector_pct": settings.MAX_SECTOR_PCT * 100,
            "max_drawdown_pct": settings.MAX_DRAWDOWN_PCT * 100,
            "daily_loss_limit_pct": settings.DAILY_LOSS_LIMIT_PCT * 100
        },
        "api_configured": {
            "zerodha": bool(settings.ZERODHA_API_KEY),
            "firecrawl": bool(settings.FIRECRAWL_API_KEY),
            "gemini": bool(settings.GEMINI_API_KEY)
        }
    }


@router.post("/mode")
async def set_mode(mode_request: ModeRequest, request: Request):
    """Switch trading mode"""
    trader = request.app.state.trader_service

    try:
        result = await trader.set_mode(mode_request.mode)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/reset")
async def reset_account(request: Request):
    """Reset paper trading account"""
    trader = request.app.state.trader_service

    try:
        result = await trader.reset()
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/capital")
async def set_capital(capital_request: CapitalRequest, request: Request):
    """Set initial capital (paper trading only)"""
    trader = request.app.state.trader_service

    if trader.mode.value == "live":
        raise HTTPException(status_code=400, detail="Cannot change capital in live mode")

    if trader.positions:
        raise HTTPException(status_code=400, detail="Close all positions before changing capital")

    trader.initial_capital = capital_request.capital
    trader.cash = capital_request.capital
    trader.peak_value = capital_request.capital
    trader.daily_start_value = capital_request.capital

    return {
        "success": True,
        "capital": capital_request.capital
    }
