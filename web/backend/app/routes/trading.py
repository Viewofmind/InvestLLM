"""
Trading API routes
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional

router = APIRouter()


class OrderRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., RELIANCE)")
    action: str = Field(..., description="BUY or SELL")
    quantity: int = Field(..., gt=0, description="Number of shares")
    price: Optional[float] = Field(None, description="Limit price (None for market)")


class ClosePositionRequest(BaseModel):
    symbol: str


@router.post("/order")
async def place_order(order: OrderRequest, request: Request):
    """Place a buy or sell order"""
    trader = request.app.state.trader_service

    try:
        trade = await trader.place_order(
            symbol=order.symbol.upper(),
            action=order.action.upper(),
            quantity=order.quantity,
            price=order.price
        )
        return {
            "success": True,
            "trade": trade.to_dict()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/close/{symbol}")
async def close_position(symbol: str, request: Request):
    """Close a specific position"""
    trader = request.app.state.trader_service

    try:
        trade = await trader.close_position(symbol.upper())
        return {
            "success": True,
            "trade": trade.to_dict()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/close-all")
async def close_all_positions(request: Request):
    """Close all positions"""
    trader = request.app.state.trader_service

    trades = await trader.close_all_positions()
    return {
        "success": True,
        "trades": [t.to_dict() for t in trades],
        "closed_count": len(trades)
    }


@router.get("/trades")
async def get_trades(request: Request, limit: int = 50):
    """Get recent trades"""
    trader = request.app.state.trader_service
    return {
        "trades": trader.get_trades(limit),
        "total": len(trader.trades)
    }


@router.post("/start")
async def start_trading(request: Request):
    """Start automated trading"""
    trader = request.app.state.trader_service
    trader.is_running = True
    return {"success": True, "message": "Trading started"}


@router.post("/stop")
async def stop_trading(request: Request):
    """Stop automated trading"""
    trader = request.app.state.trader_service
    trader.is_running = False
    return {"success": True, "message": "Trading stopped"}
