"""
Trading Signals API routes
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional

router = APIRouter()


class SignalRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    signal: str = Field(..., description="STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL")
    strength: float = Field(..., ge=0, le=1, description="Signal strength 0-1")
    source: str = Field("manual", description="Signal source")
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None


# In-memory signal storage (would use database in production)
pending_signals: List[dict] = []
signal_history: List[dict] = []


@router.post("/add")
async def add_signal(signal: SignalRequest, request: Request):
    """Add a trading signal"""
    from datetime import datetime

    signal_dict = {
        "id": len(signal_history) + 1,
        "symbol": signal.symbol.upper(),
        "signal": signal.signal.upper(),
        "strength": signal.strength,
        "source": signal.source,
        "price_target": signal.price_target,
        "stop_loss": signal.stop_loss,
        "timestamp": datetime.now().isoformat(),
        "status": "pending"
    }

    pending_signals.append(signal_dict)
    signal_history.append(signal_dict)

    # Broadcast signal
    await request.app.state.ws_manager.broadcast_signal(signal_dict)

    return {"success": True, "signal": signal_dict}


@router.get("/pending")
async def get_pending_signals():
    """Get pending signals"""
    return {
        "signals": pending_signals,
        "count": len(pending_signals)
    }


@router.get("/history")
async def get_signal_history(limit: int = 50):
    """Get signal history"""
    return {
        "signals": signal_history[-limit:],
        "total": len(signal_history)
    }


@router.post("/execute/{signal_id}")
async def execute_signal(signal_id: int, request: Request):
    """Execute a pending signal"""
    global pending_signals

    signal = next((s for s in pending_signals if s["id"] == signal_id), None)
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")

    trader = request.app.state.trader_service

    try:
        # Map signal to action
        action_map = {
            "STRONG_BUY": "BUY",
            "BUY": "BUY",
            "SELL": "SELL",
            "STRONG_SELL": "SELL"
        }

        action = action_map.get(signal["signal"])
        if not action or signal["signal"] == "HOLD":
            raise HTTPException(status_code=400, detail="Cannot execute HOLD signal")

        # Calculate quantity based on signal strength
        portfolio = trader.get_portfolio_summary()
        max_position = portfolio["total_value"] * 0.05 * signal["strength"]
        price = signal.get("price_target") or trader._get_simulated_price(signal["symbol"])
        quantity = int(max_position / price)

        if quantity < 1:
            raise HTTPException(status_code=400, detail="Position too small")

        trade = await trader.place_order(
            symbol=signal["symbol"],
            action=action,
            quantity=quantity,
            price=price
        )

        # Update signal status
        signal["status"] = "executed"
        signal["trade_id"] = trade.trade_id

        # Remove from pending
        pending_signals = [s for s in pending_signals if s["id"] != signal_id]

        return {
            "success": True,
            "signal": signal,
            "trade": trade.to_dict()
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/pending/{signal_id}")
async def dismiss_signal(signal_id: int):
    """Dismiss a pending signal"""
    global pending_signals

    signal = next((s for s in pending_signals if s["id"] == signal_id), None)
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")

    signal["status"] = "dismissed"
    pending_signals = [s for s in pending_signals if s["id"] != signal_id]

    return {"success": True, "message": "Signal dismissed"}


@router.delete("/pending")
async def clear_pending_signals():
    """Clear all pending signals"""
    global pending_signals
    count = len(pending_signals)
    pending_signals = []
    return {"success": True, "cleared": count}
