"""
Portfolio API routes
"""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/summary")
async def get_portfolio_summary(request: Request):
    """Get portfolio summary"""
    trader = request.app.state.trader_service
    return trader.get_portfolio_summary()


@router.get("/positions")
async def get_positions(request: Request):
    """Get all positions"""
    trader = request.app.state.trader_service
    return {
        "positions": trader.get_positions(),
        "count": len(trader.positions)
    }


@router.get("/positions/{symbol}")
async def get_position(symbol: str, request: Request):
    """Get specific position"""
    trader = request.app.state.trader_service

    if symbol.upper() not in trader.positions:
        return {"error": "Position not found", "symbol": symbol}

    return trader.positions[symbol.upper()].to_dict()


@router.get("/performance")
async def get_performance(request: Request):
    """Get performance metrics"""
    trader = request.app.state.trader_service
    summary = trader.get_portfolio_summary()

    # Calculate additional metrics
    winning_trades = [t for t in trader.trades if t.action == "SELL" and t.pnl > 0]
    losing_trades = [t for t in trader.trades if t.action == "SELL" and t.pnl < 0]

    return {
        "total_return_pct": summary["total_return_pct"],
        "realized_pnl": summary["realized_pnl"],
        "unrealized_pnl": summary["unrealized_pnl"],
        "total_trades": len(trader.trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": len(winning_trades) / max(len(winning_trades) + len(losing_trades), 1) * 100,
        "avg_win": sum(t.pnl for t in winning_trades) / max(len(winning_trades), 1),
        "avg_loss": sum(t.pnl for t in losing_trades) / max(len(losing_trades), 1),
        "max_drawdown_pct": trader.max_drawdown * 100
    }


@router.get("/history")
async def get_portfolio_history(request: Request, limit: int = 100):
    """Get portfolio value history"""
    # This would return historical portfolio values for charting
    # For now, return current snapshot
    trader = request.app.state.trader_service
    return {
        "history": [trader.get_portfolio_summary()],
        "count": 1
    }
