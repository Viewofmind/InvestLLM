"""
Risk Management API routes
"""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/summary")
async def get_risk_summary(request: Request):
    """Get risk metrics summary"""
    trader = request.app.state.trader_service
    return trader.get_risk_summary()


@router.get("/alerts")
async def get_risk_alerts(request: Request):
    """Get active risk alerts"""
    trader = request.app.state.trader_service
    risk = trader.get_risk_summary()

    alerts = []

    # Check drawdown
    if risk["current_drawdown_pct"] >= 15:
        alerts.append({
            "type": "CRITICAL",
            "category": "DRAWDOWN",
            "message": f"Circuit breaker triggered! Drawdown at {risk['current_drawdown_pct']:.1f}%",
            "action": "Trading halted"
        })
    elif risk["current_drawdown_pct"] >= 10:
        alerts.append({
            "type": "HIGH",
            "category": "DRAWDOWN",
            "message": f"High drawdown: {risk['current_drawdown_pct']:.1f}%",
            "action": "Reduce position sizes"
        })
    elif risk["current_drawdown_pct"] >= 5:
        alerts.append({
            "type": "MEDIUM",
            "category": "DRAWDOWN",
            "message": f"Drawdown warning: {risk['current_drawdown_pct']:.1f}%"
        })

    # Check sector exposure
    for sector, exposure in risk["sector_exposure"].items():
        if exposure > 20:
            alerts.append({
                "type": "HIGH",
                "category": "SECTOR",
                "message": f"{sector} exposure at {exposure:.1f}% (limit: 20%)",
                "action": f"Reduce {sector} positions"
            })

    # Check position concentration
    if risk["largest_position_pct"] > 10:
        alerts.append({
            "type": "MEDIUM",
            "category": "CONCENTRATION",
            "message": f"Largest position is {risk['largest_position_pct']:.1f}% of portfolio"
        })

    return {
        "alerts": alerts,
        "count": len(alerts),
        "risk_level": risk["risk_level"]
    }


@router.get("/limits")
async def get_risk_limits(request: Request):
    """Get configured risk limits"""
    trader = request.app.state.trader_service
    return trader.get_risk_summary()["limits"]
