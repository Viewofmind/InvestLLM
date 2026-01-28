"""
InvestLLM Trading Platform - FastAPI Backend
=============================================
Production-ready API for algorithmic trading.

Features:
- REST API for trading operations
- WebSocket for real-time updates
- Paper/Live trading mode switching
- Portfolio management
- Risk monitoring
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import routes
from app.routes import trading, portfolio, risk, signals, settings
from app.core.config import settings as app_settings
from app.core.websocket import ConnectionManager
from app.services.trader_service import TraderService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

# WebSocket connection manager
ws_manager = ConnectionManager()

# Trader service (singleton)
trader_service: Optional[TraderService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global trader_service

    # Startup
    logger.info("Starting InvestLLM Trading Platform...")
    trader_service = TraderService(ws_manager=ws_manager)
    await trader_service.initialize()

    # Store in app state
    app.state.trader_service = trader_service
    app.state.ws_manager = ws_manager

    logger.info("Trading platform ready!")

    yield

    # Shutdown
    logger.info("Shutting down trading platform...")
    if trader_service:
        await trader_service.shutdown()


# Create FastAPI app
app = FastAPI(
    title="InvestLLM Trading Platform",
    description="AI-powered algorithmic trading for Indian markets",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(trading.router, prefix="/api/trading", tags=["Trading"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["Portfolio"])
app.include_router(risk.router, prefix="/api/risk", tags=["Risk Management"])
app.include_router(signals.router, prefix="/api/signals", tags=["Signals"])
app.include_router(settings.router, prefix="/api/settings", tags=["Settings"])


# =============================================================================
# ROOT ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "InvestLLM Trading Platform",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    trader = app.state.trader_service

    return {
        "status": "healthy",
        "trading_mode": trader.mode.value if trader else "unknown",
        "trading_active": trader.is_running if trader else False,
        "connected_clients": len(app.state.ws_manager.active_connections),
        "timestamp": datetime.now().isoformat()
    }


# =============================================================================
# WEBSOCKET ENDPOINT
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.

    Sends:
    - Portfolio updates
    - Trade executions
    - Risk alerts
    - Price quotes
    - Signal notifications
    """
    await ws_manager.connect(websocket)

    try:
        # Send initial state
        trader = app.state.trader_service
        if trader:
            await websocket.send_json({
                "type": "init",
                "data": {
                    "mode": trader.mode.value,
                    "portfolio": trader.get_portfolio_summary(),
                    "risk": trader.get_risk_summary()
                }
            })

        # Keep connection alive and handle incoming messages
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=30.0
                )

                # Handle client messages
                msg_type = data.get("type")

                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

                elif msg_type == "subscribe":
                    # Handle subscription to specific channels
                    channels = data.get("channels", [])
                    ws_manager.subscribe(websocket, channels)

            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500
        }
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
